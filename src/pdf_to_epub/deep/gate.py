"""Evidence-aware safety gate for DeepSeek OCR corrections.

DeepSeek proposes edits; this module decides whether independent OCR evidence is
strong enough to apply them. Confidence is therefore not a global on/off switch:
a lower-confidence suggestion can pass when OCR alternatives agree, while an
unsupported rewrite remains blocked even at high confidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from ..models import DeepQueueItem
from ..ocr.scoring import diacritic_only, normalize_token, shape_key


@dataclass(frozen=True, slots=True)
class EvidenceProfile:
    old_votes: int
    new_votes: int
    candidate_count: int
    shape_preserving: bool
    segmented: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalized_words(text: str) -> list[str]:
    return [normalize_token(part) for part in text.split() if normalize_token(part)]


def _candidate_contains_sequence(candidate: str, value: str) -> bool:
    """Match one token or a short segmented phrase inside an OCR alternative."""

    candidate_words = _normalized_words(candidate)
    wanted = _normalized_words(value)
    if not wanted or len(wanted) > len(candidate_words):
        return False
    width = len(wanted)
    return any(candidate_words[index : index + width] == wanted for index in range(len(candidate_words) - width + 1))


def _evidence(item: DeepQueueItem, old: str, new: str) -> EvidenceProfile:
    old_votes = sum(1 for candidate in item.candidates if _candidate_contains_sequence(candidate, old))
    new_votes = sum(1 for candidate in item.candidates if _candidate_contains_sequence(candidate, new))
    segmented = len(new.split()) > 1
    return EvidenceProfile(
        old_votes=old_votes,
        new_votes=new_votes,
        candidate_count=len(item.candidates),
        shape_preserving=bool(shape_key(old)) and shape_key(old) == shape_key(new),
        segmented=segmented,
    )


def _token_matches(text: str, token: str) -> list[re.Match[str]]:
    """Find exact OCR-token occurrences without matching inside another word."""

    pattern = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)", re.UNICODE)
    return list(pattern.finditer(text))


def _replace_unique_token(text: str, old: str, new: str) -> tuple[str, int]:
    matches = _token_matches(text, old)
    if len(matches) != 1:
        return text, len(matches)
    match = matches[0]
    return text[: match.start()] + new + text[match.end() :], 1


def _operation_kind(raw: dict[str, Any], new: str) -> str:
    declared = str(raw.get("kind") or "").strip().casefold()
    if declared in {"replace", "segment"}:
        return declared
    return "segment" if any(char.isspace() for char in new) else "replace"


def _valid_segment(old: str, new: str) -> bool:
    """Allow one fused OCR token to become two/three real words, never a rewrite."""

    if any(char.isspace() for char in old):
        return False
    parts = new.split()
    return 2 <= len(parts) <= 3 and all(normalize_token(part) for part in parts)


def _required_confidence(base: float, evidence: EvidenceProfile) -> float | None:
    """Convert independent visual support into a dynamic confidence threshold.

    With the default base=0.97:
      - 2+ OCR alternatives supporting NEW: threshold 0.92
      - 1 OCR alternative supporting NEW: threshold 0.95
      - shape-preserving candidate support (diacritics/segmentation): up to 0.02 lower
      - no visual NEW support: only shape-preserving edits at >=0.98
    """

    if evidence.new_votes >= 2:
        threshold = max(0.90, base - 0.05)
        if evidence.shape_preserving:
            threshold = max(0.90, threshold - 0.02)
        return threshold
    if evidence.new_votes == 1:
        threshold = max(0.92, base - 0.02)
        if evidence.shape_preserving:
            threshold = max(0.90, threshold - 0.02)
        return threshold
    if evidence.shape_preserving:
        return max(base, 0.98)
    return None


def apply_ai_ops(
    item: DeepQueueItem,
    ai_ops: list[dict[str, Any]],
    min_confidence: float,
    max_ops: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply only token-local repairs supported by OCR evidence or glyph shape."""

    current = item.current
    audited: list[dict[str, Any]] = []

    for raw in ai_ops[:max_ops]:
        old = str(raw.get("old") or "")
        new = str(raw.get("new") or "")
        kind = _operation_kind(raw, new)
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        record: dict[str, Any] = {
            "kind": kind,
            "old": old,
            "new": new,
            "confidence": confidence,
            "gate": "",
            "applied": False,
        }

        if not old or not new:
            record["gate"] = "empty_token"
        elif any(char.isspace() for char in old):
            record["gate"] = "old_must_be_one_token"
        elif old == new:
            record["gate"] = "same_value"
        elif kind == "replace" and any(char.isspace() for char in new):
            record["gate"] = "replace_new_must_be_one_token"
        elif kind == "segment" and not _valid_segment(old, new):
            record["gate"] = "invalid_segmentation"
        else:
            _, occurrences = _replace_unique_token(current, old, new)
            if occurrences != 1:
                record["gate"] = "old_not_unique_token_in_current"
            else:
                evidence = _evidence(item, old, new)
                required = _required_confidence(min_confidence, evidence)
                record["evidence"] = evidence.as_dict()
                record["required_confidence"] = required

                if kind == "segment" and not (evidence.new_votes or evidence.shape_preserving):
                    record["gate"] = "segmentation_without_visual_or_shape_support"
                elif required is None:
                    record["gate"] = "unsupported_new"
                elif confidence < required:
                    record["gate"] = "insufficient_confidence_for_evidence"
                else:
                    corrected, occurrences = _replace_unique_token(current, old, new)
                    if occurrences != 1:
                        record["gate"] = "old_not_unique_token_in_current"
                    else:
                        current = corrected
                        record["gate"] = (
                            "segmentation_evidence"
                            if kind == "segment" and evidence.new_votes
                            else "segmentation_shape"
                            if kind == "segment"
                            else "ocr_evidence"
                            if evidence.new_votes
                            else "shape_preserving"
                        )
                        record["applied"] = True

        audited.append(record)

    return current, audited
