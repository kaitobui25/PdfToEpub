"""Evidence-aware safety gates for DeepSeek OCR corrections.

DeepSeek proposes edits; this module decides whether independent OCR evidence is
strong enough to apply them. The legacy token gate remains for compatibility,
while sentence mode validates every changed phrase first and then applies the
entire sentence atomically: all changes pass, or none of them are written.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import math
import re
from typing import Any

from ..models import DeepQueueItem
from ..ocr.scoring import normalize_token, shape_key


SENTENCE_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+|[^\w\s]", re.UNICODE)
WORD_TOKEN_RE = re.compile(r"^[0-9A-Za-zÀ-ỹĐđ]+$", re.UNICODE)


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
    """Match one token or a short phrase inside an OCR alternative."""

    candidate_words = _normalized_words(candidate)
    wanted = _normalized_words(value)
    if not wanted or len(wanted) > len(candidate_words):
        return False
    width = len(wanted)
    return any(candidate_words[index : index + width] == wanted for index in range(len(candidate_words) - width + 1))


def _evidence(item: DeepQueueItem, old: str, new: str) -> EvidenceProfile:
    old_votes = sum(1 for candidate in item.candidates if _candidate_contains_sequence(candidate, old))
    new_votes = sum(1 for candidate in item.candidates if _candidate_contains_sequence(candidate, new))
    segmented = len(new.split()) > len(old.split())
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
    """Convert independent visual support into a dynamic confidence threshold."""

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
    """Legacy token-local gate retained for regression compatibility."""

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


def _sentence_tokens(text: str) -> list[str]:
    return SENTENCE_TOKEN_RE.findall(text)


def _sentence_change_records(current: str, proposed: str) -> tuple[list[dict[str, Any]], str | None, int]:
    """Describe lexical replacement spans; reject bare insert/delete/reordering/punctuation edits."""

    before = _sentence_tokens(current)
    after = _sentence_tokens(proposed)
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    records: list[dict[str, Any]] = []
    changed_words = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_parts = before[i1:i2]
        new_parts = after[j1:j2]
        if tag in {"insert", "delete"} or not old_parts or not new_parts:
            return records, "sentence_insert_delete_or_reorder", changed_words
        if not all(WORD_TOKEN_RE.fullmatch(token) for token in [*old_parts, *new_parts]):
            return records, "sentence_punctuation_change", changed_words
        changed_words += max(len(old_parts), len(new_parts))
        records.append(
            {
                "kind": "sentence_span",
                "old": " ".join(old_parts),
                "new": " ".join(new_parts),
                "old_words": len(old_parts),
                "new_words": len(new_parts),
                "gate": "",
                "applied": False,
            }
        )
    return records, None, changed_words


def _whole_sentence_votes(item: DeepQueueItem, proposed: str) -> int:
    """Count OCR alternatives that directly contain the complete proposed sentence."""

    return sum(1 for candidate in item.candidates if _candidate_contains_sequence(candidate, proposed))


def _changed_word_count(current: str, proposed: str) -> int:
    before = _normalized_words(current)
    after = _normalized_words(proposed)
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    return sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


def apply_ai_sentence(
    item: DeepQueueItem,
    ai_raw: dict[str, Any],
    min_confidence: float,
    max_changed_words: int = 6,
) -> tuple[str, list[dict[str, Any]]]:
    """Validate a complete proposed sentence and apply it only as one atomic unit.

    Normal corrections are checked span-by-span. A structural sentence change is
    still allowed when Deep is >=0.98 confident AND an independent OCR alternative
    directly contains the complete proposed sentence. This rescues strongly
    supported OCR reconstructions without opening the gate to free-form rewriting.
    """

    proposed_value = ai_raw.get("corrected_sentence") if isinstance(ai_raw, dict) else None
    if proposed_value is None:
        return item.current, [
            {
                "kind": "sentence",
                "old": item.current,
                "new": "",
                "confidence": 0.0,
                "gate": "missing_corrected_sentence",
                "applied": False,
            }
        ]

    proposed = str(proposed_value).strip()
    current = item.current.strip()
    try:
        confidence = float(ai_raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    if proposed == current:
        return item.current, []
    if not proposed:
        return item.current, [
            {
                "kind": "sentence",
                "old": current,
                "new": proposed,
                "confidence": confidence,
                "gate": "empty_sentence",
                "applied": False,
            }
        ]

    changes, structural_error, changed_words = _sentence_change_records(current, proposed)
    total_words = len(_normalized_words(current))
    if structural_error:
        whole_votes = _whole_sentence_votes(item, proposed)
        whole_changed_words = _changed_word_count(current, proposed)
        required = max(0.98, min_confidence)
        max_whole_change = max(max_changed_words + 2, math.ceil(total_words * 0.75))
        if whole_votes and confidence >= required and whole_changed_words <= max_whole_change:
            return proposed, [
                {
                    "kind": "sentence",
                    "old": current,
                    "new": proposed,
                    "confidence": confidence,
                    "changed_words": whole_changed_words,
                    "evidence": {
                        "whole_sentence_votes": whole_votes,
                        "candidate_count": len(item.candidates),
                    },
                    "required_confidence": required,
                    "gate": "sentence_whole_ocr_candidate",
                    "applied": True,
                }
            ]
        return item.current, [
            {
                "kind": "sentence",
                "old": current,
                "new": proposed,
                "confidence": confidence,
                "changed_words": whole_changed_words,
                "evidence": {
                    "whole_sentence_votes": whole_votes,
                    "candidate_count": len(item.candidates),
                },
                "required_confidence": required if whole_votes else None,
                "gate": structural_error,
                "applied": False,
            }
        ]
    if not changes:
        return item.current, []

    if changed_words > max_changed_words or (total_words >= 10 and changed_words / max(1, total_words) > 0.40):
        whole_votes = _whole_sentence_votes(item, proposed)
        required = max(0.98, min_confidence)
        max_whole_change = max(max_changed_words + 2, math.ceil(total_words * 0.75))
        if whole_votes and confidence >= required and changed_words <= max_whole_change:
            return proposed, [
                {
                    "kind": "sentence",
                    "old": current,
                    "new": proposed,
                    "confidence": confidence,
                    "changed_words": changed_words,
                    "evidence": {
                        "whole_sentence_votes": whole_votes,
                        "candidate_count": len(item.candidates),
                    },
                    "required_confidence": required,
                    "gate": "sentence_whole_ocr_candidate",
                    "applied": True,
                }
            ]
        return item.current, [
            {
                "kind": "sentence",
                "old": current,
                "new": proposed,
                "confidence": confidence,
                "changed_words": changed_words,
                "gate": "sentence_rewrite_too_large",
                "applied": False,
            }
        ]

    all_pass = True
    for record in changes:
        old = str(record["old"])
        new = str(record["new"])
        evidence = _evidence(item, old, new)
        required = _required_confidence(min_confidence, evidence)
        record["confidence"] = confidence
        record["evidence"] = evidence.as_dict()
        record["required_confidence"] = required
        if required is None:
            record["gate"] = "unsupported_sentence_span"
            all_pass = False
        elif confidence < required:
            record["gate"] = "insufficient_sentence_confidence"
            all_pass = False
        else:
            record["gate"] = (
                "sentence_ocr_evidence"
                if evidence.new_votes
                else "sentence_shape_preserving"
            )

    if not all_pass:
        for record in changes:
            if record["gate"] in {"sentence_ocr_evidence", "sentence_shape_preserving"}:
                record["gate"] = "atomic_rejected_due_to_other_span"
        return item.current, changes

    for record in changes:
        record["applied"] = True
    return proposed, changes
