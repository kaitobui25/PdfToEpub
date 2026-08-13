"""Evidence-aware gates for Deep OCR corrections."""

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


@dataclass(frozen=True, slots=True)
class TrustPolicy:
    lexical_confidence: float | None
    structural_confidence: float | None
    whole_confidence: float
    extra_changed_words: int
    max_change_ratio: float


def _policy(level: str) -> TrustPolicy:
    level = str(level or "strict").casefold()
    if level == "high":
        return TrustPolicy(0.95, 0.99, 0.95, 6, 0.65)
    if level == "balanced":
        return TrustPolicy(0.98, None, 0.97, 2, 0.50)
    return TrustPolicy(None, None, 0.98, 0, 0.40)


def _normalized_words(text: str) -> list[str]:
    return [normalize_token(part) for part in text.split() if normalize_token(part)]


def _candidate_contains_sequence(candidate: str, value: str) -> bool:
    candidate_words = _normalized_words(candidate)
    wanted = _normalized_words(value)
    if not wanted or len(wanted) > len(candidate_words):
        return False
    width = len(wanted)
    return any(candidate_words[i : i + width] == wanted for i in range(len(candidate_words) - width + 1))


def _evidence(item: DeepQueueItem, old: str, new: str) -> EvidenceProfile:
    return EvidenceProfile(
        old_votes=sum(1 for c in item.candidates if _candidate_contains_sequence(c, old)),
        new_votes=sum(1 for c in item.candidates if _candidate_contains_sequence(c, new)),
        candidate_count=len(item.candidates),
        shape_preserving=bool(shape_key(old)) and shape_key(old) == shape_key(new),
        segmented=len(new.split()) > len(old.split()),
    )


def _required_confidence(base: float, evidence: EvidenceProfile) -> float | None:
    if evidence.new_votes >= 2:
        value = max(0.90, base - 0.05)
        return max(0.90, value - 0.02) if evidence.shape_preserving else value
    if evidence.new_votes == 1:
        value = max(0.92, base - 0.02)
        return max(0.90, value - 0.02) if evidence.shape_preserving else value
    if evidence.shape_preserving:
        return max(base, 0.98)
    return None


def _token_matches(text: str, token: str) -> list[re.Match[str]]:
    return list(re.finditer(rf"(?<!\w){re.escape(token)}(?!\w)", text, re.UNICODE))


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
    return "segment" if any(ch.isspace() for ch in new) else "replace"


def _valid_segment(old: str, new: str) -> bool:
    if any(ch.isspace() for ch in old):
        return False
    parts = new.split()
    return 2 <= len(parts) <= 3 and all(normalize_token(part) for part in parts)


def apply_ai_ops(
    item: DeepQueueItem,
    ai_ops: list[dict[str, Any]],
    min_confidence: float,
    max_ops: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
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
        record: dict[str, Any] = {"kind": kind, "old": old, "new": new, "confidence": confidence, "gate": "", "applied": False}
        if not old or not new:
            record["gate"] = "empty_token"
        elif any(ch.isspace() for ch in old):
            record["gate"] = "old_must_be_one_token"
        elif old == new:
            record["gate"] = "same_value"
        elif kind == "replace" and any(ch.isspace() for ch in new):
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
                        record["applied"] = True
                        record["gate"] = (
                            "segmentation_evidence" if kind == "segment" and evidence.new_votes
                            else "segmentation_shape" if kind == "segment"
                            else "ocr_evidence" if evidence.new_votes
                            else "shape_preserving"
                        )
        audited.append(record)
    return current, audited


def _sentence_tokens(text: str) -> list[str]:
    return SENTENCE_TOKEN_RE.findall(text)


def _sentence_change_records(current: str, proposed: str) -> tuple[list[dict[str, Any]], str | None, int]:
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
        records.append({
            "kind": "sentence_span",
            "old": " ".join(old_parts),
            "new": " ".join(new_parts),
            "old_words": len(old_parts),
            "new_words": len(new_parts),
            "gate": "",
            "applied": False,
        })
    return records, None, changed_words


def _whole_sentence_votes(item: DeepQueueItem, proposed: str) -> int:
    return sum(1 for candidate in item.candidates if _candidate_contains_sequence(candidate, proposed))


def _changed_word_count(current: str, proposed: str) -> int:
    before = _normalized_words(current)
    after = _normalized_words(proposed)
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")


def _small_span(record: dict[str, Any]) -> bool:
    return max(int(record.get("old_words") or 0), int(record.get("new_words") or 0)) <= 3


def apply_ai_sentence(
    item: DeepQueueItem,
    ai_raw: dict[str, Any],
    min_confidence: float,
    max_changed_words: int = 6,
    deep_trust: str = "strict",
) -> tuple[str, list[dict[str, Any]]]:
    policy = _policy(deep_trust)
    proposed_value = ai_raw.get("corrected_sentence") if isinstance(ai_raw, dict) else None
    if proposed_value is None:
        return item.current, [{"kind": "sentence", "old": item.current, "new": "", "confidence": 0.0, "gate": "missing_corrected_sentence", "applied": False}]

    proposed = str(proposed_value).strip()
    current = item.current.strip()
    try:
        confidence = float(ai_raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if proposed == current:
        return item.current, []
    if not proposed:
        return item.current, [{"kind": "sentence", "old": current, "new": proposed, "confidence": confidence, "gate": "empty_sentence", "applied": False}]

    changes, structural_error, changed_words = _sentence_change_records(current, proposed)
    total_words = len(_normalized_words(current))
    effective_max = max_changed_words + policy.extra_changed_words

    if structural_error:
        whole_votes = _whole_sentence_votes(item, proposed)
        whole_changed = _changed_word_count(current, proposed)
        whole_required = max(policy.whole_confidence, min_confidence - 0.02)
        max_whole = max(effective_max + 2, math.ceil(total_words * 0.75))
        if whole_votes and confidence >= whole_required and whole_changed <= max_whole:
            return proposed, [{
                "kind": "sentence", "old": current, "new": proposed, "confidence": confidence,
                "changed_words": whole_changed,
                "evidence": {"whole_sentence_votes": whole_votes, "candidate_count": len(item.candidates)},
                "required_confidence": whole_required, "gate": "sentence_whole_ocr_candidate", "applied": True,
            }]
        structural_limit = max(3, math.ceil(total_words * 0.20))
        if (
            structural_error == "sentence_insert_delete_or_reorder"
            and policy.structural_confidence is not None
            and confidence >= policy.structural_confidence
            and whole_changed <= structural_limit
        ):
            return proposed, [{
                "kind": "sentence", "old": current, "new": proposed, "confidence": confidence,
                "changed_words": whole_changed, "required_confidence": policy.structural_confidence,
                "gate": "sentence_deep_trust_structural", "applied": True,
            }]
        return item.current, [{
            "kind": "sentence", "old": current, "new": proposed, "confidence": confidence,
            "changed_words": whole_changed,
            "evidence": {"whole_sentence_votes": whole_votes, "candidate_count": len(item.candidates)},
            "required_confidence": whole_required if whole_votes else policy.structural_confidence,
            "gate": structural_error, "applied": False,
        }]

    if not changes:
        return item.current, []

    if changed_words > effective_max or (total_words >= 10 and changed_words / max(1, total_words) > policy.max_change_ratio):
        whole_votes = _whole_sentence_votes(item, proposed)
        whole_required = max(policy.whole_confidence, min_confidence - 0.02)
        max_whole = max(effective_max + 2, math.ceil(total_words * 0.75))
        if whole_votes and confidence >= whole_required and changed_words <= max_whole:
            return proposed, [{
                "kind": "sentence", "old": current, "new": proposed, "confidence": confidence,
                "changed_words": changed_words,
                "evidence": {"whole_sentence_votes": whole_votes, "candidate_count": len(item.candidates)},
                "required_confidence": whole_required, "gate": "sentence_whole_ocr_candidate", "applied": True,
            }]
        return item.current, [{
            "kind": "sentence", "old": current, "new": proposed, "confidence": confidence,
            "changed_words": changed_words, "gate": "sentence_rewrite_too_large", "applied": False,
        }]

    all_pass = True
    for record in changes:
        evidence = _evidence(item, str(record["old"]), str(record["new"]))
        required = _required_confidence(min_confidence, evidence)
        override = policy.lexical_confidence is not None and confidence >= policy.lexical_confidence and _small_span(record)
        record["confidence"] = confidence
        record["evidence"] = evidence.as_dict()
        record["required_confidence"] = required
        if required is None:
            if override:
                record["gate"] = "sentence_deep_trust_override"
                record["required_confidence"] = policy.lexical_confidence
            else:
                record["gate"] = "unsupported_sentence_span"
                all_pass = False
        elif confidence < required:
            if override:
                record["gate"] = "sentence_deep_trust_override"
                record["required_confidence"] = policy.lexical_confidence
            else:
                record["gate"] = "insufficient_sentence_confidence"
                all_pass = False
        else:
            record["gate"] = "sentence_ocr_evidence" if evidence.new_votes else "sentence_shape_preserving"

    if not all_pass:
        for record in changes:
            if record["gate"] in {"sentence_ocr_evidence", "sentence_shape_preserving", "sentence_deep_trust_override"}:
                record["gate"] = "atomic_rejected_due_to_other_span"
        return item.current, changes

    for record in changes:
        record["applied"] = True
    return proposed, changes
