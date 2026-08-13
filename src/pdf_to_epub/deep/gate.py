"""Evidence-aware Deep OCR gate with stable token addressing.

The model proposes a correction. Local code decides one of three outcomes:
AUTO (apply now), VERIFY (send to a closed-choice verifier later), or REJECT.
AI self-reported confidence is never enough by itself for a non-shape rewrite.
"""

from __future__ import annotations

from typing import Any

from ..models import DeepQueueItem
from ..ocr.scoring import normalize_token, strip_diacritics
from .evidence import EvidenceProfile, summarize_evidence
from .tokens import TokenRef, index_tokens


def _operation_kind(raw: dict[str, Any], new: str) -> str:
    declared = str(raw.get("kind") or "").strip().casefold()
    if declared in {"replace", "segment"}:
        return declared
    return "segment" if any(char.isspace() for char in new) else "replace"


def _valid_segment(old: str, new: str) -> bool:
    if any(char.isspace() for char in old):
        return False
    parts = new.split()
    return 2 <= len(parts) <= 3 and all(normalize_token(part) for part in parts)


def _has_diacritic(text: str) -> bool:
    return strip_diacritics(text) != text


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    left = normalize_token(left)
    right = normalize_token(right)
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    if len(left) > len(right):
        left, right = right, left
    i = j = mismatches = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
        else:
            mismatches += 1
            j += 1
            if mismatches > 1:
                return False
    return True


def _resolve_token(text: str, old: str, token_id: str | None) -> tuple[TokenRef | None, str | None]:
    tokens = index_tokens(text)
    if token_id:
        token = next((candidate for candidate in tokens if candidate.token_id == token_id), None)
        if token is None:
            return None, "unknown_token_id"
        if token.text != old:
            return None, "token_id_old_mismatch"
        return token, None

    matches = [token for token in tokens if token.text == old]
    if len(matches) != 1:
        return None, "old_not_unique_token_in_current"
    return matches[0], None


def _legacy_required_confidence(evidence: EvidenceProfile) -> float | None:
    if evidence.new_score >= 2.0:
        return 0.92
    if evidence.new_score >= 1.0:
        return 0.95
    if evidence.shape_preserving:
        return 0.98
    return None


def _decision(
    kind: str,
    old: str,
    new: str,
    confidence: float,
    evidence: EvidenceProfile,
) -> tuple[str, str, float | None]:
    """Return (decision, gate_label, compatibility_threshold)."""

    if evidence.legacy:
        required = _legacy_required_confidence(evidence)
        if kind == "segment":
            if evidence.new_score >= 1.0 and confidence >= 0.95:
                return "auto", "segmentation_evidence", 0.95
            if evidence.shape_preserving and confidence >= 0.98:
                return "auto", "segmentation_shape", 0.98
            if not (evidence.new_score or evidence.shape_preserving):
                return "reject", "segmentation_without_visual_or_shape_support", required
            return "verify", "needs_verifier_segmentation", required

        if required is None:
            return "reject", "unsupported_new", None
        if confidence >= required:
            return "auto", "ocr_evidence" if evidence.new_score else "shape_preserving", required
        return "verify", "needs_verifier_legacy", required

    if kind == "segment":
        if evidence.new_score >= 0.75 and confidence >= 0.85:
            return "auto", "segmentation_evidence", 0.85
        if (
            evidence.shape_preserving
            and evidence.old_non_dictionary
            and _has_diacritic(new)
            and confidence >= 0.90
        ):
            return "auto", "segmentation_shape", 0.90
        if evidence.shape_preserving and confidence >= 0.97:
            return "auto", "segmentation_shape", 0.97
        if evidence.new_score or evidence.shape_preserving:
            return "verify", "needs_verifier_segmentation", None
        return "reject", "segmentation_without_visual_or_shape_support", None

    # Multi-family visual agreement is the strongest automatic path.
    if evidence.new_score >= 1.60 and confidence >= 0.88:
        return "auto", "ocr_evidence", 0.88

    # Diacritic/shape-preserving spelling seen by one useful family.
    if evidence.shape_preserving and evidence.new_score >= 0.55 and confidence >= 0.90:
        return "auto", "ocr_evidence", 0.90

    # A non-dictionary OLD plus one strong independent visual witness can rescue
    # a one-glyph OCR confusion such as uui -> vui.
    if evidence.old_non_dictionary and evidence.new_score >= 0.80 and confidence >= 0.88:
        return "auto", "ocr_evidence", 0.88

    # Retain the old very-high-confidence shape fallback, but lower-confidence
    # context-dominant cases become VERIFY rather than blind accepts/rejects.
    if evidence.shape_preserving and evidence.new_score == 0 and confidence >= 0.98:
        return "auto", "shape_preserving", 0.98

    if evidence.shape_preserving and confidence >= 0.75:
        return "verify", "needs_verifier_shape", None

    if evidence.old_non_dictionary and _edit_distance_at_most_one(old, new) and confidence >= 0.80:
        return "verify", "needs_verifier_lexical", None

    if evidence.new_score > 0 and confidence >= 0.75:
        return "verify", "needs_verifier_visual", None

    return "reject", "unsupported_new", None


def apply_ai_ops(
    item: DeepQueueItem,
    ai_ops: list[dict[str, Any]],
    min_confidence: float,
    max_ops: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    """Evaluate and apply AUTO operations.

    VERIFY operations are recorded but intentionally left untouched. A later
    closed-choice verifier can handle them without weakening this gate.
    """

    del min_confidence  # retained in the public signature for CLI compatibility
    audited: list[dict[str, Any]] = []
    approved: list[tuple[int, int, str]] = []
    targeted_ids: set[str] = set()

    for raw in ai_ops[:max_ops]:
        old = str(raw.get("old") or "")
        new = str(raw.get("new") or "")
        token_id_raw = str(raw.get("token_id") or "").strip()
        token_id = token_id_raw or None
        kind = _operation_kind(raw, new)
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        record: dict[str, Any] = {
            "kind": kind,
            "token_id": token_id,
            "old": old,
            "new": new,
            "confidence": confidence,
            "decision": "reject",
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
            token, token_error = _resolve_token(item.current, old, token_id)
            if token_error:
                record["gate"] = token_error
            elif token is None:
                record["gate"] = "token_resolution_failed"
            elif token.token_id in targeted_ids:
                record["gate"] = "duplicate_token_target"
            else:
                evidence = summarize_evidence(item, old, new)
                decision, label, required = _decision(kind, old, new, confidence, evidence)
                record["token_id"] = token.token_id
                record["evidence"] = evidence.as_dict()
                record["required_confidence"] = required
                record["decision"] = decision
                record["gate"] = label

                if decision == "auto":
                    targeted_ids.add(token.token_id)
                    approved.append((token.start, token.end, new))
                    record["applied"] = True

        audited.append(record)

    current = item.current
    # Apply from right to left so original token spans stay stable when a
    # segmentation inserts spaces or another correction changes token length.
    for start, end, new in sorted(approved, key=lambda value: value[0], reverse=True):
        current = current[:start] + new + current[end:]

    return current, audited
