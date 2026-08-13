"""Evidence-aware OCR gate for constrained Deep choices.

New runs use locally generated KEEP/C1/C2/... choices.  Deep may select a choice
but cannot invent replacement text.  Strong local candidates are applied after
one closed-choice vote; medium candidates require a second binary verifier vote.
The older free-form operation path is retained only for backwards-compatible
unit tests and old serialized experiments.
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


def _legacy_decision(
    kind: str,
    old: str,
    new: str,
    confidence: float,
    evidence: EvidenceProfile,
) -> tuple[str, str, float | None]:
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
        if evidence.shape_preserving and evidence.old_non_dictionary and _has_diacritic(new) and confidence >= 0.90:
            return "auto", "segmentation_shape", 0.90
        if evidence.shape_preserving and confidence >= 0.97:
            return "auto", "segmentation_shape", 0.97
        if evidence.new_score or evidence.shape_preserving:
            return "verify", "needs_verifier_segmentation", None
        return "reject", "segmentation_without_visual_or_shape_support", None

    if evidence.new_score >= 1.60 and confidence >= 0.88:
        return "auto", "ocr_evidence", 0.88
    if evidence.shape_preserving and evidence.new_score >= 0.55 and confidence >= 0.90:
        return "auto", "ocr_evidence", 0.90
    if evidence.old_non_dictionary and evidence.new_score >= 0.80 and confidence >= 0.88:
        return "auto", "ocr_evidence", 0.88
    if evidence.shape_preserving and evidence.new_score == 0 and confidence >= 0.98:
        return "auto", "shape_preserving", 0.98
    if evidence.shape_preserving and confidence >= 0.75:
        return "verify", "needs_verifier_shape", None
    if evidence.old_non_dictionary and _edit_distance_at_most_one(old, new) and confidence >= 0.80:
        return "verify", "needs_verifier_lexical", None
    if evidence.new_score > 0 and confidence >= 0.75:
        return "verify", "needs_verifier_visual", None
    return "reject", "unsupported_new", None


def _choice_set(item: DeepQueueItem, token_id: str) -> dict[str, Any] | None:
    return next((row for row in item.choice_sets if str(row.get("token_id")) == token_id), None)


def _choice(row: dict[str, Any], choice_id: str) -> dict[str, Any] | None:
    return next((choice for choice in row.get("choices", []) if str(choice.get("choice_id")) == choice_id), None)


def render_applied_ops(item: DeepQueueItem, audited: list[dict[str, Any]]) -> str:
    """Render all approved edits against original stable token spans."""

    tokens = {token.token_id: token for token in index_tokens(item.current)}
    edits: list[tuple[int, int, str]] = []
    used: set[str] = set()
    for record in audited:
        if not record.get("applied"):
            continue
        token_id = str(record.get("token_id") or "")
        if not token_id or token_id in used:
            continue
        token = tokens.get(token_id)
        if token is None:
            continue
        new = str(record.get("new") or "")
        if not new:
            continue
        used.add(token_id)
        edits.append((token.start, token.end, new))

    current = item.current
    for start, end, new in sorted(edits, key=lambda value: value[0], reverse=True):
        current = current[:start] + new + current[end:]
    return current


def apply_verifier_votes(
    item: DeepQueueItem,
    audited: list[dict[str, Any]],
    votes: dict[str, str],
) -> str:
    """Apply medium candidates only when a second binary verifier says CHANGE."""

    for record in audited:
        if record.get("decision") != "verify":
            continue
        token_id = str(record.get("token_id") or "")
        verdict = str(votes.get(token_id) or "KEEP").strip().upper()
        record["verifier_vote"] = verdict
        if verdict == "CHANGE":
            record["decision"] = "verified"
            record["gate"] = "closed_choice_verified"
            record["applied"] = True
        else:
            record["decision"] = "keep"
            record["gate"] = "closed_choice_verifier_keep"
            record["applied"] = False
    return render_applied_ops(item, audited)


def _apply_choice_selection(
    item: DeepQueueItem,
    raw: dict[str, Any],
    targeted_ids: set[str],
) -> dict[str, Any]:
    token_id = str(raw.get("token_id") or "").strip()
    choice_id = str(raw.get("choice_id") or "").strip()
    record: dict[str, Any] = {
        "kind": "choice",
        "token_id": token_id or None,
        "choice_id": choice_id or None,
        "old": "",
        "new": "",
        "confidence": None,
        "decision": "reject",
        "gate": "",
        "applied": False,
    }

    if not token_id or not choice_id:
        record["gate"] = "missing_choice_address"
        return record
    if token_id in targeted_ids:
        record["gate"] = "duplicate_token_target"
        return record

    choice_set = _choice_set(item, token_id)
    if choice_set is None:
        record["gate"] = "unknown_token_id"
        return record
    selected = _choice(choice_set, choice_id)
    if selected is None:
        record["gate"] = "unknown_choice_id"
        return record

    old = str(choice_set.get("old") or "")
    record["old"] = old
    if choice_id == "KEEP":
        record["new"] = old
        record["decision"] = "keep"
        record["gate"] = "closed_choice_keep"
        targeted_ids.add(token_id)
        return record

    new = str(selected.get("text") or "")
    strength = str(selected.get("strength") or "weak")
    record["new"] = new
    record["kind"] = str(selected.get("kind") or ("segment" if " " in new else "replace"))
    record["choice"] = selected
    record["candidate_strength"] = strength
    targeted_ids.add(token_id)

    if not new or new == old:
        record["gate"] = "invalid_local_choice"
    elif strength == "strong":
        record["decision"] = "auto"
        record["gate"] = "closed_choice_strong"
        record["applied"] = True
    elif strength == "medium":
        record["decision"] = "verify"
        record["gate"] = "closed_choice_needs_second_vote"
    else:
        # Weak choices are shown to Deep for diagnostics/contrast but can never
        # change the book.  In particular, this prevents Vidu -> Vidụ.
        record["decision"] = "reject"
        record["gate"] = "weak_local_candidate"
    return record


def apply_ai_ops(
    item: DeepQueueItem,
    ai_ops: list[dict[str, Any]],
    min_confidence: float,
    max_ops: int = 5,
) -> tuple[str, list[dict[str, Any]]]:
    """Evaluate closed-choice selections, with legacy free-form compatibility."""

    del min_confidence
    audited: list[dict[str, Any]] = []
    targeted_ids: set[str] = set()

    for raw in ai_ops[:max_ops]:
        if "choice_id" in raw:
            audited.append(_apply_choice_selection(item, raw, targeted_ids))
            continue

        # Legacy free-form path retained for existing tests/old experiments.
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
                decision, label, required = _legacy_decision(kind, old, new, confidence, evidence)
                record["token_id"] = token.token_id
                record["evidence"] = evidence.as_dict()
                record["required_confidence"] = required
                record["decision"] = decision
                record["gate"] = label
                targeted_ids.add(token.token_id)
                if decision == "auto":
                    record["applied"] = True
        audited.append(record)

    return render_applied_ops(item, audited), audited
