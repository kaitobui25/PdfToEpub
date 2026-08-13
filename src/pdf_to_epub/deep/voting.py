"""Edit-level voting helpers for constrained OCR decisions.

The first Deep choice is one vote.  Medium candidates always receive a second
binary vote; conflicting votes get one tie-breaker.  When the first vote is
KEEP but local evidence is unusually strong, a reverse closed-choice pass may
nominate one candidate; that nomination also needs a tie-breaker before change.
"""

from __future__ import annotations

from typing import Any

from ..models import DeepQueueItem


def _choice_set(item: DeepQueueItem, token_id: str) -> dict[str, Any] | None:
    return next((row for row in item.choice_sets if str(row.get("token_id") or "") == token_id), None)


def _choice(row: dict[str, Any], choice_id: str) -> dict[str, Any] | None:
    return next((choice for choice in row.get("choices", []) if str(choice.get("choice_id") or "") == choice_id), None)


def selected_verify_rows(item: DeepQueueItem, ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for op in ops:
        if op.get("decision") != "verify":
            continue
        token_id = str(op.get("token_id") or "")
        rows.append({
            "id": f"{item.item_id}:{token_id}:verify",
            "current": item.current,
            "context": item.context,
            "token_id": token_id,
            "old": op.get("old"),
            "candidate": op.get("new"),
            "candidate_metadata": op.get("choice", {}),
        })
    return rows


def reverse_rows(item: DeepQueueItem, ops: list[dict[str, Any]], max_choices: int = 6) -> list[dict[str, Any]]:
    """Build a second-look choice only for primary KEEP tokens with strong local evidence."""

    rows: list[dict[str, Any]] = []
    for op in ops:
        if op.get("decision") != "keep" or op.get("gate") != "closed_choice_keep":
            continue
        token_id = str(op.get("token_id") or "")
        choice_set = _choice_set(item, token_id)
        if choice_set is None:
            continue
        candidates = [
            choice
            for choice in choice_set.get("choices", [])
            if str(choice.get("choice_id") or "") != "KEEP"
            and choice.get("reverse_verify")
            and str(choice.get("strength") or "weak") != "weak"
        ][:max_choices]
        if not candidates:
            continue
        rows.append({
            "id": f"{item.item_id}:{token_id}:reverse",
            "current": item.current,
            "context": item.context,
            "token_id": token_id,
            "old": choice_set.get("old"),
            "choices": [
                {"choice_id": "KEEP", "text": choice_set.get("old")},
                *[
                    {
                        "choice_id": choice.get("choice_id"),
                        "text": choice.get("text"),
                        "metadata": choice,
                    }
                    for choice in candidates
                ],
            ],
        })
    return rows


def make_reverse_op(item: DeepQueueItem, token_id: str, choice_id: str) -> dict[str, Any] | None:
    row = _choice_set(item, token_id)
    if row is None or choice_id == "KEEP":
        return None
    selected = _choice(row, choice_id)
    if selected is None or str(selected.get("strength") or "weak") == "weak":
        return None
    return {
        "kind": str(selected.get("kind") or "replace"),
        "token_id": token_id,
        "choice_id": choice_id,
        "old": str(row.get("old") or ""),
        "new": str(selected.get("text") or ""),
        "confidence": None,
        "decision": "tie_break",
        "gate": "reverse_choice_conflict",
        "applied": False,
        "choice": selected,
        "candidate_strength": selected.get("strength"),
        "votes": ["KEEP", "CHANGE"],
    }


def tie_row(item: DeepQueueItem, op: dict[str, Any], suffix: str = "tie") -> dict[str, Any]:
    token_id = str(op.get("token_id") or "")
    return {
        "id": f"{item.item_id}:{token_id}:{suffix}",
        "current": item.current,
        "context": item.context,
        "token_id": token_id,
        "old": op.get("old"),
        "candidate": op.get("new"),
        "candidate_metadata": op.get("choice", {}),
    }


def finalize_selected_vote(op: dict[str, Any], second_vote: str, tie_vote: str | None = None) -> bool:
    """Finalize a primary CHANGE candidate from 2/2 or 2/3 edit votes."""

    second = str(second_vote or "KEEP").upper()
    votes = ["CHANGE", second]
    if second == "CHANGE":
        op["votes"] = votes
        op["decision"] = "verified"
        op["gate"] = "edit_vote_2_of_2_change"
        op["applied"] = True
        return True
    if tie_vote is None:
        op["votes"] = votes
        op["decision"] = "tie_break"
        op["gate"] = "edit_vote_conflict"
        op["applied"] = False
        return False

    tie = str(tie_vote or "KEEP").upper()
    votes.append(tie)
    op["votes"] = votes
    change_votes = sum(vote == "CHANGE" for vote in votes)
    if change_votes >= 2:
        op["decision"] = "verified"
        op["gate"] = "edit_vote_2_of_3_change"
        op["applied"] = True
        return True
    op["decision"] = "keep"
    op["gate"] = "edit_vote_2_of_3_keep"
    op["applied"] = False
    return False


def finalize_reverse_vote(op: dict[str, Any], tie_vote: str) -> bool:
    """Primary KEEP + reverse CHANGE requires the third vote to CHANGE."""

    tie = str(tie_vote or "KEEP").upper()
    votes = ["KEEP", "CHANGE", tie]
    op["votes"] = votes
    if tie == "CHANGE":
        op["decision"] = "verified"
        op["gate"] = "reverse_vote_2_of_3_change"
        op["applied"] = True
        return True
    op["decision"] = "keep"
    op["gate"] = "reverse_vote_2_of_3_keep"
    op["applied"] = False
    return False
