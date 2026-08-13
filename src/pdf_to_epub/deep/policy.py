"""Post-generation policy for constrained OCR choices.

Candidate generation answers "what could this token be?".  This module answers
"how much authority should that candidate have?" and adds a small class of safe
boundary-segmentation candidates.  Keeping these concerns separate prevents the
candidate generator from accumulating context policy and voting rules.
"""

from __future__ import annotations

import re

from ..models import DeepQueueItem
from ..ocr.scoring import normalize_token
from .candidates import BookStats
from .tokens import index_tokens

ALPHA_DIGIT_RE = re.compile(r"^([A-Za-zÀ-ỹĐđ]{2,})(\d{1,3})$", re.UNICODE)
DIGIT_ALPHA_RE = re.compile(r"^(\d{1,4})([A-Za-zÀ-ỹĐđ]{2,})$", re.UNICODE)


def _non_dictionary_tokens(reasons: list[str]) -> set[str]:
    result: set[str] = set()
    for reason in reasons:
        if not str(reason).startswith("non_dictionary:"):
            continue
        payload = str(reason).split(":", 1)[1]
        for token in re.split(r"[,|]", payload):
            normalized = normalize_token(token)
            if normalized:
                result.add(normalized)
    return result


def _choice_set(item: DeepQueueItem, token_id: str) -> dict[str, object] | None:
    return next((row for row in item.choice_sets if str(row.get("token_id")) == token_id), None)


def _boundary_candidate(token_text: str, stats: BookStats) -> str | None:
    """Return a conservative letter/digit split, never a blind regex rewrite."""

    match = ALPHA_DIGIT_RE.fullmatch(token_text)
    if match:
        word, number = match.groups()
        # A lexical/proven prose word next to a short number is a plausible lost
        # space.  Codes such as A4/B2B/GPT5 are not accepted by this test unless
        # the alphabetic part is independently known in the book/lexicon.
        if stats.lexical_valid(word) or stats.frequency(word) >= 2:
            return f"{word} {number}"

    match = DIGIT_ALPHA_RE.fullmatch(token_text)
    if match:
        number, word = match.groups()
        if stats.lexical_valid(word) or stats.frequency(word) >= 2:
            return f"{number} {word}"
    return None


def _add_boundary_choices(item: DeepQueueItem, stats: BookStats) -> int:
    added = 0
    for token in index_tokens(item.current):
        split = _boundary_candidate(token.text, stats)
        if not split:
            continue
        row = _choice_set(item, token.token_id)
        if row is None:
            row = {
                "token_id": token.token_id,
                "old": token.text,
                "choices": [{
                    "choice_id": "KEEP",
                    "text": token.text,
                    "kind": "keep",
                    "strength": "keep",
                }],
            }
            item.choice_sets.append(row)

        choices = list(row.get("choices") or [])
        if any(str(choice.get("text") or "") == split for choice in choices):
            continue
        choices.append({
            "choice_id": "",
            "text": split,
            "kind": "segment",
            "strength": "medium",
            "visual_score": 0.0,
            "book_frequency": stats.frequency(split),
            "old_book_frequency": stats.frequency(token.text),
            "lexical_valid": True,
            "shape_preserving": True,
            "segmented": True,
            "base_edit_distance": 0,
            "source_tags": ["boundary_split"],
            "boundary_split": True,
            "reverse_verify": True,
        })
        row["choices"] = choices
        added += 1
    return added


def _candidate_sort_key(choice: dict[str, object]) -> tuple[object, ...]:
    strength = str(choice.get("strength") or "weak")
    rank = {"strong": 0, "medium": 1, "weak": 2}.get(strength, 3)
    reverse_rank = 0 if choice.get("reverse_verify") else 1
    return (
        rank,
        reverse_rank,
        -float(choice.get("visual_score") or 0.0),
        -int(choice.get("book_frequency") or 0),
        str(choice.get("text") or ""),
    )


def finalize_choice_policy(item: DeepQueueItem, stats: BookStats) -> dict[str, int]:
    """Apply old-word safety, reverse-verifier eligibility and boundary splits."""

    explicit_bad = _non_dictionary_tokens(item.reasons)
    boundary_added = _add_boundary_choices(item, stats)
    downgraded = 0
    reverse_marked = 0

    for row in item.choice_sets:
        old = str(row.get("old") or "")
        old_norm = normalize_token(old)
        old_valid = bool(old_norm) and stats.lexical_valid(old) and old_norm not in explicit_bad
        row["old_lexical_valid"] = old_valid

        choices = list(row.get("choices") or [])
        keep = [choice for choice in choices if str(choice.get("choice_id") or "") == "KEEP"]
        non_keep = [choice for choice in choices if str(choice.get("choice_id") or "") != "KEEP"]

        for choice in non_keep:
            choice["old_lexical_valid"] = old_valid
            strength = str(choice.get("strength") or "weak")
            # If OLD is itself a plausible word, pixels alone cannot decide the
            # semantic choice.  This directly prevents "về lâu dài" -> "lâu đài".
            if old_valid and strength == "strong":
                choice["pre_policy_strength"] = "strong"
                choice["strength"] = "medium"
                choice["old_valid_requires_verifier"] = True
                strength = "medium"
                downgraded += 1

            tags = set(choice.get("source_tags") or [])
            visual = float(choice.get("visual_score") or 0.0)
            frequency = int(choice.get("book_frequency") or 0)
            shape = bool(choice.get("shape_preserving"))
            lexical = bool(choice.get("lexical_valid"))
            reverse = (
                strength == "strong"
                or bool(choice.get("old_valid_requires_verifier"))
                or "boundary_split" in tags
                or visual >= 1.60
                or (strength == "medium" and shape and lexical and frequency >= 3)
                or (strength == "medium" and "book_phrase" in tags)
            )
            if reverse and strength != "weak":
                if not choice.get("reverse_verify"):
                    reverse_marked += 1
                choice["reverse_verify"] = True

        non_keep.sort(key=_candidate_sort_key)
        row["choices"] = (keep[:1] or [{
            "choice_id": "KEEP",
            "text": old,
            "kind": "keep",
            "strength": "keep",
        }]) + non_keep
        for index, choice in enumerate(row["choices"][1:], 1):
            choice["choice_id"] = f"C{index}"

    item.choice_sets.sort(key=lambda row: str(row.get("token_id") or ""))
    return {
        "boundary_added": boundary_added,
        "strong_downgraded": downgraded,
        "reverse_marked": reverse_marked,
    }
