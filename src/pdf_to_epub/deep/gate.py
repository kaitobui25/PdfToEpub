"""Safety gate for DeepSeek token corrections.

The validator is deliberately conservative. DeepSeek may *suggest* a repair,
but this module owns the final decision about whether that suggestion is safe
to write into the book text.

This file preserves the pre-fix V4 baseline behaviour. Keeping all acceptance
policy here makes later gate tuning isolated from OCR, OpenCode transport and
EPUB generation.
"""

from __future__ import annotations

from typing import Any

from ..models import DeepQueueItem
from ..ocr.scoring import diacritic_only, normalize_token


def _candidate_contains_token(candidate: str, token: str) -> bool:
    """Return whether *candidate* contains *token* as an OCR word.

    OCR lines commonly attach punctuation/quotes to a word (``Wallace,``,
    ``“bị”``, ``nhé!``). Comparing raw ``str.split()`` tokens would miss that
    visual evidence, so both sides are normalized before comparison.
    """

    wanted = normalize_token(token)
    if not wanted:
        return False
    return any(normalize_token(part) == wanted for part in candidate.split())


def _ocr_votes(item: DeepQueueItem, old: str, new: str) -> tuple[int, int]:
    """Count alternate line OCR candidates that support OLD and NEW."""

    old_votes = sum(1 for candidate in item.candidates if _candidate_contains_token(candidate, old))
    new_votes = sum(1 for candidate in item.candidates if _candidate_contains_token(candidate, new))
    return old_votes, new_votes


def _evidence_strongly_prefers_current(old: str, old_votes: int, new_votes: int) -> bool:
    """Protect text when repeated OCR evidence overwhelmingly supports OLD.

    Four agreeing candidate passes versus at most one NEW vote is considered a
    hard visual veto. A one-character OLD token gets the same protection at
    three votes because accidental insertion/deletion around tiny OCR fragments
    is especially risky.
    """

    if new_votes > 1:
        return False
    old_core = normalize_token(old)
    return old_votes >= 4 or (len(old_core) == 1 and old_votes >= 3)


def _candidate_change_too_large(old: str, new: str) -> bool:
    """Reject unsupported-looking one-glyph substitutions.

    A candidate pass alone is not enough evidence to turn one unrelated glyph
    into another (for example a digit into a Vietnamese letter). Diacritic-only
    changes are exempt because they preserve the same base glyph.
    """

    old_core = normalize_token(old)
    new_core = normalize_token(new)
    return (
        len(old_core) == 1
        and len(new_core) == 1
        and not diacritic_only(old_core, new_core)
        and old_core.casefold() != new_core.casefold()
    )


def _weak_diacritic_guess(
    *, confidence: float, old_votes: int, candidate_count: int, new_votes: int
) -> bool:
    """Identify the baseline's weakest unsupported accent-only suggestions.

    At the minimum accepted AI confidence (0.97), a correction with no visual
    NEW vote is withheld when at least half of the available OCR candidates
    still contain OLD. This intentionally keeps the old conservative policy;
    later quality work can tune this rule without touching other stages.
    """

    if confidence > 0.97 or new_votes:
        return False
    return candidate_count > 0 and old_votes * 2 >= candidate_count


def apply_ai_ops(
    item: DeepQueueItem,
    ai_ops: list[dict[str, Any]],
    min_confidence: float,
    max_ops: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply exact, one-token, high-confidence OCR repairs.

    Every model suggestion receives a stable ``gate`` label in the returned
    audit. The order of checks is intentional: hard safety constraints run
    before visual support and language-level diacritic fallbacks.
    """

    current = item.current
    audited: list[dict[str, Any]] = []

    for raw in ai_ops[:max_ops]:
        old = str(raw.get("old") or "")
        new = str(raw.get("new") or "")
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        record: dict[str, Any] = {
            "old": old,
            "new": new,
            "confidence": confidence,
            "gate": "",
            "applied": False,
        }

        if not old or not new:
            record["gate"] = "empty_token"
        elif any(ch.isspace() for ch in old) or any(ch.isspace() for ch in new):
            record["gate"] = "multi_token"
        elif old == new:
            record["gate"] = "same_value"
        elif confidence < min_confidence:
            record["gate"] = "low_confidence"
        elif current.count(old) != 1:
            # Exact substring uniqueness prevents replacing the wrong occurrence.
            record["gate"] = "old_not_unique_in_current"
        else:
            old_votes, new_votes = _ocr_votes(item, old, new)
            candidate_supported = new_votes > 0
            accent_only = diacritic_only(old, new)

            if _evidence_strongly_prefers_current(old, old_votes, new_votes):
                record["gate"] = "ocr_evidence_strongly_prefers_current"
            elif candidate_supported and _candidate_change_too_large(old, new):
                record["gate"] = "candidate_change_too_large"
            elif candidate_supported:
                record["gate"] = "candidate"
                record["applied"] = True
            elif accent_only and _weak_diacritic_guess(
                confidence=confidence,
                old_votes=old_votes,
                candidate_count=len(item.candidates),
                new_votes=new_votes,
            ):
                record["gate"] = "diacritic_guess_without_visual_or_phrase_gain"
            elif accent_only:
                record["gate"] = "diacritic_only"
                record["applied"] = True
            else:
                # The strict model prompt normally avoids reaching this branch,
                # but retaining it makes malformed/unsupported suggestions safe.
                record["gate"] = "unsupported_new"

        if record["applied"]:
            current = current.replace(old, new, 1)
        audited.append(record)

    return current, audited
