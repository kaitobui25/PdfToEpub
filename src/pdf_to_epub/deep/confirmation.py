"""Independent second-pass confirmation for borderline Deep OCR repairs."""

from __future__ import annotations

from typing import Any

SECOND_PASS_MIN_CONFIDENCE = 0.90
SECOND_PASS_MAX_CONFIDENCE = 0.95


def safe_confidence(raw: dict[str, Any] | None) -> float:
    if not isinstance(raw, dict):
        return 0.0
    try:
        return float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def proposed_sentence(raw: dict[str, Any] | None, current: str) -> str:
    if not isinstance(raw, dict):
        return current.strip()
    return str(raw.get("corrected_sentence") or current).strip()


def needs_second_pass(raw: dict[str, Any] | None, current: str) -> bool:
    confidence = safe_confidence(raw)
    return proposed_sentence(raw, current) != current.strip() and SECOND_PASS_MIN_CONFIDENCE <= confidence < SECOND_PASS_MAX_CONFIDENCE


def confirmation_status(first: dict[str, Any], second: dict[str, Any] | None, current: str) -> str:
    if not isinstance(second, dict):
        return "no_response"
    first_sentence = " ".join(proposed_sentence(first, current).split())
    second_sentence = " ".join(proposed_sentence(second, current).split())
    if first_sentence != second_sentence:
        return "disagreed"
    if safe_confidence(second) < SECOND_PASS_MIN_CONFIDENCE:
        return "low_confidence"
    return "confirmed"
