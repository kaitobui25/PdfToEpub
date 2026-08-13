"""Structured OCR evidence scoring for Deep correction decisions.

Votes are grouped by OCR family so several highly correlated Tesseract passes do
not count as independent witnesses. Within one family only the strongest exact
support for OLD/NEW contributes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from ..models import DeepQueueItem
from ..ocr.scoring import normalize_token, shape_key

FAMILY_WEIGHTS = {
    "line_v": 1.00,
    "line_ve": 1.00,
    "whole_v": 0.72,
    "whole_ve": 0.72,
    "legacy": 1.00,
}


@dataclass(frozen=True, slots=True)
class EvidenceProfile:
    old_score: float
    new_score: float
    old_families: tuple[str, ...]
    new_families: tuple[str, ...]
    candidate_count: int
    shape_preserving: bool
    segmented: bool
    old_non_dictionary: bool
    legacy: bool

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["old_score"] = round(self.old_score, 4)
        data["new_score"] = round(self.new_score, 4)
        return data


def _normalized_words(text: str) -> list[str]:
    return [normalize_token(part) for part in text.split() if normalize_token(part)]


def _contains_sequence(candidate: str, value: str) -> bool:
    candidate_words = _normalized_words(candidate)
    wanted = _normalized_words(value)
    if not wanted or len(wanted) > len(candidate_words):
        return False
    width = len(wanted)
    return any(candidate_words[index : index + width] == wanted for index in range(len(candidate_words) - width + 1))


def _family(candidate: dict[str, Any]) -> str:
    kind = str(candidate.get("kind") or "whole").casefold()
    source = str(candidate.get("source") or "").casefold()
    if source.startswith("legacy_"):
        return source
    is_ve = "_ve_" in source or source.startswith("line_ve") or source.startswith("fallback_ve")
    return f"{'line' if kind == 'line' else 'whole'}_{'ve' if is_ve else 'v'}"


def _confidence(candidate: dict[str, Any]) -> float:
    try:
        return max(0.0, min(100.0, float(candidate.get("conf") or candidate.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _metadata(item: DeepQueueItem) -> tuple[list[dict[str, Any]], bool]:
    if item.candidate_meta:
        return item.candidate_meta, False
    # Backward-compatible path for old queues/tests that only contain text.
    return [
        {"source": f"legacy_{index}", "kind": "legacy", "text": text, "conf": 100.0}
        for index, text in enumerate(item.candidates, 1)
    ], True


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


def summarize_evidence(item: DeepQueueItem, old: str, new: str) -> EvidenceProfile:
    candidates, legacy = _metadata(item)
    old_by_family: dict[str, float] = {}
    new_by_family: dict[str, float] = {}

    for candidate in candidates:
        text = str(candidate.get("text") or "")
        family = _family(candidate)
        conf = _confidence(candidate) / 100.0
        if _contains_sequence(text, old):
            old_by_family[family] = max(old_by_family.get(family, 0.0), conf)
        if _contains_sequence(text, new):
            new_by_family[family] = max(new_by_family.get(family, 0.0), conf)

    def score(values: dict[str, float]) -> float:
        total = 0.0
        for family, conf in values.items():
            weight = FAMILY_WEIGHTS.get(family, 1.0 if family.startswith("legacy_") else 0.65)
            total += weight * conf
        return total

    old_norm = normalize_token(old)
    non_dictionary = _non_dictionary_tokens(item.reasons)
    return EvidenceProfile(
        old_score=score(old_by_family),
        new_score=score(new_by_family),
        old_families=tuple(sorted(old_by_family)),
        new_families=tuple(sorted(new_by_family)),
        candidate_count=len(candidates),
        shape_preserving=bool(shape_key(old)) and shape_key(old) == shape_key(new),
        segmented=len(new.split()) > 1,
        old_non_dictionary=old_norm in non_dictionary,
        legacy=legacy,
    )
