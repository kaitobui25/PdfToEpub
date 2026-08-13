"""Whole-side OCR health detection and fallback decision policy.

A language model is good at repairing isolated OCR tokens; it is the wrong tool
for a page whose OCR has collapsed into symbol soup.  This module detects that
failure mode before line-level refinement and asks the local OCR layer to retry
the entire logical page with a different set of visual assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import combinations
from statistics import median
from typing import Iterable

from .scoring import WORD_RE, garbage_ratio, normalize_token, strip_diacritics


Rows = list[tuple[str, float, tuple[int, int, int, int]]]


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    min_chars: int = 80
    min_words: int = 12
    max_garbage_ratio: float = 0.055
    max_short_word_ratio: float = 0.48
    min_dictionary_ratio: float = 0.28
    min_cross_pass_similarity: float = 0.42
    min_average_confidence: float = 50.0


@dataclass(frozen=True, slots=True)
class PassHealth:
    source: str
    score: float
    chars: int
    words: int
    average_confidence: float
    garbage_ratio: float
    short_word_ratio: float
    dictionary_ratio: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SideHealthReport:
    catastrophic: bool
    best_source: str | None
    cross_pass_similarity: float
    reasons: tuple[str, ...]
    passes: tuple[PassHealth, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catastrophic": self.catastrophic,
            "best_source": self.best_source,
            "cross_pass_similarity": round(self.cross_pass_similarity, 4),
            "reasons": list(self.reasons),
            "passes": [item.as_dict() for item in self.passes],
        }


def _normalized_text(rows: Rows) -> str:
    text = " ".join(row[0] for row in rows if row[0].strip())
    return " ".join(strip_diacritics(text).casefold().split())


def _dictionary_ratio(words: list[str], lexicon: set[str]) -> float | None:
    if not lexicon:
        return None
    eligible = [normalize_token(word) for word in words if len(normalize_token(word)) >= 3 and not any(ch.isdigit() for ch in word)]
    if not eligible:
        return None
    hits = sum(1 for word in eligible if word in lexicon)
    return hits / len(eligible)


def _pass_health(source: str, rows: Rows, lexicon: set[str]) -> PassHealth:
    text = "\n".join(row[0] for row in rows if row[0].strip())
    words = WORD_RE.findall(text)
    confidences = [row[1] for row in rows if row[0].strip() and row[1] >= 0]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    short_word_ratio = sum(1 for word in words if len(normalize_token(word)) <= 2) / max(1, len(words))
    dictionary_ratio = _dictionary_ratio(words, lexicon)
    glyph_garbage = garbage_ratio(text)
    letter_ratio = sum(1 for char in text if char.isalpha()) / max(1, sum(1 for char in text if not char.isspace()))

    # Ranking is intentionally continuous; catastrophe classification below uses
    # independent thresholds so changing the preferred pass cannot silently
    # change the definition of a failed page.
    score = average_confidence * 0.55 + letter_ratio * 25.0 - glyph_garbage * 120.0 - short_word_ratio * 12.0
    if dictionary_ratio is not None:
        score += dictionary_ratio * 20.0
    score += min(len(text), 1600) / 1600 * 5.0

    return PassHealth(
        source=source,
        score=round(score, 4),
        chars=len(text.strip()),
        words=len(words),
        average_confidence=round(average_confidence, 4),
        garbage_ratio=round(glyph_garbage, 5),
        short_word_ratio=round(short_word_ratio, 5),
        dictionary_ratio=round(dictionary_ratio, 5) if dictionary_ratio is not None else None,
    )


def _cross_pass_similarity(evidence: dict[str, Rows]) -> float:
    texts = [_normalized_text(rows) for rows in evidence.values() if _normalized_text(rows)]
    if len(texts) < 2:
        return 1.0
    values = [SequenceMatcher(None, left, right).ratio() for left, right in combinations(texts, 2)]
    return median(values) if values else 1.0


def analyze_side_evidence(
    evidence: dict[str, Rows],
    lexicon: set[str],
    thresholds: HealthThresholds = HealthThresholds(),
) -> SideHealthReport:
    """Classify one logical page from all available whole-side OCR passes."""

    if not evidence:
        return SideHealthReport(True, None, 0.0, ("no_ocr_evidence",), ())

    metrics = tuple(_pass_health(source, rows, lexicon) for source, rows in evidence.items())
    best = max(metrics, key=lambda item: item.score)
    similarity = _cross_pass_similarity(evidence)

    # Tiny/blank sides are not automatically catastrophic. The cleanup layer is
    # allowed to drop them as blank/ornament pages without spending fallback OCR.
    substantial = best.chars >= thresholds.min_chars and best.words >= thresholds.min_words
    reasons: list[str] = []
    if substantial and best.garbage_ratio > thresholds.max_garbage_ratio:
        reasons.append("symbol_soup")
    if substantial and best.short_word_ratio > thresholds.max_short_word_ratio:
        reasons.append("fragmented_words")
    if substantial and best.dictionary_ratio is not None and best.dictionary_ratio < thresholds.min_dictionary_ratio:
        reasons.append("lexicon_collapse")
    if substantial and similarity < thresholds.min_cross_pass_similarity:
        reasons.append("cross_pass_instability")
    if substantial and best.average_confidence < thresholds.min_average_confidence:
        reasons.append("low_side_confidence")

    reason_set = set(reasons)
    catastrophic = substantial and (
        ("symbol_soup" in reason_set and bool(reason_set & {"lexicon_collapse", "cross_pass_instability", "fragmented_words"}))
        or ("lexicon_collapse" in reason_set and "cross_pass_instability" in reason_set)
        or ("low_side_confidence" in reason_set and "cross_pass_instability" in reason_set)
        or len(reason_set & {"symbol_soup", "fragmented_words", "lexicon_collapse", "cross_pass_instability", "low_side_confidence"}) >= 3
    )

    return SideHealthReport(
        catastrophic=catastrophic,
        best_source=best.source,
        cross_pass_similarity=similarity,
        reasons=tuple(reasons),
        passes=metrics,
    )
