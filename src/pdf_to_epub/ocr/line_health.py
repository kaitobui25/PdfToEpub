"""High-precision line health scoring after targeted line reOCR.

Whole-side health catches page collapse.  This module catches the smaller case
where one line remains symbol soup even though the rest of the side is healthy.
It intentionally combines several signals; no single low confidence or unknown
word is enough to suppress a line from Deep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import combinations
from statistics import median
from typing import Any, Iterable

from .scoring import WORD_RE, garbage_ratio, normalize_token, strip_diacritics


@dataclass(frozen=True, slots=True)
class LineHealthReport:
    catastrophic: bool
    reasons: tuple[str, ...]
    chars: int
    words: int
    garbage_ratio: float
    short_word_ratio: float
    dictionary_ratio: float | None
    cross_pass_similarity: float
    median_candidate_confidence: float
    alpha_ratio: float

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def _normalized(text: str) -> str:
    return " ".join(strip_diacritics(text).casefold().split())


def _similarity(texts: list[str]) -> float:
    unique = list(dict.fromkeys(_normalized(text) for text in texts if text.strip()))
    if len(unique) < 2:
        return 1.0
    values = [SequenceMatcher(None, left, right).ratio() for left, right in combinations(unique, 2)]
    return median(values) if values else 1.0


def _dictionary_ratio(words: list[str], lexicon: set[str]) -> float | None:
    if not lexicon:
        return None
    eligible: list[str] = []
    for raw in words:
        token = normalize_token(raw)
        if len(token) < 3 or any(char.isdigit() for char in token):
            continue
        # Proper nouns should not make an otherwise healthy line look broken.
        if raw[:1].isupper():
            continue
        eligible.append(token.casefold())
    if not eligible:
        return None
    return sum(1 for token in eligible if token in lexicon) / len(eligible)


def analyze_line_health(
    current: str,
    candidates: Iterable[dict[str, Any]],
    lexicon: Iterable[str] = (),
) -> LineHealthReport:
    """Classify one line after its whole/line OCR alternatives are available."""

    metadata = list(candidates)
    lex = {normalize_token(word).casefold() for word in lexicon if normalize_token(word)}
    words = WORD_RE.findall(current)
    visible = [char for char in current if not char.isspace()]
    alpha_ratio = sum(1 for char in visible if char.isalpha()) / max(1, len(visible))
    short_ratio = sum(1 for word in words if len(normalize_token(word)) <= 2) / max(1, len(words))
    dictionary = _dictionary_ratio(words, lex)

    candidate_texts = [current]
    confidences: list[float] = []
    for candidate in metadata:
        text = str(candidate.get("text") or "").strip()
        if text:
            candidate_texts.append(text)
        try:
            conf = float(candidate.get("conf") or candidate.get("confidence") or -1.0)
        except (TypeError, ValueError):
            conf = -1.0
        if conf >= 0:
            confidences.append(conf)

    similarity = _similarity(candidate_texts)
    med_conf = median(confidences) if confidences else 0.0
    glyph_garbage = garbage_ratio(current)
    substantial = len(current.strip()) >= 28 and len(words) >= 6

    reasons: list[str] = []
    if substantial and (glyph_garbage > 0.065 or alpha_ratio < 0.58):
        reasons.append("line_symbol_soup")
    if substantial and short_ratio > 0.48:
        reasons.append("line_fragmented_words")
    if substantial and dictionary is not None and dictionary < 0.22:
        reasons.append("line_lexicon_collapse")
    if substantial and similarity < 0.36:
        reasons.append("line_cross_pass_instability")
    if substantial and med_conf < 58.0:
        reasons.append("line_low_candidate_confidence")

    reason_set = set(reasons)
    catastrophic = substantial and (
        (
            "line_symbol_soup" in reason_set
            and bool(reason_set & {"line_fragmented_words", "line_lexicon_collapse", "line_cross_pass_instability"})
        )
        or (
            {"line_fragmented_words", "line_cross_pass_instability"}.issubset(reason_set)
            and bool(reason_set & {"line_lexicon_collapse", "line_low_candidate_confidence"})
        )
        or len(reason_set) >= 4
    )

    return LineHealthReport(
        catastrophic=catastrophic,
        reasons=tuple(reasons),
        chars=len(current.strip()),
        words=len(words),
        garbage_ratio=round(glyph_garbage, 5),
        short_word_ratio=round(short_ratio, 5),
        dictionary_ratio=round(dictionary, 5) if dictionary is not None else None,
        cross_pass_similarity=round(similarity, 5),
        median_candidate_confidence=round(med_conf, 4),
        alpha_ratio=round(alpha_ratio, 5),
    )
