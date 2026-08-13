"""OCR quality helpers shared by local refinement, side health and Deep gate."""

from __future__ import annotations

from collections import Counter
import re
import unicodedata
from typing import Iterable

from ..models import OCRCandidate


WORD_RE = re.compile(r"[\wÀ-ỹĐđ]+", re.UNICODE)
GARBAGE_RE = re.compile(r"[^\w\sÀ-ỹĐđ.,;:!?…'\"“”‘’()\-–—/+%&]", re.UNICODE)


def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")


def normalize_token(token: str) -> str:
    return token.strip(".,;:!?…'\"“”‘’()[]{}<>+-–—/*").casefold()


def shape_key(text: str) -> str:
    """Return OCR glyph shape while ignoring spaces, punctuation and diacritics.

    This is useful for safe word segmentation: ``Vidu`` and ``Ví dụ`` share the
    same shape key even though one OCR token should become two language tokens.
    """

    words = WORD_RE.findall(strip_diacritics(text).casefold())
    return "".join(words)


def garbage_score(text: str) -> float:
    """Return a small penalty for symbol soup and obvious OCR corruption."""

    if not text:
        return 100.0
    weird = len(GARBAGE_RE.findall(text))
    compact_runs = len(re.findall(r"\S{24,}", text))
    control = sum(1 for ch in text if unicodedata.category(ch).startswith("C") and ch not in "\n\t")
    return weird * 2.0 + compact_runs * 6.0 + control * 10.0


def garbage_ratio(text: str) -> float:
    """Normalized unsupported-glyph ratio used by whole-side health checks."""

    visible = sum(1 for char in text if not char.isspace())
    return len(GARBAGE_RE.findall(text)) / max(1, visible)


def side_score(lines: Iterable[tuple[str, float, tuple[int, int, int, int]]]) -> float:
    rows = list(lines)
    if not rows:
        return -10_000.0
    text = "\n".join(row[0] for row in rows)
    confs = [row[1] for row in rows if row[0].strip()]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    chars = len(text.strip())
    length_bonus = min(chars, 1800) / 1800 * 5.0
    return avg_conf + length_bonus - garbage_score(text)


def diacritic_only(old: str, new: str) -> bool:
    return old != new and strip_diacritics(old).casefold() == strip_diacritics(new).casefold()


def candidate_votes(current: str, candidates: Iterable[OCRCandidate]) -> Counter[str]:
    """Count alternate pass token spellings aligned by diacritic-insensitive shape."""

    current_tokens = current.split()
    votes: Counter[str] = Counter()
    for candidate in candidates:
        cand_tokens = candidate.text.split()
        if len(cand_tokens) != len(current_tokens):
            continue
        for old, new in zip(current_tokens, cand_tokens):
            old_core = normalize_token(old)
            new_core = normalize_token(new)
            if old_core and new_core and strip_diacritics(old_core) == strip_diacritics(new_core) and old != new:
                votes[f"{old}\0{new}"] += 1
    return votes
