"""Geometry/text cleanup after OCR and before book serialization."""

from __future__ import annotations

from collections import Counter
import re
import unicodedata

from .models import BookSide, OCRLine


PAGE_NUMBER_RE = re.compile(r"^[\s\-—–_]*[Il|1O0-9]{1,5}[\s\-—–_.]*$")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).casefold()
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9đ\s]", " ", text)
    return " ".join(text.split())


def detect_repeated_headers(sides: list[BookSide]) -> set[str]:
    """Find short text repeated near the top across many logical pages."""

    counts: Counter[str] = Counter()
    originals: dict[str, str] = {}
    for side in sides:
        for line in side.lines[:3]:
            norm = _norm(line.text)
            if 8 <= len(norm) <= 90:
                counts[norm] += 1
                originals.setdefault(norm, line.text)
    threshold = max(3, int(len(sides) * 0.08))
    return {norm for norm, count in counts.items() if count >= threshold}


def cleanup_sides(sides: list[BookSide]) -> tuple[list[BookSide], list[dict[str, object]], list[str]]:
    """Drop obvious page furniture; preserve uncertain text for later review."""

    repeated = detect_repeated_headers(sides)
    audit: list[dict[str, object]] = []
    for side in sides:
        kept: list[OCRLine] = []
        line_count = len(side.lines)
        for index, line in enumerate(side.lines):
            text = line.text.strip()
            reason: str | None = None
            if not text:
                reason = "empty"
            elif PAGE_NUMBER_RE.fullmatch(text) and (index <= 2 or index >= line_count - 3):
                reason = "page_number"
            elif _norm(text) in repeated and index <= 3:
                reason = "repeated_header"
            elif len(text) <= 2 and not text.isalnum():
                reason = "ornament"

            if reason:
                audit.append({
                    "page": [side.page_number, side.side],
                    "action": "drop",
                    "reason": reason,
                    "text": text,
                    "conf": round(line.confidence, 1),
                })
            else:
                kept.append(line)
        side.lines = kept
        side.paragraphs = paragraphize(kept)

    # Entire blank/ornament sides are intentionally omitted from book output.
    output = [side for side in sides if any(p.strip() for p in side.paragraphs)]
    repeated_display = sorted(repeated)
    return output, audit, repeated_display


def paragraphize(lines: list[OCRLine]) -> list[str]:
    """Conservative paragraph reconstruction from vertical geometry.

    Large vertical gaps start a new paragraph. Lines otherwise join with spaces;
    explicit short heading-like lines remain separate when surrounded by gaps.
    """

    if not lines:
        return []
    heights = sorted(max(1, line.h) for line in lines)
    median_h = heights[len(heights) // 2]
    paragraphs: list[list[str]] = [[lines[0].text.strip()]]
    previous = lines[0]
    for line in lines[1:]:
        gap = line.y - (previous.y + previous.h)
        current_text = line.text.strip()
        previous_text = previous.text.strip()
        heading_like = len(previous_text) < 65 and not previous_text.endswith((".", ",", ";", ":"))
        if gap > median_h * 0.95 or heading_like:
            paragraphs.append([current_text])
        else:
            paragraphs[-1].append(current_text)
        previous = line
    return [" ".join(parts).strip() for parts in paragraphs if any(parts)]
