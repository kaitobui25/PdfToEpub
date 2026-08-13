"""Layout-preserving smart projection for corrected Deep TARGET sentences."""

from __future__ import annotations

from difflib import SequenceMatcher
import re

from ..models import DeepQueueItem


TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+|[^\w\s]", re.UNICODE)


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _matches(text: str) -> list[re.Match[str]]:
    return list(TOKEN_RE.finditer(text))


def render_smart_target(item: DeepQueueItem, corrected: str) -> str | None:
    """Preserve untouched OCR whitespace while patching changed token spans.

    Repeated words are aligned by ordered token position, not unique-text search.
    Structural insert/delete edits return None so the caller can safely fall back
    to writing the clean Deep TARGET.
    """

    raw_matches = _matches(item.output_line)
    current_matches = _matches(item.current)
    corrected_matches = _matches(corrected)
    raw_tokens = [match.group(0) for match in raw_matches]
    current_tokens = [match.group(0) for match in current_matches]
    corrected_tokens = [match.group(0) for match in corrected_matches]

    if not raw_tokens or raw_tokens != current_tokens:
        return None

    edits: list[tuple[int, int, str]] = []
    matcher = SequenceMatcher(a=current_tokens, b=corrected_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"insert", "delete"} or i1 == i2 or j1 == j2:
            return None
        raw_start = raw_matches[i1].start()
        raw_end = raw_matches[i2 - 1].end()
        corrected_start = corrected_matches[j1].start()
        corrected_end = corrected_matches[j2 - 1].end()
        edits.append((raw_start, raw_end, corrected[corrected_start:corrected_end]))

    if not edits:
        return item.output_line if _collapse_ws(item.output_line) == _collapse_ws(corrected) else None

    rendered = item.output_line
    for start, end, replacement in reversed(edits):
        rendered = rendered[:start] + replacement + rendered[end:]

    return rendered if _collapse_ws(rendered) == _collapse_ws(corrected) else None
