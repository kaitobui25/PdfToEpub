"""Stable token addressing for constrained Deep OCR choices."""

from __future__ import annotations

from dataclasses import dataclass
import re

TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)
# OCR occasionally prefixes a lexical token with a symbol which is clearly not
# punctuation belonging to prose.  Keep the lexical token text stable for the
# model, but include the garbage prefix in the replacement span so `$ẽ -> sẽ`
# removes the bad glyph instead of producing `$sẽ`.
GARBAGE_PREFIX_CHARS = frozenset("$¢£€¥§¤¦")


@dataclass(frozen=True, slots=True)
class TokenRef:
    token_id: str
    text: str
    start: int
    end: int

    def as_dict(self) -> dict[str, object]:
        return {"id": self.token_id, "text": self.text}


def index_tokens(text: str) -> list[TokenRef]:
    """Tokenize a line into stable ordinal IDs and original replacement spans."""

    result: list[TokenRef] = []
    for index, match in enumerate(TOKEN_RE.finditer(text), 1):
        start = match.start()
        if start > 0 and text[start - 1] in GARBAGE_PREFIX_CHARS:
            start -= 1
        result.append(TokenRef(f"t{index:02d}", match.group(0), start, match.end()))
    return result
