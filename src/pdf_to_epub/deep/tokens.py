"""Stable token addressing for Deep OCR operations."""

from __future__ import annotations

from dataclasses import dataclass
import re

TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TokenRef:
    token_id: str
    text: str
    start: int
    end: int

    def as_dict(self) -> dict[str, object]:
        return {"id": self.token_id, "text": self.text}


def index_tokens(text: str) -> list[TokenRef]:
    """Tokenize a line into stable ordinal IDs and original character spans."""

    return [
        TokenRef(f"t{index:02d}", match.group(0), match.start(), match.end())
        for index, match in enumerate(TOKEN_RE.finditer(text), 1)
    ]
