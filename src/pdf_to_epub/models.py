"""Shared immutable-ish data structures used across pipeline stages.

Keeping stage contracts in one module prevents the OCR, cleanup, EPUB and AI
layers from depending on each other's implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OCRCandidate:
    """One OCR interpretation of the same visual line."""

    source: str
    text: str
    confidence: float
    kind: str = "whole"
    scale: float | None = None
    psm: int | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source": self.source,
            "kind": self.kind,
            "text": self.text,
            "conf": self.confidence,
        }
        if self.scale is not None:
            data["scale"] = self.scale
        if self.psm is not None:
            data["psm"] = self.psm
        return data


@dataclass(slots=True)
class OCRLine:
    """A selected OCR line plus geometry and evidence from alternate passes."""

    line_id: str
    page_number: int
    side: str
    text: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    whole_candidates: list[OCRCandidate] = field(default_factory=list)
    line_candidates: list[OCRCandidate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    edits: list[dict[str, Any]] = field(default_factory=list)

    @property
    def page_key(self) -> tuple[int, str]:
        return self.page_number, self.side


@dataclass(slots=True)
class BookSide:
    """Logical left/right book page produced from one scanned PDF spread."""

    page_number: int
    side: str
    image_path: Path
    lines: list[OCRLine] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    selected_pass: str = ""

    @property
    def tag(self) -> str:
        return f"PDF{self.page_number:03d}-{self.side}"


@dataclass(slots=True)
class DeepQueueItem:
    """Minimal evidence sent to the Deep-only validator."""

    item_id: str
    page_number: int
    side: str
    current: str
    output_line: str
    context: str
    reasons: list[str]
    candidates: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "page": [self.page_number, self.side],
            "current": self.current,
            "output_line": self.output_line,
            "context": self.context,
            "reasons": self.reasons,
            "candidates": self.candidates,
        }
