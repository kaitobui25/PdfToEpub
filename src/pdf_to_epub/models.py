"""Shared data contracts between PDF, OCR, cleanup, EPUB and Deep stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OCRCandidate:
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
    """Logical left/right page plus quality flags owned by whole-side analysis."""

    page_number: int
    side: str
    image_path: Path
    lines: list[OCRLine] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    selected_pass: str = ""
    quality_flags: list[str] = field(default_factory=list)

    @property
    def tag(self) -> str:
        return f"PDF{self.page_number:03d}-{self.side}"


@dataclass(slots=True)
class DeepQueueItem:
    item_id: str
    page_number: int
    side: str
    current: str
    output_line: str
    context: str
    reasons: list[str]
    candidates: list[str]
    # Full OCR source metadata is preserved for the local evidence gate. The
    # simple text list above remains for backwards compatibility and compact
    # prompts/old serialized queues.
    candidate_meta: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = {
            "id": self.item_id,
            "page": [self.page_number, self.side],
            "current": self.current,
            "output_line": self.output_line,
            "context": self.context,
            "reasons": self.reasons,
            "candidates": self.candidates,
        }
        if self.candidate_meta:
            data["candidate_meta"] = self.candidate_meta
        return data
