"""Central configuration for the tested V4 LOCAL_TURBO baseline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_MODEL = "opencode/deepseek-v4-flash-free"


@dataclass(frozen=True, slots=True)
class LocalTurboConfig:
    """Settings used by the local-only OCR stage.

    The defaults mirror the tested 61..100 run: five whole-side evidence passes,
    two sides batched into each Tesseract process and up to 22 host workers.
    """

    start_page: int = 61
    end_page: int = 100
    workers: int = min(22, max(1, os.cpu_count() or 8))
    batch_sides: int = 2
    render_scale: float = 2.0
    line_scale: float = 4.0
    omp_thread_limit: int = 1


@dataclass(frozen=True, slots=True)
class DeepConfig:
    """Settings for the standalone DeepSeek validation pass."""

    model: str = DEFAULT_MODEL
    batch_size: int = 6
    workers: int = 4
    min_apply_confidence: float = 0.97
    call_timeout_seconds: int = 120
    max_ops_per_item: int = 3


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Names all runtime artifacts from one local OCR range."""

    root: Path
    stem: str

    @property
    def local_txt(self) -> Path:
        return self.root / f"{self.stem}_V4_LOCAL_TURBO.txt"

    @property
    def local_epub(self) -> Path:
        return self.root / f"{self.stem}_V4_LOCAL_TURBO.epub"

    @property
    def deep_txt(self) -> Path:
        return self.root / f"{self.stem}_V4_LOCAL_TURBO_DEEP.txt"

    @property
    def deep_epub(self) -> Path:
        return self.root / f"{self.stem}_V4_LOCAL_TURBO_DEEP.epub"

    @property
    def work_dir(self) -> Path:
        return self.root / "_work"

    @property
    def deep_work_dir(self) -> Path:
        return self.root / "_deep_work"
