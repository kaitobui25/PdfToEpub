"""Render PDF spreads and split them into logical left/right book pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np


@dataclass(frozen=True, slots=True)
class SideImage:
    page_number: int
    side: str
    image: np.ndarray

    @property
    def tag(self) -> str:
        return f"{self.page_number:03d}-{self.side}"


def render_pdf_page(doc: fitz.Document, zero_based_index: int, scale: float = 2.0) -> np.ndarray:
    """Render one PDF page to BGR pixels.

    The books used by this project contain two photographed/scanned book pages in
    one PDF page. Geometry is therefore handled before OCR.
    """

    page = doc.load_page(zero_based_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _trim_outer_whitespace(image: np.ndarray) -> np.ndarray:
    """Conservatively trim nearly-white outer margins without touching text."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < 247
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image

    h, w = gray.shape
    pad_x = max(8, int(w * 0.015))
    pad_y = max(8, int(h * 0.012))
    x0 = max(0, int(xs.min()) - pad_x)
    x1 = min(w, int(xs.max()) + pad_x + 1)
    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(h, int(ys.max()) + pad_y + 1)
    return image[y0:y1, x0:x1]


def split_spread(image: np.ndarray, page_number: int, gutter_ratio: float = 0.012) -> list[SideImage]:
    """Split one spread at the center with a tiny gutter exclusion.

    This intentionally avoids clever content-based split detection. The current
    tested corpus is stable, and a deterministic midpoint split is easier to
    reason about and less likely to create cross-page bleed.
    """

    _, width = image.shape[:2]
    mid = width // 2
    gutter = max(4, int(width * gutter_ratio))
    left = image[:, : max(1, mid - gutter)]
    right = image[:, min(width - 1, mid + gutter) :]
    return [
        SideImage(page_number, "L", _trim_outer_whitespace(left)),
        SideImage(page_number, "R", _trim_outer_whitespace(right)),
    ]


def extract_sides(pdf: Path, start_page: int, end_page: int, work_dir: Path, scale: float) -> list[SideImage]:
    """Render and persist the requested PDF range as side images."""

    work_dir.mkdir(parents=True, exist_ok=True)
    sides: list[SideImage] = []
    with fitz.open(pdf) as doc:
        if start_page < 1 or end_page > len(doc) or start_page > end_page:
            raise ValueError(f"Invalid PDF range {start_page}..{end_page}; document has {len(doc)} pages")
        for page_number in range(start_page, end_page + 1):
            spread = render_pdf_page(doc, page_number - 1, scale=scale)
            for side in split_spread(spread, page_number):
                path = work_dir / f"{side.tag}_body.png"
                cv2.imwrite(str(path), side.image)
                sides.append(side)
    return sides
