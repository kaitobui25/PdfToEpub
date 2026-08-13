"""Tesseract integration and OCR evidence definitions."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import pytesseract
from pytesseract import Output

from ..models import OCRCandidate, OCRLine
from . import preprocess


@dataclass(frozen=True, slots=True)
class WholePass:
    name: str
    language: str
    psm: int
    scale: float
    transform: str


# Fast evidence set used for every logical side.
WHOLE_PASSES: tuple[WholePass, ...] = (
    WholePass("base_ve_25", "vie+eng", 4, 2.5, "gray"),
    WholePass("sharp_ve_25", "vie+eng", 4, 2.5, "sharp"),
    WholePass("base_v_25", "vie", 4, 2.5, "gray"),
    WholePass("base_v_30", "vie", 4, 3.0, "gray"),
    WholePass("base_v_30_p3", "vie", 3, 3.0, "gray"),
)

# Expensive whole-side rescue set. It runs only after health.py classifies a
# side as catastrophic, so normal pages keep the previous fast path.
FALLBACK_PASSES: tuple[WholePass, ...] = (
    WholePass("fallback_otsu_v_35_p6", "vie", 6, 3.5, "otsu"),
    WholePass("fallback_adaptive_v_35_p6", "vie", 6, 3.5, "adaptive"),
    WholePass("fallback_sharp_ve_35_p6", "vie+eng", 6, 3.5, "sharp"),
    WholePass("fallback_base_v_40_p11", "vie", 11, 4.0, "gray"),
)

LINE_PASSES: tuple[WholePass, ...] = (
    WholePass("line_v_p7", "vie", 7, 4.0, "gray"),
    WholePass("line_ve_p7", "vie+eng", 7, 4.0, "gray"),
    WholePass("line_v_p6", "vie", 6, 4.0, "gray"),
    WholePass("line_v_p13", "vie", 13, 4.0, "gray"),
    WholePass("line_ve_p13", "vie+eng", 13, 4.0, "gray"),
)


def configure_tesseract(explicit: str | None = None) -> str:
    candidates = [
        explicit,
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return str(candidate)
    raise FileNotFoundError("Tesseract not found. Install it or set TESSERACT_CMD.")


def require_languages() -> None:
    langs = set(pytesseract.get_languages(config=""))
    missing = {"vie", "eng"} - langs
    if missing:
        raise RuntimeError(f"Tesseract is missing languages: {', '.join(sorted(missing))}")


def prepare_image(image: np.ndarray, spec: WholePass) -> np.ndarray:
    """Apply one OCR pass's visual transform and scale in one canonical place."""

    transform = {
        "gray": preprocess.gray,
        "sharp": preprocess.sharpen,
        "otsu": preprocess.otsu,
        "adaptive": preprocess.adaptive,
    }[spec.transform]
    return preprocess.resize(transform(image), spec.scale)


def _clean_conf(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _line_rows(data: dict[str, list[object]]) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    grouped: dict[tuple[int, int, int], list[int]] = {}
    count = len(data.get("text", []))
    for i in range(count):
        text = str(data["text"][i]).strip()
        if not text:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        grouped.setdefault(key, []).append(i)

    rows: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for indexes in grouped.values():
        words = [str(data["text"][i]).strip() for i in indexes]
        text = " ".join(word for word in words if word)
        confs = [_clean_conf(data["conf"][i]) for i in indexes]
        confs = [c for c in confs if c >= 0]
        left = min(int(data["left"][i]) for i in indexes)
        top = min(int(data["top"][i]) for i in indexes)
        right = max(int(data["left"][i]) + int(data["width"][i]) for i in indexes)
        bottom = max(int(data["top"][i]) + int(data["height"][i]) for i in indexes)
        rows.append((text, sum(confs) / len(confs) if confs else 0.0, (left, top, right - left, bottom - top)))
    rows.sort(key=lambda row: (row[2][1], row[2][0]))
    return rows


def ocr_whole_pass(image: np.ndarray, spec: WholePass) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    transformed = prepare_image(image, spec)
    config = f"--oem 1 --psm {spec.psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(transformed, lang=spec.language, config=config, output_type=Output.DICT)
    rows = _line_rows(data)
    return [
        (text, conf, (int(x / spec.scale), int(y / spec.scale), int(w / spec.scale), int(h / spec.scale)))
        for text, conf, (x, y, w, h) in rows
    ]


def ocr_line_crop(image: np.ndarray, spec: WholePass) -> OCRCandidate:
    transformed = prepare_image(image, spec)
    config = f"--oem 1 --psm {spec.psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(transformed, lang=spec.language, config=config, output_type=Output.DICT)
    words = [str(t).strip() for t in data["text"] if str(t).strip()]
    confs = [_clean_conf(c) for c, t in zip(data["conf"], data["text"]) if str(t).strip()]
    confs = [c for c in confs if c >= 0]
    return OCRCandidate(
        source=spec.name,
        kind="line",
        scale=spec.scale,
        psm=spec.psm,
        text=" ".join(words).strip(),
        confidence=sum(confs) / len(confs) if confs else 0.0,
    )


def crop_line(image: np.ndarray, line: OCRLine) -> np.ndarray:
    height, width = image.shape[:2]
    pad_x = max(25, int(width * 0.06))
    pad_y = max(22, int(height * 0.025))
    x0 = max(0, line.x - pad_x)
    y0 = max(0, line.y - pad_y)
    x1 = min(width, line.x + line.w + pad_x)
    y1 = min(height, line.y + line.h + pad_y)
    return image[y0:y1, x0:x1]


def average_confidence(candidates: Iterable[OCRCandidate]) -> float:
    values = [c.confidence for c in candidates if c.text]
    return sum(values) / len(values) if values else 0.0
