"""Whole-side OCR scheduler optimized for CPU-heavy Windows runs.

Tesseract accepts an image-list file, so multiple side images can share one
process launch. The tested baseline uses batches of two logical sides.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import io
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Callable

import cv2

from ..pdf_layout import SideImage
from .preprocess import gray, resize, sharpen
from .tesseract import LINE_PASSES, WHOLE_PASSES, WholePass


@dataclass(slots=True)
class BatchResult:
    source: str
    side_tag: str
    rows: list[tuple[str, float, tuple[int, int, int, int]]]


def _prepare(side: SideImage, spec: WholePass, folder: Path) -> Path:
    if spec.transform == "sharp":
        image = sharpen(side.image)
    else:
        image = gray(side.image)
    image = resize(image, spec.scale)
    path = folder / f"{side.tag}_{spec.name}.png"
    cv2.imwrite(str(path), image)
    return path


def _parse_tsv(tsv: str, count: int, scale: float) -> list[list[tuple[str, float, tuple[int, int, int, int]]]]:
    pages: list[dict[tuple[int, int, int], list[dict[str, str]]]] = [dict() for _ in range(count)]
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            page_idx = max(0, int(row.get("page_num") or "1") - 1)
            if page_idx >= count:
                continue
            key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        except (ValueError, KeyError):
            continue
        pages[page_idx].setdefault(key, []).append(row)

    output: list[list[tuple[str, float, tuple[int, int, int, int]]]] = []
    for groups in pages:
        result: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for words in groups.values():
            text = " ".join((w.get("text") or "").strip() for w in words if (w.get("text") or "").strip())
            confs: list[float] = []
            boxes: list[tuple[int, int, int, int]] = []
            for w in words:
                try:
                    c = float(w.get("conf") or -1)
                    if c >= 0:
                        confs.append(c)
                    boxes.append((int(w["left"]), int(w["top"]), int(w["width"]), int(w["height"])))
                except (ValueError, KeyError):
                    continue
            if not text or not boxes:
                continue
            x0 = min(b[0] for b in boxes)
            y0 = min(b[1] for b in boxes)
            x1 = max(b[0] + b[2] for b in boxes)
            y1 = max(b[1] + b[3] for b in boxes)
            result.append((
                text,
                sum(confs) / len(confs) if confs else 0.0,
                (int(x0 / scale), int(y0 / scale), int((x1 - x0) / scale), int((y1 - y0) / scale)),
            ))
        result.sort(key=lambda row: (row[2][1], row[2][0]))
        output.append(result)
    return output


def _run_batch(
    tesseract_cmd: str,
    spec: WholePass,
    sides: list[SideImage],
    tmp_root: Path,
    omp_thread_limit: int,
) -> list[BatchResult]:
    batch_dir = Path(tempfile.mkdtemp(prefix=f"{spec.name}_", dir=tmp_root))
    paths = [_prepare(side, spec, batch_dir) for side in sides]
    list_file = batch_dir / "images.txt"
    # Tesseract's list-file parser is happiest with absolute paths and forward slashes.
    list_file.write_text("\n".join(path.resolve().as_posix() for path in paths), encoding="utf-8")

    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = str(omp_thread_limit)
    proc = subprocess.run(
        [
            tesseract_cmd,
            str(list_file),
            "stdout",
            "-l",
            spec.language,
            "--oem",
            "1",
            "--psm",
            str(spec.psm),
            "-c",
            "preserve_interword_spaces=1",
            "tsv",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Tesseract {spec.name} failed: {proc.stderr.strip()}")
    parsed = _parse_tsv(proc.stdout, len(sides), spec.scale)
    return [BatchResult(spec.name, side.tag, parsed[i]) for i, side in enumerate(sides)]


def run_whole_side_evidence(
    tesseract_cmd: str,
    sides: list[SideImage],
    work_dir: Path,
    batch_sides: int,
    workers: int,
    omp_thread_limit: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, dict[str, list[tuple[str, float, tuple[int, int, int, int]]]]], int]:
    """Run all five evidence passes and return results grouped by side/source."""

    tmp_root = work_dir / "ocr_batches"
    tmp_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[WholePass, list[SideImage]]] = []
    for spec in WHOLE_PASSES:
        for i in range(0, len(sides), batch_sides):
            jobs.append((spec, sides[i : i + batch_sides]))

    grouped: dict[str, dict[str, list[tuple[str, float, tuple[int, int, int, int]]]]] = {
        side.tag: {} for side in sides
    }
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_batch, tesseract_cmd, spec, batch, tmp_root, omp_thread_limit)
            for spec, batch in jobs
        ]
        for future in as_completed(futures):
            for result in future.result():
                grouped[result.side_tag][result.source] = result.rows
            completed += 1
            if progress:
                progress(completed, len(jobs))
    # Future completion order is nondeterministic. Rebuild each inner mapping in
    # pass-definition order so audits/prompts are reproducible across machines.
    ordered_grouped = {
        side.tag: {spec.name: grouped[side.tag][spec.name] for spec in WHOLE_PASSES if spec.name in grouped[side.tag]}
        for side in sides
    }
    return ordered_grouped, len(jobs)


def _run_line_batch(
    tesseract_cmd: str,
    spec: WholePass,
    items: list[tuple[str, object]],
    tmp_root: Path,
    omp_thread_limit: int,
) -> dict[str, tuple[str, float]]:
    """OCR many independent line crops with one Tesseract image-list process."""

    from .tesseract import crop_line
    from ..models import OCRLine

    batch_dir = Path(tempfile.mkdtemp(prefix=f"lines_{spec.name}_", dir=tmp_root))
    paths: list[Path] = []
    ids: list[str] = []
    for item_id, payload in items:
        side_image, line = payload  # type: ignore[misc]
        assert isinstance(line, OCRLine)
        crop = crop_line(side_image, line)
        if spec.transform == "sharp":
            processed = sharpen(crop)
        else:
            processed = gray(crop)
        processed = resize(processed, spec.scale)
        path = batch_dir / f"{item_id.replace('/', '_')}.png"
        cv2.imwrite(str(path), processed)
        paths.append(path)
        ids.append(item_id)

    list_file = batch_dir / "images.txt"
    list_file.write_text("\n".join(path.resolve().as_posix() for path in paths), encoding="utf-8")
    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = str(omp_thread_limit)
    proc = subprocess.run(
        [
            tesseract_cmd,
            str(list_file),
            "stdout",
            "-l",
            spec.language,
            "--oem",
            "1",
            "--psm",
            str(spec.psm),
            "-c",
            "preserve_interword_spaces=1",
            "tsv",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Tesseract line pass {spec.name} failed: {proc.stderr.strip()}")

    # Parse words by page number. Geometry is irrelevant for line crops.
    page_words: list[list[tuple[str, float]]] = [[] for _ in ids]
    reader = csv.DictReader(io.StringIO(proc.stdout), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            page_idx = max(0, int(row.get("page_num") or "1") - 1)
            conf = float(row.get("conf") or -1)
        except ValueError:
            continue
        if 0 <= page_idx < len(page_words):
            page_words[page_idx].append((text, conf))

    result: dict[str, tuple[str, float]] = {}
    for item_id, words in zip(ids, page_words):
        text = " ".join(word for word, _ in words).strip()
        confs = [conf for _, conf in words if conf >= 0]
        result[item_id] = (text, sum(confs) / len(confs) if confs else 0.0)
    return result


def run_line_evidence(
    tesseract_cmd: str,
    suspect_items: list[tuple[str, object]],
    work_dir: Path,
    workers: int,
    omp_thread_limit: int,
    batch_lines: int = 28,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, list[object]], int]:
    """Run five reOCR passes over suspicious lines in ~28-line process batches.

    With 499 suspicious lines this schedules 18 batches/pass × 5 passes = 90
    Tesseract calls, matching the observed FIX3 test run.
    """

    from ..models import OCRCandidate

    tmp_root = work_dir / "line_batches"
    tmp_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[WholePass, list[tuple[str, object]]]] = []
    for spec in LINE_PASSES:
        for i in range(0, len(suspect_items), batch_lines):
            jobs.append((spec, suspect_items[i : i + batch_lines]))

    evidence: dict[str, list[OCRCandidate]] = {item_id: [] for item_id, _ in suspect_items}
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_line_batch, tesseract_cmd, spec, batch, tmp_root, omp_thread_limit): spec
            for spec, batch in jobs
        }
        for future in as_completed(futures):
            spec = futures[future]
            for item_id, (text, conf) in future.result().items():
                evidence[item_id].append(
                    OCRCandidate(
                        source=spec.name,
                        kind="line",
                        scale=spec.scale,
                        psm=spec.psm,
                        text=text,
                        confidence=conf,
                    )
                )
            completed += 1
            if progress:
                progress(completed, len(jobs))

    # Preserve the canonical pass order even though batches finish concurrently.
    pass_order = {spec.name: index for index, spec in enumerate(LINE_PASSES)}
    for candidates in evidence.values():
        candidates.sort(key=lambda candidate: pass_order.get(candidate.source, len(pass_order)))
    return evidence, len(jobs)
