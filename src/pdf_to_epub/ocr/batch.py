"""Tesseract image-list schedulers for whole-side and suspicious-line OCR."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import io
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Callable, Sequence

import cv2

from ..pdf_layout import SideImage
from .tesseract import FALLBACK_PASSES, LINE_PASSES, WHOLE_PASSES, WholePass, crop_line, prepare_image


@dataclass(slots=True)
class BatchResult:
    source: str
    side_tag: str
    rows: list[tuple[str, float, tuple[int, int, int, int]]]


def _prepare(side: SideImage, spec: WholePass, folder: Path) -> Path:
    image = prepare_image(side.image, spec)
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
            text = " ".join((word.get("text") or "").strip() for word in words if (word.get("text") or "").strip())
            confs: list[float] = []
            boxes: list[tuple[int, int, int, int]] = []
            for word in words:
                try:
                    conf = float(word.get("conf") or -1)
                    if conf >= 0:
                        confs.append(conf)
                    boxes.append((int(word["left"]), int(word["top"]), int(word["width"]), int(word["height"])))
                except (ValueError, KeyError):
                    continue
            if not text or not boxes:
                continue
            x0 = min(box[0] for box in boxes)
            y0 = min(box[1] for box in boxes)
            x1 = max(box[0] + box[2] for box in boxes)
            y1 = max(box[1] + box[3] for box in boxes)
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
    return [BatchResult(spec.name, side.tag, parsed[index]) for index, side in enumerate(sides)]


def _run_side_passes(
    tesseract_cmd: str,
    sides: list[SideImage],
    work_dir: Path,
    pass_specs: Sequence[WholePass],
    batch_sides: int,
    workers: int,
    omp_thread_limit: int,
    folder_name: str,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, dict[str, list[tuple[str, float, tuple[int, int, int, int]]]]], int]:
    """Generic whole-side scheduler used by both fast and catastrophe passes."""

    tmp_root = work_dir / folder_name
    tmp_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[WholePass, list[SideImage]]] = []
    for spec in pass_specs:
        for index in range(0, len(sides), batch_sides):
            jobs.append((spec, sides[index : index + batch_sides]))

    grouped: dict[str, dict[str, list[tuple[str, float, tuple[int, int, int, int]]]]] = {side.tag: {} for side in sides}
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_batch, tesseract_cmd, spec, batch, tmp_root, omp_thread_limit) for spec, batch in jobs]
        for future in as_completed(futures):
            for result in future.result():
                grouped[result.side_tag][result.source] = result.rows
            completed += 1
            if progress:
                progress(completed, len(jobs))

    order = {spec.name: index for index, spec in enumerate(pass_specs)}
    ordered = {
        side.tag: dict(sorted(grouped[side.tag].items(), key=lambda item: order.get(item[0], len(order))))
        for side in sides
    }
    return ordered, len(jobs)


def run_whole_side_evidence(
    tesseract_cmd: str,
    sides: list[SideImage],
    work_dir: Path,
    batch_sides: int,
    workers: int,
    omp_thread_limit: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, dict[str, list[tuple[str, float, tuple[int, int, int, int]]]]], int]:
    return _run_side_passes(
        tesseract_cmd,
        sides,
        work_dir,
        WHOLE_PASSES,
        batch_sides,
        workers,
        omp_thread_limit,
        "ocr_batches",
        progress,
    )


def run_fallback_side_evidence(
    tesseract_cmd: str,
    sides: list[SideImage],
    work_dir: Path,
    workers: int,
    omp_thread_limit: int,
) -> tuple[dict[str, dict[str, list[tuple[str, float, tuple[int, int, int, int]]]]], int]:
    """Run expensive alternative page layouts only for catastrophic sides."""

    if not sides:
        return {}, 0
    return _run_side_passes(
        tesseract_cmd,
        sides,
        work_dir,
        FALLBACK_PASSES,
        batch_sides=1,
        workers=min(workers, max(1, len(sides) * len(FALLBACK_PASSES))),
        omp_thread_limit=omp_thread_limit,
        folder_name="fallback_side_batches",
    )


def _run_line_batch(
    tesseract_cmd: str,
    spec: WholePass,
    items: list[tuple[str, object]],
    tmp_root: Path,
    omp_thread_limit: int,
) -> dict[str, tuple[str, float]]:
    from ..models import OCRLine

    batch_dir = Path(tempfile.mkdtemp(prefix=f"lines_{spec.name}_", dir=tmp_root))
    paths: list[Path] = []
    ids: list[str] = []
    for item_id, payload in items:
        side_image, line = payload  # type: ignore[misc]
        assert isinstance(line, OCRLine)
        processed = prepare_image(crop_line(side_image, line), spec)
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
    from ..models import OCRCandidate

    if not suspect_items:
        return {}, 0
    tmp_root = work_dir / "line_batches"
    tmp_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[WholePass, list[tuple[str, object]]]] = []
    for spec in LINE_PASSES:
        for index in range(0, len(suspect_items), batch_lines):
            jobs.append((spec, suspect_items[index : index + batch_lines]))

    evidence: dict[str, list[OCRCandidate]] = {item_id: [] for item_id, _ in suspect_items}
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_line_batch, tesseract_cmd, spec, batch, tmp_root, omp_thread_limit): spec for spec, batch in jobs}
        for future in as_completed(futures):
            spec = futures[future]
            for item_id, (text, conf) in future.result().items():
                evidence[item_id].append(
                    OCRCandidate(source=spec.name, kind="line", scale=spec.scale, psm=spec.psm, text=text, confidence=conf)
                )
            completed += 1
            if progress:
                progress(completed, len(jobs))

    pass_order = {spec.name: index for index, spec in enumerate(LINE_PASSES)}
    for candidates in evidence.values():
        candidates.sort(key=lambda candidate: pass_order.get(candidate.source, len(pass_order)))
    return evidence, len(jobs)
