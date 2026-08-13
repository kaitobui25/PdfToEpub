"""LOCAL_TURBO orchestration.

This module coordinates stages but owns no OCR policy. Fast OCR, whole-side
health/fallback, line refinement, cleanup and EPUB serialization remain isolated
so each quality fix can be tested and tuned independently.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
from typing import Any, Callable

from .cleanup import cleanup_sides
from .config import LocalTurboConfig, OutputLayout
from .epub import write_epub
from .jsonio import write_json, write_text
from .models import BookSide, OCRLine
from .ocr.batch import run_fallback_side_evidence, run_line_evidence, run_whole_side_evidence
from .ocr.health import SideHealthReport, analyze_side_evidence
from .ocr.lexicon import load_vietnamese_words
from .ocr.local_refine import align_whole_candidates, refine_line, suspect_reasons
from .ocr.scoring import side_score
from .ocr.tesseract import FALLBACK_PASSES, WHOLE_PASSES, configure_tesseract, require_languages
from .pdf_layout import SideImage, extract_sides


def make_layout(pdf: Path, start_page: int, end_page: int, output: Path | None = None) -> OutputLayout:
    root = output or pdf.parent / f"{pdf.stem}_v4_local_turbo_{start_page}_{end_page}"
    stem = f"{pdf.stem}_PDF_{start_page}_{end_page}"
    return OutputLayout(root=root, stem=stem)


def _serialize_text(sides: list[BookSide]) -> str:
    parts: list[str] = []
    for side in sides:
        body = "\n\n".join(paragraph.strip() for paragraph in side.paragraphs if paragraph.strip())
        parts.append(f"===== {side.tag} =====\n\n{body}".rstrip())
    return "\n\n\n".join(parts).rstrip() + "\n"


def _make_anchor_lines(
    page_number: int,
    side_name: str,
    rows: list[tuple[str, float, tuple[int, int, int, int]]],
) -> list[OCRLine]:
    lines: list[OCRLine] = []
    for index, (text, conf, (x, y, w, h)) in enumerate(rows, 1):
        if not text.strip():
            continue
        lines.append(
            OCRLine(
                line_id=f"{page_number:03d}-{side_name}-L{index:03d}",
                page_number=page_number,
                side=side_name,
                text=text.strip(),
                confidence=conf,
                x=x,
                y=y,
                w=w,
                h=h,
            )
        )
    return lines


def _health_pass(
    side_images: list[SideImage],
    evidence: dict[str, dict[str, list[tuple[str, float, tuple[int, int, int, int]]]]],
    lexicon: set[str],
) -> dict[str, SideHealthReport]:
    return {side.tag: analyze_side_evidence(evidence.get(side.tag, {}), lexicon) for side in side_images}


def run_local_turbo(
    pdf: Path,
    layout: OutputLayout,
    config: LocalTurboConfig,
    log: Callable[[str], None],
    tesseract_cmd: str | None = None,
) -> dict[str, Any]:
    """Execute local OCR with targeted whole-side rescue before line refinement."""

    total_start = time.perf_counter()
    pdf = pdf.resolve()
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    layout.root.mkdir(parents=True, exist_ok=True)

    tesseract = configure_tesseract(tesseract_cmd)
    require_languages()
    log("=== V4 LOCAL_TURBO — LOCAL OCR + TARGETED SIDE RESCUE ===")
    log(f"PDF: {pdf}")
    log(f"Range: {config.start_page}..{config.end_page}")
    log(f"Workers: {config.workers}")
    log(f"Tesseract: {tesseract}")
    log("OCR: 5 fast whole-side passes; catastrophic sides get 4 fallback passes; suspicious healthy lines get 5x reOCR")

    lexicon = load_vietnamese_words(layout.work_dir)
    log(f"[LEXICON] {len(lexicon)} Vietnamese words from Tesseract DAWG")

    split_start = time.perf_counter()
    side_images = extract_sides(pdf, config.start_page, config.end_page, layout.work_dir / "sides", config.render_scale)
    split_seconds = time.perf_counter() - split_start
    log(f"[EXTRACT/SPLIT] {len(side_images)} sides in {split_seconds:.2f}s")

    ocr_start = time.perf_counter()
    last_reported = 0

    def whole_progress(done: int, total: int) -> None:
        nonlocal last_reported
        if done == total or done - last_reported >= 10:
            log(f"[OCR BATCH] {done}/{total} Tesseract processes complete")
            last_reported = done

    evidence, whole_processes = run_whole_side_evidence(
        tesseract,
        side_images,
        layout.work_dir,
        config.batch_sides,
        config.workers,
        config.omp_thread_limit,
        whole_progress,
    )
    initial_health = _health_pass(side_images, evidence, lexicon)
    fallback_sides = [side for side in side_images if initial_health[side.tag].catastrophic]

    fallback_calls = 0
    fallback_seconds = 0.0
    if fallback_sides:
        fallback_start = time.perf_counter()
        fallback_evidence, fallback_calls = run_fallback_side_evidence(
            tesseract,
            fallback_sides,
            layout.work_dir,
            config.workers,
            config.omp_thread_limit,
        )
        fallback_seconds = time.perf_counter() - fallback_start
        for side_tag, by_source in fallback_evidence.items():
            evidence.setdefault(side_tag, {}).update(by_source)
        log(f"[SIDE FALLBACK] {len(fallback_sides)} catastrophic sides, {fallback_calls} extra Tesseract calls, wall={fallback_seconds:.2f}s")
    else:
        log("[SIDE FALLBACK] 0 catastrophic sides")

    final_health = _health_pass(side_images, evidence, lexicon)
    rescued_sides = sum(1 for side in fallback_sides if not final_health[side.tag].catastrophic)
    unresolved_sides = [side for side in fallback_sides if final_health[side.tag].catastrophic]
    if fallback_sides:
        log(f"[SIDE HEALTH] rescued={rescued_sides} unresolved={len(unresolved_sides)}")

    ocr_wall = time.perf_counter() - ocr_start
    logical_passes = len(side_images) * len(WHOLE_PASSES) + len(fallback_sides) * len(FALLBACK_PASSES)
    log(
        f"[OCR BATCH] logical_passes={logical_passes} processes={whole_processes + fallback_calls} "
        f"base_batch_sides={config.batch_sides} wall={ocr_wall:.2f}s"
    )

    pass_specs = (*WHOLE_PASSES, *FALLBACK_PASSES)
    pass_scales = {spec.name: spec.scale for spec in pass_specs}
    image_by_tag = {side.tag: side.image for side in side_images}
    book_sides: list[BookSide] = []
    chosen_counts: Counter[str] = Counter()

    for side_image in side_images:
        by_source = evidence.get(side_image.tag, {})
        if not by_source:
            continue
        health = final_health[side_image.tag]
        chosen_source = health.best_source or max(by_source, key=lambda source: side_score(by_source[source]))
        if chosen_source not in by_source:
            chosen_source = max(by_source, key=lambda source: side_score(by_source[source]))
        chosen_counts[chosen_source] += 1
        lines = _make_anchor_lines(side_image.page_number, side_image.side, by_source[chosen_source])
        align_whole_candidates(lines, by_source, pass_scales)
        quality_flags = ["whole_side_catastrophe"] if health.catastrophic else []
        book_sides.append(
            BookSide(
                page_number=side_image.page_number,
                side=side_image.side,
                image_path=layout.work_dir / "sides" / f"{side_image.tag}_body.png",
                lines=lines,
                selected_pass=chosen_source,
                quality_flags=quality_flags,
            )
        )

    # Unresolved catastrophic sides are deliberately NOT line-reOCR'd. Their
    # layout/text evidence is already globally unreliable, so crop-level work can
    # reinforce bad geometry. They stay visible in review/audit for manual action.
    suspect_items: list[tuple[str, object]] = []
    line_by_id: dict[str, OCRLine] = {}
    for side in book_sides:
        image = image_by_tag[f"{side.page_number:03d}-{side.side}"]
        catastrophic = "whole_side_catastrophe" in side.quality_flags
        for line in side.lines:
            line.reasons = suspect_reasons(line, lexicon)
            if catastrophic and "whole_side_catastrophe" not in line.reasons:
                line.reasons.append("whole_side_catastrophe")
            if line.reasons and not catastrophic:
                suspect_items.append((line.line_id, (image, line)))
                line_by_id[line.line_id] = line

    reocr_start = time.perf_counter()
    line_evidence, line_calls = run_line_evidence(
        tesseract,
        suspect_items,
        layout.work_dir,
        config.workers,
        config.omp_thread_limit,
        batch_lines=28,
    )
    for line_id, candidates in line_evidence.items():
        refine_line(line_by_id[line_id], candidates, lexicon)
    reocr_wall = time.perf_counter() - reocr_start
    log(f"[reOCR] {len(suspect_items)} suspect healthy-side lines, {line_calls} batched Tesseract calls, wall={reocr_wall:.2f}s")

    combine_start = time.perf_counter()
    health_audit: list[dict[str, Any]] = []
    fallback_tags = {side.tag for side in fallback_sides}
    for side in side_images:
        before = initial_health[side.tag]
        after = final_health[side.tag]
        health_audit.append({
            "page": [side.page_number, side.side],
            "fallback_used": side.tag in fallback_tags,
            "rescued": before.catastrophic and not after.catastrophic,
            "before": before.as_dict(),
            "after": after.as_dict(),
        })

    local_refine_audit: list[dict[str, Any]] = []
    local_review: list[dict[str, Any]] = []
    for side in book_sides:
        if "whole_side_catastrophe" in side.quality_flags:
            local_review.append({
                "page": [side.page_number, side.side],
                "action": "review_whole_side",
                "reason": "whole_side_catastrophe_after_fallback",
                "selected_pass": side.selected_pass,
            })
        for line in side.lines:
            if line.reasons:
                local_refine_audit.append({
                    "id": line.line_id,
                    "page": [line.page_number, line.side],
                    "reasons": line.reasons,
                    "before": next((candidate.text for candidate in line.whole_candidates if candidate.source == side.selected_pass), line.text),
                    "after": line.text,
                    "edits": line.edits,
                    "whole_candidates": [candidate.as_dict() for candidate in line.whole_candidates],
                    "line_candidates": [candidate.as_dict() for candidate in line.line_candidates],
                })
            if line.confidence < 82 or (line.reasons and not line.edits):
                alternatives = sorted(
                    [*line.whole_candidates, *line.line_candidates],
                    key=lambda candidate: candidate.confidence,
                    reverse=True,
                )[:4]
                local_review.append({
                    "page": [line.page_number, line.side],
                    "action": "review",
                    "anchor": line.text,
                    "chosen": line.text,
                    "conf": round(line.confidence, 2),
                    "alts": [{"text": candidate.text, "conf": round(candidate.confidence, 1)} for candidate in alternatives],
                })

    cleaned_sides, cleanup_audit, repeated_headers = cleanup_sides(book_sides)
    text = _serialize_text(cleaned_sides)
    write_text(layout.local_txt, text)
    write_json(layout.root / "whole_side_health.json", health_audit)
    write_json(layout.root / "local_refine_audit.json", local_refine_audit)
    write_json(layout.root / "local_review.json", local_review)
    write_json(layout.root / "cleanup_audit.json", cleanup_audit)
    combine_seconds = time.perf_counter() - combine_start

    epub_start = time.perf_counter()
    write_epub(layout.local_epub, f"{pdf.stem} — V4 LOCAL TURBO — PDF {config.start_page}-{config.end_page}", cleaned_sides)
    epub_seconds = time.perf_counter() - epub_start

    total_seconds = time.perf_counter() - total_start
    summary = {
        "pdf": str(pdf),
        "range": [config.start_page, config.end_page],
        "pdf_pages": config.end_page - config.start_page + 1,
        "book_sides_input": len(side_images),
        "book_sides_output": len(cleaned_sides),
        "workers": config.workers,
        "batch_sides": config.batch_sides,
        "tesseract_calls": whole_processes + fallback_calls + line_calls,
        "whole_side_logical_passes": logical_passes,
        "whole_side_tesseract_processes": whole_processes + fallback_calls,
        "whole_side_tesseract_calls": whole_processes + fallback_calls,
        "fallback_sides_triggered": len(fallback_sides),
        "fallback_sides_rescued": rescued_sides,
        "fallback_sides_unresolved": len(unresolved_sides),
        "fallback_whole_side_calls": fallback_calls,
        "fallback_wall_seconds": round(fallback_seconds, 3),
        "line_reocr_calls": line_calls,
        "extract_split_seconds": round(split_seconds, 3),
        "ocr_wall_seconds": round(ocr_wall, 3),
        "line_reocr_wall_seconds": round(reocr_wall, 3),
        "combine_cleanup_seconds": round(combine_seconds, 3),
        "epub_write_validate_seconds": round(epub_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "chosen_pass_counts": dict(chosen_counts),
        "review_flags": len(local_review),
        "local_refine_items": len(local_refine_audit),
        "local_edits": sum(len(row["edits"]) for row in local_refine_audit),
        "cleanup_drops": len(cleanup_audit),
        "repeated_headers": repeated_headers,
        "lexicon_words": len(lexicon),
        "whole_side_health": str(layout.root / "whole_side_health.json"),
        "epub": str(layout.local_epub),
        "txt": str(layout.local_txt),
        "review": str(layout.root / "local_review.json"),
        "local_refine_audit": str(layout.root / "local_refine_audit.json"),
    }
    write_json(layout.root / "SUMMARY_V4_LOCAL_TURBO.json", summary)
    log("=== DONE ===")
    log(str(summary))
    return summary
