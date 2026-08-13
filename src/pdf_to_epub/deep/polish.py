"""Standalone Deep-only orchestration: sentence queue → parallel AI → gate → patch."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable

from ..config import DeepConfig, OutputLayout
from ..epub import sides_from_text, write_epub
from ..jsonio import write_json, write_text
from ..models import DeepQueueItem
from .client import deepseek_call, find_opencode, run_exe
from .gate import apply_ai_sentence
from .prompt import build_prompt
from .queue import build_queue


LOCK_RETRY_DELAYS = (0.75, 1.5, 3.0, 6.0)


@dataclass(slots=True)
class DeepResult:
    summary: dict[str, Any]
    audit: list[dict[str, Any]]


def _chunks(items: list[DeepQueueItem], size: int) -> list[list[DeepQueueItem]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _patch_exact_lines(text: str, replacements: dict[str, str]) -> tuple[str, int, int]:
    """Patch globally unique exact sentence substrings in the final text."""

    patched = 0
    missed = 0
    output = text
    for old, new in replacements.items():
        if not old or output.count(old) != 1:
            missed += 1
            continue
        output = output.replace(old, new, 1)
        patched += 1
    return output, patched, missed


def _worker_database(root: Path) -> Path:
    worker = re.sub(r"[^A-Za-z0-9_.-]+", "_", threading.current_thread().name)
    return root / f"run_{os.getpid()}_{worker}.db"


def _range_suffix(page_start: int | None, page_end: int | None) -> str:
    if page_start is None and page_end is None:
        return ""
    if page_start is None or page_end is None:
        raise ValueError("Deep page range requires both start and end")
    if page_start > page_end:
        raise ValueError("Deep page range start must be <= end")
    return f"_{page_start:03d}_{page_end:03d}"


def run_deep_only(
    layout: OutputLayout,
    config: DeepConfig,
    log: Callable[[str], None],
    page_start: int | None = None,
    page_end: int | None = None,
) -> DeepResult:
    """Run DeepSeek against existing local artifacts; never touch OCR/Tesseract."""

    start = time.perf_counter()
    if not layout.local_txt.exists():
        raise FileNotFoundError(f"Local TXT not found: {layout.local_txt}")
    local_audit = layout.root / "local_refine_audit.json"
    if not local_audit.exists():
        raise FileNotFoundError(f"Local refinement audit not found: {local_audit}")

    suffix = _range_suffix(page_start, page_end)
    deep_txt = layout.deep_txt if not suffix else layout.root / f"{layout.stem}_V4_LOCAL_TURBO_DEEP{suffix}.txt"
    deep_epub = layout.deep_epub if not suffix else layout.root / f"{layout.stem}_V4_LOCAL_TURBO_DEEP{suffix}.epub"
    queue_path = layout.root / f"deep_ai_queue{suffix}.json"
    audit_path = layout.root / f"deep_ai_audit{suffix}.json"
    summary_path = layout.root / f"SUMMARY_DEEP_ONLY{suffix}.json"

    queue, skipped = build_queue(layout.local_txt, local_audit, page_start=page_start, page_end=page_end)
    write_json(queue_path, [item.as_dict() for item in queue])

    opencode = find_opencode()
    if opencode is None:
        raise FileNotFoundError("OpenCode CLI not found")
    version = run_exe(opencode, ["--version"], timeout=30)
    if version.returncode != 0:
        raise RuntimeError("opencode --version failed")

    source_lines = sum(len(item.source_ids) or 1 for item in queue)
    batches = _chunks(queue, config.batch_size)
    if suffix:
        log(f"Deep page filter: PDF {page_start}..{page_end}; OCR source remains the full local run")
    log(
        f"Sentence queue: {len(queue)} sentences from {source_lines} unresolved source lines; "
        f"skipped_lines={skipped}"
    )
    log(f"Micro-batch: {config.batch_size} sentences; parallel AI workers: {config.workers}")
    log(f"AI calls: {len(batches)} total, max {config.workers} concurrent")

    ai_by_id: dict[str, dict[str, Any]] = {}
    failed_calls = 0
    lock_retries = 0
    call_seconds: list[float] = []
    ai_start = time.perf_counter()
    worker_db_root = layout.deep_work_dir / "opencode_worker_db"
    worker_db_root.mkdir(parents=True, exist_ok=True)

    def call(
        index: int,
        batch: list[DeepQueueItem],
    ) -> tuple[int, list[DeepQueueItem], dict[str, Any], float, int]:
        t0 = time.perf_counter()
        retries = 0
        database_path = _worker_database(worker_db_root)
        while True:
            try:
                result = deepseek_call(
                    opencode,
                    config.model,
                    build_prompt(batch),
                    layout.deep_work_dir,
                    f"deep_micro{suffix}_{index:03d}",
                    config.call_timeout_seconds,
                    database_path=database_path,
                )
                return index, batch, result, time.perf_counter() - t0, retries
            except RuntimeError as exc:
                locked = "database is locked" in str(exc).casefold()
                if not locked or retries >= len(LOCK_RETRY_DELAYS):
                    raise
                delay = LOCK_RETRY_DELAYS[retries]
                retries += 1
                log(
                    f"[AI RETRY {index:03d}] OpenCode DB locked; "
                    f"retry {retries}/{len(LOCK_RETRY_DELAYS)} in {delay:.2f}s"
                )
                time.sleep(delay)

    with ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="deep_ai") as pool:
        futures = [pool.submit(call, i + 1, batch) for i, batch in enumerate(batches)]
        for future in as_completed(futures):
            try:
                index, batch, response, elapsed, retries = future.result()
                lock_retries += retries
                call_seconds.append(elapsed)
                for row in response.get("items", []):
                    if isinstance(row, dict) and row.get("id"):
                        ai_by_id[str(row["id"])] = row
                retry_suffix = f" retries={retries}" if retries else ""
                log(f"[AI {index:03d}/{len(batches):03d}] sentences={len(batch)} time={elapsed:.2f}s{retry_suffix}")
            except Exception as exc:
                failed_calls += 1
                log(f"[AI FAIL] {exc}")

    ai_wall = time.perf_counter() - ai_start
    log(f"[AI WALL] {ai_wall:.2f}s")
    if failed_calls:
        raise RuntimeError(
            f"Deep-only incomplete: {failed_calls}/{len(batches)} AI calls failed. "
            "Refusing to write a partial Deep TXT/EPUB; rerun Deep-only."
        )

    audit: list[dict[str, Any]] = []
    replacements: dict[str, str] = {}
    applied_sentences = 0
    applied_spans = 0
    for item in queue:
        ai_raw = ai_by_id.get(
            item.item_id,
            {"id": item.item_id, "corrected_sentence": item.current, "confidence": 0.0},
        )
        corrected, changes = apply_ai_sentence(
            item,
            ai_raw,
            config.min_apply_confidence,
            max_changed_words=max(6, config.max_ops_per_item * 2),
        )
        changed = corrected != item.current
        if changed:
            replacements[item.output_line] = corrected
            applied_sentences += 1
            applied_spans += sum(1 for change in changes if change.get("applied"))
        audit.append(
            {
                "id": item.item_id,
                "page": [item.page_number, item.side],
                "source_ids": item.source_ids,
                "current": item.current,
                "corrected": corrected,
                "changes": changes,
                "applied_sentence": changed,
                "txt_gate": "applied_atomic_sentence" if changed else "keep_local",
                "ai_raw": ai_raw,
            }
        )

    local_text = layout.local_txt.read_text(encoding="utf-8")
    deep_text, patched, missed = _patch_exact_lines(local_text, replacements)
    write_text(deep_txt, deep_text)
    write_epub(deep_epub, f"{layout.stem} — V4 LOCAL TURBO + DeepSeek{suffix}", sides_from_text(deep_text))
    write_json(audit_path, audit)

    total = time.perf_counter() - start
    summary = {
        "mode": "deep-only-sentence-atomic",
        "source": str(layout.root),
        "model": config.model,
        "queue_unit": "sentence",
        "page_filter": [page_start, page_end] if suffix else None,
        "queue_items": len(queue),
        "queue_source_lines": source_lines,
        "skipped_items": skipped,
        "ai_batch_size": config.batch_size,
        "ai_workers": config.workers,
        "ai_calls": len(batches),
        "ai_failed_calls": failed_calls,
        "ai_database_lock_retries": lock_retries,
        "ai_wall_seconds": round(ai_wall, 3),
        "ai_sum_call_seconds": round(sum(call_seconds), 3),
        "applied_sentences": applied_sentences,
        "applied_spans": applied_spans,
        "epub_patch": {"patched_sentences": patched, "missed_sentences": missed},
        "total_seconds": round(total, 3),
        "local_txt_untouched": str(layout.local_txt),
        "local_epub_untouched": str(layout.local_epub),
        "deep_txt": str(deep_txt),
        "deep_epub": str(deep_epub),
        "audit": str(audit_path),
    }
    write_json(summary_path, summary)
    return DeepResult(summary=summary, audit=audit)
