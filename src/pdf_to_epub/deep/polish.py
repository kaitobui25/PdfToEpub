"""Standalone Deep-only orchestration: queue → parallel AI → gate → patch."""

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
from .gate import apply_ai_ops
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
    """Patch exact OCR substrings inside final paragraph lines.

    LOCAL_TURBO may join several visual OCR lines into one paragraph. Therefore
    the patch target is an exact substring, not necessarily the entire TXT line.
    A target is applied only when it occurs once in the document.
    """

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
    """Return one private OpenCode SQLite DB per executor worker for this run."""

    worker = re.sub(r"[^A-Za-z0-9_.-]+", "_", threading.current_thread().name)
    return root / f"run_{os.getpid()}_{worker}.db"


def run_deep_only(
    layout: OutputLayout,
    config: DeepConfig,
    log: Callable[[str], None],
) -> DeepResult:
    """Run DeepSeek against existing local artifacts; never touch OCR/Tesseract."""

    start = time.perf_counter()
    if not layout.local_txt.exists():
        raise FileNotFoundError(f"Local TXT not found: {layout.local_txt}")
    local_audit = layout.root / "local_refine_audit.json"
    if not local_audit.exists():
        raise FileNotFoundError(f"Local refinement audit not found: {local_audit}")

    queue, skipped = build_queue(layout.local_txt, local_audit)
    write_json(layout.root / "deep_ai_queue.json", [item.as_dict() for item in queue])

    opencode = find_opencode()
    if opencode is None:
        raise FileNotFoundError("OpenCode CLI not found")
    version = run_exe(opencode, ["--version"], timeout=30)
    if version.returncode != 0:
        raise RuntimeError("opencode --version failed")

    batches = _chunks(queue, config.batch_size)
    log(f"Queue: {len(queue)} unresolved lines; skipped={skipped}")
    log(f"Micro-batch: {config.batch_size} items; parallel AI workers: {config.workers}")
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
                    f"deep_micro_{index:03d}",
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
                log(f"[AI {index:03d}/{len(batches):03d}] items={len(batch)} time={elapsed:.2f}s{retry_suffix}")
            except Exception as exc:  # keep remaining independent batches running
                failed_calls += 1
                log(f"[AI FAIL] {exc}")

    ai_wall = time.perf_counter() - ai_start
    log(f"[AI WALL] {ai_wall:.2f}s")

    audit: list[dict[str, Any]] = []
    replacements: dict[str, str] = {}
    applied_lines = 0
    applied_ops = 0
    for item in queue:
        ai_raw = ai_by_id.get(item.item_id, {"id": item.item_id, "ops": []})
        raw_ops = ai_raw.get("ops") if isinstance(ai_raw, dict) else []
        if not isinstance(raw_ops, list):
            raw_ops = []
        corrected, ops = apply_ai_ops(item, raw_ops, config.min_apply_confidence, config.max_ops_per_item)
        changed = corrected != item.current
        if changed:
            replacements[item.output_line] = corrected
            applied_lines += 1
            applied_ops += sum(1 for op in ops if op.get("applied"))
        audit.append({
            "id": item.item_id,
            "page": [item.page_number, item.side],
            "current": item.current,
            "corrected": corrected,
            "ops": ops,
            "applied_line": changed,
            "txt_gate": "applied" if changed else "no_change",
            "ai_raw": ai_raw,
        })

    local_text = layout.local_txt.read_text(encoding="utf-8")
    deep_text, patched, missed = _patch_exact_lines(local_text, replacements)
    write_text(layout.deep_txt, deep_text)
    write_epub(layout.deep_epub, f"{layout.stem} — V4 LOCAL TURBO + DeepSeek", sides_from_text(deep_text))
    write_json(layout.root / "deep_ai_audit.json", audit)

    total = time.perf_counter() - start
    summary = {
        "mode": "deep-only",
        "source": str(layout.root),
        "model": config.model,
        "queue_items": len(queue),
        "skipped_items": skipped,
        "ai_batch_size": config.batch_size,
        "ai_workers": config.workers,
        "ai_calls": len(batches),
        "ai_failed_calls": failed_calls,
        "ai_database_lock_retries": lock_retries,
        "ai_wall_seconds": round(ai_wall, 3),
        "ai_sum_call_seconds": round(sum(call_seconds), 3),
        "applied_lines": applied_lines,
        "applied_ops": applied_ops,
        "epub_patch": {"patched_lines": patched, "missed_lines": missed},
        "total_seconds": round(total, 3),
        "local_txt_untouched": str(layout.local_txt),
        "local_epub_untouched": str(layout.local_epub),
        "deep_txt": str(layout.deep_txt),
        "deep_epub": str(layout.deep_epub),
        "audit": str(layout.root / "deep_ai_audit.json"),
    }
    write_json(layout.root / "SUMMARY_DEEP_ONLY.json", summary)
    return DeepResult(summary=summary, audit=audit)
