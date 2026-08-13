"""Enhanced Deep-only runtime with independent confirmation and smart TXT projection."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any, Callable

from ..config import DeepConfig, OutputLayout
from ..epub import sides_from_text, write_epub
from ..jsonio import write_json, write_text
from ..models import DeepQueueItem
from .client import deepseek_call, find_opencode, run_exe
from .confirmation import confirmation_status, needs_second_pass, proposed_sentence, safe_confidence
from .gate import apply_ai_sentence
from .polish import (
    DeepResult,
    LOCK_RETRY_DELAYS,
    _block_reasons,
    _chunks,
    _patch_exact_lines,
    _range_suffix,
    _render_corrected_target,
    _worker_database,
)
from .projection import render_smart_target
from .prompt import build_prompt
from .queue import build_queue


def _log_result_summary(log: Callable[[str], None], summary: dict[str, Any]) -> None:
    log("")
    log("================ DEEP RESULT ================")
    log(f"Deep trust            : {summary['deep_trust']}")
    log(f"OCR evidence gate     : {summary['ocr_evidence_gate']}")
    log(f"Patch projection      : {summary['patch_projection']}")
    log(f"Suspect TARGETs       : {summary['queue_items']} ({summary['queue_source_lines']} OCR source lines)")
    log(f"Deep proposed changes : {summary['deep_proposed_sentences']} TARGETs / {summary['deep_proposed_repair_groups']} repair groups")
    log(
        f"Second pass 0.90-0.949: {summary['second_pass_candidates']} candidates; "
        f"confirmed={summary['second_pass_confirmed']}; not_confirmed={summary['second_pass_not_confirmed']}"
    )
    log(f"Applied to TXT        : {summary['applied_sentences']} TARGETs / {summary['applied_spans']} repair groups")
    log(f"Blocked               : {summary['gate_blocked_sentences']} TARGETs")
    log(f"Patch projection fail : {summary['patch_projection_failures']} TARGETs")
    log(f"Smart layout fallback : {summary['smart_projection_fallbacks']} TARGETs")
    log(f"Deep kept unchanged   : {summary['deep_unchanged_sentences']} TARGETs")
    log(f"Skipped before Deep   : {summary['skipped_items']} items")
    reasons = summary.get("block_reasons") or {}
    if reasons:
        log("Block reasons:")
        for reason, count in sorted(reasons.items(), key=lambda pair: (-pair[1], pair[0])):
            log(f"  - {reason}: {count}")
    else:
        log("Block reasons         : none")
    log("=============================================")


def run_deep_only(
    layout: OutputLayout,
    config: DeepConfig,
    log: Callable[[str], None],
    page_start: int | None = None,
    page_end: int | None = None,
) -> DeepResult:
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
    skipped_path = layout.root / f"deep_ai_skipped{suffix}.json"
    summary_path = layout.root / f"SUMMARY_DEEP_ONLY{suffix}.json"

    queue, skipped_records = build_queue(layout.local_txt, local_audit, page_start=page_start, page_end=page_end)
    write_json(queue_path, [item.as_dict() for item in queue])
    write_json(skipped_path, skipped_records)

    opencode = find_opencode()
    if opencode is None:
        raise FileNotFoundError("OpenCode CLI not found")
    version = run_exe(opencode, ["--version"], timeout=30)
    if version.returncode != 0:
        raise RuntimeError("opencode --version failed")

    source_lines = sum(len(item.source_ids) or 1 for item in queue)
    initial_batches = _chunks(queue, config.batch_size)
    if suffix:
        log(f"Deep page filter: PDF {page_start}..{page_end}; OCR source remains the full local run")
    log(f"Sentence queue: {len(queue)} sentences from {source_lines} unresolved source lines; skipped_lines={len(skipped_records)}")
    log(f"Deep trust: {config.deep_trust}")
    log(f"OCR evidence gate: {'on' if config.ocr_evidence_gate else 'off'}")
    log(f"Patch projection: {config.patch_projection}")
    log(f"Micro-batch: {config.batch_size} sentences; parallel AI workers: {config.workers}")
    log(f"AI calls: {len(initial_batches)} initial, max {config.workers} concurrent")

    ai_by_id: dict[str, dict[str, Any]] = {}
    second_ai_by_id: dict[str, dict[str, Any]] = {}
    lock_retries = 0
    call_seconds: list[float] = []
    ai_start = time.perf_counter()
    worker_db_root = layout.deep_work_dir / "opencode_worker_db"
    worker_db_root.mkdir(parents=True, exist_ok=True)

    def call(index: int, batch: list[DeepQueueItem], stage: str) -> tuple[int, list[DeepQueueItem], dict[str, Any], float, int]:
        t0 = time.perf_counter()
        retries = 0
        database_path = _worker_database(worker_db_root)
        while True:
            try:
                response = deepseek_call(
                    opencode,
                    config.model,
                    build_prompt(batch),
                    layout.deep_work_dir,
                    f"deep_{stage}{suffix}_{index:03d}",
                    config.call_timeout_seconds,
                    database_path=database_path,
                )
                return index, batch, response, time.perf_counter() - t0, retries
            except RuntimeError as exc:
                if "database is locked" not in str(exc).casefold() or retries >= len(LOCK_RETRY_DELAYS):
                    raise
                delay = LOCK_RETRY_DELAYS[retries]
                retries += 1
                log(f"[{stage.upper()} RETRY {index:03d}] DB locked; retry {retries}/{len(LOCK_RETRY_DELAYS)} in {delay:.2f}s")
                time.sleep(delay)

    def run_batches(
        stage: str,
        batches: list[list[DeepQueueItem]],
        destination: dict[str, dict[str, Any]],
        abort_on_failure: bool,
    ) -> int:
        nonlocal lock_retries
        failures = 0
        if not batches:
            return 0
        with ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix=f"deep_{stage}") as pool:
            futures = [pool.submit(call, i + 1, batch, stage) for i, batch in enumerate(batches)]
            for future in as_completed(futures):
                try:
                    index, batch, response, elapsed, retries = future.result()
                    lock_retries += retries
                    call_seconds.append(elapsed)
                    for row in response.get("items", []):
                        if isinstance(row, dict) and row.get("id"):
                            destination[str(row["id"])] = row
                    retry_text = f" retries={retries}" if retries else ""
                    log(f"[{stage.upper()} {index:03d}/{len(batches):03d}] sentences={len(batch)} time={elapsed:.2f}s{retry_text}")
                except Exception as exc:
                    failures += 1
                    log(f"[{stage.upper()} FAIL] {exc}")
        if failures and abort_on_failure:
            raise RuntimeError(
                f"Deep-only incomplete: {failures}/{len(batches)} initial AI calls failed. "
                "Refusing to write a partial Deep TXT/EPUB; rerun Deep-only."
            )
        return failures

    initial_failed_calls = run_batches("AI", initial_batches, ai_by_id, abort_on_failure=True)

    second_items = [item for item in queue if needs_second_pass(ai_by_id.get(item.item_id), item.current)]
    second_batches = _chunks(second_items, config.batch_size)
    if second_items:
        log(f"Second pass: {len(second_items)} TARGETs at confidence 0.90..0.949; {len(second_batches)} independent call(s)")
        second_failed_calls = run_batches("AI2", second_batches, second_ai_by_id, abort_on_failure=False)
    else:
        log("Second pass: no 0.90..0.949 corrections")
        second_failed_calls = 0

    ai_wall = time.perf_counter() - ai_start
    log(f"[AI WALL] {ai_wall:.2f}s")

    second_status_by_id: dict[str, str] = {}
    effective_ai_by_id: dict[str, dict[str, Any]] = dict(ai_by_id)
    second_confirmed = 0
    second_not_confirmed = 0
    for item in second_items:
        first = ai_by_id[item.item_id]
        second = second_ai_by_id.get(item.item_id)
        status = confirmation_status(first, second, item.current)
        second_status_by_id[item.item_id] = status
        effective = dict(first)
        effective["first_pass_confidence"] = safe_confidence(first)
        effective["second_pass_confidence"] = safe_confidence(second)
        effective["second_pass_status"] = status
        if status == "confirmed":
            second_confirmed += 1
            effective["confidence"] = max(config.min_apply_confidence, safe_confidence(first), safe_confidence(second))
        else:
            second_not_confirmed += 1
            effective["confidence"] = 0.0
        effective_ai_by_id[item.item_id] = effective

    audit: list[dict[str, Any]] = []
    replacements: dict[str, str] = {}
    applied_sentences = 0
    applied_spans = 0
    projection_failures = 0
    smart_fallbacks = 0
    deep_proposed_sentences = 0
    deep_proposed_repair_groups = 0
    deep_unchanged_sentences = 0
    gate_blocked_sentences = 0
    block_reasons: Counter[str] = Counter()

    for item in queue:
        first_ai_raw = ai_by_id.get(item.item_id, {"id": item.item_id, "corrected_sentence": item.current, "confidence": 0.0})
        effective_ai_raw = effective_ai_by_id.get(item.item_id, first_ai_raw)
        proposed = proposed_sentence(first_ai_raw, item.current)
        ai_changed = proposed != item.current.strip()
        if ai_changed:
            deep_proposed_sentences += 1
        else:
            deep_unchanged_sentences += 1

        corrected, changes = apply_ai_sentence(
            item,
            effective_ai_raw,
            config.min_apply_confidence,
            max_changed_words=max(6, config.max_ops_per_item * 2),
            deep_trust=config.deep_trust,
            ocr_evidence_gate=config.ocr_evidence_gate,
        )
        if ai_changed:
            deep_proposed_repair_groups += max(1, len(changes))

        gate_changed = corrected != item.current
        second_status = second_status_by_id.get(item.item_id)
        if ai_changed and not gate_changed:
            gate_blocked_sentences += 1
            if second_status and second_status != "confirmed":
                block_reasons[f"second_pass_{second_status}"] += 1
            else:
                block_reasons.update(set(_block_reasons(changes)))

        rendered: str | None = None
        txt_applied = False
        txt_gate = "keep_local"
        if gate_changed:
            if config.patch_projection == "off":
                rendered = corrected
                txt_gate = "applied_deep_direct"
            elif config.patch_projection == "smart":
                rendered = render_smart_target(item, corrected)
                if rendered is None:
                    rendered = corrected
                    smart_fallbacks += 1
                    txt_gate = "applied_smart_fallback"
                else:
                    txt_gate = "applied_smart_projection"
            else:
                rendered = _render_corrected_target(item, corrected, changes)
                if rendered is None:
                    projection_failures += 1
                    txt_gate = "patch_projection_failed"
                else:
                    txt_gate = "applied_atomic_sentence"

            if rendered is not None:
                replacements[item.output_line] = rendered
                applied_sentences += 1
                applied_spans += sum(1 for change in changes if change.get("applied"))
                txt_applied = True

        audit.append({
            "id": item.item_id,
            "page": [item.page_number, item.side],
            "source_ids": item.source_ids,
            "current": item.current,
            "raw_target": item.output_line,
            "context_window": item.context,
            "corrected": corrected,
            "changes": changes,
            "deep_trust": config.deep_trust,
            "ocr_evidence_gate": config.ocr_evidence_gate,
            "patch_projection": config.patch_projection,
            "deep_proposed_change": ai_changed,
            "second_pass_status": second_status,
            "second_ai_raw": second_ai_by_id.get(item.item_id),
            "gate_accepted_sentence": gate_changed,
            "applied_sentence": txt_applied,
            "txt_gate": txt_gate,
            "ai_raw": first_ai_raw,
            "effective_ai_raw": effective_ai_raw,
        })

    local_text = layout.local_txt.read_text(encoding="utf-8")
    deep_text, patched, missed = _patch_exact_lines(local_text, replacements)
    write_text(deep_txt, deep_text)
    write_epub(deep_epub, f"{layout.stem} — V4 LOCAL TURBO + DeepSeek{suffix}", sides_from_text(deep_text))
    write_json(audit_path, audit)

    total = time.perf_counter() - start
    summary = {
        "mode": "deep-only-sentence-atomic-3context-confirmed-smart",
        "source": str(layout.root),
        "model": config.model,
        "deep_trust": config.deep_trust,
        "ocr_evidence_gate": "on" if config.ocr_evidence_gate else "off",
        "patch_projection": config.patch_projection,
        "queue_unit": "sentence",
        "context_window": "previous+target+next; target-only editable",
        "page_filter": [page_start, page_end] if suffix else None,
        "queue_items": len(queue),
        "queue_source_lines": source_lines,
        "skipped_items": len(skipped_records),
        "skipped_audit": str(skipped_path),
        "ai_batch_size": config.batch_size,
        "ai_workers": config.workers,
        "ai_calls": len(initial_batches) + len(second_batches),
        "ai_calls_initial": len(initial_batches),
        "ai_calls_second_pass": len(second_batches),
        "ai_failed_calls": initial_failed_calls,
        "ai_second_pass_failed_calls": second_failed_calls,
        "ai_database_lock_retries": lock_retries,
        "ai_wall_seconds": round(ai_wall, 3),
        "ai_sum_call_seconds": round(sum(call_seconds), 3),
        "second_pass_candidates": len(second_items),
        "second_pass_confirmed": second_confirmed,
        "second_pass_not_confirmed": second_not_confirmed,
        "deep_proposed_sentences": deep_proposed_sentences,
        "deep_proposed_repair_groups": deep_proposed_repair_groups,
        "deep_unchanged_sentences": deep_unchanged_sentences,
        "gate_blocked_sentences": gate_blocked_sentences,
        "block_reasons": dict(sorted(block_reasons.items())),
        "applied_sentences": applied_sentences,
        "applied_spans": applied_spans,
        "patch_projection_failures": projection_failures,
        "smart_projection_fallbacks": smart_fallbacks,
        "epub_patch": {"patched_sentences": patched, "missed_sentences": missed},
        "total_seconds": round(total, 3),
        "local_txt_untouched": str(layout.local_txt),
        "local_epub_untouched": str(layout.local_epub),
        "deep_txt": str(deep_txt),
        "deep_epub": str(deep_epub),
        "audit": str(audit_path),
    }
    write_json(summary_path, summary)
    _log_result_summary(log, summary)
    return DeepResult(summary=summary, audit=audit)
