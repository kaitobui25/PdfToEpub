"""Standalone Deep-only orchestration for constrained OCR choices."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
import time
from typing import Any, Callable

from ..config import DeepConfig, OutputLayout
from ..epub import sides_from_text, write_epub
from ..jsonio import write_json, write_text
from ..ocr.lexicon import load_vietnamese_words
from ..ocr.line_health import analyze_line_health
from .client import deepseek_call, find_opencode, run_exe
from .gate import apply_ai_ops, render_applied_ops
from .prompt import build_prompt
from .queue import build_queue
from .verify import build_reverse_prompt, build_verifier_prompt
from .voting import (
    finalize_reverse_vote,
    finalize_selected_vote,
    make_reverse_op,
    reverse_rows,
    selected_verify_rows,
    tie_row,
)


@dataclass(slots=True)
class DeepResult:
    summary: dict[str, Any]
    audit: list[dict[str, Any]]


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _side_bounds(text: str, page_number: int, side: str) -> tuple[int, int] | None:
    marker = f"===== PDF{page_number:03d}-{side} ====="
    start = text.find(marker)
    if start < 0:
        return None
    body_start = start + len(marker)
    next_marker = re.search(r"(?m)^===== PDF\d{3}-[LR] =====$", text[body_start:])
    end = body_start + next_marker.start() if next_marker else len(text)
    return body_start, end


def _patch_scoped_lines(text: str, replacements: list[dict[str, Any]]) -> tuple[str, int, int]:
    """Patch inside the owning PDF side instead of demanding global uniqueness."""

    patched = 0
    missed = 0
    output = text
    for replacement in replacements:
        old = str(replacement.get("old") or "")
        new = str(replacement.get("new") or "")
        if not old or not new:
            missed += 1
            continue
        bounds = _side_bounds(output, int(replacement["page"]), str(replacement["side"]))
        if bounds is None:
            missed += 1
            continue
        start, end = bounds
        segment = output[start:end]
        if segment.count(old) != 1:
            missed += 1
            continue
        segment = segment.replace(old, new, 1)
        output = output[:start] + segment + output[end:]
        patched += 1
    return output, patched, missed


def _run_batches(
    *,
    items: list[Any],
    batch_size: int,
    workers: int,
    opencode,
    model: str,
    workdir,
    timeout: int,
    prompt_builder: Callable[[list[Any]], str],
    prefix: str,
    label: str,
    log: Callable[[str], None],
) -> tuple[dict[str, dict[str, Any]], int, list[float], float, int]:
    batches = _chunks(items, batch_size)
    by_id: dict[str, dict[str, Any]] = {}
    failed = 0
    call_seconds: list[float] = []
    start = time.perf_counter()

    def call(index: int, batch: list[Any]) -> tuple[int, list[Any], dict[str, Any], float]:
        t0 = time.perf_counter()
        result = deepseek_call(
            opencode,
            model,
            prompt_builder(batch),
            workdir,
            f"{prefix}_{index:03d}",
            timeout,
        )
        return index, batch, result, time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(call, index + 1, batch) for index, batch in enumerate(batches)]
        for future in as_completed(futures):
            try:
                index, batch, response, elapsed = future.result()
                call_seconds.append(elapsed)
                for row in response.get("items", []):
                    if isinstance(row, dict) and row.get("id"):
                        by_id[str(row["id"])] = row
                log(f"[{label} {index:03d}/{len(batches):03d}] items={len(batch)} time={elapsed:.2f}s")
            except Exception as exc:
                failed += 1
                log(f"[{label} FAIL] {exc}")

    wall = time.perf_counter() - start
    log(f"[{label} WALL] {wall:.2f}s")
    return by_id, failed, call_seconds, wall, len(batches)


def _binary_response(mapping: dict[str, dict[str, Any]], row_id: str) -> str:
    response = mapping.get(row_id, {"decision": "KEEP"})
    return str(response.get("decision") or "KEEP").strip().upper()


def run_deep_only(
    layout: OutputLayout,
    config: DeepConfig,
    log: Callable[[str], None],
) -> DeepResult:
    """Validate existing LOCAL artifacts; OCR/Tesseract recognition is skipped."""

    start = time.perf_counter()
    if not layout.local_txt.exists():
        raise FileNotFoundError(f"Local TXT not found: {layout.local_txt}")
    local_audit = layout.root / "local_refine_audit.json"
    if not local_audit.exists():
        raise FileNotFoundError(f"Local refinement audit not found: {local_audit}")

    lexicon = load_vietnamese_words(layout.deep_work_dir)
    log(f"Vietnamese lexicon: {len(lexicon)} words (0 means book-only candidate mode)")

    queue, skipped = build_queue(layout.local_txt, local_audit, lexicon=lexicon)

    # LOCAL already performed targeted line reOCR for suspect healthy-side lines.
    # Classify the remaining evidence now; a still-collapsed line is a poor input
    # for language repair and is skipped without suppressing its entire side.
    healthy_queue = []
    line_health_skips: list[dict[str, Any]] = []
    for item in queue:
        report = analyze_line_health(item.current, item.candidate_meta, lexicon)
        if report.catastrophic:
            line_health_skips.append({
                "id": item.item_id,
                "page": [item.page_number, item.side],
                "current": item.current,
                "health": report.as_dict(),
            })
            continue
        healthy_queue.append(item)
    queue = healthy_queue
    skipped += len(line_health_skips)
    write_json(layout.root / "deep_line_health_skips.json", line_health_skips)
    write_json(layout.root / "deep_ai_queue.json", [item.as_dict() for item in queue])

    opencode = find_opencode()
    if opencode is None:
        raise FileNotFoundError("OpenCode CLI not found")
    version = run_exe(opencode, ["--version"], timeout=30)
    if version.returncode != 0:
        raise RuntimeError("opencode --version failed")

    choice_sets = sum(len(item.choice_sets) for item in queue)
    primary_batches = _chunks(queue, config.batch_size)
    log(f"Queue: {len(queue)} constrained lines; skipped/KEEP-local={skipped}")
    log(f"Line catastrophes after existing reOCR: {len(line_health_skips)}")
    log(f"Choice sets: {choice_sets}; Deep cannot invent replacement text")
    log(f"Micro-batch: {config.batch_size} items; parallel AI workers: {config.workers}")
    log(f"Primary choice calls: {len(primary_batches)}")

    primary_by_id, primary_failed, primary_seconds, primary_wall, primary_calls = _run_batches(
        items=queue,
        batch_size=config.batch_size,
        workers=config.workers,
        opencode=opencode,
        model=config.model,
        workdir=layout.deep_work_dir,
        timeout=config.call_timeout_seconds,
        prompt_builder=build_prompt,
        prefix="deep_choice",
        label="CHOICE",
        log=log,
    )

    ops_by_id: dict[str, list[dict[str, Any]]] = {}
    selected_rows: list[dict[str, Any]] = []
    reverse_choice_rows: list[dict[str, Any]] = []
    for item in queue:
        ai_raw = primary_by_id.get(item.item_id, {"id": item.item_id, "selections": []})
        raw = ai_raw.get("selections") if isinstance(ai_raw, dict) else []
        if not isinstance(raw, list):
            raw = []
        if not raw and isinstance(ai_raw, dict) and isinstance(ai_raw.get("ops"), list):
            raw = ai_raw["ops"]
        _, ops = apply_ai_ops(item, raw, config.min_apply_confidence, config.max_ops_per_item)
        ops_by_id[item.item_id] = ops
        selected_rows.extend(selected_verify_rows(item, ops))
        reverse_choice_rows.extend(reverse_rows(item, ops))

    selected_by_id: dict[str, dict[str, Any]] = {}
    selected_failed = 0
    selected_seconds: list[float] = []
    selected_wall = 0.0
    selected_calls = 0
    if selected_rows:
        log(f"Selected-medium verifier items: {len(selected_rows)}")
        selected_by_id, selected_failed, selected_seconds, selected_wall, selected_calls = _run_batches(
            items=selected_rows,
            batch_size=config.batch_size,
            workers=config.workers,
            opencode=opencode,
            model=config.model,
            workdir=layout.deep_work_dir,
            timeout=config.call_timeout_seconds,
            prompt_builder=build_verifier_prompt,
            prefix="deep_verify",
            label="VERIFY",
            log=log,
        )
    else:
        log("Selected-medium verifier items: 0")

    reverse_by_id: dict[str, dict[str, Any]] = {}
    reverse_failed = 0
    reverse_seconds: list[float] = []
    reverse_wall = 0.0
    reverse_calls = 0
    if reverse_choice_rows:
        log(f"Reverse KEEP second-look items: {len(reverse_choice_rows)}")
        reverse_by_id, reverse_failed, reverse_seconds, reverse_wall, reverse_calls = _run_batches(
            items=reverse_choice_rows,
            batch_size=config.batch_size,
            workers=config.workers,
            opencode=opencode,
            model=config.model,
            workdir=layout.deep_work_dir,
            timeout=config.call_timeout_seconds,
            prompt_builder=build_reverse_prompt,
            prefix="deep_reverse",
            label="REVERSE",
            log=log,
        )
    else:
        log("Reverse KEEP second-look items: 0")

    tie_rows: list[dict[str, Any]] = []
    reverse_ops_by_item: dict[str, list[dict[str, Any]]] = {item.item_id: [] for item in queue}

    # Primary CHANGE + verifier KEEP is a 1-1 conflict and gets one final vote.
    for item in queue:
        for op in ops_by_id[item.item_id]:
            if op.get("decision") != "verify":
                continue
            token_id = str(op.get("token_id") or "")
            row_id = f"{item.item_id}:{token_id}:verify"
            second = _binary_response(selected_by_id, row_id)
            if second == "CHANGE":
                finalize_selected_vote(op, second)
            else:
                finalize_selected_vote(op, second)
                tie_rows.append(tie_row(item, op, "tie_selected"))

    # Primary KEEP + reverse CHANGE is also a 1-1 conflict.  The reverse pass
    # may choose among several listed candidates but cannot invent a spelling.
    for item in queue:
        for row in reverse_rows(item, ops_by_id[item.item_id]):
            token_id = str(row.get("token_id") or "")
            row_id = f"{item.item_id}:{token_id}:reverse"
            response = reverse_by_id.get(row_id, {"choice_id": "KEEP"})
            choice_id = str(response.get("choice_id") or "KEEP").strip()
            primary_keep = next(
                (
                    op
                    for op in ops_by_id[item.item_id]
                    if str(op.get("token_id") or "") == token_id and op.get("decision") == "keep"
                ),
                None,
            )
            if primary_keep is not None:
                primary_keep["reverse_choice"] = choice_id
            if choice_id == "KEEP":
                if primary_keep is not None:
                    primary_keep["gate"] = "reverse_vote_2_of_2_keep"
                    primary_keep["votes"] = ["KEEP", "KEEP"]
                continue
            reverse_op = make_reverse_op(item, token_id, choice_id)
            if reverse_op is None:
                continue
            reverse_ops_by_item[item.item_id].append(reverse_op)
            tie_rows.append(tie_row(item, reverse_op, "tie_reverse"))

    tie_by_id: dict[str, dict[str, Any]] = {}
    tie_failed = 0
    tie_seconds: list[float] = []
    tie_wall = 0.0
    tie_calls = 0
    if tie_rows:
        log(f"Conflicting edit votes needing 2-of-3 tie-break: {len(tie_rows)}")
        tie_by_id, tie_failed, tie_seconds, tie_wall, tie_calls = _run_batches(
            items=tie_rows,
            batch_size=config.batch_size,
            workers=config.workers,
            opencode=opencode,
            model=config.model,
            workdir=layout.deep_work_dir,
            timeout=config.call_timeout_seconds,
            prompt_builder=build_verifier_prompt,
            prefix="deep_tie",
            label="TIE",
            log=log,
        )
    else:
        log("Conflicting edit votes needing 2-of-3 tie-break: 0")

    for item in queue:
        for op in ops_by_id[item.item_id]:
            if op.get("decision") == "tie_break" and op.get("gate") == "edit_vote_conflict":
                token_id = str(op.get("token_id") or "")
                vote = _binary_response(tie_by_id, f"{item.item_id}:{token_id}:tie_selected")
                finalize_selected_vote(op, "KEEP", vote)
        for op in reverse_ops_by_item[item.item_id]:
            token_id = str(op.get("token_id") or "")
            vote = _binary_response(tie_by_id, f"{item.item_id}:{token_id}:tie_reverse")
            finalize_reverse_vote(op, vote)
        ops_by_id[item.item_id].extend(reverse_ops_by_item[item.item_id])

    audit: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    applied_lines = 0
    applied_ops = 0
    verified_changes = 0
    vote_keeps = 0
    weak_rejects = 0

    for item in queue:
        ops = ops_by_id[item.item_id]
        corrected = render_applied_ops(item, ops)
        changed = corrected != item.current
        if changed:
            replacements.append({
                "page": item.page_number,
                "side": item.side,
                "old": item.output_line,
                "new": corrected,
            })
            applied_lines += 1
        applied_ops += sum(1 for op in ops if op.get("applied"))
        verified_changes += sum(1 for op in ops if op.get("decision") == "verified")
        vote_keeps += sum(1 for op in ops if str(op.get("gate") or "").endswith("keep"))
        weak_rejects += sum(1 for op in ops if op.get("gate") == "weak_local_candidate")

        token_ids = {str(op.get("token_id") or "") for op in ops if op.get("token_id")}
        selected_raw = [
            selected_by_id.get(f"{item.item_id}:{token_id}:verify")
            for token_id in token_ids
            if f"{item.item_id}:{token_id}:verify" in selected_by_id
        ]
        reverse_raw = [
            reverse_by_id.get(f"{item.item_id}:{token_id}:reverse")
            for token_id in token_ids
            if f"{item.item_id}:{token_id}:reverse" in reverse_by_id
        ]
        tie_raw = [
            response
            for row_id, response in tie_by_id.items()
            if row_id.startswith(f"{item.item_id}:")
        ]
        audit.append({
            "id": item.item_id,
            "page": [item.page_number, item.side],
            "current": item.current,
            "corrected": corrected,
            "ops": ops,
            "applied_line": changed,
            "txt_gate": "applied" if changed else "no_change",
            "ai_raw": primary_by_id.get(item.item_id, {"id": item.item_id, "selections": []}),
            "verifier_raw": selected_raw,
            "reverse_raw": reverse_raw,
            "tie_raw": tie_raw,
        })

    local_text = layout.local_txt.read_text(encoding="utf-8")
    deep_text, patched, missed = _patch_scoped_lines(local_text, replacements)
    write_text(layout.deep_txt, deep_text)
    write_epub(layout.deep_epub, f"{layout.stem} — V4 LOCAL TURBO + DeepSeek", sides_from_text(deep_text))
    write_json(layout.root / "deep_ai_audit.json", audit)

    total = time.perf_counter() - start
    all_seconds = primary_seconds + selected_seconds + reverse_seconds + tie_seconds
    all_wall = primary_wall + selected_wall + reverse_wall + tie_wall
    total_calls = primary_calls + selected_calls + reverse_calls + tie_calls
    total_failed = primary_failed + selected_failed + reverse_failed + tie_failed
    summary = {
        "mode": "deep-only-constrained-choice-voting",
        "source": str(layout.root),
        "model": config.model,
        "lexicon_words": len(lexicon),
        "queue_items": len(queue),
        "skipped_items": skipped,
        "line_catastrophe_skips": len(line_health_skips),
        "choice_sets": choice_sets,
        "ai_batch_size": config.batch_size,
        "ai_workers": config.workers,
        "primary_calls": primary_calls,
        "verifier_items": len(selected_rows),
        "verifier_calls": selected_calls,
        "reverse_items": len(reverse_choice_rows),
        "reverse_calls": reverse_calls,
        "tie_items": len(tie_rows),
        "tie_calls": tie_calls,
        "ai_calls": total_calls,
        "ai_failed_calls": total_failed,
        "primary_wall_seconds": round(primary_wall, 3),
        "verifier_wall_seconds": round(selected_wall, 3),
        "reverse_wall_seconds": round(reverse_wall, 3),
        "tie_wall_seconds": round(tie_wall, 3),
        "ai_wall_seconds": round(all_wall, 3),
        "ai_sum_call_seconds": round(sum(all_seconds), 3),
        "verified_changes": verified_changes,
        "vote_keeps": vote_keeps,
        "weak_candidate_rejects": weak_rejects,
        "applied_lines": applied_lines,
        "applied_ops": applied_ops,
        "epub_patch": {"patched_lines": patched, "missed_lines": missed},
        "total_seconds": round(total, 3),
        "local_txt_untouched": str(layout.local_txt),
        "local_epub_untouched": str(layout.local_epub),
        "deep_txt": str(layout.deep_txt),
        "deep_epub": str(layout.deep_epub),
        "audit": str(layout.root / "deep_ai_audit.json"),
        "line_health_skips": str(layout.root / "deep_line_health_skips.json"),
    }
    write_json(layout.root / "SUMMARY_DEEP_ONLY.json", summary)
    return DeepResult(summary=summary, audit=audit)
