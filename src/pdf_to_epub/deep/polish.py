"""Standalone Deep-only orchestration for constrained OCR choices."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
from typing import Any, Callable

from ..config import DeepConfig, OutputLayout
from ..epub import sides_from_text, write_epub
from ..jsonio import write_json, write_text
from ..ocr.lexicon import load_vietnamese_words
from .client import deepseek_call, find_opencode, run_exe
from .gate import apply_ai_ops, apply_verifier_votes, render_applied_ops
from .prompt import build_prompt
from .queue import build_queue
from .verify import build_verifier_prompt


@dataclass(slots=True)
class DeepResult:
    summary: dict[str, Any]
    audit: list[dict[str, Any]]


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _patch_exact_lines(text: str, replacements: dict[str, str]) -> tuple[str, int, int]:
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

    # Dictionary extraction is optional local metadata work, not OCR.  The book
    # vocabulary still provides candidates if Tesseract helper tools are absent.
    lexicon = load_vietnamese_words(layout.deep_work_dir)
    log(f"Vietnamese lexicon: {len(lexicon)} words (0 means book-only candidate mode)")

    queue, skipped = build_queue(layout.local_txt, local_audit, lexicon=lexicon)
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
    verify_rows: list[dict[str, Any]] = []
    verify_ids_by_item: dict[str, list[str]] = {}

    for item in queue:
        ai_raw = primary_by_id.get(item.item_id, {"id": item.item_id, "selections": []})
        raw = ai_raw.get("selections") if isinstance(ai_raw, dict) else []
        if not isinstance(raw, list):
            raw = []
        # Compatibility with an old model response during branch transitions.
        if not raw and isinstance(ai_raw, dict) and isinstance(ai_raw.get("ops"), list):
            raw = ai_raw["ops"]
        _, ops = apply_ai_ops(item, raw, config.min_apply_confidence, config.max_ops_per_item)
        ops_by_id[item.item_id] = ops

        ids: list[str] = []
        for op in ops:
            if op.get("decision") != "verify":
                continue
            token_id = str(op.get("token_id") or "")
            verify_id = f"{item.item_id}:{token_id}"
            ids.append(verify_id)
            verify_rows.append({
                "id": verify_id,
                "current": item.current,
                "context": item.context,
                "token_id": token_id,
                "old": op.get("old"),
                "candidate": op.get("new"),
                "candidate_metadata": op.get("choice", {}),
            })
        verify_ids_by_item[item.item_id] = ids

    verifier_by_id: dict[str, dict[str, Any]] = {}
    verifier_failed = 0
    verifier_seconds: list[float] = []
    verifier_wall = 0.0
    verifier_calls = 0
    if verify_rows:
        log(f"Second-pass VERIFY choices: {len(verify_rows)}")
        verifier_by_id, verifier_failed, verifier_seconds, verifier_wall, verifier_calls = _run_batches(
            items=verify_rows,
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
        log("Second-pass VERIFY choices: 0")

    audit: list[dict[str, Any]] = []
    replacements: dict[str, str] = {}
    applied_lines = 0
    applied_ops = 0
    verified_changes = 0
    verifier_keeps = 0
    weak_rejects = 0

    for item in queue:
        ops = ops_by_id[item.item_id]
        votes: dict[str, str] = {}
        verifier_raw: list[dict[str, Any]] = []
        for verify_id in verify_ids_by_item.get(item.item_id, []):
            response = verifier_by_id.get(verify_id, {"id": verify_id, "decision": "KEEP"})
            verifier_raw.append(response)
            token_id = verify_id.rsplit(":", 1)[-1]
            votes[token_id] = str(response.get("decision") or "KEEP")

        if votes:
            corrected = apply_verifier_votes(item, ops, votes)
        else:
            corrected = render_applied_ops(item, ops)

        changed = corrected != item.current
        if changed:
            replacements[item.output_line] = corrected
            applied_lines += 1
        applied_ops += sum(1 for op in ops if op.get("applied"))
        verified_changes += sum(1 for op in ops if op.get("decision") == "verified")
        verifier_keeps += sum(1 for op in ops if op.get("gate") == "closed_choice_verifier_keep")
        weak_rejects += sum(1 for op in ops if op.get("gate") == "weak_local_candidate")

        audit.append({
            "id": item.item_id,
            "page": [item.page_number, item.side],
            "current": item.current,
            "corrected": corrected,
            "ops": ops,
            "applied_line": changed,
            "txt_gate": "applied" if changed else "no_change",
            "ai_raw": primary_by_id.get(item.item_id, {"id": item.item_id, "selections": []}),
            "verifier_raw": verifier_raw,
        })

    local_text = layout.local_txt.read_text(encoding="utf-8")
    deep_text, patched, missed = _patch_exact_lines(local_text, replacements)
    write_text(layout.deep_txt, deep_text)
    write_epub(layout.deep_epub, f"{layout.stem} — V4 LOCAL TURBO + DeepSeek", sides_from_text(deep_text))
    write_json(layout.root / "deep_ai_audit.json", audit)

    total = time.perf_counter() - start
    all_seconds = primary_seconds + verifier_seconds
    summary = {
        "mode": "deep-only-constrained-choice",
        "source": str(layout.root),
        "model": config.model,
        "lexicon_words": len(lexicon),
        "queue_items": len(queue),
        "skipped_items": skipped,
        "choice_sets": choice_sets,
        "ai_batch_size": config.batch_size,
        "ai_workers": config.workers,
        "primary_calls": primary_calls,
        "verifier_items": len(verify_rows),
        "verifier_calls": verifier_calls,
        "ai_calls": primary_calls + verifier_calls,
        "ai_failed_calls": primary_failed + verifier_failed,
        "primary_wall_seconds": round(primary_wall, 3),
        "verifier_wall_seconds": round(verifier_wall, 3),
        "ai_wall_seconds": round(primary_wall + verifier_wall, 3),
        "ai_sum_call_seconds": round(sum(all_seconds), 3),
        "verified_changes": verified_changes,
        "verifier_keeps": verifier_keeps,
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
    }
    write_json(layout.root / "SUMMARY_DEEP_ONLY.json", summary)
    return DeepResult(summary=summary, audit=audit)
