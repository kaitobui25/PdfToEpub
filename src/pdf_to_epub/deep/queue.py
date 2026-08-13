"""Build a compact sentence-level Deep-only queue from local OCR evidence."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any

from ..jsonio import read_json
from ..models import DeepQueueItem


MARKER_RE = re.compile(r"^===== PDF(\d{3})-([LR]) =====$")
LEXICAL_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?…]+(?:[\"”’\)\]]+)?(?=\s+|$)")


def _page_blocks(text: str) -> dict[tuple[int, str], str]:
    """Return exact page-side text blocks, preserving original whitespace."""

    pages: dict[tuple[int, str], str] = {}
    key: tuple[int, str] | None = None
    buffer: list[str] = []
    for raw in text.splitlines(keepends=True):
        marker = MARKER_RE.match(raw.strip())
        if marker:
            if key is not None:
                pages[key] = "".join(buffer)
            key = (int(marker.group(1)), marker.group(2))
            buffer = []
            continue
        if key is not None:
            buffer.append(raw)
    if key is not None:
        pages[key] = "".join(buffer)
    return pages


def _logicalize(raw: str) -> tuple[str, list[int], list[int]]:
    """Collapse OCR line whitespace while keeping a map back to exact TXT spans."""

    chars: list[str] = []
    raw_starts: list[int] = []
    raw_ends: list[int] = []
    index = 0
    while index < len(raw):
        if raw[index].isspace():
            end = index + 1
            while end < len(raw) and raw[end].isspace():
                end += 1
            if chars and chars[-1] != " ":
                chars.append(" ")
                raw_starts.append(index)
                raw_ends.append(end)
            index = end
            continue
        chars.append(raw[index])
        raw_starts.append(index)
        raw_ends.append(index + 1)
        index += 1
    if chars and chars[-1] == " ":
        chars.pop()
        raw_starts.pop()
        raw_ends.pop()
    return "".join(chars), raw_starts, raw_ends


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _lexical(text: str) -> str:
    return " ".join(LEXICAL_RE.findall(text.casefold()))


def _high_conf_evidence(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("whole_candidates", "line_candidates"):
        for candidate in row.get(key, []):
            try:
                conf = float(candidate.get("conf") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            text = str(candidate.get("text") or "").strip()
            if conf >= 80.0 and text:
                values.append(_lexical(text))
    return values


def _trusted_stable(row: dict[str, Any]) -> bool:
    if row.get("edits"):
        return True
    current = str(row.get("after") or row.get("before") or "").strip()
    reasons = list(row.get("reasons") or [])
    evidence = _high_conf_evidence(row)
    stable = bool(evidence) and all(value == _lexical(current) for value in evidence)
    if not stable:
        return False
    if reasons == ["low_word_conf"] and len(current) > 30:
        return True
    if len(reasons) == 1 and str(reasons[0]).startswith("non_dictionary:"):
        attribution = bool(re.match(r"^\s*[—–-]\s*[A-ZÀ-ỸĐ]", current))
        if len(current) > 25 or attribution:
            return True
    return False


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return sentence spans over logical page text."""

    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        end = match.end()
        left = start
        while left < end and text[left].isspace():
            left += 1
        if left < end:
            spans.append((left, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _resolve_anchor(logical: str, current: str) -> tuple[int, int] | None:
    """Locate one audit OCR line/fragment inside whitespace-normalized page text."""

    wanted = _collapse_ws(current)
    if not wanted:
        return None
    positions = [match.start() for match in re.finditer(re.escape(wanted), logical)]
    if len(positions) == 1:
        return positions[0], positions[0] + len(wanted)

    stripped = re.sub(r"^[^0-9A-Za-zÀ-ỹĐđ]+", "", wanted).strip()
    if not stripped:
        return None
    positions = [match.start() for match in re.finditer(re.escape(stripped), logical)]
    if len(positions) == 1:
        return positions[0], positions[0] + len(stripped)
    return None


def _suspect_tokens(reasons: list[str]) -> list[str]:
    """Extract explicit suspect OCR tokens carried by audit reason strings."""

    tokens: list[str] = []
    for reason in reasons:
        value = str(reason)
        if ":" not in value:
            continue
        prefix, token = value.split(":", 1)
        token = token.strip()
        if prefix in {"non_dictionary", "low_word_conf", "suspicious_token", "diacritic_disagreement"} and token:
            tokens.append(token)
    return tokens


def _token_positions(text: str, token: str, start: int, end: int) -> list[int]:
    pattern = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)", re.UNICODE | re.IGNORECASE)
    return [match.start() for match in pattern.finditer(text, start, end)]


def _target_sentence_index(
    logical: str,
    spans: list[tuple[int, int]],
    anchor_start: int,
    anchor_end: int,
    reasons: list[str],
) -> tuple[int | None, str]:
    """Prefer the sentence containing the audited token; overlap is fallback only."""

    for token in _suspect_tokens(reasons):
        positions = _token_positions(logical, token, anchor_start, anchor_end)
        if not positions:
            continue
        position = positions[0]
        for index, (start, end) in enumerate(spans):
            if start <= position < end:
                return index, f"suspect_token:{token}"

    best_index: int | None = None
    best_overlap = 0
    for index, (start, end) in enumerate(spans):
        overlap = max(0, min(end, anchor_end) - max(start, anchor_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index
    return best_index, "anchor_overlap"


def _sentence_window(logical: str, spans: list[tuple[int, int]], target_index: int) -> str:
    """Give Deep previous + target + next while making edit scope unambiguous."""

    def value(index: int) -> str:
        if index < 0 or index >= len(spans):
            return "[NONE]"
        start, end = spans[index]
        return logical[start:end].strip()

    return (
        f"PREVIOUS: {value(target_index - 1)}\n"
        f"TARGET: {value(target_index)}\n"
        f"NEXT: {value(target_index + 1)}"
    )


def _candidate_texts(row: dict[str, Any], current: str) -> list[str]:
    """Keep useful OCR alternatives but drop giant TSV/debug blobs."""

    candidates: list[str] = []
    for source_key in ("line_candidates", "whole_candidates"):
        for candidate in row.get(source_key, []):
            text = str(candidate.get("text") or "").strip()
            if not text or text.count("\t") >= 5 or len(text) > 1200:
                continue
            text = _collapse_ws(text)
            if text and _lexical(text) != _lexical(current) and text not in candidates:
                candidates.append(text)
    return candidates


def _skip_record(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "page": list(row.get("page") or []),
        "current": str(row.get("after") or row.get("before") or "").strip(),
        "reasons": list(row.get("reasons") or []),
        "skip_reason": reason,
    }


def build_queue(
    local_txt: Path,
    local_refine_audit: Path,
    page_start: int | None = None,
    page_end: int | None = None,
) -> tuple[list[DeepQueueItem], list[dict[str, Any]]]:
    """Group suspect OCR evidence by full sentence and attach a 3-sentence window.

    The audited suspect token selects TARGET whenever the reason names one. This
    fixes OCR lines that straddle a sentence boundary: Deep sees previous/target/
    next, but it is allowed to return only the corrected TARGET sentence.
    """

    output = local_txt.read_text(encoding="utf-8")
    page_blocks = _page_blocks(output)
    audit: list[dict[str, Any]] = read_json(local_refine_audit)
    skipped: list[dict[str, Any]] = []
    groups: dict[tuple[int, str, str], dict[str, Any]] = {}

    page_cache: dict[tuple[int, str], tuple[str, list[int], list[int], list[tuple[int, int]]]] = {}
    for key, raw in page_blocks.items():
        logical, raw_starts, raw_ends = _logicalize(raw)
        page_cache[key] = (logical, raw_starts, raw_ends, _sentence_spans(logical))

    for row in audit:
        page_number, side = int(row["page"][0]), str(row["page"][1])
        if page_start is not None and page_number < page_start:
            continue
        if page_end is not None and page_number > page_end:
            continue

        reasons = list(row.get("reasons") or [])
        if not reasons or row.get("edits"):
            continue
        if "whole_side_catastrophe" in reasons:
            skipped.append(_skip_record(row, "whole_side_catastrophe"))
            continue
        if _trusted_stable(row):
            continue

        current = str(row.get("after") or row.get("before") or "").strip()
        if not current:
            skipped.append(_skip_record(row, "empty_current"))
            continue

        key = (page_number, side)
        cached = page_cache.get(key)
        if cached is None:
            skipped.append(_skip_record(row, "page_side_not_found"))
            continue
        logical, raw_starts, raw_ends, spans = cached
        if not spans:
            skipped.append(_skip_record(row, "no_sentence_spans"))
            continue

        anchor = _resolve_anchor(logical, current)
        if anchor is None:
            skipped.append(_skip_record(row, "audit_fragment_not_found_or_ambiguous"))
            continue
        target_index, target_basis = _target_sentence_index(logical, spans, anchor[0], anchor[1], reasons)
        if target_index is None:
            skipped.append(_skip_record(row, "target_sentence_not_found"))
            continue

        start, end = spans[target_index]
        sentence = logical[start:end].strip()
        if not sentence:
            skipped.append(_skip_record(row, "empty_target_sentence"))
            continue
        raw_target = page_blocks[key][raw_starts[start] : raw_ends[end - 1]]
        if not raw_target or output.count(raw_target) != 1:
            skipped.append(_skip_record(row, "target_txt_span_not_unique"))
            continue

        group_key = (page_number, side, sentence)
        group = groups.setdefault(
            group_key,
            {
                "raw_target": raw_target,
                "context": _sentence_window(logical, spans, target_index),
                "source_ids": [],
                "reasons": [],
                "candidates": [],
                "target_bases": [],
            },
        )
        source_id = str(row["id"])
        if source_id not in group["source_ids"]:
            group["source_ids"].append(source_id)
        if target_basis not in group["target_bases"]:
            group["target_bases"].append(target_basis)
        for reason in reasons:
            if reason not in group["reasons"]:
                group["reasons"].append(reason)
        for candidate in _candidate_texts(row, current):
            if candidate not in group["candidates"]:
                group["candidates"].append(candidate)

    queue: list[DeepQueueItem] = []
    side_counts: dict[tuple[int, str], int] = defaultdict(int)
    for (page_number, side, sentence), group in groups.items():
        side_counts[(page_number, side)] += 1
        index = side_counts[(page_number, side)]
        reasons = list(group["reasons"])
        reasons.extend(f"target_basis:{basis}" for basis in group["target_bases"])
        queue.append(
            DeepQueueItem(
                item_id=f"{page_number:03d}-{side}-S{index:03d}",
                page_number=page_number,
                side=side,
                current=sentence,
                output_line=str(group["raw_target"]),
                context=str(group["context"]),
                reasons=reasons,
                candidates=list(group["candidates"])[:16],
                source_ids=list(group["source_ids"]),
            )
        )
    return queue, skipped
