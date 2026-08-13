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


def _flatten_output_lines(text: str) -> dict[tuple[int, str], list[str]]:
    pages: dict[tuple[int, str], list[str]] = defaultdict(list)
    key: tuple[int, str] | None = None
    for raw in text.splitlines():
        marker = MARKER_RE.match(raw.strip())
        if marker:
            key = (int(marker.group(1)), marker.group(2))
            continue
        line = raw.strip()
        if key and line:
            pages[key].append(line)
    return dict(pages)


def _context(lines: list[str], target: str, radius_chars: int = 320) -> str:
    joined = "\n\n".join(lines)
    pos = joined.find(target)
    if pos < 0:
        return joined[: radius_chars * 2]
    start = max(0, pos - radius_chars)
    end = min(len(joined), pos + len(target) + radius_chars)
    return joined[start:end]


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


def _resolve_target(page_lines: list[str], current: str) -> str | None:
    if any(current in line for line in page_lines):
        return current
    stripped = re.sub(r"^[^0-9A-Za-zÀ-ỹĐđ]+", "", current).strip()
    if stripped and any(stripped in line for line in page_lines):
        return stripped
    return None


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return exact sentence spans without rewriting whitespace or punctuation."""

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


def _best_sentence_target(containing: str, target: str) -> str:
    """Pick the full sentence with the greatest overlap with one suspect OCR line.

    OCR visual lines often straddle a sentence boundary. Choosing the sentence
    with the largest overlap keeps one Deep item atomic instead of sending a
    half-sentence token fragment or duplicating the same source line twice.
    """

    target_start = containing.find(target)
    if target_start < 0:
        return containing.strip()
    target_end = target_start + len(target)
    best = containing.strip()
    best_overlap = -1
    for start, end in _sentence_spans(containing):
        overlap = max(0, min(end, target_end) - max(start, target_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = containing[start:end].strip()
    return best


def _candidate_texts(row: dict[str, Any], current: str) -> list[str]:
    candidates: list[str] = []
    for source_key in ("line_candidates", "whole_candidates"):
        for candidate in row.get(source_key, []):
            text = str(candidate.get("text") or "").strip()
            if text and _lexical(text) != _lexical(current) and text not in candidates:
                candidates.append(text)
    return candidates


def build_queue(local_txt: Path, local_refine_audit: Path) -> tuple[list[DeepQueueItem], int]:
    """Group suspect OCR lines by their containing sentence.

    Deep receives the complete sentence, while source_ids preserve which local
    OCR rows contributed evidence. A sentence is queued only when its exact text
    occurs once in the book, keeping the final patch deterministic and safe.
    """

    output = local_txt.read_text(encoding="utf-8")
    pages = _flatten_output_lines(output)
    audit: list[dict[str, Any]] = read_json(local_refine_audit)
    skipped = 0
    groups: dict[tuple[int, str, str], dict[str, Any]] = {}

    for row in audit:
        reasons = list(row.get("reasons") or [])
        if not reasons or row.get("edits"):
            continue
        # Collapsed/symbol-soup OCR must be recovered from pixels, never guessed
        # by a language model at sentence level.
        if "whole_side_catastrophe" in reasons:
            skipped += 1
            continue
        if _trusted_stable(row):
            continue

        page_number, side = int(row["page"][0]), str(row["page"][1])
        current = str(row.get("after") or row.get("before") or "").strip()
        if not current:
            skipped += 1
            continue

        page_lines = pages.get((page_number, side), [])
        target = _resolve_target(page_lines, current)
        if target is None:
            skipped += 1
            continue
        containing = next((line for line in page_lines if target in line), "")
        if not containing:
            skipped += 1
            continue
        if target != current:
            position = containing.find(target)
            if position <= 2 and len(containing) > len(target) * 2:
                skipped += 1
                continue

        sentence = _best_sentence_target(containing, target)
        key = (page_number, side, sentence)
        group = groups.setdefault(
            key,
            {
                "page_lines": page_lines,
                "source_ids": [],
                "reasons": [],
                "candidates": [],
            },
        )
        source_id = str(row["id"])
        if source_id not in group["source_ids"]:
            group["source_ids"].append(source_id)
        for reason in reasons:
            if reason not in group["reasons"]:
                group["reasons"].append(reason)
        for candidate in _candidate_texts(row, current):
            if candidate not in group["candidates"]:
                group["candidates"].append(candidate)

    queue: list[DeepQueueItem] = []
    side_counts: dict[tuple[int, str], int] = defaultdict(int)
    for (page_number, side, sentence), group in groups.items():
        # Sentence replacement is atomic. Refuse ambiguous repeated strings so
        # _patch_exact_lines can never alter the wrong occurrence elsewhere.
        if not sentence or output.count(sentence) != 1:
            skipped += len(group["source_ids"])
            continue
        side_counts[(page_number, side)] += 1
        index = side_counts[(page_number, side)]
        queue.append(
            DeepQueueItem(
                item_id=f"{page_number:03d}-{side}-S{index:03d}",
                page_number=page_number,
                side=side,
                current=sentence,
                output_line=sentence,
                context=_context(group["page_lines"], sentence),
                reasons=group["reasons"],
                candidates=group["candidates"][:16],
                source_ids=group["source_ids"],
            )
        )
    return queue, skipped
