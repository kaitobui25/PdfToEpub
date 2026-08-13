"""Build a compact Deep-only queue from local OCR audit evidence."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any

from ..jsonio import read_json
from ..models import DeepQueueItem


MARKER_RE = re.compile(r"^===== PDF(\d{3})-([LR]) =====$")
LEXICAL_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)


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


def _context(lines: list[str], target: str, radius_chars: int = 260) -> str:
    joined = "\n\n".join(lines)
    pos = joined.find(target)
    if pos < 0:
        return joined[: radius_chars * 2]
    start = max(0, pos - radius_chars)
    end = min(len(joined), pos + len(target) + radius_chars)
    return joined[start:end]


def _lexical(text: str) -> str:
    """Compare OCR content while ignoring bullets/ornaments and punctuation."""

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
    """Avoid paying AI latency for obvious false-positive review flags.

    This reproduces the pre-fix queue behavior: locally edited lines never go to
    DeepSeek, and long low-confidence/proper-word lines are skipped when every
    high-confidence OCR view agrees on the same lexical content.
    """

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
    """Find the exact substring that survived cleanup/paragraph joining."""

    if any(current in line for line in page_lines):
        return current

    # Local cleanup can remove bullets or one stray OCR symbol at line start.
    stripped = re.sub(r"^[^0-9A-Za-zÀ-ỹĐđ]+", "", current).strip()
    if stripped and any(stripped in line for line in page_lines):
        return stripped
    return None


def build_queue(local_txt: Path, local_refine_audit: Path) -> tuple[list[DeepQueueItem], int]:
    """Return unresolved suspicious lines plus count of unsafe/unmapped rows."""

    output = local_txt.read_text(encoding="utf-8")
    pages = _flatten_output_lines(output)
    audit: list[dict[str, Any]] = read_json(local_refine_audit)
    queue: list[DeepQueueItem] = []
    skipped = 0

    for row in audit:
        if not row.get("reasons") or row.get("edits"):
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
        # If cleanup changed the visible line AND OCR evidence itself is unstable,
        # the pre-fix baseline leaves that row alone instead of guessing a patch.
        if target is None:
            skipped += 1
            continue
        if target != current:
            containing = next((line for line in page_lines if target in line), "")
            position = containing.find(target) if containing else -1
            # Two baseline rows lost a leading bullet and were also joined into a
            # much longer paragraph.  Patching a prefix fragment there is less
            # safe than a unique mid/suffix fragment, so leave them for review.
            if containing and position <= 2 and len(containing) > len(target) * 2:
                skipped += 1
                continue

        candidates: list[str] = []
        for source_key in ("line_candidates", "whole_candidates"):
            for candidate in row.get(source_key, []):
                text = str(candidate.get("text") or "").strip()
                if text and _lexical(text) != _lexical(current) and text not in candidates:
                    candidates.append(text)
        queue.append(
            DeepQueueItem(
                item_id=str(row["id"]),
                page_number=page_number,
                side=side,
                current=current,
                output_line=target,
                context=_context(page_lines, target),
                reasons=list(row.get("reasons") or []),
                candidates=candidates[:8],
            )
        )
    return queue, skipped
