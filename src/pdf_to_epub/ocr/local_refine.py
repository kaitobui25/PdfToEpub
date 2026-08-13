"""Evidence-driven local line refinement; no network or language model calls."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import re

from ..models import OCRCandidate, OCRLine
from .scoring import diacritic_only, garbage_score, normalize_token, strip_diacritics


TOKEN_RE = re.compile(r"\S+")


def _line_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, strip_diacritics(a).casefold(), strip_diacritics(b).casefold()).ratio()


def align_whole_candidates(
    anchor_lines: list[OCRLine],
    pass_rows: dict[str, list[tuple[str, float, tuple[int, int, int, int]]]],
    pass_scales: dict[str, float],
) -> None:
    """Attach the nearest text match from every whole-side pass to each anchor line."""

    for line in anchor_lines:
        for source, rows in pass_rows.items():
            if not rows:
                continue
            best = max(rows, key=lambda row: _line_similarity(line.text, row[0]))
            score = _line_similarity(line.text, best[0])
            if score < 0.45:
                continue
            line.whole_candidates.append(
                OCRCandidate(source=source, kind="whole", scale=pass_scales[source], text=best[0], confidence=best[1])
            )


def suspect_reasons(line: OCRLine, lexicon: set[str]) -> list[str]:
    """Explain why a line deserves expensive source-pixel reOCR."""

    reasons: list[str] = []
    texts = [c.text for c in line.whole_candidates if c.text]
    normalized = {strip_diacritics(t).casefold() for t in texts}
    exact = {t.casefold() for t in texts}
    if len(exact) > 1:
        reasons.append("whole_pass_disagreement")
    if len(normalized) < len(exact):
        reasons.append("diacritic_disagreement")
    if line.confidence < 82:
        reasons.append("low_word_conf")
    if garbage_score(line.text) >= 4:
        reasons.append("glyph_garbage")

    if lexicon:
        bad: list[str] = []
        for raw in TOKEN_RE.findall(line.text):
            token = normalize_token(raw)
            if len(token) < 3 or any(ch.isdigit() for ch in token):
                continue
            if token.casefold() not in lexicon and token[0:1].isupper():
                # Names and product terms are common; do not flag them alone.
                continue
            if token.casefold() not in lexicon:
                bad.append(token)
        if bad:
            reasons.append("non_dictionary:" + ",".join(bad[:4]))
    return reasons


def _token_pairs(current: str, candidate: str) -> list[tuple[str, str]]:
    old_tokens = current.split()
    new_tokens = candidate.split()
    if len(old_tokens) != len(new_tokens):
        return []
    return list(zip(old_tokens, new_tokens))


def _apply_unique_token(text: str, old: str, new: str) -> str | None:
    if not old or text.count(old) != 1:
        return None
    return text.replace(old, new, 1)


def refine_line(
    line: OCRLine,
    line_candidates: list[OCRCandidate],
    lexicon: set[str],
) -> OCRLine:
    """Apply strong visual-consensus corrections using precomputed line reOCR.

    ReOCR is scheduled in batches elsewhere; this pure decision function keeps
    correction policy independently testable and easy to tune.
    """

    line.line_candidates = list(line_candidates)
    all_candidates = line.whole_candidates + line.line_candidates

    votes: Counter[tuple[str, str]] = Counter()
    sources: dict[tuple[str, str], set[str]] = {}
    for candidate in all_candidates:
        if not candidate.text:
            continue
        for old, new in _token_pairs(line.text, candidate.text):
            if old == new:
                continue
            old_core, new_core = normalize_token(old), normalize_token(new)
            if not old_core or not new_core:
                continue
            if strip_diacritics(old_core) != strip_diacritics(new_core) and candidate.confidence < 85:
                continue
            key = (old, new)
            votes[key] += 1
            sources.setdefault(key, set()).add(candidate.source)

    current = line.text
    for (old, new), count in votes.most_common():
        if count < 3:
            continue
        old_core, new_core = normalize_token(old), normalize_token(new)
        old_valid = old_core.casefold() in lexicon if lexicon else False
        new_valid = new_core.casefold() in lexicon if lexicon else False

        reason: str | None = None
        if lexicon and not old_valid and new_valid:
            reason = "invalid_to_valid_visual_consensus"
        elif diacritic_only(old_core, new_core) and count >= 4:
            reason = "valid_diacritic_visual_margin"
        elif old.lower() != new.lower() and old_core.casefold() == new_core.casefold() and count >= 4:
            reason = "case_visual_consensus"
        if reason is None:
            continue

        replaced = _apply_unique_token(current, old, new)
        if replaced is None:
            continue
        current = replaced
        line.edits.append({
            "action": "replace_token",
            "old": old,
            "new": new,
            "reason": reason,
            "sources": sorted(sources[(old, new)]),
        })

    line.text = current
    return line
