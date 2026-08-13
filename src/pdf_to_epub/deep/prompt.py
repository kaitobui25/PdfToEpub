"""Closed-choice DeepSeek prompt for constrained OCR validation."""

from __future__ import annotations

import json

from ..models import DeepQueueItem


SYSTEM_INSTRUCTION = """You are a STRICT Vietnamese OCR validator, not a writer.

Local code already generated every replacement you are allowed to choose.
You MUST NOT invent, rewrite, spell out, merge, split, or propose any text that
is not present in CHOICE_SETS.

For every CHOICE_SET, choose exactly ONE choice_id:
- KEEP means preserve the OCR token exactly as it is.
- C1/C2/... means use exactly that candidate text.

Use CURRENT, CONTEXT, OCR_ALTERNATIVES and candidate metadata to decide.
Prefer KEEP whenever the context does not clearly favor one listed candidate.
Never paraphrase or improve style.  Proper names, numbers and unusual wording
may be legitimate, so do not force a dictionary-looking choice.

IMPORTANT:
1. Return one selection for EVERY CHOICE_SET, even when the answer is KEEP.
2. Copy token_id and choice_id exactly.  Do not return OLD/NEW text.
3. You may not create a choice_id which is absent from CHOICE_SETS.
4. Do not omit a token merely because you are uncertain; choose KEEP instead.

Return JSON ONLY in this exact shape:
{"items":[{"id":"...","selections":[{"token_id":"t02","choice_id":"KEEP"},{"token_id":"t05","choice_id":"C1"}]}]}
"""


def build_prompt(items: list[DeepQueueItem]) -> str:
    payload = []
    for item in items:
        alternatives = item.candidate_meta or [{"text": text} for text in item.candidates]
        payload.append({
            "id": item.item_id,
            "current": item.current,
            "context": item.context,
            "ocr_alternatives": alternatives,
            "flags": item.reasons,
            "choice_sets": item.choice_sets,
        })
    return SYSTEM_INSTRUCTION + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
