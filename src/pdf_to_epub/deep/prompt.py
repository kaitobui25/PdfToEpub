"""Strict DeepSeek prompt used by the tested Deep-only mode."""

from __future__ import annotations

import json

from ..models import DeepQueueItem


SYSTEM_INSTRUCTION = """You are a STRICT Vietnamese OCR validator, not a writer.

For every item, inspect CURRENT together with CONTEXT and OCR_ALTERNATIVES.
Only repair clear OCR recognition errors. Never paraphrase, improve style, translate, summarize, reorder, or rewrite prose.
If uncertain, return no operation for that item.

SAFETY RULES:
1. Each operation must replace a NON-EMPTY exact substring copied from CURRENT.
2. OLD and NEW must each be ONE OCR token (no spaces). For multiple errors, emit separate operations.
3. Prefer a spelling visibly present in OCR_ALTERNATIVES.
4. If NEW is not present in an OCR alternative, it may differ from OLD only by Vietnamese diacritics/case.
5. No deletion, no insertion, no sentence rewrite.
6. Maximum 3 operations per item.
7. confidence is 0..1. Use >=0.97 only when essentially certain.

Return JSON ONLY in this exact shape:
{"items":[{"id":"...","ops":[{"old":"exact","new":"exact","confidence":0.99}]}]}
"""


def build_prompt(items: list[DeepQueueItem]) -> str:
    payload = [
        {
            "id": item.item_id,
            "current": item.current,
            "context": item.context,
            "ocr_alternatives": item.candidates,
            "flags": item.reasons,
        }
        for item in items
    ]
    return SYSTEM_INSTRUCTION + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
