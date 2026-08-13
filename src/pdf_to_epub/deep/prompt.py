"""Strict DeepSeek prompt for token repair and safe OCR word segmentation."""

from __future__ import annotations

import json

from ..models import DeepQueueItem
from .tokens import index_tokens


SYSTEM_INSTRUCTION = """You are a STRICT Vietnamese OCR validator, not a writer.

For every item, inspect CURRENT, TOKEN_IDS, CONTEXT and OCR_ALTERNATIVES.
Only repair clear OCR recognition errors. Never paraphrase, improve style,
translate, summarize, reorder, or rewrite prose. If uncertain, return no
operation for that item.

ALLOWED OPERATIONS:
- replace: one existing token becomes one corrected token.
- segment: one fused OCR token becomes exactly 2 or 3 words.

SAFETY RULES:
1. Every operation MUST include token_id copied from TOKEN_IDS.
2. OLD must exactly equal the text of that token_id.
3. Prefer NEW supported by high-confidence OCR alternatives, especially when
   support comes from different OCR families (line/whole, vie/vie+eng).
4. A segment operation without an exact matching alternative is allowed only
   when removing spaces/diacritics makes OLD and NEW the same glyph sequence.
5. Do not delete text, insert unrelated words, reorder, or rewrite a sentence.
6. Maximum 3 operations per item.
7. confidence is advisory only; do not inflate it to force an edit.

Return JSON ONLY:
{"items":[{"id":"...","ops":[{"kind":"replace","token_id":"t02","old":"exact","new":"exact","confidence":0.95}]}]}
"""


def build_prompt(items: list[DeepQueueItem]) -> str:
    payload = []
    for item in items:
        alternatives = item.candidate_meta or [{"text": text} for text in item.candidates]
        payload.append({
            "id": item.item_id,
            "current": item.current,
            "token_ids": [token.as_dict() for token in index_tokens(item.current)],
            "context": item.context,
            "ocr_alternatives": alternatives,
            "flags": item.reasons,
        })
    return SYSTEM_INSTRUCTION + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
