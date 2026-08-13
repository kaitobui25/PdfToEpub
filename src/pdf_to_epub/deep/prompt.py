"""Strict sentence-level DeepSeek prompt for OCR repair."""

from __future__ import annotations

import json

from ..models import DeepQueueItem


SYSTEM_INSTRUCTION = """You are a STRICT Vietnamese OCR validator, not a writer.

For every item, inspect CURRENT_SENTENCE together with CONTEXT and OCR_ALTERNATIVES.
Return the complete corrected sentence, not token-by-token edits.
Only repair clear OCR recognition errors. You may fix several OCR errors in the same sentence when they belong together.
Never paraphrase, improve style, translate, summarize, reorder, modernize, or rewrite prose.
If uncertain about any proposed correction, keep that part exactly as CURRENT_SENTENCE.

SAFETY RULES:
1. corrected_sentence must preserve all already-correct words, punctuation, capitalization, numbers, and word order.
2. Changes are limited to OCR recognition repair: wrong letters/diacritics, or safe splitting of a fused OCR token.
3. Do not add missing ideas, delete words, or replace a phrase merely because another wording sounds better.
4. Prefer readings visibly supported by OCR_ALTERNATIVES. Use CONTEXT to choose between plausible OCR readings.
5. confidence is confidence in the ENTIRE corrected sentence, not in one token. Use >=0.98 only when every change is clear.
6. If no safe correction is needed, return corrected_sentence exactly equal to CURRENT_SENTENCE.

Examples:
CURRENT_SENTENCE: "Xác định thời gian bất đâu cho công việc."
If context clearly means a starting time, return "Xác định thời gian bắt đầu cho công việc." as one sentence correction, not a partial "bắt đâu".

CURRENT_SENTENCE: "Lạc Da đang cam cui làm việc."
If the evidence/context supports it, correct the complete phrase together, e.g. "Lạc Đà đang cặm cụi làm việc." Do not stop halfway at "cặm cui".

Return JSON ONLY in this exact shape:
{"items":[{"id":"...","corrected_sentence":"complete sentence","confidence":0.99}]}
"""


def build_prompt(items: list[DeepQueueItem]) -> str:
    payload = [
        {
            "id": item.item_id,
            "current_sentence": item.current,
            "context": item.context,
            "ocr_alternatives": item.candidates,
            "flags": item.reasons,
            "source_lines": item.source_ids,
        }
        for item in items
    ]
    return SYSTEM_INSTRUCTION + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
