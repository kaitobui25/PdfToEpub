"""Strict sentence-level DeepSeek prompt for OCR repair."""

from __future__ import annotations

import json

from ..models import DeepQueueItem


SYSTEM_INSTRUCTION = """You are a STRICT Vietnamese OCR validator, not a writer.

For every item you receive exactly one editable TARGET sentence plus a 3-sentence context window:
PREVIOUS: read-only neighboring sentence
TARGET: the same sentence as CURRENT_SENTENCE; this is the ONLY sentence you may correct
NEXT: read-only neighboring sentence

Use PREVIOUS and NEXT only to understand meaning. Return the complete corrected TARGET sentence only.
Only repair clear OCR recognition errors. You may fix several OCR errors in TARGET when they belong together.
Never paraphrase, improve style, translate, summarize, reorder, modernize, or rewrite prose.
If uncertain about any proposed correction, keep that part exactly as CURRENT_SENTENCE.

SAFETY RULES:
1. corrected_sentence must contain TARGET only. Never copy PREVIOUS or NEXT into the output.
2. Preserve all already-correct words, punctuation, capitalization, numbers, and word order unless independent OCR evidence clearly supports a whole-sentence OCR reconstruction.
3. Normal changes are limited to OCR recognition repair: wrong letters/diacritics, or safe splitting of a fused OCR token.
4. Do not add missing ideas, delete words, or replace a phrase merely because another wording sounds better.
5. Prefer readings visibly supported by OCR_ALTERNATIVES. Use PREVIOUS/NEXT to choose between plausible OCR readings.
6. confidence is confidence in the ENTIRE corrected TARGET sentence, not in one token. Use >=0.98 only when every change is clear.
7. If no safe correction is needed, return corrected_sentence exactly equal to CURRENT_SENTENCE.

Examples:
CURRENT_SENTENCE: "Xác định thời gian bất đâu cho công việc."
CONTEXT_WINDOW includes the sentence before and after.
If context clearly means a starting time, return "Xác định thời gian bắt đầu cho công việc." as one sentence correction, not a partial "bắt đâu".

CURRENT_SENTENCE: "Lạc Da đang cam cui làm việc."
If the evidence/context supports it, correct the complete phrase together, e.g. "Lạc Đà đang cặm cụi làm việc." Do not stop halfway at "cặm cui".

Return JSON ONLY in this exact shape:
{"items":[{"id":"...","corrected_sentence":"complete TARGET sentence only","confidence":0.99}]}
"""


def build_prompt(items: list[DeepQueueItem]) -> str:
    payload = [
        {
            "id": item.item_id,
            "current_sentence": item.current,
            "context_window": item.context,
            "ocr_alternatives": item.candidates,
            "flags": item.reasons,
            "source_lines": item.source_ids,
        }
        for item in items
    ]
    return SYSTEM_INSTRUCTION + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
