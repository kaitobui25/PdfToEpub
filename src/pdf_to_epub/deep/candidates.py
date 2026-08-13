"""Constrained local candidate generation for Vietnamese OCR repair.

The language model is not allowed to invent replacement text.  This module
builds a small choice list from evidence already available locally:

* aligned Tesseract alternatives,
* words/phrases that already occur elsewhere in the same book/range,
* Tesseract's Vietnamese lexicon when it is available,
* conservative one-glyph neighbours for explicitly suspicious tokens.

The model later chooses KEEP/C1/C2/... only.  Candidate strength is a local
property and is deliberately independent of model self-reported confidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log1p
import re
from typing import Iterable

from ..models import DeepQueueItem
from ..ocr.scoring import normalize_token, shape_key, strip_diacritics
from .evidence import summarize_evidence
from .tokens import TokenRef, index_tokens

MARKER_RE = re.compile(r"^===== PDF\d{3}-[LR] =====$")


@dataclass(slots=True)
class BookStats:
    unigrams: Counter[str]
    phrases: Counter[str]
    shape_words: dict[str, tuple[str, ...]]
    shape_phrases: dict[str, tuple[str, ...]]
    lexicon: set[str]

    def frequency(self, value: str) -> int:
        words = _words(value)
        if len(words) == 1:
            return self.unigrams[words[0]]
        return self.phrases[" ".join(words)]

    def lexical_valid(self, value: str) -> bool:
        words = _words(value)
        if not words:
            return False
        phrase = " ".join(words)
        if len(words) > 1 and self.phrases[phrase] > 0:
            return True
        # A book token must recur before it is allowed to act as its own
        # dictionary.  This prevents a one-off OCR error from validating itself.
        return all(word in self.lexicon or self.unigrams[word] >= 2 for word in words)


def _words(text: str) -> list[str]:
    return [normalize_token(token.text) for token in index_tokens(text) if normalize_token(token.text)]


def _match_case(old: str, value: str) -> str:
    if not value:
        return value
    if old.isupper():
        return value.upper()
    if old[:1].isupper():
        return value[:1].upper() + value[1:]
    return value


def _base_edit_distance(left: str, right: str, limit: int = 1) -> int:
    """Small Levenshtein distance after removing Vietnamese diacritics."""

    left = strip_diacritics(normalize_token(left)).casefold()
    right = strip_diacritics(normalize_token(right)).casefold()
    if left == right:
        return 0
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        row_min = i
        for j, b in enumerate(right, 1):
            value = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (a != b),
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def build_book_stats(text: str, lexicon: Iterable[str] = ()) -> BookStats:
    unigrams: Counter[str] = Counter()
    phrases: Counter[str] = Counter()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or MARKER_RE.match(line):
            continue
        words = _words(line)
        unigrams.update(words)
        for width in (2, 3):
            for index in range(0, len(words) - width + 1):
                phrases[" ".join(words[index : index + width])] += 1

    normalized_lexicon = {normalize_token(word) for word in lexicon if normalize_token(word)}
    word_shapes: dict[str, set[str]] = defaultdict(set)
    for word in set(unigrams) | normalized_lexicon:
        key = shape_key(word)
        if key:
            word_shapes[key].add(word)

    phrase_shapes: dict[str, set[str]] = defaultdict(set)
    for phrase in phrases:
        key = shape_key(phrase)
        if key:
            phrase_shapes[key].add(phrase)

    def word_order(value: str) -> tuple[int, str]:
        return (-unigrams[value], value)

    def phrase_order(value: str) -> tuple[int, str]:
        return (-phrases[value], value)

    return BookStats(
        unigrams=unigrams,
        phrases=phrases,
        shape_words={key: tuple(sorted(values, key=word_order)) for key, values in word_shapes.items()},
        shape_phrases={key: tuple(sorted(values, key=phrase_order)) for key, values in phrase_shapes.items()},
        lexicon=normalized_lexicon,
    )


def _non_dictionary_tokens(reasons: list[str]) -> set[str]:
    result: set[str] = set()
    for reason in reasons:
        if not str(reason).startswith("non_dictionary:"):
            continue
        payload = str(reason).split(":", 1)[1]
        for token in re.split(r"[,|]", payload):
            token = normalize_token(token)
            if token:
                result.add(token)
    return result


def _aligned_alternatives(item: DeepQueueItem) -> dict[str, set[str]]:
    current = index_tokens(item.current)
    result: dict[str, set[str]] = defaultdict(set)
    for candidate in item.candidate_meta:
        alternative = index_tokens(str(candidate.get("text") or ""))
        if len(alternative) != len(current):
            continue
        for old, new in zip(current, alternative):
            if normalize_token(old.text) != normalize_token(new.text):
                result[old.token_id].add(new.text)
    return result


def _candidate_strength(
    item: DeepQueueItem,
    old: str,
    new: str,
    stats: BookStats,
    source_tags: set[str],
) -> tuple[str, dict[str, object]]:
    evidence = summarize_evidence(item, old, new)
    new_frequency = stats.frequency(new)
    old_frequency = stats.frequency(old)
    lexical_valid = stats.lexical_valid(new)
    distance = _base_edit_distance(old, new, limit=1)
    phrase_frequency = new_frequency if len(_words(new)) > 1 else 0

    # A non-dictionary replacement which is itself neither lexical nor observed
    # elsewhere in the book is never trusted merely because several correlated
    # OCR passes repeat it.  This is the Vidu -> Vidụ safety valve.
    if evidence.old_non_dictionary and not lexical_valid and new_frequency == 0:
        strength = "weak"
    elif evidence.segmented and evidence.shape_preserving and phrase_frequency > 0 and lexical_valid:
        strength = "strong"
    elif evidence.new_score >= 1.60 and lexical_valid:
        strength = "strong"
    elif evidence.old_non_dictionary and evidence.new_score >= 0.80 and lexical_valid:
        strength = "strong"
    elif (
        evidence.shape_preserving
        and lexical_valid
        and new_frequency >= 3
        and new_frequency >= max(3, old_frequency * 2)
    ):
        strength = "strong"
    elif lexical_valid and (
        evidence.new_score > 0
        or evidence.shape_preserving
        or "ocr_shape_expand" in source_tags
        or (evidence.old_non_dictionary and distance <= 1)
    ):
        strength = "medium"
    elif new_frequency >= 2 and distance <= 1:
        strength = "medium"
    else:
        strength = "weak"

    metadata: dict[str, object] = {
        "strength": strength,
        "visual_score": round(evidence.new_score, 4),
        "book_frequency": new_frequency,
        "old_book_frequency": old_frequency,
        "lexical_valid": lexical_valid,
        "shape_preserving": evidence.shape_preserving,
        "segmented": evidence.segmented,
        "base_edit_distance": distance,
        "source_tags": sorted(source_tags),
    }
    return strength, metadata


def _sort_key(choice: dict[str, object]) -> tuple[object, ...]:
    rank = {"strong": 0, "medium": 1, "weak": 2}.get(str(choice.get("strength")), 3)
    return (
        rank,
        -float(choice.get("visual_score") or 0.0),
        -int(choice.get("book_frequency") or 0),
        str(choice.get("text") or ""),
    )


def build_choice_sets(item: DeepQueueItem, stats: BookStats, max_tokens: int = 5, max_choices: int = 5) -> list[dict[str, object]]:
    """Build constrained replacement choices for the most suspicious tokens."""

    tokens = index_tokens(item.current)
    aligned = _aligned_alternatives(item)
    explicit_bad = _non_dictionary_tokens(item.reasons)
    broad_suspect = bool(set(item.reasons) & {"whole_pass_disagreement", "diacritic_disagreement", "low_word_conf"})

    token_priority: list[tuple[float, TokenRef]] = []
    for token in tokens:
        old = normalize_token(token.text)
        score = 0.0
        if old in explicit_bad:
            score += 4.0
        if aligned.get(token.token_id):
            score += 3.0
        if broad_suspect and stats.unigrams[old] <= 1:
            key = shape_key(old)
            alternatives = [word for word in stats.shape_words.get(key, ()) if word != old]
            if alternatives:
                score += 1.2
        if "low_word_conf" in item.reasons and stats.unigrams[old] <= 1:
            score += 0.8
        if score > 0:
            token_priority.append((score, token))

    token_priority.sort(key=lambda value: (-value[0], value[1].start))
    result: list[dict[str, object]] = []

    for _, token in token_priority[:max_tokens]:
        old = token.text
        old_norm = normalize_token(old)
        proposals: dict[str, set[str]] = defaultdict(set)

        # 1) Exact token spellings seen in aligned OCR alternatives.
        for value in aligned.get(token.token_id, set()):
            normalized = normalize_token(value)
            if normalized and normalized != old_norm:
                proposals[_match_case(old, normalized)].add("ocr_direct")

        # 2) Same-glyph words already seen in the book or known by Tesseract.
        old_shape = shape_key(old)
        for value in stats.shape_words.get(old_shape, ())[:12]:
            if value != old_norm:
                proposals[_match_case(old, value)].add("shape_lexicon")

        # 3) A fused token may match an existing 2/3-word phrase elsewhere in
        # the same book.  This deterministically generates Vidu -> Ví dụ.
        for phrase in stats.shape_phrases.get(old_shape, ())[:8]:
            proposals[_match_case(old, phrase)].add("book_phrase")

        # 4) If OCR offered an unaccented/alternate glyph seed, expand that
        # seed through the local/lexical shape index (uạn -> van -> vạn).
        for seed in list(aligned.get(token.token_id, set())):
            seed_shape = shape_key(seed)
            if not seed_shape:
                continue
            for value in stats.shape_words.get(seed_shape, ())[:8]:
                if value != old_norm:
                    proposals[_match_case(old, value)].add("ocr_shape_expand")

        # 5) Explicitly suspicious OCR tokens may use one base-glyph edit to a
        # word which actually occurs elsewhere in this book.  Book frequency
        # ranks the options; Deep later chooses only among these options.
        if old_norm in explicit_bad or "low_word_conf" in item.reasons:
            neighbours = [
                (frequency, word)
                for word, frequency in stats.unigrams.items()
                if word != old_norm and frequency > 0 and _base_edit_distance(old_norm, word, limit=1) <= 1
            ]
            neighbours.sort(key=lambda value: (-value[0], value[1]))
            for _, value in neighbours[:12]:
                proposals[_match_case(old, value)].add("book_edit_neighbor")

        choices: list[dict[str, object]] = []
        for new, tags in proposals.items():
            if normalize_token(new) == old_norm or new == old:
                continue
            strength, metadata = _candidate_strength(item, old, new, stats, tags)
            choices.append({
                "text": new,
                "kind": "segment" if len(_words(new)) > 1 else "replace",
                **metadata,
            })

        choices.sort(key=_sort_key)
        choices = choices[:max_choices]
        if not choices:
            continue

        options: list[dict[str, object]] = [{
            "choice_id": "KEEP",
            "text": old,
            "kind": "keep",
            "strength": "keep",
        }]
        for index, choice in enumerate(choices, 1):
            options.append({"choice_id": f"C{index}", **choice})
        result.append({"token_id": token.token_id, "old": old, "choices": options})

    return result
