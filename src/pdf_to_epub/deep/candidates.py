"""Constrained local candidate generation for Vietnamese OCR repair.

Deep never invents replacement text.  Local code builds KEEP/C1/C2/... from:

* token-position-aligned Tesseract alternatives,
* words/phrases recurring elsewhere in the same book/range,
* Tesseract's Vietnamese lexicon when available,
* a bounded Vietnamese diacritic-family fallback for rare one-vowel tokens,
* conservative one-base-glyph neighbours for explicitly suspicious OCR.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Iterable

from ..models import DeepQueueItem
from ..ocr.scoring import normalize_token, shape_key, strip_diacritics
from .tokens import TokenRef, index_tokens

MARKER_RE = re.compile(r"^===== PDF\d{3}-[LR] =====$")
FAMILY_WEIGHTS = {"line_v": 1.00, "line_ve": 1.00, "whole_v": 0.72, "whole_ve": 0.72}
DIACRITIC_VARIANTS = {
    "a": "aàáảãạăằắẳẵặâầấẩẫậ",
    "e": "eèéẻẽẹêềếểễệ",
    "i": "iìíỉĩị",
    "o": "oòóỏõọôồốổỗộơờớởỡợ",
    "u": "uùúủũụưừứửữự",
    "y": "yỳýỷỹỵ",
}


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
        # A one-off book token does not validate itself.  It must either belong
        # to Tesseract's Vietnamese lexicon or recur in this book/range.
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
            value = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
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

    return BookStats(
        unigrams=unigrams,
        phrases=phrases,
        shape_words={
            key: tuple(sorted(values, key=lambda value: (-unigrams[value], value)))
            for key, values in word_shapes.items()
        },
        shape_phrases={
            key: tuple(sorted(values, key=lambda value: (-phrases[value], value)))
            for key, values in phrase_shapes.items()
        },
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


def _family(candidate: dict[str, object]) -> str:
    kind = str(candidate.get("kind") or "whole").casefold()
    source = str(candidate.get("source") or "").casefold()
    is_ve = "_ve_" in source or source.startswith("line_ve") or source.startswith("fallback_ve")
    return f"{'line' if kind == 'line' else 'whole'}_{'ve' if is_ve else 'v'}"


def _confidence(candidate: dict[str, object]) -> float:
    try:
        return max(0.0, min(100.0, float(candidate.get("conf") or candidate.get("confidence") or 0.0))) / 100.0
    except (TypeError, ValueError):
        return 0.0


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


def _aligned_visual_score(item: DeepQueueItem, token_id: str, new: str) -> float:
    """Score NEW only at the target token position, never elsewhere in a line."""

    current = index_tokens(item.current)
    target_index = next((i for i, token in enumerate(current) if token.token_id == token_id), None)
    if target_index is None or len(_words(new)) != 1:
        return 0.0

    wanted = normalize_token(new)
    by_family: dict[str, float] = {}
    for candidate in item.candidate_meta:
        alternative = index_tokens(str(candidate.get("text") or ""))
        if len(alternative) != len(current):
            continue
        if normalize_token(alternative[target_index].text) != wanted:
            continue
        family = _family(candidate)
        by_family[family] = max(by_family.get(family, 0.0), _confidence(candidate))
    return sum(FAMILY_WEIGHTS.get(family, 0.65) * conf for family, conf in by_family.items())


def _single_vowel_diacritic_variants(old: str) -> set[str]:
    """Enumerate only a one-vowel Vietnamese accent family.

    This bounded fallback is useful when the installed Tesseract DAWG cannot be
    decoded.  It cannot alter consonants or word length, and every resulting
    candidate still needs two closed-choice votes unless stronger evidence exists.
    """

    core = normalize_token(old)
    positions: list[tuple[int, str]] = []
    for index, char in enumerate(core):
        base = strip_diacritics(char).casefold()
        if base in DIACRITIC_VARIANTS:
            positions.append((index, base))
    if len(positions) != 1:
        return set()
    index, base = positions[0]
    result: set[str] = set()
    for replacement in DIACRITIC_VARIANTS[base]:
        value = core[:index] + replacement + core[index + 1 :]
        if value != core:
            result.add(_match_case(old, value))
    return result


def _candidate_strength(
    item: DeepQueueItem,
    token_id: str,
    old: str,
    new: str,
    stats: BookStats,
    source_tags: set[str],
) -> tuple[str, dict[str, object]]:
    visual_score = _aligned_visual_score(item, token_id, new)
    new_frequency = stats.frequency(new)
    old_frequency = stats.frequency(old)
    lexical_valid = stats.lexical_valid(new)
    domain_valid = lexical_valid or new_frequency > 0
    old_non_dictionary = normalize_token(old) in _non_dictionary_tokens(item.reasons)
    shape_preserving = bool(shape_key(old)) and shape_key(old) == shape_key(new)
    segmented = len(_words(new)) > 1
    distance = _base_edit_distance(old, new, limit=1)
    phrase_frequency = new_frequency if segmented else 0

    if old_non_dictionary and not domain_valid and "raw_diacritic_shape" not in source_tags:
        strength = "weak"
    elif segmented and shape_preserving and phrase_frequency > 0:
        strength = "strong"
    elif visual_score >= 1.60 and domain_valid:
        strength = "strong"
    elif old_non_dictionary and visual_score >= 0.80 and domain_valid:
        strength = "strong"
    elif (
        shape_preserving
        and domain_valid
        and new_frequency >= 3
        and new_frequency >= max(3, old_frequency * 2)
    ):
        strength = "strong"
    elif "raw_diacritic_shape" in source_tags and shape_preserving:
        strength = "medium"
    elif domain_valid and (
        visual_score > 0
        or shape_preserving
        or "ocr_shape_expand" in source_tags
        or (old_non_dictionary and distance <= 1)
    ):
        strength = "medium"
    elif new_frequency >= 1 and distance <= 1:
        strength = "medium"
    else:
        strength = "weak"

    return strength, {
        "strength": strength,
        "visual_score": round(visual_score, 4),
        "book_frequency": new_frequency,
        "old_book_frequency": old_frequency,
        "lexical_valid": lexical_valid,
        "shape_preserving": shape_preserving,
        "segmented": segmented,
        "base_edit_distance": distance,
        "source_tags": sorted(source_tags),
    }


def _source_priority(choice: dict[str, object]) -> int:
    tags = set(choice.get("source_tags") or [])
    if tags & {"book_phrase", "ocr_direct"}:
        return 0
    if "ocr_shape_expand" in tags:
        return 1
    if "shape_lexicon" in tags:
        return 2
    if "raw_diacritic_shape" in tags:
        return 3
    return 4


def _sort_key(choice: dict[str, object]) -> tuple[object, ...]:
    rank = {"strong": 0, "medium": 1, "weak": 2}.get(str(choice.get("strength")), 3)
    return (
        rank,
        _source_priority(choice),
        -float(choice.get("visual_score") or 0.0),
        -int(choice.get("book_frequency") or 0),
        str(choice.get("text") or ""),
    )


def _downgrade_ambiguous_strong(choices: list[dict[str, object]]) -> None:
    strong = [choice for choice in choices if choice.get("strength") == "strong"]
    if len(strong) <= 1:
        return
    # Multiple locally plausible answers means semantic context must win twice.
    for choice in strong:
        choice["strength"] = "medium"
        choice["ambiguous_strong"] = True


def build_choice_sets(
    item: DeepQueueItem,
    stats: BookStats,
    max_tokens: int = 5,
    max_choices: int = 6,
) -> list[dict[str, object]]:
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
            alternatives = [word for word in stats.shape_words.get(shape_key(old), ()) if word != old]
            if alternatives or "diacritic_disagreement" in item.reasons:
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

        for value in aligned.get(token.token_id, set()):
            normalized = normalize_token(value)
            if normalized and normalized != old_norm:
                proposals[_match_case(old, normalized)].add("ocr_direct")

        old_shape = shape_key(old)
        for value in stats.shape_words.get(old_shape, ())[:20]:
            if value != old_norm:
                proposals[_match_case(old, value)].add("shape_lexicon")

        for phrase in stats.shape_phrases.get(old_shape, ())[:10]:
            proposals[_match_case(old, phrase)].add("book_phrase")

        for seed in aligned.get(token.token_id, set()):
            seed_shape = shape_key(seed)
            if not seed_shape:
                continue
            for value in stats.shape_words.get(seed_shape, ())[:12]:
                if value != old_norm:
                    proposals[_match_case(old, value)].add("ocr_shape_expand")

        if (
            stats.unigrams[old_norm] <= 1
            and "diacritic_disagreement" in item.reasons
            and len(_single_vowel_diacritic_variants(old))
        ):
            for value in _single_vowel_diacritic_variants(old):
                proposals[value].add("raw_diacritic_shape")

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
            _, metadata = _candidate_strength(item, token.token_id, old, new, stats, tags)
            choices.append({
                "text": new,
                "kind": "segment" if len(_words(new)) > 1 else "replace",
                **metadata,
            })

        _downgrade_ambiguous_strong(choices)
        choices.sort(key=_sort_key)
        # Rare one-vowel diacritic disputes may need the full accent family so a
        # valid rare word such as nhẫn is not crowded out by frequent homographs.
        limit = 20 if stats.unigrams[old_norm] <= 1 and "diacritic_disagreement" in item.reasons else max_choices
        choices = choices[:limit]
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
