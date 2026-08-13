from pdf_to_epub.deep.gate import apply_ai_ops
from pdf_to_epub.models import DeepQueueItem


def item(current: str, candidates: list[str]) -> DeepQueueItem:
    return DeepQueueItem(
        item_id="001-L-L001",
        page_number=1,
        side="L",
        current=current,
        output_line=current,
        context=current,
        reasons=["whole_pass_disagreement"],
        candidates=candidates,
    )


def test_candidate_supported_high_confidence_is_applied() -> None:
    source = item("Giữ cho nó đơn giân", ["Giữ cho nó đơn giản", "Giữ cho nó đơn giản"])
    corrected, audit = apply_ai_ops(source, [{"old": "giân", "new": "giản", "confidence": 0.99}], 0.97)
    assert corrected == "Giữ cho nó đơn giản"
    assert audit[0]["gate"] == "candidate"
    assert audit[0]["applied"] is True


def test_low_confidence_is_rejected() -> None:
    source = item("thói quen đứt khoát", ["thói quen dứt khoát"])
    corrected, audit = apply_ai_ops(source, [{"old": "đứt", "new": "dứt", "confidence": 0.95}], 0.97)
    assert corrected == source.current
    assert audit[0]["gate"] == "low_confidence"


def test_diacritic_only_can_apply_without_candidate() -> None:
    source = item("khán giả dé hiểu", [])
    corrected, audit = apply_ai_ops(source, [{"old": "dé", "new": "dễ", "confidence": 0.99}], 0.97)
    assert corrected == "khán giả dễ hiểu"
    assert audit[0]["gate"] == "diacritic_only"


def test_exact_substring_uniqueness_preserves_baseline_behavior() -> None:
    source = item("người kế chuyện tự do kết nối", ["người kể chuyện tự do kết nối"])
    corrected, audit = apply_ai_ops(source, [{"old": "kế", "new": "kể", "confidence": 0.99}], 0.97)
    # `kế` occurs once as a word and again inside `kết`; current baseline blocks it.
    assert corrected == source.current
    assert audit[0]["gate"] == "old_not_unique_in_current"


def test_candidate_matching_ignores_attached_punctuation() -> None:
    source = item("DeWitt WWallace,", ["DeWitt Wallace,"])
    corrected, audit = apply_ai_ops(source, [{"old": "WWallace", "new": "Wallace", "confidence": 0.97}], 0.97)
    assert corrected == "DeWitt Wallace,"
    assert audit[0]["gate"] == "candidate"


def test_strong_ocr_majority_vetoes_ai_even_with_one_new_vote() -> None:
    source = item(
        "anh bạn tré",
        ["anh bạn tré", "anh bạn tré", "anh bạn tré", "anh bạn tré", "anh bạn trẻ"],
    )
    corrected, audit = apply_ai_ops(source, [{"old": "tré", "new": "trẻ", "confidence": 0.98}], 0.97)
    assert corrected == source.current
    assert audit[0]["gate"] == "ocr_evidence_strongly_prefers_current"


def test_one_glyph_candidate_change_is_rejected_when_shape_is_unrelated() -> None:
    source = item("thấy 6 ngoài", ["thấy ở ngoài", "thấy ở ngoài", "thấy 6 ngoài"])
    corrected, audit = apply_ai_ops(source, [{"old": "6", "new": "ở", "confidence": 0.98}], 0.97)
    assert corrected == source.current
    assert audit[0]["gate"] == "candidate_change_too_large"


def test_weak_diacritic_guess_at_threshold_is_held_for_review() -> None:
    source = item("động lực đế thực hiện", ["động lực đế thực hiện", "động lực dé thực hiện"])
    corrected, audit = apply_ai_ops(source, [{"old": "đế", "new": "để", "confidence": 0.97}], 0.97)
    assert corrected == source.current
    assert audit[0]["gate"] == "diacritic_guess_without_visual_or_phrase_gain"
