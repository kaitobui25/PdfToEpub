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


def test_one_ocr_vote_lowers_threshold_to_095() -> None:
    source = item("thói quen đứt khoát", ["thói quen dứt khoát"])
    corrected, audit = apply_ai_ops(source, [{"old": "đứt", "new": "dứt", "confidence": 0.95}], 0.97)
    assert corrected == "thói quen dứt khoát"
    assert audit[0]["gate"] == "ocr_evidence"
    assert audit[0]["required_confidence"] <= 0.95


def test_two_ocr_votes_allow_stronger_evidence_discount() -> None:
    source = item("ông ta uà chờ", ["ông ta và chờ", "ông ta và chờ"])
    corrected, audit = apply_ai_ops(source, [{"old": "uà", "new": "và", "confidence": 0.93}], 0.97)
    assert corrected == "ông ta và chờ"
    assert audit[0]["applied"] is True


def test_single_glyph_change_is_allowed_when_ai_and_ocr_agree() -> None:
    source = item("sáng tạo để thấy 6 ngoài kia", ["sáng tạo để thấy ở ngoài kia"])
    corrected, audit = apply_ai_ops(source, [{"old": "6", "new": "ở", "confidence": 0.98}], 0.97)
    assert corrected == "sáng tạo để thấy ở ngoài kia"
    assert audit[0]["gate"] == "ocr_evidence"


def test_unsupported_non_shape_rewrite_is_rejected_even_at_099() -> None:
    source = item("một câu bình thường", [])
    corrected, audit = apply_ai_ops(source, [{"old": "bình", "new": "hoàn", "confidence": 0.99}], 0.97)
    assert corrected == source.current
    assert audit[0]["gate"] == "unsupported_new"


def test_shape_preserving_diacritic_without_candidate_requires_098() -> None:
    source = item("khán giả dé hiểu", [])
    corrected, audit = apply_ai_ops(source, [{"old": "dé", "new": "dễ", "confidence": 0.98}], 0.97)
    assert corrected == "khán giả dễ hiểu"
    assert audit[0]["gate"] == "shape_preserving"


def test_token_boundary_does_not_confuse_ke_with_ket() -> None:
    source = item("người kế chuyện tự do kết nối", ["người kể chuyện tự do kết nối"])
    corrected, audit = apply_ai_ops(source, [{"old": "kế", "new": "kể", "confidence": 0.98}], 0.97)
    assert corrected == "người kể chuyện tự do kết nối"
    assert audit[0]["applied"] is True


def test_segmentation_can_use_shape_preservation_without_candidate() -> None:
    source = item("4. Vidu về thói quen", [])
    corrected, audit = apply_ai_ops(
        source,
        [{"kind": "segment", "old": "Vidu", "new": "Ví dụ", "confidence": 0.99}],
        0.97,
    )
    assert corrected == "4. Ví dụ về thói quen"
    assert audit[0]["gate"] == "segmentation_shape"


def test_segmentation_with_ocr_candidate_can_apply_at_095() -> None:
    source = item("4. Vidu về thói quen", ["4. Ví dụ về thói quen"])
    corrected, audit = apply_ai_ops(
        source,
        [{"kind": "segment", "old": "Vidu", "new": "Ví dụ", "confidence": 0.95}],
        0.97,
    )
    assert corrected == "4. Ví dụ về thói quen"
    assert audit[0]["gate"] == "segmentation_evidence"


def test_unrelated_segmentation_is_rejected() -> None:
    source = item("một abc ví dụ", [])
    corrected, audit = apply_ai_ops(
        source,
        [{"kind": "segment", "old": "abc", "new": "hai chữ", "confidence": 0.99}],
        0.97,
    )
    assert corrected == source.current
    assert audit[0]["gate"] == "segmentation_without_visual_or_shape_support"
