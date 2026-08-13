import json

from pdf_to_epub.deep.gate import apply_ai_ops, apply_ai_sentence
from pdf_to_epub.deep.polish import _render_corrected_target
from pdf_to_epub.deep.queue import build_queue
from pdf_to_epub.models import DeepQueueItem


def item(current: str, candidates: list[str], raw_target: str | None = None) -> DeepQueueItem:
    return DeepQueueItem(
        item_id="001-L-L001",
        page_number=1,
        side="L",
        current=current,
        output_line=raw_target if raw_target is not None else current,
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


def test_sentence_repairs_bat_dau_as_one_atomic_phrase() -> None:
    source = item(
        "Xác định thời gian bất đâu cho công việc.",
        ["Xác định thời gian bắt đâu cho công việc."],
    )
    corrected, audit = apply_ai_sentence(
        source,
        {
            "corrected_sentence": "Xác định thời gian bắt đầu cho công việc.",
            "confidence": 0.99,
        },
        0.97,
    )
    assert corrected == "Xác định thời gian bắt đầu cho công việc."
    assert len(audit) == 1
    assert audit[0]["old"] == "bất đâu"
    assert audit[0]["new"] == "bắt đầu"
    assert audit[0]["applied"] is True


def test_sentence_repairs_cam_cui_together() -> None:
    source = item(
        "Lạc Da đang cam cui làm việc.",
        ["Lạc Đà đang cặm cui làm việc."],
    )
    corrected, audit = apply_ai_sentence(
        source,
        {
            "corrected_sentence": "Lạc Đà đang cặm cụi làm việc.",
            "confidence": 0.99,
        },
        0.97,
    )
    assert corrected == "Lạc Đà đang cặm cụi làm việc."
    assert len(audit) == 2
    assert all(change["applied"] for change in audit)


def test_sentence_context_can_fix_lau_dai_shape_preserving_phrase() -> None:
    source = item(
        "Về lau đài, những ranh giới này sẽ bảo vệ bạn.",
        ["Về lâu đài, những ranh giới này sẽ bảo vệ bạn."],
    )
    corrected, audit = apply_ai_sentence(
        source,
        {
            "corrected_sentence": "Về lâu dài, những ranh giới này sẽ bảo vệ bạn.",
            "confidence": 0.99,
        },
        0.97,
    )
    assert corrected == "Về lâu dài, những ranh giới này sẽ bảo vệ bạn."
    assert audit[0]["old"] == "lau đài"
    assert audit[0]["new"] == "lâu dài"
    assert audit[0]["gate"] == "sentence_shape_preserving"


def test_sentence_uses_full_candidate_to_choose_do_rac_nao_bo() -> None:
    source = item(
        "Hãy tạo một danh sách “dé rác não bổ”.",
        ["Hãy tạo một danh sách “đổ rác não bộ”."],
    )
    corrected, audit = apply_ai_sentence(
        source,
        {
            "corrected_sentence": "Hãy tạo một danh sách “đổ rác não bộ”.",
            "confidence": 0.95,
        },
        0.97,
    )
    assert corrected == "Hãy tạo một danh sách “đổ rác não bộ”."
    assert all(change["applied"] for change in audit)


def test_sentence_rewrite_without_evidence_is_rejected() -> None:
    source = item("Đây là một câu bình thường.", [])
    corrected, audit = apply_ai_sentence(
        source,
        {
            "corrected_sentence": "Đây là một câu hoàn hảo.",
            "confidence": 0.99,
        },
        0.97,
    )
    assert corrected == source.current
    assert audit[0]["gate"] == "unsupported_sentence_span"


def test_sentence_is_all_or_none_when_one_span_fails() -> None:
    source = item(
        "Xác định thời gian bất đâu và một việc xấu.",
        ["Xác định thời gian bắt đâu và một việc xấu."],
    )
    corrected, audit = apply_ai_sentence(
        source,
        {
            "corrected_sentence": "Xác định thời gian bắt đầu và một việc tốt.",
            "confidence": 0.99,
        },
        0.97,
    )
    assert corrected == source.current
    assert any(change["gate"] == "unsupported_sentence_span" for change in audit)
    assert not any(change["applied"] for change in audit)


def test_structural_sentence_repair_can_pass_with_direct_whole_ocr_candidate() -> None:
    source = item("Mọi việc đều tốt.", ["Mọi việc đều rất tốt."])
    corrected, audit = apply_ai_sentence(
        source,
        {"corrected_sentence": "Mọi việc đều rất tốt.", "confidence": 0.98},
        0.97,
    )
    assert corrected == "Mọi việc đều rất tốt."
    assert audit[0]["gate"] == "sentence_whole_ocr_candidate"
    assert audit[0]["applied"] is True


def test_sentence_patch_preserves_original_ocr_line_breaks() -> None:
    current = "Điều này làm cho bạn bị xao nhãăng."
    raw_target = "Điều này làm cho\n\nbạn bị xao nhãăng."
    source = item(current, ["Điều này làm cho bạn bị xao nhãng."], raw_target=raw_target)
    corrected, changes = apply_ai_sentence(
        source,
        {"corrected_sentence": "Điều này làm cho bạn bị xao nhãng.", "confidence": 0.99},
        0.97,
    )
    rendered = _render_corrected_target(source, corrected, changes)
    assert corrected == "Điều này làm cho bạn bị xao nhãng."
    assert rendered == "Điều này làm cho\n\nbạn bị xao nhãng."


def test_queue_targets_named_suspect_and_supplies_three_sentence_window(tmp_path) -> None:
    local_txt = tmp_path / "local.txt"
    local_txt.write_text(
        "===== PDF084-L =====\n\n"
        "Bạn đang làm việc rất tập trung.\n\n"
        "Điều này làm cho\n\n"
        "bạn bị xao nhãăng. Vậy thì cách tốt nhất để không\n\n"
        "bị xao nhãng đó là ngăn ngừa trước khi điều đó xảy ra.\n\n"
        "Sau đây là một cách khác.\n",
        encoding="utf-8",
    )
    audit_path = tmp_path / "local_refine_audit.json"
    audit_path.write_text(
        json.dumps(
            [
                {
                    "id": "084-L-L011",
                    "page": [84, "L"],
                    "before": "bạn bị xao nhãăng. Vậy thì cách tốt nhất để không",
                    "after": "bạn bị xao nhãăng. Vậy thì cách tốt nhất để không",
                    "reasons": [
                        "whole_pass_disagreement",
                        "diacritic_disagreement",
                        "non_dictionary:nhãăng",
                    ],
                    "edits": [],
                    "line_candidates": [
                        {
                            "text": "bạn bị xao nhãng. Vậy thì cách tốt nhất để không",
                            "conf": 92.0,
                        }
                    ],
                    "whole_candidates": [],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    queue, skipped = build_queue(local_txt, audit_path)

    assert skipped == []
    assert len(queue) == 1
    queued = queue[0]
    assert queued.current == "Điều này làm cho bạn bị xao nhãăng."
    assert queued.output_line == "Điều này làm cho\n\nbạn bị xao nhãăng."
    assert "PREVIOUS: Bạn đang làm việc rất tập trung." in queued.context
    assert "TARGET: Điều này làm cho bạn bị xao nhãăng." in queued.context
    assert "NEXT: Vậy thì cách tốt nhất để không bị xao nhãng đó là ngăn ngừa trước khi điều đó xảy ra." in queued.context
    assert "target_basis:suspect_token:nhãăng" in queued.reasons


def test_queue_records_skip_reason_instead_of_only_counting_it(tmp_path) -> None:
    local_txt = tmp_path / "local.txt"
    local_txt.write_text("===== PDF084-L =====\n\nMột câu có thật.\n", encoding="utf-8")
    audit_path = tmp_path / "local_refine_audit.json"
    audit_path.write_text(
        json.dumps(
            [
                {
                    "id": "084-L-L999",
                    "page": [84, "L"],
                    "before": "đoạn không tồn tại",
                    "after": "đoạn không tồn tại",
                    "reasons": ["non_dictionary:tại"],
                    "edits": [],
                    "line_candidates": [],
                    "whole_candidates": [],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    queue, skipped = build_queue(local_txt, audit_path)

    assert queue == []
    assert len(skipped) == 1
    assert skipped[0]["id"] == "084-L-L999"
    assert skipped[0]["skip_reason"] == "audit_fragment_not_found_or_ambiguous"
