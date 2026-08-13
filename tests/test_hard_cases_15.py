from pdf_to_epub.deep.gate import apply_ai_ops
from pdf_to_epub.deep.queue import find_legacy_catastrophic_sides
from pdf_to_epub.models import DeepQueueItem


def alt(source: str, kind: str, text: str, conf: float) -> dict[str, object]:
    return {"source": source, "kind": kind, "text": text, "conf": conf}


def make(current: str, *, reasons=None, meta=None) -> DeepQueueItem:
    return DeepQueueItem(
        item_id="hard-case",
        page_number=1,
        side="L",
        current=current,
        output_line=current,
        context=current,
        reasons=reasons or ["whole_pass_disagreement"],
        candidates=[],
        candidate_meta=meta or [],
    )


def decision(item: DeepQueueItem, op: dict[str, object]):
    corrected, audit = apply_ai_ops(item, [op], 0.97)
    return corrected, audit[0]


def test_01_vidu_shape_safe_segmentation_auto() -> None:
    item = make(
        "4. Vidu.",
        reasons=["whole_pass_disagreement", "non_dictionary:Vidu"],
        meta=[
            alt("base_ve_25", "whole", "4. Vidu.", 92),
            alt("line_ve_p7", "line", "4. Vidụ.", 90),
        ],
    )
    corrected, row = decision(item, {"kind": "segment", "token_id": "t02", "old": "Vidu", "new": "Ví dụ", "confidence": 0.90})
    assert corrected == "4. Ví dụ."
    assert row["decision"] == "auto"


def test_02_repeated_di_and_four_errors_use_token_ids() -> None:
    item = make(
        "Muốn di nhanh, hãy di một minh. Muốn di xa.",
        meta=[
            alt("base_ve_25", "whole", "Muốn di nhanh, hãy di một minh. Muốn di xa.", 95),
            alt("line_v_p13", "line", "Muốn đi nhanh, hãy đi một mình. Muốn đi xa.", 78),
        ],
    )
    corrected, audit = apply_ai_ops(
        item,
        [
            {"token_id": "t02", "old": "di", "new": "đi", "confidence": 0.99},
            {"token_id": "t05", "old": "di", "new": "đi", "confidence": 0.99},
            {"token_id": "t07", "old": "minh", "new": "mình", "confidence": 0.99},
            {"token_id": "t09", "old": "di", "new": "đi", "confidence": 0.99},
        ],
        0.97,
        max_ops=5,
    )
    assert corrected == "Muốn đi nhanh, hãy đi một mình. Muốn đi xa."
    assert [row["decision"] for row in audit] == ["auto", "auto", "auto", "auto"]


def test_03_de_to_de_context_dominant_goes_verify() -> None:
    item = make("mình đế đón nhận", meta=[alt("line_v_p7", "line", "mình đế đón nhận", 96), alt("base_ve_25", "whole", "mình đế đón nhận", 95)])
    corrected, row = decision(item, {"token_id": "t02", "old": "đế", "new": "để", "confidence": 0.97})
    assert corrected == item.current
    assert row["decision"] == "verify"


def test_04_nhan_to_nhan_context_dominant_goes_verify() -> None:
    item = make("lòng nhăn nại", meta=[alt("line_ve_p7", "line", "lòng nhăn nại", 94), alt("base_v_25", "whole", "lòng nhăn nại", 95)])
    corrected, row = decision(item, {"token_id": "t02", "old": "nhăn", "new": "nhẫn", "confidence": 0.85})
    assert corrected == item.current
    assert row["decision"] == "verify"


def test_05_ke_to_ke_visual_shape_auto() -> None:
    item = make("người kế chuyện", meta=[alt("line_v_p7", "line", "người kể chuyện", 94), alt("base_v_25", "whole", "người kể chuyện", 92)])
    corrected, row = decision(item, {"token_id": "t02", "old": "kế", "new": "kể", "confidence": 0.99})
    assert corrected == "người kể chuyện"
    assert row["decision"] == "auto"


def test_06_digit_to_letter_requires_multi_family_visual_support() -> None:
    item = make("thấy 6 ngoài", meta=[alt("line_v_p7", "line", "thấy ở ngoài", 95), alt("base_v_25", "whole", "thấy ở ngoài", 92)])
    corrected, row = decision(item, {"token_id": "t02", "old": "6", "new": "ở", "confidence": 0.98})
    assert corrected == "thấy ở ngoài"
    assert row["decision"] == "auto"


def test_07_uao_to_vao_non_dictionary_plus_visual_auto() -> None:
    item = make("tập trung uào việc", reasons=["whole_pass_disagreement", "non_dictionary:uào"], meta=[alt("line_v_p7", "line", "tập trung vào việc", 95)])
    corrected, row = decision(item, {"token_id": "t03", "old": "uào", "new": "vào", "confidence": 0.95})
    assert corrected == "tập trung vào việc"
    assert row["decision"] == "auto"


def test_08_tre_to_tre_shape_plus_visual_auto() -> None:
    item = make("anh bạn tré", reasons=["whole_pass_disagreement", "non_dictionary:tré"], meta=[alt("base_v_25", "whole", "anh bạn trẻ", 88)])
    corrected, row = decision(item, {"token_id": "t03", "old": "tré", "new": "trẻ", "confidence": 0.95})
    assert corrected == "anh bạn trẻ"
    assert row["decision"] == "auto"


def test_09_uui_to_vui_non_dictionary_line_evidence_auto() -> None:
    item = make("mua uui cho", reasons=["whole_pass_disagreement", "non_dictionary:uui"], meta=[alt("line_v_p6", "line", "mua vui cho", 92)])
    corrected, row = decision(item, {"token_id": "t02", "old": "uui", "new": "vui", "confidence": 0.90})
    assert corrected == "mua vui cho"
    assert row["decision"] == "auto"


def test_10_dut_to_dut_shape_plus_visual_auto() -> None:
    item = make("đứt khoát", meta=[alt("base_v_25", "whole", "dứt khoát", 95)])
    corrected, row = decision(item, {"token_id": "t01", "old": "đứt", "new": "dứt", "confidence": 0.97})
    assert corrected == "dứt khoát"
    assert row["decision"] == "auto"


def test_11_uan_to_van_without_visual_goes_verify() -> None:
    item = make("uạn yên", reasons=["whole_pass_disagreement", "low_word_conf", "non_dictionary:uạn"], meta=[alt("line_v_p7", "line", "uạn yên", 74)])
    corrected, row = decision(item, {"token_id": "t01", "old": "uạn", "new": "vạn", "confidence": 0.90})
    assert corrected == item.current
    assert row["decision"] == "verify"


def test_12_phan_title_with_weak_single_family_goes_verify() -> None:
    item = make("PHAN 03", meta=[alt("base_v_25", "whole", "PHẦN 03", 82), alt("line_ve_p7", "line", "PHAN 03", 85)])
    corrected, row = decision(item, {"token_id": "t01", "old": "PHAN", "new": "PHẦN", "confidence": 0.85})
    assert corrected == item.current
    assert row["decision"] == "verify"


def test_13_chan_chu_without_visual_goes_verify() -> None:
    item = make("Chẩn chừ", meta=[alt("base_ve_25", "whole", "Chẩn chừ", 90)])
    corrected, row = decision(item, {"token_id": "t01", "old": "Chẩn", "new": "Chần", "confidence": 0.75})
    assert corrected == item.current
    assert row["decision"] == "verify"


def test_14_kho_to_kho_shape_plus_visual_auto() -> None:
    item = make("khố luyện", reasons=["whole_pass_disagreement", "non_dictionary:khố"], meta=[alt("line_v_p7", "line", "khổ luyện", 93)])
    corrected, row = decision(item, {"token_id": "t01", "old": "khố", "new": "khổ", "confidence": 0.99})
    assert corrected == "khổ luyện"
    assert row["decision"] == "auto"


def test_15_legacy_collapsed_side_is_skipped_before_deep() -> None:
    audit = []
    for index in range(1, 9):
        audit.append({
            "id": f"099-L-L{index:03d}",
            "page": [99, "L"],
            "reasons": ["whole_pass_disagreement", "low_word_conf", "non_dictionary:garble"],
            "whole_candidates": [{"text": "garble", "conf": 45.0}],
            "line_candidates": [{"text": "other", "conf": 55.0}],
        })
    assert (99, "L") in find_legacy_catastrophic_sides(audit)
