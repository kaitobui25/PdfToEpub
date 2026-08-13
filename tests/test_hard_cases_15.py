from pdf_to_epub.deep.candidates import build_book_stats, build_choice_sets
from pdf_to_epub.deep.gate import apply_ai_ops, apply_verifier_votes
from pdf_to_epub.deep.queue import find_legacy_catastrophic_sides
from pdf_to_epub.models import DeepQueueItem


# Small synthetic corpus with the same error shapes as the real 61..100 run.
# Repetition intentionally models the useful signal we observed in the book:
# correct forms such as để/đi/vui/trẻ recur while OCR corruptions are sparse.
BOOK = """
Ví dụ là cách giải thích dễ hiểu.
để để để để để để để để để để
đi đi đi đi đi đi đi đi
phần phần phần phần
vui vui vui vui vui
trẻ trẻ trẻ
chần chừ một chút
khổ luyện khổ luyện
vạn yên vạn yên
người kể chuyện kể chuyện
ở ngoài ở ngoài
vào việc vào việc
sự nhẫn nại
thói quen dứt khoát
"""
LEXICON = {
    "ví", "dụ", "để", "đế", "đi", "di", "mình", "minh", "nhẫn", "nhăn",
    "kể", "kế", "ở", "ngoài", "vào", "vạn", "phần", "chần", "chẩn", "khổ",
    "vui", "trẻ", "dứt", "khoát", "yên", "nại", "chuyện",
}
STATS = build_book_stats(BOOK, LEXICON)


def alt(source: str, kind: str, text: str, conf: float) -> dict[str, object]:
    return {"source": source, "kind": kind, "text": text, "conf": conf}


def make(current: str, *, reasons=None, meta=None, context=None) -> DeepQueueItem:
    item = DeepQueueItem(
        item_id="hard-case",
        page_number=1,
        side="L",
        current=current,
        output_line=current,
        context=context or current,
        reasons=reasons or ["whole_pass_disagreement"],
        candidates=[],
        candidate_meta=meta or [],
    )
    item.choice_sets = build_choice_sets(item, STATS)
    return item


def choice_id(item: DeepQueueItem, token_id: str, text: str) -> str:
    row = next(row for row in item.choice_sets if row["token_id"] == token_id)
    return str(next(choice["choice_id"] for choice in row["choices"] if choice["text"] == text))


def choose(item: DeepQueueItem, selections: list[tuple[str, str]], verify_change: bool = True):
    raw = [
        {"token_id": token_id, "choice_id": choice_id(item, token_id, text)}
        for token_id, text in selections
    ]
    corrected, audit = apply_ai_ops(item, raw, 0.97, max_ops=5)
    votes = {
        str(row["token_id"]): "CHANGE" if verify_change else "KEEP"
        for row in audit
        if row.get("decision") == "verify"
    }
    if votes:
        corrected = apply_verifier_votes(item, audit, votes)
    return corrected, audit


def test_01_vidu_candidate_is_vi_du_and_vidu_garbage_cannot_apply() -> None:
    item = make(
        "4. Vidu...",
        reasons=["whole_pass_disagreement", "diacritic_disagreement", "non_dictionary:Vidu"],
        meta=[
            alt("base_ve_25", "whole", "4. Vidu...", 92),
            alt("base_v_25", "whole", "4. Vidụ...", 80),
            alt("line_ve_p7", "line", "4. Vidụ...", 90),
        ],
    )
    row = next(row for row in item.choice_sets if row["token_id"] == "t02")
    vi_du = next(choice for choice in row["choices"] if choice["text"] == "Ví dụ")
    bad = next(choice for choice in row["choices"] if choice["text"] == "Vidụ")
    assert vi_du["strength"] == "strong"
    assert bad["strength"] == "weak"

    wrong, audit = choose(item, [("t02", "Vidụ")])
    assert wrong == item.current
    assert audit[0]["gate"] == "weak_local_candidate"

    corrected, _ = choose(item, [("t02", "Ví dụ")])
    assert corrected == "4. Ví dụ..."


def test_02_repeated_di_and_minh_use_four_stable_token_choices() -> None:
    item = make(
        "Muốn di nhanh, hãy di một minh. Muốn di xa, phải",
        meta=[
            alt("base_ve_25", "whole", "Muốn di nhanh, hãy di một minh. Muốn di xa, phải", 95),
            alt("line_ve_p13", "line", "Muốn di nhanh, hãy di một mình. Muốn di xa, phát", 88),
            alt("line_v_p13", "line", "Muốn ải nhanh, hãy ải một mình. Muốn đi xa, phát", 78),
        ],
    )
    corrected, _ = choose(item, [("t02", "đi"), ("t05", "đi"), ("t07", "mình"), ("t09", "đi")])
    assert corrected == "Muốn đi nhanh, hãy đi một mình. Muốn đi xa, phải"


def test_03_de_to_de_is_generated_from_book_frequency() -> None:
    item = make(
        "mình đế đón nhận",
        reasons=["whole_pass_disagreement", "diacritic_disagreement"],
        meta=[alt("line_v_p7", "line", "mình đế đón nhận", 96), alt("base_ve_25", "whole", "mình đế đón nhận", 95)],
    )
    corrected, _ = choose(item, [("t02", "để")])
    assert corrected == "mình để đón nhận"


def test_04_nhan_to_nhan_is_medium_and_requires_second_vote() -> None:
    item = make(
        "lòng nhăn nại",
        reasons=["whole_pass_disagreement", "diacritic_disagreement"],
        meta=[alt("base_ve_25", "whole", "lòng nhăn nại", 94)],
    )
    raw = [{"token_id": "t02", "choice_id": choice_id(item, "t02", "nhẫn")}]
    first, audit = apply_ai_ops(item, raw, 0.97)
    assert first == item.current
    assert audit[0]["decision"] == "verify"
    corrected = apply_verifier_votes(item, audit, {"t02": "CHANGE"})
    assert corrected == "lòng nhẫn nại"


def test_05_ke_to_ke_visual_candidate_survives() -> None:
    item = make("người kế chuyện", meta=[alt("line_v_p7", "line", "người kể chuyện", 94), alt("base_v_25", "whole", "người kể chuyện", 92)])
    corrected, _ = choose(item, [("t02", "kể")])
    assert corrected == "người kể chuyện"


def test_06_digit_to_letter_requires_listed_ocr_choice() -> None:
    item = make("thấy 6 ngoài", meta=[alt("line_v_p7", "line", "thấy ở ngoài", 95), alt("base_v_25", "whole", "thấy ở ngoài", 92)])
    corrected, _ = choose(item, [("t02", "ở")])
    assert corrected == "thấy ở ngoài"


def test_07_uao_to_vao_is_generated_from_low_psm_visual_witness() -> None:
    item = make(
        "tập trung uào việc",
        reasons=["whole_pass_disagreement", "non_dictionary:uào"],
        meta=[alt("line_v_p13", "line", "tập trung vào việc", 72), alt("base_ve_25", "whole", "tập trung uào việc", 95)],
    )
    corrected, _ = choose(item, [("t03", "vào")])
    assert corrected == "tập trung vào việc"


def test_08_tre_to_tre_uses_book_and_visual_evidence() -> None:
    item = make(
        "anh bạn tré",
        reasons=["whole_pass_disagreement", "diacritic_disagreement", "non_dictionary:tré"],
        meta=[alt("sharp_ve_25", "whole", "anh bạn trẻ", 88)],
    )
    corrected, _ = choose(item, [("t03", "trẻ")])
    assert corrected == "anh bạn trẻ"


def test_09_uui_to_vui_is_constrained() -> None:
    item = make(
        "mua uui cho",
        reasons=["whole_pass_disagreement", "non_dictionary:uui"],
        meta=[alt("line_v_p6", "line", "mua vui cho", 92)],
    )
    corrected, _ = choose(item, [("t02", "vui")])
    assert corrected == "mua vui cho"


def test_10_dut_to_dut_can_use_second_vote_when_visual_support_is_single_family() -> None:
    item = make("đứt khoát", meta=[alt("sharp_ve_25", "whole", "dứt khoát", 94)])
    corrected, _ = choose(item, [("t01", "dứt")])
    assert corrected == "dứt khoát"


def test_11_uan_to_van_expands_van_seed_through_book_shape() -> None:
    item = make(
        "uạn yên được không",
        reasons=["whole_pass_disagreement", "low_word_conf", "non_dictionary:uạn"],
        meta=[alt("sharp_ve_25", "whole", "van yên được không", 78), alt("line_v_p7", "line", "uạn yên được không", 73)],
    )
    corrected, _ = choose(item, [("t01", "vạn")])
    assert corrected == "vạn yên được không"


def test_12_phan_to_phan_is_closed_choice_not_free_form() -> None:
    item = make("PHAN 03", meta=[alt("base_v_30", "whole", "PHẦN O3", 82), alt("base_ve_25", "whole", "PHAN 03", 93)])
    corrected, _ = choose(item, [("t01", "PHẦN")])
    assert corrected == "PHẦN 03"


def test_13_chan_to_chan_uses_book_lexical_candidate_and_verifier() -> None:
    item = make(
        "Chẩn chừ",
        reasons=["whole_pass_disagreement", "diacritic_disagreement"],
        meta=[alt("base_ve_25", "whole", "Chẩn chừ", 87), alt("sharp_ve_25", "whole", "Chan chừ", 88)],
    )
    corrected, _ = choose(item, [("t01", "Chần")])
    assert corrected == "Chần chừ"


def test_14_kho_to_kho_uses_line_candidate() -> None:
    item = make(
        "khố luyện",
        reasons=["whole_pass_disagreement", "diacritic_disagreement", "non_dictionary:khố"],
        meta=[alt("line_v_p6", "line", "khổ luyện", 93), alt("base_v_25", "whole", "khố luyện", 96)],
    )
    corrected, _ = choose(item, [("t01", "khổ")])
    assert corrected == "khổ luyện"


def test_15_legacy_collapsed_side_is_skipped_before_any_choice_generation() -> None:
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
