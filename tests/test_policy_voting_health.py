from pdf_to_epub.deep.candidates import build_book_stats
from pdf_to_epub.deep.policy import finalize_choice_policy
from pdf_to_epub.deep.voting import finalize_reverse_vote, finalize_selected_vote, make_reverse_op
from pdf_to_epub.models import DeepQueueItem
from pdf_to_epub.ocr.line_health import analyze_line_health


def make_item(current: str, *, reasons=None, choice_sets=None) -> DeepQueueItem:
    return DeepQueueItem(
        item_id="hard",
        page_number=1,
        side="L",
        current=current,
        output_line=current,
        context=current,
        reasons=reasons or ["whole_pass_disagreement"],
        candidates=[],
        choice_sets=choice_sets or [],
    )


def test_old_valid_strong_candidate_requires_verifier() -> None:
    stats = build_book_stats("về lâu dài\nvề lâu dài\nmột lâu đài\nmột lâu đài")
    item = make_item(
        "về lâu dài",
        choice_sets=[{
            "token_id": "t03",
            "old": "dài",
            "choices": [
                {"choice_id": "KEEP", "text": "dài", "kind": "keep", "strength": "keep"},
                {
                    "choice_id": "C1",
                    "text": "đài",
                    "kind": "replace",
                    "strength": "strong",
                    "visual_score": 2.3,
                    "book_frequency": 2,
                    "lexical_valid": True,
                    "shape_preserving": True,
                    "source_tags": ["ocr_direct"],
                },
            ],
        }],
    )
    finalize_choice_policy(item, stats)
    choice = item.choice_sets[0]["choices"][1]
    assert choice["strength"] == "medium"
    assert choice["old_valid_requires_verifier"] is True
    assert choice["reverse_verify"] is True


def test_non_dictionary_old_keeps_strong_candidate() -> None:
    stats = build_book_stats("vui vui vui")
    item = make_item(
        "uui",
        reasons=["non_dictionary:uui", "whole_pass_disagreement"],
        choice_sets=[{
            "token_id": "t01",
            "old": "uui",
            "choices": [
                {"choice_id": "KEEP", "text": "uui", "kind": "keep", "strength": "keep"},
                {
                    "choice_id": "C1",
                    "text": "vui",
                    "kind": "replace",
                    "strength": "strong",
                    "visual_score": 1.8,
                    "book_frequency": 3,
                    "lexical_valid": True,
                    "shape_preserving": False,
                    "source_tags": ["ocr_direct"],
                },
            ],
        }],
    )
    finalize_choice_policy(item, stats)
    assert item.choice_sets[0]["choices"][1]["strength"] == "strong"


def test_boundary_split_adds_chon3_without_blind_regex() -> None:
    stats = build_book_stats("Chọn cách này\nChọn cách khác\nChọn phương án")
    item = make_item("Chọn3", reasons=["whole_pass_disagreement"])
    result = finalize_choice_policy(item, stats)
    assert result["boundary_added"] == 1
    assert item.choice_sets[0]["choices"][1]["text"] == "Chọn 3"
    assert item.choice_sets[0]["choices"][1]["strength"] == "medium"


def test_boundary_split_does_not_split_a4_code() -> None:
    stats = build_book_stats("A4 là mã máy\nA4 vẫn dùng")
    item = make_item("A4", reasons=["whole_pass_disagreement"])
    result = finalize_choice_policy(item, stats)
    assert result["boundary_added"] == 0
    assert item.choice_sets == []


def test_post_reocr_line_garbage_is_catastrophic() -> None:
    current = "antes }.Ỷ/Zềể nh c cv HH1 Hóc PC nen rene xx yy zz"
    candidates = [
        {"text": "aaa / xyz nh q cv H1 H0c PC nnn", "conf": 42},
        {"text": "antes zzz p q c HHI Hoc P0 rrr", "conf": 51},
        {"text": "xxyy !! nn ccc vv H0c", "conf": 38},
    ]
    report = analyze_line_health(current, candidates, {"học", "nên", "con", "trên", "làm", "một"})
    assert report.catastrophic is True


def test_normal_vietnamese_line_is_not_catastrophic() -> None:
    current = "Bạn sẽ học được nhiều điều mới khi kiên trì mỗi ngày."
    candidates = [
        {"text": current, "conf": 94},
        {"text": "Bạn sẽ học được nhiều điều mới khi kiên trì mỗi ngày.", "conf": 92},
    ]
    lexicon = {"bạn", "sẽ", "học", "được", "nhiều", "điều", "mới", "khi", "kiên", "trì", "mỗi", "ngày"}
    assert analyze_line_health(current, candidates, lexicon).catastrophic is False


def test_selected_change_conflict_uses_third_vote() -> None:
    op = {"decision": "verify", "gate": "closed_choice_needs_second_vote", "applied": False}
    assert finalize_selected_vote(op, "KEEP") is False
    assert op["decision"] == "tie_break"
    assert finalize_selected_vote(op, "KEEP", "CHANGE") is True
    assert op["votes"] == ["CHANGE", "KEEP", "CHANGE"]


def test_reverse_change_needs_two_of_three() -> None:
    item = make_item(
        "thi nó",
        choice_sets=[{
            "token_id": "t01",
            "old": "thi",
            "choices": [
                {"choice_id": "KEEP", "text": "thi", "kind": "keep", "strength": "keep"},
                {"choice_id": "C1", "text": "thì", "kind": "replace", "strength": "medium", "reverse_verify": True},
            ],
        }],
    )
    op = make_reverse_op(item, "t01", "C1")
    assert op is not None
    assert finalize_reverse_vote(op, "KEEP") is False
    assert op["votes"] == ["KEEP", "CHANGE", "KEEP"]

    op = make_reverse_op(item, "t01", "C1")
    assert op is not None
    assert finalize_reverse_vote(op, "CHANGE") is True
    assert op["votes"] == ["KEEP", "CHANGE", "CHANGE"]
