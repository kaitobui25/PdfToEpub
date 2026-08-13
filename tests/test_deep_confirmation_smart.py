from pdf_to_epub.cli import build_parser
from pdf_to_epub.deep.confirmation import confirmation_status, needs_second_pass
from pdf_to_epub.deep.projection import render_smart_target
from pdf_to_epub.models import DeepQueueItem


def _item(current: str, raw: str | None = None) -> DeepQueueItem:
    return DeepQueueItem(
        item_id="001-L-S001",
        page_number=1,
        side="L",
        current=current,
        output_line=raw if raw is not None else current,
        context=current,
        reasons=["whole_pass_disagreement"],
        candidates=[],
    )


def test_cli_accepts_smart_projection() -> None:
    args = build_parser().parse_args(["pdf.pdf", "--deep-only", "--patch-projection", "smart"])
    assert args.patch_projection == "smart"


def test_borderline_proposal_gets_second_pass() -> None:
    current = "Nếu cùng một lúc bạn đuối hai con thỏ."
    first = {
        "corrected_sentence": "Nếu cùng một lúc bạn đuổi hai con thỏ.",
        "confidence": 0.92,
    }
    assert needs_second_pass(first, current) is True


def test_two_independent_matching_answers_confirm() -> None:
    current = "hãy đừng lại 1 phút."
    first = {"corrected_sentence": "hãy dừng lại 1 phút.", "confidence": 0.92}
    second = {"corrected_sentence": "hãy dừng lại 1 phút.", "confidence": 0.91}
    assert confirmation_status(first, second, current) == "confirmed"


def test_second_pass_disagreement_does_not_confirm() -> None:
    current = "ban sẽ uụt mất cả hai."
    first = {"corrected_sentence": "bạn sẽ vuột mất cả hai.", "confidence": 0.92}
    second = {"corrected_sentence": "bạn sẽ vụt mất cả hai.", "confidence": 0.93}
    assert confirmation_status(first, second, current) == "disagreed"


def test_smart_projection_handles_repeated_word_by_position() -> None:
    source = _item(
        "thì hãy tất tất cả những thứ gây mất tập trung.",
        "thì hãy tất\n\ntất cả những thứ gây mất tập trung.",
    )
    rendered = render_smart_target(source, "thì hãy tắt tất cả những thứ gây mất tập trung.")
    assert rendered == "thì hãy tắt\n\ntất cả những thứ gây mất tập trung."


def test_smart_projection_can_split_one_bad_ocr_token() -> None:
    source = _item("Tấttấtcả mọi thứ gây mất tập trung.")
    rendered = render_smart_target(source, "Tắt tất cả mọi thứ gây mất tập trung.")
    assert rendered == "Tắt tất cả mọi thứ gây mất tập trung."
