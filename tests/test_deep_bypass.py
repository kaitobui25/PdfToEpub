from pdf_to_epub.cli import build_parser
from pdf_to_epub.deep.gate import apply_ai_sentence
from pdf_to_epub.models import DeepQueueItem


def _item(current: str) -> DeepQueueItem:
    return DeepQueueItem(
        item_id="001-L-L001",
        page_number=1,
        side="L",
        current=current,
        output_line=current,
        context=current,
        reasons=["whole_pass_disagreement"],
        candidates=[],
    )


def test_cli_accepts_evidence_and_projection_off() -> None:
    args = build_parser().parse_args(
        [
            "pdf.pdf",
            "--deep-only",
            "--ocr-evidence-gate",
            "off",
            "--patch-projection",
            "off",
        ]
    )
    assert args.ocr_evidence_gate == "off"
    assert args.patch_projection == "off"


def test_evidence_gate_off_trusts_deep_without_ocr_candidate() -> None:
    source = _item("Đây là một câu OCR sai.")
    corrected, audit = apply_ai_sentence(
        source,
        {
            "corrected_sentence": "Đây là một câu hoàn toàn đúng.",
            "confidence": 0.98,
        },
        0.97,
        deep_trust="high",
        ocr_evidence_gate=False,
    )
    assert corrected == "Đây là một câu hoàn toàn đúng."
    assert audit[0]["gate"] == "sentence_deep_direct"
    assert audit[0]["applied"] is True


def test_evidence_gate_off_still_respects_minimum_deep_confidence() -> None:
    source = _item("Đây là một câu OCR sai.")
    corrected, audit = apply_ai_sentence(
        source,
        {
            "corrected_sentence": "Đây là một câu hoàn toàn đúng.",
            "confidence": 0.90,
        },
        0.97,
        deep_trust="high",
        ocr_evidence_gate=False,
    )
    assert corrected == source.current
    assert audit[0]["gate"] == "insufficient_deep_confidence"
    assert audit[0]["applied"] is False
