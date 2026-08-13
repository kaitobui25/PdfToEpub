from pdf_to_epub.config import DeepConfig
from pdf_to_epub.deep.gate import apply_ai_sentence
from pdf_to_epub.models import DeepQueueItem


def make_item() -> DeepQueueItem:
    current = "Đây là một việc xấu."
    return DeepQueueItem(
        item_id="001-L-S001",
        page_number=1,
        side="L",
        current=current,
        output_line=current,
        context=current,
        reasons=["whole_pass_disagreement"],
        candidates=[],
    )


def proposal(confidence: float = 0.96) -> dict[str, object]:
    return {
        "corrected_sentence": "Đây là một việc tốt.",
        "confidence": confidence,
    }


def test_default_deep_trust_is_high() -> None:
    assert DeepConfig().deep_trust == "high"


def test_strict_blocks_unsupported_lexical_change() -> None:
    source = make_item()
    corrected, audit = apply_ai_sentence(source, proposal(), 0.97, deep_trust="strict")
    assert corrected == source.current
    assert audit[0]["gate"] == "unsupported_sentence_span"


def test_balanced_still_blocks_unsupported_change_at_096() -> None:
    source = make_item()
    corrected, audit = apply_ai_sentence(source, proposal(), 0.97, deep_trust="balanced")
    assert corrected == source.current
    assert audit[0]["gate"] == "unsupported_sentence_span"


def test_high_accepts_small_unsupported_change_at_096() -> None:
    source = make_item()
    corrected, audit = apply_ai_sentence(source, proposal(), 0.97, deep_trust="high")
    assert corrected == "Đây là một việc tốt."
    assert audit[0]["gate"] == "sentence_deep_trust_override"
    assert audit[0]["applied"] is True


def test_high_does_not_allow_punctuation_rewrite() -> None:
    source = make_item()
    corrected, audit = apply_ai_sentence(
        source,
        {"corrected_sentence": "Đây là một việc tốt!", "confidence": 0.99},
        0.97,
        deep_trust="high",
    )
    assert corrected == source.current
    assert audit[0]["gate"] == "sentence_punctuation_change"
