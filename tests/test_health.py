from pdf_to_epub.ocr.health import analyze_side_evidence


BOX = (0, 0, 100, 20)


def rows(lines: list[str], confidence: float = 90.0):
    return [(text, confidence, BOX) for text in lines]


def test_normal_prose_is_not_catastrophic() -> None:
    passage = [
        "Bạn có thể thay đổi cuộc sống bằng những thói quen nhỏ mỗi ngày.",
        "Điều quan trọng là tập trung vào một việc và kiên trì thực hiện.",
        "Khi tiến bộ đủ lâu bạn sẽ nhìn thấy kết quả rõ ràng hơn.",
    ]
    evidence = {
        "base": rows(passage, 91.0),
        "sharp": rows(passage, 89.0),
        "other": rows([line.replace("rõ ràng", "rõ ràng") for line in passage], 88.0),
    }
    report = analyze_side_evidence(evidence, set())
    assert report.catastrophic is False
    assert report.best_source in evidence


def test_symbol_soup_with_multiple_severe_signals_is_catastrophic() -> None:
    evidence = {
        "base": rows([
            "cá oe ng sản €€€ tá XWN Đạy dư Ì Bis clue ứng oquate",
            "bị la lệ ng tôi MWf JAGR S MN124 %% @@@",
            "tứ 4€ xa c4 chide tổng € om Ge ### ???",
        ], 43.0),
        "sharp": rows([
            "oe 8g sảr $$ XN day dtr B1s cIue oqua",
            "bị 1a l€ ng7 toi Wf JAGR 5 MN I24",
            "tư € xa e4 ch1de tong om G€ !!!",
        ], 39.0),
        "p6": rows([
            "@ cá o€ ng sả 7 XWN dư clue",
            "lệ ng? tôi MWf ... 124",
            "€ xa c4 -- tổng __ om Ge",
        ], 35.0),
    }
    report = analyze_side_evidence(evidence, set())
    assert report.catastrophic is True
    severe = {"symbol_soup", "fragmented_words", "lexicon_collapse", "cross_pass_instability", "low_side_confidence"}
    assert len(set(report.reasons) & severe) >= 2


def test_good_fallback_can_rescue_a_bad_initial_side() -> None:
    bad = {
        "base": rows([
            "cá oe ng sản €€€ tá XWN Đạy dư Ì Bis clue ứng oquate",
            "bị la lệ ng tôi MWf JAGR S MN124 %% @@@",
            "tứ 4€ xa c4 chide tổng € om Ge ### ???",
        ], 40.0),
        "sharp": rows([
            "oe 8g sảr $$ XN day dtr B1s cIue oqua",
            "bị 1a l€ ng7 toi Wf JAGR 5 MN I24",
            "tư € xa e4 ch1de tong om G€ !!!",
        ], 38.0),
    }
    assert analyze_side_evidence(bad, set()).catastrophic is True

    rescued = dict(bad)
    good_lines = [
        "Mỗi người đều có thể chuẩn bị cho những thay đổi trong cuộc sống.",
        "Điều quan trọng là nhìn rõ vấn đề và lựa chọn cách xử lý phù hợp.",
        "Một quyết định đúng lúc có thể tạo ra kết quả rất khác biệt.",
    ]
    rescued["fallback_p6"] = rows(good_lines, 92.0)
    rescued["fallback_otsu"] = rows(good_lines, 90.0)
    assert analyze_side_evidence(rescued, set()).catastrophic is False
