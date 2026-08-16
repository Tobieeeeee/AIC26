def test_word_recall_and_hit_preserve_lab6_definition() -> None:
    from mvr.evaluation.ocr_metrics import ocr_hit, word_recall

    assert word_recall("GIÁ VÀNG TĂNG", "tin giá vàng hôm nay") == 2 / 3
    assert ocr_hit("GIÁ VÀNG TĂNG", "tin giá vàng hôm nay") is True
    assert ocr_hit("GIÁ VÀNG TĂNG", "giá hôm nay") is False


def test_word_recall_rejects_empty_ground_truth() -> None:
    import pytest

    from mvr.evaluation.ocr_metrics import word_recall

    with pytest.raises(ValueError, match="ground_truth must contain at least one word"):
        word_recall("   ", "anything")
