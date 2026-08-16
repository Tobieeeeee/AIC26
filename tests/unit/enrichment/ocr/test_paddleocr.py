import numpy as np


def test_extract_paddle_lines_supports_v3_legacy_and_deduplication() -> None:
    from mvr.enrichment.ocr.paddleocr_engine import extract_paddle_lines

    result = [
        {"rec_texts": [" GIÁ VÀNG ", "low"], "rec_scores": [0.8, 0.2]},
        [["box"], ("TĂNG", 0.3)],
        ["TĂNG", 0.3],
        {"text": "không có score"},
    ]

    assert extract_paddle_lines(result) == [
        ("GIÁ VÀNG", 0.8),
        ("TĂNG", 0.3),
        ("không có score", None),
    ]


def test_make_paddle_reader_falls_back_from_v3_to_v2_arguments() -> None:
    from mvr.enrichment.ocr.paddleocr_engine import make_paddle_reader

    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TypeError("PaddleOCR 2.x")
        return "reader"

    assert make_paddle_reader(factory) == "reader"
    assert calls == [
        {
            "lang": "vi",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "enable_mkldnn": False,
        },
        {
            "lang": "vi",
            "use_angle_cls": False,
            "show_log": False,
            "enable_mkldnn": False,
        },
    ]


def test_paddle_engine_prefers_predict_and_joins_final_text() -> None:
    from mvr.enrichment.ocr.paddleocr_engine import PaddleOcrEngine

    class Reader:
        def predict(self, image):
            assert isinstance(image, np.ndarray)
            return iter([{"rec_texts": ["xin", "chào"], "rec_scores": [0.9, 0.8]}])

    assert PaddleOcrEngine(reader=Reader()).read_text(np.zeros((2, 2, 3))) == "xin chào"
