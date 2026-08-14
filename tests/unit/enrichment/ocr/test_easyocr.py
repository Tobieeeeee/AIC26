import numpy as np


def test_easyocr_preserves_languages_gpu_and_confidence_threshold() -> None:
    from mvr.enrichment.ocr.easyocr_engine import EasyOcrEngine

    calls = []

    class Reader:
        def readtext(self, image):
            calls.append(("readtext", image.shape))
            return [
                ("box", "discarded", 0.299),
                ("box", "GIÁ VÀNG", 0.3),
                ("box", "TĂNG", 0.9),
            ]

    def factory(languages, gpu):
        calls.append(("factory", languages, gpu))
        return Reader()

    engine = EasyOcrEngine(gpu=True, reader_factory=factory)

    assert engine.read_text(np.zeros((4, 5, 3), dtype=np.uint8)) == "GIÁ VÀNG TĂNG"
    assert calls == [
        ("factory", ["vi", "en"], True),
        ("readtext", (4, 5, 3)),
    ]


def test_easyocr_default_gpu_choice_matches_lab6_cuda_device() -> None:
    from mvr.enrichment.ocr.easyocr_engine import EasyOcrEngine

    calls = []

    class Reader:
        pass

    def factory(languages, gpu):
        calls.append((languages, gpu))
        return Reader()

    EasyOcrEngine(reader_factory=factory, cuda_available=lambda: True)

    assert calls == [(["vi", "en"], True)]
