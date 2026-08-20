"""A frame whose image cannot be loaded must still produce a context.

Found by running the pipeline with OCR enabled for the first time: every
prior measurement used --no-ocr, so this path had never executed against a
real engine. Temporal expansion made it reachable -- an expanded neighbour
has no image from the retrieval backend, so `image` is None -- and EasyOCR
answered `np.asarray(None)` with an UnboundLocalError deep inside its own
input reformatting, killing the whole query.

Dropping the OCR text is the honest outcome: no image means no observed
text, not an error and not a fabricated string.
"""
from __future__ import annotations

from mvr.enrichment.qna_context import build_frame_context


class _ExplodingOcrEngine:
    """Stands in for EasyOCR, which crashes rather than refusing None."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def read_text(self, image: object) -> str:
        self.calls.append(image)
        if image is None:
            raise UnboundLocalError("cannot access local variable 'img'")
        return "TIN CHÍNH"


def test_missing_image_yields_empty_ocr_text_instead_of_crashing() -> None:
    engine = _ExplodingOcrEngine()

    context = build_frame_context(
        video_id="L30_V072",
        frame_id=1745,
        image=None,
        objects_path=None,
        ocr_engine=engine,
    )

    assert context.ocr_text == ""
    assert engine.calls == []  # the engine is never handed a None image


def test_a_real_image_still_reaches_the_engine() -> None:
    engine = _ExplodingOcrEngine()
    image = object()

    context = build_frame_context(
        video_id="L30_V072",
        frame_id=1745,
        image=image,
        objects_path=None,
        ocr_engine=engine,
    )

    assert context.ocr_text == "TIN CHÍNH"
    assert engine.calls == [image]
