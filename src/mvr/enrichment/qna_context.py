"""Per-frame structured context builder for Q&A answer reasoning.

Combines two independent evidence sources for one already-resolved
``(video_id, frame_id)``:

- the organizer's precomputed object-detection JSON, via
  ``mvr.enrichment.objects.aic_objects`` (optional -- not every video ships
  organizer object files, matching the frame registry's own "optional
  accelerator" treatment of organizer support data in
  ``mvr.ingestion.aic_dataset``);
- OCR text from any of the existing Lab 6 engines
  (``mvr.enrichment.ocr.easyocr_engine`` / ``paddleocr_engine``), injected as
  a plain ``OcrEngine`` so this module never imports EasyOCR/PaddleOCR
  itself and stays testable without the optional ``ocr`` extra installed.

A missing objects file is treated as "no organizer data for this frame" and
degrades to ``objects=None`` -- the same policy as the frame registry's
optional accelerators. A *present but malformed* objects file is not
swallowed: ``load_frame_objects`` raises ``InvalidAicObjectsError`` and this
module lets it propagate, since malformed data is a real defect and must
never be silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mvr.enrichment.objects.aic_objects import FrameObjects, load_frame_objects

# The organizer's own FiftyOne baseline notebook for AIC 2026 filters the
# provided Faster R-CNN/OpenImages detections at ``score > 0.4`` before
# using them. Real organizer files carry ~100 detections per frame down to
# confidence ~0.006; feeding those unfiltered into an answer prompt buries
# the real objects in noise and makes counting questions unanswerable.
DEFAULT_MIN_OBJECT_CONFIDENCE = 0.4


class OcrEngine(Protocol):
    def read_text(self, image: Any) -> str: ...


@dataclass(frozen=True)
class FrameContext:
    """Structured evidence for one resolved frame, ready for answer reasoning."""

    video_id: str
    frame_id: int
    ocr_text: str
    objects: FrameObjects | None

    @property
    def object_counts(self) -> dict[str, int]:
        return {} if self.objects is None else self.objects.counts


def build_frame_context(
    *,
    video_id: str,
    frame_id: int,
    image: Any,
    objects_path: Path | str | None,
    ocr_engine: OcrEngine | None,
    min_object_confidence: float = DEFAULT_MIN_OBJECT_CONFIDENCE,
) -> FrameContext:
    if not video_id.strip():
        raise ValueError("video_id must not be blank")
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")

    objects: FrameObjects | None = None
    if objects_path is not None and Path(objects_path).exists():
        objects = load_frame_objects(
            objects_path,
            video_id=video_id,
            frame_ref=str(frame_id),
            min_confidence=min_object_confidence,
        )

    ocr_text = "" if ocr_engine is None else ocr_engine.read_text(image)

    return FrameContext(
        video_id=video_id,
        frame_id=frame_id,
        ocr_text=ocr_text,
        objects=objects,
    )
