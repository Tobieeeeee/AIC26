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
    min_object_confidence: float = 0.0,
) -> FrameContext:
    """Assemble one frame's structured evidence.

    ``min_object_confidence`` defaults to ``0.0`` (no filtering) on purpose:
    **choosing a detection confidence threshold is the indexing/enrichment
    layer's decision, not Q&A's.** This module deliberately owns no
    threshold value so there is exactly one source of truth for that policy
    upstream (see ``mvr.enrichment.objects.faster_rcnn`` /
    ``mvr.enrichment.objects.yolo``, which carry their detector's own).

    Callers that read *raw, unfiltered* organizer objects JSON directly
    should pass an explicit threshold -- those files carry ~100 detections
    per frame down to confidence ~0.006, which would otherwise flood the
    answer prompt with noise.
    """
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

    # No image means no observed text. Guarding here rather than in each
    # engine: EasyOCR answers a None image with an UnboundLocalError from
    # inside its own input reformatting, which would kill the whole query.
    # Temporally-expanded neighbours routinely have no image, since no
    # retrieval backend returned one for them.
    ocr_text = "" if ocr_engine is None or image is None else ocr_engine.read_text(image)

    return FrameContext(
        video_id=video_id,
        frame_id=frame_id,
        ocr_text=ocr_text,
        objects=objects,
    )
