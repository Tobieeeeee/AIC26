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

import json
from collections.abc import Mapping
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
    #: Transcribed speech near this frame. Empty when unavailable, which is
    #: the case for every context built the query-time way -- Q&A had no
    #: path to audio at all before Task05A documents existed.
    asr_text: str = ""
    #: Where the frame image lives, so a human or VLM reviewing this
    #: candidate can actually see it. ``None`` when nothing supplied one.
    image_path: str | None = None
    #: Label counts that arrived WITHOUT the detections behind them --
    #: Task05A documents aggregate objects down to counts, keeping no
    #: per-detection confidence or box. Held separately rather than
    #: back-filled into ``FrameObjects`` because inventing detections to
    #: satisfy a dataclass would be fabricating data this frame never had.
    #: ``None`` means "derive counts from ``objects``".
    aggregated_object_counts: dict[str, int] | None = None

    @property
    def object_counts(self) -> dict[str, int]:
        if self.aggregated_object_counts is not None:
            return dict(self.aggregated_object_counts)
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
    upstream (see ``mvr.enrichment.objects.rf_detr`` /
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


def frame_context_from_record(record: "Mapping[str, Any] | Path | str") -> FrameContext:
    """Build answer context from one Task05A document.

    The document (``mvr.indexing.task05a``) already carries OCR, ASR,
    object counts and the competition ``frame_idx`` for one keyframe, so
    this replaces the query-time work -- open the JPEG, run OCR, read the
    organizer objects file -- with a single read. On CPU that work cost
    ~6.7s per candidate, which is why every measurement before this ran
    with OCR disabled.

    It also removes ``resolve_original_frame_id`` from this path: the
    document states ``frame_idx`` directly, so the failure mode where an
    unresolvable mapping silently yielded ``frame_id=None`` for every
    candidate cannot occur here.

    Malformed records raise rather than degrading to a blank context, the
    same policy ``load_frame_objects`` already applies -- bad data is a
    defect, and a silently empty context would look exactly like a frame
    that genuinely had nothing on it.
    """
    if isinstance(record, (str, Path)):
        record = json.loads(Path(record).read_text(encoding="utf-8"))
    if not isinstance(record, Mapping):
        raise ValueError(f"record must be a mapping or a path, got {type(record).__name__}")

    missing = [key for key in ("video_id", "frame_idx", "ocr", "asr", "objects") if key not in record]
    if missing:
        raise ValueError(f"Task05A record is missing required fields: {missing}")

    video_id = str(record["video_id"]).strip()
    if not video_id:
        raise ValueError("record video_id must not be blank")
    frame_id = record["frame_idx"]
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError(f"record frame_idx must be a non-negative integer, got {frame_id!r}")

    counts = {
        str(name): int(count)
        for name, count in (record["objects"].get("counts") or {}).items()
    }
    image_path = record.get("image_path")
    return FrameContext(
        video_id=video_id,
        frame_id=frame_id,
        ocr_text=str(record["ocr"].get("joined") or "").strip(),
        objects=None,
        aggregated_object_counts=counts,
        asr_text=str(record["asr"].get("joined") or "").strip(),
        image_path=None if image_path is None else str(image_path),
    )
