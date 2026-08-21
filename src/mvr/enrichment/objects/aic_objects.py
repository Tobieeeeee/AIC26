"""Organizer-provided AIC object-detection artifact loader.

Reads the per-frame object detection JSON the organizer ships under
``objects/<video_id>/<frame>.json`` (Faster R-CNN pretrained on OpenImages
V4, per the AIC2026 preliminary-round document). This is a **reader for a
precomputed artifact**, not a live detector -- it does not run inference and
is unrelated to ``mvr.enrichment.objects.yolo.YoloObjectDetector``.

Schema assumption (UNCONFIRMED against a real organizer sample -- see
``docs/AIC_QNA_IMPLEMENTATION_NOTES.md`` Task 01 §3): the organizer's linked
TensorFlow reference for this exact pretrained module
(``faster_rcnn/openimages_v4/inception_resnet_v2``) documents an output dict
with ``detection_boxes`` (``[ymin, xmin, ymax, xmax]``, normalized to
``[0, 1]``), ``detection_class_entities`` (human-readable labels), and
``detection_scores``. This loader assumes the organizer serialized that same
dict shape directly to JSON. If a real sample later shows different key
names, fix this loader rather than guessing silently -- do not widen
validation to swallow an unexpected schema.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_KEYS = ("detection_class_entities", "detection_scores", "detection_boxes")


class InvalidAicObjectsError(ValueError):
    """An organizer objects JSON file is missing, malformed, or unsafe."""


@dataclass(frozen=True)
class ObjectDetection:
    """One detected label from the organizer's Faster R-CNN/OpenImages output."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise InvalidAicObjectsError("detection label must not be blank")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise InvalidAicObjectsError(
                f"detection confidence must be finite and in [0, 1], got {self.confidence!r}"
            )
        if len(self.bbox) != 4 or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.bbox
        ):
            raise InvalidAicObjectsError(
                f"detection bbox must have four finite values in [0, 1], got {self.bbox!r}"
            )


@dataclass(frozen=True)
class FrameObjects:
    """All detections for one organizer keyframe, plus per-label counts."""

    video_id: str
    frame_ref: str
    detections: tuple[ObjectDetection, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise InvalidAicObjectsError("video_id must not be blank")
        if not self.frame_ref.strip():
            raise InvalidAicObjectsError("frame_ref must not be blank")

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(detection.label for detection in self.detections))


def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    if key not in payload:
        raise InvalidAicObjectsError(f"missing required key {key!r}")
    value = payload[key]
    if not isinstance(value, list):
        raise InvalidAicObjectsError(f"{key!r} must be a list, got {type(value).__name__}")
    return value


def load_frame_objects(
    path: Path | str, *, video_id: str, frame_ref: str, min_confidence: float = 0.0
) -> FrameObjects:
    """Parse one organizer objects JSON file into a validated ``FrameObjects``.

    Raises ``InvalidAicObjectsError`` on any schema violation rather than
    silently dropping or coercing bad detections.

    ``min_confidence`` (inclusive) drops low-confidence detections *after*
    validating them, so a corrupt entry still raises even when it would
    have been filtered out. The default of ``0.0`` keeps every detection:
    this loader is a faithful reader of the organizer artifact, and
    choosing a threshold is the consuming pipeline's policy decision (see
    ``mvr.enrichment.qna_context.build_frame_context``, which applies the
    organizer baseline's own ``0.4``).
    """
    raw_text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise InvalidAicObjectsError(f"{path}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidAicObjectsError(f"{path}: root JSON value must be an object")

    for key in REQUIRED_KEYS:
        if key not in payload:
            raise InvalidAicObjectsError(f"{path}: missing required key {key!r}")

    labels = _require_list(payload, "detection_class_entities")
    scores = _require_list(payload, "detection_scores")
    boxes = _require_list(payload, "detection_boxes")

    if not (len(labels) == len(scores) == len(boxes)):
        raise InvalidAicObjectsError(
            f"{path}: detection_class_entities/detection_scores/detection_boxes "
            f"length mismatch: {len(labels)}, {len(scores)}, {len(boxes)}"
        )

    detections: list[ObjectDetection] = []
    for label, score, box in zip(labels, scores, boxes, strict=True):
        if not isinstance(label, str):
            raise InvalidAicObjectsError(f"{path}: detection label must be a string, got {label!r}")
        if not isinstance(box, list) or len(box) != 4:
            raise InvalidAicObjectsError(f"{path}: detection bbox must have four values, got {box!r}")
        detection = ObjectDetection(
            label=label,
            confidence=float(score),
            bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
        )
        # Filter after construction so validation still runs on every entry.
        if detection.confidence >= min_confidence:
            detections.append(detection)

    return FrameObjects(video_id=video_id, frame_ref=frame_ref, detections=tuple(detections))
