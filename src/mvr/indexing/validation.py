"""Input and output validation helpers for Task05A."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ValidationIssue:
    category: str
    message: str
    identity: tuple[str, str] | None = None
    sample: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": self.category,
            "message": self.message,
        }
        if self.identity is not None:
            payload["identity"] = list(self.identity)
        if self.sample is not None:
            payload["sample"] = self.sample
        return payload


class Task05AValidationError(ValueError):
    """Raised when a required canonical input cannot be validated."""


def identity_key(video_id: object, keyframe_id: object) -> tuple[str, str]:
    return (str(video_id).strip(), str(keyframe_id).strip())


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_task05a_document(document: Mapping[str, Any]) -> None:
    """Validate one emitted document against the Task05A v1 schema."""

    required = (
        "schema_version",
        "keyframe_id",
        "video_id",
        "shot_id",
        "frame_idx",
        "timestamp",
        "image_path",
        "ocr",
        "asr",
        "objects",
        "clip",
        "indexability",
    )
    missing = [key for key in required if key not in document]
    if missing:
        raise Task05AValidationError(f"document missing required fields: {missing}")
    if not str(document["keyframe_id"]).strip() or not str(document["video_id"]).strip():
        raise Task05AValidationError("document identity fields must be non-empty")
    if isinstance(document["shot_id"], bool) or not isinstance(document["shot_id"], int):
        raise Task05AValidationError("shot_id must be an integer")
    if isinstance(document["frame_idx"], bool) or not isinstance(document["frame_idx"], int):
        raise Task05AValidationError("frame_idx must be an integer")
    if document["frame_idx"] < 0 or not _is_finite_number(document["timestamp"]):
        raise Task05AValidationError("invalid frame_idx/timestamp")
    # A blank path would satisfy a presence check while helping no consumer:
    # Q&A candidate review and the search UI both need to render the frame,
    # and neither can recover the path from anything else in the document.
    if not isinstance(document["image_path"], str) or not document["image_path"].strip():
        raise Task05AValidationError("image_path must be a non-empty string")

    for section in ("ocr", "asr", "objects", "clip", "indexability"):
        if not isinstance(document[section], Mapping):
            raise Task05AValidationError(f"{section} must be an object")
    for text_section in ("ocr", "asr"):
        if not isinstance(document[text_section].get("texts"), list):
            raise Task05AValidationError(f"{text_section}.texts must be a list")
        if not isinstance(document[text_section].get("joined"), str):
            raise Task05AValidationError(f"{text_section}.joined must be a string")
        for item in document[text_section]["texts"]:
            if not isinstance(item, Mapping):
                raise Task05AValidationError(f"{text_section}.texts item must be an object")
            if not isinstance(item.get("text"), str) or not isinstance(item.get("normalized"), str):
                raise Task05AValidationError(f"{text_section}.texts item has invalid text")
            if "conf" in item and not _is_finite_number(item["conf"]):
                raise Task05AValidationError(f"{text_section}.texts item has invalid conf")
    objects = document["objects"]
    if not isinstance(objects.get("labels"), list) or not isinstance(objects.get("counts"), Mapping):
        raise Task05AValidationError("objects requires labels and counts")
    if objects["labels"] != sorted(set(objects["labels"])):
        raise Task05AValidationError("object labels must be unique and sorted")
    for label, count in objects["counts"].items():
        if not isinstance(label, str) or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise Task05AValidationError("object counts must be non-negative integers")
    if set(objects["labels"]) != {label for label, count in objects["counts"].items() if count > 0}:
        raise Task05AValidationError("object labels/counts mismatch")
    indexability = document["indexability"]
    for key in ("lexical", "vector", "object"):
        if not isinstance(indexability.get(key), bool):
            raise Task05AValidationError(f"indexability.{key} must be boolean")


def ensure_finite_json(value: object, *, path: str = "$") -> None:
    """Reject NaN/Inf recursively before JSON serialization."""

    if isinstance(value, float) and not math.isfinite(value):
        raise Task05AValidationError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            ensure_finite_json(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            ensure_finite_json(child, path=f"{path}[{index}]")


def validate_clip_vector(
    vector: object,
    *,
    expected_dimension: int,
) -> str | None:
    """Return a stable invalid-vector reason, or ``None`` when valid."""

    array = np.asarray(vector)
    if not np.issubdtype(array.dtype, np.number):
        return "non_numeric_dtype"
    if array.ndim != 1 or array.shape[0] != expected_dimension:
        return "dimension_mismatch"
    if not np.isfinite(array).all():
        return "non_finite"
    return None
