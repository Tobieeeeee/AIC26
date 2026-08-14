"""Lab 6 search-document schema connected to Phase 02 keyframes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SearchDocument:
    frame_id: str
    video_id: str
    ts: float
    caption: str
    ocr_text: str
    objects: dict[str, int]
    clip_vector: list[float]

    def to_source(self) -> dict[str, Any]:
        return asdict(self)


def build_search_document(
    keyframe: Mapping[str, Any],
    *,
    caption: str,
    ocr_text: str,
    objects: Mapping[str, int],
    clip_vector: Sequence[float] | np.ndarray,
) -> SearchDocument:
    raw_frame_id = keyframe.get("keyframe_id")
    if raw_frame_id is None:
        raw_frame_id = keyframe.get("frame_id")
    frame_id = "" if raw_frame_id is None else str(raw_frame_id).strip()
    if not frame_id:
        raise ValueError("frame_id must be non-empty")
    return SearchDocument(
        frame_id=frame_id,
        video_id=str(keyframe["video_id"]),
        ts=float(keyframe.get("timestamp", keyframe.get("ts"))),
        caption=caption,
        ocr_text=ocr_text,
        objects={str(name): int(count) for name, count in objects.items()},
        clip_vector=np.asarray(clip_vector, dtype=np.float32).tolist(),
    )
