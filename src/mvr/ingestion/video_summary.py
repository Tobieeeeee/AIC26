"""Build video-level keyframe summaries from AIC support data."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoSummaryKeyframe:
    keyframe_id: str
    frame_id: int | None
    timestamp: float | None
    fps: float | None
    image_path: Path


@dataclass(frozen=True)
class VideoSummary:
    video_id: str
    keyframe_count: int
    mapped_count: int
    keyframes: tuple[VideoSummaryKeyframe, ...]


def _numeric_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**12, value


def _normalize_ref(value: str) -> str:
    text = str(value).strip()
    if text.lower().endswith(".jpg"):
        text = text[: -len(".jpg")]
    try:
        return str(int(text))
    except ValueError:
        return text


def _read_mapping(path: Path | None) -> dict[str, tuple[int, float, float]]:
    if path is None or not path.is_file():
        return {}
    mapping: dict[str, tuple[int, float, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            ref = _normalize_ref(row["n"])
            mapping[ref] = (
                int(row["frame_idx"]),
                float(row["pts_time"]),
                float(row["fps"]),
            )
    return mapping


def build_video_summary(
    *,
    video_id: str,
    keyframe_root: Path,
    mapping_csv: Path | None = None,
) -> VideoSummary:
    """Return organizer keyframes for one video with verified frame mappings.

    ``frame_id`` is populated only when ``map_keyframes/<video_id>.csv`` has a
    matching row. Missing mappings are left as ``None`` so callers can display
    the keyframe without inventing a competition frame ID.
    """
    video_id = video_id.strip()
    if not video_id:
        raise ValueError("video_id must not be blank")
    if video_id in {".", ".."} or "/" in video_id or "\\" in video_id:
        raise ValueError("video_id must be a single video name, not a path")

    keyframe_dir = keyframe_root / video_id
    if not keyframe_dir.is_dir():
        raise FileNotFoundError(f"no keyframe directory found for video_id={video_id!r}")

    mapping = _read_mapping(mapping_csv)
    rows: list[VideoSummaryKeyframe] = []
    for image in sorted(keyframe_dir.glob("*.jpg"), key=lambda p: _numeric_key(p.stem)):
        ref = _normalize_ref(image.stem)
        mapped = mapping.get(ref)
        frame_id = timestamp = fps = None
        if mapped is not None:
            frame_id, timestamp, fps = mapped
        rows.append(
            VideoSummaryKeyframe(
                keyframe_id=image.stem,
                frame_id=frame_id,
                timestamp=timestamp,
                fps=fps,
                image_path=image,
            )
        )

    return VideoSummary(
        video_id=video_id,
        keyframe_count=len(rows),
        mapped_count=sum(1 for row in rows if row.frame_id is not None),
        keyframes=tuple(rows),
    )
