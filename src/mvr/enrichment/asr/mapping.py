from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from mvr.ingestion.aic_dataset import AicDatasetRegistry

from .contracts import KeyframeAudioSource


def _selected(video_id: str, videos: set[str] | None) -> bool:
    return videos is None or video_id in videos


def _safe_keyframe_id(value: str) -> str:
    keyframe_id = str(value).strip()
    if not keyframe_id or Path(keyframe_id).name != keyframe_id:
        raise ValueError(f"invalid keyframe_id {value!r}")
    return keyframe_id


def _read_video_metadata(path: Path, root: Path) -> dict[str, tuple[Path, float]]:
    if not path.is_file():
        return {}
    result: dict[str, tuple[Path, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            video_path = Path(row["video_path"])
            if not video_path.is_absolute():
                video_path = root / video_path
            result[row["video_id"]] = (
                video_path.resolve(),
                float(row["duration_seconds"]),
            )
    return result


def _load_ingestion_mapping(
    root: Path, mapping_path: Path, videos: set[str] | None
) -> list[KeyframeAudioSource]:
    metadata = _read_video_metadata(root / "video_metadata.csv", root)
    rows: list[KeyframeAudioSource] = []
    with mapping_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            video_id = row["video_id"].strip()
            if not _selected(video_id, videos):
                continue
            try:
                video_path, duration = metadata[video_id]
            except KeyError as exc:
                raise ValueError(
                    f"video_metadata.csv has no row for {video_id!r}"
                ) from exc
            rows.append(
                KeyframeAudioSource(
                    video_id=video_id,
                    keyframe_id=_safe_keyframe_id(row["keyframe_id"]),
                    frame_idx=int(row["frame_idx"]),
                    timestamp=float(row["timestamp"]),
                    shot_id=int(row["shot_id"]),
                    video_path=video_path,
                    video_duration=duration,
                )
            )
    return rows


def _organizer_video_ids(root: Path, videos: set[str] | None) -> Iterable[str]:
    mapping_root = root / "map_keyframes"
    if not mapping_root.is_dir():
        return []
    available = sorted(path.stem for path in mapping_root.glob("*.csv"))
    return [video_id for video_id in available if _selected(video_id, videos)]


def _load_organizer_mapping(
    root: Path, videos: set[str] | None
) -> list[KeyframeAudioSource]:
    registry = AicDatasetRegistry(root)
    rows: list[KeyframeAudioSource] = []
    for video_id in _organizer_video_ids(root, videos):
        video_path = registry.video_path(video_id)
        if video_path is None:
            # Keep the authoritative identity so the batch can isolate/report
            # the missing video per keyframe rather than abort discovery.
            video_path = root / "videos" / f"{video_id}.mp4"
            duration = 0.0
        else:
            try:
                duration = registry.video_info(video_id).duration
            except (OSError, RuntimeError, ValueError):
                # Preserve every authoritative keyframe so the batch pipeline
                # can emit stable error artifacts instead of aborting discovery.
                duration = 0.0
        csv_path = root / "map_keyframes" / f"{video_id}.csv"
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    KeyframeAudioSource(
                        video_id=video_id,
                        keyframe_id=_safe_keyframe_id(row["n"]),
                        frame_idx=int(row["frame_idx"]),
                        timestamp=float(row["pts_time"]),
                        # Organizer map_keyframes has no shot identity. Null is
                        # explicit and auditable; inventing a shot is not.
                        shot_id=None,
                        video_path=video_path,
                        video_duration=duration,
                    )
                )
    return rows


def load_keyframe_audio_sources(
    dataset_root: Path | str, videos: Iterable[str] | None = None
) -> list[KeyframeAudioSource]:
    """Load either the repo ingestion contract or the organizer AIC contract."""

    root = Path(dataset_root)
    selected = set(videos) if videos is not None else None
    ingestion_mapping = root / "keyframe_mapping.csv"
    if ingestion_mapping.is_file():
        rows = _load_ingestion_mapping(root, ingestion_mapping, selected)
    else:
        rows = _load_organizer_mapping(root, selected)
    if not rows:
        raise ValueError(f"no authoritative keyframe mappings found under {root}")
    return sorted(rows, key=lambda row: (row.video_id, row.timestamp, row.keyframe_id))
