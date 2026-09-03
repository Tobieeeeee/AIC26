"""Decode the frames OCR reads, using the whole machine.

With the default temporal offsets every keyframe contributes three frames, so
the largest collection needs about 163,500 of them out of 497 files. Each video
costs one full ffmpeg decode, and those decodes are independent -- run one after
another they are a silent hour and a half on a thirty-core box.

The work is I/O and subprocess bound rather than interpreter bound, so threads
carry it: each worker waits on its own ffmpeg while the others run.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mvr.enrichment.ocr.scene_text import resolve_video_path
from mvr.ingestion.extract import extract_keyframes
from mvr.ingestion.models import Keyframe


# ffmpeg threads its own decode, so one worker per core oversubscribes the box
# and slows every decode down. This leaves room for that internal threading.
DEFAULT_WORKERS = 8

Resolver = Callable[[Path, str], "Path | None"]
Extractor = Callable[[Path, list[Keyframe], Path], None]


def default_workers() -> int:
    configured = os.environ.get("MVR_OCR_EXTRACT_WORKERS", "").strip()
    if configured.isdigit() and int(configured) > 0:
        return int(configured)
    return min(DEFAULT_WORKERS, os.cpu_count() or 1)


def extract_sampling_frames(
    by_video: Mapping[str, Sequence[Keyframe]],
    *,
    video_root: Path,
    frames_dir: Path,
    workers: int | None = None,
    resolver: Resolver = resolve_video_path,
    extractor: Extractor = extract_keyframes,
) -> int:
    """Decode every frame not already on disk. Returns how many were decoded.

    Videos are resolved before any decode starts: a missing source is an error
    about the dataset, and reporting it after half the pool has spent minutes
    decoding wastes exactly the time this function exists to save.
    """
    if not by_video:
        return 0

    jobs: list[tuple[Path, list[Keyframe], Path]] = []
    for video_id, frames in sorted(by_video.items()):
        video_path = resolver(video_root, video_id)
        if video_path is None:
            raise FileNotFoundError(
                f"Missing source video: {video_id}.mp4 under {video_root}"
            )
        video_frames_dir = frames_dir / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        missing = [
            frame
            for frame in frames
            if not (video_frames_dir / frame.image_path).is_file()
        ]
        if missing:
            jobs.append((video_path, missing, video_frames_dir))

    if not jobs:
        return 0

    pool_size = workers if workers and workers > 0 else default_workers()
    if pool_size == 1 or len(jobs) == 1:
        for video_path, missing, video_frames_dir in jobs:
            extractor(video_path, missing, video_frames_dir)
    else:
        with ThreadPoolExecutor(max_workers=min(pool_size, len(jobs))) as pool:
            # Consuming the results re-raises the first failure. A video that
            # only half decoded would send OCR over frames that are not there.
            list(pool.map(lambda job: extractor(*job), jobs))

    return sum(len(missing) for _path, missing, _root in jobs)
