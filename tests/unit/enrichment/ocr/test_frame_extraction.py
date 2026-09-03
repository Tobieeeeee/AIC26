"""Frame extraction must use the machine it is running on.

Before OCR infers anything it decodes every sampled frame out of the source
videos -- with the default temporal offsets that is three frames per keyframe,
so L26 alone needs roughly 163,500 of them from 497 files. Done one video after
another it is a silent, single-threaded hour and a half while twenty-nine cores
sit idle, and it produces no log line to say so.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from mvr.enrichment.ocr.frame_extraction import extract_sampling_frames
from mvr.ingestion.models import Keyframe


def _frame(video_id: str, ordinal: int) -> Keyframe:
    return Keyframe(
        keyframe_id=f"{video_id}_S0001_K01",
        video_id=video_id,
        shot_id=1,
        ordinal=ordinal,
        frame_idx=ordinal * 25,
        timestamp=float(ordinal),
        image_path=f"{ordinal:06d}.jpg",
    )


def _by_video(*video_ids: str, frames_each: int = 2) -> dict[str, list[Keyframe]]:
    return {
        video_id: [_frame(video_id, index) for index in range(frames_each)]
        for video_id in video_ids
    }


def _resolver(video_root: Path, video_id: str) -> Path | None:
    candidate = video_root / f"{video_id}.mp4"
    return candidate if candidate.is_file() else None


def _publish(output_root: Path, frames: list[Keyframe]) -> None:
    for frame in frames:
        (output_root / frame.image_path).write_bytes(b"jpeg")


def test_every_video_is_extracted(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    for video_id in ("L26_V001", "L26_V002", "L26_V003"):
        (video_root / f"{video_id}.mp4").write_bytes(b"video")
    frames_dir = tmp_path / "frames"
    seen: list[tuple[str, int]] = []

    def extractor(video_path: Path, frames: list[Keyframe], output_root: Path) -> None:
        seen.append((video_path.stem, len(frames)))
        _publish(output_root, frames)

    extracted = extract_sampling_frames(
        _by_video("L26_V001", "L26_V002", "L26_V003"),
        video_root=video_root,
        frames_dir=frames_dir,
        resolver=_resolver,
        extractor=extractor,
    )

    assert sorted(seen) == [("L26_V001", 2), ("L26_V002", 2), ("L26_V003", 2)]
    assert extracted == 6


def test_a_missing_video_is_refused_by_name(tmp_path: Path) -> None:
    """A silently skipped video would publish an artifact missing its text."""
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "L26_V001.mp4").write_bytes(b"video")

    with pytest.raises(FileNotFoundError, match="L26_V002"):
        extract_sampling_frames(
            _by_video("L26_V001", "L26_V002"),
            video_root=video_root,
            frames_dir=tmp_path / "frames",
            resolver=_resolver,
            extractor=lambda video_path, frames, output_root: _publish(output_root, frames),
        )


def test_frames_already_on_disk_are_not_decoded_again(tmp_path: Path) -> None:
    """A resumed run must not re-decode a video for frames it already has."""
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "L26_V001.mp4").write_bytes(b"video")
    frames_dir = tmp_path / "frames"
    by_video = _by_video("L26_V001", frames_each=3)
    published = frames_dir / "L26_V001"
    published.mkdir(parents=True)
    _publish(published, by_video["L26_V001"][:2])

    asked: list[str] = []

    def extractor(video_path: Path, frames: list[Keyframe], output_root: Path) -> None:
        asked.extend(frame.image_path for frame in frames)
        _publish(output_root, frames)

    extracted = extract_sampling_frames(
        by_video,
        video_root=video_root,
        frames_dir=frames_dir,
        resolver=_resolver,
        extractor=extractor,
    )

    assert asked == ["000002.jpg"]
    assert extracted == 1


def test_videos_are_decoded_concurrently(tmp_path: Path) -> None:
    """Two videos must be in flight at once, or the pool is decorative.

    A barrier settles this without timing assertions: it releases only when
    two threads reach it, so a serial implementation blocks until the timeout
    instead of passing quickly.
    """
    video_root = tmp_path / "videos"
    video_root.mkdir()
    for video_id in ("L26_V001", "L26_V002"):
        (video_root / f"{video_id}.mp4").write_bytes(b"video")
    barrier = threading.Barrier(2, timeout=5)

    def extractor(video_path: Path, frames: list[Keyframe], output_root: Path) -> None:
        barrier.wait()
        _publish(output_root, frames)

    extract_sampling_frames(
        _by_video("L26_V001", "L26_V002"),
        video_root=video_root,
        frames_dir=tmp_path / "frames",
        workers=2,
        resolver=_resolver,
        extractor=extractor,
    )

    assert barrier.n_waiting == 0


def test_one_video_failing_fails_the_whole_extraction(tmp_path: Path) -> None:
    """Partial frames would send OCR over a video it never fully decoded."""
    video_root = tmp_path / "videos"
    video_root.mkdir()
    for video_id in ("L26_V001", "L26_V002"):
        (video_root / f"{video_id}.mp4").write_bytes(b"video")

    def extractor(video_path: Path, frames: list[Keyframe], output_root: Path) -> None:
        if video_path.stem == "L26_V002":
            raise RuntimeError("ffmpeg exited with 1")
        _publish(output_root, frames)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        extract_sampling_frames(
            _by_video("L26_V001", "L26_V002"),
            video_root=video_root,
            frames_dir=tmp_path / "frames",
            workers=2,
            resolver=_resolver,
            extractor=extractor,
        )


def test_nothing_to_do_is_not_an_error(tmp_path: Path) -> None:
    assert (
        extract_sampling_frames(
            {},
            video_root=tmp_path,
            frames_dir=tmp_path / "frames",
            resolver=_resolver,
            extractor=lambda *args: None,
        )
        == 0
    )


# --- how many frames OCR is asked to read at all --------------------------


def _keyframe_dicts(count: int) -> list[dict]:
    return [
        {
            "video_id": "L26_V001",
            "keyframe_id": f"L26_V001_S0001_K{index:02d}",
            "frame_idx": 100 + index * 25,
            "timestamp": (100 + index * 25) / 25.0,
            "shot_id": 1,
        }
        for index in range(count)
    ]


def test_the_default_offsets_are_the_three_the_manifest_claims() -> None:
    """OCR's cost is a multiple of this, so it is pinned rather than inferred."""
    from mvr.enrichment.ocr.scene_text import DEFAULT_TEMPORAL_OFFSETS

    assert tuple(DEFAULT_TEMPORAL_OFFSETS) == (-1, 0, 1)


def test_the_default_offsets_cannot_be_mutated_by_a_caller() -> None:
    """A shared mutable default would change every later run in the process."""
    from mvr.enrichment.ocr.scene_text import DEFAULT_TEMPORAL_OFFSETS

    with pytest.raises((AttributeError, TypeError)):
        DEFAULT_TEMPORAL_OFFSETS.append(2)  # type: ignore[attr-defined]


def test_a_single_offset_reads_one_frame_per_keyframe() -> None:
    """Dropping the neighbours is what makes OCR three times cheaper.

    The neighbours exist to catch text that motion blur smears in the anchor
    frame, so this is a recall-for-time trade rather than a free win; the
    sampler has to honour the choice either way.
    """
    from mvr.enrichment.ocr.scene_text import generate_sampling_frames

    keyframes = _keyframe_dicts(4)
    boundaries = {("L26_V001", 1): (0, 10_000)}

    with_neighbours, _ = generate_sampling_frames(keyframes, boundaries)
    anchor_only, _ = generate_sampling_frames(keyframes, boundaries, [0])

    assert len(anchor_only) == 4
    assert len(with_neighbours) == 12
    assert [frame.frame_idx for frame in anchor_only] == [100, 125, 150, 175]
