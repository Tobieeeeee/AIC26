"""Three ways the OCR entry point failed against the organizer's data.

All three are silent or near-silent, which is why none was noticed: one
crashes before reading a frame, one reports success having read nothing,
and one clips a video's keyframes against another video's shot.

None of them can be reached by the existing tests, because there are none
for this module.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mvr.enrichment.ocr.scene_text import (
    generate_sampling_frames,
    get_shot_boundaries,
    resolve_video_path,
)


# -- shots.json is optional --------------------------------------------


def test_a_missing_shots_json_is_not_an_error(tmp_path: Path) -> None:
    """The organizer ships keyframes and no shot detection, so this is the
    normal state. `process_dataset_ocr` called this unguarded and died with
    FileNotFoundError before reading a single frame."""
    assert get_shot_boundaries(tmp_path / "shots.json") == {}


def test_shots_json_is_still_read_when_present(tmp_path: Path) -> None:
    shots = tmp_path / "shots.json"
    shots.write_text(
        json.dumps(
            {"shots": [{"video_id": "L21_V001", "shot_id": 1, "start_frame": 0, "end_frame": 99}]}
        ),
        encoding="utf-8",
    )

    assert get_shot_boundaries(shots) == {("L21_V001", 1): (0, 99)}


# -- a shot belongs to a video -----------------------------------------


def _keyframe(video_id: str, keyframe_id: str, *, shot_id: int, frame_idx: int) -> dict:
    return {
        "keyframe_id": keyframe_id,
        "video_id": video_id,
        "shot_id": shot_id,
        "frame_idx": frame_idx,
        "timestamp": frame_idx / 25.0,
        "image_path": f"{video_id}/{keyframe_id}.jpg",
    }


def test_one_videos_shot_does_not_clip_another_videos_keyframes() -> None:
    """`shot_boundaries` is built once from the whole shots.json and handed
    to every video. Keyed on shot_id alone, `boundaries[1]` was whichever
    video was parsed last -- every video numbers its shots from 1 -- so
    another video's keyframes were clipped against a range belonging to a
    video they had nothing to do with.

    Here V001 shot 1 covers frames 0-99 and V002 shot 1 covers 5000-5099.
    Under the old key, V002's keyframe at 5050 fell outside 0-99 and every
    sample for it was dropped -- the video would have been OCR'd to nothing.
    """
    boundaries = {
        ("L21_V001", 1): (0, 99),
        ("L21_V002", 1): (5000, 5099),
    }
    keyframes = [_keyframe("L21_V002", "034", shot_id=1, frame_idx=5050)]

    frames, samples = generate_sampling_frames(keyframes, boundaries)

    assert [f.frame_idx for f in frames] == [5049, 5050, 5051]
    assert list(samples) == [("L21_V002", "034")]


def test_the_shot_boundary_still_clips_inside_the_right_video() -> None:
    """Widening the key must not stop the clipping it exists to do: a
    keyframe on the first frame of its shot has no predecessor to sample."""
    boundaries = {("L21_V002", 1): (5000, 5099)}
    keyframes = [_keyframe("L21_V002", "034", shot_id=1, frame_idx=5000)]

    frames, _ = generate_sampling_frames(keyframes, boundaries)

    assert [f.frame_idx for f in frames] == [5000, 5001]


def test_no_boundaries_means_no_clipping() -> None:
    """With no shot detection every offset is kept, rather than none."""
    keyframes = [_keyframe("L21_V001", "034", shot_id=34, frame_idx=1745)]

    frames, _ = generate_sampling_frames(keyframes, {})

    assert [f.frame_idx for f in frames] == [1744, 1745, 1746]


def test_samples_are_keyed_by_video_and_keyframe() -> None:
    """Hardening rather than a bug report: both callers pass one video at a
    time today. Keyed correctly so that stops being something to remember."""
    keyframes = [
        _keyframe("L21_V001", "034", shot_id=34, frame_idx=1745),
        _keyframe("L21_V002", "034", shot_id=34, frame_idx=900),
    ]

    _, samples = generate_sampling_frames(keyframes, {})

    assert sorted(samples) == [("L21_V001", "034"), ("L21_V002", "034")]
    assert len(samples[("L21_V001", "034")]) == 3
    assert len(samples[("L21_V002", "034")]) == 3


# -- finding the video file --------------------------------------------


def test_the_flat_layout_is_found(tmp_path: Path) -> None:
    (tmp_path / "L21_V001.mp4").write_bytes(b"mp4")

    assert resolve_video_path(tmp_path, "L21_V001") == tmp_path / "L21_V001.mp4"


def test_the_raw_organizer_layout_is_found(tmp_path: Path) -> None:
    """The real download nests videos two levels and splits them across 14
    directories: Videos_L21_a/video/L21_V001.mp4. Only the flat shape was
    searched, so all 873 videos went unfound."""
    nested = tmp_path / "Videos_L21_a" / "video"
    nested.mkdir(parents=True)
    (nested / "L21_V001.mp4").write_bytes(b"mp4")

    assert resolve_video_path(tmp_path, "L21_V001") == nested / "L21_V001.mp4"


def test_a_genuinely_absent_video_returns_none(tmp_path: Path) -> None:
    assert resolve_video_path(tmp_path, "L99_V999") is None


# -- finding nothing is not success ------------------------------------


def test_finding_no_video_at_all_raises_instead_of_reporting_success(
    tmp_path: Path,
) -> None:
    """The old code warned once per video and returned a manifest saying
    zero keyframes were processed. Every downstream step accepts that as
    the data, so pointing this at the raw organizer download would have
    produced an empty OCR feature set that looked complete."""
    from mvr.enrichment.ocr.scene_text import process_dataset_ocr

    mapping = tmp_path / "keyframe_mapping.csv"
    with mapping.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["keyframe_id", "video_id", "shot_id", "ordinal", "frame_idx", "timestamp", "image_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "keyframe_id": "034", "video_id": "L21_V001", "shot_id": 34,
                "ordinal": 1, "frame_idx": 1745, "timestamp": 69.8,
                "image_path": "L21_V001/034.jpg",
            }
        )

    with pytest.raises(FileNotFoundError, match="no video found"):
        process_dataset_ocr(
            mapping_csv=mapping,
            video_root=tmp_path / "empty",
            output_dir=tmp_path / "out",
        )


# -- docker_ocr's anchor lookup ----------------------------------------


def test_the_anchor_lookup_is_keyed_by_video_and_keyframe() -> None:
    """`_build_anchor_lookup` is built over every video at once, so on
    keyframe_id alone the entry for "034" was whichever of 842 videos was
    read last. The OCR request then carried that video's anchor_frame_idx
    and anchor_timestamp -- and frame_idx is the number the competition
    scores on."""
    from mvr.enrichment.ocr.docker_ocr import _build_anchor_lookup

    keyframes = [
        _keyframe("L21_V001", "034", shot_id=34, frame_idx=1745),
        _keyframe("L21_V002", "034", shot_id=34, frame_idx=900),
    ]

    lookup = _build_anchor_lookup(keyframes)

    assert lookup[("L21_V001", "034")]["frame_idx"] == 1745
    assert lookup[("L21_V002", "034")]["frame_idx"] == 900
