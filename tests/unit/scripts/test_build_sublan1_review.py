"""Turning a voluntarily-shared team's submission into two review sheets.

Comparing it against round1_verified.json found 8 of 17 "verified" queries
naming a DIFFERENT video, not a nearby frame. That is the reason this
module exists: neither source gets trusted on its own, a person looks at
both and picks.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _module():
    path = REPOSITORY_ROOT / "scripts" / "build_sublan1_review.py"
    spec = importlib.util.spec_from_file_location("_build_sublan1_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_mapping(path: Path, rows: list[tuple[str, str, int, str]]) -> None:
    """`(video_id, keyframe_id, frame_idx, image_path)` rows."""
    lines = ["keyframe_id,video_id,shot_id,ordinal,frame_idx,timestamp,image_path"]
    for video_id, keyframe_id, frame_idx, image_path in rows:
        lines.append(f"{keyframe_id},{video_id},1,1,{frame_idx},0.0,{image_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_nearest_keyframe_is_picked_not_the_first(tmp_path) -> None:
    mapping = tmp_path / "mapping.csv"
    _write_mapping(
        mapping,
        [
            ("L1", "001", 100, "a.jpg"),
            ("L1", "002", 500, "b.jpg"),
            ("L1", "003", 1000, "c.jpg"),
        ],
    )
    module = _module()

    frame_index = module.load_frame_index(mapping)
    result = module.nearest_keyframe(frame_index, "L1", 520)

    assert result == (500, "b.jpg", 20)


def test_a_video_absent_from_the_mapping_is_reported_not_guessed(tmp_path) -> None:
    """Three of round 1's own verified answers name frames read straight
    off the video, never through search -- their video is not in the
    keyframe mapping at all. Guessing an image for them would be worse
    than saying so."""
    mapping = tmp_path / "mapping.csv"
    _write_mapping(mapping, [("L1", "001", 100, "a.jpg")])
    module = _module()

    frame_index = module.load_frame_index(mapping)

    assert module.nearest_keyframe(frame_index, "L2", 100) is None


def test_a_tie_in_distance_picks_the_earlier_keyframe(tmp_path) -> None:
    mapping = tmp_path / "mapping.csv"
    _write_mapping(mapping, [("L1", "001", 90, "a.jpg"), ("L1", "002", 110, "b.jpg")])
    module = _module()

    frame_index = module.load_frame_index(mapping)

    assert module.nearest_keyframe(frame_index, "L1", 100) == (90, "a.jpg", 10)


def test_a_video_that_agrees_is_not_a_mismatch(tmp_path) -> None:
    """sublan1 naming the SAME video the operator already confirmed is
    agreement, not a conflict -- it must not appear in the review sheet."""
    module = _module()
    ground_truth = {
        "query-p1-1-kis": {
            "kind": "kis", "video_id": "L1", "frame_idx": 100, "query_vi": "x",
        }
    }
    sublan1_dir = tmp_path / "sublan1"
    sublan1_dir.mkdir()
    (sublan1_dir / "query-p1-1-kis.csv").write_text("L1,150\n", encoding="utf-8")

    mismatch = []
    for name, entry in ground_truth.items():
        candidates = module.sublan1_candidates(sublan1_dir, name)
        sub_videos = {v for v, _ in candidates}
        if entry["video_id"] not in sub_videos:
            mismatch.append(name)

    assert mismatch == []


def test_a_video_that_disagrees_is_flagged(tmp_path) -> None:
    module = _module()
    sublan1_dir = tmp_path / "sublan1"
    sublan1_dir.mkdir()
    (sublan1_dir / "query-p1-1-kis.csv").write_text("L2,150\n", encoding="utf-8")

    candidates = module.sublan1_candidates(sublan1_dir, "query-p1-1-kis")

    assert {v for v, _ in candidates} == {"L2"}


def test_a_missing_query_in_sublan1_has_no_candidates(tmp_path) -> None:
    """sublan1 does not answer every query either -- absence is not a
    conflict and must not raise."""
    module = _module()

    assert module.sublan1_candidates(tmp_path, "query-p1-99-kis") == []


def test_the_review_row_carries_the_query_text_for_context(tmp_path) -> None:
    module = _module()
    mapping = tmp_path / "mapping.csv"
    _write_mapping(mapping, [("L1", "001", 100, "a.jpg")])
    frame_index = module.load_frame_index(mapping)

    row = module.build_row(
        query="query-p1-1-kis", source="sublan1", video_id="L1",
        frame_idx=100, frame_index=frame_index, query_vi="một câu hỏi",
    )

    assert row["query_vi"] == "một câu hỏi"
    assert row["chon_dong_nay"] == ""  # cột người phải tự điền, không suy đoán


def test_the_output_sheets_have_the_documented_columns(tmp_path) -> None:
    module = _module()
    row = {c: "" for c in module.REVIEW_COLUMNS}
    out = tmp_path / "sheet.csv"

    module.write_sheet(out, [row])

    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(module.REVIEW_COLUMNS)
