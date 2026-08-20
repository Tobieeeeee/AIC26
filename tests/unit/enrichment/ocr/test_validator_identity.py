"""A keyframe_id alone does not identify a keyframe.

The organizer's 166,550 keyframes carry only 632 distinct ids -- "001"
exists in ~850 videos -- so the validator's duplicate check, keyed on the
id alone, stopped at row 309 of the real mapping (L21_V001, then L21_V002)
and refused to validate anything at all.

This file did not exist. The bug was found by generating the real mapping
and running the consumers over it, not by reading the code.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from mvr.enrichment.ocr.validator import validate_ocr_output

MAPPING_COLUMNS = ["keyframe_id", "video_id", "shot_id", "ordinal", "frame_idx", "timestamp", "image_path"]


def _write_mapping(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row(video_id: str, keyframe_id: str, *, frame_idx: int) -> dict:
    return {
        "keyframe_id": keyframe_id,
        "video_id": video_id,
        "shot_id": int(keyframe_id),
        "ordinal": 1,
        "frame_idx": frame_idx,
        "timestamp": frame_idx / 25.0,
        "image_path": f"{video_id}/{keyframe_id}.jpg",
    }


def _write_ocr_json(output_dir: Path, row: dict) -> None:
    video_dir = output_dir / row["video_id"]
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / f"{row['keyframe_id']}.json").write_text(
        json.dumps(
            {
                "keyframe_id": row["keyframe_id"],
                "video_id": row["video_id"],
                "shot_id": row["shot_id"],
                "anchor_frame_idx": row["frame_idx"],
                "anchor_timestamp": row["timestamp"],
                "texts": [],
            }
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    mapping = tmp_path / "keyframe_mapping.csv"
    _write_mapping(mapping, rows)
    output_dir = tmp_path / "ocr"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
    for row in rows:
        _write_ocr_json(output_dir, row)
    return mapping, output_dir


def test_the_same_keyframe_id_in_two_videos_is_not_a_duplicate(tmp_path: Path) -> None:
    """The whole point. This is the organizer's normal case, not an edge one:
    every video numbers its keyframes from 001."""
    mapping, output_dir = _fixture(
        tmp_path,
        [
            _row("L21_V001", "001", frame_idx=0),
            _row("L21_V002", "001", frame_idx=0),
        ],
    )

    assert validate_ocr_output(mapping, output_dir) is True


def test_the_same_keyframe_twice_in_one_video_is_still_a_duplicate(
    tmp_path: Path,
) -> None:
    """Widening the key must not stop catching what it was there to catch."""
    mapping = tmp_path / "keyframe_mapping.csv"
    _write_mapping(
        mapping,
        [_row("L21_V001", "001", frame_idx=0), _row("L21_V001", "001", frame_idx=50)],
    )
    output_dir = tmp_path / "ocr"
    output_dir.mkdir()

    assert validate_ocr_output(mapping, output_dir) is False


def test_a_json_filed_under_the_wrong_video_is_caught(tmp_path: Path) -> None:
    """The check this replaced. Under the composite key a misfiled JSON
    misses the lookup instead of failing a cross-check, and the message
    names both sides."""
    rows = [_row("L21_V001", "001", frame_idx=0)]
    mapping, output_dir = _fixture(tmp_path, rows)
    misfiled = output_dir / "L21_V999"
    misfiled.mkdir()
    (misfiled / "001.json").write_text("{}", encoding="utf-8")

    assert validate_ocr_output(mapping, output_dir) is False


def test_the_completeness_check_counts_rows_not_distinct_ids(tmp_path: Path) -> None:
    """Keyed on the id alone, `len(keyframe_mappings)` was the number of
    distinct ids -- 632 for 166,550 organizer keyframes -- so a run missing
    almost every file would have passed."""
    rows = [
        _row("L21_V001", "001", frame_idx=0),
        _row("L21_V002", "001", frame_idx=0),
        _row("L21_V003", "001", frame_idx=0),
    ]
    mapping, output_dir = _fixture(tmp_path, rows)
    (output_dir / "L21_V003" / "001.json").unlink()

    assert validate_ocr_output(mapping, output_dir) is False
