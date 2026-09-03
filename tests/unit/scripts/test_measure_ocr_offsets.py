"""Contract checks for the OCR sampling measurement."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "operations"
    / "measure_ocr_offsets.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_measure_ocr_offsets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mapping(tmp_path: Path, *video_ids: str) -> Path:
    path = tmp_path / "keyframe_mapping.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["keyframe_id", "video_id", "frame_idx", "timestamp", "shot_id"]
        )
        writer.writeheader()
        for video_id in video_ids:
            for index in range(3):
                writer.writerow(
                    {
                        "keyframe_id": f"{video_id}_S0001_K{index:02d}",
                        "video_id": video_id,
                        "frame_idx": 100 + index * 25,
                        "timestamp": (100 + index * 25) / 25.0,
                        "shot_id": 1,
                    }
                )
    path.with_name("shots.json").write_text("{}", encoding="utf-8")
    return path


def test_the_candidate_sampling_must_keep_the_anchor_frame() -> None:
    """Offsets without 0 read only neighbours, which measures nothing useful."""
    module = _load()

    with pytest.raises(ValueError, match="anchor"):
        module.parse_offsets("-1,1")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", (0,)), ("-1,0,1", (-1, 0, 1)), (" 0 , 1 ", (0, 1))],
)
def test_offsets_are_read_as_written(raw: str, expected: tuple[int, ...]) -> None:
    assert _load().parse_offsets(raw) == expected


def test_only_the_named_video_is_measured(tmp_path: Path) -> None:
    """Measuring a whole collection would cost the hours this exists to save."""
    module = _load()
    mapping = _mapping(tmp_path, "L26_V001", "L26_V002")
    destination = tmp_path / "out" / "keyframe_mapping.csv"

    written = module.write_single_video_mapping(mapping, "L26_V001", destination)

    assert written == 3
    with destination.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["video_id"] for row in rows} == {"L26_V001"}
    assert destination.with_name("shots.json").is_file()


def test_a_video_absent_from_the_mapping_is_refused(tmp_path: Path) -> None:
    module = _load()
    mapping = _mapping(tmp_path, "L26_V001")

    with pytest.raises(ValueError, match="L26_V999"):
        module.write_single_video_mapping(
            mapping, "L26_V999", tmp_path / "out" / "keyframe_mapping.csv"
        )


def test_missing_shot_boundaries_are_refused_before_the_gpu_is_used(tmp_path: Path) -> None:
    """The sampler clips offsets to shot edges; without them it would sample across cuts."""
    module = _load()
    mapping = _mapping(tmp_path, "L26_V001")
    mapping.with_name("shots.json").unlink()

    with pytest.raises(FileNotFoundError, match="shot boundaries"):
        module.write_single_video_mapping(
            mapping, "L26_V001", tmp_path / "out" / "keyframe_mapping.csv"
        )


def test_main_reports_a_failure_without_touching_the_gpu(tmp_path, capsys) -> None:
    module = _load()
    mapping = _mapping(tmp_path, "L26_V001")

    exit_code = module.main(
        [
            "--mapping-csv", str(mapping),
            "--video-root", str(tmp_path),
            "--work-root", str(tmp_path / "work"),
            "--video-id", "L26_V999",
        ]
    )

    assert exit_code == 2
    assert "L26_V999" in json.loads(capsys.readouterr().err)["error"]
