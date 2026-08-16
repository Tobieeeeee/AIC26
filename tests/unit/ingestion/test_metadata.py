import csv
import json
from pathlib import Path


def test_write_metadata_preserves_schema_and_mapping(tmp_path: Path) -> None:
    from mvr.ingestion.metadata import write_metadata
    from mvr.ingestion.models import Keyframe, Shot

    shots = [Shot("V001", "pyscenedetect", 1, 0, 24, 25.0)]
    keyframes = [
        Keyframe(
            "V001_S0001_K01",
            "V001",
            1,
            1,
            12,
            0.48,
            "keyframes/V001/V001_S0001_K01.jpg",
        )
    ]

    write_metadata(tmp_path, shots, keyframes)

    payload = json.loads((tmp_path / "shots.json").read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": "1.0",
        "frame_indexing": "zero_based",
        "shot_end_frame": "inclusive",
        "shots": [
            {
                "video_id": "V001",
                "method": "pyscenedetect",
                "shot_id": 1,
                "start_frame": 0,
                "end_frame": 24,
                "fps": 25.0,
                "start_time": 0.0,
                "end_time": 1.0,
                "duration": 1.0,
                "frame_count": 25,
            }
        ],
        "keyframes": [keyframes[0].to_dict()],
    }

    with (tmp_path / "keyframe_mapping.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "keyframe_id": "V001_S0001_K01",
            "video_id": "V001",
            "shot_id": "1",
            "ordinal": "1",
            "frame_idx": "12",
            "timestamp": "0.48",
            "image_path": "keyframes/V001/V001_S0001_K01.jpg",
        }
    ]


def test_write_benchmark_preserves_per_method_summary(tmp_path: Path) -> None:
    from mvr.ingestion.metadata import write_benchmark

    rows = [
        {
            "method": "pyscenedetect",
            "video_id": "V001",
            "fps": 25.0,
            "frame_count": 100,
            "duration_seconds": 4.0,
            "shot_count": 3,
            "model_load_seconds": 0.1,
            "detection_seconds": 0.5,
            "realtime_factor": 0.125,
        },
        {
            "method": "pyscenedetect",
            "video_id": "V002",
            "fps": 25.0,
            "frame_count": 200,
            "duration_seconds": 8.0,
            "shot_count": 5,
            "model_load_seconds": 0.1,
            "detection_seconds": 0.7,
            "realtime_factor": 0.0875,
        },
    ]

    write_benchmark(tmp_path, rows)

    payload = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert payload["summary"] == [
        {
            "method": "pyscenedetect",
            "video_count": 2,
            "total_shots": 8,
            "mean_shots_per_video": 4.0,
            "model_load_seconds": 0.1,
            "total_detection_seconds": 1.2,
            "mean_detection_seconds": 0.6,
            "aggregate_realtime_factor": 0.1,
        }
    ]
