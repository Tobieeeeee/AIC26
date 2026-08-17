from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from mvr.evaluation.benchmark import (
    KisBenchmarkQuery,
    KisBenchmarkRecord,
    build_aic_keyframe_mapping_csv,
    run_kis_benchmark_query,
    summarize_kis_benchmark,
)


@dataclass(frozen=True)
class _FakeCandidate:
    video_id: str
    frame_id: int


def test_run_kis_benchmark_query_scores_via_official_evaluator() -> None:
    query = KisBenchmarkQuery(
        query_id="q1", query_text="a red car", video_id="L21_V001",
        start_frame=250, end_frame=270,
    )

    def fake_search(text: str):
        assert text == "a red car"
        return [_FakeCandidate("L21_V001", 261)]

    record = run_kis_benchmark_query(query, config_name="semantic", search_fn=fake_search)

    assert record.query_id == "q1"
    assert record.config_name == "semantic"
    assert record.final_score == 1.0
    assert all(v == 1.0 for v in record.r_at_k.values())
    assert record.latency_seconds >= 0.0


def test_run_kis_benchmark_query_scores_zero_when_wrong_video() -> None:
    query = KisBenchmarkQuery(
        query_id="q2", query_text="x", video_id="L21_V001",
        start_frame=250, end_frame=270,
    )

    record = run_kis_benchmark_query(
        query, config_name="semantic",
        search_fn=lambda text: [_FakeCandidate("L21_V002", 261)],
    )

    assert record.final_score == 0.0


def _record(config: str, qid: str, final_score: float, latency: float) -> KisBenchmarkRecord:
    return KisBenchmarkRecord(
        query_id=qid, config_name=config,
        r_at_k={1: final_score, 5: final_score, 20: final_score, 50: final_score, 100: final_score},
        final_score=final_score, latency_seconds=latency,
    )


def test_summarize_kis_benchmark_groups_by_config() -> None:
    records = [
        _record("semantic", "q1", 1.0, 0.10),
        _record("semantic", "q2", 0.0, 0.20),
        _record("semantic+refine", "q1", 1.0, 0.50),
        _record("semantic+refine", "q2", 1.0, 0.60),
    ]

    summary = summarize_kis_benchmark(records)

    assert set(summary.keys()) == {"semantic", "semantic+refine"}
    assert summary["semantic"]["query_count"] == 2
    assert summary["semantic"]["mean_final_score"] == pytest.approx(0.5)
    assert summary["semantic+refine"]["mean_final_score"] == pytest.approx(1.0)


def test_summarize_kis_benchmark_computes_median_and_latency_stats() -> None:
    records = [
        _record("semantic", "q1", 1.0, 0.10),
        _record("semantic", "q2", 0.5, 0.20),
        _record("semantic", "q3", 0.0, 0.30),
    ]

    summary = summarize_kis_benchmark(records)["semantic"]

    assert summary["median_final_score"] == pytest.approx(0.5)
    assert summary["mean_latency_seconds"] == pytest.approx(0.2)
    assert summary["p95_latency_seconds"] >= summary["mean_latency_seconds"]


def test_summarize_kis_benchmark_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        summarize_kis_benchmark([])


def test_summarize_kis_benchmark_single_record_p95_equals_value() -> None:
    summary = summarize_kis_benchmark([_record("semantic", "q1", 1.0, 0.42)])["semantic"]

    assert summary["p95_latency_seconds"] == pytest.approx(0.42)


# -- build_aic_keyframe_mapping_csv -------------------------------------------


def _write_map_keyframes_csv(root: Path, video_id: str, rows: list[tuple[str, str, str, str]]) -> None:
    csv_dir = root / "map_keyframes"
    csv_dir.mkdir(parents=True, exist_ok=True)
    lines = ["n,pts_time,fps,frame_idx"]
    lines.extend(",".join(row) for row in rows)
    (csv_dir / f"{video_id}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _touch_keyframe(root: Path, video_id: str, keyframe_id: str) -> None:
    directory = root / "keyframes" / video_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{keyframe_id}.jpg").write_bytes(b"")


def test_build_aic_keyframe_mapping_csv_includes_only_existing_jpegs(tmp_path: Path) -> None:
    _write_map_keyframes_csv(
        tmp_path, "L21_V001",
        [("1", "0.0", "30.0", "0"), ("3", "8.7", "30.0", "261")],
    )
    _touch_keyframe(tmp_path, "L21_V001", "001")
    # "003.jpg" deliberately NOT created -> that row must be skipped.

    output = build_aic_keyframe_mapping_csv(
        tmp_path, tmp_path / "generated" / "keyframe_mapping.csv", ["L21_V001"]
    )

    import csv as csv_module

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["keyframe_id"] == "001"
    assert rows[0]["video_id"] == "L21_V001"
    assert rows[0]["timestamp"] == "0.0"
    assert Path(rows[0]["image_path"]).name == "001.jpg"


def test_build_aic_keyframe_mapping_csv_spans_multiple_videos(tmp_path: Path) -> None:
    _write_map_keyframes_csv(tmp_path, "L21_V001", [("1", "0.0", "30.0", "0")])
    _write_map_keyframes_csv(tmp_path, "L21_V002", [("1", "0.0", "25.0", "0")])
    _touch_keyframe(tmp_path, "L21_V001", "001")
    _touch_keyframe(tmp_path, "L21_V002", "001")

    output = build_aic_keyframe_mapping_csv(
        tmp_path, tmp_path / "generated" / "keyframe_mapping.csv", ["L21_V001", "L21_V002"]
    )

    import csv as csv_module

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))

    assert {row["video_id"] for row in rows} == {"L21_V001", "L21_V002"}
