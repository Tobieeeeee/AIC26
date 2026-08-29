from __future__ import annotations

from pathlib import Path

from mvr.evaluation.sanity_contracts import (
    DeterminismResult,
    QueryRunResult,
    RetrievalHit,
    SanityQuery,
    SanityRun,
    ValidationResult,
)
from mvr.evaluation.sanity_regression import (
    build_snapshot,
    compare_snapshots,
    load_snapshot,
    save_snapshot,
)


def _run(ids: list[str]) -> SanityRun:
    hits = tuple(
        RetrievalHit(rank=rank, score=1.0, keyframe_id=item, video_id="v", frame_id=rank)
        for rank, item in enumerate(ids, start=1)
    )
    result = QueryRunResult(
        provider_id="fake",
        query_id="q1",
        repetition=1,
        top_k_requested=3,
        returned_count=len(hits),
        unique_keyframe_count=len(hits),
        latency_ms=1.0,
        expectation_met=True,
        hits=hits,
        validation=ValidationResult(True, True, True, True),
    )
    return SanityRun(
        manifest={
            "timestamp": "2026-08-20T00:00:00Z",
            "git_commit": "abc",
            "git_dirty": False,
            "query_suite_path": "queries.json",
            "query_suite_sha256": "hash",
            "top_k": 3,
            "providers": [{"provider_id": "fake"}],
        },
        queries=(SanityQuery("q1", "query"),),
        results=[result],
        determinism=[
            DeterminismResult("fake", "q1", False, None, None)
        ],
    )


def test_snapshot_round_trip(tmp_path: Path) -> None:
    snapshot = build_snapshot(_run(["a", "b"]))
    path = tmp_path / "snapshot.json"
    save_snapshot(path, snapshot)
    assert load_snapshot(path) == snapshot


def test_regression_detects_added_removed_and_rank_displacement() -> None:
    baseline = build_snapshot(_run(["a", "b", "c"]))
    current = build_snapshot(_run(["b", "a", "d"]))

    comparison = compare_snapshots(baseline, current)

    assert comparison["changed_ranking_count"] == 1
    difference = comparison["differences"][0]
    assert difference["added_keyframe_ids"] == ["d"]
    assert difference["removed_keyframe_ids"] == ["c"]
    assert difference["rank_displacement"] == {"a": 1, "b": -1}
    assert "better" not in comparison["interpretation"]


def test_regression_exact_match_is_reported_as_unchanged() -> None:
    snapshot = build_snapshot(_run(["a", "b"]))
    comparison = compare_snapshots(snapshot, snapshot)
    assert comparison["all_exact_order_match"] is True
    assert comparison["changed_ranking_count"] == 0
