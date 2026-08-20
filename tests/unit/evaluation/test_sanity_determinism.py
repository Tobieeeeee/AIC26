from __future__ import annotations

from mvr.evaluation.sanity_contracts import QueryRunResult, RetrievalHit, ValidationResult
from mvr.evaluation.sanity_determinism import compare_repetitions


VALID = ValidationResult(True, True, True, True)


def _result(
    repetition: int,
    ids: list[str],
    *,
    scores: list[float] | None = None,
) -> QueryRunResult:
    scores = scores or [1.0 - index * 0.1 for index in range(len(ids))]
    hits = tuple(
        RetrievalHit(
            rank=rank,
            score=score,
            keyframe_id=keyframe_id,
            video_id="video-a",
            frame_id=rank,
        )
        for rank, (keyframe_id, score) in enumerate(zip(ids, scores), start=1)
    )
    return QueryRunResult(
        provider_id="fake",
        query_id="q1",
        repetition=repetition,
        top_k_requested=10,
        returned_count=len(hits),
        unique_keyframe_count=len(hits),
        latency_ms=1.0,
        expectation_met=True,
        hits=hits,
        validation=VALID,
    )


def test_determinism_accepts_same_order_with_tolerated_score_drift() -> None:
    result = compare_repetitions(
        [
            _result(1, ["a", "b"], scores=[1.0, 0.5]),
            _result(2, ["a", "b"], scores=[1.0 + 1e-10, 0.5]),
            _result(3, ["a", "b"], scores=[1.0, 0.5]),
        ],
        score_tolerance=1e-8,
    )
    assert result.comparable is True
    assert result.deterministic_order is True
    assert result.score_drift_detected is False


def test_determinism_reports_order_change_and_score_drift() -> None:
    result = compare_repetitions(
        [
            _result(1, ["a", "b"], scores=[1.0, 0.5]),
            _result(2, ["b", "a"], scores=[0.9, 0.4]),
        ],
        score_tolerance=1e-8,
    )
    assert result.deterministic_order is False
    assert result.score_drift_detected is False
    assert result.ranking_changes[0]["repetition"] == 2


def test_determinism_marks_error_repetitions_not_comparable() -> None:
    failed = QueryRunResult(
        provider_id="fake",
        query_id="q1",
        repetition=2,
        top_k_requested=10,
        returned_count=0,
        unique_keyframe_count=0,
        latency_ms=1.0,
        expectation_met=False,
        error="backend unavailable",
    )
    result = compare_repetitions([_result(1, ["a"]), failed])
    assert result.comparable is False
    assert result.deterministic_order is None
