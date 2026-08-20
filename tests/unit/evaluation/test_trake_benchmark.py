import pytest

from mvr.evaluation.trake import TrakeChainPrediction, TrakeGroundTruth
from mvr.evaluation.trake_benchmark import (
    TrakeBenchmarkQuery,
    TrakeBenchmarkRecord,
    run_trake_benchmark_query,
    summarize_trake_benchmark,
)


def test_run_trake_benchmark_query_scores_via_official_evaluator() -> None:
    query = TrakeBenchmarkQuery(
        query_id="q1",
        query="SENTINEL_QUERY",  # opaque to run_trake_benchmark_query, passed straight to search_fn
        gt=TrakeGroundTruth(video_id="L21_V024", frames=(13890, 14130, 14430, 14635)),
    )

    def fake_search(passed_query):
        assert passed_query == "SENTINEL_QUERY"
        return [TrakeChainPrediction("L21_V024", (13890, 14130, 14430, 14635), score=2.5)]

    record = run_trake_benchmark_query(query, config_name="atlas", search_fn=fake_search)

    assert record.query_id == "q1"
    assert record.config_name == "atlas"
    assert record.final_score == 1.0
    assert all(v == 1.0 for v in record.r_at_k.values())
    assert record.latency_seconds >= 0.0


def test_run_trake_benchmark_query_scores_partial_credit() -> None:
    query = TrakeBenchmarkQuery(
        query_id="q2",
        query="q",
        gt=TrakeGroundTruth(video_id="L21_V018", frames=(11068, 11340, 11600)),
    )

    def fake_search(_query):
        # only 2 of 3 markers correct
        return [TrakeChainPrediction("L21_V018", (11068, 11340, 99999), score=1.0)]

    record = run_trake_benchmark_query(query, config_name="lucifer", search_fn=fake_search)

    assert record.final_score == pytest.approx(2 / 3)


def test_run_trake_benchmark_query_scores_zero_when_wrong_video() -> None:
    query = TrakeBenchmarkQuery(
        query_id="q3", query="q", gt=TrakeGroundTruth(video_id="L21_V024", frames=(1, 2))
    )

    record = run_trake_benchmark_query(
        query, config_name="ucese",
        search_fn=lambda _q: [TrakeChainPrediction("WRONG_VIDEO", (1, 2), score=5.0)],
    )

    assert record.final_score == 0.0


def _record(config: str, qid: str, final_score: float, latency: float) -> TrakeBenchmarkRecord:
    return TrakeBenchmarkRecord(
        query_id=qid, config_name=config,
        r_at_k={1: final_score, 5: final_score, 20: final_score, 50: final_score, 100: final_score},
        final_score=final_score, latency_seconds=latency,
    )


def test_summarize_trake_benchmark_groups_by_config() -> None:
    records = [
        _record("atlas", "q1", 1.0, 0.10),
        _record("atlas", "q2", 0.0, 0.20),
        _record("lucifer", "q1", 1.0, 0.50),
        _record("lucifer", "q2", 0.75, 0.60),
    ]

    summary = summarize_trake_benchmark(records)

    assert set(summary.keys()) == {"atlas", "lucifer"}
    assert summary["atlas"]["query_count"] == 2
    assert summary["atlas"]["mean_final_score"] == pytest.approx(0.5)
    assert summary["lucifer"]["mean_final_score"] == pytest.approx(0.875)


def test_summarize_trake_benchmark_computes_median_and_latency_stats() -> None:
    records = [
        _record("atlas", "q1", 1.0, 0.10),
        _record("atlas", "q2", 0.5, 0.20),
        _record("atlas", "q3", 0.0, 0.30),
    ]

    summary = summarize_trake_benchmark(records)["atlas"]

    assert summary["median_final_score"] == pytest.approx(0.5)
    assert summary["mean_latency_seconds"] == pytest.approx(0.2)
    assert summary["p95_latency_seconds"] >= summary["mean_latency_seconds"]


def test_summarize_trake_benchmark_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        summarize_trake_benchmark([])


def test_summarize_trake_benchmark_single_record_p95_equals_value() -> None:
    summary = summarize_trake_benchmark([_record("atlas", "q1", 1.0, 0.42)])["atlas"]

    assert summary["p95_latency_seconds"] == pytest.approx(0.42)
