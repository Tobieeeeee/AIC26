import pytest


def test_compute_retrieval_metrics_preserves_lab4_math() -> None:
    from mvr.evaluation.metrics import compute_retrieval_metrics

    assert compute_retrieval_metrics(
        ["x", "b", "a", "z"],
        ["a", "b"],
        k=3,
    ) == {
        "recall": 1.0,
        "hit": 1.0,
        "reciprocal_rank": 0.5,
    }


def test_compute_retrieval_metrics_preserves_lab4_empty_relevance_behavior() -> None:
    from mvr.evaluation.metrics import compute_retrieval_metrics

    assert compute_retrieval_metrics(["x"], [], 1) == {
        "recall": 0.0,
        "hit": 0.0,
        "reciprocal_rank": 0.0,
    }


def test_recall_and_mrr_aggregate_queries() -> None:
    from mvr.evaluation.metrics import mean_reciprocal_rank, recall_at_k

    rankings = [["a", "x"], ["x", "b"], ["x", "y"]]
    relevant = [{"a"}, {"b"}, {"z"}]

    assert recall_at_k(rankings, relevant, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(rankings, relevant, k=2) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank(rankings, relevant) == pytest.approx(0.5)


def test_recall_at_k_averages_fraction_of_each_relevant_set() -> None:
    from mvr.evaluation.metrics import recall_at_k

    assert recall_at_k(
        [["a", "x"], ["c", "d"]],
        [{"a", "b"}, set()],
        k=2,
    ) == pytest.approx(0.25)
