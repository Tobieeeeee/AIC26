from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "experiments"
    / "keyframe_selection"
    / "adaptive_selector.py"
)
SPEC = importlib.util.spec_from_file_location("ingest_kf02_adaptive_selector", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
adaptive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adaptive)


def test_budget_one_returns_midpoint_and_duplicate_candidate_is_suppressed() -> None:
    assert adaptive.budget_matched_diversity(
        [0, 5, 5, 10],
        np.array([[1, 0], [0, 1], [0, 1], [-1, 0]], dtype=np.float32),
        midpoint_frame=5,
        budget=1,
    ) == [5]


def test_budget_is_exact_temporally_sorted_and_uses_separated_clusters() -> None:
    selected = adaptive.budget_matched_diversity(
        [0, 10, 20, 30],
        np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32),
        midpoint_frame=20,
        budget=3,
    )
    assert selected == [0, 20, 30]


def test_budget_candidate_exhaustion_and_identical_embedding_tie_are_deterministic() -> None:
    embeddings = np.ones((3, 2), dtype=np.float32)
    first = adaptive.budget_matched_diversity(
        [30, 10, 20], embeddings, midpoint_frame=20, budget=9
    )
    second = adaptive.budget_matched_diversity(
        [30, 10, 20], embeddings, midpoint_frame=20, budget=9
    )
    assert first == second == [10, 20, 30]


def test_cosine_normalization_and_invalid_vectors() -> None:
    normalized = adaptive.normalize_embeddings(np.array([[3, 4], [0, 2]], dtype=np.float32))
    assert np.linalg.norm(normalized, axis=1) == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError, match="zero-norm"):
        adaptive.normalize_embeddings(np.array([[0, 0]], dtype=np.float32))


def test_novelty_threshold_is_strict_at_boundary() -> None:
    # Distance between the two normalized orthogonal vectors is exactly 1.
    assert adaptive.novelty_threshold_selection(
        [0, 10], np.eye(2, dtype=np.float32), midpoint_frame=0,
        novelty_threshold=1.0, minimum_gap_frames=0, maximum_keyframes=2,
    ) == [0]
    assert adaptive.novelty_threshold_selection(
        [0, 10], np.eye(2, dtype=np.float32), midpoint_frame=0,
        novelty_threshold=0.999, minimum_gap_frames=0, maximum_keyframes=2,
    ) == [0, 10]


def test_novelty_minimum_gap_hard_cap_and_temporal_order() -> None:
    selected = adaptive.novelty_threshold_selection(
        [30, 0, 20, 10],
        np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.float32),
        midpoint_frame=20,
        novelty_threshold=0.1,
        minimum_gap_frames=15,
        maximum_keyframes=2,
    )
    assert selected == [0, 20]


def test_retrieval_metrics_use_first_relevant_rank() -> None:
    path = MODULE_PATH.with_name("evaluate_retrieval.py")
    spec = importlib.util.spec_from_file_location("ingest_kf02_retrieval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.retrieval_metrics([1, 10, 100, None]) == pytest.approx(
        {
            "recall_at_1": 0.25,
            "recall_at_10": 0.5,
            "recall_at_100": 0.75,
            "mrr": (1 + 0.1 + 0.01) / 4,
        }
    )
