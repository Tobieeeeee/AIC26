from __future__ import annotations

import numpy as np

from scripts.experiments.keyframe_selection.multivideo_eval import (
    first_relevant_rank,
    retrieval_rrf_order,
    robust_adaptive_threshold,
    rrf_scores,
    select_experimental_events,
    stable_descending_ranks,
)
from mvr.ingestion.models import Shot


def test_stable_ranks_and_rrf_are_deterministic() -> None:
    assert stable_descending_ranks([0.5, 0.5, 0.2]).tolist() == [1, 2, 3]
    fused = rrf_scores([[0.9, 0.2, 0.1], [0.1, 0.8, 0.2]])
    assert fused[0] > fused[2]
    assert fused[1] > fused[2]
    assert retrieval_rrf_order([[1.0, 1.0], [1.0, 1.0]]).tolist() == [0, 1]


def test_adaptive_threshold_is_predeclared_robust_rule() -> None:
    assert robust_adaptive_threshold([0.01, 0.02, 0.03]) == 0.15
    assert np.isclose(robust_adaptive_threshold([0.1, 0.2, 0.9]), 0.5)


def test_experimental_selection_preserves_rare_and_repairs_gap() -> None:
    shot = Shot("V", "frozen", 1, 0, 600, 10.0)
    frames = list(range(50, 600, 50))
    result = select_experimental_events(
        shot,
        frames,
        [0.1, 0.2, 0.3, 0.1, 0.2, 0.0, 0.4, 0.2, 0.1, 0.2, 0.1],
        [False] * 11,
        [True, False, False, False, False, False, False, False, False, False, False],
    )
    assert 50 in result.event_indices
    assert 300 in result.selected_indices
    assert result.max_gap_sec <= 8.0
    assert result.repair_indices


def test_first_relevant_rank_requires_video_and_exact_range() -> None:
    ordered = [("OTHER", 15), ("V", 9), ("V", 15)]
    assert first_relevant_rank(ordered, video_id="V", start_frame=10, end_frame=20) == 3
    assert first_relevant_rank(ordered, video_id="V", start_frame=16, end_frame=20) is None
