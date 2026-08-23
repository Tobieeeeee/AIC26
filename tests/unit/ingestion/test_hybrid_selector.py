import numpy as np

from mvr.ingestion.hybrid_selector import (
    CandidateSignal,
    HybridSelectorConfig,
    candidate_frame_indices,
    local_clip_novelty,
    select_hybrid_frame_indices,
)
from mvr.ingestion.models import Shot


def _static_signals(shot: Shot, *, edge_events: dict[int, float] | None = None):
    frames = candidate_frame_indices(shot)
    edge_events = edge_events or {}
    return [CandidateSignal(frame, 0.0, edge_events.get(frame, 0.0)) for frame in frames]


def test_short_static_shot_selects_midpoint_only() -> None:
    shot = Shot("V", "test", 1, 0, 149, 30.0)
    result = select_hybrid_frame_indices(shot, _static_signals(shot))
    assert result.selected_indices == (74,)
    assert result.content_event_indices == ()
    assert result.temporal_repair_indices == ()


def test_midpoint_candidate_is_exact_and_grid_is_bounded() -> None:
    shot = Shot("V", "test", 1, 100, 199, 25.0)
    frames = candidate_frame_indices(shot)
    assert (shot.start_frame + shot.end_frame) // 2 in frames
    assert frames == sorted(set(frames))
    assert all(shot.start_frame <= frame <= shot.end_frame for frame in frames)


def test_lecture_slide_changes_are_retained_with_temporal_coverage() -> None:
    shot = Shot("V", "test", 1, 0, 60, 1.0)
    result = select_hybrid_frame_indices(
        shot,
        _static_signals(shot, edge_events={20: 0.40, 40: 0.35}),
    )
    assert {20, 40}.issubset(result.selected_indices)
    assert {20, 40}.issubset(result.content_event_indices)
    assert result.max_observed_gap_sec <= 8.0


def test_content_event_can_add_frame_to_short_shot() -> None:
    shot = Shot("V", "test", 1, 0, 6, 1.0)
    result = select_hybrid_frame_indices(
        shot, _static_signals(shot, edge_events={1: 0.4})
    )
    assert result.selected_indices == (1, 3)


def test_rare_transient_survives_even_near_another_event() -> None:
    shot = Shot("V", "test", 1, 0, 20, 1.0)
    signals = _static_signals(shot)
    signals = [
        CandidateSignal(item.frame_idx, 0.40 if item.frame_idx == 11 else 0.20 if item.frame_idx == 10 else 0.0, 0.0)
        for item in signals
    ]
    result = select_hybrid_frame_indices(shot, signals)
    assert 11 in result.content_event_indices


def test_deterministic_ties_prefer_earlier_frame() -> None:
    shot = Shot("V", "test", 1, 0, 10, 1.0)
    signals = [
        CandidateSignal(frame, 0.20 if frame in {4, 5} else 0.0, 0.0)
        for frame in candidate_frame_indices(shot)
    ]
    first = select_hybrid_frame_indices(shot, signals)
    second = select_hybrid_frame_indices(shot, list(reversed(signals)))
    assert first == second
    assert 4 in first.content_event_indices
    assert 5 not in first.content_event_indices


def test_local_novelty_uses_only_temporal_top_k_neighbors() -> None:
    frames = [0, 1, 2, 10]
    vectors = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    novelty = local_clip_novelty(
        frames, vectors, fps=1.0, neighbor_window_sec=2.0, knn_k=1
    )
    assert novelty.tolist() == [0.0, 0.0, 1.0, 0.0]


def test_long_static_shot_always_satisfies_max_gap_invariant() -> None:
    shot = Shot("V", "test", 1, 0, 60, 1.0)
    result = select_hybrid_frame_indices(shot, _static_signals(shot))
    points = [shot.start_frame, *result.selected_indices, shot.end_frame]
    assert max(right - left for left, right in zip(points, points[1:])) <= 8
    assert result.selected_indices == tuple(sorted(set(result.selected_indices)))
