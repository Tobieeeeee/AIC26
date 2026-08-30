import numpy as np
import pytest

from mvr.ingestion.models import Shot
from mvr.ingestion.semantic_dedup import (
    DedupCandidate,
    SemanticDedupConfig,
    candidate_frame_indices,
    cosine_similarity,
    select_semantic_dedup_frame_indices,
)

SAME = np.array([1.0, 0.0, 0.0], dtype=np.float32)
DIFFERENT = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _shot(start: int, end: int, fps: float = 30.0) -> Shot:
    return Shot("V", "test", 1, start, end, fps)


def test_candidate_frame_indices_matches_requested_fps() -> None:
    shot = _shot(0, 299, fps=30.0)
    frames = candidate_frame_indices(shot, fps=2.0)
    # 30 fps shot sampled at 2 fps -> one candidate every 15 frames.
    assert frames[:3] == [0, 15, 30]
    assert frames[0] == shot.start_frame
    assert frames == sorted(set(frames))


def test_cosine_similarity_identical_and_orthogonal_vectors() -> None:
    assert cosine_similarity(SAME, SAME) == pytest.approx(1.0)
    assert cosine_similarity(SAME, DIFFERENT) == pytest.approx(0.0)


def test_identical_candidates_are_deduplicated() -> None:
    shot = _shot(0, 100, fps=10.0)
    candidates = [
        DedupCandidate(0, SAME),
        DedupCandidate(5, SAME),
        DedupCandidate(10, SAME),
    ]
    result = select_semantic_dedup_frame_indices(shot, candidates)
    assert result.selected_indices == (0,)
    assert result.dropped_indices == (5, 10)
    assert result.forced_gap_keeps == ()


def test_dissimilar_candidate_is_kept() -> None:
    shot = _shot(0, 100, fps=10.0)
    candidates = [
        DedupCandidate(0, SAME),
        DedupCandidate(5, DIFFERENT),
    ]
    result = select_semantic_dedup_frame_indices(shot, candidates)
    assert result.selected_indices == (0, 5)
    assert result.dropped_indices == ()


def test_duplicate_forced_kept_past_max_gap() -> None:
    # Duplicate content the whole way, but the gap since the last kept frame
    # would exceed max_gap_seconds=8.0 (80 frames @ 10fps) without a forced keep.
    shot = _shot(0, 200, fps=10.0)
    candidates = [
        DedupCandidate(0, SAME),
        DedupCandidate(85, SAME),  # 8.5s later, still "similar" -> forced keep
    ]
    config = SemanticDedupConfig(max_gap_seconds=8.0)
    result = select_semantic_dedup_frame_indices(shot, candidates, config)
    assert result.selected_indices == (0, 85)
    assert result.forced_gap_keeps == (85,)
    assert result.dropped_indices == ()


def test_duplicate_within_max_gap_is_dropped() -> None:
    shot = _shot(0, 200, fps=10.0)
    candidates = [
        DedupCandidate(0, SAME),
        DedupCandidate(70, SAME),  # 7s later, still under the 8s gap budget
    ]
    config = SemanticDedupConfig(max_gap_seconds=8.0)
    result = select_semantic_dedup_frame_indices(shot, candidates, config)
    assert result.selected_indices == (0,)
    assert result.dropped_indices == (70,)


def test_first_candidate_of_shot_always_kept() -> None:
    shot = _shot(50, 150, fps=10.0)
    candidates = [DedupCandidate(50, SAME)]
    result = select_semantic_dedup_frame_indices(shot, candidates)
    assert result.selected_indices == (50,)


def test_last_kept_resets_after_a_kept_candidate() -> None:
    # A: kept (seed). B: dissimilar to A -> kept, becomes new last-kept.
    # C: similar to B (not A) -> dropped. Verifies comparison is against the
    # *last-kept* candidate, not the shot's first candidate.
    shot = _shot(0, 100, fps=10.0)
    candidates = [
        DedupCandidate(0, SAME),
        DedupCandidate(10, DIFFERENT),
        DedupCandidate(20, DIFFERENT),
    ]
    result = select_semantic_dedup_frame_indices(shot, candidates)
    assert result.selected_indices == (0, 10)
    assert result.dropped_indices == (20,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidate_fps": 0},
        {"similarity_metric": "euclidean"},
        {"similarity_threshold": 1.5},
        {"similarity_threshold": -0.1},
        {"max_gap_seconds": 0},
        {"force_first_frame_per_shot": False},
    ],
)
def test_config_validation_rejects_invalid_values(kwargs) -> None:
    config = SemanticDedupConfig(**kwargs)
    with pytest.raises(ValueError):
        config.validate()


def test_rejects_duplicate_frame_indices() -> None:
    shot = _shot(0, 100, fps=10.0)
    candidates = [DedupCandidate(0, SAME), DedupCandidate(0, DIFFERENT)]
    with pytest.raises(ValueError, match="unique"):
        select_semantic_dedup_frame_indices(shot, candidates)


def test_rejects_candidate_outside_shot_bounds() -> None:
    shot = _shot(0, 10, fps=10.0)
    candidates = [DedupCandidate(0, SAME), DedupCandidate(20, DIFFERENT)]
    with pytest.raises(ValueError, match="outside shot bounds"):
        select_semantic_dedup_frame_indices(shot, candidates)


def test_rejects_zero_norm_embedding() -> None:
    zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="positive finite norms"):
        cosine_similarity(SAME, zero)
