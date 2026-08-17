from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mvr.retrieval.frame_refinement import (
    RefinementWindowConfig,
    refine_frame,
    refine_top_candidates,
    window_bounds,
)
from mvr.retrieval.kis import KisCandidate


class _FakeEncoder:
    """Scores each image by a caller-supplied lookup keyed on frame index,
    and counts how many times encode_images is invoked (never per-frame)."""

    def __init__(self, scores_by_frame_idx: dict[int, float]) -> None:
        self._scores = scores_by_frame_idx
        self.encode_images_calls = 0

    def encode_images(self, image_paths, batch_size: int = 32):
        self.encode_images_calls += 1
        return np.array(
            [[self._scores[int(Path(p).stem)]] for p in image_paths],
            dtype=np.float32,
        )


def _fake_extractor(video_path, keyframes, work_dir) -> None:
    """Stands in for ingestion.extract.extract_keyframes without FFmpeg."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for kf in keyframes:
        (work_dir / kf.image_path).write_bytes(b"")


# -- window_bounds -----------------------------------------------------------


def test_window_bounds_clamps_lower_edge() -> None:
    assert window_bounds(center=2, radius=5, frame_count=1000) == (0, 7)


def test_window_bounds_clamps_upper_edge() -> None:
    assert window_bounds(center=997, radius=5, frame_count=1000) == (992, 999)


def test_window_bounds_mid_video_is_symmetric() -> None:
    assert window_bounds(center=500, radius=5, frame_count=1000) == (495, 505)


# -- RefinementWindowConfig: FPS/time conversion -----------------------------


def test_time_radius_converts_using_fps() -> None:
    config = RefinementWindowConfig(time_radius_seconds=2.0)
    assert config.resolve_radius_frames(fps=30.0) == 60


def test_frame_radius_takes_precedence_over_time_radius() -> None:
    config = RefinementWindowConfig(frame_radius=3, time_radius_seconds=100.0)
    assert config.resolve_radius_frames(fps=30.0) == 3


def test_radius_config_requires_one_of_frame_or_time() -> None:
    with pytest.raises(ValueError):
        RefinementWindowConfig().resolve_radius_frames(fps=30.0)


def test_time_radius_requires_positive_fps() -> None:
    with pytest.raises(ValueError):
        RefinementWindowConfig(time_radius_seconds=1.0).resolve_radius_frames(fps=0.0)


# -- refine_frame: exact original-frame return -------------------------------


def test_refine_frame_returns_exact_best_scoring_original_frame(tmp_path) -> None:
    scores = {248: 0.1, 249: 0.2, 250: 0.3, 251: 0.9, 252: 0.4}
    encoder = _FakeEncoder(scores)

    result = refine_frame(
        video_path=Path("unused.mp4"),
        video_id="L21_V001",
        coarse_frame_id=250,
        query_vector=np.array([1.0]),
        encoder=encoder,
        fps=30.0,
        frame_count=100000,
        window=RefinementWindowConfig(frame_radius=2),
        tmp_dir=tmp_path,
        extractor=_fake_extractor,
    )

    assert result.video_id == "L21_V001"
    assert result.frame_id == 251
    assert result.score == pytest.approx(0.9)
    assert result.evidence["window_lower"] == 248
    assert result.evidence["window_upper"] == 252
    assert result.evidence["candidates_scored"] == 5


# -- deterministic tie-break --------------------------------------------------


def test_tie_break_prefers_frame_closest_to_coarse_candidate(tmp_path) -> None:
    scores = {248: 0.5, 249: 0.9, 250: 0.9, 251: 0.9, 252: 0.1}
    encoder = _FakeEncoder(scores)

    result = refine_frame(
        video_path=Path("unused.mp4"),
        video_id="L21_V001",
        coarse_frame_id=250,
        query_vector=np.array([1.0]),
        encoder=encoder,
        fps=30.0,
        frame_count=100000,
        window=RefinementWindowConfig(frame_radius=2),
        tmp_dir=tmp_path,
        extractor=_fake_extractor,
    )

    assert result.frame_id == 250  # exact coarse candidate wins the 3-way tie


def test_tie_break_falls_back_to_lower_index_when_distance_tied(tmp_path) -> None:
    scores = {249: 0.9, 250: 0.5, 251: 0.9}
    encoder = _FakeEncoder(scores)

    result = refine_frame(
        video_path=Path("unused.mp4"),
        video_id="L21_V001",
        coarse_frame_id=250,
        query_vector=np.array([1.0]),
        encoder=encoder,
        fps=30.0,
        frame_count=100000,
        window=RefinementWindowConfig(frame_radius=1),
        tmp_dir=tmp_path,
        extractor=_fake_extractor,
    )

    assert result.frame_id == 249  # both tied at distance 1; lower index wins


# -- no new model load per frame ---------------------------------------------


def test_refine_frame_calls_encoder_exactly_once_regardless_of_window_size(tmp_path) -> None:
    scores = {idx: float(idx) for idx in range(240, 262)}
    encoder = _FakeEncoder(scores)

    refine_frame(
        video_path=Path("unused.mp4"),
        video_id="L21_V001",
        coarse_frame_id=250,
        query_vector=np.array([1.0]),
        encoder=encoder,
        fps=30.0,
        frame_count=100000,
        window=RefinementWindowConfig(frame_radius=10),  # 21 candidate frames
        tmp_dir=tmp_path,
        extractor=_fake_extractor,
    )

    assert encoder.encode_images_calls == 1


# -- refine_top_candidates: disabled / top-N ---------------------------------


def _candidate(video_id: str, frame_id: int, rank: int) -> KisCandidate:
    return KisCandidate(
        rank=rank, video_id=video_id, frame_id=frame_id, score=1.0 - rank / 10,
        keyframe_id=f"kf-{rank}",
    )


def test_refine_top_candidates_disabled_leaves_coarse_result_unchanged() -> None:
    candidates = [_candidate("L21_V001", 100 + i, i) for i in range(5)]

    result = refine_top_candidates(
        candidates,
        query_vector=np.array([1.0]),
        encoder=_FakeEncoder({}),
        video_path_for=lambda video_id: Path("unused.mp4"),
        video_info_for=lambda video_id: None,
        window=RefinementWindowConfig(frame_radius=2),
        top_n=5,
        enabled=False,
    )

    assert result == candidates


def test_refine_top_candidates_respects_top_n_limit(tmp_path) -> None:
    from dataclasses import dataclass

    @dataclass
    class _FakeVideoInfo:
        fps: float
        frame_count: int

    candidates = [_candidate("L21_V001", 100 + i * 10, i) for i in range(5)]
    scores = {idx: float(idx) for idx in range(0, 300)}
    encoder = _FakeEncoder(scores)

    result = refine_top_candidates(
        candidates,
        query_vector=np.array([1.0]),
        encoder=encoder,
        video_path_for=lambda video_id: Path("unused.mp4"),
        video_info_for=lambda video_id: _FakeVideoInfo(fps=30.0, frame_count=100000),
        window=RefinementWindowConfig(frame_radius=1),
        top_n=2,
        enabled=True,
        tmp_dir=tmp_path,
        extractor=_fake_extractor,
    )

    # First 2 candidates were refined (frame_id shifted to the window max).
    assert result[0].frame_id == 101
    assert result[1].frame_id == 111
    # The remaining 3 are untouched coarse candidates.
    assert result[2] == candidates[2]
    assert result[3] == candidates[3]
    assert result[4] == candidates[4]
    # encoder is reused, not reloaded: exactly one call per refined candidate.
    assert encoder.encode_images_calls == 2


def test_refine_top_candidates_empty_input_returns_empty() -> None:
    result = refine_top_candidates(
        [],
        query_vector=np.array([1.0]),
        encoder=_FakeEncoder({}),
        video_path_for=lambda video_id: Path("unused.mp4"),
        video_info_for=lambda video_id: None,
        window=RefinementWindowConfig(frame_radius=2),
    )

    assert result == []
