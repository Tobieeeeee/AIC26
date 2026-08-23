"""Deterministic hybrid temporal-content keyframe selection.

This module is model- and filesystem-independent.  Candidate extraction,
MetaCLIP inference, and cache persistence live in ``selector_cache`` so the
selection invariants remain cheap to exercise in CPU-only tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .models import Shot


HYBRID_SELECTOR_MODE = "hybrid_temporal_content"
HYBRID_SELECTOR_VERSION = "1.0"


@dataclass(frozen=True)
class HybridSelectorConfig:
    candidate_step_sec: float = 1.0
    max_temporal_gap_sec: float = 8.0
    clip_knn_k: int = 3
    clip_neighbor_window_sec: float = 4.0
    clip_novelty_threshold: float = 0.15
    clip_rare_threshold: float = 0.25
    min_event_gap_sec: float = 2.0
    edge_change_mad_multiplier: float = 3.0
    edge_change_min_score: float = 0.08
    midpoint_mandatory: bool = True

    def validate(self) -> None:
        positive = {
            "candidate_step_sec": self.candidate_step_sec,
            "max_temporal_gap_sec": self.max_temporal_gap_sec,
            "clip_neighbor_window_sec": self.clip_neighbor_window_sec,
            "clip_novelty_threshold": self.clip_novelty_threshold,
            "clip_rare_threshold": self.clip_rare_threshold,
            "min_event_gap_sec": self.min_event_gap_sec,
            "edge_change_mad_multiplier": self.edge_change_mad_multiplier,
            "edge_change_min_score": self.edge_change_min_score,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.clip_knn_k < 1:
            raise ValueError("clip_knn_k must be positive")
        if not self.midpoint_mandatory:
            raise ValueError("the canonical selector requires midpoint_mandatory=true")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSignal:
    frame_idx: int
    clip_novelty: float
    edge_change: float


@dataclass(frozen=True)
class HybridSelectionResult:
    selected_indices: tuple[int, ...]
    content_event_indices: tuple[int, ...]
    temporal_repair_indices: tuple[int, ...]
    edge_event_threshold: float
    max_observed_gap_sec: float


def candidate_frame_indices(shot: Shot, step_seconds: float = 1.0) -> list[int]:
    """Return midpoint plus an interior frame-space grid, sorted and unique."""
    if shot.fps <= 0:
        raise ValueError("shot.fps must be positive")
    if shot.end_frame < shot.start_frame:
        raise ValueError("shot.end_frame must be >= shot.start_frame")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    midpoint = (shot.start_frame + shot.end_frame) // 2
    step_frames = max(1, round(step_seconds * shot.fps))
    selected = {midpoint}
    frame_idx = shot.start_frame + step_frames
    while frame_idx < shot.end_frame:
        selected.add(frame_idx)
        frame_idx += step_frames
    return sorted(selected)


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("candidate embeddings must be a non-empty matrix")
    if not np.isfinite(values).all():
        raise ValueError("candidate embeddings must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("candidate embeddings must have positive finite norms")
    return np.ascontiguousarray(values / norms, dtype=np.float32)


def local_clip_novelty(
    frame_indices: list[int],
    embeddings: np.ndarray,
    *,
    fps: float,
    neighbor_window_sec: float,
    knn_k: int,
) -> np.ndarray:
    """Compute ``1 - mean(top-k cosine)`` over local temporal neighbors."""
    if len(frame_indices) != len(embeddings):
        raise ValueError("candidate frames and embeddings must have equal length")
    if fps <= 0 or neighbor_window_sec <= 0 or knn_k < 1:
        raise ValueError("invalid local novelty configuration")
    vectors = normalize_embeddings(embeddings)
    window_frames = neighbor_window_sec * fps
    novelty = np.zeros(len(frame_indices), dtype=np.float32)
    for position, frame_idx in enumerate(frame_indices):
        neighbors = [
            other
            for other, other_frame in enumerate(frame_indices)
            if other != position and abs(other_frame - frame_idx) <= window_frames
        ]
        if not neighbors:
            novelty[position] = 0.0
            continue
        similarities = vectors[neighbors] @ vectors[position]
        similarities = np.sort(similarities)[::-1][:knn_k]
        novelty[position] = max(0.0, 1.0 - float(np.mean(similarities)))
    return novelty


def robust_edge_event_threshold(
    edge_scores: list[float] | np.ndarray,
    *,
    mad_multiplier: float,
    minimum_score: float,
) -> float:
    values = np.asarray(edge_scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("edge scores must be a finite one-dimensional sequence")
    if mad_multiplier <= 0 or minimum_score <= 0:
        raise ValueError("edge threshold parameters must be positive")
    if not len(values):
        return minimum_score
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(minimum_score, median + mad_multiplier * mad)


def _content_score(
    signal: CandidateSignal,
    *,
    config: HybridSelectorConfig,
    edge_threshold: float,
) -> float:
    return max(
        signal.clip_novelty / config.clip_novelty_threshold,
        signal.edge_change / edge_threshold,
    )


def _max_gap_seconds(shot: Shot, selected: set[int]) -> float:
    points = [shot.start_frame, *sorted(selected), shot.end_frame]
    return max((right - left) / shot.fps for left, right in zip(points, points[1:]))


def select_hybrid_frame_indices(
    shot: Shot,
    signals: list[CandidateSignal],
    config: HybridSelectorConfig = HybridSelectorConfig(),
) -> HybridSelectionResult:
    """Select midpoint, deterministic content events, then repair temporal gaps."""
    config.validate()
    if shot.fps <= 0 or shot.end_frame < shot.start_frame:
        raise ValueError("invalid shot")
    ordered = sorted(signals, key=lambda item: item.frame_idx)
    frames = [item.frame_idx for item in ordered]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError("candidate signals must contain unique frame indices")
    if frames[0] < shot.start_frame or frames[-1] > shot.end_frame:
        raise ValueError("candidate signal is outside shot bounds")
    midpoint = (shot.start_frame + shot.end_frame) // 2
    if midpoint not in set(frames):
        raise ValueError("candidate signals must include the exact shot midpoint")

    edge_threshold = robust_edge_event_threshold(
        [item.edge_change for item in ordered],
        mad_multiplier=config.edge_change_mad_multiplier,
        minimum_score=config.edge_change_min_score,
    )
    selected = {midpoint}
    rare = [item for item in ordered if item.clip_novelty >= config.clip_rare_threshold]
    ordinary = [
        item
        for item in ordered
        if item not in rare
        and (
            item.clip_novelty >= config.clip_novelty_threshold
            or item.edge_change >= edge_threshold
        )
    ]
    priority = lambda item: (
        -_content_score(item, config=config, edge_threshold=edge_threshold),
        item.frame_idx,
    )
    event_indices: set[int] = {item.frame_idx for item in sorted(rare, key=priority)}
    min_gap_frames = round(config.min_event_gap_sec * shot.fps)
    for item in sorted(ordinary, key=priority):
        if all(abs(item.frame_idx - existing) >= min_gap_frames for existing in event_indices):
            event_indices.add(item.frame_idx)
    selected.update(event_indices)

    by_frame = {item.frame_idx: item for item in ordered}
    repair_indices: list[int] = []
    while True:
        points = [shot.start_frame, *sorted(selected), shot.end_frame]
        violating = [
            (right - left, left, right)
            for left, right in zip(points, points[1:])
            if (right - left) / shot.fps > config.max_temporal_gap_sec
        ]
        if not violating:
            break
        _, left, right = max(violating, key=lambda item: (item[0], -item[1]))
        available = [frame for frame in frames if left < frame < right and frame not in selected]
        if not available:
            raise RuntimeError(
                "candidate grid cannot satisfy max_temporal_gap_sec for "
                f"{shot.video_id}/S{shot.shot_id:04d}"
            )
        target = (left + right) / 2.0
        neighborhood = config.candidate_step_sec * shot.fps
        local = [frame for frame in available if abs(frame - target) <= neighborhood]
        pool = local or available
        chosen = min(
            pool,
            key=lambda frame: (
                -_content_score(
                    by_frame[frame], config=config, edge_threshold=edge_threshold
                ),
                abs(frame - target),
                frame,
            ),
        )
        selected.add(chosen)
        repair_indices.append(chosen)

    selected_indices = tuple(sorted(selected))
    return HybridSelectionResult(
        selected_indices=selected_indices,
        content_event_indices=tuple(sorted(event_indices)),
        temporal_repair_indices=tuple(sorted(repair_indices)),
        edge_event_threshold=edge_threshold,
        max_observed_gap_sec=_max_gap_seconds(shot, selected),
    )
