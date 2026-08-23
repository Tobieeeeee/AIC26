"""Deterministic helpers for the four-video keyframe reevaluation.

This is experiment code.  It deliberately consumes the production candidate,
novelty, and temporal-repair primitives without changing production defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mvr.ingestion.hybrid_selector import CandidateSignal, HybridSelectorConfig
from mvr.ingestion.models import Shot


RRF_K = 60


@dataclass(frozen=True)
class ExperimentalSelection:
    selected_indices: tuple[int, ...]
    event_indices: tuple[int, ...]
    repair_indices: tuple[int, ...]
    max_gap_sec: float


def stable_descending_ranks(values: Sequence[float]) -> np.ndarray:
    """Return one-based ordinal ranks, breaking equal scores by input order."""
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite one-dimensional sequence")
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def rrf_scores(score_lists: Sequence[Sequence[float]], k: int = RRF_K) -> np.ndarray:
    """Fuse independent score rankings without mixing vector spaces."""
    if k < 1 or not score_lists:
        raise ValueError("RRF requires k >= 1 and at least one score list")
    lengths = {len(values) for values in score_lists}
    if len(lengths) != 1:
        raise ValueError("all RRF score lists must have equal length")
    fused = np.zeros(next(iter(lengths)), dtype=np.float64)
    for values in score_lists:
        fused += 1.0 / (k + stable_descending_ranks(values))
    return fused


def robust_adaptive_threshold(
    novelty: Sequence[float], *, base_threshold: float = 0.15, mad_multiplier: float = 3.0
) -> float:
    """Predeclared Pass-3 threshold: max(base, median + multiplier*MAD)."""
    values = np.asarray(novelty, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("novelty must be a non-empty finite sequence")
    if base_threshold <= 0 or mad_multiplier <= 0:
        raise ValueError("adaptive threshold parameters must be positive")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(base_threshold, median + mad_multiplier * mad)


def _max_gap(shot: Shot, selected: set[int]) -> float:
    points = [shot.start_frame, *sorted(selected), shot.end_frame]
    return max((right - left) / shot.fps for left, right in zip(points, points[1:]))


def select_experimental_events(
    shot: Shot,
    frame_indices: Sequence[int],
    priority_scores: Sequence[float],
    event_mask: Sequence[bool],
    rare_mask: Sequence[bool],
    *,
    config: HybridSelectorConfig = HybridSelectorConfig(),
) -> ExperimentalSelection:
    """Select preclassified events by priority, then apply canonical gap repair.

    This supports RRF and adaptive-count experiments.  Event classification is
    supplied explicitly, while midpoint, event spacing, deterministic ties, and
    temporal repair mirror the production hybrid selector.
    """
    config.validate()
    frames = [int(value) for value in frame_indices]
    scores = np.asarray(priority_scores, dtype=np.float64)
    events = np.asarray(event_mask, dtype=bool)
    rare = np.asarray(rare_mask, dtype=bool)
    if not (len(frames) == len(scores) == len(events) == len(rare)):
        raise ValueError("candidate arrays must have equal length")
    if not frames or frames != sorted(frames) or len(frames) != len(set(frames)):
        raise ValueError("candidate frames must be sorted and unique")
    if not np.isfinite(scores).all():
        raise ValueError("priority scores must be finite")
    midpoint = (shot.start_frame + shot.end_frame) // 2
    if midpoint not in frames:
        raise ValueError("candidate frames must include the exact midpoint")

    selected = {midpoint}
    event_indices: set[int] = set()
    positions = list(range(len(frames)))
    priority = lambda position: (-float(scores[position]), frames[position])
    for position in sorted((p for p in positions if rare[p]), key=priority):
        event_indices.add(frames[position])
    minimum_gap = round(config.min_event_gap_sec * shot.fps)
    for position in sorted(
        (p for p in positions if events[p] and not rare[p]), key=priority
    ):
        frame = frames[position]
        if all(abs(frame - existing) >= minimum_gap for existing in event_indices):
            event_indices.add(frame)
    selected.update(event_indices)

    score_by_frame = dict(zip(frames, scores, strict=True))
    repairs: list[int] = []
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
            raise RuntimeError("candidate grid cannot satisfy the temporal-gap invariant")
        target = (left + right) / 2.0
        neighborhood = config.candidate_step_sec * shot.fps
        local = [frame for frame in available if abs(frame - target) <= neighborhood]
        chosen = min(
            local or available,
            key=lambda frame: (-float(score_by_frame[frame]), abs(frame - target), frame),
        )
        selected.add(chosen)
        repairs.append(chosen)
    return ExperimentalSelection(
        selected_indices=tuple(sorted(selected)),
        event_indices=tuple(sorted(event_indices)),
        repair_indices=tuple(sorted(repairs)),
        max_gap_sec=_max_gap(shot, selected),
    )


def retrieval_rrf_order(score_lists: Sequence[Sequence[float]], k: int = RRF_K) -> np.ndarray:
    """Return candidate positions ordered by rank-level RRF and stable identity."""
    fused = rrf_scores(score_lists, k=k)
    return np.argsort(-fused, kind="stable")


def first_relevant_rank(
    ordered_video_frames: Sequence[tuple[str, int]],
    *,
    video_id: str,
    start_frame: int,
    end_frame: int,
) -> int | None:
    for rank, (candidate_video, frame_idx) in enumerate(ordered_video_frames, start=1):
        if candidate_video == video_id and start_frame <= frame_idx <= end_frame:
            return rank
    return None
