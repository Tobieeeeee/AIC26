"""Deterministic experimental keyframe selectors for INGEST-KF02.

This module is intentionally outside ``src/mvr``: it does not change the
production ingestion selector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return finite, row-wise L2-normalized float32 embeddings."""

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings must not contain zero-norm rows")
    return np.ascontiguousarray(values / norms, dtype=np.float32)


def _canonical_candidates(
    frame_indices: Sequence[int], embeddings: np.ndarray
) -> tuple[list[int], np.ndarray]:
    if len(frame_indices) != len(embeddings):
        raise ValueError("one embedding is required per candidate frame")
    first_by_frame: dict[int, np.ndarray] = {}
    for frame, vector in zip(frame_indices, embeddings, strict=True):
        first_by_frame.setdefault(int(frame), vector)
    frames = sorted(first_by_frame)
    vectors = normalize_embeddings(np.stack([first_by_frame[frame] for frame in frames]))
    return frames, vectors


def budget_matched_diversity(
    frame_indices: Sequence[int],
    embeddings: np.ndarray,
    *,
    midpoint_frame: int,
    budget: int,
) -> list[int]:
    """Greedy farthest-first placement with a mandatory production midpoint."""

    if budget < 1:
        raise ValueError("budget must be at least 1")
    frames, vectors = _canonical_candidates(frame_indices, embeddings)
    try:
        midpoint_position = frames.index(int(midpoint_frame))
    except ValueError as exc:
        raise ValueError("midpoint_frame must be present in candidates") from exc

    target = min(budget, len(frames))
    selected_positions = [midpoint_position]
    remaining = set(range(len(frames))) - {midpoint_position}
    while len(selected_positions) < target:
        # Cosine distance is 1 - cosine similarity. Maximizing the minimum
        # distance is equivalent to minimizing the maximum similarity.
        best = min(
            remaining,
            key=lambda position: (
                max(float(vectors[position] @ vectors[chosen]) for chosen in selected_positions),
                frames[position],
            ),
        )
        selected_positions.append(best)
        remaining.remove(best)
    return sorted(frames[position] for position in selected_positions)


def novelty_threshold_selection(
    frame_indices: Sequence[int],
    embeddings: np.ndarray,
    *,
    midpoint_frame: int,
    novelty_threshold: float,
    minimum_gap_frames: int,
    maximum_keyframes: int,
) -> list[int]:
    """Select temporal candidates whose minimum cosine distance exceeds tau."""

    if novelty_threshold < 0:
        raise ValueError("novelty_threshold must be non-negative")
    if minimum_gap_frames < 0:
        raise ValueError("minimum_gap_frames must be non-negative")
    if maximum_keyframes < 1:
        raise ValueError("maximum_keyframes must be at least 1")
    frames, vectors = _canonical_candidates(frame_indices, embeddings)
    try:
        midpoint_position = frames.index(int(midpoint_frame))
    except ValueError as exc:
        raise ValueError("midpoint_frame must be present in candidates") from exc

    selected_positions = [midpoint_position]
    for position, frame in enumerate(frames):
        if position == midpoint_position or len(selected_positions) >= maximum_keyframes:
            continue
        if any(abs(frame - frames[chosen]) < minimum_gap_frames for chosen in selected_positions):
            continue
        minimum_distance = min(
            1.0 - float(vectors[position] @ vectors[chosen])
            for chosen in selected_positions
        )
        if minimum_distance > novelty_threshold:
            selected_positions.append(position)
    return sorted(frames[position] for position in selected_positions)
