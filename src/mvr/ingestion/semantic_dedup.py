"""Deterministic sequential semantic-dedup keyframe selection.

Proposal from the "skill pixel" team (relayed by Tobie), reviewed against
``hybrid_selector.py``: candidates are sampled at a fixed rate per shot and
compared against the *last-kept* candidate only -- an O(n) single-comparison
pass, unlike ``hybrid_selector``'s O(n * window) KNN-neighborhood novelty.
This trades some robustness (a slow content drift across many near-duplicate
frames is invisible to a last-kept-only comparison) for much lower cost,
which matters on CPU-only hardware.

This module is model- and filesystem-independent, matching
``hybrid_selector.py``'s separation of concerns: embedding computation and
caching belong in ``selector_cache.py`` (or an equivalent), not here. Callers
pass one already-computed embedding per candidate.

Not yet implemented: the spec's "optional quality-aware replacement" step
(swap a kept frame for a sharper/better-quality near-duplicate). No metric
for "quality" was specified by the proposal, so nothing is guessed here --
add it once the metric is defined.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .models import Shot

SEMANTIC_DEDUP_MODE = "semantic_dedup"
SEMANTIC_DEDUP_VERSION = "1.0"


@dataclass(frozen=True)
class SemanticDedupConfig:
    candidate_fps: float = 2.0
    similarity_metric: str = "cosine"
    similarity_threshold: float = 0.85
    max_gap_seconds: float = 8.0
    force_first_frame_per_shot: bool = True

    def validate(self) -> None:
        if self.candidate_fps <= 0:
            raise ValueError("candidate_fps must be positive")
        if self.similarity_metric != "cosine":
            raise ValueError("only cosine similarity_metric is supported")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be within [0, 1]")
        if self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        if not self.force_first_frame_per_shot:
            raise ValueError(
                "the canonical selector requires force_first_frame_per_shot=true"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DedupCandidate:
    frame_idx: int
    embedding: np.ndarray


@dataclass(frozen=True)
class SemanticDedupResult:
    selected_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...]
    forced_gap_keeps: tuple[int, ...]


def candidate_frame_indices(shot: Shot, fps: float = 2.0) -> list[int]:
    """Return a uniform frame-space grid at ``fps`` samples/sec, shot-relative."""
    if shot.fps <= 0:
        raise ValueError("shot.fps must be positive")
    if shot.end_frame < shot.start_frame:
        raise ValueError("shot.end_frame must be >= shot.start_frame")
    if fps <= 0:
        raise ValueError("fps must be positive")
    step_frames = max(1, round(shot.fps / fps))
    return list(range(shot.start_frame, shot.end_frame + 1, step_frames))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 0 or norm_b <= 0 or not np.isfinite(norm_a) or not np.isfinite(norm_b):
        raise ValueError("embeddings must have positive finite norms")
    return float(np.dot(a, b) / (norm_a * norm_b))


def select_semantic_dedup_frame_indices(
    shot: Shot,
    candidates: list[DedupCandidate],
    config: SemanticDedupConfig = SemanticDedupConfig(),
) -> SemanticDedupResult:
    """Greedy sequential dedup against the last-kept candidate only.

    Decision per candidate (matches the proposal's decision tree exactly):
      - similarity <= threshold           -> KEEP
      - similarity >  threshold, gap >= max_gap_seconds -> KEEP (forced)
      - similarity >  threshold, gap <  max_gap_seconds -> DROP

    The shot's first candidate is always kept (force_first_frame_per_shot),
    which is also this function's "new shot -> force keep" case: a caller
    invokes this once per shot, so the first candidate here always *is* the
    first frame of a (new) shot.
    """
    config.validate()
    if shot.fps <= 0 or shot.end_frame < shot.start_frame:
        raise ValueError("invalid shot")
    ordered = sorted(candidates, key=lambda c: c.frame_idx)
    frames = [c.frame_idx for c in ordered]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError("candidates must contain unique frame indices")
    if frames[0] < shot.start_frame or frames[-1] > shot.end_frame:
        raise ValueError("candidate is outside shot bounds")

    max_gap_frames = config.max_gap_seconds * shot.fps

    first = ordered[0]
    selected = [first.frame_idx]
    dropped: list[int] = []
    forced_gap_keeps: list[int] = []
    last_kept = first

    for candidate in ordered[1:]:
        similarity = cosine_similarity(last_kept.embedding, candidate.embedding)
        gap_frames = candidate.frame_idx - last_kept.frame_idx
        if similarity <= config.similarity_threshold:
            selected.append(candidate.frame_idx)
            last_kept = candidate
        elif gap_frames >= max_gap_frames:
            selected.append(candidate.frame_idx)
            forced_gap_keeps.append(candidate.frame_idx)
            last_kept = candidate
        else:
            dropped.append(candidate.frame_idx)

    return SemanticDedupResult(
        selected_indices=tuple(selected),
        dropped_indices=tuple(dropped),
        forced_gap_keeps=tuple(forced_gap_keeps),
    )
