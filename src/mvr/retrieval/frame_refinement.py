"""Exact-frame refinement on the ORIGINAL video.

Given a coarse KIS candidate ``(video_id, frame_id)``, decode a small bounded
window of nearby ORIGINAL-video frames, score each against the query's
already-computed embedding using the caller's existing encoder (never a
second duplicate model load), and return the best-scoring frame. The result
is still an exact original-video frame index -- never an organizer-keyframe
ordinal or a timestamp approximation.
"""
from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..ingestion.extract import extract_keyframes
from ..ingestion.models import Keyframe


@dataclass(frozen=True)
class RefinementWindowConfig:
    """Configures the local search window around one coarse frame.

    Provide exactly one of ``frame_radius`` (frame count) or
    ``time_radius_seconds`` (converted via a trustworthy FPS at resolve
    time); ``frame_radius`` takes explicit precedence if both are set.
    """

    frame_radius: int | None = None
    time_radius_seconds: float | None = None

    def resolve_radius_frames(self, fps: float) -> int:
        if self.frame_radius is not None:
            if self.frame_radius < 0:
                raise ValueError("frame_radius must be non-negative")
            return self.frame_radius
        if self.time_radius_seconds is not None:
            if self.time_radius_seconds < 0:
                raise ValueError("time_radius_seconds must be non-negative")
            if fps <= 0:
                raise ValueError("fps must be positive to convert a time radius")
            return round(self.time_radius_seconds * fps)
        raise ValueError("either frame_radius or time_radius_seconds must be set")


@dataclass(frozen=True)
class RefinedFrame:
    video_id: str
    frame_id: int
    score: float
    evidence: dict[str, Any]


def window_bounds(center: int, radius: int, frame_count: int) -> tuple[int, int]:
    """Clamp a [center-radius, center+radius] window to valid video bounds."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if center < 0:
        raise ValueError("center must be non-negative")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    lower = max(0, center - radius)
    upper = min(frame_count - 1, center + radius)
    return lower, upper


def refine_frame(
    *,
    video_path: Path,
    video_id: str,
    coarse_frame_id: int,
    query_vector: np.ndarray,
    encoder: Any,
    fps: float,
    frame_count: int,
    window: RefinementWindowConfig,
    tmp_dir: Path | None = None,
    extractor: Any = extract_keyframes,
) -> RefinedFrame:
    """Refine one coarse candidate via a bounded local decode + rescoring pass.

    ``encoder`` must already be constructed by the caller (its
    ``encode_images`` is called exactly once for the whole window) -- this
    function never loads a model itself. ``extractor`` defaults to the
    existing exact single-pass FFmpeg decoder from ingestion; tests inject a
    fake to avoid requiring FFmpeg for the required unit-test suite.
    """
    radius = window.resolve_radius_frames(fps)
    lower, upper = window_bounds(coarse_frame_id, radius, frame_count)
    candidate_indices = list(range(lower, upper + 1))

    keyframes = [
        Keyframe(
            keyframe_id=f"{video_id}__refine__{idx}",
            video_id=video_id,
            shot_id=0,
            ordinal=position + 1,
            frame_idx=idx,
            timestamp=idx / fps,
            image_path=f"{idx}.jpg",
        )
        for position, idx in enumerate(candidate_indices)
    ]

    owns_tmp_dir = tmp_dir is None
    work_dir = Path(tempfile.mkdtemp(prefix="mvr_refine_")) if owns_tmp_dir else Path(tmp_dir)
    try:
        extractor(video_path, keyframes, work_dir)
        image_paths = [work_dir / kf.image_path for kf in keyframes]
        embeddings = np.asarray(encoder.encode_images(image_paths), dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32)
        scores = embeddings @ query

        best_score = float(np.max(scores))
        tied = [i for i, s in enumerate(scores) if float(s) == best_score]
        # Deterministic tie-break: prefer the frame closest to the coarse
        # candidate, then the lower original frame index.
        tied.sort(key=lambda i: (abs(candidate_indices[i] - coarse_frame_id), candidate_indices[i]))
        best_i = tied[0]

        return RefinedFrame(
            video_id=video_id,
            frame_id=candidate_indices[best_i],
            score=best_score,
            evidence={
                "window_lower": lower,
                "window_upper": upper,
                "candidates_scored": len(candidate_indices),
            },
        )
    finally:
        if owns_tmp_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def refine_top_candidates(
    candidates: Sequence[Any],
    *,
    query_vector: np.ndarray,
    encoder: Any,
    video_path_for: Any,
    video_info_for: Any,
    window: RefinementWindowConfig,
    top_n: int = 5,
    enabled: bool = True,
    tmp_dir: Path | None = None,
    extractor: Any = extract_keyframes,
) -> list[Any]:
    """Refine only the first ``top_n`` ranked candidates; leave the rest as-is.

    ``candidates`` must support ``dataclasses.replace`` with ``frame_id`` and
    ``score`` fields (e.g. ``mvr.retrieval.kis.KisCandidate``). When
    ``enabled`` is False, or ``candidates`` is empty, the input is returned
    unchanged -- refinement never runs implicitly.
    """
    if not enabled:
        return list(candidates)

    refined: list[Any] = []
    for position, candidate in enumerate(candidates):
        if position >= top_n:
            refined.append(candidate)
            continue
        info = video_info_for(candidate.video_id)
        try:
            result = refine_frame(
                video_path=video_path_for(candidate.video_id),
                video_id=candidate.video_id,
                coarse_frame_id=candidate.frame_id,
                query_vector=query_vector,
                encoder=encoder,
                fps=info.fps,
                frame_count=info.frame_count,
                window=window,
                tmp_dir=tmp_dir,
                extractor=extractor,
            )
        except (OSError, RuntimeError):
            # A transient decode/subprocess failure for this one candidate
            # must not abort the whole batch: fall back to its unrefined
            # coarse result rather than crash or silently guess a frame_id.
            refined.append(candidate)
            continue
        # `score` stays the original coarse retrieval score; the refinement
        # similarity (a different score space -- see KisCandidate) is
        # recorded separately rather than overwriting it.
        refined.append(
            replace(candidate, frame_id=result.frame_id, refinement_score=result.score)
        )
    return refined
