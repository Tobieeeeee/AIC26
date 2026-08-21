"""Dense per-video TRAKE scoring: score every frame of a shortlisted video
against every event, using vectors already stored in the FAISS index.

The reference chain-alignment algorithms (``dp_align``, ``dp_align_soft``,
``two_pointer_clips`` in the course lab) all operate on a dense score matrix
spanning every frame of a candidate video, not on k independent sparse
top-k lists -- a frame highly relevant to one event but absent from another
event's top-k ``search_kis()`` hits still needs a real score, or alignment
has no way to weigh "include vs. skip" correctly for that frame.

This is affordable because ``FaissIndex`` is built as an exact
``faiss.IndexFlatIP`` that stores every L2-normalized embedding vector
verbatim and supports reconstructing it back out -- no re-encoding of
images, no extra model inference. ``faiss_index``/``aic_registry`` are
intentionally untyped (``Any``) rather than imported at module level,
matching this codebase's existing convention of not coupling the retrieval
layer to the indexing/ingestion layers at import time (see
``mvr.retrieval.kis.search_kis``'s own ``aic_registry: Any | None`` and
``mvr.retrieval.semantic.SemanticRetriever``'s untyped constructor
parameters).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kis import KeyframeFrameIndexResolver, resolve_original_frame_id


@dataclass(frozen=True)
class TrakeVideoScoreMatrix:
    """Dense per-frame scores for one video against k events.

    Row/column position ``j`` is only meaningful within this instance --
    ``timestamps[j]`` and ``original_frame_ids[j]`` are the only valid
    real-world identities for that column. ``original_frame_ids[j]`` is the
    resolved KIS ``frame_id`` (via ``resolve_original_frame_id``, reused
    unchanged) and is the ONLY value that may ever appear in a submitted
    ``TrakeChainPrediction`` -- never the raw column position ``j`` itself.
    """

    video_id: str
    timestamps: tuple[float, ...]
    original_frame_ids: tuple[int, ...]
    scores: np.ndarray  # shape (k, n_frame); scores[:, j] <-> timestamps[j] / original_frame_ids[j]


def build_video_score_matrix(
    video_id: str,
    event_text_vectors: np.ndarray,
    faiss_index: Any,
    aic_registry: Any,
    pipeline_resolver: KeyframeFrameIndexResolver | None,
) -> TrakeVideoScoreMatrix | None:
    """Reconstruct every indexed, resolvable frame of ``video_id`` from
    ``faiss_index`` and score it against ``event_text_vectors``.

    Frames whose KIS ``frame_id`` cannot be resolved (per
    ``resolve_original_frame_id()``, "never guess") are excluded entirely,
    consistent with how ``search_kis()`` itself drops unresolvable hits.
    Returns ``None`` if the video has zero resolvable frames.
    """
    resolved_rows: list[tuple[float, int, np.ndarray]] = []
    for row_index, meta in enumerate(faiss_index.metadata):
        if str(meta["video_id"]) != video_id:
            continue
        keyframe_id = str(meta["frame_id"])
        frame_id = resolve_original_frame_id(
            video_id=video_id,
            keyframe_id=keyframe_id,
            pipeline_resolver=pipeline_resolver,
            aic_registry=aic_registry,
        )
        if frame_id is None:
            continue
        vector = np.asarray(faiss_index.index.reconstruct(row_index), dtype=np.float32)
        resolved_rows.append((float(meta["timestamp"]), frame_id, vector))

    if not resolved_rows:
        return None

    resolved_rows.sort(key=lambda row: row[0])
    timestamps = tuple(row[0] for row in resolved_rows)
    original_frame_ids = tuple(row[1] for row in resolved_rows)
    vectors = np.stack([row[2] for row in resolved_rows], axis=0)  # (n_frame, embedding_dim)

    query_vectors = np.asarray(event_text_vectors, dtype=np.float32)
    scores = query_vectors @ vectors.T  # (k, n_frame)

    return TrakeVideoScoreMatrix(
        video_id=video_id,
        timestamps=timestamps,
        original_frame_ids=original_frame_ids,
        scores=scores,
    )
