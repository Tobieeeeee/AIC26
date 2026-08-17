"""KIS query orchestration: turn a Textual KIS query into ranked, deduplicated
ORIGINAL-VIDEO-frame predictions using the existing retrieval branches.

Reuses the existing semantic/hybrid services exactly as they are (never
rebuilds an index or reloads a model per request) and resolves each coarse
hit's keyframe_id to an original-video frame_id -- the only valid competition
prediction identity. A candidate whose frame_id cannot be verified is dropped
rather than guessed, consistent with the frame registry (Task 02) and
evaluator (Task 03) policy.
"""
from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class _SemanticService(Protocol):
    def search(self, query: str, top_k: int) -> Sequence[Any]: ...


class _MultimodalService(Protocol):
    def search_hybrid(
        self, query: str, *, top_k: int, rrf_k: int
    ) -> Sequence[Mapping[str, Any]]: ...


class KeyframeFrameIndexResolver:
    """Resolves THIS repo's own pipeline keyframe_id -> original-video frame_idx.

    Backed by the Phase 02 ``keyframe_mapping.csv`` (columns include
    ``keyframe_id`` and ``frame_idx``): the FAISS/Elasticsearch search index
    built from that file does not itself persist ``frame_idx``, only
    ``timestamp`` (see the Task 01/04 audit notes), so this small lookup
    supplies the missing original-frame identity without touching the
    existing, already-verified indexing code.
    """

    def __init__(self, mapping: Mapping[str, int]) -> None:
        self._mapping = dict(mapping)

    @classmethod
    def from_csv(cls, path: Path | str) -> "KeyframeFrameIndexResolver":
        mapping: dict[str, int] = {}
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                mapping[row["keyframe_id"]] = int(row["frame_idx"])
        return cls(mapping)

    def resolve(self, keyframe_id: str) -> int | None:
        return self._mapping.get(keyframe_id)


@dataclass(frozen=True)
class KisCandidate:
    rank: int
    video_id: str
    frame_id: int
    score: float
    keyframe_id: str
    timestamp: float | None = None

    def to_answer(self):
        """Convert to the Task 03 evaluator's prediction identity type."""
        from mvr.evaluation.aic_kis import KisAnswer

        return KisAnswer(video_id=self.video_id, frame_id=self.frame_id)


def resolve_original_frame_id(
    *,
    video_id: str,
    keyframe_id: str,
    pipeline_resolver: KeyframeFrameIndexResolver | None,
    aic_registry: Any | None,
) -> int | None:
    """Never guess: returns None when no resolver has a verified mapping."""
    if pipeline_resolver is not None:
        frame_id = pipeline_resolver.resolve(keyframe_id)
        if frame_id is not None:
            return frame_id
    if aic_registry is not None:
        from mvr.ingestion.aic_dataset import MissingFrameMappingError

        try:
            return aic_registry.resolve_frame(video_id, keyframe_id).frame_idx
        except MissingFrameMappingError:
            return None
    return None


def search_kis(
    query: str,
    *,
    top_k: int,
    semantic_service: _SemanticService | None,
    multimodal_service: _MultimodalService | None,
    pipeline_resolver: KeyframeFrameIndexResolver | None = None,
    aic_registry: Any | None = None,
    overfetch: int | None = None,
) -> list[KisCandidate]:
    """Run one KIS query against the existing retrieval baseline.

    Uses hybrid search (existing semantic + OCR branches fused via the
    existing RRF utility) when Elasticsearch is configured, otherwise falls
    back to semantic-only FAISS search. Never builds an index or loads a
    model itself -- both services are the same lazily-initialized singletons
    the plain `/search/*` routes already use.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if semantic_service is None and multimodal_service is None:
        raise RuntimeError("no KIS retrieval backend is configured")

    # Overfetch beyond top_k so that unresolvable/duplicate hits don't starve
    # the final result count below what the corpus could actually provide.
    fetch_count = overfetch if overfetch is not None else min(max(top_k * 3, 30), 100)

    if multimodal_service is not None:
        raw_hits: Sequence[Mapping[str, Any]] = multimodal_service.search_hybrid(
            query.strip(), top_k=fetch_count, rrf_k=60
        )
    else:
        raw_hits = [
            result.to_dict() if hasattr(result, "to_dict") else dict(result)
            for result in semantic_service.search(query.strip(), top_k=fetch_count)
        ]

    candidates: list[KisCandidate] = []
    seen: set[tuple[str, int]] = set()
    for hit in raw_hits:
        video_id = str(hit["video_id"])
        keyframe_id = str(hit["frame_id"])
        frame_id = resolve_original_frame_id(
            video_id=video_id,
            keyframe_id=keyframe_id,
            pipeline_resolver=pipeline_resolver,
            aic_registry=aic_registry,
        )
        if frame_id is None:
            continue
        key = (video_id, frame_id)
        if key in seen:
            continue
        seen.add(key)
        timestamp = hit.get("timestamp")
        candidates.append(
            KisCandidate(
                rank=len(candidates) + 1,
                video_id=video_id,
                frame_id=frame_id,
                score=float(hit["score"]),
                keyframe_id=keyframe_id,
                timestamp=None if timestamp is None else float(timestamp),
            )
        )
        if len(candidates) >= top_k:
            break
    return candidates
