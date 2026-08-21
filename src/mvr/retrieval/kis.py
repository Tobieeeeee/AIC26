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
    """Resolves ``(video_id, keyframe_id)`` -> original-video ``frame_idx``.

    Backed by the Phase 02 ``keyframe_mapping.csv`` (columns include
    ``video_id``, ``keyframe_id`` and ``frame_idx``): the legacy
    FAISS/Elasticsearch index built from that file does not itself persist
    ``frame_idx``, only ``timestamp`` (see the Task 01/04 audit notes), so this
    small lookup verifies original-frame identity for either semantic backend.

    Keyed on the PAIR, never on ``keyframe_id`` alone. Measured on the real
    mapping: 166,550 rows, every ``(video_id, keyframe_id)`` pair distinct,
    but only **632 distinct keyframe_ids**. Keying on the bare id let each
    video overwrite the one before it, leaving 632 entries -- 99.6% of the
    file gone -- and **98.8% of pairs then resolved to a frame belonging to
    some other video**.

    That is worse than losing them. ``resolve`` returning ``None`` makes
    ``search_kis`` drop the candidate, which costs recall honestly. Here it
    returned a plausible integer instead, and ``validate_kis_predictions``
    only checks that a frame_id is a non-negative int -- so the wrong frame
    was submitted as a well-formed answer, with no error anywhere.

    The bare id was correct while this repo generated its own keyframe ids
    (``L21_V001_S0007_K01`` spells the video into the id) and became wrong
    when the organizer's keyframes -- whose id is just the JPEG stem,
    ``"034"`` -- were adopted as canonical identity.
    """

    def __init__(self, mapping: Mapping[tuple[str, str], int]) -> None:
        for key in mapping:
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError(
                    "KeyframeFrameIndexResolver keys must be "
                    f"(video_id, keyframe_id) pairs, got {key!r}. A bare "
                    "keyframe_id does not identify a keyframe: the organizer's "
                    "166,550 keyframes carry only 632 distinct ids."
                )
        self._mapping = {
            (str(video_id), str(keyframe_id)): frame_idx
            for (video_id, keyframe_id), frame_idx in mapping.items()
        }

    @classmethod
    def from_csv(cls, path: Path | str) -> "KeyframeFrameIndexResolver":
        mapping: dict[tuple[str, str], int] = {}
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                identity = (row["video_id"], row["keyframe_id"])
                mapping[identity] = int(row["frame_idx"])
        return cls(mapping)

    def resolve(self, video_id: str, keyframe_id: str) -> int | None:
        return self._mapping.get((str(video_id), str(keyframe_id)))


@dataclass(frozen=True)
class KisCandidate:
    """``score`` is always the original coarse retrieval score (semantic
    cosine or hybrid RRF, depending on which branch produced this hit) and
    must never be overwritten by a different score space. Exact-frame
    refinement (Task 05) records its own local-window similarity separately
    in ``refinement_score`` -- the two are not comparable and must not be
    silently mixed.
    """

    rank: int
    video_id: str
    frame_id: int
    score: float
    keyframe_id: str
    timestamp: float | None = None
    refinement_score: float | None = None

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
        frame_id = pipeline_resolver.resolve(video_id, keyframe_id)
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
    lexical_query: str | None = None,
) -> list[KisCandidate]:
    """Run one KIS query against the existing retrieval baseline.

    Uses hybrid search (existing semantic + OCR branches fused via the
    existing RRF utility) when Elasticsearch is configured, otherwise falls
    back to the configured semantic service (Milvus in production and FAISS
    only when explicitly selected as the reference backend). Never builds an
    index or loads a model itself -- both services are the same
    lazily-initialized singletons the plain `/search/*` routes already use.

    ``query`` feeds the configured provider-matched dense encoders.
    ``lexical_query`` feeds BM25 and should be the original Vietnamese,
    because the OCR and ASR indexed under it are Vietnamese. Leave it None to
    send one string to both, as before.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not 1 <= top_k <= 500:
        raise ValueError("top_k must be between 1 and 500")
    if semantic_service is None and multimodal_service is None:
        raise RuntimeError("no KIS retrieval backend is configured")

    # Overfetch beyond top_k so that unresolvable/duplicate hits don't starve
    # the final result count below what the corpus could actually provide.
    # Deliberately NOT capped at top_k itself (that would defeat overfetch
    # entirely once top_k reaches its own max); the internal pool may
    # legitimately exceed the requested output size, up to 1500 at top_k=500.
    fetch_count = overfetch if overfetch is not None else max(top_k * 3, 30)

    if multimodal_service is not None:
        # Only forwarded when given: `ElasticsearchMultimodalService.search_hybrid`
        # does not accept this parameter, and passing it unconditionally would
        # TypeError on the /search/hybrid route that still uses that service.
        extra = {} if lexical_query is None else {"lexical_query": lexical_query}
        raw_hits: Sequence[Mapping[str, Any]] = multimodal_service.search_hybrid(
            query.strip(), top_k=fetch_count, rrf_k=60, **extra
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
        keyframe_id = str(hit.get("keyframe_id", hit["frame_id"]))
        frame_id = resolve_original_frame_id(
            video_id=video_id,
            keyframe_id=keyframe_id,
            pipeline_resolver=pipeline_resolver,
            aic_registry=aic_registry,
        )
        # A lexical hit already states its own competition frame index, so
        # requiring a keyframe->frame mapping on top of it asks twice. Same
        # rule `search_qna` has used since it gained a lexical branch (see
        # `qna.py`): prefer what the hit states, and treat the two sources
        # disagreeing as a real inconsistency rather than picking a side.
        hit_frame_idx = hit.get("frame_idx")
        if hit_frame_idx is not None:
            hit_frame_idx = int(hit_frame_idx)
            if frame_id is not None and frame_id != hit_frame_idx:
                raise ValueError(
                    f"frame_idx disagreement for ({video_id!r}, {keyframe_id!r}): "
                    f"hit says {hit_frame_idx}, resolver says {frame_id}"
                )
            frame_id = hit_frame_idx
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


def parse_keyframe_ref(ref: object) -> tuple[str, str]:
    """``"L21_V001_001"`` -> ``("L21_V001", "001")``.

    Splits on the LAST separator: `video_id` contains an underscore of its
    own, so splitting on the first yields ``("L21", "V001_001")`` and
    resolves to nothing. The slash form printed by the interactive
    listing (``L21_V001/001``) is accepted too, because copying what is
    on screen is the obvious thing to do.
    """
    text = str(ref).strip().replace("/", "_")
    video_id, separator, keyframe_id = text.rpartition("_")
    if not separator or not video_id or not keyframe_id:
        raise ValueError(
            f"không đọc được keyframe từ {ref!r}; cần dạng "
            "<video_id>_<keyframe_id>, ví dụ L21_V001_001. Một id trần "
            "không xác định được keyframe: 166.550 keyframe của BTC chỉ "
            "có 632 id phân biệt"
        )
    return video_id, keyframe_id


def resolve_keyframe_ref(
    resolver: KeyframeFrameIndexResolver, ref: object
) -> tuple[str, str, int | None]:
    """``(video_id, keyframe_id, frame_idx)``; ``frame_idx`` is None on a miss.

    Retries a numeric id zero-padded to the organizer's three-digit JPEG
    stem, so ``L21_V001_50`` still finds ``050``. A miss is reported, never
    guessed -- handing back a plausible integer for an unmapped pair is the
    exact failure `KeyframeFrameIndexResolver` exists to prevent.
    """
    video_id, keyframe_id = parse_keyframe_ref(ref)
    frame_idx = resolver.resolve(video_id, keyframe_id)
    if frame_idx is None and keyframe_id.isdigit():
        padded = keyframe_id.zfill(3)
        if padded != keyframe_id:
            retry = resolver.resolve(video_id, padded)
            if retry is not None:
                return video_id, padded, retry
    return video_id, keyframe_id, frame_idx
