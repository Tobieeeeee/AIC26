"""Q&A query orchestration: turn an event description + a question into
ranked, deduplicated, ANSWERED original-video-frame predictions.

Deliberately duplicates `mvr.retrieval.kis.search_kis()`'s retrieval/
resolution shape (overfetch -> resolve original frame -> dedupe -> rank)
rather than calling it, since a Q&A candidate carries an extra `answer`
field `KisCandidate` has no room for -- but it reuses
`resolve_original_frame_id()` and `KeyframeFrameIndexResolver` from that
module verbatim (Task 01 §1: never duplicate frame-resolution logic).

Answer reasoning (Task 03: `mvr.enrichment.qna_answer.AnswerReasoner`) only
runs on the final top-`k` resolved candidates, not the whole overfetch pool
-- it is comparatively expensive (an LLM call per candidate), unlike the
coarse retrieval score.

Object-detection context (Task 02: `mvr.enrichment.qna_context`) is wired
through an injected `objects_path_resolver` that returns a path to the
organizer's per-frame objects JSON, or `None` when unavailable -- never
guessed. As recorded in the Task 01/03 notes, this repository now has
*multiple, non-interchangeable* object-detection label spaces (organizer
OpenImages-labeled files, and this repo's own COCO-labeled live detectors
-- YOLO and the newer torchvision Faster R-CNN from PR #4). This module
deliberately wires only the organizer's file-based objects through
`build_frame_context` for now; falling back to a live COCO detector when
the organizer file is absent is an explicit, separately-scoped follow-up
(see `docs/AIC_QNA_IMPLEMENTATION_NOTES.md`), not silently done here.

`min_object_confidence` defaults to `0.0` (no filtering). Choosing a
detection confidence threshold belongs to the indexing/enrichment layer
that produces the detections, NOT to Q&A -- this module owns no threshold
value so the policy has a single owner upstream. Callers feeding raw,
unfiltered organizer JSON must pass a threshold explicitly.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mvr.enrichment.qna_answer import AnswerReasoner, AnswerSource
from mvr.enrichment.qna_context import OcrEngine, build_frame_context
from mvr.retrieval.kis import KeyframeFrameIndexResolver, resolve_original_frame_id


class _SemanticService(Protocol):
    def search(self, query: str, top_k: int) -> Sequence[Any]: ...


class _MultimodalService(Protocol):
    def search_hybrid(
        self, query: str, *, top_k: int, rrf_k: int
    ) -> Sequence[Mapping[str, Any]]: ...


ObjectsPathResolver = Callable[[str, str], "Path | str | None"]
ImageLoader = Callable[[str], Any]


@dataclass(frozen=True)
class QnaCandidate:
    """One ranked, answered Q&A prediction.

    ``score`` is always the original coarse retrieval score (semantic
    cosine or hybrid RRF), never overwritten -- same invariant as
    ``KisCandidate.score``. ``answer_source`` is the reasoner's own
    self-report (context-grounded / knowledge-grounded / mixed), not
    independently verified.
    """

    rank: int
    video_id: str
    frame_id: int
    score: float
    keyframe_id: str
    answer: str
    answer_source: AnswerSource
    timestamp: float | None = None

    def to_answer(self):
        """Convert to the Task 04 evaluator's prediction identity type."""
        from mvr.evaluation.aic_qna import QnaAnswer

        return QnaAnswer(video_id=self.video_id, frame_id=self.frame_id, answer=self.answer)


def search_qna(
    event_description: str,
    question: str,
    *,
    top_k: int,
    semantic_service: _SemanticService | None,
    multimodal_service: _MultimodalService | None,
    answer_reasoner: AnswerReasoner | None,
    pipeline_resolver: KeyframeFrameIndexResolver | None = None,
    aic_registry: Any | None = None,
    ocr_engine: OcrEngine | None = None,
    image_loader: ImageLoader | None = None,
    objects_path_resolver: ObjectsPathResolver | None = None,
    min_object_confidence: float = 0.0,
    overfetch: int | None = None,
) -> list[QnaCandidate]:
    """Run one Q&A query against the existing retrieval baseline plus the
    Task 03 answer reasoner.

    Uses hybrid search (existing semantic + OCR branches fused via RRF)
    when Elasticsearch is configured, otherwise falls back to semantic-only
    FAISS search over ``event_description`` -- exactly the same branch
    selection as ``search_kis``. Never builds an index, reloads a retrieval
    model, or loads the answer-reasoning model itself -- all three are the
    same lazily-initialized singletons other routes already use.
    """
    if not isinstance(event_description, str) or not event_description.strip():
        raise ValueError("event_description must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if semantic_service is None and multimodal_service is None:
        raise RuntimeError("no QnA retrieval backend is configured")
    if answer_reasoner is None:
        raise RuntimeError("no QnA answer reasoner is configured")
    if (ocr_engine is None) != (image_loader is None):
        raise ValueError("ocr_engine and image_loader must both be provided or both omitted")

    fetch_count = overfetch if overfetch is not None else max(top_k * 3, 30)

    if multimodal_service is not None:
        raw_hits: Sequence[Mapping[str, Any]] = multimodal_service.search_hybrid(
            event_description.strip(), top_k=fetch_count, rrf_k=60
        )
    else:
        raw_hits = [
            result.to_dict() if hasattr(result, "to_dict") else dict(result)
            for result in semantic_service.search(event_description.strip(), top_k=fetch_count)
        ]

    question = question.strip()
    candidates: list[QnaCandidate] = []
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

        image = image_loader(str(hit["image_path"])) if image_loader is not None else None
        objects_path = (
            objects_path_resolver(video_id, keyframe_id)
            if objects_path_resolver is not None
            else None
        )
        context = build_frame_context(
            video_id=video_id,
            frame_id=frame_id,
            image=image,
            objects_path=objects_path,
            ocr_engine=ocr_engine,
            min_object_confidence=min_object_confidence,
        )
        result = answer_reasoner.answer(question, context)

        timestamp = hit.get("timestamp")
        candidates.append(
            QnaCandidate(
                rank=len(candidates) + 1,
                video_id=video_id,
                frame_id=frame_id,
                score=float(hit["score"]),
                keyframe_id=keyframe_id,
                timestamp=None if timestamp is None else float(timestamp),
                answer=result.text,
                answer_source=result.source,
            )
        )
        if len(candidates) >= top_k:
            break
    return candidates
