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

Temporal expansion (`temporal_radius`) is opt-in and off by default. See
`_expansion_plan` for why it exists and how the budget is shared between
"more videos" and "more frames of the same video".
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mvr.enrichment.qna_answer import AnswerReasoner, AnswerSource
from mvr.enrichment.qna_context import OcrEngine, build_frame_context
from mvr.enrichment.qna_translation import QueryTranslator, translate_for_retrieval
from mvr.retrieval.kis import KeyframeFrameIndexResolver, resolve_original_frame_id


class _SemanticService(Protocol):
    def search(self, query: str, top_k: int) -> Sequence[Any]: ...


class _MultimodalService(Protocol):
    def search_hybrid(
        self, query: str, *, top_k: int, rrf_k: int
    ) -> Sequence[Mapping[str, Any]]: ...


ObjectsPathResolver = Callable[[str, str], "Path | str | None"]
KeyframePathResolver = Callable[[str, str], "Path | str | None"]
ImageLoader = Callable[[str], Any]

_TRAILING_NUMBER = re.compile(r"^(.*?)(\d+)$")


def shift_keyframe_id(keyframe_id: str, offset: int) -> str | None:
    """Return the id of the keyframe `offset` positions away, or `None`.

    Derived from the id *string* rather than rebuilt from a bare number so
    that whatever zero-padding and prefix the indexing layer chose survive
    intact: resolvers key on the exact string, so `"0028"` finds a mapping
    where `"28"` does not.

    Returns `None` when the id has no trailing number to shift, or when the
    shift would run off the front of the video. Existence is NOT checked
    here -- that stays with the frame resolver, which is the only component
    entitled to say whether a keyframe is real.
    """
    match = _TRAILING_NUMBER.match(str(keyframe_id))
    if match is None:
        return None
    prefix, digits = match.group(1), match.group(2)
    shifted = int(digits) + offset
    if shifted < 0:
        return None
    return f"{prefix}{shifted:0{len(digits)}d}"


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
    #: Where the retrieval hit's keyframe image lives, carried through so a
    #: human or VLM reviewing this candidate can actually see the frame.
    #: ``None`` when the retrieval backend did not supply one.
    image_path: str | None = None
    #: How many keyframes away from the retrieval hit this prediction sits.
    #: ``0`` means retrieval returned this frame itself.
    temporal_offset: int = 0
    #: The keyframe_id retrieval actually returned, when this candidate was
    #: produced by temporal expansion rather than by retrieval. Non-``None``
    #: means ``score`` is INHERITED from that hit, not earned -- the two
    #: fields exist so that borrowing is explicit instead of silent.
    expanded_from: str | None = None


    def to_answer(self):
        """Convert to the Task 04 evaluator's prediction identity type."""
        from mvr.evaluation.aic_qna import QnaAnswer

        return QnaAnswer(video_id=self.video_id, frame_id=self.frame_id, answer=self.answer)


def _expansion_plan(
    hit_count: int, radius: int, neighbour_cost: float
) -> list[tuple[int, int]]:
    """Order every ``(hit index, keyframe offset)`` pair into one submission
    sequence, best first.

    The competition scores ``mean(R@1, R@5, R@20, R@50, R@100)``, so the
    100-answer budget has to be split between two different bets: *more
    videos* (retrieval picked the wrong one) and *more frames of the same
    video* (retrieval picked the right one but the wrong moment). Measured
    on real ground truth, the second bet is the one currently losing --
    p1-15 retrieved the right video at ranks #1 and #2 and still scored
    zero, four keyframes short.

    Each pair costs ``hit_index + neighbour_cost * abs(offset)`` and the
    plan is that cost, ascending. One knob controls the whole trade-off:
    a large ``neighbour_cost`` submits every retrieval hit before any
    neighbour, a small one exhausts the top hit's neighbourhood first, and
    ``1.0`` treats "one rank further down" and "one keyframe to the side"
    as equally speculative.

    Ties break towards the smaller ``abs(offset)``, so an actual retrieval
    hit always beats a hedge that landed on the same cost.
    """
    plan = [
        (rank_index, offset)
        for rank_index in range(hit_count)
        for offset in range(-radius, radius + 1)
    ]
    plan.sort(
        key=lambda pair: (
            pair[0] + neighbour_cost * abs(pair[1]),
            abs(pair[1]),
            pair[0],
            pair[1] < 0,  # +1 before -1, so expansion leads with what follows
            pair[1],
        )
    )
    return plan


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
    keyframe_path_resolver: KeyframePathResolver | None = None,
    min_object_confidence: float = 0.0,
    query_translator: QueryTranslator | None = None,
    temporal_radius: int = 0,
    neighbour_cost: float = 1.0,
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

    ``temporal_radius`` additionally predicts each hit's neighbouring
    keyframes (see ``_expansion_plan``); ``0``, the default, is retrieval's
    ranking untouched. Neighbours are resolved through the same frame
    resolver as the hits themselves, so one that does not exist is dropped,
    never guessed.

    ``keyframe_path_resolver`` supplies an image path for frames the
    retrieval backend did not return one for -- which is every expanded
    neighbour. Without it neighbours are answered with no OCR or object
    context at all, which for Q&A is most of the signal.
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
    if temporal_radius < 0:
        raise ValueError("temporal_radius must be >= 0")
    if neighbour_cost <= 0:
        raise ValueError("neighbour_cost must be > 0")

    fetch_count = overfetch if overfetch is not None else max(top_k * 3, 30)

    # Only the retrieval query is translated. The question keeps its original
    # language for the answer reasoner -- see mvr.enrichment.qna_translation.
    retrieval_query = translate_for_retrieval(event_description.strip(), query_translator)

    if multimodal_service is not None:
        raw_hits: Sequence[Mapping[str, Any]] = multimodal_service.search_hybrid(
            retrieval_query, top_k=fetch_count, rrf_k=60
        )
    else:
        raw_hits = [
            result.to_dict() if hasattr(result, "to_dict") else dict(result)
            for result in semantic_service.search(retrieval_query, top_k=fetch_count)
        ]
    raw_hits = list(raw_hits)

    question = question.strip()
    candidates: list[QnaCandidate] = []
    seen: set[tuple[str, int]] = set()
    for rank_index, offset in _expansion_plan(len(raw_hits), temporal_radius, neighbour_cost):
        hit = raw_hits[rank_index]
        video_id = str(hit["video_id"])
        hit_keyframe_id = str(hit["frame_id"])
        keyframe_id = (
            hit_keyframe_id if offset == 0 else shift_keyframe_id(hit_keyframe_id, offset)
        )
        if keyframe_id is None:
            continue
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

        image_path: str | None = None
        if offset == 0:
            raw_image_path = hit.get("image_path")
            image_path = None if raw_image_path is None else str(raw_image_path)
        if image_path is None and keyframe_path_resolver is not None:
            resolved_path = keyframe_path_resolver(video_id, keyframe_id)
            image_path = None if resolved_path is None else str(resolved_path)

        image = image_loader(image_path) if image_loader is not None and image_path else None
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

        # A neighbour has no timestamp of its own: the hit's belongs to a
        # different frame, and interpolating one would invent data.
        timestamp = hit.get("timestamp") if offset == 0 else None
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
                image_path=image_path,
                temporal_offset=offset,
                expanded_from=None if offset == 0 else hit_keyframe_id,
            )
        )
        if len(candidates) >= top_k:
            break
    return candidates
