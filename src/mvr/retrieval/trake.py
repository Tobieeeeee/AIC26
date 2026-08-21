"""TRAKE orchestration: wire event encoding, per-event KIS shortlisting,
dense per-video scoring, and chain alignment into one pure-Python function.

``search_kis()`` (unchanged) is called once per event, ONLY to shortlist
candidate videos -- this is what "TRAKE = KIS x k" means concretely. Dense
scoring (Phase 04, Part 1) reuses the encoded event vectors and the FAISS
index directly; it never calls ``search_kis()`` again.

``TrakeChainPrediction`` is imported lazily, matching ``trake_align.py``'s
"Import direction" note (mirrors ``KisCandidate.to_answer()``'s pattern) --
this module never depends on the evaluation layer at import time.

Per-event and per-video failures are logged (``logging``, matching
``mvr.api.app``'s convention) and skipped rather than aborting the whole
query. This is a deliberate departure from
``frame_refinement.refine_top_candidates()``'s silent-fallback pattern
(that function has no visibility mechanism at all): a orchestration-level
failure spanning shortlisting or scoring is a more significant event than
one candidate's refinement failing, and needs to be inspectable (e.g. via
``pytest``'s ``caplog``) rather than silently invisible.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..enrichment.qna_translation import QueryTranslator, translate_for_retrieval
from .kis import KeyframeFrameIndexResolver, search_kis
from .trake_align import TrakeAlignmentStrategy
from .trake_events import TrakeQuery
from .trake_scoring import build_video_score_matrix

if TYPE_CHECKING:
    from mvr.evaluation.trake import TrakeChainPrediction

logger = logging.getLogger(__name__)


def search_trake(
    query: TrakeQuery,
    *,
    alignment_strategy: TrakeAlignmentStrategy,
    semantic_service: Any,
    aic_registry: Any,
    top_k: int = 100,
    shortlist_top_k: int = 500,
    pipeline_resolver: KeyframeFrameIndexResolver | None = None,
    multimodal_service: Any | None = None,
    query_translator: QueryTranslator | None = None,
) -> "list[TrakeChainPrediction]":
    """Run one TRAKE query end to end: encode -> shortlist -> dense score ->
    align -> merge, entirely in pure Python (no HTTP surface).

    Never raises on a per-event shortlisting failure or a per-video scoring
    failure -- both are logged and skipped, consistent with the "never let
    one bad item abort the batch" principle already established for KIS
    (``refine_top_candidates()``). Returns an empty list (not an exception)
    if every event fails to shortlist.

    ``query_translator`` translates each event's text for retrieval only
    (both shortlisting and dense scoring use the SAME translated text, so
    the two stay comparable), mirroring ``search_qna()``'s
    ``translate_for_retrieval`` usage -- see ``mvr.enrichment.qna_translation``
    for why: the index's encoder is English-trained (measured: a Vietnamese
    query ranked the correct frame 192/268 vs. #1/268 in English on the same
    data, see docs/AIC_TRAKE_BENCHMARK_RESULTS.md). ``None`` (the default)
    keeps every event's text unchanged, identical to all prior behavior.

    The shortlisting call also passes the event's ORIGINAL (untranslated)
    text as ``search_kis()``'s ``lexical_query`` -- same split ``search_kis``
    itself documents: dense/semantic wants English, lexical/BM25 indexes
    OCR/ASR text that is predominantly Vietnamese, so it wants the original.
    Only takes effect if ``multimodal_service`` is configured; otherwise
    ``search_kis()`` ignores it, matching TRAKE's documented semantic-only
    default.
    """
    retrieval_texts = [
        translate_for_retrieval(event.text, query_translator) for event in query.events
    ]
    event_text_vectors = semantic_service.text_encoder.encode_text(retrieval_texts)

    video_ids: set[str] = set()
    any_event_succeeded = False
    for event, retrieval_text in zip(query.events, retrieval_texts):
        try:
            candidates = search_kis(
                retrieval_text,
                top_k=shortlist_top_k,
                semantic_service=semantic_service,
                multimodal_service=multimodal_service,
                pipeline_resolver=pipeline_resolver,
                aic_registry=aic_registry,
                # Same split as search_kis()'s own docstring: the dense
                # branch wants English (retrieval_text, possibly
                # translated); the lexical/BM25 branch indexes OCR/ASR text,
                # which is predominantly Vietnamese, so it wants the
                # ORIGINAL event text -- never the translated one. Only
                # takes effect when multimodal_service is configured;
                # search_kis() ignores it otherwise.
                lexical_query=event.text,
            )
        except Exception:
            logger.exception(
                "search_trake: search_kis failed while shortlisting for event "
                "index=%d text=%r; skipping this event",
                event.index,
                retrieval_text,
            )
            continue
        any_event_succeeded = True
        video_ids.update(candidate.video_id for candidate in candidates)

    if not any_event_succeeded:
        logger.error(
            "search_trake: all %d events failed to shortlist; returning no chains",
            len(query.events),
        )
        return []

    faiss_index = semantic_service.index
    chains: list[TrakeChainPrediction] = []
    for video_id in sorted(video_ids):
        try:
            matrix = build_video_score_matrix(
                video_id,
                event_text_vectors,
                faiss_index,
                aic_registry,
                pipeline_resolver,
            )
        except Exception:
            logger.exception(
                "search_trake: build_video_score_matrix failed for video_id=%r; skipping",
                video_id,
            )
            continue
        if matrix is None:
            continue
        chains.extend(alignment_strategy.align(matrix, query.gaps))

    chains.sort(key=lambda chain: (-chain.score, chain.video_id))
    return chains[:top_k]
