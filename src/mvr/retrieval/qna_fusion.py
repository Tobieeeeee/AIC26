"""Fuse semantic (FAISS/CLIP) retrieval with Task05B lexical (BM25) for Q&A.

Built for a measured failure, not a hunch. Of the three human-verified
queries, two score zero and both are *retrieval* failures -- neither has a
single candidate from the correct video, so no amount of answering or
budget reallocation reaches them. Temporal expansion cannot help either: it
only expands around videos retrieval already found.

What distinguishes both is text rather than scene appearance::

    p1-19  "Hai câu thơ đó là gì?"      the poetry is RECITED   -> asr_text
    p1-22  "tiêu đề công thức nấu ăn"   the title is ON SCREEN  -> ocr_text

which are precisely the two fields ``lexical_index.SEARCHABLE_FIELDS``
indexes. Locating those two videos by hand earlier meant scanning
``media_info`` titles -- doing BM25's job manually.

Separate from ``mvr.api.app.MultimodalService`` on purpose. That one fuses
the older Elasticsearch schema and keys RRF on ``frame_id`` alone, which is
safe only while keyframe ids are globally unique (they are for this repo's
own ``<video>_S0007_K01`` ids). It is NOT safe for organizer-style ids: the
real index holds 166,550 keyframes with just **632 distinct ids**, so "001"
appears in roughly 850 videos and fusing on the bare id would merge
unrelated frames into one entry. This module keys on
``(video_id, keyframe_id)`` throughout.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from mvr.retrieval.hybrid import rrf_fuse

logger = logging.getLogger(__name__)

#: How many candidates to pull from each branch before fusing. Larger than
#: the requested ``top_k`` so a frame ranked well by only one branch can
#: still surface -- which is the entire point of fusing.
DEFAULT_PER_BRANCH_TOP_K = 200

#: Joins (video_id, keyframe_id) into the single string ``rrf_fuse`` ranks
#: by. NUL cannot occur in either id, so the pair can never be split at the
#: wrong place. Written as ``chr(0)`` rather than an escape so it survives
#: every editor and diff tool unambiguously.
_KEY_SEPARATOR = chr(0)


class _SemanticService(Protocol):
    def search(self, query: str, top_k: int) -> Sequence[Any]: ...


#: Signature of ``search_task05b_lexical`` with its client and index bound.
LexicalSearch = Callable[..., Sequence[Any]]


def _semantic_payload(result: Any) -> dict[str, Any]:
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def _key(video_id: Any, keyframe_id: Any) -> str:
    return f"{video_id}{_KEY_SEPARATOR}{keyframe_id}"


class FusedRetriever:
    """``search_hybrid`` over semantic + lexical, in the shape ``search_qna`` reads.

    Emits ``frame_id`` for the KEYFRAME id, matching what ``search_qna``
    already consumes, and passes ``frame_idx`` through when the lexical
    branch supplied one. The two names differ by three characters and mean
    entirely different things -- keyframe versus original-video frame --
    and swapping them makes every candidate fail to resolve, silently.
    """

    def __init__(
        self,
        *,
        semantic_service: _SemanticService,
        lexical_search: LexicalSearch | None = None,
        per_branch_top_k: int = DEFAULT_PER_BRANCH_TOP_K,
    ) -> None:
        if per_branch_top_k <= 0:
            raise ValueError("per_branch_top_k must be positive")
        self._semantic = semantic_service
        self._lexical = lexical_search
        self._per_branch_top_k = per_branch_top_k

    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int,
        rrf_k: int,
        lexical_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """``query`` đi vào nhánh dense, ``lexical_query`` vào nhánh BM25.

        Hai nhánh muốn hai ngôn ngữ khác nhau. Encoder dense là OpenCLIP
        ``RN50/openai``, tiếng Anh thuần; index lexical chứa OCR/ASR tiếng
        Việt. Ép cùng một chuỗi vào cả hai thì luôn có một nhánh chịu thiệt.

        ``lexical_query=None`` giữ nguyên hành vi cũ -- một chuỗi cho cả hai.
        """
        lexical_text = query if lexical_query is None else lexical_query

        semantic_hits = [
            _semantic_payload(result)
            for result in self._semantic.search(query, self._per_branch_top_k)
        ]

        lexical_hits: list[Any] = []
        if self._lexical is not None:
            try:
                lexical_hits = list(
                    self._lexical(lexical_text, top_k=self._per_branch_top_k)
                )
            except Exception:
                # Elasticsearch being unreachable should cost recall, not the
                # whole query: semantic alone still returns something useful.
                # The semantic branch is deliberately NOT wrapped -- it is the
                # primary path, and swallowing its failure would hide an
                # outage behind quietly worse results.
                logger.warning(
                    "lexical branch unavailable; using semantic only", exc_info=True
                )

        payloads: dict[tuple[str, str], dict[str, Any]] = {}
        for hit in semantic_hits:
            key = (str(hit["video_id"]), str(hit["frame_id"]))
            payloads.setdefault(key, {**hit, "video_id": key[0], "frame_id": key[1]})
        for hit in lexical_hits:
            key = (str(hit.video_id), str(hit.keyframe_id))
            payload = payloads.setdefault(
                key,
                {
                    "video_id": key[0],
                    "frame_id": key[1],
                    "timestamp": hit.timestamp,
                    "score": hit.score,
                },
            )
            # The lexical document states the competition frame index, so a
            # candidate found this way needs no keyframe->frame resolution.
            payload["frame_idx"] = hit.frame_idx

        fused = rrf_fuse(
            (
                [_key(hit["video_id"], hit["frame_id"]) for hit in semantic_hits],
                [_key(hit.video_id, hit.keyframe_id) for hit in lexical_hits],
            ),
            rrf_k=rrf_k,
            top_k=top_k,
        )

        results: list[dict[str, Any]] = []
        for rank, (fused_key, score) in enumerate(fused, start=1):
            video_id, keyframe_id = fused_key.split(_KEY_SEPARATOR, 1)
            payload = dict(payloads[(video_id, keyframe_id)])
            payload.update(rank=rank, score=float(score))
            results.append(payload)
        return results


#: Tên cũ, giữ lại vì `scripts/run_qna_pipeline.py` và các test Q&A import nó.
#: Lớp này không còn chỉ phục vụ Q&A: nhánh KIS dùng chung nó để tách truy vấn
#: tiếng Việt khỏi truy vấn tiếng Anh.
QnaFusedRetriever = FusedRetriever
