"""Fusing semantic (FAISS) with Task05B lexical (BM25 over OCR + ASR).

This targets the two of three verified queries that score zero, and it does
so for a measured reason rather than a hunch. Both are retrieval failures:
neither has a single candidate from the correct video, because CLIP ranks
that video #153 and outside the top hits respectively. Temporal expansion
cannot help -- it only expands around videos retrieval already found.

What distinguishes both queries is *text*, not scene appearance:

    p1-19  "Hai câu thơ đó là gì?"        the poetry is RECITED  -> asr_text
    p1-22  "tiêu đề công thức nấu ăn"     the title is ON SCREEN -> ocr_text

Those are exactly the two fields `lexical_index.SEARCHABLE_FIELDS` holds.
Finding those two videos by hand earlier meant scanning media_info titles --
doing BM25's job manually.

Fusion is keyed on **(video_id, keyframe_id)**, never keyframe_id alone.
The real index holds 166,550 keyframes with only 632 distinct ids: "001"
exists in roughly 850 videos. Fusing on the bare id would silently merge
unrelated frames from unrelated videos into one ranked entry.
"""
from __future__ import annotations

import pytest

from mvr.retrieval.lexical_bm25 import LexicalSearchHit
from mvr.retrieval.qna_fusion import QnaFusedRetriever


def _resolver(mapping, videos=("V", "A", "B")):
    """Fixtures pre-dating the pair key: one keyframe id per video.

    `KeyframeFrameIndexResolver` now keys on `(video_id, keyframe_id)`,
    because a bare keyframe id does not identify a keyframe -- the
    organizer's 166,550 keyframes carry only 632 distinct ids. These
    fixtures use synthetic ids that were meant to resolve whatever video
    asked, so registering each id under every video the test uses keeps
    exactly the behaviour they were written to check.
    """
    from mvr.retrieval.kis import KeyframeFrameIndexResolver

    return KeyframeFrameIndexResolver(
        {(video, kf): idx for video in videos for kf, idx in mapping.items()}
    )




class _FakeSemanticResult:
    def __init__(self, video_id: str, frame_id: str, score: float, rank: int) -> None:
        self.video_id = video_id
        self.frame_id = frame_id
        self.score = score
        self.rank = rank
        self.timestamp = 1.0 * rank
        self.image_path = f"/frames/{video_id}/{frame_id}.jpg"

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "score": self.score,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
        }


class _FakeSemantic:
    def __init__(self, results) -> None:
        self._results = results
        self.queries: list[str] = []

    def search(self, query: str, top_k: int):
        self.queries.append(query)
        return self._results[:top_k]


class _FakeLexical:
    """Stands in for `search_task05b_lexical`, same return type."""

    def __init__(self, hits) -> None:
        self._hits = hits
        self.queries: list[str] = []

    def __call__(self, query: str, *, top_k: int):
        self.queries.append(query)
        return self._hits[:top_k]


def _lex(video_id: str, keyframe_id: str, frame_idx: int, rank: int) -> LexicalSearchHit:
    return LexicalSearchHit(
        rank=rank,
        score=10.0 - rank,
        keyframe_id=keyframe_id,
        video_id=video_id,
        frame_idx=frame_idx,
        shot_id=None,
        timestamp=float(frame_idx) / 25.0,
    )


def _retriever(semantic_results, lexical_hits) -> QnaFusedRetriever:
    return QnaFusedRetriever(
        semantic_service=_FakeSemantic(semantic_results),
        lexical_search=_FakeLexical(lexical_hits),
    )


# -- the collision this class exists to prevent -------------------------


def test_fusion_keys_on_video_and_keyframe_not_the_bare_id() -> None:
    """"001" exists in ~850 videos. Merging them would be catastrophic."""
    semantic = [_FakeSemanticResult("A", "001", 0.9, 1)]
    lexical = [_lex("B", "001", 500, 1)]

    results = _retriever(semantic, lexical).search_hybrid("q", top_k=10, rrf_k=60)

    assert len(results) == 2
    assert {(r["video_id"], r["frame_id"]) for r in results} == {("A", "001"), ("B", "001")}


def test_a_frame_found_by_both_branches_outranks_one_found_by_either() -> None:
    semantic = [
        _FakeSemanticResult("A", "001", 0.9, 1),
        _FakeSemanticResult("A", "002", 0.8, 2),
    ]
    lexical = [_lex("A", "002", 500, 1)]

    results = _retriever(semantic, lexical).search_hybrid("q", top_k=10, rrf_k=60)

    assert (results[0]["video_id"], results[0]["frame_id"]) == ("A", "002")


# -- the shape search_qna consumes --------------------------------------


def test_hits_use_frame_id_for_the_keyframe_as_search_qna_expects() -> None:
    """`frame_id` in a hit means the KEYFRAME id; `frame_idx` means the
    original-video frame. Two near-identical names, opposite meanings --
    swapping them makes every candidate fail to resolve, silently."""
    results = _retriever([], [_lex("A", "L21_V001_S0007_K01", 578, 1)]).search_hybrid(
        "q", top_k=10, rrf_k=60
    )

    assert results[0]["frame_id"] == "L21_V001_S0007_K01"
    assert results[0]["frame_idx"] == 578


def test_the_lexical_branch_supplies_frame_idx_so_no_resolution_is_needed() -> None:
    results = _retriever([], [_lex("A", "001", 1745, 1)]).search_hybrid(
        "q", top_k=10, rrf_k=60
    )

    assert results[0]["frame_idx"] == 1745


def test_a_semantic_only_hit_carries_no_frame_idx_rather_than_a_guess() -> None:
    results = _retriever([_FakeSemanticResult("A", "001", 0.9, 1)], []).search_hybrid(
        "q", top_k=10, rrf_k=60
    )

    assert results[0].get("frame_idx") is None
    assert results[0]["image_path"] == "/frames/A/001.jpg"


def test_ranks_are_contiguous_and_scores_are_the_fused_ones() -> None:
    semantic = [_FakeSemanticResult("A", f"{n:03d}", 0.9, n) for n in (1, 2, 3)]
    results = _retriever(semantic, []).search_hybrid("q", top_k=10, rrf_k=60)

    assert [r["rank"] for r in results] == [1, 2, 3]
    assert all(0.0 < r["score"] < 1.0 for r in results)


# -- degradation --------------------------------------------------------


def test_either_branch_may_be_empty() -> None:
    only_lexical = _retriever([], [_lex("A", "001", 10, 1)]).search_hybrid(
        "q", top_k=5, rrf_k=60
    )
    only_semantic = _retriever([_FakeSemanticResult("A", "001", 0.9, 1)], []).search_hybrid(
        "q", top_k=5, rrf_k=60
    )

    assert len(only_lexical) == 1
    assert len(only_semantic) == 1


def test_a_failing_lexical_backend_degrades_to_semantic_instead_of_failing() -> None:
    """Elasticsearch being down must cost recall, not the whole query."""

    def _broken(query: str, *, top_k: int):
        raise RuntimeError("elasticsearch unreachable")

    retriever = QnaFusedRetriever(
        semantic_service=_FakeSemantic([_FakeSemanticResult("A", "001", 0.9, 1)]),
        lexical_search=_broken,
    )

    results = retriever.search_hybrid("q", top_k=5, rrf_k=60)

    assert [(r["video_id"], r["frame_id"]) for r in results] == [("A", "001")]


def test_a_failing_semantic_backend_is_not_swallowed() -> None:
    """Semantic is the primary branch; losing it silently would hide a real
    outage behind quietly worse results."""

    class _Broken:
        def search(self, query: str, top_k: int):
            raise RuntimeError("index not loaded")

    retriever = QnaFusedRetriever(
        semantic_service=_Broken(), lexical_search=_FakeLexical([])
    )

    with pytest.raises(RuntimeError):
        retriever.search_hybrid("q", top_k=5, rrf_k=60)


def test_top_k_is_honoured() -> None:
    semantic = [_FakeSemanticResult("A", f"{n:03d}", 0.9, n) for n in range(1, 20)]
    results = _retriever(semantic, []).search_hybrid("q", top_k=4, rrf_k=60)

    assert len(results) == 4


def test_both_branches_see_the_same_query() -> None:
    semantic = _FakeSemantic([])
    lexical = _FakeLexical([])
    QnaFusedRetriever(semantic_service=semantic, lexical_search=lexical).search_hybrid(
        "đình thần", top_k=5, rrf_k=60
    )

    assert semantic.queries == ["đình thần"]
    assert lexical.queries == ["đình thần"]


# -- search_qna consuming a hit that already knows its frame_idx --------


def _run_search(hits, **kwargs):
    import json

    from mvr.enrichment.qna_answer import AnswerReasoner
    from mvr.retrieval.qna import search_qna

    class _Backend:
        def generate(self, prompt: str) -> str:
            return json.dumps({"answer": "x", "source": "context-grounded"})

    class _Service:
        def search_hybrid(self, query, *, top_k, rrf_k):
            return hits[:top_k]

    return search_qna(
        "event", "question", top_k=10,
        semantic_service=None, multimodal_service=_Service(),
        answer_reasoner=AnswerReasoner(_Backend()),
        **kwargs,
    )


def test_a_hit_carrying_frame_idx_needs_no_resolver_at_all() -> None:
    """The lexical document states the competition frame index, so requiring
    a keyframe->frame mapping on top of it would be asking twice."""
    hits = [
        {"video_id": "V", "frame_id": "kf-1", "frame_idx": 1745, "score": 0.9,
         "timestamp": 1.0}
    ]

    results = _run_search(hits)

    assert [(c.video_id, c.frame_id) for c in results] == [("V", 1745)]


def test_a_neighbour_never_inherits_the_hit_frame_idx() -> None:
    """The subtle one. `frame_idx` on a hit describes the hit's OWN keyframe.
    A temporally expanded neighbour is a different frame, so reusing the
    parent's index would submit the parent's frame under the neighbour's id
    -- a wrong answer that looks perfectly well-formed."""
    from mvr.retrieval.kis import KeyframeFrameIndexResolver

    hits = [
        {"video_id": "V", "frame_id": "kf-2", "frame_idx": 200, "score": 0.9,
         "timestamp": 1.0}
    ]

    results = _run_search(
        hits,
        temporal_radius=1,
        pipeline_resolver=_resolver({"kf-2": 200, "kf-3": 300}),
    )

    assert [(c.keyframe_id, c.frame_id) for c in results] == [
        ("kf-2", 200),
        ("kf-3", 300),
    ]


def test_a_hit_frame_idx_disagreeing_with_the_resolver_is_not_papered_over() -> None:
    from mvr.retrieval.kis import KeyframeFrameIndexResolver

    hits = [
        {"video_id": "V", "frame_id": "kf-1", "frame_idx": 999, "score": 0.9,
         "timestamp": 1.0}
    ]

    with pytest.raises(ValueError):
        _run_search(hits, pipeline_resolver=_resolver({"kf-1": 261}))


# -- one query string cannot serve two branches -------------------------


def test_each_branch_receives_its_own_query_string() -> None:
    """Nhánh dense là OpenCLIP RN50, tiếng Anh thuần. Nhánh lexical là BM25
    trên OCR/ASR tiếng Việt. Một chuỗi duy nhất không thể phục vụ cả hai:
    dịch sang tiếng Anh làm hỏng BM25, giữ tiếng Việt làm hỏng CLIP."""
    from mvr.retrieval.qna_fusion import FusedRetriever

    seen: dict[str, str] = {}

    class _Semantic:
        def search(self, query, top_k):
            seen["dense"] = query
            return []

    def _lexical(query, *, top_k):
        seen["lexical"] = query
        return []

    FusedRetriever(semantic_service=_Semantic(), lexical_search=_lexical).search_hybrid(
        "a red motorbike on a rainy street",
        top_k=10,
        rrf_k=60,
        lexical_query="xe máy màu đỏ trên phố mưa",
    )

    assert seen["dense"] == "a red motorbike on a rainy street"
    assert seen["lexical"] == "xe máy màu đỏ trên phố mưa"


def test_omitting_lexical_query_sends_one_string_to_both_branches() -> None:
    """Hành vi mặc định phải y hệt hiện tại, nếu không toàn bộ đường Q&A đang
    chấm điểm sẽ đổi kết quả mà không ai yêu cầu."""
    from mvr.retrieval.qna_fusion import FusedRetriever

    seen: dict[str, str] = {}

    class _Semantic:
        def search(self, query, top_k):
            seen["dense"] = query
            return []

    def _lexical(query, *, top_k):
        seen["lexical"] = query
        return []

    FusedRetriever(semantic_service=_Semantic(), lexical_search=_lexical).search_hybrid(
        "chỉ một chuỗi", top_k=10, rrf_k=60
    )

    assert seen["dense"] == "chỉ một chuỗi"
    assert seen["lexical"] == "chỉ một chuỗi"


def test_the_old_class_name_still_works() -> None:
    """`QnaFusedRetriever` được import ở `scripts/run_qna_pipeline.py`; đổi tên
    mà không giữ alias sẽ làm vỡ đường Q&A đang chấm điểm."""
    from mvr.retrieval.qna_fusion import FusedRetriever, QnaFusedRetriever

    assert QnaFusedRetriever is FusedRetriever
