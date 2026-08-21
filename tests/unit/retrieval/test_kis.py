from __future__ import annotations

from dataclasses import dataclass

import pytest

from mvr.retrieval.kis import (


    KeyframeFrameIndexResolver,
    KisCandidate,
    resolve_original_frame_id,
    search_kis,
)


def _resolver(mapping, videos=("L21_V001", "L21_V002")):
    """Fixtures pre-dating the pair key: one keyframe id per video.

    `KeyframeFrameIndexResolver` now keys on `(video_id, keyframe_id)`,
    because a bare keyframe id does not identify a keyframe -- the
    organizer's 166,550 keyframes carry only 632 distinct ids. These
    fixtures use synthetic ids that were meant to resolve whatever video
    asked, so registering each id under every video the test uses keeps
    exactly the behaviour they were written to check.
    """
    return KeyframeFrameIndexResolver(
        {(video, kf): idx for video in videos for kf, idx in mapping.items()}
    )


@dataclass
class _FakeSemanticResult:
    frame_id: str
    video_id: str
    score: float
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "video_id": self.video_id,
            "score": self.score,
            "timestamp": self.timestamp,
        }


class _FakeSemanticService:
    def __init__(self, results: list[_FakeSemanticResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return self._results[:top_k]


class _FakeMultimodalService:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.calls: list[tuple[str, int, int]] = []

    def search_hybrid(self, query: str, *, top_k: int, rrf_k: int):
        self.calls.append((query, top_k, rrf_k))
        return self._hits[:top_k]


def _hit(video_id: str, keyframe_id: str, score: float, timestamp: float = 1.0) -> dict:
    return {"video_id": video_id, "frame_id": keyframe_id, "score": score, "timestamp": timestamp}


# -- resolve_original_frame_id --------------------------------------------


def test_resolve_prefers_pipeline_resolver_over_aic_registry() -> None:
    resolver = _resolver({"L21_V001_S0001_K01": 42})

    result = resolve_original_frame_id(
        video_id="L21_V001",
        keyframe_id="L21_V001_S0001_K01",
        pipeline_resolver=resolver,
        aic_registry=None,
    )

    assert result == 42


def test_resolve_falls_back_to_aic_registry_when_pipeline_resolver_misses() -> None:
    from mvr.ingestion.aic_dataset import AicFrameMapping

    class _StubRegistry:
        def resolve_frame(self, video_id: str, keyframe_ref: str):
            return AicFrameMapping(
                video_id=video_id, keyframe_ref=keyframe_ref, frame_idx=261, timestamp=8.7, fps=30.0
            )

    resolver = _resolver({})  # empty: always misses

    result = resolve_original_frame_id(
        video_id="L21_V001",
        keyframe_id="003",
        pipeline_resolver=resolver,
        aic_registry=_StubRegistry(),
    )

    assert result == 261


def test_resolve_never_guesses_when_unresolvable() -> None:
    result = resolve_original_frame_id(
        video_id="L21_V001",
        keyframe_id="unknown",
        pipeline_resolver=None,
        aic_registry=None,
    )

    assert result is None


def test_keyframe_frame_index_resolver_from_csv(tmp_path) -> None:
    csv_path = tmp_path / "keyframe_mapping.csv"
    csv_path.write_text(
        "keyframe_id,video_id,shot_id,ordinal,frame_idx,timestamp,image_path\n"
        "L01_V001_S0001_K01,L01_V001,1,1,120,4.0,keyframes/L01_V001/x.jpg\n",
        encoding="utf-8",
    )

    resolver = KeyframeFrameIndexResolver.from_csv(csv_path)

    assert resolver.resolve("L01_V001", "L01_V001_S0001_K01") == 120
    assert resolver.resolve("L01_V001", "missing") is None


# -- search_kis: validation -------------------------------------------------


def test_search_kis_rejects_blank_query() -> None:
    with pytest.raises(ValueError):
        search_kis("   ", top_k=5, semantic_service=_FakeSemanticService([]), multimodal_service=None)


def test_search_kis_rejects_top_k_zero() -> None:
    with pytest.raises(ValueError):
        search_kis("query", top_k=0, semantic_service=_FakeSemanticService([]), multimodal_service=None)


def test_search_kis_rejects_top_k_101() -> None:
    with pytest.raises(ValueError):
        search_kis("query", top_k=101, semantic_service=_FakeSemanticService([]), multimodal_service=None)


def test_search_kis_requires_at_least_one_backend() -> None:
    with pytest.raises(RuntimeError):
        search_kis("query", top_k=5, semantic_service=None, multimodal_service=None)


# -- search_kis: correctness -------------------------------------------------


def test_search_kis_correct_original_frame_mapping() -> None:
    resolver = _resolver({"kf-1": 261})
    service = _FakeSemanticService(
        [_FakeSemanticResult("kf-1", "L21_V001", score=0.9, timestamp=8.7)]
    )

    results = search_kis(
        "a query", top_k=5, semantic_service=service, multimodal_service=None,
        pipeline_resolver=resolver,
    )

    assert len(results) == 1
    assert results[0] == KisCandidate(
        rank=1, video_id="L21_V001", frame_id=261, score=0.9, keyframe_id="kf-1", timestamp=8.7,
    )


def test_search_kis_top_100_works() -> None:
    mapping = {f"kf-{i}": i for i in range(150)}
    resolver = _resolver(mapping)
    hits = [_hit("L21_V001", f"kf-{i}", score=1.0 - i / 1000) for i in range(150)]
    service = _FakeMultimodalService(hits)

    results = search_kis(
        "a query", top_k=100, semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, overfetch=150,
    )

    assert len(results) == 100
    assert [r.rank for r in results] == list(range(1, 101))


def test_search_kis_default_overfetch_fills_top_100_past_unmapped_and_duplicates() -> None:
    """Hardening Fix 3: at top_k=100, the DEFAULT internal fetch pool (no
    explicit `overfetch`) must exceed 100, so mapping/dedup losses in the
    first 100 raw ranks don't starve the final result below 100 when enough
    valid unique candidates exist further down the raw pool."""
    # 200 raw hits: only even-indexed ones resolve (100 valid unique total);
    # odd-indexed ones are unmapped. Half of the first 100 raw ranks are
    # therefore unusable, exactly the scenario the fix must handle.
    resolver = _resolver({f"kf-{i}": i for i in range(0, 200, 2)})
    hits = [_hit("L21_V001", f"kf-{i}", score=1.0 - i / 1000) for i in range(200)]
    service = _FakeMultimodalService(hits)

    results = search_kis(
        "a query", top_k=100, semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver,
        # no explicit overfetch: exercises the real default formula
    )

    assert len(results) == 100
    assert [r.rank for r in results] == list(range(1, 101))


def test_search_kis_default_overfetch_never_exceeds_top_k_bounds() -> None:
    resolver = _resolver({f"kf-{i}": i for i in range(300)})
    hits = [_hit("L21_V001", f"kf-{i}", score=1.0 - i / 1000) for i in range(300)]

    small = search_kis(
        "q", top_k=20, semantic_service=None,
        multimodal_service=_FakeMultimodalService(hits), pipeline_resolver=resolver,
    )
    large = search_kis(
        "q", top_k=100, semantic_service=None,
        multimodal_service=_FakeMultimodalService(hits), pipeline_resolver=resolver,
    )

    assert len(small) <= 20
    assert len(large) <= 100


def test_search_kis_stable_ranking_preserves_input_order() -> None:
    resolver = _resolver({"kf-a": 10, "kf-b": 20, "kf-c": 30})
    hits = [
        _hit("L21_V001", "kf-a", score=0.9),
        _hit("L21_V001", "kf-b", score=0.8),
        _hit("L21_V001", "kf-c", score=0.7),
    ]
    service = _FakeMultimodalService(hits)

    results = search_kis(
        "a query", top_k=5, semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver,
    )

    assert [r.frame_id for r in results] == [10, 20, 30]
    assert [r.rank for r in results] == [1, 2, 3]


def test_search_kis_deduplicates_same_video_and_frame_id() -> None:
    # Two different keyframes that both resolve to the same original frame.
    resolver = _resolver({"kf-a": 10, "kf-b": 10})
    hits = [
        _hit("L21_V001", "kf-a", score=0.9),
        _hit("L21_V001", "kf-b", score=0.8),
    ]
    service = _FakeMultimodalService(hits)

    results = search_kis(
        "a query", top_k=5, semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver,
    )

    assert len(results) == 1
    assert results[0].keyframe_id == "kf-a"  # first occurrence kept


def test_search_kis_drops_unresolvable_candidates_without_guessing() -> None:
    resolver = _resolver({"kf-a": 10})  # kf-b unknown
    hits = [
        _hit("L21_V001", "kf-a", score=0.9),
        _hit("L21_V001", "kf-b", score=0.8),
    ]
    service = _FakeMultimodalService(hits)

    results = search_kis(
        "a query", top_k=5, semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver,
    )

    assert len(results) == 1
    assert results[0].frame_id == 10


def test_search_kis_prefers_hybrid_over_semantic_when_both_available() -> None:
    resolver = _resolver({"kf-hybrid": 1, "kf-semantic": 2})
    semantic = _FakeSemanticService(
        [_FakeSemanticResult("kf-semantic", "L21_V001", score=0.5, timestamp=1.0)]
    )
    multimodal = _FakeMultimodalService([_hit("L21_V001", "kf-hybrid", score=0.9)])

    results = search_kis(
        "a query", top_k=5, semantic_service=semantic, multimodal_service=multimodal,
        pipeline_resolver=resolver,
    )

    assert len(results) == 1
    assert results[0].keyframe_id == "kf-hybrid"
    assert semantic.calls == []  # semantic branch never called when hybrid is available


def test_kis_candidate_to_answer_matches_evaluator_identity() -> None:
    from mvr.evaluation.aic_kis import KisAnswer

    candidate = KisCandidate(
        rank=1, video_id="L21_V001", frame_id=261, score=0.9, keyframe_id="003",
    )

    assert candidate.to_answer() == KisAnswer(video_id="L21_V001", frame_id=261)


# -- the Vietnamese query the lexical branch needs ----------------------


def test_lexical_query_is_forwarded_when_given() -> None:
    seen: dict[str, object] = {}

    class _Service:
        def search_hybrid(self, query, *, top_k, rrf_k, lexical_query=None):
            seen["query"] = query
            seen["lexical_query"] = lexical_query
            return []

    search_kis(
        "a red motorbike",
        top_k=10,
        semantic_service=None,
        multimodal_service=_Service(),
        lexical_query="xe máy màu đỏ",
    )

    assert seen == {"query": "a red motorbike", "lexical_query": "xe máy màu đỏ"}


def test_a_service_without_lexical_query_support_is_not_broken() -> None:
    """`ElasticsearchMultimodalService` chưa nhận tham số này. Truyền vô điều
    kiện sẽ TypeError và làm hỏng route /search/hybrid đang chạy."""
    seen: dict[str, object] = {}

    class _OldService:
        def search_hybrid(self, query, *, top_k, rrf_k):
            seen["query"] = query
            return []

    search_kis(
        "a red motorbike",
        top_k=10,
        semantic_service=None,
        multimodal_service=_OldService(),
    )

    assert seen["query"] == "a red motorbike"


# -- the resolver must know which video it is answering about -----------


def test_resolver_keys_on_video_and_keyframe(tmp_path) -> None:
    """Đo trên mapping thật: 166.550 dòng, mọi cặp (video_id, keyframe_id)
    đều duy nhất, nhưng chỉ 632 keyframe_id phân biệt. Key trên keyframe_id
    trần thì mỗi video ghi đè video trước và còn lại 632 mục -- 98,8% số cặp
    trả về frame của một video khác."""
    import csv

    path = tmp_path / "keyframe_mapping.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["keyframe_id", "video_id", "frame_idx"])
        writer.writeheader()
        writer.writerow({"keyframe_id": "034", "video_id": "L21_V001", "frame_idx": 3782})
        writer.writerow({"keyframe_id": "034", "video_id": "L21_V002", "frame_idx": 3813})

    resolver = KeyframeFrameIndexResolver.from_csv(path)

    assert resolver.resolve("L21_V001", "034") == 3782
    assert resolver.resolve("L21_V002", "034") == 3813
    assert resolver.resolve("L21_V999", "034") is None


def test_a_bare_string_key_is_refused_loudly() -> None:
    """Một keyframe_id trần không định danh keyframe nào. Chấp nhận nó lặng lẽ
    là cách lỗi này sống sót qua PR #8: nó trả về một con số trông hợp lệ."""
    with pytest.raises(TypeError, match="video_id"):
        KeyframeFrameIndexResolver({"034": 3782})


def test_search_kis_prefers_the_frame_idx_a_hit_states() -> None:
    """Hit lexical tự khai `frame_idx` của nó, nên hỏi resolver là hỏi thừa --
    `search_qna` đã ưu tiên nó từ trước (qna.py:388). `search_kis` thì chưa,
    nên nó vứt con số đúng đi để lấy con số sai từ resolver."""

    class _Service:
        def search_hybrid(self, query, *, top_k, rrf_k, lexical_query=None):
            return [
                {
                    "video_id": "L21_V002",
                    "frame_id": "034",
                    "frame_idx": 3813,
                    "timestamp": 152.5,
                    "score": 9.9,
                }
            ]

    candidates = search_kis(
        "q",
        top_k=10,
        semantic_service=None,
        multimodal_service=_Service(),
        pipeline_resolver=KeyframeFrameIndexResolver({("L21_V001", "034"): 3782}),
    )

    assert [(c.video_id, c.frame_id) for c in candidates] == [("L21_V002", 3813)]


def test_a_frame_idx_disagreeing_with_the_resolver_raises() -> None:
    """Cùng quy tắc `search_qna` dùng: hai nguồn bất đồng là một mâu thuẫn
    thật, phải nổ chứ không phải im lặng chọn một bên."""

    class _Service:
        def search_hybrid(self, query, *, top_k, rrf_k, lexical_query=None):
            return [
                {
                    "video_id": "L21_V001",
                    "frame_id": "034",
                    "frame_idx": 999,
                    "timestamp": 1.0,
                    "score": 1.0,
                }
            ]

    with pytest.raises(ValueError, match="frame_idx disagreement"):
        search_kis(
            "q",
            top_k=10,
            semantic_service=None,
            multimodal_service=_Service(),
            pipeline_resolver=KeyframeFrameIndexResolver({("L21_V001", "034"): 3782}),
        )
