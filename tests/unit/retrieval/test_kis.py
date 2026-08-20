from __future__ import annotations

from dataclasses import dataclass

import pytest

from mvr.retrieval.kis import (
    KeyframeFrameIndexResolver,
    KisCandidate,
    resolve_original_frame_id,
    search_kis,
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
    resolver = KeyframeFrameIndexResolver({"L21_V001_S0001_K01": 42})

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

    resolver = KeyframeFrameIndexResolver({})  # empty: always misses

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

    assert resolver.resolve("L01_V001_S0001_K01") == 120
    assert resolver.resolve("missing") is None


# -- search_kis: validation -------------------------------------------------


def test_search_kis_rejects_blank_query() -> None:
    with pytest.raises(ValueError):
        search_kis("   ", top_k=5, semantic_service=_FakeSemanticService([]), multimodal_service=None)


def test_search_kis_rejects_top_k_zero() -> None:
    with pytest.raises(ValueError):
        search_kis("query", top_k=0, semantic_service=_FakeSemanticService([]), multimodal_service=None)


def test_search_kis_rejects_top_k_501() -> None:
    with pytest.raises(ValueError):
        search_kis("query", top_k=501, semantic_service=_FakeSemanticService([]), multimodal_service=None)


def test_search_kis_accepts_top_k_above_100_for_trake_shortlisting() -> None:
    """TRAKE's video-shortlisting step needs top_k up to 500 (see
    agent_task/trake_competition_ready/01_REPO_CONTRACT_AUDIT_AND_HARDENING.md);
    plain KIS callers are unaffected since the public API/CLI never pass
    more than 100."""
    mapping = {f"kf-{i}": i for i in range(300)}
    resolver = KeyframeFrameIndexResolver(mapping)
    hits = [_hit("L21_V001", f"kf-{i}", score=1.0 - i / 1000) for i in range(300)]
    service = _FakeMultimodalService(hits)

    results = search_kis(
        "a query", top_k=300, semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, overfetch=300,
    )

    assert len(results) == 300
    assert [r.rank for r in results] == list(range(1, 301))


def test_search_kis_requires_at_least_one_backend() -> None:
    with pytest.raises(RuntimeError):
        search_kis("query", top_k=5, semantic_service=None, multimodal_service=None)


# -- search_kis: correctness -------------------------------------------------


def test_search_kis_correct_original_frame_mapping() -> None:
    resolver = KeyframeFrameIndexResolver({"kf-1": 261})
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
    resolver = KeyframeFrameIndexResolver(mapping)
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
    resolver = KeyframeFrameIndexResolver({f"kf-{i}": i for i in range(0, 200, 2)})
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
    resolver = KeyframeFrameIndexResolver({f"kf-{i}": i for i in range(300)})
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
    resolver = KeyframeFrameIndexResolver({"kf-a": 10, "kf-b": 20, "kf-c": 30})
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
    resolver = KeyframeFrameIndexResolver({"kf-a": 10, "kf-b": 10})
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
    resolver = KeyframeFrameIndexResolver({"kf-a": 10})  # kf-b unknown
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
    resolver = KeyframeFrameIndexResolver({"kf-hybrid": 1, "kf-semantic": 2})
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
