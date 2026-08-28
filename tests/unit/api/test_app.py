from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from mvr.api.app import LazyFaissSemanticService, LazyQueryTranslator, create_app, _default_trake_query_translator
from mvr.retrieval.kis import KeyframeFrameIndexResolver
from mvr.retrieval.semantic import SearchResult


def _resolver(mapping, videos=("video-1", "video-2")):
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
class FakeSemanticService:
    loaded: bool = True
    image_path: str = "/tmp/frame-high.jpg"

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        results = [
            SearchResult(
                rank=1,
                frame_id="frame-high",
                video_id="video-1",
                timestamp=1.25,
                score=0.9,
                image_path=self.image_path,
            ),
            SearchResult(
                rank=2,
                frame_id="frame-low",
                video_id="video-2",
                timestamp=2.5,
                score=0.4,
                image_path="/tmp/frame-low.jpg",
            ),
        ]
        return results[:top_k]


class FakeMultimodalService:
    available = True

    def document_count(self) -> int:
        return 2

    def search_text(self, query: str, *, top_k: int, person_min: int, car_min: int):
        if query == "no-match":
            return {"hits": {"total": {"value": 0}, "hits": []}}
        return _response(_hit("frame-high", score=3.5, ocr_text="hello world"))

    def search_object(self, want: str, *, top_k: int):
        return _response(
            _hit("frame-high", score=1.0, objects={"person": 2})
        )

    def search_hybrid(self, query: str, *, top_k: int, rrf_k: int, lexical_query: str | None = None):
        return [
            {
                **_public_hit("frame-high", score=0.0325),
                "rank_lexical": 1,
                "rank_semantic": 2,
            },
            {
                **_public_hit("frame-low", score=0.0320),
                "rank_lexical": 2,
                "rank_semantic": 1,
            },
        ][:top_k]


class UnavailableSemanticService(FakeSemanticService):
    loaded = False

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        raise RuntimeError("semantic index is unavailable")


class FailingHybridService(FakeMultimodalService):
    def search_hybrid(self, query: str, *, top_k: int, rrf_k: int, lexical_query: str | None = None):
        raise ConnectionError("Elasticsearch connection failed")


def _public_hit(frame_id: str, *, score: float) -> dict:
    return {
        "rank": 1,
        "frame_id": frame_id,
        "video_id": "video-1",
        "timestamp": 1.25,
        "score": score,
        "image_path": f"/keyframes/{frame_id}.jpg",
        "ocr_text": "",
        "caption": "",
        "objects": {},
        "highlight": {},
    }


def _hit(
    frame_id: str,
    *,
    score: float,
    ocr_text: str = "",
    objects: dict[str, int] | None = None,
) -> dict:
    return {
        "_id": frame_id,
        "_score": score,
        "_source": {
            "frame_id": frame_id,
            "video_id": "video-1",
            "ts": 1.25,
            "caption": "",
            "ocr_text": ocr_text,
            "objects": objects or {},
        },
        "highlight": {},
    }


def _response(*hits: dict) -> dict:
    return {"hits": {"total": {"value": len(hits)}, "hits": list(hits)}}


def _client(*, multimodal: bool = True) -> TestClient:
    return TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=FakeMultimodalService() if multimodal else None,
        )
    )


def test_health_reports_real_service_readiness() -> None:
    response = _client().get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "api": "ready",
        "semantic": {
            "configured": True,
            "available": True,
            "loaded": True,
        },
        "elasticsearch": {
            "configured": True,
            "available": True,
            "documents": 2,
        },
    }


def test_health_reports_missing_semantic_runtime_as_degraded() -> None:
    client = TestClient(
        create_app(
            semantic_service=None,
            multimodal_service=None,
            elasticsearch_configured=False,
        )
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "api": "ready",
        "semantic": {
            "configured": False,
            "available": False,
            "loaded": False,
        },
        "elasticsearch": {
            "configured": False,
            "available": False,
        },
    }


def test_health_does_not_report_corrupt_semantic_artifacts_as_available(
    tmp_path,
) -> None:
    for name in ("keyframes.faiss", "metadata.json", "manifest.json"):
        (tmp_path / name).write_text("not a valid artifact", encoding="utf-8")
    service = LazyFaissSemanticService(tmp_path, device="cpu")
    client = TestClient(
        create_app(semantic_service=service, multimodal_service=None)
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["semantic"] == {
        "configured": True,
        "available": False,
        "loaded": False,
    }


def test_semantic_search_returns_ranked_response_contract() -> None:
    response = _client().post(
        "/search/semantic",
        json={"query": "a cyclist", "top_k": 1},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "a cyclist",
        "count": 1,
        "results": [
            {
                "rank": 1,
                "frame_id": "frame-high",
                "video_id": "video-1",
                "timestamp": 1.25,
                "score": 0.9,
                "image_path": "/tmp/frame-high.jpg",
                "ocr_text": None,
                "caption": None,
                "objects": None,
                "highlight": None,
                "rank_lexical": None,
                "rank_semantic": None,
            }
        ],
    }


def test_semantic_result_uses_mounted_keyframe_url(tmp_path) -> None:
    image = tmp_path / "video-1" / "frame-high.jpg"
    image.parent.mkdir()
    image.write_bytes(b"jpeg-smoke")
    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(image_path=str(image)),
            multimodal_service=None,
            keyframe_dir=tmp_path,
        )
    )

    response = client.post(
        "/search/semantic",
        json={"query": "a cyclist", "top_k": 1},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["image_path"] == (
        "/keyframes/video-1/frame-high.jpg"
    )
    image_response = client.get("/keyframes/video-1/frame-high.jpg")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"


def test_semantic_search_rejects_blank_query_and_invalid_top_k() -> None:
    client = _client()

    assert client.post(
        "/search/semantic", json={"query": "", "top_k": 5}
    ).status_code == 422
    assert client.post(
        "/search/semantic", json={"query": "   ", "top_k": 5}
    ).status_code == 422
    assert client.post(
        "/search/semantic", json={"query": "valid", "top_k": 0}
    ).status_code == 422
    assert client.post(
        "/search/semantic", json={"query": "valid", "top_k": 101}
    ).status_code == 422


def test_optional_routes_are_absent_without_elasticsearch_service() -> None:
    client = _client(multimodal=False)

    assert client.post("/search/text", json={"query": "hello"}).status_code == 404
    assert client.post("/search/object", json={"want": "person:1"}).status_code == 404
    assert client.post("/search/hybrid", json={"query": "hello"}).status_code == 404
    paths = client.get("/openapi.json").json()["paths"]
    assert "/search/semantic" in paths
    assert "/search/text" not in paths


def test_text_search_supports_results_and_empty_results() -> None:
    client = _client()

    success = client.post("/search/text", json={"query": "hello", "top_k": 5})
    empty = client.post("/search/text", json={"query": "no-match", "top_k": 5})

    assert success.status_code == 200
    assert success.json()["count"] == 1
    assert success.json()["results"][0]["ocr_text"] == "hello world"
    assert empty.status_code == 200
    assert empty.json() == {"query": "no-match", "count": 0, "results": []}


def test_object_search_translates_parser_errors_to_http_400() -> None:
    response = _client().post(
        "/search/object",
        json={"want": "person:abc", "top_k": 5},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Count for 'person' must be a non-negative integer"
    }


def test_object_and_hybrid_responses_preserve_retrieval_metadata_and_order() -> None:
    client = _client()

    objects = client.post(
        "/search/object", json={"want": "person:2", "top_k": 5}
    )
    hybrid = client.post(
        "/search/hybrid",
        json={"query": "person", "top_k": 2, "rrf_k": 60},
    )

    assert objects.status_code == 200
    assert objects.json()["results"][0]["objects"] == {"person": 2}
    assert hybrid.status_code == 200
    assert [item["frame_id"] for item in hybrid.json()["results"]] == [
        "frame-high",
        "frame-low",
    ]
    assert hybrid.json()["results"][0]["rank_lexical"] == 1
    assert hybrid.json()["rrf_k"] == 60


def test_semantic_runtime_failure_is_not_reported_as_success() -> None:
    client = TestClient(
        create_app(
            semantic_service=UnavailableSemanticService(),
            multimodal_service=None,
        )
    )

    response = client.post(
        "/search/semantic",
        json={"query": "a cyclist", "top_k": 5},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "semantic index is unavailable"


def test_kis_search_returns_ranked_response_contract() -> None:
    resolver = _resolver({"frame-high": 261, "frame-low": 45})
    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=FakeMultimodalService(),
            kis_pipeline_resolver=resolver,
            kis_aic_registry=None,
        )
    )

    response = client.post("/search/kis", json={"query": "a cyclist", "top_k": 2})

    assert response.status_code == 200
    assert response.json() == {
        "query": "a cyclist",
        "count": 2,
        "results": [
            {
                "rank": 1,
                "video_id": "video-1",
                "frame_id": 261,
                "score": 0.0325,
                "keyframe_id": "frame-high",
                "timestamp": 1.25,
            },
            {
                "rank": 2,
                "video_id": "video-1",
                "frame_id": 45,
                "score": 0.0320,
                "keyframe_id": "frame-low",
                "timestamp": 1.25,
            },
        ],
    }


def test_ui_search_adapts_kis_results_for_frontend_contract(tmp_path) -> None:
    keyframe_dir = tmp_path / "keyframes"
    video_dir = tmp_path / "videos"
    (keyframe_dir / "video-1").mkdir(parents=True)
    video_dir.mkdir()
    (keyframe_dir / "video-1" / "frame-high.jpg").write_bytes(b"jpeg")
    (video_dir / "video-1.mp4").write_bytes(b"mp4")

    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=FakeMultimodalService(),
            kis_pipeline_resolver=_resolver({"frame-high": 261, "frame-low": 45}),
            keyframe_dir=keyframe_dir,
            video_dir=video_dir,
        )
    )

    response = client.post(
        "/api/v1/search",
        json={
            "task_mode": "KIS",
            "query": "a cyclist",
            "model_engine": "use all",
            "sim_reorder": True,
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["total_results"] == 1
    assert body["results"][0] == {
        "id": "video-1:261:1",
        "video_id": "video-1",
        "frame_id": 261,
        "timestamp": "00:00:01",
        "timestamp_sec": 1.25,
        "thumbnail_url": "/keyframes/video-1/frame-high.jpg",
        "video_url": "/videos/video-1.mp4",
        "score": 0.0325,
        "objects": [],
        "answer": None,
        "frames": None,
    }
    assert client.get("/keyframes/video-1/frame-high.jpg").status_code == 200
    assert client.get("/videos/video-1.mp4").status_code == 200


def test_ui_submit_previews_official_submission_rows() -> None:
    client = _client(multimodal=False)

    kis = client.post(
        "/api/v1/submit",
        json={
            "query_id": "query-1-kis",
            "task_mode": "KIS",
            "items": [{"video_id": "L01_V028", "frame_id": 25300}],
        },
    )
    qna = client.post(
        "/api/v1/submit",
        json={
            "query_id": "query-2-qa",
            "task_mode": "Q&A",
            "items": [
                {"video_id": "L01_V028", "frame_id": 3450, "answer": 'Năm "người"'}
            ],
        },
    )
    trake = client.post(
        "/api/v1/submit",
        json={
            "query_id": "query-3-trake",
            "task_mode": "TRAKE",
            "items": [
                {"video_id": "L10_V001", "frame_id": 1200},
                {"video_id": "L10_V001", "frame_id": 1850},
                {"video_id": "L10_V001", "frame_id": 2100},
            ],
        },
    )

    assert kis.status_code == 200
    assert kis.json()["csv_preview"] == "L01_V028,25300"
    assert qna.status_code == 200
    assert qna.json()["csv_preview"] == 'L01_V028,3450,"Năm ""người"""'
    assert trake.status_code == 200
    assert trake.json()["csv_preview"] == "L10_V001,1200,1850,2100"


def test_kis_search_rejects_blank_query_and_invalid_top_k() -> None:
    resolver = _resolver({"frame-high": 261})
    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=FakeMultimodalService(),
            kis_pipeline_resolver=resolver,
        )
    )

    assert client.post("/search/kis", json={"query": "", "top_k": 5}).status_code == 422
    assert client.post("/search/kis", json={"query": "   ", "top_k": 5}).status_code == 422
    assert client.post("/search/kis", json={"query": "valid", "top_k": 0}).status_code == 422
    assert client.post("/search/kis", json={"query": "valid", "top_k": 101}).status_code == 422


def test_kis_search_unresolvable_candidates_are_dropped_not_guessed() -> None:
    # No resolver/registry configured at all: nothing can be verified.
    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=FakeMultimodalService(),
            kis_pipeline_resolver=None,
            kis_aic_registry=None,
        )
    )

    response = client.post("/search/kis", json={"query": "a cyclist", "top_k": 5})

    assert response.status_code == 200
    assert response.json() == {"query": "a cyclist", "count": 0, "results": []}


def test_kis_search_returns_503_when_no_retrieval_backend_configured() -> None:
    client = TestClient(
        create_app(
            semantic_service=None,
            multimodal_service=None,
            elasticsearch_configured=False,
        )
    )

    response = client.post("/search/kis", json={"query": "a cyclist", "top_k": 5})

    assert response.status_code == 503


class _RaisingSemanticService:
    """Fails loudly if ever invoked -- proves the KIS route did not fall back
    to (or duplicate work through) the semantic branch when hybrid is used."""

    def search(self, query: str, top_k: int = 5):
        raise AssertionError("semantic branch must not be called when hybrid is available")


class _CallCountingMultimodalService(FakeMultimodalService):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def search_hybrid(self, query: str, *, top_k: int, rrf_k: int, lexical_query: str | None = None):
        self.calls.append((query, top_k, rrf_k))
        return super().search_hybrid(query, top_k=top_k, rrf_k=rrf_k)


def test_kis_search_never_rebuilds_index_or_reloads_model_per_request() -> None:
    resolver = _resolver({"frame-high": 261, "frame-low": 45})
    multimodal_service = _CallCountingMultimodalService()
    client = TestClient(
        create_app(
            semantic_service=_RaisingSemanticService(),
            multimodal_service=multimodal_service,
            kis_pipeline_resolver=resolver,
        )
    )

    first = client.post("/search/kis", json={"query": "first", "top_k": 2})
    second = client.post("/search/kis", json={"query": "second", "top_k": 2})

    assert first.status_code == 200
    assert second.status_code == 200
    # The same injected multimodal_service instance handled both requests
    # (called exactly once per request, no extra calls), and the semantic
    # branch was never touched -- proving no per-request rebuild/reload.
    assert [call[0] for call in multimodal_service.calls] == ["first", "second"]


def test_hybrid_backend_failure_returns_service_unavailable() -> None:
    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=FailingHybridService(),
        ),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/search/hybrid",
        json={"query": "a cyclist", "top_k": 5, "rrf_k": 60},
    )

    assert response.status_code == 503


class _FakeAnswerReasoner:
    def __init__(self, answer: str = "5", source: str = "context-grounded") -> None:
        self._answer = answer
        self._source = source
        self.calls: list[tuple[str, object]] = []

    def answer(self, question: str, context):
        self.calls.append((question, context))
        from mvr.enrichment.qna_answer import QnaAnswer

        return QnaAnswer(text=self._answer, source=self._source, raw_output="{}")


def test_qna_search_returns_ranked_answered_response_contract() -> None:
    resolver = _resolver({"frame-high": 261, "frame-low": 45})
    reasoner = _FakeAnswerReasoner(answer="blue", source="context-grounded")
    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=FakeMultimodalService(),
            kis_pipeline_resolver=resolver,
            kis_aic_registry=None,
            qna_answer_reasoner=reasoner,
        )
    )

    response = client.post(
        "/search/qna",
        json={"event_description": "a party scene", "question": "what color?", "top_k": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["event_description"] == "a party scene"
    assert body["question"] == "what color?"
    assert body["count"] == 2
    assert body["results"][0] == {
        "rank": 1,
        "video_id": "video-1",
        "frame_id": 261,
        "score": 0.0325,
        "keyframe_id": "frame-high",
        "timestamp": 1.25,
        "answer": "blue",
        "answer_source": "context-grounded",
        # No keyframe directory configured in this test, so nothing servable.
        "image_path": None,
    }
    assert reasoner.calls[0][0] == "what color?"


def test_qna_search_exposes_servable_image_url_when_keyframes_are_mounted(tmp_path) -> None:
    """The UI needs a URL it can actually load to show the frame to a human
    reviewer -- and must never receive a raw local filesystem path."""
    keyframe_dir = tmp_path / "keyframes"
    (keyframe_dir / "video-1").mkdir(parents=True)
    image = keyframe_dir / "video-1" / "frame-high.jpg"
    image.write_bytes(b"jpeg")

    class _AbsolutePathService(FakeMultimodalService):
        def search_hybrid(self, query: str, *, top_k: int, rrf_k: int, lexical_query: str | None = None):
            hits = super().search_hybrid(query, top_k=top_k, rrf_k=rrf_k, lexical_query=lexical_query)
            hits[0] = {**hits[0], "image_path": str(image)}
            return hits

    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=_AbsolutePathService(),
            kis_pipeline_resolver=_resolver({"frame-high": 261, "frame-low": 45}),
            keyframe_dir=keyframe_dir,
            qna_answer_reasoner=_FakeAnswerReasoner(),
        )
    )

    response = client.post(
        "/search/qna",
        json={"event_description": "e", "question": "q", "top_k": 1},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["image_path"] == "/keyframes/video-1/frame-high.jpg"


def test_qna_search_hides_image_paths_outside_the_served_directory(tmp_path) -> None:
    """A candidate image that the API cannot serve must degrade to null,
    never leak the local absolute path of the competition dataset."""
    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=FakeMultimodalService(),
            kis_pipeline_resolver=_resolver({"frame-high": 261}),
            keyframe_dir=tmp_path / "somewhere-else",
            qna_answer_reasoner=_FakeAnswerReasoner(),
        )
    )

    response = client.post(
        "/search/qna",
        json={"event_description": "e", "question": "q", "top_k": 1},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["image_path"] is None


def test_qna_search_rejects_blank_fields_and_invalid_top_k() -> None:
    resolver = _resolver({"frame-high": 261})
    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=FakeMultimodalService(),
            kis_pipeline_resolver=resolver,
            qna_answer_reasoner=_FakeAnswerReasoner(),
        )
    )

    assert client.post(
        "/search/qna", json={"event_description": "", "question": "q", "top_k": 5}
    ).status_code == 422
    assert client.post(
        "/search/qna", json={"event_description": "e", "question": "  ", "top_k": 5}
    ).status_code == 422
    assert client.post(
        "/search/qna", json={"event_description": "e", "question": "q", "top_k": 0}
    ).status_code == 422
    assert client.post(
        "/search/qna", json={"event_description": "e", "question": "q", "top_k": 101}
    ).status_code == 422


def test_qna_search_returns_503_when_no_answer_reasoner_configured() -> None:
    resolver = _resolver({"frame-high": 261})
    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=FakeMultimodalService(),
            kis_pipeline_resolver=resolver,
            qna_answer_reasoner=None,
        )
    )

    response = client.post(
        "/search/qna",
        json={"event_description": "e", "question": "q", "top_k": 5},
    )

    assert response.status_code == 503


def test_qna_search_returns_503_when_no_retrieval_backend_configured() -> None:
    client = TestClient(
        create_app(
            semantic_service=None,
            multimodal_service=None,
            elasticsearch_configured=False,
            qna_answer_reasoner=_FakeAnswerReasoner(),
        )
    )

    response = client.post(
        "/search/qna",
        json={"event_description": "e", "question": "q", "top_k": 5},
    )

    assert response.status_code == 503


def test_qna_search_unresolvable_candidates_are_dropped_not_guessed() -> None:
    client = TestClient(
        create_app(
            semantic_service=FakeSemanticService(),
            multimodal_service=FakeMultimodalService(),
            kis_pipeline_resolver=None,
            kis_aic_registry=None,
            qna_answer_reasoner=_FakeAnswerReasoner(),
        )
    )

    response = client.post(
        "/search/qna",
        json={"event_description": "e", "question": "q", "top_k": 5},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 0


def test_default_trake_query_translator_is_none_without_the_env_var(monkeypatch) -> None:
    monkeypatch.delenv("MVR_TRAKE_TRANSLATE_QUERIES", raising=False)
    assert _default_trake_query_translator() is None


def test_default_trake_query_translator_is_constructed_when_opted_in(monkeypatch) -> None:
    """Only checks the lazy wrapper is built when the env var is set -- does
    NOT load the actual translation model (LazyQueryTranslator defers that
    to first .translate() call, so this needs no cached model/network)."""
    monkeypatch.setenv("MVR_TRAKE_TRANSLATE_QUERIES", "1")
    translator = _default_trake_query_translator()

    assert isinstance(translator, LazyQueryTranslator)


def test_lazy_query_translator_fails_closed_and_caches_the_error() -> None:
    """A bogus model name must raise (mapped to a 503 by translate_for_retrieval's
    caller, or safely swallowed by translate_for_retrieval itself), and must
    not attempt to reload -- and therefore re-fail -- on every call."""
    translator = LazyQueryTranslator(
        model_name="this-model-does-not-exist/bogus", allow_model_download=False, device="cpu"
    )

    with pytest.raises(RuntimeError):
        translator.translate("hello")
    first_error = translator._error
    with pytest.raises(RuntimeError):
        translator.translate("hello again")

    assert translator._error is first_error  # same cached failure, not reloaded
