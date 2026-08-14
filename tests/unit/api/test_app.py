from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from mvr.api.app import LazyFaissSemanticService, create_app
from mvr.retrieval.semantic import SearchResult


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

    def search_hybrid(self, query: str, *, top_k: int, rrf_k: int):
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
    def search_hybrid(self, query: str, *, top_k: int, rrf_k: int):
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
