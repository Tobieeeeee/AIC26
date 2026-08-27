from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mvr.api.app import (
    ElasticsearchMultimodalService,
    LazyFaissSemanticService,
    _default_semantic_service,
    create_app,
)
from mvr.enrichment.embeddings.query_encoders import QueryEmbedding
from mvr.indexing.milvus_backend import MilvusProviderBackend
from mvr.indexing.milvus_client import MilvusClient, MilvusConnectionError
from mvr.indexing.milvus_schema import EXPECTED_DIMENSIONS, VECTOR_FIELDS
from mvr.retrieval.milvus_semantic import MilvusSemanticService


class _DeterministicEncoders:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def encode_query(self, model_id, text, *, index_contract):
        self.calls.append((model_id, text, index_contract.collection_name))
        dimension = EXPECTED_DIMENSIONS[model_id]
        vector = np.zeros(dimension, dtype=np.float32)
        vector[0] = 1.0
        return QueryEmbedding(model_id, vector, dimension, "float32", True)


def _ready_service() -> tuple[MilvusSemanticService, MagicMock, _DeterministicEncoders]:
    client = MagicMock(spec=MilvusClient)
    client.health.return_value = {"status": "ok"}
    client.collection_info.return_value = {
        "fields": ["keyframe_id", *VECTOR_FIELDS],
        "dimensions": {
            "beit3_vector": 768,
            "metaclip2_vector": 512,
            "pe_vector": 1024,
        },
        "num_entities": 2,
        "indexes": [{"field": field, "type": "FLAT"} for field in VECTOR_FIELDS],
        "loaded": True,
    }
    backend = MagicMock(spec=MilvusProviderBackend)
    backend.client = client

    def search_provider(*, query_vector, provider_id, top_k, collection_name):
        assert query_vector.shape == (EXPECTED_DIMENSIONS[provider_id],)
        assert collection_name == "aic26_keyframes"
        provider_order = {
            "beit3": ("kf-a", "kf-b"),
            "metaclip2": ("kf-b", "kf-a"),
            "perception_encoder": ("kf-a", "kf-b"),
        }[provider_id]
        return [
            {
                "keyframe_id": keyframe_id,
                "video_id": "V01",
                "frame_id": 101 if keyframe_id == "kf-a" else 202,
                "shot_id": 1,
                "timestamp": 1.25 if keyframe_id == "kf-a" else 2.5,
                "batch_id": "B1",
                "collection_id": "C1",
                "distance": 1.0 - rank * 0.1,
            }
            for rank, keyframe_id in enumerate(provider_order[:top_k])
        ]

    backend.search_provider.side_effect = search_provider
    encoders = _DeterministicEncoders()
    return MilvusSemanticService(backend, query_encoders=encoders), backend, encoders


def test_default_backend_is_milvus(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.delenv("MVR_SEMANTIC_BACKEND", raising=False)
    with patch(
        "mvr.retrieval.milvus_semantic.MilvusSemanticService.from_env",
        return_value=sentinel,
    ) as factory:
        assert _default_semantic_service() is sentinel
    factory.assert_called_once_with(index_type="FLAT")


def test_faiss_reference_requires_explicit_selection(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MVR_SEMANTIC_BACKEND", "faiss_reference")
    monkeypatch.setenv("MVR_SEMANTIC_INDEX", str(tmp_path))

    service = _default_semantic_service()

    assert isinstance(service, LazyFaissSemanticService)
    assert service.index_root == tmp_path.resolve()


def test_milvus_factory_failure_never_falls_back_to_faiss(monkeypatch) -> None:
    monkeypatch.setenv("MVR_SEMANTIC_BACKEND", "milvus")
    monkeypatch.setenv("MVR_SEMANTIC_INDEX", "must-not-be-used")
    with patch(
        "mvr.retrieval.milvus_semantic.MilvusSemanticService.from_env",
        side_effect=RuntimeError("bad Milvus configuration"),
    ), pytest.raises(RuntimeError, match="bad Milvus configuration"):
        _default_semantic_service()


def test_unknown_backend_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("MVR_SEMANTIC_BACKEND", "auto")
    with pytest.raises(RuntimeError, match="milvus.*faiss_reference"):
        _default_semantic_service()


def test_service_routes_three_providers_and_fuses_by_rrf() -> None:
    service, backend, encoders = _ready_service()

    results = service.search("fixture query", top_k=2)

    assert [result.keyframe_id for result in results] == ["kf-a", "kf-b"]
    assert results[0].frame_id == 101
    assert results[0].provider_ranks == {
        "beit3": 1,
        "metaclip2": 2,
        "perception_encoder": 1,
    }
    assert results[0].score == pytest.approx(2 / 61 + 1 / 62)
    assert backend.search_provider.call_count == 3
    assert [call[0] for call in encoders.calls] == [
        "beit3",
        "metaclip2",
        "perception_encoder",
    ]
    assert all(call[2].endswith(f"__{call[0]}") for call in encoders.calls)


def test_health_distinguishes_reachability_and_collection_readiness() -> None:
    service, _, _ = _ready_service()
    response = TestClient(
        create_app(semantic_service=service, multimodal_service=None)
    ).get("/health")

    assert response.status_code == 200
    assert response.json()["semantic"] == {
        "configured": True,
        "available": True,
        "loaded": True,
        "backend": "milvus",
        "reachable": True,
        "collection_ready": True,
        "collection": "aic26_keyframes",
        "entities": 2,
    }


def test_unreachable_milvus_degrades_health_and_never_falls_back() -> None:
    service, backend, _ = _ready_service()
    backend.client.health.side_effect = MilvusConnectionError("connection refused")
    client = TestClient(create_app(semantic_service=service, multimodal_service=None))

    health = client.get("/health")
    response = client.post(
        "/search/semantic",
        json={"query": "fixture query", "top_k": 1},
    )

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["semantic"] == {
        "configured": True,
        "available": False,
        "loaded": False,
        "backend": "milvus",
        "reachable": False,
        "collection_ready": False,
        "collection": "aic26_keyframes",
        "error": "Milvus unreachable: connection refused",
    }
    assert response.status_code == 503
    assert "Milvus unreachable" in response.json()["detail"]
    backend.search_provider.assert_not_called()


def test_semantic_api_preserves_canonical_milvus_identity() -> None:
    service, _, _ = _ready_service()
    response = TestClient(
        create_app(semantic_service=service, multimodal_service=None)
    ).post("/search/semantic", json={"query": "fixture query", "top_k": 1})

    assert response.status_code == 200
    hit = response.json()["results"][0]
    assert hit["keyframe_id"] == "kf-a"
    assert hit["video_id"] == "V01"
    assert hit["frame_id"] == 101
    assert hit["frame_idx"] == 101
    assert hit["timestamp"] == 1.25
    assert hit["image_path"] == "/keyframes/V01/kf-a.jpg"


def test_kis_prefers_explicit_keyframe_id_from_milvus_result() -> None:
    service, _, _ = _ready_service()

    class _Resolver:
        def resolve(self, video_id: str, keyframe_id: str):
            assert video_id == "V01"
            return {"kf-a": 101}.get(keyframe_id)

    response = TestClient(
        create_app(
            semantic_service=service,
            multimodal_service=None,
            kis_pipeline_resolver=_Resolver(),
            kis_aic_registry=None,
        )
    ).post("/search/kis", json={"query": "fixture query", "top_k": 1})

    assert response.status_code == 200
    assert response.json()["results"][0]["keyframe_id"] == "kf-a"
    assert response.json()["results"][0]["frame_id"] == 101


def test_hybrid_uses_shared_milvus_semantic_service() -> None:
    service, backend, _ = _ready_service()
    lexical_response = {
        "hits": {
            "total": {"value": 1},
            "hits": [
                {
                    "_score": 2.0,
                    "_source": {
                        "frame_id": "kf-a",
                        "video_id": "V01",
                        "ts": 1.25,
                        "ocr_text": "fixture",
                        "caption": "",
                        "objects": {},
                    },
                }
            ],
        }
    }
    hybrid = ElasticsearchMultimodalService(
        object(), semantic_service=service
    )

    with patch("mvr.api.app.search_lexical", return_value=lexical_response):
        results = hybrid.search_hybrid("fixture query", top_k=2, rrf_k=60)

    assert backend.search_provider.call_count == 3
    assert results[0]["video_id"] == "V01"
    assert results[0]["rank_lexical"] == 1
    assert results[0]["rank_semantic"] == 1
