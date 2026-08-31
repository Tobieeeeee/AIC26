"""Opt-in live Milvus production-backend smoke test.

Uses only a tiny deterministic five-entity fixture. It never downloads model
weights and refuses to recreate a collection unless the target is the local
development service and the canonical collection name.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mvr.api.app import create_app
from mvr.enrichment.embeddings.query_encoders import QueryEmbedding
from mvr.indexing.milvus_backend import MilvusProviderBackend
from mvr.indexing.milvus_client import MilvusClient, MilvusConfig
from mvr.indexing.milvus_schema import (
    COLLECTION_NAME,
    EXPECTED_DIMENSIONS,
    INDEX_TYPE_PARAMS,
    VECTOR_FIELDS,
    build_collection_schema,
    make_entity_id,
    vector_field_for_provider,
)
from mvr.retrieval.milvus_semantic import MilvusSemanticService


pytestmark = pytest.mark.skipif(
    os.environ.get("MVR_RUN_MILVUS_RUNTIME_TESTS") != "1",
    reason="set MVR_RUN_MILVUS_RUNTIME_TESTS=1 for the local Milvus smoke test",
)

PROVIDERS = ("beit3", "metaclip2", "perception_encoder")


def _normalized_fixture(dimension: int, count: int = 5) -> np.ndarray:
    values = np.zeros((count, dimension), dtype=np.float32)
    for row in range(count):
        values[row, row] = 1.0
    return values


class _FixtureEncoders:
    def encode_query(self, model_id, text, *, index_contract):
        assert text == "fixture query"
        assert index_contract.model_id == model_id
        dimension = EXPECTED_DIMENSIONS[model_id]
        vector = np.zeros(dimension, dtype=np.float32)
        vector[0] = 1.0
        return QueryEmbedding(model_id, vector, dimension, "float32", True)


def _build_fixture() -> tuple[MilvusSemanticService, dict[str, np.ndarray]]:
    config = MilvusConfig.from_env()
    if config.host not in {"localhost", "127.0.0.1"} or config.uri not in {
        "",
        "http://localhost:19530",
        "http://127.0.0.1:19530",
    }:
        pytest.fail("live fixture may only recreate the local development Milvus")
    if config.collection != COLLECTION_NAME or config.token:
        pytest.fail("live fixture requires the canonical local collection without a token")

    client = MilvusClient(config)
    client.create_collection(
        build_collection_schema(), COLLECTION_NAME, drop_existing=True
    )
    vectors = {
        provider: _normalized_fixture(dimension)
        for provider, dimension in EXPECTED_DIMENSIONS.items()
    }
    entities = []
    for row in range(5):
        keyframe_id = f"kf-{row}"
        video_id = "fixture-video"
        entities.append(
            {
                "entity_id": make_entity_id(video_id, keyframe_id),
                "keyframe_id": keyframe_id,
                "video_id": video_id,
                "frame_id": 100 + row,
                "shot_id": row,
                "timestamp": float(row) * 1.25,
                "batch_id": "fixture-batch",
                "collection_id": "fixture-collection",
                "beit3_vector": vectors["beit3"][row].tolist(),
                "metaclip2_vector": vectors["metaclip2"][row].tolist(),
                "pe_vector": vectors["perception_encoder"][row].tolist(),
            }
        )
    # Collision-safety proof: a different video sharing keyframe_id "kf-0"
    # with row 0 above must become its own entity, not overwrite row 0 --
    # keyframe_id is not globally unique across the organizer's dataset
    # (see task05a.py). Its vectors deliberately do not match the unit-query
    # fixture, so if entity_id ever collapses back to bare keyframe_id, this
    # row would silently corrupt row 0's data and the parity assertions
    # below would fail.
    collision_video_id = "fixture-video-2"
    collision_entity = {
        "entity_id": make_entity_id(collision_video_id, "kf-0"),
        "keyframe_id": "kf-0",
        "video_id": collision_video_id,
        "frame_id": 999,
        "shot_id": 99,
        "timestamp": 99.0,
        "batch_id": "fixture-batch",
        "collection_id": "fixture-collection",
        "beit3_vector": (-vectors["beit3"][0]).tolist(),
        "metaclip2_vector": (-vectors["metaclip2"][0]).tolist(),
        "pe_vector": (-vectors["perception_encoder"][0]).tolist(),
    }
    entities.append(collision_entity)
    inserted = client.upsert_batch(entities, collection_name=COLLECTION_NAME)
    assert len(inserted) == 6
    client.flush_collection(COLLECTION_NAME)
    for field in VECTOR_FIELDS:
        client.create_index(
            field,
            index_type="HNSW",
            metric_type="IP",
            params=INDEX_TYPE_PARAMS["HNSW"],
            collection_name=COLLECTION_NAME,
        )
    client.load_collection(COLLECTION_NAME)
    backend = MilvusProviderBackend(
        client, index_type="HNSW", collection_name=COLLECTION_NAME
    )
    assert client.entity_count(COLLECTION_NAME) == 6
    return MilvusSemanticService(backend, query_encoders=_FixtureEncoders()), vectors


def test_live_milvus_fixture_parity_and_fastapi_runtime() -> None:
    faiss = pytest.importorskip("faiss", reason="FAISS reference backend unavailable")
    service, vectors = _build_fixture()

    health = service.runtime_status()
    assert health["backend"] == "milvus"
    assert health["reachable"] is True
    assert health["collection_ready"] is True
    assert health["entities"] == 6

    for provider in PROVIDERS:
        query = np.zeros(EXPECTED_DIMENSIONS[provider], dtype=np.float32)
        query[0] = 1.0
        contract = service.contracts[provider]
        milvus_hits = service.provider_search_backend.search(contract, query, top_k=5)

        reference = faiss.IndexFlatIP(EXPECTED_DIMENSIONS[provider])
        reference.add(vectors[provider])
        _, indices = reference.search(query.reshape(1, -1), 5)
        expected_ids = [f"kf-{index}" for index in indices[0]]
        actual_ids = [str(hit.metadata["keyframe_id"]) for hit in milvus_hits]
        assert actual_ids == expected_ids
        assert vector_field_for_provider(provider) in VECTOR_FIELDS

    fused = service.search("fixture query", top_k=5)
    assert [hit.keyframe_id for hit in fused] == [
        "kf-0",
        "kf-1",
        "kf-2",
        "kf-3",
        "kf-4",
    ]
    assert fused[0].provider_ranks == {
        "beit3": 1,
        "metaclip2": 1,
        "perception_encoder": 1,
    }

    client = TestClient(create_app(semantic_service=service, multimodal_service=None))
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["semantic"]["backend"] == "milvus"
    assert health_response.json()["semantic"]["collection_ready"] is True

    response = client.post(
        "/search/semantic", json={"query": "fixture query", "top_k": 3}
    )
    assert response.status_code == 200
    first = response.json()["results"][0]
    assert first["keyframe_id"] == "kf-0"
    assert first["video_id"] == "fixture-video"
    assert first["frame_id"] == 100
    assert first["timestamp"] == 0.0
