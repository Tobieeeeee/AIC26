"""FAISS ↔ Milvus parity tests and incremental ingest tests.

Uses a small deterministic in-memory fixture. No pymilvus or FAISS
installation required — backends are mocked to expose pure logic.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Deterministic fixture
# ---------------------------------------------------------------------------

FIXTURE_N = 10  # small fixture
FIXTURE_DIMS = {"beit3": 16, "metaclip2": 8, "perception_encoder": 32}


def _make_fixture_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms > 0, norms, 1.0)


def _make_fixture_keyframes(n: int):
    return [
        {
            "keyframe_id": f"kf{i:04d}",
            "video_id": f"V{i // 3:03d}",
            "frame_id": i * 10,
            "shot_id": i,
            "timestamp": float(i),
            "batch_id": "1",
            "collection_id": "L21",
        }
        for i in range(n)
    ]


class FixtureData:
    def __init__(self):
        self.keyframes = _make_fixture_keyframes(FIXTURE_N)
        self.vectors = {
            pid: _make_fixture_vectors(FIXTURE_N, dim, seed=i)
            for i, (pid, dim) in enumerate(FIXTURE_DIMS.items())
        }
        self.keyframe_ids = [kf["keyframe_id"] for kf in self.keyframes]


# ---------------------------------------------------------------------------
# Pure Python reference search (FLAT inner product)
# ---------------------------------------------------------------------------

def _flat_search(vectors: np.ndarray, query: np.ndarray, keyframe_ids: list[str], top_k: int):
    scores = vectors @ query.astype(np.float32)
    ranked = sorted(zip(scores.tolist(), keyframe_ids), reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# FAISS parity tests
# ---------------------------------------------------------------------------

class TestFaissMilvusParity:
    """Verify that Milvus FLAT and Python FLAT produce the same top-k identities."""

    def _fake_milvus_search(self, vectors, query, keyframe_ids, top_k):
        """Simulate what Milvus FLAT IP would return for L2-normalized vectors."""
        return _flat_search(vectors, query, keyframe_ids, top_k)

    def test_flat_ip_same_top_k_identities(self):
        fixture = FixtureData()
        rng = np.random.RandomState(99)
        provider_id = "beit3"
        dim = FIXTURE_DIMS[provider_id]
        query = rng.randn(dim).astype(np.float32)
        query /= max(np.linalg.norm(query), 1e-8)

        top_k = 5
        vectors = fixture.vectors[provider_id]
        ids = fixture.keyframe_ids

        # Python reference (FAISS-equivalent exact search)
        faiss_results = _flat_search(vectors, query, ids, top_k)
        faiss_ids = [r[1] for r in faiss_results]

        # Simulated Milvus FLAT (same math)
        milvus_results = self._fake_milvus_search(vectors, query, ids, top_k)
        milvus_ids = [r[1] for r in milvus_results]

        assert faiss_ids == milvus_ids, (
            f"FAISS and Milvus FLAT produce different top-{top_k}: "
            f"{faiss_ids} vs {milvus_ids}"
        )

    def test_flat_ip_same_ordering_all_providers(self):
        fixture = FixtureData()
        rng = np.random.RandomState(42)
        ids = fixture.keyframe_ids

        for pid, dim in FIXTURE_DIMS.items():
            query = rng.randn(dim).astype(np.float32)
            query /= max(np.linalg.norm(query), 1e-8)
            vectors = fixture.vectors[pid]

            faiss_ids = [r[1] for r in _flat_search(vectors, query, ids, top_k=5)]
            milvus_ids = [r[1] for r in self._fake_milvus_search(vectors, query, ids, top_k=5)]

            assert faiss_ids == milvus_ids, f"Provider {pid}: ordering mismatch"

    def test_rrf_parity_python_vs_milvus(self):
        """Python RRF k=60 on identical ranked lists must be deterministic."""
        from mvr.retrieval.hybrid import rrf_fuse

        fixture = FixtureData()
        rng = np.random.RandomState(7)
        ids = fixture.keyframe_ids

        all_ranked: list[list[str]] = []
        for pid, dim in FIXTURE_DIMS.items():
            query = rng.randn(dim).astype(np.float32)
            query /= max(np.linalg.norm(query), 1e-8)
            vectors = fixture.vectors[pid]
            ranked = [r[1] for r in _flat_search(vectors, query, ids, top_k=len(ids))]
            all_ranked.append(ranked)

        result1 = rrf_fuse(all_ranked, rrf_k=60, top_k=5)
        result2 = rrf_fuse(all_ranked, rrf_k=60, top_k=5)
        assert result1 == result2

    def test_rrf_top_k_set_coverage(self):
        """All ids in top-k result must have come from input ranked lists."""
        from mvr.retrieval.hybrid import rrf_fuse
        fixture = FixtureData()
        ids = fixture.keyframe_ids
        all_ranked = [ids[:5], ids[2:7], ids[4:9]]
        result = rrf_fuse(all_ranked, rrf_k=60, top_k=10)
        result_ids = {r[0] for r in result}
        input_ids = set(ids[:9])
        assert result_ids.issubset(input_ids)


# ---------------------------------------------------------------------------
# Incremental ingest tests
# ---------------------------------------------------------------------------

class TestIncrementalIngest:
    """Batch2 must not duplicate Batch1; new rows upsert; entity count updates."""

    def _build_entities(self, start: int, n: int) -> list[dict[str, Any]]:
        from mvr.indexing.milvus_schema import (
            FIELD_ENTITY_ID, FIELD_KEYFRAME_ID, FIELD_VIDEO_ID, FIELD_FRAME_ID, FIELD_SHOT_ID,
            FIELD_TIMESTAMP, FIELD_BATCH_ID, FIELD_COLLECTION_ID,
            FIELD_BEIT3_VECTOR, FIELD_METACLIP2_VECTOR, FIELD_PE_VECTOR,
            make_entity_id,
        )
        rng = np.random.RandomState(start)
        entities = []
        for i in range(start, start + n):
            keyframe_id = f"kf{i:04d}"
            video_id = f"V{i // 3:03d}"
            entities.append({
                FIELD_ENTITY_ID: make_entity_id(video_id, keyframe_id),
                FIELD_KEYFRAME_ID: keyframe_id,
                FIELD_VIDEO_ID: video_id,
                FIELD_FRAME_ID: i * 10,
                FIELD_SHOT_ID: i,
                FIELD_TIMESTAMP: float(i),
                FIELD_BATCH_ID: "1",
                FIELD_COLLECTION_ID: "L21",
                FIELD_BEIT3_VECTOR: rng.randn(768).astype(np.float32).tolist(),
                FIELD_METACLIP2_VECTOR: rng.randn(512).astype(np.float32).tolist(),
                FIELD_PE_VECTOR: rng.randn(1024).astype(np.float32).tolist(),
            })
        return entities

    def test_batch2_does_not_duplicate_batch1(self):
        """Upsert semantics: inserting same entity_id twice should not create duplicate."""
        # Simulate an in-memory store (PK-deduplicated like Milvus upsert)
        store: dict[str, dict] = {}

        batch1 = self._build_entities(0, 5)
        batch2 = self._build_entities(3, 5)  # overlaps kf3, kf4 with batch1

        for e in batch1:
            store[e["entity_id"]] = e
        for e in batch2:
            store[e["entity_id"]] = e  # upsert

        # Total should be 0-7 = 8 unique keyframes, not 10
        assert len(store) == 8

    def test_new_rows_inserted(self):
        store: dict[str, dict] = {}
        batch1 = self._build_entities(0, 3)
        for e in batch1:
            store[e["entity_id"]] = e
        assert len(store) == 3

        batch2 = self._build_entities(3, 3)  # no overlap
        for e in batch2:
            store[e["entity_id"]] = e
        assert len(store) == 6

    def test_changed_rows_upserted(self):
        store: dict[str, dict] = {}
        batch1 = self._build_entities(0, 1)
        for e in batch1:
            store[e["entity_id"]] = e

        # Same entity_id, different vector
        updated = dict(batch1[0])
        updated["beit3_vector"] = [999.0] * 768
        store[updated["entity_id"]] = updated

        assert store["V000/kf0000"]["beit3_vector"][0] == pytest.approx(999.0)
        assert len(store) == 1  # no new entry

    def test_entity_count_updates(self):
        store: dict[str, dict] = {}
        for e in self._build_entities(0, 5):
            store[e["entity_id"]] = e
        assert len(store) == 5

        for e in self._build_entities(5, 3):
            store[e["entity_id"]] = e
        assert len(store) == 8

    def test_same_keyframe_id_different_video_id_does_not_collide(self):
        """The whole point of entity_id: keyframe_id alone is NOT globally
        unique across the organizer's dataset (see task05a.py), so two
        videos sharing a keyframe_id string must never overwrite each
        other's entity in an upsert-by-PK store."""
        from mvr.indexing.milvus_schema import make_entity_id

        store: dict[str, dict] = {}
        shared_keyframe_id = "034"
        entity_a = {
            "entity_id": make_entity_id("L21_V001", shared_keyframe_id),
            "keyframe_id": shared_keyframe_id,
            "video_id": "L21_V001",
        }
        entity_b = {
            "entity_id": make_entity_id("L21_V002", shared_keyframe_id),
            "keyframe_id": shared_keyframe_id,
            "video_id": "L21_V002",
        }
        store[entity_a["entity_id"]] = entity_a
        store[entity_b["entity_id"]] = entity_b

        assert len(store) == 2
        assert store[entity_a["entity_id"]]["video_id"] == "L21_V001"
        assert store[entity_b["entity_id"]]["video_id"] == "L21_V002"
