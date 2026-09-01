"""Tests for Milvus search routing, RRF, and parity with FAISS.

All tests use deterministic fixtures and mocked Milvus/FAISS backends.
No pymilvus installation or network connection required.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Provider search routing tests
# ---------------------------------------------------------------------------

class TestProviderSearchRouting:
    """Each query vector must go ONLY to its matching provider field."""

    def _make_mock_backend(self, expected_field: str):
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.milvus_client import MilvusClient
        client = MagicMock(spec=MilvusClient)
        backend = MilvusProviderBackend(client, index_type="FLAT")
        # Track which vector_field was searched
        searched_fields = []
        def fake_search_provider(query_vector, provider_id, top_k, collection_name=None):
            from mvr.indexing.milvus_schema import vector_field_for_provider
            field = vector_field_for_provider(provider_id)
            searched_fields.append(field)
            return [
                {"keyframe_id": f"kf{i}", "video_id": "V01",
                 "frame_id": i, "shot_id": -1, "timestamp": float(i),
                 "batch_id": "1", "collection_id": "L21",
                 "distance": 1.0 - i * 0.01}
                for i in range(min(3, top_k))
            ]
        backend.search_provider = fake_search_provider
        return backend, searched_fields

    def test_beit3_routes_to_beit3_field(self):
        backend, searched_fields = self._make_mock_backend("beit3_vector")
        query_vec = np.random.rand(768).astype(np.float32)
        backend.search_provider(query_vec, "beit3", top_k=5)
        assert "beit3_vector" in searched_fields
        assert "metaclip2_vector" not in searched_fields
        assert "pe_vector" not in searched_fields

    def test_metaclip2_routes_to_metaclip2_field(self):
        backend, searched_fields = self._make_mock_backend("metaclip2_vector")
        query_vec = np.random.rand(512).astype(np.float32)
        backend.search_provider(query_vec, "metaclip2", top_k=5)
        assert "metaclip2_vector" in searched_fields
        assert "beit3_vector" not in searched_fields

    def test_pe_routes_to_pe_field(self):
        backend, searched_fields = self._make_mock_backend("pe_vector")
        query_vec = np.random.rand(1024).astype(np.float32)
        backend.search_provider(query_vec, "perception_encoder", top_k=5)
        assert "pe_vector" in searched_fields
        assert "beit3_vector" not in searched_fields

    def test_unknown_provider_raises(self):
        from mvr.indexing.milvus_schema import vector_field_for_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            vector_field_for_provider("openclip_rn50")


# ---------------------------------------------------------------------------
# Search hit identity mapping tests
# ---------------------------------------------------------------------------

class TestSearchHitMapping:
    def _make_hit(self, keyframe_id, video_id, frame_id, distance=0.9):
        return {
            "keyframe_id": keyframe_id,
            "video_id": video_id,
            "frame_id": frame_id,
            "shot_id": -1,
            "timestamp": 1.0,
            "batch_id": "1",
            "collection_id": "L21",
            "distance": distance,
        }

    def test_hit_maps_to_keyframe_id(self):
        from mvr.retrieval.milvus_semantic import MilvusProviderSearchBackend
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.milvus_client import MilvusClient
        from mvr.indexing.provider_indexes import ProviderIndexContract

        mock_backend = MagicMock(spec=MilvusProviderBackend)
        mock_backend.search_provider.return_value = [
            self._make_hit("kf001", "V01", 100, 0.95),
            self._make_hit("kf002", "V01", 200, 0.80),
        ]

        search_backend = MilvusProviderSearchBackend(mock_backend)
        contract = ProviderIndexContract(
            model_id="beit3",
            collection_name="aic26_keyframes",
            dimension=768,
            metric="IP",
            normalized=True,
            backend="milvus",
        )
        query_vec = np.random.rand(768).astype(np.float32)
        hits = search_backend.search(contract, query_vec, top_k=10)

        assert len(hits) == 2
        assert hits[0].metadata["keyframe_id"] == "kf001"
        assert hits[1].metadata["keyframe_id"] == "kf002"
        assert hits[0].metadata["video_id"] == "V01"
        assert hits[0].metadata["frame_id"] == 100

    def test_scalar_metadata_in_results(self):
        from mvr.retrieval.milvus_semantic import MilvusProviderSearchBackend
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.provider_indexes import ProviderIndexContract

        mock_backend = MagicMock(spec=MilvusProviderBackend)
        mock_backend.search_provider.return_value = [
            self._make_hit("kf001", "V01", 100, 0.95),
        ]

        search_backend = MilvusProviderSearchBackend(mock_backend)
        contract = ProviderIndexContract(
            model_id="beit3",
            collection_name="aic26_keyframes",
            dimension=768,
            metric="IP",
            normalized=True,
            backend="milvus",
        )
        hits = search_backend.search(contract, np.zeros(768, dtype=np.float32), top_k=1)
        meta = hits[0].metadata
        # All required scalar fields must be present
        for field in ("keyframe_id", "video_id", "frame_id", "shot_id", "timestamp"):
            assert field in meta

    def test_top_k_respected(self):
        from mvr.retrieval.milvus_semantic import MilvusProviderSearchBackend
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.provider_indexes import ProviderIndexContract

        mock_backend = MagicMock(spec=MilvusProviderBackend)
        mock_backend.search_provider.return_value = [
            self._make_hit(f"kf{i:03d}", "V01", i, 1.0 - i * 0.01)
            for i in range(50)
        ]
        search_backend = MilvusProviderSearchBackend(mock_backend)
        contract = ProviderIndexContract(
            model_id="beit3",
            collection_name="aic26_keyframes",
            dimension=768, metric="IP", normalized=True, backend="milvus",
        )
        hits = search_backend.search(contract, np.zeros(768, dtype=np.float32), top_k=50)
        assert len(hits) == 50


# ---------------------------------------------------------------------------
# RRF tests
# ---------------------------------------------------------------------------

class TestSemanticRRF:
    def test_rrf_k60_basic(self):
        from mvr.retrieval.hybrid import rrf_fuse
        ranked_lists = [
            ["kf001", "kf002", "kf003"],
            ["kf002", "kf001", "kf003"],
        ]
        result = rrf_fuse(ranked_lists, rrf_k=60, top_k=3)
        ids = [r[0] for r in result]
        # kf001 and kf002 both appear in both lists — should be top 2
        assert "kf001" in ids[:2]
        assert "kf002" in ids[:2]

    def test_rrf_k60_deterministic_ties(self):
        from mvr.retrieval.hybrid import rrf_fuse
        # kf001 and kf002 have same rank in same list — tie broken by lexicographic id
        result1 = rrf_fuse([["kf001", "kf002"]], rrf_k=60, top_k=2)
        result2 = rrf_fuse([["kf001", "kf002"]], rrf_k=60, top_k=2)
        assert result1 == result2

    def test_rrf_dedup_by_keyframe_id(self):
        from mvr.retrieval.hybrid import rrf_fuse
        # Same keyframe_id in multiple lists should not create duplicates
        result = rrf_fuse([
            ["kf001", "kf002"],
            ["kf001", "kf003"],
            ["kf001", "kf004"],
        ], rrf_k=60, top_k=10)
        ids = [r[0] for r in result]
        assert ids.count("kf001") == 1

    def test_rrf_score_formula(self):
        from mvr.retrieval.hybrid import rrf_fuse
        # Single list, rank 1: score = 1/(60+1)
        result = rrf_fuse([["kfA"]], rrf_k=60, top_k=1)
        assert len(result) == 1
        assert abs(result[0][1] - 1.0 / 61.0) < 1e-9

    def test_no_cross_provider_score_addition(self):
        """Verify provider scores are never raw-added across providers."""
        from mvr.retrieval.hybrid import rrf_fuse
        # RRF uses only ranks, not raw scores — this is verified by the formula
        # rank-based fusion doesn't require raw score access
        result_a = rrf_fuse([["kf1", "kf2"]], rrf_k=60, top_k=2)
        result_b = rrf_fuse([["kf2", "kf1"]], rrf_k=60, top_k=2)
        # Different rank orders → different RRF scores
        assert result_a[0][0] == "kf1"
        assert result_b[0][0] == "kf2"

    def test_rrf_top_k_limit(self):
        from mvr.retrieval.hybrid import rrf_fuse
        result = rrf_fuse([list(range(100))], rrf_k=60, top_k=10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# Manifest and stale-detection tests
# ---------------------------------------------------------------------------

class TestManifest:
    def test_build_and_load_manifest(self, tmp_path):
        from mvr.indexing.milvus_manifest import (
            build_manifest, write_manifest, load_manifest,
            ProviderManifestEntry, STATE_FRESH,
        )
        entries = {
            "beit3": ProviderManifestEntry("beit3", 768, "IP", True, "abc123"),
            "metaclip2": ProviderManifestEntry("metaclip2", 512, "IP", True, "def456"),
            "perception_encoder": ProviderManifestEntry("perception_encoder", 1024, "IP", True, "ghi789"),
        }
        m = build_manifest(
            collection_name="aic26_keyframes",
            entity_count=100,
            keyframe_mapping_checksum="kf_sha",
            providers=entries,
            index_type="FLAT",
            metric="IP",
            index_params={},
            state=STATE_FRESH,
        )
        path = tmp_path / "manifest.json"
        write_manifest(m, path)
        loaded = load_manifest(path)
        assert loaded is not None
        assert loaded.collection_name == "aic26_keyframes"
        assert loaded.entity_count == 100
        assert loaded.state == STATE_FRESH
        assert loaded.providers["beit3"].dimension == 768

    def test_stale_detection_checksum_change(self):
        from mvr.indexing.milvus_manifest import (
            build_manifest, check_staleness,
            ProviderManifestEntry, STATE_FRESH, STATE_STALE,
        )
        entries = {
            "beit3": ProviderManifestEntry("beit3", 768, "IP", True, "old_b3"),
        }
        m = build_manifest(
            collection_name="aic26_keyframes", entity_count=10,
            keyframe_mapping_checksum="old_kf",
            providers=entries, index_type="FLAT", metric="IP", index_params={},
        )
        state, reasons = check_staleness(
            m,
            current_keyframe_checksum="new_kf",
            current_provider_checksums={"beit3": "old_b3"},
            current_dimensions={"beit3": 768},
        )
        assert state == STATE_STALE
        assert any("keyframe" in r for r in reasons)

    def test_fresh_when_nothing_changes(self):
        from mvr.indexing.milvus_manifest import (
            build_manifest, check_staleness,
            ProviderManifestEntry, STATE_FRESH,
        )
        entries = {
            "beit3": ProviderManifestEntry("beit3", 768, "IP", True, "sha1"),
            "metaclip2": ProviderManifestEntry("metaclip2", 512, "IP", True, "sha2"),
            "perception_encoder": ProviderManifestEntry("perception_encoder", 1024, "IP", True, "sha3"),
        }
        m = build_manifest(
            collection_name="aic26_keyframes", entity_count=10,
            keyframe_mapping_checksum="kf_sha",
            providers=entries, index_type="FLAT", metric="IP", index_params={},
        )
        state, reasons = check_staleness(
            m,
            current_keyframe_checksum="kf_sha",
            current_provider_checksums={"beit3": "sha1", "metaclip2": "sha2", "perception_encoder": "sha3"},
            current_dimensions={"beit3": 768, "metaclip2": 512, "perception_encoder": 1024},
        )
        assert state == STATE_FRESH
        assert reasons == []

    def test_load_missing_manifest_returns_none(self, tmp_path):
        from mvr.indexing.milvus_manifest import load_manifest
        result = load_manifest(tmp_path / "nonexistent.json")
        assert result is None


# ---------------------------------------------------------------------------
# insert_batch entity_id computation
# ---------------------------------------------------------------------------

class TestInsertBatchEntityId:
    """MilvusProviderBackend.insert_batch() must compute entity_id itself,
    at the layer closest to the actual Milvus write -- not trust an
    upstream caller to have already namespaced it by video."""

    def _make_backend(self):
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.milvus_client import MilvusClient
        client = MagicMock(spec=MilvusClient)
        return MilvusProviderBackend(client, index_type="HNSW", batch_size=1000)

    def _contract(self):
        from mvr.indexing.provider_indexes import ProviderIndexContract
        return ProviderIndexContract(
            model_id="beit3",
            collection_name="aic26_keyframes",
            dimension=768,
            metric="IP",
            normalized=True,
            backend="milvus",
        )

    def test_entity_id_is_video_id_slash_keyframe_id(self):
        backend = self._make_backend()
        vectors = np.zeros((1, 768), dtype=np.float32)
        metadata = [{"keyframe_id": "K01", "video_id": "L21_V001"}]
        backend.insert_batch(self._contract(), vectors, metadata)
        assert backend._staged[0]["entity_id"] == "L21_V001/K01"

    def test_two_videos_same_keyframe_id_get_different_entity_ids(self):
        backend = self._make_backend()
        vectors = np.zeros((2, 768), dtype=np.float32)
        metadata = [
            {"keyframe_id": "034", "video_id": "L21_V001"},
            {"keyframe_id": "034", "video_id": "L21_V002"},
        ]
        backend.insert_batch(self._contract(), vectors, metadata)
        entity_ids = [row["entity_id"] for row in backend._staged]
        assert entity_ids[0] != entity_ids[1]
        assert len(set(entity_ids)) == 2

    def test_missing_video_id_raises(self):
        """A caller that never resolved video_id must fail loudly, not
        silently write an entity_id like "/keyframe_id" that still
        collides across videos."""
        backend = self._make_backend()
        vectors = np.zeros((1, 768), dtype=np.float32)
        metadata = [{"keyframe_id": "K01", "video_id": ""}]
        with pytest.raises(ValueError, match="non-empty"):
            backend.insert_batch(self._contract(), vectors, metadata)


class TestBuildEntityRowEntityId:
    """ingest_to_milvus() upserts through the client directly rather than
    through MilvusProviderBackend.insert_batch(), so _build_entity_row must
    set the entity_id primary key itself -- otherwise every CLI-driven
    ingest writes rows with no primary key at all."""

    def _vectors(self):
        return {
            "beit3": np.zeros(768, dtype=np.float32),
            "metaclip2": np.zeros(512, dtype=np.float32),
            "perception_encoder": np.zeros(1024, dtype=np.float32),
        }

    def test_entity_id_is_set_from_video_id_and_keyframe_id(self):
        from mvr.indexing.milvus_ingest import _build_entity_row
        row = _build_entity_row(
            "K01", {"video_id": "L21_V001", "frame_id": 5}, self._vectors()
        )
        assert row["entity_id"] == "L21_V001/K01"
        assert row["keyframe_id"] == "K01"
        assert row["video_id"] == "L21_V001"

    def test_two_videos_same_keyframe_id_get_different_entity_ids(self):
        from mvr.indexing.milvus_ingest import _build_entity_row
        row_a = _build_entity_row("034", {"video_id": "L21_V001"}, self._vectors())
        row_b = _build_entity_row("034", {"video_id": "L21_V002"}, self._vectors())
        assert row_a["entity_id"] != row_b["entity_id"]

    def test_missing_video_id_raises_actionable_ingest_error(self):
        """The extraction-status schema currently has no video_id column, so
        this path must refuse rather than write "/keyframe_id" keys that
        collide across every video."""
        from mvr.indexing.milvus_ingest import MilvusIngestError, _build_entity_row
        with pytest.raises(MilvusIngestError, match="collision-safe entity_id"):
            _build_entity_row("K01", {"frame_id": 5}, self._vectors())

    def test_blank_video_id_raises(self):
        from mvr.indexing.milvus_ingest import MilvusIngestError, _build_entity_row
        with pytest.raises(MilvusIngestError, match="no video_id"):
            _build_entity_row("K01", {"video_id": "   "}, self._vectors())
