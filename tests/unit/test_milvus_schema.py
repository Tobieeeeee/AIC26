"""Tests for Milvus collection schema, eligibility gate, and provider routing.

All tests run without pymilvus installed (mocked) and without any network calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Schema and field tests
# ---------------------------------------------------------------------------

class TestMilvusSchema:
    def test_scalar_fields(self):
        from mvr.indexing.milvus_schema import SCALAR_FIELDS, FIELD_KEYFRAME_ID
        assert FIELD_KEYFRAME_ID in SCALAR_FIELDS
        for f in ("video_id", "frame_id", "shot_id", "timestamp", "batch_id", "collection_id"):
            assert f in SCALAR_FIELDS

    def test_three_vector_fields(self):
        from mvr.indexing.milvus_schema import VECTOR_FIELDS
        assert len(VECTOR_FIELDS) == 3
        assert "beit3_vector" in VECTOR_FIELDS
        assert "metaclip2_vector" in VECTOR_FIELDS
        assert "pe_vector" in VECTOR_FIELDS

    def test_canonical_metric(self):
        from mvr.indexing.milvus_client import CANONICAL_METRIC
        assert CANONICAL_METRIC == "IP"

    def test_expected_dimensions(self):
        from mvr.indexing.milvus_schema import EXPECTED_DIMENSIONS
        assert EXPECTED_DIMENSIONS["beit3"] == 768
        assert EXPECTED_DIMENSIONS["metaclip2"] == 512
        assert EXPECTED_DIMENSIONS["perception_encoder"] == 1024

    def test_provider_to_vector_field_mapping(self):
        from mvr.indexing.milvus_schema import PROVIDER_TO_VECTOR_FIELD
        assert PROVIDER_TO_VECTOR_FIELD["beit3"] == "beit3_vector"
        assert PROVIDER_TO_VECTOR_FIELD["metaclip2"] == "metaclip2_vector"
        assert PROVIDER_TO_VECTOR_FIELD["perception_encoder"] == "pe_vector"

    def test_no_cross_provider_mapping(self):
        from mvr.indexing.milvus_schema import vector_field_for_provider
        # Each provider routes to its own field only
        assert vector_field_for_provider("beit3") == "beit3_vector"
        assert vector_field_for_provider("metaclip2") == "metaclip2_vector"
        assert vector_field_for_provider("perception_encoder") == "pe_vector"

    def test_unknown_provider_raises(self):
        from mvr.indexing.milvus_schema import vector_field_for_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            vector_field_for_provider("openclip")

    def test_supported_index_types(self):
        from mvr.indexing.milvus_schema import SUPPORTED_INDEX_TYPES
        assert "FLAT" in SUPPORTED_INDEX_TYPES
        assert "HNSW" in SUPPORTED_INDEX_TYPES
        assert "IVF_FLAT" in SUPPORTED_INDEX_TYPES

    def test_collection_name(self):
        from mvr.indexing.milvus_schema import COLLECTION_NAME
        assert COLLECTION_NAME == "aic26_keyframes"

    def test_search_output_fields(self):
        from mvr.indexing.milvus_schema import SEARCH_OUTPUT_FIELDS
        for f in ("keyframe_id", "video_id", "frame_id", "shot_id", "timestamp"):
            assert f in SEARCH_OUTPUT_FIELDS


# ---------------------------------------------------------------------------
# Dimension validation tests
# ---------------------------------------------------------------------------

class TestDimensionValidation:
    def test_valid_dimensions(self):
        from mvr.indexing.milvus_schema import validate_provider_dimensions, EXPECTED_DIMENSIONS
        errors = validate_provider_dimensions(dict(EXPECTED_DIMENSIONS))
        assert errors == []

    def test_wrong_beit3_dimension(self):
        from mvr.indexing.milvus_schema import validate_provider_dimensions, EXPECTED_DIMENSIONS
        dims = dict(EXPECTED_DIMENSIONS)
        dims["beit3"] = 512  # wrong
        errors = validate_provider_dimensions(dims)
        assert any("beit3" in e and "512" in e for e in errors)

    def test_missing_provider(self):
        from mvr.indexing.milvus_schema import validate_provider_dimensions
        errors = validate_provider_dimensions({"beit3": 768})
        assert any("metaclip2" in e for e in errors)
        assert any("perception_encoder" in e for e in errors)


# ---------------------------------------------------------------------------
# Milvus client config tests
# ---------------------------------------------------------------------------

class TestMilvusConfig:
    def test_from_env_defaults(self, monkeypatch):
        from mvr.indexing.milvus_client import MilvusConfig
        monkeypatch.delenv("MILVUS_URI", raising=False)
        monkeypatch.delenv("MILVUS_HOST", raising=False)
        monkeypatch.delenv("MILVUS_PORT", raising=False)
        monkeypatch.delenv("MILVUS_TOKEN", raising=False)
        monkeypatch.delenv("MILVUS_DATABASE", raising=False)
        monkeypatch.delenv("MILVUS_COLLECTION", raising=False)
        cfg = MilvusConfig.from_env()
        assert cfg.host == "localhost"
        assert cfg.port == 19530
        assert cfg.database == "default"
        assert cfg.collection == "aic26_keyframes"
        assert cfg.token == ""

    def test_effective_uri_from_uri_env(self, monkeypatch):
        from mvr.indexing.milvus_client import MilvusConfig
        monkeypatch.setenv("MILVUS_URI", "http://milvus.prod:19530")
        cfg = MilvusConfig.from_env()
        assert cfg.effective_uri == "http://milvus.prod:19530"

    def test_effective_uri_from_host_port(self, monkeypatch):
        from mvr.indexing.milvus_client import MilvusConfig
        monkeypatch.delenv("MILVUS_URI", raising=False)
        monkeypatch.setenv("MILVUS_HOST", "192.168.1.1")
        monkeypatch.setenv("MILVUS_PORT", "19531")
        cfg = MilvusConfig.from_env()
        assert cfg.effective_uri == "http://192.168.1.1:19531"

    def test_invalid_port_raises(self, monkeypatch):
        from mvr.indexing.milvus_client import MilvusConfig, MilvusConfigError
        monkeypatch.delenv("MILVUS_URI", raising=False)
        monkeypatch.setenv("MILVUS_PORT", "notanumber")
        with pytest.raises(MilvusConfigError, match="MILVUS_PORT"):
            MilvusConfig.from_env()


# ---------------------------------------------------------------------------
# Eligibility gate tests (require polars)
# ---------------------------------------------------------------------------

polars = pytest.importorskip("polars")


class TestEligibilityGate:
    def _make_status_parquet(self, tmp_path, rows):
        df = polars.DataFrame(rows)
        path = tmp_path / "status.parquet"
        df.write_parquet(path)
        return path

    def test_all_done_eligible(self, tmp_path):
        from mvr.indexing.milvus_ingest import discover_eligible_keyframes
        path = self._make_status_parquet(tmp_path, {
            "keyframe_id": ["kf1", "kf2"],
            "beit3_status": ["DONE", "DONE"],
            "metaclip2_status": ["DONE", "DONE"],
            "pe_status": ["DONE", "DONE"],
        })
        df = discover_eligible_keyframes(path)
        assert df.height == 2

    def test_missing_provider_status_ineligible(self, tmp_path):
        from mvr.indexing.milvus_ingest import discover_eligible_keyframes
        path = self._make_status_parquet(tmp_path, {
            "keyframe_id": ["kf1", "kf2"],
            "beit3_status": ["DONE", "TODO"],
            "metaclip2_status": ["DONE", "DONE"],
            "pe_status": ["DONE", "DONE"],
        })
        df = discover_eligible_keyframes(path)
        assert df.height == 1
        assert df["keyframe_id"].to_list() == ["kf1"]

    def test_all_todo_zero_eligible(self, tmp_path):
        from mvr.indexing.milvus_ingest import discover_eligible_keyframes
        path = self._make_status_parquet(tmp_path, {
            "keyframe_id": ["kf1"],
            "beit3_status": ["TODO"],
            "metaclip2_status": ["TODO"],
            "pe_status": ["TODO"],
        })
        df = discover_eligible_keyframes(path)
        assert df.height == 0

    def test_failed_status_ineligible(self, tmp_path):
        from mvr.indexing.milvus_ingest import discover_eligible_keyframes
        path = self._make_status_parquet(tmp_path, {
            "keyframe_id": ["kf1"],
            "beit3_status": ["FAILED"],
            "metaclip2_status": ["DONE"],
            "pe_status": ["DONE"],
        })
        df = discover_eligible_keyframes(path)
        assert df.height == 0


# ---------------------------------------------------------------------------
# Vector validation tests
# ---------------------------------------------------------------------------

class TestVectorValidation:
    def test_finite_vector_valid(self):
        from mvr.indexing.milvus_ingest import _is_finite_vector
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assert _is_finite_vector(vec)

    def test_nan_vector_invalid(self):
        from mvr.indexing.milvus_ingest import _is_finite_vector
        vec = np.array([0.1, float("nan"), 0.3], dtype=np.float32)
        assert not _is_finite_vector(vec)

    def test_inf_vector_invalid(self):
        from mvr.indexing.milvus_ingest import _is_finite_vector
        vec = np.array([0.1, float("inf"), 0.3], dtype=np.float32)
        assert not _is_finite_vector(vec)

    def test_empty_vector_invalid(self):
        from mvr.indexing.milvus_ingest import _is_finite_vector
        vec = np.array([], dtype=np.float32)
        assert not _is_finite_vector(vec)


# ---------------------------------------------------------------------------
# MilvusBackend index type tests (no pymilvus needed)
# ---------------------------------------------------------------------------

class TestMilvusBackendInit:
    def test_valid_index_types(self):
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.milvus_client import MilvusClient
        client = MagicMock(spec=MilvusClient)
        for idx_type in ("FLAT", "HNSW", "IVF_FLAT"):
            backend = MilvusProviderBackend(client, index_type=idx_type)
            assert backend.index_type == idx_type

    def test_invalid_index_type_raises(self):
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.milvus_client import MilvusClient
        client = MagicMock(spec=MilvusClient)
        with pytest.raises(ValueError, match="Unsupported index type"):
            MilvusProviderBackend(client, index_type="SCANN")

    def test_provider_routes_to_correct_field(self):
        from mvr.indexing.milvus_schema import vector_field_for_provider
        # These must route correctly for search isolation
        assert vector_field_for_provider("beit3") != vector_field_for_provider("metaclip2")
        assert vector_field_for_provider("metaclip2") != vector_field_for_provider("perception_encoder")
        assert vector_field_for_provider("beit3") != vector_field_for_provider("perception_encoder")
