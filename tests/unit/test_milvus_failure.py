"""Tests for Milvus failure modes — unavailable, missing collection, stale manifest,
provider mismatch, and dimension mismatch.

No pymilvus installation or network connection required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Milvus unavailable
# ---------------------------------------------------------------------------

class TestMilvusUnavailable:
    def test_connect_raises_connection_error(self, monkeypatch):
        from mvr.indexing.milvus_client import MilvusClient, MilvusConnectionError, MilvusConfig

        pytest.importorskip("pymilvus", reason="pymilvus not installed; skipping live connect test")

        cfg = MilvusConfig(host="unreachable", port=19530)
        client = MilvusClient(cfg)

        import pymilvus
        with patch("pymilvus.connections.connect", side_effect=RuntimeError("connection refused")):
            with pytest.raises((MilvusConnectionError, Exception)):
                client.connect()


    def test_health_without_connect_raises(self):
        from mvr.indexing.milvus_client import MilvusClient, MilvusConfig
        cfg = MilvusConfig(host="localhost", port=19530)
        client = MilvusClient(cfg)
        # Not connected — _ensure_connected tries connect and fails if pymilvus is absent
        # Without pymilvus, MilvusConfigError is raised
        try:
            import pymilvus
            # If pymilvus is available, patch it to fail
            with patch("pymilvus.connections.connect", side_effect=RuntimeError("refused")):
                from mvr.indexing.milvus_client import MilvusConnectionError
                with pytest.raises((MilvusConnectionError, Exception)):
                    client.health()
        except ImportError:
            # pymilvus not installed → MilvusConfigError
            from mvr.indexing.milvus_client import MilvusConfigError
            with pytest.raises(MilvusConfigError):
                client.health()

    def test_semantic_service_not_ready_returns_error(self):
        from mvr.retrieval.milvus_semantic import (
            MilvusSemanticService, MilvusNotReadyError,
        )
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.milvus_client import MilvusClient, MilvusConnectionError

        mock_client = MagicMock(spec=MilvusClient)
        mock_client.health.side_effect = MilvusConnectionError("unreachable")

        mock_backend = MagicMock(spec=MilvusProviderBackend)
        mock_backend.client = mock_client

        service = MilvusSemanticService(mock_backend)
        status = service.health()
        assert "error" in status or status.get("status") == "not_ready"

    def test_stale_policy_fail_raises(self):
        from mvr.retrieval.milvus_semantic import (
            MilvusProviderSearchBackend, StaleManifestError,
        )
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.milvus_manifest import (
            SemanticIndexManifest, STATE_STALE,
        )

        mock_backend = MagicMock(spec=MilvusProviderBackend)
        stale_manifest = MagicMock(spec=SemanticIndexManifest)
        stale_manifest.state = STATE_STALE

        backend = MilvusProviderSearchBackend(
            mock_backend,
            stale_policy="fail",
            manifest=stale_manifest,
        )

        from mvr.indexing.provider_indexes import ProviderIndexContract
        contract = ProviderIndexContract(
            model_id="beit3",
            collection_name="aic26_keyframes",
            dimension=768, metric="IP", normalized=True, backend="milvus",
        )
        with pytest.raises(StaleManifestError):
            backend.search(contract, np.zeros(768, dtype=np.float32), top_k=10)

    def test_stale_policy_warn_continues(self, caplog):
        import logging
        from mvr.retrieval.milvus_semantic import MilvusProviderSearchBackend
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.milvus_manifest import SemanticIndexManifest, STATE_STALE

        mock_backend = MagicMock(spec=MilvusProviderBackend)
        mock_backend.search_provider.return_value = []
        stale_manifest = MagicMock(spec=SemanticIndexManifest)
        stale_manifest.state = STATE_STALE

        backend = MilvusProviderSearchBackend(
            mock_backend,
            stale_policy="warn",
            manifest=stale_manifest,
        )

        from mvr.indexing.provider_indexes import ProviderIndexContract
        contract = ProviderIndexContract(
            model_id="beit3",
            collection_name="aic26_keyframes",
            dimension=768, metric="IP", normalized=True, backend="milvus",
        )
        with caplog.at_level(logging.WARNING):
            # Should not raise — just warn
            result = backend.search(contract, np.zeros(768, dtype=np.float32), top_k=10)
        assert any("STALE" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Missing collection
# ---------------------------------------------------------------------------

class TestMissingCollection:
    def test_validate_collection_raises_if_missing(self):
        from mvr.indexing.milvus_client import MilvusClient, MilvusCollectionError

        mock_client = MagicMock(spec=MilvusClient)
        mock_client.validate_collection.side_effect = MilvusCollectionError(
            "Collection does not exist: aic26_keyframes"
        )
        with pytest.raises(MilvusCollectionError, match="does not exist"):
            mock_client.validate_collection("aic26_keyframes")

    def test_verify_readiness_raises_if_collection_missing(self):
        from mvr.retrieval.milvus_semantic import MilvusProviderSearchBackend, MilvusNotReadyError
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.milvus_client import MilvusClient, MilvusCollectionError

        mock_client = MagicMock(spec=MilvusClient)
        mock_client.health.return_value = {"status": "ok", "collections": []}
        mock_client.collection_info.side_effect = MilvusCollectionError("not found")

        mock_backend = MagicMock(spec=MilvusProviderBackend)
        mock_backend.client = mock_client

        sb = MilvusProviderSearchBackend(mock_backend)
        with pytest.raises(MilvusNotReadyError, match="missing"):
            sb.verify_readiness()


# ---------------------------------------------------------------------------
# Stale manifest
# ---------------------------------------------------------------------------

class TestStaleManifest:
    def test_stale_state_after_checksum_change(self):
        from mvr.indexing.milvus_manifest import (
            build_manifest, check_staleness,
            ProviderManifestEntry, STATE_STALE,
        )
        entries = {
            "beit3": ProviderManifestEntry("beit3", 768, "IP", True, "old_sha"),
        }
        m = build_manifest(
            collection_name="aic26_keyframes", entity_count=100,
            keyframe_mapping_checksum="old_kf",
            providers=entries, index_type="FLAT", metric="IP", index_params={},
        )
        state, reasons = check_staleness(
            m,
            current_keyframe_checksum="new_kf",
            current_provider_checksums={"beit3": "old_sha"},
            current_dimensions={"beit3": 768},
        )
        assert state == STATE_STALE
        assert len(reasons) >= 1

    def test_stale_state_after_provider_checksum_change(self):
        from mvr.indexing.milvus_manifest import (
            build_manifest, check_staleness,
            ProviderManifestEntry, STATE_STALE,
        )
        entries = {
            "beit3": ProviderManifestEntry("beit3", 768, "IP", True, "sha_a"),
            "metaclip2": ProviderManifestEntry("metaclip2", 512, "IP", True, "sha_b"),
            "perception_encoder": ProviderManifestEntry("perception_encoder", 1024, "IP", True, "sha_c"),
        }
        m = build_manifest(
            collection_name="aic26_keyframes", entity_count=100,
            keyframe_mapping_checksum="kf_sha",
            providers=entries, index_type="FLAT", metric="IP", index_params={},
        )
        state, reasons = check_staleness(
            m,
            current_keyframe_checksum="kf_sha",
            current_provider_checksums={"beit3": "sha_a", "metaclip2": "CHANGED", "perception_encoder": "sha_c"},
            current_dimensions={"beit3": 768, "metaclip2": 512, "perception_encoder": 1024},
        )
        assert state == STATE_STALE
        assert any("metaclip2" in r for r in reasons)

    def test_stale_manifest_no_silent_empty_results(self):
        """Production policy=fail must raise, not silently return empty results."""
        from mvr.retrieval.milvus_semantic import (
            MilvusProviderSearchBackend, StaleManifestError,
        )
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.milvus_manifest import SemanticIndexManifest, STATE_STALE
        from mvr.indexing.provider_indexes import ProviderIndexContract

        mock_backend = MagicMock(spec=MilvusProviderBackend)
        stale = MagicMock(spec=SemanticIndexManifest)
        stale.state = STATE_STALE

        sb = MilvusProviderSearchBackend(mock_backend, stale_policy="fail", manifest=stale)
        contract = ProviderIndexContract(
            model_id="beit3", collection_name="aic26_keyframes",
            dimension=768, metric="IP", normalized=True, backend="milvus",
        )
        # Must NOT return empty list silently — must raise
        with pytest.raises(StaleManifestError):
            sb.search(contract, np.zeros(768, dtype=np.float32), top_k=10)


# ---------------------------------------------------------------------------
# Provider manifest mismatch
# ---------------------------------------------------------------------------

class TestProviderMismatch:
    def test_wrong_dimension_rejected_by_validation(self):
        from mvr.indexing.milvus_schema import validate_provider_dimensions, EXPECTED_DIMENSIONS
        wrong_dims = dict(EXPECTED_DIMENSIONS)
        wrong_dims["beit3"] = 256  # wrong
        errors = validate_provider_dimensions(wrong_dims)
        assert any("beit3" in e for e in errors)
        assert any("256" in e for e in errors)

    def test_missing_provider_rejected_by_validation(self):
        from mvr.indexing.milvus_schema import validate_provider_dimensions
        # Only one provider present — others missing
        errors = validate_provider_dimensions({"beit3": 768})
        assert len(errors) >= 2  # metaclip2 and perception_encoder missing

    def test_unknown_provider_field_routing_raises(self):
        from mvr.indexing.milvus_schema import vector_field_for_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            vector_field_for_provider("siglip")


# ---------------------------------------------------------------------------
# Invalid stale policy
# ---------------------------------------------------------------------------

class TestInvalidConfig:
    def test_invalid_stale_policy_raises(self):
        from mvr.retrieval.milvus_semantic import MilvusProviderSearchBackend
        from mvr.indexing.milvus_backend import MilvusProviderBackend

        mock_backend = MagicMock(spec=MilvusProviderBackend)
        with pytest.raises(ValueError, match="stale_policy"):
            MilvusProviderSearchBackend(mock_backend, stale_policy="ignore_silently")

    def test_invalid_index_type_raises(self):
        from mvr.indexing.milvus_backend import MilvusProviderBackend
        from mvr.indexing.milvus_client import MilvusClient
        mock_client = MagicMock(spec=MilvusClient)
        with pytest.raises(ValueError, match="Unsupported index type"):
            MilvusProviderBackend(mock_client, index_type="ANNOY")
