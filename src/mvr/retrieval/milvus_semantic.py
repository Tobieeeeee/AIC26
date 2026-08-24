"""Milvus semantic retrieval service.

Implements ProviderSearchBackend Protocol so it plugs into the existing
IndependentSemanticRetriever without any interface changes.

Provider-specific routing contract:
    BEiT-3 query vector  → search beit3_vector field only
    MetaCLIP-2 vector    → search metaclip2_vector field only
    PE-Core-L vector     → search pe_vector field only

No cross-provider raw score addition. No per-query R2 or Parquet reads.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from mvr.indexing.milvus_backend import MilvusProviderBackend
from mvr.indexing.milvus_client import (
    MilvusClient,
    MilvusCollectionError,
    MilvusConnectionError,
)
from mvr.indexing.milvus_manifest import (
    STATE_STALE,
    SemanticIndexManifest,
    load_manifest,
)
from mvr.indexing.milvus_schema import (
    COLLECTION_NAME,
    SEARCH_OUTPUT_FIELDS,
    vector_field_for_provider,
)
from mvr.indexing.provider_indexes import ProviderIndexContract
from mvr.retrieval.provider_semantic import BackendSearchHit

LOGGER = logging.getLogger(__name__)


class MilvusSemanticError(RuntimeError):
    """Raised when Milvus semantic retrieval cannot proceed safely."""


class MilvusNotReadyError(MilvusSemanticError):
    """Milvus is unavailable or the collection is not ready."""


class StaleManifestError(MilvusSemanticError):
    """The semantic index manifest is STALE — serving stale results is not allowed."""


class MilvusProviderSearchBackend:
    """Implements ProviderSearchBackend Protocol backed by Milvus.

    Plugs into IndependentSemanticRetriever in provider_semantic.py.

    Each call to search() routes query_vector to its provider-specific
    vector field in Milvus. No cross-provider contamination.
    """

    def __init__(
        self,
        backend: MilvusProviderBackend,
        *,
        collection_name: str = COLLECTION_NAME,
        stale_policy: str = "fail",
        manifest: SemanticIndexManifest | None = None,
    ) -> None:
        """
        stale_policy:
            'fail'  — raise StaleManifestError if manifest is STALE (default, production)
            'warn'  — log a warning and continue
            'allow' — silently allow stale serving (reference/debug only)
        """
        if stale_policy not in {"fail", "warn", "allow"}:
            raise ValueError(f"Invalid stale_policy: {stale_policy!r}")
        self.backend = backend
        self.collection_name = collection_name
        self.stale_policy = stale_policy
        self.manifest = manifest
        self._readiness_verified = False

    def _check_manifest_freshness(self) -> None:
        if self.manifest is None:
            return
        if self.manifest.state == STATE_STALE:
            msg = (
                f"Semantic index manifest is STALE for collection "
                f"{self.collection_name!r}. Do not silently serve stale results."
            )
            if self.stale_policy == "fail":
                raise StaleManifestError(msg)
            elif self.stale_policy == "warn":
                LOGGER.warning(msg)

    def verify_readiness(self) -> dict[str, Any]:
        """Verify Milvus is reachable and the collection is ready.

        Checks:
          - Milvus reachable
          - collection exists
          - schema has all three vector fields
          - collection is loaded / index is ready
          - manifest state is compatible

        Returns a status dict. Raises MilvusNotReadyError if not ready.
        """
        try:
            health = self.backend.client.health()
        except MilvusConnectionError as exc:
            raise MilvusNotReadyError(f"Milvus unreachable: {exc}") from exc

        try:
            info = self.backend.client.collection_info(self.collection_name)
        except MilvusCollectionError as exc:
            raise MilvusNotReadyError(
                f"Collection {self.collection_name!r} missing: {exc}"
            ) from exc

        from mvr.indexing.milvus_schema import VECTOR_FIELDS
        missing = set(VECTOR_FIELDS) - set(info["fields"])
        if missing:
            raise MilvusNotReadyError(
                f"Collection {self.collection_name!r} missing vector fields: "
                f"{sorted(missing)}"
            )

        manifest_ok = True
        if self.manifest is not None and self.manifest.state == STATE_STALE:
            manifest_ok = False

        self._readiness_verified = True
        return {
            "milvus_reachable": True,
            "collection": self.collection_name,
            "collection_exists": True,
            "schema_valid": True,
            "entity_count": info["num_entities"],
            "indexes": info["indexes"],
            "manifest_ok": manifest_ok,
            "manifest_state": self.manifest.state if self.manifest else "UNKNOWN",
        }

    def search(
        self,
        contract: ProviderIndexContract,
        vector: np.ndarray,
        top_k: int,
    ) -> Sequence[BackendSearchHit]:
        """Search Milvus for the given provider's vector field.

        Routes vector ONLY to its matching Milvus field.
        Returns BackendSearchHit sequence compatible with IndependentSemanticRetriever.
        """
        self._check_manifest_freshness()

        provider_id = contract.model_id
        hits_raw = self.backend.search_provider(
            query_vector=vector.astype(np.float32),
            provider_id=provider_id,
            top_k=top_k,
            collection_name=self.collection_name,
        )

        return [
            BackendSearchHit(
                score=float(hit["distance"]),
                metadata={
                    "keyframe_id": hit.get("keyframe_id", ""),
                    "video_id": hit.get("video_id", ""),
                    "frame_id": hit.get("frame_id", 0),
                    "shot_id": hit.get("shot_id", -1),
                    "timestamp": hit.get("timestamp", 0.0),
                    "batch_id": hit.get("batch_id", ""),
                    "collection_id": hit.get("collection_id", ""),
                },
            )
            for hit in hits_raw
        ]


class MilvusSemanticService:
    """High-level semantic service: query → IndependentSemanticResponse.

    This is the production API surface. Uses Python RRF k=60 via the
    existing semantic_rrf.py — NOT Milvus-native RRF in v1.
    """

    def __init__(
        self,
        backend: MilvusProviderBackend,
        *,
        collection_name: str = COLLECTION_NAME,
        index_type: str = "FLAT",
        stale_policy: str = "fail",
        manifest_path: str | None = None,
    ) -> None:
        manifest: SemanticIndexManifest | None = None
        if manifest_path:
            from pathlib import Path
            manifest = load_manifest(Path(manifest_path))

        self._search_backend = MilvusProviderSearchBackend(
            backend=backend,
            collection_name=collection_name,
            stale_policy=stale_policy,
            manifest=manifest,
        )
        self.collection_name = collection_name
        self.index_type = index_type

    @classmethod
    def from_env(
        cls,
        *,
        index_type: str = "FLAT",
        stale_policy: str = "fail",
        manifest_path: str | None = None,
    ) -> "MilvusSemanticService":
        """Construct from environment variables."""
        client = MilvusClient()
        backend = MilvusProviderBackend(
            client,
            index_type=index_type,
            collection_name=COLLECTION_NAME,
        )
        return cls(
            backend,
            collection_name=COLLECTION_NAME,
            index_type=index_type,
            stale_policy=stale_policy,
            manifest_path=manifest_path,
        )

    def health(self) -> dict[str, Any]:
        """Return service readiness status."""
        try:
            return self._search_backend.verify_readiness()
        except MilvusSemanticError as exc:
            return {"status": "not_ready", "error": str(exc)}

    @property
    def provider_search_backend(self) -> MilvusProviderSearchBackend:
        """Expose backend for use with IndependentSemanticRetriever."""
        return self._search_backend
