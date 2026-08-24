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
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mvr.indexing.milvus_backend import MilvusProviderBackend
from mvr.indexing.milvus_client import (
    MilvusClient,
    MilvusCollectionError,
    MilvusConfigError,
    MilvusConnectionError,
)
from mvr.indexing.milvus_manifest import (
    STATE_STALE,
    SemanticIndexManifest,
    load_manifest,
)
from mvr.indexing.milvus_schema import (
    COLLECTION_NAME,
    EXPECTED_DIMENSIONS,
    SEARCH_OUTPUT_FIELDS,
    validate_provider_dimensions,
    vector_field_for_provider,
)
from mvr.indexing.provider_indexes import ProviderIndexContract
from mvr.retrieval.provider_semantic import BackendSearchHit, IndependentSemanticRetriever
from mvr.retrieval.semantic_rrf import fuse_independent_response

LOGGER = logging.getLogger(__name__)

PRODUCTION_PROVIDERS = ("beit3", "metaclip2", "perception_encoder")
DEFAULT_RRF_K = 60
DEFAULT_MAX_TOP_K = 1500


class MilvusSemanticError(RuntimeError):
    """Raised when Milvus semantic retrieval cannot proceed safely."""


class MilvusNotReadyError(MilvusSemanticError):
    """Milvus is unavailable or the collection is not ready."""


class StaleManifestError(MilvusSemanticError):
    """The semantic index manifest is STALE — serving stale results is not allowed."""


@dataclass(frozen=True)
class MilvusSemanticHit:
    """API-neutral fused semantic hit with canonical identity intact."""

    rank: int
    keyframe_id: str
    video_id: str
    frame_id: int
    timestamp: float
    score: float
    provider_ranks: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "keyframe_id": self.keyframe_id,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "frame_idx": self.frame_id,
            "timestamp": self.timestamp,
            "score": self.score,
            "provider_ranks": dict(sorted(self.provider_ranks.items())),
        }


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
        except (MilvusConfigError, MilvusConnectionError) as exc:
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

        dimensions = info.get("dimensions", {})
        if dimensions:
            expected_by_field = {
                vector_field_for_provider(provider_id): dimension
                for provider_id, dimension in EXPECTED_DIMENSIONS.items()
            }
            wrong_dimensions = {
                field: (expected, dimensions.get(field))
                for field, expected in expected_by_field.items()
                if dimensions.get(field) != expected
            }
            if wrong_dimensions:
                raise MilvusNotReadyError(
                    f"Collection {self.collection_name!r} vector dimensions do not "
                    f"match provider contracts: {wrong_dimensions}"
                )

        indexed_fields = {
            str(index.get("field")) for index in info.get("indexes", [])
        }
        missing_indexes = set(VECTOR_FIELDS) - indexed_fields
        if missing_indexes:
            raise MilvusNotReadyError(
                f"Collection {self.collection_name!r} missing vector indexes: "
                f"{sorted(missing_indexes)}"
            )
        if info.get("loaded") is False:
            raise MilvusNotReadyError(
                f"Collection {self.collection_name!r} is not loaded"
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
            "dimensions": dimensions,
            "loaded": bool(info.get("loaded", True)),
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
        query_encoders: Any | None = None,
        provider_dimensions: dict[str, int] | None = None,
        provider_ids: Sequence[str] = PRODUCTION_PROVIDERS,
        rrf_k: int = DEFAULT_RRF_K,
        max_top_k: int = DEFAULT_MAX_TOP_K,
    ) -> None:
        manifest: SemanticIndexManifest | None = None
        if manifest_path:
            manifest = load_manifest(Path(manifest_path))
            if manifest is None:
                raise MilvusSemanticError(
                    f"Milvus manifest is unavailable: {manifest_path}"
                )

        dimensions = dict(provider_dimensions or EXPECTED_DIMENSIONS)
        dimension_errors = validate_provider_dimensions(dimensions)
        if dimension_errors:
            raise MilvusSemanticError("; ".join(dimension_errors))
        providers = tuple(dict.fromkeys(provider_ids))
        if set(providers) != set(PRODUCTION_PROVIDERS):
            raise MilvusSemanticError(
                "production Milvus requires exactly beit3, metaclip2, and "
                "perception_encoder"
            )
        if rrf_k <= 0 or max_top_k <= 0:
            raise ValueError("rrf_k and max_top_k must be positive")
        if manifest is not None:
            if manifest.collection_name != collection_name:
                raise MilvusSemanticError(
                    "Milvus manifest collection does not match runtime collection"
                )
            manifest_dimensions = {
                provider_id: entry.dimension
                for provider_id, entry in manifest.providers.items()
            }
            manifest_errors = validate_provider_dimensions(manifest_dimensions)
            if manifest_errors:
                raise MilvusSemanticError("; ".join(manifest_errors))
            for provider_id in providers:
                if manifest_dimensions[provider_id] != dimensions[provider_id]:
                    raise MilvusSemanticError(
                        f"Milvus manifest/config dimension mismatch for {provider_id}"
                    )

        self._search_backend = MilvusProviderSearchBackend(
            backend=backend,
            collection_name=collection_name,
            stale_policy=stale_policy,
            manifest=manifest,
        )
        self.collection_name = collection_name
        self.index_type = index_type
        self.provider_ids = providers
        self.rrf_k = rrf_k
        self.max_top_k = max_top_k
        self.query_encoders = query_encoders
        self.contracts = {
            provider_id: ProviderIndexContract(
                model_id=provider_id,
                collection_name=f"{collection_name}__{provider_id}",
                dimension=dimensions[provider_id],
                metric="inner_product",
                normalized=True,
                backend="milvus",
            )
            for provider_id in providers
        }
        self._retriever = (
            None
            if query_encoders is None
            else IndependentSemanticRetriever(
                query_encoders=query_encoders,
                backend=self._search_backend,
                contracts=self.contracts,
                failure_policy="fail_closed",
                required_providers=providers,
                max_top_k=max_top_k,
            )
        )

    @classmethod
    def from_env(
        cls,
        *,
        index_type: str = "FLAT",
        stale_policy: str = "fail",
        manifest_path: str | None = None,
    ) -> "MilvusSemanticService":
        """Construct from environment variables."""
        from mvr.enrichment.embeddings.provider_specs import load_provider_specs
        from mvr.enrichment.embeddings.query_encoders import (
            ProviderQueryEncoderConfig,
            QueryEncoderRegistry,
        )

        def optional_path(name: str) -> Path | None:
            value = os.environ.get(name, "").strip()
            return Path(value).resolve() if value else None

        client = MilvusClient()
        collection_name = client.config.collection
        backend = MilvusProviderBackend(
            client,
            index_type=index_type,
            collection_name=collection_name,
        )
        specs_path = Path(
            os.environ.get(
                "MVR_EMBEDDING_PROVIDER_SPECS",
                "configs/embedding_providers.json",
            )
        ).resolve()
        specs = load_provider_specs(specs_path)
        dimensions = {
            provider_id: specs.get(provider_id).embedding_dimension
            for provider_id in PRODUCTION_PROVIDERS
        }
        encoder_config = ProviderQueryEncoderConfig(
            device=os.environ.get("MVR_DEVICE", "auto"),
            beit3_checkpoint=optional_path("MVR_BEIT3_CHECKPOINT"),
            beit3_tokenizer=optional_path("MVR_BEIT3_TOKENIZER"),
            metaclip2_model=(
                os.environ.get("MVR_METACLIP2_MODEL", "").strip() or None
            ),
            metaclip2_allow_download=os.environ.get(
                "MVR_METACLIP2_ALLOW_DOWNLOAD", ""
            ).strip().lower() in {"1", "true", "yes"},
            perception_encoder_checkpoint=optional_path("MVR_PE_CHECKPOINT"),
        )
        query_encoders = QueryEncoderRegistry(specs, config=encoder_config)
        resolved_manifest = (
            manifest_path
            if manifest_path is not None
            else os.environ.get("MVR_MILVUS_MANIFEST", "").strip() or None
        )
        return cls(
            backend,
            collection_name=collection_name,
            index_type=index_type,
            stale_policy=stale_policy,
            manifest_path=resolved_manifest,
            query_encoders=query_encoders,
            provider_dimensions=dimensions,
        )

    def health(self) -> dict[str, Any]:
        """Return service readiness status."""
        try:
            return self._search_backend.verify_readiness()
        except MilvusSemanticError as exc:
            return {"status": "not_ready", "error": str(exc)}

    @property
    def configured(self) -> bool:
        return True

    @property
    def available(self) -> bool:
        return bool(self.runtime_status()["available"])

    @property
    def loaded(self) -> bool:
        return bool(self.runtime_status()["collection_ready"])

    def runtime_status(self) -> dict[str, Any]:
        """Return FastAPI readiness fields without any FAISS fallback."""
        status = self.health()
        reachable = status.get("milvus_reachable") is True
        collection_ready = (
            reachable
            and status.get("collection_exists") is True
            and status.get("schema_valid") is True
            and status.get("loaded") is True
            and status.get("manifest_ok") is not False
        )
        return {
            "configured": True,
            "available": bool(collection_ready),
            "loaded": bool(collection_ready),
            "backend": "milvus",
            "reachable": bool(reachable),
            "collection_ready": bool(collection_ready),
            "collection": self.collection_name,
            "entities": status.get("entity_count"),
            "error": status.get("error"),
        }

    def search(self, query: str, top_k: int = 5) -> list[MilvusSemanticHit]:
        """Search all provider fields and fuse their ranks with RRF k=60."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0 or top_k > self.max_top_k:
            raise ValueError(f"top_k must be between 1 and {self.max_top_k}")
        self._search_backend.verify_readiness()
        if self._retriever is None:
            raise MilvusSemanticError("Milvus query encoders are not configured")
        response = self._retriever.search(
            query.strip(),
            provider_ids=self.provider_ids,
            top_k=top_k,
        )
        fused = fuse_independent_response(
            response,
            rrf_k=self.rrf_k,
            top_k=top_k,
        )
        return [
            MilvusSemanticHit(
                rank=result.rank,
                keyframe_id=result.keyframe_id,
                video_id=result.video_id,
                frame_id=result.frame_id,
                timestamp=result.timestamp,
                score=result.rrf_score,
                provider_ranks=result.provider_ranks,
            )
            for result in fused.results
        ]

    @property
    def provider_search_backend(self) -> MilvusProviderSearchBackend:
        """Expose backend for use with IndependentSemanticRetriever."""
        return self._search_backend
