"""Milvus client for AIC26 semantic vector database.

Credentials come only from environment variables — nothing is ever committed.

Required env vars:
    MILVUS_URI       Full URI e.g. http://localhost:19530 (takes precedence)
    MILVUS_HOST      Hostname (used if MILVUS_URI not set)
    MILVUS_PORT      Port     (used if MILVUS_URI not set, default 19530)
    MILVUS_TOKEN     API token for Milvus Cloud (optional for local dev)
    MILVUS_DATABASE  Database name (default: default)
    MILVUS_COLLECTION  Collection name (default: aic26_keyframes)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 19530
DEFAULT_DATABASE = "default"
DEFAULT_COLLECTION = "aic26_keyframes"

# Metric compatible with existing normalized vector contract
CANONICAL_METRIC = "IP"  # inner product = cosine for L2-normalized vectors


class MilvusConfigError(RuntimeError):
    """Missing or invalid Milvus configuration."""


class MilvusConnectionError(RuntimeError):
    """Could not connect to Milvus."""


class MilvusCollectionError(RuntimeError):
    """Collection does not exist or schema mismatch."""


class MilvusSearchError(RuntimeError):
    """A Milvus search operation failed."""


@dataclass
class MilvusConfig:
    uri: str = ""
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token: str = ""
    database: str = DEFAULT_DATABASE
    collection: str = DEFAULT_COLLECTION

    @classmethod
    def from_env(cls) -> "MilvusConfig":
        uri = os.environ.get("MILVUS_URI", "").strip()
        host = os.environ.get("MILVUS_HOST", DEFAULT_HOST).strip()
        port_str = os.environ.get("MILVUS_PORT", str(DEFAULT_PORT)).strip()
        token = os.environ.get("MILVUS_TOKEN", "").strip()
        database = os.environ.get("MILVUS_DATABASE", DEFAULT_DATABASE).strip()
        collection = os.environ.get("MILVUS_COLLECTION", DEFAULT_COLLECTION).strip()
        try:
            port = int(port_str)
        except ValueError:
            raise MilvusConfigError(
                f"MILVUS_PORT must be an integer, got: {port_str!r}"
            )
        return cls(uri=uri, host=host, port=port, token=token,
                   database=database, collection=collection)

    @property
    def effective_uri(self) -> str:
        if self.uri:
            return self.uri
        return f"http://{self.host}:{self.port}"


def _require_pymilvus():
    try:
        import pymilvus
        return pymilvus
    except ImportError as exc:
        raise MilvusConfigError(
            "pymilvus is required for Milvus operations; "
            "install with: pip install pymilvus"
        ) from exc


class MilvusClient:
    """Reusable Milvus client with lazy connection."""

    def __init__(self, config: MilvusConfig | None = None) -> None:
        self.config = config or MilvusConfig.from_env()
        self._connected = False

    def connect(self) -> None:
        """Establish connection to Milvus."""
        pm = _require_pymilvus()
        kwargs: dict[str, Any] = {
            "uri": self.config.effective_uri,
            "db_name": self.config.database,
        }
        if self.config.token:
            kwargs["token"] = self.config.token
        try:
            pm.connections.connect("default", **kwargs)
            self._connected = True
            LOGGER.info("Connected to Milvus at %s", self.config.effective_uri)
        except Exception as exc:
            raise MilvusConnectionError(
                f"Could not connect to Milvus at {self.config.effective_uri}: {exc}"
            ) from exc

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def health(self) -> dict[str, Any]:
        """Check Milvus reachability and return status info."""
        self._ensure_connected()
        pm = _require_pymilvus()
        try:
            utility = pm.utility
            collections = utility.list_collections()
            return {
                "status": "ok",
                "uri": self.config.effective_uri,
                "database": self.config.database,
                "collections": collections,
            }
        except Exception as exc:
            raise MilvusConnectionError(f"Milvus health check failed: {exc}") from exc

    def collection_exists(self, name: str | None = None) -> bool:
        """Check if a collection exists."""
        self._ensure_connected()
        pm = _require_pymilvus()
        target = name or self.config.collection
        return pm.utility.has_collection(target)

    def create_collection(
        self,
        schema: Any,
        name: str | None = None,
        *,
        drop_existing: bool = False,
    ) -> None:
        """Create a Milvus collection from a CollectionSchema."""
        self._ensure_connected()
        pm = _require_pymilvus()
        target = name or self.config.collection
        if drop_existing and pm.utility.has_collection(target):
            LOGGER.warning("Dropping existing collection: %s", target)
            pm.utility.drop_collection(target)
        if pm.utility.has_collection(target):
            LOGGER.info("Collection already exists: %s", target)
            return
        pm.Collection(name=target, schema=schema, using="default")
        LOGGER.info("Created collection: %s", target)

    def validate_collection(
        self,
        name: str | None = None,
        *,
        expected_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate that a collection exists and has the expected fields."""
        self._ensure_connected()
        pm = _require_pymilvus()
        target = name or self.config.collection
        if not pm.utility.has_collection(target):
            raise MilvusCollectionError(f"Collection does not exist: {target}")
        col = pm.Collection(name=target)
        actual_fields = [f.name for f in col.schema.fields]
        if expected_fields:
            missing = set(expected_fields) - set(actual_fields)
            if missing:
                raise MilvusCollectionError(
                    f"Collection {target} is missing fields: {sorted(missing)}"
                )
        return {"collection": target, "fields": actual_fields, "schema_valid": True}

    def create_index(
        self,
        field_name: str,
        *,
        index_type: str = "HNSW",
        metric_type: str = CANONICAL_METRIC,
        params: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Create an ANN index on a vector field."""
        self._ensure_connected()
        pm = _require_pymilvus()
        target = collection_name or self.config.collection
        col = pm.Collection(name=target)
        index_params: dict[str, Any] = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": params or {},
        }
        col.create_index(field_name=field_name, index_params=index_params)
        LOGGER.info(
            "Created %s index on %s.%s (metric=%s)",
            index_type, target, field_name, metric_type,
        )

    def load_collection(self, name: str | None = None) -> None:
        """Load a collection into memory for search."""
        self._ensure_connected()
        pm = _require_pymilvus()
        target = name or self.config.collection
        col = pm.Collection(name=target)
        col.load()
        LOGGER.info("Loaded collection: %s", target)

    def flush_collection(self, name: str | None = None) -> None:
        """Flush pending inserts/upserts so counts and indexes see them."""
        self._ensure_connected()
        pm = _require_pymilvus()
        target = name or self.config.collection
        pm.Collection(name=target).flush()
        LOGGER.info("Flushed collection: %s", target)

    def insert_batch(
        self,
        entities: list[dict[str, Any]],
        *,
        collection_name: str | None = None,
    ) -> list[Any]:
        """Insert a batch of entities. Returns inserted IDs."""
        self._ensure_connected()
        pm = _require_pymilvus()
        target = collection_name or self.config.collection
        col = pm.Collection(name=target)
        result = col.insert(entities)
        LOGGER.debug("Inserted %d entities into %s", len(entities), target)
        return list(result.primary_keys)

    def upsert_batch(
        self,
        entities: list[dict[str, Any]],
        *,
        collection_name: str | None = None,
    ) -> list[Any]:
        """Upsert a batch of entities (insert or update by PK)."""
        self._ensure_connected()
        pm = _require_pymilvus()
        target = collection_name or self.config.collection
        col = pm.Collection(name=target)
        result = col.upsert(entities)
        LOGGER.debug("Upserted %d entities into %s", len(entities), target)
        return list(result.primary_keys)

    def search(
        self,
        query_vectors: list[list[float]],
        vector_field: str,
        *,
        top_k: int = 30,
        metric_type: str = CANONICAL_METRIC,
        search_params: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
        collection_name: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """ANN search on a specific vector field.

        Returns per-query lists of hit dicts with at least:
        id, distance, and all requested output_fields.
        """
        self._ensure_connected()
        pm = _require_pymilvus()
        target = collection_name or self.config.collection
        col = pm.Collection(name=target)
        sp = search_params or {"metric_type": metric_type}
        results = col.search(
            data=query_vectors,
            anns_field=vector_field,
            param=sp,
            limit=top_k,
            output_fields=output_fields or [],
        )
        hits: list[list[dict[str, Any]]] = []
        for query_result in results:
            query_hits = []
            for hit in query_result:
                hit_dict: dict[str, Any] = {
                    "id": hit.id,
                    "distance": hit.distance,
                }
                for f in (output_fields or []):
                    hit_dict[f] = hit.entity.get(f)
                query_hits.append(hit_dict)
            hits.append(query_hits)
        return hits

    def entity_count(self, name: str | None = None) -> int:
        """Return the number of entities in a collection."""
        self._ensure_connected()
        pm = _require_pymilvus()
        target = name or self.config.collection
        col = pm.Collection(name=target)
        return col.num_entities

    def collection_info(self, name: str | None = None) -> dict[str, Any]:
        """Return collection metadata."""
        self._ensure_connected()
        pm = _require_pymilvus()
        target = name or self.config.collection
        if not pm.utility.has_collection(target):
            raise MilvusCollectionError(f"Collection does not exist: {target}")
        col = pm.Collection(name=target)
        dimensions = {
            field.name: int(field.params["dim"])
            for field in col.schema.fields
            if "dim" in field.params
        }
        try:
            loaded = str(pm.utility.load_state(target)).endswith("Loaded")
        except Exception:
            loaded = False
        return {
            "name": target,
            "fields": [f.name for f in col.schema.fields],
            "dimensions": dimensions,
            "num_entities": col.num_entities,
            "loaded": loaded,
            "indexes": [
                {"field": idx.field_name, "type": idx.params.get("index_type")}
                for idx in col.indexes
            ],
        }

    def disconnect(self) -> None:
        if self._connected:
            pm = _require_pymilvus()
            pm.connections.disconnect("default")
            self._connected = False
