"""MilvusProviderBackend — implements ProviderIndexBackend Protocol for Milvus.

Plugs into the existing provider_indexes.py infrastructure with no interface
changes required.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .milvus_client import CANONICAL_METRIC, MilvusClient, MilvusConfig
from .milvus_schema import (
    FIELD_BATCH_ID,
    FIELD_BEIT3_VECTOR,
    FIELD_COLLECTION_ID,
    FIELD_ENTITY_ID,
    FIELD_FRAME_ID,
    FIELD_KEYFRAME_ID,
    FIELD_METACLIP2_VECTOR,
    FIELD_PE_VECTOR,
    FIELD_SHOT_ID,
    FIELD_TIMESTAMP,
    FIELD_VIDEO_ID,
    INDEX_TYPE_PARAMS,
    SEARCH_OUTPUT_FIELDS,
    SEARCH_PARAMS,
    SUPPORTED_INDEX_TYPES,
    VECTOR_FIELDS,
    build_collection_schema,
    make_entity_id,
    vector_field_for_provider,
)
from .provider_indexes import ProviderIndexContract

LOGGER = logging.getLogger(__name__)

# Default insert/upsert batch size
DEFAULT_BATCH_SIZE = 1000


class MilvusProviderBackend:
    """Milvus backend implementing the ProviderIndexBackend Protocol.

    Supports FLAT, HNSW, IVF_FLAT index types.
    Uses IP metric (inner product) compatible with L2-normalized vector contract.
    """

    name = "milvus"

    def __init__(
        self,
        client: MilvusClient | None = None,
        *,
        index_type: str = "HNSW",
        batch_size: int = DEFAULT_BATCH_SIZE,
        collection_name: str | None = None,
    ) -> None:
        if index_type not in SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"Unsupported index type: {index_type!r}. "
                f"Supported: {sorted(SUPPORTED_INDEX_TYPES)}"
            )
        self.client = client or MilvusClient()
        self.index_type = index_type
        self.batch_size = batch_size
        self.collection_name = collection_name
        self._staged: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # ProviderIndexBackend Protocol
    # ------------------------------------------------------------------

    def prepare(self, contract: ProviderIndexContract, policy: str) -> int:
        """Prepare the Milvus collection for ingest.

        policy: 'create' | 'upsert' | 'create_or_skip'
        Returns entity count before ingest.
        """
        collection_name = self.collection_name or contract.collection_name
        self.client.connect()

        if policy == "create":
            schema = build_collection_schema()
            self.client.create_collection(schema, collection_name, drop_existing=True)
        elif policy == "upsert":
            if not self.client.collection_exists(collection_name):
                schema = build_collection_schema()
                self.client.create_collection(schema, collection_name)
        elif policy == "create_or_skip":
            if not self.client.collection_exists(collection_name):
                schema = build_collection_schema()
                self.client.create_collection(schema, collection_name)
        else:
            raise ValueError(f"Unknown prepare policy: {policy!r}")

        self._staged = []
        try:
            return self.client.entity_count(collection_name)
        except Exception:
            return 0

    def insert_batch(
        self,
        contract: ProviderIndexContract,
        vectors: np.ndarray,
        metadata: Sequence[Mapping[str, Any]],
    ) -> Sequence[str]:
        """Stage and flush a batch of (vector, metadata) entities.

        Returns keyframe_ids of inserted entities.
        """
        collection_name = self.collection_name or contract.collection_name
        vector_field = vector_field_for_provider(contract.model_id)

        for i, (vec, meta) in enumerate(zip(vectors, metadata)):
            video_id = meta.get(FIELD_VIDEO_ID, "")
            keyframe_id = meta[FIELD_KEYFRAME_ID]
            entity: dict[str, Any] = {
                FIELD_ENTITY_ID: make_entity_id(video_id, keyframe_id),
                FIELD_KEYFRAME_ID: keyframe_id,
                FIELD_VIDEO_ID: video_id,
                FIELD_FRAME_ID: int(meta.get(FIELD_FRAME_ID, 0)),
                FIELD_SHOT_ID: int(meta.get(FIELD_SHOT_ID, -1)),
                FIELD_TIMESTAMP: float(meta.get(FIELD_TIMESTAMP, 0.0)),
                FIELD_BATCH_ID: meta.get(FIELD_BATCH_ID, ""),
                FIELD_COLLECTION_ID: meta.get(FIELD_COLLECTION_ID, ""),
                vector_field: vec.tolist(),
            }
            # Fill other vector fields with zeros (will be upserted later)
            for vf in VECTOR_FIELDS:
                if vf not in entity:
                    entity[vf] = []
            self._staged.append(entity)

        inserted_ids: list[str] = []
        while len(self._staged) >= self.batch_size:
            batch = self._staged[: self.batch_size]
            self._staged = self._staged[self.batch_size :]
            ids = self.client.upsert_batch(batch, collection_name=collection_name)
            inserted_ids.extend(str(i) for i in ids)

        return inserted_ids

    def finalize(self, contract: ProviderIndexContract) -> int:
        """Flush remaining staged entities, create indexes, load collection.

        Returns final entity count.
        """
        collection_name = self.collection_name or contract.collection_name

        # Flush remaining staged entities
        if self._staged:
            self.client.upsert_batch(self._staged, collection_name=collection_name)
            self._staged = []

        self.client.flush_collection(collection_name)

        # Create indexes on all vector fields
        for vector_field in VECTOR_FIELDS:
            self.client.create_index(
                vector_field,
                index_type=self.index_type,
                metric_type=CANONICAL_METRIC,
                params=INDEX_TYPE_PARAMS.get(self.index_type, {}),
                collection_name=collection_name,
            )

        # Load collection into memory
        self.client.load_collection(collection_name)
        count = self.client.entity_count(collection_name)
        LOGGER.info(
            "Finalized Milvus collection %s: %d entities, index=%s",
            collection_name, count, self.index_type,
        )
        return count

    # ------------------------------------------------------------------
    # Search (used by MilvusSemanticService)
    # ------------------------------------------------------------------

    def search_provider(
        self,
        query_vector: np.ndarray,
        provider_id: str,
        *,
        top_k: int = 30,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search a single provider's vector field.

        Returns list of hits with identity fields + score.
        Query vector routes ONLY to its matching provider field.
        """
        vector_field = vector_field_for_provider(provider_id)
        target = collection_name or self.collection_name or COLLECTION_NAME
        search_params = {
            "metric_type": CANONICAL_METRIC,
            **SEARCH_PARAMS.get(self.index_type, {}),
        }
        results = self.client.search(
            query_vectors=[query_vector.tolist()],
            vector_field=vector_field,
            top_k=top_k,
            metric_type=CANONICAL_METRIC,
            search_params=search_params,
            output_fields=SEARCH_OUTPUT_FIELDS,
            collection_name=target,
        )
        return results[0] if results else []


# Avoid circular import
from .milvus_schema import COLLECTION_NAME  # noqa: E402
