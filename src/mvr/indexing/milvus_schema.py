"""Milvus collection schema for AIC26 semantic indexing.

Collection: aic26_keyframes
One entity per keyframe_id, three independent vector fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .milvus_client import CANONICAL_METRIC, MilvusConfigError

# ---------------------------------------------------------------------------
# Canonical field names
# ---------------------------------------------------------------------------
FIELD_ENTITY_ID = "entity_id"
FIELD_KEYFRAME_ID = "keyframe_id"
FIELD_VIDEO_ID = "video_id"
FIELD_FRAME_ID = "frame_id"
FIELD_SHOT_ID = "shot_id"
FIELD_TIMESTAMP = "timestamp"
FIELD_BATCH_ID = "batch_id"
FIELD_COLLECTION_ID = "collection_id"

FIELD_BEIT3_VECTOR = "beit3_vector"
FIELD_METACLIP2_VECTOR = "metaclip2_vector"
FIELD_PE_VECTOR = "pe_vector"

SCALAR_FIELDS = (
    FIELD_ENTITY_ID,
    FIELD_KEYFRAME_ID,
    FIELD_VIDEO_ID,
    FIELD_FRAME_ID,
    FIELD_SHOT_ID,
    FIELD_TIMESTAMP,
    FIELD_BATCH_ID,
    FIELD_COLLECTION_ID,
)

VECTOR_FIELDS = (
    FIELD_BEIT3_VECTOR,
    FIELD_METACLIP2_VECTOR,
    FIELD_PE_VECTOR,
)

# Map: provider model_id → Milvus vector field name
PROVIDER_TO_VECTOR_FIELD: dict[str, str] = {
    "beit3": FIELD_BEIT3_VECTOR,
    "metaclip2": FIELD_METACLIP2_VECTOR,
    "perception_encoder": FIELD_PE_VECTOR,
}

# Expected dimensions from pinned provider specs (configs/embedding_providers.json)
EXPECTED_DIMENSIONS: dict[str, int] = {
    "beit3": 768,
    "metaclip2": 512,
    "perception_encoder": 1024,
}

COLLECTION_NAME = "aic26_keyframes"
SCHEMA_VERSION = "1.0"

# Output fields returned with every semantic search hit
SEARCH_OUTPUT_FIELDS = [
    FIELD_ENTITY_ID,
    FIELD_KEYFRAME_ID,
    FIELD_VIDEO_ID,
    FIELD_FRAME_ID,
    FIELD_SHOT_ID,
    FIELD_TIMESTAMP,
    FIELD_BATCH_ID,
    FIELD_COLLECTION_ID,
]

# Supported index types
SUPPORTED_INDEX_TYPES = {"FLAT", "HNSW", "IVF_FLAT"}

# Index params for each type
INDEX_TYPE_PARAMS: dict[str, dict[str, Any]] = {
    "FLAT": {},
    "HNSW": {"M": 16, "efConstruction": 200},
    "IVF_FLAT": {"nlist": 128},
}

SEARCH_PARAMS: dict[str, dict[str, Any]] = {
    "FLAT": {},
    "HNSW": {"ef": 64},
    "IVF_FLAT": {"nprobe": 16},
}


def make_entity_id(video_id: str, keyframe_id: str) -> str:
    """Build the deterministic Milvus primary key for a keyframe.

    `keyframe_id` alone is not globally unique across the organizer's
    dataset (the same id string recurs across many videos), so the actual
    Milvus primary key namespaces it by video: ``video_id/keyframe_id``.
    Every write path must go through this single helper.
    """
    video_id = video_id.strip()
    keyframe_id = keyframe_id.strip()
    if not video_id or not keyframe_id:
        raise ValueError("video_id and keyframe_id must be non-empty")
    return f"{video_id}/{keyframe_id}"


@dataclass(frozen=True)
class VectorFieldSpec:
    name: str
    provider_id: str
    dimension: int
    metric: str = CANONICAL_METRIC


def build_collection_schema(
    *,
    beit3_dim: int = EXPECTED_DIMENSIONS["beit3"],
    metaclip2_dim: int = EXPECTED_DIMENSIONS["metaclip2"],
    pe_dim: int = EXPECTED_DIMENSIONS["perception_encoder"],
    max_entity_id_len: int = 512,
    max_keyframe_id_len: int = 256,
    max_string_len: int = 128,
):
    """Build and return the Milvus CollectionSchema for aic26_keyframes.

    Dimensions should be validated against provider manifests before calling.
    """
    try:
        from pymilvus import CollectionSchema, DataType, FieldSchema
    except ImportError as exc:
        raise MilvusConfigError(
            "pymilvus is required; install with: pip install pymilvus"
        ) from exc

    fields = [
        FieldSchema(
            name=FIELD_ENTITY_ID,
            dtype=DataType.VARCHAR,
            max_length=max_entity_id_len,
            is_primary=True,
            auto_id=False,
            description="Unique entity identifier (primary key): video_id/keyframe_id",
        ),
        FieldSchema(
            name=FIELD_KEYFRAME_ID,
            dtype=DataType.VARCHAR,
            max_length=max_keyframe_id_len,
            description="Keyframe identifier (not globally unique; scalar metadata only)",
        ),
        FieldSchema(
            name=FIELD_VIDEO_ID,
            dtype=DataType.VARCHAR,
            max_length=max_string_len,
            description="Source video identifier",
        ),
        FieldSchema(
            name=FIELD_FRAME_ID,
            dtype=DataType.INT64,
            description="Competition-facing frame index",
        ),
        FieldSchema(
            name=FIELD_SHOT_ID,
            dtype=DataType.INT64,
            description="Shot index (nullable as -1)",
        ),
        FieldSchema(
            name=FIELD_TIMESTAMP,
            dtype=DataType.FLOAT,
            description="Timestamp in seconds",
        ),
        FieldSchema(
            name=FIELD_BATCH_ID,
            dtype=DataType.VARCHAR,
            max_length=max_string_len,
            description="Dataset batch identifier",
        ),
        FieldSchema(
            name=FIELD_COLLECTION_ID,
            dtype=DataType.VARCHAR,
            max_length=max_string_len,
            description="Dataset collection identifier",
        ),
        FieldSchema(
            name=FIELD_BEIT3_VECTOR,
            dtype=DataType.FLOAT_VECTOR,
            dim=beit3_dim,
            description=f"BEiT-3 visual embedding (dim={beit3_dim}, IP-normalized)",
        ),
        FieldSchema(
            name=FIELD_METACLIP2_VECTOR,
            dtype=DataType.FLOAT_VECTOR,
            dim=metaclip2_dim,
            description=f"MetaCLIP-2 visual embedding (dim={metaclip2_dim}, IP-normalized)",
        ),
        FieldSchema(
            name=FIELD_PE_VECTOR,
            dtype=DataType.FLOAT_VECTOR,
            dim=pe_dim,
            description=f"PE-Core-L visual embedding (dim={pe_dim}, IP-normalized)",
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description=(
            f"AIC26 keyframe semantic index v{SCHEMA_VERSION}. "
            "Three independent normalized vector fields (BEiT-3, MetaCLIP-2, PE-Core-L). "
            "No cross-provider concatenation or raw score addition."
        ),
    )


def validate_provider_dimensions(
    manifest_dims: dict[str, int],
) -> list[str]:
    """Validate that manifest dimensions match expected canonical values.

    Returns a list of validation error strings. Empty = valid.
    """
    errors: list[str] = []
    for provider_id, expected in EXPECTED_DIMENSIONS.items():
        if provider_id not in manifest_dims:
            errors.append(f"provider {provider_id!r} missing from manifest dimensions")
            continue
        actual = manifest_dims[provider_id]
        if actual != expected:
            errors.append(
                f"provider {provider_id!r} dimension mismatch: "
                f"expected {expected}, manifest says {actual}"
            )
    return errors


def vector_field_for_provider(provider_id: str) -> str:
    """Return the Milvus vector field name for a provider."""
    if provider_id not in PROVIDER_TO_VECTOR_FIELD:
        raise ValueError(
            f"Unknown provider {provider_id!r}. "
            f"Valid providers: {sorted(PROVIDER_TO_VECTOR_FIELD)}"
        )
    return PROVIDER_TO_VECTOR_FIELD[provider_id]
