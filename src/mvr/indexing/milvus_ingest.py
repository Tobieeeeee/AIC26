"""Milvus ingest pipeline — canonical Data-02 Parquet → Milvus.

Flow:
    canonical Parquet (extraction_status)
            ↓
    Polars eligibility gate
    (beit3 DONE AND metaclip2 DONE AND pe_status DONE)
            ↓
    provider mapping join + dimension validation
            ↓
    finite vector validation
            ↓
    Milvus upsert batches
            ↓
    manifest write
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .milvus_backend import MilvusProviderBackend
from .milvus_client import MilvusClient, MilvusConfig
from .milvus_manifest import (
    ProviderManifestEntry,
    SemanticIndexManifest,
    STATE_FRESH,
    build_manifest,
    check_staleness,
    load_manifest,
    write_manifest,
)
from .milvus_schema import (
    COLLECTION_NAME,
    EXPECTED_DIMENSIONS,
    FIELD_BATCH_ID,
    FIELD_COLLECTION_ID,
    FIELD_FRAME_ID,
    FIELD_KEYFRAME_ID,
    FIELD_SHOT_ID,
    FIELD_TIMESTAMP,
    FIELD_VIDEO_ID,
    build_collection_schema,
    validate_provider_dimensions,
)

LOGGER = logging.getLogger(__name__)

REQUIRED_PROVIDERS = ("beit3", "metaclip2", "perception_encoder")

# Status column names in extraction_status Parquet
_STATUS_COLS = {
    "beit3": "beit3_status",
    "metaclip2": "metaclip2_status",
    "perception_encoder": "pe_status",
}


class MilvusIngestError(RuntimeError):
    """Raised when ingest cannot proceed safely."""


@dataclass
class EligibilityReport:
    total_keyframes: int = 0
    eligible: int = 0
    ineligible: int = 0
    rejected_invalid_vector: int = 0
    rejected_dim_mismatch: int = 0
    rejected_missing_provider: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class IngestResult:
    collection_name: str
    entity_count_before: int
    entity_count_after: int
    upserted: int
    eligibility: EligibilityReport
    manifest_path: str
    index_type: str


def _require_polars():
    try:
        import polars as pl
        return pl
    except ImportError as exc:
        raise MilvusIngestError(
            "polars is required for Milvus ingest; install with: pip install polars"
        ) from exc


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_vectors(vector_path: Path) -> np.ndarray:
    """Load vectors from .npy or .npz file."""
    if vector_path.suffix == ".npy":
        return np.load(vector_path, mmap_mode="r")
    if vector_path.suffix == ".npz":
        data = np.load(vector_path)
        key = list(data.keys())[0]
        return data[key]
    raise MilvusIngestError(f"Unsupported vector file format: {vector_path}")


def _load_mapping(mapping_path: Path) -> dict[str, int]:
    """Load keyframe_id → row_index mapping from Parquet."""
    pl = _require_polars()
    df = pl.read_parquet(mapping_path)
    if "keyframe_id" not in df.columns or "row_idx" not in df.columns:
        raise MilvusIngestError(
            f"Mapping Parquet must have 'keyframe_id' and 'row_idx' columns: {mapping_path}"
        )
    return dict(zip(df["keyframe_id"].to_list(), df["row_idx"].to_list()))


def _is_finite_vector(vec: np.ndarray) -> bool:
    return bool(np.isfinite(vec).all()) and vec.ndim == 1 and len(vec) > 0


def discover_eligible_keyframes(
    status_parquet: Path,
) -> "Any":
    """Return a Polars DataFrame of eligible keyframe rows.

    Eligibility: all three provider statuses == DONE.
    """
    pl = _require_polars()
    df = pl.read_parquet(status_parquet)
    for provider_id, col in _STATUS_COLS.items():
        if col not in df.columns:
            LOGGER.warning("Column %s not found in extraction_status; treating as ineligible", col)
            df = df.with_columns(pl.lit("TODO").alias(col))
    eligible = df.filter(
        (pl.col("beit3_status") == "DONE")
        & (pl.col("metaclip2_status") == "DONE")
        & (pl.col("pe_status") == "DONE")
    )
    return eligible


def _build_entity_row(
    keyframe_id: str,
    meta_row: dict[str, Any],
    provider_vectors: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Build a flat entity dict for Milvus upsert."""
    from .milvus_schema import (
        FIELD_BEIT3_VECTOR, FIELD_METACLIP2_VECTOR, FIELD_PE_VECTOR,
    )
    return {
        FIELD_KEYFRAME_ID: keyframe_id,
        FIELD_VIDEO_ID: meta_row.get(FIELD_VIDEO_ID, ""),
        FIELD_FRAME_ID: int(meta_row.get(FIELD_FRAME_ID, 0)),
        FIELD_SHOT_ID: int(meta_row.get(FIELD_SHOT_ID, -1)),
        FIELD_TIMESTAMP: float(meta_row.get(FIELD_TIMESTAMP, 0.0)),
        FIELD_BATCH_ID: meta_row.get(FIELD_BATCH_ID, ""),
        FIELD_COLLECTION_ID: meta_row.get(FIELD_COLLECTION_ID, ""),
        FIELD_BEIT3_VECTOR: provider_vectors["beit3"].tolist(),
        FIELD_METACLIP2_VECTOR: provider_vectors["metaclip2"].tolist(),
        FIELD_PE_VECTOR: provider_vectors["perception_encoder"].tolist(),
    }


def ingest_to_milvus(
    *,
    status_parquet: Path,
    keyframe_mapping_csv: Path,
    provider_artifacts: dict[str, dict[str, Path]],
    # provider_artifacts = {
    #   "beit3": {"vectors": Path, "mapping": Path, "manifest": Path},
    #   "metaclip2": {...}, "perception_encoder": {...}
    # }
    manifest_path: Path,
    client: MilvusClient | None = None,
    index_type: str = "HNSW",
    collection_name: str = COLLECTION_NAME,
    policy: str = "upsert",
    batch_size: int = 1000,
) -> IngestResult:
    """Ingest eligible keyframes from canonical Data-02 Parquet into Milvus.

    Only inserts/upserts complete 3-provider rows. No partial entities.
    """
    pl = _require_polars()
    eligibility = EligibilityReport()

    # ----------------------------------------------------------------
    # 1. Validate provider artifacts and dimensions
    # ----------------------------------------------------------------
    manifest_dims: dict[str, int] = {}
    provider_manifests: dict[str, dict[str, Any]] = {}
    provider_checksums: dict[str, str] = {}

    for pid in REQUIRED_PROVIDERS:
        if pid not in provider_artifacts:
            raise MilvusIngestError(f"Provider artifacts missing for: {pid}")
        art = provider_artifacts[pid]
        if "manifest" in art and art["manifest"].is_file():
            pm = json.loads(art["manifest"].read_text(encoding="utf-8"))
            provider_manifests[pid] = pm
            manifest_dims[pid] = pm.get("dimension", EXPECTED_DIMENSIONS[pid])
        else:
            manifest_dims[pid] = EXPECTED_DIMENSIONS[pid]
        if "vectors" in art and art["vectors"].is_file():
            provider_checksums[pid] = _sha256_path(art["vectors"])
        else:
            provider_checksums[pid] = ""

    dim_errors = validate_provider_dimensions(manifest_dims)
    if dim_errors:
        raise MilvusIngestError(
            "Provider dimension validation failed:\n" + "\n".join(dim_errors)
        )

    # ----------------------------------------------------------------
    # 2. Stale detection against existing manifest
    # ----------------------------------------------------------------
    keyframe_checksum = _sha256_path(keyframe_mapping_csv) if keyframe_mapping_csv.is_file() else ""
    existing_manifest = load_manifest(manifest_path)
    if existing_manifest is not None and policy == "upsert":
        state, reasons = check_staleness(
            existing_manifest,
            current_keyframe_checksum=keyframe_checksum,
            current_provider_checksums=provider_checksums,
            current_dimensions=manifest_dims,
        )
        if state == "STALE":
            LOGGER.warning("Semantic index manifest is STALE: %s", "; ".join(reasons))

    # ----------------------------------------------------------------
    # 3. Load provider vectors and mappings
    # ----------------------------------------------------------------
    provider_vectors: dict[str, np.ndarray] = {}
    provider_mappings: dict[str, dict[str, int]] = {}

    for pid in REQUIRED_PROVIDERS:
        art = provider_artifacts[pid]
        if "vectors" not in art or not art["vectors"].is_file():
            raise MilvusIngestError(f"Vector file missing for provider {pid}: {art.get('vectors')}")
        provider_vectors[pid] = _load_vectors(art["vectors"])
        if "mapping" in art and art["mapping"].is_file():
            provider_mappings[pid] = _load_mapping(art["mapping"])
        else:
            # Assume positional alignment with status parquet order
            provider_mappings[pid] = {}

    # ----------------------------------------------------------------
    # 4. Discover eligible keyframes
    # ----------------------------------------------------------------
    eligible_df = discover_eligible_keyframes(status_parquet)
    eligibility.total_keyframes = pl.read_parquet(status_parquet).height
    eligibility.eligible = eligible_df.height
    eligibility.ineligible = eligibility.total_keyframes - eligibility.eligible
    LOGGER.info(
        "Eligibility: %d/%d keyframes eligible for Milvus ingest",
        eligibility.eligible, eligibility.total_keyframes,
    )

    if eligibility.eligible == 0:
        raise MilvusIngestError("No eligible keyframes found (all three providers must be DONE)")

    # ----------------------------------------------------------------
    # 5. Prepare Milvus collection
    # ----------------------------------------------------------------
    milvus_client = client or MilvusClient()
    backend = MilvusProviderBackend(
        milvus_client,
        index_type=index_type,
        batch_size=batch_size,
        collection_name=collection_name,
    )

    from .provider_indexes import ProviderIndexContract
    dummy_contract = ProviderIndexContract(
        model_id="beit3",
        collection_name=collection_name,
        dimension=manifest_dims["beit3"],
        metric="IP",
        normalized=True,
        backend="milvus",
    )
    entity_count_before = backend.prepare(dummy_contract, policy)

    # ----------------------------------------------------------------
    # 6. Upsert eligible entities in batches
    # ----------------------------------------------------------------
    eligible_rows = eligible_df.to_dicts()
    upserted = 0
    batch_entities: list[dict[str, Any]] = []

    for row in eligible_rows:
        kf_id = row[FIELD_KEYFRAME_ID]
        # Retrieve vectors for each provider
        provider_vecs: dict[str, np.ndarray] = {}
        rejected = False
        for pid in REQUIRED_PROVIDERS:
            mapping = provider_mappings[pid]
            if mapping:
                if kf_id not in mapping:
                    eligibility.rejected_missing_provider += 1
                    eligibility.errors.append(f"{kf_id}: missing in {pid} mapping")
                    rejected = True
                    break
                row_idx = mapping[kf_id]
            else:
                # No mapping file: this provider skips (should not happen in production)
                eligibility.rejected_missing_provider += 1
                rejected = True
                break

            vec = provider_vectors[pid][row_idx].astype(np.float32)

            if vec.shape[0] != manifest_dims[pid]:
                eligibility.rejected_dim_mismatch += 1
                eligibility.errors.append(
                    f"{kf_id}/{pid}: dim {vec.shape[0]} != {manifest_dims[pid]}"
                )
                rejected = True
                break

            if not _is_finite_vector(vec):
                eligibility.rejected_invalid_vector += 1
                eligibility.errors.append(f"{kf_id}/{pid}: non-finite vector")
                rejected = True
                break

            provider_vecs[pid] = vec

        if rejected:
            continue

        entity = _build_entity_row(kf_id, row, provider_vecs)
        batch_entities.append(entity)

        if len(batch_entities) >= batch_size:
            milvus_client.upsert_batch(batch_entities, collection_name=collection_name)
            upserted += len(batch_entities)
            batch_entities = []

    if batch_entities:
        milvus_client.upsert_batch(batch_entities, collection_name=collection_name)
        upserted += len(batch_entities)

    # ----------------------------------------------------------------
    # 7. Create indexes and load collection
    # ----------------------------------------------------------------
    backend.finalize(dummy_contract)
    entity_count_after = milvus_client.entity_count(collection_name)

    # ----------------------------------------------------------------
    # 8. Write manifest
    # ----------------------------------------------------------------
    provider_entries: dict[str, ProviderManifestEntry] = {}
    for pid in REQUIRED_PROVIDERS:
        art = provider_artifacts[pid]
        pm = provider_manifests.get(pid, {})
        provider_entries[pid] = ProviderManifestEntry(
            provider_id=pid,
            dimension=manifest_dims[pid],
            metric="IP",
            normalized=pm.get("normalized", True),
            checksum=provider_checksums.get(pid, ""),
            vector_file=str(art.get("vectors", "")),
            mapping_file=str(art.get("mapping", "")),
        )

    final_manifest = build_manifest(
        collection_name=collection_name,
        entity_count=entity_count_after,
        keyframe_mapping_checksum=keyframe_checksum,
        providers=provider_entries,
        index_type=index_type,
        metric="IP",
        index_params={},
        state=STATE_FRESH,
    )
    write_manifest(final_manifest, manifest_path)

    return IngestResult(
        collection_name=collection_name,
        entity_count_before=entity_count_before,
        entity_count_after=entity_count_after,
        upserted=upserted,
        eligibility=eligibility,
        manifest_path=str(manifest_path),
        index_type=index_type,
    )
