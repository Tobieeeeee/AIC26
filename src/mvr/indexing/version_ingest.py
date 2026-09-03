"""Ingest one immutable enrichment version into Milvus.

Composition only: it validates the version, reads its catalog, downloads
exactly the vector files that catalog references, builds entities keyed on the
``(video_id, keyframe_id)`` pair, upserts them, and records the version id in
the semantic-index manifest so an index can always be traced back to the
version it was built from.

It deliberately never touches ``extraction_status.parquet``. See
``mvr.operations.enrichment_catalog`` for why.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mvr.operations.enrichment_catalog import read_enrichment_catalog

from .milvus_backend import MilvusProviderBackend
from .milvus_client import MilvusClient
from .milvus_ingest import (
    MilvusIngestError,
    IngestResult,
    REQUIRED_PROVIDERS,
    build_entities_from_catalog,
)
from .milvus_manifest import (
    ProviderManifestEntry,
    STATE_FRESH,
    build_manifest,
    write_manifest,
)
from .milvus_schema import (
    COLLECTION_NAME,
    EXPECTED_DIMENSIONS,
    validate_provider_dimensions,
)

LOGGER = logging.getLogger(__name__)


def _provider_dimensions(
    *, r2_client: Any, prefix: str, batch: str, collection: str, work_root: Path
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """Read each provider's declared dimension from its published manifest."""
    dimensions: dict[str, int] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for provider in REQUIRED_PROVIDERS:
        key = (
            f"{prefix}/batch={batch}/collection={collection}"
            f"/embeddings/{provider}/manifest.json"
        )
        local = work_root / "manifests" / f"{provider}.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        r2_client.download_file(key, local)
        try:
            payload = json.loads(local.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise MilvusIngestError(f"unreadable provider manifest: {key}") from exc
        manifests[provider] = payload
        dimensions[provider] = int(
            payload.get("dimension", EXPECTED_DIMENSIONS[provider])
        )
    return dimensions, manifests


def _download_vectors(
    *, r2_client: Any, prefix: str, batch: str, entries: Sequence[Any], root: Path
) -> dict[str, Path]:
    """Fetch each referenced vector file once, mirroring the version layout."""
    wanted: dict[tuple[str, str], str] = {}
    for entry in entries:
        for provider, (feature_file, _index) in entry.vectors.items():
            wanted.setdefault(
                (provider, feature_file),
                f"{prefix}/batch={batch}/collection={entry.collection_id}"
                f"/embeddings/{provider}/{feature_file}",
            )
    downloaded: dict[str, Path] = {}
    for (provider, feature_file), key in sorted(wanted.items()):
        local = root / "embeddings" / provider / feature_file
        local.parent.mkdir(parents=True, exist_ok=True)
        r2_client.download_file(key, local)
        downloaded[key] = local
    return downloaded


def ingest_version_to_milvus(
    *,
    r2_client: Any,
    version: str,
    work_root: Path,
    manifest_path: Path,
    collections: Sequence[str] | None = None,
    milvus_client: MilvusClient | None = None,
    index_type: str = "HNSW",
    collection_name: str = COLLECTION_NAME,
    policy: str = "upsert",
    batch_size: int = 1000,
) -> IngestResult:
    """Build the Milvus semantic index from a published enrichment version."""
    work_root = Path(work_root)
    catalog = read_enrichment_catalog(
        client=r2_client,
        version=version,
        work_root=work_root,
        collections=collections,
    )
    entries = catalog["entries"]
    prefix = catalog["prefix"]
    batch = catalog["batch"]

    dimensions, provider_manifests = _provider_dimensions(
        r2_client=r2_client,
        prefix=prefix,
        batch=batch,
        collection=catalog["collections"][0],
        work_root=work_root,
    )
    dimension_errors = validate_provider_dimensions(dimensions)
    if dimension_errors:
        raise MilvusIngestError(
            "provider dimension validation failed: " + "; ".join(dimension_errors)
        )

    vectors_root = work_root / "vectors"
    _download_vectors(
        r2_client=r2_client,
        prefix=prefix,
        batch=batch,
        entries=entries,
        root=vectors_root,
    )

    entities, eligibility = build_entities_from_catalog(
        entries=entries,
        vectors_root=vectors_root,
        batch_id=f"batch{batch}",
        dimensions=dimensions,
    )
    if not entities:
        raise MilvusIngestError(
            f"version {version} produced no ingestible entities: "
            f"{eligibility.errors[:5]}"
        )

    client = milvus_client or MilvusClient()
    backend = MilvusProviderBackend(
        client,
        index_type=index_type,
        batch_size=batch_size,
        collection_name=collection_name,
    )
    from .provider_indexes import ProviderIndexContract

    contract = ProviderIndexContract(
        model_id="beit3",
        collection_name=collection_name,
        dimension=dimensions["beit3"],
        metric="IP",
        normalized=True,
        backend="milvus",
    )
    entity_count_before = backend.prepare(contract, policy)

    upserted = 0
    for start in range(0, len(entities), batch_size):
        chunk = entities[start : start + batch_size]
        client.upsert_batch(chunk, collection_name=collection_name)
        upserted += len(chunk)

    backend.finalize(contract)
    entity_count_after = client.entity_count(collection_name)

    provider_entries = {
        provider: ProviderManifestEntry(
            provider_id=provider,
            dimension=dimensions[provider],
            metric="IP",
            normalized=bool(provider_manifests[provider].get("normalized", True)),
            checksum="",
            vector_file=f"{prefix}/batch={batch}/collection=*/embeddings/{provider}/",
            mapping_file=(
                f"{prefix}/batch={batch}/collection=*/embeddings/{provider}"
                "/feature_mapping.csv"
            ),
        )
        for provider in REQUIRED_PROVIDERS
    }
    write_manifest(
        build_manifest(
            collection_name=collection_name,
            entity_count=entity_count_after,
            keyframe_mapping_checksum="",
            providers=provider_entries,
            index_type=index_type,
            metric="IP",
            index_params={},
            dataset_version=version,
            state=STATE_FRESH,
        ),
        manifest_path,
    )

    LOGGER.info(
        "ingested %d entities from version %s into %s",
        upserted,
        version,
        collection_name,
    )
    return IngestResult(
        collection_name=collection_name,
        entity_count_before=entity_count_before,
        entity_count_after=entity_count_after,
        upserted=upserted,
        eligibility=eligibility,
        manifest_path=str(manifest_path),
        index_type=index_type,
    )
