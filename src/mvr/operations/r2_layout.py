"""Canonical R2 object-key layout for the AIC26 data lifecycle.

    data/
    ├── raw/batchN/videos/Lxx/video.mp4
    ├── staging/{feature}/{run_id}/...
    ├── versions/{feature}/{version}/...
    └── corpus/
        ├── dataset_manifest.json
        ├── ingest/
        ├── keyframes/
        ├── metadata/
        ├── features/
        ├── indexes/
        └── reports/
"""
from __future__ import annotations

from pathlib import PurePosixPath


# Top-level prefixes
RAW_PREFIX = "data/raw"
STAGING_PREFIX = "data/staging"
VERSIONS_PREFIX = "data/versions"
CORPUS_PREFIX = "data/corpus"


def raw_batch_segment(batch_id: str) -> str:
    """Normalize operator-facing ``1`` and layout-facing ``batch1`` alike."""
    value = str(batch_id).strip().strip("/")
    if not value:
        raise ValueError("batch_id must be non-empty")
    if value.lower().startswith("batch"):
        value = value[5:]
    if not value:
        raise ValueError("batch_id must include a value after 'batch'")
    return f"batch{value}"


def raw_video_prefix(batch_id: str, collection_id: str) -> str:
    """Object-key prefix containing one collection's raw videos."""
    return f"{RAW_PREFIX}/{raw_batch_segment(batch_id)}/videos/{collection_id}/"


def raw_video_key(batch_id: str, collection_id: str, video_id: str) -> str:
    """Object key for a raw video in R2."""
    return f"{raw_video_prefix(batch_id, collection_id)}{video_id}.mp4"


def staging_key(feature: str, run_id: str, *parts: str) -> str:
    """Object key for a staging artifact.

    feature: ingestion | ocr | asr | objects | embeddings/beit3 | etc.
    run_id: unique run identifier
    parts: additional path segments
    """
    segments = [STAGING_PREFIX, feature, run_id] + list(parts)
    return "/".join(segments)


def _require_segment(value: str, name: str) -> str:
    """Return a value that is safe to use as exactly one object-key segment."""
    segment = str(value).strip()
    if not segment:
        raise ValueError(f"{name} must be non-empty")
    if "/" in segment or segment in {".", ".."}:
        raise ValueError(f"{name} must be a single path segment: {value!r}")
    return segment


def version_key(feature: str, version: str, *parts: str) -> str:
    """Object key inside an immutable, run-scoped version tree.

    Versions live outside ``data/corpus`` on purpose: ``R2Sync.pull_corpus``
    pulls the corpus prefix blindly, so an unreleased version stored under it
    would be downloaded by every ``r2 pull --corpus``.  Only the pointer in
    ``dataset_manifest.json`` decides which version is active.
    """
    segments = [
        VERSIONS_PREFIX,
        _require_segment(feature, "feature"),
        _require_segment(version, "version"),
    ]
    segments.extend(parts)
    return "/".join(segments)


def enrichment_version_prefix(version: str) -> str:
    """Run-scoped root holding one immutable enrichment version."""
    return version_key("enrichment", version)


def corpus_key(*parts: str) -> str:
    """Object key for a canonical corpus artifact."""
    return "/".join([CORPUS_PREFIX] + list(parts))


# ---- Corpus sub-paths ----

def corpus_ingest_key(*parts: str) -> str:
    return corpus_key("ingest", *parts)


def corpus_keyframe_key(video_id: str, filename: str) -> str:
    return corpus_key("keyframes", video_id, filename)


def corpus_metadata_key(
    feature: str,
    batch_id: str,
    collection_id: str,
    part_name: str = "part-000.parquet",
) -> str:
    """Parquet metadata key with batch/collection partitioning.

    Example: data/corpus/metadata/ocr/batch=1/collection=L21/part-000.parquet
    """
    return corpus_key(
        "metadata", feature,
        f"batch={batch_id}", f"collection={collection_id}",
        part_name,
    )


def corpus_embedding_key(provider: str, *parts: str) -> str:
    """Key for a provider's embedding artifact.

    Example: data/corpus/features/embeddings/beit3/vectors.npy
    """
    return corpus_key("features", "embeddings", provider, *parts)


def corpus_index_key(*parts: str) -> str:
    return corpus_key("indexes", *parts)


def corpus_report_key(*parts: str) -> str:
    return corpus_key("reports", *parts)


# ---- Manifest keys ----

def dataset_manifest_key() -> str:
    return corpus_key("dataset_manifest.json")


def batch_manifest_key(batch_id: str) -> str:
    return corpus_ingest_key("manifests", f"{batch_id}.json")


def ingest_manifest_key() -> str:
    return corpus_ingest_key("ingest_manifest.json")


# ---- Extraction ----

def extraction_key(video_id: str) -> str:
    """Key prefix for all extraction outputs of a video."""
    return f"{STAGING_PREFIX}/extraction/{video_id}"


# ---- Utility ----

def parse_object_key(key: str) -> dict[str, str]:
    """Parse an R2 object key into its components."""
    parts = PurePosixPath(key).parts
    result: dict[str, str] = {"key": key}
    if len(parts) >= 2:
        result["prefix"] = parts[0]
        result["lifecycle"] = parts[1] if len(parts) > 1 else ""
    return result
