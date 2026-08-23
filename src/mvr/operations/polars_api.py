"""Polars metadata access layer for canonical AIC26 corpus.

Provides lazy scanning, filtering, joining, and missing-feature discovery
over partitioned Parquet metadata. This module does NOT replace FAISS or
Elasticsearch — it is for metadata operations only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _require_polars():
    """Import and return polars, raising a clear error if unavailable."""
    try:
        import polars as pl
        return pl
    except ImportError as exc:
        raise RuntimeError(
            "polars is required for metadata operations; "
            "install with: pip install polars"
        ) from exc


def scan_parquet(path: Path | str):
    """Lazy scan a single Parquet file or a Hive-partitioned directory."""
    pl = _require_polars()
    path = Path(path)
    if path.is_file():
        return pl.scan_parquet(path)
    if path.is_dir():
        # Hive-partitioned: data/batch=1/collection=L21/part-000.parquet
        return pl.scan_parquet(
            path / "**/*.parquet",
            hive_partitioning=True,
        )
    raise FileNotFoundError(f"Parquet source not found: {path}")


def filter_by_batch(lf, batch_id: str):
    """Filter a LazyFrame by batch_id."""
    pl = _require_polars()
    return lf.filter(pl.col("batch_id") == batch_id)


def filter_by_collection(lf, collection_id: str):
    """Filter a LazyFrame by collection_id."""
    pl = _require_polars()
    return lf.filter(pl.col("collection_id") == collection_id)


def filter_by_video(lf, video_id: str):
    """Filter a LazyFrame by video_id."""
    pl = _require_polars()
    return lf.filter(pl.col("video_id") == video_id)


def find_duplicate_keyframe_ids(lf) -> list[str]:
    """Find duplicate keyframe_id values. Returns a list of duplicate IDs."""
    pl = _require_polars()
    df = lf.select("keyframe_id").collect()
    dupes = df.filter(df["keyframe_id"].is_duplicated())
    return sorted(dupes["keyframe_id"].unique().to_list())


def find_missing_feature(
    keyframes_lf,
    feature_lf,
    *,
    join_col: str = "keyframe_id",
):
    """Find keyframes missing from a feature dataset.

    Returns a LazyFrame of keyframe rows with no matching feature record.
    """
    pl = _require_polars()
    return keyframes_lf.join(
        feature_lf.select(join_col),
        on=join_col,
        how="anti",
    )


def find_missing_ocr(keyframes_path: Path, ocr_path: Path) -> list[str]:
    """Find keyframe_ids present in keyframes but missing OCR."""
    kf = scan_parquet(keyframes_path)
    ocr = scan_parquet(ocr_path)
    missing = find_missing_feature(kf, ocr).select("keyframe_id").collect()
    return sorted(missing["keyframe_id"].to_list())


def find_missing_asr(keyframes_path: Path, asr_path: Path) -> list[str]:
    """Find keyframe_ids present in keyframes but missing ASR."""
    kf = scan_parquet(keyframes_path)
    asr = scan_parquet(asr_path)
    missing = find_missing_feature(kf, asr).select("keyframe_id").collect()
    return sorted(missing["keyframe_id"].to_list())


def find_missing_objects(keyframes_path: Path, objects_path: Path) -> list[str]:
    """Find keyframe_ids present in keyframes but missing object detections."""
    kf = scan_parquet(keyframes_path)
    objs = scan_parquet(objects_path)
    missing = find_missing_feature(kf, objs).select("keyframe_id").collect()
    return sorted(missing["keyframe_id"].to_list())


def find_missing_embedding(
    keyframes_path: Path,
    embedding_mapping_path: Path,
) -> list[str]:
    """Find keyframe_ids missing from an embedding provider's mapping."""
    kf = scan_parquet(keyframes_path)
    emb = scan_parquet(embedding_mapping_path)
    missing = find_missing_feature(kf, emb).select("keyframe_id").collect()
    return sorted(missing["keyframe_id"].to_list())


def join_metadata(
    keyframes_path: Path,
    *,
    ocr_path: Path | None = None,
    asr_path: Path | None = None,
    objects_path: Path | None = None,
):
    """Join keyframe metadata with available feature datasets.

    Returns a LazyFrame with all available columns joined on keyframe_id.
    """
    pl = _require_polars()
    result = scan_parquet(keyframes_path)

    if ocr_path and Path(ocr_path).exists():
        ocr = scan_parquet(ocr_path)
        result = result.join(
            ocr.rename({"text": "ocr_text", "confidence": "ocr_confidence"}),
            on="keyframe_id",
            how="left",
        )
    if asr_path and Path(asr_path).exists():
        asr = scan_parquet(asr_path)
        # Avoid column conflicts
        asr_cols = {"text": "asr_text"}
        if "timestamp" in asr.columns:
            asr_cols["timestamp"] = "asr_timestamp"
        result = result.join(
            asr.rename(asr_cols),
            on="keyframe_id",
            how="left",
        )
    if objects_path and Path(objects_path).exists():
        # Objects have multiple rows per keyframe; aggregate
        objs = scan_parquet(objects_path)
        obj_agg = objs.group_by("keyframe_id").agg(
            pl.col("label").implode().alias("object_labels"),
            pl.count().alias("object_count"),
        )
        result = result.join(obj_agg, on="keyframe_id", how="left")

    return result


def prepare_task05_documents(
    keyframes_path: Path,
    *,
    ocr_path: Path | None = None,
    asr_path: Path | None = None,
    objects_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Prepare Task05A searchable documents from joined metadata.

    Each document contains: keyframe identity, OCR text, ASR text, and
    object metadata, ready for Elasticsearch indexing.
    """
    lf = join_metadata(
        keyframes_path,
        ocr_path=ocr_path,
        asr_path=asr_path,
        objects_path=objects_path,
    )
    df = lf.collect()
    documents: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        doc: dict[str, Any] = {
            "keyframe_id": row["keyframe_id"],
            "video_id": row["video_id"],
            "frame_idx": row.get("frame_idx"),
        }
        if "ocr_text" in row and row["ocr_text"]:
            doc["ocr_text"] = row["ocr_text"]
        if "asr_text" in row and row["asr_text"]:
            doc["asr_text"] = row["asr_text"]
        if "object_labels" in row and row["object_labels"]:
            doc["object_labels"] = row["object_labels"]
        if "object_count" in row and row["object_count"]:
            doc["object_count"] = row["object_count"]
        documents.append(doc)
    return documents
