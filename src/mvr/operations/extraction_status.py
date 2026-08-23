"""Extraction status control plane.

Tracks per-keyframe extraction progress across all pipeline stages.
Workers can discover missing work without enumerating R2 directories.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parquet_schemas import (
    COL_ASR_STATUS,
    COL_BATCH_ID,
    COL_BEIT3_STATUS,
    COL_COLLECTION_ID,
    COL_INGEST_STATUS,
    COL_KEYFRAME_ID,
    COL_METACLIP2_STATUS,
    COL_OBJECT_STATUS,
    COL_OCR_STATUS,
    COL_PE_STATUS,
    COL_UPDATED_AT,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STALE,
    STATUS_TODO,
    VALID_STATUSES,
)

LOGGER = logging.getLogger(__name__)

# Feature columns in extraction status
FEATURE_COLUMNS = (
    COL_INGEST_STATUS,
    COL_OCR_STATUS,
    COL_ASR_STATUS,
    COL_OBJECT_STATUS,
    COL_BEIT3_STATUS,
    COL_METACLIP2_STATUS,
    COL_PE_STATUS,
)


def _require_polars():
    try:
        import polars as pl
        return pl
    except ImportError as exc:
        raise RuntimeError(
            "polars is required for extraction status; "
            "install with: pip install polars"
        ) from exc


def initialize_status(
    keyframe_ids: list[str],
    *,
    batch_id: str,
    collection_id: str,
    output_path: Path,
) -> Path:
    """Create an initial extraction status Parquet with all features at TODO.

    Does not overwrite if the file already exists.
    """
    pl = _require_polars()
    output_path = Path(output_path)
    if output_path.is_file():
        LOGGER.info("extraction status already exists: %s", output_path)
        return output_path

    now = datetime.now(timezone.utc).isoformat()
    records = [
        {
            COL_KEYFRAME_ID: kf_id,
            COL_BATCH_ID: batch_id,
            COL_COLLECTION_ID: collection_id,
            COL_INGEST_STATUS: STATUS_TODO,
            COL_OCR_STATUS: STATUS_TODO,
            COL_ASR_STATUS: STATUS_TODO,
            COL_OBJECT_STATUS: STATUS_TODO,
            COL_BEIT3_STATUS: STATUS_TODO,
            COL_METACLIP2_STATUS: STATUS_TODO,
            COL_PE_STATUS: STATUS_TODO,
            COL_UPDATED_AT: now,
        }
        for kf_id in keyframe_ids
    ]
    df = pl.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    LOGGER.info("initialized extraction status for %d keyframes", len(keyframe_ids))
    return output_path


def load_status(path: Path):
    """Load extraction status as a Polars DataFrame."""
    pl = _require_polars()
    return pl.read_parquet(path)


def scan_status(path: Path):
    """Lazy scan extraction status."""
    pl = _require_polars()
    return pl.scan_parquet(path)


def update_status(
    path: Path,
    keyframe_ids: list[str],
    feature: str,
    status: str,
) -> int:
    """Update the status of specific keyframes for a feature.

    Returns the number of rows updated.
    """
    pl = _require_polars()
    if feature not in FEATURE_COLUMNS:
        raise ValueError(
            f"unknown feature column: {feature}; "
            f"valid: {FEATURE_COLUMNS}"
        )
    if status not in VALID_STATUSES:
        raise ValueError(
            f"invalid status: {status}; valid: {sorted(VALID_STATUSES)}"
        )
    df = pl.read_parquet(path)
    now = datetime.now(timezone.utc).isoformat()
    kf_set = set(keyframe_ids)
    mask = df[COL_KEYFRAME_ID].is_in(list(kf_set))
    count = mask.sum()
    df = df.with_columns(
        pl.when(mask).then(pl.lit(status)).otherwise(pl.col(feature)).alias(feature),
        pl.when(mask).then(pl.lit(now)).otherwise(pl.col(COL_UPDATED_AT)).alias(COL_UPDATED_AT),
    )
    df.write_parquet(path)
    return count


def find_todo(path: Path, feature: str) -> list[str]:
    """Find keyframe_ids that still need processing for a feature."""
    pl = _require_polars()
    if feature not in FEATURE_COLUMNS:
        raise ValueError(f"unknown feature column: {feature}")
    df = pl.read_parquet(path)
    todo = df.filter(pl.col(feature) == STATUS_TODO)
    return sorted(todo[COL_KEYFRAME_ID].to_list())


def find_failed(path: Path, feature: str) -> list[str]:
    """Find keyframe_ids that failed processing for a feature."""
    pl = _require_polars()
    if feature not in FEATURE_COLUMNS:
        raise ValueError(f"unknown feature column: {feature}")
    df = pl.read_parquet(path)
    failed = df.filter(pl.col(feature) == STATUS_FAILED)
    return sorted(failed[COL_KEYFRAME_ID].to_list())


def summary(path: Path) -> dict[str, dict[str, int]]:
    """Return per-feature status counts."""
    pl = _require_polars()
    df = pl.read_parquet(path)
    result: dict[str, dict[str, int]] = {}
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            counts = df.group_by(col).len().sort(col)
            result[col] = {
                row[col]: row["len"] for row in counts.iter_rows(named=True)
            }
    return result
