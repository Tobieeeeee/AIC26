"""Consolidate staging JSON/records into canonical Parquet.

    staging JSON/records
      ↓
    Polars
      ↓
    schema normalization
    identity join/check
    dedup
    partition
      ↓
    Parquet
      ↓
    validate
      ↓
    promote
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .parquet_schemas import (
    SCHEMAS,
    SchemaField,
    validate_columns,
)

LOGGER = logging.getLogger(__name__)


class ConsolidationError(ValueError):
    """Consolidation failed due to schema or identity issues."""


def _read_json_records(source_dir: Path, glob: str = "*.json") -> list[dict[str, Any]]:
    """Read all JSON files under a directory into a flat list of records."""
    records: list[dict[str, Any]] = []
    if not source_dir.is_dir():
        raise FileNotFoundError(f"staging directory not found: {source_dir}")
    for path in sorted(source_dir.rglob(glob)):
        if path.name.startswith(".") or path.name in (
            "manifest.json", "errors.json", "warnings.json",
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                records.extend(payload)
            elif isinstance(payload, dict):
                records.append(payload)
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("skipping unreadable JSON %s: %s", path, exc)
    return records


def _validate_schema(
    records: list[dict[str, Any]],
    schema: tuple[SchemaField, ...],
    schema_name: str,
) -> None:
    """Validate that all records have the required columns."""
    if not records:
        raise ConsolidationError(f"no records to consolidate for {schema_name}")
    sample_keys = list(records[0].keys())
    missing = validate_columns(schema_name, sample_keys)
    if missing:
        raise ConsolidationError(
            f"{schema_name} records missing required columns: {missing}"
        )


def _dedup_records(
    records: list[dict[str, Any]],
    identity_columns: list[str],
) -> list[dict[str, Any]]:
    """Remove duplicate records based on identity columns.

    Keeps the last occurrence (latest write wins).
    """
    seen: dict[tuple, int] = {}
    for i, record in enumerate(records):
        key = tuple(record.get(col) for col in identity_columns)
        seen[key] = i
    indices = sorted(seen.values())
    return [records[i] for i in indices]


def consolidate_to_parquet(
    source_dir: Path,
    output_path: Path,
    schema_name: str,
    *,
    batch_id: str | None = None,
    collection_id: str | None = None,
    identity_columns: list[str] | None = None,
    json_glob: str = "*.json",
) -> dict[str, Any]:
    """Consolidate staging JSON into a single Parquet file.

    Returns a summary dict with record counts and the output path.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ConsolidationError(
            "polars is required for Parquet consolidation; "
            "install with: pip install polars"
        ) from exc

    if schema_name not in SCHEMAS:
        raise ConsolidationError(f"unknown schema: {schema_name}")

    schema = SCHEMAS[schema_name]
    records = _read_json_records(source_dir, json_glob)
    _validate_schema(records, schema, schema_name)

    # Dedup
    if identity_columns is not None:
        id_cols = identity_columns
    elif schema_name == "objects":
        # Object metadata is one row per detection, not one row per keyframe.
        # Include the class and coordinates so multiple detections on the same
        # keyframe survive consolidation while exact duplicate rows do not.
        id_cols = [
            "video_id",
            "keyframe_id",
            "class_id",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    else:
        id_cols = ["keyframe_id"]
    original_count = len(records)
    records = _dedup_records(records, id_cols)
    dedup_count = original_count - len(records)

    # Build DataFrame with schema enforcement
    schema_columns = [f.name for f in schema]
    clean_records = []
    for record in records:
        clean = {col: record.get(col) for col in schema_columns}
        # Inject batch/collection if provided and present in schema
        if batch_id and "batch_id" in clean:
            clean["batch_id"] = batch_id
        if collection_id and "collection_id" in clean:
            clean["collection_id"] = collection_id
        clean_records.append(clean)

    df = pl.DataFrame(clean_records)

    # Write Parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    summary = {
        "schema": schema_name,
        "source_dir": str(source_dir),
        "output_path": str(output_path),
        "total_records": len(records),
        "duplicates_removed": dedup_count,
        "columns": schema_columns,
        "batch_id": batch_id,
        "collection_id": collection_id,
    }
    LOGGER.info(
        "consolidated %d records (%d deduped) → %s",
        len(records), dedup_count, output_path,
    )
    return summary


def consolidate_partitioned(
    source_dir: Path,
    output_root: Path,
    schema_name: str,
    *,
    batch_id: str,
    collection_id: str,
    identity_columns: list[str] | None = None,
    json_glob: str = "*.json",
) -> dict[str, Any]:
    """Consolidate into a partitioned Parquet layout.

    Output: output_root/batch={batch_id}/collection={collection_id}/part-000.parquet
    """
    part_dir = output_root / f"batch={batch_id}" / f"collection={collection_id}"
    part_path = part_dir / "part-000.parquet"
    return consolidate_to_parquet(
        source_dir,
        part_path,
        schema_name,
        batch_id=batch_id,
        collection_id=collection_id,
        identity_columns=identity_columns,
        json_glob=json_glob,
    )
