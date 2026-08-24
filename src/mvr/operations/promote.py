"""Staging → validate → promote → canonical corpus contract.

A failed worker must not corrupt canonical corpus. Promotion is:
- Schema check
- Identity check
- Count check where known
- Checksum where practical
- Idempotent rerun
- No partial canonical file on failure
- Manifest/status update only after successful canonical write
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class PromotionError(ValueError):
    """Promotion validation failed."""


class PromotionResult:
    """Result of a promote operation."""

    def __init__(
        self,
        *,
        success: bool,
        source: str,
        target: str,
        checks: dict[str, Any],
        error: str | None = None,
    ) -> None:
        self.success = success
        self.source = source
        self.target = target
        self.checks = checks
        self.error = error
        self.promoted_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source": self.source,
            "target": self.target,
            "checks": self.checks,
            "error": self.error,
            "promoted_at": self.promoted_at,
        }


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _check_file_exists(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def validate_parquet(
    path: Path,
    *,
    schema_name: str | None = None,
    expected_columns: list[str] | None = None,
    min_rows: int = 0,
    expected_row_count: int | None = None,
) -> dict[str, Any]:
    """Validate a Parquet file. Returns a checks dict.

    Raises PromotionError on failure.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise PromotionError("polars is required for Parquet validation") from exc

    checks: dict[str, Any] = {"path": str(path)}

    if not _check_file_exists(path):
        raise PromotionError(f"staging artifact not found or empty: {path}")
    checks["exists"] = True

    df = pl.read_parquet(path)
    checks["rows"] = len(df)
    checks["columns"] = df.columns

    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise PromotionError(
                f"staging Parquet missing columns: {sorted(missing)}"
            )
        checks["columns_valid"] = True

    if schema_name:
        from .parquet_schemas import validate_columns
        missing = validate_columns(schema_name, df.columns)
        if missing:
            raise PromotionError(
                f"staging Parquet missing required {schema_name} columns: {missing}"
            )
        checks["schema_valid"] = True

    if len(df) < min_rows:
        raise PromotionError(
            f"staging Parquet has {len(df)} rows, minimum {min_rows} required"
        )
    checks["min_rows_valid"] = True

    if expected_row_count is not None and len(df) != expected_row_count:
        raise PromotionError(
            f"staging Parquet has {len(df)} rows, expected {expected_row_count}"
        )
    if expected_row_count is not None:
        checks["row_count_valid"] = True

    # Identity check: no duplicate keyframe_id if that column exists
    if "keyframe_id" in df.columns:
        dupes = df.filter(df["keyframe_id"].is_duplicated())
        if len(dupes) > 0:
            dupe_ids = dupes["keyframe_id"].unique().to_list()[:5]
            raise PromotionError(
                f"staging Parquet has duplicate keyframe_id values: {dupe_ids}"
            )
        checks["identity_unique"] = True

    checks["sha256"] = _sha256_file(path)
    checks["size_bytes"] = path.stat().st_size
    return checks


def promote_file(
    staging_path: Path,
    corpus_path: Path,
    *,
    schema_name: str | None = None,
    expected_columns: list[str] | None = None,
    min_rows: int = 0,
    expected_row_count: int | None = None,
    manifest_path: Path | None = None,
) -> PromotionResult:
    """Validate and promote a staging file to canonical corpus.

    The promotion is atomic: uses temp file + rename so no partial corpus
    file exists on failure.
    """
    try:
        # Validate
        is_parquet = staging_path.suffix == ".parquet"
        checks: dict[str, Any] = {}
        if is_parquet:
            checks = validate_parquet(
                staging_path,
                schema_name=schema_name,
                expected_columns=expected_columns,
                min_rows=min_rows,
                expected_row_count=expected_row_count,
            )
        else:
            if not _check_file_exists(staging_path):
                raise PromotionError(f"staging artifact not found: {staging_path}")
            checks["exists"] = True
            checks["sha256"] = _sha256_file(staging_path)
            checks["size_bytes"] = staging_path.stat().st_size

        # Atomic copy to corpus
        corpus_path = Path(corpus_path)
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=corpus_path.parent,
            prefix=f".{corpus_path.name}.",
            suffix=".tmp",
        )
        import os
        temp_path = Path(temp_path_str)
        try:
            os.close(temp_fd)
            shutil.copy2(staging_path, temp_path)
            temp_path.replace(corpus_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        # Update manifest only after successful write
        if manifest_path:
            _update_manifest(manifest_path, staging_path, corpus_path, checks)

        result = PromotionResult(
            success=True,
            source=str(staging_path),
            target=str(corpus_path),
            checks=checks,
        )
        LOGGER.info("promoted %s → %s", staging_path, corpus_path)
        return result

    except PromotionError as exc:
        return PromotionResult(
            success=False,
            source=str(staging_path),
            target=str(corpus_path),
            checks=checks if "checks" in dir() else {},
            error=str(exc),
        )


def _update_manifest(
    manifest_path: Path,
    source: Path,
    target: Path,
    checks: dict[str, Any],
) -> None:
    """Append promotion record to a manifest JSON file."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append({
        "source": str(source),
        "target": str(target),
        "checks": checks,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    })
    manifest_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
