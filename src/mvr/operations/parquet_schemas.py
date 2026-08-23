"""Canonical Parquet schemas for the AIC26 corpus metadata.

All structured metadata uses Parquet. Schemas are defined here to ensure
consistent column names, types, and validation across the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Column name constants to avoid typos
COL_KEYFRAME_ID = "keyframe_id"
COL_VIDEO_ID = "video_id"
COL_COLLECTION_ID = "collection_id"
COL_BATCH_ID = "batch_id"
COL_SHOT_ID = "shot_id"
COL_ORDINAL = "ordinal"
COL_FRAME_IDX = "frame_idx"
COL_TIMESTAMP = "timestamp"
COL_IMAGE_PATH = "image_path"
COL_VIDEO_OBJECT_KEY = "video_object_key"

# OCR-specific
COL_TEXT = "text"
COL_CONFIDENCE = "confidence"

# Object-specific
COL_LABEL = "label"
COL_CLASS_ID = "class_id"
COL_X1 = "x1"
COL_Y1 = "y1"
COL_X2 = "x2"
COL_Y2 = "y2"

# Extraction status
COL_INGEST_STATUS = "ingest_status"
COL_OCR_STATUS = "ocr_status"
COL_ASR_STATUS = "asr_status"
COL_OBJECT_STATUS = "object_status"
COL_BEIT3_STATUS = "beit3_status"
COL_METACLIP2_STATUS = "metaclip2_status"
COL_PE_STATUS = "pe_status"
COL_UPDATED_AT = "updated_at"

# Status enum values
STATUS_TODO = "TODO"
STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"
STATUS_STALE = "STALE"
VALID_STATUSES = frozenset({STATUS_TODO, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED, STATUS_STALE})


@dataclass(frozen=True)
class SchemaField:
    name: str
    dtype: str  # polars dtype name: Utf8, Int64, Float64, etc.
    nullable: bool = False
    description: str = ""


# --------------------------------------------------------------------------
# Schema definitions
# --------------------------------------------------------------------------

KEYFRAME_MAPPING_SCHEMA = (
    SchemaField(COL_KEYFRAME_ID, "Utf8", description="Unique keyframe identifier"),
    SchemaField(COL_VIDEO_ID, "Utf8", description="Source video identifier"),
    SchemaField(COL_COLLECTION_ID, "Utf8", description="Collection (e.g. L21)"),
    SchemaField(COL_BATCH_ID, "Utf8", description="Batch identifier"),
    SchemaField(COL_SHOT_ID, "Int64", nullable=True, description="Shot index within video"),
    SchemaField(COL_ORDINAL, "Int64", description="Keyframe ordinal within shot/video"),
    SchemaField(COL_FRAME_IDX, "Int64", description="Zero-based frame index in source video"),
    SchemaField(COL_TIMESTAMP, "Float64", description="Timestamp in seconds"),
    SchemaField(COL_IMAGE_PATH, "Utf8", nullable=True, description="Local or object path to keyframe image"),
    SchemaField(COL_VIDEO_OBJECT_KEY, "Utf8", nullable=True, description="R2 object key for source video"),
)

OCR_SCHEMA = (
    SchemaField(COL_KEYFRAME_ID, "Utf8"),
    SchemaField(COL_VIDEO_ID, "Utf8"),
    SchemaField(COL_FRAME_IDX, "Int64"),
    SchemaField(COL_TEXT, "Utf8", nullable=True, description="Consolidated OCR text"),
    SchemaField(COL_CONFIDENCE, "Float64", nullable=True, description="Average OCR confidence"),
)

ASR_SCHEMA = (
    SchemaField(COL_KEYFRAME_ID, "Utf8"),
    SchemaField(COL_VIDEO_ID, "Utf8"),
    SchemaField(COL_FRAME_IDX, "Int64"),
    SchemaField(COL_TIMESTAMP, "Float64"),
    SchemaField(COL_TEXT, "Utf8", nullable=True, description="Transcribed speech text"),
    SchemaField(COL_SHOT_ID, "Int64", nullable=True),
)

OBJECT_SCHEMA = (
    SchemaField(COL_KEYFRAME_ID, "Utf8"),
    SchemaField(COL_VIDEO_ID, "Utf8"),
    SchemaField(COL_FRAME_IDX, "Int64"),
    SchemaField(COL_LABEL, "Utf8", description="Object class label"),
    SchemaField(COL_CLASS_ID, "Int64", nullable=True, description="Object class ID"),
    SchemaField(COL_CONFIDENCE, "Float64", description="Detection confidence"),
    SchemaField(COL_X1, "Float64"),
    SchemaField(COL_Y1, "Float64"),
    SchemaField(COL_X2, "Float64"),
    SchemaField(COL_Y2, "Float64"),
)

EXTRACTION_STATUS_SCHEMA = (
    SchemaField(COL_KEYFRAME_ID, "Utf8"),
    SchemaField(COL_BATCH_ID, "Utf8"),
    SchemaField(COL_COLLECTION_ID, "Utf8"),
    SchemaField(COL_INGEST_STATUS, "Utf8"),
    SchemaField(COL_OCR_STATUS, "Utf8"),
    SchemaField(COL_ASR_STATUS, "Utf8"),
    SchemaField(COL_OBJECT_STATUS, "Utf8"),
    SchemaField(COL_BEIT3_STATUS, "Utf8"),
    SchemaField(COL_METACLIP2_STATUS, "Utf8"),
    SchemaField(COL_PE_STATUS, "Utf8"),
    SchemaField(COL_UPDATED_AT, "Utf8", description="ISO 8601 timestamp"),
)

SCHEMAS: dict[str, tuple[SchemaField, ...]] = {
    "keyframe_mapping": KEYFRAME_MAPPING_SCHEMA,
    "ocr": OCR_SCHEMA,
    "asr": ASR_SCHEMA,
    "objects": OBJECT_SCHEMA,
    "extraction_status": EXTRACTION_STATUS_SCHEMA,
}


def schema_columns(schema_name: str) -> list[str]:
    """Return column names for a named schema."""
    return [f.name for f in SCHEMAS[schema_name]]


def validate_columns(schema_name: str, columns: list[str]) -> list[str]:
    """Return list of missing required columns."""
    expected = {f.name for f in SCHEMAS[schema_name] if not f.nullable}
    return sorted(expected - set(columns))


def polars_schema(schema_name: str) -> dict[str, Any]:
    """Return a Polars-compatible schema dict.

    Requires polars to be importable. The caller should handle ImportError.
    """
    import polars as pl

    dtype_map = {
        "Utf8": pl.Utf8,
        "Int64": pl.Int64,
        "Float64": pl.Float64,
    }
    return {f.name: dtype_map[f.dtype] for f in SCHEMAS[schema_name]}
