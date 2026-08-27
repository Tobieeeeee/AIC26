"""Tests for Parquet schemas, consolidation, promotion, and Polars API."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mvr.operations.parquet_schemas import (
    SCHEMAS,
    VALID_STATUSES,
    schema_columns,
    validate_columns,
)


# --------------------------------------------------------------------------
# Schema definition tests
# --------------------------------------------------------------------------

class TestParquetSchemas:
    def test_all_schemas_defined(self):
        expected = {"keyframe_mapping", "ocr", "asr", "objects", "extraction_status"}
        assert expected == set(SCHEMAS.keys())

    def test_keyframe_mapping_has_identity(self):
        cols = schema_columns("keyframe_mapping")
        assert "keyframe_id" in cols
        assert "video_id" in cols
        assert "frame_idx" in cols

    def test_ocr_schema_columns(self):
        cols = schema_columns("ocr")
        assert "keyframe_id" in cols
        assert "text" in cols

    def test_asr_schema_columns(self):
        cols = schema_columns("asr")
        assert "keyframe_id" in cols
        assert "text" in cols

    def test_objects_schema_columns(self):
        cols = schema_columns("objects")
        for col in ("keyframe_id", "frame_id", "frame_idx", "label", "confidence", "x1", "y1", "x2", "y2"):
            assert col in cols

    def test_extraction_status_columns(self):
        cols = schema_columns("extraction_status")
        for col in ("keyframe_id", "ingest_status", "ocr_status", "beit3_status"):
            assert col in cols

    def test_validate_columns_finds_missing(self):
        missing = validate_columns("ocr", ["keyframe_id", "video_id"])
        assert "frame_idx" in missing

    def test_validate_columns_passes(self):
        all_cols = schema_columns("ocr")
        missing = validate_columns("ocr", all_cols)
        assert missing == []

    def test_valid_statuses(self):
        assert "TODO" in VALID_STATUSES
        assert "DONE" in VALID_STATUSES
        assert "FAILED" in VALID_STATUSES


# --------------------------------------------------------------------------
# Consolidation tests (requires polars)
# --------------------------------------------------------------------------

polars = pytest.importorskip("polars")


class TestConsolidation:
    def _make_staging_jsons(self, tmp_path: Path, schema_name: str = "ocr"):
        staging = tmp_path / "staging"
        staging.mkdir()
        records = [
            {"keyframe_id": "kf001", "video_id": "V01", "frame_idx": 100, "text": "hello", "confidence": 0.9},
            {"keyframe_id": "kf002", "video_id": "V01", "frame_idx": 200, "text": "world", "confidence": 0.8},
        ]
        (staging / "kf001.json").write_text(json.dumps(records[0]))
        (staging / "kf002.json").write_text(json.dumps(records[1]))
        return staging

    def test_consolidate_basic(self, tmp_path):
        from mvr.operations.consolidate import consolidate_to_parquet
        staging = self._make_staging_jsons(tmp_path)
        output = tmp_path / "output.parquet"
        result = consolidate_to_parquet(staging, output, "ocr")
        assert result["total_records"] == 2
        assert output.is_file()
        df = polars.read_parquet(output)
        assert len(df) == 2
        assert "keyframe_id" in df.columns

    def test_consolidate_dedup(self, tmp_path):
        from mvr.operations.consolidate import consolidate_to_parquet
        staging = tmp_path / "staging"
        staging.mkdir()
        # Two records with same keyframe_id
        (staging / "a.json").write_text(json.dumps(
            {"keyframe_id": "kf001", "video_id": "V01", "frame_idx": 100, "text": "old", "confidence": 0.5}
        ))
        (staging / "b.json").write_text(json.dumps(
            {"keyframe_id": "kf001", "video_id": "V01", "frame_idx": 100, "text": "new", "confidence": 0.9}
        ))
        output = tmp_path / "dedup.parquet"
        result = consolidate_to_parquet(staging, output, "ocr")
        assert result["duplicates_removed"] == 1
        assert result["total_records"] == 1

    def test_objects_preserve_multiple_detections_per_keyframe(self, tmp_path):
        import json
        import polars as pl

        from mvr.operations.consolidate import consolidate_to_parquet

        staging = tmp_path / "objects"
        staging.mkdir()
        common = {
            "keyframe_id": "kf001",
            "video_id": "V01",
            "frame_id": 100,
            "frame_idx": 100,
        }
        (staging / "kf001.json").write_text(
            json.dumps(
                [
                    {**common, "label": "person", "class_id": 1, "confidence": 0.9, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
                    {**common, "label": "person", "class_id": 1, "confidence": 0.8, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
                    {**common, "video_id": "V02", "label": "person", "class_id": 1, "confidence": 0.7, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
                ]
            ),
            encoding="utf-8",
        )

        output = tmp_path / "objects.parquet"
        result = consolidate_to_parquet(staging, output, "objects")

        assert result["total_records"] == 3
        assert pl.read_parquet(output).height == 3

    def test_consolidate_empty_raises(self, tmp_path):
        from mvr.operations.consolidate import ConsolidationError, consolidate_to_parquet
        staging = tmp_path / "empty_staging"
        staging.mkdir()
        with pytest.raises(ConsolidationError, match="no records"):
            consolidate_to_parquet(staging, tmp_path / "out.parquet", "ocr")


# --------------------------------------------------------------------------
# Promotion tests (requires polars)
# --------------------------------------------------------------------------

class TestPromotion:
    def _make_valid_parquet(self, tmp_path: Path) -> Path:
        df = polars.DataFrame({
            "keyframe_id": ["kf001", "kf002"],
            "video_id": ["V01", "V01"],
            "frame_idx": [100, 200],
            "text": ["hello", "world"],
            "confidence": [0.9, 0.8],
        })
        path = tmp_path / "staging.parquet"
        df.write_parquet(path)
        return path

    def test_promote_success(self, tmp_path):
        from mvr.operations.promote import promote_file
        staging = self._make_valid_parquet(tmp_path)
        corpus = tmp_path / "corpus" / "ocr.parquet"
        result = promote_file(staging, corpus, schema_name="ocr")
        assert result.success
        assert corpus.is_file()
        df = polars.read_parquet(corpus)
        assert len(df) == 2

    def test_promote_idempotent(self, tmp_path):
        from mvr.operations.promote import promote_file
        staging = self._make_valid_parquet(tmp_path)
        corpus = tmp_path / "corpus" / "ocr.parquet"
        r1 = promote_file(staging, corpus, schema_name="ocr")
        r2 = promote_file(staging, corpus, schema_name="ocr")
        assert r1.success and r2.success
        assert r1.checks["sha256"] == r2.checks["sha256"]

    def test_promote_duplicate_identity_fails(self, tmp_path):
        from mvr.operations.promote import promote_file
        df = polars.DataFrame({
            "keyframe_id": ["kf001", "kf001"],  # duplicate!
            "video_id": ["V01", "V01"],
            "frame_idx": [100, 100],
            "text": ["a", "b"],
            "confidence": [0.9, 0.8],
        })
        staging = tmp_path / "dupe.parquet"
        df.write_parquet(staging)
        result = promote_file(staging, tmp_path / "corpus.parquet", schema_name="ocr")
        assert not result.success
        assert "duplicate" in result.error.lower()

    def test_promote_missing_file_fails(self, tmp_path):
        from mvr.operations.promote import promote_file
        result = promote_file(
            tmp_path / "nonexistent.parquet",
            tmp_path / "corpus.parquet",
        )
        assert not result.success

    def test_promote_updates_manifest(self, tmp_path):
        from mvr.operations.promote import promote_file
        staging = self._make_valid_parquet(tmp_path)
        manifest = tmp_path / "manifest.json"
        promote_file(staging, tmp_path / "corpus.parquet",
                     schema_name="ocr", manifest_path=manifest)
        assert manifest.is_file()
        data = json.loads(manifest.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert "promoted_at" in data[0]

    def test_failed_validation_does_not_alter_corpus(self, tmp_path):
        from mvr.operations.promote import promote_file
        # Create existing corpus file
        corpus = tmp_path / "corpus.parquet"
        corpus.write_text("existing")
        original = corpus.read_text()
        # Try to promote nonexistent file
        result = promote_file(tmp_path / "bad.parquet", corpus)
        assert not result.success
        assert corpus.read_text() == original  # unchanged


# --------------------------------------------------------------------------
# Extraction status tests (requires polars)
# --------------------------------------------------------------------------

class TestExtractionStatus:
    def test_initialize_and_load(self, tmp_path):
        from mvr.operations.extraction_status import initialize_status, load_status
        path = tmp_path / "status.parquet"
        initialize_status(["kf001", "kf002"], batch_id="1",
                          collection_id="L21", output_path=path)
        assert path.is_file()
        df = load_status(path)
        assert len(df) == 2
        assert all(df["ingest_status"] == "TODO")

    def test_update_status(self, tmp_path):
        from mvr.operations.extraction_status import (
            initialize_status, load_status, update_status,
        )
        path = tmp_path / "status.parquet"
        initialize_status(["kf001", "kf002"], batch_id="1",
                          collection_id="L21", output_path=path)
        count = update_status(path, ["kf001"], "ocr_status", "DONE")
        assert count == 1
        df = load_status(path)
        ocr_vals = dict(zip(df["keyframe_id"].to_list(), df["ocr_status"].to_list()))
        assert ocr_vals["kf001"] == "DONE"
        assert ocr_vals["kf002"] == "TODO"

    def test_find_todo(self, tmp_path):
        from mvr.operations.extraction_status import (
            find_todo, initialize_status, update_status,
        )
        path = tmp_path / "status.parquet"
        initialize_status(["kf001", "kf002", "kf003"], batch_id="1",
                          collection_id="L21", output_path=path)
        update_status(path, ["kf001"], "beit3_status", "DONE")
        todo = find_todo(path, "beit3_status")
        assert "kf001" not in todo
        assert "kf002" in todo
        assert "kf003" in todo

    def test_summary(self, tmp_path):
        from mvr.operations.extraction_status import (
            initialize_status, summary, update_status,
        )
        path = tmp_path / "status.parquet"
        initialize_status(["kf001", "kf002"], batch_id="1",
                          collection_id="L21", output_path=path)
        update_status(path, ["kf001"], "ocr_status", "DONE")
        s = summary(path)
        assert "ocr_status" in s
        assert s["ocr_status"]["DONE"] == 1
        assert s["ocr_status"]["TODO"] == 1
