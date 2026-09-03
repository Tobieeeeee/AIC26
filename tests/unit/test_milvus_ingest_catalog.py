"""Entity building from an enrichment catalog.

This is the adapter between what an enrichment version publishes
(``feature_mapping.csv`` with ``video_id`` and ``feature_index``) and what
Milvus needs (one entity per keyframe with three aligned vectors).  It is the
layer where the historic identity bug lived: keying on ``keyframe_id`` alone
let one video's vector be written under another video's keyframe.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mvr.indexing.milvus_ingest import (
    MilvusIngestError,
    build_entities_from_catalog,
)
from mvr.indexing.milvus_schema import (
    FIELD_BATCH_ID,
    FIELD_BEIT3_VECTOR,
    FIELD_COLLECTION_ID,
    FIELD_ENTITY_ID,
    FIELD_KEYFRAME_ID,
    FIELD_METACLIP2_VECTOR,
    FIELD_PE_VECTOR,
    FIELD_VIDEO_ID,
)
from mvr.operations.enrichment_catalog import CatalogEntry

PROVIDERS = ("beit3", "metaclip2", "perception_encoder")
DIMENSIONS = {"beit3": 4, "metaclip2": 4, "perception_encoder": 4}


def _entry(video_id: str, keyframe_id: str, index: int) -> CatalogEntry:
    return CatalogEntry(
        video_id=video_id,
        keyframe_id=keyframe_id,
        collection_id="L21",
        shot_id=1,
        frame_id=10,
        timestamp=1.5,
        image_path=f"data/corpus/keyframes/L21/{video_id}/{keyframe_id}.jpg",
        vectors={
            provider: (f"{video_id}.npy", index) for provider in PROVIDERS
        },
    )


def _write_vectors(root: Path, video_id: str, rows: int, *, offset: float = 0.0) -> None:
    """Row i is [i + offset, 0, 0, 0] so a wrong row is detectable."""
    for provider in PROVIDERS:
        directory = root / "embeddings" / provider
        directory.mkdir(parents=True, exist_ok=True)
        block = np.zeros((rows, DIMENSIONS[provider]), dtype=np.float32)
        for i in range(rows):
            block[i, 0] = i + offset
        np.save(directory / f"{video_id}.npy", block)


def test_one_entity_per_catalog_entry_keyed_by_video_and_keyframe(tmp_path) -> None:
    _write_vectors(tmp_path, "L21_V001", rows=3)
    entries = [_entry("L21_V001", "L21_V001_S0001_K01", 0)]

    built, report = build_entities_from_catalog(
        entries=entries,
        vectors_root=tmp_path,
        batch_id="batch1",
        dimensions=DIMENSIONS,
    )

    assert len(built) == 1
    entity = built[0]
    assert entity[FIELD_ENTITY_ID] == "L21_V001/L21_V001_S0001_K01"
    assert entity[FIELD_VIDEO_ID] == "L21_V001"
    assert entity[FIELD_KEYFRAME_ID] == "L21_V001_S0001_K01"
    assert entity[FIELD_BATCH_ID] == "batch1"
    assert entity[FIELD_COLLECTION_ID] == "L21"
    assert len(entity[FIELD_BEIT3_VECTOR]) == 4
    assert report.rejected_missing_provider == 0


def test_the_vector_row_comes_from_feature_index_not_iteration_order(tmp_path) -> None:
    _write_vectors(tmp_path, "L21_V001", rows=5)
    entries = [_entry("L21_V001", "L21_V001_S0003_K01", 3)]

    built, _ = build_entities_from_catalog(
        entries=entries,
        vectors_root=tmp_path,
        batch_id="batch1",
        dimensions=DIMENSIONS,
    )

    assert built[0][FIELD_BEIT3_VECTOR][0] == pytest.approx(3.0)
    assert built[0][FIELD_METACLIP2_VECTOR][0] == pytest.approx(3.0)
    assert built[0][FIELD_PE_VECTOR][0] == pytest.approx(3.0)


def test_two_videos_sharing_a_keyframe_id_produce_two_distinct_entities(tmp_path) -> None:
    _write_vectors(tmp_path, "L21_V001", rows=2, offset=0.0)
    _write_vectors(tmp_path, "L21_V002", rows=2, offset=100.0)
    entries = [_entry("L21_V001", "034", 0), _entry("L21_V002", "034", 0)]

    built, _ = build_entities_from_catalog(
        entries=entries,
        vectors_root=tmp_path,
        batch_id="batch1",
        dimensions=DIMENSIONS,
    )

    assert {entity[FIELD_ENTITY_ID] for entity in built} == {
        "L21_V001/034",
        "L21_V002/034",
    }
    by_id = {entity[FIELD_ENTITY_ID]: entity for entity in built}
    assert by_id["L21_V001/034"][FIELD_BEIT3_VECTOR][0] == pytest.approx(0.0)
    assert by_id["L21_V002/034"][FIELD_BEIT3_VECTOR][0] == pytest.approx(100.0)


def test_a_dimension_mismatch_is_rejected_and_counted(tmp_path) -> None:
    _write_vectors(tmp_path, "L21_V001", rows=2)
    entries = [_entry("L21_V001", "L21_V001_S0001_K01", 0)]

    built, report = build_entities_from_catalog(
        entries=entries,
        vectors_root=tmp_path,
        batch_id="batch1",
        dimensions={**DIMENSIONS, "metaclip2": 8},
    )

    assert built == []
    assert report.rejected_dim_mismatch == 1
    assert any("metaclip2" in message for message in report.errors)


def test_a_non_finite_vector_is_rejected_and_counted(tmp_path) -> None:
    _write_vectors(tmp_path, "L21_V001", rows=2)
    broken = tmp_path / "embeddings" / "beit3" / "L21_V001.npy"
    block = np.load(broken)
    block[0, 0] = np.nan
    np.save(broken, block)
    entries = [_entry("L21_V001", "L21_V001_S0001_K01", 0)]

    built, report = build_entities_from_catalog(
        entries=entries,
        vectors_root=tmp_path,
        batch_id="batch1",
        dimensions=DIMENSIONS,
    )

    assert built == []
    assert report.rejected_invalid_vector == 1


def test_a_feature_index_past_the_end_of_the_vector_file_is_rejected(tmp_path) -> None:
    _write_vectors(tmp_path, "L21_V001", rows=2)
    entries = [_entry("L21_V001", "L21_V001_S0009_K01", 9)]

    built, report = build_entities_from_catalog(
        entries=entries,
        vectors_root=tmp_path,
        batch_id="batch1",
        dimensions=DIMENSIONS,
    )

    assert built == []
    assert report.rejected_missing_provider == 1
    assert any("9" in message for message in report.errors)


def test_a_missing_vector_file_stops_the_build(tmp_path) -> None:
    entries = [_entry("L21_V001", "L21_V001_S0001_K01", 0)]

    with pytest.raises(MilvusIngestError, match="L21_V001.npy"):
        build_entities_from_catalog(
            entries=entries,
            vectors_root=tmp_path,
            batch_id="batch1",
            dimensions=DIMENSIONS,
        )
