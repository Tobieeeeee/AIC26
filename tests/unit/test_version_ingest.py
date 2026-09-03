"""Ingesting an enrichment version into Milvus end to end.

Covers the composition: validated version -> catalog -> downloaded vectors ->
entities -> Milvus upserts -> manifest. The Milvus side is exercised through a
recording double, so batching, prepare/finalize order and the manifest are
verified without a server.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from mvr.indexing.milvus_ingest import MilvusIngestError
from mvr.indexing.milvus_schema import (
    EXPECTED_DIMENSIONS,
    FIELD_ENTITY_ID,
)
from mvr.indexing.version_ingest import ingest_version_to_milvus

PROVIDERS = ("beit3", "metaclip2", "perception_encoder")
VERSION = "enrich-v1"
PREFIX = f"data/versions/enrichment/{VERSION}"
NEWLINE = chr(10)


def _npy(rows: int, dimension: int, offset: float) -> bytes:
    block = np.zeros((rows, dimension), dtype=np.float32)
    for i in range(rows):
        block[i, 0] = i + offset
    buffer = io.BytesIO()
    np.save(buffer, block)
    return buffer.getvalue()


def _mapping_csv(rows: list[dict]) -> bytes:
    columns = [
        "keyframe_id", "video_id", "feature_file", "feature_index",
        "frame_id", "shot_id", "timestamp", "image_path",
    ]
    lines = [",".join(columns)]
    lines.extend(",".join(str(row[column]) for column in columns) for row in rows)
    return (NEWLINE.join(lines) + NEWLINE).encode()


def _bucket(*, dimensions: dict[str, int] | None = None) -> dict[str, bytes]:
    dims = dimensions or dict(EXPECTED_DIMENSIONS)
    videos = ("L21_V001", "L21_V002")
    per_video = 2
    rows = [
        {
            "keyframe_id": f"{video_id}_S{i + 1:04d}_K01",
            "video_id": video_id,
            "feature_file": f"{video_id}.npy",
            "feature_index": i,
            "frame_id": i * 10,
            "shot_id": i + 1,
            "timestamp": f"{i}.5",
            "image_path": f"data/corpus/keyframes/L21/{video_id}/k{i}.jpg",
        }
        for video_id in videos
        for i in range(per_video)
    ]
    root = f"{PREFIX}/batch=1/collection=L21"
    marker = {
        "schema_version": "2.0",
        "status": "DONE",
        "run_id": VERSION,
        "batch_id": "batch1",
        "collection_id": "L21",
        "video_count": len(videos),
        "keyframe_count": len(rows),
        "providers": list(PROVIDERS),
        "features": ["objects", *PROVIDERS, "ocr", "asr"],
        "object_transport": "task05a/objects.tar.gz",
        "object_metadata": "metadata/objects/part-000.parquet",
        "ocr_transport": "task05a/ocr.tar.gz",
        "ocr_metadata": "metadata/ocr/part-000.parquet",
        "asr_transport": "task05a/asr.tar.gz",
        "asr_metadata": "metadata/asr/part-000.parquet",
    }
    objects: dict[str, bytes] = {
        f"{PREFIX}/_READY.json": json.dumps(
            {
                "schema_version": "2.0",
                "status": "READY",
                "run_id": VERSION,
                "batch_id": "batch1",
                "collections": ["L21"],
                "providers": list(PROVIDERS),
                "video_count": len(videos),
                "keyframe_count": len(rows),
            }
        ).encode(),
        f"{root}/_SUCCESS.json": json.dumps(marker).encode(),
        f"{root}/ocr/manifest.json": b"{}",
        f"{root}/asr/manifest.json": b"{}",
    }
    for feature in ("objects", "ocr", "asr"):
        objects[f"{root}/task05a/{feature}.tar.gz"] = b"archive"
        objects[f"{root}/metadata/{feature}/part-000.parquet"] = b"parquet"
    for provider in PROVIDERS:
        base = f"{root}/embeddings/{provider}"
        objects[f"{base}/manifest.json"] = json.dumps(
            {"model_id": provider, "dimension": dims[provider], "normalized": True}
        ).encode()
        objects[f"{base}/feature_mapping.csv"] = _mapping_csv(rows)
        for index, video_id in enumerate(videos):
            objects[f"{base}/{video_id}.npy"] = _npy(
                per_video, dims[provider], offset=index * 100.0
            )
    return objects


class _FakeR2:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.downloads: list[str] = []

    def list_objects(self, prefix, **kwargs):
        return [
            {"Key": key, "Size": len(value)}
            for key, value in sorted(self.objects.items())
            if key.startswith(prefix)
        ]

    def download_file(self, key, local_path, **kwargs):
        self.downloads.append(key)
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[key])
        return destination


class _FakeMilvus:
    def __init__(self) -> None:
        self.upserts: list[list[dict]] = []
        self.created_indexes: list[str] = []
        self.loaded = False
        self.flushed = False
        self.connected = False
        self.collections: set[str] = set()

    def connect(self) -> None:
        self.connected = True

    def collection_exists(self, name=None) -> bool:
        return name in self.collections

    def create_collection(self, schema, name=None, drop_existing=False) -> None:
        self.collections.add(name)

    def entity_count(self, name=None) -> int:
        return sum(len(batch) for batch in self.upserts)

    def upsert_batch(self, entities, collection_name=None):
        self.upserts.append(list(entities))
        return [entity[FIELD_ENTITY_ID] for entity in entities]

    def flush_collection(self, name=None) -> None:
        self.flushed = True

    def create_index(self, field, **kwargs) -> None:
        self.created_indexes.append(field)

    def load_collection(self, name=None) -> None:
        self.loaded = True


def test_every_catalog_keyframe_is_upserted_with_a_pair_keyed_entity_id(tmp_path) -> None:
    milvus = _FakeMilvus()

    result = ingest_version_to_milvus(
        r2_client=_FakeR2(_bucket()),
        version=VERSION,
        work_root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        milvus_client=milvus,
    )

    upserted_ids = {
        entity[FIELD_ENTITY_ID] for batch in milvus.upserts for entity in batch
    }
    assert upserted_ids == {
        "L21_V001/L21_V001_S0001_K01",
        "L21_V001/L21_V001_S0002_K01",
        "L21_V002/L21_V002_S0001_K01",
        "L21_V002/L21_V002_S0002_K01",
    }
    assert result.upserted == 4
    assert milvus.flushed and milvus.loaded


def test_upserts_are_split_by_batch_size(tmp_path) -> None:
    milvus = _FakeMilvus()

    ingest_version_to_milvus(
        r2_client=_FakeR2(_bucket()),
        version=VERSION,
        work_root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        milvus_client=milvus,
        batch_size=2,
    )

    assert [len(batch) for batch in milvus.upserts] == [2, 2]


def test_each_vector_file_is_downloaded_once_not_once_per_keyframe(tmp_path) -> None:
    r2 = _FakeR2(_bucket())

    ingest_version_to_milvus(
        r2_client=r2,
        version=VERSION,
        work_root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        milvus_client=_FakeMilvus(),
    )

    npy_downloads = [key for key in r2.downloads if key.endswith(".npy")]
    assert len(npy_downloads) == 6  # 3 providers x 2 videos, 4 keyframes total
    assert len(set(npy_downloads)) == 6


def test_manifest_records_the_version_it_was_built_from(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"

    ingest_version_to_milvus(
        r2_client=_FakeR2(_bucket()),
        version=VERSION,
        work_root=tmp_path,
        manifest_path=manifest_path,
        milvus_client=_FakeMilvus(),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == VERSION
    assert manifest["entity_count"] == 4
    assert set(manifest["providers"]) == set(PROVIDERS)


def test_a_provider_manifest_dimension_that_contradicts_the_schema_stops_ingest(
    tmp_path,
) -> None:
    dims = {**EXPECTED_DIMENSIONS, "metaclip2": 999}

    with pytest.raises(MilvusIngestError, match="metaclip2"):
        ingest_version_to_milvus(
            r2_client=_FakeR2(_bucket(dimensions=dims)),
            version=VERSION,
            work_root=tmp_path,
            manifest_path=tmp_path / "manifest.json",
            milvus_client=_FakeMilvus(),
        )
