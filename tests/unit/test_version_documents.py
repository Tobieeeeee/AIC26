"""Search documents built from an enrichment version.

The lexical side had the same identity flaw as the vector side: the
Elasticsearch document ``_id`` was the keyframe id alone, so two videos sharing
one organizer keyframe id overwrote each other.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import polars as pl
import pytest

from mvr.indexing.version_documents import build_documents_from_version

PROVIDERS = ("beit3", "metaclip2", "perception_encoder")
VERSION = "enrich-v1"
PREFIX = f"data/versions/enrichment/{VERSION}"
ROOT = f"{PREFIX}/batch=1/collection=L21"
NEWLINE = chr(10)


def _parquet(rows: list[dict]) -> bytes:
    buffer = io.BytesIO()
    pl.DataFrame(rows).write_parquet(buffer)
    return buffer.getvalue()


def _mapping_csv(rows: list[dict]) -> bytes:
    columns = [
        "keyframe_id", "video_id", "feature_file", "feature_index",
        "frame_id", "shot_id", "timestamp", "image_path",
    ]
    lines = [",".join(columns)]
    lines.extend(",".join(str(row[column]) for column in columns) for row in rows)
    return (NEWLINE.join(lines) + NEWLINE).encode()


def _bucket(
    *,
    identities: list[tuple[str, str]] | None = None,
    ocr: list[dict] | None = None,
    asr: list[dict] | None = None,
    objects: list[dict] | None = None,
) -> dict[str, bytes]:
    pairs = identities or [
        ("L21_V001", "L21_V001_S0001_K01"),
        ("L21_V002", "L21_V002_S0001_K01"),
    ]
    mapping_rows = [
        {
            "keyframe_id": keyframe_id,
            "video_id": video_id,
            "feature_file": f"{video_id}.npy",
            "feature_index": index,
            "frame_id": 10 * index,
            "shot_id": 1,
            "timestamp": f"{index}.5",
            "image_path": f"data/corpus/keyframes/L21/{video_id}/{keyframe_id}.jpg",
        }
        for index, (video_id, keyframe_id) in enumerate(pairs)
    ]
    marker = {
        "schema_version": "2.0",
        "status": "DONE",
        "run_id": VERSION,
        "batch_id": "batch1",
        "collection_id": "L21",
        "video_count": len({video_id for video_id, _ in pairs}),
        "keyframe_count": len(pairs),
        "providers": list(PROVIDERS),
        "features": ["objects", *PROVIDERS, "ocr", "asr"],
        "object_transport": "task05a/objects.tar.gz",
        "object_metadata": "metadata/objects/part-000.parquet",
        "ocr_transport": "task05a/ocr.tar.gz",
        "ocr_metadata": "metadata/ocr/part-000.parquet",
        "asr_transport": "task05a/asr.tar.gz",
        "asr_metadata": "metadata/asr/part-000.parquet",
    }
    bucket: dict[str, bytes] = {
        f"{PREFIX}/_READY.json": json.dumps(
            {
                "schema_version": "2.0",
                "status": "READY",
                "run_id": VERSION,
                "batch_id": "batch1",
                "collections": ["L21"],
                "providers": list(PROVIDERS),
                "video_count": marker["video_count"],
                "keyframe_count": len(pairs),
            }
        ).encode(),
        f"{ROOT}/_SUCCESS.json": json.dumps(marker).encode(),
        f"{ROOT}/ocr/manifest.json": b"{}",
        f"{ROOT}/asr/manifest.json": b"{}",
    }
    for feature in ("objects", "ocr", "asr"):
        bucket[f"{ROOT}/task05a/{feature}.tar.gz"] = b"archive"
    default_ocr = [
        {
            "keyframe_id": keyframe_id,
            "video_id": video_id,
            "frame_idx": 0,
            "text": f"ocr {video_id}",
            "confidence": 0.9,
        }
        for video_id, keyframe_id in pairs
    ]
    default_asr = [
        {
            "keyframe_id": keyframe_id,
            "video_id": video_id,
            "frame_idx": 0,
            "timestamp": 1.0,
            "text": f"asr {video_id}",
            "shot_id": 1,
        }
        for video_id, keyframe_id in pairs
    ]
    default_objects = [
        {
            "keyframe_id": keyframe_id,
            "video_id": video_id,
            "frame_id": 0,
            "frame_idx": 0,
            "label": "person",
            "class_id": 1,
            "confidence": 0.9,
            "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0,
        }
        for video_id, keyframe_id in pairs
    ]
    bucket[f"{ROOT}/metadata/ocr/part-000.parquet"] = _parquet(
        ocr if ocr is not None else default_ocr
    )
    bucket[f"{ROOT}/metadata/asr/part-000.parquet"] = _parquet(
        asr if asr is not None else default_asr
    )
    bucket[f"{ROOT}/metadata/objects/part-000.parquet"] = _parquet(
        objects if objects is not None else default_objects
    )
    for provider in PROVIDERS:
        base = f"{ROOT}/embeddings/{provider}"
        bucket[f"{base}/manifest.json"] = b"{}"
        bucket[f"{base}/feature_mapping.csv"] = _mapping_csv(mapping_rows)
        for video_id, _ in pairs:
            bucket[f"{base}/{video_id}.npy"] = b"vectors"
    return bucket


class _FakeR2:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)

    def list_objects(self, prefix, **kwargs):
        return [
            {"Key": key, "Size": len(value)}
            for key, value in sorted(self.objects.items())
            if key.startswith(prefix)
        ]

    def download_file(self, key, local_path, **kwargs):
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[key])
        return destination


def test_one_document_per_keyframe_merging_ocr_asr_and_objects(tmp_path) -> None:
    documents = build_documents_from_version(
        r2_client=_FakeR2(_bucket()), version=VERSION, work_root=tmp_path
    )

    assert len(documents) == 2
    first = next(d for d in documents if d["video_id"] == "L21_V001")
    assert first["frame_id"] == "L21_V001_S0001_K01"
    assert first["ocr_text"] == "ocr L21_V001"
    assert first["caption"] == "asr L21_V001"
    assert first["objects"] == {"person": 1}
    assert first["ts"] == pytest.approx(0.5)


def test_two_videos_sharing_a_keyframe_id_keep_their_own_text(tmp_path) -> None:
    pairs = [("L21_V001", "034"), ("L21_V002", "034")]

    documents = build_documents_from_version(
        r2_client=_FakeR2(_bucket(identities=pairs)),
        version=VERSION,
        work_root=tmp_path,
    )

    by_video = {document["video_id"]: document for document in documents}
    assert set(by_video) == {"L21_V001", "L21_V002"}
    assert by_video["L21_V001"]["ocr_text"] == "ocr L21_V001"
    assert by_video["L21_V002"]["ocr_text"] == "ocr L21_V002"
    assert {document["document_id"] for document in documents} == {
        "L21_V001/034",
        "L21_V002/034",
    }


def test_a_keyframe_without_text_still_produces_a_document(tmp_path) -> None:
    documents = build_documents_from_version(
        r2_client=_FakeR2(_bucket(ocr=[], asr=[], objects=[])),
        version=VERSION,
        work_root=tmp_path,
    )

    assert len(documents) == 2
    assert all(document["ocr_text"] == "" for document in documents)
    assert all(document["caption"] == "" for document in documents)
    assert all(document["objects"] == {} for document in documents)


def test_repeated_labels_are_counted_per_keyframe(tmp_path) -> None:
    pairs = [("L21_V001", "L21_V001_S0001_K01")]
    objects = [
        {
            "keyframe_id": "L21_V001_S0001_K01",
            "video_id": "L21_V001",
            "frame_id": 0, "frame_idx": 0,
            "label": label, "class_id": 1, "confidence": 0.9,
            "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0,
        }
        for label in ("person", "person", "car")
    ]

    documents = build_documents_from_version(
        r2_client=_FakeR2(_bucket(identities=pairs, objects=objects)),
        version=VERSION,
        work_root=tmp_path,
    )

    assert documents[0]["objects"] == {"person": 2, "car": 1}


def test_multiple_ocr_rows_for_one_keyframe_are_joined(tmp_path) -> None:
    pairs = [("L21_V001", "L21_V001_S0001_K01")]
    ocr = [
        {
            "keyframe_id": "L21_V001_S0001_K01",
            "video_id": "L21_V001",
            "frame_idx": 0,
            "text": text,
            "confidence": 0.9,
        }
        for text in ("hello", "world")
    ]

    documents = build_documents_from_version(
        r2_client=_FakeR2(_bucket(identities=pairs, ocr=ocr)),
        version=VERSION,
        work_root=tmp_path,
    )

    assert documents[0]["ocr_text"] == "hello world"
