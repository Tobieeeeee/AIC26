"""Opt-in real CUDA gate for the pinned RF-DETR Large production runtime.

Set ``MVR_RFDETR_CHECKPOINT`` and ``MVR_RFDETR_SMOKE_MAPPING`` to run. The
test deliberately selects only 20 existing keyframes and never performs a
full extraction.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import pytest

pl = pytest.importorskip("polars")
torch = pytest.importorskip("torch")

from mvr.enrichment.feature_extraction import (
    extract_objects,
    sort_keyframes_deterministically,
)
from mvr.enrichment.objects.rfdetr import RFDETRConfig, RFDETRDetector
from mvr.indexing.documents import build_search_document
from mvr.indexing.elasticsearch_index import build_mappings
from mvr.indexing.task05a import build_index_documents
from mvr.retrieval.object_search import build_object_filters


CHECKPOINT = os.environ.get("MVR_RFDETR_CHECKPOINT")
MAPPING = os.environ.get("MVR_RFDETR_SMOKE_MAPPING")

pytestmark = pytest.mark.skipif(
    not CHECKPOINT or not MAPPING,
    reason="set MVR_RFDETR_CHECKPOINT and MVR_RFDETR_SMOKE_MAPPING",
)


def _representative_rows(mapping: Path, count: int = 20) -> list[dict[str, str]]:
    """Choose one deterministic frame from each of multiple videos."""

    rows = sort_keyframes_deterministically(mapping)
    selected: list[dict[str, str]] = []
    seen_videos: set[str] = set()
    for row in rows:
        if row["video_id"] in seen_videos:
            continue
        seen_videos.add(row["video_id"])
        selected.append(row)
        if len(selected) == count:
            break
    assert len(selected) == count
    return selected


def _write_mapping(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assert_repeatable(
    first: list[dict], second: list[dict], *, confidence_abs: float, bbox_abs: float
) -> None:
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert left["class_id"] == right["class_id"]
        assert left["label"] == right["label"]
        assert left["confidence"] == pytest.approx(
            right["confidence"], abs=confidence_abs
        )
        for coordinate in ("x1", "y1", "x2", "y2"):
            assert left[coordinate] == pytest.approx(
                right[coordinate], abs=bbox_abs
            )


def test_rfdetr_large_real_cuda_data02_task05_smoke(tmp_path) -> None:
    import polars as pl

    from mvr.operations.consolidate import consolidate_to_parquet
    from mvr.operations.polars_api import prepare_task05_documents, scan_parquet
    from mvr.operations.promote import promote_file, validate_parquet

    checkpoint = Path(CHECKPOINT)
    source_mapping = Path(MAPPING)
    keyframe_root = source_mapping.parent
    selected = _representative_rows(source_mapping)
    smoke_mapping = tmp_path / "mapping" / "keyframe_mapping.csv"
    _write_mapping(smoke_mapping, selected)

    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    detector = RFDETRDetector(
        RFDETRConfig(checkpoint_path=checkpoint, device="cuda")
    )

    load_started = time.perf_counter()
    detector._load_model()
    load_seconds = time.perf_counter() - load_started
    assert detector.device == "cuda"
    assert len(detector.categories) == 80

    first_output = tmp_path / "first"
    torch.cuda.synchronize()
    first_started = time.perf_counter()
    extract_objects(
        "rfdetr_smoke",
        selected,
        detector,
        first_output,
        keyframe_root,
        smoke_mapping,
    )
    torch.cuda.synchronize()
    first_seconds = time.perf_counter() - first_started

    first_by_keyframe: dict[str, list[dict]] = {}
    for path in sorted((first_output / "objects_staging").rglob("*.json")):
        if path.name == "manifest.json":
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        if records:
            first_by_keyframe[records[0]["keyframe_id"]] = records
        else:
            first_by_keyframe[path.stem] = []

    second_by_keyframe: dict[str, list[dict]] = {}
    torch.cuda.synchronize()
    second_started = time.perf_counter()
    for row in selected:
        second_by_keyframe[row["keyframe_id"]] = detector.detect_keyframe(
            keyframe_root / row["image_path"],
            keyframe_id=row["keyframe_id"],
            video_id=row["video_id"],
            frame_id=int(row["frame_idx"]),
        )
    torch.cuda.synchronize()
    second_seconds = time.perf_counter() - second_started

    assert set(first_by_keyframe) == set(second_by_keyframe)
    for keyframe_id in sorted(first_by_keyframe):
        _assert_repeatable(
            first_by_keyframe[keyframe_id],
            second_by_keyframe[keyframe_id],
            confidence_abs=1e-5,
            bbox_abs=1e-3,
        )

    staging_parquet = tmp_path / "staging" / "objects.parquet"
    consolidation = consolidate_to_parquet(
        first_output / "objects_staging",
        staging_parquet,
        "objects",
    )
    checks = validate_parquet(staging_parquet, schema_name="objects", min_rows=1)
    corpus_parquet = tmp_path / "corpus" / "objects.parquet"
    promotion = promote_file(
        staging_parquet,
        corpus_parquet,
        schema_name="objects",
        min_rows=1,
    )
    assert promotion.success, promotion.error

    objects = scan_parquet(corpus_parquet).collect()
    assert objects.height == consolidation["total_records"]
    assert objects.select((pl.col("frame_id") == pl.col("frame_idx")).all()).item()
    mapping_frame = pl.DataFrame(
        [
            {
                "keyframe_id": row["keyframe_id"],
                "video_id_expected": row["video_id"],
                "frame_idx_expected": int(row["frame_idx"]),
            }
            for row in selected
        ]
    )
    joined = objects.join(mapping_frame, on="keyframe_id", how="inner")
    assert joined.height == objects.height
    assert joined.select(
        (
            (pl.col("video_id") == pl.col("video_id_expected"))
            & (pl.col("frame_id") == pl.col("frame_idx_expected"))
        ).all()
    ).item()

    keyframes_parquet = tmp_path / "corpus" / "keyframes.parquet"
    pl.DataFrame(
        [
            {
                "keyframe_id": row["keyframe_id"],
                "video_id": row["video_id"],
                "frame_idx": int(row["frame_idx"]),
            }
            for row in selected
        ]
    ).write_parquet(keyframes_parquet)
    polars_documents = prepare_task05_documents(
        keyframes_parquet,
        objects_path=corpus_parquet,
    )
    assert len(polars_documents) == len(selected)
    assert any(document.get("object_labels") for document in polars_documents)

    task05_output = tmp_path / "task05"
    task05 = build_index_documents(
        features_root=first_output,
        mapping_csv=smoke_mapping,
        output_root=task05_output,
        objects_root=first_output / "objects",
    )
    assert task05.document_count == len(selected)
    task05_documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((task05_output / "documents").rglob("*.json"))
    ]
    object_document = next(
        document for document in task05_documents if document["objects"]["counts"]
    )
    label, count = next(iter(object_document["objects"]["counts"].items()))
    assert build_object_filters([(label, count)]) == [
        {"range": {f"objects.{label}": {"gte": count}}}
    ]
    es_document = build_search_document(
        object_document,
        caption="",
        ocr_text="",
        objects=object_document["objects"]["counts"],
        clip_vector=[1.0, 0.0],
    )
    assert es_document.objects == object_document["objects"]["counts"]
    assert build_mappings(2)["properties"]["objects"]["properties"]["person"] == {
        "type": "integer"
    }

    summary = {
        "model": detector.provider_info,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "images": len(selected),
        "videos": len({row["video_id"] for row in selected}),
        "batch_size": 1,
        "load_seconds": load_seconds,
        "first_inference_seconds": first_seconds,
        "first_images_per_second": len(selected) / first_seconds,
        "second_inference_seconds": second_seconds,
        "second_images_per_second": len(selected) / second_seconds,
        "detections": objects.height,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "parquet_checks": checks,
        "repeatability": {"confidence_abs": 1e-5, "bbox_abs": 1e-3},
    }
    print("RFDETR_SMOKE_SUMMARY=" + json.dumps(summary, sort_keys=True))
