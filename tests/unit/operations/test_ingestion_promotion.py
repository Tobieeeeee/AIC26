from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from mvr.operations.ingestion_promotion import (
    IngestionPromotionError,
    SourceAssignment,
    discover_staged_videos,
    parse_source_assignment,
    prepare_promotion_bundle,
    upload_prepared_bundle,
)


class FakeR2Client:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.upload_order: list[str] = []

    def list_objects(self, prefix: str, **_: object) -> list[dict[str, object]]:
        return [
            {"Key": key, "Size": len(value), "ETag": f'"etag-{len(value)}"'}
            for key, value in sorted(self.objects.items())
            if key.startswith(prefix)
        ]

    def object_exists(self, key: str) -> bool:
        return key in self.objects

    def download_file(self, key: str, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.objects[key])
        return path

    def upload_file(
        self, path: Path, key: str, *, content_type: str | None = None
    ) -> dict[str, object]:
        del content_type
        value = Path(path).read_bytes()
        self.objects[key] = value
        self.upload_order.append(key)
        return {"key": key, "size": len(value), "sha256": "test"}


def _csv_bytes(columns: list[str], rows: list[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _video_artifacts(
    run_id: str, collection: str, video_id: str, keyframe_id: str
) -> dict[str, bytes]:
    prefix = (
        f"data/staging/ingestion/{run_id}/batch=1/"
        f"collection={collection}/{video_id}/"
    )
    image_relative = f"keyframes/{video_id}/{keyframe_id}.jpg"
    mapping = _csv_bytes(
        [
            "keyframe_id",
            "video_id",
            "shot_id",
            "ordinal",
            "frame_idx",
            "timestamp",
            "image_path",
        ],
        [
            {
                "keyframe_id": keyframe_id,
                "video_id": video_id,
                "shot_id": 1,
                "ordinal": 1,
                "frame_idx": 12,
                "timestamp": 0.48,
                "image_path": image_relative,
            }
        ],
    )
    shot = {
        "video_id": video_id,
        "method": "transnetv2",
        "shot_id": 1,
        "start_frame": 0,
        "end_frame": 24,
        "fps": 25.0,
        "start_time": 0.0,
        "end_time": 1.0,
        "duration": 1.0,
        "frame_count": 25,
    }
    shots = {
        "schema_version": "1.0",
        "shots": [shot],
        "keyframes": [
            {
                "keyframe_id": keyframe_id,
                "video_id": video_id,
                "shot_id": 1,
                "ordinal": 1,
                "frame_idx": 12,
                "timestamp": 0.48,
                "image_path": image_relative,
            }
        ],
    }
    selector = {
        "schema_version": "1.0",
        "selector_mode": "hybrid_temporal_content",
        "config": {"max_temporal_gap_sec": 8.0},
        "metaclip_model": "facebook/metaclip-2-worldwide-b16",
        "metaclip_revision": "revision",
        "selected_keyframe_count": 1,
    }
    video_metadata = _csv_bytes(
        [
            "video_id",
            "video_path",
            "duration_seconds",
            "fps",
            "frame_count",
            "width",
            "height",
        ],
        [
            {
                "video_id": video_id,
                "video_path": f"C:/{video_id}.mp4",
                "duration_seconds": 1.0,
                "fps": 25.0,
                "frame_count": 25,
                "width": 1920,
                "height": 1080,
            }
        ],
    )
    artifacts = {
        "ingest_manifest.json": json.dumps(
            {"video_count": 1, "shot_count": 1, "keyframe_count": 1}
        ).encode(),
        "keyframe_mapping.csv": mapping,
        "selector_manifest.json": json.dumps(selector).encode(),
        "shots.json": json.dumps(shots).encode(),
        "video_metadata.csv": video_metadata,
        image_relative: b"jpeg",
    }
    artifact_bytes = sum(len(value) for value in artifacts.values())
    marker = {
        "schema_version": "1.0",
        "status": "DONE",
        "run_id": run_id,
        "batch_id": "1",
        "collection_id": collection,
        "video_id": video_id,
        "artifact_count": len(artifacts),
        "artifact_bytes": artifact_bytes,
    }
    return {
        **{prefix + name: value for name, value in artifacts.items()},
        prefix + "_SUCCESS.json": json.dumps(marker).encode(),
    }


def _complete_client() -> FakeR2Client:
    objects = {
        "data/raw/batch1/videos/L21/L21_V001.mp4": b"video-1",
        "data/raw/batch1/videos/L22/L22_V001.mp4": b"video-2",
    }
    objects.update(_video_artifacts("local", "L21", "L21_V001", "L21_V001_S0001_K01"))
    objects.update(_video_artifacts("vm", "L22", "L22_V001", "L22_V001_S0001_K01"))
    return FakeR2Client(objects)


def test_parse_source_assignment() -> None:
    assert parse_source_assignment("run-1:L21,L22") == SourceAssignment(
        "run-1", ("L21", "L22")
    )
    with pytest.raises(IngestionPromotionError):
        parse_source_assignment("run-1")


def test_discovery_fails_when_a_raw_video_has_no_success_marker() -> None:
    client = _complete_client()
    del client.objects[
        "data/staging/ingestion/vm/batch=1/collection=L22/L22_V001/_SUCCESS.json"
    ]

    with pytest.raises(IngestionPromotionError, match="not complete"):
        discover_staged_videos(
            client,
            batch_id="1",
            sources=[
                SourceAssignment("local", ("L21",)),
                SourceAssignment("vm", ("L22",)),
            ],
        )


def test_prepare_consolidates_runs_and_rewrites_canonical_paths(tmp_path: Path) -> None:
    client = _complete_client()
    bundle = prepare_promotion_bundle(
        client,
        batch_id="1",
        sources=[
            SourceAssignment("local", ("L21",)),
            SourceAssignment("vm", ("L22",)),
        ],
        promotion_id="batch1-test",
        work_root=tmp_path,
        workers=2,
    )

    assert bundle.report["summary"] == {
        "video_count": 2,
        "shot_count": 2,
        "keyframe_count": 2,
        "legacy_success_marker_count": 0,
    }
    assert bundle.report["safe_to_activate"] is True
    with (bundle.root / "ingest/keyframe_mapping.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["video_id"] for row in rows] == ["L21_V001", "L22_V001"]
    assert rows[0]["image_path"] == (
        "data/corpus/keyframes/L21/L21_V001/L21_V001_S0001_K01.jpg"
    )
    assert rows[0]["video_object_key"] == (
        "data/raw/batch1/videos/L21/L21_V001.mp4"
    )
    assert "promotion_report.json" not in bundle.report["checksums"]
    assert "_READY.json" not in bundle.report["checksums"]


def test_prepare_accepts_stale_selector_cache_not_counted_by_current_marker(
    tmp_path: Path,
) -> None:
    client = _complete_client()
    cache_key = (
        "data/staging/ingestion/local/batch=1/collection=L21/L21_V001/"
        "selector_cache/L21_V001/candidate.jpg"
    )
    client.objects[cache_key] = b"stale-cache"

    bundle = prepare_promotion_bundle(
        client,
        batch_id="1",
        sources=[
            SourceAssignment("local", ("L21",)),
            SourceAssignment("vm", ("L22",)),
        ],
        promotion_id="cache-test",
        work_root=tmp_path,
        workers=2,
    )

    assert bundle.report["summary"]["legacy_success_marker_count"] == 0


def test_existing_downstream_features_block_activation(tmp_path: Path) -> None:
    client = _complete_client()
    existing_mapping = _csv_bytes(
        ["keyframe_id", "video_id", "frame_idx", "timestamp"],
        [
            {
                "keyframe_id": "L21_V001_S0001_K01",
                "video_id": "L21_V001",
                "frame_idx": 9,
                "timestamp": 0.36,
            }
        ],
    )
    client.objects["data/corpus/ingest/keyframe_mapping.csv"] = existing_mapping
    client.objects["data/corpus/dataset_manifest.json"] = json.dumps(
        {
            "collections": {
                "L21": {
                    "coverage": {"objects": 1, "asr": 1},
                    "embeddings": {"metaclip2": {"count": 1}},
                }
            }
        }
    ).encode()

    bundle = prepare_promotion_bundle(
        client,
        batch_id="1",
        sources=[
            SourceAssignment("local", ("L21",)),
            SourceAssignment("vm", ("L22",)),
        ],
        promotion_id="impact-test",
        work_root=tmp_path,
        workers=2,
    )

    impact = bundle.report["canonical_impact"]
    assert bundle.report["safe_to_activate"] is False
    assert impact["stale_downstream_collections"] == ["L21"]
    assert impact["changed_overlapping_identities"] == 1


def test_upload_writes_ready_marker_last(tmp_path: Path) -> None:
    client = _complete_client()
    bundle = prepare_promotion_bundle(
        client,
        batch_id="1",
        sources=[
            SourceAssignment("local", ("L21",)),
            SourceAssignment("vm", ("L22",)),
        ],
        promotion_id="upload-test",
        work_root=tmp_path,
        workers=2,
    )

    result = upload_prepared_bundle(client, bundle)

    assert result["prefix"] == (
        "data/staging/ingestion-consolidated/upload-test/"
    )
    assert client.upload_order[-1].endswith("/_READY.json")
