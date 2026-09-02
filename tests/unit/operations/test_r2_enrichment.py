from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.ingestion.models import Keyframe
from mvr.ingestion.selector_cache import _write_selected_embedding_artifact
from mvr.operations.r2_enrichment import (
    EnrichmentRunConfig,
    R2EnrichmentError,
    R2EnrichmentRunner,
    build_input_transfers,
    build_raw_video_transfers,
    prepare_collection_work,
)
from mvr.operations.r2_sync import R2Sync


def _row(video_id: str, keyframe_id: str, frame_idx: int) -> dict[str, str]:
    collection = video_id.split("_", maxsplit=1)[0]
    return {
        "keyframe_id": keyframe_id,
        "video_id": video_id,
        "collection_id": collection,
        "batch_id": "batch1",
        "shot_id": "1",
        "ordinal": "1",
        "frame_idx": str(frame_idx),
        "timestamp": "1.0",
        "image_path": (
            f"data/corpus/keyframes/{collection}/{video_id}/{keyframe_id}.jpg"
        ),
        "video_object_key": f"data/raw/batch1/videos/{collection}/{video_id}.mp4",
    }


def test_collection_plan_uses_active_ingestion_run_and_exact_r2_keys(tmp_path: Path) -> None:
    rows = [
        _row("L21_V001", "L21_V001_S0001_K01", 10),
        _row("L21_V002", "L21_V002_S0001_K01", 20),
    ]
    plans = prepare_collection_work(
        dataset_manifest={
            "collections": {"L21": {"run_id": "ingest-batch1-local"}}
        },
        mapping_rows=rows,
        collections=("L21",),
        batch_id="1",
        work_root=tmp_path,
    )

    assert len(plans) == 1
    assert plans[0].source_run_id == "ingest-batch1-local"
    assert plans[0].video_ids == ("L21_V001", "L21_V002")
    with plans[0].mapping_csv.open(newline="", encoding="utf-8") as stream:
        assert [row["keyframe_id"] for row in csv.DictReader(stream)] == [
            "L21_V001_S0001_K01",
            "L21_V002_S0001_K01",
        ]

    transfers = build_input_transfers(
        plans, local_r2_root=tmp_path / "r2", batch_id="batch1"
    )
    keys = [key for key, _ in transfers]
    assert "data/corpus/keyframes/L21/L21_V001/L21_V001_S0001_K01.jpg" in keys
    assert (
        "data/staging/ingestion/ingest-batch1-local/batch=1/collection=L21/"
        "L21_V001/selector_embeddings/metaclip2/L21_V001.npy"
    ) in keys
    assert len(transfers) == 8  # 2 keyframes + 3 selector files per video
    assert [
        key
        for key, _ in build_raw_video_transfers(
            plans, local_r2_root=tmp_path / "r2"
        )
    ] == [
        "data/raw/batch1/videos/L21/L21_V001.mp4",
        "data/raw/batch1/videos/L21/L21_V002.mp4",
    ]


def test_collection_plan_rejects_image_key_outside_canonical_video(tmp_path: Path) -> None:
    row = _row("L21_V001", "L21_V001_S0001_K01", 10)
    row["image_path"] = "data/corpus/keyframes/L21/L21_V999/wrong.jpg"

    with pytest.raises(R2EnrichmentError, match="outside its collection/video"):
        prepare_collection_work(
            dataset_manifest={
                "collections": {"L21": {"run_id": "ingest-batch1-local"}}
            },
            mapping_rows=[row],
            collections=("L21",),
            batch_id="1",
            work_root=tmp_path,
        )


def test_collection_plan_rejects_unknown_requested_video(tmp_path: Path) -> None:
    with pytest.raises(R2EnrichmentError, match="requested videos are absent"):
        prepare_collection_work(
            dataset_manifest={
                "collections": {"L21": {"run_id": "ingest-batch1-local"}}
            },
            mapping_rows=[_row("L21_V001", "L21_V001_S0001_K01", 10)],
            collections=("L21",),
            batch_id="1",
            work_root=tmp_path,
            video_ids={"L21_V001", "L21_V999"},
        )


def test_synthetic_runner_stages_ready_bundle_without_per_keyframe_object_puts(
    tmp_path: Path,
) -> None:
    row = _row("L21_V001", "L21_V001_S0001_K01", 10)
    mapping_columns = list(row)
    mapping_path = tmp_path / "source_mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=mapping_columns)
        writer.writeheader()
        writer.writerow(row)
    shots_path = tmp_path / "shots.parquet"
    pytest.importorskip("polars").DataFrame(
        [
            {
                "video_id": row["video_id"],
                "collection_id": "L21",
                "batch_id": "batch1",
                "method": "transnetv2",
                "shot_id": 1,
                "start_frame": 0,
                "end_frame": 100,
                "fps": 10.0,
                "start_time": 0.0,
                "end_time": 10.1,
                "duration": 10.1,
                "frame_count": 101,
            }
        ]
    ).write_parquet(shots_path)

    specs = load_provider_specs(Path("configs/embedding_providers.json"))
    metaclip_spec = specs.get("metaclip2")
    keyframe = Keyframe(
        row["keyframe_id"],
        row["video_id"],
        1,
        1,
        10,
        1.0,
        row["image_path"],
    )
    selector = _write_selected_embedding_artifact(
        output_root=tmp_path / "selector",
        keyframes=[keyframe],
        vectors_by_identity={
            (keyframe.video_id, keyframe.shot_id, keyframe.frame_idx): np.r_[
                1.0, np.zeros(511, dtype=np.float32)
            ]
        },
        spec=metaclip_spec,
    )

    class FakeClient:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {
                "data/corpus/dataset_manifest.json": json.dumps(
                    {"collections": {"L21": {"run_id": "ingest-local"}}}
                ).encode(),
                "data/corpus/ingest/keyframe_mapping.csv": mapping_path.read_bytes(),
                "data/corpus/ingest/shots.parquet": shots_path.read_bytes(),
                row["image_path"]: b"synthetic-jpeg",
                row["video_object_key"]: b"synthetic-mp4",
            }
            selector_prefix = (
                "data/staging/ingestion/ingest-local/batch=1/collection=L21/"
                "L21_V001/selector_embeddings/metaclip2"
            )
            for path in selector.iterdir():
                if path.is_file():
                    self.objects[f"{selector_prefix}/{path.name}"] = path.read_bytes()

        def download_file(self, key, local_path, **kwargs) -> None:
            destination = Path(local_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.objects[key])

        def upload_file(self, local_path, key, **kwargs):
            payload = Path(local_path).read_bytes()
            self.objects[key] = payload
            return {"key": key, "size": len(payload), "sha256": "unused"}

        def list_objects(self, prefix, **kwargs):
            return [
                {"Key": key, "Size": len(payload)}
                for key, payload in sorted(self.objects.items())
                if key.startswith(prefix)
            ]

    class FakeDetector:
        model_family = "RF-DETR"
        model_name = "rfdetr_large"
        score_threshold = 0.4
        nms_enabled = False
        nms_threshold = None
        nms_mode = "none"
        nms_implementation = "nms_free_set_prediction"
        provider_info = {"provider_id": "rfdetr", "checkpoint": "fake"}

        def detect_keyframe(self, path, *, keyframe_id, video_id, frame_id):
            return [
                {
                    "keyframe_id": keyframe_id,
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "frame_idx": frame_id,
                    "label": "person",
                    "class_id": 1,
                    "confidence": 0.9,
                    "x1": 1.0,
                    "y1": 2.0,
                    "x2": 3.0,
                    "y2": 4.0,
                    "score": 0.9,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                }
            ]

    class FakeEncoder:
        def __init__(self, dimension: int) -> None:
            self.dimension = dimension
            self.device_request = "fake-cuda"

        def encode_images(self, paths):
            vectors = np.zeros((len(paths), self.dimension), dtype=np.float32)
            vectors[:, 0] = 1.0
            return vectors

    fake_client = FakeClient()
    detector_loads = []
    encoder_loads = []
    ocr_runs = []
    asr_loads = []

    def detector_factory():
        detector_loads.append("rfdetr")
        return FakeDetector()

    def encoder_factory(provider_id, spec):
        encoder_loads.append(provider_id)
        return FakeEncoder(spec.embedding_dimension)

    def ocr_runner(mapping, videos, output, temp):
        del videos, temp
        with Path(mapping).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for item in rows:
            target = Path(output) / item["video_id"] / f"{item['keyframe_id']}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    {
                        "keyframe_id": item["keyframe_id"],
                        "video_id": item["video_id"],
                        "shot_id": int(item["shot_id"]),
                        "anchor_frame_idx": int(item["frame_idx"]),
                        "anchor_timestamp": float(item["timestamp"]),
                        "texts": [
                            {
                                "text": "TIN CHÍNH",
                                "conf": 0.9,
                                "support_count": 2,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        manifest = {
            "schema_version": "2.0",
            "keyframe_count": len(rows),
            "json_file_count": len(rows),
            "canonical_text_count": len(rows),
        }
        (Path(output) / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        ocr_runs.append(tuple(item["keyframe_id"] for item in rows))
        return manifest

    class FakeASRStage:
        def __init__(self):
            asr_loads.append("asr")

        def __call__(self, dataset_root, output_root):
            with (Path(dataset_root) / "keyframe_mapping.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            for item in rows:
                target = (
                    Path(output_root)
                    / item["video_id"]
                    / f"{item['keyframe_id']}.json"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(
                        {
                            "video_id": item["video_id"],
                            "keyframe_id": item["keyframe_id"],
                            "frame_idx": int(item["frame_idx"]),
                            "timestamp": float(item["timestamp"]),
                            "shot_id": int(item["shot_id"]),
                            "text": "xin chào",
                            "segments": [],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            manifest = {
                "format": "mvr-asr-keyframe-v1",
                "audio": {
                    "sample_rate": 16000,
                    "pre_seconds": 3.0,
                    "post_seconds": 3.0,
                },
                "vad": {"merge_gap_ms": 250},
                "counts": {"keyframes_total": len(rows), "keyframes_failed": 0},
            }
            Path(output_root).mkdir(parents=True, exist_ok=True)
            (Path(output_root) / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (Path(output_root) / "errors.json").write_text(
                '{"errors": [], "warnings": []}', encoding="utf-8"
            )
            return manifest

        def close(self):
            return None

    work = tmp_path / "work"
    report = R2EnrichmentRunner(
        sync=R2Sync(client=fake_client, local_root=work),
        config=EnrichmentRunConfig(
            run_id="enrich-test",
            batch_id="1",
            collections=("L21",),
            work_root=work,
            ocr_model_lock_sha256="0" * 64,
            keyframe_download_workers=2,
            upload_workers=2,
            embedding_batch_size=1,
        ),
        provider_specs=specs,
        detector_factory=detector_factory,
        encoder_factory=encoder_factory,
        ocr_runner=ocr_runner,
        asr_stage_factory=FakeASRStage,
        duration_resolver=lambda _: 10.0,
    ).run()

    assert report["status"] == "READY"
    assert report["schema_version"] == "2.0"
    assert report["features"] == [
        "objects",
        "beit3",
        "metaclip2",
        "perception_encoder",
        "ocr",
        "asr",
    ]
    assert detector_loads == ["rfdetr"]
    assert encoder_loads == ["beit3", "perception_encoder"]
    assert ocr_runs == [(row["keyframe_id"],)]
    assert asr_loads == ["asr"]
    prefix = "data/staging/enrichment/enrich-test/"
    remote_keys = {key for key in fake_client.objects if key.startswith(prefix)}
    assert f"{prefix}_READY.json" in remote_keys
    ready = json.loads(fake_client.objects[f"{prefix}_READY.json"])
    assert ready["features"][-2:] == ["ocr", "asr"]
    assert any(key.endswith("/task05a/objects.tar.gz") for key in remote_keys)
    assert any(key.endswith("/metadata/objects/part-000.parquet") for key in remote_keys)
    assert any(key.endswith("/metadata/ocr/part-000.parquet") for key in remote_keys)
    assert any(key.endswith("/metadata/asr/part-000.parquet") for key in remote_keys)
    assert any(key.endswith("/task05a/ocr.tar.gz") for key in remote_keys)
    assert any(key.endswith("/task05a/asr.tar.gz") for key in remote_keys)
    assert any(key.endswith("/embeddings/metaclip2/L21_V001.npy") for key in remote_keys)
    assert not any(key.endswith(f"/{row['keyframe_id']}.json") for key in remote_keys)

    with pytest.raises(R2EnrichmentError, match="resume state does not match"):
        R2EnrichmentRunner(
            sync=R2Sync(client=fake_client, local_root=work),
            config=EnrichmentRunConfig(
                run_id="enrich-test",
                batch_id="1",
                collections=("L21",),
                work_root=work,
                ocr_model_lock_sha256="0" * 64,
                asr_pre_seconds=4.0,
            ),
            provider_specs=specs,
            detector_factory=detector_factory,
            encoder_factory=encoder_factory,
            ocr_runner=ocr_runner,
            asr_stage_factory=FakeASRStage,
            duration_resolver=lambda _: 10.0,
        ).run()
