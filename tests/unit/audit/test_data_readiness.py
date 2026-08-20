from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from mvr.audit.data_readiness import (
    BLOCKED,
    READY_WITH_WARNINGS,
    AuditConfig,
    run_audit,
)


def _write(path: Path, value: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")
    return path


def _json(path: Path, payload: object) -> Path:
    return _write(path, json.dumps(payload))


def _mapping(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["keyframe_id", "video_id", "shot_id", "frame_idx", "timestamp", "image_path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _base_fixture(tmp_path: Path, *, with_optional: bool = True) -> dict[str, Path]:
    dataset = tmp_path / "dataset"
    videos = dataset / "videos"
    keyframes = dataset / "keyframes"
    mapping = dataset / "keyframe_mapping.csv"
    media = dataset / "media_info"
    ocr = tmp_path / "ocr"
    asr = tmp_path / "asr"
    objects = tmp_path / "objects"
    embeddings = tmp_path / "embeddings"
    processed = tmp_path / "processed"
    indexes = tmp_path / "indexes"
    _write(videos / "V1.mp4", b"video")
    _write(keyframes / "V1" / "K1.jpg", b"\xff\xd8\xff\xd9")
    _mapping(
        mapping,
        [
            {
                "keyframe_id": "K1",
                "video_id": "V1",
                "shot_id": 1,
                "frame_idx": 4,
                "timestamp": 1.0,
                "image_path": "V1/K1.jpg",
            }
        ],
    )
    _write(media / "video_metadata.csv", "video_id,duration_seconds,fps,frame_count\nV1,10,4,20\n")
    if with_optional:
        _json(
            ocr / "V1" / "K1.json",
            {
                "keyframe_id": "K1",
                "video_id": "V1",
                "shot_id": 1,
                "anchor_frame_idx": 4,
                "anchor_timestamp": 1.0,
                "texts": [{"text": "hello", "conf": 0.9}],
            },
        )
        _json(
            asr / "V1" / "K1.json",
            {
                "keyframe_id": "K1",
                "video_id": "V1",
                "shot_id": 1,
                "frame_idx": 4,
                "timestamp": 1.0,
                "text": "hello",
                "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            },
        )
        _json(
            objects / "V1" / "K1.json",
            {
                "keyframe_id": "K1",
                "video_id": "V1",
                "shot_id": 1,
                "frame_idx": 4,
                "timestamp": 1.0,
                "objects": [{"label": "person", "score": 0.9, "bbox": [0, 0, 1, 1]}],
                "person_count": 1,
                "car_count": 0,
            },
        )
    _json(
        embeddings / "clip" / "manifest.json",
        {
            "feature_type": "clip_image",
            "embedding_dimension": 2,
            "vector_count_total": 1,
            "mapping_file": "mapping.csv",
        },
    )
    _write(embeddings / "clip" / "mapping.csv", "video_id,keyframe_id\nV1,K1\n")
    _json(embeddings / "clip" / "K1.json", {"video_id": "V1", "keyframe_id": "K1", "vector": [0.1, 0.2]})
    return {
        "dataset": dataset,
        "mapping": mapping,
        "videos": videos,
        "keyframes": keyframes,
        "media": media,
        "ocr": ocr,
        "asr": asr,
        "objects": objects,
        "embeddings": embeddings,
        "processed": processed,
        "indexes": indexes,
    }


def _config(paths: dict[str, Path], output: Path) -> AuditConfig:
    return AuditConfig(
        dataset_root=paths["dataset"],
        output_root=output,
        mapping_path=paths["mapping"],
        videos_root=paths["videos"],
        keyframes_root=paths["keyframes"],
        media_info_root=paths["media"],
        ocr_root=paths["ocr"],
        asr_root=paths["asr"],
        objects_root=paths["objects"],
        embeddings_root=paths["embeddings"],
        processed_root=paths["processed"],
        indexes_root=paths["indexes"],
        run_id="test-run",
    )


def test_clean_synthetic_audit_serializes_and_marks_pending_providers(tmp_path: Path) -> None:
    paths = _base_fixture(tmp_path)
    result = run_audit(_config(paths, tmp_path / "reports"))

    assert result.summary["overall_status"] == READY_WITH_WARNINGS
    assert result.summary["sections"]["canonical_mapping"]["records"] == 1
    assert result.summary["sections"]["embeddings"]["providers"]["clip"]["vector_count"] == 1
    assert {"beit3", "metaclip2", "perception_encoder"} <= set(
        result.summary["sections"]["embeddings"]["pending_providers"]
    )
    assert result.coverage[0]["keyframe"] == "present"
    assert result.coverage[0]["clip"] == "present"
    for name in ("manifest.json", "summary.json", "issues.jsonl", "coverage.csv", "report.md"):
        assert (result.output_dir / name).is_file()


def test_mapping_duplicate_pair_missing_video_and_keyframe_orphan_are_blockers(tmp_path: Path) -> None:
    paths = _base_fixture(tmp_path, with_optional=False)
    _mapping(
        paths["mapping"],
        [
            {"keyframe_id": "K1", "video_id": "V1", "shot_id": 1, "frame_idx": 4, "timestamp": 1.0, "image_path": "V1/K1.jpg"},
            {"keyframe_id": "K1", "video_id": "V2", "shot_id": 1, "frame_idx": 4, "timestamp": 1.0, "image_path": "V2/K1.jpg"},
            {"keyframe_id": "K2", "video_id": "V1", "shot_id": 1, "frame_idx": 4, "timestamp": 1.0, "image_path": "V1/K1.jpg"},
        ],
    )
    _write(paths["keyframes"] / "orphan.jpg", b"\xff\xd8\xff\xd9")
    result = run_audit(_config(paths, tmp_path / "reports"))
    categories = {issue.category for issue in result.issues}

    assert result.summary["overall_status"] == BLOCKED
    assert "duplicate_canonical_keyframe_id" in categories
    assert "duplicate_output_pair" in categories
    assert "missing_referenced_video" in categories
    assert "missing_keyframe_file" in categories
    assert "orphan_keyframe_file" in categories


def test_asr_metadata_is_excluded_but_malformed_keyframe_json_is_reported(tmp_path: Path) -> None:
    paths = _base_fixture(tmp_path, with_optional=False)
    _json(paths["asr"] / "manifest.json", {"format": "mvr-asr-keyframe-v1"})
    _json(paths["asr"] / "errors.json", {"errors": [], "warnings": []})
    _write(paths["asr"] / "V1" / "K1.json", "{malformed")
    result = run_audit(_config(paths, tmp_path / "reports"))
    section = result.summary["sections"]["asr"]

    assert section["metadata_excluded"] == 2
    assert section["malformed_artifacts"] == 1
    assert any(issue.category == "asr_invalid_json" for issue in result.issues)


def test_ocr_orphan_and_object_invalid_schema_are_detected(tmp_path: Path) -> None:
    paths = _base_fixture(tmp_path, with_optional=False)
    _json(paths["ocr"] / "V1" / "orphan.json", {"keyframe_id": "UNKNOWN", "video_id": "V1", "texts": []})
    _json(
        paths["objects"] / "V1" / "K1.json",
        {
            "keyframe_id": "K1",
            "video_id": "V1",
            "objects": [{"label": "person", "score": "nan"}],
            "person_count": -1,
        },
    )
    result = run_audit(_config(paths, tmp_path / "reports"))
    categories = {issue.category for issue in result.issues}

    assert "orphan_ocr_artifact" in categories
    assert "objects_non_finite_score" in categories
    assert "objects_invalid_count" in categories
    assert "objects_invalid_schema" in categories


def test_embedding_non_finite_dimension_and_count_mismatch_and_required_pending(tmp_path: Path) -> None:
    paths = _base_fixture(tmp_path, with_optional=False)
    _json(
        paths["embeddings"] / "clip" / "manifest.json",
        {"feature_type": "clip_image", "embedding_dimension": 3, "vector_count_total": 2, "mapping_file": "mapping.csv"},
    )
    _json(paths["embeddings"] / "clip" / "K1.json", {"video_id": "V1", "keyframe_id": "K1", "vector": [0.1, "nan"]})
    config = replace(_config(paths, tmp_path / "reports"), required_providers=("beit3",))
    result = run_audit(config)
    categories = {issue.category for issue in result.issues}

    assert result.summary["overall_status"] == BLOCKED
    assert "embedding_non_finite" in categories
    assert "embedding_dimension_mismatch" in categories
    assert "embedding_count_mismatch" in categories
    assert "required_provider_missing" in categories


def test_processed_duplicate_documents_and_indexability_are_audited(tmp_path: Path) -> None:
    paths = _base_fixture(tmp_path, with_optional=False)
    document = {
        "keyframe_id": "K1",
        "video_id": "V1",
        "frame_idx": 4,
        "timestamp": 1.0,
        "indexability": {"lexical": True, "object": False, "vector": True},
    }
    _json(paths["processed"] / "documents" / "K1.json", document)
    _json(paths["processed"] / "documents" / "duplicate.json", document)
    _json(paths["processed"] / "manifest.json", {"task": "task05a", "document_count": 2})
    result = run_audit(_config(paths, tmp_path / "reports"))

    assert result.summary["sections"]["processed_documents"]["document_files"] == 2
    assert any(issue.category == "duplicate_processed_document" for issue in result.issues)
