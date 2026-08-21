from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mvr.indexing.filtering import (
    AsrFilterConfig,
    OcrFilterConfig,
    deduplicate_asr_entries,
    deduplicate_ocr_entries,
    filter_asr_entries,
    filter_ocr_entries,
    project_object_metadata,
)
from mvr.indexing.task05a import Task05AConfig, build_index_documents
from mvr.indexing.validation import validate_clip_vector


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_mapping(path: Path) -> list[dict[str, object]]:
    rows = [
        {
            "keyframe_id": "V1_S0001_K01",
            "video_id": "V1",
            "shot_id": 1,
            "ordinal": 1,
            "frame_idx": 10,
            "timestamp": 0.4,
            "image_path": "keyframes/V1/V1_S0001_K01.jpg",
        },
        {
            "keyframe_id": "V1_S0002_K01",
            "video_id": "V1",
            "shot_id": 2,
            "ordinal": 1,
            "frame_idx": 20,
            "timestamp": 0.8,
            "image_path": "keyframes/V1/V1_S0002_K01.jpg",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _make_features(tmp_path: Path, *, corrupt_second_clip: bool = False) -> tuple[Path, Path]:
    features = tmp_path / "features_demo"
    mapping = tmp_path / "ingest" / "keyframe_mapping.csv"
    rows = _write_mapping(mapping)

    clip = features / "clip"
    clip.mkdir(parents=True)
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if corrupt_second_clip:
        vectors[1, 0] = np.nan
    np.save(clip / "V1.npy", vectors)
    _write_json(
        clip / "manifest.json",
        {
            "schema_version": "1.1",
            "embedding_dimension": 3,
            "dtype": "float32",
            "mapping_file": "clip_feature_mapping.csv",
        },
    )
    with (clip / "clip_feature_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "feature_file",
                "feature_index",
                "keyframe_id",
                "shot_id",
                "frame_idx",
                "timestamp",
                "image_path",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "video_id": row["video_id"],
                    "feature_file": "V1.npy",
                    "feature_index": index,
                    "keyframe_id": row["keyframe_id"],
                    "shot_id": row["shot_id"],
                    "frame_idx": row["frame_idx"],
                    "timestamp": row["timestamp"],
                    "image_path": row["image_path"],
                }
            )

    objects = features / "objects"
    _write_json(objects / "manifest.json", {"schema_version": "1.0"})
    _write_json(
        objects / "V1" / "V1_S0001_K01.json",
        {
            **{key: rows[0][key] for key in ("keyframe_id", "video_id", "shot_id", "frame_idx", "timestamp")},
            "person_count": 2,
            "car_count": 0,
            "objects": [
                {"label": "person"},
                {"label": "person"},
                {"label": "chair"},
            ],
        },
    )
    _write_json(
        objects / "V1" / "V1_S0002_K01.json",
        {
            **{key: rows[1][key] for key in ("keyframe_id", "video_id", "shot_id", "frame_idx", "timestamp")},
            "person_count": 0,
            "car_count": 0,
            "objects": [],
        },
    )

    text = features / "text"
    _write_json(text / "manifest.json", {"schema_version": "2.0"})
    _write_json(
        text / "V1" / "V1_S0001_K01.json",
        {
            "keyframe_id": rows[0]["keyframe_id"],
            "video_id": rows[0]["video_id"],
            "shot_id": rows[0]["shot_id"],
            "anchor_frame_idx": rows[0]["frame_idx"],
            "anchor_timestamp": rows[0]["timestamp"],
            "texts": [
                {
                    "text": "  GIÁ   VÀNG ",
                    "conf": 0.70,
                    "support_count": 3,
                    "frame_idx": 9,
                },
                {
                    "text": "giá vàng",
                    "conf": 0.99,
                    "support_count": 1,
                    "frame_idx": 10,
                },
                {"text": "---", "conf": 0.99, "support_count": 1},
                {"text": "123", "conf": 0.86, "support_count": 1},
            ],
        },
    )
    _write_json(
        text / "V1" / "V1_S0002_K01.json",
        {
            "keyframe_id": rows[1]["keyframe_id"],
            "video_id": rows[1]["video_id"],
            "shot_id": rows[1]["shot_id"],
            "anchor_frame_idx": rows[1]["frame_idx"],
            "anchor_timestamp": rows[1]["timestamp"],
            "texts": [],
        },
    )

    asr = features / "asr"
    _write_json(asr / "manifest.json", {"schema_version": "1.0"})
    _write_json(
        asr / "V1" / "V1_S0001_K01.json",
        {
            "keyframe_id": rows[0]["keyframe_id"],
            "video_id": rows[0]["video_id"],
            "shot_id": rows[0]["shot_id"],
            "frame_idx": rows[0]["frame_idx"],
            "timestamp": rows[0]["timestamp"],
            "segments": [
                {"transcript": "  Xin   chào "},
                {"transcript": "xin chào"},
                {"transcript": "   "},
            ],
        },
    )
    _write_json(
        asr / "V1" / "V1_S0002_K01.json",
        {
            "keyframe_id": rows[1]["keyframe_id"],
            "video_id": rows[1]["video_id"],
            "shot_id": rows[1]["shot_id"],
            "frame_idx": rows[1]["frame_idx"],
            "timestamp": rows[1]["timestamp"],
            "segments": [],
        },
    )
    return features, mapping


@pytest.mark.parametrize(
    ("conf", "support", "expected"),
    [
        (0.99, 1, True),
        (0.86, 1, True),
        (0.70, 3, True),
        (0.55, 2, True),
        (0.70, 1, False),
        (0.45, 3, False),
    ],
)
def test_ocr_approved_quality_cases(conf: float, support: int, expected: bool) -> None:
    result = filter_ocr_entries(
        [{"text": "valid", "conf": conf, "support_count": support}]
    )
    assert bool(result.kept) is expected


def test_ocr_drop_reasons_and_numeric_preservation() -> None:
    result = filter_ocr_entries(
        [
            {"text": None, "conf": 0.9, "support_count": 1},
            {"text": "x", "conf": float("nan"), "support_count": 1},
            {"text": "x", "conf": 0.9, "support_count": 0},
            {"text": "x", "conf": 0.4, "support_count": 3},
            {"text": "x", "conf": 0.7, "support_count": 1},
            {"text": "!!!", "conf": 0.99, "support_count": 1},
            {"text": "12345", "conf": 0.86, "support_count": 1},
        ]
    )
    assert [entry["text"] for entry in result.kept] == ["12345"]
    assert result.dropped_by_reason == {
        "empty": 1,
        "invalid_conf": 1,
        "invalid_support": 1,
        "below_absolute_conf": 1,
        "low_support_and_not_high_conf": 1,
        "punctuation_only": 1,
    }


def test_ocr_dedup_prioritizes_support_before_confidence() -> None:
    entries = [
        {"text": "GIÁ VÀNG", "normalized": "giá vàng", "conf": 0.99, "support_count": 1, "frame_idx": 10},
        {"text": "giá vàng", "normalized": "giá vàng", "conf": 0.70, "support_count": 3, "frame_idx": 8},
    ]
    kept, removed = deduplicate_ocr_entries(entries, anchor_frame_idx=10)
    assert removed == 1
    assert kept[0]["support_count"] == 3


def test_asr_empty_filter_and_intra_keyframe_dedup_without_confidence() -> None:
    filtered = filter_asr_entries(
        [{"text": None}, {"text": "  "}, {"text": " Xin  chào "}]
    )
    kept, removed = deduplicate_asr_entries(
        [*filtered.kept, {"text": "xin chào", "normalized": "xin chào"}]
    )
    assert filtered.dropped_by_reason["empty"] == 2
    assert kept == [{"text": "Xin chào", "normalized": "xin chào"}]
    assert removed == 1
    assert "conf" not in kept[0]


def test_asr_confidence_filter_only_when_actual_field_is_supplied() -> None:
    result = filter_asr_entries(
        [
            {"text": "low", "confidence": 0.4},
            {"text": "high", "confidence": 0.9},
        ],
        config=AsrFilterConfig(min_conf=0.5),
        confidence_field="confidence",
    )
    assert [entry["text"] for entry in result.kept] == ["high"]
    assert result.dropped_by_reason["low_conf"] == 1


def test_object_projection_is_deterministic_and_preserves_zero_object() -> None:
    projected, count = project_object_metadata(
        {
            "person_count": 2,
            "car_count": 1,
            "objects": [
                {"label": "chair"},
                {"label": "person"},
                {"label": "person"},
                {"label": "car"},
            ],
        }
    )
    assert projected == {
        "labels": ["car", "chair", "person"],
        "counts": {"car": 1, "chair": 1, "person": 2},
    }
    assert count == 4
    assert project_object_metadata(
        {"person_count": 0, "car_count": 0, "objects": []}
    ) == ({"labels": [], "counts": {}}, 0)
    with pytest.raises(ValueError, match="non-negative"):
        project_object_metadata({"objects": {"person": -1}})


def test_clip_vector_validation_cases() -> None:
    assert validate_clip_vector(np.asarray([1.0, 0.0]), expected_dimension=2) is None
    assert validate_clip_vector(np.asarray([np.nan, 0.0]), expected_dimension=2) == "non_finite"
    assert validate_clip_vector(np.asarray([np.inf, 0.0]), expected_dimension=2) == "non_finite"
    assert validate_clip_vector(np.asarray([1.0]), expected_dimension=2) == "dimension_mismatch"
    assert validate_clip_vector(np.asarray(["x", "y"]), expected_dimension=2) == "non_numeric_dtype"


def test_builder_emits_one_document_per_mapping_and_independent_flags(tmp_path: Path) -> None:
    features, mapping = _make_features(tmp_path, corrupt_second_clip=True)
    result = build_index_documents(
        features_root=features,
        mapping_csv=mapping,
        output_root=tmp_path / "processed",
        dataset="demo",
    )
    assert result.document_count == 2
    first = json.loads(
        (result.output_root / "documents" / "V1" / "V1_S0001_K01.json").read_text(
            encoding="utf-8"
        )
    )
    second = json.loads(
        (result.output_root / "documents" / "V1" / "V1_S0002_K01.json").read_text(
            encoding="utf-8"
        )
    )
    assert first["ocr"]["texts"][0]["text"] == "123"
    assert any(item["text"] == "GIÁ VÀNG" for item in first["ocr"]["texts"])
    assert first["asr"]["joined"] == "Xin chào"
    assert first["indexability"] == {
        "lexical": True,
        "object": True,
        "vector": True,
        "embeddings": {"clip": True},
    }
    assert second["indexability"] == {
        "lexical": False,
        "object": True,
        "vector": False,
        "embeddings": {"clip": False},
    }
    assert result.manifest["asr"]["confidence_field"] is None
    assert result.manifest["asr"]["confidence_filter_active"] is False
    assert result.manifest["document_count"] == 2
    assert result.manifest["clip"]["vectors_expected"] == 2
    assert result.manifest["clip"]["vectors_valid"] == 1
    assert result.manifest["clip"]["vectors_invalid"] == 1


def test_builder_reports_identity_mismatch_without_dropping_keyframe(tmp_path: Path) -> None:
    features, mapping = _make_features(tmp_path)
    object_path = features / "objects" / "V1" / "V1_S0002_K01.json"
    payload = json.loads(object_path.read_text(encoding="utf-8"))
    payload["frame_idx"] = 999
    _write_json(object_path, payload)
    result = build_index_documents(
        features_root=features,
        mapping_csv=mapping,
        output_root=tmp_path / "processed",
    )
    document = json.loads(
        (result.output_root / "documents" / "V1" / "V1_S0002_K01.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.document_count == 2
    assert document["indexability"]["object"] is False
    assert result.manifest["validation_issue_counts"]["identity_mismatch"] >= 1


def test_builder_keeps_lexical_and_vector_flags_independent(tmp_path: Path) -> None:
    features, mapping = _make_features(tmp_path)
    vector_path = features / "clip" / "V1.npy"
    vectors = np.load(vector_path)
    vectors[0, 0] = np.nan
    np.save(vector_path, vectors)
    result = build_index_documents(
        features_root=features,
        mapping_csv=mapping,
        output_root=tmp_path / "processed",
    )
    first = json.loads(
        (result.output_root / "documents" / "V1" / "V1_S0001_K01.json").read_text(
            encoding="utf-8"
        )
    )
    second = json.loads(
        (result.output_root / "documents" / "V1" / "V1_S0002_K01.json").read_text(
            encoding="utf-8"
        )
    )
    assert first["indexability"]["lexical"] is True
    assert first["indexability"]["vector"] is False
    assert second["indexability"]["lexical"] is False
    assert second["indexability"]["vector"] is True


def test_builder_is_deterministic_except_created_at(tmp_path: Path) -> None:
    features, mapping = _make_features(tmp_path)
    first = build_index_documents(
        features_root=features,
        mapping_csv=mapping,
        output_root=tmp_path / "out-a",
    )
    second = build_index_documents(
        features_root=features,
        mapping_csv=mapping,
        output_root=tmp_path / "out-b",
    )
    first_documents = {
        path.relative_to(first.output_root): path.read_bytes()
        for path in (first.output_root / "documents").rglob("*.json")
    }
    second_documents = {
        path.relative_to(second.output_root): path.read_bytes()
        for path in (second.output_root / "documents").rglob("*.json")
    }
    assert first_documents == second_documents
    manifest_a = dict(first.manifest)
    manifest_b = dict(second.manifest)
    manifest_a.pop("created_at")
    manifest_b.pop("created_at")
    assert manifest_a == manifest_b
    assert first.filtering_report == second.filtering_report
    assert not list(first.output_root.rglob("*.jsonl"))


def test_builder_rejects_duplicate_canonical_identity(tmp_path: Path) -> None:
    features, mapping = _make_features(tmp_path)
    content = mapping.read_text(encoding="utf-8").splitlines()
    mapping.write_text("\n".join([*content, content[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate canonical identity"):
        build_index_documents(
            features_root=features,
            mapping_csv=mapping,
            output_root=tmp_path / "processed",
            config=Task05AConfig(ocr=OcrFilterConfig()),
        )


def test_asr_branch_metadata_is_ignored_but_malformed_keyframe_is_reported(
    tmp_path: Path,
) -> None:
    features, mapping = _make_features(tmp_path)
    asr_root = features / "asr"
    _write_json(asr_root / "errors.json", {"errors": [{"message": "diagnostic"}]})
    result = build_index_documents(
        features_root=features,
        mapping_csv=mapping,
        output_root=tmp_path / "processed",
    )
    assert result.manifest["validation_issue_counts"].get("invalid_schema", 0) == 0
    assert result.manifest["asr"]["input_entries"] > 0

    broken_path = asr_root / "V1" / "BROKEN.json"
    _write_json(broken_path, {"segments": [{"transcript": "not valid identity"}]})
    result_with_broken = build_index_documents(
        features_root=features,
        mapping_csv=mapping,
        output_root=tmp_path / "processed-broken",
    )
    assert result_with_broken.manifest["validation_issue_counts"]["invalid_schema"] >= 1
