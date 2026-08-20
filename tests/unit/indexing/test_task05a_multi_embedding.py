from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mvr.indexing.task05a import Task05AConfig, build_index_documents
from mvr.indexing.validation import Task05AValidationError, validate_task05a_document


PROVIDERS = {"beit3": 2, "metaclip2": 3, "perception_encoder": 4}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_mapping(path: Path) -> list[dict[str, object]]:
    rows = [
        {
            "keyframe_id": f"V1_K{i:02d}",
            "video_id": "V1",
            "shot_id": i,
            "frame_idx": i * 10,
            "timestamp": i / 2,
            "image_path": f"V1/V1_K{i:02d}.jpg",
        }
        for i in range(1, 6)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_provider(
    root: Path,
    provider_id: str,
    dimension: int,
    canonical: list[dict[str, object]],
    *,
    order: list[int] | None = None,
    dtype: np.dtype = np.dtype("float32"),
) -> Path:
    provider = root / provider_id
    provider.mkdir(parents=True)
    vectors = np.zeros((len(canonical), dimension), dtype=dtype)
    vectors[:, 0] = 1
    np.save(provider / "V1.npy", vectors)
    mapping = provider / "feature_mapping.csv"
    with mapping.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "keyframe_id",
            "video_id",
            "feature_file",
            "feature_index",
            "frame_id",
            "shot_id",
            "timestamp",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in order or list(range(len(canonical))):
            row = canonical[index]
            writer.writerow(
                {
                    "keyframe_id": row["keyframe_id"],
                    "video_id": row["video_id"],
                    "feature_file": "V1.npy",
                    "feature_index": index,
                    "frame_id": row["frame_idx"],
                    "shot_id": row["shot_id"],
                    "timestamp": row["timestamp"],
                }
            )
    manifest = {
        "schema_version": "1.0",
        "model_id": provider_id,
        "model_name": f"test/{provider_id}",
        "checkpoint_or_pretrained": "test-checkpoint",
        "model_revision": "deadbeef",
        "embedding_dimension": dimension,
        "dtype": str(dtype),
        "normalized": True,
        "mapping_path": mapping.name,
        "mapping_sha256": _sha256(mapping),
        "keyframe_count": len(canonical),
        "created_at": "2026-01-01T00:00:00Z",
    }
    (provider / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return provider


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    features = tmp_path / "features"
    features.mkdir()
    mapping = tmp_path / "ingest" / "keyframe_mapping.csv"
    canonical = _canonical_mapping(mapping)
    providers = {
        provider_id: _write_provider(
            features / "embeddings",
            provider_id,
            dimension,
            canonical,
            order=[4, 0, 3, 1, 2],
        )
        for provider_id, dimension in PROVIDERS.items()
    }
    return features, mapping, providers


def _refresh_mapping_hash(provider: Path, *, keyframe_count: int | None = None) -> None:
    manifest_path = provider / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mapping_sha256"] = _sha256(provider / "feature_mapping.csv")
    if keyframe_count is not None:
        manifest["keyframe_count"] = keyframe_count
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _build(tmp_path: Path, features: Path, mapping: Path):
    return build_index_documents(
        features_root=features,
        mapping_csv=mapping,
        output_root=tmp_path / "processed",
        config=Task05AConfig(required_embedding_providers=tuple(PROVIDERS)),
    )


def test_three_providers_are_joined_by_keyframe_id_not_mapping_position(
    tmp_path: Path,
) -> None:
    features, mapping, _ = _fixture(tmp_path)
    result = _build(tmp_path, features, mapping)
    assert result.document_count == 5
    assert result.manifest["indexability_counts"]["embeddings"] == {
        "beit3": 5,
        "metaclip2": 5,
        "perception_encoder": 5,
    }
    assert result.manifest["embedding_provider_policy"] == {
        "required": sorted(PROVIDERS),
        "optional": [],
        "required_satisfied": True,
        "invalid_required": [],
    }
    for provider_id, dimension in PROVIDERS.items():
        stats = result.manifest["embeddings"][provider_id]
        assert stats["dimension"] == dimension
        assert stats["vectors_valid"] == 5
        assert stats["missing_ids"] == stats["extra_ids"] == 0
        assert stats["duplicate_ids"] == stats["mapping_mismatches"] == 0
    document = json.loads(
        (result.output_root / "documents" / "V1" / "V1_K01.json").read_text(
            encoding="utf-8"
        )
    )
    assert document["indexability"]["embeddings"] == {
        provider_id: True for provider_id in sorted(PROVIDERS)
    }
    assert document["indexability"]["vector"] is True
    assert all(locator["row_idx"] == 0 for locator in document["embeddings"].values())


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("wrong_dimension", "dimension_mismatch"),
        ("wrong_dtype", "dtype_mismatch"),
        ("nan", "non_finite"),
        ("inf", "non_finite"),
        ("locator", "invalid_locator"),
    ],
)
def test_provider_vector_failures_are_isolated(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    features, mapping, providers = _fixture(tmp_path)
    provider = providers["beit3"]
    vectors = np.load(provider / "V1.npy")
    if mutation == "wrong_dimension":
        vectors = np.ones((5, 3), dtype=np.float32)
    elif mutation == "wrong_dtype":
        vectors = vectors.astype(np.float64)
    elif mutation == "nan":
        vectors[0, 0] = np.nan
    elif mutation == "inf":
        vectors[0, 0] = np.inf
    elif mutation == "locator":
        rows = list(csv.DictReader((provider / "feature_mapping.csv").open(encoding="utf-8")))
        rows[0]["feature_index"] = "99"
        with (provider / "feature_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _refresh_mapping_hash(provider)
    if mutation != "locator":
        np.save(provider / "V1.npy", vectors)
    result = _build(tmp_path, features, mapping)
    assert result.manifest["embeddings"]["beit3"]["vectors_valid"] < 5
    assert result.manifest["embeddings"]["beit3"]["invalid_by_reason"][reason] > 0
    assert result.manifest["embeddings"]["metaclip2"]["vectors_valid"] == 5
    assert result.manifest["embeddings"]["perception_encoder"]["vectors_valid"] == 5


def test_duplicate_missing_extra_and_count_mismatch_are_reported(tmp_path: Path) -> None:
    features, mapping, providers = _fixture(tmp_path)
    provider = providers["beit3"]
    mapping_path = provider / "feature_mapping.csv"
    rows = list(csv.DictReader(mapping_path.open(encoding="utf-8")))
    rows[1] = dict(rows[0])
    rows.pop(2)
    rows.append({**rows[-1], "keyframe_id": "EXTRA", "video_id": "VX"})
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _refresh_mapping_hash(provider, keyframe_count=4)
    result = _build(tmp_path, features, mapping)
    stats = result.manifest["embeddings"]["beit3"]
    assert stats["duplicate_ids"] == 1
    assert stats["missing_ids"] >= 1
    assert stats["extra_ids"] == 1
    assert stats["mapping_mismatches"] >= 1


@pytest.mark.parametrize(
    ("flags", "aggregate"),
    [
        ({"a": False, "b": False}, False),
        ({"a": False, "b": True}, True),
        ({"a": True, "b": True}, True),
    ],
)
def test_document_validator_enforces_aggregate_vector_flag(
    flags: dict[str, bool], aggregate: bool
) -> None:
    locators = {key: ({"row_idx": 0} if value else None) for key, value in flags.items()}
    document = {
        "schema_version": "1.0",
        "keyframe_id": "K1",
        "video_id": "V1",
        "shot_id": 1,
        "frame_idx": 1,
        "timestamp": 0.1,
        "ocr": {"texts": [], "joined": ""},
        "asr": {"texts": [], "joined": ""},
        "objects": {"labels": [], "counts": {}},
        "clip": {},
        "embeddings": locators,
        "indexability": {
            "lexical": False,
            "object": False,
            "vector": aggregate,
            "embeddings": flags,
        },
    }
    validate_task05a_document(document)
    document["indexability"]["vector"] = not aggregate
    with pytest.raises(Task05AValidationError, match="must equal any"):
        validate_task05a_document(document)
