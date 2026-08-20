from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mvr.indexing.provider_indexes import (
    ProviderIndexContract,
    ProviderIndexError,
    build_provider_index,
    provider_collection_name,
)


DIMS = {"beit3": 768, "metaclip2": 512, "perception_encoder": 1024}


class FakeBackend:
    name = "fake"

    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()
        self.calls: list[tuple[str, str]] = []
        self.rows: dict[str, list[dict[str, object]]] = {}

    def prepare(self, contract: ProviderIndexContract, policy: str) -> int:
        self.calls.append(("prepare", contract.collection_name))
        if policy == "rebuild":
            self.rows[contract.collection_name] = []
        return len(self.rows.setdefault(contract.collection_name, []))

    def insert_batch(self, contract, vectors, metadata):
        self.calls.append(("insert", contract.collection_name))
        assert vectors.shape[1] == contract.dimension
        failed = []
        for row in metadata:
            if row["keyframe_id"] in self.fail_ids:
                failed.append(row["keyframe_id"])
            else:
                self.rows[contract.collection_name].append(dict(row))
        return failed

    def finalize(self, contract: ProviderIndexContract) -> int:
        self.calls.append(("finalize", contract.collection_name))
        return len(self.rows[contract.collection_name])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    embeddings = tmp_path / "features" / "embeddings"
    processed = tmp_path / "processed"
    docs_root = processed / "documents" / "V1"
    docs_root.mkdir(parents=True)
    rows = [
        {
            "keyframe_id": f"K{i}",
            "video_id": "V1",
            "frame_idx": i * 10,
            "shot_id": i,
            "timestamp": i / 2,
        }
        for i in range(1, 6)
    ]
    manifest_refs = {}
    for provider_id, dimension in DIMS.items():
        root = embeddings / provider_id
        root.mkdir(parents=True)
        vectors = np.zeros((5, dimension), dtype=np.float32)
        vectors[:, 0] = 1
        np.save(root / "V1.npy", vectors)
        mapping = root / "feature_mapping.csv"
        with mapping.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "keyframe_id",
                    "video_id",
                    "feature_file",
                    "feature_index",
                    "frame_id",
                    "shot_id",
                    "timestamp",
                ],
            )
            writer.writeheader()
            for index in [4, 1, 3, 0, 2]:
                row = rows[index]
                writer.writerow(
                    {
                        "keyframe_id": row["keyframe_id"],
                        "video_id": row["video_id"],
                        "feature_file": "V1.npy",
                        "feature_index": index,
                        "frame_id": row["frame_idx"],
                    }
                )
        provider_manifest = {
            "schema_version": "1.0",
            "model_id": provider_id,
            "model_name": provider_id,
            "checkpoint_or_pretrained": "test",
            "model_revision": "a" * 40,
            "embedding_dimension": dimension,
            "dtype": "float32",
            "normalized": True,
            "mapping_path": mapping.name,
            "mapping_sha256": _sha(mapping),
            "keyframe_count": 5,
            "created_at": "2026-01-01T00:00:00Z",
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(provider_manifest), encoding="utf-8")
        manifest_refs[provider_id] = {
            "path": str(manifest_path),
            "sha256": _sha(manifest_path),
        }
    for index, row in enumerate(rows):
        locators = {
            provider_id: {
                "provider": provider_id,
                "video_id": "V1",
                "feature_file": "V1.npy",
                "row_idx": index,
            }
            for provider_id in DIMS
        }
        document = {
            "schema_version": "1.0",
            **row,
            "ocr": {"texts": [], "joined": ""},
            "asr": {"texts": [], "joined": ""},
            "objects": {"labels": [], "counts": {}},
            "clip": {},
            "embeddings": locators,
            "indexability": {
                "lexical": False,
                "object": False,
                "vector": True,
                "embeddings": {provider_id: True for provider_id in DIMS},
            },
        }
        (docs_root / f"{row['keyframe_id']}.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
    (processed / "manifest.json").write_text(
        json.dumps(
            {"source_branch_manifests": {"embeddings": manifest_refs}}
        ),
        encoding="utf-8",
    )
    return embeddings, processed


def _build(tmp_path: Path, provider_id: str, backend: FakeBackend, **kwargs):
    fixture = kwargs.pop("fixture", None)
    embeddings, processed = fixture if fixture is not None else _fixture(tmp_path)
    return build_provider_index(
        provider_root=embeddings / provider_id,
        processed_root=processed,
        output_root=tmp_path / "indexes",
        provider_specs_path=Path("configs/embedding_providers.json"),
        backend=backend,
        environment="test",
        dataset="fasttest",
        batch_size=2,
        **kwargs,
    )


def test_three_providers_build_three_independent_contracts_and_metadata(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    backend = FakeBackend()
    results = {
        provider_id: _build(tmp_path, provider_id, backend, fixture=fixture)
        for provider_id in DIMS
    }
    assert {result.manifest["collection_name"] for result in results.values()} == {
        f"test__fasttest__{provider_id}" for provider_id in DIMS
    }
    assert {key: value.manifest["dimension"] for key, value in results.items()} == DIMS
    assert all(result.manifest["indexed_count"] == 5 for result in results.values())
    for provider_id in DIMS:
        rows = backend.rows[f"test__fasttest__{provider_id}"]
        assert rows[0] == {
            "keyframe_id": "K1",
            "video_id": "V1",
            "frame_id": 10,
            "shot_id": 1,
            "timestamp": 0.5,
        }


def test_source_manifest_hash_mismatch_is_rejected_before_backend(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = fixture[0] / "beit3" / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(ProviderIndexError, match="manifest hash mismatch"):
        _build(tmp_path, "beit3", backend, fixture=fixture)
    assert backend.calls == []


def test_dimension_mismatch_is_rejected_before_backend(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = fixture[0] / "beit3" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["embedding_dimension"] = 512
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    processed_manifest_path = fixture[1] / "manifest.json"
    processed_manifest = json.loads(processed_manifest_path.read_text(encoding="utf-8"))
    processed_manifest["source_branch_manifests"]["embeddings"]["beit3"]["sha256"] = _sha(manifest_path)
    processed_manifest_path.write_text(json.dumps(processed_manifest), encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(ProviderIndexError, match="artifact dimension"):
        _build(tmp_path, "beit3", backend, fixture=fixture)
    assert backend.calls == []


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_missing_or_duplicate_processed_identity_is_rejected(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    docs = fixture[1] / "documents"
    if mutation == "missing":
        (docs / "V1" / "K5.json").unlink()
    else:
        duplicate_dir = docs / "duplicate"
        duplicate_dir.mkdir()
        (duplicate_dir / "copy.json").write_bytes((docs / "V1" / "K1.json").read_bytes())
    with pytest.raises(ProviderIndexError, match="identity mismatch|duplicate processed"):
        _build(tmp_path, "beit3", FakeBackend(), fixture=fixture)


def test_cross_provider_route_is_rejected_before_backend(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    backend = FakeBackend()
    wrong = provider_collection_name("test", "fasttest", "metaclip2")
    with pytest.raises(ProviderIndexError, match="cross-provider"):
        _build(
            tmp_path,
            "beit3",
            backend,
            fixture=fixture,
            collection_name=wrong,
        )
    assert backend.calls == []


def test_create_rerun_is_idempotent_without_second_backend_write(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    backend = FakeBackend()
    first = _build(tmp_path, "beit3", backend, fixture=fixture)
    call_count = len(backend.calls)
    second = _build(tmp_path, "beit3", backend, fixture=fixture)
    assert first.manifest == second.manifest
    assert second.reused is True
    assert len(backend.calls) == call_count


def test_partial_backend_failures_are_reported(tmp_path: Path) -> None:
    backend = FakeBackend({"K2", "K4"})
    result = _build(tmp_path, "beit3", backend)
    assert result.manifest["indexed_count"] == 3
    assert result.manifest["failed_count"] == 2
    assert result.manifest["failed_ids"] == ["K2", "K4"]
