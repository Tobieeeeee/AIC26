from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mvr.enrichment.embeddings.artifacts import (
    EmbeddingArtifactError,
    EmbeddingProviderRegistry,
    discover_embedding_providers,
    load_mapping,
    load_provider_artifact,
    sha256_file,
)


def _create_provider(
    root: Path,
    model_id: str,
    *,
    dimension: int = 8,
    directory_name: str | None = None,
    mapping_path: str = "feature_mapping.csv",
) -> Path:
    provider_root = root / (directory_name or model_id)
    provider_root.mkdir(parents=True)
    mapping = provider_root / "feature_mapping.csv"
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
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
        writer.writerow(
            {
                "keyframe_id": f"{model_id}_kf_1",
                "video_id": "L21_V001",
                "feature_file": "L21_V001.npy",
                "feature_index": 0,
                "frame_id": 42,
                "shot_id": 3,
                "timestamp": 1.4,
            }
        )
    manifest = {
        "schema_version": "1.0",
        "model_id": model_id,
        "model_name": f"Test {model_id}",
        "checkpoint_or_pretrained": "test/checkpoint",
        "model_revision": "0123456789abcdef",
        "embedding_dimension": dimension,
        "dtype": "float32",
        "normalized": True,
        "mapping_path": mapping_path,
        "mapping_sha256": sha256_file(mapping),
        "keyframe_count": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (provider_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return provider_root


def test_discovers_three_provider_fixtures_without_provider_branches(tmp_path):
    root = tmp_path / "embeddings"
    _create_provider(root, "perception_encoder", dimension=1024)
    _create_provider(root, "beit3", dimension=768)
    _create_provider(root, "metaclip2", dimension=1280)
    (root / "README.txt").write_text("not a provider", encoding="utf-8")
    (root / "cache").mkdir()

    registry = discover_embedding_providers(
        root, verify_checksum=True, verify_mapping=True
    )

    assert registry.model_ids == ("beit3", "metaclip2", "perception_encoder")
    assert [item.manifest.embedding_dimension for item in registry] == [768, 1280, 1024]


def test_empty_or_missing_provider_root_returns_empty_registry(tmp_path):
    assert discover_embedding_providers(tmp_path / "missing").model_ids == ()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert len(discover_embedding_providers(empty)) == 0


def test_registry_rejects_duplicate_model_id(tmp_path):
    artifact = load_provider_artifact(_create_provider(tmp_path, "beit3"))
    with pytest.raises(EmbeddingArtifactError, match="duplicate model_id: beit3"):
        EmbeddingProviderRegistry([artifact, artifact])


def test_rejects_directory_manifest_model_id_mismatch(tmp_path):
    root = tmp_path / "embeddings"
    _create_provider(root, "beit3", directory_name="wrong")
    with pytest.raises(EmbeddingArtifactError, match="does not match model_id"):
        discover_embedding_providers(root)


@pytest.mark.parametrize("dimension", [0, -1, True, "768"])
def test_rejects_invalid_dimension(tmp_path, dimension):
    provider = _create_provider(tmp_path, "beit3")
    manifest_path = provider / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["embedding_dimension"] = dimension
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EmbeddingArtifactError, match="positive integer"):
        load_provider_artifact(provider)


@pytest.mark.parametrize("mapping_path", ["../outside.csv", "/absolute.csv"])
def test_rejects_mapping_path_outside_provider_root(tmp_path, mapping_path):
    provider = _create_provider(tmp_path, "beit3", mapping_path=mapping_path)
    with pytest.raises(EmbeddingArtifactError, match="mapping_path"):
        load_provider_artifact(provider)


def test_checksum_verification_is_optional_and_rejects_mismatch(tmp_path):
    provider = _create_provider(tmp_path, "beit3")
    mapping = provider / "feature_mapping.csv"
    mapping.write_text(mapping.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    load_provider_artifact(provider, verify_checksum=False)
    with pytest.raises(EmbeddingArtifactError, match="checksum mismatch"):
        load_provider_artifact(provider, verify_checksum=True)


def test_mapping_preserves_source_identity_and_uses_explicit_locator(tmp_path):
    artifact = load_provider_artifact(
        _create_provider(tmp_path, "beit3"), verify_checksum=True
    )
    records = load_mapping(artifact)

    assert records[0].keyframe_id == "beit3_kf_1"
    assert records[0].feature_file == "L21_V001.npy"
    assert records[0].feature_index == 0
    assert records[0].frame_id == "42"
    assert records[0].shot_id == "3"
    assert records[0].timestamp == "1.4"


def test_discovery_order_is_deterministic(tmp_path):
    root = tmp_path / "embeddings"
    for model_id in ("zeta", "alpha", "middle"):
        _create_provider(root, model_id)

    first = discover_embedding_providers(root).model_ids
    second = discover_embedding_providers(root).model_ids
    assert first == second == ("alpha", "middle", "zeta")


def test_invalid_manifest_reports_provider_directory(tmp_path):
    root = tmp_path / "embeddings"
    provider = _create_provider(root, "beit3")
    (provider / "manifest.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(EmbeddingArtifactError, match="invalid provider 'beit3'"):
        discover_embedding_providers(root)
