from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvr.enrichment.embeddings.provider_specs import (
    ProviderSpecError,
    load_provider_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = REPO_ROOT / "configs" / "embedding_providers.json"


def test_locked_provider_specs_parse_without_network_access():
    registry = load_provider_specs(SPEC_PATH)

    assert registry.schema_version == "1.0"
    assert registry.model_ids == ("beit3", "metaclip2", "perception_encoder")
    assert all(provider.status == "ready" for provider in registry.providers)


def test_dimensions_are_provider_specific_and_match_locked_checkpoints():
    registry = load_provider_specs(SPEC_PATH)

    assert registry.get("beit3").embedding_dimension == 768
    assert registry.get("metaclip2").embedding_dimension == 512
    assert registry.get("perception_encoder").embedding_dimension == 1024


def test_each_provider_locks_pair_preprocessing_tokenizer_and_sources():
    registry = load_provider_specs(SPEC_PATH)

    for provider in registry.providers:
        assert provider.image_encoder_entry_point
        assert provider.text_encoder_entry_point
        assert provider.image_preprocessing
        assert provider.tokenizer
        assert provider.official_sources
        assert provider.dependencies
        assert provider.normalized is True
        assert provider.similarity_metric == "inner_product"
        provider.validate_artifact_compatibility()


def test_model_and_source_revisions_are_not_mutable_branch_names():
    registry = load_provider_specs(SPEC_PATH)

    for provider in registry.providers:
        assert len(provider.source_revision) == 40
        if provider.model_revision is not None:
            assert len(provider.model_revision) == 40
    assert "immutable" in registry.get("beit3").reproducibility_notes.lower()


def test_perception_checkpoint_has_published_sha256():
    provider = load_provider_specs(SPEC_PATH).get("perception_encoder")
    assert provider.checkpoint_sha256 == (
        "0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8"
    )


def test_beit3_locks_checkpoint_and_tokenizer_sha256():
    provider = load_provider_specs(SPEC_PATH).get("beit3")
    assert provider.checkpoint_sha256 == (
        "d483a41d92b9aea7ef87d17551413b8c22faae927110666b01ac3aa7a199a25b"
    )
    assert provider.tokenizer_sha256 == (
        "6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("embedding_dimension", 0, "positive integer"),
        ("dtype", "int8", "unsupported dtype"),
        ("similarity_metric", "euclidean", "unsupported similarity"),
        ("source_revision", "main", "40-character commit"),
        ("normalized", False, "must emit normalized"),
    ],
)
def test_rejects_unlocked_checkpoint_dependent_values(
    tmp_path, field, value, message
):
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    payload["providers"][0][field] = value
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProviderSpecError, match=message):
        load_provider_specs(path)


def test_rejects_duplicate_provider_id(tmp_path):
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    payload["providers"].append(dict(payload["providers"][0]))
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProviderSpecError, match="must be unique"):
        load_provider_specs(path)


def test_rejects_beit3_without_locked_tokenizer_sha256(tmp_path):
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    payload["providers"][0]["tokenizer_sha256"] = None
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProviderSpecError, match="requires a locked tokenizer"):
        load_provider_specs(path)


@pytest.mark.parametrize("model_id", ["beit3", "perception_encoder"])
def test_rejects_local_checkpoint_provider_without_sha256(tmp_path, model_id):
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    provider = next(row for row in payload["providers"] if row["model_id"] == model_id)
    provider["checkpoint_sha256"] = None
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProviderSpecError, match="requires a locked checkpoint"):
        load_provider_specs(path)


def test_blocked_provider_requires_reason(tmp_path):
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    payload["providers"][0]["status"] = "blocked"
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProviderSpecError, match="require blocker_reason"):
        load_provider_specs(path)
