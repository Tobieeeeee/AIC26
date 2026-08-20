from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mvr.enrichment.embeddings.artifacts import sha256_file
from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from scripts.verification.run_real_multi_embedding_retrieval import (
    _load_contract,
    load_recorded_query_artifact,
)


SPECS = load_provider_specs(Path("configs/embedding_providers.json"))


def _write_query_artifact(root: Path, provider: str = "beit3") -> Path:
    spec = SPECS.get(provider)
    root.mkdir(parents=True)
    vectors = np.zeros((2, spec.embedding_dimension), dtype=np.float32)
    vectors[:, 0] = 1.0
    vector_path = root / "query_vectors.npy"
    np.save(vector_path, vectors, allow_pickle=False)
    manifest = {
        "schema_version": "1.0",
        "model_id": provider,
        "model_name": spec.model_name,
        "checkpoint_or_pretrained": spec.checkpoint_or_pretrained,
        "model_revision": spec.model_revision,
        "source_revision": spec.source_revision,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "tokenizer_sha256": spec.tokenizer_sha256,
        "dimension": spec.embedding_dimension,
        "dtype": spec.dtype,
        "normalized": spec.normalized,
        "adapter": (
            "mvr.enrichment.embeddings.query_encoders.Beit3TextEncoder"
        ),
        "tokenizer": spec.tokenizer,
        "query_count": 2,
        "queries": [
            {"id": "Q1", "text": "first"},
            {"id": "Q2", "text": "second"},
        ],
        "vector_file": vector_path.name,
        "vector_sha256": sha256_file(vector_path),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_recorded_query_artifact_verifies_full_contract(tmp_path):
    root = _write_query_artifact(tmp_path / "beit3")
    manifest, vectors, ids, texts = load_recorded_query_artifact(
        root, "beit3", SPECS.get("beit3")
    )
    assert manifest["checkpoint_sha256"] == SPECS.get("beit3").checkpoint_sha256
    assert vectors.shape == (2, 768)
    assert ids == ["Q1", "Q2"]
    assert texts == ["first", "second"]


def test_recorded_query_artifact_rejects_tampered_vectors(tmp_path):
    root = _write_query_artifact(tmp_path / "beit3")
    with (root / "query_vectors.npy").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="vector SHA256 mismatch"):
        load_recorded_query_artifact(root, "beit3", SPECS.get("beit3"))


def test_index_contract_verifies_provider_and_source_manifest_hash(tmp_path):
    provider = "beit3"
    spec = SPECS.get(provider)
    embeddings_root = tmp_path / "embeddings"
    source = embeddings_root / provider / "manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    payload = {
        "schema_version": "1.0",
        "model_id": provider,
        "collection_name": "test__fast5__beit3",
        "dimension": spec.embedding_dimension,
        "metric": spec.similarity_metric,
        "normalized": spec.normalized,
        "backend": "faiss_index_flat_ip",
        "indexed_count": 5,
        "failed_count": 0,
        "missing_ids": 0,
        "extra_ids": 0,
        "duplicate_ids": 0,
        "source_embedding_manifest_sha256": sha256_file(source),
    }
    path = tmp_path / "index_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    contract = _load_contract(path, provider, spec, embeddings_root)
    assert contract.model_id == provider

    payload["source_embedding_manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source embedding manifest SHA256 mismatch"):
        _load_contract(path, provider, spec, embeddings_root)
