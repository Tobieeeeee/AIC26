from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mvr.enrichment.embeddings.offline import extract_embedding_artifacts
from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.enrichment.embeddings.query_encoders import (
    ProviderQueryEncoderConfig,
    QueryEncoderRegistry,
)
from mvr.indexing.provider_indexes import (
    ProviderIndexContract,
    build_provider_index,
    provider_collection_name,
)
from mvr.indexing.task05a import Task05AConfig, build_index_documents
from mvr.retrieval.provider_semantic import (
    BackendSearchHit,
    IndependentSemanticRetriever,
)
from mvr.retrieval.semantic_rrf import fuse_independent_response


PROVIDERS = ("beit3", "metaclip2", "perception_encoder")


class OfflineEncoder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def encode_images(self, image_paths):
        values = np.zeros((len(image_paths), self.dimension), dtype=np.float32)
        for index in range(len(image_paths)):
            values[index, index] = 1.0
        return values


class StoredIndexBackend:
    name = "fasttest_fake"

    def __init__(self) -> None:
        self.values = {}
        self.metadata = {}

    def prepare(self, contract, policy):
        self.values[contract.collection_name] = []
        self.metadata[contract.collection_name] = []
        return 0

    def insert_batch(self, contract, vectors, metadata):
        self.values[contract.collection_name].extend(np.asarray(row) for row in vectors)
        self.metadata[contract.collection_name].extend(dict(row) for row in metadata)
        return []

    def finalize(self, contract):
        return len(self.metadata[contract.collection_name])

    def search(self, contract, vector, top_k):
        values = np.asarray(self.values[contract.collection_name])
        scores = values @ vector
        order = sorted(
            range(len(scores)),
            key=lambda index: (
                -float(scores[index]),
                self.metadata[contract.collection_name][index]["keyframe_id"],
            ),
        )[:top_k]
        return [
            BackendSearchHit(
                float(scores[index]), self.metadata[contract.collection_name][index]
            )
            for index in order
        ]


class QueryAdapter:
    def __init__(self, model_id: str, dimension: int) -> None:
        self.model_id = model_id
        self.dimension = dimension

    def encode_text(self, text: str):
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[0] = 1.0
        return vector


def test_five_keyframe_three_provider_structural_e2e(tmp_path: Path) -> None:
    specs = load_provider_specs(Path("configs/embedding_providers.json"))
    keyframe_root = tmp_path / "keyframes"
    mapping_path = tmp_path / "ingest" / "keyframe_mapping.csv"
    mapping_path.parent.mkdir()
    rows = []
    for index in range(1, 6):
        image_path = f"V1/K{index}.jpg"
        target = keyframe_root / image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
        rows.append(
            {
                "keyframe_id": f"K{index}",
                "video_id": "V1",
                "shot_id": index,
                "frame_idx": index * 10,
                "timestamp": index / 2,
                "image_path": image_path,
            }
        )
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    features_root = tmp_path / "features"
    embeddings_root = features_root / "embeddings"
    for provider_id in PROVIDERS:
        spec = specs.get(provider_id)
        extract_embedding_artifacts(
            rows,
            keyframe_root=keyframe_root,
            embeddings_root=embeddings_root,
            spec=spec,
            encoder=OfflineEncoder(spec.embedding_dimension),
            batch_size=5,
        )

    processed = build_index_documents(
        features_root=features_root,
        mapping_csv=mapping_path,
        output_root=tmp_path / "processed",
        config=Task05AConfig(required_embedding_providers=PROVIDERS),
    )
    assert processed.document_count == 5
    for provider_id in PROVIDERS:
        stats = processed.manifest["embeddings"][provider_id]
        assert stats["vectors_valid"] == 5
        assert stats["missing_ids"] == stats["extra_ids"] == 0
        assert stats["duplicate_ids"] == 0

    backend = StoredIndexBackend()
    index_manifests = {}
    contracts = {}
    for provider_id in PROVIDERS:
        result = build_provider_index(
            provider_root=embeddings_root / provider_id,
            processed_root=processed.output_root,
            output_root=tmp_path / "indexes",
            provider_specs_path=Path("configs/embedding_providers.json"),
            backend=backend,
            environment="test",
            dataset="fasttest",
            policy="create",
        )
        index_manifests[provider_id] = result.manifest
        spec = specs.get(provider_id)
        contracts[provider_id] = ProviderIndexContract(
            model_id=provider_id,
            collection_name=provider_collection_name("test", "fasttest", provider_id),
            dimension=spec.embedding_dimension,
            metric=spec.similarity_metric,
            normalized=spec.normalized,
            backend=backend.name,
        )
        assert result.manifest["indexed_count"] == 5
        assert result.manifest["failed_count"] == 0

    query_encoders = QueryEncoderRegistry(
        specs,
        ProviderQueryEncoderConfig(),
        factories={
            provider_id: (
                lambda spec, provider_id=provider_id: QueryAdapter(
                    provider_id, spec.embedding_dimension
                )
            )
            for provider_id in PROVIDERS
        },
    )
    retrieval = IndependentSemanticRetriever(
        query_encoders=query_encoders,
        backend=backend,
        contracts=contracts,
        failure_policy="fail_closed",
        required_providers=PROVIDERS,
    )
    response = retrieval.search(
        "a red bus", provider_ids=PROVIDERS, top_k=5
    )
    assert len(response.ranked_lists) == 3
    assert all(len(results) == 5 for results in response.ranked_lists.values())
    fused = fuse_independent_response(response, top_k=5)
    assert len(fused.results) == 5
    assert {item.keyframe_id for item in fused.results} == {
        "K1", "K2", "K3", "K4", "K5"
    }
    assert all(item.video_id == "V1" for item in fused.results)
