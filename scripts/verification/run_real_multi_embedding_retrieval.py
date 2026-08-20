"""Search three real provider indexes and fuse their independent rankings."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mvr.enrichment.embeddings.artifacts import sha256_file
from mvr.enrichment.embeddings.query_encoders import QueryEncoderRegistry, QueryRoutingError
from mvr.enrichment.embeddings.provider_specs import ProviderSpec, load_provider_specs
from mvr.indexing.provider_indexes import ProviderIndexContract
from mvr.retrieval.provider_semantic import (
    FaissProviderSearchBackend,
    IndependentSemanticRetriever,
)
from mvr.retrieval.semantic_rrf import fuse_independent_response


PROVIDERS = ("beit3", "metaclip2", "perception_encoder")
EXPECTED_ADAPTERS = {
    "beit3": "mvr.enrichment.embeddings.query_encoders.Beit3TextEncoder",
    "metaclip2": "mvr.enrichment.embeddings.query_encoders.MetaClip2TextEncoder",
    "perception_encoder": (
        "mvr.enrichment.embeddings.query_encoders.PerceptionEncoderTextEncoder"
    ),
}


class _RecordedTextAdapter:
    def __init__(self, model_id: str, vectors: np.ndarray, texts: list[str]) -> None:
        self.model_id = model_id
        self._vectors = vectors
        self._rows = {text: index for index, text in enumerate(texts)}

    def encode_text(self, text: str) -> np.ndarray:
        try:
            return self._vectors[self._rows[text]]
        except KeyError as exc:
            raise ValueError(f"query was not encoded by {self.model_id}: {text}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-specs", type=Path, required=True)
    parser.add_argument("--query-artifacts", type=Path, required=True)
    parser.add_argument("--indexes-root", type=Path, required=True)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser


def _safe_child(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative")
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes its artifact root") from exc
    return resolved


def load_recorded_query_artifact(
    root: Path, provider: str, spec: ProviderSpec
) -> tuple[dict, np.ndarray, list[str], list[str]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version": "1.0",
        "model_id": provider,
        "model_name": spec.model_name,
        "checkpoint_or_pretrained": spec.checkpoint_or_pretrained,
        "model_revision": spec.model_revision,
        "source_revision": spec.source_revision,
        "dimension": spec.embedding_dimension,
        "dtype": spec.dtype,
        "normalized": spec.normalized,
        "adapter": EXPECTED_ADAPTERS[provider],
        "tokenizer": spec.tokenizer,
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"query manifest {field} mismatch for {provider}: "
                f"{manifest.get(field)!r} != {expected!r}"
            )
    if provider == "beit3":
        if manifest.get("checkpoint_sha256") != spec.checkpoint_sha256:
            raise ValueError("BEiT-3 query checkpoint SHA256 does not match provider lock")
        if manifest.get("tokenizer_sha256") != spec.tokenizer_sha256:
            raise ValueError("BEiT-3 query tokenizer SHA256 does not match provider lock")
    elif spec.checkpoint_sha256 is not None:
        if manifest.get("checkpoint_sha256") != spec.checkpoint_sha256:
            raise ValueError(f"{provider} query checkpoint SHA256 does not match provider lock")

    vector_path = _safe_child(root, manifest.get("vector_file"), "vector_file")
    if not vector_path.is_file():
        raise FileNotFoundError(f"query vector file not found: {vector_path}")
    if manifest.get("vector_sha256") != sha256_file(vector_path):
        raise ValueError(f"query vector SHA256 mismatch for {provider}")
    vectors = np.load(vector_path, allow_pickle=False)
    queries = manifest.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"query manifest contains no queries for {provider}")
    ids = [row.get("id") for row in queries if isinstance(row, dict)]
    texts = [row.get("text") for row in queries if isinstance(row, dict)]
    if (
        len(ids) != len(queries)
        or any(
            not isinstance(value, str) or not value.strip()
            for value in (*ids, *texts)
        )
        or len(ids) != len(set(ids))
    ):
        raise ValueError(f"invalid or duplicate query identities for {provider}")
    expected_shape = (len(queries), spec.embedding_dimension)
    if vectors.shape != expected_shape or str(vectors.dtype) != spec.dtype:
        raise ValueError(
            f"query vector contract mismatch for {provider}: "
            f"shape={vectors.shape}, dtype={vectors.dtype}"
        )
    if manifest.get("query_count") != len(queries) or not np.isfinite(vectors).all():
        raise ValueError(f"invalid query vector count/data for {provider}")
    if spec.normalized and not np.allclose(
        np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3, rtol=1e-3
    ):
        raise ValueError(f"query vectors are not normalized for {provider}")
    return manifest, vectors, ids, texts


def _load_contract(
    path: Path, provider: str, spec: ProviderSpec, embeddings_root: Path
) -> ProviderIndexContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "1.0",
        "model_id": provider,
        "dimension": spec.embedding_dimension,
        "metric": spec.similarity_metric,
        "normalized": spec.normalized,
        "backend": "faiss_index_flat_ip",
        "failed_count": 0,
        "missing_ids": 0,
        "extra_ids": 0,
        "duplicate_ids": 0,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"index manifest {field} mismatch for {provider}")
    collection = payload.get("collection_name")
    if not isinstance(collection, str) or not collection.endswith(f"__{provider}"):
        raise ValueError(f"index collection is not provider-namespaced for {provider}")
    if not isinstance(payload.get("indexed_count"), int) or payload["indexed_count"] <= 0:
        raise ValueError(f"index contains no validated vectors for {provider}")
    source_manifest = embeddings_root / provider / "manifest.json"
    if payload.get("source_embedding_manifest_sha256") != sha256_file(source_manifest):
        raise ValueError(f"source embedding manifest SHA256 mismatch for {provider}")
    return ProviderIndexContract(
        model_id=payload["model_id"],
        collection_name=payload["collection_name"],
        dimension=payload["dimension"],
        metric=payload["metric"],
        normalized=payload["normalized"],
        backend=payload["backend"],
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = load_provider_specs(args.provider_specs)
    manifests = {}
    adapters = {}
    query_ids: list[str] | None = None
    query_texts: list[str] | None = None
    for provider in PROVIDERS:
        root = args.query_artifacts / provider
        manifest, vectors, ids, texts = load_recorded_query_artifact(
            root, provider, specs.get(provider)
        )
        if query_ids is None:
            query_ids, query_texts = ids, texts
        elif ids != query_ids or texts != query_texts:
            raise ValueError("provider query identities/texts do not match")
        adapters[provider] = _RecordedTextAdapter(provider, vectors, texts)
        manifests[provider] = manifest

    factories = {
        provider: (lambda _spec, adapter=adapter: adapter)
        for provider, adapter in adapters.items()
    }
    encoders = QueryEncoderRegistry(specs, factories=factories)
    contracts = {
        provider: _load_contract(
            args.indexes_root / provider / "index_manifest.json",
            provider,
            specs.get(provider),
            args.embeddings_root,
        )
        for provider in PROVIDERS
    }

    cross_route_rejections = []
    try:
        encoders.encode_query("beit3", query_texts[0], index_contract=contracts["metaclip2"])
    except QueryRoutingError as exc:
        cross_route_rejections.append(str(exc))
    else:
        raise AssertionError("cross-provider query/index route was not rejected")

    retriever = IndependentSemanticRetriever(
        query_encoders=encoders,
        backend=FaissProviderSearchBackend(args.indexes_root / "collections"),
        contracts=contracts,
        failure_policy="fail_closed",
        required_providers=PROVIDERS,
        candidate_multiplier=1,
    )
    results = []
    for query_id, query_text in zip(query_ids, query_texts):
        search_started = time.perf_counter()
        response = retriever.search(query_text, provider_ids=PROVIDERS, top_k=args.top_k)
        search_total_ms = (time.perf_counter() - search_started) * 1000.0
        fusion_started = time.perf_counter()
        fused = fuse_independent_response(response, rrf_k=args.rrf_k, top_k=args.top_k)
        fusion_ms = (time.perf_counter() - fusion_started) * 1000.0
        real_encoding_ms = {
            provider: manifests[provider]["queries"][query_ids.index(query_id)]["encoding_ms"]
            for provider in PROVIDERS
        }
        results.append(
            {
                "query_id": query_id,
                "query": query_text,
                "independent": response.to_dict(),
                "rrf": fused.to_dict(),
                "latency_ms": {
                    "real_encoding_by_provider": real_encoding_ms,
                    "search_orchestration": search_total_ms,
                    "fusion": fusion_ms,
                    "total": sum(real_encoding_ms.values()) + search_total_ms + fusion_ms,
                },
            }
        )

    payload = {
        "schema_version": "1.0",
        "providers": list(PROVIDERS),
        "rrf_k": args.rrf_k,
        "cross_route_rejections": cross_route_rejections,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
