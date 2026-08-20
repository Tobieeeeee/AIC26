"""Search three real provider indexes and fuse their independent rankings."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mvr.enrichment.embeddings.query_encoders import QueryEncoderRegistry, QueryRoutingError
from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.indexing.provider_indexes import ProviderIndexContract
from mvr.retrieval.provider_semantic import (
    FaissProviderSearchBackend,
    IndependentSemanticRetriever,
)
from mvr.retrieval.semantic_rrf import fuse_independent_response


PROVIDERS = ("beit3", "metaclip2", "perception_encoder")


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser


def _load_contract(path: Path) -> ProviderIndexContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
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
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        vectors = np.load(root / manifest["vector_file"], allow_pickle=False)
        ids = [row["id"] for row in manifest["queries"]]
        texts = [row["text"] for row in manifest["queries"]]
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
        provider: _load_contract(args.indexes_root / provider / "index_manifest.json")
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
