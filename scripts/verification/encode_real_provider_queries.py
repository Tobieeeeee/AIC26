"""Encode a shared query set inside one isolated real provider runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mvr.enrichment.embeddings.query_encoders import (
    ProviderQueryEncoderConfig,
    QueryEncoderRegistry,
)
from mvr.enrichment.embeddings.provider_specs import load_provider_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--provider-specs", type=Path, required=True)
    parser.add_argument("--queries-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beit3-checkpoint", type=Path)
    parser.add_argument("--beit3-tokenizer", type=Path)
    parser.add_argument("--metaclip2-model")
    parser.add_argument("--perception-encoder-checkpoint", type=Path)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    query_config = json.loads(args.queries_json.read_text(encoding="utf-8"))
    queries = query_config.get("queries")
    if not isinstance(queries, list) or len(queries) < 3:
        raise ValueError("queries-json must contain at least three queries")
    if any(not isinstance(row, dict) or not row.get("id") or not row.get("text") for row in queries):
        raise ValueError("every query requires non-empty id and text")

    specs = load_provider_specs(args.provider_specs)
    registry = QueryEncoderRegistry(
        specs,
        ProviderQueryEncoderConfig(
            device=args.device,
            beit3_checkpoint=args.beit3_checkpoint,
            beit3_tokenizer=args.beit3_tokenizer,
            metaclip2_model=args.metaclip2_model,
            perception_encoder_checkpoint=args.perception_encoder_checkpoint,
        ),
    )
    vectors = []
    query_evidence = []
    for query in queries:
        started = time.perf_counter()
        embedding = registry.encode_query(args.provider, str(query["text"]))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        vectors.append(embedding.vector)
        query_evidence.append(
            {
                "id": str(query["id"]),
                "text": str(query["text"]),
                "encoding_ms": elapsed_ms,
                "norm": float(np.linalg.norm(embedding.vector)),
            }
        )

    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    output = args.output_root / args.provider
    if output.exists():
        raise FileExistsError(f"query artifact already exists: {output}")
    output.mkdir(parents=True)
    vector_path = output / "query_vectors.npy"
    np.save(vector_path, matrix, allow_pickle=False)
    spec = specs.get(args.provider)
    adapter = registry._cache[args.provider]
    manifest = {
        "schema_version": "1.0",
        "model_id": args.provider,
        "model_name": spec.model_name,
        "checkpoint_or_pretrained": spec.checkpoint_or_pretrained,
        "model_revision": spec.model_revision,
        "source_revision": spec.source_revision,
        "adapter": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "tokenizer": spec.tokenizer,
        "dimension": spec.embedding_dimension,
        "dtype": str(matrix.dtype),
        "normalized": spec.normalized,
        "device": args.device,
        "query_count": len(queries),
        "queries": query_evidence,
        "vector_file": vector_path.name,
        "vector_sha256": _sha256(vector_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
