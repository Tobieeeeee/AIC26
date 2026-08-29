"""Fixed-provider secondary retrieval evaluation for the KF02 pilot."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import faiss
import numpy as np

from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.enrichment.embeddings.query_encoders import MetaClip2TextEncoder


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def retrieval_metrics(first_relevant_ranks: list[int | None]) -> dict[str, float]:
    if not first_relevant_ranks:
        raise ValueError("at least one query rank is required")
    return {
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in first_relevant_ranks) / len(first_relevant_ranks),
        "recall_at_10": sum(rank is not None and rank <= 10 for rank in first_relevant_ranks) / len(first_relevant_ranks),
        "recall_at_100": sum(rank is not None and rank <= 100 for rank in first_relevant_ranks) / len(first_relevant_ranks),
        "mrr": sum(0.0 if rank is None else 1.0 / rank for rank in first_relevant_ranks) / len(first_relevant_ranks),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-language", choices=["en", "vi"], default="en")
    args = parser.parse_args()

    gt = _read_csv(args.groundtruth)
    candidates = _read_csv(args.experiment_root / "candidate_mapping.csv")
    vectors = np.load(
        args.experiment_root / "selector_embeddings" / "metaclip2" / "candidate_embeddings.npy"
    ).astype(np.float32)
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.ascontiguousarray(vectors / vector_norms, dtype=np.float32)
    position_by_identity = {
        (row["video_id"], int(row["shot_id"]), int(row["frame_idx"])): position
        for position, row in enumerate(candidates)
    }

    spec = load_provider_specs(Path("configs/embedding_providers.json")).get("metaclip2")
    encoder = MetaClip2TextEncoder(spec=spec, device=args.device, allow_download=False)
    query_field = f"query_{args.query_language}"
    query_start = time.perf_counter()
    query_vectors = []
    for row in gt:
        vector = np.asarray(encoder.encode_text(row[query_field]), dtype=np.float32).reshape(-1)
        query_vectors.append(vector / np.linalg.norm(vector))
    query_seconds = time.perf_counter() - query_start

    metric_rows = []
    per_query_rows = []
    index_root = args.experiment_root / "retrieval_indexes"
    index_root.mkdir(exist_ok=True)
    for strategy_dir in sorted((args.experiment_root / "strategies").iterdir()):
        mapping = _read_csv(strategy_dir / "keyframe_mapping.csv")
        positions = [
            position_by_identity[(row["video_id"], int(row["shot_id"]), int(row["frame_idx"]))]
            for row in mapping
        ]
        strategy_vectors = np.ascontiguousarray(vectors[positions], dtype=np.float32)
        index = faiss.IndexFlatIP(strategy_vectors.shape[1])
        index.add(strategy_vectors)
        index_path = index_root / f"{strategy_dir.name}.faiss"
        faiss.write_index(index, str(index_path))
        embedding_path = index_root / f"{strategy_dir.name}.npy"
        np.save(embedding_path, strategy_vectors)

        ranks: list[int | None] = []
        high_ranks: list[int | None] = []
        search_start = time.perf_counter()
        for row, query_vector in zip(gt, query_vectors, strict=True):
            _, neighbors = index.search(query_vector[None, :], len(mapping))
            rank = None
            for offset, neighbor in enumerate(neighbors[0], start=1):
                frame = int(mapping[int(neighbor)]["frame_idx"])
                if row["video_id"] == mapping[int(neighbor)]["video_id"] and int(row["start_frame"]) <= frame <= int(row["end_frame"]):
                    rank = offset
                    break
            ranks.append(rank)
            if row["annotation_confidence"] == "high":
                high_ranks.append(rank)
            per_query_rows.append(
                {
                    "strategy": strategy_dir.name,
                    "query_id": row["query_id"],
                    "query_language": args.query_language,
                    "annotation_confidence": row["annotation_confidence"],
                    "first_relevant_rank": "" if rank is None else rank,
                }
            )
        search_seconds = time.perf_counter() - search_start
        metrics = retrieval_metrics(ranks)
        high_metrics = retrieval_metrics(high_ranks)
        metric_rows.append(
            {
                "strategy": strategy_dir.name,
                "query_language": args.query_language,
                "query_count": len(ranks),
                **metrics,
                "high_query_count": len(high_ranks),
                "high_recall_at_1": high_metrics["recall_at_1"],
                "high_recall_at_10": high_metrics["recall_at_10"],
                "high_recall_at_100": high_metrics["recall_at_100"],
                "high_mrr": high_metrics["mrr"],
                "query_embedding_seconds_shared": query_seconds,
                "search_seconds": search_seconds,
                "retrieval_embedding_bytes": embedding_path.stat().st_size,
                "faiss_index_bytes": index_path.stat().st_size,
            }
        )

    _write_csv(args.experiment_root / "retrieval_metrics.csv", metric_rows, list(metric_rows[0]))
    _write_csv(args.experiment_root / "retrieval_per_query.csv", per_query_rows, list(per_query_rows[0]))
    (args.experiment_root / "retrieval_manifest.json").write_text(
        json.dumps(
            {
                "provider": "metaclip2",
                "checkpoint": spec.checkpoint_or_pretrained,
                "revision": spec.model_revision,
                "query_language": args.query_language,
                "query_count": len(gt),
                "query_embedding_seconds_shared": query_seconds,
                "search": "FAISS IndexFlatIP over normalized float32 vectors",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metric_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
