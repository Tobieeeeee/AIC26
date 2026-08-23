"""One-video regression comparison for the canonical hybrid selector."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.enrichment.embeddings.query_encoders import MetaClip2TextEncoder
from mvr.ingestion.keyframes import make_keyframes
from mvr.ingestion.models import Keyframe, Shot


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _shots(path: Path) -> list[Shot]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Shot(
            row["video_id"], row["method"], int(row["shot_id"]),
            int(row["start_frame"]), int(row["end_frame"]), float(row["fps"]),
        )
        for row in payload["shots"]
    ]


def _mapping(path: Path) -> list[Keyframe]:
    return [
        Keyframe(
            row["keyframe_id"], row["video_id"], int(row["shot_id"]),
            int(row["ordinal"]), int(row["frame_idx"]), float(row["timestamp"]),
            row["image_path"],
        )
        for row in _csv(path)
    ]


def _max_gap(shots: list[Shot], frames: list[Keyframe]) -> float:
    by_shot: dict[int, list[int]] = defaultdict(list)
    for frame in frames:
        by_shot[frame.shot_id].append(frame.frame_idx)
    observed = 0.0
    for shot in shots:
        points = [shot.start_frame, *sorted(by_shot[shot.shot_id]), shot.end_frame]
        observed = max(
            observed,
            max((right - left) / shot.fps for left, right in zip(points, points[1:])),
        )
    return observed


def _selection_metrics(gt: list[dict[str, str]], frames: list[Keyframe]):
    indices = [frame.frame_idx for frame in frames]
    covered = [
        any(int(row["start_frame"]) <= frame <= int(row["end_frame"]) for frame in indices)
        for row in gt
    ]
    high = [position for position, row in enumerate(gt) if row["annotation_confidence"] == "high"]
    return sum(covered), sum(covered[position] for position in high), len(high)


def _retrieval_metrics(ranks: list[int | None]) -> dict[str, float]:
    return {
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / len(ranks),
        "recall_at_100": sum(rank is not None and rank <= 100 for rank in ranks) / len(ranks),
        "mrr": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--provider-specs", type=Path, default=Path("configs/embedding_providers.json"))
    args = parser.parse_args()

    shots = _shots(args.ingest_root / "shots.json")
    gt = _csv(args.groundtruth)
    selector_manifest = json.loads(
        (args.ingest_root / "selector_manifest.json").read_text(encoding="utf-8")
    )
    candidate_mapping = _mapping(
        args.ingest_root / "selector_cache" / shots[0].video_id / "candidate_mapping.csv"
    )
    candidate_vectors = np.load(
        args.ingest_root / "selector_cache" / shots[0].video_id / "candidate_embeddings.npy",
        allow_pickle=False,
    ).astype(np.float32)
    candidate_vectors /= np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
    candidate_position = {
        (item.shot_id, item.frame_idx): position
        for position, item in enumerate(candidate_mapping)
    }

    selection_start = time.perf_counter()
    midpoint = make_keyframes(shots, float("inf"), 8.0)
    midpoint_seconds = time.perf_counter() - selection_start
    selection_start = time.perf_counter()
    periodic = make_keyframes(shots, 12.0, 8.0)
    periodic_seconds = time.perf_counter() - selection_start
    hybrid = _mapping(args.ingest_root / "keyframe_mapping.csv")
    strategies = {
        "midpoint": midpoint,
        "periodic_12s_8s": periodic,
        "hybrid_temporal_content": hybrid,
    }

    spec = load_provider_specs(args.provider_specs).get("metaclip2")
    text_encoder = MetaClip2TextEncoder(spec=spec, device=args.device, allow_download=False)
    query_start = time.perf_counter()
    query_vectors = []
    for row in gt:
        vector = np.asarray(text_encoder.encode_text(row["query_en"]), dtype=np.float32).reshape(-1)
        query_vectors.append(vector / np.linalg.norm(vector))
    query_embedding_seconds = time.perf_counter() - query_start

    rows = []
    per_query = []
    selector_seconds = {
        "midpoint": midpoint_seconds,
        "periodic_12s_8s": periodic_seconds,
        "hybrid_temporal_content": selector_manifest["selector_only_seconds"],
    }
    for name, frames in strategies.items():
        positions = [candidate_position[(item.shot_id, item.frame_idx)] for item in frames]
        vectors = candidate_vectors[positions]
        ranks: list[int | None] = []
        for gt_row, query in zip(gt, query_vectors, strict=True):
            scores = vectors @ query
            order = np.argsort(-scores, kind="stable")
            rank = None
            for offset, selected_position in enumerate(order, start=1):
                frame = frames[int(selected_position)].frame_idx
                if int(gt_row["start_frame"]) <= frame <= int(gt_row["end_frame"]):
                    rank = offset
                    break
            ranks.append(rank)
            per_query.append(
                {
                    "strategy": name,
                    "query_id": gt_row["query_id"],
                    "annotation_confidence": gt_row["annotation_confidence"],
                    "first_relevant_rank": "" if rank is None else rank,
                }
            )
        covered, high_covered, high_total = _selection_metrics(gt, frames)
        jpeg_bytes = sum(
            (args.ingest_root / candidate_mapping[position].image_path).stat().st_size
            for position in positions
        )
        row = {
            "strategy": name,
            "total_keyframes": len(frames),
            "gt_covered": covered,
            "gt_total": len(gt),
            "selection_recall": covered / len(gt),
            "high_gt_covered": high_covered,
            "high_gt_total": high_total,
            "high_selection_recall": high_covered / high_total,
            "max_observed_temporal_gap_sec": _max_gap(shots, frames),
            "content_event_selections": selector_manifest["content_event_selection_count"] if name == "hybrid_temporal_content" else 0,
            "temporal_repair_selections": selector_manifest["temporal_repair_selection_count"] if name == "hybrid_temporal_content" else 0,
            "candidate_count": len(candidate_mapping) if name == "hybrid_temporal_content" else 0,
            "jpeg_bytes": jpeg_bytes,
            "candidate_extraction_seconds": selector_manifest["candidate_extraction_seconds"] if name == "hybrid_temporal_content" else 0.0,
            "candidate_embedding_seconds": selector_manifest["candidate_embedding_seconds"] if name == "hybrid_temporal_content" else 0.0,
            "selector_only_seconds": selector_seconds[name],
            **_retrieval_metrics(ranks),
        }
        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "strategy_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "retrieval_per_query.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_query[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_query)
    summary = {
        "video_id": shots[0].video_id,
        "shot_count": len(shots),
        "groundtruth_rows": len(gt),
        "high_confidence_rows": sum(row["annotation_confidence"] == "high" for row in gt),
        "query_language": "en",
        "retrieval_provider": spec.checkpoint_or_pretrained,
        "retrieval_revision": spec.model_revision,
        "query_embedding_seconds_shared": query_embedding_seconds,
        "strategies": rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
