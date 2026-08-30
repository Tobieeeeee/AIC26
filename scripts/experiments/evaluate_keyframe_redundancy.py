"""Internal redundancy metric for keyframe-selector strategies.

Complements evaluate_keyframe_coverage.py, which only asks "is there some
kept frame near this candidate" -- a metric a strategy can win purely by
keeping more frames, even near-duplicates of each other. This script asks
the opposite question: among a strategy's OWN kept frames, how similar are
they to EACH OTHER? High internal redundancy means some of the "coverage"
win is just padding (near-duplicate keyframes), not genuinely distinct
content.

Reuses the exact same real ingest output as evaluate_keyframe_coverage.py;
no new video processing needed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mvr.enrichment.embeddings.metaclip2 import MetaClip2ImageEncoder
from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.ingestion.extract import extract_keyframes
from mvr.ingestion.keyframes import make_keyframes
from mvr.ingestion.models import Keyframe, Shot

STRATEGY_NAMES = ("midpoint", "periodic_12s_8s", "hybrid_temporal_content", "semantic_dedup")


def _shots(path: Path) -> list[Shot]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Shot(
            row["video_id"], row["method"], int(row["shot_id"]),
            int(row["start_frame"]), int(row["end_frame"]), float(row["fps"]),
        )
        for row in payload["shots"]
    ]


def _mapping_csv(path: Path) -> list[Keyframe]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        Keyframe(
            row["keyframe_id"], row["video_id"], int(row["shot_id"]),
            int(row["ordinal"]), int(row["frame_idx"]), float(row["timestamp"]),
            row["image_path"],
        )
        for row in rows
    ]


def _selected_vectors_from_own_grid(
    selected: list[Keyframe], candidate_mapping: list[Keyframe], candidate_vectors: np.ndarray
) -> np.ndarray:
    position_by_identity = {
        (item.shot_id, item.frame_idx): position
        for position, item in enumerate(candidate_mapping)
    }
    positions = [position_by_identity[(item.shot_id, item.frame_idx)] for item in selected]
    return candidate_vectors[positions]


def redundancy_score(vectors: np.ndarray, near_duplicate_threshold: float = 0.9) -> dict:
    """For each kept keyframe, similarity to its most similar OTHER kept
    keyframe (self excluded). High values -> many near-duplicates kept."""
    n = len(vectors)
    if n < 2:
        return {
            "mean_nearest_other_similarity": 0.0,
            "max_nearest_other_similarity": 0.0,
            "near_duplicate_fraction": 0.0,
            "near_duplicate_pairs": 0,
        }
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -1.0)  # exclude self-match
    nearest_other = similarities.max(axis=1)
    near_duplicate_pairs = int((np.triu(similarities, k=1) >= near_duplicate_threshold).sum())
    return {
        "mean_nearest_other_similarity": float(nearest_other.mean()),
        "max_nearest_other_similarity": float(nearest_other.max()),
        "near_duplicate_fraction": float((nearest_other >= near_duplicate_threshold).mean()),
        "near_duplicate_pairs": near_duplicate_pairs,
    }


def evaluate_video(
    *,
    video_id: str,
    video_path: Path,
    hybrid_root: Path,
    semantic_dedup_root: Path,
    image_encoder: MetaClip2ImageEncoder,
    scratch_root: Path,
) -> list[dict]:
    shots = _shots(hybrid_root / "shots.json")

    hybrid_cache = hybrid_root / "selector_cache" / video_id
    semantic_cache = semantic_dedup_root / "selector_cache" / video_id
    hybrid_candidates = _mapping_csv(hybrid_cache / "candidate_mapping.csv")
    hybrid_vectors = np.load(hybrid_cache / "candidate_embeddings.npy", allow_pickle=False)
    hybrid_vectors = hybrid_vectors / np.linalg.norm(hybrid_vectors, axis=1, keepdims=True)
    semantic_candidates = _mapping_csv(semantic_cache / "candidate_mapping.csv")
    semantic_vectors = np.load(semantic_cache / "candidate_embeddings.npy", allow_pickle=False)
    semantic_vectors = semantic_vectors / np.linalg.norm(semantic_vectors, axis=1, keepdims=True)

    hybrid_selected = _mapping_csv(hybrid_root / "keyframe_mapping.csv")
    semantic_selected = _mapping_csv(semantic_dedup_root / "keyframe_mapping.csv")
    midpoint = make_keyframes(shots, float("inf"), 8.0)
    periodic = make_keyframes(shots, 12.0, 8.0)

    midpoint_root = scratch_root / video_id / "midpoint"
    periodic_root = scratch_root / video_id / "periodic"
    if not midpoint_root.exists():
        extract_keyframes(video_path, midpoint, midpoint_root)
    if not periodic_root.exists():
        extract_keyframes(video_path, periodic, periodic_root)

    def encode(root: Path, frames: list[Keyframe]) -> np.ndarray:
        vectors = image_encoder.encode_images([root / f.image_path for f in frames])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    selected_vectors = {
        "midpoint": encode(midpoint_root, midpoint),
        "periodic_12s_8s": encode(periodic_root, periodic),
        "hybrid_temporal_content": _selected_vectors_from_own_grid(
            hybrid_selected, hybrid_candidates, hybrid_vectors
        ),
        "semantic_dedup": _selected_vectors_from_own_grid(
            semantic_selected, semantic_candidates, semantic_vectors
        ),
    }
    selected_counts = {
        "midpoint": len(midpoint),
        "periodic_12s_8s": len(periodic),
        "hybrid_temporal_content": len(hybrid_selected),
        "semantic_dedup": len(semantic_selected),
    }

    rows = []
    for name in STRATEGY_NAMES:
        metrics = redundancy_score(selected_vectors[name])
        rows.append(
            {"video_id": video_id, "strategy": name, "selected_keyframes": selected_counts[name], **metrics}
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--videos-root", type=Path, required=True)
    parser.add_argument("--ingest-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--provider-specs", type=Path, default=Path("configs/embedding_providers.json"))
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.9)
    args = parser.parse_args()

    gt = json.loads(args.groundtruth.read_text(encoding="utf-8"))["test_cases"]
    video_ids = sorted(
        {
            row["video_id"]
            for row in gt
            if (args.ingest_root / f"{row['video_id']}_hybrid").is_dir()
            and (args.ingest_root / f"{row['video_id']}_semantic_dedup").is_dir()
        }
    )
    print(f"videos: {video_ids}")

    spec = load_provider_specs(args.provider_specs).get("metaclip2")
    image_encoder = MetaClip2ImageEncoder(spec=spec, device=args.device, allow_download=False)

    all_rows: list[dict] = []
    for video_id in video_ids:
        collection = video_id.split("_")[0]
        video_path = args.videos_root / f"Videos_{collection}_a" / "video" / f"{video_id}.mp4"
        print(f"=== {video_id} ===")
        all_rows.extend(
            evaluate_video(
                video_id=video_id,
                video_path=video_path,
                hybrid_root=args.ingest_root / f"{video_id}_hybrid",
                semantic_dedup_root=args.ingest_root / f"{video_id}_semantic_dedup",
                image_encoder=image_encoder,
                scratch_root=args.scratch_root,
            )
        )

    agg_fields = (
        "mean_nearest_other_similarity", "max_nearest_other_similarity",
        "near_duplicate_fraction", "near_duplicate_pairs", "selected_keyframes",
    )
    per_strategy: dict[str, dict[str, list[float]]] = {
        name: {field: [] for field in agg_fields} for name in STRATEGY_NAMES
    }
    for row in all_rows:
        for field in agg_fields:
            per_strategy[row["strategy"]][field].append(row[field])
    summary = [
        {"strategy": name, "videos": len(video_ids), **{f: float(np.mean(v[f])) for f in agg_fields}}
        for name, v in per_strategy.items()
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "redundancy_per_video_strategy.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output / "redundancy_summary.json").write_text(
        json.dumps({"videos": video_ids, "strategies": summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
