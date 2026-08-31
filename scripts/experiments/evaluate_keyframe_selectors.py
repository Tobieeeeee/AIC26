"""Multi-video, 4-strategy keyframe-selector comparison.

Strategies: midpoint, periodic (12s/8s, former baseline),
hybrid_temporal_content, semantic_dedup. Each strategy's own selected
keyframes are extracted (if not already on disk) and encoded directly with
MetaCLIP-2 -- unlike the original single-video
evaluate_hybrid_keyframe_selector.py, this does not rely on a shared dense
candidate grid + position lookup, which breaks when a strategy's frame
indices don't happen to coincide with that grid's sampling points.

Requires real hybrid_temporal_content and semantic_dedup ingest output per
video (run scripts/run_ingestion.py with each --selector-mode first).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mvr.enrichment.embeddings.metaclip2 import MetaClip2ImageEncoder
from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.enrichment.embeddings.query_encoders import MetaClip2TextEncoder
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


def _retrieval_metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    return {
        "recall_at_1": sum(r is not None and r <= 1 for r in ranks) / n,
        "recall_at_10": sum(r is not None and r <= 10 for r in ranks) / n,
        "recall_at_100": sum(r is not None and r <= 100 for r in ranks) / n,
        "mrr": sum(0.0 if r is None else 1.0 / r for r in ranks) / n,
    }


def _encode_frames(encoder: MetaClip2ImageEncoder, keyframe_root: Path, frames: list[Keyframe]) -> np.ndarray:
    paths = [keyframe_root / frame.image_path for frame in frames]
    vectors = encoder.encode_images(paths)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def evaluate_video(
    *,
    video_id: str,
    video_path: Path,
    hybrid_root: Path,
    semantic_dedup_root: Path,
    gt_rows: list[dict],
    image_encoder: MetaClip2ImageEncoder,
    text_encoder: MetaClip2TextEncoder,
    scratch_root: Path,
) -> dict:
    shots = _shots(hybrid_root / "shots.json")
    hybrid_manifest = json.loads((hybrid_root / "selector_manifest.json").read_text(encoding="utf-8"))
    semantic_manifest = json.loads((semantic_dedup_root / "selector_manifest.json").read_text(encoding="utf-8"))

    start = time.perf_counter()
    midpoint = make_keyframes(shots, float("inf"), 8.0)
    midpoint_seconds = time.perf_counter() - start

    start = time.perf_counter()
    periodic = make_keyframes(shots, 12.0, 8.0)
    periodic_seconds = time.perf_counter() - start

    hybrid = _mapping_csv(hybrid_root / "keyframe_mapping.csv")
    semantic_dedup = _mapping_csv(semantic_dedup_root / "keyframe_mapping.csv")

    # midpoint/periodic were never extracted by any real ingest run for this
    # video (only hybrid_temporal_content and semantic_dedup were) -- extract
    # them fresh from the source video so every strategy is scored on its
    # own real images, not a borrowed/coincidental candidate grid position.
    midpoint_root = scratch_root / video_id / "midpoint"
    periodic_root = scratch_root / video_id / "periodic"
    extract_keyframes(video_path, midpoint, midpoint_root)
    extract_keyframes(video_path, periodic, periodic_root)

    strategies = {
        "midpoint": (midpoint, midpoint_root, midpoint_seconds),
        "periodic_12s_8s": (periodic, periodic_root, periodic_seconds),
        "hybrid_temporal_content": (
            hybrid, hybrid_root, hybrid_manifest["selector_only_seconds"],
        ),
        "semantic_dedup": (
            semantic_dedup, semantic_dedup_root, semantic_manifest["selector_only_seconds"],
        ),
    }

    video_gt = [row for row in gt_rows if row["video_id"] == video_id]
    query_vectors = []
    for row in video_gt:
        vector = np.asarray(text_encoder.encode_text(row["query_en"]), dtype=np.float32).reshape(-1)
        query_vectors.append(vector / np.linalg.norm(vector))

    rows = []
    per_query = []
    for name in STRATEGY_NAMES:
        frames, root, selector_seconds = strategies[name]
        encode_start = time.perf_counter()
        vectors = _encode_frames(image_encoder, root, frames)
        encoding_seconds = time.perf_counter() - encode_start

        ranks: list[int | None] = []
        for gt_row, query in zip(video_gt, query_vectors, strict=True):
            scores = vectors @ query
            order = np.argsort(-scores, kind="stable")
            rank = None
            for offset, position in enumerate(order, start=1):
                frame_idx = frames[int(position)].frame_idx
                if gt_row["start_frame"] <= frame_idx <= gt_row["end_frame"]:
                    rank = offset
                    break
            ranks.append(rank)
            per_query.append(
                {
                    "video_id": video_id,
                    "strategy": name,
                    "query_id": gt_row["query_id"],
                    "first_relevant_rank": "" if rank is None else rank,
                }
            )

        rows.append(
            {
                "video_id": video_id,
                "strategy": name,
                "total_keyframes": len(frames),
                "max_observed_temporal_gap_sec": _max_gap(shots, frames),
                "selector_only_seconds": selector_seconds,
                "encoding_seconds": encoding_seconds,
                **_retrieval_metrics(ranks),
            }
        )

    return {"strategy_rows": rows, "per_query": per_query, "gt_count": len(video_gt)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument(
        "--videos-root", type=Path, required=True,
        help="Dataset root containing Videos_<collection>_a/video/<video_id>.mp4",
    )
    parser.add_argument(
        "--ingest-root", type=Path, required=True,
        help="Root containing <video_id>_hybrid and <video_id>_semantic_dedup ingest dirs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--provider-specs", type=Path, default=Path("configs/embedding_providers.json"))
    args = parser.parse_args()

    payload = json.loads(args.groundtruth.read_text(encoding="utf-8"))
    gt = payload["test_cases"]
    video_ids = sorted(
        {
            row["video_id"]
            for row in gt
            if (args.ingest_root / f"{row['video_id']}_hybrid").is_dir()
            and (args.ingest_root / f"{row['video_id']}_semantic_dedup").is_dir()
        }
    )
    if not video_ids:
        raise SystemExit(
            "no GT video has both <video_id>_hybrid and <video_id>_semantic_dedup "
            f"ingest output under {args.ingest_root}"
        )
    print(f"videos with usable GT + ingest output: {video_ids}")

    spec = load_provider_specs(args.provider_specs).get("metaclip2")
    image_encoder = MetaClip2ImageEncoder(spec=spec, device=args.device, allow_download=False)
    text_encoder = MetaClip2TextEncoder(spec=spec, device=args.device, allow_download=False)

    scratch_root = args.output / "_scratch_midpoint_periodic"
    all_rows: list[dict] = []
    all_per_query: list[dict] = []
    total_gt = 0
    for video_id in video_ids:
        collection = video_id.split("_")[0]
        video_path = args.videos_root / f"Videos_{collection}_a" / "video" / f"{video_id}.mp4"
        if not video_path.is_file():
            raise SystemExit(f"raw video not found for {video_id}: {video_path}")
        print(f"=== evaluating {video_id} ===")
        result = evaluate_video(
            video_id=video_id,
            video_path=video_path,
            hybrid_root=args.ingest_root / f"{video_id}_hybrid",
            semantic_dedup_root=args.ingest_root / f"{video_id}_semantic_dedup",
            gt_rows=gt,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            scratch_root=scratch_root,
        )
        all_rows.extend(result["strategy_rows"])
        all_per_query.extend(result["per_query"])
        total_gt += result["gt_count"]
        print(f"    {result['gt_count']} GT queries scored")

    agg_fields = (
        "recall_at_1", "recall_at_10", "recall_at_100", "mrr",
        "selector_only_seconds", "encoding_seconds", "total_keyframes",
        "max_observed_temporal_gap_sec",
    )
    per_strategy: dict[str, dict[str, list[float]]] = {
        name: {field: [] for field in agg_fields} for name in STRATEGY_NAMES
    }
    for row in all_rows:
        for field in agg_fields:
            per_strategy[row["strategy"]][field].append(row[field])

    summary_rows = []
    for name in STRATEGY_NAMES:
        values = per_strategy[name]
        summary_rows.append(
            {
                "strategy": name,
                "videos": len(video_ids),
                "total_gt_queries": total_gt,
                **{field: float(np.mean(values[field])) for field in agg_fields},
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_video_strategy_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output / "retrieval_per_query.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_per_query[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_per_query)
    with (args.output / "summary_across_videos.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.output / "summary.json").write_text(
        json.dumps({"videos": video_ids, "total_gt_queries": total_gt, "strategies": summary_rows},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
