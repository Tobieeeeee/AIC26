"""Produce long-shot, diversity, and paired-bootstrap KF02 evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _pairwise_distance(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    distances = 1.0 - normalized @ normalized.T
    upper = distances[np.triu_indices(len(vectors), 1)]
    return float(np.mean(upper))


def _bootstrap_delta(
    left: list[int | None], right: list[int | None], *, seed: int = 260822
) -> dict:
    rng = np.random.default_rng(seed)
    left_r10 = np.array([rank is not None and rank <= 10 for rank in left], dtype=float)
    right_r10 = np.array([rank is not None and rank <= 10 for rank in right], dtype=float)
    left_rr = np.array([0 if rank is None else 1 / rank for rank in left], dtype=float)
    right_rr = np.array([0 if rank is None else 1 / rank for rank in right], dtype=float)
    samples = rng.integers(0, len(left), size=(10_000, len(left)))
    r10_delta = (right_r10[samples] - left_r10[samples]).mean(axis=1)
    mrr_delta = (right_rr[samples] - left_rr[samples]).mean(axis=1)
    return {
        "query_count": len(left),
        "recall_at_10_delta": float(right_r10.mean() - left_r10.mean()),
        "recall_at_10_delta_95ci": [float(x) for x in np.quantile(r10_delta, [0.025, 0.975])],
        "mrr_delta": float(right_rr.mean() - left_rr.mean()),
        "mrr_delta_95ci": [float(x) for x in np.quantile(mrr_delta, [0.025, 0.975])],
        "bootstrap_replicates": 10_000,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root
    frozen = json.loads((root / "frozen_shots.json").read_text(encoding="utf-8"))
    shots = frozen["shots"]
    gt = _csv(args.groundtruth)
    candidates = _csv(root / "candidate_mapping.csv")
    position = {
        (row["video_id"], int(row["shot_id"]), int(row["frame_idx"])): i
        for i, row in enumerate(candidates)
    }
    embeddings = {
        provider: np.load(root / "selector_embeddings" / provider / "candidate_embeddings.npy")
        for provider in ("metaclip2", "perception_encoder")
    }
    strategy_names = [
        "S0-midpoint", "S1-12s-8s", "S2-15s-5s",
        "S3-metaclip2-budget-S1", "S3-perception_encoder-budget-S1",
    ]
    mappings = {name: _csv(root / "strategies" / name / "keyframe_mapping.csv") for name in strategy_names}
    by_strategy_shot: dict[str, dict[int, list[dict[str, str]]]] = {}
    for name, rows in mappings.items():
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["shot_id"])].append(row)
        by_strategy_shot[name] = grouped

    long_shot_summary = []
    for threshold in (12.0, 30.0):
        eligible = [shot for shot in shots if float(shot["duration"]) >= threshold]
        for name in strategy_names:
            rows = [row for shot in eligible for row in by_strategy_shot[name][int(shot["shot_id"])]]
            per_shot_diversity = []
            for shot in eligible:
                chosen = by_strategy_shot[name][int(shot["shot_id"])]
                indices = [position[(row["video_id"], int(row["shot_id"]), int(row["frame_idx"]))] for row in chosen]
                per_shot_diversity.append(_pairwise_distance(embeddings["metaclip2"][indices]))
            overlapping_gt = [
                row for row in gt
                if any(int(row["start_frame"]) <= int(shot["end_frame"]) and int(row["end_frame"]) >= int(shot["start_frame"]) for shot in eligible)
            ]
            covered = sum(
                any(int(row["start_frame"]) <= int(item["frame_idx"]) <= int(row["end_frame"]) for item in rows)
                for row in overlapping_gt
            )
            long_shot_summary.append(
                {
                    "minimum_duration_seconds": threshold,
                    "strategy": name,
                    "shot_count": len(eligible),
                    "selected_keyframes": len(rows),
                    "keyframes_per_shot": len(rows) / len(eligible) if eligible else None,
                    "mean_per_shot_pairwise_cosine_distance_metaclip2": float(np.mean(per_shot_diversity)) if eligible else None,
                    "overlapping_gt_segments": len(overlapping_gt),
                    "covered_overlapping_gt_segments": covered,
                }
            )

    distance_distributions = {}
    candidates_by_shot: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(candidates):
        candidates_by_shot[int(row["shot_id"])].append(index)
    for provider, vectors in embeddings.items():
        normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        distances = []
        for shot in shots:
            indices = candidates_by_shot[int(shot["shot_id"])]
            midpoint = (int(shot["start_frame"]) + int(shot["end_frame"])) // 2
            midpoint_index = next(i for i in indices if int(candidates[i]["frame_idx"]) == midpoint)
            distances.extend(
                1.0 - float(normalized[i] @ normalized[midpoint_index])
                for i in indices if i != midpoint_index
            )
        distance_distributions[provider] = {
            "count": len(distances),
            "min_p25_p50_p75_p95_max": [
                float(x) for x in np.quantile(distances, [0, 0.25, 0.5, 0.75, 0.95, 1])
            ],
        }

    longest = sorted(shots, key=lambda row: (-float(row["duration"]), int(row["shot_id"])))[:5]
    examples = []
    for shot in longest:
        shot_id = int(shot["shot_id"])
        example = {
            "shot_id": shot_id,
            "duration": shot["duration"],
            "start_frame": shot["start_frame"],
            "end_frame": shot["end_frame"],
            "relevant_query_ids": [
                row["query_id"] for row in gt
                if int(row["start_frame"]) <= int(shot["end_frame"]) and int(row["end_frame"]) >= int(shot["start_frame"])
            ],
            "selections": {},
        }
        for name in strategy_names:
            rows = by_strategy_shot[name][shot_id]
            indices = [position[(row["video_id"], shot_id, int(row["frame_idx"]))] for row in rows]
            example["selections"][name] = {
                "frames": [int(row["frame_idx"]) for row in rows],
                "pairwise_cosine_distance_metaclip2": _pairwise_distance(embeddings["metaclip2"][indices]),
            }
        examples.append(example)

    ranks_by_strategy: dict[str, list[int | None]] = defaultdict(list)
    for row in _csv(root / "retrieval_per_query.csv"):
        ranks_by_strategy[row["strategy"]].append(
            None if row["first_relevant_rank"] == "" else int(row["first_relevant_rank"])
        )
    comparisons = {}
    for left, right in (
        ("S0-midpoint", "S1-12s-8s"),
        ("S1-12s-8s", "S2-15s-5s"),
        ("S1-12s-8s", "S3-metaclip2-budget-S1"),
        ("S1-12s-8s", "S3-perception_encoder-budget-S1"),
    ):
        comparisons[f"{right}_minus_{left}"] = _bootstrap_delta(
            ranks_by_strategy[left], ranks_by_strategy[right]
        )

    frame_overlap = {}
    s1_identities = {(int(row["shot_id"]), int(row["frame_idx"])) for row in mappings["S1-12s-8s"]}
    for name in ("S3-metaclip2-budget-S1", "S3-perception_encoder-budget-S1"):
        identities = {(int(row["shot_id"]), int(row["frame_idx"])) for row in mappings[name]}
        frame_overlap[name] = {
            "same_count": len(identities) == len(s1_identities),
            "shared_frames": len(identities & s1_identities),
            "different_placements_each_side": len(identities - s1_identities),
        }

    payload = {
        "corpus": {
            "videos": 1,
            "shots": len(shots),
            "shots_ge_12s": sum(float(shot["duration"]) >= 12 for shot in shots),
            "shots_ge_30s": sum(float(shot["duration"]) >= 30 for shot in shots),
        },
        "distance_distributions": distance_distributions,
        "long_shot_summary": long_shot_summary,
        "five_longest_shot_examples": examples,
        "paired_bootstrap": comparisons,
        "budget_s1_frame_overlap": frame_overlap,
    }
    (root / "analysis_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
