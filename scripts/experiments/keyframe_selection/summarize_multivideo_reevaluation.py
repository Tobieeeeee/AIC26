"""Derive comparison evidence from the measured four-video artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _rank(value: str) -> float:
    return math.inf if value == "" else float(value)


def _pareto(rows: Sequence[dict[str, str]]) -> list[str]:
    """Nondominated pooled quality / KF / embedding-time / storage strategies."""
    result = []
    for candidate in rows:
        dominated = False
        candidate_values = (
            -_number(candidate, "pooled_selection_recall"),
            -_number(candidate, "pooled_recall_at_1"),
            -_number(candidate, "pooled_recall_at_10"),
            -_number(candidate, "pooled_recall_at_100"),
            -_number(candidate, "pooled_mrr"),
            _number(candidate, "selected_keyframes"),
            _number(candidate, "embedding_seconds_shared"),
            _number(candidate, "vector_bytes") + _number(candidate, "index_bytes"),
        )
        for other in rows:
            if other is candidate:
                continue
            other_values = (
                -_number(other, "pooled_selection_recall"),
                -_number(other, "pooled_recall_at_1"),
                -_number(other, "pooled_recall_at_10"),
                -_number(other, "pooled_recall_at_100"),
                -_number(other, "pooled_mrr"),
                _number(other, "selected_keyframes"),
                _number(other, "embedding_seconds_shared"),
                _number(other, "vector_bytes") + _number(other, "index_bytes"),
            )
            if all(left <= right for left, right in zip(other_values, candidate_values)) and any(
                left < right for left, right in zip(other_values, candidate_values)
            ):
                dominated = True
                break
        if not dominated:
            result.append(candidate["strategy"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    metrics = _csv(args.root / "all_metrics_fixed_metaclip2.csv")
    per_query = _csv(args.root / "all_per_query_fixed_metaclip2.csv")
    matched_metrics = _csv(args.root / "all_metrics.csv")
    novelty = _csv(args.root / "provider_novelty_comparison.csv")
    aggregate = [row for row in metrics if row["scope"] == "aggregate"]
    videos = [row for row in metrics if row["scope"] == "video"]

    invariant_failures = [
        {"regime": row["regime"], "strategy": row["strategy"], "gap": row["max_gap_sec"]}
        for row in aggregate
        if row["strategy"] != "S0_midpoint" and _number(row, "max_gap_sec") > 8.0 + 1e-9
    ]

    pooled_ranks: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in per_query:
        if row["scope"] == "pooled":
            pooled_ranks[(row["regime"], row["strategy"])][row["query_id"]] = _rank(
                row["first_relevant_rank"]
            )
    wins_losses = []
    strategies = sorted({row["strategy"] for row in aggregate})
    for regime in ("pyscenedetect", "transnetv2"):
        baseline = pooled_ranks[(regime, "S1_periodic_12s_8s")]
        for strategy in strategies:
            if strategy in ("S0_midpoint", "S1_periodic_12s_8s"):
                continue
            comparison = pooled_ranks[(regime, strategy)]
            wins = losses = ties = 0
            for query_id, base_rank in baseline.items():
                rank = comparison[query_id]
                if rank < base_rank:
                    wins += 1
                elif rank > base_rank:
                    losses += 1
                else:
                    ties += 1
            wins_losses.append(
                {
                    "regime": regime,
                    "strategy": strategy,
                    "wins_vs_periodic": wins,
                    "losses_vs_periodic": losses,
                    "ties_vs_periodic": ties,
                }
            )

    novelty_aggregate = []
    grouped_novelty: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in novelty:
        grouped_novelty[(row["regime"], row["provider_pair"])].append(row)
    for (regime, pair), rows in sorted(grouped_novelty.items()):
        weight = sum(int(row["candidate_count"]) for row in rows)
        novelty_aggregate.append(
            {
                "regime": regime,
                "provider_pair": pair,
                "shots": len(rows),
                "candidate_weighted_spearman": sum(
                    float(row["spearman_rank_correlation"]) * int(row["candidate_count"])
                    for row in rows
                )
                / weight,
                "same_top1_shots": sum(int(row["same_top1"]) for row in rows),
                "same_top1_rate": sum(int(row["same_top1"]) for row in rows) / len(rows),
                "different_top1_rate": 1.0
                - sum(int(row["same_top1"]) for row in rows) / len(rows),
            }
        )

    pareto = {
        regime: _pareto([row for row in aggregate if row["regime"] == regime])
        for regime in ("pyscenedetect", "transnetv2")
    }

    lookup = {(row["regime"], row["strategy"]): row for row in aggregate}
    deltas = []
    comparisons = (
        ("S5_metaclip2_edge", "S2_metaclip2"),
        ("F_RRF_perception_encoder+metaclip2", "S3_perception_encoder"),
        ("F_RRF_perception_encoder+beit3", "S3_perception_encoder"),
        ("F_RRF_metaclip2+perception_encoder+beit3", "F_RRF_perception_encoder+beit3"),
        ("KNN_ADAPTIVE_perception_encoder", "S3_perception_encoder"),
        ("KNN_ADAPTIVE_beit3", "S4_beit3"),
    )
    for regime in ("pyscenedetect", "transnetv2"):
        for changed, baseline in comparisons:
            if (regime, changed) not in lookup:
                continue
            left, right = lookup[(regime, changed)], lookup[(regime, baseline)]
            deltas.append(
                {
                    "regime": regime,
                    "changed": changed,
                    "baseline": baseline,
                    "keyframe_delta": int(left["selected_keyframes"]) - int(right["selected_keyframes"]),
                    "macro_r10_delta": _number(left, "recall_at_10") - _number(right, "recall_at_10"),
                    "pooled_r10_delta": _number(left, "pooled_recall_at_10") - _number(right, "pooled_recall_at_10"),
                    "macro_mrr_delta": _number(left, "mrr") - _number(right, "mrr"),
                    "pooled_mrr_delta": _number(left, "pooled_mrr") - _number(right, "pooled_mrr"),
                    "embedding_seconds_delta": _number(left, "embedding_seconds_shared") - _number(right, "embedding_seconds_shared"),
                }
            )

    compact_aggregate = [
        {
            key: row[key]
            for key in (
                "regime", "strategy", "providers", "selector_providers",
                "retrieval_providers", "selected_keyframes", "gt_covered",
                "gt_total", "pooled_selection_recall", "high_gt_covered", "high_gt_total",
                "recall_at_1", "recall_at_10", "recall_at_100", "mrr",
                "pooled_recall_at_1", "pooled_recall_at_10", "pooled_recall_at_100",
                "pooled_mrr", "worst_selection_recall", "worst_recall_at_10",
                "max_gap_sec", "event_count", "repair_count", "embedding_seconds_shared",
                "retrieval_embedding_seconds_shared",
                "edge_signal_seconds_shared", "jpeg_bytes", "vector_bytes", "index_bytes",
            )
        }
        for row in aggregate
    ]
    compact_videos = [
        {
            key: row[key]
            for key in (
                "regime", "strategy", "video_id", "shot_count", "selected_keyframes",
                "gt_covered", "gt_total", "selection_recall", "high_gt_covered",
                "high_gt_total", "high_selection_recall", "recall_at_1", "recall_at_10",
                "recall_at_100", "mrr", "max_gap_sec",
            )
        }
        for row in videos
    ]
    payload: dict[str, Any] = {
        "primary_retrieval_protocol": "fixed MetaCLIP-2 for every selector",
        "secondary_retrieval_protocol": "selector-provider-matched diagnostic",
        "aggregate_metrics": compact_aggregate,
        "provider_matched_aggregate_metrics": [
            row for row in matched_metrics if row["scope"] == "aggregate"
        ],
        "per_video_metrics": compact_videos,
        "periodic_vs_content_pooled_query_wins_losses": wins_losses,
        "provider_novelty": novelty_aggregate,
        "quality_cost_pareto": pareto,
        "key_comparison_deltas": deltas,
        "temporal_gap_invariant_failures": invariant_failures,
    }
    (args.root / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("quality_cost_pareto", "provider_novelty", "key_comparison_deltas", "temporal_gap_invariant_failures")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
