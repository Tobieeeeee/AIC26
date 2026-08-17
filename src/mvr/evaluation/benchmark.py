"""Real-data KIS benchmark: query records, dataset-level summary statistics,
and an AIC-organizer-data adapter that lets the existing FAISS index builder
be reused unchanged.

This module is pure orchestration/aggregation: it calls the existing KIS
evaluator (Task 03) rather than reimplementing scoring, and does not itself
run retrieval -- callers inject a `search_fn`.
"""
from __future__ import annotations

import csv
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aic_kis import KIS_CUTOFFS, KisAnswer, KisGroundTruth, evaluate_kis


@dataclass(frozen=True)
class KisBenchmarkQuery:
    """One team-curated benchmark query. `label` must stay "internal" unless
    the query set is organizer-provided official GT."""

    query_id: str
    query_text: str
    video_id: str
    start_frame: int
    end_frame: int
    label: str = "internal"


@dataclass(frozen=True)
class KisBenchmarkRecord:
    query_id: str
    config_name: str
    r_at_k: dict[int, float]
    final_score: float
    latency_seconds: float


def run_kis_benchmark_query(
    query: KisBenchmarkQuery,
    *,
    config_name: str,
    search_fn: Callable[[str], Sequence[Any]],
) -> KisBenchmarkRecord:
    """Time one query through `search_fn` and score it with the official evaluator.

    `search_fn` must return an ordered (rank 1 first) sequence of objects
    exposing `.video_id` and `.frame_id` (e.g. `KisCandidate`).
    """
    gt = KisGroundTruth(
        video_id=query.video_id, start_frame=query.start_frame, end_frame=query.end_frame
    )
    started = time.perf_counter()
    candidates = search_fn(query.query_text)
    latency = time.perf_counter() - started
    answers = [KisAnswer(video_id=c.video_id, frame_id=c.frame_id) for c in candidates]
    score = evaluate_kis(answers, gt)
    return KisBenchmarkRecord(
        query_id=query.query_id,
        config_name=config_name,
        r_at_k=dict(score.recall_at_k),
        final_score=score.final_score,
        latency_seconds=latency,
    )


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must be non-empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _summarize_one_config(records: Sequence[KisBenchmarkRecord]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"query_count": len(records)}
    for cutoff in KIS_CUTOFFS:
        values = [record.r_at_k[cutoff] for record in records]
        summary[f"mean_recall@{cutoff}"] = statistics.fmean(values)
    final_scores = [record.final_score for record in records]
    summary["mean_final_score"] = statistics.fmean(final_scores)
    summary["median_final_score"] = statistics.median(final_scores)
    latencies = sorted(record.latency_seconds for record in records)
    summary["mean_latency_seconds"] = statistics.fmean(latencies)
    summary["p95_latency_seconds"] = _percentile(latencies, 0.95)
    return summary


def summarize_kis_benchmark(
    records: Sequence[KisBenchmarkRecord],
) -> dict[str, dict[str, float | int]]:
    """Group records by `config_name` and compute mean/median/p95 dataset
    statistics for each configuration. Labeled per-query "mean per-query ..."
    per the task's own naming rule -- callers must not present this as the
    organizer's official aggregate score."""
    if not records:
        raise ValueError("cannot summarize an empty benchmark run")
    by_config: dict[str, list[KisBenchmarkRecord]] = {}
    for record in records:
        by_config.setdefault(record.config_name, []).append(record)
    return {name: _summarize_one_config(group) for name, group in by_config.items()}


def build_aic_keyframe_mapping_csv(
    dataset_root: Path | str,
    output_path: Path | str,
    video_ids: Sequence[str],
) -> Path:
    """Adapt AIC organizer `map_keyframes/<video_id>.csv` files into the
    `keyframe_mapping.csv` shape `indexing.semantic_builder.build_semantic_index`
    already expects, so the existing (verified) index-building code can be
    reused unchanged rather than duplicated for organizer data.

    Uses the verified organizer `frame_idx`-mapping identity (n -> frame_idx)
    indirectly: `keyframe_id` here is the organizer keyframe reference (e.g.
    "003"), which `AicDatasetRegistry.resolve_frame()` can resolve to the
    exact original-video `frame_idx` later -- this function does not itself
    resolve or emit `frame_idx`. Rows whose keyframe JPEG does not exist
    locally are skipped rather than guessed.
    """
    dataset_root = Path(dataset_root)
    rows: list[dict[str, str]] = []
    for video_id in video_ids:
        csv_path = dataset_root / "map_keyframes" / f"{video_id}.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                keyframe_id = f"{int(row['n'].strip()):03d}"
                image_path = dataset_root / "keyframes" / video_id / f"{keyframe_id}.jpg"
                if not image_path.is_file():
                    continue
                rows.append(
                    {
                        "keyframe_id": keyframe_id,
                        "video_id": video_id,
                        "timestamp": row["pts_time"],
                        "image_path": str(image_path),
                    }
                )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["keyframe_id", "video_id", "timestamp", "image_path"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path
