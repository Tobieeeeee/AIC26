"""Real-data TRAKE benchmark: query records and dataset-level summary
statistics for comparing chain-alignment strategies.

Mirrors `mvr.evaluation.benchmark`'s KIS benchmark shape for consistency.
Pure orchestration/aggregation: calls the existing TRAKE evaluator
(`mvr.evaluation.trake.evaluate_trake`) rather than reimplementing scoring,
and does not itself run retrieval -- callers inject a `search_fn` (in
practice, `mvr.retrieval.trake.search_trake` bound to one alignment
strategy).
"""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .trake import TRAKE_CUTOFFS, TrakeGroundTruth, evaluate_trake


@dataclass(frozen=True)
class TrakeBenchmarkQuery:
    """One team-curated benchmark query. `label` must stay "internal" unless
    the query set is organizer-provided official GT."""

    query_id: str
    query: Any  # a mvr.retrieval.trake_events.TrakeQuery
    gt: TrakeGroundTruth
    label: str = "internal"


@dataclass(frozen=True)
class TrakeBenchmarkRecord:
    query_id: str
    config_name: str
    r_at_k: dict[int, float]
    final_score: float
    latency_seconds: float


def run_trake_benchmark_query(
    query: TrakeBenchmarkQuery,
    *,
    config_name: str,
    search_fn: Callable[[Any], Sequence[Any]],
) -> TrakeBenchmarkRecord:
    """Time one query through `search_fn` and score it with the official
    evaluator.

    `search_fn` must return a ranked (best first) sequence of
    `TrakeChainPrediction`-shaped objects (e.g. `search_trake()`'s return
    value, or a `functools.partial` of it bound to one alignment strategy).
    Unlike the KIS benchmark's `run_kis_benchmark_query`, no conversion step
    is needed here: `search_trake()` already returns the evaluator's own
    input type.
    """
    started = time.perf_counter()
    predictions = search_fn(query.query)
    latency = time.perf_counter() - started
    score = evaluate_trake(predictions, query.gt)
    return TrakeBenchmarkRecord(
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


def _summarize_one_config(records: Sequence[TrakeBenchmarkRecord]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"query_count": len(records)}
    for cutoff in TRAKE_CUTOFFS:
        values = [record.r_at_k[cutoff] for record in records]
        summary[f"mean_recall@{cutoff}"] = statistics.fmean(values)
    final_scores = [record.final_score for record in records]
    summary["mean_final_score"] = statistics.fmean(final_scores)
    summary["median_final_score"] = statistics.median(final_scores)
    latencies = sorted(record.latency_seconds for record in records)
    summary["mean_latency_seconds"] = statistics.fmean(latencies)
    summary["p95_latency_seconds"] = _percentile(latencies, 0.95)
    return summary


def summarize_trake_benchmark(
    records: Sequence[TrakeBenchmarkRecord],
) -> dict[str, dict[str, float | int]]:
    """Group records by `config_name` (one per alignment strategy) and
    compute mean/median/p95 dataset statistics for each. Labeled per-query
    "mean per-query ..." per the task's own naming rule -- callers must not
    present this as an organizer-official aggregate score."""
    if not records:
        raise ValueError("cannot summarize an empty benchmark run")
    by_config: dict[str, list[TrakeBenchmarkRecord]] = {}
    for record in records:
        by_config.setdefault(record.config_name, []).append(record)
    return {name: _summarize_one_config(group) for name, group in by_config.items()}
