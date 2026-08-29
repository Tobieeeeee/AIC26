"""Execution and aggregation for the GT-free retrieval sanity harness."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sanity_adapters import RetrieverAdapter
from .sanity_contracts import QueryRunResult, SanityQuery, SanityRun
from .sanity_determinism import compare_repetitions
from .sanity_validators import validate_hits


def load_sanity_queries(path: Path) -> tuple[SanityQuery, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("query suite must be a JSON list")
    queries_list: list[SanityQuery] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"query suite row {index} must be a JSON object")
        queries_list.append(SanityQuery.from_dict(row))
    queries = tuple(queries_list)
    if not queries:
        raise ValueError("query suite must not be empty")
    query_ids = [query.query_id for query in queries]
    duplicates = sorted(
        query_id for query_id in set(query_ids) if query_ids.count(query_id) > 1
    )
    if duplicates:
        raise ValueError(f"duplicate query IDs: {duplicates}")
    return queries


def query_suite_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_value(repository_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_run_manifest(
    *,
    repository_root: Path,
    query_suite_path: Path,
    adapter: RetrieverAdapter,
    top_k: int,
    repetitions: int,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    git_status = _git_value(repository_root, "status", "--porcelain")
    return {
        "run_id": (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + f"-{adapter.provider_id}"
        ),
        "timestamp": timestamp,
        "git_commit": _git_value(repository_root, "rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "git_status_porcelain": git_status.splitlines(),
        "query_suite_path": str(Path(query_suite_path).resolve()),
        "query_suite_sha256": query_suite_sha256(query_suite_path),
        "providers": [adapter.provenance()],
        "top_k": top_k,
        "repetitions": repetitions,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "evaluation_scope": "GT-free retrieval sanity and regression only",
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("values must be non-empty")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_run(run: SanityRun) -> dict[str, Any]:
    latencies = [result.latency_ms for result in run.results]
    latency_by_repetition = {
        str(repetition): [
            result.latency_ms
            for result in run.results
            if result.repetition == repetition
        ]
        for repetition in range(1, run.manifest["repetitions"] + 1)
    }
    validations = [
        result.validation for result in run.results if result.validation is not None
    ]
    successful_attempts = sum(
        result.expectation_met
        and (
            result.error is not None
            or (result.validation is not None and result.validation.valid)
        )
        for result in run.results
    )
    comparable = [item for item in run.determinism if item.comparable]
    return {
        "total_queries": len(run.queries),
        "total_attempts": len(run.results),
        "successful_attempts": successful_attempts,
        "failed_attempts": len(run.results) - successful_attempts,
        "expected_error_queries": sum(query.expect_error for query in run.queries),
        "provider_errors": sum(result.error is not None for result in run.results),
        "contract_failures": sum(
            not validation.contract_valid for validation in validations
        ),
        "top_k_failures": sum(not validation.top_k_valid for validation in validations),
        "invalid_rank_results": sum(
            not validation.ranks_valid for validation in validations
        ),
        "non_finite_score_results": sum(
            not validation.scores_finite for validation in validations
        ),
        "duplicate_keyframe_occurrences": sum(
            len(validation.duplicate_keyframe_ids) for validation in validations
        ),
        "identity_conflict_occurrences": sum(
            len(validation.identity_conflicts) for validation in validations
        ),
        "latency_ms": {
            "count": len(latencies),
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "by_repetition": {
                repetition: {
                    "count": len(values),
                    "mean": statistics.fmean(values) if values else None,
                    "p50": statistics.median(values) if values else None,
                    "p95": _percentile(values, 0.95) if values else None,
                }
                for repetition, values in latency_by_repetition.items()
            },
        },
        "determinism": {
            "comparable_query_count": len(comparable),
            "deterministic_order_count": sum(
                item.deterministic_order is True for item in comparable
            ),
            "changed_order_count": sum(
                item.deterministic_order is False for item in comparable
            ),
            "score_drift_count": sum(
                item.score_drift_detected is True for item in comparable
            ),
        },
        "scope_notice": (
            "This summary checks correctness, stability, latency, and ranking "
            "change. It does not measure relevance quality."
        ),
    }


def run_sanity_suite(
    *,
    adapter: RetrieverAdapter,
    queries: Sequence[SanityQuery],
    top_k: int,
    repetitions: int = 3,
    score_tolerance: float = 1e-8,
    manifest: dict[str, Any],
) -> SanityRun:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")

    results: list[QueryRunResult] = []
    for query in queries:
        for repetition in range(1, repetitions + 1):
            started = time.perf_counter()
            try:
                hits = tuple(adapter.search(query.text, top_k))
                latency_ms = (time.perf_counter() - started) * 1000.0
                validation = validate_hits(hits, top_k=top_k)
                results.append(
                    QueryRunResult(
                        provider_id=adapter.provider_id,
                        query_id=query.query_id,
                        repetition=repetition,
                        top_k_requested=top_k,
                        returned_count=len(hits),
                        unique_keyframe_count=len(
                            {hit.keyframe_id for hit in hits}
                        ),
                        latency_ms=latency_ms,
                        expectation_met=not query.expect_error,
                        hits=hits,
                        validation=validation,
                        error=None,
                    )
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000.0
                results.append(
                    QueryRunResult(
                        provider_id=adapter.provider_id,
                        query_id=query.query_id,
                        repetition=repetition,
                        top_k_requested=top_k,
                        returned_count=0,
                        unique_keyframe_count=0,
                        latency_ms=latency_ms,
                        expectation_met=query.expect_error,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    determinism = []
    for query in queries:
        query_results = [
            result for result in results if result.query_id == query.query_id
        ]
        if query.expect_error:
            determinism.append(
                compare_repetitions(query_results, score_tolerance=score_tolerance)
            )
        else:
            determinism.append(
                compare_repetitions(query_results, score_tolerance=score_tolerance)
            )

    run = SanityRun(
        manifest=manifest,
        queries=tuple(queries),
        results=results,
        determinism=determinism,
    )
    run.summary = summarize_run(run)
    return run
