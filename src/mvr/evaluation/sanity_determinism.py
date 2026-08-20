"""Same-configuration ranking determinism checks."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .sanity_contracts import DeterminismResult, QueryRunResult


def compare_repetitions(
    results: Sequence[QueryRunResult],
    *,
    score_tolerance: float = 1e-8,
) -> DeterminismResult:
    if not results:
        raise ValueError("results must be non-empty")
    if score_tolerance < 0 or not math.isfinite(score_tolerance):
        raise ValueError("score_tolerance must be finite and non-negative")

    provider_id = results[0].provider_id
    query_id = results[0].query_id
    if any(
        result.provider_id != provider_id or result.query_id != query_id
        for result in results
    ):
        raise ValueError("all repetitions must share provider_id and query_id")
    if len(results) < 2:
        return DeterminismResult(
            provider_id=provider_id,
            query_id=query_id,
            comparable=False,
            deterministic_order=None,
            score_drift_detected=None,
            error="at least two successful repetitions are required",
        )
    if any(result.error is not None for result in results):
        return DeterminismResult(
            provider_id=provider_id,
            query_id=query_id,
            comparable=False,
            deterministic_order=None,
            score_drift_detected=None,
            error="one or more repetitions returned an error",
        )

    baseline = results[0]
    baseline_order = [
        (hit.rank, hit.keyframe_id)
        for hit in baseline.hits
    ]
    changes: list[dict[str, Any]] = []
    score_drift = False

    for result in results[1:]:
        current_order = [(hit.rank, hit.keyframe_id) for hit in result.hits]
        if current_order != baseline_order:
            changes.append(
                {
                    "repetition": result.repetition,
                    "baseline": [
                        {"rank": rank, "keyframe_id": keyframe_id}
                        for rank, keyframe_id in baseline_order
                    ],
                    "current": [
                        {"rank": rank, "keyframe_id": keyframe_id}
                        for rank, keyframe_id in current_order
                    ],
                }
            )
        if len(result.hits) != len(baseline.hits):
            score_drift = True
            continue
        for previous, current in zip(baseline.hits, result.hits):
            if previous.keyframe_id != current.keyframe_id:
                continue
            if not math.isclose(
                previous.score,
                current.score,
                rel_tol=score_tolerance,
                abs_tol=score_tolerance,
            ):
                score_drift = True
                break

    return DeterminismResult(
        provider_id=provider_id,
        query_id=query_id,
        comparable=True,
        deterministic_order=not changes,
        score_drift_detected=score_drift,
        ranking_changes=tuple(changes),
    )
