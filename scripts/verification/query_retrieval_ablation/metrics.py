"""Scoring, aggregation, and pairwise comparison for the TKIS ablation.

Scoring is not reimplemented here. `mvr.evaluation.aic_kis.evaluate_kis`
already encodes the official R-Score / R@k / Final Score rule -- interval
containment at each cutoff, not set recall -- and this module only shapes
its output into the records `runner.py` and the CLI report read.

The one rule this module owns is the separation the empty file was
committed three times to make (`0040a9b` "separate TKIS ablation quality
metrics from failures"): **a variant that could not run is not a variant
that scored zero.** Averaging a translation outage as 0.0 makes a broken
dependency look like a retrieval result, so failures move `failure_rate`
and the quality mean stays over the queries that actually ran.
"""
from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

from mvr.evaluation.aic_kis import (
    KisAnswer,
    KisGroundTruth,
    evaluate_kis,
    kis_r_score,
)

from .models import AblationVariantRecord, PairwiseComparison, VariantSummary

#: Answers past this rank cannot score, per the official rule.
MAX_RANKED_ANSWERS = 100


def _answers(candidates: Sequence[Any]) -> list[KisAnswer]:
    return [
        KisAnswer(video_id=str(c.video_id), frame_id=int(c.frame_id))
        for c in candidates[:MAX_RANKED_ANSWERS]
    ]


def _first_correct_rank(answers: Sequence[KisAnswer], gt: KisGroundTruth) -> int | None:
    for rank, answer in enumerate(answers, start=1):
        if kis_r_score(answer, gt) == 1.0:
            return rank
    return None


def evaluate_ablation_variant(
    query_id: str,
    variant: str,
    candidates: Sequence[Any],
    gt: KisGroundTruth,
    latency_ms: float,
    error: str | None = None,
) -> AblationVariantRecord:
    """Score one variant's answers for one query.

    ``candidates`` must already be rank-ordered and expose ``video_id`` and
    ``frame_id`` -- `mvr.retrieval.kis.KisCandidate` does.

    ``error`` is taken at face value. Whether an empty result counts as a
    failure is the caller's judgement (``runner.py`` calls it
    ``empty_mapped_candidates``); second-guessing it here would give two
    places an opinion on the same question.
    """
    if error is not None:
        return AblationVariantRecord(
            query_id=query_id,
            variant=variant,
            first_correct_rank=None,
            r_at_1=0.0,
            r_at_5=0.0,
            r_at_20=0.0,
            r_at_50=0.0,
            r_at_100=0.0,
            final_score=0.0,
            retrieval_latency_ms=float(latency_ms),
            error=error,
        )

    answers = _answers(candidates)
    score = evaluate_kis(answers, gt)
    return AblationVariantRecord(
        query_id=query_id,
        variant=variant,
        first_correct_rank=_first_correct_rank(answers, gt),
        r_at_1=score.recall_at_k[1],
        r_at_5=score.recall_at_k[5],
        r_at_20=score.recall_at_k[20],
        r_at_50=score.recall_at_k[50],
        r_at_100=score.recall_at_k[100],
        final_score=score.final_score,
        retrieval_latency_ms=float(latency_ms),
        error=None,
    )


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_variant(
    variant: str, records: Sequence[AblationVariantRecord]
) -> VariantSummary:
    """Aggregate one variant, keeping quality and failures apart.

    Every quality figure is over the queries that ran. A variant where
    nothing ran reports ``None`` rather than ``0.0``: zero is a measurement,
    ``None`` is the absence of one, and a report printing ``0.0000`` for a
    variant that never executed invites exactly the wrong conclusion.
    """
    attempted = len(records)
    ran = [record for record in records if record.error is None]
    failed = [record for record in records if record.error is not None]
    latencies = [record.retrieval_latency_ms for record in ran]

    return VariantSummary(
        variant=variant,
        attempted_queries=attempted,
        successful_queries=len(ran),
        failed_queries=len(failed),
        failure_rate=len(failed) / attempted if attempted else 0.0,
        error_counts=dict(sorted(Counter(r.error for r in failed).items())),
        r_at_1=_mean([r.r_at_1 for r in ran]),
        r_at_5=_mean([r.r_at_5 for r in ran]),
        r_at_20=_mean([r.r_at_20 for r in ran]),
        r_at_50=_mean([r.r_at_50 for r in ran]),
        r_at_100=_mean([r.r_at_100 for r in ran]),
        final_score=_mean([r.final_score for r in ran]),
        mean_latency_ms=_mean(latencies),
        median_latency_ms=statistics.median(latencies) if latencies else None,
    )


def _by_query_id(
    records: Sequence[AblationVariantRecord], label: str
) -> dict[str, AblationVariantRecord]:
    indexed: dict[str, AblationVariantRecord] = {}
    for record in records:
        if record.query_id in indexed:
            raise ValueError(
                f"duplicate query_id {record.query_id!r} in variant {label!r}"
            )
        indexed[record.query_id] = record
    return indexed


def compare_variants(
    variant_a: str,
    variant_b: str,
    records_a: Sequence[AblationVariantRecord],
    records_b: Sequence[AblationVariantRecord],
) -> PairwiseComparison:
    """Compare two variants query by query.

    **Direction: ``variant_a`` is the BASELINE, ``variant_b`` the CANDIDATE,
    and every counter reads from B's side.** ``wins`` means B beat A;
    ``rescues`` means B found something A missed entirely.

    Said plainly because reversing it does not fail anything -- it silently
    inverts the conclusion of the whole benchmark.

    A pair where either side failed is excluded rather than scored: crediting
    the candidate for the baseline's translation outage would measure the
    outage, not the change.
    """
    left = _by_query_id(records_a, variant_a)
    right = _by_query_id(records_b, variant_b)
    shared = sorted(set(left) & set(right))

    comparison = PairwiseComparison(variant_a=variant_a, variant_b=variant_b)
    comparison.attempted_pairs = len(shared)

    for query_id in shared:
        a, b = left[query_id], right[query_id]
        if a.error is not None or b.error is not None:
            comparison.excluded_pairs += 1
            continue
        comparison.valid_pairs += 1

        if b.final_score > a.final_score:
            comparison.wins += 1
        elif b.final_score < a.final_score:
            comparison.losses += 1
        else:
            comparison.ties += 1

        a_rank, b_rank = a.first_correct_rank, b.first_correct_rank
        if a_rank is None and b_rank is not None:
            comparison.rescues += 1
        elif a_rank is not None and b_rank is None:
            comparison.regressions += 1
        elif a_rank is not None and b_rank is not None:
            if b_rank < a_rank:
                comparison.moves_earlier += 1
            elif b_rank > a_rank:
                comparison.moves_later += 1

    return comparison


__all__ = [
    "MAX_RANKED_ANSWERS",
    "compare_variants",
    "evaluate_ablation_variant",
    "summarize_variant",
]
