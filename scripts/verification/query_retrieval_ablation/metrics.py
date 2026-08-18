"""Metrics calculation for the ablation benchmark."""
from __future__ import annotations

from typing import Sequence

from mvr.evaluation.aic_kis import KisAnswer, KisGroundTruth, evaluate_kis, kis_r_score
from mvr.retrieval.kis import KisCandidate
from .models import AblationVariantRecord, PairwiseComparison


def evaluate_ablation_variant(
    query_id: str,
    variant_name: str,
    candidates: Sequence[KisCandidate],
    gt: KisGroundTruth,
    latency_ms: float,
    error: str | None = None,
) -> AblationVariantRecord:
    """Evaluate one variant's candidates against the ground truth."""
    answers = [c.to_answer() for c in candidates]
    score = evaluate_kis(answers, gt)

    first_correct_rank = None
    for rank, ans in enumerate(answers[:100], start=1):
        if kis_r_score(ans, gt) > 0:
            first_correct_rank = rank
            break

    return AblationVariantRecord(
        query_id=query_id,
        variant=variant_name,
        first_correct_rank=first_correct_rank,
        r_at_1=score.recall_at_k.get(1, 0.0),
        r_at_5=score.recall_at_k.get(5, 0.0),
        r_at_20=score.recall_at_k.get(20, 0.0),
        r_at_50=score.recall_at_k.get(50, 0.0),
        r_at_100=score.recall_at_k.get(100, 0.0),
        final_score=score.final_score,
        retrieval_latency_ms=latency_ms,
        error=error,
    )


def compute_pairwise_comparison(
    records_a: Sequence[AblationVariantRecord],
    records_b: Sequence[AblationVariantRecord],
    variant_a_name: str,
    variant_b_name: str,
) -> PairwiseComparison:
    """Compare variant A against variant B."""
    map_a = {r.query_id: r for r in records_a}
    map_b = {r.query_id: r for r in records_b}

    comp = PairwiseComparison(variant_a=variant_a_name, variant_b=variant_b_name)

    for query_id, r_a in map_a.items():
        r_b = map_b.get(query_id)
        if not r_b:
            continue

        # Wins/Ties/Losses based on Final Score
        if r_a.final_score > r_b.final_score:
            comp.wins += 1
        elif r_a.final_score < r_b.final_score:
            comp.losses += 1
        else:
            comp.ties += 1

        # Rescues/Regressions
        a_correct = r_a.first_correct_rank is not None
        b_correct = r_b.first_correct_rank is not None

        if a_correct and not b_correct:
            comp.rescues += 1
        elif not a_correct and b_correct:
            comp.regressions += 1

        # Rank movement
        if a_correct and b_correct:
            if r_a.first_correct_rank < r_b.first_correct_rank:
                comp.moves_earlier += 1
            elif r_a.first_correct_rank > r_b.first_correct_rank:
                comp.moves_later += 1

    return comp
