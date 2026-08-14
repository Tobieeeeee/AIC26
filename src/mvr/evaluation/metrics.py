"""Retrieval metrics adapted from Lab 4."""
from __future__ import annotations

from collections.abc import Collection, Sequence


def compute_retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> dict[str, float]:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        return {"recall": 0.0, "hit": 0.0, "reciprocal_rank": 0.0}
    top_ids = list(retrieved_ids[:k])
    retrieved_relevant = relevant.intersection(top_ids)
    first_rank = next(
        (rank for rank, item_id in enumerate(top_ids, start=1) if item_id in relevant),
        None,
    )
    return {
        "recall": len(retrieved_relevant) / len(relevant),
        "hit": float(bool(retrieved_relevant)),
        "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
    }


def recall_at_k(
    rankings: Sequence[Sequence[str]],
    relevant_sets: Sequence[Collection[str]],
    *,
    k: int,
) -> float:
    if len(rankings) != len(relevant_sets) or not rankings:
        raise ValueError("rankings and relevant_sets must have equal non-zero length")
    if k <= 0:
        raise ValueError("k must be positive")
    recalls = []
    for ranking, relevant in zip(rankings, relevant_sets):
        relevant_lookup = set(relevant)
        recalls.append(
            len(set(ranking[:k]).intersection(relevant_lookup)) / len(relevant_lookup)
            if relevant_lookup
            else 0.0
        )
    return sum(recalls) / len(recalls)


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]],
    relevant_sets: Sequence[Collection[str]],
) -> float:
    if len(rankings) != len(relevant_sets) or not rankings:
        raise ValueError("rankings and relevant_sets must have equal non-zero length")
    reciprocal_ranks = []
    for ranking, relevant in zip(rankings, relevant_sets):
        relevant_lookup = set(relevant)
        first_rank = next(
            (rank for rank, item_id in enumerate(ranking, start=1) if item_id in relevant_lookup),
            None,
        )
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
