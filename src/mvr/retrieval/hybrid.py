"""Pure fusion utilities shared by API and evaluation code."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping


def rrf_fuse(
    ranked_lists: Iterable[Iterable[str]],
    *,
    rrf_k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    scores: Counter[str] = Counter()
    for ranked_ids in ranked_lists:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] += 1.0 / (rrf_k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def min_max_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return {item_id: 1.0 for item_id in scores}
    scale = high - low
    return {item_id: (score - low) / scale for item_id, score in scores.items()}


def weighted_fuse(
    lexical_scores: Mapping[str, float],
    semantic_scores: Mapping[str, float],
    *,
    alpha: float = 0.25,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    lexical = min_max_normalize(lexical_scores)
    semantic = min_max_normalize(semantic_scores)
    candidates = set(lexical_scores) | set(semantic_scores)
    combined = {
        item_id: alpha * lexical.get(item_id, 0.0)
        + (1.0 - alpha) * semantic.get(item_id, 0.0)
        for item_id in candidates
    }

    def tie_key(item_id: str):
        active_present = (
            (alpha > 0.0 and item_id in lexical_scores)
            or (alpha < 1.0 and item_id in semantic_scores)
        )
        active_scores = []
        if alpha > 0.0:
            active_scores.append(lexical.get(item_id, 0.0))
        if alpha < 1.0:
            active_scores.append(semantic.get(item_id, 0.0))
        return (
            -combined[item_id],
            not active_present,
            -max(active_scores, default=0.0),
            item_id,
        )

    ordered = sorted(candidates, key=tie_key)[:top_k]
    return [(item_id, combined[item_id]) for item_id in ordered]
