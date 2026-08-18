"""RRF fusion logic for KIS candidates."""
from __future__ import annotations

from typing import Sequence

from mvr.retrieval.kis import KisCandidate


def reciprocal_rank_fusion(
    rankings: list[Sequence[KisCandidate]],
    rrf_k: int = 60,
    top_k: int = 100,
) -> list[KisCandidate]:
    """Fuse multiple KIS candidate rankings using RRF by candidate identity.
    
    Candidate identity is defined as `(video_id, frame_id)`.
    """
    scores: dict[tuple[str, int], float] = {}
    candidates_map: dict[tuple[str, int], KisCandidate] = {}

    for ranking in rankings:
        for rank, candidate in enumerate(ranking, start=1):
            key = (candidate.video_id, candidate.frame_id)
            if key not in candidates_map:
                candidates_map[key] = candidate
            
            score = 1.0 / (rrf_k + rank)
            scores[key] = scores.get(key, 0.0) + score

    # Sort by RRF score descending, then by identity for determinism
    fused_keys = sorted(
        scores.keys(),
        key=lambda k: (-scores[k], k[0], k[1])
    )

    fused_candidates: list[KisCandidate] = []
    for rank, key in enumerate(fused_keys[:top_k], start=1):
        original = candidates_map[key]
        fused_candidates.append(
            KisCandidate(
                rank=rank,
                video_id=original.video_id,
                frame_id=original.frame_id,
                score=scores[key],
                keyframe_id=original.keyframe_id,
                timestamp=original.timestamp,
                refinement_score=original.refinement_score,
            )
        )

    return fused_candidates
