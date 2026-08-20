"""Integrity validators for ranked retrieval results."""
from __future__ import annotations

import math
from collections.abc import Sequence

from .sanity_contracts import RetrievalHit, ValidationResult


def _valid_frame_id(value: int | str) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    return isinstance(value, str) and bool(value.strip())


def validate_hits(
    hits: Sequence[RetrievalHit],
    *,
    top_k: int,
) -> ValidationResult:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    errors: list[str] = []
    identity_conflicts: set[str] = set()
    seen_identity: dict[str, tuple[str, int | str]] = {}
    keyframe_ids: list[str] = []
    ranks: list[int] = []
    contract_valid = True
    scores_finite = True

    for index, hit in enumerate(hits):
        if not isinstance(hit, RetrievalHit):
            contract_valid = False
            errors.append(f"hit[{index}] is not a RetrievalHit")
            continue
        keyframe_id_valid = (
            isinstance(hit.keyframe_id, str) and bool(hit.keyframe_id.strip())
        )
        video_id_valid = isinstance(hit.video_id, str) and bool(hit.video_id.strip())
        if not keyframe_id_valid:
            contract_valid = False
            errors.append(f"hit[{index}] has invalid keyframe_id")
        if not video_id_valid:
            contract_valid = False
            errors.append(f"hit[{index}] has invalid video_id")
        frame_id_valid = _valid_frame_id(hit.frame_id)
        if not frame_id_valid:
            contract_valid = False
            errors.append(f"hit[{index}] has invalid frame_id")
        if isinstance(hit.rank, bool) or not isinstance(hit.rank, int):
            contract_valid = False
            errors.append(f"hit[{index}] rank is not an integer")
        else:
            ranks.append(hit.rank)
        if isinstance(hit.score, bool) or not isinstance(hit.score, (int, float)):
            scores_finite = False
            errors.append(f"hit[{index}] score is not numeric")
        elif not math.isfinite(float(hit.score)):
            scores_finite = False
            errors.append(f"hit[{index}] score is not finite")

        if keyframe_id_valid:
            keyframe_ids.append(hit.keyframe_id)
            if video_id_valid and frame_id_valid:
                identity = (hit.video_id, hit.frame_id)
                previous = seen_identity.setdefault(hit.keyframe_id, identity)
                if previous != identity:
                    identity_conflicts.add(hit.keyframe_id)

    duplicates = tuple(
        sorted(
            keyframe_id
            for keyframe_id in set(keyframe_ids)
            if keyframe_ids.count(keyframe_id) > 1
        )
    )
    top_k_valid = len(hits) <= top_k
    if not top_k_valid:
        errors.append(f"returned {len(hits)} hits for top_k={top_k}")

    ranks_valid = True
    if hits:
        ranks_valid = (
            len(ranks) == len(hits)
            and ranks[0] == 1
            and len(set(ranks)) == len(ranks)
            and all(current < following for current, following in zip(ranks, ranks[1:]))
        )
    if not ranks_valid:
        errors.append("ranks must start at 1 and be unique in increasing order")
    if duplicates:
        errors.append(f"duplicate keyframe IDs: {list(duplicates)}")
    if identity_conflicts:
        errors.append(
            f"canonical identity conflicts: {sorted(identity_conflicts)}"
        )

    return ValidationResult(
        contract_valid=contract_valid,
        top_k_valid=top_k_valid,
        ranks_valid=ranks_valid,
        scores_finite=scores_finite,
        duplicate_keyframe_ids=duplicates,
        identity_conflicts=tuple(sorted(identity_conflicts)),
        errors=tuple(errors),
    )
