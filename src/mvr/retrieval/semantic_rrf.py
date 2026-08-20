"""Deterministic rank-only fusion for semantic model ranked lists."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .provider_semantic import IndependentSemanticResponse, ProviderSemanticResult


class SemanticRrfError(ValueError):
    """Raised when semantic provider rankings cannot be fused safely."""


@dataclass(frozen=True)
class FusedSemanticResult:
    rank: int
    keyframe_id: str
    video_id: str
    frame_id: int
    shot_id: int | None
    timestamp: float
    rrf_score: float
    provider_ranks: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provider_ranks"] = dict(sorted(self.provider_ranks.items()))
        return payload


@dataclass(frozen=True)
class SemanticRrfResponse:
    results: tuple[FusedSemanticResult, ...]
    rrf_k: int
    participating_providers: tuple[str, ...]
    missing_providers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "rrf_k": self.rrf_k,
            "participating_providers": list(self.participating_providers),
            "missing_providers": list(self.missing_providers),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def fuse_semantic_rankings(
    ranked_lists: Mapping[str, Sequence[ProviderSemanticResult]],
    *,
    rrf_k: int = 60,
    top_k: int = 100,
    expected_providers: Sequence[str] = (),
    failure_policy: str = "fail_closed",
) -> SemanticRrfResponse:
    """Fuse by positive ranks only; provider-local raw scores are ignored."""

    if rrf_k <= 0:
        raise SemanticRrfError("rrf_k must be positive")
    if top_k <= 0:
        raise SemanticRrfError("top_k must be positive")
    if failure_policy not in {"fail_closed", "degraded"}:
        raise SemanticRrfError("failure_policy must be fail_closed or degraded")
    if any(not isinstance(model_id, str) or not model_id for model_id in ranked_lists):
        raise SemanticRrfError("model_id keys must be non-empty strings")
    if len(set(expected_providers)) != len(tuple(expected_providers)):
        raise SemanticRrfError("expected provider IDs must be unique")

    participating = tuple(sorted(ranked_lists))
    missing = tuple(sorted(set(expected_providers) - set(ranked_lists)))
    if missing and failure_policy == "fail_closed":
        raise SemanticRrfError(f"required semantic provider rankings missing: {missing}")

    warnings: list[str] = [
        f"missing_provider:{model_id}" for model_id in missing
    ]
    # Everything below is keyed on (video_id, keyframe_id).
    #
    # On keyframe_id alone this quietly threw away results. The organizer's
    # 166,550 keyframes carry 632 distinct ids -- "034" is in 842 videos --
    # so `best_in_provider` treated two different frames as one, kept the
    # better-ranked, and recorded the loss as a `duplicate:` warning rather
    # than an error. Measured on two providers each returning L21_V001/034
    # and L21_V002/034: two candidates in, one result out.
    #
    # It was correct while the team generated its own ids
    # (`L21_V001_S0007_K01` spells the video into the id) and became wrong
    # when the organizer's keyframes were adopted as canonical identity.
    identities: dict[tuple[str, str], tuple[int, int | None, float]] = {}
    provider_ranks: dict[tuple[str, str], dict[str, int]] = {}
    for model_id in participating:
        best_in_provider: dict[tuple[str, str], ProviderSemanticResult] = {}
        for item in ranked_lists[model_id]:
            if item.model_id != model_id:
                raise SemanticRrfError(
                    f"ranked list key {model_id} contains result from {item.model_id}"
                )
            if isinstance(item.rank, bool) or not isinstance(item.rank, int) or item.rank <= 0:
                raise SemanticRrfError(
                    f"provider {model_id} has invalid rank for "
                    f"{item.video_id}/{item.keyframe_id}: {item.rank}"
                )
            key = (item.video_id, item.keyframe_id)
            previous = best_in_provider.get(key)
            if previous is not None:
                warnings.append(f"duplicate:{model_id}:{item.video_id}/{item.keyframe_id}")
                if item.rank >= previous.rank:
                    continue
            best_in_provider[key] = item
        for key, item in best_in_provider.items():
            identity = (item.frame_id, item.shot_id, item.timestamp)
            existing = identities.get(key)
            if existing is not None and existing != identity:
                raise SemanticRrfError(
                    f"conflicting identity metadata for keyframe {key[0]}/{key[1]}"
                )
            identities[key] = identity
            provider_ranks.setdefault(key, {})[model_id] = item.rank

    scores = {
        key: sum(1.0 / (rrf_k + rank) for rank in ranks.values())
        for key, ranks in provider_ranks.items()
    }
    # The final tiebreak is the full identity, so ordering stays total and
    # deterministic: keyframe_id alone does not order the corpus.
    ordered_ids = sorted(
        scores,
        key=lambda key: (
            -scores[key],
            min(provider_ranks[key].values()),
            -len(provider_ranks[key]),
            key,
        ),
    )[:top_k]
    results = []
    for output_rank, key in enumerate(ordered_ids, start=1):
        video_id, keyframe_id = key
        frame_id, shot_id, timestamp = identities[key]
        results.append(
            FusedSemanticResult(
                rank=output_rank,
                keyframe_id=keyframe_id,
                video_id=video_id,
                frame_id=frame_id,
                shot_id=shot_id,
                timestamp=timestamp,
                rrf_score=scores[key],
                provider_ranks=dict(sorted(provider_ranks[key].items())),
            )
        )
    return SemanticRrfResponse(
        results=tuple(results),
        rrf_k=rrf_k,
        participating_providers=participating,
        missing_providers=missing,
        warnings=tuple(sorted(set(warnings))),
    )


def fuse_independent_response(
    response: IndependentSemanticResponse,
    *,
    rrf_k: int = 60,
    top_k: int = 100,
) -> SemanticRrfResponse:
    """Orchestration hook after Task05C.3 independent retrieval."""

    return fuse_semantic_rankings(
        response.ranked_lists,
        rrf_k=rrf_k,
        top_k=top_k,
        expected_providers=response.requested_providers,
        failure_policy=response.failure_policy,
    )
