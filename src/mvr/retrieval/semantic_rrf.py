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
    identities: dict[str, tuple[str, int, int | None, float]] = {}
    provider_ranks: dict[str, dict[str, int]] = {}
    for model_id in participating:
        best_in_provider: dict[str, ProviderSemanticResult] = {}
        for item in ranked_lists[model_id]:
            if item.model_id != model_id:
                raise SemanticRrfError(
                    f"ranked list key {model_id} contains result from {item.model_id}"
                )
            if isinstance(item.rank, bool) or not isinstance(item.rank, int) or item.rank <= 0:
                raise SemanticRrfError(
                    f"provider {model_id} has invalid rank for {item.keyframe_id}: {item.rank}"
                )
            previous = best_in_provider.get(item.keyframe_id)
            if previous is not None:
                warnings.append(f"duplicate:{model_id}:{item.keyframe_id}")
                if item.rank >= previous.rank:
                    continue
            best_in_provider[item.keyframe_id] = item
        for keyframe_id, item in best_in_provider.items():
            identity = (item.video_id, item.frame_id, item.shot_id, item.timestamp)
            existing = identities.get(keyframe_id)
            if existing is not None and existing != identity:
                raise SemanticRrfError(
                    f"conflicting identity metadata for keyframe_id {keyframe_id}"
                )
            identities[keyframe_id] = identity
            provider_ranks.setdefault(keyframe_id, {})[model_id] = item.rank

    scores = {
        keyframe_id: sum(
            1.0 / (rrf_k + rank) for rank in ranks.values()
        )
        for keyframe_id, ranks in provider_ranks.items()
    }
    ordered_ids = sorted(
        scores,
        key=lambda keyframe_id: (
            -scores[keyframe_id],
            min(provider_ranks[keyframe_id].values()),
            -len(provider_ranks[keyframe_id]),
            keyframe_id,
        ),
    )[:top_k]
    results = []
    for output_rank, keyframe_id in enumerate(ordered_ids, start=1):
        video_id, frame_id, shot_id, timestamp = identities[keyframe_id]
        results.append(
            FusedSemanticResult(
                rank=output_rank,
                keyframe_id=keyframe_id,
                video_id=video_id,
                frame_id=frame_id,
                shot_id=shot_id,
                timestamp=timestamp,
                rrf_score=scores[keyframe_id],
                provider_ranks=dict(sorted(provider_ranks[keyframe_id].items())),
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
