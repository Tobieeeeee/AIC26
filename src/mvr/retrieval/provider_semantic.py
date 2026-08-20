"""Independent semantic retrieval over provider-matched vector spaces."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from mvr.enrichment.embeddings.query_encoders import QueryEmbedding, QueryRoutingError
from mvr.indexing.provider_indexes import ProviderIndexContract, ProviderIndexError


class ProviderSearchError(RuntimeError):
    """Raised when an independent provider search cannot complete safely."""


@dataclass(frozen=True)
class BackendSearchHit:
    score: float
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ProviderSemanticResult:
    rank: int
    score: float
    model_id: str
    keyframe_id: str
    video_id: str
    frame_id: int
    shot_id: int | None
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IndependentSemanticResponse:
    ranked_lists: dict[str, tuple[ProviderSemanticResult, ...]]
    requested_providers: tuple[str, ...]
    successful_providers: tuple[str, ...]
    failed_providers: tuple[str, ...]
    errors: dict[str, str]
    failure_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked_lists": {
                model_id: [result.to_dict() for result in results]
                for model_id, results in self.ranked_lists.items()
            },
            "coverage": {
                "requested": list(self.requested_providers),
                "successful": list(self.successful_providers),
                "failed": list(self.failed_providers),
            },
            "errors": dict(self.errors),
            "failure_policy": self.failure_policy,
        }


class ProviderSearchBackend(Protocol):
    def search(
        self,
        contract: ProviderIndexContract,
        vector: np.ndarray,
        top_k: int,
    ) -> Sequence[BackendSearchHit]: ...


class FaissProviderSearchBackend:
    """Read Task05C provider collections and execute exact FAISS search."""

    def __init__(self, collections_root: Path) -> None:
        self.collections_root = Path(collections_root)

    @staticmethod
    def _faiss():
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError('FAISS is missing; install with ".[semantic]"') from exc
        return faiss

    def _target(self, contract: ProviderIndexContract) -> Path:
        if not contract.collection_name.endswith(f"__{contract.model_id}"):
            raise QueryRoutingError("provider collection suffix/model mismatch")
        target = (self.collections_root / contract.collection_name).resolve()
        try:
            target.relative_to(self.collections_root.resolve())
        except ValueError as exc:
            raise QueryRoutingError("provider collection escapes configured root") from exc
        return target

    def search(self, contract, vector, top_k):
        target = self._target(contract)
        faiss = self._faiss()
        index = faiss.read_index(str(target / "keyframes.faiss"))
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        if index.d != contract.dimension:
            raise QueryRoutingError("loaded provider index dimension mismatch")
        if len(metadata) != index.ntotal:
            raise ProviderSearchError("provider index metadata count mismatch")
        query = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        scores, indices = index.search(query, min(top_k, index.ntotal))
        return [
            BackendSearchHit(float(score), metadata[int(index_id)])
            for score, index_id in zip(scores[0], indices[0])
            if index_id >= 0
        ]


class IndependentSemanticRetriever:
    def __init__(
        self,
        *,
        query_encoders: Any,
        backend: ProviderSearchBackend,
        contracts: Mapping[str, ProviderIndexContract],
        failure_policy: str = "fail_closed",
        required_providers: Sequence[str] = (),
        max_top_k: int = 100,
        candidate_multiplier: int = 2,
    ) -> None:
        if failure_policy not in {"fail_closed", "degraded"}:
            raise ValueError("failure_policy must be fail_closed or degraded")
        if max_top_k <= 0 or candidate_multiplier <= 0:
            raise ValueError("top-k limits must be positive")
        self.query_encoders = query_encoders
        self.backend = backend
        self.contracts = dict(contracts)
        self.failure_policy = failure_policy
        self.required_providers = frozenset(required_providers)
        self.max_top_k = max_top_k
        self.candidate_multiplier = candidate_multiplier

    @staticmethod
    def _higher_is_better(metric: str) -> bool:
        if metric in {"inner_product", "cosine"}:
            return True
        if metric in {"l2", "euclidean"}:
            return False
        raise ProviderSearchError(f"unsupported provider index metric: {metric}")

    def search_provider(
        self, model_id: str, query: str, *, top_k: int
    ) -> tuple[ProviderSemanticResult, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        effective_top_k = min(top_k, self.max_top_k)
        try:
            contract = self.contracts[model_id]
        except KeyError as exc:
            raise QueryRoutingError(f"no index contract for provider {model_id}") from exc
        if contract.model_id != model_id or not contract.collection_name.endswith(
            f"__{model_id}"
        ):
            raise QueryRoutingError(f"cross-provider index route for {model_id}")
        embedding: QueryEmbedding = self.query_encoders.encode_query(
            model_id, query, index_contract=contract
        )
        if embedding.model_id != model_id or embedding.dimension != contract.dimension:
            raise QueryRoutingError("encoder result does not match target index contract")
        hits = self.backend.search(
            contract,
            embedding.vector,
            effective_top_k * self.candidate_multiplier,
        )
        higher_is_better = self._higher_is_better(contract.metric)
        best: dict[str, BackendSearchHit] = {}
        required_metadata = {
            "keyframe_id",
            "video_id",
            "frame_id",
            "shot_id",
            "timestamp",
        }
        for hit in hits:
            if not np.isfinite(hit.score) or not required_metadata.issubset(hit.metadata):
                continue
            keyframe_id = str(hit.metadata["keyframe_id"])
            current = best.get(keyframe_id)
            if current is None or (
                hit.score > current.score if higher_is_better else hit.score < current.score
            ):
                best[keyframe_id] = hit
        ordered = sorted(
            best.values(),
            key=lambda hit: (
                -hit.score if higher_is_better else hit.score,
                str(hit.metadata["keyframe_id"]),
            ),
        )[:effective_top_k]
        return tuple(
            ProviderSemanticResult(
                rank=rank,
                score=float(hit.score),
                model_id=model_id,
                keyframe_id=str(hit.metadata["keyframe_id"]),
                video_id=str(hit.metadata["video_id"]),
                frame_id=int(hit.metadata["frame_id"]),
                shot_id=(
                    None
                    if hit.metadata["shot_id"] is None
                    else int(hit.metadata["shot_id"])
                ),
                timestamp=float(hit.metadata["timestamp"]),
            )
            for rank, hit in enumerate(ordered, start=1)
        )

    def search(
        self,
        query: str,
        *,
        provider_ids: Sequence[str],
        top_k: int,
    ) -> IndependentSemanticResponse:
        requested = tuple(dict.fromkeys(provider_ids))
        if not requested:
            raise ValueError("at least one semantic provider is required")
        ranked_lists: dict[str, tuple[ProviderSemanticResult, ...]] = {}
        errors: dict[str, str] = {}
        for model_id in requested:
            try:
                ranked_lists[model_id] = self.search_provider(
                    model_id, query, top_k=top_k
                )
            except Exception as exc:
                errors[model_id] = f"{type(exc).__name__}: {exc}"
                if self.failure_policy == "fail_closed" and model_id in self.required_providers:
                    raise ProviderSearchError(
                        f"required semantic provider failed: {model_id}: {type(exc).__name__}"
                    ) from exc
        successful = tuple(model_id for model_id in requested if model_id in ranked_lists)
        failed = tuple(model_id for model_id in requested if model_id in errors)
        return IndependentSemanticResponse(
            ranked_lists=ranked_lists,
            requested_providers=requested,
            successful_providers=successful,
            failed_providers=failed,
            errors=errors,
            failure_policy=self.failure_policy,
        )
