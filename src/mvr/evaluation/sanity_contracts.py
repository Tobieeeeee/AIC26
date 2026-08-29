"""Provider-independent contracts for GT-free retrieval sanity evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


FrameId = int | str


@dataclass(frozen=True)
class SanityQuery:
    query_id: str
    text: str
    language: str | None = None
    tags: tuple[str, ...] = ()
    expect_error: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SanityQuery":
        query_id = payload.get("query_id")
        text = payload.get("text")
        language = payload.get("language")
        tags = payload.get("tags", [])
        expect_error = payload.get("expect_error", False)
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if not isinstance(text, str):
            raise ValueError(f"{query_id}: text must be a string")
        if language is not None and not isinstance(language, str):
            raise ValueError(f"{query_id}: language must be a string or null")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"{query_id}: tags must be a list of strings")
        if not isinstance(expect_error, bool):
            raise ValueError(f"{query_id}: expect_error must be boolean")
        return cls(
            query_id=query_id.strip(),
            text=text,
            language=language,
            tags=tuple(tags),
            expect_error=expect_error,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    score: float
    keyframe_id: str
    video_id: str
    frame_id: FrameId
    shot_id: int | str | None = None
    timestamp: float | None = None
    provider: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationResult:
    contract_valid: bool
    top_k_valid: bool
    ranks_valid: bool
    scores_finite: bool
    duplicate_keyframe_ids: tuple[str, ...] = ()
    identity_conflicts: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return (
            self.contract_valid
            and self.top_k_valid
            and self.ranks_valid
            and self.scores_finite
            and not self.duplicate_keyframe_ids
            and not self.identity_conflicts
            and not self.errors
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["valid"] = self.valid
        return payload


@dataclass(frozen=True)
class QueryRunResult:
    provider_id: str
    query_id: str
    repetition: int
    top_k_requested: int
    returned_count: int
    unique_keyframe_count: int
    latency_ms: float
    expectation_met: bool
    hits: tuple[RetrievalHit, ...] = ()
    validation: ValidationResult | None = None
    error: str | None = None

    @property
    def ordered_keyframe_ids(self) -> tuple[str, ...]:
        return tuple(hit.keyframe_id for hit in self.hits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "query_id": self.query_id,
            "repetition": self.repetition,
            "top_k_requested": self.top_k_requested,
            "returned_count": self.returned_count,
            "unique_keyframe_count": self.unique_keyframe_count,
            "latency_ms": self.latency_ms,
            "expectation_met": self.expectation_met,
            "ordered_keyframe_ids": list(self.ordered_keyframe_ids),
            "hits": [hit.to_dict() for hit in self.hits],
            "validation": None if self.validation is None else self.validation.to_dict(),
            "error": self.error,
        }


@dataclass(frozen=True)
class DeterminismResult:
    provider_id: str
    query_id: str
    comparable: bool
    deterministic_order: bool | None
    score_drift_detected: bool | None
    ranking_changes: tuple[dict[str, Any], ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SanityRun:
    manifest: dict[str, Any]
    queries: tuple[SanityQuery, ...]
    results: list[QueryRunResult]
    determinism: list[DeterminismResult]
    summary: dict[str, Any] = field(default_factory=dict)
