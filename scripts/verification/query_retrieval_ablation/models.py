"""Data models for the TKIS query-variant retrieval ablation benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryVariant:
    """A frozen set of query variants for a single query."""

    query_id: str
    query_vi: str
    query_en_full: str
    keyword_query: str
    translation_model: str
    qwen_model: str
    qwen_schema_version: str
    translation_success: bool
    decomposition_success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_vi": self.query_vi,
            "query_en_full": self.query_en_full,
            "keyword_query": self.keyword_query,
            "translation_model": self.translation_model,
            "qwen_model": self.qwen_model,
            "qwen_schema_version": self.qwen_schema_version,
            "translation_success": self.translation_success,
            "decomposition_success": self.decomposition_success,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryVariant:
        return cls(**data)


@dataclass
class AblationVariantRecord:
    """Retrieval metrics for a specific query and variant."""

    query_id: str
    variant: str
    first_correct_rank: int | None
    r_at_1: float
    r_at_5: float
    r_at_20: float
    r_at_50: float
    r_at_100: float
    final_score: float
    retrieval_latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "variant": self.variant,
            "first_correct_rank": self.first_correct_rank,
            "R@1": round(self.r_at_1, 4),
            "R@5": round(self.r_at_5, 4),
            "R@20": round(self.r_at_20, 4),
            "R@50": round(self.r_at_50, 4),
            "R@100": round(self.r_at_100, 4),
            "Final Score": round(self.final_score, 4),
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "error": self.error,
        }


@dataclass
class PairwiseComparison:
    """Comparison between two variants across the benchmark."""

    variant_a: str
    variant_b: str
    attempted_pairs: int = 0
    valid_pairs: int = 0
    excluded_pairs: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0
    moves_earlier: int = 0
    moves_later: int = 0
    rescues: int = 0
    regressions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_a": self.variant_a,
            "variant_b": self.variant_b,
            "attempted_pairs": self.attempted_pairs,
            "valid_pairs": self.valid_pairs,
            "excluded_pairs": self.excluded_pairs,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "moves_earlier": self.moves_earlier,
            "moves_later": self.moves_later,
            "rescues": self.rescues,
            "regressions": self.regressions,
        }


@dataclass
class VariantSummary:
    """Aggregate metrics for a variant."""

    variant: str
    attempted_queries: int
    successful_queries: int
    failed_queries: int
    failure_rate: float
    error_counts: dict[str, int]
    r_at_1: float | None
    r_at_5: float | None
    r_at_20: float | None
    r_at_50: float | None
    r_at_100: float | None
    final_score: float | None
    mean_latency_ms: float | None
    median_latency_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "attempted_queries": self.attempted_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "failure_rate": round(self.failure_rate, 4),
            "error_counts": self.error_counts,
            "R@1": round(self.r_at_1, 4) if self.r_at_1 is not None else None,
            "R@5": round(self.r_at_5, 4) if self.r_at_5 is not None else None,
            "R@20": round(self.r_at_20, 4) if self.r_at_20 is not None else None,
            "R@50": round(self.r_at_50, 4) if self.r_at_50 is not None else None,
            "R@100": round(self.r_at_100, 4) if self.r_at_100 is not None else None,
            "Final Score": round(self.final_score, 4) if self.final_score is not None else None,
            "mean_latency_ms": round(self.mean_latency_ms, 2) if self.mean_latency_ms is not None else None,
            "median_latency_ms": round(self.median_latency_ms, 2) if self.median_latency_ms is not None else None,
        }
