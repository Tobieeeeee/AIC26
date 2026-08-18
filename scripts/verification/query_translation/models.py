"""Data models for the TKIS query translation benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranslationResult:
    """One translation attempt for one query through one method."""

    query_id: str
    query_vi: str
    method: str
    translated_text: str
    latency_ms: float
    success: bool
    error: str | None = None
    model_or_provider: str = ""
    source_language: str = "vi"
    target_language: str = "en"
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_vi": self.query_vi,
            "method": self.method,
            "translated_text": self.translated_text,
            "latency_ms": round(self.latency_ms, 2),
            "success": self.success,
            "error": self.error,
            "model_or_provider": self.model_or_provider,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "timestamp": self.timestamp,
        }


@dataclass
class MethodSummary:
    """Aggregate statistics for one translation method."""

    method: str
    total_queries: int
    success_count: int
    failure_count: int
    failure_rate: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "total_queries": self.total_queries,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "failure_rate": round(self.failure_rate, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "median_latency_ms": round(self.median_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }
