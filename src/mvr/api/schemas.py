"""Pydantic request and response contracts for the MVR search API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _require_non_blank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)

    _validate_query = field_validator("query")(_require_non_blank)


class TextSearchRequest(BaseModel):
    query: str = ""
    top_k: int = Field(default=5, ge=1, le=100)
    person_min: int = Field(default=0, ge=0)
    car_min: int = Field(default=0, ge=0)


class ObjectSearchRequest(BaseModel):
    want: str
    top_k: int = Field(default=5, ge=1, le=100)


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=1000)

    _validate_query = field_validator("query")(_require_non_blank)


class KisSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)

    _validate_query = field_validator("query")(_require_non_blank)


class KisHit(BaseModel):
    rank: int = Field(ge=1)
    video_id: str
    frame_id: int
    score: float
    keyframe_id: str | None = None
    timestamp: float | None = None


class KisSearchResponse(BaseModel):
    query: str
    count: int = Field(ge=0)
    results: list[KisHit]


class SearchHit(BaseModel):
    rank: int = Field(ge=1)
    frame_id: str
    video_id: str
    timestamp: float
    score: float
    image_path: str
    ocr_text: str | None = None
    caption: str | None = None
    objects: dict[str, int] | None = None
    highlight: dict[str, Any] | None = None
    rank_lexical: int | None = Field(default=None, ge=1)
    rank_semantic: int | None = Field(default=None, ge=1)


class SearchResponse(BaseModel):
    query: str
    count: int = Field(ge=0)
    results: list[SearchHit]


class HybridSearchResponse(SearchResponse):
    rrf_k: int = Field(ge=1)


class SemanticRuntimeStatus(BaseModel):
    configured: bool
    available: bool
    loaded: bool


class ElasticsearchRuntimeStatus(BaseModel):
    configured: bool
    available: bool
    documents: int | None = Field(default=None, ge=0)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api: Literal["ready"] = "ready"
    semantic: SemanticRuntimeStatus
    elasticsearch: ElasticsearchRuntimeStatus
