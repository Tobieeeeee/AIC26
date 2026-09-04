"""Pydantic request and response contracts for the MVR search API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class TrakeEventRequest(BaseModel):
    text: str = Field(min_length=1)

    _validate_text = field_validator("text")(_require_non_blank)


class TrakeSearchRequest(BaseModel):
    """Accepts an already-decomposed k-event TRAKE query. Gap timing is
    expressed ONLY as connector phrases (e.g. "rồi", "ngay sau đó"),
    translated via ``translate_gap_phrase()`` -- raw [min,max] second
    numbers are deliberately not accepted here, since the real-data
    benchmark (docs/AIC_TRAKE_BENCHMARK_RESULTS.md) measured the phrase
    table's own defaults outperforming hand-picked numbers (0.639 vs
    0.633 mean final score)."""

    events: list[TrakeEventRequest] = Field(min_length=2)
    gap_phrases: list[str] = Field(
        default_factory=list,
        description='One entry per gap between consecutive events (len(events)-1 total). '
        'Empty string "" falls back to the phrase table\'s no-clue default.',
    )
    top_k: int = Field(default=100, ge=1, le=100)
    alignment_strategy: Literal["atlas", "lucifer", "ucese"] | None = Field(
        default=None,
        description="Override the evidence-backed default (atlas) for manual testing only.",
    )

    @model_validator(mode="after")
    def _validate_gap_phrase_count(self) -> "TrakeSearchRequest":
        expected = len(self.events) - 1
        if len(self.gap_phrases) != expected:
            raise ValueError(
                f"expected {expected} gap_phrases for {len(self.events)} events, "
                f"got {len(self.gap_phrases)}"
            )
        return self


class TrakeChainHit(BaseModel):
    rank: int = Field(ge=1)
    video_id: str
    frames: list[int | None]
    score: float


class TrakeSearchResponse(BaseModel):
    event_count: int = Field(ge=2)
    alignment_strategy: str
    count: int = Field(ge=0)
    results: list[TrakeChainHit]


class QnaSearchRequest(BaseModel):
    event_description: str = Field(min_length=1)
    question: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)

    _validate_event_description = field_validator("event_description")(_require_non_blank)
    _validate_question = field_validator("question")(_require_non_blank)


class QnaHit(BaseModel):
    rank: int = Field(ge=1)
    video_id: str
    frame_id: int
    score: float
    keyframe_id: str | None = None
    timestamp: float | None = None
    answer: str
    answer_source: Literal["context-grounded", "knowledge-grounded", "mixed"]
    #: Served URL for this candidate's keyframe (e.g. "/keyframes/L21_V001/003.jpg"),
    #: so a reviewer can see the frame. ``None`` when the API cannot serve it --
    #: a raw local filesystem path is never exposed here.
    image_path: str | None = None


class QnaSearchResponse(BaseModel):
    event_description: str
    question: str
    count: int = Field(ge=0)
    results: list[QnaHit]


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int = Field(ge=1)
    frame_id: str | int
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


class UiSpatialFilter(BaseModel):
    enabled: bool = False
    logic: Literal["AND", "OR"] = "AND"
    objects: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)


class UiVisualQuery(BaseModel):
    image_url: str = ""


class UiSearchRequest(BaseModel):
    task_mode: Literal["KIS", "TRAKE", "Q&A"]
    query: str = Field(min_length=1)
    model_engine: str = "use all"
    sim_reorder: bool = True
    spatial_filter: UiSpatialFilter | None = None
    visual_query: UiVisualQuery | None = None
    top_k: int = Field(default=100, ge=1, le=100)
    question: str | None = None
    event_description: str | None = None
    trake_events: list[str] | None = None
    gap_phrases: list[str] | None = None

    _validate_query = field_validator("query")(_require_non_blank)


class UiKeyframeResult(BaseModel):
    id: str
    video_id: str
    frame_id: int
    timestamp: str
    timestamp_sec: float
    thumbnail_url: str
    video_url: str
    score: float
    objects: list[str] = Field(default_factory=list)
    answer: str | None = None
    frames: list[int | None] | None = None


class UiSearchResponse(BaseModel):
    status: Literal["ok"] = "ok"
    total_results: int = Field(ge=0)
    results: list[UiKeyframeResult]


class VideoSummaryKeyframe(BaseModel):
    keyframe_id: str
    frame_id: int | None = None
    timestamp: str | None = None
    timestamp_sec: float | None = None
    thumbnail_url: str
    video_url: str
    mapped: bool


class VideoSummaryResponse(BaseModel):
    video_id: str
    keyframe_count: int = Field(ge=0)
    mapped_count: int = Field(ge=0)
    results: list[VideoSummaryKeyframe]


class UiSubmissionItem(BaseModel):
    video_id: str
    frame_id: int
    answer: str | None = None


class UiSubmitRequest(BaseModel):
    query_id: str = Field(min_length=1)
    task_mode: Literal["KIS", "TRAKE", "Q&A"]
    items: list[UiSubmissionItem] = Field(default_factory=list, max_length=100)

    _validate_query_id = field_validator("query_id")(_require_non_blank)


class UiSubmitResponse(BaseModel):
    status: Literal["ok"] = "ok"
    query_id: str
    task_mode: Literal["KIS", "TRAKE", "Q&A"]
    row_count: int = Field(ge=0)
    csv_preview: str


class SemanticRuntimeStatus(BaseModel):
    configured: bool
    available: bool
    loaded: bool
    backend: str | None = None
    reachable: bool | None = None
    collection_ready: bool | None = None
    collection: str | None = None
    entities: int | None = Field(default=None, ge=0)
    error: str | None = None


class ElasticsearchRuntimeStatus(BaseModel):
    configured: bool
    available: bool
    documents: int | None = Field(default=None, ge=0)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api: Literal["ready"] = "ready"
    semantic: SemanticRuntimeStatus
    elasticsearch: ElasticsearchRuntimeStatus
