"""Task05B v1.2 exact/folded/fuzzy retrieval over Task05A projections."""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from mvr.indexing.lexical_index import FOLDED_SEARCHABLE_FIELDS, SEARCHABLE_FIELDS
from mvr.indexing.normalization import normalize_lexical_query


logger = logging.getLogger(__name__)
LexicalStrategy = Literal["exact", "folded", "fuzzy"]


@dataclass(frozen=True)
class LexicalSearchHit:
    rank: int
    score: float
    keyframe_id: str
    video_id: str
    frame_idx: int
    shot_id: int | None
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "keyframe_id": self.keyframe_id,
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "shot_id": self.shot_id,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class LexicalSearchResult:
    """Observable fallback outcome without changing the public hit schema."""

    strategy: LexicalStrategy
    hits: tuple[LexicalSearchHit, ...]


def build_task05b_lexical_request(
    query: str,
    *,
    top_k: int = 5,
    strategy: LexicalStrategy = "exact",
) -> dict[str, Any]:
    normalized_query = normalize_lexical_query(query)
    if not normalized_query:
        raise ValueError("query must be non-empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be positive")
    if strategy not in ("exact", "folded", "fuzzy"):
        raise ValueError("strategy must be exact, folded, or fuzzy")

    if strategy == "exact":
        multi_match: dict[str, Any] = {
            "query": normalized_query,
            "type": "phrase",
            "fields": [f"{SEARCHABLE_FIELDS[0]}^2", SEARCHABLE_FIELDS[1]],
        }
    elif strategy == "folded":
        multi_match = {
            "query": normalized_query,
            "type": "phrase",
            "fields": [
                f"{FOLDED_SEARCHABLE_FIELDS[0]}^2",
                FOLDED_SEARCHABLE_FIELDS[1],
            ],
        }
    else:
        multi_match = {
            "query": normalized_query,
            "fields": [
                f"{FOLDED_SEARCHABLE_FIELDS[0]}^2",
                FOLDED_SEARCHABLE_FIELDS[1],
            ],
            "fuzziness": "AUTO",
            "prefix_length": 1,
            "max_expansions": 50,
            "minimum_should_match": "75%",
        }
    return {
        "size": top_k,
        "query": {"multi_match": multi_match},
        "sort": [
            {"_score": {"order": "desc"}},
            {"keyframe_id": {"order": "asc"}},
        ],
        "_source": [
            "keyframe_id",
            "video_id",
            "frame_idx",
            "shot_id",
            "timestamp",
        ],
    }


def elasticsearch_total_hits(response: Any) -> int:
    """Read Elasticsearch 7/8 ``hits.total`` in integer or object form."""

    if not isinstance(response, Mapping):
        response = getattr(response, "body", response)
    if not isinstance(response, Mapping):
        raise RuntimeError("malformed search response")
    hits_wrapper = response.get("hits")
    if not isinstance(hits_wrapper, Mapping):
        raise RuntimeError("malformed search response: missing hits")
    total = hits_wrapper.get("total")
    if isinstance(total, Mapping):
        total = total.get("value")
    if total is None:
        raw_hits = hits_wrapper.get("hits")
        if isinstance(raw_hits, list):
            return len(raw_hits)
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise RuntimeError("malformed search response: hits.total must be non-negative")
    return total


def _source_string(source: Mapping[str, Any], field: str) -> str:
    value = source.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"malformed search response: {field} must be non-empty")
    return value


def _source_int(
    source: Mapping[str, Any],
    field: str,
    *,
    nullable: bool = False,
) -> int | None:
    value = source.get(field)
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"malformed search response: {field} must be an integer")
    return value


def map_task05b_lexical_response(
    response: Any,
    *,
    top_k: int,
) -> list[LexicalSearchHit]:
    if not isinstance(response, Mapping):
        response = getattr(response, "body", response)
    if not isinstance(response, Mapping):
        raise RuntimeError("malformed search response")
    hits_wrapper = response.get("hits")
    if not isinstance(hits_wrapper, Mapping):
        raise RuntimeError("malformed search response: missing hits")
    raw_hits = hits_wrapper.get("hits")
    if not isinstance(raw_hits, list):
        raise RuntimeError("malformed search response: hits.hits must be a list")

    mapped: list[tuple[float, str, str, int, int | None, float]] = []
    for raw_hit in raw_hits[:top_k]:
        if not isinstance(raw_hit, Mapping):
            raise RuntimeError("malformed search response hit")
        source = raw_hit.get("_source")
        if not isinstance(source, Mapping):
            raise RuntimeError("malformed search response: missing _source")
        keyframe_id = _source_string(source, "keyframe_id")
        video_id = _source_string(source, "video_id")
        # `_id` là cặp video/keyframe, không phải keyframe trần -- xem
        # `LexicalIndexDocument.document_id`, phía sinh ra nó. Một `_id` trần
        # nghĩa là index được dựng bằng code trước PR #8, nơi "034" của 842
        # video cùng ghi vào một document; đọc tiếp sẽ giấu mất mất mát đó.
        if raw_hit.get("_id") != f"{video_id}/{keyframe_id}":
            raise RuntimeError("malformed search response: _id/keyframe_id mismatch")
        raw_score = raw_hit.get("_score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise RuntimeError("malformed search response: score must be numeric")
        score = float(raw_score)
        if not math.isfinite(score):
            raise RuntimeError("malformed search response: score must be finite")
        timestamp_value = source.get("timestamp")
        if (
            isinstance(timestamp_value, bool)
            or not isinstance(timestamp_value, (int, float))
            or not math.isfinite(float(timestamp_value))
        ):
            raise RuntimeError("malformed search response: timestamp must be finite")
        mapped.append(
            (
                score,
                keyframe_id,
                video_id,
                int(_source_int(source, "frame_idx")),
                _source_int(source, "shot_id", nullable=True),
                float(timestamp_value),
            )
        )

    mapped.sort(key=lambda item: (-item[0], item[1]))
    return [
        LexicalSearchHit(
            rank=rank,
            score=item[0],
            keyframe_id=item[1],
            video_id=item[2],
            frame_idx=item[3],
            shot_id=item[4],
            timestamp=item[5],
        )
        for rank, item in enumerate(mapped, start=1)
    ]


def search_task05b_lexical(
    client: Any,
    query: str,
    *,
    index_name: str,
    top_k: int = 5,
) -> list[LexicalSearchHit]:
    return list(
        search_task05b_lexical_with_strategy(
            client,
            query,
            index_name=index_name,
            top_k=top_k,
        ).hits
    )


def search_task05b_lexical_with_strategy(
    client: Any,
    query: str,
    *,
    index_name: str,
    top_k: int = 5,
) -> LexicalSearchResult:
    if not isinstance(index_name, str) or not index_name.strip():
        raise ValueError("index_name must be non-empty")
    normalized_query = normalize_lexical_query(query)
    if not normalized_query:
        raise ValueError("query must be non-empty")

    for strategy in ("exact", "folded", "fuzzy"):
        request = build_task05b_lexical_request(
            normalized_query,
            top_k=top_k,
            strategy=strategy,
        )
        response = client.search(index=index_name, **request)
        total = elasticsearch_total_hits(response)
        if total > 0 or strategy == "fuzzy":
            logger.info(
                "Task05B lexical search selected strategy=%s index=%s hits=%d",
                strategy,
                index_name,
                total,
            )
            return LexicalSearchResult(
                strategy=strategy,
                hits=tuple(map_task05b_lexical_response(response, top_k=top_k)),
            )

    raise AssertionError("unreachable lexical fallback state")
