"""BM25 OCR/ASR request construction with Vietnamese fallback."""
from __future__ import annotations

import logging
from typing import Any

from mvr.indexing.normalization import normalize_lexical_query
from mvr.retrieval.lexical_bm25 import (
    LexicalStrategy,
    build_task05b_lexical_request,
    elasticsearch_total_hits,
)


DEFAULT_INDEX_NAME = "keyframes"
logger = logging.getLogger(__name__)


def build_lexical_request(
    query: str,
    *,
    top_k: int = 5,
    person_min: int = 0,
    car_min: int = 0,
    strategy: LexicalStrategy = "exact",
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if person_min < 0 or car_min < 0:
        raise ValueError("object minimum counts must be non-negative")
    normalized_query = normalize_lexical_query(query)
    if normalized_query:
        lexical_query = build_task05b_lexical_request(
            normalized_query,
            top_k=top_k,
            strategy=strategy,
        )["query"]
    else:
        lexical_query = {"match_all": {}}
    bool_query = {
        "must": (
            [lexical_query]
        ),
        "filter": [],
    }
    if person_min:
        bool_query["filter"].append(
            {"range": {"objects.person": {"gte": person_min}}}
        )
    if car_min:
        bool_query["filter"].append(
            {"range": {"objects.car": {"gte": car_min}}}
        )
    return {
        "size": top_k,
        "query": {"bool": bool_query},
        "highlight": {"fields": {"ocr_text": {}}},
    }


def search_lexical(
    client: Any,
    query: str,
    *,
    top_k: int = 5,
    person_min: int = 0,
    car_min: int = 0,
    index_name: str = DEFAULT_INDEX_NAME,
) -> dict[str, Any]:
    response, _ = search_lexical_with_strategy(
        client,
        query,
        top_k=top_k,
        person_min=person_min,
        car_min=car_min,
        index_name=index_name,
    )
    return response


def search_lexical_with_strategy(
    client: Any,
    query: str,
    *,
    top_k: int = 5,
    person_min: int = 0,
    car_min: int = 0,
    index_name: str = DEFAULT_INDEX_NAME,
) -> tuple[dict[str, Any], LexicalStrategy]:
    normalized_query = normalize_lexical_query(query)
    strategies: tuple[LexicalStrategy, ...] = (
        ("exact", "folded", "fuzzy") if normalized_query else ("exact",)
    )
    for strategy in strategies:
        request = build_lexical_request(
            normalized_query,
            top_k=top_k,
            person_min=person_min,
            car_min=car_min,
            strategy=strategy,
        )
        response = client.search(index=index_name, **request)
        total = elasticsearch_total_hits(response)
        if total > 0 or strategy == strategies[-1]:
            logger.info(
                "lexical search selected strategy=%s index=%s hits=%d",
                strategy,
                index_name,
                total,
            )
            return response, strategy
    raise AssertionError("unreachable lexical fallback state")
