"""Lab 6 BM25 OCR/caption request construction and execution."""
from __future__ import annotations

from typing import Any


DEFAULT_INDEX_NAME = "keyframes"


def build_lexical_request(
    query: str,
    *,
    top_k: int = 5,
    person_min: int = 0,
    car_min: int = 0,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if person_min < 0 or car_min < 0:
        raise ValueError("object minimum counts must be non-negative")
    bool_query = {
        "must": (
            [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["ocr_text^2", "caption"],
                    }
                }
            ]
            if query
            else [{"match_all": {}}]
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
    request = build_lexical_request(
        query,
        top_k=top_k,
        person_min=person_min,
        car_min=car_min,
    )
    return client.search(index=index_name, **request)
