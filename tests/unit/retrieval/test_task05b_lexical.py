from __future__ import annotations

import pytest

from mvr.retrieval.lexical_bm25 import (
    build_task05b_lexical_request,
    map_task05b_lexical_response,
    search_task05b_lexical,
)


def _hit(
    keyframe_id: str,
    *,
    score: float,
    frame_idx: int = 10,
) -> dict:
    return {
        "_id": keyframe_id,
        "_score": score,
        "_source": {
            "keyframe_id": keyframe_id,
            "video_id": "video-a",
            "frame_idx": frame_idx,
            "shot_id": 1,
            "timestamp": 0.4,
        },
    }


def test_request_searches_exact_fields_with_equal_weights_and_deterministic_sort() -> None:
    request = build_task05b_lexical_request("tin chính", top_k=3)
    assert request["query"] == {
        "multi_match": {
            "query": "tin chính",
            "fields": ["ocr_text", "asr_text"],
        }
    }
    assert request["sort"] == [
        {"_score": {"order": "desc"}},
        {"keyframe_id": {"order": "asc"}},
    ]
    assert request["size"] == 3
    assert "combined_text" not in str(request)


@pytest.mark.parametrize("query", ["", " ", "\t"])
def test_empty_or_whitespace_query_is_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="query must be non-empty"):
        build_task05b_lexical_request(query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        build_task05b_lexical_request("query", top_k=top_k)


def test_response_mapping_preserves_identity_and_breaks_score_ties_by_id() -> None:
    results = map_task05b_lexical_response(
        {"hits": {"hits": [_hit("frame-b", score=2.0), _hit("frame-a", score=2.0)]}},
        top_k=5,
    )
    assert [result.keyframe_id for result in results] == ["frame-a", "frame-b"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].frame_idx == 10


def test_search_respects_top_k_and_selected_index() -> None:
    calls = []

    class Client:
        def search(self, **kwargs):
            calls.append(kwargs)
            return {
                "hits": {
                    "hits": [
                        _hit("frame-a", score=2.0),
                        _hit("frame-b", score=1.0),
                    ]
                }
            }

    results = search_task05b_lexical(
        Client(),
        "query",
        index_name="lexical",
        top_k=1,
    )
    assert len(results) == 1
    assert calls[0]["index"] == "lexical"
    assert calls[0]["size"] == 1


def test_official_client_object_api_response_body_is_supported() -> None:
    class Response:
        body = {"hits": {"hits": [_hit("frame-a", score=1.0)]}}

    results = map_task05b_lexical_response(Response(), top_k=5)
    assert [result.keyframe_id for result in results] == ["frame-a"]


@pytest.mark.parametrize(
    "response",
    [{}, {"hits": {}}, {"hits": {"hits": [None]}}, {"hits": {"hits": [{"_id": "x"}]}}],
)
def test_malformed_search_response_is_rejected(response: dict) -> None:
    with pytest.raises(RuntimeError, match="malformed search response"):
        map_task05b_lexical_response(response, top_k=5)
