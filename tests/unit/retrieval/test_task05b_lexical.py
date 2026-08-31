from __future__ import annotations

import pytest

from mvr.retrieval.lexical_bm25 import (
    build_task05b_lexical_request,
    elasticsearch_total_hits,
    map_task05b_lexical_response,
    search_task05b_lexical,
    search_task05b_lexical_with_strategy,
)


def _hit(
    keyframe_id: str,
    *,
    score: float,
    frame_idx: int = 10,
    video_id: str = "video-a",
) -> dict:
    return {
        # Đúng định dạng `LexicalIndexDocument.document_id` sinh ra.
        "_id": f"{video_id}/{keyframe_id}",
        "_score": score,
        "_source": {
            "keyframe_id": keyframe_id,
            "video_id": video_id,
            "frame_idx": frame_idx,
            "shot_id": 1,
            "timestamp": 0.4,
        },
    }


def test_exact_request_is_a_diacritic_preserving_phrase() -> None:
    request = build_task05b_lexical_request("tin chính", top_k=3)
    assert request["query"] == {
        "multi_match": {
            "query": "tin chính",
            "type": "phrase",
            "fields": ["ocr_text^2", "asr_text"],
        }
    }
    assert request["sort"] == [
        {"_score": {"order": "desc"}},
        {"keyframe_id": {"order": "asc"}},
    ]
    assert request["size"] == 3
    assert "combined_text" not in str(request)


def test_folded_phrase_has_no_fuzziness_and_fuzzy_is_bounded() -> None:
    folded = build_task05b_lexical_request(
        "gia vang tang", strategy="folded"
    )["query"]["multi_match"]
    assert folded == {
        "query": "gia vang tang",
        "type": "phrase",
        "fields": ["ocr_text.folded^2", "asr_text.folded"],
    }
    assert "fuzziness" not in folded

    fuzzy = build_task05b_lexical_request(
        "gia vagn tang", strategy="fuzzy"
    )["query"]["multi_match"]
    assert fuzzy["fields"] == ["ocr_text.folded^2", "asr_text.folded"]
    assert fuzzy["fuzziness"] == "AUTO"
    assert fuzzy["prefix_length"] == 1
    assert fuzzy["max_expansions"] == 50
    assert fuzzy["minimum_should_match"] == "75%"
    assert "type" not in fuzzy


def test_query_normalization_is_nfc_whitespace_trim_and_casefold() -> None:
    request = build_task05b_lexical_request("  GIA\u0301   VA\u0300NG TĂNG  ")
    assert request["query"]["multi_match"]["query"] == "giá vàng tăng"


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
                    "total": {"value": 2, "relation": "eq"},
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
    assert len(calls) == 1


def test_exact_hit_short_circuits_folded_and_fuzzy() -> None:
    calls = []

    class Client:
        def search(self, **kwargs):
            calls.append(kwargs)
            return {
                "hits": {
                    "total": {"value": 1},
                    "hits": [_hit("a", score=2.0)],
                }
            }

    result = search_task05b_lexical_with_strategy(
        Client(), "GIÁ VÀNG TĂNG", index_name="lexical"
    )
    assert result.strategy == "exact"
    assert len(calls) == 1


def test_folded_hit_runs_only_after_empty_exact_and_skips_fuzzy() -> None:
    calls = []

    class Client:
        def search(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return {"hits": {"total": 0, "hits": []}}
            return {
                "hits": {
                    "total": {"value": 1},
                    "hits": [_hit("a", score=1.0)],
                }
            }

    result = search_task05b_lexical_with_strategy(
        Client(), "gia vang tang", index_name="lexical"
    )
    assert result.strategy == "folded"
    assert len(calls) == 2
    assert "fuzziness" not in calls[1]["query"]["multi_match"]


def test_fuzzy_runs_only_after_both_phrase_stages_are_empty() -> None:
    calls = []

    class Client:
        def search(self, **kwargs):
            calls.append(kwargs)
            if len(calls) < 3:
                return {"hits": {"total": {"value": 0}, "hits": []}}
            return {"hits": {"total": 1, "hits": [_hit("a", score=0.5)]}}

    result = search_task05b_lexical_with_strategy(
        Client(), "gia vagn tang", index_name="lexical"
    )
    assert result.strategy == "fuzzy"
    assert len(calls) == 3
    assert calls[2]["query"]["multi_match"]["minimum_should_match"] == "75%"


def test_all_three_empty_returns_fuzzy_strategy_with_no_hits() -> None:
    calls = []

    class Client:
        def search(self, **kwargs):
            calls.append(kwargs)
            return {"hits": {"total": {"value": 0}, "hits": []}}

    result = search_task05b_lexical_with_strategy(
        Client(), "nothing here", index_name="lexical"
    )
    assert result.strategy == "fuzzy"
    assert result.hits == ()
    assert len(calls) == 3


@pytest.mark.parametrize("failure_call", [1, 2])
def test_elasticsearch_exception_is_not_reinterpreted_as_zero_hits(
    failure_call: int,
) -> None:
    calls = []

    class Client:
        def search(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == failure_call:
                raise TimeoutError("Elasticsearch timed out")
            return {"hits": {"total": {"value": 0}, "hits": []}}

    with pytest.raises(TimeoutError, match="timed out"):
        search_task05b_lexical_with_strategy(
            Client(), "query", index_name="lexical"
        )
    assert len(calls) == failure_call


@pytest.mark.parametrize("query", ["gia", "gia vang", "gia vang tang"])
def test_fuzzy_request_supports_one_two_and_three_token_queries(query: str) -> None:
    request = build_task05b_lexical_request(query, strategy="fuzzy")
    multi_match = request["query"]["multi_match"]
    assert multi_match["query"] == query
    assert multi_match["minimum_should_match"] == "75%"


@pytest.mark.parametrize(
    ("total", "expected"),
    [(3, 3), ({"value": 4, "relation": "eq"}, 4)],
)
def test_total_hits_supports_integer_and_elasticsearch_8_object(
    total, expected
) -> None:
    assert elasticsearch_total_hits({"hits": {"total": total, "hits": []}}) == expected


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


def test_reader_accepts_the_id_the_writer_actually_produces() -> None:
    """Phía ghi và phía đọc phải dùng chung một định dạng `_id`.

    Trước đây mỗi phía được test bằng một chuỗi tự bịa -- phía ghi khẳng định
    "L21_V010/frame-a", phía đọc dựng fixture với `_id` là keyframe_id trần --
    nên cả hai pass trong khi ở production chúng không khớp nhau. Test này lấy
    `_id` từ chính object phía ghi, nên không thể lệch lần nữa.
    """
    from mvr.indexing.lexical_index import LexicalIndexDocument

    document = LexicalIndexDocument(
        keyframe_id="034",
        video_id="L21_V001",
        frame_idx=1745,
        shot_id=7,
        timestamp=69.8,
        ocr_text="công thức nấu ăn",
        asr_text="",
    )
    response = {
        "hits": {
            "hits": [
                {
                    "_id": document.document_id,
                    "_score": 12.5,
                    "_source": document.to_source(),
                }
            ]
        }
    }

    results = map_task05b_lexical_response(response, top_k=5)

    assert len(results) == 1
    assert (results[0].video_id, results[0].keyframe_id) == ("L21_V001", "034")
    assert results[0].frame_idx == 1745


def test_bare_keyframe_id_as_elasticsearch_id_is_rejected() -> None:
    """`_id` trần là dấu hiệu index được dựng bằng code trước PR #8, nơi "034"
    của 842 video đè lên nhau còn đúng một document. Đọc nó như dữ liệu hợp lệ
    sẽ giấu mất việc index đã mất 99,6% số dòng."""
    response = {
        "hits": {
            "hits": [
                {
                    "_id": "034",
                    "_score": 1.0,
                    "_source": {
                        "keyframe_id": "034",
                        "video_id": "L21_V001",
                        "frame_idx": 1745,
                        "shot_id": 7,
                        "timestamp": 69.8,
                    },
                }
            ]
        }
    }

    with pytest.raises(RuntimeError, match="_id/keyframe_id mismatch"):
        map_task05b_lexical_response(response, top_k=5)
