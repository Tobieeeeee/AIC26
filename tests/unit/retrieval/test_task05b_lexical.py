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
