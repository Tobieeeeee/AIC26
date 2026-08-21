"""Kiểm chứng analyzer `vi_folded` trên một Elasticsearch thật.

Unit test chỉ khẳng định payload tạo index có đúng settings; nó không chứng
minh Elasticsearch thật sự bỏ dấu. Test này cần một ES đang chạy nên tắt mặc
định -- máy phát triển hiện không có Docker.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MVR_ES_URL"),
    reason="đặt MVR_ES_URL trỏ tới một Elasticsearch đang chạy để chạy test này",
)


def test_unaccented_query_matches_accented_vietnamese_text() -> None:
    from elasticsearch import Elasticsearch

    from mvr.indexing.lexical_index import LEXICAL_MAPPINGS, LEXICAL_SETTINGS
    from mvr.retrieval.lexical_bm25 import search_task05b_lexical

    client = Elasticsearch(os.environ["MVR_ES_URL"])
    index_name = "task05b-analyzer-test"
    client.indices.delete(index=index_name, ignore_unavailable=True)
    client.indices.create(
        index=index_name,
        mappings=LEXICAL_MAPPINGS,
        settings=LEXICAL_SETTINGS,
    )
    try:
        client.index(
            index=index_name,
            id="L21_V001/034",
            document={
                "keyframe_id": "034",
                "video_id": "L21_V001",
                "frame_idx": 1745,
                "shot_id": 7,
                "timestamp": 69.8,
                "ocr_text": "Hà Nội mùa thu",
                "asr_text": "",
            },
            refresh=True,
        )

        for query in ("Hà Nội", "Ha Noi", "ha noi"):
            results = search_task05b_lexical(
                client, query, index_name=index_name, top_k=5
            )
            assert [r.keyframe_id for r in results] == ["034"], query
    finally:
        client.indices.delete(index=index_name, ignore_unavailable=True)
