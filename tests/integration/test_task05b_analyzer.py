"""Opt-in Elasticsearch 8 integration for Vietnamese lexical fallback."""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MVR_ES_URL"),
    reason="set MVR_ES_URL to an Elasticsearch 8 runtime",
)


def test_exact_folded_fuzzy_runtime_and_analyzers() -> None:
    from elasticsearch import Elasticsearch

    from mvr.indexing.lexical_index import LEXICAL_MAPPINGS, LEXICAL_SETTINGS
    from mvr.retrieval.lexical_bm25 import search_task05b_lexical_with_strategy

    client = Elasticsearch(os.environ["MVR_ES_URL"])
    index_name = "task05b-analyzer-test"
    client.indices.delete(index=index_name, ignore_unavailable=True)
    client.indices.create(
        index=index_name,
        mappings=LEXICAL_MAPPINGS,
        settings=LEXICAL_SETTINGS,
    )
    try:
        documents = (
            (
                "L21_V001/034",
                {
                    "keyframe_id": "034",
                    "video_id": "L21_V001",
                    "frame_idx": 1745,
                    "shot_id": 7,
                    "timestamp": 69.8,
                    "ocr_text": "GIÁ VÀNG TĂNG",
                    "asr_text": "",
                },
            ),
            (
                "L21_V002/035",
                {
                    "keyframe_id": "035",
                    "video_id": "L21_V002",
                    "frame_idx": 1800,
                    "shot_id": 8,
                    "timestamp": 72.0,
                    "ocr_text": "gia",
                    "asr_text": "",
                },
            ),
        )
        for document_id, document in documents:
            client.index(index=index_name, id=document_id, document=document)
        client.indices.refresh(index=index_name)

        exact_tokens = client.indices.analyze(
            index=index_name,
            analyzer="vi_exact",
            text="GIÁ VÀNG TĂNG",
        )["tokens"]
        folded_tokens = client.indices.analyze(
            index=index_name,
            analyzer="vi_folded",
            text="GIÁ VÀNG TĂNG",
        )["tokens"]
        assert [token["token"] for token in exact_tokens] == ["giá", "vàng", "tăng"]
        assert [token["token"] for token in folded_tokens] == ["gia", "vang", "tang"]

        expected = {
            "GIÁ VÀNG TĂNG": "exact",
            "giá vàng tăng": "exact",
            "gia vang tang": "folded",
            "gia vagn tang": "fuzzy",
        }
        for query, strategy in expected.items():
            result = search_task05b_lexical_with_strategy(
                client,
                query,
                index_name=index_name,
                top_k=5,
            )
            assert result.strategy == strategy, query
            assert [(hit.video_id, hit.keyframe_id) for hit in result.hits] == [
                ("L21_V001", "034")
            ], query

        mapping = client.indices.get_mapping(index=index_name)[index_name]["mappings"]
        settings = client.indices.get_settings(index=index_name)[index_name]["settings"]
        assert mapping["properties"]["ocr_text"] == LEXICAL_MAPPINGS["properties"][
            "ocr_text"
        ]
        analyzers = settings["index"]["analysis"]["analyzer"]
        assert analyzers["vi_exact"]["filter"] == ["lowercase"]
        assert analyzers["vi_folded"]["filter"] == ["lowercase", "asciifolding"]
    finally:
        client.indices.delete(index=index_name, ignore_unavailable=True)
