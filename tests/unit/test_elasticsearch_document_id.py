"""Elasticsearch document ids must be collision-safe.

Keyframe ids are not globally unique across the dataset, so an ``_id`` built
from ``frame_id`` alone lets one video's document overwrite another's.
"""
from __future__ import annotations

from typing import Any

import pytest

from mvr.indexing.documents import SearchDocument
from mvr.indexing.elasticsearch_index import ElasticsearchIndex


class _FakeIndices:
    def refresh(self, index=None) -> None:
        return None


class _FakeClient:
    def __init__(self) -> None:
        self.indices = _FakeIndices()

    def count(self, index=None) -> dict[str, Any]:
        return {"count": 0}


def _capture() -> tuple[list[dict], Any]:
    captured: list[dict] = []

    def bulk_fn(client, actions, **kwargs):
        captured.extend(actions)
        return len(captured), []

    return captured, bulk_fn


def _document(video_id: str, frame_id: str) -> SearchDocument:
    return SearchDocument(
        frame_id=frame_id,
        video_id=video_id,
        ts=1.0,
        caption="",
        ocr_text="",
        objects={},
        clip_vector=[0.0, 1.0],
    )


def test_document_id_combines_the_video_and_keyframe_id() -> None:
    captured, bulk_fn = _capture()
    index = ElasticsearchIndex(_FakeClient())

    index.index_documents([_document("L21_V001", "034")], bulk_fn=bulk_fn)

    assert captured[0]["_id"] == "L21_V001/034"


def test_two_videos_sharing_a_keyframe_id_produce_two_documents() -> None:
    captured, bulk_fn = _capture()
    index = ElasticsearchIndex(_FakeClient())

    index.index_documents(
        [_document("L21_V001", "034"), _document("L21_V002", "034")],
        bulk_fn=bulk_fn,
    )

    assert {action["_id"] for action in captured} == {
        "L21_V001/034",
        "L21_V002/034",
    }


def test_an_explicit_document_id_is_used_as_is() -> None:
    captured, bulk_fn = _capture()
    index = ElasticsearchIndex(_FakeClient())

    index.index_documents(
        [
            {
                "document_id": "L21_V001/034",
                "frame_id": "034",
                "video_id": "L21_V001",
                "ts": 1.0,
                "caption": "",
                "ocr_text": "",
                "objects": {},
            }
        ],
        bulk_fn=bulk_fn,
    )

    assert captured[0]["_id"] == "L21_V001/034"


def test_a_document_without_a_video_id_is_refused() -> None:
    _captured, bulk_fn = _capture()
    index = ElasticsearchIndex(_FakeClient())

    with pytest.raises(ValueError, match="video_id"):
        index.index_documents(
            [{"frame_id": "034", "ts": 1.0, "caption": "", "ocr_text": "", "objects": {}}],
            bulk_fn=bulk_fn,
        )
