"""Elasticsearch 8.x mapping and bulk indexing adapted from Lab 6."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .documents import SearchDocument
# Shared with the vector side on purpose: both indexes must derive a document
# identity the same way, or a keyframe resolves differently in each.
from .milvus_schema import make_entity_id


DEFAULT_INDEX_NAME = "keyframes"
SETTINGS = {
    "analysis": {
        "analyzer": {
            "vi_folded": {
                "tokenizer": "standard",
                "filter": ["lowercase", "asciifolding"],
            }
        }
    }
}


def build_mappings(embedding_dimension: int) -> dict[str, Any]:
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be positive")
    return {
        "properties": {
            "frame_id": {"type": "keyword"},
            "video_id": {"type": "keyword"},
            "ts": {"type": "float"},
            "caption": {"type": "text", "analyzer": "vi_folded"},
            "ocr_text": {"type": "text", "analyzer": "vi_folded"},
            "objects": {
                "properties": {
                    "person": {"type": "integer"},
                    "car": {"type": "integer"},
                }
            },
            "clip_vector": {
                "type": "dense_vector",
                "dims": int(embedding_dimension),
                "index": True,
                "similarity": "cosine",
            },
        }
    }


def document_id(source: Mapping[str, Any]) -> str:
    """Return a collision-safe Elasticsearch ``_id`` for one document.

    Keyframe ids are not globally unique: the organizer's ids are per-video
    ordinals, so 177,321 of their keyframes share only 632 distinct id strings.
    An ``_id`` taken from ``frame_id`` alone therefore lets one video's
    document overwrite another's. A document that cannot name its video is
    refused rather than indexed under an ambiguous id.
    """
    explicit = str(source.get("document_id") or "").strip()
    if explicit:
        return explicit
    frame_id = str(source.get("frame_id") or "").strip()
    if not frame_id:
        raise ValueError("search document is missing frame_id")
    video_id = str(source.get("video_id") or "").strip()
    if not video_id:
        raise ValueError(
            f"search document {frame_id!r} is missing video_id: keyframe ids are "
            "not globally unique, so an _id built from frame_id alone would "
            "overwrite another video's keyframe"
        )
    return make_entity_id(video_id, frame_id)


class ElasticsearchIndex:
    def __init__(self, client: Any, index_name: str = DEFAULT_INDEX_NAME) -> None:
        self.client = client
        self.index_name = index_name

    def recreate(self, embedding_dimension: int) -> None:
        self.client.indices.delete(
            index=self.index_name,
            ignore_unavailable=True,
        )
        self.client.indices.create(
            index=self.index_name,
            settings=SETTINGS,
            mappings=build_mappings(embedding_dimension),
        )

    def index_documents(
        self,
        documents: Iterable[SearchDocument | Mapping[str, Any]],
        *,
        bulk_fn: Callable[..., Any] | None = None,
    ) -> int:
        if bulk_fn is None:
            try:
                from elasticsearch.helpers import bulk
            except ImportError as exc:  # pragma: no cover - optional integration
                raise RuntimeError(
                    "Elasticsearch client is missing; install the project's search extra"
                ) from exc
            bulk_fn = bulk

        def actions():
            for document in documents:
                source = (
                    document.to_source()
                    if isinstance(document, SearchDocument)
                    else dict(document)
                )
                yield {
                    "_index": self.index_name,
                    "_id": document_id(source),
                    **source,
                }

        bulk_fn(
            self.client,
            actions(),
            chunk_size=50,
            request_timeout=120,
        )
        self.client.indices.refresh(index=self.index_name)
        return int(self.client.count(index=self.index_name)["count"])
