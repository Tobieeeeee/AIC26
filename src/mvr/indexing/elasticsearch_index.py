"""Elasticsearch 8.x mapping and bulk indexing adapted from Lab 6."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .documents import SearchDocument


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
                    "_id": source["frame_id"],
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
