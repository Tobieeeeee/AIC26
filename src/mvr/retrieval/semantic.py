"""Text-to-keyframe retrieval over the Phase 03 FAISS index."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable
from typing import Any

import numpy as np


LAB6_MODEL_NAME = "RN50"
LAB6_PRETRAINED = "openai"
LAB6_IMAGE_BATCH_SIZE = 64


class Lab6OpenClipEncoder:
    """Apply Lab 6's batch defaults without changing the Phase 03 adapter."""

    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self.model_name = encoder.model_name
        self.pretrained = encoder.pretrained

    def encode_images(self, image_paths, batch_size: int = LAB6_IMAGE_BATCH_SIZE):
        return self.encoder.encode_images(image_paths, batch_size=batch_size)

    def encode_text(self, texts, batch_size: int = 64):
        return self.encoder.encode_text(texts, batch_size=batch_size)


def create_lab6_openclip_encoder(
    *,
    device: str = "auto",
    encoder_factory: Callable[..., Any] | None = None,
) -> Any:
    if encoder_factory is None:
        from ..enrichment.embeddings.openclip import OpenClipEncoder

        encoder_factory = OpenClipEncoder
    return Lab6OpenClipEncoder(
        encoder_factory(
            model_name=LAB6_MODEL_NAME,
            pretrained=LAB6_PRETRAINED,
            device=device,
        )
    )


@dataclass(frozen=True)
class SearchResult:
    rank: int
    frame_id: str
    video_id: str
    timestamp: float
    score: float
    image_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticRetriever:
    def __init__(self, text_encoder, index) -> None:
        self.text_encoder = text_encoder
        self.index = index

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        query_vector = self.text_encoder.encode_text([query.strip()])
        hits = self.index.search(query_vector, top_k=top_k)
        return [
            SearchResult(
                rank=rank,
                frame_id=str(hit.metadata["frame_id"]),
                video_id=str(hit.metadata["video_id"]),
                timestamp=float(hit.metadata["timestamp"]),
                score=hit.score,
                image_path=str(hit.metadata["image_path"]),
            )
            for rank, hit in enumerate(hits, start=1)
        ]


class ElasticsearchSemanticRetriever:
    """Execute Lab 6 cosine kNN requests with an injected CLIP encoder."""

    def __init__(
        self,
        text_encoder,
        client: Any,
        *,
        index_name: str = "keyframes",
        vector_field: str = "clip_vector",
        num_candidates: int = 100,
    ) -> None:
        self.text_encoder = text_encoder
        self.client = client
        self.index_name = index_name
        self.vector_field = vector_field
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        self.num_candidates = num_candidates

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > self.num_candidates:
            raise ValueError("top_k cannot exceed num_candidates")
        vector = np.asarray(
            self.text_encoder.encode_text([query.strip()]),
            dtype=np.float32,
        )
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError("text encoder must return one two-dimensional vector")
        if not np.isfinite(vector).all():
            raise ValueError("text encoder vector must contain only finite values")
        if not np.allclose(
            np.linalg.norm(vector, axis=1),
            1.0,
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("text encoder vector must be L2-normalized")
        return self.client.search(
            index=self.index_name,
            size=top_k,
            knn={
                "field": self.vector_field,
                "query_vector": vector[0].tolist(),
                "k": top_k,
                "num_candidates": self.num_candidates,
            },
        )
