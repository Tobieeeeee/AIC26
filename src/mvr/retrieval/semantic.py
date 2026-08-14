"""Text-to-keyframe retrieval over the Phase 03 FAISS index."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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
