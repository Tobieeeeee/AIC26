from __future__ import annotations

import numpy as np
import pytest


class StubTextEncoder:
    def encode_text(self, texts, batch_size: int = 64) -> np.ndarray:
        assert list(texts) == ["motorcycle"]
        return np.array([[0.8, 0.6]], dtype=np.float32)


def test_semantic_search_reconstructs_keyframe_results() -> None:
    from mvr.indexing.faiss_index import FaissIndex
    from mvr.retrieval.semantic import SemanticRetriever

    pytest.importorskip("faiss")
    index = FaissIndex.build(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [
            {
                "frame_id": "frame-a",
                "video_id": "video-1",
                "timestamp": 0.4,
                "image_path": "keyframes/video-1/frame-a.jpg",
            },
            {
                "frame_id": "frame-b",
                "video_id": "video-1",
                "timestamp": 1.6,
                "image_path": "keyframes/video-1/frame-b.jpg",
            },
        ],
    )
    retriever = SemanticRetriever(StubTextEncoder(), index)

    results = retriever.search("  motorcycle  ", top_k=2)

    assert [result.to_dict() for result in results] == [
        {
            "rank": 1,
            "frame_id": "frame-a",
            "video_id": "video-1",
            "timestamp": 0.4,
            "score": pytest.approx(0.8),
            "image_path": "keyframes/video-1/frame-a.jpg",
        },
        {
            "rank": 2,
            "frame_id": "frame-b",
            "video_id": "video-1",
            "timestamp": 1.6,
            "score": pytest.approx(0.6),
            "image_path": "keyframes/video-1/frame-b.jpg",
        },
    ]


@pytest.mark.parametrize("query", ["", "   ", None])
def test_semantic_search_rejects_empty_query(query) -> None:
    from mvr.retrieval.semantic import SemanticRetriever

    retriever = SemanticRetriever(StubTextEncoder(), object())
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.search(query)
