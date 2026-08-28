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


def test_elasticsearch_semantic_request_preserves_lab6_knn_defaults() -> None:
    from mvr.retrieval.semantic import ElasticsearchSemanticRetriever

    calls = []

    class Encoder:
        def encode_text(self, texts):
            assert texts == ["a dog running on grass"]
            return np.array([[0.6, 0.8]], dtype=np.float32)

    class Client:
        def search(self, **kwargs):
            calls.append(kwargs)
            return {"hits": {"hits": []}}

    retriever = ElasticsearchSemanticRetriever(Encoder(), Client())

    assert retriever.search("  a dog running on grass  ", top_k=5) == {
        "hits": {"hits": []}
    }
    assert calls == [
        {
            "index": "keyframes",
            "size": 5,
            "knn": {
                "field": "clip_vector",
                "query_vector": [0.6000000238418579, 0.800000011920929],
                "k": 5,
                "num_candidates": 100,
            },
        }
    ]


def test_lab6_encoder_factory_keeps_rn50_separate_from_phase03() -> None:
    from mvr.retrieval.semantic import create_lab6_openclip_encoder

    calls = []

    class Encoder:
        model_name = "RN50"
        pretrained = "openai"

        def encode_images(self, paths, batch_size):
            calls.append(("images", list(paths), batch_size))
            return "images"

        def encode_text(self, texts, batch_size):
            calls.append(("text", list(texts), batch_size))
            return "text"

    def factory(**kwargs):
        calls.append(kwargs)
        return Encoder()

    encoder = create_lab6_openclip_encoder(device="cpu", encoder_factory=factory)

    assert encoder.encode_images(["frame.jpg"]) == "images"
    assert encoder.encode_text(["query"]) == "text"
    assert encoder.model_name == "RN50"
    assert encoder.pretrained == "openai"
    assert calls == [
        {"model_name": "RN50", "pretrained": "openai", "device": "cpu"},
        ("images", ["frame.jpg"], 64),
        ("text", ["query"], 64),
    ]


def test_elasticsearch_semantic_validates_candidate_and_vector_contracts() -> None:
    from mvr.retrieval.semantic import ElasticsearchSemanticRetriever

    class Client:
        def search(self, **_):
            pytest.fail("invalid requests must not reach Elasticsearch")

    class UnitEncoder:
        def encode_text(self, _):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="num_candidates must be positive"):
        ElasticsearchSemanticRetriever(UnitEncoder(), Client(), num_candidates=0)

    retriever = ElasticsearchSemanticRetriever(
        UnitEncoder(), Client(), num_candidates=100
    )
    with pytest.raises(ValueError, match="cannot exceed num_candidates"):
        retriever.search("query", top_k=101)

    class UnnormalizedEncoder:
        def encode_text(self, _):
            return np.array([[1.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="L2-normalized"):
        ElasticsearchSemanticRetriever(UnnormalizedEncoder(), Client()).search("query")


def _portable_index(image_root: str):
    """An index built the portable way: paths relative, root in manifest."""
    from mvr.indexing.faiss_index import FaissIndex

    return FaissIndex.build(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [
            {
                "frame_id": "001",
                "video_id": "L21_V001",
                "timestamp": 0.4,
                "image_path": "Keyframes_L21/keyframes/L21_V001/001.jpg",
            }
        ],
        manifest={"image_root": image_root},
    )


def test_a_relative_hit_path_is_joined_to_the_root_the_index_recorded() -> None:
    """Every consumer reaches image_path through this class, so resolving
    it anywhere else would mean fixing the same thing three times."""
    from pathlib import Path

    from mvr.retrieval.semantic import SemanticRetriever

    pytest.importorskip("faiss")

    result = SemanticRetriever(
        StubTextEncoder(), _portable_index("E:/AIC26_data")
    ).search("motorcycle", top_k=1)[0]

    assert Path(result.image_path) == Path(
        "E:/AIC26_data/Keyframes_L21/keyframes/L21_V001/001.jpg"
    )


def test_an_image_root_override_reaches_every_result() -> None:
    """The point of the whole exercise: one index, read on a machine that
    mounts the corpus somewhere else."""
    from pathlib import Path

    from mvr.retrieval.semantic import SemanticRetriever

    pytest.importorskip("faiss")

    result = SemanticRetriever(
        StubTextEncoder(),
        _portable_index("E:/AIC26_data"),
        image_root="/kaggle/input/aic26",
    ).search("motorcycle", top_k=1)[0]

    assert Path(result.image_path) == Path(
        "/kaggle/input/aic26/Keyframes_L21/keyframes/L21_V001/001.jpg"
    )
