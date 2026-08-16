import json
from pathlib import Path

import numpy as np
import pytest


def test_indexflatip_search_order_and_persistence(tmp_path: Path) -> None:
    from mvr.indexing.faiss_index import FaissIndex

    pytest.importorskip("faiss")
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    metadata = [
        {"frame_id": "frame-a", "video_id": "video", "timestamp": 0.0, "image_path": "a.jpg"},
        {"frame_id": "frame-b", "video_id": "video", "timestamp": 1.0, "image_path": "b.jpg"},
        {"frame_id": "frame-c", "video_id": "video", "timestamp": 2.0, "image_path": "c.jpg"},
    ]
    index = FaissIndex.build(
        vectors,
        metadata,
        manifest={
            "embedding_model": "test-model",
            "pretrained": "test-pretrained",
        },
    )

    hits = index.search(np.array([[0.8, 0.6]], dtype=np.float32), top_k=2)

    assert [(hit.metadata["frame_id"], round(hit.score, 6)) for hit in hits] == [
        ("frame-a", 0.8),
        ("frame-b", 0.6),
    ]

    index.save(tmp_path)
    loaded = FaissIndex.load(tmp_path)
    loaded_hits = loaded.search(
        np.array([[0.8, 0.6]], dtype=np.float32),
        top_k=2,
    )

    assert [hit.metadata for hit in loaded_hits] == metadata[:2]
    assert loaded.manifest == {
        "schema_version": "1.0",
        "index_type": "IndexFlatIP",
        "embedding_dimension": 2,
        "vector_count": 3,
        "normalization": "l2",
        "embedding_model": "test-model",
        "pretrained": "test-pretrained",
    }
    assert (tmp_path / "keyframes.faiss").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert (tmp_path / "manifest.json").is_file()


@pytest.mark.parametrize("top_k", [0, -1])
def test_faiss_search_rejects_non_positive_top_k(top_k: int) -> None:
    from mvr.indexing.faiss_index import FaissIndex

    pytest.importorskip("faiss")
    index = FaissIndex.build(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [
            {
                "frame_id": "frame-a",
                "video_id": "video",
                "timestamp": 0.0,
                "image_path": "a.jpg",
            }
        ],
        manifest={
            "embedding_model": "test-model",
            "pretrained": "test-pretrained",
        },
    )

    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=top_k)


def test_faiss_rejects_vectors_that_violate_cosine_contract() -> None:
    from mvr.indexing.faiss_index import FaissIndex

    pytest.importorskip("faiss")
    with pytest.raises(ValueError, match="L2-normalized"):
        FaissIndex.build(
            np.array([[2.0, 0.0]], dtype=np.float32),
            [{"frame_id": "frame-a"}],
        )

    index = FaissIndex.build(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [
            {
                "frame_id": "frame-a",
                "video_id": "video",
                "timestamp": 0.0,
                "image_path": "a.jpg",
            }
        ],
        manifest={
            "embedding_model": "test-model",
            "pretrained": "test-pretrained",
        },
    )
    with pytest.raises(ValueError, match="L2-normalized"):
        index.search(np.array([[0.5, 0.0]], dtype=np.float32))


def test_faiss_protects_and_validates_persistence_invariants(tmp_path: Path) -> None:
    from mvr.indexing.faiss_index import FaissIndex

    pytest.importorskip("faiss")
    with pytest.raises(ValueError, match="reserved manifest keys"):
        FaissIndex.build(
            np.array([[1.0, 0.0]], dtype=np.float32),
            [{"frame_id": "frame-a"}],
            manifest={"vector_count": 999},
        )

    index = FaissIndex.build(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [
            {
                "frame_id": "frame-a",
                "video_id": "video",
                "timestamp": 0.0,
                "image_path": "a.jpg",
            }
        ],
        manifest={
            "embedding_model": "test-model",
            "pretrained": "test-pretrained",
        },
    )
    index.save(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["embedding_dimension"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="embedding_dimension"):
        FaissIndex.load(tmp_path)



def test_model_agnostic_faiss_index_round_trips(tmp_path: Path) -> None:
    from mvr.indexing.faiss_index import FaissIndex

    pytest.importorskip("faiss")
    index = FaissIndex.build(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [
            {
                "frame_id": "frame-a",
                "video_id": "video",
                "timestamp": 0.0,
                "image_path": "a.jpg",
            }
        ],
    )
    index.save(tmp_path)

    loaded = FaissIndex.load(tmp_path)

    assert loaded.index.ntotal == 1
    assert loaded.metadata[0]["frame_id"] == "frame-a"
