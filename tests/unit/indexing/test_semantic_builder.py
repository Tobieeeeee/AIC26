from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


class StubImageEncoder:
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"

    def encode_images(self, image_paths, batch_size: int = 32) -> np.ndarray:
        paths = list(image_paths)
        assert [path.name for path in paths] == ["frame-a.jpg", "frame-b.jpg"]
        assert batch_size == 2
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def test_builder_consumes_phase02_mapping_and_persists_embeddings(
    tmp_path: Path,
) -> None:
    from mvr.indexing.faiss_index import FaissIndex
    from mvr.indexing.semantic_builder import build_semantic_index

    pytest.importorskip("faiss")
    ingestion_root = tmp_path / "ingestion"
    image_root = ingestion_root / "keyframes" / "video"
    image_root.mkdir(parents=True)
    (image_root / "frame-a.jpg").touch()
    (image_root / "frame-b.jpg").touch()
    with (ingestion_root / "keyframe_mapping.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "keyframe_id",
                "video_id",
                "shot_id",
                "ordinal",
                "frame_idx",
                "timestamp",
                "image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "keyframe_id": "frame-a",
                    "video_id": "video",
                    "shot_id": 1,
                    "ordinal": 1,
                    "frame_idx": 10,
                    "timestamp": 0.4,
                    "image_path": "keyframes/video/frame-a.jpg",
                },
                {
                    "keyframe_id": "frame-b",
                    "video_id": "video",
                    "shot_id": 2,
                    "ordinal": 1,
                    "frame_idx": 40,
                    "timestamp": 1.6,
                    "image_path": "keyframes/video/frame-b.jpg",
                },
            ]
        )

    index_root = tmp_path / "index"
    build_semantic_index(
        ingestion_root,
        index_root,
        StubImageEncoder(),
        batch_size=2,
    )

    embeddings = np.load(index_root / "image_embeddings.npy")
    assert embeddings.dtype == np.float32
    assert embeddings.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    loaded = FaissIndex.load(index_root)
    assert loaded.metadata == [
        {
            "frame_id": "frame-a",
            "video_id": "video",
            "timestamp": 0.4,
            "image_path": str((image_root / "frame-a.jpg").resolve()),
        },
        {
            "frame_id": "frame-b",
            "video_id": "video",
            "timestamp": 1.6,
            "image_path": str((image_root / "frame-b.jpg").resolve()),
        },
    ]
    assert all(Path(row["image_path"]).is_file() for row in loaded.metadata)
    hit = loaded.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=1)[0]
    assert Path(hit.metadata["image_path"]).is_file()
    assert loaded.manifest["embedding_model"] == "ViT-B-32"
    assert loaded.manifest["pretrained"] == "laion2b_s34b_b79k"
