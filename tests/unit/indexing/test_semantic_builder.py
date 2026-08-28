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
    from mvr.indexing.faiss_index import FaissIndex, resolve_image_path
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
    # image_path is stored relative to the ingestion root the manifest
    # carries; it was absolute until portability was added, which pinned
    # the index to the one disk that built it.
    assert loaded.metadata == [
        {
            "frame_id": "frame-a",
            "video_id": "video",
            "timestamp": 0.4,
            "image_path": "keyframes/video/frame-a.jpg",
        },
        {
            "frame_id": "frame-b",
            "video_id": "video",
            "timestamp": 1.6,
            "image_path": "keyframes/video/frame-b.jpg",
        },
    ]
    assert all(
        resolve_image_path(row, loaded.manifest).is_file() for row in loaded.metadata
    )
    hit = loaded.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=1)[0]
    assert resolve_image_path(hit.metadata, loaded.manifest).is_file()
    assert (image_root / "frame-a.jpg").is_file()
    assert loaded.manifest["embedding_model"] == "ViT-B-32"
    assert loaded.manifest["pretrained"] == "laion2b_s34b_b79k"


def _phase02_root(tmp_path: Path) -> tuple[Path, Path]:
    """The smallest ingestion output a builder will accept."""
    ingestion_root = tmp_path / "ingestion"
    image_root = ingestion_root / "keyframes" / "video"
    image_root.mkdir(parents=True)
    (image_root / "frame-a.jpg").touch()
    (image_root / "frame-b.jpg").touch()
    with (ingestion_root / "keyframe_mapping.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        handle.write(
            "keyframe_id,video_id,shot_id,ordinal,frame_idx,timestamp,image_path\n"
            "frame-a,video,1,1,10,0.4,keyframes/video/frame-a.jpg\n"
            "frame-b,video,2,1,40,1.6,keyframes/video/frame-b.jpg\n"
        )
    return ingestion_root, image_root


def test_the_index_stores_paths_relative_to_the_ingestion_root(tmp_path: Path) -> None:
    """An absolute path pins the index to one machine. `answer_queries`
    puts this column in the review sheet a person opens by eye, so a
    teammate who receives the index sees rows they cannot open."""
    from mvr.indexing.faiss_index import FaissIndex
    from mvr.indexing.semantic_builder import build_semantic_index

    pytest.importorskip("faiss")
    ingestion_root, _ = _phase02_root(tmp_path)

    build_semantic_index(
        ingestion_root, tmp_path / "index", StubImageEncoder(), batch_size=2
    )

    loaded = FaissIndex.load(tmp_path / "index")
    assert [row["image_path"] for row in loaded.metadata] == [
        "keyframes/video/frame-a.jpg",
        "keyframes/video/frame-b.jpg",
    ]


def test_the_recorded_root_still_finds_the_images_on_this_machine(
    tmp_path: Path,
) -> None:
    """Relative paths are only useful if the root travels with them."""
    from mvr.indexing.faiss_index import FaissIndex, resolve_image_path
    from mvr.indexing.semantic_builder import build_semantic_index

    pytest.importorskip("faiss")
    ingestion_root, _ = _phase02_root(tmp_path)

    build_semantic_index(
        ingestion_root, tmp_path / "index", StubImageEncoder(), batch_size=2
    )

    loaded = FaissIndex.load(tmp_path / "index")
    # Both halves matter: a stored path that is still absolute would also
    # resolve here, on the machine that wrote it, and prove nothing.
    assert not any(Path(row["image_path"]).is_absolute() for row in loaded.metadata)
    assert all(
        resolve_image_path(row, loaded.manifest).is_file() for row in loaded.metadata
    )


def test_the_index_records_the_mapping_it_was_built_from(tmp_path: Path) -> None:
    """Without this the index and the CSV drift apart silently: the frame
    resolver finds nothing for changed rows and drops them, so the run
    succeeds and the submission is short."""
    from mvr.indexing.faiss_index import (
        FaissIndex,
        mapping_fingerprint,
        verify_mapping_binding,
    )
    from mvr.indexing.semantic_builder import build_semantic_index

    pytest.importorskip("faiss")
    ingestion_root, _ = _phase02_root(tmp_path)
    mapping_csv = ingestion_root / "keyframe_mapping.csv"

    build_semantic_index(
        ingestion_root, tmp_path / "index", StubImageEncoder(), batch_size=2
    )

    loaded = FaissIndex.load(tmp_path / "index")
    assert loaded.manifest["mapping_sha256"] == mapping_fingerprint(mapping_csv)["mapping_sha256"]
    assert loaded.manifest["mapping_row_count"] == 2
    verify_mapping_binding(loaded.manifest, mapping_csv)
