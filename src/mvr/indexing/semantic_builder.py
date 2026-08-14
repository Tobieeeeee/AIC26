"""Build a persistent semantic keyframe index from Phase 02 output."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .faiss_index import FaissIndex


def build_semantic_index(
    ingestion_root: Path,
    output_root: Path,
    encoder,
    batch_size: int = 32,
) -> FaissIndex:
    ingestion_root = Path(ingestion_root)
    output_root = Path(output_root)
    mapping_path = ingestion_root / "keyframe_mapping.csv"
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no keyframes found in {mapping_path}")

    image_paths = [ingestion_root / row["image_path"] for row in rows]
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"keyframe image not found: {missing[0]}")

    embeddings = np.ascontiguousarray(
        encoder.encode_images(image_paths, batch_size=batch_size),
        dtype=np.float32,
    )
    metadata = [
        {
            "frame_id": row["keyframe_id"],
            "video_id": row["video_id"],
            "timestamp": float(row["timestamp"]),
            "image_path": str(image_path.resolve()),
        }
        for row, image_path in zip(rows, image_paths)
    ]
    index = FaissIndex.build(
        embeddings,
        metadata,
        manifest={
            "embedding_model": encoder.model_name,
            "pretrained": encoder.pretrained,
        },
    )
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / "image_embeddings.npy", embeddings)
    index.save(output_root)
    return index
