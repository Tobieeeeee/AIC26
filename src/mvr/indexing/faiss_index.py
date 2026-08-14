"""Exact inner-product FAISS index adapted from Lab 4."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


INDEX_FILENAME = "keyframes.faiss"
METADATA_FILENAME = "metadata.json"
MANIFEST_FILENAME = "manifest.json"
CORE_MANIFEST_KEYS = {
    "schema_version",
    "index_type",
    "embedding_dimension",
    "vector_count",
    "normalization",
}


def _faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "FAISS is missing; install with pip install -e \".[semantic]\""
        ) from exc
    return faiss


def _require_normalized(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} must be L2-normalized")


@dataclass(frozen=True)
class FaissHit:
    score: float
    metadata: dict[str, Any]


class FaissIndex:
    def __init__(
        self,
        index,
        metadata: list[dict[str, Any]],
        manifest: dict[str, Any],
    ) -> None:
        self.index = index
        self.metadata = metadata
        self.manifest = manifest

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
        manifest: dict[str, Any] | None = None,
    ) -> "FaissIndex":
        values = np.ascontiguousarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("vectors must be a non-empty two-dimensional array")
        if len(metadata) != values.shape[0]:
            raise ValueError("metadata count must match vector count")
        _require_normalized(values, "vectors")
        overlap = CORE_MANIFEST_KEYS.intersection(manifest or {})
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"reserved manifest keys cannot be overridden: {names}")

        faiss = _faiss()
        index = faiss.IndexFlatIP(values.shape[1])
        index.add(values)
        details = {
            "schema_version": "1.0",
            "index_type": "IndexFlatIP",
            "embedding_dimension": int(values.shape[1]),
            "vector_count": int(values.shape[0]),
            "normalization": "l2",
        }
        if manifest:
            details.update(manifest)
        return cls(index, list(metadata), details)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[FaissHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        values = np.ascontiguousarray(query, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != 1:
            raise ValueError("query must have shape (1, embedding_dimension)")
        if values.shape[1] != self.index.d:
            raise ValueError("query embedding dimension does not match the index")
        _require_normalized(values, "query")

        count = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(values, count)
        return [
            FaissHit(float(score), self.metadata[int(index)])
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def save(self, output_root: Path) -> None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        _faiss().write_index(self.index, str(output_root / INDEX_FILENAME))
        (output_root / METADATA_FILENAME).write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_root / MANIFEST_FILENAME).write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, input_root: Path) -> "FaissIndex":
        input_root = Path(input_root)
        faiss = _faiss()
        index = faiss.read_index(str(input_root / INDEX_FILENAME))
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("persisted index must be an exact FAISS IndexFlatIP")
        metadata = json.loads(
            (input_root / METADATA_FILENAME).read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (input_root / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        expected = {
            "schema_version": "1.0",
            "index_type": "IndexFlatIP",
            "embedding_dimension": int(index.d),
            "vector_count": int(index.ntotal),
            "normalization": "l2",
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"persisted {key} does not match the loaded FAISS index"
                )
        if not isinstance(metadata, list) or not all(
            isinstance(row, dict) for row in metadata
        ):
            raise ValueError("persisted metadata must be a list of objects")
        if len(metadata) != index.ntotal:
            raise ValueError("persisted metadata count does not match the FAISS index")
        required_metadata = {"frame_id", "video_id", "timestamp", "image_path"}
        if any(not required_metadata.issubset(row) for row in metadata):
            raise ValueError("persisted metadata rows are missing required fields")
        return cls(index, metadata, manifest)
