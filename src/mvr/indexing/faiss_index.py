"""Exact inner-product FAISS index adapted from Lab 4."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
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


class IndexMappingMismatch(RuntimeError):
    """The keyframe mapping is not the one this index was built from."""


# `E:\...`, `E:/...` or a `\\server\share` UNC root. Matched by pattern
# rather than by `Path.is_absolute()` because the deciding machine is
# usually not the one that wrote the path: an index built on Windows is
# read on Linux, where `Path(r"E:\a\b")` is one relative filename.
_WINDOWS_ROOT = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _pure(value: Any) -> PurePath:
    """The stored text read in the flavour it was written in."""
    text = str(value)
    return PureWindowsPath(text) if _WINDOWS_ROOT.match(text) else PurePosixPath(text)


def _is_absolute(value: Any) -> bool:
    return _pure(value).is_absolute()


def relative_image_path(image_path: Any, image_root: Any) -> str:
    """`image_path` as text relative to `image_root`, forward-slashed.

    Refuses a path outside the root instead of falling back to the
    absolute one: a single unresolvable row in an otherwise portable
    index would only surface later, on someone else's machine.
    """
    relative_to_root = _pure(image_path)
    root = _pure(image_root)
    try:
        return relative_to_root.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"{relative_to_root} is outside the image root {root}"
        ) from exc


def resolve_image_path(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    image_root: Any | None = None,
) -> Path:
    """Where this row's keyframe image lives on the machine reading it.

    `image_root` overrides the root recorded at build time, which is what
    lets one index serve a corpus mounted somewhere else entirely.

    A stored path that is already absolute is handed back untouched --
    that is every index built before the root was recorded, and joining a
    root onto it would invent a path that never existed.
    """
    stored = row["image_path"]
    if _is_absolute(stored):
        return Path(str(stored))
    root = image_root if image_root is not None else manifest.get("image_root")
    return Path(str(root)) / str(stored) if root is not None else Path(str(stored))


def mapping_fingerprint(mapping_csv: Any) -> dict[str, Any]:
    """Identity of the keyframe mapping an index is built from.

    The checksum decides; the row count is what makes a mismatch
    actionable to a person reading the error.
    """
    path = Path(mapping_csv)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    return {"mapping_sha256": digest.hexdigest(), "mapping_row_count": row_count}


def verify_mapping_binding(manifest: Mapping[str, Any], mapping_csv: Any) -> None:
    """Raise unless this mapping is the one the index was built from.

    An index stores `frame_id`, never the competition `frame_idx`, so
    every submission row is resolved through this CSV. Pairing an index
    with a different one does not fail loudly today: the resolver finds
    nothing for the changed rows and drops them, so the run succeeds and
    the submission is silently short.

    A manifest with no recorded mapping only warns. Every index built so
    far lacks the binding, and raising would strand all of them.
    """
    path = Path(mapping_csv)
    if not path.is_file():
        raise FileNotFoundError(f"keyframe mapping not found: {path}")
    recorded = manifest.get("mapping_sha256")
    if not recorded:
        warnings.warn(
            f"index is not bound to a keyframe mapping, so {path.name} cannot be "
            "checked against the file it was built from",
            UserWarning,
            stacklevel=2,
        )
        return
    actual = mapping_fingerprint(path)
    if actual["mapping_sha256"] == recorded:
        return
    raise IndexMappingMismatch(
        f"{path} is not the keyframe mapping this index was built from "
        f"(built over {manifest.get('mapping_row_count', 'an unrecorded number of')} "
        f"rows, this file has {actual['mapping_row_count']})"
    )


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
        if details.get("image_root") is not None:
            # Builders hand this over as a Path, which json.dumps refuses --
            # and it would refuse it in save(), long after the expensive
            # encoding work was done.
            details["image_root"] = _pure(details["image_root"]).as_posix()
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
    def load(cls, input_root: Path, *, mapping_csv: Any | None = None) -> "FaissIndex":
        """Read a persisted index.

        Naming ``mapping_csv`` also checks that it is the mapping this
        index was built from. Callers that never resolve a competition
        frame -- the API, score_verified_set -- leave it out and are not
        warned about a file they do not use.
        """
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
        if mapping_csv is not None:
            verify_mapping_binding(manifest, mapping_csv)
        return cls(index, metadata, manifest)
