"""Identity-safe, atomic writer for offline embedding provider artifacts."""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from .artifacts import EmbeddingManifest, sha256_file
from .provider_specs import ProviderSpec


_SAFE_SHARD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class OfflineEmbeddingError(ValueError):
    """Raised when offline extraction cannot publish a valid artifact."""


class ImageEmbeddingEncoder(Protocol):
    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray: ...


def run_selected_embedding_providers(provider_ids, runners) -> None:
    """Run exactly the requested provider callbacks, with no fallback."""
    for provider_id in provider_ids:
        try:
            runner = runners[provider_id]
        except KeyError as exc:
            raise ValueError(f"unknown or unavailable embedding provider: {provider_id}") from exc
        runner()


def load_canonical_keyframes(mapping_csv: Path) -> list[dict[str, str]]:
    with Path(mapping_csv).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = {"keyframe_id", "video_id", "image_path"}
        missing = sorted(required - fields)
        if missing:
            raise OfflineEmbeddingError(
                f"canonical mapping is missing columns: {', '.join(missing)}"
            )
        return [dict(row) for row in reader]


def _validate_inputs(rows: Sequence[Mapping[str, str]], keyframe_root: Path) -> None:
    # Keyed on (video_id, keyframe_id). A keyframe_id does not identify a
    # keyframe: the organizer's 166,550 keyframes carry 632 distinct ids,
    # one per video, so "034" is in 842 of them. On the id alone this raised
    # at the second video and no extraction could start at all.
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=2):
        keyframe_id = (row.get("keyframe_id") or "").strip()
        video_id = (row.get("video_id") or "").strip()
        image_path = (row.get("image_path") or "").strip()
        if not keyframe_id or not video_id or not image_path:
            raise OfflineEmbeddingError(f"mapping row {index} has an empty required field")
        if (video_id, keyframe_id) in seen:
            raise OfflineEmbeddingError(f"duplicate keyframe: {video_id}/{keyframe_id}")
        seen.add((video_id, keyframe_id))
        if not _SAFE_SHARD_ID.fullmatch(video_id):
            raise OfflineEmbeddingError(f"unsafe video_id for shard filename: {video_id}")
        resolved_root = keyframe_root.resolve()
        candidate = (keyframe_root / image_path).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise OfflineEmbeddingError(
                f"image_path escapes keyframe root for {keyframe_id}: {image_path}"
            ) from exc
        if not candidate.is_file():
            raise OfflineEmbeddingError(f"missing image for {keyframe_id}: {candidate}")


def _validated_vectors(vectors: np.ndarray, spec: ProviderSpec, expected: int) -> np.ndarray:
    array = np.asarray(vectors)
    expected_shape = (expected, spec.embedding_dimension)
    if array.shape != expected_shape:
        raise OfflineEmbeddingError(
            f"{spec.model_id} returned shape {array.shape}; expected {expected_shape}"
        )
    array = np.ascontiguousarray(array, dtype=np.dtype(spec.dtype))
    if not np.isfinite(array).all():
        if np.isnan(array).any():
            raise OfflineEmbeddingError(f"{spec.model_id} returned NaN vectors")
        raise OfflineEmbeddingError(f"{spec.model_id} returned Inf vectors")
    if spec.normalized:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise OfflineEmbeddingError(
                f"{spec.model_id} returned a zero or invalid vector norm"
            )
        array = np.ascontiguousarray(array / norms, dtype=np.dtype(spec.dtype))
    return array


def _write_mapping(path: Path, rows: Sequence[dict[str, object]]) -> None:
    columns = [
        "keyframe_id",
        "video_id",
        "feature_file",
        "feature_index",
        "frame_id",
        "shot_id",
        "timestamp",
        "image_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def extract_embedding_artifacts(
    rows: Sequence[Mapping[str, str]],
    *,
    keyframe_root: Path,
    embeddings_root: Path,
    spec: ProviderSpec,
    encoder: ImageEmbeddingEncoder,
    batch_size: int = 32,
    checkpoint_sha256: str | None = None,
) -> Path:
    """Extract one provider without exposing partial output as successful."""
    extraction_started = time.perf_counter()
    if spec.status != "ready":
        raise OfflineEmbeddingError(
            f"provider {spec.model_id} is blocked: {spec.blocker_reason}"
        )
    if batch_size <= 0:
        raise OfflineEmbeddingError("batch_size must be positive")
    rows = list(rows)
    if not rows:
        raise OfflineEmbeddingError("canonical mapping must not be empty")
    keyframe_root = Path(keyframe_root)
    _validate_inputs(rows, keyframe_root)

    target = Path(embeddings_root) / spec.model_id
    if target.exists():
        raise FileExistsError(f"provider artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{spec.model_id}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()

    try:
        grouped: OrderedDict[str, list[tuple[int, Mapping[str, str]]]] = OrderedDict()
        for source_index, row in enumerate(rows):
            grouped.setdefault(row["video_id"], []).append((source_index, row))

        output_rows: list[dict[str, object] | None] = [None] * len(rows)
        for video_id, group in grouped.items():
            image_paths = [keyframe_root / row["image_path"] for _, row in group]
            chunks = []
            for start in range(0, len(image_paths), batch_size):
                batch = image_paths[start : start + batch_size]
                chunks.append(np.asarray(encoder.encode_images(batch)))
            vectors = _validated_vectors(
                np.concatenate(chunks, axis=0), spec, expected=len(group)
            )
            np.save(staging / f"{video_id}.npy", vectors, allow_pickle=False)
            for feature_index, (source_index, row) in enumerate(group):
                output_rows[source_index] = {
                    "keyframe_id": row["keyframe_id"],
                    "video_id": video_id,
                    "feature_file": f"{video_id}.npy",
                    "feature_index": feature_index,
                    "frame_id": row.get("frame_id") or row.get("frame_idx") or "",
                    "shot_id": row.get("shot_id") or "",
                    "timestamp": row.get("timestamp") or "",
                    "image_path": row["image_path"],
                }

        mapping_path = staging / "feature_mapping.csv"
        _write_mapping(mapping_path, [row for row in output_rows if row is not None])
        manifest = EmbeddingManifest(
            schema_version="1.0",
            model_id=spec.model_id,
            model_name=spec.model_name,
            checkpoint_or_pretrained=spec.checkpoint_or_pretrained,
            model_revision=spec.model_revision,
            embedding_dimension=spec.embedding_dimension,
            dtype=spec.dtype,
            normalized=spec.normalized,
            mapping_path="feature_mapping.csv",
            mapping_sha256=sha256_file(mapping_path),
            keyframe_count=len(rows),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        manifest_payload = manifest.to_dict()
        manifest_payload.update(
            {
                "source_repository": spec.source_repository,
                "source_revision": spec.source_revision,
                "checkpoint_sha256": checkpoint_sha256 or spec.checkpoint_sha256,
                "execution": {
                    "adapter": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
                    "device_request": str(getattr(encoder, "device_request", "unknown")),
                    "batch_size": batch_size,
                    "latency_ms": (time.perf_counter() - extraction_started) * 1000.0,
                    "image_preprocessing": spec.image_preprocessing,
                    "vector_files_sha256": {
                        path.name: sha256_file(path)
                        for path in sorted(staging.glob("*.npy"))
                    },
                },
            }
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def audit_embedding_artifact(provider_root: Path) -> dict[str, int]:
    """Audit identity, shape, and finiteness without positional joins."""
    from .artifacts import load_mapping, load_provider_artifact

    artifact = load_provider_artifact(
        provider_root, verify_checksum=True, verify_mapping=True
    )
    records = load_mapping(artifact)
    unique_ids = {record.keyframe_id for record in records}
    invalid = 0
    for record in records:
        vector_file = artifact.resolve_vector_path(record.feature_file)
        try:
            matrix = np.load(vector_file, mmap_mode="r", allow_pickle=False)
            vector = matrix[record.feature_index]
        except (OSError, ValueError, IndexError):
            invalid += 1
            continue
        if vector.shape != (artifact.manifest.embedding_dimension,):
            invalid += 1
        elif not np.isfinite(vector).all():
            invalid += 1
        elif artifact.manifest.normalized and not math.isclose(
            float(np.linalg.norm(vector)), 1.0, rel_tol=1e-5, abs_tol=1e-5
        ):
            invalid += 1
    expected = artifact.manifest.keyframe_count
    return {
        "expected_keyframes": expected,
        "vectors": len(records),
        "unique_ids": len(unique_ids),
        "missing_ids": max(expected - len(unique_ids), 0),
        "extra_ids": max(len(unique_ids) - expected, 0),
        "invalid_vectors": invalid,
    }


def reuse_embedding_artifact(
    source_root: Path,
    *,
    embeddings_root: Path,
    rows: Sequence[Mapping[str, str]],
    spec: ProviderSpec,
) -> Path:
    """Publish a validated selector-produced provider artifact without encoding."""
    from .artifacts import load_mapping, load_provider_artifact, sha256_file

    source = load_provider_artifact(
        source_root, verify_checksum=True, verify_mapping=True
    )
    manifest = source.manifest
    expected_contract = (
        spec.model_id,
        spec.model_name,
        spec.checkpoint_or_pretrained,
        spec.model_revision,
        spec.embedding_dimension,
        spec.dtype,
        spec.normalized,
    )
    actual_contract = (
        manifest.model_id,
        manifest.model_name,
        manifest.checkpoint_or_pretrained,
        manifest.model_revision,
        manifest.embedding_dimension,
        manifest.dtype,
        manifest.normalized,
    )
    if actual_contract != expected_contract:
        raise OfflineEmbeddingError(
            "selector MetaCLIP artifact does not match the downstream provider identity"
        )
    records = load_mapping(source)
    expected_ids = {
        (
            (row.get("video_id") or "").strip(),
            (row.get("keyframe_id") or "").strip(),
            str(row.get("frame_idx") or row.get("frame_id") or "").strip(),
        )
        for row in rows
    }
    actual_ids = {
        (record.video_id, record.keyframe_id, str(record.frame_id or "").strip())
        for record in records
    }
    if expected_ids != actual_ids or len(expected_ids) != len(rows):
        raise OfflineEmbeddingError(
            "selector MetaCLIP artifact candidate identity does not match canonical mapping"
        )
    audit = audit_embedding_artifact(source_root)
    if audit["invalid_vectors"] or audit["missing_ids"] or audit["extra_ids"]:
        raise OfflineEmbeddingError(f"selector MetaCLIP artifact failed audit: {audit}")
    try:
        payload = json.loads((Path(source_root) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineEmbeddingError("cannot read selector MetaCLIP manifest") from exc
    execution = payload.get("execution")
    if not isinstance(execution, dict) or execution.get("image_preprocessing") != spec.image_preprocessing:
        raise OfflineEmbeddingError(
            "selector MetaCLIP preprocessing does not match downstream provider"
        )
    hashes = execution.get("vector_files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise OfflineEmbeddingError("selector MetaCLIP artifact lacks vector hashes")
    for name, expected_hash in hashes.items():
        path = source.resolve_vector_path(name)
        if sha256_file(path) != expected_hash:
            raise OfflineEmbeddingError(
                f"selector MetaCLIP vector hash mismatch: {name}"
            )

    target = Path(embeddings_root) / spec.model_id
    if target.exists():
        raise FileExistsError(f"provider artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{spec.model_id}.{uuid.uuid4().hex}.reuse.tmp"
    try:
        shutil.copytree(source.root, staging)
        staging.replace(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def reuse_embedding_artifacts(
    source_roots: Sequence[Path],
    *,
    embeddings_root: Path,
    rows: Sequence[Mapping[str, str]],
    spec: ProviderSpec,
) -> Path:
    """Merge validated selector artifacts into one provider artifact.

    Canonical ingestion publishes one MetaCLIP-2 artifact per video.  Offline
    enrichment operates on a collection, so this function validates every
    source against the locked provider contract and atomically publishes one
    collection-level artifact without running MetaCLIP inference again.
    """
    from .artifacts import load_mapping, load_provider_artifact, sha256_file

    rows = list(rows)
    roots = [Path(root) for root in source_roots]
    if not rows:
        raise OfflineEmbeddingError("canonical mapping must not be empty")
    if not roots:
        raise OfflineEmbeddingError("selector artifact sources must not be empty")

    expected_contract = (
        spec.model_id,
        spec.model_name,
        spec.checkpoint_or_pretrained,
        spec.model_revision,
        spec.embedding_dimension,
        spec.dtype,
        spec.normalized,
    )
    expected_rows: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for row in rows:
        identity = (
            (row.get("video_id") or "").strip(),
            (row.get("keyframe_id") or "").strip(),
            str(row.get("frame_idx") or row.get("frame_id") or "").strip(),
        )
        if not all(identity):
            raise OfflineEmbeddingError(
                "canonical mapping has an empty selector reuse identity"
            )
        if identity in expected_rows:
            raise OfflineEmbeddingError(
                f"duplicate canonical selector identity: {identity[0]}/{identity[1]}"
            )
        expected_rows[identity] = row

    records_by_identity = {}
    vector_sources: dict[str, tuple[Path, str]] = {}
    source_manifests: list[dict[str, str | int]] = []
    for root in roots:
        source = load_provider_artifact(
            root, verify_checksum=True, verify_mapping=True
        )
        manifest = source.manifest
        actual_contract = (
            manifest.model_id,
            manifest.model_name,
            manifest.checkpoint_or_pretrained,
            manifest.model_revision,
            manifest.embedding_dimension,
            manifest.dtype,
            manifest.normalized,
        )
        if actual_contract != expected_contract:
            raise OfflineEmbeddingError(
                f"selector artifact contract mismatch: {root}"
            )
        try:
            payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OfflineEmbeddingError(
                f"cannot read selector MetaCLIP manifest: {root}"
            ) from exc
        execution = payload.get("execution")
        if (
            not isinstance(execution, dict)
            or execution.get("image_preprocessing") != spec.image_preprocessing
        ):
            raise OfflineEmbeddingError(
                f"selector MetaCLIP preprocessing mismatch: {root}"
            )
        hashes = execution.get("vector_files_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise OfflineEmbeddingError(
                f"selector MetaCLIP artifact lacks vector hashes: {root}"
            )

        records = load_mapping(source)
        referenced_files = {record.feature_file for record in records}
        if not referenced_files.issubset(hashes):
            raise OfflineEmbeddingError(
                f"selector MetaCLIP artifact lacks referenced vector hashes: {root}"
            )
        for filename in sorted(referenced_files):
            vector_path = source.resolve_vector_path(filename)
            actual_hash = sha256_file(vector_path)
            if actual_hash != hashes[filename]:
                raise OfflineEmbeddingError(
                    f"selector MetaCLIP vector hash mismatch: {vector_path}"
                )
            previous = vector_sources.get(filename)
            if previous is not None and previous != (vector_path, actual_hash):
                raise OfflineEmbeddingError(
                    f"selector artifacts collide on vector filename: {filename}"
                )
            vector_sources[filename] = (vector_path, actual_hash)

        for record in records:
            identity = (
                record.video_id,
                record.keyframe_id,
                str(record.frame_id or "").strip(),
            )
            if identity in records_by_identity:
                raise OfflineEmbeddingError(
                    "selector artifacts contain duplicate identity: "
                    f"{record.video_id}/{record.keyframe_id}"
                )
            records_by_identity[identity] = record
        source_manifests.append(
            {
                "root": str(root),
                "keyframe_count": manifest.keyframe_count,
                "manifest_sha256": sha256_file(root / "manifest.json"),
            }
        )

    expected_ids = set(expected_rows)
    actual_ids = set(records_by_identity)
    if expected_ids != actual_ids:
        missing = len(expected_ids - actual_ids)
        extra = len(actual_ids - expected_ids)
        raise OfflineEmbeddingError(
            "selector MetaCLIP artifacts do not match canonical mapping: "
            f"missing={missing}, extra={extra}"
        )

    target = Path(embeddings_root) / spec.model_id
    if target.exists():
        raise FileExistsError(f"provider artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{spec.model_id}.{uuid.uuid4().hex}.reuse.tmp"
    staging.mkdir()
    try:
        for filename, (source_path, _) in sorted(vector_sources.items()):
            destination = staging / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

        output_rows = []
        for identity, row in expected_rows.items():
            record = records_by_identity[identity]
            output_rows.append(
                {
                    "keyframe_id": record.keyframe_id,
                    "video_id": record.video_id,
                    "feature_file": record.feature_file,
                    "feature_index": record.feature_index,
                    "frame_id": row.get("frame_id") or row.get("frame_idx") or "",
                    "shot_id": row.get("shot_id") or "",
                    "timestamp": row.get("timestamp") or "",
                    "image_path": row.get("image_path") or "",
                }
            )
        mapping_path = staging / "feature_mapping.csv"
        _write_mapping(mapping_path, output_rows)
        manifest = EmbeddingManifest(
            schema_version="1.0",
            model_id=spec.model_id,
            model_name=spec.model_name,
            checkpoint_or_pretrained=spec.checkpoint_or_pretrained,
            model_revision=spec.model_revision,
            embedding_dimension=spec.embedding_dimension,
            dtype=spec.dtype,
            normalized=spec.normalized,
            mapping_path="feature_mapping.csv",
            mapping_sha256=sha256_file(mapping_path),
            keyframe_count=len(rows),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        payload = manifest.to_dict()
        payload.update(
            {
                "source_repository": spec.source_repository,
                "source_revision": spec.source_revision,
                "checkpoint_sha256": spec.checkpoint_sha256,
                "execution": {
                    "adapter": "reuse.selector_artifacts",
                    "image_preprocessing": spec.image_preprocessing,
                    "source_artifacts": source_manifests,
                    "vector_files_sha256": {
                        name: digest
                        for name, (_, digest) in sorted(vector_sources.items())
                    },
                },
            }
        )
        (staging / "manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target
