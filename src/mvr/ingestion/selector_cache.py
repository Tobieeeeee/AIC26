"""Validated candidate cache and production hybrid-selector orchestration."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mvr.enrichment.embeddings.artifacts import (
    EmbeddingManifest,
    load_mapping,
    load_provider_artifact,
    sha256_file,
)
from mvr.enrichment.embeddings.provider_specs import ProviderSpec

from .extract import extract_keyframes
from .hybrid_selector import (
    HYBRID_SELECTOR_MODE,
    HYBRID_SELECTOR_VERSION,
    CandidateSignal,
    HybridSelectorConfig,
    candidate_frame_indices,
    local_clip_novelty,
    normalize_embeddings,
    select_hybrid_frame_indices,
)
from .keyframes import make_keyframes_from_indices
from .models import Keyframe, Shot, VideoInfo


CACHE_SCHEMA_VERSION = "1.0"


class SelectorCacheError(RuntimeError):
    """Raised when a selector cache is absent, incomplete, or incompatible."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _source_fingerprint(video_path: Path) -> dict[str, object]:
    stat = video_path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(video_path),
    }


def _shot_fingerprint(shots: Sequence[Shot]) -> str:
    return _sha256_json(
        [
            {
                "video_id": shot.video_id,
                "method": shot.method,
                "shot_id": shot.shot_id,
                "start_frame": shot.start_frame,
                "end_frame": shot.end_frame,
                "fps": shot.fps,
            }
            for shot in shots
        ]
    )


def _candidate_keyframes(
    shots: Sequence[Shot], output_root: Path, config: HybridSelectorConfig
) -> list[Keyframe]:
    candidates: list[Keyframe] = []
    for shot in shots:
        for ordinal, frame_idx in enumerate(
            candidate_frame_indices(shot, config.candidate_step_sec), start=1
        ):
            candidate_id = (
                f"{shot.video_id}_S{shot.shot_id:04d}_C{ordinal:04d}"
            )
            image_path = (
                Path("selector_cache")
                / shot.video_id
                / "candidates"
                / f"{frame_idx:012d}.jpg"
            ).as_posix()
            candidates.append(
                Keyframe(
                    keyframe_id=candidate_id,
                    video_id=shot.video_id,
                    shot_id=shot.shot_id,
                    ordinal=ordinal,
                    frame_idx=frame_idx,
                    timestamp=frame_idx / shot.fps,
                    image_path=image_path,
                )
            )
    return candidates


def _write_candidate_mapping(path: Path, candidates: Sequence[Keyframe]) -> None:
    columns = [
        "keyframe_id",
        "video_id",
        "shot_id",
        "ordinal",
        "frame_idx",
        "timestamp",
        "image_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(item.to_dict() for item in candidates)


def _load_candidate_mapping(path: Path) -> list[Keyframe]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise SelectorCacheError(f"cannot read candidate mapping: {path}") from exc
    try:
        return [
            Keyframe(
                keyframe_id=row["keyframe_id"],
                video_id=row["video_id"],
                shot_id=int(row["shot_id"]),
                ordinal=int(row["ordinal"]),
                frame_idx=int(row["frame_idx"]),
                timestamp=float(row["timestamp"]),
                image_path=row["image_path"],
            )
            for row in rows
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectorCacheError(f"invalid candidate mapping: {path}") from exc


def _jpeg_aggregate_sha256(output_root: Path, candidates: Sequence[Keyframe]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        path = output_root / candidate.image_path
        if not path.is_file():
            raise SelectorCacheError(f"cached candidate JPEG is missing: {path}")
        digest.update(candidate.image_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def compute_edge_change_scores(
    image_paths: Sequence[Path], shot_ids: Sequence[int]
) -> np.ndarray:
    """Return deterministic grayscale/edge consecutive-change scores.

    Each image is resized to 320x180 grayscale.  The score is the equally
    weighted mean absolute grayscale difference and binary Canny-edge XOR
    fraction.  Both terms are normalized to [0, 1].  The first candidate in a
    shot has score zero.
    """
    if len(image_paths) != len(shot_ids):
        raise ValueError("image_paths and shot_ids must have equal length")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - base dependency
        raise SelectorCacheError("OpenCV is required for hybrid edge signals") from exc
    scores = np.zeros(len(image_paths), dtype=np.float32)
    previous_by_shot: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for position, (path, shot_id) in enumerate(zip(image_paths, shot_ids, strict=True)):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SelectorCacheError(f"cannot decode cached candidate JPEG: {path}")
        gray = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(gray, 100, 200)
        previous = previous_by_shot.get(shot_id)
        if previous is not None:
            previous_gray, previous_edges = previous
            gray_change = float(
                np.mean(np.abs(gray.astype(np.float32) - previous_gray)) / 255.0
            )
            edge_change = float(np.mean(edges != previous_edges))
            scores[position] = 0.5 * gray_change + 0.5 * edge_change
        previous_by_shot[shot_id] = (gray.astype(np.float32), edges)
    return scores


def _encode_in_batches(encoder: Any, paths: list[Path], batch_size: int) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("embedding batch size must be positive")
    chunks = [
        np.asarray(encoder.encode_images(paths[start : start + batch_size]))
        for start in range(0, len(paths), batch_size)
    ]
    return normalize_embeddings(np.concatenate(chunks, axis=0))


def _expected_identity(
    video: VideoInfo,
    shots: Sequence[Shot],
    config: HybridSelectorConfig,
    spec: ProviderSpec,
) -> dict[str, object]:
    return {
        "video_id": video.video_id,
        "source_fingerprint": _source_fingerprint(video.path),
        "shot_manifest_sha256": _shot_fingerprint(shots),
        "candidate_step_sec": config.candidate_step_sec,
        "metaclip": {
            "model_id": spec.model_id,
            "checkpoint_or_pretrained": spec.checkpoint_or_pretrained,
            "model_revision": spec.model_revision,
            "image_preprocessing": spec.image_preprocessing,
            "embedding_dimension": spec.embedding_dimension,
            "dtype": "float32",
            "normalized": True,
        },
    }


def validate_selector_cache_manifest(
    payload: dict[str, Any], expected_identity: dict[str, object]
) -> None:
    """Fail closed unless every cache-defining identity field matches."""
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise SelectorCacheError("selector cache schema_version mismatch")
    for field in (
        "video_id",
        "source_fingerprint",
        "shot_manifest_sha256",
        "candidate_step_sec",
        "metaclip",
    ):
        if payload.get(field) != expected_identity.get(field):
            raise SelectorCacheError(f"selector cache {field} mismatch")


def load_or_build_selector_cache(
    *,
    video: VideoInfo,
    shots: Sequence[Shot],
    output_root: Path,
    config: HybridSelectorConfig,
    spec: ProviderSpec,
    encoder: Any,
    embedding_batch_size: int,
) -> tuple[list[Keyframe], np.ndarray, np.ndarray, dict[str, Any]]:
    """Reuse a fully validated cache or build it once; never silently fallback."""
    config.validate()
    output_root = Path(output_root)
    cache_root = output_root / "selector_cache" / video.video_id
    manifest_path = cache_root / "manifest.json"
    mapping_path = cache_root / "candidate_mapping.csv"
    vectors_path = cache_root / "candidate_embeddings.npy"
    edge_path = cache_root / "edge_change_scores.npy"
    expected = _expected_identity(video, shots, config, spec)

    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SelectorCacheError(f"invalid selector cache manifest: {manifest_path}") from exc
        validate_selector_cache_manifest(payload, expected)
        candidates = _load_candidate_mapping(mapping_path)
        expected_candidates = _candidate_keyframes(shots, output_root, config)
        if [item.to_dict() for item in candidates] != [
            item.to_dict() for item in expected_candidates
        ]:
            raise SelectorCacheError(
                "selector cache candidate mapping does not match frozen shots/config"
            )
        artifacts = payload.get("artifacts", {})
        if sha256_file(mapping_path) != artifacts.get("candidate_mapping_sha256"):
            raise SelectorCacheError("selector cache candidate mapping hash mismatch")
        if _jpeg_aggregate_sha256(output_root, candidates) != artifacts.get(
            "candidate_jpegs_aggregate_sha256"
        ):
            raise SelectorCacheError("selector cache candidate JPEG hash mismatch")
        if sha256_file(vectors_path) != artifacts.get("candidate_embeddings_sha256"):
            raise SelectorCacheError("selector cache embedding hash mismatch")
        if sha256_file(edge_path) != artifacts.get("edge_change_scores_sha256"):
            raise SelectorCacheError("selector cache edge-score hash mismatch")
        vectors = np.load(vectors_path, allow_pickle=False)
        edges = np.load(edge_path, allow_pickle=False)
        if vectors.shape != (len(candidates), spec.embedding_dimension):
            raise SelectorCacheError("selector cache embedding shape mismatch")
        if edges.shape != (len(candidates),):
            raise SelectorCacheError("selector cache edge-score shape mismatch")
        vectors = normalize_embeddings(vectors)
        payload = dict(payload)
        payload["cache_reused"] = True
        return candidates, vectors, np.asarray(edges, dtype=np.float32), payload

    if cache_root.exists() and any(cache_root.iterdir()):
        raise SelectorCacheError(
            f"incomplete selector cache without manifest: {cache_root}; use a fresh output"
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_keyframes(shots, output_root, config)
    _write_candidate_mapping(mapping_path, candidates)

    extraction_start = time.perf_counter()
    extract_keyframes(video.path, candidates, output_root)
    extraction_seconds = time.perf_counter() - extraction_start
    image_paths = [output_root / item.image_path for item in candidates]

    embedding_start = time.perf_counter()
    vectors = _encode_in_batches(encoder, image_paths, embedding_batch_size)
    embedding_seconds = time.perf_counter() - embedding_start
    if vectors.shape[1] != spec.embedding_dimension:
        raise SelectorCacheError(
            f"MetaCLIP returned dimension {vectors.shape[1]}; expected {spec.embedding_dimension}"
        )
    np.save(vectors_path, vectors, allow_pickle=False)
    edges = compute_edge_change_scores(
        image_paths, [item.shot_id for item in candidates]
    )
    np.save(edge_path, edges, allow_pickle=False)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        **expected,
        "candidate_count": len(candidates),
        "candidate_mapping_sha256": sha256_file(mapping_path),
        "timing": {
            "candidate_extraction_seconds": extraction_seconds,
            "candidate_embedding_seconds": embedding_seconds,
        },
        "artifacts": {
            "candidate_mapping_sha256": sha256_file(mapping_path),
            "candidate_jpegs_aggregate_sha256": _jpeg_aggregate_sha256(
                output_root, candidates
            ),
            "candidate_embeddings_sha256": sha256_file(vectors_path),
            "edge_change_scores_sha256": sha256_file(edge_path),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    payload["cache_reused"] = False
    return candidates, vectors, edges, payload


def _write_selected_embedding_artifact(
    *,
    output_root: Path,
    keyframes: Sequence[Keyframe],
    vectors_by_identity: dict[tuple[str, int, int], np.ndarray],
    spec: ProviderSpec,
) -> Path:
    target = output_root / "selector_embeddings" / spec.model_id
    mapping_rows: list[dict[str, object]] = []
    existing_count = 0
    if target.exists():
        artifact = load_provider_artifact(
            target, verify_checksum=True, verify_mapping=True
        )
        manifest = artifact.manifest
        expected = (
            spec.model_id,
            spec.model_name,
            spec.checkpoint_or_pretrained,
            spec.model_revision,
            spec.embedding_dimension,
            "float32",
            True,
        )
        actual = (
            manifest.model_id,
            manifest.model_name,
            manifest.checkpoint_or_pretrained,
            manifest.model_revision,
            manifest.embedding_dimension,
            manifest.dtype,
            manifest.normalized,
        )
        if actual != expected:
            raise SelectorCacheError("selected MetaCLIP artifact identity mismatch")
        existing_records = load_mapping(artifact)
        incoming_videos = {item.video_id for item in keyframes}
        overlap = incoming_videos & {item.video_id for item in existing_records}
        if overlap:
            incoming_identity = {
                (item.video_id, item.keyframe_id, str(item.frame_idx)) for item in keyframes
            }
            existing_identity = {
                (item.video_id, item.keyframe_id, str(item.frame_id))
                for item in existing_records
                if item.video_id in incoming_videos
            }
            if overlap == incoming_videos and incoming_identity == existing_identity:
                return target
            raise SelectorCacheError(
                "selected MetaCLIP artifact contains conflicting video identity"
            )
        with artifact.mapping_path.open("r", encoding="utf-8", newline="") as stream:
            mapping_rows.extend(dict(row) for row in csv.DictReader(stream))
        existing_count = len(existing_records)
    else:
        target.mkdir(parents=True)
    grouped: dict[str, list[Keyframe]] = defaultdict(list)
    for frame in keyframes:
        grouped[frame.video_id].append(frame)
    vector_hashes: dict[str, str] = {
        path.name: sha256_file(path) for path in sorted(target.glob("*.npy"))
    }
    for video_id in sorted(grouped):
        frames = sorted(grouped[video_id], key=lambda item: (item.frame_idx, item.keyframe_id))
        matrix = np.stack(
            [vectors_by_identity[(item.video_id, item.shot_id, item.frame_idx)] for item in frames]
        ).astype(np.float32)
        vector_path = target / f"{video_id}.npy"
        np.save(vector_path, matrix, allow_pickle=False)
        vector_hashes[vector_path.name] = sha256_file(vector_path)
        for index, frame in enumerate(frames):
            mapping_rows.append(
                {
                    "keyframe_id": frame.keyframe_id,
                    "video_id": frame.video_id,
                    "feature_file": vector_path.name,
                    "feature_index": index,
                    "frame_id": frame.frame_idx,
                    "shot_id": frame.shot_id,
                    "timestamp": frame.timestamp,
                    "image_path": frame.image_path,
                }
            )
    mapping_path = target / "feature_mapping.csv"
    columns = [
        "keyframe_id", "video_id", "feature_file", "feature_index", "frame_id",
        "shot_id", "timestamp", "image_path",
    ]
    with mapping_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(mapping_rows)
    manifest = EmbeddingManifest(
        schema_version="1.0",
        model_id=spec.model_id,
        model_name=spec.model_name,
        checkpoint_or_pretrained=spec.checkpoint_or_pretrained,
        model_revision=spec.model_revision,
        embedding_dimension=spec.embedding_dimension,
        dtype="float32",
        normalized=True,
        mapping_path="feature_mapping.csv",
        mapping_sha256=sha256_file(mapping_path),
        keyframe_count=existing_count + len(keyframes),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    payload = manifest.to_dict()
    payload.update(
        {
            "source_repository": spec.source_repository,
            "source_revision": spec.source_revision,
            "execution": {
                "adapter": "mvr.ingestion.selector_cache",
                "reuse_source": "canonical hybrid selector candidate embeddings",
                "image_preprocessing": spec.image_preprocessing,
                "vector_files_sha256": vector_hashes,
            },
        }
    )
    (target / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target


def run_hybrid_selector(
    *,
    video: VideoInfo,
    shots: Sequence[Shot],
    output_root: Path,
    config: HybridSelectorConfig,
    spec: ProviderSpec,
    encoder: Any,
    embedding_batch_size: int = 16,
) -> tuple[list[Keyframe], dict[str, Any]]:
    """Run the canonical selector for one video and publish reuse artifacts."""
    candidates, vectors, edge_scores, cache_manifest = load_or_build_selector_cache(
        video=video,
        shots=shots,
        output_root=output_root,
        config=config,
        spec=spec,
        encoder=encoder,
        embedding_batch_size=embedding_batch_size,
    )
    positions_by_shot: dict[int, list[int]] = defaultdict(list)
    for position, candidate in enumerate(candidates):
        positions_by_shot[candidate.shot_id].append(position)
    selected_by_shot: dict[tuple[str, int], list[int]] = {}
    decisions = []
    selection_start = time.perf_counter()
    for shot in shots:
        positions = positions_by_shot[shot.shot_id]
        frame_indices = [candidates[position].frame_idx for position in positions]
        novelty = local_clip_novelty(
            frame_indices,
            vectors[positions],
            fps=shot.fps,
            neighbor_window_sec=config.clip_neighbor_window_sec,
            knn_k=config.clip_knn_k,
        )
        signals = [
            CandidateSignal(frame_idx, float(novelty[offset]), float(edge_scores[position]))
            for offset, (position, frame_idx) in enumerate(zip(positions, frame_indices, strict=True))
        ]
        result = select_hybrid_frame_indices(shot, signals, config)
        selected_by_shot[(shot.video_id, shot.shot_id)] = list(result.selected_indices)
        decisions.append(
            {
                "video_id": shot.video_id,
                "shot_id": shot.shot_id,
                "selected_frame_indices": list(result.selected_indices),
                "content_event_frame_indices": list(result.content_event_indices),
                "temporal_repair_frame_indices": list(result.temporal_repair_indices),
                "edge_event_threshold": result.edge_event_threshold,
                "max_observed_gap_sec": result.max_observed_gap_sec,
            }
        )
    selection_seconds = time.perf_counter() - selection_start
    keyframes = make_keyframes_from_indices(list(shots), selected_by_shot)
    candidate_by_identity = {
        (item.video_id, item.shot_id, item.frame_idx): (item, vectors[position])
        for position, item in enumerate(candidates)
    }
    vectors_by_identity: dict[tuple[str, int, int], np.ndarray] = {}
    for frame in keyframes:
        source, vector = candidate_by_identity[(frame.video_id, frame.shot_id, frame.frame_idx)]
        destination = output_root / frame.image_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_root / source.image_path, destination)
        vectors_by_identity[(frame.video_id, frame.shot_id, frame.frame_idx)] = vector
    artifact = _write_selected_embedding_artifact(
        output_root=output_root,
        keyframes=keyframes,
        vectors_by_identity=vectors_by_identity,
        spec=spec,
    )
    manifest = {
        "schema_version": "1.0",
        "selector_mode": HYBRID_SELECTOR_MODE,
        "selector_version": HYBRID_SELECTOR_VERSION,
        "config": config.to_dict(),
        "metaclip_model": spec.checkpoint_or_pretrained,
        "metaclip_revision": spec.model_revision,
        "candidate_count": len(candidates),
        "selected_keyframe_count": len(keyframes),
        "candidate_extraction_seconds": cache_manifest["timing"]["candidate_extraction_seconds"],
        "candidate_embedding_seconds": cache_manifest["timing"]["candidate_embedding_seconds"],
        "selector_only_seconds": selection_seconds,
        "content_event_selection_count": sum(len(row["content_event_frame_indices"]) for row in decisions),
        "temporal_repair_selection_count": sum(len(row["temporal_repair_frame_indices"]) for row in decisions),
        "max_observed_temporal_gap_sec": max(row["max_observed_gap_sec"] for row in decisions),
        "candidate_cache_manifest": str(
            Path("selector_cache") / video.video_id / "manifest.json"
        ),
        "selected_embedding_artifact": artifact.relative_to(output_root).as_posix(),
        "candidate_cache_reused": bool(cache_manifest.get("cache_reused")),
        "shots": decisions,
    }
    return keyframes, manifest
