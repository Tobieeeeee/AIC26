"""Resumable enrichment-only execution over canonical R2 keyframes.

The runner deliberately publishes to a run-scoped staging prefix.  It never
mutates ``data/corpus``: collection ``_SUCCESS.json`` markers and a run-level
``_READY.json`` make the output safe to validate and promote separately.
"""
from __future__ import annotations

import csv
import gc
import json
import logging
import os
import shutil
import subprocess
import tarfile
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from mvr.enrichment.embeddings.artifacts import (
    load_mapping,
    load_provider_artifact,
    sha256_file,
)
from mvr.enrichment.embeddings.offline import (
    audit_embedding_artifact,
    extract_embedding_artifacts,
    reuse_embedding_artifacts,
)
from mvr.enrichment.embeddings.provider_specs import ProviderSpec, ProviderSpecRegistry
from mvr.enrichment.feature_extraction import extract_objects
from mvr.operations.consolidate import consolidate_to_parquet
from mvr.operations.r2_layout import raw_video_key, staging_key
from mvr.operations.r2_sync import R2Sync, SyncStats


LOGGER = logging.getLogger(__name__)
CANONICAL_PROVIDERS = ("beit3", "metaclip2", "perception_encoder")
CANONICAL_FEATURES = ("objects", *CANONICAL_PROVIDERS, "ocr", "asr")
DATASET_MANIFEST_KEY = "data/corpus/dataset_manifest.json"
CANONICAL_MAPPING_KEY = "data/corpus/ingest/keyframe_mapping.csv"
CANONICAL_SHOTS_KEY = "data/corpus/ingest/shots.parquet"


class R2EnrichmentError(RuntimeError):
    """Raised when enrichment cannot preserve corpus or artifact contracts."""


@dataclass(frozen=True)
class CollectionWork:
    collection_id: str
    source_run_id: str
    rows: tuple[dict[str, str], ...]
    mapping_csv: Path
    artifact_root: Path
    selector_roots: tuple[Path, ...]

    @property
    def video_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row["video_id"] for row in self.rows))


@dataclass(frozen=True)
class EnrichmentRunConfig:
    run_id: str
    batch_id: str
    collections: tuple[str, ...]
    work_root: Path
    ocr_model_lock_sha256: str
    keyframe_download_workers: int = 24
    raw_video_download_workers: int = 4
    upload_workers: int = 12
    embedding_batch_size: int = 8
    asr_sample_rate: int = 16000
    asr_pre_seconds: float = 3.0
    asr_post_seconds: float = 3.0
    asr_merge_gap_ms: int = 250
    evict_after_push: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.collections:
            raise ValueError("collections must be non-empty")
        if any(
            value < 1
            for value in (
                self.keyframe_download_workers,
                self.raw_video_download_workers,
                self.upload_workers,
                self.embedding_batch_size,
                self.asr_sample_rate,
            )
        ):
            raise ValueError(
                "worker, batch-size, and sample-rate values must be positive"
            )
        if self.asr_pre_seconds < 0 or self.asr_post_seconds < 0:
            raise ValueError("ASR window sizes must be non-negative")
        if self.asr_merge_gap_ms < 0:
            raise ValueError("ASR merge gap must be non-negative")
        digest = self.ocr_model_lock_sha256.lower()
        if len(digest) != 64 or any(
            value not in "0123456789abcdef" for value in digest
        ):
            raise ValueError("ocr_model_lock_sha256 must be a SHA-256 hex digest")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_batch_id(value: str) -> str:
    normalized = str(value).strip()
    if normalized.lower().startswith("batch"):
        normalized = normalized[5:]
    if not normalized:
        raise R2EnrichmentError("batch_id must be non-empty")
    return normalized


def _collection_sort(value: str) -> tuple[int, str]:
    suffix = value[1:] if value.startswith("L") else value
    return (int(suffix), value) if suffix.isdigit() else (10**9, value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R2EnrichmentError(f"cannot read JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise R2EnrichmentError(f"JSON root must be an object: {path}")
    return payload


def _load_mapping(path: Path) -> list[dict[str, str]]:
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {
                "keyframe_id",
                "video_id",
                "collection_id",
                "batch_id",
                "shot_id",
                "ordinal",
                "frame_idx",
                "timestamp",
                "image_path",
                "video_object_key",
            }
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise R2EnrichmentError(
                    "canonical mapping is missing columns: " + ", ".join(missing)
                )
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise R2EnrichmentError(f"cannot read canonical mapping: {path}") from exc
    if not rows:
        raise R2EnrichmentError("canonical mapping is empty")
    return rows


def _write_mapping(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _source_run_by_collection(dataset_manifest: Mapping[str, Any]) -> dict[str, str]:
    raw = dataset_manifest.get("collections")
    if not isinstance(raw, dict):
        raise R2EnrichmentError("dataset manifest has no collection registry")
    result: dict[str, str] = {}
    for collection, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            result[str(collection)] = run_id.strip()
    return result


def prepare_collection_work(
    *,
    dataset_manifest: Mapping[str, Any],
    mapping_rows: Sequence[Mapping[str, str]],
    collections: Sequence[str],
    batch_id: str,
    work_root: Path,
    video_ids: set[str] | None = None,
) -> tuple[CollectionWork, ...]:
    """Validate canonical identity and materialize per-collection plans."""
    batch = _canonical_batch_id(batch_id)
    requested = tuple(sorted(dict.fromkeys(collections), key=_collection_sort))
    if not requested:
        raise R2EnrichmentError("at least one collection is required")
    run_ids = _source_run_by_collection(dataset_manifest)
    missing_runs = [collection for collection in requested if collection not in run_ids]
    if missing_runs:
        raise R2EnrichmentError(
            "dataset manifest has no authoritative ingestion run for: "
            + ",".join(missing_runs)
        )

    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict(
        (collection, []) for collection in requested
    )
    seen: set[tuple[str, str]] = set()
    for source in mapping_rows:
        row = {str(key): str(value) for key, value in source.items()}
        collection = row.get("collection_id", "").strip()
        if collection not in grouped:
            continue
        if video_ids is not None and row.get("video_id") not in video_ids:
            continue
        row_batch = _canonical_batch_id(row.get("batch_id", ""))
        if row_batch != batch:
            raise R2EnrichmentError(
                f"mapping batch mismatch for {row.get('keyframe_id')}: {row_batch} != {batch}"
            )
        video_id = row.get("video_id", "").strip()
        keyframe_id = row.get("keyframe_id", "").strip()
        identity = (video_id, keyframe_id)
        if not all(identity) or identity in seen:
            raise R2EnrichmentError(
                f"invalid or duplicate canonical identity: {video_id}/{keyframe_id}"
            )
        seen.add(identity)
        image_key = PurePosixPath(row.get("image_path", "").replace("\\", "/"))
        expected_prefix = PurePosixPath(
            "data", "corpus", "keyframes", collection, video_id
        )
        if image_key.is_absolute() or ".." in image_key.parts:
            raise R2EnrichmentError(f"unsafe canonical image key: {image_key}")
        if image_key.parts[: len(expected_prefix.parts)] != expected_prefix.parts:
            raise R2EnrichmentError(
                f"canonical image key is outside its collection/video: {image_key}"
            )
        row["image_path"] = image_key.as_posix()
        video_key = PurePosixPath(row.get("video_object_key", "").replace("\\", "/"))
        expected_video_key = raw_video_key(batch, collection, video_id)
        if (
            video_key.is_absolute()
            or ".." in video_key.parts
            or video_key.as_posix() != expected_video_key
        ):
            raise R2EnrichmentError(
                f"canonical raw-video key mismatch for {video_id}: "
                f"{video_key.as_posix()} != {expected_video_key}"
            )
        row["video_object_key"] = video_key.as_posix()
        grouped[collection].append(row)

    plans: list[CollectionWork] = []
    selector_base = Path(work_root) / "selector_sources"
    for collection, rows in grouped.items():
        if not rows:
            raise R2EnrichmentError(f"canonical mapping has no rows for {collection}")
        rows.sort(
            key=lambda row: (
                row["video_id"],
                int(row["shot_id"]),
                int(row["ordinal"]),
                int(row["frame_idx"]),
                row["keyframe_id"],
            )
        )
        mapping_csv = (
            Path(work_root) / "mappings" / collection / "keyframe_mapping.csv"
        )
        _write_mapping(mapping_csv, rows)
        video_order = tuple(dict.fromkeys(row["video_id"] for row in rows))
        plans.append(
            CollectionWork(
                collection_id=collection,
                source_run_id=run_ids[collection],
                rows=tuple(rows),
                mapping_csv=mapping_csv,
                artifact_root=Path(work_root) / "artifacts" / collection,
                selector_roots=tuple(
                    selector_base / collection / video_id / "metaclip2"
                    for video_id in video_order
                ),
            )
        )
    if video_ids is not None:
        planned = {video for plan in plans for video in plan.video_ids}
        missing_videos = sorted(video_ids - planned)
        if missing_videos:
            raise R2EnrichmentError(
                "requested videos are absent from selected canonical collections: "
                + ",".join(missing_videos)
            )
    return tuple(plans)


def build_input_transfers(
    plans: Sequence[CollectionWork], *, local_r2_root: Path, batch_id: str
) -> list[tuple[str, Path]]:
    """Return exact canonical-keyframe and selector-artifact downloads."""
    batch = _canonical_batch_id(batch_id)
    transfers: list[tuple[str, Path]] = []
    for plan in plans:
        for row in plan.rows:
            key = row["image_path"]
            transfers.append((key, Path(local_r2_root) / Path(*PurePosixPath(key).parts)))
        for video_id, selector_root in zip(plan.video_ids, plan.selector_roots):
            source_prefix = staging_key(
                "ingestion",
                plan.source_run_id,
                f"batch={batch}",
                f"collection={plan.collection_id}",
                video_id,
                "selector_embeddings",
                "metaclip2",
            )
            for filename in (f"{video_id}.npy", "feature_mapping.csv", "manifest.json"):
                transfers.append(
                    (f"{source_prefix}/{filename}", selector_root / filename)
                )
    return transfers


def build_raw_video_transfers(
    plans: Sequence[CollectionWork], *, local_r2_root: Path
) -> list[tuple[str, Path]]:
    """Return one exact canonical raw-video download per selected video."""
    transfers: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for plan in plans:
        by_video: dict[str, str] = {}
        for row in plan.rows:
            video_id = row["video_id"]
            key = row["video_object_key"]
            previous = by_video.setdefault(video_id, key)
            if previous != key:
                raise R2EnrichmentError(
                    f"multiple raw-video keys for {video_id}: {previous}, {key}"
                )
        for video_id in plan.video_ids:
            key = by_video[video_id]
            if key in seen:
                continue
            seen.add(key)
            transfers.append(
                (key, Path(local_r2_root) / Path(*PurePosixPath(key).parts))
            )
    return transfers


def _write_collection_shot_manifests(
    shots_parquet: Path,
    plans: Sequence[CollectionWork],
) -> None:
    try:
        import polars as pl
    except ImportError as exc:
        raise R2EnrichmentError(
            "polars is required to materialize canonical OCR shot boundaries"
        ) from exc
    try:
        rows = pl.read_parquet(shots_parquet).to_dicts()
    except Exception as exc:
        raise R2EnrichmentError(f"cannot read canonical shots: {shots_parquet}") from exc
    by_collection: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_collection.setdefault(str(row.get("collection_id", "")), []).append(row)
    for plan in plans:
        video_ids = set(plan.video_ids)
        selected = [
            row
            for row in by_collection.get(plan.collection_id, [])
            if str(row.get("video_id")) in video_ids
        ]
        identities = {
            (str(row["video_id"]), int(row["shot_id"])) for row in selected
        }
        expected = {
            (row["video_id"], int(row["shot_id"])) for row in plan.rows
        }
        missing = sorted(expected - identities)
        if missing:
            raise R2EnrichmentError(
                f"canonical shots missing {len(missing)} OCR boundaries for "
                f"{plan.collection_id}; first={missing[0]}"
            )
        _atomic_json(
            plan.mapping_csv.with_name("shots.json"),
            {"schema_version": "1.0", "shots": selected},
        )


def _write_ocr_job_shots(
    target: Path,
    plans: Sequence[CollectionWork],
) -> None:
    shots: list[dict[str, Any]] = []
    for plan in plans:
        payload = _read_json(plan.mapping_csv.with_name("shots.json"))
        values = payload.get("shots")
        if not isinstance(values, list):
            raise R2EnrichmentError(
                f"invalid collection shot manifest for {plan.collection_id}"
            )
        shots.extend(item for item in values if isinstance(item, dict))
    _atomic_json(target, {"schema_version": "1.0", "shots": shots})


def _provider_artifact_is_complete(
    root: Path, spec: ProviderSpec, rows: Sequence[Mapping[str, str]]
) -> bool:
    if not (root / "manifest.json").is_file():
        return False
    artifact = load_provider_artifact(root, verify_checksum=True, verify_mapping=True)
    contract = (
        artifact.manifest.model_id,
        artifact.manifest.model_name,
        artifact.manifest.checkpoint_or_pretrained,
        artifact.manifest.model_revision,
        artifact.manifest.embedding_dimension,
        artifact.manifest.dtype,
        artifact.manifest.normalized,
    )
    expected_contract = (
        spec.model_id,
        spec.model_name,
        spec.checkpoint_or_pretrained,
        spec.model_revision,
        spec.embedding_dimension,
        spec.dtype,
        spec.normalized,
    )
    if contract != expected_contract or artifact.manifest.keyframe_count != len(rows):
        raise R2EnrichmentError(f"existing provider artifact contract mismatch: {root}")
    actual_ids = {(row.video_id, row.keyframe_id) for row in load_mapping(artifact)}
    expected_ids = {(row["video_id"], row["keyframe_id"]) for row in rows}
    audit = audit_embedding_artifact(root)
    if actual_ids != expected_ids or any(
        audit[key] for key in ("missing_ids", "extra_ids", "invalid_vectors")
    ):
        raise R2EnrichmentError(f"existing provider artifact failed audit: {root}")
    return True


def _archive_tree(source: Path, target: Path) -> Path:
    if target.is_file() and target.stat().st_size > 0:
        return target
    if not source.is_dir():
        raise R2EnrichmentError(f"artifact directory is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with tarfile.open(temporary, mode="w:gz", compresslevel=1) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(source.parent).as_posix())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _text_artifact_is_complete(
    root: Path,
    rows: Sequence[Mapping[str, str]],
    feature: str,
    *,
    expected_asr_config: Mapping[str, int | float] | None = None,
) -> bool:
    """Validate the resumable OCR/ASR per-keyframe transport contract."""
    if feature not in {"ocr", "asr"}:
        raise ValueError(f"unsupported text feature: {feature}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    if feature == "asr" and expected_asr_config is not None:
        audio = manifest.get("audio", {})
        vad = manifest.get("vad", {})
        actual = {
            "sample_rate": audio.get("sample_rate"),
            "pre_seconds": audio.get("pre_seconds"),
            "post_seconds": audio.get("post_seconds"),
            "merge_gap_ms": vad.get("merge_gap_ms"),
        }
        if actual != dict(expected_asr_config):
            raise R2EnrichmentError(
                f"ASR artifact configuration mismatch under {root}: "
                f"expected={dict(expected_asr_config)} actual={actual}"
            )
    expected = {(row["video_id"], row["keyframe_id"]): row for row in rows}
    paths = sorted(root.glob("*/*.json"))
    actual = {(path.parent.name, path.stem): path for path in paths}
    if set(actual) != set(expected):
        raise R2EnrichmentError(
            f"{feature} artifact identity mismatch under {root}: "
            f"expected={len(expected)} actual={len(actual)}"
        )
    if feature == "asr" and (root / "errors.json").is_file():
        errors = _read_json(root / "errors.json").get("errors", [])
        if errors:
            return False
    for identity, row in expected.items():
        payload = _read_json(actual[identity])
        if payload.get("video_id") != row["video_id"]:
            raise R2EnrichmentError(f"{feature} video identity mismatch: {actual[identity]}")
        if payload.get("keyframe_id") != row["keyframe_id"]:
            raise R2EnrichmentError(
                f"{feature} keyframe identity mismatch: {actual[identity]}"
            )
        if feature == "ocr":
            if payload.get("shot_id") != int(row["shot_id"]):
                raise R2EnrichmentError(f"OCR shot mismatch: {actual[identity]}")
            if payload.get("anchor_frame_idx") != int(row["frame_idx"]):
                raise R2EnrichmentError(f"OCR frame mismatch: {actual[identity]}")
            if abs(float(payload.get("anchor_timestamp", -1)) - float(row["timestamp"])) > 1e-6:
                raise R2EnrichmentError(f"OCR timestamp mismatch: {actual[identity]}")
            if not isinstance(payload.get("texts"), list):
                raise R2EnrichmentError(f"OCR texts must be a list: {actual[identity]}")
        else:
            if payload.get("frame_idx") != int(row["frame_idx"]):
                raise R2EnrichmentError(f"ASR frame mismatch: {actual[identity]}")
            if abs(float(payload.get("timestamp", -1)) - float(row["timestamp"])) > 1e-6:
                raise R2EnrichmentError(f"ASR timestamp mismatch: {actual[identity]}")
            if payload.get("shot_id") != int(row["shot_id"]):
                raise R2EnrichmentError(f"ASR shot mismatch: {actual[identity]}")
            if not isinstance(payload.get("text"), str) or not isinstance(
                payload.get("segments"), list
            ):
                raise R2EnrichmentError(f"invalid ASR artifact: {actual[identity]}")
    return True


def _write_text_metadata_staging(
    source_root: Path,
    staging_root: Path,
    rows: Sequence[Mapping[str, str]],
    feature: str,
) -> None:
    """Project canonical OCR/ASR envelopes into one metadata row per keyframe."""
    for row in rows:
        source = source_root / row["video_id"] / f"{row['keyframe_id']}.json"
        payload = _read_json(source)
        if feature == "ocr":
            texts = [
                item
                for item in payload.get("texts", [])
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            confidences = [
                float(item["conf"])
                for item in texts
                if isinstance(item.get("conf"), (int, float))
            ]
            record: dict[str, Any] = {
                "keyframe_id": row["keyframe_id"],
                "video_id": row["video_id"],
                "frame_idx": int(row["frame_idx"]),
                "text": " ".join(str(item["text"]).strip() for item in texts),
                "confidence": (
                    sum(confidences) / len(confidences) if confidences else None
                ),
            }
        elif feature == "asr":
            record = {
                "keyframe_id": row["keyframe_id"],
                "video_id": row["video_id"],
                "frame_idx": int(row["frame_idx"]),
                "timestamp": float(row["timestamp"]),
                "text": str(payload.get("text", "")),
                "shot_id": int(row["shot_id"]),
            }
        else:
            raise ValueError(f"unsupported text feature: {feature}")
        _atomic_json(
            staging_root / row["video_id"] / f"{row['keyframe_id']}.json",
            record,
        )


def _probe_duration(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise R2EnrichmentError(f"cannot probe duration for {video_path}: {exc}") from exc
    if duration <= 0:
        raise R2EnrichmentError(f"non-positive duration for {video_path}: {duration}")
    return duration


def _prepare_asr_dataset(
    plan: CollectionWork,
    *,
    local_r2_root: Path,
    dataset_root: Path,
    duration_resolver: Callable[[Path], float],
) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    _write_mapping(dataset_root / "keyframe_mapping.csv", plan.rows)
    metadata_path = dataset_root / "video_metadata.csv"
    existing_ids: set[str] = set()
    if metadata_path.is_file():
        try:
            with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
                existing = list(csv.DictReader(stream))
            if all(
                Path(row["video_path"]).is_file()
                and float(row["duration_seconds"]) > 0
                for row in existing
            ):
                existing_ids = {row["video_id"] for row in existing}
        except (OSError, KeyError, ValueError):
            existing_ids = set()
    if existing_ids == set(plan.video_ids):
        return dataset_root

    by_video = {row["video_id"]: row["video_object_key"] for row in plan.rows}
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("video_id", "video_path", "duration_seconds"),
            lineterminator="\n",
        )
        writer.writeheader()
        for video_id in plan.video_ids:
            video_path = (
                Path(local_r2_root)
                / Path(*PurePosixPath(by_video[video_id]).parts)
            ).resolve()
            if not video_path.is_file():
                raise R2EnrichmentError(f"raw video is missing: {video_path}")
            writer.writerow(
                {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "duration_seconds": duration_resolver(video_path),
                }
            )
    os.replace(temporary, metadata_path)
    return dataset_root


def _materialize_ocr_collection(
    shared_root: Path,
    plan: CollectionWork,
    global_manifest: Mapping[str, Any],
) -> Path:
    target = plan.artifact_root / "ocr"
    target.mkdir(parents=True, exist_ok=True)
    canonical_text_count = 0
    for video_id in plan.video_ids:
        source_video = shared_root / video_id
        if not source_video.is_dir():
            raise R2EnrichmentError(f"OCR output missing video directory: {source_video}")
        shutil.copytree(source_video, target / video_id, dirs_exist_ok=True)
        for path in source_video.glob("*.json"):
            canonical_text_count += len(_read_json(path).get("texts", []))
    manifest = dict(global_manifest)
    manifest.update(
        {
            "collection_id": plan.collection_id,
            "mapping_source": str(plan.mapping_csv),
            "keyframe_count": len(plan.rows),
            "json_file_count": len(plan.rows),
            "canonical_text_count": canonical_text_count,
        }
    )
    _atomic_json(target / "manifest.json", manifest)
    return target


def _free_accelerator(value: Any = None) -> None:
    del value
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class R2EnrichmentRunner:
    """Run all six canonical enrichment branches into run-scoped R2 staging."""

    def __init__(
        self,
        *,
        sync: R2Sync,
        config: EnrichmentRunConfig,
        provider_specs: ProviderSpecRegistry,
        detector_factory: Callable[[], Any],
        encoder_factory: Callable[[str, ProviderSpec], Any],
        ocr_runner: Callable[[Path, Path, Path, Path], Mapping[str, Any]],
        asr_stage_factory: Callable[[], Callable[[Path, Path], Mapping[str, Any]]],
        duration_resolver: Callable[[Path], float] = _probe_duration,
        video_ids: set[str] | None = None,
    ) -> None:
        self.sync = sync
        self.config = config
        self.provider_specs = provider_specs
        self.detector_factory = detector_factory
        self.encoder_factory = encoder_factory
        self.ocr_runner = ocr_runner
        self.asr_stage_factory = asr_stage_factory
        self.duration_resolver = duration_resolver
        self.video_ids = video_ids
        self.work_root = Path(config.work_root).resolve()
        self.local_r2_root = self.work_root / "r2"
        self.state_path = self.work_root / "enrichment_state.json"
        self.report_path = self.work_root / "enrichment_report.json"
        self.state: dict[str, Any] = {}
        self.timings: dict[str, float] = {}

    def _asr_config_identity(self) -> dict[str, int | float]:
        return {
            "sample_rate": self.config.asr_sample_rate,
            "pre_seconds": self.config.asr_pre_seconds,
            "post_seconds": self.config.asr_post_seconds,
            "merge_gap_ms": self.config.asr_merge_gap_ms,
        }

    def _load_state(self, mapping_sha256: str) -> None:
        fingerprint = {
            "run_id": self.config.run_id,
            "batch_id": _canonical_batch_id(self.config.batch_id),
            "collections": list(self.config.collections),
            "mapping_sha256": mapping_sha256,
            "providers": list(CANONICAL_PROVIDERS),
            "features": list(CANONICAL_FEATURES),
            "video_ids": sorted(self.video_ids) if self.video_ids is not None else None,
            "ocr_model_lock_sha256": self.config.ocr_model_lock_sha256,
            "asr": self._asr_config_identity(),
        }
        if self.state_path.is_file():
            state = _read_json(self.state_path)
            if state.get("fingerprint") != fingerprint:
                raise R2EnrichmentError(
                    "resume state does not match this run/mapping/configuration"
                )
            self.state = state
            return
        self.state = {
            "schema_version": "2.0",
            "fingerprint": fingerprint,
            "completed": {},
            "created_at": _utc_now(),
        }
        self._save_state()

    def _save_state(self) -> None:
        self.state["updated_at"] = _utc_now()
        _atomic_json(self.state_path, self.state)

    def _done(self, stage: str, collection: str) -> bool:
        return collection in self.state.get("completed", {}).get(stage, [])

    def _mark(self, stage: str, collection: str) -> None:
        completed = self.state.setdefault("completed", {}).setdefault(stage, [])
        if collection not in completed:
            completed.append(collection)
            completed.sort(key=_collection_sort)
            self._save_state()

    def _stage_prefix(self, collection: str) -> str:
        return staging_key(
            "enrichment",
            self.config.run_id,
            f"batch={_canonical_batch_id(self.config.batch_id)}",
            f"collection={collection}",
        )

    def _upload(
        self,
        plan: CollectionWork,
        stage: str,
        files: Sequence[tuple[Path, str]],
    ) -> SyncStats:
        if self._done(f"{stage}_uploaded", plan.collection_id):
            return SyncStats(skipped_up_to_date=len(files))
        prefix = self._stage_prefix(plan.collection_id)
        transfers = [(path, f"{prefix}/{relative}") for path, relative in files]
        stats = self.sync.push_files_parallel(
            transfers,
            workers=self.config.upload_workers,
            existing_prefix=prefix,
        )
        if stats.errors:
            raise R2EnrichmentError(
                f"R2 upload failed for {plan.collection_id}/{stage}: {stats.to_dict()}"
            )
        self._mark(f"{stage}_uploaded", plan.collection_id)
        return stats

    def _download_control_files(self) -> tuple[Path, Path, Path]:
        manifest = self.local_r2_root / Path(*PurePosixPath(DATASET_MANIFEST_KEY).parts)
        mapping = self.local_r2_root / Path(*PurePosixPath(CANONICAL_MAPPING_KEY).parts)
        shots = self.local_r2_root / Path(*PurePosixPath(CANONICAL_SHOTS_KEY).parts)
        stats = self.sync.pull_objects_parallel(
            [
                (DATASET_MANIFEST_KEY, manifest),
                (CANONICAL_MAPPING_KEY, mapping),
                (CANONICAL_SHOTS_KEY, shots),
            ],
            force=True,
            workers=3,
        )
        if stats.errors:
            raise R2EnrichmentError("failed to download canonical corpus control files")
        return manifest, mapping, shots

    def _run_objects(self, plans: Sequence[CollectionWork]) -> None:
        pending = [
            plan
            for plan in plans
            if not self._done("objects_uploaded", plan.collection_id)
            and not self._done("objects_extracted", plan.collection_id)
        ]
        if pending:
            LOGGER.info("loading RF-DETR once for %d collections", len(pending))
            detector = self.detector_factory()
            try:
                for plan in pending:
                    started = time.perf_counter()
                    extract_objects(
                        plan.collection_id,
                        list(plan.rows),
                        detector,
                        plan.artifact_root,
                        self.local_r2_root,
                        plan.mapping_csv,
                        resume=True,
                    )
                    self.timings[f"objects:{plan.collection_id}"] = (
                        time.perf_counter() - started
                    )
                    self._mark("objects_extracted", plan.collection_id)
            finally:
                del detector
                _free_accelerator()

        for plan in plans:
            if self._done("objects_uploaded", plan.collection_id):
                continue
            source = plan.artifact_root / "objects_staging"
            parquet = plan.artifact_root / "metadata" / "objects" / "part-000.parquet"
            if not parquet.is_file():
                consolidate_to_parquet(
                    source,
                    parquet,
                    "objects",
                    batch_id=_canonical_batch_id(self.config.batch_id),
                    collection_id=plan.collection_id,
                )
            archive = _archive_tree(
                plan.artifact_root / "objects",
                plan.artifact_root / "packages" / "objects.tar.gz",
            )
            self._upload(
                plan,
                "objects",
                (
                    (parquet, "metadata/objects/part-000.parquet"),
                    (archive, "task05a/objects.tar.gz"),
                ),
            )

    def _run_provider(self, plans: Sequence[CollectionWork], provider_id: str) -> None:
        spec = self.provider_specs.get(provider_id)
        pending_compute: list[CollectionWork] = []
        for plan in plans:
            if self._done(f"{provider_id}_uploaded", plan.collection_id):
                continue
            root = plan.artifact_root / "embeddings" / provider_id
            if _provider_artifact_is_complete(root, spec, plan.rows):
                self._mark(provider_id, plan.collection_id)
            elif self._done(provider_id, plan.collection_id):
                raise R2EnrichmentError(
                    f"state marks {provider_id} complete but artifact is missing: {root}"
                )
            else:
                pending_compute.append(plan)

        if pending_compute and provider_id == "metaclip2":
            for plan in pending_compute:
                started = time.perf_counter()
                reuse_embedding_artifacts(
                    plan.selector_roots,
                    embeddings_root=plan.artifact_root / "embeddings",
                    rows=plan.rows,
                    spec=spec,
                )
                self.timings[f"{provider_id}:{plan.collection_id}"] = (
                    time.perf_counter() - started
                )
                self._mark(provider_id, plan.collection_id)
        elif pending_compute:
            LOGGER.info(
                "loading %s once for %d collections", provider_id, len(pending_compute)
            )
            encoder = self.encoder_factory(provider_id, spec)
            try:
                for plan in pending_compute:
                    started = time.perf_counter()
                    extract_embedding_artifacts(
                        plan.rows,
                        keyframe_root=self.local_r2_root,
                        embeddings_root=plan.artifact_root / "embeddings",
                        spec=spec,
                        encoder=encoder,
                        batch_size=self.config.embedding_batch_size,
                    )
                    self.timings[f"{provider_id}:{plan.collection_id}"] = (
                        time.perf_counter() - started
                    )
                    self._mark(provider_id, plan.collection_id)
            finally:
                del encoder
                _free_accelerator()

        for plan in plans:
            if self._done(f"{provider_id}_uploaded", plan.collection_id):
                continue
            root = plan.artifact_root / "embeddings" / provider_id
            if not _provider_artifact_is_complete(root, spec, plan.rows):
                raise R2EnrichmentError(
                    f"provider artifact was not published locally: {root}"
                )
            files = [
                (path, f"embeddings/{provider_id}/{path.relative_to(root).as_posix()}")
                for path in sorted(root.rglob("*"))
                if path.is_file()
            ]
            self._upload(plan, provider_id, files)

    def _upload_text_feature(self, plans: Sequence[CollectionWork], feature: str) -> None:
        for plan in plans:
            if self._done(f"{feature}_uploaded", plan.collection_id):
                continue
            source = plan.artifact_root / feature
            if not _text_artifact_is_complete(
                source,
                plan.rows,
                feature,
                expected_asr_config=(
                    self._asr_config_identity() if feature == "asr" else None
                ),
            ):
                raise R2EnrichmentError(
                    f"{feature} artifact was not published locally: {source}"
                )
            staging = plan.artifact_root / f"{feature}_staging"
            _write_text_metadata_staging(source, staging, plan.rows, feature)
            parquet = (
                plan.artifact_root
                / "metadata"
                / feature
                / "part-000.parquet"
            )
            if not parquet.is_file():
                consolidate_to_parquet(
                    staging,
                    parquet,
                    feature,
                    batch_id=_canonical_batch_id(self.config.batch_id),
                    collection_id=plan.collection_id,
                    identity_columns=["video_id", "keyframe_id"],
                )
            archive = _archive_tree(
                source,
                plan.artifact_root / "packages" / f"{feature}.tar.gz",
            )
            self._upload(
                plan,
                feature,
                (
                    (parquet, f"metadata/{feature}/part-000.parquet"),
                    (archive, f"task05a/{feature}.tar.gz"),
                    (source / "manifest.json", f"{feature}/manifest.json"),
                ),
            )

    def _run_ocr(self, plans: Sequence[CollectionWork]) -> None:
        pending: list[CollectionWork] = []
        for plan in plans:
            if self._done("ocr_uploaded", plan.collection_id):
                continue
            root = plan.artifact_root / "ocr"
            if _text_artifact_is_complete(root, plan.rows, "ocr"):
                self._mark("ocr_extracted", plan.collection_id)
            elif self._done("ocr_extracted", plan.collection_id):
                raise R2EnrichmentError(
                    f"state marks OCR complete but artifact is missing: {root}"
                )
            else:
                pending.append(plan)

        if pending:
            all_rows = [row for plan in pending for row in plan.rows]
            plan_path = self.work_root / "ocr_jobs" / "pending_mapping.csv"
            _write_mapping(plan_path, all_rows)
            job_root = self.work_root / "ocr_jobs" / sha256_file(plan_path)[:24]
            mapping_path = job_root / "keyframe_mapping.csv"
            _write_mapping(mapping_path, all_rows)
            _write_ocr_job_shots(mapping_path.with_name("shots.json"), pending)
            shared_output = job_root / "output"
            temp_root = job_root / "runtime"
            started = time.perf_counter()
            LOGGER.info(
                "running one OCR Docker job for %d collections / %d keyframes",
                len(pending),
                len(all_rows),
            )
            global_manifest = self.ocr_runner(
                mapping_path,
                self.local_r2_root / "data" / "raw",
                shared_output,
                temp_root,
            )
            if not isinstance(global_manifest, Mapping):
                raise R2EnrichmentError("OCR runner returned a non-mapping manifest")
            for plan in pending:
                root = _materialize_ocr_collection(
                    shared_output,
                    plan,
                    global_manifest,
                )
                if not _text_artifact_is_complete(root, plan.rows, "ocr"):
                    raise R2EnrichmentError(
                        f"OCR collection artifact is incomplete: {root}"
                    )
            elapsed = time.perf_counter() - started
            self.timings["ocr:all_pending"] = elapsed
            for plan in pending:
                self._mark("ocr_extracted", plan.collection_id)

        self._upload_text_feature(plans, "ocr")
        if self.config.evict_after_push and all(
            self._done("ocr_uploaded", plan.collection_id) for plan in plans
        ):
            job_root = self.work_root / "ocr_jobs"
            if job_root.is_dir():
                shutil.rmtree(job_root)

    def _run_asr(self, plans: Sequence[CollectionWork]) -> None:
        pending: list[CollectionWork] = []
        for plan in plans:
            if self._done("asr_uploaded", plan.collection_id):
                continue
            root = plan.artifact_root / "asr"
            if _text_artifact_is_complete(
                root,
                plan.rows,
                "asr",
                expected_asr_config=self._asr_config_identity(),
            ):
                self._mark("asr_extracted", plan.collection_id)
            elif self._done("asr_extracted", plan.collection_id):
                raise R2EnrichmentError(
                    f"state marks ASR complete but artifact is missing: {root}"
                )
            else:
                pending.append(plan)

        inputs = {
            plan.collection_id: _prepare_asr_dataset(
                plan,
                local_r2_root=self.local_r2_root,
                dataset_root=self.work_root / "asr_inputs" / plan.collection_id,
                duration_resolver=self.duration_resolver,
            )
            for plan in pending
        }
        if pending:
            LOGGER.info("loading ASR/VAD once for %d collections", len(pending))
            stage = self.asr_stage_factory()
            try:
                for plan in pending:
                    started = time.perf_counter()
                    manifest = stage(
                        inputs[plan.collection_id],
                        plan.artifact_root / "asr",
                    )
                    if not isinstance(manifest, Mapping):
                        raise R2EnrichmentError(
                            "ASR stage returned a non-mapping manifest"
                        )
                    if int(manifest.get("counts", {}).get("keyframes_failed", 0)):
                        raise R2EnrichmentError(
                            f"ASR reported failed keyframes for {plan.collection_id}"
                        )
                    if not _text_artifact_is_complete(
                        plan.artifact_root / "asr",
                        plan.rows,
                        "asr",
                        expected_asr_config=self._asr_config_identity(),
                    ):
                        raise R2EnrichmentError(
                            f"ASR collection artifact is incomplete: {plan.artifact_root / 'asr'}"
                        )
                    self.timings[f"asr:{plan.collection_id}"] = (
                        time.perf_counter() - started
                    )
                    self._mark("asr_extracted", plan.collection_id)
            finally:
                close = getattr(stage, "close", None)
                if callable(close):
                    close()
                del stage
                _free_accelerator()

        self._upload_text_feature(plans, "asr")
        if self.config.evict_after_push:
            for plan in plans:
                if self._done("asr_uploaded", plan.collection_id):
                    target = self.work_root / "asr_inputs" / plan.collection_id
                    if target.is_dir():
                        shutil.rmtree(target)
            if all(self._done("asr_uploaded", plan.collection_id) for plan in plans):
                cache_root = self.work_root / "asr_audio_cache"
                if cache_root.is_dir():
                    shutil.rmtree(cache_root)

    def _finalize_collection(self, plan: CollectionWork) -> None:
        if self._done("collection", plan.collection_id):
            return
        required = CANONICAL_FEATURES
        incomplete = [
            stage
            for stage in required
            if not self._done(f"{stage}_uploaded", plan.collection_id)
        ]
        if incomplete:
            raise R2EnrichmentError(
                f"cannot mark {plan.collection_id} complete; missing={incomplete}"
            )
        report = {
            "schema_version": "2.0",
            "status": "DONE",
            "run_id": self.config.run_id,
            "batch_id": _canonical_batch_id(self.config.batch_id),
            "collection_id": plan.collection_id,
            "source_ingestion_run_id": plan.source_run_id,
            "video_count": len(plan.video_ids),
            "keyframe_count": len(plan.rows),
            "providers": list(CANONICAL_PROVIDERS),
            "features": list(CANONICAL_FEATURES),
            "feature_configuration": {
                "ocr_model_lock_sha256": self.config.ocr_model_lock_sha256,
                "asr": self._asr_config_identity(),
            },
            "object_transport": "task05a/objects.tar.gz",
            "object_metadata": "metadata/objects/part-000.parquet",
            "ocr_transport": "task05a/ocr.tar.gz",
            "ocr_metadata": "metadata/ocr/part-000.parquet",
            "asr_transport": "task05a/asr.tar.gz",
            "asr_metadata": "metadata/asr/part-000.parquet",
            "canonical_mutation": False,
            "completed_at": _utc_now(),
        }
        local_report = plan.artifact_root / "collection_report.json"
        _atomic_json(local_report, report)
        self._upload(plan, "collection_report", ((local_report, "collection_report.json"),))
        marker = plan.artifact_root / "_SUCCESS.json"
        _atomic_json(marker, report)
        marker_key = f"{self._stage_prefix(plan.collection_id)}/_SUCCESS.json"
        if not self.sync.push_file(marker, marker_key, content_type="application/json"):
            raise R2EnrichmentError(
                f"failed to upload success marker for {plan.collection_id}"
            )
        self._mark("collection", plan.collection_id)
        if self.config.evict_after_push:
            self._evict_collection(plan)

    def _evict_collection(self, plan: CollectionWork) -> None:
        raw_paths = {
            self.local_r2_root / Path(*PurePosixPath(row["video_object_key"]).parts)
            for row in plan.rows
        }
        targets = (
            plan.artifact_root,
            self.local_r2_root / "data/corpus/keyframes" / plan.collection_id,
            *plan.selector_roots,
            *raw_paths,
        )
        for target in targets:
            resolved = target.resolve()
            try:
                resolved.relative_to(self.work_root)
            except ValueError as exc:
                raise R2EnrichmentError(
                    f"refusing to evict path outside work root: {resolved}"
                ) from exc
            if resolved.is_dir():
                shutil.rmtree(resolved)
            elif resolved.is_file():
                resolved.unlink()

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        manifest_path, mapping_path, shots_path = self._download_control_files()
        dataset_manifest = _read_json(manifest_path)
        mapping_rows = _load_mapping(mapping_path)
        self._load_state(sha256_file(mapping_path))
        plans = prepare_collection_work(
            dataset_manifest=dataset_manifest,
            mapping_rows=mapping_rows,
            collections=self.config.collections,
            batch_id=self.config.batch_id,
            work_root=self.work_root,
            video_ids=self.video_ids,
        )
        _write_collection_shot_manifests(shots_path, plans)

        if self.state.get("status") == "READY":
            LOGGER.info("run already READY: %s", self.config.run_id)
            return self.state

        visual_plans = [
            plan
            for plan in plans
            if any(
                not self._done(f"{feature}_uploaded", plan.collection_id)
                for feature in ("objects", *CANONICAL_PROVIDERS)
            )
        ]
        input_started = time.perf_counter()
        input_stats = self.sync.pull_objects_parallel(
            build_input_transfers(
                visual_plans,
                local_r2_root=self.local_r2_root,
                batch_id=self.config.batch_id,
            ),
            workers=self.config.keyframe_download_workers,
        )
        self.timings["download_inputs"] = time.perf_counter() - input_started
        if input_stats.errors:
            raise R2EnrichmentError(
                f"R2 input download failed: {input_stats.to_dict()}"
            )

        self._run_objects(plans)
        for provider_id in CANONICAL_PROVIDERS:
            self._run_provider(plans, provider_id)

        raw_plans = [
            plan
            for plan in plans
            if not self._done("ocr_uploaded", plan.collection_id)
            or not self._done("asr_uploaded", plan.collection_id)
        ]
        raw_started = time.perf_counter()
        raw_input_stats = self.sync.pull_objects_parallel(
            build_raw_video_transfers(raw_plans, local_r2_root=self.local_r2_root),
            workers=self.config.raw_video_download_workers,
        )
        self.timings["download_raw_videos"] = time.perf_counter() - raw_started
        if raw_input_stats.errors:
            raise R2EnrichmentError(
                f"R2 raw-video download failed: {raw_input_stats.to_dict()}"
            )

        self._run_ocr(plans)
        self._run_asr(plans)
        for plan in plans:
            self._finalize_collection(plan)

        report = {
            "schema_version": "2.0",
            "status": "READY",
            "run_id": self.config.run_id,
            "batch_id": _canonical_batch_id(self.config.batch_id),
            "collections": [plan.collection_id for plan in plans],
            "video_count": sum(len(plan.video_ids) for plan in plans),
            "keyframe_count": sum(len(plan.rows) for plan in plans),
            "providers": list(CANONICAL_PROVIDERS),
            "features": list(CANONICAL_FEATURES),
            "feature_configuration": {
                "ocr_model_lock_sha256": self.config.ocr_model_lock_sha256,
                "asr": self._asr_config_identity(),
            },
            "visual_input_sync": input_stats.to_dict(),
            "raw_video_input_sync": raw_input_stats.to_dict(),
            "timings_seconds": {
                key: round(value, 3) for key, value in sorted(self.timings.items())
            },
            "total_seconds": round(time.perf_counter() - started, 3),
            "canonical_mutation": False,
            "promotion_required": True,
            "completed_at": _utc_now(),
        }
        _atomic_json(self.report_path, report)
        run_prefix = staging_key("enrichment", self.config.run_id)
        if not self.sync.push_file(
            self.report_path,
            f"{run_prefix}/enrichment_report.json",
            content_type="application/json",
        ):
            raise R2EnrichmentError("failed to upload enrichment report")
        ready_path = self.work_root / "_READY.json"
        _atomic_json(ready_path, report)
        if not self.sync.push_file(
            ready_path,
            f"{run_prefix}/_READY.json",
            content_type="application/json",
        ):
            raise R2EnrichmentError("failed to upload enrichment ready marker")
        self.state["status"] = "READY"
        self.state["report"] = report
        self._save_state()
        return report
