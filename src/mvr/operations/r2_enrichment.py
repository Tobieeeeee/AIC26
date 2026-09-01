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
from mvr.operations.r2_layout import staging_key
from mvr.operations.r2_sync import R2Sync, SyncStats


LOGGER = logging.getLogger(__name__)
CANONICAL_PROVIDERS = ("beit3", "metaclip2", "perception_encoder")
DATASET_MANIFEST_KEY = "data/corpus/dataset_manifest.json"
CANONICAL_MAPPING_KEY = "data/corpus/ingest/keyframe_mapping.csv"


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
    keyframe_download_workers: int = 24
    upload_workers: int = 12
    embedding_batch_size: int = 8
    evict_after_push: bool = False


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
        mapping_csv = Path(work_root) / "mappings" / f"{collection}.csv"
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


def _archive_objects(source: Path, target: Path) -> Path:
    if target.is_file() and target.stat().st_size > 0:
        return target
    if not source.is_dir():
        raise R2EnrichmentError(f"object artifact directory is missing: {source}")
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
    """Run RF-DETR and three canonical embedding artifacts from R2 keyframes."""

    def __init__(
        self,
        *,
        sync: R2Sync,
        config: EnrichmentRunConfig,
        provider_specs: ProviderSpecRegistry,
        detector_factory: Callable[[], Any],
        encoder_factory: Callable[[str, ProviderSpec], Any],
        video_ids: set[str] | None = None,
    ) -> None:
        self.sync = sync
        self.config = config
        self.provider_specs = provider_specs
        self.detector_factory = detector_factory
        self.encoder_factory = encoder_factory
        self.video_ids = video_ids
        self.work_root = Path(config.work_root).resolve()
        self.local_r2_root = self.work_root / "r2"
        self.state_path = self.work_root / "enrichment_state.json"
        self.report_path = self.work_root / "enrichment_report.json"
        self.state: dict[str, Any] = {}
        self.timings: dict[str, float] = {}

    def _load_state(self, mapping_sha256: str) -> None:
        fingerprint = {
            "run_id": self.config.run_id,
            "batch_id": _canonical_batch_id(self.config.batch_id),
            "collections": list(self.config.collections),
            "mapping_sha256": mapping_sha256,
            "providers": list(CANONICAL_PROVIDERS),
            "video_ids": sorted(self.video_ids) if self.video_ids is not None else None,
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
            "schema_version": "1.0",
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

    def _download_control_files(self) -> tuple[Path, Path]:
        manifest = self.local_r2_root / Path(*PurePosixPath(DATASET_MANIFEST_KEY).parts)
        mapping = self.local_r2_root / Path(*PurePosixPath(CANONICAL_MAPPING_KEY).parts)
        stats = self.sync.pull_objects_parallel(
            [(DATASET_MANIFEST_KEY, manifest), (CANONICAL_MAPPING_KEY, mapping)],
            force=True,
            workers=2,
        )
        if stats.errors:
            raise R2EnrichmentError("failed to download canonical corpus control files")
        return manifest, mapping

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
            archive = _archive_objects(
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

    def _finalize_collection(self, plan: CollectionWork) -> None:
        if self._done("collection", plan.collection_id):
            return
        required = ("objects", *CANONICAL_PROVIDERS)
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
            "schema_version": "1.0",
            "status": "DONE",
            "run_id": self.config.run_id,
            "batch_id": _canonical_batch_id(self.config.batch_id),
            "collection_id": plan.collection_id,
            "source_ingestion_run_id": plan.source_run_id,
            "video_count": len(plan.video_ids),
            "keyframe_count": len(plan.rows),
            "providers": list(CANONICAL_PROVIDERS),
            "object_transport": "task05a/objects.tar.gz",
            "object_metadata": "metadata/objects/part-000.parquet",
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
        for target in (plan.artifact_root, self.local_r2_root / "data/corpus/keyframes" / plan.collection_id):
            resolved = target.resolve()
            try:
                resolved.relative_to(self.work_root)
            except ValueError as exc:
                raise R2EnrichmentError(
                    f"refusing to evict path outside work root: {resolved}"
                ) from exc
            if resolved.is_dir():
                shutil.rmtree(resolved)

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        manifest_path, mapping_path = self._download_control_files()
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

        if self.state.get("status") == "READY":
            LOGGER.info("run already READY: %s", self.config.run_id)
            return self.state

        pending_plans = [
            plan for plan in plans if not self._done("collection", plan.collection_id)
        ]
        input_started = time.perf_counter()
        input_stats = self.sync.pull_objects_parallel(
            build_input_transfers(
                pending_plans,
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
        for plan in plans:
            self._finalize_collection(plan)

        report = {
            "schema_version": "1.0",
            "status": "READY",
            "run_id": self.config.run_id,
            "batch_id": _canonical_batch_id(self.config.batch_id),
            "collections": [plan.collection_id for plan in plans],
            "video_count": sum(len(plan.video_ids) for plan in plans),
            "keyframe_count": sum(len(plan.rows) for plan in plans),
            "providers": list(CANONICAL_PROVIDERS),
            "input_sync": input_stats.to_dict(),
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
