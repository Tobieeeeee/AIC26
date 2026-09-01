"""Validate and consolidate run-scoped R2 ingestion into a promotion bundle.

Per-video ingestion workers intentionally publish only to
``data/staging/ingestion/<run_id>/...``.  This module is the single-writer
boundary that verifies those independent runs and prepares one corpus-level
mapping.  Preparing a bundle never mutates ``data/corpus``; activation is a
separate, explicit operation because changing the canonical keyframe identity
invalidates downstream embeddings and indexes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .r2_layout import raw_batch_segment, raw_video_key, raw_video_prefix, staging_key


LOGGER = logging.getLogger(__name__)

REQUIRED_METADATA = (
    "_SUCCESS.json",
    "ingest_manifest.json",
    "keyframe_mapping.csv",
    "selector_manifest.json",
    "shots.json",
    "video_metadata.csv",
)
KEYFRAME_COLUMNS = (
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
)
SHOT_COLUMNS = (
    "video_id",
    "collection_id",
    "batch_id",
    "method",
    "shot_id",
    "start_frame",
    "end_frame",
    "fps",
    "start_time",
    "end_time",
    "duration",
    "frame_count",
)
VIDEO_COLUMNS = (
    "video_id",
    "collection_id",
    "batch_id",
    "video_object_key",
    "duration_seconds",
    "fps",
    "frame_count",
    "width",
    "height",
)


class IngestionPromotionError(ValueError):
    """A source run or consolidated bundle violates the promotion contract."""


@dataclass(frozen=True)
class SourceAssignment:
    """One ingestion run and the collections for which it is authoritative."""

    run_id: str
    collections: tuple[str, ...]


@dataclass(frozen=True)
class StagedVideo:
    """A complete per-video staging tree discovered on R2."""

    run_id: str
    batch_id: str
    collection_id: str
    video_id: str
    prefix: str
    objects: Mapping[str, Mapping[str, Any]]

    def key(self, relative: str) -> str:
        return f"{self.prefix}{relative}"


@dataclass(frozen=True)
class PromotionBundle:
    """Prepared local bundle and its validation report."""

    promotion_id: str
    root: Path
    report: Mapping[str, Any]


def parse_source_assignment(value: str) -> SourceAssignment:
    """Parse ``RUN_ID:L21,L22`` into a validated assignment."""

    run_id, separator, raw_collections = value.partition(":")
    run_id = run_id.strip()
    collections = tuple(
        part.strip() for part in raw_collections.split(",") if part.strip()
    )
    if not separator or not run_id or not collections:
        raise IngestionPromotionError(
            f"invalid source assignment {value!r}; expected RUN_ID:L21,L22"
        )
    if any("/" in part or "\\" in part for part in (run_id, *collections)):
        raise IngestionPromotionError("run IDs and collections must be path segments")
    return SourceAssignment(run_id=run_id, collections=collections)


def discover_staged_videos(
    client: Any,
    *,
    batch_id: str,
    sources: Sequence[SourceAssignment],
) -> tuple[list[StagedVideo], dict[str, Any]]:
    """Match every raw video to exactly one complete staging prefix.

    Raw videos are authoritative for membership.  A missing success marker,
    an extra staged video, overlapping collection assignments, or a missing
    required metadata artifact fails the entire audit.
    """

    if not sources:
        raise IngestionPromotionError("at least one source assignment is required")

    collection_owner: dict[str, str] = {}
    for source in sources:
        for collection in source.collections:
            previous = collection_owner.setdefault(collection, source.run_id)
            if previous != source.run_id:
                raise IngestionPromotionError(
                    f"collection {collection} is assigned to both {previous} and {source.run_id}"
                )

    staged: list[StagedVideo] = []
    collection_reports: dict[str, Any] = {}
    for collection in sorted(collection_owner, key=_collection_sort_key):
        run_id = collection_owner[collection]
        LOGGER.info("audit collection=%s run_id=%s", collection, run_id)
        raw_objects = client.list_objects(raw_video_prefix(batch_id, collection))
        raw_ids = {
            PurePosixPath(str(obj["Key"])).stem
            for obj in raw_objects
            if str(obj["Key"]).lower().endswith(".mp4")
        }
        if not raw_ids:
            raise IngestionPromotionError(
                f"raw collection {collection} contains no MP4 videos"
            )

        collection_prefix = staging_key(
            "ingestion", run_id, f"batch={batch_id}", f"collection={collection}"
        ).rstrip("/") + "/"
        stage_objects = client.list_objects(collection_prefix)
        by_video: dict[str, dict[str, Mapping[str, Any]]] = {}
        for obj in stage_objects:
            key = str(obj["Key"])
            relative = key.removeprefix(collection_prefix)
            parts = PurePosixPath(relative).parts
            if len(parts) < 2:
                continue
            video_id = parts[0]
            artifact = PurePosixPath(*parts[1:]).as_posix()
            by_video.setdefault(video_id, {})[artifact] = obj

        completed_ids = {
            video_id
            for video_id, objects in by_video.items()
            if int(objects.get("_SUCCESS.json", {}).get("Size", 0)) > 0
        }
        missing = sorted(raw_ids - completed_ids, key=_video_sort_key)
        extra = sorted(completed_ids - raw_ids, key=_video_sort_key)
        if missing or extra:
            raise IngestionPromotionError(
                f"{collection}/{run_id} is not complete: missing={missing} extra={extra}"
            )

        for video_id in sorted(raw_ids, key=_video_sort_key):
            objects = by_video[video_id]
            absent = [name for name in REQUIRED_METADATA if name not in objects]
            if absent:
                raise IngestionPromotionError(
                    f"{collection}/{video_id} missing required artifacts: {absent}"
                )
            prefix = f"{collection_prefix}{video_id}/"
            staged.append(
                StagedVideo(
                    run_id=run_id,
                    batch_id=str(batch_id),
                    collection_id=collection,
                    video_id=video_id,
                    prefix=prefix,
                    objects=objects,
                )
            )

        collection_reports[collection] = {
            "run_id": run_id,
            "raw_video_count": len(raw_ids),
            "successful_video_count": len(completed_ids),
            "staging_object_count": len(stage_objects),
        }
        LOGGER.info(
            "audit complete collection=%s videos=%d staging_objects=%d",
            collection,
            len(raw_ids),
            len(stage_objects),
        )

    report = {
        "batch_id": raw_batch_segment(batch_id),
        "video_count": len(staged),
        "collections": collection_reports,
    }
    return staged, report


def prepare_promotion_bundle(
    client: Any,
    *,
    batch_id: str,
    sources: Sequence[SourceAssignment],
    promotion_id: str,
    work_root: Path,
    workers: int = 10,
) -> PromotionBundle:
    """Download only metadata, validate it, and build one promotion bundle."""

    if not promotion_id.strip() or "/" in promotion_id or "\\" in promotion_id:
        raise IngestionPromotionError("promotion_id must be one non-empty path segment")
    if workers < 1:
        raise IngestionPromotionError("workers must be at least 1")

    staged, audit = discover_staged_videos(
        client, batch_id=str(batch_id), sources=sources
    )
    root = Path(work_root) / promotion_id
    metadata_root = root / "source_metadata"
    bundle_root = root / "bundle"
    metadata_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)

    downloads: list[tuple[str, Path, int]] = []
    for video in staged:
        video_root = metadata_root / video.collection_id / video.video_id
        for relative in REQUIRED_METADATA:
            obj = video.objects[relative]
            downloads.append(
                (video.key(relative), video_root / relative, int(obj.get("Size", 0)))
            )

    # Create the boto client before sharing the wrapper between worker threads.
    if hasattr(client, "check_credentials"):
        client.check_credentials()

    LOGGER.info(
        "materializing metadata files=%d videos=%d workers=%d",
        len(downloads),
        len(staged),
        workers,
    )

    def download_one(item: tuple[str, Path, int]) -> None:
        key, destination, expected_size = item
        if (
            destination.is_file()
            and destination.stat().st_size == expected_size
            and expected_size > 0
        ):
            return
        client.download_file(key, destination)
        if destination.stat().st_size != expected_size:
            raise IngestionPromotionError(
                f"downloaded size mismatch for {key}: "
                f"{destination.stat().st_size} != {expected_size}"
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(download_one, downloads))
    LOGGER.info("metadata materialization complete")

    keyframe_rows: list[dict[str, Any]] = []
    shot_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    copy_rows: list[dict[str, Any]] = []
    seen_keyframes: set[str] = set()
    selector_profile: dict[str, Any] | None = None
    legacy_marker_count = 0
    collection_counts: dict[str, dict[str, Any]] = {}

    for video_index, video in enumerate(staged, 1):
        local = metadata_root / video.collection_id / video.video_id
        marker = _read_json(local / "_SUCCESS.json")
        _validate_marker(marker, video)

        all_artifacts = {
            name: obj for name, obj in video.objects.items() if name != "_SUCCESS.json"
        }
        publishable = {
            name: obj
            for name, obj in all_artifacts.items()
            if PurePosixPath(name).parts[0] != "selector_cache"
        }
        marker_count = int(marker.get("artifact_count", -1))
        marker_bytes = int(marker.get("artifact_bytes", -1))
        publishable_bytes = sum(int(obj.get("Size", 0)) for obj in publishable.values())
        all_bytes = sum(int(obj.get("Size", 0)) for obj in all_artifacts.values())
        if (marker_count, marker_bytes) == (len(all_artifacts), all_bytes):
            legacy_marker_count += int(len(all_artifacts) != len(publishable))
        elif (marker_count, marker_bytes) != (len(publishable), publishable_bytes):
            raise IngestionPromotionError(
                f"{video.video_id} success marker count/bytes do not match its staging tree"
            )

        mapping_rows = _read_mapping(local / "keyframe_mapping.csv", video)
        ingest_manifest = _read_json(local / "ingest_manifest.json")
        shots_payload = _read_json(local / "shots.json")
        selector_manifest = _read_json(local / "selector_manifest.json")
        profile = _selector_profile(selector_manifest)
        if selector_profile is None:
            selector_profile = profile
        elif profile != selector_profile:
            raise IngestionPromotionError(
                f"{video.video_id} selector profile differs from earlier videos"
            )

        shots = shots_payload.get("shots")
        if not isinstance(shots, list):
            raise IngestionPromotionError(f"{video.video_id} shots.json has no shots list")
        if int(ingest_manifest.get("video_count", -1)) != 1:
            raise IngestionPromotionError(f"{video.video_id} ingest manifest video_count != 1")
        if int(ingest_manifest.get("shot_count", -1)) != len(shots):
            raise IngestionPromotionError(f"{video.video_id} ingest manifest shot count mismatch")
        if int(ingest_manifest.get("keyframe_count", -1)) != len(mapping_rows):
            raise IngestionPromotionError(
                f"{video.video_id} ingest manifest keyframe count mismatch"
            )
        if int(selector_manifest.get("selected_keyframe_count", -1)) != len(mapping_rows):
            raise IngestionPromotionError(
                f"{video.video_id} selector selected-keyframe count mismatch"
            )

        mapped_ids = {row["keyframe_id"] for row in mapping_rows}
        payload_keyframes = shots_payload.get("keyframes")
        if (
            not isinstance(payload_keyframes, list)
            or len(payload_keyframes) != len(mapping_rows)
            or {
                str(row.get("keyframe_id", "")) for row in payload_keyframes
            }
            != mapped_ids
        ):
            raise IngestionPromotionError(
                f"{video.video_id} shots.json keyframes differ from keyframe_mapping.csv"
            )

        collection_count = collection_counts.setdefault(
            video.collection_id,
            {"run_id": video.run_id, "video_count": 0, "shot_count": 0, "keyframe_count": 0},
        )
        collection_count["video_count"] += 1
        collection_count["shot_count"] += len(shots)
        collection_count["keyframe_count"] += len(mapping_rows)

        for row in mapping_rows:
            keyframe_id = row["keyframe_id"]
            if keyframe_id in seen_keyframes:
                raise IngestionPromotionError(
                    f"duplicate canonical keyframe identity: {keyframe_id}"
                )
            seen_keyframes.add(keyframe_id)
            source_relative = PurePosixPath(str(row["image_path"]).replace("\\", "/")).as_posix()
            expected_image_root = f"keyframes/{video.video_id}/"
            if not source_relative.startswith(expected_image_root):
                raise IngestionPromotionError(
                    f"{video.video_id}/{keyframe_id} has an invalid image path: "
                    f"{source_relative}"
                )
            source_obj = video.objects.get(source_relative)
            if source_obj is None or int(source_obj.get("Size", 0)) <= 0:
                raise IngestionPromotionError(
                    f"{video.video_id}/{keyframe_id} image is absent or empty: {source_relative}"
                )
            filename = PurePosixPath(source_relative).name
            destination = (
                f"data/corpus/keyframes/{video.collection_id}/{video.video_id}/{filename}"
            )
            canonical = {
                "keyframe_id": keyframe_id,
                "video_id": video.video_id,
                "collection_id": video.collection_id,
                "batch_id": raw_batch_segment(batch_id),
                "shot_id": int(row["shot_id"]),
                "ordinal": int(row["ordinal"]),
                "frame_idx": int(row["frame_idx"]),
                "timestamp": float(row["timestamp"]),
                "image_path": destination,
                "video_object_key": raw_video_key(
                    batch_id, video.collection_id, video.video_id
                ),
            }
            if canonical["frame_idx"] < 0 or canonical["timestamp"] < 0:
                raise IngestionPromotionError(
                    f"{keyframe_id} has a negative frame or timestamp"
                )
            keyframe_rows.append(canonical)
            copy_rows.append(
                {
                    "source": video.key(source_relative),
                    "destination": destination,
                    "size": int(source_obj.get("Size", 0)),
                    "etag": str(source_obj.get("ETag", "")).strip('"'),
                }
            )

        for shot in shots:
            if str(shot.get("video_id")) != video.video_id:
                raise IngestionPromotionError(
                    f"{video.video_id} shots.json contains another video identity"
                )
            shot_rows.append(
                {
                    "video_id": video.video_id,
                    "collection_id": video.collection_id,
                    "batch_id": raw_batch_segment(batch_id),
                    "method": str(shot["method"]),
                    "shot_id": int(shot["shot_id"]),
                    "start_frame": int(shot["start_frame"]),
                    "end_frame": int(shot["end_frame"]),
                    "fps": float(shot["fps"]),
                    "start_time": float(shot["start_time"]),
                    "end_time": float(shot["end_time"]),
                    "duration": float(shot["duration"]),
                    "frame_count": int(shot["frame_count"]),
                }
            )

        with (local / "video_metadata.csv").open(newline="", encoding="utf-8") as handle:
            metadata_rows = list(csv.DictReader(handle))
        if len(metadata_rows) != 1 or metadata_rows[0].get("video_id") != video.video_id:
            raise IngestionPromotionError(
                f"{video.video_id} video_metadata.csv must contain exactly its own row"
            )
        metadata = metadata_rows[0]
        video_rows.append(
            {
                "video_id": video.video_id,
                "collection_id": video.collection_id,
                "batch_id": raw_batch_segment(batch_id),
                "video_object_key": raw_video_key(
                    batch_id, video.collection_id, video.video_id
                ),
                "duration_seconds": float(metadata["duration_seconds"]),
                "fps": float(metadata["fps"]),
                "frame_count": int(metadata["frame_count"]),
                "width": int(metadata["width"]),
                "height": int(metadata["height"]),
            }
        )
        if video_index % 50 == 0 or video_index == len(staged):
            LOGGER.info(
                "validated videos=%d/%d keyframes=%d",
                video_index,
                len(staged),
                len(keyframe_rows),
            )

    keyframe_rows.sort(key=_canonical_row_sort_key)
    shot_rows.sort(
        key=lambda row: (
            _collection_sort_key(row["collection_id"]),
            _video_sort_key(row["video_id"]),
            row["shot_id"],
        )
    )
    video_rows.sort(
        key=lambda row: (
            _collection_sort_key(row["collection_id"]),
            _video_sort_key(row["video_id"]),
        )
    )
    copy_rows.sort(key=lambda row: row["destination"])

    ingest_root = bundle_root / "ingest"
    manifest_root = ingest_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    _write_csv(ingest_root / "keyframe_mapping.csv", KEYFRAME_COLUMNS, keyframe_rows)
    _write_parquet(ingest_root / "shots.parquet", shot_rows, SHOT_COLUMNS)
    _write_parquet(ingest_root / "video_metadata.parquet", video_rows, VIDEO_COLUMNS)
    _write_json_lines(bundle_root / "keyframe_copy_manifest.jsonl", copy_rows)

    selector_profile = selector_profile or {}
    batch_manifest = {
        "schema_version": "1.0",
        "batch_id": raw_batch_segment(batch_id),
        "promotion_id": promotion_id,
        "status": "STAGED_VALIDATED",
        "collections": {
            key: collection_counts[key]
            for key in sorted(collection_counts, key=_collection_sort_key)
        },
    }
    ingest_manifest = {
        "schema_version": "1.0",
        "batch_id": raw_batch_segment(batch_id),
        "promotion_id": promotion_id,
        "status": "STAGED_VALIDATED",
        "source_runs": [
            {"run_id": source.run_id, "collections": list(source.collections)}
            for source in sources
        ],
        "video_count": len(video_rows),
        "shot_count": len(shot_rows),
        "keyframe_count": len(keyframe_rows),
        "selector_profile": selector_profile,
    }
    _write_json(manifest_root / f"{raw_batch_segment(batch_id)}.json", batch_manifest)
    _write_json(ingest_root / "ingest_manifest.json", ingest_manifest)

    impact = _inspect_existing_corpus(
        client,
        work_root=root / "existing_corpus",
        collections=set(collection_counts),
        new_rows=keyframe_rows,
    )
    output_files = sorted(
        path
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name not in {"promotion_report.json", "_READY.json"}
    )
    checksums = {
        path.relative_to(bundle_root).as_posix(): {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in output_files
    }
    report = {
        "schema_version": "1.0",
        "promotion_id": promotion_id,
        "status": "PREPARED",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "audit": audit,
        "summary": {
            "video_count": len(video_rows),
            "shot_count": len(shot_rows),
            "keyframe_count": len(keyframe_rows),
            "legacy_success_marker_count": legacy_marker_count,
        },
        "selector_profile": selector_profile,
        "canonical_impact": impact,
        "safe_to_activate": not impact["activation_blockers"],
        "checksums": checksums,
    }
    _write_json(bundle_root / "promotion_report.json", report)
    LOGGER.info(
        "prepared promotion_id=%s videos=%d shots=%d keyframes=%d safe_to_activate=%s",
        promotion_id,
        len(video_rows),
        len(shot_rows),
        len(keyframe_rows),
        report["safe_to_activate"],
    )
    return PromotionBundle(promotion_id=promotion_id, root=bundle_root, report=report)


def upload_prepared_bundle(client: Any, bundle: PromotionBundle) -> dict[str, Any]:
    """Upload a prepared bundle to staging, with ``_READY.json`` last."""

    prefix = staging_key("ingestion-consolidated", bundle.promotion_id).rstrip("/")
    files = sorted(
        path
        for path in bundle.root.rglob("*")
        if path.is_file() and path.name != "_READY.json"
    )
    uploaded = 0
    uploaded_bytes = 0
    for path in files:
        relative = path.relative_to(bundle.root).as_posix()
        key = f"{prefix}/{relative}"
        content_type = _content_type(path)
        client.upload_file(path, key, content_type=content_type)
        uploaded += 1
        uploaded_bytes += path.stat().st_size

    ready = bundle.root / "_READY.json"
    _write_json(
        ready,
        {
            "schema_version": "1.0",
            "status": "READY",
            "promotion_id": bundle.promotion_id,
            "artifact_count": uploaded,
            "artifact_bytes": uploaded_bytes,
            "safe_to_activate": bool(bundle.report["safe_to_activate"]),
        },
    )
    client.upload_file(ready, f"{prefix}/_READY.json", content_type="application/json")
    return {
        "prefix": f"{prefix}/",
        "uploaded": uploaded + 1,
        "bytes": uploaded_bytes + ready.stat().st_size,
    }


def _validate_marker(marker: Mapping[str, Any], video: StagedVideo) -> None:
    expected = {
        "schema_version": "1.0",
        "status": "DONE",
        "run_id": video.run_id,
        "batch_id": video.batch_id,
        "collection_id": video.collection_id,
        "video_id": video.video_id,
    }
    mismatches = {
        key: {"expected": value, "actual": marker.get(key)}
        for key, value in expected.items()
        if str(marker.get(key)) != str(value)
    }
    if mismatches:
        raise IngestionPromotionError(
            f"{video.video_id} has an invalid success marker: {mismatches}"
        )


def _read_mapping(path: Path, video: StagedVideo) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "keyframe_id", "video_id", "shot_id", "ordinal",
            "frame_idx", "timestamp", "image_path",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise IngestionPromotionError(
                f"{video.video_id} mapping missing columns: {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise IngestionPromotionError(f"{video.video_id} mapping is empty")
    if any(row.get("video_id") != video.video_id for row in rows):
        raise IngestionPromotionError(
            f"{video.video_id} mapping contains another video identity"
        )
    return rows


def _selector_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = ("selector_mode", "config", "metaclip_model", "metaclip_revision")
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise IngestionPromotionError(
            f"selector manifest missing profile fields: {missing}"
        )
    return {key: payload[key] for key in required}


def _inspect_existing_corpus(
    client: Any,
    *,
    work_root: Path,
    collections: set[str],
    new_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mapping_key = "data/corpus/ingest/keyframe_mapping.csv"
    dataset_key = "data/corpus/dataset_manifest.json"
    existing_rows: list[dict[str, str]] = []
    dataset_manifest: dict[str, Any] = {}
    work_root.mkdir(parents=True, exist_ok=True)

    if client.object_exists(mapping_key):
        mapping_path = work_root / "keyframe_mapping.csv"
        client.download_file(mapping_key, mapping_path)
        with mapping_path.open(newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
    if client.object_exists(dataset_key):
        manifest_path = work_root / "dataset_manifest.json"
        client.download_file(dataset_key, manifest_path)
        dataset_manifest = _read_json(manifest_path)

    new_by_id = {str(row["keyframe_id"]): row for row in new_rows}
    existing_by_id = {str(row.get("keyframe_id", "")): row for row in existing_rows}
    overlapping_ids = set(existing_by_id) & set(new_by_id)
    changed_overlaps = 0
    for keyframe_id in overlapping_ids:
        old = existing_by_id[keyframe_id]
        new = new_by_id[keyframe_id]
        try:
            same = (
                int(old.get("frame_idx", -1)) == int(new["frame_idx"])
                and abs(float(old.get("timestamp", -1)) - float(new["timestamp"])) < 1e-6
            )
        except (TypeError, ValueError):
            same = False
        changed_overlaps += int(not same)

    manifest_collections = dataset_manifest.get("collections", {})
    stale_collections: list[str] = []
    for collection in sorted(collections, key=_collection_sort_key):
        entry = (
            manifest_collections.get(collection, {})
            if isinstance(manifest_collections, dict)
            else {}
        )
        coverage = entry.get("coverage", {}) if isinstance(entry, dict) else {}
        if entry and (
            entry.get("embeddings")
            or int(coverage.get("objects", 0) or 0) > 0
            or int(coverage.get("asr", 0) or 0) > 0
        ):
            stale_collections.append(collection)

    blockers: list[str] = []
    if stale_collections:
        blockers.append(
            "canonical downstream artifacts would become stale for: "
            + ",".join(stale_collections)
        )
    return {
        "existing_mapping_rows": len(existing_rows),
        "existing_mapping_videos": len(
            {row.get("video_id") for row in existing_rows if row.get("video_id")}
        ),
        "overlapping_keyframe_ids": len(overlapping_ids),
        "changed_overlapping_identities": changed_overlaps,
        "new_only_keyframe_ids": len(set(new_by_id) - set(existing_by_id)),
        "existing_only_keyframe_ids": len(set(existing_by_id) - set(new_by_id)),
        "stale_downstream_collections": stale_collections,
        "activation_blockers": blockers,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestionPromotionError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IngestionPromotionError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    if not rows:
        raise IngestionPromotionError(f"cannot write empty Parquet: {path}")
    try:
        import polars as pl
    except ImportError as exc:
        raise IngestionPromotionError("polars is required for ingestion consolidation") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).select(list(columns)).write_parquet(path)


def _write_json_lines(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_type(path: Path) -> str | None:
    return {
        ".csv": "text/csv",
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".parquet": "application/octet-stream",
    }.get(path.suffix.lower())


def _collection_sort_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"L(\d+)", str(value), flags=re.IGNORECASE)
    return (int(match.group(1)), str(value)) if match else (10**9, str(value))


def _video_sort_key(value: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"L(\d+)_V(\d+)", str(value), flags=re.IGNORECASE)
    return (
        (int(match.group(1)), int(match.group(2)), str(value))
        if match
        else (10**9, 10**9, str(value))
    )


def _canonical_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _collection_sort_key(str(row["collection_id"])),
        _video_sort_key(str(row["video_id"])),
        int(row["shot_id"]),
        int(row["ordinal"]),
        int(row["frame_idx"]),
        str(row["keyframe_id"]),
    )
