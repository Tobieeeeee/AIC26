"""Activate a validated ingestion promotion on the canonical R2 corpus.

Activation is deliberately separate from promotion preparation.  It verifies
the immutable candidate, snapshots the currently active ingestion metadata and
affected keyframes, copies the promoted keyframes server-side, and only then
switches the canonical manifests.  Superseded downstream artifacts are kept on
R2 but are no longer declared active by ``dataset_manifest.json``.
"""
from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .ingestion_promotion import IngestionPromotionError
from .r2_layout import staging_key


LOGGER = logging.getLogger(__name__)

CANONICAL_INGEST_PREFIX = "data/corpus/ingest/"
CANONICAL_KEYFRAME_PREFIX = "data/corpus/keyframes/"
DATASET_MANIFEST_KEY = "data/corpus/dataset_manifest.json"
BACKUP_ROOT = "data/staging/ingestion-promotion-backups"


@dataclass(frozen=True)
class ActivationCandidate:
    """A locally verified promotion bundle and its keyframe copy contract."""

    promotion_id: str
    root: Path
    report: Mapping[str, Any]
    ingest_manifest: Mapping[str, Any]
    batch_manifest: Mapping[str, Any]
    copy_rows: tuple[Mapping[str, Any], ...]


def activate_promotion(
    client: Any,
    *,
    bundle_root: Path,
    promotion_id: str,
    acknowledged_stale_collections: Sequence[str] = (),
    work_root: Path,
    workers: int = 10,
) -> dict[str, Any]:
    """Activate one prepared promotion with backup, verification, and cleanup.

    ``acknowledged_stale_collections`` is required when the preparation audit
    found downstream artifacts whose keyframe identity will be superseded.
    Unknown preparation blockers cannot be overridden by this operation.
    """

    if workers < 1:
        raise IngestionPromotionError("workers must be at least 1")
    candidate = load_activation_candidate(bundle_root, promotion_id=promotion_id)
    stale_collections = set(
        candidate.report.get("canonical_impact", {}).get(
            "stale_downstream_collections", []
        )
    )
    acknowledged = {str(item) for item in acknowledged_stale_collections}
    missing_acknowledgements = sorted(stale_collections - acknowledged)
    if missing_acknowledgements:
        raise IngestionPromotionError(
            "activation requires explicit stale-downstream acknowledgement for: "
            + ",".join(missing_acknowledgements)
        )
    unexpected_acknowledgements = sorted(acknowledged - stale_collections)
    if unexpected_acknowledgements:
        raise IngestionPromotionError(
            "stale-downstream acknowledgement was not reported by the candidate: "
            + ",".join(unexpected_acknowledgements)
        )
    blockers = list(
        candidate.report.get("canonical_impact", {}).get("activation_blockers", [])
    )
    stale_blocker_prefix = "canonical downstream artifacts would become stale for: "
    unknown_blockers = [
        str(blocker)
        for blocker in blockers
        if not str(blocker).startswith(stale_blocker_prefix)
    ]
    if unknown_blockers:
        raise IngestionPromotionError(
            f"candidate has activation blockers that cannot be acknowledged: {unknown_blockers}"
        )

    activation_root = Path(work_root) / promotion_id
    activation_root.mkdir(parents=True, exist_ok=True)
    _verify_remote_candidate(client, candidate, activation_root / "candidate")

    collections = tuple(
        sorted(candidate.batch_manifest["collections"], key=_collection_sort_key)
    )
    desired_by_key = {
        str(row["destination"]): row for row in candidate.copy_rows
    }
    if len(desired_by_key) != len(candidate.copy_rows):
        raise IngestionPromotionError("copy manifest contains duplicate destinations")

    backup = _ensure_backup(
        client,
        candidate=candidate,
        collections=collections,
        activation_root=activation_root,
        workers=workers,
    )
    LOGGER.info(
        "canonical backup ready prefix=%s objects=%d",
        backup["prefix"],
        backup["object_count"],
    )

    existing_before = _list_collection_keyframes(client, collections)
    stale_keys = sorted(set(existing_before) - set(desired_by_key))
    for key in stale_keys:
        if (
            not key.startswith(CANONICAL_KEYFRAME_PREFIX)
            or PurePosixPath(key).suffix.lower() not in {".jpg", ".jpeg", ".png"}
        ):
            raise IngestionPromotionError(f"refusing to delete unexpected key: {key}")
    copied, skipped = _copy_promoted_keyframes(
        client,
        copy_rows=candidate.copy_rows,
        existing=existing_before,
        workers=workers,
    )
    _verify_promoted_keyframes(client, desired_by_key, collections)

    activated_at = datetime.now(timezone.utc).isoformat()
    generated_root = activation_root / "canonical"
    generated = _write_activated_manifests(
        candidate,
        output_root=generated_root,
        activated_at=activated_at,
        backup_prefix=str(backup["prefix"]),
        stale_collections=stale_collections,
    )
    batch_id = str(candidate.batch_manifest["batch_id"])
    _upload_canonical_metadata(client, generated, batch_id=batch_id)

    # Exact-key cleanup happens only after backup and the new active pointer
    # exist. Every removed object remains recoverable from the backup prefix.
    for index, key in enumerate(stale_keys, 1):
        client.delete_object(key)
        if index % 1000 == 0 or index == len(stale_keys):
            LOGGER.info("removed stale canonical keyframes=%d/%d", index, len(stale_keys))

    final_objects = _verify_promoted_keyframes(client, desired_by_key, collections)
    unexpected_final = sorted(set(final_objects) - set(desired_by_key))
    if unexpected_final:
        raise IngestionPromotionError(
            f"canonical keyframe cleanup left {len(unexpected_final)} unexpected objects; "
            f"first={unexpected_final[0]}"
        )
    _verify_active_metadata(
        client,
        candidate,
        generated,
        activation_root / "final",
        batch_id=batch_id,
    )

    result = {
        "schema_version": "1.0",
        "status": "ACTIVATED",
        "promotion_id": promotion_id,
        "activated_at": activated_at,
        "backup_prefix": backup["prefix"],
        "collections": list(collections),
        "acknowledged_stale_collections": sorted(stale_collections),
        "keyframes": {
            "expected": len(candidate.copy_rows),
            "copied": copied,
            "skipped_identical": skipped,
            "removed_stale": len(stale_keys),
            "final": len(final_objects),
        },
        "summary": candidate.report["summary"],
        "downstream_status": "REBUILD_REQUIRED",
    }
    report_path = activation_root / "activation_report.json"
    _write_json(report_path, result)
    report_prefix = f"data/corpus/reports/ingestion/{promotion_id}"
    client.upload_file(
        report_path,
        f"{report_prefix}/activation_report.json",
        content_type="application/json",
    )
    client.upload_file(
        report_path,
        f"{report_prefix}/_ACTIVATED.json",
        content_type="application/json",
    )
    return result


def load_activation_candidate(
    bundle_root: Path, *, promotion_id: str
) -> ActivationCandidate:
    """Validate the local promotion bundle before allowing any remote write."""

    root = Path(bundle_root).resolve()
    if not root.is_dir():
        raise IngestionPromotionError(f"promotion bundle not found: {root}")
    report = _read_json(root / "promotion_report.json")
    if report.get("promotion_id") != promotion_id or report.get("status") != "PREPARED":
        raise IngestionPromotionError("promotion report identity/status is invalid")
    checksums = report.get("checksums")
    if not isinstance(checksums, dict) or not checksums:
        raise IngestionPromotionError("promotion report contains no checksums")
    for relative, expected in checksums.items():
        relative_path = PurePosixPath(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise IngestionPromotionError(f"unsafe bundle path: {relative}")
        path = root.joinpath(*relative_path.parts)
        if not path.is_file():
            raise IngestionPromotionError(f"bundle artifact is missing: {relative}")
        expected_size = int(expected.get("size", -1))
        expected_sha = str(expected.get("sha256", ""))
        if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha:
            raise IngestionPromotionError(f"bundle checksum mismatch: {relative}")

    ingest_manifest = _read_json(root / "ingest" / "ingest_manifest.json")
    batch_id = str(ingest_manifest.get("batch_id", ""))
    batch_manifest = _read_json(root / "ingest" / "manifests" / f"{batch_id}.json")
    for name, manifest in (
        ("ingest", ingest_manifest),
        ("batch", batch_manifest),
    ):
        if manifest.get("promotion_id") != promotion_id:
            raise IngestionPromotionError(f"{name} manifest promotion ID mismatch")
        if manifest.get("status") != "STAGED_VALIDATED":
            raise IngestionPromotionError(f"{name} manifest is not staged/validated")

    copy_rows = tuple(_read_copy_rows(root / "keyframe_copy_manifest.jsonl"))
    expected_keyframes = int(report.get("summary", {}).get("keyframe_count", -1))
    if len(copy_rows) != expected_keyframes:
        raise IngestionPromotionError(
            f"copy manifest count mismatch: {len(copy_rows)} != {expected_keyframes}"
        )
    if int(ingest_manifest.get("keyframe_count", -1)) != expected_keyframes:
        raise IngestionPromotionError("ingest manifest keyframe count mismatch")
    return ActivationCandidate(
        promotion_id=promotion_id,
        root=root,
        report=report,
        ingest_manifest=ingest_manifest,
        batch_manifest=batch_manifest,
        copy_rows=copy_rows,
    )


def _verify_remote_candidate(
    client: Any, candidate: ActivationCandidate, work_root: Path
) -> None:
    prefix = staging_key(
        "ingestion-consolidated", candidate.promotion_id
    ).rstrip("/")
    remote = {
        str(obj["Key"]): obj for obj in client.list_objects(f"{prefix}/")
    }
    ready_key = f"{prefix}/_READY.json"
    report_key = f"{prefix}/promotion_report.json"
    if ready_key not in remote or report_key not in remote:
        raise IngestionPromotionError("remote candidate is missing READY/report objects")
    work_root.mkdir(parents=True, exist_ok=True)
    ready_path = client.download_file(ready_key, work_root / "_READY.json")
    report_path = client.download_file(report_key, work_root / "promotion_report.json")
    ready = _read_json(ready_path)
    remote_report = _read_json(report_path)
    if ready.get("status") != "READY" or ready.get("promotion_id") != candidate.promotion_id:
        raise IngestionPromotionError("remote candidate READY identity/status is invalid")
    if remote_report != candidate.report:
        raise IngestionPromotionError("remote promotion report differs from local bundle")
    remote_artifacts = [obj for key, obj in remote.items() if key != ready_key]
    if (
        int(ready.get("artifact_count", -1)) != len(remote_artifacts)
        or int(ready.get("artifact_bytes", -1))
        != sum(int(obj.get("Size", 0)) for obj in remote_artifacts)
    ):
        raise IngestionPromotionError("remote candidate READY count/bytes are invalid")
    for relative, expected in candidate.report["checksums"].items():
        key = f"{prefix}/{relative}"
        obj = remote.get(key)
        if obj is None or int(obj.get("Size", -1)) != int(expected["size"]):
            raise IngestionPromotionError(f"remote candidate artifact mismatch: {relative}")


def _ensure_backup(
    client: Any,
    *,
    candidate: ActivationCandidate,
    collections: Sequence[str],
    activation_root: Path,
    workers: int,
) -> dict[str, Any]:
    backup_prefix = f"{BACKUP_ROOT}/{candidate.promotion_id}"
    marker_key = f"{backup_prefix}/_BACKUP_READY.json"
    manifest_key = f"{backup_prefix}/backup_manifest.json"
    if client.object_exists(marker_key):
        local = activation_root / "backup-existing"
        marker_path = client.download_file(marker_key, local / "_BACKUP_READY.json")
        manifest_path = client.download_file(manifest_key, local / "backup_manifest.json")
        marker = _read_json(marker_path)
        manifest = _read_json(manifest_path)
        if (
            marker.get("status") != "BACKUP_READY"
            or marker.get("promotion_id") != candidate.promotion_id
            or manifest.get("promotion_id") != candidate.promotion_id
        ):
            raise IngestionPromotionError("existing backup marker/manifest is invalid")
        _verify_backup_objects(client, manifest)
        return manifest

    canonical: dict[str, Mapping[str, Any]] = {
        str(obj["Key"]): obj for obj in client.list_objects(CANONICAL_INGEST_PREFIX)
    }
    if client.object_exists(DATASET_MANIFEST_KEY):
        head = client.head_object(DATASET_MANIFEST_KEY)
        canonical[DATASET_MANIFEST_KEY] = {
            "Key": DATASET_MANIFEST_KEY,
            "Size": int(head.get("ContentLength", head.get("Size", 0))),
            "ETag": head.get("ETag", ""),
        }
    canonical.update(_list_collection_keyframes(client, collections))

    rows = []
    for key, obj in sorted(canonical.items()):
        backup_key = f"{backup_prefix}/snapshot/{key}"
        rows.append(
            {
                "source": key,
                "backup": backup_key,
                "size": int(obj.get("Size", obj.get("ContentLength", 0))),
                "etag": _normalise_etag(obj.get("ETag", "")),
            }
        )

    def copy_one(row: Mapping[str, Any]) -> None:
        client.copy_object(str(row["source"]), str(row["backup"]))

    _run_parallel_copies(
        rows,
        copy_one,
        workers=workers,
        log_label="backed up canonical objects",
    )

    manifest = {
        "schema_version": "1.0",
        "status": "BACKUP_READY",
        "promotion_id": candidate.promotion_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prefix": f"{backup_prefix}/",
        "object_count": len(rows),
        "object_bytes": sum(int(row["size"]) for row in rows),
        "objects": rows,
    }
    _verify_backup_objects(client, manifest)
    backup_root = activation_root / "backup"
    manifest_path = backup_root / "backup_manifest.json"
    marker_path = backup_root / "_BACKUP_READY.json"
    _write_json(manifest_path, manifest)
    client.upload_file(manifest_path, manifest_key, content_type="application/json")
    _write_json(
        marker_path,
        {
            "schema_version": "1.0",
            "status": "BACKUP_READY",
            "promotion_id": candidate.promotion_id,
            "object_count": len(rows),
            "object_bytes": manifest["object_bytes"],
        },
    )
    client.upload_file(marker_path, marker_key, content_type="application/json")
    return manifest


def _verify_backup_objects(client: Any, manifest: Mapping[str, Any]) -> None:
    objects = manifest.get("objects", [])
    backup_prefix = str(manifest.get("prefix", ""))
    remote = {
        str(obj["Key"]): obj for obj in client.list_objects(backup_prefix)
    }
    missing = [
        str(row["backup"])
        for row in objects
        if str(row["backup"]) not in remote
        or int(remote[str(row["backup"])].get("Size", -1)) != int(row["size"])
    ]
    if missing:
        raise IngestionPromotionError(
            f"backup verification failed for {len(missing)} objects; first={missing[0]}"
        )


def _copy_promoted_keyframes(
    client: Any,
    *,
    copy_rows: Sequence[Mapping[str, Any]],
    existing: Mapping[str, Mapping[str, Any]],
    workers: int,
) -> tuple[int, int]:
    pending = []
    skipped = 0
    for row in copy_rows:
        destination = str(row["destination"])
        current = existing.get(destination)
        if (
            current is not None
            and int(current.get("Size", -1)) == int(row["size"])
            and _normalise_etag(current.get("ETag", "")) == str(row["etag"])
        ):
            skipped += 1
        else:
            pending.append(row)

    if hasattr(client, "check_credentials"):
        client.check_credentials()

    def copy_one(row: Mapping[str, Any]) -> None:
        client.copy_object(str(row["source"]), str(row["destination"]))

    _run_parallel_copies(
        pending,
        copy_one,
        workers=workers,
        log_label="copied promoted keyframes",
    )
    return len(pending), skipped


def _verify_promoted_keyframes(
    client: Any,
    desired: Mapping[str, Mapping[str, Any]],
    collections: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    remote = _list_collection_keyframes(client, collections)
    mismatches = [
        key
        for key, row in desired.items()
        if key not in remote or int(remote[key].get("Size", -1)) != int(row["size"])
    ]
    if mismatches:
        raise IngestionPromotionError(
            f"canonical keyframe verification failed for {len(mismatches)} objects; "
            f"first={mismatches[0]}"
        )
    return remote


def _write_activated_manifests(
    candidate: ActivationCandidate,
    *,
    output_root: Path,
    activated_at: str,
    backup_prefix: str,
    stale_collections: set[str],
) -> dict[str, Path]:
    ingest = dict(candidate.ingest_manifest)
    ingest.update(
        {
            "status": "ACTIVE",
            "activated_at": activated_at,
            "backup_prefix": backup_prefix,
        }
    )
    batch = dict(candidate.batch_manifest)
    batch.update(
        {
            "status": "ACTIVE",
            "activated_at": activated_at,
            "backup_prefix": backup_prefix,
        }
    )
    collection_entries: dict[str, Any] = {}
    for collection in sorted(batch["collections"], key=_collection_sort_key):
        counts = batch["collections"][collection]
        entry: dict[str, Any] = {
            "collection": collection,
            "batch_id": batch["batch_id"],
            "run_id": counts["run_id"],
            "ingestion_promotion_id": candidate.promotion_id,
            "completeness": "INGESTION_ONLY",
            "downstream_status": "REBUILD_REQUIRED",
            "coverage": {
                "videos": int(counts["video_count"]),
                "shots": int(counts["shot_count"]),
                "keyframes": int(counts["keyframe_count"]),
                "objects": 0,
                "asr": 0,
            },
            "embeddings": {},
            "ocr": {"status": "NOT_BUILT"},
            "canonical_artifact_keys": {
                "keyframes": f"{CANONICAL_KEYFRAME_PREFIX}{collection}/",
                "keyframe_mapping": f"{CANONICAL_INGEST_PREFIX}keyframe_mapping.csv",
                "shots": f"{CANONICAL_INGEST_PREFIX}shots.parquet",
                "video_metadata": f"{CANONICAL_INGEST_PREFIX}video_metadata.parquet",
                "ingest_manifest": f"{CANONICAL_INGEST_PREFIX}ingest_manifest.json",
                "batch_manifest": (
                    f"{CANONICAL_INGEST_PREFIX}manifests/{batch['batch_id']}.json"
                ),
            },
        }
        if collection in stale_collections:
            entry["superseded_downstream_artifacts"] = {
                "status": "STALE_RETAINED_ON_R2",
                "reason": "canonical keyframe identity replaced by ingestion promotion",
                "previous_dataset_manifest_backup": (
                    f"{backup_prefix}snapshot/{DATASET_MANIFEST_KEY}"
                ),
            }
        collection_entries[collection] = entry

    dataset = {
        "schema_version": "1.0",
        "updated_at": activated_at,
        "active_ingestion_promotion": candidate.promotion_id,
        "batch_id": batch["batch_id"],
        "completeness": "INGESTION_ONLY",
        "downstream_status": "REBUILD_REQUIRED",
        "summary": candidate.report["summary"],
        "source_runs": ingest["source_runs"],
        "selector_profile": ingest["selector_profile"],
        "canonical_artifact_keys": {
            "ingest_manifest": f"{CANONICAL_INGEST_PREFIX}ingest_manifest.json",
            "batch_manifest": f"{CANONICAL_INGEST_PREFIX}manifests/{batch['batch_id']}.json",
            "keyframe_mapping": f"{CANONICAL_INGEST_PREFIX}keyframe_mapping.csv",
            "shots": f"{CANONICAL_INGEST_PREFIX}shots.parquet",
            "video_metadata": f"{CANONICAL_INGEST_PREFIX}video_metadata.parquet",
        },
        "collections": collection_entries,
    }

    paths = {
        "keyframe_mapping": candidate.root / "ingest" / "keyframe_mapping.csv",
        "shots": candidate.root / "ingest" / "shots.parquet",
        "video_metadata": candidate.root / "ingest" / "video_metadata.parquet",
        "batch_manifest": output_root / "ingest" / "manifests" / f"{batch['batch_id']}.json",
        "ingest_manifest": output_root / "ingest" / "ingest_manifest.json",
        "dataset_manifest": output_root / "dataset_manifest.json",
    }
    _write_json(paths["batch_manifest"], batch)
    _write_json(paths["ingest_manifest"], ingest)
    _write_json(paths["dataset_manifest"], dataset)
    return paths


def _upload_canonical_metadata(
    client: Any, paths: Mapping[str, Path], *, batch_id: str
) -> None:
    uploads = (
        ("keyframe_mapping", f"{CANONICAL_INGEST_PREFIX}keyframe_mapping.csv"),
        ("shots", f"{CANONICAL_INGEST_PREFIX}shots.parquet"),
        ("video_metadata", f"{CANONICAL_INGEST_PREFIX}video_metadata.parquet"),
        ("batch_manifest", f"{CANONICAL_INGEST_PREFIX}manifests/{batch_id}.json"),
        ("ingest_manifest", f"{CANONICAL_INGEST_PREFIX}ingest_manifest.json"),
        ("dataset_manifest", DATASET_MANIFEST_KEY),
    )
    for name, key in uploads:
        client.upload_file(paths[name], key, content_type=_content_type(paths[name]))


def _verify_active_metadata(
    client: Any,
    candidate: ActivationCandidate,
    paths: Mapping[str, Path],
    work_root: Path,
    *,
    batch_id: str,
) -> None:
    expected_sizes = {
        f"{CANONICAL_INGEST_PREFIX}keyframe_mapping.csv": paths["keyframe_mapping"].stat().st_size,
        f"{CANONICAL_INGEST_PREFIX}shots.parquet": paths["shots"].stat().st_size,
        f"{CANONICAL_INGEST_PREFIX}video_metadata.parquet": paths["video_metadata"].stat().st_size,
        f"{CANONICAL_INGEST_PREFIX}manifests/{batch_id}.json": paths["batch_manifest"].stat().st_size,
        f"{CANONICAL_INGEST_PREFIX}ingest_manifest.json": paths["ingest_manifest"].stat().st_size,
        DATASET_MANIFEST_KEY: paths["dataset_manifest"].stat().st_size,
    }
    for key, size in expected_sizes.items():
        head = client.head_object(key)
        actual = int(head.get("ContentLength", head.get("Size", -1)))
        if actual != size:
            raise IngestionPromotionError(f"active metadata size mismatch: {key}")
    work_root.mkdir(parents=True, exist_ok=True)
    active_path = client.download_file(
        DATASET_MANIFEST_KEY, work_root / "dataset_manifest.json"
    )
    active = _read_json(active_path)
    if active.get("active_ingestion_promotion") != candidate.promotion_id:
        raise IngestionPromotionError("dataset manifest did not activate the promotion")


def _list_collection_keyframes(
    client: Any, collections: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    selected = set(collections)
    result: dict[str, Mapping[str, Any]] = {}
    for obj in client.list_objects(CANONICAL_KEYFRAME_PREFIX):
        key = str(obj["Key"])
        relative = key.removeprefix(CANONICAL_KEYFRAME_PREFIX)
        parts = PurePosixPath(relative).parts
        if not parts or parts[0] not in selected:
            continue
        if key in result:
            raise IngestionPromotionError(f"duplicate listed canonical key: {key}")
        result[key] = obj
    return result


def _read_copy_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise IngestionPromotionError(f"cannot read copy manifest {path}: {exc}") from exc
    with handle:
        destinations: set[str] = set()
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IngestionPromotionError(
                    f"invalid copy manifest JSON at line {line_number}"
                ) from exc
            source = str(row.get("source", ""))
            destination = str(row.get("destination", ""))
            if not source.startswith("data/staging/ingestion/"):
                raise IngestionPromotionError(f"unsafe copy source at line {line_number}")
            if not destination.startswith(CANONICAL_KEYFRAME_PREFIX):
                raise IngestionPromotionError(f"unsafe copy destination at line {line_number}")
            if destination in destinations:
                raise IngestionPromotionError(f"duplicate copy destination: {destination}")
            destinations.add(destination)
            size = int(row.get("size", -1))
            etag = _normalise_etag(row.get("etag", ""))
            if size <= 0 or not etag:
                raise IngestionPromotionError(
                    f"invalid copy size/etag at line {line_number}"
                )
            yield {
                "source": source,
                "destination": destination,
                "size": size,
                "etag": etag,
            }


def _run_parallel_copies(
    rows: Sequence[Mapping[str, Any]],
    operation: Any,
    *,
    workers: int,
    log_label: str,
) -> None:
    """Run copy operations in bounded chunks to avoid 100k queued futures."""

    if not rows:
        return
    completed = 0
    chunk_size = max(100, workers * 20)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(rows), chunk_size):
            chunk = rows[offset : offset + chunk_size]
            futures = [executor.submit(operation, row) for row in chunk]
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 1000 == 0 or completed == len(rows):
                    LOGGER.info("%s=%d/%d", log_label, completed, len(rows))


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _content_type(path: Path) -> str | None:
    return {
        ".csv": "text/csv",
        ".json": "application/json",
        ".parquet": "application/octet-stream",
    }.get(path.suffix.lower())


def _collection_sort_key(value: str) -> tuple[int, str]:
    raw = str(value)
    if len(raw) > 1 and raw[0].upper() == "L" and raw[1:].isdigit():
        return int(raw[1:]), raw
    return 10**9, raw
