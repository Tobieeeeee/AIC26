"""Active-version pointer for immutable enrichment outputs.

An enrichment run publishes into ``data/versions/enrichment/<version>/`` and
never touches ``data/corpus``.  ``dataset_manifest.json`` carries the single
pointer that decides which version is active, so activating a validated
version and rolling back to the previous one are each one PUT.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mvr.operations.r2_layout import (
    enrichment_version_prefix,
    raw_batch_segment,
    version_key,
)


DATASET_MANIFEST_KEY = "data/corpus/dataset_manifest.json"
ACTIVE_VERSION_FIELD = "active_enrichment_version"
ENRICHMENT_FIELD = "enrichment"


class EnrichmentVersionError(RuntimeError):
    """Raised when the manifest pointer is unusable."""


def resolve_active_enrichment_version(manifest: Mapping[str, Any]) -> str | None:
    """Return the active enrichment version, or ``None`` when none is active.

    A malformed pointer is an error rather than a silent ``None``: treating it
    as "nothing activated" would make consumers fall back to reading no
    enrichment at all while the manifest claims otherwise.
    """
    if ACTIVE_VERSION_FIELD not in manifest:
        return None
    pointer = manifest[ACTIVE_VERSION_FIELD]
    if pointer is None:
        return None
    if not isinstance(pointer, str):
        raise EnrichmentVersionError(
            f"{ACTIVE_VERSION_FIELD} must be a string or null, got {type(pointer).__name__}"
        )
    version = pointer.strip()
    if not version or "/" in version or version in {".", ".."}:
        raise EnrichmentVersionError(
            f"{ACTIVE_VERSION_FIELD} must be a single path segment: {pointer!r}"
        )
    return version


def apply_enrichment_pointer(
    manifest: Mapping[str, Any],
    *,
    version: str,
    activated_at: str,
    report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of ``manifest`` with the enrichment pointer set to ``version``.

    The input mapping is not mutated so a caller can back up the previous
    manifest bytes before writing the new ones.
    """
    version_id = str(version).strip()
    prefix = enrichment_version_prefix(version_id)
    previous = resolve_active_enrichment_version(manifest)
    updated = deepcopy(dict(manifest))
    updated[ACTIVE_VERSION_FIELD] = version_id
    entry: dict[str, Any] = {
        "version": version_id,
        "prefix": prefix,
        "activated_at": activated_at,
        "previous_version": previous,
    }
    if report is not None:
        entry["run_report"] = {
            key: report[key]
            for key in ("run_id", "batch_id", "collections", "features", "keyframe_count")
            if key in report
        }
    updated[ENRICHMENT_FIELD] = entry
    updated["downstream_status"] = "ENRICHED"
    return updated


def _backup_segment(activated_at: str) -> str:
    return "".join(char for char in activated_at if char.isalnum()) or "backup"


def activate_enrichment_version(
    *,
    client: Any,
    version: str,
    work_root: Path,
    activated_at: str | None = None,
) -> dict[str, Any]:
    """Point ``dataset_manifest.json`` at a validated enrichment version.

    Fail-closed: the version root must carry ``_READY.json`` and every
    collection it declares must carry ``_SUCCESS.json``.  No artifact is
    copied; activation and rollback are each one manifest PUT, preceded by a
    backup of the manifest being replaced.
    """
    stamp = activated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prefix = enrichment_version_prefix(version)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    remote_keys = {str(obj["Key"]) for obj in client.list_objects(f"{prefix}/")}
    ready_key = f"{prefix}/_READY.json"
    if ready_key not in remote_keys:
        raise EnrichmentVersionError(
            f"version is not publishable; missing _READY marker: {ready_key}"
        )

    ready_path = work_root / "_READY.json"
    client.download_file(ready_key, ready_path)
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("status") != "READY":
        raise EnrichmentVersionError(
            f"version _READY marker is not READY: {ready.get('status')!r}"
        )
    collections = ready.get("collections")
    if not isinstance(collections, list) or not collections:
        raise EnrichmentVersionError("version _READY marker declares no collections")
    batch = raw_batch_segment(str(ready.get("batch_id", "1")))[len("batch"):]
    missing = [
        collection
        for collection in collections
        if f"{prefix}/batch={batch}/collection={collection}/_SUCCESS.json"
        not in remote_keys
    ]
    if missing:
        raise EnrichmentVersionError(
            f"version is incomplete; missing collection _SUCCESS markers: {missing}"
        )

    manifest_path = work_root / "dataset_manifest.json"
    client.download_file(DATASET_MANIFEST_KEY, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous = resolve_active_enrichment_version(manifest)

    backup_key = version_key(
        "manifest-backups", _backup_segment(stamp), "dataset_manifest.json"
    )
    client.upload_file(manifest_path, backup_key, content_type="application/json")

    updated = apply_enrichment_pointer(
        manifest, version=version, activated_at=stamp, report=ready
    )
    updated_path = work_root / "dataset_manifest.activated.json"
    updated_path.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    client.upload_file(
        updated_path, DATASET_MANIFEST_KEY, content_type="application/json"
    )

    return {
        "status": "ACTIVATED",
        "version": updated[ACTIVE_VERSION_FIELD],
        "prefix": prefix,
        "previous_version": previous,
        "collections": list(collections),
        "activated_at": stamp,
        "dataset_manifest_backup": backup_key,
        "artifacts_copied": 0,
    }
