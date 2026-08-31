"""High-level R2 sync orchestration for the AIC26 pipeline.

Bridges Cloudflare R2 object storage with the local filesystem layout
expected by the ingestion, enrichment, and indexing CLIs.

Pull path:  R2 -> local cache -> local working tree
Push path:  local working tree -> R2 (under data/staging or data/corpus)
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .cache import BoundedCache
from .r2_client import R2Client, R2ConfigError
from .r2_layout import (
    CORPUS_PREFIX,
    RAW_PREFIX,
    STAGING_PREFIX,
    corpus_embedding_key,
    corpus_index_key,
    corpus_metadata_key,
    corpus_report_key,
    dataset_manifest_key,
    raw_video_key,
    staging_key,
)

LOGGER = logging.getLogger(__name__)

INGESTION_UPLOAD_EXCLUDED_ROOTS = frozenset({"selector_cache"})


@dataclass
class SyncStats:
    """Counters for a sync operation."""

    downloaded: int = 0
    skipped_up_to_date: int = 0
    uploaded: int = 0
    errors: int = 0
    bytes_transferred: int = 0

    def to_dict(self) -> dict:
        return {
            "downloaded": self.downloaded,
            "skipped_up_to_date": self.skipped_up_to_date,
            "uploaded": self.uploaded,
            "errors": self.errors,
            "bytes_transferred": self.bytes_transferred,
        }


class R2Sync:
    """Pull/push orchestration between R2 and the local working tree."""

    def __init__(
        self,
        *,
        client: R2Client | None = None,
        local_root: str | Path = ".",
        cache: BoundedCache | None = None,
        raw_prefix: str = RAW_PREFIX,
        corpus_prefix: str = CORPUS_PREFIX,
        staging_prefix: str = STAGING_PREFIX,
    ) -> None:
        self.client = client or R2Client()
        self.local_root = Path(local_root)
        self.cache = cache or BoundedCache()
        self.raw_prefix = raw_prefix
        self.corpus_prefix = corpus_prefix
        self.staging_prefix = staging_prefix

    # ---- pull helpers ---------------------------------------------------

    def pull_object(
        self,
        key: str,
        local_path: Path,
        *,
        expected_sha256: str | None = None,
        force: bool = False,
    ) -> bool:
        """Download a single object into the cache and materialize a copy.

        Returns True if a download happened.
        """
        local_path = Path(local_path)
        if not force and local_path.is_file() and local_path.stat().st_size > 0:
            LOGGER.debug("skip %s (local file already exists)", key)
            return False
        target = self.cache.local_path(key)

        def _do_download(temp_path: Path) -> None:
            self.client.download_file(
                key, temp_path, expected_sha256=expected_sha256
            )

        cached = self.cache.put(key, _do_download)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if cached != local_path.resolve():
            if local_path.exists():
                local_path.unlink()
            try:
                shutil.copy2(cached, local_path)
            except OSError:
                try:
                    local_path.symlink_to(cached.resolve())
                except OSError:
                    local_path.hardlink_to(cached)
        return True

    def pull_videos(
        self,
        batch_id: str,
        collection_id: str,
        video_ids: Iterable[str],
        *,
        force: bool = False,
    ) -> SyncStats:
        """Pull raw videos into <local_root>/<raw_prefix>/batchN/videos/..."""
        stats = SyncStats()
        for video_id in video_ids:
            key = raw_video_key(batch_id, collection_id, video_id)
            local = self.local_root / key
            try:
                if self.pull_object(key, local, force=force):
                    stats.downloaded += 1
                    stats.bytes_transferred += local.stat().st_size
                else:
                    stats.skipped_up_to_date += 1
            except (R2ConfigError, Exception) as exc:
                LOGGER.error("failed to pull %s: %s", key, exc)
                stats.errors += 1
        return stats

    def pull_prefix(
        self,
        prefix: str,
        *,
        force: bool = False,
        max_keys: int | None = None,
    ) -> SyncStats:
        """Pull every object under an R2 prefix into the local tree."""
        stats = SyncStats()
        objects = self.client.list_objects(prefix)
        if max_keys is not None:
            objects = objects[:max_keys]
        for obj in objects:
            key = obj["Key"]
            size = int(obj.get("Size", 0))
            local = self.local_root / key
            try:
                if self.pull_object(key, local, force=force):
                    stats.downloaded += 1
                    stats.bytes_transferred += size
                else:
                    stats.skipped_up_to_date += 1
            except (R2ConfigError, Exception) as exc:
                LOGGER.error("failed to pull %s: %s", key, exc)
                stats.errors += 1
        return stats

    def pull_corpus(self, *, force: bool = False, max_keys: int | None = None) -> SyncStats:
        """Pull the canonical corpus tree (keyframes/metadata/features/indexes)."""
        return self.pull_prefix(self.corpus_prefix, force=force, max_keys=max_keys)

    def pull_dataset_manifest(self, *, force: bool = False) -> Path | None:
        """Pull the dataset manifest JSON and return its local path."""
        key = dataset_manifest_key()
        local = self.local_root / key
        try:
            self.pull_object(key, local, force=force)
            return local
        except (R2ConfigError, Exception) as exc:
            LOGGER.error("failed to pull dataset manifest: %s", exc)
            return None

    # ---- push helpers ---------------------------------------------------

    def push_file(
        self,
        local_path: Path,
        key: str,
        *,
        content_type: str | None = None,
    ) -> bool:
        """Upload one local file to R2. Returns True on success."""
        try:
            result = self.client.upload_file(local_path, key, content_type=content_type)
            LOGGER.debug("pushed %s -> %s (sha256=%s)", local_path, key, result.get("sha256"))
            return True
        except Exception as exc:
            LOGGER.error("failed to push %s -> %s: %s", local_path, key, exc)
            return False

    def push_staging(
        self,
        feature: str,
        run_id: str,
        local_paths: Iterable[Path],
    ) -> SyncStats:
        """Push local artifacts under the staging prefix."""
        stats = SyncStats()
        for path in local_paths:
            path = Path(path)
            if not path.is_file():
                LOGGER.warning("skip missing staging file: %s", path)
                continue
            key = staging_key(feature, run_id, path.name)
            if self.push_file(path, key):
                stats.uploaded += 1
                stats.bytes_transferred += path.stat().st_size
            else:
                stats.errors += 1
        return stats

    def push_ingestion_run(
        self,
        output_root: Path,
        *,
        run_id: str,
        batch_id: str,
        collection_id: str,
        video_id: str,
    ) -> SyncStats:
        """Upload one publishable per-video ingestion tree to staging.

        Published artifacts keep their paths relative to ``output_root``.
        Regenerable selector candidates under ``selector_cache/`` are excluded
        from this upload; final keyframes, selected embeddings, and metadata
        remain publishable. A ``_SUCCESS.json`` marker is uploaded only after
        every preceding file succeeds, so an unattended runner can safely
        distinguish complete videos from partial uploads.
        """
        root = Path(output_root)
        stats = SyncStats()
        if not root.is_dir():
            LOGGER.error("ingestion output root does not exist: %s", root)
            stats.errors = 1
            return stats

        marker = root / "_SUCCESS.json"
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path != marker
            and path.relative_to(root).parts[0]
            not in INGESTION_UPLOAD_EXCLUDED_ROOTS
        )
        if not files:
            LOGGER.error("ingestion output root contains no files: %s", root)
            stats.errors = 1
            return stats

        prefix_parts = (
            f"batch={batch_id}",
            f"collection={collection_id}",
            video_id,
        )
        content_types = {
            ".csv": "text/csv",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".json": "application/json",
            ".npy": "application/octet-stream",
        }
        for path in files:
            relative = path.relative_to(root)
            key = staging_key(
                "ingestion", run_id, *prefix_parts, *relative.parts
            )
            if self.push_file(
                path,
                key,
                content_type=content_types.get(path.suffix.lower()),
            ):
                stats.uploaded += 1
                stats.bytes_transferred += path.stat().st_size
            else:
                stats.errors += 1

        if stats.errors:
            LOGGER.error(
                "ingestion upload incomplete for %s: uploaded=%d errors=%d",
                video_id,
                stats.uploaded,
                stats.errors,
            )
            return stats

        marker.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "DONE",
                    "run_id": run_id,
                    "batch_id": batch_id,
                    "collection_id": collection_id,
                    "video_id": video_id,
                    "artifact_count": len(files),
                    "artifact_bytes": stats.bytes_transferred,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        marker_key = staging_key(
            "ingestion", run_id, *prefix_parts, marker.name
        )
        if self.push_file(marker, marker_key, content_type="application/json"):
            stats.uploaded += 1
            stats.bytes_transferred += marker.stat().st_size
        else:
            stats.errors += 1
        return stats

    def push_corpus_metadata(
        self,
        feature: str,
        batch_id: str,
        collection_id: str,
        local_parquet: Path,
    ) -> SyncStats:
        """Push a canonical metadata Parquet into the corpus tree."""
        stats = SyncStats()
        key = corpus_metadata_key(feature, batch_id, collection_id)
        if self.push_file(local_parquet, key, content_type="application/octet-stream"):
            stats.uploaded += 1
            stats.bytes_transferred += Path(local_parquet).stat().st_size
        else:
            stats.errors += 1
        return stats

    def push_corpus_embeddings(
        self,
        provider: str,
        local_path: Path,
        filename: str = "vectors.npy",
    ) -> SyncStats:
        """Push provider embedding vectors into the corpus tree."""
        stats = SyncStats()
        key = corpus_embedding_key(provider, filename)
        local_path = Path(local_path)
        if self.push_file(local_path, key, content_type="application/octet-stream"):
            stats.uploaded += 1
            stats.bytes_transferred += local_path.stat().st_size
        else:
            stats.errors += 1
        return stats

    def push_corpus_index(self, local_path: Path, *parts: str) -> SyncStats:
        """Push an index artifact into the corpus tree."""
        stats = SyncStats()
        key = corpus_index_key(*parts)
        local_path = Path(local_path)
        if self.push_file(local_path, key, content_type="application/octet-stream"):
            stats.uploaded += 1
            stats.bytes_transferred += local_path.stat().st_size
        else:
            stats.errors += 1
        return stats

    def push_corpus_report(self, local_path: Path, *parts: str) -> SyncStats:
        """Push a report JSON into the corpus tree."""
        stats = SyncStats()
        key = corpus_report_key(*parts)
        local_path = Path(local_path)
        if self.push_file(local_path, key, content_type="application/json"):
            stats.uploaded += 1
            stats.bytes_transferred += local_path.stat().st_size
        else:
            stats.errors += 1
        return stats

    # ---- introspection --------------------------------------------------

    def list_remote(self, prefix: str) -> list[dict]:
        """List objects under an R2 prefix."""
        return self.client.list_objects(prefix)

    def check_credentials(self) -> None:
        """Fail fast if R2 credentials are unusable."""
        self.client.list_objects(self.raw_prefix, max_keys=1)
