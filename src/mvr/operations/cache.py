"""Bounded local cache for R2 objects.

Optimized for latency, duplicate-read avoidance, and worker efficiency.
R2 Internet egress is $0 so this cache optimizes for local I/O, not bandwidth cost.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ".cache/mvr"
DEFAULT_MAX_SIZE_GB = 50.0


@dataclass
class CacheStats:
    """Cache hit/miss statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    downloads: int = 0
    bytes_downloaded: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "downloads": self.downloads,
            "bytes_downloaded": self.bytes_downloaded,
            "hit_rate": round(self.hit_rate, 4),
        }


def _object_key_to_local_path(cache_dir: Path, object_key: str) -> Path:
    """Deterministic mapping from R2 object key to local cache path.

    Uses the full key hierarchy to avoid collisions.
    """
    # Normalize separators and strip leading slash
    normalized = object_key.strip("/").replace("\\", "/")
    if not normalized:
        raise ValueError("object key must not be empty")
    return cache_dir / normalized


class BoundedCache:
    """Bounded local cache manager for R2 objects.

    Thread-safe for concurrent reads. Downloads are atomic (temp file + rename).
    Eviction uses LRU based on file access time.
    """

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        max_size_gb: float = DEFAULT_MAX_SIZE_GB,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self._lock = threading.Lock()
        self._stats = CacheStats()
        self._in_progress: set[str] = set()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def local_path(self, object_key: str) -> Path:
        """Return the deterministic local path for an object key."""
        return _object_key_to_local_path(self.cache_dir, object_key)

    def contains(self, object_key: str) -> bool:
        """Check if an object is in the cache."""
        path = self.local_path(object_key)
        return path.is_file() and path.stat().st_size > 0

    def get(self, object_key: str) -> Path | None:
        """Get the local path for a cached object, or None if not cached.

        Updates access time on hit to support LRU eviction.
        """
        path = self.local_path(object_key)
        if path.is_file() and path.stat().st_size > 0:
            with self._lock:
                self._stats.hits += 1
            # Touch access time for LRU
            try:
                os.utime(path)
            except OSError:
                pass
            return path
        with self._lock:
            self._stats.misses += 1
        return None

    def put(
        self,
        object_key: str,
        download_fn: Any,
        *,
        expected_size: int | None = None,
    ) -> Path:
        """Ensure an object is cached. Downloads on miss using download_fn.

        download_fn(local_path: Path) -> None should download the object to
        the given path. It will be called with a temporary path that is
        atomically renamed on success.

        Concurrent calls for the same key will wait rather than duplicate
        the download.
        """
        # Fast path: already cached
        cached = self.get(object_key)
        if cached is not None:
            return cached

        # Slow path: need to download
        target = self.local_path(object_key)

        with self._lock:
            # Double-check after acquiring lock
            if target.is_file() and target.stat().st_size > 0:
                self._stats.hits += 1
                return target

            if object_key in self._in_progress:
                # Another thread is downloading this key; wait
                pass
            else:
                self._in_progress.add(object_key)

        # Wait for in-progress download by another thread
        wait_count = 0
        while True:
            with self._lock:
                if object_key not in self._in_progress:
                    if target.is_file() and target.stat().st_size > 0:
                        self._stats.hits += 1
                        return target
                    # Previous attempt failed; we take over
                    self._in_progress.add(object_key)
                    break
                # We are the owner
                if object_key in self._in_progress:
                    break
            time.sleep(0.1)
            wait_count += 1
            if wait_count > 600:  # 60 second timeout
                raise TimeoutError(f"timeout waiting for download of {object_key}")

        try:
            self._ensure_space(expected_size or 0)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_fd, temp_path_str = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
            temp_path = Path(temp_path_str)
            try:
                os.close(temp_fd)
                download_fn(temp_path)
                # Verify download is not empty/partial
                if not temp_path.is_file() or temp_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"download produced empty or missing file for {object_key}"
                    )
                temp_path.replace(target)
                with self._lock:
                    self._stats.downloads += 1
                    self._stats.bytes_downloaded += target.stat().st_size
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
        finally:
            with self._lock:
                self._in_progress.discard(object_key)

        return target

    def _ensure_space(self, needed_bytes: int) -> None:
        """Evict LRU entries until there is enough space."""
        if self.max_size_bytes <= 0:
            return
        current = self._cache_size_bytes()
        target = self.max_size_bytes - needed_bytes
        if current <= target:
            return
        # Collect all cached files sorted by access time (oldest first)
        entries: list[tuple[float, Path]] = []
        for path in self.cache_dir.rglob("*"):
            if path.is_file() and not path.name.startswith("."):
                try:
                    entries.append((path.stat().st_atime, path))
                except OSError:
                    continue
        entries.sort(key=lambda e: e[0])
        for _, path in entries:
            if current <= target:
                break
            try:
                size = path.stat().st_size
                path.unlink()
                current -= size
                with self._lock:
                    self._stats.evictions += 1
                LOGGER.debug("evicted %s (%d bytes)", path, size)
            except OSError:
                continue

    def _cache_size_bytes(self) -> int:
        """Total size of all cached files."""
        total = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def clear(self) -> int:
        """Remove all cached files. Returns count of removed files."""
        count = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                try:
                    path.unlink()
                    count += 1
                except OSError:
                    continue
        return count
