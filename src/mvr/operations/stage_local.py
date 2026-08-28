"""Real-R2 fetch adapter for the worker orchestrator.

This is the only DATA-03 module that touches live R2 and local disk. It turns an
object key into a locally-resident path using the DATA-02 `BoundedCache` +
`R2Client`, and reports the exact per-fetch counters (GET/HEAD, bytes, cache hit)
the orchestrator needs for honest metrics.

Keeping this separate from `worker_orchestrate` means the orchestration logic is
unit-testable with a fake fetcher and never imports boto3 or any model.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from .cache import BoundedCache
from .r2_client import R2Client
from .worker_orchestrate import FetchOutcome

LOGGER = logging.getLogger(__name__)


class R2StageFetcher:
    """Materialize object keys through the shared BoundedCache.

    A cache hit costs no network I/O; a miss issues exactly one GET (via
    `client.download_file`) plus one HEAD when a size probe is requested. The
    cache dedupes concurrent misses for the same key, so a prefetch and a
    foreground request for the same bytes never double-download.
    """

    def __init__(
        self,
        *,
        client: R2Client,
        cache: BoundedCache,
        probe_size: bool = False,
    ) -> None:
        self.client = client
        self.cache = cache
        self.probe_size = probe_size

    async def fetch(self, object_key: str) -> FetchOutcome:
        # Run the blocking boto3 / filesystem work off the event loop so the
        # foreground stage keeps making progress while a prefetch downloads.
        return await asyncio.to_thread(self._fetch_sync, object_key)

    def _fetch_sync(self, object_key: str) -> FetchOutcome:
        started = time.perf_counter()
        cached = self.cache.get(object_key)
        if cached is not None:
            return FetchOutcome(
                object_key=object_key,
                local_path=cached,
                size_bytes=cached.stat().st_size,
                cache_hit=True,
                seconds=time.perf_counter() - started,
            )

        head_requests = 0
        expected_size: int | None = None
        if self.probe_size:
            try:
                meta = self.client.head_object(object_key)
                expected_size = int(meta.get("ContentLength", 0)) or None
                head_requests = 1
            except Exception as exc:  # HEAD is best-effort; a miss still downloads
                LOGGER.debug("HEAD %s failed: %s", object_key, exc)

        get_requests = 0

        def _download(temp_path: Path) -> None:
            nonlocal get_requests
            self.client.download_file(object_key, temp_path)
            get_requests = 1

        local_path = self.cache.put(
            object_key, _download, expected_size=expected_size
        )
        return FetchOutcome(
            object_key=object_key,
            local_path=local_path,
            size_bytes=local_path.stat().st_size,
            cache_hit=False,
            get_requests=get_requests,
            head_requests=head_requests,
            seconds=time.perf_counter() - started,
        )

    def delete_local(self, path: Path) -> None:
        """Evict a materialized file from local disk (working-set eviction)."""
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:
            LOGGER.warning("failed to delete %s: %s", path, exc)
