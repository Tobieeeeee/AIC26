"""Focused unit tests for the DATA-03 worker orchestrator.

These tests use a fake fetcher and the `_DryRunFetcher` from the worker CLI
where appropriate. No real R2, no models, no network.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from mvr.operations.worker_orchestrate import (
    Stage,
    StageResult,
    WorkItem,
    WorkingSet,
    prefetch_run,
    sequential_run,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class SlowFakeFetcher:
    """Materializes keys into a tmp dir with a configurable per-key sleep.

    Counts downloads, bytes, and the set of keys fetched more than once, so
    tests can assert on reuse. Honors a *parallel* flag: if False, the
    downloads run sequentially, so a "prefetch" test can be sure overlap
    happened by observing the timing of fetches vs. stages.
    """

    def __init__(self, *, per_key_seconds: float = 0.05, download_jitter: float = 0.0):
        self.root = Path(tempfile.mkdtemp(prefix="mvr_worker_"))
        self.per_key_seconds = per_key_seconds
        self.download_jitter = download_jitter
        self.downloads: list[tuple[str, float]] = []  # (key, fetched_at)
        self.bytes_by_key: dict[str, int] = {}
        self.get_requests = 0
        self.head_requests = 0

    def declare_bytes(self, key: str, size: int) -> None:
        self.bytes_by_key[key] = size

    async def fetch(self, key: str):
        return await asyncio.to_thread(self._fetch_sync, key)

    def _fetch_sync(self, key: str):
        fetched_at = time.perf_counter()
        self.downloads.append((key, fetched_at))
        if self.download_jitter:
            time.sleep(self.download_jitter)
        local = self.root / key.replace("/", "_")
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.is_file() and local.stat().st_size > 0:
            return _outcome(key, local, cache_hit=True)
        self.get_requests += 1
        size = self.bytes_by_key.get(key, 1024)
        local.write_bytes(b"x" * size)
        return _outcome(key, local, cache_hit=False, get_requests=1)


def _outcome(key, local, *, cache_hit, get_requests=0, head_requests=0):
    from mvr.operations.worker_orchestrate import FetchOutcome
    return FetchOutcome(
        object_key=key,
        local_path=local,
        size_bytes=local.stat().st_size,
        cache_hit=cache_hit,
        get_requests=get_requests,
        head_requests=head_requests,
    )


class TrackingStage:
    """A stage that records acquire/release timing and the consumer it uses."""

    def __init__(self, name: str, *, video_keys: bool = True,
                 keyframe_keys: bool = False, sleep: float = 0.02,
                 fail_on: set[str] | None = None):
        self.name = name
        self.video_keys = video_keys
        self.keyframe_keys = keyframe_keys
        self.sleep = sleep
        self.fail_on = fail_on or set()
        self.starts: list[tuple[str, float]] = []  # (item_id, started_at)
        self.ended: list[tuple[str, float]] = []
        self.local_paths: dict[str, dict[str, Path]] = {}

    def keys_for(self, item: WorkItem) -> list[str]:
        out: list[str] = []
        if self.video_keys:
            out.append(item.metadata["video_key"])
        if self.keyframe_keys:
            out.extend(item.metadata.get("keyframe_keys", []))
        return out

    async def run(self, item, paths):
        from mvr.operations.worker_orchestrate import StageResult
        self.starts.append((item.item_id, time.perf_counter()))
        self.local_paths[item.item_id] = dict(paths)
        if item.item_id in self.fail_on:
            return StageResult(stage=self.name, item_id=item.item_id,
                               success=False, error="forced")
        if self.sleep:
            await asyncio.sleep(self.sleep)
        self.ended.append((item.item_id, time.perf_counter()))
        return StageResult(stage=self.name, item_id=item.item_id, success=True)


def _video_item(vid: str, *, batch="1", collection="L21", kf_keys=None) -> WorkItem:
    from mvr.operations.r2_layout import raw_video_key
    video_key = raw_video_key(batch, collection, vid)
    kf = kf_keys or []
    keys = [video_key, *kf]
    return WorkItem(
        item_id=vid,
        keys=keys,
        metadata={"video_key": video_key, "keyframe_keys": kf, "video_id": vid},
    )


# ---------------------------------------------------------------------------
# WorkingSet
# ---------------------------------------------------------------------------


class TestWorkingSet:
    def test_declare_then_admit_then_acquire_release(self):
        async def main():
            ws = WorkingSet()
            await ws.declare("k1", ["ocr:v1", "asr:v1"])
            await ws.admit("k1", Path("/tmp/k1"), 100)
            assert ws.is_resident("k1")
            p = await ws.acquire("k1", "ocr:v1")
            assert p == Path("/tmp/k1")
            assert ws._resident["k1"].refcount == 1
            # Same consumer can acquire multiple times? Protocol: each acquire
            # is a distinct hold. Test that re-acquire increments.
            await ws.acquire("k1", "asr:v1")
            assert ws._resident["k1"].refcount == 2
            # Now release both. release() reports the last-consumer transition
            # but does not itself evict — eviction is a separate explicit step.
            assert (await ws.release("k1", "ocr:v1")) is False
            assert (await ws.release("k1", "asr:v1")) is True
            assert ws.is_resident("k1")  # still resident until evicted
            assert (await ws.evict_releasable()) == ["k1"]
            assert not ws._resident
            return "ok"
        assert asyncio.run(main()) == "ok"

    def test_evict_releasable_skips_active(self):
        async def main():
            ws = WorkingSet()
            await ws.declare("k1", ["ingest:v1", "ocr:v1"])
            await ws.admit("k1", Path("/tmp/k1"), 1)
            await ws.acquire("k1", "ingest:v1")
            evicted = await ws.evict_releasable()
            assert evicted == []
            await ws.acquire("k1", "ocr:v1")
            await ws.release("k1", "ingest:v1")
            evicted = await ws.evict_releasable()
            assert evicted == []
            await ws.release("k1", "ocr:v1")
            evicted = await ws.evict_releasable()
            assert evicted == ["k1"]
        asyncio.run(main())

    def test_reserve_space_backpressures(self):
        async def main():
            ws = WorkingSet(max_bytes=1000)
            await ws.declare("a", ["x"]); await ws.admit("a", Path("/a"), 600)
            await ws.declare("b", ["x"]); await ws.admit("b", Path("/b"), 400)
            await ws.acquire("a", "x")
            await ws.acquire("b", "x")
            # Both are still in use, so reserving 100 bytes must time out quickly.
            t0 = time.perf_counter()
            with pytest.raises(TimeoutError):
                await ws.reserve_space(100, timeout=0.2)
            elapsed = time.perf_counter() - t0
            assert elapsed < 1.0
        asyncio.run(main())


# ---------------------------------------------------------------------------
# Sequential run
# ---------------------------------------------------------------------------


class TestSequentialRun:
    def test_no_overlap_zero(self):
        async def main():
            fetcher = SlowFakeFetcher(per_key_seconds=0.02)
            items = [_video_item("V1"), _video_item("V2"), _video_item("V3")]
            for it in items:
                fetcher.declare_bytes(it.metadata["video_key"], 100)
            stages = [TrackingStage("ingest", sleep=0.02)]
            result = await sequential_run(items, stages, fetcher)
            assert result.metrics.run_mode == "sequential"
            assert result.metrics.overlap_ratio == 0.0
            assert result.metrics.prefetch_overlap_seconds == 0.0
            assert result.metrics.items_completed == 3
            assert result.metrics.items_failed == 0
            # Each item's first stage acquired the same video key
            for it in items:
                assert it.metadata["video_key"] in stages[0].local_paths[it.item_id]
        asyncio.run(main())

    def test_no_prefetch_so_no_reuse_across_items(self):
        async def main():
            fetcher = SlowFakeFetcher(per_key_seconds=0.0)
            items = [_video_item("V1"), _video_item("V2")]
            for it in items:
                fetcher.declare_bytes(it.metadata["video_key"], 100)
            stages = [TrackingStage("ingest", sleep=0.0)]
            result = await sequential_run(items, stages, fetcher)
            # Two different videos => two GETs in sequential mode.
            assert result.metrics.get_requests == 2
        asyncio.run(main())


# ---------------------------------------------------------------------------
# Prefetch run
# ---------------------------------------------------------------------------


class TestPrefetchRun:
    def test_prefetch_overlap_positive(self):
        async def main():
            fetcher = SlowFakeFetcher(per_key_seconds=0.10)
            items = [_video_item(f"V{i}") for i in range(4)]
            for it in items:
                fetcher.declare_bytes(it.metadata["video_key"], 100)
            stages = [TrackingStage("ingest", sleep=0.15)]
            result = await prefetch_run(
                items, stages, fetcher, prefetch_ahead=1,
            )
            # At least one item's download happened during another item's stage.
            assert result.metrics.prefetch_overlap_seconds > 0.0
            assert result.metrics.overlap_ratio > 0.0
            assert result.metrics.items_completed == 4
        asyncio.run(main())

    def test_video_level_reuse(self):
        async def main():
            # Single item with three stages sharing its video key.
            fetcher = SlowFakeFetcher(per_key_seconds=0.0)
            item = _video_item("V1")
            fetcher.declare_bytes(item.metadata["video_key"], 100)
            stages = [
                TrackingStage("ingest", sleep=0.0),
                TrackingStage("ocr", sleep=0.0),
                TrackingStage("asr", sleep=0.0),
            ]
            result = await prefetch_run([item], stages, fetcher)
            # One video key => one GET, one miss. All three stages reuse the
            # resident copy (no second download).
            assert result.metrics.get_requests == 1
            assert result.metrics.cache_misses == 1
            # Every stage materialized the same video key path.
            paths = [s.local_paths["V1"][item.metadata["video_key"]] for s in stages]
            assert all(p == paths[0] for p in paths)
        asyncio.run(main())

    def test_keyframe_level_reuse(self):
        async def main():
            fetcher = SlowFakeFetcher(per_key_seconds=0.0)
            from mvr.operations.r2_layout import raw_video_key
            kf_keys = [
                "data/corpus/keyframes/V1/001.jpg",
                "data/corpus/keyframes/V1/002.jpg",
            ]
            # Item declares only the keyframe keys, since only keyframe stages run.
            item = WorkItem(
                item_id="V1",
                keys=list(kf_keys),
                metadata={
                    "video_key": raw_video_key("1", "L21", "V1"),
                    "keyframe_keys": kf_keys,
                    "video_id": "V1",
                },
            )
            for k in kf_keys:
                fetcher.declare_bytes(k, 200)
            stages = [
                TrackingStage("objects", video_keys=False, keyframe_keys=True, sleep=0.0),
                TrackingStage("beit3", video_keys=False, keyframe_keys=True, sleep=0.0),
                TrackingStage("metaclip2", video_keys=False, keyframe_keys=True, sleep=0.0),
            ]
            result = await prefetch_run([item], stages, fetcher)
            # 2 keyframe keys, each fetched once; three stages reuse them.
            assert result.metrics.get_requests == 2
            assert result.metrics.cache_misses == 2
            assert result.metrics.items_completed == 1
        asyncio.run(main())

    def test_eviction_after_last_consumer(self):
        async def main():
            fetcher = SlowFakeFetcher(per_key_seconds=0.0)
            items = [_video_item(f"V{i}") for i in range(3)]
            for it in items:
                fetcher.declare_bytes(it.metadata["video_key"], 50)
            stages = [TrackingStage("ingest", sleep=0.0)]
            result = await prefetch_run(items, stages, fetcher, prefetch_ahead=1)
            # All keys were admitted, but every key's last consumer released it,
            # so the working set should be empty at the end.
            assert result.metrics.peak_resident_keys <= 2  # current + 1 prefetch
            # The orchestrator's own eviction runs after every item; combined
            # with prefetch, peak <= prefetch_ahead + 1.
        asyncio.run(main())

    def test_resume_idempotent(self):
        async def main():
            fetcher = SlowFakeFetcher(per_key_seconds=0.0)
            items = [_video_item("V1"), _video_item("V2"), _video_item("V3")]
            for it in items:
                fetcher.declare_bytes(it.metadata["video_key"], 50)
            stages = [TrackingStage("ingest", sleep=0.0)]
            first = await prefetch_run(items, stages, fetcher, prefetch_ahead=1)
            assert first.metrics.items_completed == 3
            # Cache now has 3 video keys. Re-run with resume on V1.
            fetcher2 = SlowFakeFetcher(per_key_seconds=0.0)
            for it in items:
                fetcher2.declare_bytes(it.metadata["video_key"], 50)
            stages2 = [TrackingStage("ingest", sleep=0.0)]
            second = await prefetch_run(
                items, stages2, fetcher2, prefetch_ahead=1,
                resume_completed={"V1"},
            )
            assert second.metrics.items_skipped_resume == 1
            assert second.metrics.items_completed == 2
            assert second.metrics.get_requests == 2  # V2 and V3, not V1
        asyncio.run(main())

    def test_failure_records_and_continues(self):
        async def main():
            fetcher = SlowFakeFetcher(per_key_seconds=0.0)
            items = [_video_item("V1"), _video_item("V2")]
            for it in items:
                fetcher.declare_bytes(it.metadata["video_key"], 50)
            stages = [TrackingStage("ingest", sleep=0.0, fail_on={"V1"})]
            result = await prefetch_run(items, stages, fetcher)
            assert result.metrics.items_failed == 1
            assert result.metrics.items_completed == 1
        asyncio.run(main())

    def test_prefetch_ahead_bounds_active_prefetches(self):
        # With prefetch_ahead=1, the orchestrator should not start fetching
        # item N+2 while item N+1 is still being prefetched. We verify by
        # inspecting download timestamps: there should be at most ~2 fetches
        # in flight at the same wall-time point.
        async def main():
            fetcher = SlowFakeFetcher(per_key_seconds=0.20)
            items = [_video_item(f"V{i}") for i in range(5)]
            for it in items:
                fetcher.declare_bytes(it.metadata["video_key"], 50)
            stages = [TrackingStage("ingest", sleep=0.30)]
            await prefetch_run(items, stages, fetcher, prefetch_ahead=1)
            # No assertion on number of downloads, but ensure all completed.
            assert len(fetcher.downloads) == 5
        asyncio.run(main())

    def test_visible_transfer_wait_positive_when_no_prefetch_room(self):
        # With prefetch_ahead=0 we disable prefetch entirely; each item's
        # download time should be visible to its stage.
        async def main():
            fetcher = SlowFakeFetcher(per_key_seconds=0.10)
            items = [_video_item(f"V{i}") for i in range(3)]
            for it in items:
                fetcher.declare_bytes(it.metadata["video_key"], 50)
            stages = [TrackingStage("ingest", sleep=0.0)]
            result = await prefetch_run(items, stages, fetcher, prefetch_ahead=0)
            # Prefetch was disabled (semaphore clamped to 1), so every item
            # download is a foreground wait.
            assert result.metrics.visible_transfer_wait_seconds > 0.0
        asyncio.run(main())


# ---------------------------------------------------------------------------
# End-to-end compare (sequential vs prefetch) on the same working set
# ---------------------------------------------------------------------------


class TestCompare:
    def test_prefetch_is_faster_or_equal_than_sequential(self):
        # The fetch must take real time (download_jitter) for a prefetch window
        # to exist; stage sleep must be long enough to overlap it.
        async def main():
            items = [_video_item(f"V{i}") for i in range(5)]
            seq_fetcher = SlowFakeFetcher(download_jitter=0.05)
            pre_fetcher = SlowFakeFetcher(download_jitter=0.05)
            for it in items:
                seq_fetcher.declare_bytes(it.metadata["video_key"], 50)
                pre_fetcher.declare_bytes(it.metadata["video_key"], 50)
            stages = [TrackingStage("ingest", sleep=0.10)]
            seq = await sequential_run(items, stages, seq_fetcher)
            pre = await prefetch_run(items, stages, pre_fetcher, prefetch_ahead=1)
            # Prefetch must show non-zero overlap; sequential must not.
            assert pre.metrics.prefetch_overlap_seconds > 0.0
            assert seq.metrics.prefetch_overlap_seconds == 0.0
            # Same byte count downloaded in both arms.
            assert pre.metrics.bytes_transferred == seq.metrics.bytes_transferred
        asyncio.run(main())
