"""DATA-03 worker prefetch orchestration.

Layered on the DATA-02 primitives (R2Client / BoundedCache / R2Sync). This module
owns *intent* — which object keys a stage needs, who still needs them, and when
bytes may be evicted — while the DATA-02 cache owns the bytes themselves.

Two ideas do the work:

1. Bounded N+1 prefetch. While item N is processing, item N+1's bytes are fetched
   on a background task. `prefetch_ahead` bounds how far ahead we run, so a fast
   network cannot outrun a slow GPU and fill the disk.

2. Working-set reuse. A key downloaded for one stage stays resident for every
   later same-worker stage that declared it, and is evicted only after the last
   declared consumer releases it.

No full-corpus sync: only keys explicitly declared by the queued items are fetched.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Protocol, Sequence

LOGGER = logging.getLogger(__name__)

DEFAULT_PREFETCH_AHEAD = 1


class Fetcher(Protocol):
    """Materializes an object key locally. Implemented by stage_local.R2StageFetcher."""

    async def fetch(self, object_key: str) -> "FetchOutcome": ...


@dataclass
class FetchOutcome:
    """Result of materializing one object key."""

    object_key: str
    local_path: Path
    size_bytes: int
    cache_hit: bool
    get_requests: int = 0
    head_requests: int = 0
    seconds: float = 0.0


@dataclass
class ResidentEntry:
    """One key held in the working set, with the consumers that still need it."""

    object_key: str
    local_path: Path
    size_bytes: int
    pending_consumers: set[str] = field(default_factory=set)
    active_consumers: set[str] = field(default_factory=set)

    @property
    def refcount(self) -> int:
        return len(self.active_consumers)

    @property
    def releasable(self) -> bool:
        """True once no consumer holds it and none is still expected to."""
        return not self.active_consumers and not self.pending_consumers


@dataclass
class WorkerMetrics:
    """Measured run counters. Every field is observed, never estimated."""

    run_mode: str = "prefetch"
    items_total: int = 0
    items_completed: int = 0
    items_failed: int = 0
    items_skipped_resume: int = 0
    stages_run: int = 0

    wall_seconds: float = 0.0
    # Time a foreground stage was blocked waiting for bytes it did not have.
    # This is the cost prefetch exists to remove.
    visible_transfer_wait_seconds: float = 0.0
    # Time spent processing while at least one prefetch was in flight.
    prefetch_overlap_seconds: float = 0.0
    processing_seconds: float = 0.0

    get_requests: int = 0
    head_requests: int = 0
    bytes_transferred: int = 0

    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    evicted_bytes: int = 0

    peak_resident_keys: int = 0
    peak_resident_bytes: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    @property
    def overlap_ratio(self) -> float:
        """Share of processing time that had a prefetch in flight."""
        return (
            self.prefetch_overlap_seconds / self.processing_seconds
            if self.processing_seconds > 0
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "items_total": self.items_total,
            "items_completed": self.items_completed,
            "items_failed": self.items_failed,
            "items_skipped_resume": self.items_skipped_resume,
            "stages_run": self.stages_run,
            "wall_seconds": round(self.wall_seconds, 4),
            "visible_transfer_wait_seconds": round(
                self.visible_transfer_wait_seconds, 4
            ),
            "prefetch_overlap_seconds": round(self.prefetch_overlap_seconds, 4),
            "processing_seconds": round(self.processing_seconds, 4),
            "overlap_ratio": round(self.overlap_ratio, 4),
            "get_requests": self.get_requests,
            "head_requests": self.head_requests,
            "bytes_transferred": self.bytes_transferred,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "evictions": self.evictions,
            "evicted_bytes": self.evicted_bytes,
            "peak_resident_keys": self.peak_resident_keys,
            "peak_resident_bytes": self.peak_resident_bytes,
        }


class WorkingSet:
    """Explicit residency + refcount state for the keys a worker is using.

    The bytes live in the DATA-02 BoundedCache; this tracks who needs them so a
    key is evicted only after its last declared consumer releases it. All state
    transitions take the same lock, so prefetch and foreground stages can race
    freely.
    """

    def __init__(self, *, max_bytes: int | None = None) -> None:
        self.max_bytes = max_bytes
        self._resident: dict[str, ResidentEntry] = {}
        self._lock = asyncio.Lock()
        self._space = asyncio.Condition(self._lock)
        self.evictions = 0
        self.evicted_bytes = 0
        self.peak_keys = 0
        self.peak_bytes = 0

    @property
    def resident_bytes(self) -> int:
        return sum(e.size_bytes for e in self._resident.values())

    @property
    def resident_keys(self) -> list[str]:
        return sorted(self._resident)

    def is_resident(self, object_key: str) -> bool:
        """True if the key's bytes are locally present right now (no locking)."""
        entry = self._resident.get(object_key)
        return entry is not None and entry.size_bytes > 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "resident_keys": len(self._resident),
            "resident_bytes": self.resident_bytes,
            "max_bytes": self.max_bytes,
            "evictions": self.evictions,
            "evicted_bytes": self.evicted_bytes,
            "entries": [
                {
                    "object_key": e.object_key,
                    "size_bytes": e.size_bytes,
                    "refcount": e.refcount,
                    "active_consumers": sorted(e.active_consumers),
                    "pending_consumers": sorted(e.pending_consumers),
                }
                for e in sorted(self._resident.values(), key=lambda x: x.object_key)
            ],
        }

    async def declare(self, object_key: str, consumers: Iterable[str]) -> None:
        """Register that these consumers will eventually need this key.

        Declaring before fetching is what makes safe eviction possible: a key is
        never dropped while a later stage of the same item still intends to read it.
        """
        async with self._lock:
            entry = self._resident.get(object_key)
            if entry is None:
                entry = ResidentEntry(
                    object_key=object_key, local_path=Path(), size_bytes=0
                )
                self._resident[object_key] = entry
            entry.pending_consumers.update(consumers)

    async def admit(
        self, object_key: str, local_path: Path, size_bytes: int
    ) -> None:
        """Record that a key's bytes are now resident locally."""
        async with self._lock:
            entry = self._resident.get(object_key)
            if entry is None:
                entry = ResidentEntry(
                    object_key=object_key, local_path=local_path, size_bytes=size_bytes
                )
                self._resident[object_key] = entry
            else:
                entry.local_path = local_path
                entry.size_bytes = size_bytes
            self._note_peak()

    async def acquire(self, object_key: str, consumer: str) -> Path:
        """Mark a consumer as actively reading a key. Must already be resident."""
        async with self._lock:
            entry = self._resident.get(object_key)
            if entry is None:
                raise KeyError(
                    f"{object_key} is not resident; fetch it before acquiring"
                )
            entry.pending_consumers.discard(consumer)
            entry.active_consumers.add(consumer)
            return entry.local_path

    async def release(self, object_key: str, consumer: str) -> bool:
        """Release one consumer's hold. Returns True if that was the last one."""
        async with self._lock:
            entry = self._resident.get(object_key)
            if entry is None:
                return False
            entry.active_consumers.discard(consumer)
            entry.pending_consumers.discard(consumer)
            return entry.releasable

    def _note_peak(self) -> None:
        self.peak_keys = max(self.peak_keys, len(self._resident))
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)

    async def evict_releasable(
        self, *, delete: Callable[[Path], None] | None = None
    ) -> list[str]:
        """Evict every key whose last declared consumer has released it.

        This is the safe-eviction rule: no active reader, and no later stage still
        expecting the bytes. Anything else stays resident for reuse.
        """
        async with self._lock:
            evicted = self._evict_releasable_locked(delete)
            if evicted:
                self._space.notify_all()
        return evicted

    def _evict_releasable_locked(
        self, delete: Callable[[Path], None] | None
    ) -> list[str]:
        evicted: list[str] = []
        for key in list(self._resident):
            entry = self._resident[key]
            if not entry.releasable:
                continue
            if delete is not None and entry.size_bytes > 0:
                try:
                    delete(entry.local_path)
                except OSError as exc:
                    LOGGER.warning("failed to evict %s: %s", entry.local_path, exc)
            del self._resident[key]
            self.evictions += 1
            self.evicted_bytes += entry.size_bytes
            evicted.append(key)
        return evicted

    async def reserve_space(
        self,
        size_bytes: int,
        *,
        delete: Callable[[Path], None] | None = None,
        timeout: float = 300.0,
    ) -> float:
        """Block until `size_bytes` fits in the working-set budget.

        This is the backpressure valve. Returns the seconds spent waiting so the
        caller can account for it honestly — a prefetch that had to wait for room
        did not overlap anything.
        """
        if self.max_bytes is None or size_bytes <= 0:
            return 0.0
        started = time.perf_counter()
        deadline = started + timeout
        async with self._lock:
            while self.resident_bytes + size_bytes > self.max_bytes:
                self._evict_releasable_locked(delete)
                if self.resident_bytes + size_bytes <= self.max_bytes:
                    break
                if not self._resident:
                    # Nothing left to evict; the item is larger than the budget.
                    # Admit it rather than deadlocking on an impossible reservation.
                    LOGGER.warning(
                        "working-set budget %d < item %d; admitting anyway",
                        self.max_bytes,
                        size_bytes,
                    )
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out reserving {size_bytes} bytes in working set"
                    )
                try:
                    await asyncio.wait_for(self._space.wait(), timeout=remaining)
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"timed out reserving {size_bytes} bytes in working set"
                    ) from exc
        return time.perf_counter() - started


class Stage(Protocol):
    """One processing step a worker runs against an item.

    A stage declares the object keys it consumes (so the working set can reason
    about reuse and eviction) and does its work. The orchestrator only depends on
    this interface, so it stays unit-testable with fakes and never imports models.
    """

    name: str

    def keys_for(self, item: "WorkItem") -> Sequence[str]:
        """Object keys this stage reads for the given item."""
        ...

    async def run(self, item: "WorkItem", paths: dict[str, Path]) -> "StageResult":
        """Process the item given the locally-resident paths for its keys."""
        ...


@dataclass
class WorkItem:
    """One unit of work (a video) plus its declared object keys.

    `keys` is the full set of object keys every stage of this item will need.
    Declaring them up front is what lets the working set admit and evict keys
    with full knowledge of future demand.
    """

    item_id: str
    keys: list[str]
    estimated_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Outcome of one stage run against one item."""

    stage: str
    item_id: str
    success: bool
    seconds: float = 0.0
    error: str | None = None


@dataclass
class RunResult:
    """Outcome of a full worker run."""

    metrics: WorkerMetrics
    stage_results: list[StageResult] = field(default_factory=list)
    items: list[WorkItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "items_total": len(self.items),
            "stage_results": [
                {
                    "stage": r.stage,
                    "item_id": r.item_id,
                    "success": r.success,
                    "seconds": round(r.seconds, 4),
                    "error": r.error,
                }
                for r in self.stage_results
            ],
        }


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


async def sequential_run(
    items: Sequence[WorkItem],
    stages: Sequence[Stage],
    fetcher: Fetcher,
    *,
    working_set: WorkingSet | None = None,
    resume_completed: set[str] | None = None,
    on_delete: Callable[[Path], None] | None = None,
) -> RunResult:
    """Run every stage of every item in order, with no prefetch.

    This is the baseline arm: each item is fully materialized and processed
    before the next begins, so there is no overlap and no cross-item reuse.
    """
    metrics = WorkerMetrics(run_mode="sequential")
    metrics.items_total = len(items)
    working_set = working_set or WorkingSet()
    resume_completed = resume_completed or set()
    results: list[StageResult] = []
    wall_start = time.perf_counter()

    for item in items:
        if item.item_id in resume_completed:
            metrics.items_skipped_resume += 1
            continue
        consumers = [_consumer_id(s.name, item.item_id) for s in stages]
        item_start = time.perf_counter()
        try:
            await _materialize_item(
                item, fetcher, working_set, metrics, on_delete, consumers
            )
            for stage in stages:
                outcome = await _run_stage(stage, item, working_set, metrics)
                results.append(outcome)
                if not outcome.success:
                    metrics.items_failed += 1
                    break
            else:
                metrics.items_completed += 1
        finally:
            await working_set.evict_releasable(delete=on_delete)
        metrics.processing_seconds += time.perf_counter() - item_start

    metrics.wall_seconds = time.perf_counter() - wall_start
    _finalize_metrics(metrics, working_set)
    return RunResult(metrics=metrics, stage_results=results, items=list(items))


async def prefetch_run(
    items: Sequence[WorkItem],
    stages: Sequence[Stage],
    fetcher: Fetcher,
    *,
    prefetch_ahead: int = DEFAULT_PREFETCH_AHEAD,
    working_set: WorkingSet | None = None,
    resume_completed: set[str] | None = None,
    on_delete: Callable[[Path], None] | None = None,
) -> RunResult:
    """Bounded N+1 prefetch run.

    A bounded number of background tasks (`prefetch_ahead`) materialize upcoming
    items while the foreground processes the current one. The semaphore is the
    backpressure valve: once `prefetch_ahead` fetches are in flight, the next one
    waits, so a fast network cannot outrun a slow GPU and fill the disk.
    """
    metrics = WorkerMetrics(run_mode="prefetch")
    metrics.items_total = len(items)
    working_set = working_set or WorkingSet()
    resume_completed = resume_completed or set()
    results: list[StageResult] = []
    wall_start = time.perf_counter()
    ahead = asyncio.Semaphore(max(prefetch_ahead, 1))
    prefetch_lock = asyncio.Lock()
    # Overlap is measured precisely: we record the wall-clock intervals during
    # which a prefetch was in flight and, separately, during which a foreground
    # stage was processing, then intersect them at the end. This is race-free —
    # it does not depend on sampling a flag at the right instant.
    prefetch_intervals: list[list[float]] = []
    processing_intervals: list[tuple[float, float]] = []
    active_prefetches = 0

    async def _prefetch_one(idx: int) -> None:
        nonlocal active_prefetches
        if idx >= len(items):
            return
        item = items[idx]
        if item.item_id in resume_completed:
            return
        async with prefetch_lock:
            active_prefetches += 1
            if active_prefetches == 1:
                prefetch_intervals.append([time.perf_counter(), 0.0])
        try:
            consumers = [_consumer_id(s.name, item.item_id) for s in stages]
            await _materialize_item(
                item, fetcher, working_set, metrics, on_delete, consumers
            )
        finally:
            async with prefetch_lock:
                active_prefetches -= 1
                if active_prefetches == 0 and prefetch_intervals:
                    prefetch_intervals[-1][1] = time.perf_counter()

    for i, item in enumerate(items):
        if item.item_id in resume_completed:
            metrics.items_skipped_resume += 1
            continue

        # Launch prefetches for the next `prefetch_ahead` items, bounded.
        prefetch_tasks: list[asyncio.Task[None]] = []
        for offset in range(1, prefetch_ahead + 1):
            idx = i + offset
            if idx >= len(items) or items[idx].item_id in resume_completed:
                continue
            await ahead.acquire()
            task = asyncio.create_task(_prefetch_one(idx))

            def _release(t: asyncio.Task[None], sem: asyncio.Semaphore = ahead) -> None:
                sem.release()

            task.add_done_callback(_release)
            prefetch_tasks.append(task)

        item_start = time.perf_counter()
        try:
            # If this item was prefetched, this is a cache hit and costs nothing.
            consumers = [_consumer_id(s.name, item.item_id) for s in stages]
            await _materialize_item(
                item, fetcher, working_set, metrics, on_delete, consumers
            )
            proc_start = time.perf_counter()
            for stage in stages:
                outcome = await _run_stage(stage, item, working_set, metrics)
                results.append(outcome)
                if not outcome.success:
                    metrics.items_failed += 1
                    break
            else:
                metrics.items_completed += 1
            processing_intervals.append((proc_start, time.perf_counter()))
        finally:
            await working_set.evict_releasable(delete=on_delete)
        metrics.processing_seconds += time.perf_counter() - item_start

        if prefetch_tasks:
            await asyncio.gather(*prefetch_tasks, return_exceptions=True)

    # Close any prefetch interval still open (shouldn't happen after gather).
    for iv in prefetch_intervals:
        if iv[1] == 0.0:
            iv[1] = time.perf_counter()
    metrics.prefetch_overlap_seconds = _interval_intersection(
        [(a, b) for a, b in prefetch_intervals], processing_intervals
    )
    metrics.wall_seconds = time.perf_counter() - wall_start
    _finalize_metrics(metrics, working_set)
    return RunResult(metrics=metrics, stage_results=results, items=list(items))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _materialize_item(
    item: WorkItem,
    fetcher: Fetcher,
    working_set: WorkingSet,
    metrics: WorkerMetrics,
    on_delete: Callable[[Path], None] | None,
    stage_consumers: list[str],
) -> None:
    """Ensure every key of an item is resident, declaring future demand first.

    Called at most twice per item (once by a prefetch task, once by the
    foreground). The second call sees every key already resident and does no
    network I/O — that reuse is exactly what prefetch buys.
    """
    consumers = set(stage_consumers)
    for key in item.keys:
        await working_set.declare(key, consumers)
    for key in item.keys:
        if working_set.is_resident(key):
            metrics.cache_hits += 1
            continue
        started = time.perf_counter()
        outcome = await fetcher.fetch(key)
        wait = time.perf_counter() - started
        if outcome.cache_hit:
            metrics.cache_hits += 1
        else:
            metrics.cache_misses += 1
            metrics.get_requests += outcome.get_requests
            metrics.head_requests += outcome.head_requests
            metrics.bytes_transferred += outcome.size_bytes
            metrics.visible_transfer_wait_seconds += wait
        await working_set.admit(key, outcome.local_path, outcome.size_bytes)


def _consumer_id(stage_name: str, item_id: str) -> str:
    return f"{stage_name}:{item_id}"


async def _run_stage(
    stage: Stage,
    item: WorkItem,
    working_set: WorkingSet,
    metrics: WorkerMetrics,
) -> StageResult:
    """Acquire the keys a stage needs, run it, then release them."""
    paths: dict[str, Path] = {}
    needed = list(stage.keys_for(item))
    cid = _consumer_id(stage.name, item.item_id)
    started = time.perf_counter()
    try:
        for key in needed:
            paths[key] = await working_set.acquire(key, cid)
        result = await stage.run(item, paths)
        metrics.stages_run += 1
        return StageResult(
            stage=stage.name,
            item_id=item.item_id,
            success=result.success,
            seconds=time.perf_counter() - started,
            error=result.error,
        )
    except Exception as exc:
        metrics.stages_run += 1
        return StageResult(
            stage=stage.name,
            item_id=item.item_id,
            success=False,
            seconds=time.perf_counter() - started,
            error=str(exc),
        )
    finally:
        for key in needed:
            await working_set.release(key, cid)


def _finalize_metrics(
    metrics: WorkerMetrics, working_set: WorkingSet
) -> None:
    metrics.evictions = working_set.evictions
    metrics.evicted_bytes = working_set.evicted_bytes
    metrics.peak_resident_keys = working_set.peak_keys
    metrics.peak_resident_bytes = working_set.peak_bytes


def _interval_intersection(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Total wall time covered by both interval sets (their overlap).

    Used to measure how much processing time genuinely coincided with an
    in-flight prefetch, without depending on sampling a flag at the right moment.
    """
    total = 0.0
    for a0, a1 in a:
        for b0, b1 in b:
            lo = max(a0, b0)
            hi = min(a1, b1)
            if hi > lo:
                total += hi - lo
    return total
