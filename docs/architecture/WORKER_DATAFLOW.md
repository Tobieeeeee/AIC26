# Worker Dataflow — DATA-03 Prefetch Orchestration

Version: 1.0
Builds on: DATA-02 shared-corpus primitives (R2Client / BoundedCache / R2Sync)
Status: Implemented; real-R2 canary pending (see
[DATA03_WORKER_PREFETCH_REPORT](../reviews/DATA03_WORKER_PREFETCH_REPORT.md)).

---

## What this adds

DATA-02 gave us the pieces that move bytes between Cloudflare R2 and the local
tree: an S3 client with retry + atomic download (`R2Client`), a bounded LRU cache
(`BoundedCache`), pull/push orchestration (`R2Sync`), the canonical key layout,
and the validate→promote contract.

DATA-03 adds **bounded N+1 prefetch orchestration** on top of those pieces so a
worker overlaps the download of the *next* video with processing of the *current*
one, reuses the same local bytes across the stages that a single worker runs, and
evicts only after the last consumer is done. It does **not** replace or modify any
DATA-02 code, and it never syncs the full R2 corpus — only the object keys the
queued work items explicitly declare.

Canonical model defaults (TransNetV2 @0.5 + `hybrid_temporal_content`, pinned RF-DETR,
BEiT-3 + MetaCLIP-2 + PE-Core-L, DeepSolo + Vietnamese PARSeq + F-1/F/F+1,
Silero VAD + ChunkFormer) are untouched — the orchestrator only sequences the I/O
around the stages that run those models; it does not choose or configure them.

---

## Module layout

```text
src/mvr/operations/worker_orchestrate.py   orchestration core (pure, no R2, no models)
src/mvr/operations/stage_local.py          real-R2 fetch adapter (the only I/O module)
src/mvr/worker/cli.py                       `mvr worker run|compare|status`
```

The split matters: `worker_orchestrate` depends only on a `Fetcher` protocol, so
its logic (reuse, refcounting, eviction, backpressure, metrics) is unit-tested
with a fake fetcher and never imports boto3 or any model.

---

## Working-set model

A worker processes a queue of **WorkItems** (one per video). Each item declares the
full set of object keys every one of its stages will read:

- the raw `.mp4` video key (`raw_video_key`) — read by ingestion, OCR, ASR
- the keyframe image keys (`corpus_keyframe_key`) — read by object detection and
  the three embedding providers

The **WorkingSet** tracks, per resident key: its local path, byte size, the set of
consumers still *expected* to read it (`pending_consumers`, set by `declare`), and
the set *currently* reading it (`active_consumers`, set by `acquire`). A key is
**releasable** — and therefore evictable — only when both sets are empty: no active
reader, and no later stage still expecting it.

```text
declare(key, consumers)   register future demand         (before any fetch)
admit(key, path, size)    record bytes are now local     (after fetch)
acquire(key, consumer)    a stage starts reading          pending--, active++
release(key, consumer)    a stage finishes                active--; returns last?
evict_releasable()        drop every releasable key       (safe eviction)
```

This ordering is what makes eviction safe: because demand is declared up front, a
key is never dropped while a later same-item stage still intends to read it.

---

## N+1 prefetch and backpressure

`prefetch_run` walks the queue. Before processing item *i*, it launches background
tasks to materialize items *i+1 … i+prefetch_ahead* into the shared cache. A
bounded `asyncio.Semaphore(prefetch_ahead)` is the backpressure valve — once that
many prefetches are in flight, the next one waits, so a fast network cannot outrun
a slow GPU and fill the disk.

An optional working-set byte budget (`WorkingSet(max_bytes=...)`,
`--working-set-mb`) adds a second backpressure layer: `reserve_space` blocks a
prefetch until releasable keys can be evicted to make room. Time spent blocked here
is counted as **visible transfer wait**, not overlap — a prefetch that had to wait
for room did not actually overlap anything.

```text
           ┌─ prefetch item i+1 (background) ─────────┐
process i: ├─ ingest ─ ocr ─ asr ─ objects ───────────┤   ← overlap window
           └──────────────────────────────────────────┘
process i+1: bytes already resident → cache hit, no wait
```

---

## Two levels of reuse

- **Video level** — ingestion, OCR, and ASR all declare the same `.mp4` key. The
  first stage's fetch is a cache miss (one GET); the other two are cache hits
  against the resident copy. One download serves three stages.
- **Keyframe level** — object detection and the BEiT-3 / MetaCLIP-2 / PE embedding
  stages all declare the same keyframe image keys. First consumer downloads; the
  rest reuse.

Reuse is realized through the DATA-02 `BoundedCache`: the orchestrator asks the
fetcher for a key, the fetcher returns the cached path on a hit, and the working
set records residency so no stage re-requests bytes that are already local.

---

## Eviction

After each item finishes all its stages, `evict_releasable` drops every key whose
last declared consumer has released it. Keys still needed by a not-yet-run stage of
a prefetched future item stay resident. The `BoundedCache` LRU remains the final
backstop for anything the working set does not explicitly evict (e.g. across runs).

---

## Metrics (measured, never estimated)

`WorkerMetrics` records only observed counters:

| Field | Meaning |
|---|---|
| `wall_seconds` | total run wall-clock |
| `visible_transfer_wait_seconds` | time a foreground stage blocked waiting for bytes — the cost prefetch removes |
| `prefetch_overlap_seconds` | processing time that had a prefetch in flight |
| `overlap_ratio` | overlap / processing time |
| `get_requests`, `head_requests` | R2 requests actually issued |
| `bytes_transferred` | bytes actually downloaded |
| `cache_hits`, `cache_misses`, `cache_hit_rate` | from `BoundedCache.stats` |
| `evictions`, `evicted_bytes` | working-set + cache evictions |
| `peak_resident_keys`, `peak_resident_bytes` | working-set high-water marks |

No performance-gain claim is made anywhere without a sequential-vs-prefetch
`compare` run producing these numbers on the actual data.

---

## Resume / idempotency

A worker records completed `item_id`s. On restart, `resume_completed` skips them:
their stages don't re-run and their bytes aren't re-fetched. Because promotion is
atomic and manifest-after-success (DATA-02 `promote.py`), a crash mid-item leaves
no partial canonical output — only fully-completed items are promoted, so a resume
is safe and produces the same corpus as an uninterrupted run.

---

## Operator entrypoint

```bash
# Prefetch run over an explicit working set (no full-corpus sync)
mvr worker run --videos L21_V001,L21_V002,L21_V003 \
  --batch 1 --collection L21 --run-mode prefetch --prefetch 1

# Baseline arm
mvr worker run ... --run-mode sequential

# Sequential vs prefetch on the same working set → metric deltas
mvr worker compare --videos L21_V001,L21_V002,L21_V003 --batch 1 --collection L21

# Cache + working-set state
mvr worker status
```

`--keyframe-keys-file` supplies per-video keyframe object keys (JSON: `{video_id:
[key, ...]}`) to exercise keyframe-level reuse; without it the worker operates at
video granularity only.

Real-R2 `run` requires `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`,
`R2_ENDPOINT` and the `boto3` package. `compare` and `status` run without
credentials using a deterministic dry-run fetcher for mechanism validation.
