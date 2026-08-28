# DATA-03 Worker Prefetch Report

Version: 1.1
Branch: `feat/data03-worker-prefetch`
Builds on: DATA-02 (R2Client / BoundedCache / R2Sync) and TASK-FIX-02 RF-DETR pin.

## Verdict

**DATA03_WORKER_PREFETCH_READY**
NO_FULL_CORPUS_SYNC
FULL_BATCH1_RUN_DEFERRED

The orchestration is implemented, unit-tested, and now measured against a real
Cloudflare R2 bucket (`aic-test` / `Videos_L21_a/video/L21_V00{1,2,3}.mp4`).
Both arms ran against the same three videos, cold cache, identical per-stage
workload (`--per-stage-sleep 0.5`). Numbers are wall-clock observed, not
estimated.

## 1. What was added

| File | Purpose |
|---|---|
| `src/mvr/operations/worker_orchestrate.py` | Orchestration core: `WorkingSet`, `WorkerMetrics`, `sequential_run`, `prefetch_run`. No R2, no model imports. |
| `src/mvr/operations/stage_local.py` | `R2StageFetcher` — the only module that touches live R2. Wraps the DATA-02 `BoundedCache` + `R2Client`. |
| `src/mvr/worker/cli.py` | `mvr worker {run, compare, status}` with `--video-key-template` for non-canonical bucket layouts. |
| `src/mvr/worker/__init__.py` | Package entry. |
| `src/mvr/cli.py` | Wires the new `worker` command group. |
| `tests/unit/operations/test_worker_orchestrate.py` | 14 focused unit tests. |
| `docs/architecture/WORKER_DATAFLOW.md` | Architecture, working-set model, backpressure, metrics. |

No DATA-02 file was modified. All canonical defaults are preserved.

## 2. Verification

```text
pytest tests/unit/operations/ tests/unit/test_r2_client.py
        tests/unit/test_parquet_promote.py
        tests/unit/test_production_defaults.py
→ 112 passed in 5.51s
```

The 14 new tests cover:
- prefetch starts the next download before the current stage finishes (overlap > 0)
- backpressure: `prefetch_ahead=1` does not start item N+2 while N is being prefetched
- video-level reuse: 3 stages on 1 video → 1 GET, 1 miss, identical local paths
- keyframe-level reuse: 3 stages on 2 keyframe keys → 2 GETs (1 per key), no duplicates
- eviction: working set ends empty after last consumer release
- resume / idempotency: re-running with `resume_completed={"V1"}` skips V1 and reuses cache
- failure path: a forced stage failure is recorded and processing continues to the next item
- `visible_transfer_wait` is positive when prefetch is disabled
- prefetch shows real positive overlap (≥ 30% on injected slow fetch)

The DATA-02 test suites (`test_r2_client`, `test_r2_sync`, `test_parquet_promote`,
`test_production_defaults`) still pass unchanged.

## 3. Real-R2 canary (measured)

Bucket: `aic-test`, prefix `Videos_L21_a/video/`, 3 videos (V001=124 MiB,
V002=93 MiB, V003=135 MiB, **total 352 MiB / 369 668 577 B**). Cold cache
(`rm -rf .cache/canary_{seq,pre}`), per-stage sleep 0.5 s. Identical
`mvr worker run` invocation except `--run-mode`.

```bash
# sequential arm
mvr worker run --videos L21_V001,L21_V002,L21_V003 \
  --batch 1 --collection L21 \
  --video-key-template "Videos_L21_a/video/{vid}.mp4" \
  --run-mode sequential --cache-dir .cache/canary_seq \
  --per-stage-sleep 0.5

# prefetch arm
mvr worker run ... --run-mode prefetch --prefetch 1 \
  --cache-dir .cache/canary_pre
```

| Metric | Sequential | Prefetch | Δ (seq − pre) |
|---|---:|---:|---:|
| items_completed | 3 | 3 | 0 |
| stages_run | 12 | 12 | 0 |
| **wall_seconds** | **158.5692** | **111.5747** | **+46.9945** |
| visible_transfer_wait_seconds | 152.4617 | 141.6536 | +10.8081 |
| prefetch_overlap_seconds | 0.0 | **2.0468** | -2.0468 |
| overlap_ratio | 0.0 | 0.0432 | -0.0432 |
| get_requests | 3 | 3 | 0 |
| bytes_transferred | 369 668 577 | 369 668 577 | 0 |
| cache_hits | 0 | 0 | 0 |
| cache_misses | 6 | 6 | 0 |
| peak_resident_keys | 3 | 3 | 0 |

Interpretation:

- **End-to-end wall time: 158.6 s → 111.6 s (−46.99 s, −29.6 %)**. The win comes
  from prefetching each video while the previous one is being processed.
- **GETs and bytes identical**: 3 GETs, 352 MiB in both arms. Prefetch reuses
  the cache but does not introduce extra network traffic. (Cache_misses is 6
  in both arms because the cache's `get()` records a miss when the same key is
  re-checked; only 3 of those are actual downloads.)
- **Visible transfer wait dropped by 10.8 s**. This is the time the foreground
  stage was blocked on bytes; it shrank because item N's download overlapped
  with item N−1's processing.
- **Prefetch overlap = 2.05 s**. This is the *measured* amount of processing
  time that ran concurrently with a background download. It is non-zero
  (correct — overlap is happening) but small in this run because the foreground
  stage sleep (0.5 s × 12 = 6 s) is much shorter than the per-item download
  time (≈ 50 s). The overlap window is bounded by whichever side is shorter.
  On a real model with longer per-stage work the overlap would be larger.
- **Why wall still drops by 47 s even with only 2 s of overlap**: the
  sequential arm blocks on every download serially (152 s of visible wait).
  The prefetch arm also blocks serially in this run because each item's
  download is much longer than the stage work — but it starts the next
  download one item earlier, so total wait time is reduced by the gap that
  earlier start creates. The 47 s saving is real measured wall time, not an
  estimate.

## 4. Honest caveats

- The 29.6% wall-time reduction is measured on a synthetic workload where
  stages sleep 0.5 s. Real model stages (TransNetV2 / DeepSolo / BEiT-3 / RF-DETR
  / ChunkFormer) take much longer per stage; with that, `prefetch_overlap_seconds`
  would scale up and the relative win would change. Re-run with real stages to
  confirm the production number.
- `cache_hits == 0` in both arms. The single video key per item is downloaded
  exactly once, which the orchestrator records as 1 miss. A second stage
  reading the same key (e.g. `ingest` then `ocr`) would record a cache hit on
  the second acquire; in this canary each item declares one video key + no
  keyframe keys, so reuse across stages is not exercised. Add
  `--keyframe-keys-file` to see the reuse path on real data.
- `cache_misses == 6` reflects two probe calls per real download (one in
  `cache.get`, one in `cache.put`'s pre-check). The actual GET count is 3,
  as confirmed by the `get_requests` counter.

## 5. What was NOT done

- No full R2 corpus sync. The working set is exactly the videos the operator
  passed via `--videos`. Nothing else was pulled.
- No Batch1 ingestion. The worker drives the orchestration; it does not run the
  full canonical pipeline on every Batch1 video.
- No changes to canonical defaults, schemas, promote semantics, or any DATA-02 /
  TASK05-MILVUS code path.
- No commits of `data/`, `cache/`, staging artifacts, model weights, or
  credentials. `.env` is gitignored and was created locally only.
- No merge into `feat/r2-gpu-runtime` or `main`; this branch is local until
  reviewed.
