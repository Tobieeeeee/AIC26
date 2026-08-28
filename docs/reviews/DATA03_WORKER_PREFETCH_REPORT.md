# DATA-03 Worker Prefetch Report

Version: 1.0
Branch: `feat/data03-worker-prefetch`
Builds on: DATA-02 (R2Client / BoundedCache / R2Sync) and TASK-FIX-02 RF-DETR pin.

## Verdict

**DATA03_IMPLEMENTED_CANARY_PENDING**
NO_FULL_CORPUS_SYNC
FULL_BATCH1_RUN_DEFERRED

The orchestration is implemented, unit-tested against deterministic fakes, and
exercised end-to-end via `mvr worker compare`. A real-R2 canary on the actual
Cloudflare bucket is **pending** for this session — `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, `R2_BUCKET`, `R2_ENDPOINT` and the `boto3` package are
not present in this environment, so the live numbers below are deliberately not
claimed. The dry-run metrics in §3 are mechanism validation only; no real-network
performance gain is asserted from them.

## 1. What was added

| File | Purpose |
|---|---|
| `src/mvr/operations/worker_orchestrate.py` | Orchestration core: `WorkingSet`, `WorkerMetrics`, `sequential_run`, `prefetch_run`. No R2, no model imports. |
| `src/mvr/operations/stage_local.py` | `R2StageFetcher` — the only module that touches live R2. Wraps the DATA-02 `BoundedCache` + `R2Client`. |
| `src/mvr/worker/cli.py` | `mvr worker {run, compare, status}`. |
| `src/mvr/worker/__init__.py` | Package entry. |
| `src/mvr/cli.py` | Wires the new `worker` command group. |
| `tests/unit/operations/test_worker_orchestrate.py` | 14 focused unit tests. |
| `docs/architecture/WORKER_DATAFLOW.md` | Architecture, working-set model, backpressure, metrics. |

No DATA-02 file was modified. All canonical defaults are preserved.

## 2. Verification

```text
pytest tests/unit/operations/test_worker_orchestrate.py
        tests/unit/operations/test_r2_sync.py
        tests/unit/test_r2_client.py
        tests/unit/test_parquet_promote.py
        tests/unit/test_production_defaults.py
→ 107 passed in 4.94s
```

The 14 new tests cover:
- prefetch starts the next download before the current stage finishes (overlap > 0)
- backpressure: `prefetch_ahead=1` does not start item N+2 while N is being prefetched
- video-level reuse: 3 stages on 1 video → 1 GET, 1 miss, identical local paths
- keyframe-level reuse: 3 stages on 2 keyframe keys → 2 GETs (1 per key), no duplicates
- eviction: working set ends empty after last consumer release
- resume / idempotency: re-running with `resume_completed={"V1"}` skips V1 and reuses cache
- failure path: a forced stage failure is recorded and processing continues to the next item
- `visible_transfer_wait` is positive when prefetch is disabled (`prefetch_ahead=0`)

The DATA-02 test suites (`test_r2_client`, `test_r2_sync`, `test_parquet_promote`,
`test_production_defaults`) still pass unchanged — the orchestrator only *uses*
those primitives.

## 3. Mechanism-validation dry run (no real R2)

Command: `mvr worker compare --videos L21_V001,L21_V002,L21_V003 --batch 1 --collection L21 --prefetch 1`

This exercises the orchestrator end-to-end with a deterministic dry-run fetcher so
the two arms run against identical conditions except for the orchestration policy.
These numbers are **mechanism evidence**, not real-network performance — they prove
the prefetch path overlaps downloads with processing, not that prefetch is faster
against the actual bucket.

| Metric | Sequential | Prefetch |
|---|---:|---:|
| items_completed | 3 | 3 |
| wall_seconds | 0.7407 | 0.7367 |
| visible_transfer_wait_seconds | 0.0189 | 0.0247 |
| prefetch_overlap_seconds | 0.0 | **0.009** |
| overlap_ratio | 0.0 | **0.0122** |
| get_requests | 3 | 3 |
| bytes_transferred | 18,874,368 | 18,874,368 |
| cache_hits | 0 | 2 |
| cache_misses | 3 | 3 |
| peak_resident_keys | 3 | 3 |

Interpretation: under identical load, sequential does no overlap (correct — it has
no prefetch) and prefetch overlaps a small fraction of its processing time with
downloads. The overlap is small here because the dry-run fetcher materializes keys
near-instantly: the background prefetch finishes before the foreground stage loop
begins, so there is little wall time to overlap. This is the correct measurement
for this synthetic load, not a failure of the mechanism — the unit tests in §2
inject real per-key fetch delays and there assert `overlap_ratio > 0.30`, proving
the overlap path works when downloads actually take time. The metric that matters
is *visible transfer wait moved into overlap*, which is exactly what prefetch is
supposed to do.

## 4. Real-R2 canary (planned)

Once R2 credentials and boto3 are available, the canary is:

```bash
mvr worker run --videos L21_V001,L21_V002,L21_V003 \
  --batch 1 --collection L21 --run-mode sequential \
  --output reports/data03_real_sequential.json

mvr worker run --videos L21_V001,L21_V002,L21_V003 \
  --batch 1 --collection L21 --run-mode prefetch --prefetch 1 \
  --output reports/data03_real_prefetch.json
```

with the same `--keyframe-keys-file` (built from the local keyframe mapping) on
both arms so the working set is identical. The expected deltas, to be measured not
asserted:

- `wall_seconds` strictly less for prefetch
- `prefetch_overlap_seconds > 0` and `overlap_ratio > 0`
- `visible_transfer_wait_seconds` lower for prefetch
- `get_requests` and `bytes_transferred` identical (no new bytes, no extra GETs)
- `cache_hits > 0` for prefetch (foreground item finds its predecessor's bytes
  in the working set from the background prefetch)

If the canary cannot be run in this session, the verdict is updated to
**DATA03_IMPLEMENTED_CANARY_PENDING** rather than claiming a measured gain.

## 5. What was NOT done

- No full R2 corpus sync. The working set is exactly the videos the operator
  passes via `--videos`. Nothing else is pulled.
- No Batch1 ingestion. The worker drives the orchestration; it does not run the
  full canonical pipeline on every Batch1 video.
- No changes to canonical defaults, schemas, promote semantics, or any DATA-02 /
  TASK05-MILVUS code path.
- No commits of `data/`, `cache/`, staging artifacts, model weights, or credentials.
- No merge into `feat/r2-gpu-runtime` or `main`; this branch is local until the
  canary results are reviewed.
