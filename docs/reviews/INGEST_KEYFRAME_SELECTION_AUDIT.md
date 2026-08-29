# INGEST-KF01 — Exact Keyframe Selection Audit

## 1. Repository state

- Audited HEAD: `37696aa568ce634458ee21bd91fc9a6a863576bf`
- Branch: `fix/ocr-production-checkpoints`
- Audit date: 2026-08-22 (Asia/Saigon)
- Scope: audit report and audit-only tests. No production semantics were changed.
- Worktree note: unrelated pre-existing untracked files were present and were not modified. Generated fixture output was written below ignored `artifacts/verification/ingest_kf01/`.

## 2. Files and callsites inspected

The static trace covered:

- `src/mvr/ingestion/keyframes.py:6-88` — selector and metadata construction.
- `src/mvr/ingestion/models.py:17-56` — inclusive shot model, duration, and serialized timestamps.
- `src/mvr/ingestion/pipeline.py:26-99` — detector-to-selector-to-extractor call flow.
- `src/mvr/ingestion/shot_detection/__init__.py:8-18` — detector factory.
- `src/mvr/ingestion/shot_detection/pyscenedetect.py:6-70` and `transnetv2.py:9-87` — boundary producers.
- `src/mvr/ingestion/extract.py` — exact requested-frame extraction.
- `src/mvr/cli.py:49-73,371-387` — package CLI defaults and dispatch.
- `scripts/run_ingestion.py:5-31` — standalone script defaults and dispatch.
- `configs/default.yaml:1-6` — declarative ingestion values.
- Existing ingestion tests under `tests/unit/ingestion/` and `tests/unit/test_cli.py`.
- New executable audit locks in `tests/benchmark/test_ingest_keyframe_selection_audit.py`.
- Read-only source lab `shot-boundary-lab-student/student/src/shotlab/keyframes.py` and its tests. `git diff --no-index` found no content difference from the production selector except the repository file's final newline.

Repository-wide searches found exactly two runnable ingestion argument parsers: `mvr.cli ingest` and `scripts/run_ingestion.py`. No code loads `configs/default.yaml`; it is documentation/configuration data, not an active config-driven entrypoint.

## 3. Exact algorithm

For a shot with zero-based inclusive bounds `start_frame = s`, `end_frame = e`, and FPS `f`:

```text
require f > 0, e >= s, long_threshold >= 0, interval > 0
frame_count = e - s + 1
duration = frame_count / f
midpoint = (s + e) // 2
selected = {midpoint}

if duration >= long_threshold:
    step = max(1, round(interval * f))
    candidate = s + step
    while candidate < e:
        selected.add(candidate)
        candidate += step

return sorted(i for i in selected if s <= i <= e)
```

Exact consequences:

- Every valid shot produces at least one keyframe: its midpoint.
- The midpoint uses integer floor division over the inclusive endpoints. Equivalently it is `s + (frame_count - 1) // 2`; for an even frame count it selects the lower of the two central frames.
- A shot is long at equality: `duration >= long_shot_seconds`.
- Periodic sampling starts one full step after the shot start, not at the start and not relative to the midpoint.
- `round` is Python's built-in nearest-integer rounding, including ties-to-even (`round(2.5) == 2`, `round(1.5) == 2`). `max(1, ...)` prevents a zero-frame step.
- Periodic candidates use `candidate < end_frame`; the inclusive shot end is deliberately excluded from the periodic sequence. The midpoint remains bounded with inclusive checks.
- Selection is performed in frame space. There is no independent timestamp-to-frame conversion. A `Keyframe.timestamp` is subsequently computed as the absolute video timestamp `frame_idx / fps`.
- A set suppresses midpoint/periodic collisions, after which numerical sorting determines order. Thus adding one frame can reduce the count: at 30 fps under 12s/8s, a 480-frame shot selects `[239, 240]`, while a 481-frame shot selects only `[240]` because the midpoint and first periodic sample collide and the next periodic point equals the excluded end.
- `make_keyframes` enumerates the sorted indices from 1 independently for each shot. IDs are `<video_id>_S<shot_id:04d>_K<ordinal:02d>`; for example `AUDIT_S0007_K01`, `K02`, and so on. The image path is `keyframes/<video_id>/<keyframe_id>.jpg`.
- Raw `Keyframe.timestamp` retains the division result; `Keyframe.to_dict()` and persisted metadata round it to six decimal places.

For one shot, the count is the cardinality of `{midpoint} ∪ {s + k*step | k >= 1 and s + k*step < e}`. There is no fixed maximum other than that implied by shot length and step.

## 4. Defaults by entrypoint

| Entrypoint/profile | Detector default | Primary detector | Long-shot threshold | Sample interval | Other relevant defaults |
|---|---|---|---:|---:|---|
| `python -m mvr.cli ingest` | `pyscenedetect` | `pyscenedetect` | 12s | 8s | PyScene 27.0; TransNet 0.5; device `auto`; methods list contains PyScene only |
| `python scripts/run_ingestion.py` | `pyscenedetect` | `pyscenedetect` | 15s | 5s | PyScene 27.0; TransNet 0.5; device `cpu`; methods is comma-separated text |
| `configs/default.yaml` | `pyscenedetect` | implied by `shot_detector` | 12s | 8s | PyScene 27.0; TransNet 0.5; no runtime loader/callsite exists |
| `select_frame_indices(...)` direct API defaults | supplied in `Shot` | supplied in `Shot` | 12s | 8s | `make_keyframes(...)` and `run_pipeline(...)` themselves require explicit density arguments |

The current runnable defaults are legacy PySceneDetect defaults. TransNetV2 is an optional selectable detector (`--methods transnetv2 --primary transnetv2`); there is no active entrypoint whose default is the purported project-target TransNetV2 backend. Selecting TransNetV2 does not implicitly select a different keyframe-density profile.

## 5. Deterministic boundary tests

All rows below were generated through the real `make_keyframes` implementation with the 12s/8s profile. Timestamps shown are raw relative timestamps because these synthetic shots start at frame 0. “Epsilon” is one frame at the stated FPS.

| Case | FPS | Frames / actual duration | Count | Exact frame indices | Relative timestamps (seconds) |
|---|---:|---:|---:|---|---|
| 0.1s | 30 | 3 / 0.1s | 1 | `[1]` | `[1/30 = 0.033333…]` |
| 1s | 25 | 25 / 1s | 1 | `[12]` | `[0.48]` |
| threshold − ε | 25 | 299 / 11.96s | 1 | `[149]` | `[5.96]` |
| threshold | 25 | 300 / 12s | 2 | `[149, 200]` | `[5.96, 8]` |
| threshold + ε | 25 | 301 / 12.04s | 2 | `[150, 200]` | `[6, 8]` |
| 2 × interval | 30 | 480 / 16s | 2 | `[239, 240]` | `[7.966667, 8]` |
| 2 × interval + ε | 30 | 481 / 16.033333…s | 1 | `[240]` | `[8]` |
| nominal 30s at 29.97 | 29.97 | 899 / 29.996663…s | 4 | `[240, 449, 480, 720]` | `[8.008008, 14.981648, 16.016016, 24.024024]` |
| nominal 60s at 29.97 | 29.97 | 1798 / 59.993327…s | 8 | `[240, 480, 720, 898, 960, 1200, 1440, 1680]` | `[8.008008, 16.016016, 24.024024, 29.963297, 32.032032, 40.040040, 48.048048, 56.056056]` |

The audit tests additionally lock between-frame step rounding:

- 25 fps × 0.1s = 2.5 frames → step 2 (ties-to-even).
- 30 fps × 0.05s = 1.5 frames → step 2 (ties-to-even).
- 29.97 fps × 0.1s = 2.997 frames → step 3.
- An exact periodic candidate at `end_frame` is excluded, and a midpoint/periodic collision receives one ordinal only.

## 6. Exact duration examples at 30 fps

These values were generated by the real selector with start frame 0 and `end_frame = duration*30 - 1`. Displayed timestamps are `frame_idx / 30`, rounded to six decimals as persisted metadata does.

### Midpoint-only comparison

This is an audit comparison, not an active default; periodic sampling was disabled by setting the long-shot threshold above 60 seconds.

| Shot duration | Count | Frame indices | Relative timestamps (s) |
|---:|---:|---|---|
| 5 | 1 | `[74]` | `[2.466667]` |
| 10 | 1 | `[149]` | `[4.966667]` |
| 12 | 1 | `[179]` | `[5.966667]` |
| 15 | 1 | `[224]` | `[7.466667]` |
| 20 | 1 | `[299]` | `[9.966667]` |
| 30 | 1 | `[449]` | `[14.966667]` |
| 45 | 1 | `[674]` | `[22.466667]` |
| 60 | 1 | `[899]` | `[29.966667]` |

### Package CLI and declarative config: 12s/8s

| Shot duration | Count | Frame indices | Relative timestamps (s) |
|---:|---:|---|---|
| 5 | 1 | `[74]` | `[2.466667]` |
| 10 | 1 | `[149]` | `[4.966667]` |
| 12 | 2 | `[179, 240]` | `[5.966667, 8]` |
| 15 | 2 | `[224, 240]` | `[7.466667, 8]` |
| 20 | 3 | `[240, 299, 480]` | `[8, 9.966667, 16]` |
| 30 | 4 | `[240, 449, 480, 720]` | `[8, 14.966667, 16, 24]` |
| 45 | 6 | `[240, 480, 674, 720, 960, 1200]` | `[8, 16, 22.466667, 24, 32, 40]` |
| 60 | 8 | `[240, 480, 720, 899, 960, 1200, 1440, 1680]` | `[8, 16, 24, 29.966667, 32, 40, 48, 56]` |

### `scripts/run_ingestion.py`: 15s/5s

| Shot duration | Count | Frame indices | Relative timestamps (s) |
|---:|---:|---|---|
| 5 | 1 | `[74]` | `[2.466667]` |
| 10 | 1 | `[149]` | `[4.966667]` |
| 12 | 1 | `[179]` | `[5.966667]` |
| 15 | 3 | `[150, 224, 300]` | `[5, 7.466667, 10]` |
| 20 | 4 | `[150, 299, 300, 450]` | `[5, 9.966667, 10, 15]` |
| 30 | 6 | `[150, 300, 449, 450, 600, 750]` | `[5, 10, 14.966667, 15, 20, 25]` |
| 45 | 9 | `[150, 300, 450, 600, 674, 750, 900, 1050, 1200]` | `[5, 10, 15, 20, 22.466667, 25, 30, 35, 40]` |
| 60 | 12 | `[150, 300, 450, 600, 750, 899, 900, 1050, 1200, 1350, 1500, 1650]` | `[5, 10, 15, 20, 25, 29.966667, 30, 35, 40, 45, 50, 55]` |

Thus “current CLI” and explicit 12s/8s are identical, while “current script” and explicit 15s/5s are identical.

## 7. Real-video sample

The real package CLI was run end-to-end with its defaults:

```text
.venv/Scripts/python.exe -m mvr.cli ingest <fixture> \
  --output artifacts/verification/ingest_kf01/cli_12_8 --device cpu
```

Fixture: read-only source-lab video `L01_V001.mp4`, 41,162 bytes, SHA-256 `b7470e9cd2e22cf8fdbf14c6cf334cf46346919f78d328ad59709158ce7d8855`. Runtime detector: PySceneDetect 0.7.1 at threshold 27.0. The path performed probing, real detection, real selection, FFmpeg extraction, and metadata writing successfully.

The selector writes absolute video timestamps. Relative time below is measured from each shot's start.

| Shot | Inclusive frames | Duration | KF count | Frame indices | Absolute timestamps | Relative timestamps |
|---:|---|---:|---:|---|---|---|
| 1 | 0–24 | 1.0s | 1 | `[12]` | `[0.48]` | `[0.48]` |
| 2 | 25–54 | 1.2s | 1 | `[39]` | `[1.56]` | `[0.56]` |
| 3 | 55–89 | 1.4s | 1 | `[72]` | `[2.88]` | `[0.68]` |
| 4 | 90–109 | 0.8s | 1 | `[99]` | `[3.96]` | `[0.36]` |
| 5 | 110–134 | 1.0s | 1 | `[122]` | `[4.88]` | `[0.48]` |

Summary: 5 shots; 5 keyframes; mean 1.0; median 1; p95 1; maximum 1; 100% of shots have exactly one keyframe; 0% have more than one.

TransNetV2 could not be run because `transnetv2_pytorch` is not installed in this environment. This is an optional runtime limitation, not a gap in the selector proof. PySceneDetect was available and was run.

## 8. Detector versus selector responsibility

**Does TransNetV2 decide the final keyframe timestamps/frame indices? NO.**

The proven call graph is:

```text
video
  -> create_detector(method, threshold, device)
  -> detector.detect(video) returns Shot boundaries and FPS
  -> pipeline selects only the primary detector's Shot objects
  -> make_keyframes(primary_shots, caller_threshold, caller_interval)
  -> select_frame_indices(shot, ...)
  -> extract_keyframes(video, requested Keyframe objects, output_root)
```

Changing PySceneDetect to TransNetV2 can change the number of shots, each shot's inclusive start/end frames, shot IDs, and therefore the midpoint/periodic results induced by those different boundaries. It does not change the sampling formula, density arguments, timestamp formula, deduplication, ordering, or ordinal assignment. `select_frame_indices` never branches on `Shot.method`. The audit test passes otherwise identical PySceneDetect- and TransNetV2-labelled `Shot` objects through the real selector and locks identical results. The existing pipeline test separately proves that `primary="transnetv2"` supplies TransNetV2 boundaries to the same `make_keyframes` call.

## 9. Keyframe-density sensitivity on the real fixture

Each profile below was applied through real `make_keyframes` calls to the exact five detected shots above.

| Profile | Shots | Keyframes | KF/shot | Increase vs midpoint-only |
|---|---:|---:|---:|---:|
| Midpoint-only | 5 | 5 | 1.0 | 0% |
| Current package CLI | 5 | 5 | 1.0 | 0% |
| Current `scripts/run_ingestion.py` | 5 | 5 | 1.0 | 0% |
| Explicit 12s/8s | 5 | 5 | 1.0 | 0% |
| Explicit 15s/5s | 5 | 5 | 1.0 | 0% |

This fixture contains no shot reaching 12 seconds, so it validates the real path but cannot discriminate long-shot density. The synthetic duration tables show the deterministic density differences without extrapolating any Batch1 count.

## 10. Default inconsistencies

There are two conflicting runnable defaults: 12s/8s in the package CLI and 15s/5s in the standalone script. The declarative config agrees with the package CLI but is not consumed by either path. Consequently, the command used to launch ingestion changes keyframe density while keeping the same nominal detector defaults, and generated artifacts do not currently carry a named canonical density profile.

No production behavior was changed in this audit. A canonical project ingestion profile should be explicitly selected and then wired consistently into both entrypoints and configuration in a separate implementation task.

## 11. Recommendation classification

**NEEDS_EXPLICIT_PROJECT_DEFAULT**

This classification is based on the proven entrypoint inconsistency, not on a claim that either density is better. Selecting the best density requires a separate retrieval-quality experiment that holds detector boundaries and models fixed while measuring:

```text
keyframe density
  versus retrieval recall / KIS success
  versus extraction time / storage / embedding and index cost
```

That follow-up should use representative evaluation queries and long shots. Intuition or this five-shot short fixture cannot choose an optimal density.

## 12. Verification and final verdict

Focused command:

```text
.venv/Scripts/python.exe -m pytest \
  tests/benchmark/test_ingest_keyframe_selection_audit.py \
  tests/unit/ingestion/test_keyframes.py \
  tests/unit/ingestion/test_pipeline.py \
  tests/unit/test_cli.py -q
```

The audit tests invoke the real selector and real entrypoint parsers; they contain literal expected indices, timestamps, ordinals, IDs, rounding cases, endpoint behavior, duplicate suppression, detector-label invariance, duration examples, and both runnable default profiles.

Result: **37 passed in 0.23s**. `git diff --check` exited 0. `git status --short` was inspected; only the two audit deliverables were added by this task, while all unrelated pre-existing untracked work remained untouched. No merge, push, or commit was performed.

**Final verdict: NEEDS_EXPLICIT_PROJECT_DEFAULT**

The exact current selector behavior is proven. The unresolved issue is choosing and declaring one canonical project default, not ambiguity in what the existing code does.
