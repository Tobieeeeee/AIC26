# AIC TRAKE — Real-Data Benchmark Results

Phase 06 of the TRAKE task pack. Compares the three chain-alignment
strategies on real AIC L21 data to pick an evidence-backed production
default, the same way the KIS benchmark chose semantic-only retrieval over
exact-frame refinement.

**Ground truth is internal/self-authored, NOT organizer-official.** Every
target frame below was read directly from the organizer's own
`map_keyframes/<video_id>.csv` `frame_idx` column for a keyframe whose
image was visually inspected first — no frame number was inferred or
guessed. Do not present these scores as an official competition result.

Reproduce with:

```powershell
$env:MVR_RUN_AIC_TRAKE_E2E="1"; $env:MVR_AIC_DATASET_ROOT="E:\AIC\AIC_DATASET_ROOT"
python -m pytest tests/integration/test_aic_trake_e2e.py -q -s
```

## Setup

- Index: `artifacts/indexes/aic_l21` — 7800 vectors, 29 real L21 videos,
  `ViT-B-32` / `laion2b_s34b_b79k`, exact `IndexFlatIP`.
- Retrieval backend: semantic-only FAISS. The hybrid/Elasticsearch branch is
  out of scope for TRAKE dense scoring (no local vector store to
  reconstruct from — see `docs/AIC_TRAKE_IMPLEMENTATION_NOTES.md`).
- 5 queries, k = 2..4 events, 5 different videos, 5 different topics.

## Benchmark queries

| Query id | Video | k | Topic | GT frames |
|---|---|---|---|---|
| `traffic_accident_night` | L21_V024 | 4 | night traffic accident scene | 13890, 14130, 14430, 14635 |
| `flood_embankment` | L21_V015 | 4 | flooding / embankment overflow | 8279, 8636, 8988, 9378 |
| `house_fire_alley` | L21_V018 | 3 | house fire in a narrow alley | 11068, 11340, 11600 |
| `ocean_debris_cleanup` | L21_V010 | 2 | divers collecting seabed debris | 9150, 9730 |
| `suspect_arrest` | L21_V026 | 4 | suspect arrest at night | 16083, 16355, 16697, 17021 |

`suspect_arrest` was chosen deliberately because two of its events look
nearly identical to a human (same dark scene, same subjects), to give
Lucifer's skip mechanism a realistic chance to matter rather than
benchmarking only easy chains.

## Query language: English, and why that is not a stylistic choice

The first benchmark run used Vietnamese event text and scored **0.000 on
every query for every strategy**. Root cause: the index's embedding model
(`ViT-B-32` / `laion2b_s34b_b79k`, see
`artifacts/indexes/aic_l21/manifest.json`) is English-trained, not
multilingual. Measured on identical target frames in `L21_V010`:

| Event | Vietnamese text | English text |
|---|---|---|
| diver collecting seabed debris (frame 9150) | rank 192 / 268 | **rank 1 / 268** |
| loading trash onto a boat (frame 9730) | rank 144 / 268 | **rank 2 / 268** |

This is a property of the current index, not of TRAKE. It is recorded here
because it silently produces a total-zero benchmark that looks like a
broken pipeline, and because it means **any future TRAKE query path must
translate Vietnamese competition queries to English before retrieval**
while this index/model is in use.

## Three structural defects found and fixed before the numbers were trusted

The first complete run produced ATLAS 0.417 / Lucifer 0.417 / U-CESE 0.100.
Those numbers were discarded: investigation showed two of the three
strategies were running with parameters that made them structurally
incapable of a fair result.

### 1. U-CESE's `topk` default was invalid at per-video scope

The lab's `topk=300` selected an event's strongest frames across the whole
corpus. This port is per-video (Phase 04 redesign), and a typical AIC video
contributes only ~200-320 indexed frames, so at `topk=300` essentially
every frame qualified as a hit for every event and the first-touch rule
collapsed to "earliest frame in the window". Measured output on
`L21_V015`: `(8820, 8820, 8820, 8820)` — the same frame for all four
events, scoring 0.000.

| topk | score | distinct frames chosen |
|---|---|---|
| 300 (lab default) | 0.000 | 1 of 4 |
| 20 | 0.000 | 1 of 4 |
| 5 | 0.500 | 2 of 4 |

U-CESE's original 0.100 was therefore a parameter artifact, not a verdict
on the algorithm.

### 2. Lucifer's `min_sim` default made its skip mechanism unreachable

Real CLIP cosine scores on this index span roughly -0.05..0.31, with
correct target frames at 0.21..0.31. The lab's `min_sim=0.05` sits below
essentially every frame, so a full chain was always available and the skip
path — Lucifer's entire reason to exist — never fired. Its output was
byte-identical to ATLAS on all five queries.

This also explains an earlier misreading: Lucifer originally measured
~12.7 s/query versus ATLAS's ~0.42 s, which looked like an inherent
algorithmic cost. With a meaningful `min_sim` the DP state space is pruned
and Lucifer runs at ~0.43 s/query. **The "30x slower" finding was an
artifact of the same bad default, not a property of the algorithm.**

### 3. One chain per video wasted 71% of the submission budget

Every strategy returned exactly one chain per video (faithful to the lab
reference), so a 29-video shortlist produced 29 predictions against a
100-prediction allowance — and, more importantly, the correct video got
exactly one attempt. This is why `R@1 == R@100` in the original run.

All three strategies now return up to `top_n` distinct ranked chains per
video. Effect on ATLAS, holding everything else fixed:

| top_n | mean final score |
|---|---|
| 1 (lab behavior) | 0.417 |
| 3 | 0.630 |
| **5** | **0.660** |
| 10 | 0.660 |
| 20 | 0.660 |

A **+58% relative gain** purely from using the ranking budget, and the
plateau from 5 onward is flat.

### One hypothesis that was tested and rejected

The tolerance of 1 frame was suspected of artificially depressing scores.
It does not: holding the ATLAS chain fixed on `flood_embankment` and
widening tolerance from 1 frame to 450 frames (15 s) leaves the score at
0.500 throughout. The missed markers are 16-27 s away — genuinely
different scenes, not near-misses. The absolute scores below are real, not
a metric artifact.

## Parameter sweeps (real data, 5 queries)

```text
ATLAS   top_n:    1 -> 0.417   3 -> 0.630   5 -> 0.660  10 -> 0.660  20 -> 0.660
LUCIFER min_sim: 0.05 -> 0.660  0.15 -> 0.660  0.20 -> 0.670  0.22 -> 0.530  0.25 -> 0.473  0.28 -> 0.403
UCESE   topk:     1 -> 0.190   3 -> 0.300   5 -> 0.380  10 -> 0.413  20 -> 0.253
```

Defaults were set from these measurements, not guessed: ATLAS `top_n=5`,
Lucifer `min_sim=0.20`, U-CESE `topk=10`.

## Final results

| Strategy | mean final score | R@1 | R@5 | R@20 | mean latency | p95 latency |
|---|---|---|---|---|---|---|
| ATLAS | 0.660 | 0.417 | 0.683 | 0.733 | 0.371 s | 0.442 s |
| Lucifer | **0.670** | **0.467** | 0.683 | 0.733 | 0.434 s | 0.538 s |
| U-CESE | 0.413 | 0.300 | 0.367 | 0.467 | 0.371 s | 0.482 s |

Per-query final scores:

| Query | ATLAS | Lucifer | U-CESE |
|---|---|---|---|
| traffic_accident_night | 0.350 | 0.350 | 0.300 |
| flood_embankment | 0.700 | 0.750 | 0.500 |
| house_fire_alley | 0.600 | 0.600 | 0.267 |
| ocean_debris_cleanup | 0.900 | 0.900 | 0.500 |
| suspect_arrest | 0.750 | 0.750 | 0.500 |

## Recommended production default: ATLAS

Lucifer scores 0.670 against ATLAS's 0.660. That gap is **one marker in one
query across a five-query set** — noise, not a result, and the two are
identical from R@5 onward (0.683 / 0.733). The decision therefore rests on
robustness, where the two differ sharply:

- **ATLAS sits on a plateau.** Its only tunable, `top_n`, gives an identical
  0.660 anywhere from 5 to 20, and its `min_sim` default of 0.0 introduces
  no sensitivity at all.
- **Lucifer sits on a cliff.** `min_sim=0.20` gives 0.670; 0.22 gives 0.530;
  0.25 gives 0.473. The one good value is immediately adjacent to a sharp
  drop, and it was tuned on five queries. Below 0.20 Lucifer stops skipping
  entirely and degenerates into ATLAS.

In a competition where parameters cannot be re-tuned per query, a flat
plateau beats a knife edge that happens to peak 0.01 higher on a small
sample. ATLAS is also marginally faster (0.371 s vs 0.434 s mean).

**Lucifer remains a genuine close second**, and its better R@1 (0.467 vs
0.417) is worth revisiting if the benchmark set grows large enough for a
0.01 difference to carry signal. U-CESE is clearly behind (0.413) even
after its parameter fix, consistent with the lab's own description of it as
a fast manual-review suggestion generator rather than a primary scored
strategy.

## Limitations

- **Five queries.** Differences under roughly 0.05 in mean final score
  should not be treated as meaningful. The ATLAS/Lucifer ordering is
  explicitly within that noise band.
- **Parameters were tuned on the same five queries they are evaluated on.**
  There is no held-out split. The chosen defaults are best read as
  "reasonable and measured", not "validated".
- **Ground truth is internal.** Five self-authored chains, not organizer
  labels.
- **Tolerance is unverified.** `TrakeGroundTruth.tolerance` defaults to 1
  frame, matching the lab reference's own `tol=1`. The organizer's real
  tolerance is unknown; it was shown above not to affect these particular
  results, but that finding does not generalize.
- **Semantic-only backend.** The hybrid/Elasticsearch path was not
  benchmarked.
- **Lucifer's skip mechanism is still lightly exercised.** Even
  `suspect_arrest`, chosen for its near-identical events, did not force
  ATLAS to fail where Lucifer succeeded.

## Trigger condition for the deferred fourth strategy

`04_CHAIN_ALIGNMENT_ALGORITHMS.md` defers the AIC 2025 winner's multi-stage
windowed + harmonic-mean approach, to be revisited only if the three lab
strategies underperform badly. That condition is **not** met: ATLAS
resolves a non-zero chain for all five queries and reaches R@20 = 0.733.
No action needed.
