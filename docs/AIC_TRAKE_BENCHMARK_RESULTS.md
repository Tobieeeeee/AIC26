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
| `wide_gap_bulletin_segments` | L21_V008 | 4 | three unrelated bulletin segments minutes apart | 4620, 12367, 19710, 26652 |

`suspect_arrest` was chosen deliberately because two of its events look
nearly identical to a human (same dark scene, same subjects), to give
Lucifer's skip mechanism a realistic chance to matter rather than
benchmarking only easy chains.

`wide_gap_bulletin_segments` was added after the first five were found to
share a blind spot: their real inter-event gaps are all 8-23 s, comfortably
inside any sane budget, so the gap penalty never fired once and ATLAS's and
Lucifer's two-sided `[min,max]` machinery — a core part of both algorithms
— was never actually exercised. Its real gaps are 310 s / 294 s / 278 s.
Its third event (the studio anchor) also recurs many times across the
bulletin, so only the timing constraint can select the right occurrence.

All gap budgets are derived through Phase 03's `translate_gap_phrase()`
rather than hard-coded, so the benchmark runs exactly what production would
run for the same query text.

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

### 4. The gap constraint was never exercised, and the values used were wrong

The original five queries all had real gaps of 8-23 s against budgets that
never bound, so `alpha`, `beta` and the entire two-sided penalty were inert
— a large part of both DP strategies was being reported on without being
run. Adding `wide_gap_bulletin_segments` (gaps of 310/294/278 s) exercises
it, and the mechanism demonstrably decides the answer:

| gap budget on that query | ATLAS final | chain picked |
|---|---|---|
| `(0, 60)` — far too tight | 0.250 | 4620, 5987, 7104, 7859 |
| `(30, 300)` — from `"một lúc sau"` | 0.500 | 10775, 12390, 19710, 26652 |
| effectively unconstrained | **0.750** | 4620, 12390, 19710, 26652 |

That single query suggested constraints only hurt. Measuring all six says
the opposite, and the initial hand-picked budgets were the real problem:

| gap configuration (ATLAS, 6 queries) | mean final |
|---|---|
| hand-picked `(0, 60)` per query | 0.633 |
| **`translate_gap_phrase()` no-clue default `(0, 300)`** | **0.639** |
| no gap constraint at all | 0.594 |

So the constraint contributes **+0.045 over running unconstrained** —
`house_fire_alley` alone swings 0.600 to 0.267 without it — and the phrase
table's own default beats values chosen by hand. The benchmark now derives
every budget from the table.

### One hypothesis that was tested, rejected, then partly reinstated

The 1-frame tolerance was first suspected of artificially depressing
scores, and on `flood_embankment` it clearly does not: widening tolerance
from 1 frame to 450 frames (15 s) leaves that score at 0.500 throughout,
because its missed markers are 16-27 s away — genuinely different scenes.

The wide-gap query shows that conclusion does not generalise. Its best
chain is `(4620, 12390, 19710, 26652)` against a ground truth of
`(4620, 12367, 19710, 26652)`: three markers exact, one **23 frames
(0.92 s) off** — the adjacent keyframe. At the default 1-frame tolerance
that scores 0.750; at 25 frames (1 s) it scores **1.000**.

| tolerance | wide-gap final |
|---|---|
| 1 frame (0.04 s) | 0.750 |
| 25 frames (1 s) | 1.000 |
| 250 frames (10 s) | 1.000 |

The absolute scores in this report are therefore genuinely sensitive to the
organizer's real (still unknown) tolerance, in a way the first check
suggested they were not. Relative strategy ordering is unaffected — all
three face the same tolerance.

## Parameter sweeps (real data, 5 queries)

```text
ATLAS   top_n:    1 -> 0.417   3 -> 0.630   5 -> 0.660  10 -> 0.660  20 -> 0.660
LUCIFER min_sim: 0.05 -> 0.660  0.15 -> 0.660  0.20 -> 0.670  0.22 -> 0.530  0.25 -> 0.473  0.28 -> 0.403
UCESE   topk:     1 -> 0.190   3 -> 0.300   5 -> 0.380  10 -> 0.413  20 -> 0.253
```

Defaults were set from these measurements, not guessed: ATLAS `top_n=5`,
Lucifer `min_sim=0.20`, U-CESE `topk=10`.

## Final results

6 queries, gap budgets from `translate_gap_phrase()`, defaults as shipped.

| Strategy | mean final score | R@1 | R@5 | R@20 | mean latency | p95 latency |
|---|---|---|---|---|---|---|
| **ATLAS** | **0.639** | 0.292 | 0.694 | 0.736 | **0.372 s** | 0.444 s |
| Lucifer | **0.639** | 0.292 | 0.694 | 0.736 | 0.410 s | 0.489 s |
| U-CESE | 0.344 | 0.250 | 0.306 | 0.389 | 0.372 s | 0.432 s |

Per-query final scores:

| Query | ATLAS | Lucifer | U-CESE |
|---|---|---|---|
| traffic_accident_night | 0.600 | 0.600 | 0.300 |
| flood_embankment | 0.550 | 0.550 | 0.500 |
| house_fire_alley | 0.533 | 0.533 | 0.267 |
| ocean_debris_cleanup | 0.900 | 0.900 | 0.500 |
| suspect_arrest | 0.750 | 0.750 | 0.500 |
| wide_gap_bulletin_segments | 0.500 | 0.500 | 0.000 |

## Recommended production default: ATLAS

**ATLAS and Lucifer are exactly tied** — identical mean, identical value on
every single query, identical at every cutoff. With the phrase table's
wider `(0, 300)` budgets in play, penalties rarely bind, full chains are
always available, and Lucifer's skip path stops firing, so it reduces to
ATLAS. (An earlier run with narrower hand-picked budgets showed Lucifer at
0.670 vs ATLAS 0.660 — a one-marker difference on five queries, i.e. noise,
and it did not survive using realistic budgets.)

With quality identical, the decision rests on cost and robustness, and
ATLAS wins both:

- **Faster:** 0.372 s vs 0.410 s mean per query.
- **ATLAS sits on a parameter plateau.** Its only tunable, `top_n`, gives an
  identical score anywhere from 5 to 20, and its `min_sim` default of 0.0
  introduces no sensitivity at all.
- **Lucifer sits on a cliff.** `min_sim` 0.20 → 0.670, 0.22 → 0.530, 0.25 →
  0.473 (measured on the five-query set). The one good value is immediately
  adjacent to a sharp drop; below it, Lucifer degenerates into ATLAS anyway.

In a competition where parameters cannot be re-tuned per query, a flat
plateau beats a knife edge that never actually wins.

### Lucifer's skip mechanism, measured in isolation

Because the six benchmark queries never trigger a skip, the mechanism was
verified separately in
`test_lucifer_skip_beats_atlas_when_an_unfindable_event` — deliberately
kept out of the benchmark mean, since an impossible query is not a
representative one and averaging it in would distort the comparison rather
than inform it.

The test takes `house_fire_alley`'s real first and third events and splices
an arctic scene between them that nothing in a Vietnamese city bulletin can
match. Its ground-truth marker for that event is set beyond the end of the
video: unscoreable by construction, so the maximum achievable score is 2/3
and what is actually being measured is whether the impossible event drags
its two findable neighbours down with it.

| strategy | best `L21_V018` chain | final |
|---|---|---|
| ATLAS | `(13108, 16534, 17730)` | 0.000 |
| Lucifer | `(13108, `**`None`**`, 17730)` | **0.267** |

Lucifer declines to invent a frame; ATLAS has no skip path and is
structurally forced to pick one, and that forced pick constrains its other
two markers. So the mechanism is real and it does pay off — **but only
under a condition none of the six realistic queries produced.** That is why
it does not change the production default.

**Lucifer stays worth revisiting** if real competition queries turn out to
contain unfindable events more often than this benchmark suggests — that is
the only condition under which it can beat ATLAS, and even `suspect_arrest`,
chosen for its near-identical events, did not create it.
U-CESE is clearly behind (0.344)
even after its parameter fix, consistent with the lab's own description of
it as a fast manual-review suggestion generator rather than a primary
scored strategy.

## Limitations

- **Six queries.** Differences under roughly 0.05 in mean final score should
  not be treated as meaningful.
- **Parameters were tuned on the same queries they are evaluated on.** There
  is no held-out split. The chosen defaults are best read as "reasonable and
  measured", not "validated".
- **Ground truth is internal.** Six self-authored chains, not organizer
  labels.
- **Tolerance is unverified and it matters.** `TrakeGroundTruth.tolerance`
  defaults to 1 frame, matching the lab reference's own `tol=1`. The
  organizer's real value is unknown, and the wide-gap query above shows a
  0.92 s near-miss being scored as a total failure (0.750 at tol=1 vs 1.000
  at tol=25). Absolute scores here should be read as a floor under the
  strictest plausible tolerance.
- **Shortlisting is untested at scale.** This corpus has 29 videos, and
  `shortlist_top_k` of 20, 100 and 500 all produce an identical 0.660 —
  every video is shortlisted every time, so Phase 05's narrowing step is
  effectively a no-op here and unverified against a competition-sized
  corpus.
- **Semantic-only backend.** The hybrid/Elasticsearch path was not
  benchmarked.
- **Lucifer's skip mechanism never fires on realistic queries.** Under
  realistic gap budgets a full chain is always available, which is precisely
  why it ties ATLAS. It is verified separately (see above) on a
  deliberately impossible query, where it does beat ATLAS — but no
  representative query in this set produces that condition, so how often it
  would matter in the real competition is unknown.

## Trigger condition for the deferred fourth strategy

`04_CHAIN_ALIGNMENT_ALGORITHMS.md` defers the AIC 2025 winner's multi-stage
windowed + harmonic-mean approach, to be revisited only if the three lab
strategies underperform badly. That condition is **not** met: ATLAS
resolves a non-zero chain for all five queries and reaches R@20 = 0.733.
No action needed.
