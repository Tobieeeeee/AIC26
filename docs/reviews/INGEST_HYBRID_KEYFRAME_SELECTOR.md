# INGEST-KF04 — Canonical Hybrid Temporal-Content Selector

Date: 2026-08-23

Branch: `feat/ingest-hybrid-keyframe-selector`

Base: fetched `origin/main` at `e637af712b2e7f0e879c612abbac1c18e7fce73b`

Verdict: **CANONICAL_HYBRID_SELECTOR_READY**

## Decision and scope

The runnable production ingestion default is now
`hybrid_temporal_content`. It combines mandatory midpoint preservation,
pinned MetaCLIP-2 local novelty, a cheap grayscale/edge change proxy,
rare/transient preservation, and deterministic temporal repair. Every measured
shot satisfies the configured boundary/inter-keyframe maximum gap of 8 seconds.

The old midpoint plus 12s/8s periodic algorithm is unchanged and remains
available only through explicit `selector_mode=periodic`. No Transformer,
fixed first/middle/last, pure KNN-density selection, or implicit fallback was
introduced. PySceneDetect or TransNetV2 selects shot boundaries only; the same
post-boundary selector policy applies to either backend.

Canonical `keyframe_id`, `video_id`, `shot_id`, zero-based `frame_idx`,
timestamp, ordinal, image path, and CSV schema semantics are unchanged.

## Implementation

- `src/mvr/ingestion/hybrid_selector.py`: model-free deterministic candidate,
  novelty, event, and temporal-repair algorithm.
- `src/mvr/ingestion/selector_cache.py`: exact-frame candidate extraction,
  edge signal, normalized MetaCLIP cache validation, canonical JPEG
  materialization, provenance, and selected-vector publication.
- `src/mvr/ingestion/keyframes.py`: shared canonical identity formatter for an
  externally selected sorted frame set; legacy periodic functions are
  unchanged.
- `src/mvr/ingestion/pipeline.py`: production mode routing, fail-closed pinned
  provider construction, selector manifest, and multi-video orchestration.
- `src/mvr/cli.py`, `scripts/run_ingestion.py`, `configs/default.yaml`: aligned
  canonical defaults and explicit legacy controls.
- `src/mvr/enrichment/embeddings/offline.py` and
  `scripts/run_feature_extraction.py`: validate and reuse selector-produced
  MetaCLIP vectors when provider, revision, preprocessing, normalization,
  canonical identity/frame, and hashes match.
- `scripts/experiments/evaluate_hybrid_keyframe_selector.py`: fixed-shot,
  fixed-candidate one-video regression scorer.
- `tests/unit/ingestion/test_hybrid_selector.py`,
  `test_selector_cache.py`, and `test_hybrid_defaults_and_reuse.py`: CPU-only
  correctness, integration, cache, reuse, and default-alignment coverage.

## Canonical configuration

| Field | Default |
|---|---:|
| `selector.mode` | `hybrid_temporal_content` |
| `candidate_step_sec` | 1.0 |
| `max_temporal_gap_sec` | 8.0 |
| `clip_knn_k` | 3 |
| `clip_neighbor_window_sec` | 4.0 |
| `clip_novelty_threshold` | 0.15 |
| `clip_rare_threshold` | 0.25 |
| `min_event_gap_sec` | 2.0 |
| `edge_change_mad_multiplier` | 3.0 |
| `edge_change_min_score` | 0.08 |
| `midpoint_mandatory` | `true` |
| Legacy periodic threshold/interval | 12.0s / 8.0s |

All defaults agree in the library signature, `python -m mvr.cli ingest`,
`scripts/run_ingestion.py`, and `configs/default.yaml`.

## Exact deterministic algorithm

For an inclusive shot `[start_frame, end_frame]`, midpoint is
`(start_frame + end_frame) // 2`. Candidate step is
`max(1, round(candidate_step_sec * fps))`; the grid starts at
`start_frame + step`, advances in frame space while `< end_frame`, then unions
the exact midpoint, deduplicates, and sorts. Thus transition endpoints are
avoided unless a very short shot makes the midpoint itself a boundary.

Candidate embeddings use
`facebook/metaclip-2-worldwide-b16`, revision
`8cc8591f32b43a809f3325192d3b77aa56968ddb`, the repository provider's pinned
processor, contiguous `float32`, and explicit L2 normalization. For candidate
`c`, temporal neighbors are other candidates within 4 seconds. Novelty is:

```text
1 - mean(top 3 cosine similarities to local temporal neighbors)
```

Fewer neighbors use all available neighbors; no neighbor gives novelty zero.

The edge/text-sensitive proxy resizes each JPEG to 320x180 grayscale, computes
Canny edges with thresholds 100/200, and compares consecutive candidates in
the same shot:

```text
edge_change = 0.5 * mean(abs(gray_t - gray_t-1)) / 255
            + 0.5 * mean(canny_t != canny_t-1)
```

The first candidate in each shot has score zero. The robust shot threshold is
`max(0.08, median + 3 * MAD)`; MAD zero therefore deterministically uses the
absolute floor.

Selection starts with the midpoint. A candidate is an event when novelty is at
least 0.15 or edge change reaches the robust threshold. Novelty at least 0.25
marks a strong rare event. Strong rare events bypass ordinary 2-second event
suppression. Ordinary competitors are ordered by the larger of normalized
novelty and normalized edge score, then earlier `frame_idx`; an ordinary event
within 2 seconds of an already retained event is suppressed.

Temporal repair repeatedly takes the largest violating interval, breaking an
equal interval tie toward the earlier interval. Among candidates within one
candidate step of its temporal midpoint it chooses strongest normalized
content score, then nearest midpoint, then earlier frame. If that neighborhood
is empty, all unselected interior candidates are considered. The loop ends only
when start-to-first, every inter-keyframe, and last-to-end gaps are at most 8
seconds. Output is in-bounds, unique, sorted, and assigned `K01`, `K02`, ... in
that order.

## Cache and downstream reuse

The ignored per-video cache records source size/mtime/SHA256, frozen-shot hash,
candidate step, candidate mapping hash, aggregate JPEG hash, exact MetaCLIP
identity/revision/preprocessing, dimension, dtype, normalization, vector hash,
edge-score hash, counts, and timings. Reuse recomputes and checks these fields
and also regenerates the expected candidate mapping from the frozen shots. A
missing, partial, stale, or mismatched cache raises `SelectorCacheError`; it is
never treated as valid.

Selected rows are published under `selector_embeddings/metaclip2` using the
existing provider artifact contract. The real downstream feature runner reused
all 380 rows without model initialization. Artifact audit result:

```text
expected=380, vectors=380, unique=380,
missing=0, extra=0, invalid=0
```

## L21_V001 pilot regression

Pilot corpus: **1 annotated video, L21_V001, 32 manually annotated visual
temporal GT segments** (30 high confidence, 2 medium confidence). This is
research GT v0.1, not official BTC/AIC ground truth. Both medium-confidence
rows remain unchanged and are included in the ALL result.

Frozen conditions:

- video: 37,849 frames, 30 fps, 1,261.633333 seconds;
- PySceneDetect threshold 27, run once through production ingestion;
- 342 frozen shots;
- 1,398 shared 1-second-grid plus midpoint candidates;
- identical candidate JPEGs and normalized MetaCLIP-2 vectors for all three
  strategies;
- English-only query run, exact normalized inner-product ranking, stable ties;
- Windows, NVIDIA GeForce RTX 4050 Laptop GPU, OpenCV 4.11.0, NumPy 2.1.2,
  Torch 2.5.1+cu124, Transformers 4.57.1.

### Measured selection and cost

| Strategy | KFs | ALL covered | HIGH covered | Max gap | Content events | Repairs | JPEG bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Midpoint | 342 | 32/32 (1.0000) | 30/30 (1.0000) | 16.1s | 0 | 0 | 49,337,589 |
| Legacy 12s/8s | 363 | 32/32 (1.0000) | 30/30 (1.0000) | 8.0s | 0 | 0 | 52,693,928 |
| Canonical hybrid | 380 | 32/32 (1.0000) | 30/30 (1.0000) | **8.0s** | 22 | 16 | 55,013,094 |

Hybrid shared-stage timings were 172.253835s candidate extraction,
140.500018s candidate embedding, and 0.090294s selector-only time. Midpoint
and periodic selection logic took 0.006539s and 0.004955s respectively and do
not incur the hybrid dense-candidate stages.

### Fixed MetaCLIP-2 retrieval sanity check

| Strategy | Recall@1 | Recall@10 | Recall@100 | MRR |
|---|---:|---:|---:|---:|
| Midpoint | 0.40625 | 0.81250 | 0.96875 | 0.516471 |
| Legacy 12s/8s | 0.40625 | **0.84375** | 0.96875 | **0.520467** |
| Canonical hybrid | 0.40625 | 0.78125 | 0.96875 | 0.516230 |

The hybrid selector does **not** beat legacy 12s/8s retrieval on this pilot.
Most notably Q003 moved rank 10 to 11 and Q016 rank 4 to 14, while some queries
improved (Q015 5 to 4; Q019 19 to 13). Correctness and temporal-safety gates
pass, but these results provide no evidence for corpus-wide retrieval
superiority. Additional annotated videos and query-level error analysis are
required before making a quality generalization or tuning any threshold.

## Verification

Commands executed included:

```powershell
python -m pytest -q tests/unit/ingestion tests/unit/test_cli.py `
  tests/unit/enrichment/embeddings/test_metaclip2_offline.py `
  tests/unit/enrichment/test_feature_extraction.py

python -m mvr.cli ingest <L21_V001.mp4> `
  --output artifacts/verification/ingest_kf04/L21_V001_hybrid `
  --device cuda --embedding-batch-size 32

python scripts/experiments/evaluate_hybrid_keyframe_selector.py `
  --ingest-root artifacts/verification/ingest_kf04/L21_V001_hybrid `
  --groundtruth <L21_V001 GT CSV> `
  --output artifacts/verification/ingest_kf04/evaluation --device cuda

python scripts/run_feature_extraction.py `
  --mapping-csv <hybrid keyframe_mapping.csv> `
  --keyframe-root <hybrid ingest root> --output <ignored reuse_check> `
  --embedding-providers metaclip2 --skip-objects

git diff --check
git status --short
```

Focused/relevant result: **64 passed, 1 skipped** (the skip is the existing
optional TransNetV2 runtime). A broader non-API run produced **875 passed, 14
skipped, 1 unrelated failure**: a pre-existing Task05A test imports
`tests.unit...` but this environment resolves no `tests.unit` package. The
unrestricted full-suite attempt additionally stopped at collection because
the isolated MetaCLIP environment does not contain the optional FastAPI extra.
Neither issue touches ingestion or the changed files. Python compilation and
`git diff --check` passed.

Evidence hashes:

| Artifact | SHA256 |
|---|---|
| `shots.json` | `d6aa4fd59e977ef0e3d0fc929f15784a4ce578e85669f1281d2def7cc264746b` |
| Frozen shot contract | `4ba9822a2f652ca75455aa3a881ab76db93e959a615f55bb7941eef53a57bfbe` |
| Candidate mapping | `55bf64bc9571732c9a3e1905a4fc49ef107fc96001bddca8bfa6e9faf66d79f6` |
| Candidate embeddings | `a1108030fdff5bb41f4a7ba4205657088bc9660d4dcb2377ad8b2c5bf3c43e92` |
| Canonical hybrid mapping | `17afde454eb8d1ed48f80d78f413a40cda220a6f95ab4a59c89e008c5193fe6b` |
| Strategy metrics CSV | `994e135992e8a4073ce8a6579bd23224be660865c4a3cf8469b9faaf261b0329` |
| Evaluation summary | `690fd9253dbec00a531676d1df2b98ae956c826f9d8104423f126d961c005719` |

Raw video, GT data, candidate JPEGs, embeddings, model cache, reuse output,
and evaluation artifacts remain ignored and unstaged.

## Known limitations and stop state

- One manually annotated video cannot establish a Batch1/corpus-wide optimum.
- The edge signal is an inexpensive slide/text-sensitive proxy, not OCR or
  semantic text understanding.
- Hybrid preprocessing adds substantial offline cost and 17 frames over the
  legacy control on this pilot; downstream MetaCLIP reuse avoids paying the
  selected-frame encode cost again.
- MetaCLIP-2 carries its documented CC-BY-NC-4.0 restriction.
- Threshold quality needs a multi-video evaluation; no intuition-based tuning
  was performed after observing this pilot.

No raw data or generated artifacts are staged. No merge and no push were
performed.
