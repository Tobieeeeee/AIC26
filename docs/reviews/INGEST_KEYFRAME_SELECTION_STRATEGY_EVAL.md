# INGEST-KF02 — Keyframe Selection Strategy Pilot Evaluation

## Verdict

**NEEDS_MORE_EVALUATION**

The pilot produced complete measured evidence for midpoint, both current periodic profiles, PE-Core-L and MetaCLIP-2 budget-matched diversity, two-second candidate-grid sensitivity, and ten novelty variants. On this video, midpoint already covers every manually annotated segment. The 12s/8s profile has the best secondary Recall@10 and MRR, but its advantage is one query out of 32 and is not sufficient to establish a corpus-wide production optimum.

No production selector or ingestion default was changed.

## 1. Repository and pilot scope

- HEAD: `37696aa568ce634458ee21bd91fc9a6a863576bf`
- Branch: `fix/ocr-production-checkpoints`
- Experiment date: 2026-08-22, Asia/Saigon
- **Pilot corpus: 1 annotated video, `L21_V001`, 32 manually annotated visual temporal GT segments.**
- Confidence split: 30 high-confidence rows and 2 visible medium-confidence rows (`Q020`, `Q030`).
- GT status: manually annotated research GT v0.1, not official BTC/AIC ground truth.
- Video: 1280×720, 30 fps, 37,849 frames, 1,261.633333 seconds, 130,322,332 bytes.

The supplied data used versioned filenames:

- `L21_V001_keyframe_selection_groundtruth_v0_1.csv`
- `L21_V001_keyframe_selection_groundtruth_v0_1.json`
- `L21_V001_keyframe_selection_groundtruth_README.md`
- `evaluate_keyframe_selection_recall.py`

These are the available equivalents of the shorthand paths stated in the request. The raw video remained under ignored `data/raw/`; it was not staged or copied into a tracked location.

This explicitly authorized one-video pilot supersedes the original task's representative-corpus stopping rule. The resulting manifest has 342 shots, but only 12 shots ≥12 seconds and one shot ≥30 seconds. It does not satisfy the original multi-video/multi-collection representative target and must not be generalized to all Batch1.

## 2. Frozen shots and fairness

TransNetV2 was unavailable because `transnetv2_pytorch` was not installed. PySceneDetect 0.7.1 was therefore run exactly once at threshold 27.0. Detection took 90.178147 seconds and produced one frozen 342-shot manifest. SHA-256 of `frozen_shots.json` is `b79882f0988c25dba18a4144637b7b299a901b656e4f294caab30223d6a0a088`.

Every strategy consumed that same manifest. No selector redetected shots.

The shared candidate pool was the production midpoint plus a deterministic one-second temporal grid, bounded, deduplicated, and temporally sorted. The production FFmpeg extractor scanned the source video once:

- 1,398 candidate JPEGs;
- 207,163,002 candidate-cache bytes;
- 85.872009 seconds measured extraction wall time;
- no full-frame-rate decode cache;
- no repeated decode between selector sweeps.

The two-second adaptive sensitivity used a deterministic subset of this same cache, so it introduced neither new detection nor extraction.

## 3. Strategy definitions and model identity

| Label | Exact strategy |
|---|---|
| S0 | Current production floor midpoint only |
| S1 | Real production selector with 12s threshold / 8s periodic interval |
| S2 | Real production selector with 15s threshold / 5s periodic interval |
| S3 Meta | Midpoint mandatory; greedy maximum-minimum cosine distance; exact per-shot S1 budget |
| S3 PE | Same adaptive algorithm and exact per-shot S1 budget, with PE embeddings |
| S4 | Midpoint mandatory; temporal scan; cosine novelty strictly `> tau`; 2s minimum gap; cap 16 |

Adaptive candidate generation used a one-second grid as primary and a two-second grid as sensitivity. S3 also ran budget-matched to S2; those results are retained in machine-readable output.

| Selector encoder | Locked identity | Resolution | Dim. | Normalization | Device / batch | Candidate embedding time |
|---|---|---:|---:|---|---|---:|
| MetaCLIP-2 | `facebook/metaclip-2-worldwide-b16@8cc8591f32b43a809f3325192d3b77aa56968ddb` | 224² | 512 | L2 contract | CPU / 16 | 469.211258s |
| PE-Core-L | `PE-Core-L14-336.pt@ee1fdb467ab42497fa2e60e94a79e9d7383c1cbd`; SHA-256 locked by provider spec | 336² | 1024 | `encode_image(normalize=True)` | RTX 4050 CUDA / 8 | 157.370238s |

The PE and Meta timings are not a hardware speed comparison because their devices differ. Both embedding artifacts are normalized `float32`; sizes are 5,726,336 and 2,863,232 bytes respectively. Selector-only CPU time was 0.015625–0.03125 seconds per adaptive strategy at the Windows timer's resolution.

## 4. Primary selection-recall results

A GT record is covered only when video identity matches and an exact selected `frame_idx` lies in its inclusive annotated frame range. The repository-supplied helper independently reproduced 32/32 for S0, S1, S2, S3 Meta budget-S1, and S3 PE budget-S1.

| Strategy | Covered all | Recall all | Covered high | Recall high | KFs | KF/shot | Relative KFs | JPEG bytes | Segments / 100 KFs | Selector embedding | Selector CPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 midpoint | 32/32 | 1.0000 | 30/30 | 1.0000 | 342 | 1.0000 | 1.0000× | 49,337,589 | 9.3567 | 0s | 0s |
| S1 12s/8s | 32/32 | 1.0000 | 30/30 | 1.0000 | 363 | 1.0614 | 1.0614× | 52,693,928 | 8.8154 | 0s | ≤0.015625s |
| S2 15s/5s | 32/32 | 1.0000 | 30/30 | 1.0000 | 374 | 1.0936 | 1.0936× | 54,421,457 | 8.5561 | 0s | ≤0.015625s |
| S3 Meta budget-S1 | 32/32 | 1.0000 | 30/30 | 1.0000 | 363 | 1.0614 | 1.0614× | 52,624,883 | 8.8154 | 469.211258s | 0.015625s |
| S3 PE budget-S1 | 32/32 | 1.0000 | 30/30 | 1.0000 | 363 | 1.0614 | 1.0614× | 52,584,867 | 8.8154 | 157.370238s | 0.015625s |

Count distribution:

- S0: median/p95/max = 1/1/1; 100% of shots have one KF.
- S1 and both budget-S1 adaptives: median/p95/max = 1/1/5; 3.5088% of shots have multiple KFs.
- S2: median/p95/max = 1/1/7; 2.6316% have multiple KFs. This percentage is lower than S1 despite a higher total because S2 concentrates more frames in fewer shots.

Direct equal-budget answer: **No. Adaptive placement did not cover more GT segments than 12s/8s at the same 363-keyframe count.** All three covered 32/32. Each adaptive strategy changed 20 placements relative to S1 while retaining 343 common frames. PE and Meta differed from each other by 14 placements on each side, but not in selection recall.

The two-second S3 candidate-grid variants also used exactly 363 KFs and covered 32/32. Their secondary retrieval metrics were identical to the one-second S3 variants, so the denser candidate grid added no measured GT benefit in this pilot.

## 5. Novelty sweep

The requested thresholds were not selected from intuition. Across 1,056 non-midpoint candidate-to-midpoint comparisons, measured cosine-distance quantiles `[min, p25, p50, p75, p95, max]` were:

- MetaCLIP-2: `[0.000210, 0.014784, 0.025880, 0.047552, 0.148668, 0.528626]`.
- PE: `[0.000672, 0.011598, 0.025489, 0.054961, 0.128237, 0.538727]`.

Thus the fixed sweep 0.05–0.25 spans approximately the upper quartile through beyond p95 and is non-degenerate.

| S4 strategy | KFs | KF/shot | Relative | JPEG bytes | Recall all/high | Segments/100 KFs | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Meta τ=.05 | 406 | 1.1871 | 1.1871× | 58,464,240 | 1.000/1.000 | 7.8818 | .8125 | .511096 |
| Meta τ=.10 | 379 | 1.1082 | 1.1082× | 54,313,177 | 1.000/1.000 | 8.4433 | .78125 | .511349 |
| Meta τ=.15 | 358 | 1.0468 | 1.0468× | 51,453,480 | 1.000/1.000 | 8.9385 | .8125 | .516860 |
| Meta τ=.20 | 351 | 1.0263 | 1.0263× | 50,376,097 | 1.000/1.000 | 9.1168 | .8125 | .516312 |
| Meta τ=.25 | 347 | 1.0146 | 1.0146× | 49,897,154 | 1.000/1.000 | 9.2219 | .8125 | .516351 |
| PE τ=.05 | 420 | 1.2281 | 1.2281× | 61,017,443 | 1.000/1.000 | 7.6190 | .78125 | .512712 |
| PE τ=.10 | 373 | 1.0906 | 1.0906× | 53,588,802 | 1.000/1.000 | 8.5791 | .8125 | .516213 |
| PE τ=.15 | 356 | 1.0409 | 1.0409× | 51,163,122 | 1.000/1.000 | 8.9888 | .8125 | .517413 |
| PE τ=.20 | 351 | 1.0263 | 1.0263× | 50,623,493 | 1.000/1.000 | 9.1168 | .8125 | .516271 |
| PE τ=.25 | 345 | 1.0088 | 1.0088× | 49,744,924 | 1.000/1.000 | 9.2754 | .8125 | .516351 |

No novelty threshold improved selection recall, Recall@1, Recall@10, or Recall@100 over midpoint. No threshold is promoted.

All S4 rows reuse the same 85.872009-second candidate extraction and their provider's cached embedding pass; selector-only CPU time is 0.015625–0.03125 seconds. The S3 budget-S2 variants also completed: both used 374 KFs, covered 32/32, and reproduced S2's `.8125` R@10 / `.516417` MRR while adding their respective selector embedding overhead.

## 6. Secondary fixed-provider retrieval

This optional evaluation was feasible and was run. It used English query text only—Vietnamese and English were not mixed. All strategies used the exact same pinned MetaCLIP-2 query/image model, normalization, query preprocessing, exact FAISS `IndexFlatIP`, 32 queries, and temporal relevance ranges. Only the selected frame set varied.

The 1,398-candidate Meta image pass was shared and measured once at 469.211258 seconds; it served both Meta selection and fixed retrieval rather than being charged twice. The final warm CUDA text-query pass took 11.901736 seconds for all 32 queries. Strategy-specific normalized arrays and FAISS indexes were physically materialized and measured.

| Strategy | R@1 | R@10 | R@100 | MRR | High R@10 | High MRR | Embedding bytes | FAISS bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 midpoint | .40625 | .81250 | .96875 | .516472 | .86667 | .548223 | 700,544 | 700,461 |
| S1 12s/8s | .40625 | **.84375** | .96875 | **.520468** | **.90000** | **.552485** | 743,552 | 743,469 |
| S2 15s/5s | .40625 | .81250 | .96875 | .516417 | .86667 | .548189 | 766,080 | 765,997 |
| S3 Meta budget-S1 | .40625 | .81250 | .96875 | .516308 | .86667 | .548097 | 743,552 | 743,469 |
| S3 PE budget-S1 | .40625 | .81250 | .96875 | .516308 | .86667 | .548097 | 743,552 | 743,469 |

S3 one-second and two-second variants have identical retrieval metrics. All strategies retrieved 31/32 relevant segments within 100; the remaining high-confidence `Q006` first appeared at rank 174 for S1, rather than being absent from its selected frames.

This secondary test is query-to-keyframe retrieval on one video, not an official multi-video BTC ranking benchmark. With only one video, target-video hit rate is uninformative, and no TRAKE sequence label exists.

## 7. Paired uncertainty

Deterministic paired bootstrap: 10,000 query resamples, seed 260822.

| Right minus left | ΔR@10 | 95% bootstrap interval | ΔMRR | 95% bootstrap interval |
|---|---:|---|---:|---|
| S1 − S0 | +.03125 (one query) | [.00000, .09375] | +.003996 | [−.004687, .016741] |
| S2 − S1 | −.03125 | [−.09375, .00000] | −.004051 | [−.016774, .004631] |
| S3 Meta − S1 | −.03125 | [−.09375, .00000] | −.004160 | [−.016915, .004454] |
| S3 PE − S1 | −.03125 | [−.09375, .00000] | −.004160 | [−.016915, .004454] |

The pilot does not establish a stable non-zero MRR improvement. Absolute query deltas are reported because describing one query as a general improvement would be misleading.

## 8. Long-shot-only analysis

There were 12 shots ≥12 seconds and only one ≥30 seconds. Four GT segments overlap the ≥12-second shots; all five compared strategies cover all four. No GT record overlaps the sole ≥30-second shot.

| Scope / strategy | Shots | Selected KFs | KF/shot | Mean per-shot pairwise Meta cosine distance | Overlapping GT covered |
|---|---:|---:|---:|---:|---:|
| ≥12s, S0 | 12 | 12 | 1.00 | 0 | 4/4 |
| ≥12s, S1 | 12 | 33 | 2.75 | .036823 | 4/4 |
| ≥12s, S2 | 12 | 44 | 3.67 | .033238 | 4/4 |
| ≥12s, S3 Meta | 12 | 33 | 2.75 | **.121873** | 4/4 |
| ≥12s, S3 PE | 12 | 33 | 2.75 | .104554 | 4/4 |
| ≥30s, S0/S1/S2/S3 Meta/S3 PE | 1 | 1/5/7/5/5 | 1/5/7/5/5 | 0/.020517/.018856/.028267/.025855 | 0/0 |

Adaptive placement increased a fixed MetaCLIP visual-diversity proxy, but this did not improve selection recall or fixed-provider retrieval. Diversity alone is therefore not adoption evidence.

The five examples below are the five longest shots, chosen deterministically by descending duration then shot ID—not cherry-picked successes:

| Shot | Duration | GT | S0 frames | S1 frames | S2 frames | S3 Meta frames | S3 PE frames |
|---:|---:|---|---|---|---|---|---|
| 14 | 32.2s | none | `1403` | `1161,1401,1403,1641,1881` | `1071,1221,1371,1403,1521,1671,1821` | `951,1131,1191,1403,1551` | `951,1041,1101,1403,1491` |
| 49 | 22.867s | Q008 | `5935` | `5833,5935,6073` | `5743,5893,5935,6043,6193` | `5803,5935,6193` | `5935,6193,6253` |
| 281 | 20.933s | Q029 | `30976` | `30903,30976,31143` | `30813,30963,30976,31113,31263` | `30693,30843,30976` | `30693,30976,31113` |
| 58 | 20.333s | Q009 | `7506` | `7442,7506,7682` | `7352,7502,7506,7652,7802` | `7262,7506,7712` | `7232,7506,7772` |
| 144 | 16.9s | none | `15609` | `15596,15609,15836` | `15506,15609,15656,15806` | `15566,15609,15776` | `15566,15609,15776` |

Exact per-example diversity values are retained in `analysis_summary.json`.

## 9. Cost, size, and Pareto interpretation

Relative to S0:

- S1 adds 21 KFs (+6.14%), 3,356,339 JPEG bytes, 43,008 retrieval-array bytes, and 43,008 FAISS bytes.
- S2 adds 32 KFs (+9.36%), 5,083,868 JPEG bytes, 65,536 retrieval-array bytes, and 65,536 FAISS bytes.
- Budget-S1 adaptive selectors keep the same count/index size as S1 but add a 1,398-frame candidate cache and 157.37s PE or 469.21s Meta embedding pass.
- S4 variants reuse those embedding passes but provide no quality gain over the cheaper midpoint/S1 choices.

OCR, object detection, and ASR were not rerun and no runtime for them is claimed. For keyframe-native downstream stages, work and artifact cardinality scale with selected keyframe count as a derived relation, not a measured stage cost.

Using Recall@10 as the secondary quality dimension, S0 and S1 form the useful pilot frontier: S0 is cheapest and most efficient per keyframe; S1 spends 6.14% more keyframes for one additional Top-10 hit. S2 is dominated by S0 on R@1/R@10/R@100, MRR, count, JPEG bytes, index bytes, and selector overhead. Budget-S1 adaptive strategies are dominated by S1 on R@10/MRR and overhead at identical count. Novelty strategies add overhead without improving the primary selection metric or retrieval Recall@K.

## 10. Level-3 Attention/Transformer feasibility

No learned selector was trained or integrated.

Credible supervision would require many videos with query-to-segment labels and negative candidates, ideally official KIS/TRAKE outcomes or logged retrieval success. Thirty-two segments from one video are insufficient for train/validation/test separation and would strongly overfit visual topics in this news compilation. Training labels could rank candidates by whether they improve fixed-retriever success under a budget; evaluation would still need held-out videos, density/cost controls, and comparison against S1 and the best Level-2 selector. A learned summarization objective risks selecting visually representative frames rather than query-retrievable evidence. Level 3 is not justified by this pilot.

## 11. Reproducible artifacts

Ignored outputs are under `artifacts/verification/ingest_kf02/`, including:

- `frozen_shots.json`
- `candidate_mapping.csv` and shared candidate JPEGs
- `strategy_summary.csv` — SHA-256 `b0fd25f90f82bd08eb41354a84032ea289ebb1eaff802d0fd464335011c1e9d9`
- `per_shot_selection.csv` and per-strategy mappings
- `selector_embeddings/{metaclip2,perception_encoder}/`
- `retrieval_metrics.csv`
- `retrieval_per_query.csv`, physical arrays, and FAISS indexes
- `analysis_summary.json`
- model/runtime caches under ignored locations only

Experimental implementation:

- `scripts/experiments/keyframe_selection/adaptive_selector.py`
- `scripts/experiments/keyframe_selection/run_strategy_eval.py`
- `scripts/experiments/keyframe_selection/evaluate_retrieval.py`
- `scripts/experiments/keyframe_selection/summarize_results.py`
- deterministic tests in `tests/benchmark/test_ingest_keyframe_adaptive_selector.py`

Verification completed:

- focused ingestion, adaptive-selector, and real-provider adapter tests: **53 passed in 0.86s**;
- `git diff --check`: exit 0;
- `git status --short`: inspected; raw video and `artifacts/verification/ingest_kf02/` remained ignored;
- no merge, push, commit, or production-default change was performed.

## 12. Required direct answers

1. **Which strategy produced the best retrieval quality?** S1 12s/8s on this pilot: R@10 .84375 and MRR .520468; all key strategies tie at R@1 .40625 and R@100 .96875.
2. **Which strategy produced the best quality per keyframe?** S0 midpoint: the same 32/32 selection recall and 26/32 R@10 with the fewest KFs; its selection efficiency is 9.3567 covered segments per 100 KFs. S1 buys one extra Top-10 query for 21 extra frames.
3. **Does adaptive placement beat periodic placement at the same keyframe count?** No. Selection recall ties 32/32; both S3 encoders score 26/32 R@10 versus S1's 27/32.
4. **Is 15s/5s worth its extra keyframes over 12s/8s?** No on this pilot. It uses 11 more KFs than S1 and scores one fewer Top-10 query.
5. **Does midpoint-only lose meaningful retrieval recall?** It loses one R@10 query versus S1 but nothing at R@1/R@100 or selection recall. The 32-query bootstrap does not justify calling this corpus-wide meaningful.
6. **Which adaptive encoder is better, PE or MetaCLIP?** Neither on quality: their selection and retrieval metrics tie. They choose 14 different placements per side. PE's shorter measured time used CUDA while Meta used CPU, so it is not a fair encoder-speed win.
7. **Added offline compute/storage cost?** Shared extraction 85.872s/207,163,002 candidate bytes, plus 157.370s/5,726,336 PE-vector bytes or 469.211s/2,863,232 Meta-vector bytes; budget-S1 output count/index storage equals S1.
8. **What should become the canonical production default?** No corpus-wide default can be selected from one annotated video. Keep production unchanged. If an interim incumbent must be named for further evaluation, test 12s/8s first because it leads this pilot and is simpler than adaptive selection.
9. **Is Level-3 work justified?** No. Expand manual GT to multiple representative videos and resolve Level-2/generalization evidence first.

## Final verdict

**NEEDS_MORE_EVALUATION**

The next experiment should add multiple annotated videos/collections with substantially more ≥12s and ≥30s shots, retain the same fixed-provider retrieval protocol, and pre-register whether selection recall, Recall@10, or another KIS metric controls the canonical-default decision.
