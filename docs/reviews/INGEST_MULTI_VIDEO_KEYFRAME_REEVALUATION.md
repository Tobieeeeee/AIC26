# INGEST multi-video keyframe re-evaluation

**Status:** measured four-video pilot; production defaults unchanged

**Date:** 2026-08-23
**Branch / base:** `feat/ingest-hybrid-keyframe-selector` / `e637af712b2e7f0e879c612abbac1c18e7fce73b`

## Verdict

The four-video result does **not** justify changing the production default. The safest strategy in this pilot is still the explicit periodic `12s/8s` control: it has no selector-model dependency, covers `98/99` GT segments under PySceneDetect and `99/99` under TransNetV2, enforces the 8-second gap, and is tied for the best fixed-retrieval pooled Recall@10 under TransNetV2 with fewer keyframes than every content-aware Pass-1 strategy.

The content encoders are useful, but not as a stable replacement based on these four videos. With selector placement isolated by fixed MetaCLIP-2 retrieval, PE is the best single selector under PySceneDetect, MetaCLIP-2 is best under TransNetV2, and no single encoder wins both regimes. With provider-matched retrieval, PE is the strongest single retrieval encoder and PE+BEiT RRF is the strongest fusion; that is a different, confounded question and is reported separately.

Direct answers:

1. **Does periodic still lead under TransNetV2? Yes.** It achieves `99/99` Selection Recall, macro R@10 `79.2%`, pooled R@10 `78.8%`, and 1,182 KFs. Its macro R@10 is higher than all content-aware placements under fixed retrieval; pooled R@10 ties Meta, Meta+edge, and PE+Meta, with fewer KFs and zero selector embedding cost.
2. **Which single encoder wins? No detector-independent winner.** Fixed-retrieval placement: PE wins PySceneDetect (macro R@10 `80.9%`), while Meta wins TransNetV2 (pooled R@10 `78.8%`; macro Meta and BEiT tie at `78.5%`). Averaged over the two detector macros, PE is narrowly highest (`79.2%`) but is not robust across regimes. Provider-matched retrieval identifies PE as the strongest retrieval encoder (`83.4%/84.4%` macro R@10 for PyScene/TransNet).
3. **Does edge assistance help? Not reliably.** Versus Meta-only, it adds 54 PyScene and 50 TransNet KFs. PyScene pooled R@10 increases one query (`+1.01 pp`) but pooled MRR falls; TransNet pooled R@10 is unchanged and macro R@10 falls `1.0 pp`. It is not a safe net improvement.
4. **Does PE+Meta beat the best single? No.** Under fixed retrieval it adds 30/36 KFs. It lowers PyScene macro R@10 by `2.78 pp` versus PE and only ties pooled R@10; under TransNet it trails Meta macro R@10 and only ties pooled R@10. Provider-matched RRF improves some R@10 values but lowers pooled MRR versus PE in both regimes.
5. **Is triple fusion worth the compute? No.** It costs all three embedding passes (`866.5 s` incremental selector-image time) and 27/28 more KFs than PE+BEiT. It does not improve fixed-retrieval R@10; in provider-matched retrieval it is also below PE+BEiT on macro and pooled R@10 in both regimes.
6. **Can adaptive KNN reduce KFs without retrieval loss? Yes, but only slightly in this pilot.** Adaptive PE removes 4 PyScene and 7 TransNet KFs (`0.3%/0.6%`) with unchanged R@1/10/100 and a negligible positive MRR delta. Adaptive BEiT removes 3/8 KFs with the same result. This is real but too small to motivate a default change.
7. **Which strategy is safest across all four videos? Periodic `12s/8s`.** It has the best combined robustness/cost profile. The worst-video fixed-retrieval R@10 is `50.0%` under PyScene and `43.8%` under TransNet; content-aware strategies do not improve both. L24_V043 remains the hard video for every strategy.
8. **Enough evidence to change production? No.** Four videos and 99 manually annotated, non-official GT segments are stronger evidence than the one-video pilot, but still not a corpus-wide evaluation. The result is also detector-sensitive. Keep the current production default unchanged and treat any change as requiring a larger preregistered evaluation.

## Corpus and ground truth

The authoritative repository file used was `data/eval/keyframe_selection/groundtruth_4videos_v0_2.csv` (the repository filename corresponding to the task's `AIC26_keyframe_selection_groundtruth_4videos_v0_2.csv`), SHA-256:

`51cc5e1d10f28bb88fc04772fa969938887a9bcaec8b1736c26d8c7fcdd75369`

It contains exactly 99 detector-independent manual GT rows: 86 high-confidence and 13 medium-confidence.

| Video | All GT | High | Medium | Duration |
|---|---:|---:|---:|---:|
| L21_V001 | 32 | 30 | 2 | 1,261.73 s |
| L21_V029 | 26 | 24 | 2 | 1,156.38 s |
| L22_V018 | 25 | 16 | 9 | 1,377.06 s |
| L24_V043 | 16 | 16 | 0 | 688.10 s |
| **Total** | **99** | **86** | **13** | **4,483.26 s** |

All retrieval runs use `query_en` consistently. No BTC-provided keyframes were used as GT. These annotations are research GT, not official BTC/AIC ground truth.

## Frozen detector regimes

Each selector consumes the same immutable manifest inside its video/detector regime. PySceneDetect uses threshold 27; TransNetV2 uses threshold 0.5. TransNet was run with `CUBLAS_WORKSPACE_CONFIG=:4096:8` after rejecting an initial, non-frozen run that emitted CUDA determinism warnings.

| Video | Detector | Shots | Min / median / p95 / max shot | 1 s candidate records | Detector time |
|---|---|---:|---|---:|---:|
| L21_V001 | PySceneDetect | 342 | 0.60 / 2.73 / 10.20 / 32.20 s | 1,398 | reused frozen KF04 manifest |
| L21_V001 | TransNetV2 | 336 | 0.17 / 2.85 / 9.78 / 32.20 s | 1,400 | 41.20 s |
| L21_V029 | PySceneDetect | 287 | 0.60 / 2.60 / 13.13 / 30.00 s | 1,279 | 79.25 s |
| L21_V029 | TransNetV2 | 282 | 0.70 / 2.70 / 13.16 / 22.97 s | 1,283 | 37.84 s |
| L22_V018 | PySceneDetect | 392 | 0.57 / 2.40 / 11.59 / 49.13 s | 1,531 | 93.27 s |
| L22_V018 | TransNetV2 | 383 | 0.10 / 2.53 / 11.30 / 29.60 s | 1,526 | 44.78 s |
| L24_V043 | PySceneDetect | 105 | 0.84 / 4.64 / 18.00 / 38.28 s | 726 | 39.59 s |
| L24_V043 | TransNetV2 | 84 | 0.88 / 6.42 / 20.60 / 38.28 s | 719 | 20.46 s |

PyScene manifests cover every source frame contiguously. The real repository TransNet adapter leaves two uncovered one-frame positions in L21_V029 (frame 0 and one interior hole) and three interior one-frame holes in L22_V018. These manifests were not repaired because doing so would change the detector regime. No GT segment is limited to those holes.

The only non-midpoint GT miss for all PyScene Pass-1 selectors is high-confidence `L21_V029_Q026`, frames 33840–33929. PyScene makes it part of a 25.8 s shot whose midpoint and 8 s grid miss the range; TransNet isolates a 5.67 s shot with midpoint 33866 and covers it. This is direct evidence that detector boundaries can matter more than the tested content signal.

## Protocol and preregistered mechanics

Pass 1 was completed and persisted before the competitiveness gate ran:

- S0 midpoint;
- S1 periodic `12s/8s`;
- S2 MetaCLIP-2 only;
- S3 PE-Core-L only;
- S4 BEiT-3 only;
- S5 MetaCLIP-2 plus the existing grayscale/Canny edge/text-sensitive change score.

S2/S3/S4 use the same production mechanics: 1 s candidates, exact midpoint mandatory, temporal-local cosine novelty (`k=3`, ±4 s), ordinary threshold 0.15, rare threshold 0.25, minimum event spacing 2 s, and deterministic repair until maximum gap is at most 8 s. S5 adds only the existing edge signal. No threshold was changed after inspecting misses.

All three encoders won at least one preregistered competitiveness dimension, so Pass 2 ran all requested RRF `k=60` fusions: PE+Meta, Meta+BEiT, PE+BEiT, and PE+Meta+BEiT. Fusion ranks novelty inside each shot; it never concatenates raw vectors.

Pass 3 used the top two single providers from Pass 1 (PE and BEiT). Its fixed adaptive rule is `max(0.15, median novelty + 3*MAD)` per shot/provider, while retaining the fixed 0.25 rare threshold and 8 s repair. No pair KNN was run because the optional extension was unnecessary after the single-provider reduction proved very small.

## Cache, model, runtime, and storage evidence

Candidate JPEGs were unioned by source frame across both detector regimes. There are 9,862 detector-specific candidate records but only 5,581 unique source frames, so the union avoids 4,281 duplicate JPEG/embedding operations (`43.4%`). The shared JPEG pool is 991,270,150 bytes.

| Video | Unique union frames | Reused seed JPEGs | Newly extracted | Incremental extraction | JPEG bytes |
|---|---:|---:|---:|---:|---:|
| L21_V001 | 1,543 | 1,398 | 145 | 82.99 s | 226,992,677 |
| L21_V029 | 1,442 | 0 | 1,442 | 80.81 s | 239,823,260 |
| L22_V018 | 1,775 | 0 | 1,775 | 94.19 s | 297,716,126 |
| L24_V043 | 821 | 0 | 821 | 41.36 s | 226,738,087 |

Pinned providers:

- MetaCLIP-2 `facebook/metaclip-2-worldwide-b16`, revision `8cc8591f32b43a809f3325192d3b77aa56968ddb`, 512-D;
- PE-Core-L/14-336, revision `ee1fdb467ab42497fa2e60e94a79e9d7383c1cbd`, checkpoint SHA-256 `0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8`, 1024-D;
- BEiT-3 base ITC, UniLM source `69961945a2fd283f2beed74d1f3c543265c9930c`, checkpoint SHA-256 `d483a41d92b9aea7ef87d17551413b8c22faae927110666b01ac3aa7a199a25b`, tokenizer SHA-256 `6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4`, 768-D.

| Provider | New images this run | Seed vectors reused | Incremental image time | Full vector-cache bytes | 99-query time |
|---|---:|---:|---:|---:|---:|
| MetaCLIP-2 | 4,183 | 1,398 | 235.64 s | 11,430,400 | 16.34 s |
| PE-Core-L | 5,581 | 0 | 504.47 s | 22,860,288 | 18.81 s |
| BEiT-3 | 5,581 | 0 | 126.37 s | 17,145,344 | 10.29 s |

The reused Meta seed's earlier measured image time was 140.50 s; it is not added to incremental task time. Edge arrays were recomputed and matched the accepted cache exactly. Edge timing is 22.15 s for all PyScene candidates and 21.98 s for all TransNet candidates. Selector-only CPU totals range from 0.125–0.391 s per detector corpus; Windows CPU timer granularity makes smaller differences non-actionable.

## Retrieval fairness

The **primary** retrieval comparison fixes MetaCLIP-2 image/text vectors, English queries, normalization, exhaustive inner-product ranking, and the exact GT ranges for every selector. Thus only selected frame placement changes. The **secondary** diagnostic uses each selector provider's paired text/image space and RRFs provider ranks for fusion. It measures an end-to-end encoder stack, not placement alone.

Recall@K means at least one selected frame inside the query's exact GT interval appears in the first K ranked results. MRR uses the first such result. Per-video metrics search only that video's selected frames; pooled metrics search all four videos. “Macro” is the unweighted mean of the four per-video metrics. “Worst” is the minimum video metric.

## Primary aggregate results: fixed MetaCLIP-2 retrieval

| Detector | Strategy | KFs | Selection | High selection | Macro R@1/10/100 | MRR | Pooled R@1/10/100 | Pooled MRR | Worst R@10 |
|---|---|---:|---:|---:|---|---:|---|---:|---:|
| PyScene | S0 midpoint | 1,126 | 96/99 | 83/86 | 42.1/79.3/95.1% | .538 | 37.4/77.8/94.9% | .500 | 43.8% |
| PyScene | S1 periodic | 1,227 | 98/99 | 85/86 | 42.1/80.6/98.3% | .540 | 37.4/79.8/97.0% | .504 | 50.0% |
| PyScene | S2 Meta | 1,243 | 98/99 | 85/86 | 44.0/77.5/98.3% | .553 | 38.4/77.8/97.0% | .512 | 43.8% |
| PyScene | S3 PE | 1,243 | 98/99 | 85/86 | 41.1/80.9/98.3% | .533 | 37.4/78.8/97.0% | .500 | 50.0% |
| PyScene | S4 BEiT | 1,223 | 98/99 | 85/86 | 42.1/79.3/98.3% | .540 | 37.4/76.8/97.0% | .501 | 43.8% |
| PyScene | S5 Meta+edge | 1,297 | 98/99 | 85/86 | 44.0/78.1/96.7% | .552 | 38.4/78.8/96.0% | .511 | 50.0% |
| PyScene | PE+Meta RRF | 1,273 | 98/99 | 85/86 | 43.0/78.1/98.3% | .546 | 38.4/78.8/97.0% | .510 | 50.0% |
| PyScene | Meta+BEiT RRF | 1,252 | 98/99 | 85/86 | 44.0/77.5/98.3% | .552 | 38.4/76.8/97.0% | .512 | 43.8% |
| PyScene | PE+BEiT RRF | 1,248 | 98/99 | 85/86 | 41.1/80.9/98.3% | .532 | 37.4/77.8/97.0% | .500 | 50.0% |
| PyScene | triple RRF | 1,275 | 98/99 | 85/86 | 43.0/78.1/98.3% | .545 | 38.4/77.8/97.0% | .510 | 50.0% |
| PyScene | adaptive PE | 1,239 | 98/99 | 85/86 | 41.1/80.9/98.3% | .534 | 37.4/78.8/97.0% | .500 | 50.0% |
| PyScene | adaptive BEiT | 1,220 | 98/99 | 85/86 | 42.1/79.3/98.3% | .540 | 37.4/76.8/97.0% | .501 | 43.8% |
| TransNet | S0 midpoint | 1,085 | 97/99 | 84/86 | 39.4/80.0/96.1% | .530 | 35.4/78.8/97.0% | .492 | 50.0% |
| TransNet | S1 periodic | 1,182 | 99/99 | 86/86 | 39.4/79.2/99.2% | .530 | 36.4/78.8/99.0% | .500 | 43.8% |
| TransNet | S2 Meta | 1,229 | 99/99 | 86/86 | 41.3/78.5/99.2% | .542 | 36.4/78.8/98.0% | .504 | 43.8% |
| TransNet | S3 PE | 1,221 | 99/99 | 86/86 | 39.4/77.5/99.2% | .528 | 35.4/77.8/98.0% | .493 | 43.8% |
| TransNet | S4 BEiT | 1,208 | 99/99 | 86/86 | 39.4/78.5/99.2% | .529 | 35.4/76.8/98.0% | .494 | 43.8% |
| TransNet | S5 Meta+edge | 1,279 | 99/99 | 86/86 | 41.3/77.5/99.2% | .538 | 36.4/78.8/97.0% | .502 | 43.8% |
| TransNet | PE+Meta RRF | 1,257 | 99/99 | 86/86 | 41.3/77.5/99.2% | .541 | 36.4/78.8/98.0% | .504 | 43.8% |
| TransNet | Meta+BEiT RRF | 1,240 | 99/99 | 86/86 | 41.3/78.5/99.2% | .541 | 36.4/77.8/98.0% | .504 | 43.8% |
| TransNet | PE+BEiT RRF | 1,236 | 99/99 | 86/86 | 39.4/77.5/99.2% | .527 | 35.4/76.8/98.0% | .493 | 43.8% |
| TransNet | triple RRF | 1,264 | 99/99 | 86/86 | 41.3/77.5/99.2% | .540 | 36.4/77.8/98.0% | .504 | 43.8% |
| TransNet | adaptive PE | 1,214 | 99/99 | 86/86 | 39.4/77.5/99.2% | .529 | 35.4/77.8/98.0% | .493 | 43.8% |
| TransNet | adaptive BEiT | 1,200 | 99/99 | 86/86 | 39.4/78.5/99.2% | .529 | 35.4/76.8/98.0% | .494 | 43.8% |

Every strategy except midpoint satisfies the measured maximum-gap invariant at `<=8.0 s`; there are zero invariant failures.

## Mandatory Pass-1 per-video results

These are fixed-retrieval metrics. The complete per-video rows for Pass 2/3 and exact per-row JPEG/vector/index bytes and runtimes are in `artifacts/verification/ingest_kf04_multivideo/all_metrics_fixed_metaclip2.csv`.

| Detector | Video | Strategy | KFs | Selection | High | R@1/10/100 | MRR | Max gap |
|---|---|---|---:|---:|---:|---|---:|---:|
| PyScene | L21_V001 | S0 | 342 | 32/32 | 30/30 | 40.6/81.2/96.9% | .516 | 16.10 s |
| PyScene | L21_V001 | S1 | 363 | 32/32 | 30/30 | 40.6/84.4/96.9% | .520 | 8.00 s |
| PyScene | L21_V001 | S2 | 371 | 32/32 | 30/30 | 40.6/78.1/96.9% | .516 | 7.93 s |
| PyScene | L21_V001 | S3 | 372 | 32/32 | 30/30 | 40.6/81.2/96.9% | .515 | 8.00 s |
| PyScene | L21_V001 | S4 | 369 | 32/32 | 30/30 | 40.6/81.2/96.9% | .514 | 8.00 s |
| PyScene | L21_V001 | S5 | 380 | 32/32 | 30/30 | 40.6/78.1/96.9% | .516 | 8.00 s |
| PyScene | L21_V029 | S0 | 287 | 25/26 | 23/24 | 65.4/96.2/96.2% | .772 | 15.00 s |
| PyScene | L21_V029 | S1 | 315 | 25/26 | 23/24 | 65.4/96.2/96.2% | .772 | 8.00 s |
| PyScene | L21_V029 | S2 | 320 | 25/26 | 23/24 | 73.1/96.2/96.2% | .822 | 8.00 s |
| PyScene | L21_V029 | S3 | 320 | 25/26 | 23/24 | 61.5/96.2/96.2% | .741 | 8.00 s |
| PyScene | L21_V029 | S4 | 311 | 25/26 | 23/24 | 65.4/96.2/96.2% | .772 | 8.00 s |
| PyScene | L21_V029 | S5 | 336 | 25/26 | 23/24 | 73.1/96.2/96.2% | .822 | 8.00 s |
| PyScene | L22_V018 | S0 | 392 | 25/25 | 16/16 | 56.0/96.0/100% | .659 | 24.57 s |
| PyScene | L22_V018 | S1 | 424 | 25/25 | 16/16 | 56.0/92.0/100% | .659 | 8.00 s |
| PyScene | L22_V018 | S2 | 428 | 25/25 | 16/16 | 56.0/92.0/100% | .660 | 7.90 s |
| PyScene | L22_V018 | S3 | 427 | 25/25 | 16/16 | 56.0/96.0/100% | .658 | 7.90 s |
| PyScene | L22_V018 | S4 | 418 | 25/25 | 16/16 | 56.0/96.0/100% | .661 | 7.90 s |
| PyScene | L22_V018 | S5 | 450 | 25/25 | 16/16 | 56.0/88.0/100% | .654 | 7.90 s |
| PyScene | L24_V043 | S0 | 105 | 14/16 | 14/16 | 6.2/43.8/87.5% | .204 | 19.12 s |
| PyScene | L24_V043 | S1 | 125 | 16/16 | 16/16 | 6.2/50.0/100% | .210 | 8.00 s |
| PyScene | L24_V043 | S2 | 124 | 16/16 | 16/16 | 6.2/43.8/100% | .215 | 7.80 s |
| PyScene | L24_V043 | S3 | 124 | 16/16 | 16/16 | 6.2/50.0/100% | .220 | 7.80 s |
| PyScene | L24_V043 | S4 | 125 | 16/16 | 16/16 | 6.2/43.8/100% | .214 | 7.80 s |
| PyScene | L24_V043 | S5 | 131 | 16/16 | 16/16 | 6.2/50.0/93.8% | .214 | 7.80 s |
| TransNet | L21_V001 | S0 | 336 | 32/32 | 30/30 | 37.5/78.1/96.9% | .510 | 16.10 s |
| TransNet | L21_V001 | S1 | 356 | 32/32 | 30/30 | 37.5/81.2/96.9% | .514 | 8.00 s |
| TransNet | L21_V001 | S2 | 370 | 32/32 | 30/30 | 37.5/78.1/96.9% | .508 | 7.93 s |
| TransNet | L21_V001 | S3 | 374 | 32/32 | 30/30 | 37.5/78.1/96.9% | .507 | 8.00 s |
| TransNet | L21_V001 | S4 | 368 | 32/32 | 30/30 | 37.5/78.1/96.9% | .506 | 8.00 s |
| TransNet | L21_V001 | S5 | 381 | 32/32 | 30/30 | 37.5/78.1/96.9% | .507 | 8.00 s |
| TransNet | L21_V029 | S0 | 282 | 26/26 | 24/24 | 57.7/100/100% | .742 | 11.47 s |
| TransNet | L21_V029 | S1 | 305 | 26/26 | 24/24 | 57.7/100/100% | .733 | 8.00 s |
| TransNet | L21_V029 | S2 | 324 | 26/26 | 24/24 | 65.4/100/100% | .786 | 8.00 s |
| TransNet | L21_V029 | S3 | 319 | 26/26 | 24/24 | 57.7/100/100% | .731 | 8.00 s |
| TransNet | L21_V029 | S4 | 314 | 26/26 | 24/24 | 57.7/100/100% | .736 | 8.00 s |
| TransNet | L21_V029 | S5 | 338 | 26/26 | 24/24 | 65.4/100/100% | .779 | 8.00 s |
| TransNet | L22_V018 | S0 | 383 | 25/25 | 16/16 | 56.0/92.0/100% | .662 | 14.80 s |
| TransNet | L22_V018 | S1 | 406 | 25/25 | 16/16 | 56.0/92.0/100% | .662 | 8.00 s |
| TransNet | L22_V018 | S2 | 422 | 25/25 | 16/16 | 56.0/92.0/100% | .661 | 8.00 s |
| TransNet | L22_V018 | S3 | 415 | 25/25 | 16/16 | 56.0/88.0/100% | .661 | 8.00 s |
| TransNet | L22_V018 | S4 | 412 | 25/25 | 16/16 | 56.0/92.0/100% | .662 | 8.00 s |
| TransNet | L22_V018 | S5 | 441 | 25/25 | 16/16 | 56.0/88.0/100% | .657 | 8.00 s |
| TransNet | L24_V043 | S0 | 84 | 14/16 | 14/16 | 6.2/50.0/87.5% | .207 | 19.12 s |
| TransNet | L24_V043 | S1 | 115 | 16/16 | 16/16 | 6.2/43.8/100% | .209 | 8.00 s |
| TransNet | L24_V043 | S2 | 113 | 16/16 | 16/16 | 6.2/43.8/100% | .212 | 7.48 s |
| TransNet | L24_V043 | S3 | 113 | 16/16 | 16/16 | 6.2/43.8/100% | .215 | 7.00 s |
| TransNet | L24_V043 | S4 | 114 | 16/16 | 16/16 | 6.2/43.8/100% | .212 | 7.48 s |
| TransNet | L24_V043 | S5 | 119 | 16/16 | 16/16 | 6.2/43.8/100% | .209 | 7.52 s |

## Event, repair, bytes, and compute totals

Exact per-video values are in the primary CSV. Aggregate fixed-retrieval vector/index bytes use one 512-D float32 Meta vector per selected frame and a real serialized FAISS `IndexFlatIP`; selector embedding time is separate.

| Detector | Strategy | Events / repairs | Selector CPU | Selector embedding | Edge | JPEG MiB | Vector / index MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| PyScene | S1 periodic | 0 / 0 | 0 | 0 | 0 | 201.4 | 2.40 / 2.40 |
| PyScene | S2 Meta | 53 / 68 | .156 s | 235.6 s | 0 | 203.0 | 2.43 / 2.43 |
| PyScene | S3 PE | 53 / 65 | .172 s | 504.5 s | 0 | 202.1 | 2.43 / 2.43 |
| PyScene | S4 BEiT | 37 / 67 | .188 s | 126.4 s | 0 | 199.7 | 2.39 / 2.39 |
| PyScene | S5 Meta+edge | 113 / 64 | .156 s | 235.6 s | 22.2 s | 211.4 | 2.53 / 2.53 |
| PyScene | PE+Meta | 86 / 65 | .266 s | 740.1 s | 0 | 206.5 | 2.49 / 2.49 |
| PyScene | PE+BEiT | 64 / 65 | .250 s | 630.8 s | 0 | 202.7 | 2.44 / 2.44 |
| PyScene | triple | 94 / 64 | .391 s | 866.5 s | 0 | 206.9 | 2.49 / 2.49 |
| TransNet | S1 periodic | 0 / 0 | 0 | 0 | 0 | 193.9 | 2.31 / 2.31 |
| TransNet | S2 Meta | 82 / 68 | .141 s | 235.6 s | 0 | 200.1 | 2.40 / 2.40 |
| TransNet | S3 PE | 72 / 68 | .156 s | 504.5 s | 0 | 198.4 | 2.38 / 2.38 |
| TransNet | S4 BEiT | 59 / 68 | .188 s | 126.4 s | 0 | 196.6 | 2.36 / 2.36 |
| TransNet | S5 Meta+edge | 138 / 64 | .125 s | 235.6 s | 22.0 s | 208.0 | 2.50 / 2.50 |
| TransNet | PE+Meta | 111 / 68 | .250 s | 740.1 s | 0 | 203.7 | 2.46 / 2.46 |
| TransNet | PE+BEiT | 89 / 68 | .281 s | 630.8 s | 0 | 200.3 | 2.41 / 2.41 |
| TransNet | triple | 119 / 68 | .375 s | 866.5 s | 0 | 204.8 | 2.47 / 2.47 |

## Provider-matched retrieval diagnostic

This diagnostic changes retrieval space with the selector and therefore must not be interpreted as selector-placement causality.

| Detector | Strategy | Macro R@10 | Pooled R@10 | Macro / pooled MRR |
|---|---|---:|---:|---:|
| PyScene | Meta | 77.5% | 77.8% | .553 / .512 |
| PyScene | PE | 83.4% | 82.8% | .583 / .583 |
| PyScene | BEiT | 81.6% | 80.8% | .551 / .544 |
| PyScene | PE+Meta | 84.2% | 84.8% | .584 / .553 |
| PyScene | PE+BEiT | **86.5%** | **85.9%** | **.595 / .581** |
| PyScene | triple | 83.4% | 82.8% | .590 / .579 |
| TransNet | Meta | 78.5% | 78.8% | .542 / .504 |
| TransNet | PE | 84.4% | 83.8% | .587 / .580 |
| TransNet | BEiT | 84.2% | 81.8% | .546 / .535 |
| TransNet | PE+Meta | 82.8% | 84.8% | .588 / .556 |
| TransNet | PE+BEiT | **87.5%** | **85.9%** | **.595 / .577** |
| TransNet | triple | 85.9% | 84.8% | .587 / .566 |

## Per-query wins/losses and provider agreement

Using pooled fixed-Meta ranks, content-aware placement often changes nothing. Representative wins/losses/ties versus periodic across the 99 queries:

| Detector | Strategy | Wins | Losses | Ties |
|---|---|---:|---:|---:|
| PyScene | Meta | 14 | 14 | 71 |
| PyScene | PE | 14 | 17 | 68 |
| PyScene | BEiT | 14 | 11 | 74 |
| PyScene | Meta+edge | 11 | 18 | 70 |
| PyScene | PE+Meta | 12 | 19 | 68 |
| PyScene | PE+BEiT | 14 | 19 | 66 |
| PyScene | triple | 12 | 20 | 67 |
| TransNet | Meta | 12 | 19 | 68 |
| TransNet | PE | 12 | 17 | 70 |
| TransNet | BEiT | 12 | 15 | 72 |
| TransNet | Meta+edge | 7 | 24 | 68 |
| TransNet | PE+Meta | 13 | 19 | 67 |
| TransNet | PE+BEiT | 12 | 19 | 68 |
| TransNet | triple | 13 | 20 | 66 |

The providers are related but not interchangeable. Candidate-weighted per-shot Spearman novelty correlations are `0.628` Meta/PE, `0.634` Meta/BEiT, and `0.588` PE/BEiT under PyScene; under TransNet they are `0.613`, `0.621`, and `0.574`. Across all shots, the top novelty candidate differs in 26.0–26.9% of PyScene shots and 27.8–28.8% of TransNet shots, depending on pair. Restricting to multi-candidate shots increases those ranges to 27.7–28.7% and 29.2–30.2%. This diversity is real, but the tested RRFs do not convert it into a consistent fixed-retrieval gain.

## Quality/cost Pareto frontier

The frontier treats pooled Selection Recall, R@1/10/100, and MRR as qualities to maximize, and KF count, selector embedding time, and vector/index bytes as costs to minimize. Nondominated strategies are:

- PyScene: midpoint, periodic, Meta, Meta+edge, PE+Meta RRF, adaptive BEiT;
- TransNet: midpoint, periodic, Meta.

This is a multi-objective frontier, not a recommendation by itself. Periodic is the only non-midpoint strategy on both detector frontiers with zero selector-model compute.

## Reproducibility and artifacts

Environment: Python 3.12.10, PyTorch 2.5.1+cu124, CUDA 12.4, NumPy 2.1.2, PySceneDetect 0.7.1, `transnetv2-pytorch` 1.0.5, Transformers 4.57.1, and FAISS CPU 1.15.0. BEiT used its isolated `timm==0.4.12`; PE used its pinned source/runtime.

Generated artifacts remain ignored under `artifacts/verification/ingest_kf04_multivideo/`. Key evidence files:

- `evaluation_manifest.json`: pass ordering, competitiveness gate, selected fusion/adaptive providers;
- `frozen_shots/<detector>/<video>.json`: exact manifests and source hashes;
- `preparation_summary.json`: union extraction/cache evidence;
- `embeddings/<provider>/<video>/manifest.json`: identity, mapping hash, vector hash, runtime, bytes;
- `all_selections.csv`: every selected frame for every detector/strategy;
- `all_metrics_fixed_metaclip2.csv`: primary per-video and aggregate metrics, runtime, JPEG/vector/index bytes;
- `all_per_query_fixed_metaclip2.csv`: primary local and pooled query ranks;
- `all_metrics.csv` and `all_per_query.csv`: provider-matched diagnostic;
- `provider_novelty_comparison.csv` and `analysis_summary.json`: correlations, top-1 agreement, wins/losses, deltas, Pareto frontier;
- `edge_scores/**/*.json`: exact recomputation timings and cache-validation result.

No raw video, candidate JPEG, vector, index, checkpoint, tokenizer, or model source is tracked. No production default was changed. No merge or push was performed.

## Limitations and next experiment

This is still only four videos from a much larger corpus, with manually authored research GT and very uneven retrieval difficulty (especially L24_V043). It has no preregistered uncertainty interval, no additional held-out threshold-validation split, and no official AIC relevance judgments. The small adaptive-count reduction and detector-dependent encoder ordering need replication on more annotated videos before any production decision.

The appropriate follow-up is a preregistered held-out expansion that freezes thresholds and the competitiveness gate before evaluating additional videos. It should retain fixed-provider retrieval as the primary placement test and provider-matched retrieval as a separate encoder-stack diagnostic.
