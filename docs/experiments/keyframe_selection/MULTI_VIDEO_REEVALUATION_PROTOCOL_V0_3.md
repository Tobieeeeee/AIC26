# AIC26 — Multi-Video Keyframe Selector Re-evaluation Protocol v0.3

## Goal

Re-evaluate keyframe selection on 4 videos / 99 detector-independent temporal GT segments, while separating:

1. shot-detector effects;
2. temporal-only selection;
3. single-encoder content-aware selection;
4. edge/text-change assistance;
5. multi-encoder fusion;
6. adaptive-count/KNN-style exploratory selection.

Do not change the production default during this experiment.

---

# Dataset

Videos:

```text
L21_V001
L21_V029
L22_V018
L24_V043
```

Ground truth:

```text
AIC26_keyframe_selection_groundtruth_4videos_v0_2.csv
```

Total:

```text
99 query → temporal-segment GT records
```

GT is manual and independent of PySceneDetect and TransNetV2.

---

# A. Shot regimes

For every video create two frozen shot manifests:

```text
D1 = PySceneDetect
D2 = TransNetV2
```

Every selector under one detector must consume the exact same frozen shot manifest.

Report detector statistics before selector evaluation:

```text
shot count
median shot duration
p95 shot duration
max shot duration
% shots >= 12s
% shots >= 30s
% shots >= 60s
```

---

# B. Tier 0 — temporal controls

Mandatory:

```text
S0 midpoint-only
S1 periodic 12s/8s
```

These establish the temporal-only baseline.

---

# C. Tier 1 — single-encoder selectors

Use the SAME generic selector mechanics for each encoder so only the visual representation changes.

Candidate grid:

```text
1 second
midpoint mandatory
max temporal gap <= 8s
```

Test:

```text
S2-Meta
MetaCLIP-2 only

S3-PE
PE-Core-L only

S4-BEiT
BEiT-3 only
```

Single-encoder selector should use:

```text
local temporal novelty
rare/transient preservation
temporal coverage repair
```

Do NOT include edge-change in these three variants.

This answers:

```text
Which single visual embedding is best for keyframe placement?
```

Provider identities must match the project's pinned retrieval/feature providers.

---

# D. Tier 2 — encoder + cheap text/slide-change signal

Test:

```text
S5-MetaEdge
MetaCLIP-2 novelty
+ edge/text-sensitive visual change
+ temporal safety

S6-PEEdge
PE-Core-L novelty
+ edge/text-sensitive visual change
+ temporal safety

S7-BEiTEdge
BEiT-3 novelty
+ edge/text-sensitive visual change
+ temporal safety
```

`S5-MetaEdge` is equivalent to the current hybrid temporal-content idea when configuration matches.

This tier asks whether the cheap edge/slide proxy adds retrieval value beyond the encoder alone.

---

# E. Tier 3 — multi-encoder fusion

Primary fusion variants:

```text
S8-PEMeta
PE + MetaCLIP

S9-MetaBEiT
MetaCLIP + BEiT-3

S10-PEBEiT
PE + BEiT-3

S11-Triple
PE + MetaCLIP + BEiT-3
```

Use rank-level fusion first, not raw vector concatenation.

Primary fusion:

```text
RRF with k=60
```

At each candidate-selection step:

1. rank candidates independently by each provider's local novelty;
2. fuse provider ranks;
3. select under the same midpoint / rare-event / <=8s temporal constraints.

For two encoders:

```text
RRF(c) = 1/(60 + rank_a(c)) + 1/(60 + rank_b(c))
```

For three encoders add all three terms.

Do not allow fusion variants a larger keyframe budget merely because they use more models.

---

# F. Tier 4 — fusion + edge signal

Only run if Tier 3 is computationally feasible after cache reuse.

Test:

```text
S12-PEMetaEdge
S13-TripleEdge
```

These combine fused semantic/visual novelty with the cheap edge/text-change event signal and temporal coverage repair.

Do not expand this tier unless Tier 3 shows evidence of complementarity.

---

# G. Tier 5 — adaptive-count / KNN exploratory variants

These are exploratory and must NOT replace the main fixed-safety evaluation.

Test at minimum:

```text
S14-MetaKNN
S15-PEKNN
```

Optional:

```text
S16-PEMetaKNN
```

Purpose:

```text
let visual-state structure influence HOW MANY keyframes are retained,
not only WHERE a fixed/safety-constrained selector places them.
```

Requirements:

```text
candidate grid = 1s
temporal-local KNN only
rare isolated/transient states are retained
max temporal gap <= 8s remains mandatory
hard deterministic per-shot cap
```

Do NOT use global KNN density that can discard rare events.

Suggested initial KNN values:

```text
k = 3 and 5
temporal neighborhood <= 8s
```

Treat threshold sweeps as development evidence; do not tune and evaluate on the same held-out claim set.

---

# H. Shared candidate/embedding cache

Extract the 1-second candidate pool once per:

```text
video + detector frozen-shot manifest
```

Reuse candidate JPEGs for all content-aware selectors.

Compute each provider embedding once per candidate pool:

```text
MetaCLIP-2
PE-Core-L
BEiT-3
```

Reuse them across single, edge, fusion, and KNN variants.

Validate cache identity:

```text
video/source fingerprint
shot manifest hash
candidate mapping hash
provider identity
provider revision/checkpoint
preprocessing
embedding dimension
normalization
artifact hash
```

Do not re-encode the same provider/candidate pool for each strategy.

---

# I. Fairness rules

Hold fixed across selectors within each detector regime:

```text
video
frozen shots
candidate pool
ground truth
query language
retrieval provider(s)
query preprocessing
FAISS/index type
top_k
tie-breaking
relevance intervals
```

The only changing variable should be keyframe selection.

Do not change thresholds after examining one selector's query failures and then present the same evaluation as unbiased.

---

# J. Required metrics

Per:

```text
video × detector × selector
```

report:

```text
selected keyframes
KF/shot
JPEG bytes
Selection Recall
high-confidence Selection Recall
Recall@1
Recall@10
Recall@100
MRR
max temporal gap
content-event count
temporal-repair count
rare-event count
candidate extraction time
provider embedding time
selector-only time
index/vector bytes
```

Also report:

```text
macro-average across videos
pooled-query metrics
median per-video metric
worst-video metric
```

The worst-video metric matters because a selector that wins only one video type is not a safe canonical default.

---

# K. Encoder complementarity analysis

For MetaCLIP, PE, and BEiT-3 compare candidate novelty rankings.

Report:

```text
Spearman correlation
same top-1 candidate rate
different top-1 candidate rate
average temporal distance between provider choices
per-long-shot disagreement
```

This determines whether fusion is justified.

If two providers rank almost identically, their fusion should be deprioritized.

---

# L. Video-type analysis

Classify pilot videos descriptively, not as hard labels:

```text
many-cut/news-like
long visually similar sequence
mixed scenes/events
long static-camera/presentation-like if present
```

For each type answer:

```text
Does temporal-only work?
Does single Meta help?
Does single PE help?
Does single BEiT help?
Does edge assistance help?
Does fusion help?
Does adaptive KNN reduce KFs without hurting retrieval?
```

Special attention:

```text
L24_V043
```

because its long lion-dance sequence provides a useful visually similar but action-changing stress case.

---

# M. Pareto analysis

Do not select a winner by Recall@10 alone.

Construct Pareto comparisons over:

```text
retrieval quality
selected KF count
candidate preprocessing cost
embedding cost
storage/index size
```

Primary quality dimensions:

```text
Recall@10
MRR
Recall@100 safety
```

Primary cost dimensions:

```text
selected KFs
total candidate/provider GPU time
downstream keyframe-native work implied by KF count
```

---

# N. Suggested execution tiers

To control runtime:

## Pass 1 — mandatory

```text
S0 midpoint
S1 periodic
S2 Meta
S3 PE
S4 BEiT
S5 MetaEdge
```

If these are enough to clearly eliminate providers, stop eliminated branches.

## Pass 2 — fusion

Run only providers that survived Pass 1:

```text
best pair fusion
triple fusion only if all three remain competitive
```

## Pass 3 — adaptive-count/KNN

Run on the best one or two representations from Pass 1/2.

This avoids blindly running every combinatorial strategy.

---

# O. Decision questions

The final report must answer:

1. Does periodic 12s/8s still lead under TransNetV2?
2. Is PySceneDetect's short-shot distribution helping midpoint/periodic?
3. Which single encoder is best: MetaCLIP-2, PE-Core-L, or BEiT-3?
4. Does edge/text-sensitive change help each single encoder?
5. Are PE, MetaCLIP, and BEiT sufficiently complementary to justify fusion?
6. Does PE+Meta beat the best single model?
7. Does triple fusion beat the best pair enough to justify extra compute?
8. Can KNN/adaptive-count reduce keyframes without hurting Recall@10/100?
9. Which strategy has the best worst-video behavior?
10. Which detector + selector lies on the best quality/cost Pareto frontier?
11. Is there enough evidence to change the production default?

---

# P. Decision safety

Do not change production default during the experiment.

If thresholds are tuned using these four videos, these four videos become development data.

A final threshold-tuned production claim then requires additional held-out annotated videos.

No merge.
No push.
