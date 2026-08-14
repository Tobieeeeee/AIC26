# 00 — Master Plan

## Objective

Build the first runnable Multimedia Video Retrieval baseline by reusing code already created in the existing labs.

Core principle:

```text
Reuse
  ↓
Make runnable
  ↓
Verify
  ↓
Document
  ↓
Commit
  ↓
Push
```

Only after the baseline is stable should later work:

```text
Refactor
Benchmark
Replace
Optimize
Extend
```

## Source labs

Read-only source directory:

```text
D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\New folder
```

## Target outcome

The minimum runnable vertical slice is:

```text
VIDEO
  ↓
Shot Detection
  ↓
Keyframe Extraction
  ↓
OpenCLIP Image Embeddings
  ↓
FAISS
  ↓
Text Query
  ↓
OpenCLIP Text Embedding
  ↓
Top-K Keyframes
  ↓
video_id + timestamp + score
```

## Phase order

### Phase 01 — Discover and map

Read all labs and create `docs/LAB_REUSE_MAP.md`.

No production migration yet.

### Phase 02 — Reuse ingestion

Reuse the Shot Boundary Lab:

```text
video
  ↓
probe
  ↓
shot detection
  ↓
keyframe selection
  ↓
frame extraction
  ↓
metadata
```

### Phase 03 — Build semantic baseline

Reuse:

- OpenCLIP from Lab 3
- FAISS from Lab 4

Connect them to Phase 02.

This phase must create the first real end-to-end baseline.

### Phase 04 — Reuse multimodal components

Reuse Lab 6:

- EasyOCR
- PaddleOCR
- YOLOv8n
- Elasticsearch
- BM25
- object retrieval
- semantic kNN
- RRF
- weighted hybrid

Do not tune or upgrade them.

### Phase 05 — API and tests

Reuse FastAPI patterns from the FastAPI Lab and Lab 6.

Expose only endpoints backed by real implementations.

### Phase 06 — Verify and publish

Run final verification, inspect Git, push the feature branch, and create a PR to `main`.

Do not merge automatically.

## Execution rule

Execute **ONE PHASE AT A TIME**.

After each phase:

1. run required tests;
2. update `docs/CODEX_PROGRESS.md`;
3. create a logical commit;
4. stop;
5. report what was completed and what remains.

Do not automatically continue to the next phase.

## Recommended commit sequence

```text
docs: map reusable lab implementations

feat: reuse shot boundary ingestion pipeline

feat: reuse CLIP and FAISS semantic baseline

feat: reuse retrieval evaluation metrics

feat: reuse Lab 6 multimodal retrieval components

feat: reuse FastAPI serving baseline

test: verify reused MVR baseline
```

Do not create meaningless commits just to match this list.
