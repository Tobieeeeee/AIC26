# 01 — Discover and Map Existing Labs

## Goal

Inspect the existing lab implementations and map reusable components into the target MVR repository.

This phase is analysis/documentation only.

Do **not** migrate production code yet.

## Required preflight

Read:

```text
AGENTS.md
codex/00_MASTER_PLAN.md
docs/CODEX_PROGRESS.md
README.md
docs/ARCHITECTURE.md
docs/ROADMAP.md
docs/SOURCE_FILE_ANALYSIS.md
docs/GIT_WORKFLOW.md
configs/default.yaml
pyproject.toml
.gitignore
```

If some files do not exist, note that clearly.

## Inspect source directory

Read-only source:

```text
D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\New folder
```

Inspect the full directory tree and identify the actual names/locations of:

- Shot Boundary Lab
- Lab 3
- Lab 4
- Lab 6
- FastAPI Lab

Do not assume names blindly.

## Inspect expected reusable components

### Shot Boundary Lab

Locate:

- video probing
- `VideoInfo`, `Shot`, `Keyframe`
- PySceneDetect integration
- TransNetV2 integration
- keyframe selection
- FFmpeg extraction
- metadata generation
- pipeline orchestration
- tests

### Lab 3

Locate:

- OpenCLIP model loading
- image preprocessing
- image encoding
- text encoding
- L2 normalization
- similarity logic
- SigLIP / SigLIP2 experiments
- Vietnamese-related retrieval experiments

### Lab 4

Locate:

- FAISS `IndexFlatIP`
- vector normalization
- indexing
- searching
- Recall@K
- MRR
- latency measurement
- candidate retrieval
- reranking experiment

### Lab 6

Locate:

- EasyOCR
- PaddleOCR
- YOLOv8n
- CLIP embeddings
- Elasticsearch mapping
- BM25 retrieval
- object retrieval
- semantic kNN retrieval
- RRF
- weighted hybrid fusion
- FastAPI prototype

### FastAPI Lab

Locate:

- FastAPI structure
- Pydantic schemas
- request validation
- API tests
- GitHub Actions CI

## Required output

Create or update:

```text
docs/LAB_REUSE_MAP.md
```

Use a table such as:

| Source Lab | Original Source | Reused Component | Destination | Expected Changes | Status |
|---|---|---|---|---|---|

Every important reusable component should have a mapping.

## Also inspect Git

Run:

```bash
git rev-parse --show-toplevel
git status -sb
git branch --show-current
git remote -v
```

Do not change branches or push in this phase unless needed to establish the feature branch safely.

If currently on `main`, create:

```bash
git switch -c feat/reuse-lab-baseline
```

Do not overwrite an existing branch without inspecting it first.

## Acceptance criteria

- All relevant labs were located.
- Their reusable components were identified.
- `docs/LAB_REUSE_MAP.md` exists and is useful.
- No source lab was modified.
- No production algorithm was migrated yet.
- No model/algorithm replacement occurred.

## Finish

Update `docs/CODEX_PROGRESS.md`.

Commit with a message similar to:

```text
docs: map reusable lab implementations
```

Stop after this phase.

Do not begin Phase 02.
