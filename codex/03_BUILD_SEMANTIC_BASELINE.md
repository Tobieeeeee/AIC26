# 03 — Build the First Semantic End-to-End Baseline

## Goal

Create the first true runnable Multimedia Video Retrieval vertical slice by reusing:

- OpenCLIP logic from Lab 3
- FAISS logic from Lab 4
- ingestion pipeline from Phase 02

Target:

```text
Video
  ↓
Keyframes
  ↓
OpenCLIP image embeddings
  ↓
FAISS IndexFlatIP
  ↓
Text query
  ↓
OpenCLIP text embedding
  ↓
Top-K keyframes
```

## Required preflight

Read:

```text
AGENTS.md
codex/00_MASTER_PLAN.md
codex/03_BUILD_SEMANTIC_BASELINE.md
docs/LAB_REUSE_MAP.md
docs/CODEX_PROGRESS.md
```

Confirm Phase 02 is complete.

## Part A — Reuse Lab 3 OpenCLIP

Inspect the actual Lab 3 notebook/source under:

```text
D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\New folder
```

Extract the existing OpenCLIP baseline behavior.

Create:

```text
src/mvr/enrichment/embeddings/openclip.py
```

Use a minimal production interface such as:

```python
class OpenClipEncoder:
    def encode_images(...):
        ...

    def encode_text(...):
        ...
```

Preserve original:

- model selection
- preprocessing
- image encoding
- text encoding
- normalization
- similarity assumptions

Do not upgrade to a different model during this phase.

SigLIP/SigLIP2 may remain documented/reference-only unless they are needed by the original baseline.

## Part B — Reuse Lab 4 FAISS

Extract the existing FAISS implementation.

Create:

```text
src/mvr/indexing/faiss_index.py
```

Preserve:

```text
IndexFlatIP
L2-normalized vectors
inner-product search
```

Provide only the minimum wrapper needed:

```python
build(...)
search(...)
save(...)
load(...)
```

Add metadata mapping between FAISS rows and at minimum:

```text
frame_id
video_id
timestamp
image_path
```

Do not change retrieval mathematics.

## Part C — Semantic retriever

Create:

```text
src/mvr/retrieval/semantic.py
```

It should use:

- the embedding module for text encoding
- the FAISS wrapper for search
- metadata mapping for result reconstruction

Avoid duplicating model logic.

## Part D — CLI

Extend:

```text
src/mvr/cli.py
```

Target commands:

```powershell
python -m mvr.cli ingest "<video-directory>"

python -m mvr.cli build-index

python -m mvr.cli search "a man riding a motorcycle"
```

Exact syntax may differ if needed, but document only verified commands.

Search output must include at minimum:

```text
rank
frame_id
video_id
timestamp
score
image_path
```

## Evaluation reuse

If lightweight and practical in this phase, extract Lab 4 metrics:

```text
Recall@K
MRR
```

into:

```text
src/mvr/evaluation/metrics.py
```

Otherwise leave it for a small follow-up commit within this phase.

## Tests

Add lightweight tests for:

- FAISS wrapper with tiny synthetic normalized vectors
- metadata mapping
- semantic result reconstruction
- Recall@K
- MRR
- CLI argument validation where practical

Heavy OpenCLIP integration tests may be marked/skipped if model download is unavailable.

## Definition of done

This phase is complete only when a semantic video retrieval vertical slice is runnable or all blockers are explicitly documented.

The expected flow is:

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
Top-K Keyframes
```

## Git

Before commit:

```bash
git status -sb
git diff --stat
git diff
pytest -q
```

Suggested commits:

```text
feat: reuse CLIP and FAISS semantic baseline
```

and, if separate:

```text
feat: reuse retrieval evaluation metrics
```

## Finish

Update:

```text
docs/LAB_REUSE_MAP.md
docs/CODEX_PROGRESS.md
README.md
```

Stop after this phase.

Do not begin OCR/YOLO/Elasticsearch work automatically.
