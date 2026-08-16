# 04 — Reuse Lab 6 Multimodal Retrieval Components

## Goal

After the semantic baseline works, reuse the existing Lab 6 components without redesigning or tuning them.

Reuse where practical:

- EasyOCR
- PaddleOCR
- YOLOv8n
- Elasticsearch mapping
- BM25 retrieval
- object retrieval
- semantic kNN
- RRF
- weighted hybrid fusion

## Required preflight

Read:

```text
AGENTS.md
codex/00_MASTER_PLAN.md
codex/04_REUSE_MULTIMODAL.md
docs/LAB_REUSE_MAP.md
docs/CODEX_PROGRESS.md
```

Confirm Phase 03 is complete.

## OCR

Create:

```text
src/mvr/enrichment/ocr/easyocr_engine.py
src/mvr/enrichment/ocr/paddleocr_engine.py
```

Reuse original Lab 6 behavior.

Preserve confidence thresholds and defaults where practical.

Do not tune OCR.

## Object detection

Create:

```text
src/mvr/enrichment/objects/yolo.py
```

Reuse YOLOv8n logic from Lab 6.

Preserve original object-count behavior.

Expected conceptual output:

```python
{
    "person": 2,
    "car": 1
}
```

Do not upgrade YOLO in this phase.

## Elasticsearch

Create:

```text
src/mvr/indexing/elasticsearch_index.py
```

Reuse the original mapping concepts:

- `frame_id`
- `video_id`
- `timestamp`
- OCR text
- object metadata
- dense vector

Extend only what is necessary to connect real video keyframes.

Do not redesign the Elasticsearch architecture.

## Retrieval modules

Create or complete:

```text
src/mvr/retrieval/lexical.py
src/mvr/retrieval/object_search.py
src/mvr/retrieval/semantic.py
src/mvr/retrieval/hybrid.py
```

Reuse:

- BM25 logic
- object query logic
- semantic kNN behavior
- RRF
- weighted hybrid
- per-query normalization

Inspect existing `hybrid.py` before replacing anything.

Reconcile duplicate implementations and document which behavior is the source of truth.

## Important policy

Do not tune:

- alpha weights
- OCR thresholds
- YOLO confidence
- retrieval K values
- model choices

unless the original implementation cannot run as-is.

If a compatibility fix is required, document it in `docs/LAB_REUSE_MAP.md`.

## Tests

Add lightweight tests for:

- object query parsing
- fusion behavior
- metadata/index document construction
- lexical query request construction
- normalization
- error handling

Heavy Elasticsearch/model tests may be marked as integration tests.

## Git

Before commit:

```bash
git status -sb
git diff --stat
git diff
pytest -q
```

Suggested commit:

```text
feat: reuse Lab 6 multimodal retrieval components
```

## Finish

Update:

```text
docs/LAB_REUSE_MAP.md
docs/CODEX_PROGRESS.md
README.md
```

README must distinguish:

- actually working features
- optional integrations
- features requiring Elasticsearch/models

Stop after Phase 04.

Do not begin API work automatically.
