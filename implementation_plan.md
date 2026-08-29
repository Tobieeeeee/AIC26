# DATA-02 — Shared Corpus, Parquet/Polars Layer, and Canonical System Defaults

## Background

This is a large infrastructure task that builds the shared data-platform layer for AIC26. **None of the required infrastructure exists yet** — no R2 client, no Polars layer, no Parquet schemas, no staging/promote workflow, no bounded cache, no `configs/production.yaml`, no RF-DETR provider identity, and no canonical system defaults document.

The existing `configs/default.yaml` has **stale defaults** that conflict with the task's frozen production contract (PySceneDetect instead of TransNetV2, PaddleOCR instead of DeepSolo, YOLOv8n instead of RF-DETR, single CLIP instead of 3-provider stack).

### Current state vs. required state

| Area | Current | Required |
|---|---|---|
| Ingestion default | `pyscenedetect` | `transnetv2` |
| Object default | `yolov8n` | `rfdetr` |
| OCR default | `paddleocr` | `deepsolo + parseq_vietnamese` |
| Embedding default | `clip` only | `beit3, metaclip2, perception_encoder` |
| ASR default | ✅ Already correct (Silero+ChunkFormer) | Same |
| R2 client | ❌ None | S3-compatible client layer |
| Bounded cache | ❌ None | Configurable cache manager |
| Parquet metadata | ❌ None | Partitioned Parquet per feature type |
| Polars layer | ❌ None | Lazy scan/filter/join API |
| Staging → Promote | ❌ None | Validate + promote workflow |
| Extraction status | ❌ None | Status control plane |
| `configs/production.yaml` | ❌ None | Canonical system defaults |

---

## User Review Required

> [!IMPORTANT]
> This is a **very large task** (~23 sections, ~15-20 new files, extensive testing). It will require significant implementation time. I recommend proceeding in phases to keep reviews manageable.

> [!WARNING]
> The task requires changing `configs/default.yaml` to align with the frozen production contract. The current defaults (PySceneDetect, PaddleOCR, YOLOv8n, single CLIP) will be **replaced** with TransNetV2, DeepSolo+PARSeq, RF-DETR, and the 3-provider stack. The existing defaults represent stale lab-era configurations.

---

## Open Questions

> [!IMPORTANT]
> **Git base**: The task says to start from `feat/data02-shared-corpus-parquet` branch. Current branch is `fix/ocr-production-checkpoints`. Should I:
> - Create the new branch from the current HEAD (`37696aa`)?
> - Or is there a specific base branch/commit you'd prefer?

> [!IMPORTANT]
> **RF-DETR**: The task acknowledges that a fully pinned RF-DETR variant/checkpoint may not exist yet. Should I create the provider identity as `rfdetr` with explicit "pin pending" status and fail-fast behavior, per the task instructions?

> [!IMPORTANT]
> **R2 credentials**: The task references environment variables (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, etc.). The R2 access audit already verified these exist. Should I run a tiny canary against `aic-test` bucket during verification, or keep tests fully mocked?

---

## Proposed Changes

I'll organize implementation into **5 phases** to keep things tractable:

---

### Phase 1: Canonical System Defaults & Config

Freeze the production defaults first so everything else builds on them.

#### [NEW] [production.yaml](file:///d:/project/STUDY/NAM_2/SELF_PRJ/AIC26/SkillPixel/mvr-competition-starter/configs/production.yaml)
- Canonical production config: TransNetV2 @ 0.5, periodic 12/8, RF-DETR, BEiT-3/MetaCLIP-2/PE, DeepSolo+PARSeq+F-1/F/F+1, Silero+ChunkFormer

#### [MODIFY] [default.yaml](file:///d:/project/STUDY/NAM_2/SELF_PRJ/AIC26/SkillPixel/mvr-competition-starter/configs/default.yaml)
- Align defaults with production contract (TransNetV2, rfdetr, deepsolo, 3-provider stack)
- Keep legacy values available only via explicit override

#### [NEW] [CANONICAL_SYSTEM_DEFAULTS.md](file:///d:/project/STUDY/NAM_2/SELF_PRJ/AIC26/SkillPixel/mvr-competition-starter/docs/architecture/CANONICAL_SYSTEM_DEFAULTS.md)
- Human-readable summary of the frozen production stack

---

### Phase 2: R2 Client & Bounded Cache

#### [NEW] `src/mvr/operations/r2_client.py`
- S3-compatible R2 client using `boto3`
- Credentials from env vars only
- list/head/download/upload/delete
- Retry policy, multipart upload, deterministic key handling

#### [NEW] `src/mvr/operations/cache.py`
- Bounded local cache manager
- Configurable size limit, directory, eviction
- Atomic downloads, concurrent-safe reads
- Hit/miss stats

#### [NEW] `src/mvr/operations/r2_layout.py`
- Canonical R2 object-key layout (raw/staging/corpus)
- Path resolution helpers

---

### Phase 3: Parquet/Polars Layer & Staging→Promote

#### [NEW] `src/mvr/operations/parquet_schemas.py`
- Pyarrow/Polars schemas for keyframes, OCR, ASR, objects, extraction status
- Batch/collection partitioning logic

#### [NEW] `src/mvr/operations/consolidate.py`
- Staging JSON → canonical Parquet consolidation
- Schema normalization, identity join, dedup, partition

#### [NEW] `src/mvr/operations/promote.py`
- Validate + promote contract
- Schema check, identity check, count check, checksum
- Idempotent, no partial writes
- Manifest/status update after success

#### [NEW] `src/mvr/operations/polars_api.py`
- Lazy Parquet scans
- Filter by batch/collection/video
- Duplicate detection, missing-feature discovery
- Join metadata on `keyframe_id`
- Task05A document preparation

#### [NEW] `src/mvr/operations/extraction_status.py`
- Status control plane (TODO/RUNNING/DONE/FAILED/STALE)
- Schema: `keyframe_id`, `batch_id`, `collection_id`, per-feature status, `updated_at`

---

### Phase 4: RF-DETR Provider Identity & Manifest Provenance

#### [NEW] `src/mvr/enrichment/objects/rfdetr.py`
- Provider identity `rfdetr`
- Explicit variant/checkpoint/revision fields
- Fail-fast if production runtime invoked without pinned model
- Same output contract as existing detectors (joinable by `keyframe_id`)

#### [MODIFY] `src/mvr/enrichment/objects/__init__.py`
- Register RF-DETR as available backend
- Keep Faster R-CNN as legacy/experimental

---

### Phase 5: CLI Commands, Tests, and Report

#### [MODIFY] [cli.py](file:///d:/project/STUDY/NAM_2/SELF_PRJ/AIC26/SkillPixel/mvr-competition-starter/src/mvr/cli.py)
- Add commands: `r2-status`, `r2-download`, `r2-upload`, `validate-staging`, `promote`, `consolidate-parquet`, `audit-status`, `show-defaults`

#### [NEW] `tests/unit/test_production_defaults.py`
- Assert canonical defaults resolve correctly
- Assert no default Faster R-CNN, no default EasyOCR, no default PySceneDetect, no default single-CLIP

#### [NEW] `tests/unit/test_r2_client.py`
- Object-key normalization, mock roundtrip, missing object, checksum mismatch

#### [NEW] `tests/unit/test_cache.py`
- Hit, miss, no-redownload, size-bound eviction, partial download rejection

#### [NEW] `tests/unit/test_parquet_schemas.py`
- Schema validation, partitions, dedup rejection, consolidation, lazy scans, empty partition

#### [NEW] `tests/unit/test_promote.py`
- Validation before promote, idempotent promote, failed validation, manifest update

#### [NEW] `docs/reviews/DATA02_SHARED_CORPUS_REPORT.md`
- Final report per task spec §22

---

## Verification Plan

### Automated Tests
```powershell
pytest tests/unit/test_production_defaults.py -v
pytest tests/unit/test_r2_client.py -v
pytest tests/unit/test_cache.py -v
pytest tests/unit/test_parquet_schemas.py -v
pytest tests/unit/test_promote.py -v
pytest tests/ -k "not benchmark" --tb=short
```

### Manual Verification
- `python -m mvr.cli show-defaults` prints canonical production stack
- `git diff --check` passes
- No credentials, model weights, or large data committed
