# DATA-02: Shared Corpus, Parquet/Polars Layer, and System Defaults — Report

**Status**: ✅ IMPLEMENTED  
> Superseded status (2026-08-24): TASK-FIX-02 completed the RF-DETR runtime pin.
> Pending references below describe DATA-02's state at its original freeze.

**Branch**: `feat/data02-shared-corpus-parquet`  
**Date**: 2026-08-24  

---

## Summary

This report documents the implementation of DATA-02 infrastructure:

1. **Canonical system defaults** — frozen production stack aligned across configs
2. **R2/S3-compatible client** — with retry, atomic downloads, checksum verification
3. **Bounded local cache** — LRU eviction, concurrent-safe, hit/miss stats
4. **Parquet metadata schemas** — canonical schemas for all feature types
5. **Staging → Consolidation → Promote** — validated data lifecycle
6. **Polars metadata access** — lazy scans, joins, missing-feature discovery
7. **Extraction status control plane** — per-keyframe progress tracking
8. **RF-DETR provider identity** — fail-fast for unpinned runtime
9. **CLI commands** — `show-defaults`, `audit-status`, `consolidate-parquet`, `validate-staging`, `promote`

---

## Canonical System Defaults

| Component | Production Default | Legacy (opt-in only) |
|---|---|---|
| Shot detection | TransNetV2 @ 0.5 | PySceneDetect |
| Keyframe selection | periodic 12s/8s | — |
| Object detection | RF-DETR (pin pending) | Faster R-CNN, YOLOv8n |
| Visual embeddings | BEiT-3, MetaCLIP-2, PE-Core-L | Single OpenCLIP |
| OCR | DeepSolo + Vietnamese PARSeq | EasyOCR, PaddleOCR |
| ASR | Silero VAD + ChunkFormer CTC | — |

**Config hierarchy**:
- `configs/production.yaml` — authoritative source of truth
- `configs/default.yaml` — runtime defaults (aligned)
- `configs/embedding_providers.json` — pinned provider specs

---

## Files Created

### Configs & Docs
| File | Purpose |
|---|---|
| `configs/production.yaml` | Canonical production defaults |
| `docs/architecture/CANONICAL_SYSTEM_DEFAULTS.md` | Human-readable defaults reference |

### Operations Layer (`src/mvr/operations/`)
| Module | Purpose |
|---|---|
| `r2_client.py` | S3-compatible R2 client with retry/atomic/checksum |
| `cache.py` | Bounded local cache with LRU eviction |
| `r2_layout.py` | Canonical R2 object-key layout |
| `parquet_schemas.py` | Schema definitions for all feature types |
| `consolidate.py` | Staging JSON → Parquet consolidation |
| `promote.py` | Validate + promote to canonical corpus |
| `polars_api.py` | Lazy Parquet scans, joins, missing-feature discovery |
| `extraction_status.py` | Per-keyframe extraction progress tracking |

### Enrichment
| Module | Purpose |
|---|---|
| `src/mvr/enrichment/objects/rfdetr.py` | RF-DETR provider identity with fail-fast |

### Files Modified
| File | Change |
|---|---|
| `configs/default.yaml` | Aligned with production contract |
| `src/mvr/enrichment/objects/__init__.py` | Registered RF-DETR as production default |
| `src/mvr/cli.py` | Added 5 new CLI commands |

### Tests
| File | Coverage |
|---|---|
| `tests/unit/test_production_defaults.py` | 36 tests: config defaults, RFDETR fail-fast, no-legacy-fallback |
| `tests/unit/test_r2_client.py` | 21 tests: layout, client mocks, cache hit/miss/eviction |
| `tests/unit/test_parquet_promote.py` | (polars-dependent) consolidation, promotion, extraction status |

---

## Test Results

```
57 passed, 1 skipped
```

- All 36 production defaults tests: **PASSED**
- All 21 R2/cache tests: **PASSED**
- Parquet/promote tests: **SKIPPED** (polars not installed in base environment; tests are correctly guarded with `pytest.importorskip`)

---

## RF-DETR Status

RF-DETR is registered as the production default object detector, but the exact
variant/checkpoint are **NOT YET PINNED**. The runtime enforces fail-fast:

- `RFDETRConfig.is_pinned` → `False`
- `RFDETRDetector.detect()` → raises `RFDETRPinError`
- No silent fallback to Faster R-CNN or YOLOv8n

Full object extraction is **BLOCKED** until the RF-DETR checkpoint is resolved.

---

## What Remains

| Item | Status |
|---|---|
| RF-DETR checkpoint pinning | 🔴 Blocked — needs variant selection |
| Polars integration tests | ⏳ Requires `pip install polars` |
| R2 live canary test | ⏳ Optional; unit tests use mocks |
| Full Batch1 ingestion | ❌ Explicitly out of scope |
| Production pipeline run | ❌ Requires pinned RF-DETR + deployed workers |

---

## Architecture Alignment

The implementation follows the DATA-02 spec:

- ✅ R2 credentials from env vars only (never committed)
- ✅ Canonical object-key layout: raw/staging/corpus
- ✅ Parquet schemas with batch/collection partitioning
- ✅ Staging → validate → promote contract (atomic, idempotent)
- ✅ Extraction status control plane
- ✅ No vector concatenation across embedding providers
- ✅ Ingestion frozen (TransNetV2 @ 0.5 + periodic 12s/8s)
- ✅ ASR frozen (Silero VAD + ChunkFormer)
