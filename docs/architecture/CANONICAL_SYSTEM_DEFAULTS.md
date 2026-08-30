# Canonical System Defaults — AIC26

Version: 1.3
Frozen by: DATA-02, TASK05-MILVUS, TASK-FIX-02, SELECTOR-HYBRID-MIGRATION
Config source of truth: `configs/production.yaml`

---

## Production Stack

```text
Canonical Production Defaults
------------------------------
Ingestion: TransNetV2 @ 0.5 + hybrid_temporal_content (KNN-novelty + edge-change, max gap 8s)
Objects:   RFDETRLarge / rf-detr-large-2026.pth @ 0.4
Visual:    BEiT-3, MetaCLIP-2, PE-Core-L
OCR:       DeepSolo + Vietnamese PARSeq + F-1/F/F+1
ASR:       Silero VAD + ChunkFormer CTC Large Vie
Semantic:  Milvus (3 vector fields: beit3, metaclip2, pe_vector)
Lexical:   Elasticsearch BM25
```

---

## Ingestion

| Parameter | Value |
|---|---|
| Shot detector | TransNetV2 |
| Detector threshold | 0.5 |
| Keyframe selector | hybrid_temporal_content |
| Candidate step | 1.0 seconds |
| Max temporal gap (forced keep) | 8.0 seconds |
| CLIP novelty threshold | 0.15 |
| CLIP rare threshold | 0.25 |
| Edge-change min score | 0.08 (robust median + 3·MAD) |
| Min event gap | 2.0 seconds |
| Midpoint mandatory | true |

Status: **Frozen**. Do not silently fall back to PySceneDetect.

Chosen over `semantic_dedup` on a 4-video/7-query real-data benchmark
(2026-08-31): Recall@1 0.583 vs 0.333, MRR 0.667 vs 0.417. `semantic_dedup`
scored better on raw information coverage and had lower internal redundancy,
but lost decisively on the retrieval task itself — see
`scripts/experiments/evaluate_keyframe_selectors.py`,
`evaluate_keyframe_coverage.py`, `evaluate_keyframe_redundancy.py` and their
`results/*_full4/summary*.json` outputs. `semantic_dedup` is kept available
(not legacy) pending a larger-scale re-evaluation, not deleted.

---

## Object Detection

| Parameter | Value |
|---|---|
| Provider | RF-DETR |
| Variant / class | Large / `RFDETRLarge` |
| Package | `rfdetr==1.9.3` (`3f7748b1783533bc2e62804ca5975fe9b9cfc041`) |
| Checkpoint | `rf-detr-large-2026.pth` |
| Checkpoint bytes | 135,954,129 |
| Checkpoint SHA-256 | `0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38` |
| Resolution | 704 x 704 |
| Vocabulary | MS COCO 2017, 80 labels, sparse IDs in 90 slots |
| Threshold / NMS | 0.4 / none (set prediction) |
| Runtime | local checkpoint, CUDA batch 1, downloads disabled |

Status: **Production runtime pinned and real-CUDA verified**. Full Batch1
object extraction remains a separately scheduled offline operation.

Faster R-CNN and YOLO are reference/legacy backends by explicit opt-in only.
OpenImages V4 describes organizer-provided historical object artifacts, not
the production detector vocabulary. The production RF-DETR vocabulary is COCO.

---

## Visual Semantic Embeddings

Three independent providers. No vector concatenation. No cross-provider raw
score addition.

| Provider | Model ID | Dimension | Normalized | Status |
|---|---|---:|---|---|
| BEiT-3 | `beit3` | 768 | L2 | Ready, pinned |
| MetaCLIP-2 | `metaclip2` | 512 | L2 | Ready, pinned |
| PE-Core-L | `perception_encoder` | 1024 | L2 | Ready, pinned |

Pinned provider specifications: `configs/embedding_providers.json`

---

## OCR

| Parameter | Value |
|---|---|
| Text detector | DeepSolo |
| Text recognizer | Vietnamese-finetuned PARSeq |
| Temporal context | F-1 / F / F+1 |
| Text similarity | Normalized Levenshtein ≥ 0.80 |
| Spatial similarity | BBox IoU ≥ 0.30 |

Status: **Frozen**. Checkpoint provenance pinned in `docker/ocr/model_sources.lock.json`.

Legacy EasyOCR and PaddleOCR are available by explicit opt-in only.

---

## ASR

| Parameter | Value |
|---|---|
| VAD | Silero |
| Recognizer | ChunkFormer CTC Large Vietnamese |
| Model | `khanhld/chunkformer-ctc-large-vie` |
| Revision | `14a68262fa5fc6d497cb107c18bc5a9308ef21ca` |
| Audio window | ±3.0 seconds around keyframe timestamp |
| Sample rate | 16,000 Hz |

Status: **Frozen**. Do not silently replace with Whisper or another ASR model.

---

## What Is NOT the Default

The following are available as legacy/experimental backends but are **not**
production defaults:

| Component | Legacy backend | Why not default |
|---|---|---|
| Keyframe selection | periodic (12s/8s), midpoint | Replaced by hybrid_temporal_content (see benchmark note above); still selectable via `--selector-mode periodic`/`midpoint` |
| Shot detection | PySceneDetect | Replaced by TransNetV2 |
| Object detection | Faster R-CNN / YOLOv8n | Replaced by RF-DETR |
| OCR | EasyOCR / PaddleOCR | Replaced by DeepSolo + PARSeq |
| Embeddings | Single OpenCLIP (ViT-B-32) | Replaced by 3-provider stack |
| ASR | — | No legacy; current system is canonical |
| Semantic DB | FAISS (faiss_reference) | Replaced by Milvus (TASK05-MILVUS) |

---

## Semantic Backend (TASK05-MILVUS)

Production semantic backend: **Milvus**

| Parameter | Value |
|---|---|
| Collection | `aic26_keyframes` |
| Vector fields | `beit3_vector` (768), `metaclip2_vector` (512), `pe_vector` (1024) |
| Metric | IP (inner product = cosine for L2-normalized vectors) |
| Index types | FLAT (exact/reference), HNSW (production ANN), IVF_FLAT (alternative) |
| Semantic fusion | Python RRF k=60 (no Milvus-native RRF in v1) |
| Recovery source | Cloudflare R2 canonical vector artifacts |

FAISS is retained as `faiss_reference` for:
- Parity testing
- Exact/reference comparisons
- Migration regression

FAISS must NOT be the production default.

---

## Config File Hierarchy

```text
configs/production.yaml      ← authoritative production defaults
configs/default.yaml          ← runtime defaults (aligned with production.yaml)
configs/embedding_providers.json  ← pinned provider specifications
```

CLI/runtime code should resolve defaults from these configs rather than
hardcoding values in Python modules.
