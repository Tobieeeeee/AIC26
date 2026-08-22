# AIC26 Shared Data Architecture — R2 + Parquet + Polars

## Status
PROPOSED_CANONICAL_DATA_ARCHITECTURE_V1

## Purpose

Tài liệu này mô tả cấu trúc dữ liệu chính thức đề xuất cho hệ thống AIC26 của đội.

Mục tiêu:

- tránh để dữ liệu extraction chỉ tồn tại trên máy local của một thành viên;
- hỗ trợ nhiều GPU worker / teammate cùng xử lý dữ liệu;
- hỗ trợ Batch1, Batch2, Batch3 theo cách incremental;
- tách rõ raw input, staging output và canonical corpus;
- dùng Cloudflare R2 làm shared persistent storage;
- dùng Parquet + Polars cho structured metadata;
- dùng FAISS cho semantic retrieval;
- dùng Elasticsearch cho lexical retrieval;
- giữ một global corpus duy nhất cho Online Search.

## 1. Nguyên tắc tổng quát

```text
RAW
= dữ liệu BTC cung cấp

STAGING
= dữ liệu worker/GPU vừa sinh, chưa canonical

CORPUS
= dữ liệu đã validate và được đội chấp nhận

PARQUET
= canonical structured metadata

POLARS
= query / transform / audit / batch processing metadata

FAISS
= semantic vector search

ELASTICSEARCH
= lexical/BM25 search

R2
= shared persistent storage

GITHUB
= code, config, schema, docs
  KHÔNG chứa dataset/model binaries
```

## 2. Data lifecycle tổng thể

```text
BTC VIDEO
   ↓
CLOUDFLARE R2 — RAW
   ↓
INGESTION
   ↓
STAGING
   ↓
VALIDATE
   ↓
CANONICAL CORPUS
   ↓
FEATURE EXTRACTION
   ↓
STAGING
   ↓
VALIDATE
   ↓
PARQUET / VECTOR ARTIFACTS
   ↓
POLARS / TASK05
   ↓
ELASTICSEARCH + FAISS
   ↓
RRF / FILTERING
   ↓
API
   ↓
UI
   ↓
video_id + frame_id
```

## 3. Flow tổng thể

```text
                    BTC / ORGANIZER
                          │
                    gửi VIDEO raw
                          │
                          ▼
                CLOUDFLARE R2 — RAW
             raw/batchN/videos/Lxx/...
                          │
                          ▼
══════════════════════ INGESTION ══════════════════════

                    TransNetV2
                 Shot Detection
                          │
                          ▼
                  Keyframe Selection
                          │
                          ▼
                       FFmpeg
                  Frame Extraction
                          │
                          ├───────────────┐
                          ▼               ▼
                      Keyframes       Ingest metadata
                                         │
                                         ├─ shots
                                         ├─ video metadata
                                         └─ keyframe mapping
                          │
                          ▼
                     VALIDATION
                          │
                          ▼
                  CANONICAL CORPUS
                          │
           ┌──────────────┼──────────────┬───────────────┐
           │              │              │               │
           ▼              ▼              ▼               ▼
          OCR            ASR          VISUAL           OBJECT
           │              │              │               │
       DeepSolo       Silero VAD       BEiT-3        Faster R-CNN
           ↓              ↓           MetaCLIP-2      OpenImages V4
       crop/rectify    ChunkFormer     PE-Core-L
           ↓
    Vietnamese PARSeq
           ↓
     F-1 / F / F+1
           │              │              │               │
           └──────────────┴──────────────┴───────────────┘
                          │
                          ▼
                       STAGING
                          │
                 schema / identity /
                 completeness check
                          │
                          ▼
                 CANONICAL FEATURES
                          │
            ┌─────────────┴──────────────┐
            │                            │
            ▼                            ▼
     Parquet metadata              Vector artifacts
   OCR / ASR / Objects       BEiT / MetaCLIP / PE
            │                            │
            ▼                            ▼
         POLARS                    FAISS builders
            │                            │
            └────────────┬───────────────┘
                         ▼
══════════════════════ TASK 05 ══════════════════════════

                     Task05A
              searchable documents
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
       Elasticsearch              FAISS
          BM25                  semantic
       OCR + ASR          ┌────────┬────────┐
                          │        │        │
                        BEiT3   MetaCLIP2   PE
                          │        │        │
                          └──── semantic RRF
                                   │
                    semantic + lexical RRF
                                   │
                     object eligibility/filter
                                   │
                                   ▼
                             FINAL RANKING
                                   │
                                   ▼
                              API / UI
```

## 4. Shared storage layout

```text
data/
│
├── raw/
│   ├── batch1/
│   │   └── videos/
│   │       ├── L21/
│   │       │   ├── L21_V001.mp4
│   │       │   ├── L21_V002.mp4
│   │       │   └── ...
│   │       ├── L22/
│   │       │   ├── L22_V001.mp4
│   │       │   ├── L22_V002.mp4
│   │       │   └── ...
│   │       └── ...
│   ├── batch2/
│   │   └── videos/
│   │       └── ...
│   └── batch3/
│       └── videos/
│           └── ...
│
├── staging/
│   ├── ingestion/
│   │   └── <run_id>/
│   ├── ocr/
│   │   └── <run_id>/
│   ├── asr/
│   │   └── <run_id>/
│   ├── objects/
│   │   └── <run_id>/
│   └── embeddings/
│       ├── beit3/
│       ├── metaclip2/
│       └── perception_encoder/
│
└── corpus/
    ├── dataset_manifest.json
    ├── ingest/
    │   ├── keyframe_mapping.csv
    │   ├── shots.parquet
    │   ├── video_metadata.parquet
    │   ├── ingest_manifest.json
    │   └── manifests/
    │       ├── batch1.json
    │       ├── batch2.json
    │       └── batch3.json
    ├── keyframes/
    │   ├── L21/
    │   │   ├── L21_V001/
    │   │   │   ├── L21_V001_S0001_K01.jpg
    │   │   │   ├── L21_V001_S0002_K01.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   ├── L22/
    │   │   └── ...
    │   └── ...
    ├── metadata/
    │   ├── keyframes/
    │   │   ├── batch=1/
    │   │   │   ├── collection=L21/
    │   │   │   │   └── part-000.parquet
    │   │   │   ├── collection=L22/
    │   │   │   │   └── part-000.parquet
    │   │   │   └── ...
    │   │   ├── batch=2/
    │   │   └── batch=3/
    │   ├── ocr/
    │   │   ├── batch=1/
    │   │   ├── batch=2/
    │   │   └── batch=3/
    │   ├── asr/
    │   │   ├── batch=1/
    │   │   ├── batch=2/
    │   │   └── batch=3/
    │   ├── objects/
    │   │   ├── batch=1/
    │   │   ├── batch=2/
    │   │   └── batch=3/
    │   └── extraction_status/
    │       ├── batch=1/
    │       ├── batch=2/
    │       └── batch=3/
    ├── features/
    │   └── embeddings/
    │       ├── beit3/
    │       │   ├── vectors.*
    │       │   ├── mapping.parquet
    │       │   └── manifest.json
    │       ├── metaclip2/
    │       │   ├── vectors.*
    │       │   ├── mapping.parquet
    │       │   └── manifest.json
    │       └── perception_encoder/
    │           ├── vectors.*
    │           ├── mapping.parquet
    │           └── manifest.json
    ├── indexes/
    │   ├── lexical/
    │   │   └── manifest.json
    │   └── semantic/
    │       ├── beit3/
    │       ├── metaclip2/
    │       └── perception_encoder/
    └── reports/
        ├── data_readiness/
        ├── extraction/
        └── indexing/
```

## 5. Ý nghĩa ba tầng

### `raw/`

`raw/` chỉ chứa byte mà BTC cung cấp.

Hiện tại:

```text
raw/batchN/videos/
```

Không đặt trong `raw/`:

```text
keyframes/
map_keyframes/
media_info/
clip-features/
objects/
ocr/
asr/
embeddings/
```

vì tất cả là output do pipeline của đội sinh.

`raw/` phải được coi là immutable.

### `staging/`

`staging/` là workspace output tạm của GPU/job.

Ví dụ:

```text
GPU OCR
↓
staging/ocr/run_001/
↓
validate
↓
promote
↓
corpus/metadata/ocr/
```

Mục tiêu là worker chết giữa chừng không làm corrupt canonical corpus.

### `corpus/`

`corpus/` là dataset canonical của hệ thống.

Chỉ chứa dữ liệu đã:

```text
extract
→ validate
→ identity check
→ schema check
→ promote
```

Task05 và Online Pipeline chỉ tin `corpus/`.

## 6. Ingestion

### Input

```text
raw/batchN/videos/
```

### Pipeline

```text
video
 ↓
FFprobe
 ↓
TransNetV2
 ↓
shot boundaries
 ↓
keyframe selection
 ↓
FFmpeg frame extraction
```

### Canonical output

```text
corpus/ingest/
├── keyframe_mapping.csv
├── shots.parquet
├── video_metadata.parquet
├── ingest_manifest.json
└── manifests/

corpus/keyframes/
└── Lxx/
    └── video_id/
        └── keyframe.jpg
```

## 7. `corpus/ingest/`

### `keyframe_mapping.csv`

Identity source of truth cho toàn corpus.

Conceptual schema:

```text
keyframe_id
video_id
collection_id
batch_id
shot_id
ordinal
frame_idx
timestamp
image_path
video_object_key
```

Ví dụ:

```csv
keyframe_id,video_id,collection_id,batch_id,shot_id,ordinal,frame_idx,timestamp,image_path,video_object_key
L21_V001_S0007_K01,L21_V001,L21,1,7,1,578,19.267,keyframes/L21/L21_V001/L21_V001_S0007_K01.jpg,raw/batch1/videos/L21/L21_V001.mp4
```

`frame_idx` là frame trong video gốc và là competition-facing `frame_id`.

### `shots.parquet`

Conceptual schema:

```text
video_id
collection_id
batch_id
shot_id
start_frame
end_frame
start_time
end_time
duration
```

### `video_metadata.parquet`

Conceptual schema:

```text
video_id
collection_id
batch_id
filename
duration
fps
frame_count
width
height
codec
has_audio
source_object_key
```

### `ingest_manifest.json`

Mô tả pipeline ingestion:

```text
shot detector
revision
keyframe strategy
ffmpeg/runtime information
schema version
mapping path
```

### `manifests/batchN.json`

Theo dõi trạng thái ingestion theo batch:

```text
batch id
source path
video count
shot count
keyframe count
status
```

## 8. Keyframe folder hierarchy

Binary keyframes chia theo:

```text
Lxx
 ↓
video_id
 ↓
keyframe files
```

Ví dụ:

```text
corpus/keyframes/
├── L21/
│   ├── L21_V001/
│   ├── ...
│   └── L21_V023/
└── L22/
    ├── L22_V001/
    ├── ...
    └── L22_V046/
```

Ví dụ path:

```text
corpus/keyframes/L21/L21_V001/L21_V001_S0007_K01.jpg
```

## 9. Parquet hierarchy

Structured metadata không chia tới từng video.

Partition theo:

```text
batch
 ↓
collection
 ↓
part-N.parquet
```

Ví dụ:

```text
corpus/metadata/keyframes/
└── batch=1/
    ├── collection=L21/
    │   └── part-000.parquet
    └── collection=L22/
        └── part-000.parquet
```

Tránh tạo quá nhiều file Parquet nhỏ.

## 10. OCR

Production OCR:

```text
Keyframe / neighboring frames
        ↓
DeepSolo
Total-Text ResNet-50
        ↓
crop / rectify
        ↓
Vietnamese-finetuned PARSeq
lynguyenminh
        ↓
F-1 / F / F+1 temporal consolidation
        ↓
OCR records
```

Worker output:

```text
staging/ocr/<run_id>/
```

Sau validation:

```text
corpus/metadata/ocr/*.parquet
```

JSON per-keyframe có thể giữ trong staging/debug; Parquet là canonical structured representation.

## 11. ASR

```text
video/audio
 ↓
Silero VAD
 ↓
ChunkFormer CTC Large Vietnamese
 ↓
per-keyframe transcript
 ↓
staging
 ↓
validation
 ↓
corpus/metadata/asr/*.parquet
```

Identity join qua `keyframe_id`.

## 12. Object Detection

Target architecture:

```text
keyframe
 ↓
Faster R-CNN
OpenImages V4
 ↓
bbox / label / score / count
 ↓
objects parquet
```

Canonical object table nên giữ:

```text
keyframe_id
video_id
frame_id
label
class_id
confidence
x1
y1
x2
y2
```

Lưu ý: implementation cần được align chắc chắn với target OpenImages V4 trước full extraction nếu runtime hiện tại vẫn dùng torchvision default weights.

## 13. Visual embeddings

Ba provider độc lập:

```text
Keyframes
 ├── BEiT-3
 ├── MetaCLIP-2
 └── PE-Core-L
```

Không concatenate vector và không cộng raw score giữa các model.

Storage:

```text
corpus/features/embeddings/
├── beit3/
├── metaclip2/
└── perception_encoder/
```

Mỗi provider nên có:

```text
vectors
mapping.parquet
manifest.json
model identity
source revision
dimension
normalization
```

## 14. Polars

Polars là metadata/data-access layer.

```text
Parquet
 ↓
Polars
 ↓
audit
filter
merge
job discovery
Task05 preparation
```

Polars không thay FAISS hoặc Elasticsearch.

Use cases:

```text
find missing OCR
find missing PE
merge Batch1 + Batch2 metadata
detect duplicate keyframe_id
filter collection L21
audit coverage
prepare Task05A documents
```

Không cần Polars Cloud cho v1.

Khuyến nghị:

```text
Polars OSS
+
Parquet
+
Cloudflare R2
```

## 15. Extraction status

Nên có:

```text
corpus/metadata/extraction_status/
```

Conceptual schema:

```text
keyframe_id
batch_id
collection_id
ocr_status
asr_status
object_status
beit3_status
metaclip2_status
pe_status
```

Ví dụ:

```text
KF001 | 1 | L21 | DONE | DONE | DONE | DONE | DONE | DONE
KF002 | 1 | L21 | DONE | TODO | DONE | DONE | TODO | DONE
```

GPU worker có thể query `PE == TODO` và chỉ xử lý keyframe còn thiếu.

## 16. Batch1 / Batch2 / Batch3

Batch chỉ tồn tại ở tầng raw/provenance/incremental processing.

Online Search không search từng batch riêng.

```text
Batch1 ─┐
Batch2 ─┼── GLOBAL CORPUS
Batch3 ─┘
            ↓
          Search
```

`batch_id` chỉ phục vụ:

```text
provenance
incremental extraction
audit
rollback
debug
filtering
```

## 17. Khi Batch2/Batch3 đến

Không rerun Batch1.

```text
raw/batch2/videos/
        ↓
ingest Batch2
        ↓
extract chỉ Batch2
        ↓
validate
        ↓
append/promote vào global corpus
        ↓
update keyframe_mapping
        ↓
update parquet partitions
        ↓
update Elasticsearch
        ↓
append/rebuild FAISS từ vectors đã có
```

Feature extraction GPU đắt tiền không chạy lại nếu artifact cũ vẫn hợp lệ.

FAISS rebuild không đồng nghĩa với re-extract embedding.

## 18. Task05

Structured metadata:

```text
keyframes.parquet
ocr.parquet
asr.parquet
objects.parquet
       ↓
Task05A
       ↓
canonical searchable documents
```

Lexical:

```text
OCR + ASR
 ↓
Elasticsearch / BM25
```

Semantic:

```text
BEiT3 vectors       → FAISS BEiT3
MetaCLIP2 vectors   → FAISS MetaCLIP2
PE vectors          → FAISS PE
```

Online flow:

```text
query
 │
 ├── semantic
 │     ├→ BEiT3
 │     ├→ MetaCLIP2
 │     └→ PE
 │          ↓
 │     semantic RRF
 │
 ├── lexical
 │        ↓
 │       BM25
 │
 └── object constraints
          ↓

semantic + lexical RRF
          ↓
object eligibility/filter
          ↓
dedup
          ↓
top-k
          ↓
video_id + frame_id
```

## 19. `dataset_manifest.json`

Nên giữ vì corpus có nhiều batch, nhưng không phải identity source of truth.

Ví dụ:

```json
{
  "dataset": "AIC26",
  "schema_version": 1,
  "batches": {
    "batch1": {
      "status": "ready"
    },
    "batch2": {
      "status": "not_received"
    },
    "batch3": {
      "status": "not_received"
    }
  }
}
```

Identity source of truth:

```text
corpus/ingest/keyframe_mapping.csv
```

Data-platform representation:

```text
corpus/metadata/keyframes/*.parquet
```

## 20. Indexes

`indexes/` là derived artifacts, rebuildable từ canonical metadata/features.

```text
corpus/indexes/
├── lexical/
└── semantic/
    ├── beit3/
    ├── metaclip2/
    └── perception_encoder/
```

Index manifest nên record:

```text
dataset version
provider
mapping checksum
vector checksum
build timestamp
index type
```

để phát hiện index stale khi Batch2/Batch3 được thêm.

## 21. Cloudflare R2 role

R2 là shared persistent storage cho cả đội.

```text
local PC
GPU thuê A
GPU thuê B
teammate machine
backend machine
```

đều đọc/ghi từ cùng một corpus.

Local/GPU disk chỉ là:

```text
cache
temporary working directory
model cache
download buffer
```

Canonical artifact phải được promote về R2.

## 22. Không duplicate dữ liệu

Không copy lại raw video vào corpus nếu không cần.

Ví dụ:

```text
raw/batch1/videos/L21/L21_V001.mp4
```

là source object.

Metadata chỉ lưu:

```text
video_object_key =
raw/batch1/videos/L21/L21_V001.mp4
```

Tránh duplicate binary artifact chỉ để thay đổi logical namespace.

## 23. Data-02 position in roadmap

Data-02 là task hạ tầng dữ liệu nằm giữa feature implementation và full extraction/indexing.

```text
Ingestion implementation
        ↓
Feature Extraction implementation
        ↓
Data-02
Shared Corpus Storage & Parquet Metadata
        ↓
Full Batch extraction
        ↓
Data Readiness
        ↓
Task05 indexing
        ↓
API / UI
```

Data-02 v1 nên có tối thiểu:

```text
Cloudflare R2 bucket/layout
R2 credentials via environment variables
upload/download/sync tooling
keyframes Parquet
extraction_status Parquet
Polars access layer
Batch partitioning
Data-01 compatibility
Task05A compatibility
```

Chưa cần:

```text
Polars Cloud
Kubernetes
Airflow
Milvus
Redis worker queue
distributed scheduler
```

## 24. Final architecture summary

```text
RAW
    BTC videos
      ↓
INGESTION
    TransNetV2 + FFmpeg
      ↓
CANONICAL KEYFRAMES + MAPPING
      ↓
FEATURE EXTRACTION
    OCR / ASR / Object / Embeddings
      ↓
STAGING
      ↓
VALIDATION
      ↓
CORPUS
    Parquet metadata + vector artifacts
      ↓
POLARS
    audit / transform / job discovery
      ↓
TASK05
    Elasticsearch + FAISS
      ↓
RRF / filtering
      ↓
API / UI
      ↓
video_id + frame_id
```

## 25. Canonical architecture decision

```text
Shared storage:
Cloudflare R2

Raw organization:
batch → collection(Lxx) → video

Keyframe organization:
collection(Lxx) → video → keyframe

Structured metadata:
Parquet partitioned by batch and collection

Metadata engine:
Polars OSS

Semantic retrieval:
FAISS, one namespace/index per visual provider

Lexical retrieval:
Elasticsearch / BM25

Canonical identity:
keyframe_id

Global competition output identity:
video_id + frame_id

Batch handling:
incremental offline processing,
single global online retrieval corpus
```

Tài liệu này được dùng làm proposed canonical data architecture cho Batch1 và là compatibility target khi Batch2, Batch3 được thêm.
