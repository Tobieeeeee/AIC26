# Kiến trúc đề xuất

## Offline pipeline

```text
raw videos
  -> probe
  -> shot detection
  -> keyframe selection
  -> exact extraction
  -> OCR
  -> object detection
  -> visual embedding
  -> metadata/document builder
  -> indexes
```

### `src/mvr/ingestion/`
Chịu trách nhiệm mọi thứ từ video gốc đến keyframe + timestamp.

### `src/mvr/enrichment/`
Biến keyframe thành metadata có thể search:
- `ocr/`: chữ xuất hiện trong hình;
- `objects/`: vật thể và số lượng;
- `embeddings/`: vector semantic image-text.

### `src/mvr/indexing/`
Xây/làm mới các index. Không chứa business logic của query.

## Online pipeline

```text
query
  -> optional query understanding
  -> lexical branch
  -> semantic branch
  -> object/metadata filters
  -> fusion
  -> optional reranking
  -> result payload
  -> UI / submission
```

### `src/mvr/retrieval/`
Mỗi retrieval strategy là một module độc lập. Hybrid chỉ nhận ranked lists/scores, không tự chạy model/index.

### `src/mvr/api/`
HTTP contract. API gọi retrieval service; không đặt model loading/index creation trực tiếp trong route.

### `src/mvr/evaluation/`
Query set, qrels, Recall@K, MRR, latency, ablation. Mọi quyết định đổi model phải có benchmark ở đây.

## Folder dữ liệu

### `data/raw/`
Video gốc. Không commit.

### `data/interim/`
Metadata trung gian như probe results, shot boundaries. Có thể tái tạo.

### `data/processed/`
Manifest/document cuối trước khi index. Với corpus lớn nên để ngoài Git/object storage.

### `artifacts/keyframes/`
Ảnh keyframe sinh từ video. Không commit.

### `artifacts/embeddings/`
Vector `.npy/.parquet/...`. Không commit.

### `artifacts/indexes/`
FAISS/Elasticsearch snapshots/index artifacts. Không commit.

### `artifacts/metrics/`
Benchmark local. Chỉ commit report nhỏ/curated nếu cần; raw output lớn bỏ Git.

## Những phần còn thiếu so với một hệ thống thi hoàn chỉnh

1. UI để xem grid keyframe/video nhanh.
2. Query history và relevance feedback trong phiên thi.
3. Temporal/sequence retrieval cho truy vấn nhiều sự kiện liên tiếp.
4. Reranker chuyên cho top candidates.
5. Submission adapter theo format/server của cuộc thi.
6. Cache để query lặp lại trả nhanh.
7. Monitoring latency và lỗi model/index.
8. Benchmark trên dữ liệu thật của cuộc thi thay vì Flickr8k/synthetic sample.
