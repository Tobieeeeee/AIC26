# Lộ trình xây hệ thống

## Phase 0 — Chuẩn hóa repo và dữ liệu

**Mục tiêu:** mọi thành viên chạy cùng một codebase.

- Chuyển code production ra khỏi notebook.
- Chốt schema `Video -> Shot -> Keyframe -> SearchDocument`.
- Tạo config chung và deterministic seeds.
- Dọn `.venv`, cache, generated keyframes khỏi Git.
- CI chạy unit test.

**Done khi:** clone mới -> setup -> test pass.

## Phase 1 — Video ingestion

- Probe video bằng ffprobe.
- So sánh PySceneDetect và TransNetV2 trên video thật.
- Freeze CANONICAL INGESTION V1: TransNetV2 `0.5` plus deterministic periodic
  midpoint + 12s/8s sampling. Keep PySceneDetect, midpoint-only, hybrid, and
  Meta/PE/BEiT selector experiments available only through explicit modes.
- Xuất manifest có `video_id`, `shot_id`, `frame_idx`, `timestamp`, `image_path`.

**Metric:** coverage thủ công của cảnh quan trọng + thời gian preprocess/video.

## Phase 2 — Enrichment

Chạy độc lập trên cùng keyframe set:

- OCR: EasyOCR/PaddleOCR hoặc engine mới.
- Object: YOLO.
- Embedding: CLIP baseline, sau đó SigLIP2/model tốt hơn nếu benchmark chứng minh lợi ích.

**Output:** một `SearchDocument` thống nhất cho mỗi keyframe.

## Phase 3 — Indexing

- BM25 index cho OCR/caption.
- Vector kNN index cho visual embedding.
- Object/timestamp/video fields cho filter.
- Script rebuild index từ processed manifest.

**Done khi:** index có thể xóa và dựng lại hoàn toàn từ artifacts offline.

## Phase 4 — Retrieval baselines

Tách 3 nhánh trước khi hybrid:

1. Lexical: OCR/caption BM25.
2. Semantic: text -> visual embedding -> kNN.
3. Structured: object counts + metadata filters.

Benchmark từng nhánh riêng để biết nó giải loại query nào.

## Phase 5 — Hybrid + reranking

- Giữ RRF làm baseline đối xứng.
- Weighted Hybrid với normalize theo từng query.
- Tune alpha trên validation query set, không tune trên test.
- Thử reranking top-30/top-50 nếu latency cho phép.

**Metric:** Recall@1, Recall@5/10, MRR, latency/query theo từng query type.

## Phase 6 — API

- `/health`
- `/search/text`
- `/search/semantic`
- `/search/object`
- `/search/hybrid`
- `/frame/{id}` hoặc static keyframe serving
- `/video/{id}?ts=...` nếu UI cần jump đúng timestamp.

Route chỉ validate/request/response; retrieval logic nằm ở service modules.

## Phase 7 — Competition UI

Tối ưu cho tốc độ thao tác chứ không chỉ đẹp:

- grid nhiều keyframe;
- mở rộng theo video/time neighborhood;
- filter object/OCR;
- keyboard shortcuts;
- query history;
- chọn result và submit nhanh.

## Phase 8 — Evaluation & ablation

Tạo query set gần format thi:

- visual semantic;
- OCR/text;
- object/count;
- mixed query;
- temporal sequence nếu cuộc thi có.

Mỗi lần đổi detector/OCR/embedding/fusion phải ghi:
- dataset/query version;
- config;
- Recall@K/MRR;
- latency;
- hardware;
- commit hash.

## Phase 9 — Freeze trước thi

- Tag release ổn định.
- Precompute toàn bộ corpus.
- Backup index + metadata.
- Có script one-command start.
- Test offline/restart/recovery.
- Không thay model vào phút cuối nếu chưa benchmark.
