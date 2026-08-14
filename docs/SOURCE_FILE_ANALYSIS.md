# Phân tích các file nguồn

## 1. `SkillPixel_Lab3_Homework_Completed.ipynb`

**Vai trò:** nền tảng image-text retrieval.

Các hạng mục chính:

1. Tải Flickr8k và lấy sample ảnh/caption.
2. OpenCLIP `ViT-B-32` tạo image embedding.
3. L2-normalize embedding để dot product tương đương cosine similarity.
4. Lưu embedding (`.npy`) và mapping (`.parquet`).
5. Encode text query rồi tính similarity với ảnh.
6. Đánh giá Recall@1/5/10 và MRR.
7. Mở rộng từ 100 lên tối đa 500 ảnh.
8. Dịch English -> Vietnamese bằng NLLB để đo cross-lingual retrieval.
9. So sánh OpenCLIP, SigLIP, SigLIP2; tùy chọn GVE/Qwen2.5-VL embedding.
10. Fusion OpenCLIP + SigLIP2 bằng vector concatenation.

**Ý nghĩa cho cuộc thi:** đây là nơi học/benchmark encoder. Không nên giữ logic này trong notebook khi triển khai; chuyển encoder vào `src/mvr/enrichment/embeddings/` và evaluation vào `src/mvr/evaluation/`.

**Kết quả baseline đã chạy:** sample 100 ảnh với OpenCLIP đạt Recall@1 89%, Recall@5 98%, Recall@10 100%, MRR 0.9277. Đây chỉ là lab self-retrieval nhỏ, không phải chất lượng dự kiến trên corpus thi thật.

---

## 2. `HCM_AIC_Lab4_semifinal.ipynb`

**Vai trò:** nâng retrieval từ tính ma trận thủ công thành search/index/evaluation bài bản.

### Part A — Text-to-Image
- Flickr8k.
- OpenCLIP.
- Batch image embedding.
- FAISS `IndexFlatIP` cho exact cosine search trên vector đã normalize.
- Hàm search ảnh.
- Recall@K + MRR + latency.

### Part B — Text-to-Text
- SciFact corpus/query/qrels.
- SentenceTransformer dense retrieval.
- FAISS index.
- CrossEncoder reranking.
- Đánh giá dense-only và dense+reranker.

**Ý nghĩa cho cuộc thi:** cung cấp ba ý tưởng production quan trọng:
1. vector index thay vì quét toàn bộ embedding;
2. qrels/ground truth để benchmark đúng;
3. two-stage retrieval: candidate generation -> reranking.

---

## 3. `shot-boundary-lab-student/student/`

**Vai trò:** biến video dài thành các đơn vị có thể index.

### `src/shotlab/probe.py`
Đọc metadata video như FPS, frame count, duration. Đây là bước bắt buộc để timestamp/frame mapping chính xác.

### `src/shotlab/detectors/pyscenedetect.py`
Shot boundary detection bằng PySceneDetect `ContentDetector`. Chuyển end frame từ exclusive sang inclusive.

### `src/shotlab/detectors/transnetv2_pytorch.py`
Shot boundary detection bằng TransNetV2. Model được load một lần và tái sử dụng cho nhiều video.

### `src/shotlab/keyframes.py`
Mỗi shot luôn chọn midpoint; shot dài lấy thêm frame theo chu kỳ. Sinh `keyframe_id`, `frame_idx`, `timestamp`, `image_path`.

### `src/shotlab/extract.py`
Trích đúng frame bằng FFmpeg, tối ưu theo hướng scan video một lần thay vì seek lại cho từng frame.

### `src/shotlab/models.py`
Data model `VideoInfo`, `Shot`, `Keyframe`; đây là hợp đồng dữ liệu quan trọng giữa các stage.

### `src/shotlab/metadata.py`
Xuất:
- `shots.json`: shot boundaries + schema.
- `keyframe_mapping.csv`: keyframe -> video/shot/frame/timestamp/path.
- benchmark JSON/CSV.

### `src/shotlab/pipeline.py`
Orchestrator: discover video -> probe -> chạy detector -> benchmark -> chọn keyframe -> extract -> ghi metadata.

### `output/benchmark*`
Kết quả sample 10 video: PySceneDetect nhanh hơn TransNetV2 trong benchmark hiện có, nhưng sample synthetic nhỏ nên không đủ để kết luận chất lượng boundary trên dữ liệu thi thật.

### `output_test/`, `.venv/`, `.pytest_cache/`
Artifacts/môi trường sinh ra khi chạy. Không phải source code và không nên commit Git.

**Ý nghĩa cho cuộc thi:** đây là tầng ingestion/offline đầu tiên. Nếu keyframe bỏ sót cảnh quan trọng, mọi OCR/CLIP/search phía sau đều không cứu được.

---

## 4. `HCM_AIC_Lab6.ipynb`

**Vai trò:** prototype end-to-end gần hệ thống thi nhất.

### Offline enrichment
1. Tạo 300 keyframe giả lập từ Flickr8k.
2. EasyOCR và PaddleOCR trên cùng frame, cùng confidence threshold và cùng metric `word_recall`.
3. YOLOv8n đếm object theo lớp COCO.
4. CLIP RN50 tạo visual embedding 1024 chiều.
5. Đóng gói mỗi frame thành document: `frame_id`, `video_id`, `ts`, `caption`, `ocr_text`, `objects`, `clip_vector`.

### Indexing
- Elasticsearch 8.x.
- Analyzer lowercase + asciifolding cho OCR tiếng Việt.
- Text fields cho BM25.
- Dynamic object counts cho structured filtering.
- Dense vector field cho cosine kNN.

### Retrieval
- BM25 trên OCR/caption.
- Flexible object search như `person:2,car:1`.
- CLIP kNN semantic search.
- RRF hybrid.
- Weighted Hybrid với min-max normalize độc lập theo từng query.

### Benchmark đã chạy
- BM25 mạnh với OCR-text nhưng gần như thất bại ở semantic query.
- CLIP mạnh với semantic query.
- RRF cân bằng hai nhánh nhưng benchmark hiện tại cho semantic thấp hơn pure CLIP.
- Weighted fusion cho phép điều khiển trade-off; trong thử nghiệm này `alpha=0.25` là lựa chọn cân bằng được notebook ghi nhận.

### API
FastAPI có:
- `/health`
- `/search/text`
- `/search/object`
- `/search/hybrid`
- static `/keyframes/...`

**Ý nghĩa cho cuộc thi:** Lab 6 chứng minh kiến trúc multi-modal hợp lý, nhưng production code phải tách notebook thành module offline/index/retrieval/API.

---

## 5. `lab1-fastapi-student/`

**Vai trò:** lớp serving/API cơ bản.

### `app/main.py`
Endpoint `/search`, validation query, limit, sort, response contract.

### `app/models.py`
Pydantic response model.

### `tests/test_app.py`
Contract tests: success, case-insensitive, empty result, validation, limit, sort.

### `.github/workflows/ci.yml`
GitHub Actions chạy test khi push/PR.

**Ý nghĩa cho cuộc thi:** cấu trúc API/test/CI này nên giữ, nhưng thay mock search bằng retrieval service thật.

---

# Kết luận mapping

| File/nhóm file | Vai trò trong hệ thống thi |
|---|---|
| Shot Boundary Lab | Video -> shot -> keyframe |
| Lab 3 | Học và benchmark image-text embedding |
| Lab 4 | Vector index + retrieval metrics + reranking |
| Lab 6 | OCR + object + CLIP + Elasticsearch + hybrid + API prototype |
| FastAPI Lab 1 | API contract + test + CI |

Không file nào một mình là “hệ thống thi”; giá trị thật nằm ở việc ghép các phần này thành một pipeline modular.
