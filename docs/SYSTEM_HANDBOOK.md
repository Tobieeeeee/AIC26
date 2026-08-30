# Handbook hệ thống MVR / AIC 2026

> Tài liệu này giải thích **toàn bộ** hệ thống hiện tại: mỗi module làm gì, input/output
> ra sao, lý thuyết/kỹ thuật đứng đằng sau nó, và hạn chế thật đang tồn tại. Viết cho
> người đã biết Python/ML cơ bản nhưng chưa biết gì về repo này. Số liệu, tên hàm, tên
> file trong tài liệu này đều lấy trực tiếp từ code và các báo cáo đã verify thật trong
> `docs/`, không suy đoán.
>
> Repo đã lớn hơn rất nhiều so với baseline ban đầu — không chỉ còn KIS, mà đã có đủ
> 3 track thi (KIS, TRAKE, Q&A), 4 embedding provider, ASR, OCR thật, lexical BM25 thật.
> Tài liệu này phản ánh đúng trạng thái đó, không phải bản kiến trúc dự kiến ban đầu.

---

## 0. Cách đọc tài liệu này

- Nếu chỉ có 10 phút: đọc mục 1 (bức tranh tổng) + mục 6 (ba track thi) + mục 10 (hạn chế).
- Nếu cần code thật: mục 2 là bảng tra "muốn làm X thì đọc file nào".
- Mọi con số benchmark trong tài liệu này đều ghi rõ nguồn và cỡ mẫu — **đừng trích dẫn
  chúng như kết quả thi chính thức**, tất cả đều tự đo trên tập L21 nội bộ (29 video),
  ground truth tự soạn, chưa phải dữ liệu/đề thi thật của ban tổ chức.

---

## 1. Bức tranh tổng quan

### 1.1 Bài toán

AIC 2026 có (ít nhất) ba dạng đề dùng chung một kho video: **KIS** (tìm đúng 1 khung hình
từ mô tả), **TRAKE** (tìm đúng k khung hình theo thứ tự thời gian của k sự kiện liên
tiếp), **Q&A** (tìm đúng 1 khung hình + trả lời một câu hỏi về khung đó). Cả ba đều quy
về cùng một lõi: video → biểu diễn có thể tìm kiếm được → truy vấn → xếp hạng ứng viên →
map về đúng frame gốc trên video của ban tổ chức (không phải số thứ tự keyframe).

### 1.2 Nguyên tắc kiến trúc xuyên suốt (đọc trước khi đọc phần còn lại)

Những quy tắc này lặp lại ở khắp các module, nên nêu một lần ở đây:

- **Reuse-first**: mọi thứ ưu tiên tái dùng logic từ các lab gốc hoặc module đã có, ghi
  lại chính xác vào `docs/LAB_REUSE_MAP.md` khi hành vi bị đổi khác bản gốc.
- **Fail closed, không đoán**: khi không map được một định danh (frame gốc, provider,
  model...), hệ thống luôn **bỏ qua/raise lỗi rõ ràng** thay vì suy đoán ra một giá trị.
  Cụm từ "refusing to guess" xuất hiện lặp lại trong nhiều module khác nhau — đây là
  triết lý thiết kế, không phải trùng hợp.
- **Offline/online tách biệt**: xử lý nặng (OCR, ASR, embedding, index) chạy offline một
  lần; lúc truy vấn chỉ load index/model đã có sẵn (lazy-singleton, không load lại mỗi
  request) rồi tính.
- **`(video_id, keyframe_id)` luôn là khoá, không bao giờ dùng `keyframe_id` một mình**
  — xem hộp cảnh báo ở mục 3.9, đây là lỗi tốn nhiều công sửa nhất trong lịch sử repo.
- **Test bằng dữ liệu giả cho CI, verify bằng dữ liệu thật riêng (opt-in qua biến môi
  trường)** trước khi tin một con số.

### 1.3 Sơ đồ luồng lớn nhất

```text
video gốc (BTC hoặc tự quay)
        │
        ▼
ingestion: probe (ffprobe) → shot detection → chọn keyframe → trích frame chính xác
        │
        ▼
enrichment (chạy song song, độc lập nhau, mỗi thứ ghi ra 1 thư mục JSON riêng):
   OCR (chữ trong khung) · ASR (lời nói quanh khung) · Object detection ·
   Embedding ảnh (4 model độc lập: OpenCLIP, BEiT-3, MetaCLIP-2, Perception Encoder)
        │
        ▼
Task05A: GHÉP tất cả nhánh trên lại thành 1 "canonical document" / keyframe
 (đây là lớp keo nối toàn hệ thống — xem mục 3.9)
        │
        ▼
indexing: FAISS (1 index/provider) + Elasticsearch lexical (BM25 trên ocr_text/asr_text)
        │
        ▼
retrieval branches: semantic (FAISS) · lexical (BM25) · object filter · RRF fusion ·
 dịch truy vấn Việt→Anh (opt-in)
        │
        ▼
orchestration theo TỪNG TRACK THI (khác nhau, không dùng chung 1 pipeline):
   KIS (search_kis) · TRAKE (search_trake, thêm bước DP alignment) ·
   Q&A (search_qna, thêm bước LLM answer reasoning)
        │
        ▼
FastAPI (POST /search/kis · /search/trake · /search/qna · /search/semantic · ...)
        │
        ▼
submission export (CSV theo đúng format ban tổ chức, hoặc file debug nội bộ)
```

---

## 2. Bảng tra module → file (đúng theo code hiện tại)

| Nhóm | Thư mục | Vai trò |
|---|---|---|
| Ingestion | `src/mvr/ingestion/` | probe video, shot detection, chọn/trích keyframe, đọc dataset của BTC |
| Enrichment — OCR | `src/mvr/enrichment/ocr/` | EasyOCR/PaddleOCR adapter, OCR theo thời gian (`scene_text.py`), OCR chạy trong Docker (`docker_ocr.py`), kiểm định OCR output (`validator.py`) |
| Enrichment — ASR | `src/mvr/enrichment/asr/` | VAD (`vad.py`), giải mã audio (`audio.py`), ASR model (`chunkformer.py`), ghép pipeline (`pipeline.py`) |
| Enrichment — Objects | `src/mvr/enrichment/objects/` | YOLOv8 (`yolo.py`), Faster R-CNN (`faster_rcnn.py`), đọc file objects BTC cấp sẵn (`aic_objects.py`) |
| Enrichment — Embeddings | `src/mvr/enrichment/embeddings/` | OpenCLIP gốc (`openclip.py`) + 3 provider mới: BEiT-3, MetaCLIP-2, Perception Encoder, cộng `provider_specs.py` (khoá checkpoint), `query_encoders.py` (encode query đúng không gian từng provider), `artifacts.py`/`offline.py` (đóng gói/đọc artifact) |
| Enrichment — Q&A | `src/mvr/enrichment/qna_*.py` | context builder, answer reasoner (LLM), dịch truy vấn, batch xử lý qua Kaggle |
| Indexing | `src/mvr/indexing/` | FAISS (`faiss_index.py`, `semantic_builder.py`), Elasticsearch (`elasticsearch_index.py` — cũ/Lab6, `lexical_index.py` — mới/thật), **canonical document builder `task05a.py`**, `provider_indexes.py` (index riêng từng embedding provider), `filtering.py`/`normalization.py`/`validation.py` |
| Retrieval | `src/mvr/retrieval/` | `semantic.py`, `lexical.py`/`lexical_bm25.py`, `object_search.py`, `hybrid.py` (RRF), `semantic_rrf.py` (RRF nhiều provider), `provider_semantic.py`, `kis.py`, `trake.py`+`trake_events.py`+`trake_scoring.py`+`trake_align.py`, `qna.py`+`qna_fusion.py`, `frame_refinement.py` |
| Evaluation | `src/mvr/evaluation/` | `aic_kis.py`, `trake.py`, `aic_qna.py` (3 evaluator riêng theo luật từng track), `benchmark.py`, `trake_benchmark.py`, `metrics.py` (generic, không dùng cho 3 track thi) |
| API | `src/mvr/api/` | `app.py` (FastAPI, mọi route), `schemas.py` (Pydantic contract) |
| Submission | `src/mvr/submission/` | `kis_export.py`, `trake_export.py`, `qna_export.py` |
| Operations | `src/mvr/operations/full_corpus.py` | state machine chạy build toàn bộ corpus thật (có cổng chặn, xem mục 4.5) |
| CLI | `src/mvr/cli.py` | 7 lệnh: `ingest`, `build-index`, `search`, `build-aic-index`, `build-asr-features`, `benchmark-kis`, `enrich-ocr` |
| Scripts độc lập | `scripts/` | mọi thứ **không** nằm trong `mvr.cli` — build lexical index, build provider index, chạy Q&A pipeline, chạy TRAKE pipeline, các script kiểm chứng — xem mục 8 |

---

## 3. Offline pipeline — từng bước, kèm lý thuyết

### 3.1 Video probing

`ingestion/probe.py: probe_video()` gọi `ffprobe` lấy fps, số frame, thời lượng. Vấn đề
kỹ thuật: video có thể là **VFR** (variable frame rate — số frame/giây không cố định,
rất phổ biến với video tải từ mạng/điện thoại), nên `frame_count` đôi khi phải ước lượng
(`duration × fps`) nếu `ffprobe` không đếm được chính xác — đây là nguồn sai số tiềm ẩn
nếu pipeline sau tin tuyệt đối vào `frame_count`.

### 3.2 Shot boundary detection — lý thuyết

Một video là chuỗi khung hình liên tục; một **shot** là đoạn khung hình quay liên tục
không cắt cảnh. Việc tìm ranh giới shot dùng 1 trong 2 kỹ thuật:

- **PySceneDetect `ContentDetector`** (`shot_detection/pyscenedetect.py`, mặc định):
  thuật toán **content-based** kinh điển — chuyển mỗi khung sang không gian màu HSV, đo
  độ khác biệt trung bình giữa khung hiện tại và khung trước; nếu vượt `threshold` (mặc
  định 27.0) thì coi là 1 lần cắt cảnh. Không cần model, chạy CPU nhanh, nhưng nhạy với
  chuyển cảnh mượt (dissolve/fade) vì độ khác biệt tăng từ từ chứ không đột ngột.
- **TransNetV2** (`shot_detection/transnetv2.py`, tuỳ chọn): mạng CNN 3D được huấn luyện
  trực tiếp để dự đoán xác suất "đây là ranh giới shot" cho từng khung — bắt được cả cắt
  cứng lẫn chuyển cảnh mượt, nhưng cần GPU/model weight và chưa được verify runtime thật
  trong repo này (`README.md` ghi rõ "preserved as optional backend, not runtime-verified").

### 3.3 Chọn keyframe

`ingestion/keyframes.py: select_frame_indices()` — luôn lấy khung **giữa** mỗi shot (đại
diện ổn định nhất, tránh khung đầu/cuối dễ mờ do đang cắt). Với shot dài (mặc định ≥12
giây) lấy thêm khung mỗi 8 giây, vì một shot dài thực tế có thể chứa nhiều nội dung khác
nhau (camera đứng yên nhưng hành động thay đổi) mà 1 khung giữa không đại diện đủ.

### 3.4 Trích khung chính xác — kỹ thuật đáng chú ý

`ingestion/extract.py` **không dùng OpenCV** để đọc video mà dùng 2 tiến trình `ffmpeg`
nối pipe với nhau: 1 tiến trình decode toàn bộ video **đúng 1 lần** ra raw frame
(`rgb24`), Python đếm frame bay qua và chỉ forward đúng những frame cần vào tiến trình
`ffmpeg` thứ 2 để ghi JPEG. Lý do làm vậy thay vì seek từng frame: seek ngẫu nhiên trên
video nén (H.264/H.265) phải decode lại từ keyframe/GOP gần nhất mỗi lần seek, chậm hơn
nhiều so với 1 lần quét tuần tự nếu cần nhiều frame trong cùng video.

**Đánh đổi quan trọng cần nhớ**: kỹ thuật này rất hiệu quả khi trích **nhiều frame gần
đầu video**, nhưng cực kỳ chậm khi cần 1 frame **ở gần cuối** video (phải quét qua hết
phần trước). Đây chính xác là nguyên nhân gốc khiến tính năng "soi kỹ lại" của KIS chậm
205 giây/câu (mục 6.1) — nó tái dùng đúng hàm này cho 1 cửa sổ nhỏ nhưng nằm sâu trong
video. **Chưa có bản seek-based decode** để thay thế.

### 3.5 OCR — lý thuyết + 2 tầng trong repo

OCR gồm 2 bài toán con: **text detection** (khoanh vùng có chữ) rồi **text recognition**
(đọc ra ký tự). Repo có 2 tầng OCR:

- **Tầng cũ (Lab 6)**: `EasyOcrEngine`/`PaddleOcrEngine` — nhận 1 ảnh, trả về 1 chuỗi
  text đã lọc theo ngưỡng tin cậy 0.3. Đơn giản, dùng cho baseline KIS ban đầu.
- **Tầng mới, thật hơn**: `enrichment/ocr/scene_text.py` chạy OCR **theo shot**, không
  theo từng khung độc lập — vì cùng 1 dòng chữ (biển hiệu, phụ đề) thường xuất hiện
  nhiều khung liên tiếp trong 1 shot; module này gộp/khử trùng lặp theo thời gian
  (`text_similarity_threshold`, `bbox_iou_threshold`) để không sinh ra 5 bản ghi giống hệt
  nhau cho cùng 1 dòng chữ. Có bản chạy qua Docker (`docker_ocr.py`) để cô lập
  dependency. Bug thật đã tìm và sửa: OCR theo shot từng khoá `shot_boundaries` bằng
  `shot_id` trần (không kèm `video_id`) → shot số 1 của video sau ghi đè shot số 1 của
  video trước, khiến 872/873 video bị cắt sai khoảng và OCR ra rỗng không một cảnh báo
  (xem `docs/KEYFRAME_IDENTITY.md`).

### 3.6 ASR — lý thuyết + thiết kế

Repo **không transcribe nguyên video rồi cắt theo timestamp** — cách đó dễ nhưng tốn
gấp đôi (mỗi giây audio bị nhận dạng lại nhiều lần vì các cửa sổ chồng lấp), ASR ở đây
được thiết kế **theo từng keyframe**:

```text
với mỗi keyframe tại thời điểm t:
  cắt audio [t-3s, t+3s] (mono 16kHz)
  → Silero VAD: có tiếng nói trong đoạn này không?
  → nếu có: ChunkFormer transcribe đúng đoạn có tiếng nói
  → nếu không: text="" (không gọi ASR, tiết kiệm)
```

**VAD (Voice Activity Detection)** là bài toán phân loại nhị phân từng frame audio
"có/không có giọng nói" — Silero VAD dùng mạng nhẹ dự đoán xác suất này, lọc bỏ đoạn
im lặng/nhạc nền trước khi đưa vào ASR (tránh ASR "bịa" chữ từ nhiễu). **ChunkFormer**
là model ASR tiếng Việt dạng CTC (Connectionist Temporal Classification — cho phép model
dự đoán chuỗi ký tự dài hơn/ngắn hơn audio input mà không cần alignment thủ công), chạy
theo chunk để xử lý audio dài hiệu quả.

Chi phí đo thật: **~6.7 giây/ảnh trên CPU** (đây là lý do phần lớn benchmark Q&A ban đầu
phải chạy với `--no-ocr`/không ASR). Trên GPU tốc độ chưa đo chính thức nhưng theo thiết
kế nhanh hơn nhiều bậc — xem `docs/AIC_QNA_TEAM_TASKS.md` phần T0 (đo tốc độ trước khi
quyết định chạy toàn corpus 166.550 keyframe).

### 3.7 Object detection — 3 nguồn song song, cố tình không gộp

Có tới **3 nguồn nhãn đối tượng khác nhau** trong hệ thống, cố ý giữ tách biệt vì chúng
dùng **3 tập nhãn khác nhau** (gộp sẽ trộn lẫn nhãn không tương thích):

- **YOLOv8n** (`yolo.py`, từ Lab 6) — nhãn COCO (80 lớp), single-stage/anchor-free
  detector: 1 mạng dự đoán đồng thời vị trí + lớp cho mọi vùng trên ảnh trong 1 lần chạy
  (khác 2-stage detector phải đề xuất vùng rồi mới phân loại), nên nhanh, phù hợp
  real-time nhưng độ chính xác không bằng 2-stage với vật thể nhỏ.
- **Faster R-CNN** (`faster_rcnn.py`, `torchvision fasterrcnn_resnet50_fpn`, COCO-pretrained)
  — 2-stage detector: tầng 1 (**RPN — Region Proposal Network**) đề xuất các vùng khả
  nghi có vật thể, tầng 2 phân loại + tinh chỉnh bounding box cho từng vùng đó. Chính xác
  hơn YOLO ở vật thể nhỏ/chi tiết, đổi lại chậm hơn.
- **File objects do BTC cấp sẵn** (`aic_objects.py`) — Faster R-CNN pretrained trên
  **OpenImages V4** (không phải COCO, tập nhãn rộng hơn), đã chạy sẵn, chỉ cần đọc JSON
  (`detection_class_entities`, `detection_scores`, `detection_boxes`). **Chỉ phủ 59/873
  video (6.8%)** trong dữ liệu team đang có cục bộ — không phải nguồn phủ đầy đủ, OCR vẫn
  là bằng chứng chính cho Q&A trên phần lớn video.

### 3.8 Embedding ảnh/text — lý thuyết CLIP + 4 biến thể trong repo

**Lý thuyết chung (CLIP — Contrastive Language-Image Pretraining)**: huấn luyện đồng
thời 1 image encoder và 1 text encoder sao cho vector của 1 ảnh và vector của đúng câu
mô tả ảnh đó có **tích vô hướng (cosine similarity) cao**, còn với mọi cặp ảnh/câu không
khớp thì thấp — học trên hàng trăm triệu cặp (ảnh, caption) từ Internet bằng **contrastive
loss** (kéo cặp đúng lại gần, đẩy mọi cặp sai trong cùng batch ra xa). Kết quả: một câu
tiếng Anh và một khung hình có thể so sánh trực tiếp bằng phép nhân vector — đây chính là
nền tảng của toàn bộ semantic search trong hệ thống này.

Repo hiện có **4 model CLIP-họ độc lập**, mỗi model là 1 không gian vector *riêng*
(không thể trộn lẫn — vector của model A không so sánh được với vector của model B):

| Provider | Checkpoint khoá cứng | Chiều vector | Vai trò |
|---|---|---:|---|
| OpenCLIP (gốc) | `ViT-B-32` / `laion2b_s34b_b79k` | 512 | Index chính cho KIS/TRAKE (`build-aic-index`) |
| OpenCLIP (Lab 6, legacy) | `RN50` / `openai` | 1024 | Chỉ dùng cho nhánh Elasticsearch index A cũ, không phải baseline chính |
| BEiT-3 | `beit3_base_itc_patch16_224.pth` | 768 | Provider multi-embedding thứ 2 |
| MetaCLIP-2 | `facebook/metaclip-2-worldwide-b16` | 512 | Provider đa ngữ (worldwide), **CC-BY-NC-4.0** — cần xác nhận điều khoản phi thương mại trước khi dùng thi |
| Perception Encoder | `facebook/PE-Core-L14-336` | 1024 | Provider Meta, Apache-2.0, checkpoint ~2.68GB |

**Nguyên tắc khoá cứng** (`configs/embedding_providers.json` là nguồn sự thật duy nhất):
không được tự đổi checkpoint/revision/tiền xử lý mà không cập nhật hợp đồng này kèm bằng
chứng; mọi model đều `local_files_only` mặc định — nếu chưa tải sẵn thì lỗi rõ ràng chứ
không tự tải ngầm hay âm thầm dùng model khác thay thế. `QueryEncoderRegistry` đảm bảo
1 câu query được encode đúng không gian vector của provider cần tìm, không cho phép mã
hoá lẫn provider rồi tìm nhầm index.

**BEiT-3** dùng kiến trúc *Multiway Transformer* của Microsoft, huấn luyện bằng
masked-data modeling đa phương thức trước khi fine-tune cho retrieval (ITC — Image-Text
Contrastive). **MetaCLIP-2** là biến thể CLIP huấn luyện lại với dữ liệu chọn lọc theo
thuật toán metadata-curation công khai của Meta, bản "worldwide" hỗ trợ đa ngôn ngữ qua
tokenizer XLM-R — tiềm năng bỏ bớt bước dịch Việt→Anh nếu đo cho thấy lợi (**chưa đo**).
**Perception Encoder** là họ CLIP mới nhất của Meta (2025), tối ưu cho cả retrieval lẫn
các tác vụ thị giác khác.

**Trạng thái thật**: 3 provider mới (BEiT-3/MetaCLIP-2/Perception Encoder) đã qua đủ
"code + structural release gate" (kiểm tra pipeline đúng, artifact đúng schema), nhưng
**"operational/competition release gate" đang BLOCKED** — chưa build embedding cho toàn
bộ corpus thật, chỉ mới verify trên 5 keyframe mẫu mỗi provider. Không có tuyên bố nào
rằng 3 model mới cải thiện chất lượng tìm kiếm — vì chưa có ground truth đại diện để đo.

### 3.9 Canonical document (Task05A) — lớp keo nối toàn hệ thống

`indexing/task05a.py: build_index_documents()` đọc **đồng thời** mapping keyframe gốc +
4 thư mục artifact riêng biệt (OCR, ASR, objects, embeddings-mỗi-provider) rồi ghép thành
**1 file JSON/keyframe** tại `documents/<video_id>/<keyframe_id>.json`:

```json
{
  "schema_version": "1.0",
  "keyframe_id": "...", "video_id": "...", "shot_id": 0,
  "frame_idx": 1745, "timestamp": 58.2, "image_path": "...",
  "ocr":  {"texts": [...], "joined": "..."},
  "asr":  {"texts": [...], "joined": "..."},
  "objects": {"labels": [...], "counts": {...}},
  "clip": {"video_id": "...", "feature_file": "...", "row_idx": 12},
  "embeddings": {"beit3": {...}, "metaclip2": {...}, "perception_encoder": {...}},
  "indexability": {"lexical": true, "vector": true, "object": false, "embeddings": {...}}
}
```

Đây là bước **validate cực kỳ nghiêm ngặt**: một keyframe thiếu OCR không sao (mảng
rỗng), nhưng nếu 1 file artifact khai `video_id` khác với mapping gốc, hay 1 vector CLIP
không hữu hạn (`NaN`/`Inf`), hay không chuẩn hoá L2 đúng — toàn bộ bị từ chối và ghi vào
`filtering_report.json` kèm mẫu lỗi cụ thể (`invalid_schema`, `identity_mismatch`,
`missing_branch_artifact`, `duplicate_identity`...) chứ không âm thầm bỏ qua.

> ### ⚠️ Bài học tốn kém nhất repo: khoá phải là `(video_id, keyframe_id)`
>
> Keyframe do pipeline tự sinh có id kiểu `L21_V001_S0007_K01` — tự mang tên video bên
> trong nên luôn **duy nhất toàn corpus**. Keyframe do **ban tổ chức cấp sẵn** thì
> `keyframe_id` chỉ là số thứ tự trong video (`"034"`) — đo thật trên 166.550 keyframe
> BTC thì chỉ có **632 tên phân biệt**, `"034"` xuất hiện ở 842 video khác nhau. Khi đội
> chuyển sang dùng keyframe BTC, code cũ coi `keyframe_id` một mình là khoá duy nhất đã
> **âm thầm sai ở 6 chỗ khác nhau** cùng lúc: Elasticsearch `_id=keyframe_id` ghi đè
> 166.550 document xuống còn 632 (bộ kiểm tra cũng dùng chính khoá sai đó nên không bắt
> được), OCR anchor lấy nhầm frame của video khác, RRF multi-embedding gộp nhầm 2 frame
> khác nhau thành 1 (top-100 chỉ còn tối đa 632 dòng)... Bài học duy nhất cần nhớ khi
> viết code mới đụng tới keyframe: **luôn khoá bằng `(video_id, keyframe_id)`**, không
> bao giờ dùng `keyframe_id` một mình — kể cả khi "chắc chắn nó không trùng" (đã có 3
> test cũ mã hoá luôn giả định sai này thành assertion). Chi tiết đầy đủ:
> `docs/KEYFRAME_IDENTITY.md`.

---

## 4. Indexing

### 4.1 FAISS `IndexFlatIP` — lý thuyết

`indexing/faiss_index.py`. **Exact nearest-neighbor search** (không xấp xỉ như HNSW/IVF)
— với vector đã chuẩn hoá L2 (độ dài = 1), **tích vô hướng = cosine similarity**, nên
`IndexFlatIP` (Inner Product) tìm đúng top-K vector gần nhất theo cosine bằng cách nhân
ma trận trực tiếp với mọi vector trong kho, không index cây/đồ thị nào cả — chính xác
tuyệt đối, chi phí tuyến tính theo kích thước kho. Với quy mô hiện tại (7.800–166.550
vector) vẫn đủ nhanh trên CPU (đo thật KIS: 0.058s/câu — mục 6.1); sẽ cần đổi sang ANN
(HNSW/IVF-PQ) nếu index tăng lên hàng triệu vector và độ trễ trở thành vấn đề — **chưa
cần làm việc này ở quy mô hiện tại**.

Lớp `FaissIndex` tự kiểm tra nghiêm ngặt: vector phải hữu hạn, phải chuẩn hoá L2 (dung
sai `1e-4`), field bắt buộc trong metadata (`frame_id`, `video_id`, `timestamp`,
`image_path`), và khi load lại một index đã lưu, so khớp `schema_version`/`index_type`/
`embedding_dimension`/`normalization` với file manifest — không cho load nhầm index xây
bằng model khác.

### 4.2 Provider-specific indexes (Task05C)

`indexing/provider_indexes.py`. Mỗi embedding provider có **1 FAISS index riêng**, đặt
tên `<environment>__<dataset>__<model_id>` để không lẫn giữa các model/config. Chính
sách ghi có 3 loại: `create` (mặc định, an toàn — dừng nếu đã tồn tại), `append` (báo lỗi
với ID trùng), `rebuild` (duy nhất được phép ghi đè — luôn cần soát kỹ trước khi dùng).

### 4.3 Elasticsearch — hai index không tương thích nhau

Đây là điểm dễ nhầm nhất khi đọc code cũ:

| | Index A — `keyframes` (Lab 6 cũ) | Index B — Task05B (mới, thật) |
|---|---|---|
| Định nghĩa | `indexing/elasticsearch_index.py` | `indexing/lexical_index.py` |
| Trường text | `caption`, `ocr_text` | `ocr_text`, `asr_text` |
| Trường vector | `clip_vector` (dense_vector, cosine) | **không có** |
| `_id` | `frame_id` | `video_id/keyframe_id` |
| Có builder chạy production không | **Không — chỉ xuất hiện trong test** | Có (`scripts/indexing/build_lexical_index.py`) |

Nói cách khác: **nhánh "hybrid" cũ (`/search/hybrid`, dựa vào Index A) về lý thuyết vẫn
chạy được nhưng không có dữ liệu thật đứng sau nó** trong hệ thống hiện tại — Index A
chưa từng được build từ dữ liệu production. Lexical retrieval **thật** (dùng cho Q&A) đi
qua Index B. KIS/TRAKE hiện tại **chỉ chạy semantic-only** (FAISS), chưa nối với Index B.

**Lý thuyết BM25** (thuật toán xếp hạng của Elasticsearch mặc định): tính điểm 1 tài
liệu theo tần suất từ khoá xuất hiện trong tài liệu đó (TF), có bù trừ theo độ hiếm của
từ trong toàn kho (IDF) và độ dài tài liệu — khác hẳn semantic search (so vector), BM25
tìm theo **khớp từ chính xác** (sau khi qua analyzer), nên mạnh với truy vấn có tên
riêng/con số/chữ xuất hiện đúng y hệt trên màn hình (rất hợp với OCR text).

**Bug thật, đã tìm ra và có thiết kế sửa đã duyệt (chưa triển khai xong):** Index B ghi
`_id = f"{video_id}/{keyframe_id}"` (`lexical_index.py`) nhưng code đọc
(`lexical_bm25.py:106`) vẫn đòi `_id == keyframe_id` và raise
`RuntimeError("malformed search response: _id/keyframe_id mismatch")` nếu không khớp —
**mọi hit từ 1 Index B build đúng đều raise lỗi này**. Sở dĩ không ai bắt được bằng test:
`tests/unit/indexing/test_task05b.py:153` (phía ghi) dùng `_id: "L21_V010/frame-a"`,
còn `tests/unit/retrieval/test_task05b_lexical.py:19` (phía đọc) tự bịa `_id: keyframe_id`
— hai bên test cùng 1 hợp đồng bằng 2 định dạng khác nhau, cả hai đều pass, nên không
test nào phát hiện ra 2 phía không khớp nhau. Ở tầng `QnaFusedRetriever`, exception này
bị nuốt thành `logger.warning("lexical branch unavailable")`, nên nhánh BM25 âm thầm trả
rỗng vĩnh viễn mà không có cảnh báo nào nổi lên cho người vận hành thấy. Lỗi này do PR #8
gây ra (sửa phía ghi, quên truy theo phía đọc).

Thiết kế sửa đầy đủ nằm ở `docs/superpowers/specs/2026-08-21-tkis-query-branch-split-design.md`,
chia 2 đợt:
- **Đợt 1** (không phụ thuộc gì, làm được ngay): sửa hợp đồng `_id` ở trên + thêm
  analyzer tiếng Việt `vi_folded` (standard + lowercase + asciifolding, copy từ Index A)
  vào Index B, nâng `TASK05B_SCHEMA_VERSION` lên `"1.1"` để index cũ không bị nhầm là
  tương thích (cần reindex lại).
- **Đợt 2** (phải chờ nhánh Q&A merge vào `main` vì đụng `qna_fusion.py`): tách hẳn 1
  câu truy vấn tiếng Việt riêng cho BM25 và 1 câu tiếng Anh riêng (đã dịch) cho dense
  search — hiện tại `search_hybrid()` nhận đúng 1 chuỗi và đưa nguyên cho cả 2 nhánh, nên
  dù sửa xong `_id`, nhánh BM25 vẫn đang bị "hỏi" bằng tiếng Anh trong khi index của nó là
  tiếng Việt. Thêm tham số tuỳ chọn `lexical_query` (mặc định `None` = giữ hành vi cũ) để
  không phá vỡ chữ ký hàm hiện có.

Ngoài phạm vi thiết kế này (cố ý chưa làm): hợp nhất Index A/B thành 1 schema, đổi
OpenCLIP RN50 sang encoder đa ngữ để có thể bỏ bước dịch hoàn toàn, và 1 lỗi định danh
cùng lớp còn sống ở `ElasticsearchMultimodalService` (fuse trên `frame_id` đơn) — chưa
gây hại vì đường ống đó (Index A) chưa có dữ liệu thật.

**Bộ đo A/B so sánh trước/sau khi tách truy vấn cũng đang hỏng**: 3 file của bộ ablation
(`scripts/verification/query_retrieval_ablation/metrics.py`,
`scripts/verification/run_tkis_query_retrieval_ablation.py`,
`tests/unit/test_tkis_query_retrieval_ablation.py`) là **0 byte trong mọi commit** từng
đụng tới chúng — `runner.py` import 1 hàm từ module rỗng nên `ImportError` ngay khi chạy.
Nghĩa là **hiện chưa có bất kỳ con số nào chứng minh việc tách truy vấn thực sự cải thiện
kết quả** — việc sửa `_id` được làm vì nó đúng và rẻ, không phải vì đã đo thấy lợi.

### 4.4 Semantic RRF nhiều provider — lý thuyết fusion

`retrieval/semantic_rrf.py`. **Reciprocal Rank Fusion**: với mỗi danh sách xếp hạng độc
lập (ở đây là kết quả tìm kiếm riêng của từng embedding provider), 1 tài liệu ở hạng `r`
được cộng điểm `1 / (rrf_k + r)`; cộng dồn điểm từ mọi danh sách rồi xếp hạng lại theo
tổng điểm đó. Điểm mấu chốt về lý thuyết: RRF **chỉ dùng thứ hạng (rank), không bao giờ
dùng điểm tương đồng thô** — vì điểm cosine của BEiT-3 và của MetaCLIP-2 không cùng
thang đo (một model có thể "hào phóng" điểm hơn model khác một cách hệ thống), trộn điểm
thô sẽ để 1 model lấn át các model khác một cách sai lệch. Đây cũng đúng là nguyên tắc mà
`retrieval/hybrid.py: rrf_fuse()` (dùng cho KIS/TRAKE) đã áp dụng từ đầu.

### 4.5 Vận hành build corpus thật — cổng chặn có chủ đích

`operations/full_corpus.py` là 1 state machine bắt buộc đi đúng thứ tự: mapping toàn
corpus → trích embedding từng provider → audit định danh (phải ra đúng `valid=7800`,
mọi số thiếu/thừa/trùng đều = 0) mới cho sang provider kế → build index → smoke online →
benchmark. Có `--dry-run` (chỉ đọc, không tải/không build gì) và cờ
`--approved-expensive-run` bắt buộc mới cho chạy thật — **cố ý** làm khó để không ai vô
tình chạy nhầm 1 job tốn nhiều giờ GPU/hàng chục GB đĩa. Tại thời điểm viết tài liệu này,
lệnh thật **chưa từng chạy** — mọi placeholder trong `configs/full_corpus_build.json`
vẫn cần điền trước khi được duyệt chạy.

---

## 5. Retrieval branches (khối build-block, dùng lại ở cả 3 track)

| Nhánh | File | Input → Output | Ghi chú |
|---|---|---|---|
| Semantic (1 provider) | `retrieval/semantic.py` | text query → `SemanticRetriever.search()` → `list[SearchResult]` | Encode câu query 1 lần, tìm trong FAISS đã load sẵn |
| Semantic (nhiều provider độc lập) | `retrieval/provider_semantic.py` | text query → dict `{provider_id: ranked list}` | Không tự fuse — để `semantic_rrf` làm |
| Lexical BM25 | `retrieval/lexical_bm25.py`, `lexical.py` | text query → ES BM25 response | 2 phiên bản cho 2 index A/B khác nhau (mục 4.3) |
| Object filter | `retrieval/object_search.py` | `"person:2,car:1"` → ES range filter | Cú pháp `class:count_tối_thiểu`, validate theo 80 lớp COCO |
| Hybrid RRF (Lab 6 style) | `retrieval/hybrid.py` | 2+ ranked list → 1 list đã fuse | Hàm thuần, không sở hữu model/index — dùng cả cho KIS/TRAKE/Q&A |
| Frame refinement | `retrieval/frame_refinement.py` | 1 candidate thô → "soi kỹ" quanh nó | Xem cảnh báo hiệu năng ở mục 3.4 và 6.1 |
| Dịch truy vấn | `enrichment/qna_translation.py` | câu tiếng Việt → tiếng Anh | `Helsinki-NLP/opus-mt-vi-en`, opt-in, fail-open (lỗi thì dùng câu gốc, không 503) |

---

## 6. Ba track thi — kiến trúc, input/output, số liệu thật, hạn chế

### 6.1 KIS (Known-Item Search)

**Bài toán**: 1 câu mô tả → tối đa 100 cặp `(video_id, frame_id)` xếp hạng, `frame_id`
phải là frame thật trên video gốc.

**Luật chấm**: `R@k` (k∈{1,5,20,50,100}) = có ít nhất 1 đáp án đúng trong top-k hay
không (0/1); Final Score = trung bình cộng 5 giá trị đó. Đúng khi
`video_id` khớp **và** `frame_id` nằm trong khoảng `[start_frame, end_frame]` (không
phải khớp tuyệt đối 1 số). Cài trong `evaluation/aic_kis.py`.

**Orchestration** (`retrieval/kis.py: search_kis()`):
1. Overfetch `max(top_k*3, 30)` kết quả thô từ FAISS (hoặc hybrid nếu có ES).
2. Map từng keyframe → frame gốc, ưu tiên resolver của chính pipeline, sau đó registry
   dữ liệu BTC (`map_keyframes/<video>.csv`, cột `frame_idx` — **không bao giờ dùng cột
   `n`**). Không map được → bỏ, không đoán.
3. Loại trùng theo `(video_id, frame_id)`, dừng khi đủ `top_k`.

**Exact-frame refinement** (tuỳ chọn, KHÔNG chạy trong API mặc định): giải mã 1 cửa sổ
nhỏ quanh candidate thô, chấm lại bằng CLIP, chọn frame khớp nhất. Benchmark thật (6 câu,
L21, 7800 keyframe): **độ chính xác y hệt bản không refine (Final Score 0.633 cả hai)**,
nhưng **chậm hơn ~3500 lần** (0.058s → 205s/câu) — nguyên nhân gốc là bộ giải mã
tuần-tự-từ-frame-0 (mục 3.4), không phải do bản chất kỹ thuật soi lại chậm. → quyết
định: tắt mặc định, chỉ dùng qua CLI `benchmark-kis`, chưa nối vào API.

**Endpoint**: `POST /search/kis` — luôn đăng ký (chỉ cần FAISS, không bắt buộc ES).

**Hạn chế đã biết**: index chỉ phủ L21 (29 video/7800 keyframe, không phải toàn bộ
corpus); ground truth benchmark tự soạn nội bộ (6 câu, người viết câu biết trước đáp án
— không phải đo khách quan); nhánh hybrid/OCR chưa benchmark trên dữ liệu thật; nhánh
lexical hiện **chưa từng được bật khi đo KIS** — script đo hiện tại
(`scripts/verification/run_real_data_kis_smoke.py`) truyền thẳng `multimodal_service=None`,
nên mọi con số KIS trong tài liệu này là **semantic-only**, chưa phản ánh khi có BM25.

### 6.2 TRAKE (Temporal Retrieval & Alignment of Key Events)

**Bài toán**: `TRAKE = KIS × k sự kiện, cùng 1 video, đúng thứ tự`. Input: k (≥2) mô tả
sự kiện + cụm từ nối giữa chúng (`"rồi"`, `"ngay sau đó"`, `"một lúc sau"`...). Output:
tối đa 100 chuỗi `(video_id, [frame_1, ..., frame_k])` xếp hạng.

**Luật chấm** (`evaluation/trake.py`): khác KIS — cho **điểm từng phần**: đúng video thì
điểm = tỷ lệ số mốc khớp trong dung sai (mặc định 1 khung) / tổng số sự kiện; sai video =
0 tuyệt đối. R@k và Final Score tính giống hệt cấu trúc KIS (trung bình 5 mốc cắt).

**Orchestration, 3 bước** (`retrieval/trake.py`, `trake_scoring.py`, `trake_align.py`):
1. **Shortlist**: gọi `search_kis()` **y nguyên, k lần** — mỗi sự kiện là 1 lượt KIS con,
   gộp video được rút gọn (không re-encode).
2. **Dense scoring**: với mỗi video lọt shortlist, dựng ma trận điểm
   `(k sự kiện) × (số frame trong video)` bằng cách **đọc thẳng vector đã có sẵn trong
   FAISS** (`IndexFlatIP.reconstruct()`) — không encode lại ảnh.
3. **Alignment**: chọn 1 trong 3 thuật toán (port nguyên từ `trake_align.py` của lab, đã
   verify khớp giá trị self-check gốc) để chọn chuỗi frame tăng dần thời gian tốt nhất
   trong video đó — xem bảng dưới.

| Thuật toán | Ý tưởng | Tham số đã tune từ đo thật |
|---|---|---|
| **ATLAS** (mặc định) | DP cứng, mọi sự kiện phải khớp 1 frame, phạt 2 phía theo `[min,max]` gap | `alpha=0.01, beta=0.01, min_sim=0.0, top_n=5` |
| Lucifer | DP mềm, cho phép skip sự kiện không tự tin (nhân phạt λ=0.8) | `min_sim=0.20` — dưới ngưỡng này cơ chế skip không bao giờ kích hoạt (tie với ATLAS) |
| U-CESE | Cửa sổ trượt 2 con trỏ, nhanh, **không ép thứ tự thời gian** | `topk=10` — mặc định gốc của lab (300) làm mọi sự kiện collapse về 1 frame duy nhất |

**Số liệu thật** (6 câu, L21, real FAISS/CLIP CPU): ATLAS = Lucifer = **Final 0.639**,
latency 0.372s; U-CESE = 0.344 (không ép thứ tự nên thua rõ). ATLAS được chọn làm mặc
định vì cùng độ chính xác nhưng nhanh hơn và không có "vách" tham số nguy hiểm như
Lucifer (`min_sim` 0.20→0.22 đã tụt điểm còn 0.530). Lucifer chỉ thắng khi có 1 sự kiện
thực sự **không thể tìm thấy** trong video — đã verify riêng (không tính vào benchmark
chính) chứ 6 câu benchmark thật không tạo ra tình huống đó.

**Never-blank fallback** (`submission/trake_export.py`): trước khi nộp, nếu Lucifer để
lại 1 mốc `None`, hệ thống nội suy tuyến tính (nếu có 2 mốc thật 2 bên) hoặc lấy trung
điểm khoảng gap khai báo (nếu ở đầu/cuối chuỗi) — không bao giờ để trống, vì để trống
chắc chắn mất điểm còn đoán có cơ hội đúng.

**Endpoint**: `POST /search/trake` — luôn đăng ký, trả **chuỗi thô** (không áp
never-blank fallback ở tầng API — fallback là việc của bước xuất bài nộp).

**Dịch truy vấn**: có (`MVR_TRAKE_TRANSLATE_QUERIES=1`, tắt mặc định). Đo thật trên 6 câu
tiếng Việt tự nhiên: không dịch = 0.000 mọi câu; có dịch = 0.547 trung bình (so với 0.639
nếu viết tay bằng tiếng Anh) — dịch máy khôi phục phần lớn nhưng không toàn bộ khoảng
cách, chủ yếu vì model dịch nghĩa đen tên riêng.

**Hạn chế đã biết**: chưa có lệnh trong `mvr.cli` (chỉ có script rời
`scripts/run_trake_pipeline.py` và API); nhánh hybrid/ES chưa dùng (dense scoring cần
đọc thẳng vector FAISS, ES không có cơ chế tương đương); index vẫn chỉ L21; đóng gói file
nộp bài chính thức (zip theo đúng cấu trúc BTC) là thao tác thủ công, chưa tự động hoá.

### 6.3 Q&A

**Bài toán**: 1 mô tả sự kiện + 1 câu hỏi → tối đa 100 bộ ba
`(video_id, frame_id, answer)` xếp hạng. `answer` có thể tiếng Việt hoặc Anh, chấm theo
khớp **ngữ nghĩa** (không phải chuỗi ký tự tuyệt đối — ví dụ đề bài BTC chấp nhận cả
`"5"` lẫn `"Năm"`).

**Đây là track phức tạp nhất và cũng đang dở dang nhất**, vì đòi hỏi 1 bước hoàn toàn mới
so với KIS/TRAKE: **suy luận trả lời** từ ngữ cảnh (không chỉ tìm đúng khung). Kiến trúc:

```text
search_qna(event_description, question)
   → tái dùng search_kis()-shape để lấy candidate frame gốc  (không gọi search_kis trực
     tiếp vì QnaCandidate cần thêm field answer, nhưng dùng lại đúng resolver)
   → (tuỳ chọn) mở rộng theo thời gian quanh video đã tìm được — xem bên dưới
   → build_frame_context(): gom OCR + ASR + object-count cho từng candidate
   → AnswerReasoner (LLM local, Qwen2.5) sinh câu trả lời + gắn nhãn nguồn:
        context-grounded / knowledge-grounded / mixed
```

**Phát hiện quan trọng nhất qua đo thật** (3 câu hỏi có ground-truth người xác nhận):
retrieval ở **cấp video** đã hoạt động khá tốt (1 câu đúng video ở hạng #1–2), nhưng thất
bại ở **cấp frame** — đáp án đúng cách candidate gần nhất 4 keyframe (~10 giây). Đây là
lý do 2 cải tiến sau ra đời:

- **Mở rộng theo thời gian (temporal expansion)**: nếu 1 video được xếp hạng cao, tự
  động "đoán" thêm các keyframe lân cận (bán kính cấu hình được) như ứng viên phụ, mỗi
  ứng viên phụ được trả lời lại độc lập (không copy câu trả lời của candidate gốc). Biến
  4-keyframe-miss ở trên thành 1 hit thật ở hạng #18. Chi phí: 1 lần mở rộng "ăn" nhiều
  slot trong ngân sách 100 câu trả lời (đo thật: cấu hình sâu 13 video × nhiều frame/video
  thắng cấu hình rộng 73 video × ít frame/video, vì độ chính xác tập trung ở đúng video
  chứ không rải đều).
- **Lexical fusion (BM25 + semantic)**: 2/3 câu hỏi ground-truth thất bại **hoàn toàn vì
  retrieval** (0/100 candidate thuộc đúng video) — 1 câu cần ASR (nội dung *được đọc*,
  không phải chữ hiện trên màn hình), 1 câu cần OCR/BM25 (tên trên màn hình). Mở rộng
  thời gian không cứu được vì nó chỉ mở rộng quanh video **đã tìm thấy** — nếu retrieval
  gốc trật hoàn toàn thì không có gì để mở rộng.

**Số liệu thật đầu tiên có nghĩa** (chuỗi đủ real: retrieval → mở rộng → ảnh → EasyOCR →
Qwen2.5-7B trên Kaggle → chấm điểm): **Mean Final Score 0.2000** (n=3, 1/3 câu đúng hoàn
toàn) — từ 0.000 trước đó. Không phải kết quả tốt, nhưng là bằng chứng đầu tiên toàn
chuỗi hoạt động khi được nối đúng.

**Vận hành đặc thù — offload GPU qua Kaggle**: máy phát triển Q&A chỉ có 3.8GB RAM,
không GPU (model 1.5B không load nổi, model 0.5B thì lặp lại câu hỏi thay vì trả lời).
Giải pháp: xuất `prompts.json` (chỉ chứa câu hỏi + text OCR/object count, **không** ảnh/
video) → chạy Qwen2.5-7B trên Kaggle T4×2 miễn phí → nhập kết quả trả lời về máy local để
xuất bài nộp. Quy trình đầy đủ: `docs/QNA_KAGGLE_RUNBOOK.md`.

**Endpoint**: `POST /search/qna` — luôn đăng ký (503 nếu chưa cấu hình `MVR_QNA_MODEL`).

**Hạn chế đã biết (nhiều, ghi trung thực)**:
- n=3 là **trần vĩnh viễn** cho ground-truth đáng tin — mọi tham số (`temporal_radius=4`,
  `neighbour_cost=0.5`) hiện tune trên n=1 hiệu quả, không phải giá trị tối ưu đã kiểm
  chứng. Có xây thêm 1 bộ **synthetic set** (chọn sẵn frame → tự sinh câu hỏi từ OCR của
  chính frame đó) để tăng n lên ~14 case đo được, nhưng bộ đó chỉ đo được khả năng "trích
  câu trả lời từ ngữ cảnh cho sẵn", **không đo được chất lượng retrieval**.
- Index KIS/TRAKE (ViT-B-32/laion2b, 29 video) và index Q&A
  (`qna_organizer_all`, ViT-B-32-quickgelu/openai, 850 video, dựng từ CLIP feature có sẵn
  của BTC) là **2 không gian vector và 2 phạm vi corpus khác nhau**, chưa hợp nhất — đây
  là quyết định ở cấp team, không phải lỗi.
- Object context gần như vô dụng ở quy mô hiện tại (chỉ phủ 6.8% video); OCR là bằng
  chứng chủ lực thật sự.
- Lexical fusion đã có code (`QnaFusedRetriever`, 14 test) nhưng **chưa đo được** vì phụ
  thuộc Elasticsearch chạy thật + bug `_id` ở mục 4.3 chưa sửa xong.
- Trả lời bị chặn bởi tài nguyên máy (không phải chặn bởi thiết kế pipeline) — xem
  `docs/AIC_QNA_TEAM_TASKS.md` để rõ việc gì cần máy có GPU/Docker làm hộ.

---

## 7. API — toàn bộ route hiện có

`src/mvr/api/app.py: create_app()`. Quy ước lỗi xuyên suốt: `ValueError` → HTTP 400,
`RuntimeError` (thiếu cấu hình/model chưa sẵn sàng) → HTTP 503 — không bao giờ để lộ
500 thô cho lỗi đã biết trước.

| Route | Luôn đăng ký? | Input chính | Ghi chú |
|---|---|---|---|
| `GET /health` | có | — | Báo trạng thái semantic index + Elasticsearch, không coi phụ thuộc thiếu là "khoẻ" |
| `POST /search/semantic` | có | `query`, `top_k≤100` | FAISS thuần, 503 nếu chưa cấu hình `MVR_SEMANTIC_INDEX` |
| `POST /search/kis` | có | `query`, `top_k≤100` | Chỉ cần FAISS |
| `POST /search/trake` | có | `events[≥2]`, `gap_phrases`, `top_k≤100`, `alignment_strategy?` | Trả chuỗi thô, chưa never-blank |
| `POST /search/qna` | có | `event_description`, `question`, `top_k≤100` | 503 nếu chưa cấu hình `MVR_QNA_MODEL` |
| `POST /search/text` | chỉ khi có ES | `query`, `person_min`, `car_min` | BM25 trên Index A (Lab 6) |
| `POST /search/object` | chỉ khi có ES | `want="person:2,car:1"` | |
| `POST /search/hybrid` | chỉ khi có ES | `query`, `rrf_k` | RRF semantic(Index A)+lexical |

Biến môi trường điều khiển: `MVR_SEMANTIC_INDEX`, `MVR_DEVICE`, `MVR_ES_URL`,
`MVR_ES_INDEX`, `MVR_KEYFRAME_DIR`, `MVR_KEYFRAME_MAPPING_CSV`, `MVR_AIC_DATASET_ROOT`,
`MVR_QNA_MODEL`, `MVR_QNA_ALLOW_MODEL_DOWNLOAD`, `MVR_TRAKE_TRANSLATE_QUERIES`,
`MVR_TRAKE_ALLOW_MODEL_DOWNLOAD`. **Không có `configs/default.yaml` nào được đọc bởi
code** — mọi default đều là hằng số Python tại nơi dùng (ghi rõ trong comment của chính
`app.py`); đây là khoảng trống về config tập trung đã tồn tại từ bản đầu, chưa được xử lý.

---

## 8. CLI và scripts — bảng tra lệnh

`mvr.cli` (7 lệnh, dùng `python -m mvr.cli <lệnh>`):

| Lệnh | Việc gì |
|---|---|
| `ingest` | video → shot detection → keyframe (pipeline gốc, không cần dữ liệu BTC) |
| `build-index` | keyframe đã ingest → FAISS index (OpenCLIP) |
| `search` | tìm nhanh 1 câu qua CLI, không qua API |
| `build-aic-index` | dựng FAISS thẳng từ dataset BTC (không cần chạy `ingest` trước) |
| `build-asr-features` | ASR theo keyframe (VAD + ChunkFormer) |
| `enrich-ocr` | OCR theo shot (scene_text) |
| `benchmark-kis` | chấm điểm KIS thật (semantic vs semantic+refine) |

**Ngoài `mvr.cli`** (script độc lập dưới `scripts/`, không gõ qua `python -m mvr.cli`):
`run_feature_extraction.py` (trích embedding multi-provider), `indexing/build_provider_indexes.py`,
`indexing/build_lexical_index.py`, `indexing/verify_fasttest_baseline.py`,
`operations/run_full_corpus.py`, `operations/audit_provider_artifact.py`,
`verification/run_multi_embedding_ablation.py`, `verification/audit_multi_embedding_release.py`,
`verification/run_real_data_kis_smoke.py`, `verification/query_retrieval_ablation/` (bộ đo
A/B truy vấn — hiện hỏng, xem mục 4.3), `build_keyframe_mapping.py`, `run_qna_pipeline.py`,
`kaggle_qna_generate.py`, `import_qna_answers.py`, `build_qna_index_from_organizer_features.py`,
`find_qna_source_videos.py`, `verify_qna_ground_truth.py`, `run_trake_pipeline.py`.
Lý do TRAKE/Q&A/multi-embedding chưa vào `mvr.cli`: mỗi track được phát triển trên
nhánh riêng, script độc lập không đụng code chung — hợp nhất vào CLI chính thức là việc
dọn dẹp còn để sau (ghi rõ là **cố ý chưa làm**, không phải quên).

---

## 9. Test & CI — mức độ tin cậy thật

- Bộ test hiện tại: **860 passed / 14 skipped / 4 failed** (môi trường có cài `semantic`
  extras). 4 fail là lỗi môi trường cũ không liên quan (`test_weight_validation.py` gọi
  nhầm Python hệ thống thiếu `cv2`), không phải lỗi logic mới.
- **CI GitHub Actions** (`.github/workflows/ci.yml`, đã verify trực tiếp) chỉ chạy đúng
  1 job: `pip install -e .[dev,api,semantic]` rồi `pytest -q` — đã mở rộng so với bản đầu
  (trước chỉ `dev,api`) nhưng **vẫn chưa cài** `ocr`, `objects`, `asr`, `qna`, `search`
  (Elasticsearch) — nghĩa là toàn bộ OCR thật, ASR thật, object detection thật, answer
  reasoning thật, lexical BM25 thật **không bao giờ được CI chạy**; các test liên quan tự
  skip. Không có bước lint/type-check nào trong CI (không ruff, không mypy riêng). Mọi
  verify "thật" đều làm thủ công, ghi báo cáo tay (`*_PRE_PUSH_REPORT.md`,
  `*_FINAL_REPORT.md`), không có cổng tự động.
- Kỷ luật xác nhận trước khi tin số liệu: mọi task-pack (KIS/TRAKE) đều đối chiếu bằng
  môi trường sạch y hệt CI trước khi báo "xong", vì có lần phát hiện máy dev cài dư thư
  viện khiến số liệu không đại diện cho CI thật (Lượt 1 trong phần rà soát KIS).

---

## 10. Hạn chế / rủi ro tổng hợp (đọc kỹ trước khi thi thật)

Xếp theo mức độ ảnh hưởng tới khả năng thi thật:

**Nghiêm trọng — chặn đường thi nếu không xử lý**
1. Index KIS/TRAKE hiện chỉ phủ **L21 (29/1.478 video theo quy mô AIC 2025)** — chưa có
   dữ liệu cho phần lớn corpus thật. Build toàn corpus bị chặn chủ động
   (`operations/full_corpus.py`) chờ duyệt tài nguyên.
2. **Câu hỏi thi thật sẽ là tiếng Việt**, nhưng model CLIP đang dùng huấn luyện tiếng
   Anh thuần. Dịch máy đã có (KIS: chưa nối; TRAKE: có, tắt mặc định, phục hồi ~86%
   khoảng cách điểm) — chưa verify đủ để bật mặc định cho thi thật.
3. Q&A — track phức tạp nhất — mới có **1 kết quả end-to-end thật, n=3, Final Score
   trung bình 0.200**. Retrieval cấp frame còn yếu (mục 6.3), answer reasoning bị giới
   hạn bởi máy không GPU cần offload Kaggle thủ công mỗi lần.

**Đáng kể — ảnh hưởng chất lượng nhưng có đường vòng**
4. Lexical/BM25 thật đang **bị chặn bởi 1 bug hợp đồng `_id`** đã xác định rõ nguyên
   nhân (mục 4.3) — thiết kế sửa đã duyệt (2 đợt), code chưa xong. Trong lúc chờ, mọi
   nhánh cần BM25 (fusion Q&A, `/search/text` thật) coi như không có dữ liệu. Bộ đo A/B
   để chứng minh việc sửa này có lợi cũng đang hỏng (3 file rỗng 0 byte).
5. Exact-frame refinement (KIS) đúng nhưng vô dụng ở tốc độ hiện tại (205s/câu) — cần
   viết lại decode kiểu seek trước khi bật lại.
6. 3 embedding provider mới (BEiT-3/MetaCLIP-2/Perception Encoder) mới xong hạ tầng, chưa
   có bằng chứng nào cải thiện chất lượng — đừng giả định chúng tốt hơn OpenCLIP chỉ vì
   mới hơn.
7. Object detection chỉ phủ 6.8% video (file BTC cấp) — không dùng được ở diện rộng; YOLO/
   Faster R-CNN tự chạy là nguồn thay thế nhưng khác không gian nhãn, chưa được đưa vào
   Q&A context để tránh trộn nhãn sai.

**Nhẹ — biết trước, không bất ngờ**
8. `configs/default.yaml` vẫn không được code nào đọc — mọi default nằm rải rác trong
   source, đã tồn tại từ bản đầu.
9. CI vẫn chưa phủ OCR/ASR/objects/Q&A/Elasticsearch thật — chỉ phủ semantic (đã verify
   trực tiếp nội dung `ci.yml`, không phải suy đoán).
10. TRAKE/Q&A/multi-embedding chưa có lệnh CLI thống nhất, chỉ script rời — biết là nợ kỹ
    thuật, chưa xếp lịch dọn.
11. MetaCLIP-2 là CC-BY-NC-4.0 (phi thương mại) — cần xác nhận điều khoản thi đấu có xung
    đột không trước khi dùng chính thức.
12. Ground truth của cả 3 track hiện đều **tự soạn nội bộ** (không phải đề/đáp án chính
    thức của BTC) — mọi con số Final Score trong tài liệu này chỉ chứng minh pipeline nối
    đúng, không phải dự đoán điểm thi thật.

---

## 11. Khuyến nghị hướng đi tiếp theo (mentor's take)

Nhìn tổng thể, hệ thống đã đi rất xa: cả 3 track đều có đường chạy end-to-end thật, không
còn ở giai đoạn "khung sườn". Vấn đề bây giờ không phải là **thiếu tính năng** mà là
**thiếu 3 thứ liên kết các tính năng đã có lại với nhau ở quy mô thật**:

1. **Sửa bug `_id` lexical (mục 4.3) trước tiên** — đây là 1 sửa nhỏ nhưng mở khoá cả
   BM25 thật lẫn Q&A fusion, hiện đang là nút thắt cổ chai rẻ nhất để gỡ. Đợt 1 của thiết
   kế đã duyệt (sửa `_id` + thêm analyzer tiếng Việt) không phụ thuộc nhánh Q&A, làm được
   ngay trên `main`.
2. **Quyết định dứt điểm chuyện dịch Việt→Anh có bật mặc định hay không** cho cả 3 track
   (hiện KIS chưa có, TRAKE có nhưng tắt) — vì đề thi thật chắc chắn tiếng Việt, đây
   không phải tính năng phụ.
3. **Build 1 lần index thật trên toàn corpus** (ít nhất cho track được ưu tiên nhất) sau
   khi đã duyệt tài nguyên — mọi benchmark hiện tại đang đo trên 2% quy mô thật, rủi ro
   con số không đại diện là có thật.

Việc tối ưu decode seek-based cho refinement, hay đánh giá 3 provider mới có thật sự tốt
hơn không, tôi xếp sau — vì cả hai đều là "làm tốt hơn cái đã chạy được", trong khi 3 việc
trên là "làm cho cái chưa chạy được bắt đầu chạy".

---

## 12. Đọc thêm

| Muốn biết sâu hơn về | Đọc file |
|---|---|
| KIS đầy đủ | `docs/AIC_KIS_FINAL_REPORT.md`, `docs/AIC_KIS_IMPLEMENTATION_NOTES.md` |
| TRAKE đầy đủ | `docs/AIC_TRAKE_FINAL_REPORT.md`, `docs/AIC_TRAKE_BENCHMARK_RESULTS.md` |
| Q&A đầy đủ (rất chi tiết, có log vận hành thật) | `docs/AIC_QNA_IMPLEMENTATION_NOTES.md`, `docs/AIC_QNA_TEAM_TASKS.md`, `docs/QNA_KAGGLE_RUNBOOK.md` |
| Multi-embedding (BEiT-3/MetaCLIP-2/Perception Encoder) | `docs/MULTI_EMBEDDING_FINAL_RELEASE_REPORT.md`, `docs/EMBEDDING_PROVIDER_SPECS.md`, `docs/FULL_CORPUS_MULTI_EMBEDDING_RUNBOOK.md` |
| ASR | `docs/ASR_KEYFRAME_PIPELINE.md`, `docs/ASR_PRE_PUSH_REPORT.md` |
| Quy tắc định danh keyframe (đọc trước khi viết code mới đụng keyframe) | `docs/KEYFRAME_IDENTITY.md` |
| Bug lexical `_id` đang sửa, thiết kế đầy đủ | `docs/superpowers/specs/2026-08-21-tkis-query-branch-split-design.md` |
| Quy ước code/git/reuse | `AGENTS.md`, `docs/GIT_WORKFLOW.md`, `docs/LAB_REUSE_MAP.md` |
| Lịch sử từng commit/phase | `docs/CODEX_PROGRESS.md` |
