# Việc cần người có GPU / Docker làm

Máy chạy phần Q&A **không có GPU, không có Docker, RAM 3,8 GB (còn trống 0,5 GB)**:

```
torch 2.13.0+cpu   |  cuda: False
nvidia-smi         : không có
docker --version   : không có
```

Bốn việc dưới đây vì thế không ai bên Q&A làm hộ được. Mọi thứ khác trong
chuỗi đã xong hoặc đang được xử lý — xem mục "Không cần các bạn làm".

---

## Bối cảnh: vì sao cần bốn việc này

Retrieval hiện tại chỉ có **semantic (CLIP)**. Hai trong ba câu hỏi thật
(p1-19, p1-22) trượt vì **retrieval**, không phải vì trả lời sai: trong 100
ứng viên **không có lấy một frame nào thuộc đúng video**. Mở rộng theo thời
gian không cứu được — không có gì để mở rộng quanh.

Thứ cứu được là **tìm theo chữ** (BM25): người ta hỏi "xã Giang Ly", chữ đó
nằm ngay trên màn hình. Nhưng index chữ chỉ tìm trên đúng hai trường:

```python
SEARCHABLE_FIELDS = ["ocr_text", "asr_text"]      # indexing/lexical_index.py:20
"lexical": bool(ocr or asr)                        # indexing/task05a.py:1134
```

**Chưa chạy OCR và ASR thì index rỗng — không phải index nhỏ, mà là rỗng.**

Chuỗi phụ thuộc:

```
keyframe_mapping.csv  ✅ xong (166.550 dòng)
   ├─ OCR   ❌  ← T1, cần GPU
   ├─ ASR   ❌  ← T2, cần GPU
   ▼
Task05A documents  (chờ T1+T2)
   ▼
Elasticsearch  ❌  ← T3, cần Docker/RAM
   ▼
lexical fusion  ✅ code sẵn sàng
```

---

## T0 — Đo tốc độ OCR trên GPU  ⏱ 15 phút  ⚠️ LÀM TRƯỚC TIÊN

**Đây là con số quyết định T1 và T2 có đáng làm không.** Chưa ai đo. Đừng
đổ 10 giờ vào T1 trước khi có nó.

Lấy 200 ảnh keyframe bất kỳ, chạy OCR trên GPU, ghi lại **giây mỗi ảnh**.

```powershell
# 200 ảnh mẫu
python -c "import glob,random,shutil,os; os.makedirs('bench200',exist_ok=True); [shutil.copy(p,'bench200') for p in random.sample(glob.glob('D:/AIC26_data/Keyframes_*/keyframes/*/*.jpg'), 200)]"
```

Rồi chạy engine OCR của các bạn trên thư mục đó và báo lại:

```
số ảnh : 200
tổng    : ___ giây
→ ___ giây/ảnh
```

### Con số đó nghĩa là gì

166.550 keyframe. Bảng quy đổi:

| giây/ảnh | OCR ảnh có sẵn (166.550) | OCR kèm ±1 frame (499.650) |
|---|---|---|
| 0,05 | **2,3 giờ** ✅ | 6,9 giờ |
| 0,10 | **4,6 giờ** ✅ | 13,9 giờ ❌ |
| 0,20 | 9,3 giờ ⚠️ | 27,8 giờ ❌ |
| 0,50 | 23 giờ ❌ | 69 giờ ❌ |
| 6,7 (CPU, đã đo) | 310 giờ ❌ | 930 giờ ❌ |

**Nếu > 0,2 giây/ảnh thì bỏ hướng toàn corpus**, chuyển sang OCR tập con
(xem cuối file).

---

## T1 — OCR toàn corpus  ⚠️ CHƯA CHẠY ĐƯỢC, xem phần chặn

### Dùng đường nào

Có hai đường. **Dùng `enrich-ocr`, đừng dùng `docker_ocr`:**

| | `mvr.cli enrich-ocr` | `docker_ocr` |
|---|---|---|
| Docker | không cần ✅ | cần |
| GPU | `--device cuda` ✅ | qua container |
| Xử lý | **từng video một** ✅ | nạp cả 873 video một lượt |
| Định danh | an toàn ✅ | **hỏng** — xem dưới |

`docker_ocr` gom `samples_by_keyframe[keyframe_id]` trên toàn bộ 873 video
cùng lúc. Keyframe BTC chỉ có 632 id phân biệt (`"001"` có ở ~850 video),
nên sample của mọi video dồn hết vào một khoá. Kết quả OCR gắn sai frame.
`enrich-ocr` nhóm theo `video_id` trước nên không dính.

### Lệnh

```powershell
# 1. Sinh mapping (11 giây, chạy trên máy bạn)
python scripts/build_keyframe_mapping.py `
    --dataset-root D:/AIC26_data `
    --output artifacts/keyframe_mapping.csv

# 2. OCR
python -m mvr.cli enrich-ocr `
    --mapping-csv artifacts/keyframe_mapping.csv `
    --video-root <thư mục video PHẲNG> `
    --output artifacts/features/text `
    --device cuda
```

### ⚠️ Ba thứ làm lệnh trên hỏng ngay — tôi đang sửa, đừng chạy trước khi tôi báo xong

1. **Thiếu `shots.json` → crash.** `process_dataset_ocr` gọi
   `get_shot_boundaries()` không kiểm tra file có tồn tại không
   (`scene_text.py:378`), nên `FileNotFoundError` ngay dòng đầu.

2. **`--video-root` bắt buộc phẳng.** Code tìm `video_root/<video_id>.mp4`
   (`scene_text.py:382`). Data gốc lồng hai cấp:
   `D:/AIC26_data/Videos_L21_a/video/L21_V001.mp4` — 14 thư mục.
   Không khớp thì nó **in warning rồi bỏ qua**, chạy xong báo thành công
   với 0 file. `D:/AIC26_test_root/videos/` phẳng nhưng chỉ có **60/873** video.

3. **`shot_boundaries` khoá theo `shot_id` trần** (`scene_text.py:174`).
   `shot_id` 1 của video A và của video B đè nhau.

Tôi sửa cả ba. **Sẽ nhắn khi xong** — chạy T1 trước đó thì kết quả sai mà
không báo lỗi.

---

## T2 — ASR toàn corpus

```powershell
python -m mvr.cli build-asr-features `
    --dataset-root <root> `
    --output artifacts/features/asr `
    --device cuda
```

### Trước khi chạy, cân nhắc đổi cách làm

Hiện tại ASR cắt cửa sổ **±3 giây quanh từng keyframe**
(`cli.py:144-145`). Keyframe BTC cách nhau 2–4 giây, nên các cửa sổ
**chồng lên nhau khoảng 2 lần**:

```
166.550 keyframe × 6 giây  =  277 giờ audio phải nhận dạng
video thật                 =  130 giờ
                              → phí ~2,1 lần
```

**Đề xuất:** nhận dạng mỗi video **một lần từ đầu đến cuối** (130 giờ), lưu
transcript kèm timestamp, rồi cắt theo keyframe sau. Rẻ hơn 2,1 lần và
context tốt hơn hẳn — câu nói không bị cắt cụt giữa chừng.

Đây là thay đổi bên ASR pipeline nên tôi để các bạn quyết. Nếu chốt làm,
báo tôi biết format transcript để bên Q&A đọc.

Cũng cần đo trước như T0: chạy 3 video, ghi lại **hệ số realtime** (1 giờ
audio mất bao nhiêu phút).

---

## T3 — Dựng Elasticsearch

Cần Docker hoặc server riêng, và RAM cho JVM heap. Máy Q&A còn 0,5 GB nên
không dựng được.

```powershell
docker run -d --name es -p 9200:9200 `
  -e "discovery.type=single-node" -e "xpack.security.enabled=false" `
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" `
  docker.elastic.co/elasticsearch/elasticsearch:8.14.0
```

Chỉ cần **URL** (`http://localhost:9200` hoặc IP nội bộ). Phần build index
và nối vào Q&A tôi làm.

### ⚠️ Đừng build index trước khi tôi sửa xong — sẽ mất dữ liệu im lặng

Đã dựng 2 document (`L21_V001/001`, `L21_V002/001`) chạy thử:

```
CHẶN 1 — load_task05a_lexical_source: Task05BValidationError
   duplicate Task05A keyframe IDs: ['001']

CHẶN 2 — _id của Elasticsearch: ['001', '001']
   2 tài liệu → 1 _id phân biệt
```

Chặn 2 nguy hơn: nếu ai gỡ chặn 1, Elasticsearch **ghi đè im lặng** —
166.550 document co lại còn **632**, không một dòng báo lỗi.

---

## Gốc rễ: vì sao cùng một lỗi cứ xuất hiện

`keyframe_id` do đội tự sinh là:

```python
f"{video_id}_S{shot_id:04d}_K{ordinal:02d}"     # → L21_V001_S0007_K01
```

**Duy nhất trên toàn corpus.** Nên mọi chỗ dùng `keyframe_id` làm khoá đều
đúng, suốt từ đầu.

`keyframe_id` của BTC là `034` — chỉ duy nhất **trong một video**.
Chuyển sang keyframe BTC đã phá vỡ bất biến mà cả codebase dựa vào, và
chưa ai tuyên bố điều đó. Đây là lý do cùng một lỗi xuất hiện ở sáu chỗ,
không phải sáu lỗi rời rạc.

**Từ giờ: mọi khoá phải là `(video_id, keyframe_id)`.** Nếu các bạn viết
thêm code đụng tới keyframe, đây là quy tắc duy nhất cần nhớ.

---

## Không cần các bạn làm — bên Q&A đang lo

| Việc | Trạng thái |
|---|---|
| `keyframe_mapping.csv` từ data BTC (166.550 dòng, 11 giây) | ✅ xong |
| `ocr/validator.py` — khoá `(video_id, keyframe_id)` | ✅ xong, kèm 4 test |
| `lexical_index.py` — 2 chỗ định danh | 🔧 đang làm |
| `scene_text.py` — 3 chỗ (shots.json, video-root, shot_boundaries) | 🔧 đang làm |
| `docker_ocr.py` — gộp sample sai video | 🔧 đang làm |
| Task05A build documents | ⏳ chờ T1+T2 |
| Build lexical index + nối fusion | ⏳ chờ T3 |

---

## Nếu T0 cho ra > 0,2 giây/ảnh

Bỏ hướng toàn corpus. Làm tập con thay thế:

1. Chạy semantic search cho p1-19 và p1-22, lấy **top 300 video**.
2. OCR chỉ những video đó (~60.000 keyframe thay vì 166.550).
3. Đo xem lexical có kéo được đúng video vào top-100 không.

Mất khoảng 1/3 thời gian, và trả lời được câu hỏi "hướng này có ăn thua
không" trước khi cam kết. Bên Q&A dựng được danh sách video, chỉ cần các
bạn chạy OCR.

---

## Tóm tắt: làm gì bây giờ

1. **T0 ngay** — 15 phút, quyết định mọi thứ còn lại.
2. **T3 song song** — dựng Elasticsearch, không phụ thuộc gì.
3. **T1/T2 chờ** hai điều kiện: T0 cho số khả thi, **và** tôi báo đã sửa xong.

Báo lại con số T0 sớm nhất có thể — còn ~10 giờ.
