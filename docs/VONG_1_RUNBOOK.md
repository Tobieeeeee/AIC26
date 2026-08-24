# Vòng sơ tuyển — quy trình thao tác

Đọc trang này lúc thi. Mọi thứ khác để sau.

---

## Cài đặt cho máy mới (đồng đội đọc mục này trước)

Merge code là chưa đủ. Hai thứ hệ thống cần đều **không nằm trong git**:

```
artifacts/indexes/qna_organizer_all/    376 MB
artifacts/keyframe_mapping.csv           15 MB
```

**Đừng chép chúng từ máy người khác.** Cả hai nhúng đường dẫn tuyệt đối
kiểu `E:\AIC26_data\Keyframes_L21\...`, sang máy khác là trỏ vào hư không —
mà không báo lỗi, chỉ ra kết quả sai. Dựng lại tại chỗ, mất vài phút.

**Cũng phải dựng lại nếu bạn di chuyển thư mục dữ liệu.** Đã xảy ra một
lần: dataset chuyển từ `D:` sang `E:`, tìm kiếm vẫn chạy bình thường nhưng
**100% đường dẫn ảnh chết** — `:o` không mở được ảnh và cột `image_path`
trong file review Q&A trỏ vào hư không. Kiểm nhanh:

```powershell
.venv\Scripts\python.exe -c "import csv,os;r=next(csv.DictReader(open('artifacts/keyframe_mapping.csv',newline='',encoding='utf-8')));print(r['image_path'], os.path.isfile(r['image_path']))"
```

In ra `False` là phải chạy lại hai lệnh dựng ở trên.

### Cần có trước

Bộ dữ liệu BTC. Đủ để chạy vòng 1 chỉ cần bốn thư mục này (~30 GB):

```
<DATA>/Keyframes_*/keyframes/<video_id>/*.jpg
<DATA>/map-keyframes-aic25-b1/map-keyframes/<video_id>.csv
<DATA>/clip-features-32-aic25-b1/clip-features-32/<video_id>.npy
<DATA>/media-info-aic25-b1/media-info/<video_id>.json
```

**Không cần** `Videos_*` (78 GB) — vòng 1 không giải mã video.

### Ba lệnh

```powershell
git clone https://github.com/Tobieeeeee/AIC26.git
cd AIC26
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

.venv\Scripts\python.exe scripts\build_keyframe_mapping.py ^
    --dataset-root <DATA> --output artifacts\keyframe_mapping.csv

.venv\Scripts\python.exe scripts\build_qna_index_from_organizer_features.py ^
    --dataset-root <DATA> --output artifacts\indexes\qna_organizer_all
```

Lệnh hai mất ~16 giây, lệnh ba vài phút. Thay `<DATA>` bằng đường dẫn bộ
dữ liệu trên máy đó.

### Kiểm

```powershell
.venv\Scripts\python.exe -m pytest tests/unit -q
```

Rồi làm bước 1 bên dưới. Nếu ra `100 dòng` là máy đó sẵn sàng.

### Con số phải khớp

```
mapping   177.321 dòng
index     177.321 vector, 873 video
```

Lệch nghĩa là bộ dữ liệu thiếu thư mục nào đó — kiểm lại `Keyframes_*`
(phải có đủ 14 thư mục, gồm cả `Keyframes_L29`).

---

## 0. Chuẩn bị trước khi BTC phát đề (làm ngay, một lần)

```powershell
cd D:\AIC26
.venv\Scripts\activate
python -m pytest tests/unit -q
```

Phải ra `796 passed`. Nếu không, dừng lại và sửa trước — đừng đợi tới lúc có đề.

Kiểm hai file bắt buộc:

```powershell
dir artifacts\indexes\qna_organizer_all\keyframes.faiss    # index 177.321 vector, 873 video
dir artifacts\keyframe_mapping.csv                          # bảng tra frame
```

Thiếu `keyframe_mapping.csv` thì sinh lại, 10 giây:

```powershell
.venv\Scripts\python.exe scripts\build_keyframe_mapping.py --dataset-root E:\AIC26_data --output artifacts\keyframe_mapping.csv
```

---

## 1. Khi có gói truy vấn

Đặt tất cả `query-*.txt` vào một thư mục, ví dụ `artifacts\queries\`.

### Bước bắt buộc: viết bản tiếng Anh cho từng truy vấn

Đây là bước quan trọng nhất và **không được bỏ**. Đo trên ground truth thật:

| Đầu vào | Mean Final |
|---|---|
| tiếng Việt gốc | **0.0000** |
| tiếng Anh viết tay | **0.3333** |

Encoder là OpenCLIP tiếng Anh thuần. Tiếng Việt vào là 0 điểm, không phải "kém hơn".

Với mỗi `query-1-kis.txt`, tạo thêm `query-1-kis.en.txt`.

**Cách viết**: mô tả **cái nhìn thấy trong khung hình**, không dịch câu hỏi.

```
Gốc   "Trong đoạn video có 2 câu thơ ca ngợi anh hùng Nguyễn Trung Trực
       trong đình thần Nguyễn Trung Trực tại Kiên Giang."

TỐT   "a stone tablet with two lines of Vietnamese poetry inside a
       traditional temple"

TỆ    "two lines of poetry praising hero Nguyen Trung Truc in the Nguyen
       Trung Truc temple in Kien Giang"
```

Bản "TỆ" dịch đúng nghĩa nhưng đầy tên riêng — CLIP không biết Nguyễn Trung Trực là ai. Bản "TỐT" nói **màu gì, vật gì, ở đâu, ai đang làm gì**.

Ba mẹo:
- Bỏ hết tên riêng, tên tổ chức, địa danh
- Thêm chi tiết thị giác: màu áo, loại phòng, trong nhà hay ngoài trời, số người
- Ngắn gọn: CLIP chỉ đọc 77 token, câu dài bị cắt

---

## 2. Chạy

```powershell
.venv\Scripts\python.exe scripts\answer_queries.py --queries artifacts\queries --output artifacts\submissions
```

Mất ~40 giây khởi động, rồi **0,2–2 giây mỗi truy vấn**. Chạy một lần cho cả gói, đừng chạy từng câu.

Đọc kỹ output. Có dòng `CẢNH BÁO ... không có ....en.txt` nghĩa là truy vấn đó đang chạy bằng tiếng Việt và **sẽ được 0 điểm** — viết file `.en.txt` rồi chạy lại.

Tên file ra **tự động đúng chuẩn BTC** — đề tên `query-p1-23-kis.txt` vẫn cho
ra `query-23-kis.csv`, không phải đổi tay.

Kết quả trong `artifacts\submissions\`:

```
query-1-kis.csv           ← nộp được ngay
query-3-qa.csv            ← nộp được, cột answer trống
query-3-qa.review.csv     ← file làm việc, không nộp
```

---

## 2b. Chưa ưng kết quả? Dùng mode tương tác

Sửa một chữ rồi chạy lại cả script là mất 40 giây nạp model. Mode tương tác
nạp một lần rồi mỗi lần thử chỉ **0,2 giây**:

```powershell
.venv\Scripts\python.exe scripts\answer_queries.py --interactive
```

```
> volunteers in matching club shirts handing charity gift boxes to villagers
  100 dòng / 61 video  (2.69s)
    1.   L30_V072/1776    0.364  E:\AIC26_data\...\L30_V072.jpg
    2.   L21_V009/387     0.343  ...
> :o                      ← mở ảnh hạng 1 để xem có đúng không
> :v 1                    ← mỗi video chỉ 1 dòng, trải rộng hơn
> <sửa lại câu mô tả>     ← 0,2 giây
> :w kis 1                ← ghi ra query-1-kis.csv
> :q
```

Lệnh: `:n <số>` đổi số dòng hiển thị · `:v <số>` đổi giới hạn mỗi video ·
`:o` mở ảnh hạng 1 · `:w kis|qa <stt>` ghi file · `:q` thoát · `:h` trợ giúp.

**Dùng `:o` nhiều vào.** Nhìn ảnh là cách nhanh nhất biết câu mô tả có đúng
hướng không — điểm số trên màn hình không nói được điều đó.

---

## 2c. Đã tự tìm ra ảnh đúng bằng mắt? Dùng `:top`

Lật thư mục keyframe thấy đúng khung hình rồi thì khỏi cần tìm nữa — chèn
thẳng lên hạng 1 và ghi file trong một lệnh:

```
> :top L21_V001_050 kis 23
   L21_V001/050 -> frame 5760, đã đưa lên hạng 1
   đã ghi artifacts\submissions\query-23-kis.csv  (100 dòng)
```

Cú pháp `<video_id>_<keyframe_id>`, lấy thẳng từ đường dẫn ảnh:

```
E:\AIC26_data\Keyframes_L21\keyframes\L21_V001\050.jpg
                                      ^^^^^^^^ ^^^
                                      video_id id

->  :top L21_V001_050 kis 23
```

Gõ `_50` thay `_050` cũng được, nó tự đệm số 0.

**`keyframe_id` không phải `frame_idx`.** `050` là tên file JPEG; cột thứ hai
của file nộp là số khung hình trong video gốc (`5760`). `:top` tra bảng
`keyframe_mapping.csv` để đổi. Cùng một `050` ở video khác ra số khác hẳn —
nên bắt buộc phải có `video_id` đi kèm.

Tra không ra thì nó **báo lỗi chứ không đoán bừa**. Thấy báo lỗi là kiểm lại
tên thư mục và tên file ảnh.

---

## 3. Q&A: điền câu trả lời

`query-3-qa.csv` nộp được luôn nhưng cột answer trống, nên chỉ ăn điểm phần frame.

Muốn có điểm answer, mở `query-3-qa.review.csv`:

```
rank,video_id,frame_idx,answer,score,image_path
1,L25_V045,15300,,0.346,E:\AIC26_data\Keyframes_L25\...\170.jpg
```

Mở ảnh ở cột `image_path`, nhìn, gõ câu trả lời vào cột `answer`. Chỉ cần làm **20–30 dòng đầu** — R@k thưởng việc xếp đáp án đúng lên sớm.

Rồi dựng lại file nộp:

```powershell
.venv\Scripts\python.exe scripts\answer_queries.py --from-review artifacts\submissions\query-3-qa.review.csv --output artifacts\submissions\query-3-qa.csv
```

Sửa file bằng Excel cũng được — lệnh trên đọc được cả file Excel lưu ra (có BOM).

---

## 4. Kiểm trước khi nộp

```powershell
notepad artifacts\submissions\query-1-kis.csv
```

Nhìn ba thứ:

- **Dòng đầu là dữ liệu**, không phải tên cột
- **KIS đúng 2 cột**, **Q&A đúng 3 cột**
- **Không quá 100 dòng**

BTC nhắc riêng: **phải là `.csv`, không phải `.xlsx`**. Nếu bạn mở bằng Excel rồi lưu, chọn *CSV (Comma delimited)*.

---

## 5. Khi hỏng

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `không thấy artifacts\keyframe_mapping.csv` | chưa sinh | chạy `build_keyframe_mapping.py` ở mục 0 |
| `không thấy file .txt nào` | sai thư mục `--queries` | kiểm lại đường dẫn |
| `BỎ QUA ...: không rõ loại truy vấn` | tên file không kết thúc bằng `-kis` / `-qa` | đổi tên file |
| File CSV ra 0 dòng | truy vấn rỗng, hoặc mapping không khớp index | kiểm file `.txt` có nội dung |
| Kết quả toàn một video | bình thường nếu mô tả rất đặc thù; thêm `--per-video 2` để trải rộng hơn |

Cần nộp gấp mà chưa kịp gì: **file `.csv` rỗng vẫn hợp lệ** và giữ được suất nộp. Không nộp gì mới là mất trắng.

---

## Những gì vòng 1 KHÔNG dùng

Nói rõ để khỏi mất thời gian tìm:

- **OCR, ASR, Elasticsearch** — không kịp trong vòng này. Nhánh BM25 hiện trả rỗng vì index lexical chưa có dữ liệu. Đây là việc vòng 2.
- **GPU** — toàn bộ vòng 1 chạy CPU trên máy có `E:\AIC26_data`. Index đã dựng sẵn từ CLIP feature BTC cung cấp.
- **TRAKE** — do người khác phụ trách. Script tự bỏ qua file `-trake`.

## Con số đã đo, để so sánh về sau

```
index            177.321 vector, ViT-B-32-quickgelu/openai, 873 video (gồm cả L29)
khởi động        ~40 giây (index 4s + encoder 30s + mapping 2s)
mỗi truy vấn     0,2–2 giây khi đã ấm
Mean Final       0.0000 tiếng Việt  |  0.3333 tiếng Anh   (n=3, ground truth p1)
đa dạng          100 dòng / 61 video với --per-video 3 (mặc định)
```

`n=3` là rất nhỏ. Đừng coi 0.3333 là dự báo điểm thi — nó chỉ chứng minh **tiếng Anh ăn đứt tiếng Việt**, và đó là điều duy nhất cần nhớ.
