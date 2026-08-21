# Vòng sơ tuyển — quy trình thao tác

Đọc trang này lúc thi. Mọi thứ khác để sau.

---

## 0. Chuẩn bị trước khi BTC phát đề (làm ngay, một lần)

```powershell
cd D:\AIC26
.venv\Scripts\activate
python -m pytest tests/unit -q
```

Phải ra `773 passed`. Nếu không, dừng lại và sửa trước — đừng đợi tới lúc có đề.

Kiểm hai file bắt buộc:

```powershell
dir artifacts\indexes\qna_organizer_all\keyframes.faiss    # index 177.321 vector, 873 video
dir artifacts\keyframe_mapping.csv                          # bảng tra frame
```

Thiếu `keyframe_mapping.csv` thì sinh lại, 10 giây:

```powershell
python scripts\build_keyframe_mapping.py --dataset-root D:\AIC26_data --output artifacts\keyframe_mapping.csv
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
python scripts\answer_queries.py --queries artifacts\queries --output artifacts\submissions
```

Mất ~40 giây khởi động, rồi **0,2–2 giây mỗi truy vấn**. Chạy một lần cho cả gói, đừng chạy từng câu.

Đọc kỹ output. Có dòng `CẢNH BÁO ... không có ....en.txt` nghĩa là truy vấn đó đang chạy bằng tiếng Việt và **sẽ được 0 điểm** — viết file `.en.txt` rồi chạy lại.

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
python scriptsnswer_queries.py --interactive
```

```
> volunteers in matching club shirts handing charity gift boxes to villagers
  100 dòng / 61 video  (2.69s)
    1.   L30_V072/1776    0.364  D:\AIC26_data\...\L30_V072.jpg
    2.   L21_V009/387     0.343  ...
> :o                      ← mở ảnh hạng 1 để xem có đúng không
> :v 1                    ← mỗi video chỉ 1 dòng, trải rộng hơn
> <sửa lại câu mô tả>     ← 0,2 giây
> :w kis query-1-kis      ← ghi ra query-1-kis.csv
> :q
```

Lệnh: `:n <số>` đổi số dòng hiển thị · `:v <số>` đổi giới hạn mỗi video ·
`:o` mở ảnh hạng 1 · `:w kis|qa <tên>` ghi file · `:q` thoát · `:h` trợ giúp.

**Dùng `:o` nhiều vào.** Nhìn ảnh là cách nhanh nhất biết câu mô tả có đúng
hướng không — điểm số trên màn hình không nói được điều đó.

---

## 3. Q&A: điền câu trả lời

`query-3-qa.csv` nộp được luôn nhưng cột answer trống, nên chỉ ăn điểm phần frame.

Muốn có điểm answer, mở `query-3-qa.review.csv`:

```
rank,video_id,frame_idx,answer,score,image_path
1,L25_V045,15300,,0.346,D:\AIC26_data\Keyframes_L25\...\170.jpg
```

Mở ảnh ở cột `image_path`, nhìn, gõ câu trả lời vào cột `answer`. Chỉ cần làm **20–30 dòng đầu** — R@k thưởng việc xếp đáp án đúng lên sớm.

Rồi dựng lại file nộp:

```powershell
python scripts\answer_queries.py --from-review artifacts\submissions\query-3-qa.review.csv --output artifacts\submissions\query-3-qa.csv
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
- **GPU** — toàn bộ vòng 1 chạy CPU trên máy có `D:\AIC26_data`. Index đã dựng sẵn từ CLIP feature BTC cung cấp.
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
