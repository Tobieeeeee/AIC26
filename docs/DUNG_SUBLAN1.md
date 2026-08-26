# Dùng file sublan1 để sửa và mở rộng ground truth

`artifacts/sublan1` là bài nộp vòng 1 của một đội khác, **tự nguyện chia sẻ** với đội mình để học hỏi. Không phải dữ liệu bị lộ. File này nói cách dùng nó cho đúng.

## Phát hiện quan trọng nhất, đo được trước khi làm gì cả

Đối chiếu sublan1 với `data/ground_truth/round1_verified.json` (17 câu đội mình đã xác minh bằng mắt):

```
khớp video       9/17
KHÁC VIDEO       8/17   (2, 3, 5, 9, 11, 12, 16, 17)
```

Không phải lệch vài frame — **khác hẳn video**. Đây là bằng chứng trực tiếp rằng **không nguồn nào được tin tuyệt đối**: ground truth của đội mình có thể sai ở tối đa 8 chỗ, và sublan1 "điểm cao nhất tổng thể" không có nghĩa đúng ở từng câu cụ thể.

Vì vậy: **không bao giờ tự động merge sublan1 vào ground truth.** Mọi thay đổi đi qua một người xem ảnh và xác nhận.

## Quy trình ba bước

### Bước 1 — dựng sheet để xem

```bash
python scripts/build_sublan1_review.py
```

Ghi ra `artifacts/sublan1_review/`:

| File | Nội dung |
|---|---|
| `video_mismatch.csv` | 7 câu KIS/QA lệch video (loại trừ câu 16-trake — chain nhiều frame, không so kiểu này được) — mỗi câu có 1 dòng "ta" + các dòng "sublan1" |
| `new_candidates.csv` | 8 câu vòng 1 chưa từng xác minh (8, 14, 18, 19, 20, 21, 23, 24) — chỉ có ứng viên từ sublan1 |

Mỗi dòng có `image_path` trỏ thẳng đến ảnh keyframe **gần frame đó nhất** (không nhất thiết đúng khung, vì `frame_idx` là chỉ số khung hình thi đấu, không phải id keyframe) và `distance_frames` — độ lệch giữa hai cái, để biết ảnh có đại diện đúng hay không.

### Bước 2 — bạn mở ảnh, đánh dấu

Mở từng CSV (Excel hoặc trình soạn thảo), với mỗi câu:
1. Mở lần lượt các `image_path` trong cùng nhóm câu đó.
2. Xem dòng nào khớp mô tả `query_vi`.
3. Gõ `x` vào cột `chon_dong_nay` **ở đúng một dòng** — kể cả khi đó là dòng "ta (đã xác minh)", nghĩa là bạn xem lại và xác nhận đáp án cũ đúng.
4. Không chắc thì để trống cả câu đó — an toàn hơn đoán.

### Bước 3 — áp lại vào ground truth

```bash
python scripts/apply_sublan1_review.py            # xem trước, không ghi gì
python scripts/apply_sublan1_review.py --write     # ghi thật
```

Quy tắc áp dụng:

- `video_mismatch.csv`: dòng `x` **thay thế** mục cũ trong `round1_verified.json`. Nếu dòng `x` là dòng "ta", **không đổi gì** — không tự gắn lại `verified_by`.
- `new_candidates.csv`: dòng `x` trở thành **mục mới**. Không bao giờ ghi đè một mục đã có sẵn.
- Mọi mục do bước này tạo/sửa mang `"verified_by": "human (đối chiếu sublan1)"` — khác với `"human"` của 17 mục gốc, để nhìn vào file là biết nguồn.
- Một câu có **hai dòng `x`** → script dừng và báo lỗi, không đoán dòng nào đúng.

### Sau đó — đo lại

```bash
python scripts/score_verified_set.py --against data/ground_truth/baseline_en.json
```

## Kết quả thật, ngày 26/08/2026

Đã làm hết quy trình trên. Kết quả:

- **7/7 xung đột** (câu 2, 3, 5, 9, 11, 12, 17) đều chọn đáp án của sublan1, không câu nào giữ đáp án cũ.
- **8/8 câu chưa xác minh** (8, 14, 18, 19, 20, 21, 23, 24) đều xác minh được — ground truth giờ **n=25, phủ trọn vòng 1**.
- Dò lại hạng của 7 video bị bác trong chính kết quả tìm kiếm hệ thống mình: **5/7 đứng hạng 1**. Đây là bằng chứng đội mình mắc lỗi **vòng lặp tự xác nhận** — xác minh vòng 1 bằng cách xem top kết quả tìm kiếm rồi chấp nhận, tức lấy chính hệ thống cần đo để tạo ra thước đo cho nó.
- Mean Final Score (mức video, n=24 KIS+QA) tụt từ **0.8375 xuống 0.5083** — không phải hệ thống kém đi, mà là con số cũ phần lớn ảo do vòng lặp trên.

Chi tiết đầy đủ, gồm bảng hạng trước/sau và số đo mức frame, xem [DO_LUONG_VONG_2.md](DO_LUONG_VONG_2.md).

## Vì sao không tự động hoá bước 2

Đã thử tính "sublan1 đúng hơn vì họ điểm cao" — không dùng được, vì 8/17 câu chính họ cũng khác với đội mình, và không có cách nào tự động biết bên nào đúng mà không nhìn ảnh. Nhìn bằng mắt là bước rẻ nhất còn lại; phần đắt (dò 177.321 keyframe, tính khoảng cách) đã làm sẵn ở bước 1.
