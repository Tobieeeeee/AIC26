# Đo cải tiến vòng 2

BTC **không công bố điểm vòng 1**. Không có thước đo nào từ bên ngoài. Thứ duy nhất còn lại là 17 câu mà người vận hành đã mở ảnh ra xem và xác nhận đáp án đúng hoặc gần đúng.

Tài liệu này nói bộ đo đó đo được gì, **không** đo được gì, và cách dùng.

```
data/ground_truth/round1_verified.json      17 câu đã xác minh
scripts/build_round1_ground_truth.py        dựng lại file trên
scripts/score_verified_set.py               chạy đo
src/mvr/evaluation/verified_set.py          luật chấm
```

## Chạy

```bash
# đo hiện trạng, ghi lại làm mốc
python scripts/score_verified_set.py --save data/ground_truth/baseline_en.json

# ... sửa hệ thống ...

# đo lại và xem câu nào đổi
python scripts/score_verified_set.py --against data/ground_truth/baseline_en.json
```

`--compare` chấm cả tiếng Việt lẫn tiếng Anh trong một lần nạp model (~40 giây nạp, ~5 giây cho 16 câu).

## Số đo được, ngày 24/08/2026

Index 177.321 vector, `ViT-B-32-quickgelu/openai`, CPU, `--per-video 3`.

### Mức video — "có tìm ra đúng video không, ở hạng mấy"

| | tiếng Việt | tiếng Anh |
|---|---|---|
| R@1 | 0.0000 | **0.6875** |
| R@5 | 0.0000 | 0.7500 |
| R@20 | 0.0000 | 0.8750 |
| R@50 | 0.0000 | 0.9375 |
| R@100 | 0.0625 | 0.9375 |
| **Mean Final** | **0.0125** | **0.8375** |
| tìm ra video | 1/16 | 15/16 |

Đây là bằng chứng thứ hai, trên n=16 thay vì n=3, cho cùng một kết luận: **encoder không hiểu tiếng Việt**. Không phải "hiểu kém" — là 1/16 so với 15/16.

### Mức frame — "có trúng đúng khoảnh khắc không"

Chấm với `--frame-window N`, tức chỉ tính đúng khi frame nằm trong `frame_idx ± N`:

| cửa sổ | R@1 | R@5 | Final |
|---|---|---|---|
| cả video | 0.6875 | 0.7500 | **0.8375** |
| ± 0 | 0.3125 | 0.4375 | 0.4500 |
| ± 25 | 0.3750 | 0.5000 | 0.5125 |
| ± 125 | 0.3750 | 0.5000 | 0.5125 |
| ± 500 | 0.3750 | 0.5625 | 0.5750 |

**Khoảng cách 0.8375 → 0.51 là chỗ mất điểm lớn nhất còn lại.** Tìm đúng video rồi nhưng rơi vào sai khoảnh khắc trong video đó.

Vì sao nới cửa sổ ±25 lên ±125 không đổi gì — đo trên 15 câu tìm đúng video, độ lệch giữa frame ta trả về và frame đã xác minh:

```
0, 0, 0, 0, 0, 0, 0, 0, 15  |  300, 475, 675, 1046, 1080, 1481
```

Phân bố **hai cụm, rỗng ở giữa**. Trúng thì trúng chính xác; trượt thì trượt 300–1500 frame, tức 12–60 giây — một khoảnh khắc khác hẳn. Không có kiểu lỗi "lệch sang keyframe kế bên", nên mọi cửa sổ nằm trong vùng chết 15–300 đều cho cùng một con số.

**Không phải do lưới keyframe thưa.** Keyframe BTC phát sẵn cách nhau trung vị **2,16 giây** (55 frame), p95 là 6,00 giây — thích ứng và có trần cứng. Trích dày hơn sẽ không cứu được các câu đang hỏng: chúng không thiếu keyframe, chúng thiếu khả năng phân biệt hai khoảnh khắc trong cùng một video. Đó là việc của OCR, ASR và re-rank theo thời gian, không phải của selector.

## Bộ này đo được gì, và KHÔNG đo được gì

Phải đọc kỹ mục này trước khi tin bất kỳ con số nào ở trên.

**17 câu này được chọn vì chúng là những câu đã làm được.** 8 câu còn lại của vòng 1 (8, 14, 18, 19, 20, 21, 23, 24) không có mặt ở đây, vì không ai biết đáp án đúng của chúng. Nên:

- **0.8375 không phải điểm hệ thống.** Nó là điểm trên tập con hệ thống vốn đã giải được. Điểm thi thật thấp hơn.
- **Bắt tụt điểm thì rất tốt.** 11/16 câu đang ở hạng 1. Một thay đổi làm chúng tụt xuống là hồi quy thật, và bộ này thấy ngay.
- **Chứng minh cải tiến thì yếu — ở mức video.** Trần đã gần chạm: chỉ còn 1 câu chưa tìm ra video (p1-25) và 2 câu ở hạng > 5.
- **Chứng minh cải tiến thì mạnh — ở mức frame.** Chỉ 5/16 frame trong ground truth trùng với frame máy trả về; 11/16 còn lại do người tự tìm trong video. Phần này không bị lệch chọn mẫu, và còn nhiều dư địa (0.45–0.51).

Kết luận thực dụng: **dùng mức video để canh hồi quy, dùng mức frame để đo tiến bộ.**

## Vì sao chấm ở mức video, không phải mức frame

Người xác minh nhìn **một frame** và nói "đúng hoặc tương đối đúng". BTC chấm KIS theo khoảng `[start_frame, end_frame]` — và **không công bố khoảng đó**.

Nên video là kiến thức chắc chắn; frame là một mẫu rút ra từ một khoảng không biết rộng bao nhiêu. Chấm frame theo một khoảng do mình tự đặt ra thì đang đo chính cái mình đặt ra. Vì vậy `--frame-window` được ghi rõ là **ước lượng**, dùng để so tương đối giữa hai lần chạy, không bao giờ báo cáo như điểm BTC.

Luật chấm không viết lại: `evaluate_kis` trong `mvr/evaluation/aic_kis.py` đã mã hoá đúng R@1/5/20/50/100 và Final Score của BTC, và ground truth mức video chính là luật đó với khoảng phủ trọn video.

## Bẫy trùng tên, đã tránh

`artifacts/qna/ground_truth.json` đã có sẵn 3 câu mang tên `query-p1-15-qa`, `query-p1-19-qa`, `query-p1-22-qa` — **từ một gói đề luyện tập khác** đánh số trùng cách. `query-p1-15-qa` của nó hỏi tên một xã; `query-p1-15-qa` của vòng 1 hỏi về bản đồ động đất. Hai câu khác nhau, cùng tên.

Vì thế bộ mới nằm ở file riêng và không bao giờ trộn vào file cũ. File cũ vẫn còn giá trị: nó là bộ duy nhất **không bị lệch chọn mẫu**, vì được xác minh trước khi biết hệ thống trả về gì.

## Khi xác minh thêm câu

Sửa `VERIFIED_ROUND_1` trong `scripts/build_round1_ground_truth.py` rồi chạy lại, hoặc truyền thẳng:

```bash
python scripts/build_round1_ground_truth.py --verified 1 2 3 ... 19
```

Câu càng khó mà thêm được thì bộ đo càng bớt lệch. Ưu tiên 8 câu vòng 1 chưa giải được.
