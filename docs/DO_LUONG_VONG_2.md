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

## CẢNH BÁO: số đo ngày 24/08 bị vòng lặp tự xác nhận, đã sửa ngày 26/08

Số 0.8375 dưới đây **không dùng làm mốc nữa**. Lý do, đo được chứ không phải suy đoán:

Ground truth 17 câu ban đầu được dựng bằng cách: chạy tìm kiếm, xem top kết quả, thấy hợp lý thì chấp nhận làm "đã xác minh". Đối chiếu với `artifacts/sublan1` (bài nộp một đội khác, chia sẻ tự nguyện) phát hiện **8/17 câu lệch hẳn video**. Xem lại bằng mắt qua quy trình ở [DUNG_SUBLAN1.md](DUNG_SUBLAN1.md), 7/7 trường hợp tranh chấp — không phải một vài — sublan1 đúng, đội mình sai.

Bằng chứng vòng lặp: dò lại hạng của **video bị bác** trong chính kết quả tìm kiếm của hệ thống mình:

```
câu               hạng video CŨ (bị bác)   hạng video MỚI (đúng)
query-p1-2-kis                   1                    6
query-p1-3-qa                    1                   52
query-p1-5-kis                   1                   56
query-p1-9-qa                    1                   97
query-p1-11-kis                  1                    2
query-p1-12-kis                  3                    7
query-p1-17-qa                   1                    5
```

**5/7 video bị bác đứng hạng 1**, một hạng 3. Người xác minh không hề sai lười — họ nhìn top kết quả, thấy có vẻ hợp lý, chấp nhận. Nhưng đó là lấy chính hệ thống cần đo để tạo ra thước đo cho nó: hệ thống luôn "đúng" theo ground truth do chính nó gợi ý. 0.8375 phần lớn là ảo giác từ vòng lặp này.

Ground truth hiện đã sửa 7 câu này và thêm 8 câu chưa từng xác minh (8, 14, 18, 19, 20, 21, 23, 24) từ cùng nguồn đối chiếu — **n=25, phủ trọn vòng 1**, hết lệch chọn mẫu theo hướng "chỉ chọn câu dễ".

## Số đo được, ngày 26/08/2026 (sau khi sửa)

Index 177.321 vector, `ViT-B-32-quickgelu/openai`, CPU, `--per-video 3`, n=24 (KIS+QA, trake loại trừ).

### Mức video — "có tìm ra đúng video không, ở hạng mấy"

| | trước khi sửa (n=16, có vòng lặp) | sau khi sửa (n=24) |
|---|---|---|
| R@1 | 0.6875 | **0.2083** |
| R@5 | 0.7500 | 0.3333 |
| R@20 | 0.8750 | 0.5833 |
| R@50 | 0.9375 | 0.6250 |
| R@100 | 0.9375 | 0.7917 |
| **Mean Final** | 0.8375 | **0.5083** |
| tìm ra video | 15/16 | 19/24 |

0.5083 là con số trung thực đầu tiên. Vẫn cao hơn thực tế thi thật một chút — 5 câu (25, 8, 14, 18, 21) không tìm ra video ở bất kỳ hạng nào, và các câu này lại nằm trong nhóm "khó nhất" (chưa ai xác minh nổi ở vòng 1).

Kết luận tiếng Việt-vs-Anh **không đổi**, vì đó là so sánh có kiểm soát (cùng ground truth, chỉ đổi ngôn ngữ câu hỏi) — không phải chỗ vòng lặp xảy ra: Việt 0.0125, Anh giờ 0.5083 (thay vì 0.8375), vẫn là chênh lệch áp đảo.

### Mức frame — "có trúng đúng khoảnh khắc không"

| cửa sổ | R@1 | R@5 | Final |
|---|---|---|---|
| cả video | 0.2083 | 0.3333 | **0.5083** |
| ± 0 | 0.0417 | 0.1250 | 0.1083 |
| ± 25 | 0.0417 | 0.1667 | 0.1417 |
| ± 125 | 0.0417 | 0.1667 | 0.1583 |
| ± 500 | 0.0417 | 0.2083 | 0.2833 |

Vẫn cùng một mẫu hình như trước khi sửa: khoảng cách video→frame là chỗ mất điểm lớn nhất (0.51 → 0.11–0.28), và nới cửa sổ chỉ giúp chút ít cho tới ±500. Không phải do lưới keyframe thưa — lý do y hệt bản đo trước (xem lịch sử file này): keyframe BTC cách nhau trung vị 2,16s, phân bố lệch frame vẫn kiểu "trúng chính xác hoặc trượt xa", không có vùng giữa.

## Bộ này đo được gì, và KHÔNG đo được gì

Phải đọc kỹ mục này trước khi tin bất kỳ con số nào ở trên.

**n=25, phủ trọn vòng 1** — lệch chọn mẫu theo hướng "chỉ giữ câu dễ" đã hết, vì 8 câu khó nhất (chưa ai xác minh nổi bằng mắt) giờ đã có trong bộ nhờ đối chiếu sublan1. Nhưng còn hai giới hạn khác, phải nhớ:

- **Ground truth vẫn chỉ là "khả năng cao đúng", không phải BTC xác nhận.** 7 câu sửa và 8 câu thêm đều dựa trên phán đoán của một người xem ảnh, cộng với việc một đội khác (không phải BTC) tự tin vào đáp án của họ. Sai số dư vẫn có thể còn.
- **Mean Final 0.5083 vẫn cao hơn thực tế** một chút, vì 5 câu chưa tìm ra video ở bất kỳ hạng nào lại chính là các câu khó nhất — nếu có ground truth cho toàn bộ query vòng 1 kể cả những câu không ai từng thử xác minh, điểm thật sẽ còn thấp hơn.
- **Bắt tụt điểm vẫn rất tốt.** 5 câu đang ở hạng 1 (1, 4, 6, 10, 15). Một thay đổi làm chúng tụt xuống là hồi quy thật.
- **Chứng minh cải tiến giờ có chỗ để đo cả hai mức** — mức video còn nhiều dư địa (0.5083, không còn gần trần như 0.8375 cũ), mức frame cũng vậy (0.11–0.28).

Kết luận thực dụng không đổi: **dùng mức video để canh hồi quy, dùng mức frame để đo tiến bộ** — chỉ là con số mốc giờ trung thực hơn.

## Vì sao không tự chạy lại `build_round1_ground_truth.py` mà không cẩn thận

Script đó dựng ground truth từ `artifacts/submissions` — nguồn duy nhất nó biết là 17 câu gốc. Chạy lại với tham số mặc định sẽ **ghi đè cả 7 câu vừa sửa bằng đáp án cũ (đã biết sai)**, vì nó không biết `apply_sublan1_review.py` đã sửa gì.

Từ commit sửa lỗi này, script tự bảo vệ: nó đọc file ground truth hiện có, và **bỏ qua** bất kỳ câu nào có `verified_by` khác `"human"` — trừ khi truyền `--force`. Chạy bình thường thì an toàn; `--force` chỉ dùng khi thật sự muốn ghi đè một câu đã sửa.

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
