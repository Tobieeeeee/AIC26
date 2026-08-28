# Câu hỏi gửi mentor — hệ thống AIC26

Ngày soạn: 24/08/2026. Sau vòng 1, trước khi bắt tay vòng 2.

Nội dung gốc do **tobie** nêu; các con số kèm theo lấy từ phép đo trên 17 câu vòng 1 đã xác minh bằng mắt (xem [DO_LUONG_VONG_2.md](DO_LUONG_VONG_2.md)).

**Nếu mentor ít thời gian, hỏi nhóm 1, 2 và 7 trước.** Đó là ba nhóm có đòn bẩy lớn nhất lên điểm.

---

## Đối chiếu trước khi hỏi

Hai ví dụ tobie nêu đều nằm trong tập ground truth đã xác minh, nên hỏi được đích danh:

| tobie mô tả | Video thật | Nguồn |
|---|---|---|
| "video cô @Hồng Nhung dạy, cảnh im, người im, chỉ có text thay đổi" | **L25_V041** | câu p1-22-kis, áo dài hồng, dạy động từ *remember* |
| "video múa lân thay đổi cảnh liên tục" | **L24_V035** | câu p1-16-trake |

Và con số đáng giá nhất: trong 15 câu tìm ra đúng video, **L25_V041 đứng hạng 46 — tệ nhất**, trong khi phần lớn câu khác ở hạng 1. Trực giác của tobie về video tĩnh là **đúng, và đo được**.

---

## 1. Detector và selector keyframe

> Hệ thống bọn em đang dùng **TransNetV2** cho shot boundary, selector lấy keyframe cố định mỗi **8–12s**.
>
> Bọn em vừa đo keyframe **BTC phát sẵn** (177.321 ảnh / 873 video) để so, thì thấy BTC làm khác hẳn:
>
> ```
> trung vị      2,16 s   (55 frame)
> p75           4,40 s
> p95           6,00 s   ← trần cứng
> p5            0,04 s   ← dày đặc ngay chỗ cắt cảnh
> ```
>
> Tức BTC dùng **thích ứng, trung vị 2,16s, không bao giờ quá ~6s** — dày hơn mốc 8–12s của bọn em khoảng 4–5 lần.
>
> Hỏi:
>
> 1. Vì sao mốc 8–12s lại phổ biến, trong khi chính BTC ship keyframe cách nhau trung vị 2,16s? Có phải nó thuần tuý là ngân sách tính toán (số ảnh phải trích đặc trưng) chứ không dựa trên nội dung?
> 2. Selector cố định so với selector theo ngưỡng khác biệt hình ảnh (frame difference / embedding distance), có thêm trần cứng kiểu ~6s như BTC — cái nào các đội mạnh hay dùng?
> 3. Có detector nào bắt được **thay đổi nội dung trong cùng một shot** (text/slide đổi mà cảnh không đổi) không, hay việc đó phải giao hẳn cho OCR?
>
> Hai ca khó trong batch 1, để mentor hình dung:
>
> - **L25_V041** — cô giáo giảng bài, máy quay tĩnh, người gần như không cử động, **chỉ có text trên màn hình thay đổi**. Câu này bọn em truy xuất ra ở **hạng 46**, tệ nhất trong các câu tìm được.
> - **L24_V035** — múa lân, cảnh đổi liên tục. Ngược lại: rất nhiều boundary, số keyframe phình to.

**Lưu ý khi hỏi:** đừng đặt vấn đề thành "lưới keyframe của bọn em quá thưa nên trượt frame". Bọn em đã đo và **không phải vậy** — xem mục dưới. Nhóm câu hỏi này là để chuẩn bị cho vòng 2/3 khi đội tự trích keyframe từ data mới, chứ không phải để vá lỗi đang có.

## 2. Xin ground truth

> BTC **không công bố kết quả vòng 1**, nên bọn em đang không có thước đo. Bọn em tự dựng một bộ từ 17 câu đã xác minh bằng mắt, và vừa phát hiện ra một lỗi nghiêm trọng trong chính cách xác minh: người xác minh chạy tìm kiếm, thấy top kết quả có vẻ hợp lý, rồi chấp nhận làm "đáp án đúng" — tức **lấy chính hệ thống cần đo để tạo ra thước đo cho nó**.
>
> Đối chiếu với bài nộp một đội khác (họ tự nguyện chia sẻ), tụi em phát hiện 8/17 câu "đã xác minh" đó lệch hẳn **video**, không phải lệch frame. Dò lại thì **5/7 video bị bác đứng hạng 1** trong chính kết quả tìm kiếm hệ thống của tụi em — đúng như nghi ngờ: người xác minh vô tình chọn cái hệ thống tự đưa ra, không phải cái đúng thật. Sau khi sửa, Mean Final Score tụt từ 0.8375 xuống còn **0.5083** — con số cũ phần lớn là ảo giác.
>
> Hỏi:
>
> 1. Thầy/anh có **ground truth của các mùa trước (AIC 2023/2024/2025)** hoặc bộ query có đáp án nào bọn em dùng để benchmark độc lập, không tự chấm bằng chính hệ thống của mình không?
> 2. Cần khoảng bao nhiêu câu thì con số mới đáng tin?
> 3. Ngoài việc nhờ một nguồn độc lập đối chiếu (như tụi em vừa làm), có quy trình chuẩn nào để tránh vòng lặp tự xác nhận này khi không có ground truth chính thức không?

## 3. Lưu trữ thứ 3 (Cloudflare R2)

> Kiến trúc đề xuất của đội có ba tầng: **FAISS** (vector), **Elasticsearch** (BM25), và **R2** làm shared persistent storage, cộng Parquet cho metadata.
>
> Bọn em đang vướng đúng vấn đề R2 định giải: index 177.321 vector và file `keyframe_mapping.csv` **chỉ tồn tại trên máy một người**. Nhưng R2 tốn tiền và tốn công.
>
> Cập nhật 28/08: bọn em vừa sửa được một nửa vấn đề **mà R2 không giải nổi**. Index cũ nhúng đường dẫn tuyệt đối `E:\AIC26_data\...` vào từng dòng metadata, nên copy nguyên khối sang máy khác thì nó nạp được mà **không mở nổi một tấm ảnh nào** — đường dẫn hỏng nằm *bên trong* file mà R2 sẽ chuyển. Giờ đường dẫn lưu tương đối cộng một gốc cấu hình được, đã dựng lại và kiểm chứng: file FAISS giống bản cũ từng byte, điểm vẫn đúng 0,5083. Nửa còn lại — **đưa 363 MB đó tới máy người khác** — vẫn là việc của R2.
>
> Hỏi:
>
> 1. R2 có thật sự cần cho quy mô một đội 5 người không, hay chỉ cần một ổ cứng ngoài luân chuyển + Google Drive là đủ?
> 2. Nếu cần thì **dùng ở khâu nào và lưu cái gì** — raw video của BTC, keyframe đã trích, embedding/index đã dựng, hay chỉ metadata Parquet?
> 3. Cái gì **không nên** đưa lên R2 vì tải xuống còn lâu hơn tự sinh lại?

## 4. Quy trình ghép bài lúc thi

> Giả sử đề 100 câu, đội chia mỗi người 20 câu, **mỗi người một máy riêng**. Định dạng nộp là mỗi câu một file CSV, tên file theo tên file đề.
>
> Rủi ro bọn em đã gặp thật ở vòng 1: **12/25 file bị lỗi định dạng** — 5 file dính BOM do mở bằng Excel, 4 file thừa khoảng trắng quanh dấu phẩy, và 1 file có **72/99 dòng trùng nhau** do ghép tay. Nhìn bằng mắt thì file trông hoàn toàn bình thường.
>
> Hỏi:
>
> 1. Quy trình ghép mà các đội có kinh nghiệm hay dùng là gì — mỗi người tự xuất file rồi gom, hay **một máy làm server trung tâm**, mọi người đẩy kết quả về đó và chỉ máy đó được ghi file nộp?
> 2. Nếu gom file thì gom qua đâu cho nhanh và không phụ thuộc mạng — LAN, USB, hay Git?
> 3. Có nên có người chuyên trách **chỉ làm việc kiểm định dạng và nộp**, không tham gia tìm kiếm?

## 5. Mạng không ổn định ở vòng chung kết

> Chung kết thi tập trung ở hội trường, mạng nhiều khả năng không ổn định.
>
> Hỏi:
>
> 1. Nên thiết kế hệ thống **chạy hoàn toàn offline trên từng máy** (mỗi máy một bản index đầy đủ), hay **một máy chủ + các máy khác làm client qua LAN**? Đánh đổi ở đâu?
> 2. Nếu chọn client–server qua LAN thì cần chuẩn bị gì trước — switch/router riêng, cáp mạng, IP tĩnh?
> 3. Những thứ nào bắt buộc phải cache sẵn trước khi vào phòng thi (model weight, index, keyframe ảnh)? Dung lượng thực tế cần bao nhiêu mỗi máy?
> 4. Nộp bài lên `sotuyenaic.oj.io.vn` **có cần internet không**, và nếu mạng rớt đúng lúc nộp thì phương án dự phòng là gì?

## 6. FastAPI và latency

> Backend bọn em là FastAPI, các endpoint `/search/semantic`, `/search/kis`, `/search/qna`, `/search/trake`, `/search/hybrid`, `/health`.
>
> Số đo hiện tại: **khởi động ~40 giây** (nạp FAISS index 4s + CLIP text encoder 30s + mapping 2s), sau khi ấm thì **mỗi truy vấn 0,2–2 giây**.
>
> Hỏi:
>
> 1. Model nên nạp ở **lifespan/startup** hay lazy lần gọi đầu? Lúc thi thì 40 giây khởi động một lần có chấp nhận được không?
> 2. Inference CLIP là **CPU-bound và blocking**. Endpoint nên để `def` (FastAPI đẩy vào threadpool) hay `async def` + `run_in_executor`? Bọn em đang dùng `def`, có đúng không?
> 3. Có nên **batch nhiều truy vấn** vào một lần encode khi nhiều người cùng tìm không, hay độ phức tạp không đáng?
> 4. Cache ở tầng nào là hiệu quả nhất — cache text embedding theo câu truy vấn, hay cache cả kết quả tìm kiếm?
> 5. **Schema response cuối cùng nên có những trường gì** để UI không phải gọi thêm lần thứ hai? Hiện bọn em trả `video_id`, `frame_idx`, `score`, `image_path` — thiếu gì không (timestamp, shot_id, đoạn OCR/ASR khớp, link ảnh thumbnail)?

## 7. Rào cản ngôn ngữ

Nhóm này tobie chưa nêu, nhưng theo số đo thì nó là câu hỏi có đòn bẩy lớn nhất.

> Encoder bọn em là OpenCLIP `ViT-B-32-quickgelu/openai`, **chỉ hiểu tiếng Anh**. Đo trên 24 câu: truy vấn tiếng Việt được **0.0333**, tiếng Anh **0.5083** — tìm ra đúng video 2/24 so với 19/24. Máy dịch tự động (`opus-mt-vi-en`) cũng ra 0.0000 vì dịch sai danh từ riêng.
>
> Hiện đội đang **viết tay câu mô tả tiếng Anh** cho từng truy vấn, rất tốn thời gian lúc thi.
>
> Hỏi: các đội khác xử lý rào cản ngôn ngữ này thế nào — dùng model đa ngữ (multilingual CLIP, SigLIP, jina-clip), hay dùng LLM dịch + viết lại thành mô tả thị giác? Cái nào đáng đầu tư hơn cho vòng 2?

---

## Bối cảnh: chỗ hệ thống thật sự đang trượt

Đọc mục này trước khi hỏi nhóm 1, để không mô tả sai vấn đề cho mentor.

Bọn em tìm **đúng video 0.5083** nhưng **đúng frame chỉ 0.11–0.28**. Câu hỏi tự nhiên là "lưới keyframe thưa quá à?" — đã đo, và **không phải**.

Độ lệch giữa frame trả về và frame đã xác minh, trên 15 câu tìm đúng video:

```
0, 0, 0, 0, 0, 0, 0, 0, 15  |  300, 475, 675, 1046, 1080, 1481
```

Phân bố **hai cụm, rỗng hoàn toàn ở giữa**. Trúng thì trúng chính xác tuyệt đối; trượt thì trượt 300–1500 frame, tức **12–60 giây** — một khoảnh khắc khác hẳn trong cùng video. Không hề có kiểu lỗi "lệch sang keyframe kế bên".

Cộng với việc keyframe BTC đã cách nhau trung vị 2,16s, kết luận là: **trích keyframe dày hơn sẽ không cứu được các câu đang hỏng.** Chúng không thiếu keyframe. Chúng thiếu khả năng phân biệt hai khoảnh khắc trong cùng một video — việc của OCR, ASR và re-rank theo thời gian.

Nên khi hỏi mentor, nhóm 1 là **chuẩn bị cho vòng 2/3** (tự trích keyframe từ data mới), còn thứ cần hỏi để vá lỗi hiện tại nằm ở nhóm 7 và ở OCR/ASR.

## Nguồn các con số

| Con số | Đo ở đâu |
|---|---|
| 0.5083 video / 0.11–0.28 frame | `scripts/score_verified_set.py`, n=24 (sau khi sửa bằng sublan1) |
| L25_V041 hạng 46 | cùng phép đo, bảng chi tiết từng câu |
| Việt 0.0333 so với Anh 0.5083 | `--compare` cùng script |
| 5/7 video bị bác đứng hạng 1, vòng lặp tự xác nhận | đối chiếu `artifacts/sublan1`, xem `docs/DO_LUONG_VONG_2.md` |
| 12/25 file lỗi định dạng | `mvr.submission.lint`, quét `artifacts/submissions/` |
| 40s khởi động, 0,2–2s mỗi truy vấn | `scripts/answer_queries.py`, đo lúc chạy vòng 1 |
| 177.321 vector, 873 video | manifest của `artifacts/indexes/qna_organizer_all` |
| keyframe BTC cách nhau trung vị 2,16s / p95 6,00s | cột `timestamp` của `artifacts/keyframe_mapping.csv`, 176.448 khoảng |
| phân bố lệch frame hai cụm (0…15 rồi 300…1481) | truy xuất lại 16 câu, lấy frame gần nhất trong đúng video |
