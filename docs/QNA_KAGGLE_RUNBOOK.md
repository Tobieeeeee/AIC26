# Q&A trên Kaggle — Hướng dẫn chạy

Sinh câu trả lời Q&A bằng GPU miễn phí của Kaggle, trong khi retrieval,
OCR và toàn bộ dữ liệu video vẫn nằm ở máy local.

**Chỉ có text rời khỏi máy.** File `prompts.json` chứa câu hỏi + text OCR +
object counts của các frame ứng viên (vài trăm KB). Không upload video,
không upload keyframe, không upload repo.

---

## Vì sao phải làm vậy

Máy phát triển có 3.8GB RAM, không GPU. Đo thực tế:

| Model | Kết quả trên máy local |
|---|---|
| Qwen2.5-1.5B | Không load được (lỗi paging file) |
| Qwen2.5-0.5B | Chạy được nhưng **lặp lại câu hỏi**, bịa đáp án |

Kaggle cho T4 x2 (2×16GB) — đủ chạy model 7B, gấp ~14 lần.

---

## Bước 1 — Local: xuất prompt

```powershell
py -3.12 -m pip install -e ".[dev,api,semantic,ocr]"

py -3.12 scripts\run_qna_pipeline.py `
  --query-file "D:\AIC_query\query-p1-groupA\query-p1-15-qa.txt" `
  --index      "artifacts\indexes\qna_l21_l22" `
  --dataset-root "D:\AIC26_test_root" `
  --top-k 20 `
  --export-prompts "artifacts\qna\query-p1-15-qa.prompts.json"
```

Không có `--export-prompts` thì script sẽ cố chạy model local. Có cờ này
thì **không model nào được load** — chỉ thu prompt.

Thêm `--min-object-confidence 0.4` nếu đọc file objects thô của BTC (chưa
lọc). Bỏ `--no-ocr` để **bật OCR** — đây là nguồn bằng chứng chính.

Kiểm tra file sinh ra:

```powershell
py -3.12 -c "import json; b=json.load(open(r'artifacts\qna\query-p1-15-qa.prompts.json',encoding='utf-8')); print(b['query_id'], len(b['items']), 'prompts')"
```

---

## Bước 2 — Kaggle: thiết lập

1. Vào <https://www.kaggle.com> → **Create → New Notebook**

2. **Bật GPU**: panel bên phải → **Session options** → **Accelerator** →
   chọn **GPU T4 x2**
   *(Lần đầu dùng GPU cần xác minh số điện thoại.)*

3. **Bật Internet**: cùng panel → **Internet** → **On**
   *(Cần để tải model từ HuggingFace. Nếu không muốn bật, xem Phụ lục A.)*

4. **Upload prompts.json**: panel phải → **Input → Upload → New Dataset**
   - Kéo file `query-p1-15-qa.prompts.json` vào
   - Đặt tên dataset, ví dụ `aic-qna-prompts`
   - **Create**
   - File sẽ nằm ở `/kaggle/input/aic-qna-prompts/query-p1-15-qa.prompts.json`

---

## Bước 3 — Kaggle: dán code

**Cell 1** — dán toàn bộ nội dung `scripts/kaggle_qna_generate.py`
(trừ khối `if __name__ == "__main__":` ở cuối, không cần).

**Cell 2** — chạy:

```python
run_batch(
    prompts_path="/kaggle/input/aic-qna-prompts/query-p1-15-qa.prompts.json",
    answers_path="/kaggle/working/query-p1-15-qa.answers.json",
    model_name="Qwen/Qwen2.5-7B-Instruct",
)
```

Đầu ra in từng đáp án khi sinh xong, ví dụ:

```
query_id : query-p1-15-qa
question : Hỏi xã này có tên là gì? (tại thời điểm đó)
prompts  : 20
[1/20] {"answer": "Ninh Hòa", "source": "context-grounded"}
[2/20] {"answer": "Vạn Ninh", "source": "context-grounded"}
...
done in 214s -> /kaggle/working/query-p1-15-qa.answers.json
```

### Chọn model theo GPU

| GPU đang có | Cấu hình | Cách gọi |
|---|---|---|
| **T4 x2** (khuyến nghị) | 7B bf16 | mặc định như trên |
| T4 x1 / P100 | 7B 4-bit | thêm `load_in_4bit=True` |
| Muốn chất lượng cao hơn | 14B 4-bit | `model_name="Qwen/Qwen2.5-14B-Instruct", load_in_4bit=True` |

Nếu gặp `CUDA out of memory` → thêm `load_in_4bit=True`, hoặc đổi sang
`Qwen/Qwen2.5-3B-Instruct`.

---

## Bước 4 — Kaggle: chạy nền và tải kết quả

**Quan trọng:** phiên interactive của Kaggle **tự ngắt sau ~20 phút không
thao tác**. Với batch dài, dùng chế độ chạy nền:

- Góc trên phải → **Save Version** → chọn **Save & Run All (Commit)** →
  **Save**
- Notebook chạy headless (tối đa 12h), bạn có thể đóng trình duyệt
- Xong thì vào tab **Output** của version đó → tải `answers.json` về

Nếu batch ngắn (<15 phút) thì chạy interactive rồi tải trực tiếp từ panel
**Output → /kaggle/working** cũng được.

Đặt file tải về vào `artifacts\qna\`.

---

## Bước 5 — Local: nhập kết quả và xuất bài nộp

```powershell
py -3.12 scripts\import_qna_answers.py `
  --prompts "artifacts\qna\query-p1-15-qa.prompts.json" `
  --answers "artifacts\qna\query-p1-15-qa.answers.json" `
  --output  "artifacts\qna\query-p1-15-qa.predictions.csv"
```

Bước này **không cần** index, OCR, model hay GPU — chạy trong vài giây,
vì `prompts.json` đã mang sẵn mọi trường của retrieval.

Kết quả:

```
[query_id] query-p1-15-qa
[question] Hỏi xã này có tên là gì? (tại thời điểm đó)
#1 L21_V018 frame=8208 score=0.4106 answer='Ninh Hòa' source=context-grounded
...
wrote 20 validated predictions to artifacts\qna\query-p1-15-qa.predictions.csv
```

Đáp án nào model trả về sai định dạng JSON sẽ bị **loại bỏ**, không bịa
thay thế — script báo rõ số lượng bị loại.

---

## Hạn mức Kaggle cần biết

| | |
|---|---|
| GPU quota | **30 giờ/tuần** |
| Phiên tối đa | 12 giờ |
| Idle timeout (interactive) | **~20 phút** → dùng Save & Run All |
| `/kaggle/working` | 20GB, giữ lại sau khi commit |

Ước lượng: ~10s/prompt với 7B trên T4. 20 prompt ≈ 3-4 phút. Cả 9 câu Q&A
với top-k 20 ≈ 30 phút. Thoải mái trong quota.

---

## Phụ lục A — Không muốn bật Internet

Kaggle có sẵn nhiều model trong mục **Models**:

1. Panel phải → **Input → Add Input → Models** → tìm `Qwen2.5`
2. Thêm vào notebook
3. Trỏ `model_name` tới đường dẫn local, ví dụ:
   `/kaggle/input/qwen2.5/transformers/7b-instruct/1`

Cách này không cần internet, và tải nhanh hơn.

---

## Phụ lục B — Chạy nhiều câu một lượt

Gộp mọi `prompts.json` vào cùng một dataset, rồi **nạp model một lần** và
lặp qua tất cả (nạp model 7B mất vài phút, không nên lặp lại mỗi file):

```python
from pathlib import Path

loaded = load_model("Qwen/Qwen2.5-7B-Instruct")   # nạp DUY NHẤT 1 lần

for p in sorted(Path("/kaggle/input/aic-qna-prompts").glob("*.prompts.json")):
    run_batch(
        prompts_path=p,
        answers_path=f"/kaggle/working/{p.name.replace('.prompts', '.answers')}",
        loaded=loaded,
    )
```

Xong sẽ có đủ số file `.answers.json` trong `/kaggle/working/`. Tải cả
thư mục về (nút **Download** ở panel Output) rồi giải nén vào
`artifacts\qna\`.

---

## Lưu ý về dữ liệu

Prompt gửi lên Kaggle chứa text OCR và object counts của frame. Dữ liệu
AIC batch 1 là dataset công khai (AIC 2025), rủi ro thấp — nhưng nếu đội
có ràng buộc riêng về việc gửi dữ liệu ra dịch vụ ngoài thì cần xác nhận
trước.
