# Tách luồng truy vấn: tiếng Việt cho BM25, tiếng Anh cho dense

Ngày: 2026-08-21
Trạng thái: đã duyệt design, chưa triển khai

## 1. Vấn đề

Dữ liệu OCR/ASR của corpus là **tiếng Việt**. Pipeline hiện tại dịch mọi truy vấn
sang tiếng Anh trước khi đưa vào cả hai nhánh tìm kiếm, nên nhánh BM25 khớp một
truy vấn tiếng Anh với một index tiếng Việt.

Điểm ghép nằm ở một chỗ duy nhất — `ElasticsearchMultimodalService.search_hybrid`
(`src/mvr/api/app.py:229`) và `QnaFusedRetriever.search_hybrid`
(`src/mvr/retrieval/qna_fusion.py:88`) — cả hai nhận **một** chuỗi truy vấn và
chuyển nguyên chuỗi đó cho cả nhánh dense lẫn nhánh lexical.

Nhánh dense đúng là cần tiếng Anh: encoder là OpenCLIP `RN50/openai`
(`src/mvr/retrieval/semantic.py:11-12`), tiếng Anh thuần, ngữ cảnh 77 token.

## 2. Khảo sát: hai index Elasticsearch, không tương thích

| | Index A — `keyframes` | Index B — Task05B |
|---|---|---|
| Định nghĩa | `src/mvr/indexing/elasticsearch_index.py` | `src/mvr/indexing/lexical_index.py` |
| Trường text | `caption`, `ocr_text` | `ocr_text`, `asr_text` |
| Trường vector | `clip_vector` (dense_vector, cosine) | **không có** |
| Analyzer | `vi_folded` (standard + lowercase + asciifolding) | mặc định `standard` |
| `_id` | `frame_id` | `video_id/keyframe_id` |
| Builder production | **không có** — chỉ xuất hiện trong test | `scripts/indexing/build_lexical_index.py` |

Hai kết luận cưỡng chế thiết kế:

1. **Index B không có trường vector**, nên `ElasticsearchSemanticRetriever` (kNN
   trên `clip_vector`) không chạy được trên B. Sau khi tách, hai nhánh buộc phải
   nằm ở hai backend khác nhau.
2. **Index A không có builder production.** `ElasticsearchIndex` và
   `build_search_document` chỉ được gọi từ `tests/`. Nó là schema Lab 6 để lại.

Đường duy nhất có dữ liệu ở cả hai đầu: **dense = FAISS, lexical = index B.**

Milvus không tồn tại trong repo (`grep -rn -i milvus src/ scripts/ tests/` → 0 kết
quả). Trong bản mô tả tác vụ, "Milvus" chỉ nhánh vector nói chung; hiện thực là FAISS.

## 3. Blocker: hợp đồng `_id` của Task05B đã vỡ

`src/mvr/indexing/lexical_index.py` ghi `_id = f"{video_id}/{keyframe_id}"`.
`src/mvr/retrieval/lexical_bm25.py:106` vẫn đòi `_id == keyframe_id`:

```python
if raw_hit.get("_id") != keyframe_id:
    raise RuntimeError("malformed search response: _id/keyframe_id mismatch")
```

Kiểm chứng — đưa đúng thứ `build_lexical_index` ghi vào ES cho hàm đọc:

```
_id thực tế trong index = L21_V001/034
-> RuntimeError: malformed search response: _id/keyframe_id mismatch
```

**Mọi hit từ một index B dựng đúng đều raise.** `QnaFusedRetriever` nuốt exception
này thành `logger.warning("lexical branch unavailable")` (`qna_fusion.py:96-98`),
nên nhánh BM25 trả rỗng vĩnh viễn mà không có lỗi nào nổi lên.

Nguyên nhân không ai bắt được: hai phía của cùng một hợp đồng được test bằng hai
định dạng `_id` khác nhau, và cả hai đều pass.

```
tests/unit/indexing/test_task05b.py:153         "_id": "L21_V010/frame-a"   ← phía ghi
tests/unit/retrieval/test_task05b_lexical.py:19 "_id": keyframe_id          ← phía đọc
```

Lỗi này do PR #8 gây ra: phía ghi được sửa, phía đọc không được truy theo.

## 4. Thiết kế

### 4.1 Mục 0 — Sửa hợp đồng `_id`

`map_task05b_lexical_response` kiểm tra `_id == f"{video_id}/{keyframe_id}"`.

Thêm test nối hai phía: fixture phía đọc lấy `_id` từ
`LexicalIndexDocument.document_id` thật, thay vì mỗi bên tự bịa một chuỗi. Đây là
loại test lẽ ra phải có ở PR #8.

Không có mục này thì việc tách query là vô nghĩa — nhánh tiếng Việt vẫn trả rỗng.

### 4.2 Mục 1 — Tách query, không đổi chữ ký công khai

`QnaFusedRetriever` → `FusedRetriever` (giữ alias tên cũ), thêm tham số tùy chọn:

```python
def search_hybrid(self, query, *, top_k, rrf_k, lexical_query=None):
    dense_text   = query                    # En, cho OpenCLIP RN50
    lexical_text = lexical_query or query   # Vi, cho BM25
```

`lexical_query=None` cho hành vi y hệt hiện tại. Protocol `_MultimodalService` ở
`retrieval/kis.py:25` và `retrieval/qna.py:61` vẫn khớp; không file nào của track
khác vỡ.

`search_kis` nhận thêm `lexical_query: str | None = None` và **chỉ chuyển tiếp khi
khác `None`**. Nếu truyền vô điều kiện, một service chưa nhận tham số này —
`ElasticsearchMultimodalService` chẳng hạn — sẽ `TypeError`. Vì lý do đó, thay đổi ở
`search_kis` thuộc đợt 2 cùng với `FusedRetriever`, không tách ra đợt 1 được.

`ElasticsearchMultimodalService` **không đụng tới** — nó phục vụ index A vốn không
có dữ liệu; giữ nguyên cho route `/search/hybrid` cũ.

### 4.3 Mục 2 — Analyzer tiếng Việt cho index B

Thêm settings analyzer `vi_folded` (standard + lowercase + asciifolding) vào index
B, copy nguyên từ `elasticsearch_index.py:11-21`. Nâng `TASK05B_SCHEMA_VERSION` lên
`"1.1"` để index cũ không bị nhầm là tương thích. Cần reindex.

Không có mục này, tiếng Việt vào được index nhưng mất phần lớn lợi ích: `"Hà Nội"`
không khớp `"Ha Noi"`, trong khi OCR tiếng Việt sai dấu rất nhiều.

### 4.4 Mục 3 — Bật nhánh lexical cho KIS

`scripts/verification/run_real_data_kis_smoke.py:105` đang truyền
`multimodal_service=None`: KIS baseline hôm nay là FAISS đơn thuần, nhánh lexical
chưa từng được bật trong đo đạc KIS. Đổi sang `FusedRetriever` khi có `MVR_ES_URL`,
giữ `None` khi không, để vẫn chạy được trên máy không có Elasticsearch.

### 4.5 Mục 4 — Test

- Round-trip `_id`: writer → reader, dùng object thật ở cả hai đầu.
- `lexical_query` khác `query`: khẳng định hai nhánh nhận hai chuỗi khác nhau.
- Bỏ trống `lexical_query`: hai nhánh nhận cùng chuỗi (không hồi quy Q&A).
- Analyzer: `"Hà Nội"` và `"Ha Noi"` cùng khớp một document.

Mỗi test được kiểm chứng bằng cách stash bản sửa và xác nhận nó **fail** trên code cũ.

## 5. Phụ thuộc giữa các nhánh git

| File | Có trên `origin/main`? |
|---|---|
| `src/mvr/retrieval/lexical_bm25.py` | có |
| `src/mvr/indexing/lexical_index.py` | có |
| `src/mvr/retrieval/kis.py` | có |
| `scripts/verification/run_real_data_kis_smoke.py` | có |
| `src/mvr/retrieval/qna_fusion.py` | **không** — chỉ có trên nhánh Q&A |

Hệ quả: Mục 0 và Mục 2 làm được ngay trên nhánh tách từ `origin/main`. Mục 1 và Mục 3
phụ thuộc `qna_fusion.py`, tức phải chờ PR Q&A merge.

Vì vậy chia hai đợt:

- **Đợt 1** (không phụ thuộc ai): Mục 0 + Mục 2 + phần test tương ứng.
- **Đợt 2** (sau khi PR Q&A merge): Mục 1 + Mục 3.

Mục 0 sửa một lỗi đang làm hỏng nhánh lexical của Task05B trên `main`, ảnh hưởng cả
track Q&A lẫn KIS, nên đợt 1 nên đi thành PR riêng và nhỏ.

## 6. Giai đoạn C — dựng lại bộ đo

Ba file của bộ đo ablation rỗng 0 byte trong **mọi** commit đã đụng tới chúng
(`8f4e32a`, `6456027`, `0040a9b`):

```
scripts/verification/query_retrieval_ablation/metrics.py   0 byte
scripts/verification/run_tkis_query_retrieval_ablation.py  0 byte
tests/unit/test_tkis_query_retrieval_ablation.py           0 byte
```

`runner.py:9` import `evaluate_ablation_variant` từ module rỗng, nên bộ đo chưa từng
chạy được:

```
ImportError: cannot import name 'evaluate_ablation_variant'
```

Giai đoạn C dựng lại ba file đó và thêm biến thể thứ năm `vi_lexical_en_dense` vào
`runner.py` để so trực tiếp với `en_full` hiện có. C cần một Elasticsearch chạy thật,
nên phụ thuộc đồng đội (xem T3 trong `docs/AIC_QNA_TEAM_TASKS.md`).

Trước C, chưa có số liệu nào chứng minh việc tách cải thiện điểm. A được làm vì nó
sửa lỗi và rẻ, không phải vì đã đo.

## 7. Ngoài phạm vi

- Hợp nhất index A và B thành một schema.
- Thay OpenCLIP RN50 bằng encoder đa ngữ (MetaCLIP-2 đa ngữ có thể khiến bước dịch
  thành thừa) — cần đo, và đụng track multi-embedding.
- Lỗi định danh còn sống ở `ElasticsearchMultimodalService` (fuse trên `frame_id`
  đơn, `app.py:256`) và `KeyframeFrameIndexResolver` (`kis.py:44-52`). Cùng lớp lỗi
  PR #8 đã sửa nơi khác, nhưng đường ống đó không có dữ liệu nên chưa gây hại.
