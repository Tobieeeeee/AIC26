# Tách luồng truy vấn Vi/En — Implementation Plan (Phase A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nhánh BM25 nhận truy vấn tiếng Việt gốc, nhánh dense nhận truy vấn đã dịch sang tiếng Anh, và index Task05B khớp được tiếng Việt không dấu.

**Architecture:** Dense chạy trên FAISS (OpenCLIP RN50, tiếng Anh), lexical chạy trên index Elasticsearch Task05B (BM25, tiếng Việt). Hai nhánh gặp nhau ở `FusedRetriever.search_hybrid`, nhận hai chuỗi truy vấn thay vì một. Trước đó phải sửa hợp đồng `_id` đang làm nhánh lexical trả rỗng.

**Tech Stack:** Python 3.12, pytest, Elasticsearch 8.x client, FAISS, OpenCLIP.

**Spec:** `docs/superpowers/specs/2026-08-21-tkis-query-branch-split-design.md`

## Global Constraints

- Nhánh làm việc: `feat/tkis-query-branch-split`, tách từ `origin/main` (`6198a08`).
- Định danh keyframe luôn là cặp `(video_id, keyframe_id)`. `keyframe_id` một mình **không** định danh keyframe: 166.550 keyframe của BTC chỉ có 632 tên phân biệt, `"034"` xuất hiện ở 842 video.
- `_id` của Elasticsearch index Task05B là `f"{video_id}/{keyframe_id}"`, do `LexicalIndexDocument.document_id` sinh ra. Đây là nguồn sự thật duy nhất; không viết lại chuỗi này bằng tay ở nơi khác.
- Commit **không** được có trailer `Co-Authored-By`. Author duy nhất là `ipvfml <256519431+ipvfml@users.noreply.github.com>`.
- Máy phát triển không có GPU, không có Docker, không có Elasticsearch chạy thật. Mọi test bắt buộc phải chạy được offline; test cần Elasticsearch thật phải `skipif` theo biến môi trường.
- Chạy test bằng `./.venv/Scripts/python.exe -m pytest` từ `D:\AIC26`. Python toàn cục **không** có pytest.
- Baseline trước khi bắt đầu: `737 passed, 1 skipped` trên `origin/main` = `1b15f36`.

## Phạm vi

Kế hoạch này chỉ phủ **Phase A** của spec (mục 4.1 → 4.5). Giai đoạn C (dựng lại bộ
đo ablation, mục 6 của spec) cần một Elasticsearch chạy thật nên sẽ có kế hoạch riêng
khi đồng đội dựng xong T3.

Cả 5 task đều chạy được: PR Q&A đã merge (`origin/main` = `1b15f36`), nên
`src/mvr/retrieval/qna_fusion.py` đã có trên `main` và nhánh làm việc đã rebase lên đó.

Thứ tự khuyến nghị: Task 1 → Task 2 → Task 3 → Task 4 → Task 5. Task 1 và Task 2 độc
lập với nhau; Task 4 cần Task 3, Task 5 cần cả hai.

---

### Task 1: Sửa hợp đồng `_id` giữa phía ghi và phía đọc Task05B

Phía ghi (`LexicalIndexDocument.document_id`) sinh `"L21_V001/034"`. Phía đọc
(`map_task05b_lexical_response`) vẫn đòi `_id == keyframe_id`, nên **mọi hit từ một
index dựng đúng đều raise**. `QnaFusedRetriever` nuốt exception này thành một dòng
warning, nên nhánh BM25 trả rỗng mà không có lỗi nào nổi lên.

Hai phía được test bằng hai định dạng `_id` khác nhau và cả hai đều pass — nên test
mới phải lấy `_id` từ chính object phía ghi, không tự bịa chuỗi.

**Files:**
- Modify: `src/mvr/retrieval/lexical_bm25.py:104-107`
- Modify: `tests/unit/retrieval/test_task05b_lexical.py:13-28` (hàm `_hit`)
- Test: `tests/unit/retrieval/test_task05b_lexical.py`

**Interfaces:**
- Consumes: `mvr.indexing.lexical_index.LexicalIndexDocument` (đã tồn tại; thuộc tính `document_id`, phương thức `to_source()`).
- Produces: `map_task05b_lexical_response` chấp nhận `_id == f"{video_id}/{keyframe_id}"` và từ chối mọi định dạng khác. Chữ ký không đổi.

- [ ] **Step 1: Viết test round-trip nối phía ghi với phía đọc**

Thêm vào cuối `tests/unit/retrieval/test_task05b_lexical.py`:

```python
def test_reader_accepts_the_id_the_writer_actually_produces() -> None:
    """Phía ghi và phía đọc phải dùng chung một định dạng `_id`.

    Trước đây mỗi phía được test bằng một chuỗi tự bịa -- phía ghi khẳng định
    "L21_V010/frame-a", phía đọc dựng fixture với `_id` là keyframe_id trần --
    nên cả hai pass trong khi ở production chúng không khớp nhau. Test này lấy
    `_id` từ chính object phía ghi, nên không thể lệch lần nữa.
    """
    from mvr.indexing.lexical_index import LexicalIndexDocument

    document = LexicalIndexDocument(
        keyframe_id="034",
        video_id="L21_V001",
        frame_idx=1745,
        shot_id=7,
        timestamp=69.8,
        ocr_text="công thức nấu ăn",
        asr_text="",
    )
    response = {
        "hits": {
            "hits": [
                {
                    "_id": document.document_id,
                    "_score": 12.5,
                    "_source": document.to_source(),
                }
            ]
        }
    }

    results = map_task05b_lexical_response(response, top_k=5)

    assert len(results) == 1
    assert (results[0].video_id, results[0].keyframe_id) == ("L21_V001", "034")
    assert results[0].frame_idx == 1745


def test_bare_keyframe_id_as_elasticsearch_id_is_rejected() -> None:
    """`_id` trần là dấu hiệu index được dựng bằng code trước PR #8, nơi "034"
    của 842 video đè lên nhau còn đúng một document. Đọc nó như dữ liệu hợp lệ
    sẽ giấu mất việc index đã mất 99,6% số dòng."""
    response = {
        "hits": {
            "hits": [
                {
                    "_id": "034",
                    "_score": 1.0,
                    "_source": {
                        "keyframe_id": "034",
                        "video_id": "L21_V001",
                        "frame_idx": 1745,
                        "shot_id": 7,
                        "timestamp": 69.8,
                    },
                }
            ]
        }
    }

    with pytest.raises(RuntimeError, match="_id/keyframe_id mismatch"):
        map_task05b_lexical_response(response, top_k=5)
```

- [ ] **Step 2: Chạy test để xác nhận nó fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/retrieval/test_task05b_lexical.py::test_reader_accepts_the_id_the_writer_actually_produces -v`

Expected: FAIL với `RuntimeError: malformed search response: _id/keyframe_id mismatch`.

Test thứ hai (`test_bare_keyframe_id_as_elasticsearch_id_is_rejected`) sẽ **pass** ngay
ở bước này — đó là điều đúng: nó tồn tại để giữ hành vi sau khi sửa, không phải để bắt lỗi.

- [ ] **Step 3: Sửa phía đọc**

Trong `src/mvr/retrieval/lexical_bm25.py`, thay đoạn:

```python
        keyframe_id = _source_string(source, "keyframe_id")
        if raw_hit.get("_id") != keyframe_id:
            raise RuntimeError("malformed search response: _id/keyframe_id mismatch")
```

bằng:

```python
        keyframe_id = _source_string(source, "keyframe_id")
        video_id = _source_string(source, "video_id")
        # `_id` là cặp video/keyframe, không phải keyframe trần -- xem
        # `LexicalIndexDocument.document_id`, phía sinh ra nó. Một `_id` trần
        # nghĩa là index được dựng bằng code trước PR #8, nơi "034" của 842
        # video cùng ghi vào một document; đọc tiếp sẽ giấu mất mất mát đó.
        if raw_hit.get("_id") != f"{video_id}/{keyframe_id}":
            raise RuntimeError("malformed search response: _id/keyframe_id mismatch")
```

Ở dưới, trong `mapped.append(...)`, thay `_source_string(source, "video_id")` bằng
biến `video_id` vừa có để không đọc hai lần:

```python
        mapped.append(
            (
                score,
                keyframe_id,
                video_id,
                int(_source_int(source, "frame_idx")),
                _source_int(source, "shot_id", nullable=True),
                float(timestamp_value),
            )
        )
```

- [ ] **Step 4: Sửa fixture `_hit` trong test cũ**

`_hit` đang sinh `_id` trần nên 5 test hiện có sẽ fail sau Step 3. Chúng khẳng định
lỗi như thể đó là ý đồ. Sửa hàm `_hit`:

```python
def _hit(
    keyframe_id: str,
    *,
    score: float,
    frame_idx: int = 10,
    video_id: str = "video-a",
) -> dict:
    return {
        # Đúng định dạng `LexicalIndexDocument.document_id` sinh ra.
        "_id": f"{video_id}/{keyframe_id}",
        "_score": score,
        "_source": {
            "keyframe_id": keyframe_id,
            "video_id": video_id,
            "frame_idx": frame_idx,
            "shot_id": 1,
            "timestamp": 0.4,
        },
    }
```

- [ ] **Step 5: Chạy lại toàn bộ file test**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/retrieval/test_task05b_lexical.py -v`

Expected: tất cả PASS, gồm hai test mới.

- [ ] **Step 6: Chạy toàn bộ test suite để bắt hồi quy**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit -q`

Expected: không có test nào fail thêm so với trước khi sửa. Nếu có, đọc kỹ trước khi
nới lỏng bất cứ điều gì — một test fail ở đây nhiều khả năng là một fixture khác cũng
đang bịa `_id`.

- [ ] **Step 7: Kiểm chứng test thật sự bắt được lỗi**

```bash
git stash
./.venv/Scripts/python.exe -m pytest tests/unit/retrieval/test_task05b_lexical.py::test_reader_accepts_the_id_the_writer_actually_produces -v
git stash pop
```

Expected: FAIL khi đã stash. Nếu nó pass, test không kiểm tra thứ ta nghĩ.

- [ ] **Step 8: Commit**

```bash
git add src/mvr/retrieval/lexical_bm25.py tests/unit/retrieval/test_task05b_lexical.py
git commit -m "fix: read the Task05B _id the writer actually produces

PR #8 changed the index _id to video_id/keyframe_id but left the reader
demanding a bare keyframe_id, so every hit from a correctly built index
raised. QnaFusedRetriever swallows that into a warning, so the lexical
branch returned nothing with no error surfacing anywhere.

Both sides were tested against a different _id format and both passed;
the new test takes its _id from LexicalIndexDocument itself."
```

---

### Task 2: Analyzer tiếng Việt cho index Task05B

Index Task05B dùng analyzer mặc định `standard`, không có `asciifolding`. Index A —
index có `vi_folded` — lại là schema Lab 6 không có builder production nào. Nên dữ
liệu OCR/ASR tiếng Việt thật đang nằm ở index không biết bỏ dấu, khiến `"Hà Nội"`
không khớp `"Ha Noi"`. OCR tiếng Việt sai dấu rất thường xuyên.

**Files:**
- Modify: `src/mvr/indexing/lexical_index.py:22` (`TASK05B_SCHEMA_VERSION`), `:24-34` (`LEXICAL_MAPPINGS`), `:352-365` (`create_lexical_index`)
- Test: `tests/unit/indexing/test_task05b.py`
- Create: `tests/integration/test_task05b_analyzer.py`

**Interfaces:**
- Consumes: không gì từ Task 1 (độc lập; làm song song được).
- Produces: `mvr.indexing.lexical_index.LEXICAL_SETTINGS` — `dict[str, Any]` chứa analyzer `vi_folded`. `create_lexical_index` truyền nó vào `client.indices.create(..., settings=LEXICAL_SETTINGS)`. `TASK05B_SCHEMA_VERSION` thành `"1.1"`.

- [ ] **Step 1: Viết test cho payload tạo index**

Thêm vào `tests/unit/indexing/test_task05b.py`:

```python
def test_index_is_created_with_a_diacritic_folding_vietnamese_analyzer(
    tmp_path: Path,
) -> None:
    """OCR và ASR của corpus là tiếng Việt, và OCR sai dấu rất thường xuyên.
    Không có `asciifolding` thì "Ha Noi" không khớp "Hà Nội", nên một truy vấn
    tiếng Việt đúng vẫn trượt document đúng."""
    from mvr.indexing.lexical_index import LEXICAL_SETTINGS

    processed = _write_processed(tmp_path, [_document("frame-a")])
    client = _Client(exists=False)
    build_lexical_index(
        client,
        processed_root=processed,
        index_name="lexical-v1",
        recreate=False,
        bulk_fn=lambda *args, **kwargs: (1, []),
        scan_fn=lambda *args, **kwargs: [{"_id": "L21_V010/frame-a"}],
    )

    create_call = next(call for call in client.calls if call[0] == "create")
    assert create_call[1]["settings"] == LEXICAL_SETTINGS
    analyzer = LEXICAL_SETTINGS["analysis"]["analyzer"]["vi_folded"]
    assert analyzer["filter"] == ["lowercase", "asciifolding"]
    assert create_call[1]["mappings"]["properties"]["ocr_text"] == {
        "type": "text",
        "analyzer": "vi_folded",
    }
    assert create_call[1]["mappings"]["properties"]["asr_text"] == {
        "type": "text",
        "analyzer": "vi_folded",
    }
```

Lưu ý: `_Client` và `_write_processed` đã có sẵn trong file này. Đọc chúng trước khi
viết, để khớp đúng chữ ký `bulk_fn`/`scan_fn` mà các test lân cận đang dùng — đừng đoán.

- [ ] **Step 2: Chạy test để xác nhận nó fail**

Run: `./.venv/Scripts/python.exe -m pytest "tests/unit/indexing/test_task05b.py::test_index_is_created_with_a_diacritic_folding_vietnamese_analyzer" -v`

Expected: FAIL với `ImportError: cannot import name 'LEXICAL_SETTINGS'`.

- [ ] **Step 3: Thêm settings và analyzer vào module**

Trong `src/mvr/indexing/lexical_index.py`:

```python
TASK05B_SCHEMA_VERSION = "1.1"

#: Sao nguyên từ `elasticsearch_index.SETTINGS`, index Lab 6 vốn đã dùng đúng
#: analyzer này cho tiếng Việt. `asciifolding` bỏ dấu ở CẢ lúc index lẫn lúc
#: truy vấn, nên "Ha Noi" khớp "Hà Nội" theo cả hai chiều -- cần thiết vì OCR
#: tiếng Việt sai dấu rất thường xuyên.
LEXICAL_SETTINGS: dict[str, Any] = {
    "analysis": {
        "analyzer": {
            "vi_folded": {
                "tokenizer": "standard",
                "filter": ["lowercase", "asciifolding"],
            }
        }
    }
}
LEXICAL_MAPPINGS: dict[str, Any] = {
    "properties": {
        "keyframe_id": {"type": "keyword"},
        "video_id": {"type": "keyword"},
        "frame_idx": {"type": "integer"},
        "shot_id": {"type": "integer"},
        "timestamp": {"type": "float"},
        "ocr_text": {"type": "text", "analyzer": "vi_folded"},
        "asr_text": {"type": "text", "analyzer": "vi_folded"},
    }
}
```

Rồi trong `create_lexical_index`:

```python
    if not exists:
        client.indices.create(
            index=index_name,
            mappings=LEXICAL_MAPPINGS,
            settings=LEXICAL_SETTINGS,
        )
```

- [ ] **Step 4: Chạy test để xác nhận nó pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/indexing/test_task05b.py -v`

Expected: tất cả PASS. Test cũ `create_call[1]["mappings"] == LEXICAL_MAPPINGS` vẫn
đúng vì nó so với chính hằng số đã đổi.

- [ ] **Step 5: Viết integration test cho hành vi analyzer thật**

Unit test ở trên chỉ khẳng định payload gửi đi, không chứng minh Elasticsearch xử lý
đúng. Tạo `tests/integration/test_task05b_analyzer.py`:

```python
"""Kiểm chứng analyzer `vi_folded` trên một Elasticsearch thật.

Unit test chỉ khẳng định payload tạo index có đúng settings; nó không chứng
minh Elasticsearch thật sự bỏ dấu. Test này cần một ES đang chạy nên tắt mặc
định -- máy phát triển hiện không có Docker.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MVR_ES_URL"),
    reason="đặt MVR_ES_URL trỏ tới một Elasticsearch đang chạy để chạy test này",
)


def test_unaccented_query_matches_accented_vietnamese_text() -> None:
    from elasticsearch import Elasticsearch

    from mvr.indexing.lexical_index import LEXICAL_MAPPINGS, LEXICAL_SETTINGS
    from mvr.retrieval.lexical_bm25 import search_task05b_lexical

    client = Elasticsearch(os.environ["MVR_ES_URL"])
    index_name = "task05b-analyzer-test"
    client.indices.delete(index=index_name, ignore_unavailable=True)
    client.indices.create(
        index=index_name,
        mappings=LEXICAL_MAPPINGS,
        settings=LEXICAL_SETTINGS,
    )
    try:
        client.index(
            index=index_name,
            id="L21_V001/034",
            document={
                "keyframe_id": "034",
                "video_id": "L21_V001",
                "frame_idx": 1745,
                "shot_id": 7,
                "timestamp": 69.8,
                "ocr_text": "Hà Nội mùa thu",
                "asr_text": "",
            },
            refresh=True,
        )

        for query in ("Hà Nội", "Ha Noi", "ha noi"):
            results = search_task05b_lexical(
                client, query, index_name=index_name, top_k=5
            )
            assert [r.keyframe_id for r in results] == ["034"], query
    finally:
        client.indices.delete(index=index_name, ignore_unavailable=True)
```

- [ ] **Step 6: Xác nhận integration test được skip sạch trên máy này**

Run: `./.venv/Scripts/python.exe -m pytest tests/integration/test_task05b_analyzer.py -v`

Expected: SKIPPED, không ERROR. Nếu ERROR thì có import chạy trước `skipif` — kiểm tra
lại rằng mọi import `elasticsearch` nằm **trong** thân hàm test.

- [ ] **Step 7: Chạy toàn bộ suite**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit -q`

Expected: không fail thêm.

- [ ] **Step 8: Commit**

```bash
git add src/mvr/indexing/lexical_index.py tests/unit/indexing/test_task05b.py tests/integration/test_task05b_analyzer.py
git commit -m "feat: fold diacritics in the Task05B lexical index

The corpus OCR and ASR are Vietnamese and OCR gets the diacritics wrong
often, so without asciifolding a correct Vietnamese query misses the
document it should match. The analyzer is copied from the Lab 6 index,
which had it right; the index that actually holds the data did not.

Schema version 1.1: existing indexes need a reindex to pick this up."
```

---

### Task 3: `FusedRetriever` nhận hai chuỗi truy vấn

**Files:**
- Modify: `src/mvr/retrieval/qna_fusion.py:65` (tên lớp), `:88-140` (`search_hybrid`)
- Test: `tests/unit/retrieval/test_qna_fused_retriever.py`

**Interfaces:**
- Consumes: `LexicalSearchHit` từ `mvr.retrieval.lexical_bm25` (Task 1 giữ nguyên chữ ký của nó).
- Produces: `mvr.retrieval.qna_fusion.FusedRetriever` với
  `search_hybrid(self, query: str, *, top_k: int, rrf_k: int, lexical_query: str | None = None) -> list[dict[str, Any]]`.
  Alias `QnaFusedRetriever = FusedRetriever` giữ nguyên để code Q&A hiện có không vỡ.

- [ ] **Step 1: Viết test cho việc hai nhánh nhận hai chuỗi khác nhau**

Thêm vào `tests/unit/retrieval/test_qna_fused_retriever.py`:

```python
def test_each_branch_receives_its_own_query_string() -> None:
    """Nhánh dense là OpenCLIP RN50, tiếng Anh thuần. Nhánh lexical là BM25
    trên OCR/ASR tiếng Việt. Một chuỗi duy nhất không thể phục vụ cả hai:
    dịch sang tiếng Anh làm hỏng BM25, giữ tiếng Việt làm hỏng CLIP."""
    from mvr.retrieval.qna_fusion import FusedRetriever

    seen: dict[str, str] = {}

    class _Semantic:
        def search(self, query, top_k):
            seen["dense"] = query
            return []

    def _lexical(query, *, top_k):
        seen["lexical"] = query
        return []

    FusedRetriever(semantic_service=_Semantic(), lexical_search=_lexical).search_hybrid(
        "a red motorbike on a rainy street",
        top_k=10,
        rrf_k=60,
        lexical_query="xe máy màu đỏ trên phố mưa",
    )

    assert seen["dense"] == "a red motorbike on a rainy street"
    assert seen["lexical"] == "xe máy màu đỏ trên phố mưa"


def test_omitting_lexical_query_sends_one_string_to_both_branches() -> None:
    """Hành vi mặc định phải y hệt hiện tại, nếu không toàn bộ đường Q&A đang
    chấm điểm sẽ đổi kết quả mà không ai yêu cầu."""
    from mvr.retrieval.qna_fusion import FusedRetriever

    seen: dict[str, str] = {}

    class _Semantic:
        def search(self, query, top_k):
            seen["dense"] = query
            return []

    def _lexical(query, *, top_k):
        seen["lexical"] = query
        return []

    FusedRetriever(semantic_service=_Semantic(), lexical_search=_lexical).search_hybrid(
        "chỉ một chuỗi", top_k=10, rrf_k=60
    )

    assert seen["dense"] == "chỉ một chuỗi"
    assert seen["lexical"] == "chỉ một chuỗi"


def test_the_old_class_name_still_works() -> None:
    """`QnaFusedRetriever` được import ở `scripts/run_qna_pipeline.py`; đổi tên
    mà không giữ alias sẽ làm vỡ đường Q&A đang chấm điểm."""
    from mvr.retrieval.qna_fusion import FusedRetriever, QnaFusedRetriever

    assert QnaFusedRetriever is FusedRetriever
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/retrieval/test_qna_fused_retriever.py -k "own_query or omitting or old_class" -v`

Expected: FAIL với `ImportError: cannot import name 'FusedRetriever'`.

- [ ] **Step 3: Đổi tên lớp và thêm tham số**

Trong `src/mvr/retrieval/qna_fusion.py`, đổi `class QnaFusedRetriever:` thành
`class FusedRetriever:`, và sau thân lớp thêm:

```python
#: Tên cũ, giữ lại vì `scripts/run_qna_pipeline.py` và các test Q&A import nó.
#: Lớp này không còn chỉ phục vụ Q&A: nhánh KIS dùng chung nó để tách truy vấn
#: tiếng Việt khỏi truy vấn tiếng Anh.
QnaFusedRetriever = FusedRetriever
```

Đổi `search_hybrid`:

```python
    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int,
        rrf_k: int,
        lexical_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """``query`` đi vào nhánh dense, ``lexical_query`` vào nhánh BM25.

        Hai nhánh muốn hai ngôn ngữ khác nhau. Encoder dense là OpenCLIP
        ``RN50/openai``, tiếng Anh thuần; index lexical chứa OCR/ASR tiếng
        Việt. Ép cùng một chuỗi vào cả hai thì luôn có một nhánh chịu thiệt.

        ``lexical_query=None`` giữ nguyên hành vi cũ -- một chuỗi cho cả hai.
        """
        lexical_text = query if lexical_query is None else lexical_query

        semantic_hits = [
            _semantic_payload(result)
            for result in self._semantic.search(query, self._per_branch_top_k)
        ]

        lexical_hits: list[Any] = []
        if self._lexical is not None:
            try:
                lexical_hits = list(
                    self._lexical(lexical_text, top_k=self._per_branch_top_k)
                )
            except Exception:
                # Elasticsearch chết thì mất recall, không mất cả truy vấn.
                # Nhánh semantic cố ý KHÔNG bọc try: nó là đường chính, nuốt
                # lỗi của nó là giấu một sự cố sau kết quả tệ đi trong im lặng.
                logger.warning(
                    "lexical branch unavailable; using semantic only", exc_info=True
                )
```

Phần còn lại của phương thức giữ nguyên.

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/retrieval/test_qna_fused_retriever.py -v`

Expected: tất cả PASS, gồm ba test mới.

- [ ] **Step 5: Commit**

```bash
git add src/mvr/retrieval/qna_fusion.py tests/unit/retrieval/test_qna_fused_retriever.py
git commit -m "feat: let each retrieval branch take its own query string

The dense branch runs OpenCLIP RN50, English-only; the lexical branch runs
BM25 over Vietnamese OCR and ASR. One string cannot serve both. Passing
lexical_query is optional, so every existing caller keeps its behaviour.

Renamed to FusedRetriever because KIS now shares it; the old name stays
as an alias."
```

---

### Task 4: `search_kis` chuyển tiếp truy vấn lexical

Phụ thuộc Task 3. `search_kis` chỉ được truyền `lexical_query` khi nó khác `None`. Truyền vô điều kiện
sẽ làm `ElasticsearchMultimodalService.search_hybrid` — chưa nhận tham số này —
`TypeError`.

**Files:**
- Modify: `src/mvr/retrieval/kis.py:107-140` (`search_kis`)
- Test: `tests/unit/retrieval/test_kis.py`

**Interfaces:**
- Consumes: `FusedRetriever.search_hybrid(..., lexical_query=...)` từ Task 3.
- Produces: `search_kis(query, *, top_k, semantic_service, multimodal_service, pipeline_resolver=None, aic_registry=None, overfetch=None, lexical_query=None) -> list[KisCandidate]`.

- [ ] **Step 1: Viết test**

Thêm vào `tests/unit/retrieval/test_kis.py`:

```python
def test_lexical_query_is_forwarded_when_given() -> None:
    seen: dict[str, object] = {}

    class _Service:
        def search_hybrid(self, query, *, top_k, rrf_k, lexical_query=None):
            seen["query"] = query
            seen["lexical_query"] = lexical_query
            return []

    search_kis(
        "a red motorbike",
        top_k=10,
        semantic_service=None,
        multimodal_service=_Service(),
        lexical_query="xe máy màu đỏ",
    )

    assert seen == {"query": "a red motorbike", "lexical_query": "xe máy màu đỏ"}


def test_a_service_without_lexical_query_support_is_not_broken() -> None:
    """`ElasticsearchMultimodalService` chưa nhận tham số này. Truyền vô điều
    kiện sẽ TypeError và làm hỏng route /search/hybrid đang chạy."""
    seen: dict[str, object] = {}

    class _OldService:
        def search_hybrid(self, query, *, top_k, rrf_k):
            seen["query"] = query
            return []

    search_kis(
        "a red motorbike",
        top_k=10,
        semantic_service=None,
        multimodal_service=_OldService(),
    )

    assert seen["query"] == "a red motorbike"
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/retrieval/test_kis.py -k "lexical_query or without_lexical" -v`

Expected: test đầu FAIL (`lexical_query` là `None` vì `search_kis` chưa nhận tham số);
test thứ hai PASS sẵn.

- [ ] **Step 3: Thêm tham số vào `search_kis`**

Thêm `lexical_query: str | None = None,` vào cuối danh sách tham số keyword của
`search_kis`, rồi đổi chỗ gọi:

```python
    if multimodal_service is not None:
        # Chỉ truyền khi có, vì `ElasticsearchMultimodalService.search_hybrid`
        # chưa nhận tham số này và sẽ TypeError nếu truyền vô điều kiện.
        extra = {} if lexical_query is None else {"lexical_query": lexical_query}
        raw_hits: Sequence[Mapping[str, Any]] = multimodal_service.search_hybrid(
            query.strip(), top_k=fetch_count, rrf_k=60, **extra
        )
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/retrieval/test_kis.py -v`

Expected: tất cả PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mvr/retrieval/kis.py tests/unit/retrieval/test_kis.py
git commit -m "feat: forward a Vietnamese lexical query through search_kis

Only forwarded when given, so ElasticsearchMultimodalService -- which does
not accept the parameter -- keeps working."
```

---

### Task 5: Bật nhánh lexical cho KIS smoke test

Phụ thuộc Task 3 và Task 4.

`run_real_data_kis_smoke.py` đang truyền `multimodal_service=None`, nên KIS baseline
hôm nay là FAISS đơn thuần: nhánh lexical mà cả kế hoạch này phục vụ chưa từng được
bật trong đo đạc KIS.

**Files:**
- Modify: `scripts/verification/run_real_data_kis_smoke.py:105`

**Interfaces:**
- Consumes: `FusedRetriever` (Task 3), `search_kis(..., lexical_query=...)` (Task 4), `search_task05b_lexical` (đã có).
- Produces: không có API mới; chỉ đổi cách nối dây trong script.

- [ ] **Step 1: Đọc chỗ gọi hiện tại**

Run: `sed -n '90,120p' scripts/verification/run_real_data_kis_smoke.py`

Ghi lại tên biến chứa semantic service, tên biến chứa truy vấn tiếng Việt gốc, và cách
script nhận tham số dòng lệnh, để các bước sau khớp đúng phong cách file — đừng đoán tên.

- [ ] **Step 2: Thêm hàm dựng multimodal service**

Thêm vào script, ngay trên hàm chứa dòng 105:

```python
def _build_multimodal_service(semantic_service):
    """Nối nhánh BM25 khi có Elasticsearch, ngược lại trả None.

    Trả None chứ không raise: máy phát triển không có Elasticsearch, và một
    smoke test FAISS-only vẫn có giá trị. Nhưng nó KHÔNG đo được nhánh
    lexical -- báo cáo phải nói rõ điều đó thay vì để người đọc tự đoán.
    """
    import os

    es_url = os.environ.get("MVR_ES_URL", "").strip()
    index_name = os.environ.get("MVR_ES_LEXICAL_INDEX", "").strip()
    if not es_url or not index_name:
        return None

    from elasticsearch import Elasticsearch

    from mvr.retrieval.lexical_bm25 import search_task05b_lexical
    from mvr.retrieval.qna_fusion import FusedRetriever

    client = Elasticsearch(es_url)

    def lexical_search(query, *, top_k):
        return search_task05b_lexical(client, query, index_name=index_name, top_k=top_k)

    return FusedRetriever(
        semantic_service=semantic_service, lexical_search=lexical_search
    )
```

- [ ] **Step 3: Nối vào chỗ gọi `search_kis`**

Thay `multimodal_service=None,` bằng
`multimodal_service=_build_multimodal_service(semantic_service),` và truyền thêm
`lexical_query=<biến chứa truy vấn tiếng Việt gốc>`, tên lấy từ Step 1.

- [ ] **Step 4: Chạy script ở chế độ không có Elasticsearch**

Run: `python scripts/verification/run_real_data_kis_smoke.py --help`

Expected: in ra trợ giúp, không lỗi import. Không đặt `MVR_ES_URL` thì đường chạy phải
giống hệt trước.

- [ ] **Step 5: Commit**

```bash
git add scripts/verification/run_real_data_kis_smoke.py
git commit -m "feat: run the KIS smoke test with the lexical branch on

It passed multimodal_service=None, so every KIS measurement so far was
FAISS alone -- the lexical branch this work exists to feed was never in
the numbers. Still None when no Elasticsearch is configured."
```

---

## Self-Review

**Phủ spec:**

| Mục spec | Task |
|---|---|
| 4.1 Sửa hợp đồng `_id` | Task 1 |
| 4.2 Tách query, giữ chữ ký | Task 3 + Task 4 |
| 4.3 Analyzer tiếng Việt | Task 2 |
| 4.4 Bật lexical cho KIS | Task 5 |
| 4.5 Test | nằm trong từng task |
| 5 Phụ thuộc nhánh git | đã hết hiệu lực: PR Q&A merge rồi, ghi ở "Phạm vi" |
| 6 Giai đoạn C | ngoài phạm vi kế hoạch này, nêu rõ ở "Phạm vi" |
| 7 Ngoài phạm vi | không có task, đúng ý định |

**Nhất quán kiểu:** `FusedRetriever.search_hybrid(query, *, top_k, rrf_k, lexical_query=None)` dùng đúng tên đó ở Task 3, 4, 5. `LEXICAL_SETTINGS` dùng đúng tên ở Task 2 và trong integration test. `LexicalSearchHit` không đổi chữ ký.

**Không placeholder:** mọi bước có mã thật. Hai chỗ duy nhất yêu cầu người thực thi đọc file trước (`_Client` ở Task 2 Step 1, tên biến ở Task 5 Step 1) đều nói rõ đọc gì và vì sao không nên đoán.
