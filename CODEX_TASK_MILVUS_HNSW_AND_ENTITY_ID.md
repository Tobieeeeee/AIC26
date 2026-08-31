# Codex Task — Milvus `entity_id` + HNSW cho ba vector field

## 1. Mục tiêu

Sửa riêng layer Milvus của repo AIC26 để:

1. Dùng primary key không collision giữa các video:

   ```text
   entity_id = video_id + "/" + keyframe_id
   ```

2. Giữ `video_id` và `keyframe_id` thành scalar metadata riêng.
3. Tạo HNSW index cho cả ba vector field bằng metric IP.
4. Dùng search param HNSW đúng khi query.
5. Thêm test và smoke verification tập trung vào phần thay đổi.

Đây là task Milvus độc lập. **Không sửa Elasticsearch.**

## 2. Quyết định đã chốt — không cần đề xuất lại

```text
Index type: HNSW
Metric: IP
M: 16
efConstruction: 200
Search ef: 64
```

Áp dụng cho:

```text
beit3_vector      dim=768
metaclip2_vector  dim=512
pe_vector         dim=1024
```

Không đo recall và latency. Không benchmark HNSW với FLAT/IVF_FLAT vì hiện không có Ground Truth và lựa chọn HNSW đã được chốt.

## 3. Baseline đã được kiểm tra

Không audit lại toàn bộ repo. Dùng các kết luận sau làm baseline và chỉ xác minh tối thiểu tại các file/call site trực tiếp bị sửa:

- Milvus image: `milvusdb/milvus:v2.4.13`.
- Milvus, etcd và MinIO đang healthy.
- Database runtime: `default`.
- Runtime hiện có `0 collection`.
- Collection mặc định `aic26_keyframes` chưa tồn tại.
- Runtime hiện chưa có vector index và chưa dùng ANN Search nào.
- Source schema nằm tại `src/mvr/indexing/milvus_schema.py`.
- Production client/search nằm tại `src/mvr/indexing/milvus_client.py`.
- Source hiện dự kiến ba vector field: `beit3_vector`, `metaclip2_vector`, `pe_vector`.
- Metric source là `IP`.
- Source đang dùng `keyframe_id` làm primary key.
- Đường nạp dữ liệu dùng `upsert`, nên có nguy cơ ghi đè nếu `keyframe_id` lặp giữa các video.
- `col.search(...)` hiện không có metadata filter; filter không thuộc phạm vi task này.
- Manifest mặc định `data/milvus_index_manifest.json` chưa tồn tại.
- Các environment variable Milvus hiện không được đặt; collection default là `aic26_keyframes`.

Không được suy luận rằng runtime đang dùng FLAT chỉ vì source có fallback FLAT. Runtime hiện là `NONE` vì collection chưa tồn tại.

## 4. Phạm vi file cần đọc

Trước khi sửa, chỉ cần đọc:

1. `src/mvr/indexing/milvus_schema.py`.
2. `src/mvr/indexing/milvus_client.py`.
3. Các file định nghĩa record/entity được truyền trực tiếp vào Milvus.
4. Các call site trực tiếp gọi create collection, create index, upsert hoặc search.
5. Test hiện có liên quan trực tiếp đến Milvus schema/client/indexing.
6. File cấu hình hoặc `.env.example` có khai báo các biến Milvus bị tác động.

Chỉ mở rộng sang file khác khi một dependency/call site trực tiếp yêu cầu. Không audit lại Elasticsearch, API, frontend, toàn bộ video pipeline hoặc toàn bộ repository.

## 5. Yêu cầu triển khai

### 5.1 Tạo `entity_id` dùng chung

Tạo một helper deterministic, có type hint và test:

```python
def make_entity_id(video_id: str, keyframe_id: str) -> str:
    ...
```

Yêu cầu:

- output chính xác theo format `video_id/keyframe_id`;
- validate hai thành phần không rỗng sau khi xử lý whitespace theo convention hiện có của repo;
- không dùng hash làm mất khả năng đọc/debug;
- mọi đường nạp vector phải dùng cùng helper;
- không ghép `entity_id` rải rác ở nhiều nơi;
- không thay đổi `video_id` hoặc `keyframe_id` gốc.

Nếu source hiện dùng object/model cho entity, thêm `entity_id` tại tầng gần record Milvus nhất và giữ backward compatibility hợp lý cho dữ liệu đầu vào chưa có sẵn field này.

### 5.2 Sửa schema Milvus

Schema mục tiêu phải có tối thiểu:

| Field | Kiểu/vai trò |
|---|---|
| `entity_id` | `VARCHAR`, primary key, không auto-id |
| `video_id` | Scalar metadata |
| `keyframe_id` | Scalar metadata |
| `beit3_vector` | Float vector, dim 768 |
| `metaclip2_vector` | Float vector, dim 512 |
| `pe_vector` | Float vector, dim 1024 |

Giữ nguyên các metadata field hợp lệ đang có như frame/timestamp/path/batch nếu source đã định nghĩa; không xóa field ngoài phạm vi.

Chọn `max_length` cho `entity_id` đủ chứa ID thực tế và validate trước khi upsert. Không dùng `keyframe_id` làm primary key nữa.

Collection runtime đang chưa tồn tại, nên không cần viết migration dữ liệu cũ. Tuy nhiên logic phải fail rõ nếu gặp collection cùng tên nhưng schema cũ/không tương thích; không âm thầm tiếp tục upsert vào schema sai.

### 5.3 Nạp vector vào Milvus

Lưu ý thuật ngữ: pipeline “video ingestion” sinh keyframe; bước trong task này là **nạp metadata và embedding đã có vào Milvus**.

Tại đường upsert:

- tạo hoặc xác minh `entity_id` trước khi ghi;
- ghi đủ `entity_id`, `video_id`, `keyframe_id`;
- validate dimension của ba vector theo convention hiện có;
- giữ kiểm tra NaN/Inf nếu source đã có;
- không sinh lại keyframe;
- không sinh lại embedding nếu không cần cho smoke fixture;
- cùng `entity_id` được phép upsert idempotent;
- hai video khác nhau có cùng `keyframe_id` phải tạo hai entity riêng.

### 5.4 Cố định HNSW cho production path

Tạo index trên cả ba vector field:

```python
{
    "index_type": "HNSW",
    "metric_type": "IP",
    "params": {
        "M": 16,
        "efConstruction": 200,
    },
}
```

Yêu cầu:

- cả ba vector field đều dùng cùng cấu hình trên;
- production default phải là HNSW, không còn fallback ngầm sang FLAT khi env chưa đặt;
- nếu vẫn giữ `MVR_MILVUS_INDEX_TYPE` để tương thích, default phải là `HNSW` và giá trị không hợp lệ phải fail với thông báo rõ;
- không thêm benchmark hoặc auto-selection;
- cập nhật `.env.example`/config docs trực tiếp liên quan nếu tồn tại;
- manifest sau khi build phải phản ánh HNSW thực tế, không ghi cấu hình dự kiến sai runtime.

Không tự ý rebuild hoặc nạp full corpus. Chỉ tạo collection/index smoke nếu cần để kiểm chứng và dữ liệu runtime đang rỗng.

### 5.5 Search HNSW

Production search trên từng provider phải dùng:

```python
{
    "metric_type": "IP",
    "params": {"ef": 64},
}
```

Yêu cầu:

- `anns_field` vẫn chọn đúng vector field theo provider;
- không dùng search params của IVF như `nprobe` cho HNSW;
- output fields tối thiểu phải đủ nhận diện hit: `entity_id`, `video_id`, `keyframe_id`, cộng metadata hiện có mà downstream cần;
- giữ behavior RRF hiện tại, chỉ sửa tương thích nếu primary key mới làm thay đổi cách đọc hit ID;
- downstream không được nhầm `entity_id` với `keyframe_id`;
- không thêm prefilter/post-filter trong task này.

### 5.6 Load collection

Sau khi schema và index đã sẵn sàng, collection phải được load trước search theo lifecycle hiện có. Không load trước khi index creation hoàn tất nếu điều đó gây trạng thái không nhất quán.

## 6. Test bắt buộc

### 6.1 Unit tests

Thêm hoặc cập nhật test cho ít nhất các trường hợp:

1. `make_entity_id("L21_V001", "K01") == "L21_V001/K01"`.
2. Thiếu `video_id` hoặc `keyframe_id` bị reject rõ ràng.
3. Hai video có cùng `keyframe_id` tạo hai `entity_id` khác nhau.
4. Schema dùng `entity_id` làm primary key, `auto_id=False`.
5. `video_id` và `keyframe_id` vẫn là scalar field.
6. Ba vector field có đúng dimension 768/512/1024.
7. Index params của cả ba field là HNSW/IP, `M=16`, `efConstruction=200`.
8. Search params là IP với `ef=64`.
9. Output fields có `entity_id`, `video_id`, `keyframe_id`.
10. Config không hợp lệ fail rõ, không âm thầm fallback sang FLAT.

### 6.2 Milvus integration/smoke test

Nếu runtime Milvus hiện có thể sử dụng, chạy smoke test nhỏ, không nạp full corpus:

1. Tạo collection test hoặc collection đích rỗng theo quy tắc an toàn hiện có.
2. Xác minh schema runtime dùng `entity_id` làm primary key.
3. Chèn/upsert ít nhất hai record có cùng `keyframe_id` nhưng khác `video_id`.
4. Xác minh entity count là 2 và không record nào bị ghi đè.
5. Xác minh cả ba vector field có HNSW index với metric IP và build params đúng.
6. Load collection.
7. Chạy một query trên từng provider với `ef=64`.
8. Xác minh hit trả về đủ `entity_id`, `video_id`, `keyframe_id`.
9. Xác minh không có duplicate do cùng primary key trong kết quả mỗi provider.

Dùng vector fixture nhỏ, deterministic và đúng dimension. Không cần tính recall hoặc latency.

Nếu runtime không cho phép chạy integration test, không giả lập PASS. Báo rõ test nào chưa chạy và nguyên nhân; unit tests vẫn phải hoàn tất.

### 6.3 Regression

Chạy test suite trực tiếp liên quan đến Milvus/indexing/retrieval. Không bắt buộc chạy toàn bộ repo nếu suite đó kéo theo model/GPU hoặc dịch vụ ngoài phạm vi. Báo chính xác command và kết quả.

## 7. Tiêu chí nghiệm thu

Task chỉ PASS khi thỏa tất cả điểm có thể kiểm chứng trong môi trường:

- [ ] `entity_id` là primary key duy nhất của collection.
- [ ] `entity_id` được tạo bằng một helper dùng chung.
- [ ] `video_id` và `keyframe_id` vẫn được lưu riêng.
- [ ] Hai video có cùng `keyframe_id` không ghi đè nhau.
- [ ] Có đủ ba vector field, đúng dimension.
- [ ] Cả ba field dùng HNSW/IP.
- [ ] `M=16`, `efConstruction=200`.
- [ ] Query dùng `ef=64`.
- [ ] Collection được load trước search.
- [ ] Ba provider search đọc đúng primary key và metadata.
- [ ] Unit tests liên quan PASS.
- [ ] Smoke integration PASS, hoặc được ghi rõ BLOCKED với bằng chứng nếu runtime không khả dụng.
- [ ] Không sửa Elasticsearch.
- [ ] Không đo recall/latency và không benchmark index khác.
- [ ] Không nạp full corpus ngoài yêu cầu.

## 8. Ngoài phạm vi — không thực hiện

- Không sửa Elasticsearch mapping/analyzer/query.
- Không triển khai exact/folded/fuzzy lexical search.
- Không sinh keyframe từ video.
- Không thay đổi OCR/ASR.
- Không thiết kế mới RRF/hybrid fusion.
- Không thêm Milvus prefilter hoặc post-filter.
- Không triển khai frontend/API public.
- Không deploy public.
- Không benchmark FLAT, IVF_FLAT hoặc HNSW.
- Không đo recall, latency, QPS hoặc memory.
- Không audit lại toàn bộ repository.

## 9. Yêu cầu báo cáo đầu ra của agent

Khi hoàn tất, trả report ngắn nhưng có bằng chứng, theo cấu trúc:

1. **Verdict:** PASS / PARTIAL / BLOCKED / FAIL.
2. **Files changed:** từng file và mục đích thay đổi.
3. **Schema sau sửa:** primary key, scalar fields, vector fields/dimensions.
4. **Index runtime:** index type, metric, `M`, `efConstruction` cho từng field.
5. **Search runtime:** `ef` và output fields cho từng provider.
6. **Collision proof:** hai record cùng `keyframe_id`, khác `video_id`, entity count và ID thực tế.
7. **Tests:** command, số test pass/fail/skip.
8. **Runtime mutations:** collection/index/entity nào đã được tạo; có phải smoke data hay không.
9. **Unresolved items:** chỉ các blocker/thứ chưa thể xác minh.
10. **Scope confirmation:** xác nhận Elasticsearch không bị sửa, full corpus không bị nạp và recall/latency không được đo.

Không chỉ mô tả source code. Nếu integration test chạy được, phải report trạng thái runtime thực tế sau thay đổi.
