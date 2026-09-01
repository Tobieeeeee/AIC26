# AIC26 Milvus Runtime Audit & Architecture Decision Report

## 1. Kết luận tổng thể

**Runtime audit: FAIL — Milvus service khỏe nhưng collection chưa tồn tại.**

Milvus `v2.4.13`, etcd và MinIO đang hoạt động bình thường. Tuy nhiên database `default` hiện có `0 collection`; collection đích `aic26_keyframes` chưa được tạo. Vì vậy runtime chưa có entity, schema, vector index hoặc ANN Search thực tế.

Quyết định triển khai đã được chốt:

- Sửa primary key trước lần nạp vector đầu tiên.
- Dùng `entity_id = video_id + "/" + keyframe_id` làm primary key.
- Giữ `video_id` và `keyframe_id` thành hai scalar field riêng.
- Dùng **HNSW** cho cả ba vector field.
- Dùng metric **IP**.
- Cấu hình mặc định: `M=16`, `efConstruction=200`, search `ef=64`.
- Không benchmark recall hoặc latency vì hiện chưa có Ground Truth.
- Task Milvus này độc lập và không sửa Elasticsearch.

## 2. Phân biệt thuật ngữ

Trong pipeline có hai thao tác khác nhau, không nên cùng gọi chung là “ingest”:

| Giai đoạn | Ý nghĩa |
|---|---|
| Video ingestion | Đọc video, phát hiện shot và sinh keyframe |
| Nạp vector vào Milvus | Ghi metadata và embedding của keyframe vào collection Milvus |

Flow được thống nhất:

```text
Video
→ Video ingestion
→ Sinh keyframe
→ Sinh ba embedding
→ Validate vector
→ Tạo entity_id duy nhất
→ Nạp metadata + vector vào Milvus
→ Xây HNSW trên ba vector field
→ Load collection
→ Vector search từng provider
→ Hợp nhất kết quả bằng RRF
```

## 3. Baseline đã kiểm tra

Các điểm sau đã được audit trực tiếp và được xem là baseline. Agent triển khai không cần audit lại toàn bộ repo trước khi sửa; chỉ cần xác minh tối thiểu tại các file trực tiếp bị tác động.

### 3.1 Runtime

| Thành phần | Kết quả |
|---|---|
| Milvus image | `milvusdb/milvus:v2.4.13` |
| Database | `default` |
| Health endpoint | PASS — HTTP 200 |
| gRPC `19530` | PASS — reachable |
| etcd | Healthy |
| MinIO | Healthy |
| Authentication token | Không cấu hình |

Các biến môi trường Milvus hiện không được đặt, nên code sử dụng default:

```text
MILVUS_HOST=<unset>       → localhost
MILVUS_PORT=<unset>       → 19530
MILVUS_DATABASE=<unset>   → default
MILVUS_COLLECTION=<unset> → aic26_keyframes
MVR_MILVUS_INDEX_TYPE=<unset>
MVR_MILVUS_MANIFEST=<unset>
```

### 3.2 Collection và ANN index

Runtime trả về danh sách collection rỗng:

```json
{"code": 0, "data": []}
```

Collection `aic26_keyframes` không tồn tại. Vì vậy trạng thái thực tế là:

| Thuộc tính | Runtime hiện tại |
|---|---|
| Collection | Không tồn tại |
| Entity count | N/A |
| Schema | Không có |
| Primary key | Không có |
| Vector fields | Không có |
| Vector indexes | Không có |
| ANN Search | **NONE** |
| Load state | Không loaded |

Không được mô tả runtime hiện tại là đang dùng FLAT, HNSW hoặc IVF_FLAT. Default `FLAT` trong source chỉ là fallback của code, chưa phải trạng thái runtime.

### 3.3 Schema dự kiến trong source

Source hiện dự kiến ba vector field:

| Field | Dimension |
|---|---:|
| `beit3_vector` | 768 |
| `metaclip2_vector` | 512 |
| `pe_vector` | 1024 |

Source hỗ trợ `FLAT`, `HNSW` và `IVF_FLAT`, metric mặc định là `IP`. Manifest mặc định `data/milvus_index_manifest.json` chưa tồn tại.

### 3.4 Rủi ro collision

Schema source hiện dùng `keyframe_id` làm primary key và đường nạp dữ liệu dùng `upsert`. Nếu `keyframe_id` chỉ duy nhất trong phạm vi từng video, hai video có cùng `keyframe_id` có thể ghi đè entity của nhau.

Thiết kế cần sửa trước khi tạo collection:

```text
entity_id = video_id + "/" + keyframe_id   # primary key
video_id                                  # scalar field
keyframe_id                               # scalar field
```

Ví dụ:

```text
video_id    = L21_V001
keyframe_id = L21_V001_S0011_K01
entity_id   = L21_V001/L21_V001_S0011_K01
```

Việc tạo `entity_id` phải có một hàm dùng chung và deterministic; không ghép chuỗi rải rác tại nhiều call site.

### 3.5 Search filter

Đường production search hiện gọi `col.search(...)` mà không truyền `expr` hoặc metadata predicate. Do đó chưa có server-side prefilter. Audit cũng không thấy post-filter trong đường search Milvus này.

Task hiện tại không bổ sung filter nếu không cần thiết để tương thích với schema/primary key mới. Việc thiết kế prefilter/post-filter nên là task riêng.

## 4. Quyết định HNSW

HNSW được chọn cố định cho collection production ở thời điểm này:

```text
index_type: HNSW
metric_type: IP
M: 16
efConstruction: 200
search ef: 64
```

Áp dụng cho:

- `beit3_vector`
- `metaclip2_vector`
- `pe_vector`

Không thực hiện:

- benchmark với FLAT hoặc IVF_FLAT;
- đo recall;
- đo latency;
- chọn index tự động dựa trên benchmark.

Lý do: chưa có Ground Truth để tính recall có ý nghĩa, và quyết định vận hành hiện tại đã chốt HNSW.

## 5. Trạng thái mục tiêu sau triển khai

| Hạng mục | Trạng thái yêu cầu |
|---|---|
| Collection | `aic26_keyframes` tồn tại |
| Primary key | `entity_id` |
| Collision | Không ghi đè giữa các video |
| Vector fields | Đủ ba field, đúng dimension |
| Index type | HNSW trên cả ba vector field |
| Metric | IP trên cả ba index |
| Build params | `M=16`, `efConstruction=200` |
| Search params | `ef=64` |
| Load state | Loaded trước khi query |
| Provider search | Cả ba provider trả được hit hợp lệ khi có dữ liệu |
| Metadata | Hit trả về đủ `entity_id`, `video_id`, `keyframe_id` và metadata cần thiết |
| Elasticsearch | Không bị thay đổi |

## 6. Ranh giới triển khai

Task Milvus chỉ bao gồm:

1. Sửa schema và primary key.
2. Tạo `entity_id` nhất quán.
3. Nạp metadata và vector vào Milvus mà không collision.
4. Tạo HNSW cho cả ba vector field.
5. Dùng đúng search params HNSW.
6. Thêm test và smoke verification cần thiết.

Task không bao gồm:

- sinh keyframe;
- thay đổi pipeline OCR/ASR;
- sửa Elasticsearch hoặc lexical search;
- thay đổi flow exact/folded/fuzzy của Elasticsearch;
- bổ sung hybrid fusion hoặc sửa RRF ngoài phần tương thích bắt buộc;
- thiết kế prefilter/post-filter;
- rebuild hoặc nạp toàn bộ corpus nếu chưa được yêu cầu rõ;
- deploy public;
- đo recall/latency.

## 7. Kết luận

Milvus runtime khỏe nhưng đang rỗng, nên hiện chưa dùng ANN Search nào. Đây là thời điểm phù hợp để sửa primary key trước lần nạp vector đầu tiên. Sau khi sửa, hệ thống phải dùng `entity_id` chống collision và xây HNSW/IP trên cả ba vector field với cấu hình đã chốt.
