# Git workflow cho nhóm

## Branches

- `main`: luôn chạy được, chỉ merge qua PR.
- `dev`: nhánh tích hợp trước khi lên `main` nếu nhóm muốn thêm một lớp kiểm soát.
- Feature branches:
  - `feat/shot-detection`
  - `feat/keyframe-extraction`
  - `feat/semantic-retrieval`
  - `feat/ocr`
  - `feat/object-detection`
  - `feat/indexing`
  - `feat/hybrid-search`
  - `feat/temporal-retrieval`
  - `feat/api`
  - `feat/ui`
  - `eval/<experiment-name>`
  - `fix/<bug-name>`

Không chia branch theo tên thành viên; chia theo chức năng để lịch sử repo nói rõ thay đổi gì.

## Commit convention

```text
feat: add TransNetV2 shot detector
feat: add PaddleOCR adapter
perf: cache CLIP text encoder
fix: correct inclusive shot end frame
test: cover weighted fusion edge cases
docs: document offline indexing flow
chore: update dependencies
```

Mỗi commit nên làm một việc logic và chạy test được.

## Pull request

PR cần ghi:

1. Thay đổi gì.
2. Vì sao cần.
3. Cách test.
4. Benchmark trước/sau nếu thay retrieval/model.
5. Artifact lớn nằm ở đâu, không attach vào Git history.

## Quy tắc merge

- Không push thẳng `main`.
- CI phải xanh.
- Ít nhất 1 người review phần thay đổi lớn.
- Dùng squash merge cho feature nhỏ để history sạch.
- Tag milestone: `v0.1-ingestion`, `v0.2-index`, `v0.3-retrieval`, `v1.0-competition`.

## Large files

Không commit:
- `.venv/`;
- video gốc;
- hàng nghìn keyframe;
- model weights;
- `.npy/.parquet` lớn;
- FAISS/Elasticsearch index;
- output benchmark raw rất lớn.

Nếu bắt buộc version một file lớn, cân nhắc Git LFS; mặc định nên dùng shared storage/object storage và giữ manifest/checksum trong repo.

## Luồng làm việc mẫu

```bash
git checkout main
git pull
git checkout -b feat/ocr

# code + test
git add src/mvr/enrichment/ocr tests/
git commit -m "feat: add OCR engine interface"
git push -u origin feat/ocr

# mở PR -> review -> CI -> merge
```
