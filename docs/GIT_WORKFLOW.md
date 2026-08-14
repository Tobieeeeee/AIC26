# Git Workflow

## Branches

`main` is the stable shared integration branch. Normal feature implementation must not be performed directly on `main`.

Repository-wide governance or documentation changes may be committed directly to `main` only when the active task explicitly authorizes it. Otherwise, changes reach `main` through a reviewed Pull Request with passing CI.

Current feature branches include:

- `feat/e2e-baseline`
- `feat/shot-keyframe`
- `feat/semantic-retrieval`
- `feat/ocr-object`
- `feat/indexing`
- `feat/hybrid-search`
- `feat/query-understanding`
- `feat/temporal-retrieval`
- `feat/reranking`
- `feat/vlm-caption`
- `feat/evaluation`
- `feat/api`
- `feat/ui`
- `feat/submission`
- `feat/deployment`

The reuse migration may use `feat/reuse-lab-baseline` when specified by the active task.

Branches describe functionality, not team-member names. Short-lived branches such as `fix/<bug-name>` or `eval/<experiment-name>` may be used when their purpose is clear.

## Feature workflow

The normal development workflow is:

```text
main
  ↓
feature branch
  ↓
implementation
  ↓
tests
  ↓
commit
  ↓
push
  ↓
Pull Request
  ↓
CI / review
  ↓
merge to main
```

## Notebook and production-code rule

- Production code belongs under `src/mvr/` as reusable, importable Python modules.
- Experiments belong under `notebooks/experiments/`.
- Original lab and reference notebooks remain under `notebooks/reference/`.
- Production features must not depend on manually executed notebook state.

When an experiment becomes production functionality, use this promotion path:

```text
experiment notebook
    ↓
validate
    ↓
reusable Python module
    ↓
tests
    ↓
integration
```

## Commit convention

Use meaningful prefixes when practical:

- `feat:`
- `fix:`
- `test:`
- `docs:`
- `refactor:`
- `perf:`
- `chore:`

```text
feat: add TransNetV2 shot detector
feat: add PaddleOCR adapter
perf: cache CLIP text encoder
fix: correct inclusive shot end frame
test: cover weighted fusion edge cases
docs: document offline indexing flow
chore: update dependencies
```

Each commit should represent one logical change and should be testable. Do not create meaningless empty commits merely to make branch histories look different.

## Pull request

PR cần ghi:

1. Thay đổi gì.
2. Vì sao cần.
3. Cách test.
4. Benchmark trước/sau nếu thay retrieval/model.
5. Artifact lớn nằm ở đâu, không attach vào Git history.

## Merge rules

- Normal feature work must not be pushed directly to `main`.
- Direct governance/documentation commits to `main` require explicit authorization.
- CI must pass before a normal feature PR is merged.
- Significant changes should receive at least one review.
- Squash merge may be used for small features to keep history focused.
- Milestones may be tagged as `v0.1-ingestion`, `v0.2-index`, `v0.3-retrieval`, and `v1.0-competition`.

## Git safety

Contributors and agents must not:

- force-push unless explicitly authorized;
- silently discard unrelated user changes;
- rewrite unrelated history;
- commit datasets, generated keyframes, embeddings, indexes, model weights, or other generated artifacts;
- commit credentials, API keys, tokens, `.env` files, or other secrets;
- merge into `main` without the required workflow or explicit authorization.

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

## Example workflow

```bash
git checkout main
git pull
git checkout -b feat/ocr-object

# code + test
git add src/mvr/enrichment/ocr tests/
git commit -m "feat: add OCR engine interface"
git push -u origin feat/ocr-object

# mở PR -> review -> CI -> merge
```
