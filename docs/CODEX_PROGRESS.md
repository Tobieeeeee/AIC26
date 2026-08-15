# Codex Progress

This file is the persistent checkpoint for the lab-reuse migration.

Codex must update it at the end of every phase.

## Phase 01 — Discover and Map

Status: COMPLETE

Commit: `docs: map reusable lab implementations` (this phase commit; hash reported after commit)

Completed:
- Installed the Codex prompt pack at the required repository-root paths.
- Located the Shot Boundary Lab, Lab 3, Lab 4, Lab 6, and FastAPI Lab in the read-only source root.
- Inspected the source implementations for every component required by Phase 01.
- Added 47 provenance-backed component mappings to `docs/LAB_REUSE_MAP.md`.
- Verified that all three repository reference notebooks are byte-identical to their source originals.
- Documented behavior contracts, compatibility concerns, deferred experiments, and target destinations.
- Kept production code unchanged and did not begin migration.

Remaining:
- None for Phase 01.

Tests:
- `python -m pytest -q`: 4 passed.
- Reuse-map validation: 47 component rows, all `MAPPED`, no `TBD`/`TODO` placeholders.
- Production diff check (`src/`, `tests/`, `configs/`, `pyproject.toml`, `.github/`): no changes.
- Read-only source integrity: unchanged at 11,177 files / 1,221,999,740 bytes / metadata SHA-256 `07b34aa47d766429a2667a682d504df837a304b707c9b999aacc737ef057bfbd`.

Blockers:
- None.

---

## Phase 02 — Reuse Ingestion

Status: COMPLETE

Commit: `feat: reuse shot boundary ingestion pipeline` (this phase commit; hash reported after commit)

Completed:
- Reused the Shot Boundary Lab models, ffprobe probing, detector factory, PySceneDetect backend, TransNetV2 backend, keyframe selection, exact FFmpeg extraction, metadata writers, and pipeline under `src/mvr/ingestion/`.
- Added `python -m mvr.cli ingest` packaging while preserving the lab defaults and validation.
- Preserved zero-based frame indices, inclusive shot ends, timestamps, thresholds, primary-detector behavior, long-shot sampling, output schemas, and benchmark semantics.
- Added the lab's lightweight PySceneDetect, OpenCV, and tqdm dependency ranges.
- Verified the complete default ingestion flow on source sample `L01_V001.mp4`.
- Recorded all reused files and compatibility-only changes in `docs/LAB_REUSE_MAP.md`.
- Kept TransNetV2 lazy/optional and did not start Phase 03 or any OpenCLIP, FAISS, OCR, YOLO, Elasticsearch, or FastAPI work.

Remaining:
- None for Phase 02.

Tests:
- `py -3.12 -m pytest -q`: 23 passed, 1 skipped (optional TransNetV2 runtime dependencies unavailable).
- `py -3.12 -m mvr.cli --help`: passed.
- `py -3.12 -m mvr.cli ingest --help`: passed.
- Verified smoke command: `py -3.12 -m mvr.cli ingest "D:\project\STUDY\NAM_2\SELF_PRJ\AIC26\New folder\shot-boundary-lab-student\student\data\videos\L01_V001.mp4" --output "artifacts\keyframes\phase02-smoke"`.
- Smoke outputs: 5 shots, 5 keyframe mappings, 5 JPEGs, and JSON/CSV benchmark metadata.
- Read-only source integrity: unchanged at 11,177 files / 1,221,999,740 bytes / metadata SHA-256 `07b34aa47d766429a2667a682d504df837a304b707c9b999aacc737ef057bfbd`.

Blockers:
- None. TransNetV2 runtime verification is intentionally skipped as an optional heavy dependency, not a Phase 02 blocker.

---

## Phase 03 — Semantic End-to-End Baseline

Status: COMPLETE

Commit: `feat: reuse CLIP and FAISS semantic baseline` (this phase commit; hash reported after commit)

Completed:
- Reused Lab 3's OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` model loading, RGB preprocessing, image/text batching, and L2 normalization in `src/mvr/enrichment/embeddings/openclip.py`.
- Reused Lab 4's contiguous normalized `float32` vectors, exact `faiss.IndexFlatIP`, inner-product search, descending top-K ranking, and retrieval metrics.
- Added persistent FAISS index, JSON row metadata, schema-versioned manifest, and `.npy` embedding output without changing retrieval mathematics.
- Connected the Phase 02 `keyframe_mapping.csv` and JPEG output to `build-index`, then connected the saved index to OpenCLIP text encoding and `search`.
- Added optional semantic dependency packaging and kept heavy imports lazy so Phase 02 ingestion remains independently usable.
- Verified real OpenCLIP image/text inference, FAISS save/load, and text-to-keyframe search over five real extracted keyframes.
- Enforced normalized-vector and persistence invariants, preserved multi-relevance Recall@K, and made returned image paths directly resolvable.
- Documented every reused component and all compatibility-only changes in `docs/LAB_REUSE_MAP.md`.
- Did not begin OCR, YOLO, Elasticsearch, FastAPI, UI, temporal retrieval, reranking, or any Phase 04+ work.

Remaining:
- None for Phase 03.

Tests:
- `py -3.12 -m pip install -e ".[dev,semantic]"`: passed.
- `py -3.12 -m pytest -q`: 42 passed, 2 skipped (real OpenCLIP opt-in test and optional TransNetV2 runtime).
- `$env:MVR_RUN_OPENCLIP_TESTS='1'; py -3.12 -m pytest -q`: 43 passed, 1 skipped (only optional TransNetV2 runtime).
- Real OpenCLIP integration: image and text outputs each had shape `(2, 512)`, `float32` dtype, and unit L2 norms.
- Verified build command: `py -3.12 -m mvr.cli build-index --input artifacts\keyframes\phase02-smoke --output artifacts\indexes\phase03-smoke --device cpu`.
- Persisted index check: 5 normalized vectors of dimension 512, exact `IndexFlatIP`, 5 FAISS rows, and 5 metadata rows after reload.
- Verified search command: `py -3.12 -m mvr.cli search "a man riding a motorcycle" --index artifacts\indexes\phase03-smoke --top-k 5 --device cpu`.
- Search rank 1: `L01_V001_S0002_K01`, video `L01_V001`, timestamp `1.56`, score `0.16651125252246857`.
- Read-only source integrity count/size remained unchanged at 11,177 files / 1,221,999,740 bytes.

Blockers:
- None. TransNetV2 remains an intentionally optional Phase 02 backend and is not a Phase 03 blocker.

---

## Phase 04 — Multimodal Reuse

Status: COMPLETE

Commits: `feat: reuse Lab 6 multimodal retrieval components`; `test: verify Lab 6 multimodal retrieval runtime` (runtime hash reported after commit)

Completed:
- Reused Lab 6 EasyOCR (`vi`, `en`) filtering with automatic CUDA choice and confidence threshold `0.3` in a lazy adapter.
- Reused PaddleOCR Vietnamese 3.x/2.x construction compatibility, result traversal, confidence threshold `0.3`, deduplication, and final joined OCR text.
- Reused YOLOv8n `yolov8n.pt`, confidence `0.4`, COCO name lookup, and per-frame object counts.
- Reused lowercase whitespace-token OCR recall and the Lab 6 hit threshold `0.60`.
- Added the Lab 6 search document schema and an adapter from Phase 02 keyframe IDs/timestamps.
- Reused the Elasticsearch 8.x `vi_folded` analyzer, field mapping, cosine dense vector, explicit recreate, frame-ID bulk indexing, chunk size `50`, timeout `120`, refresh, and count semantics.
- Reused BM25 OCR emphasis (`ocr_text^2`), caption fallback, match-all behavior, optional person/car filters, and highlighting.
- Reused all-COCO flexible object parsing, validation, and Elasticsearch `range gte` filters.
- Reused semantic kNN request behavior with `clip_vector`, normalized injected text embeddings, and notebook `num_candidates=100`.
- Made Lab 6's `RN50` / `openai` / batch `64` embedding identity explicit without replacing or conflating the Phase 03 ViT-B-32 FAISS baseline.
- Added compatibility validation for frame IDs, OCR metric preconditions, normalized semantic vectors, and Elasticsearch candidate limits.
- Verified RRF one-based ranks/default `60` and weighted-fusion independent normalization, equal-score handling, endpoint ties, and configured alpha `0.25` without tuning.
- Kept EasyOCR, PaddleOCR/PaddlePaddle, Ultralytics, and Elasticsearch as optional dependency groups with lazy model/backend imports.
- Completed real CPU inference with EasyOCR, PaddleOCR, YOLOv8n, and exact Lab 6 RN50/OpenAI over five existing Phase 02 keyframes.
- Started checksum-verified Elasticsearch 8.11.4 locally with Lab 6's single-node/security/heap settings, recreated the preserved `keyframes` mapping, and indexed five real enriched documents.
- Executed real BM25, object, semantic kNN, RRF, and weighted retrieval requests and printed representative metadata-rich Top-K results.
- Added `tests/integration/test_lab6_runtime.py` as an opt-in reproducible real-runtime gate.
- Did not start FastAPI, UI, temporal retrieval, reranking, or any Phase 05+ work.

Remaining:
- None for Phase 04 packaging or runtime integration verification.
- The five-frame synthetic corpus produced no YOLO objects, so object retrieval executed successfully with zero hits; a positive real object-match case remains unobserved rather than fabricated.
- Retrieval quality remains unbenchmarked; Phase 03 and Phase 04 are engineering baselines only.

Tests:
- Baseline before Phase 04: `py -3.12 -m pytest -q`: 42 passed, 2 skipped.
- Verified runtime dependency command: `py -3.12 -m pip install -e ".[dev,semantic,search,ocr,objects]"`.
- Phase 04 focused tests: 37 passed.
- Full lightweight suite: `py -3.12 -m pytest -q`: 71 passed, 3 skipped (real OpenCLIP, real Lab 6 runtime, and optional TransNetV2 opt-in tests).
- Real Phase 04 runtime test: `$env:MVR_RUN_PHASE04_RUNTIME_TESTS='1'; py -3.12 -m pytest -q -s tests\integration\test_lab6_runtime.py`: 1 passed, 9 upstream deprecation/runtime warnings.
- Combined real-runtime suite: `$env:MVR_RUN_OPENCLIP_TESTS='1'; $env:MVR_RUN_PHASE04_RUNTIME_TESTS='1'; py -3.12 -m pytest -q`: 73 passed, 1 skipped, 9 upstream warnings (only optional TransNetV2 skipped).
- Runtime versions: EasyOCR 1.7.2, PaddleOCR 3.7.0, PaddlePaddle 3.3.1, Ultralytics 8.4.120, OpenCLIP 3.3.0, Elasticsearch client 8.19.3, and Elasticsearch server 8.11.4.
- Real RN50 output: five 1024-dimensional normalized `float32` image vectors; Elasticsearch index count: five.
- Read-only source integrity count/size remained unchanged at 11,177 files / 1,221,999,740 bytes.

Blockers:
- None for Phase 04 runtime execution. The smoke corpus has no YOLO-detectable objects, so positive-hit object retrieval was not demonstrated; the real `person:1` branch was executed and returned zero results.

---

## Phase 05 — API and Tests

Status: COMPLETE

Commit: `feat: reuse FastAPI serving baseline` (this phase commit; hash reported after commit)

Completed:
- Reused the FastAPI Lab application/Pydantic/TestClient structure and Lab 6 health, BM25, object, RRF hybrid, lazy CLIP, validation, stable hit, OpenAPI, and static-file patterns.
- Added `GET /health` and the required `POST /search/semantic` backed by the persisted Phase 03 FAISS index and its manifest-selected OpenCLIP model.
- Added `POST /search/text`, `/search/object`, and `/search/hybrid` only when explicitly configured Elasticsearch is reachable and the configured index exists at application startup.
- Kept index/model loading lazy and reusable; request handlers call existing retrieval services instead of rebuilding indexes or reimplementing retrieval mathematics.
- Added explicit Pydantic request/response schemas, nonblank query and numeric bound validation, framework-neutral object-error translation to HTTP `400`, and runtime failures as HTTP `503`.
- Preserved Lab 6 hybrid API defaults: branch candidate size `30`, RRF `60`, and API kNN candidates `200`; no model, retrieval, or fusion tuning was performed.
- Added optional real static JPEG serving from `MVR_KEYFRAME_DIR` and adapted URLs to the Phase 02 nested keyframe layout.
- Added twelve lightweight API contract tests using injected retrieval services and changed CI to install only the lightweight `dev,api` extras.
- Verified that `httpx2` is the intended modern Starlette TestClient dependency and used a compatible range rather than copying the lab pin blindly.
- Verified real Uvicorn startup and every implemented endpoint over the existing five-keyframe FAISS/Elasticsearch smoke corpus.
- Updated README with only the exact startup and request commands exercised in Phase 05.
- Did not begin UI, query understanding, temporal retrieval, reranking, submission, deployment, publication, or Phase 06.

Remaining:
- None for Phase 05.
- Retrieval quality remains unbenchmarked; the API smoke run verifies serving/runtime integration only.

Tests:
- Baseline before Phase 05: `py -3.12 -m pytest -q`: 71 passed, 3 skipped.
- Test-first red gate: `py -3.12 -m pytest -q tests\unit\api\test_app.py` failed at collection because FastAPI and the serving modules did not yet exist.
- Focused API suite: `py -3.12 -m pytest -q tests\unit\api\test_app.py`: 12 passed.
- API compilation: `py -3.12 -m compileall -q src\mvr\api`: passed.
- Lightweight suite after implementation: `py -3.12 -m pytest -q`: 83 passed, 3 skipped (real OpenCLIP, real Lab 6 runtime, and optional TransNetV2 opt-in tests).
- Independent Phase 05 code review: all four Important findings were fixed; follow-up assessment reported no Critical/Important issues and readiness `Yes`.
- Real Uvicorn health: semantic configured/available and initially lazy; Elasticsearch available with 5 documents; semantic reported loaded after the first request.
- Real semantic POST `a man riding a motorcycle`: 3 hits; rank 1 `L01_V001_S0002_K01` at `1.56s`, score `0.16651125252246857`.
- Real text POST `00`: 1 hit; `L01_V001_S0001_K01` with OCR text `00:00:00.480 12`, score `1.6309344`.
- Real object POST `person:1`: request executed successfully with 0 hits, consistent with the Phase 04 smoke corpus.
- Real hybrid POST `00`, `rrf_k=60`: 3 hits; rank 1 `L01_V001_S0001_K01`, RRF score `0.032266458495966696`.
- Malformed object POST returned `400`; static keyframe request returned `200 image/jpeg`; OpenAPI contained all five implemented routes.

Blockers:
- None. Elasticsearch is required only for the conditionally exposed text/object/hybrid routes.

---

## Phase 06 — Verify and Publish

Status: COMPLETE

Commit: `chore: add agent_output to gitignore and record Phase 06 completion` (hash recorded after push)

Completed:
- Confirmed branch `feat/reuse-lab-baseline` with local HEAD and `origin/feat/reuse-lab-baseline` both at `83fe5e1` before the phase started.
- Ran full branch diff against `origin/main` (`3c64022`): 64 files changed, 6706 insertions, 14 deletions — all legitimate source/doc/test/config/CI files.
- Verified no generated artifacts, model weights, datasets, raw videos, FAISS indexes, embeddings, `.env`, secrets, or Elasticsearch runtime data are tracked. Only `.gitkeep` placeholders exist under ignored artifact directories.
- Found `agent_output/` (AI agent output directory) was untracked but not gitignored; added `agent_output/` to `.gitignore` — this is the only legitimate Phase 06 repository change.
- Ran lightweight test suite: **83 passed, 3 skipped** in 2.32s.
- 3 skipped: real OpenCLIP opt-in test (`MVR_RUN_OPENCLIP_TESTS` not set), real Lab 6 runtime test (`MVR_RUN_PHASE04_RUNTIME_TESTS` not set), and optional TransNetV2 runtime.
- Verified all documented README commands against actual implementation:
  - `py -3.12 -m mvr.cli --help` → passed, shows `ingest`, `build-index`, `search`.
  - `py -3.12 -m mvr.cli ingest --help` → passed, all flags present.
  - `py -3.12 -m mvr.cli build-index --help` → passed, all flags present.
  - `py -3.12 -m mvr.cli search --help` → passed, all flags present.
  - `import mvr.api.app` → passed.
  - `pytest -q` → passed.
- Verified GitHub CLI (`gh` 2.97.0) authenticated as `Tobieeeeee` with `repo` scope.
- Verified remote `origin` is the intended repository `https://github.com/Tobieeeeee/AIC26.git`.
- Committed `.gitignore` and this progress update.
- Pushed `feat/reuse-lab-baseline` to `origin`.
- Created Pull Request `feat/reuse-lab-baseline → main` titled `feat: build runnable MVR baseline from existing labs`.
- Did not merge the PR.

Remaining:
- None for Phase 06.
- PR merge to `main` requires explicit human approval.
- Retrieval quality benchmarking is future work.
- TransNetV2 real runtime verification remains optional/deferred.

Tests:
- Lightweight suite: `py -3.12 -m pytest -q`: **83 passed, 3 skipped** in 2.32s.
- Real OpenCLIP tests: skipped (opt-in, requires `MVR_RUN_OPENCLIP_TESTS=1`).
- Real Lab 6 runtime tests: skipped (opt-in, requires `MVR_RUN_PHASE04_RUNTIME_TESTS=1`).
- Optional TransNetV2 runtime: skipped (intentional; heavy optional dependency).

Git:
- Branch: `feat/reuse-lab-baseline`
- Remote: `origin → https://github.com/Tobieeeeee/AIC26.git`
- Push: COMPLETE
- PR: CREATED (feat/reuse-lab-baseline → main)

Blockers:
- None.

---

# Current active phase

Phase 06 — Verify and Publish: COMPLETE

All planned phases are complete.

Next action: human review and merge of the open Pull Request.

Stop gate: do not begin post-baseline improvements without explicit approval.

# Resume instruction

If work is interrupted, read:

```text
AGENTS.md
codex/00_MASTER_PLAN.md
docs/CODEX_PROGRESS.md
```

All six phases are complete. The feature branch is pushed and a PR to `main` is open.
