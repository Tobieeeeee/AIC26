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

Commits:
1. `chore: add agent_output to gitignore and record Phase 06 completion` (`5522202`) — initial Phase 06 commit
2. `chore: extend gitignore for agent_task and requirements-dev; record full Phase 06 E2E verification` — E2E/fresh-clone results

Completed:
- Confirmed branch `feat/reuse-lab-baseline` with local HEAD and `origin/feat/reuse-lab-baseline` both at `83fe5e1` before the phase started.
- Ran full branch diff against `origin/main` (`3c64022`): 64 files changed, 6706 insertions, 14 deletions — all legitimate source/doc/test/config/CI files.
- Verified no generated artifacts, model weights, datasets, raw videos, FAISS indexes, embeddings, `.env`, secrets, or Elasticsearch runtime data are tracked. Only `.gitkeep` placeholders exist under ignored artifact directories.
- Added `agent_output/`, `agent_task/`, and `requirements-dev.txt` to `.gitignore` — these are AI agent and developer convenience files, not source.
- Verified no hard-coded local absolute paths exist in any production source file (`src/`).
- Ran Phase 05 API runtime verification (separate task): confirmed `/health` was degraded only due to missing env vars; no production code changes needed.
- **E2E smoke path verified** (Phase 06 task):
  - Artifacts present: `artifacts/indexes/phase03-smoke/` (4 files) and `artifacts/keyframes/phase02-smoke/keyframes/L01_V001/` (5 JPEGs) ✅
  - Elasticsearch 8.11.4 at `http://localhost:9200`, `keyframes` index with 5 documents ✅
  - Uvicorn started with all 5 env vars configured ✅
  - `GET /health` → `status=ok`, `semantic.configured=true, available=true`, `elasticsearch.configured=true, available=true, documents=5` ✅
  - `POST /search/semantic` `"a man riding a motorcycle"` → rank 1 `L01_V001_S0002_K01` at `1.56s`, score `0.16651` (matches Phase 03 baseline exactly) ✅
  - `POST /search/text` `"00"` → rank 1 `L01_V001_S0001_K01`, OCR `00:00:00.480 12`, score `1.6309` (matches Phase 04/05) ✅
  - `POST /search/hybrid` `"00", rrf_k=60` → rank 1 `L01_V001_S0001_K01`, RRF score `0.03226` ✅
  - `GET /keyframes/L01_V001/L01_V001_S0002_K01.jpg` → `200 image/jpeg` ✅
- **Fresh-clone verification**:
  - Cloned `https://github.com/Tobieeeeee/AIC26.git` into `AIC26-clean-verify/`
  - Checked out `feat/reuse-lab-baseline`
  - Created new `.venv-verify` Python 3.12 environment
  - Installed `.[dev,api]` — no undeclared dependencies, no missing source files
  - `import mvr` → OK in clean clone
  - `python -m pytest -q -rs` → **75 passed, 11 skipped** — PASS
    - 3 expected skips (real OpenCLIP, real Lab 6 runtime, optional TransNetV2)
    - 8 additional FAISS skips: expected — `faiss-cpu` is in the `semantic` extra, not `dev,api`; tests correctly self-skip when optional extras are absent
    - 0 failures ✅
- Ran full test suite in working repo: **83 passed, 3 skipped** in 1.20s.
- Committed and pushed final `.gitignore` and progress updates.
- All six planned phases verified; PR #1 open for human merge.

Remaining:
- None for Phase 06.
- PR merge to `main` requires explicit human approval.
- Retrieval quality benchmarking is future work.
- TransNetV2 real runtime verification remains optional/deferred.
- Fresh-clone `semantic` extra tests require `.[dev,api,semantic]` install to be fully exercised.

Tests:
- Working repo normal suite: `py -3.12 -m pytest -q`: **83 passed, 3 skipped** in 1.20s.
- Fresh-clone `.[dev,api]` suite: **75 passed, 11 skipped** (8 FAISS skips expected without `semantic` extra) — PASS.
- Real OpenCLIP tests: previously PASS (`MVR_RUN_OPENCLIP_TESTS=1`).
- Real Lab 6 runtime tests: previously PASS (`MVR_RUN_PHASE04_RUNTIME_TESTS=1`).
- Phase 05 API runtime verification: PASS (separate task, no code changes).
- Phase 06 E2E smoke: PASS.

Git:
- Branch: `feat/reuse-lab-baseline`
- Remote: `origin → https://github.com/Tobieeeeee/AIC26.git`
- Push: COMPLETE
- PR: CREATED — https://github.com/Tobieeeeee/AIC26/pull/1 (OPEN, not merged)

Blockers:
- None.

---

---

# Post-baseline work: Textual KIS Competition-Ready task pack

Status: COMPLETE (all 9 tasks)

Branch: `feat/kis-competition-ready` (branched from `main` after PR #1 merged)

This work followed a separate task pack (kept out of Git as agent working
material, like `codex/` phases but for KIS) rather than the `codex/` phase
files. It was explicitly approved post-baseline work, executed one gated task
at a time with a stop-and-report boundary after each.

Commits, in order:

1. `9b373dc` — Task 01, repository contract audit → `docs/AIC_KIS_IMPLEMENTATION_NOTES.md`
2. `37576a6` — Task 02, AIC original-frame registry / organizer keyframe adapter
3. `bfcf125` — Task 03, official KIS evaluator (R-Score, R@k, per-query Final Score)
4. `dc297b1` — Task 04, KIS orchestration, Top-100 contract, `POST /search/kis`
5. `9e688c0` — Task 05, exact-frame refinement on the original video
6. `77c89aa` — Task 06, real-data KIS benchmark and refinement decision
7. `47cd694` — Task 07, prediction contract validation and local export
8. `245bda6` — Task 08, real AIC KIS end-to-end verification
9. `6551e58` — Task 09, `docs/AIC_KIS_FINAL_REPORT.md`

Completed:

- Audited and reused the existing retrieval hit types, probe/extract utilities, and
  API conventions instead of creating parallel abstractions.
- Added an original-video frame registry that resolves organizer keyframes via the
  verified `map_keyframes` `frame_idx` column (never the `n` ordinal or filename),
  keyed compositely on `(video_id, keyframe_ref)`, and never guesses a missing mapping.
- Implemented the documented competition scoring rule exactly, deliberately not
  reusing the generic ID-set Recall@K/MRR in `evaluation/metrics.py`.
- Exposed Top-100 KIS predictions through both a service and an always-registered
  `POST /search/kis` route that reuses the existing lazily-loaded index/model.
- Implemented and real-video-verified exact-frame refinement, then benchmarked it.
- Added prediction validation plus a clearly-labeled local verification export;
  no organizer transport format was invented.
- Created `src/mvr/submission/`, which did not previously exist.

Tests (current state, after the PR #2 hardening pass recorded below):

- Working repo (with `semantic` extras installed): 173 passed, 5 skipped, 0 failed.
- CI-equivalent environment (`.[dev,api]` only, matching `.github/workflows/ci.yml`):
  165 passed, 13 skipped, 0 failed — the 8 extra skips are the existing
  faiss-dependent tests, consistent with the Phase 06 fresh-clone result.
- Real AIC KIS E2E (`MVR_RUN_AIC_KIS_E2E=1`): PASS in 60.37s.
- Real exact-frame refinement decode (`MVR_RUN_FRAME_REFINEMENT_REAL_TESTS=1`): PASS,
  cross-verified against an independent OpenCV decode of the same frame index.
- Real `POST /search/kis` smoke over HTTP against a live Uvicorn process: PASS.
- Real benchmark over 29 real L21 videos / 7800 real keyframes: executed; results and
  the refinement decision are recorded in `docs/AIC_KIS_FINAL_REPORT.md`.

Remaining / explicitly out of scope:

- Hybrid and text/OCR modes were not benchmarked on real data in this environment
  (no local Java/Elasticsearch/OCR runtime); the code paths exist and are unit-tested.
- Exact-frame refinement is correct but not latency-viable (measured ~205s/query mean)
  and is therefore not enabled by default; a seek-based decode would be needed first.
- Ground truth is a small team-curated internal set, not organizer-labeled.
- Q&A, TRAKE/temporal, Frontend, VLM, and deployment remain untouched per the pack's
  stop gate.

Blockers:

- None.

## PR #2 pre-merge hardening pass

Status: COMPLETE

Commit: `b6c35b1` — `fix: harden KIS correctness and candidate handling`

An independent review of PR #2 (then-head `50cdc47`) identified three issues.
Each was re-verified as a real defect against the actual code before being
fixed, rather than accepted on the review's word:

1. **Weak E2E correctness assertion.** The real E2E test asserted only
   `0.0 <= final_score <= 1.0`, which a completely wrong retrieval also
   satisfies. The known query's correct answer was measured to sit at real
   rank 6, so the test now asserts `score.recall_at_k[20] == 1.0` — the
   strongest official cutoff supported by evidence — and fails on regression.
2. **Coarse and refinement scores were mixed.** `refine_top_candidates` wrote
   the refinement window's local similarity into `KisCandidate.score`,
   discarding the coarse retrieval score and conflating two non-comparable
   score spaces (harmless while only semantic cosine is used, dangerous once
   hybrid RRF scores are refined). `KisCandidate` gained a separate
   `refinement_score: float | None`; `score` is now immutable across refinement.
3. **Top-100 overfetch was capped at 100.** `search_kis`'s internal pool was
   `min(top_k * 3, 100)`, so at `top_k=100` no overfetching occurred at all
   and unmapped/duplicate hits inside the first 100 raw ranks could starve the
   output below 100. Now `max(top_k * 3, 30)`, uncapped.

Tests: 8 added. Focused suites 49 passed. Real AIC KIS E2E re-verified PASS
(72.67s) with the strengthened assertion. Real frame-refinement decode
re-verified PASS (47.67s). Full regression 173 passed / 5 skipped / 0 failed;
CI-equivalent 165 passed / 13 skipped / 0 failed.

Handoff gap closed in the same pass: the README told a reader to point
`MVR_SEMANTIC_INDEX` at `artifacts/indexes/aic_l21`, but nothing documented or
exposed how to *create* that index from an AIC organizer dataset — it had only
ever been built by an uncommitted ad-hoc script, so no teammate could
reproduce it. `python -m mvr.cli build-aic-index` now exposes the existing,
already-tested `build_aic_keyframe_mapping_csv` + `build_semantic_index` path
(discovering every video that has both local keyframes and a mapping CSV), and
the README documents it. Verified for real on `L21_V001`: 307 vectors, all
four index files written.

Deliberate scope decisions (recorded so they are not mistaken for oversights):

- **Fix 3 has no measured effect on the current benchmark.** Re-running all six
  internal queries against the live code confirmed identical R@k and an
  identical mean Final Score of `0.6333`, because every query already returned
  a full 100 candidates under the old formula — the current index is built from
  organizer keyframes, all of which resolve, so there were no unmapped losses
  to recover. Fix 3 removes a latent defect; it is not a score improvement, and
  the benchmark table in `docs/AIC_KIS_FINAL_REPORT.md` remains accurate.
- **Fix 3 only fully applies to the semantic path.** The Elasticsearch hybrid
  branch is still bounded by the pre-existing baseline `TOP_N_EACH = 30`
  per-branch candidate limit in `mvr.api.app`, which caps the fused union at
  roughly 60 regardless of the KIS overfetch value. That constant is Lab 6
  baseline code and was intentionally left untouched.
- **`refinement_score` is intentionally not exposed on `KisHit`.** The
  `POST /search/kis` route does not run refinement (it is default-off for
  latency), so no data is lost today. If refinement is ever wired into that
  route, `KisHit` must be extended at the same time or the field will silently
  disappear from API responses.

Blockers:

- None.

---

# Current active phase

None. Phases 01–06 (reuse baseline) are COMPLETE and merged via PR #1. The
Textual KIS competition-ready task pack (Tasks 01–09) is also COMPLETE on
`feat/kis-competition-ready`.

Next action: human review of `feat/kis-competition-ready`.

Stop gate: do not begin Q&A, TRAKE, Frontend, VLM, advanced reranking, or
deployment without explicit approval.

# Resume instruction

If work is interrupted, read:

```text
AGENTS.md
codex/00_MASTER_PLAN.md
docs/CODEX_PROGRESS.md
docs/AIC_KIS_IMPLEMENTATION_NOTES.md
docs/AIC_KIS_FINAL_REPORT.md
```
