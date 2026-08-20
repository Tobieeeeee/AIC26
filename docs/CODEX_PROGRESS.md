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

# Post-baseline work: per-keyframe ASR features

Status: CODE COMPLETE; CPU REAL RUNTIME VERIFIED

Branch: `feat/keyframe-asr`

Completed:

- Added a provider-separated per-keyframe pipeline under
  `src/mvr/enrichment/asr/`: authoritative mapping adapters, FFmpeg mono 16 kHz
  window decoding, Silero VAD, nearby-region merge, exact Vietnamese
  ChunkFormer wrapping, canonical JSON/artifact validation, atomic writes,
  manifest/reporting, resume, and force behavior.
- Preserved the required semantic contract: every keyframe independently runs
  `[t-3s,t+3s]` (clamped) -> VAD -> ASR only on speech regions. No full-video
  transcript alignment path was introduced.
- Reused the existing repo ingestion `keyframe_mapping.csv` and
  `video_metadata.csv` contracts and the existing organizer
  `AicDatasetRegistry`. Organizer `map_keyframes` supplies `n`, `pts_time`, and
  `frame_idx`; because it has no `shot_id`, canonical organizer ASR artifacts
  explicitly use `null` rather than guessing.
- Added `build-asr-features` to `mvr.cli`, optional `asr` dependencies and
  defaults, a pinned CPU Dockerfile, detailed runtime documentation, 15 offline
  tests, and one opt-in real integration test.
- No lab implementation was migrated for this task, and OCR/object/visual/
  retrieval behavior was not modified.

Verification:

- Focused ASR suite after BOM regression coverage: 15 passed, 1 skipped.
- Full regression suite after syncing `origin/main`: 253 passed, 6 skipped, 0 failed.
- Python compileall and `git diff --check`: passed.
- Dependency resolution reproduced an upstream blocker: clean installation of
  `chunkformer==1.2.2` reaches its unconditional `deepspeed>=0.14` dependency,
  whose isolated build cannot import the already installed Torch on the local
  Windows/Python 3.13 host. The Dockerfile applies the minimal isolated Linux
  workaround: pinned CPU Torch first, then pinned DeepSpeed with build
  isolation disabled, then the project ASR extra.
- Built `mvr-asr:cpu` successfully from `docker/asr/Dockerfile.cpu`.
- Ran a real Docker smoke test over five authoritative keyframe mappings from
  `L21_V001` and its original video. The pinned Silero VAD and exact
  `khanhld/chunkformer-ctc-large-vie` revision loaded successfully on CPU.
- Smoke manifest: 5 keyframes total, 4 with speech, 1 without speech, 0 failed,
  4 segments, 4 ASR calls, and 69.8 seconds elapsed. `errors.json` contained no
  errors or warnings. All five canonical per-keyframe JSON artifacts were
  generated, including the required empty artifact for the no-speech frame.
- A second real run over keyframes 6-10 loaded the model entirely from the
  read-only cache and produced 5 speech artifacts, 8 segments, 8 ASR calls,
  and 0 failures in 73.2 seconds. It exposed and then regression-verified CSV
  header handling for both UTF-8 and UTF-8-with-BOM organizer mappings.
- A third real run over keyframes 11-20 produced 10 speech artifacts, 14
  segments, 14 ASR calls, and 0 failures in 88.0 seconds. Across all three
  runs, 20 real `L21_V001` keyframes have now been verified.

Remaining runtime gates:

- GPU visibility and CUDA inference have not been verified.
- The 20-keyframe smoke runs verify execution and schema, not ASR quality or
  word-error rate. Full-dataset inference was intentionally not run.

## Task05A v1 — Feature Filtering & Index Document Preparation

Status: COMPLETE

Branch: `feat/filtering`

Completed:

- Added reusable NFC text normalization, OCR/ASR filtering and intra-keyframe
  deduplication, Task04 object projection, CLIP vector/mapping validation, and
  Task05A document validation under `src/mvr/indexing/`.
- Added `scripts/indexing/build_index_documents.py` with configurable Task04
  branch roots and filter thresholds; it performs no model inference and does
  not create an Elasticsearch/Milvus index.
- Preserved one output document per canonical mapping keyframe even when OCR,
  ASR, Object, or CLIP content is absent or invalid.
- Added deterministic UTF-8 atomic JSON output, `manifest.json`, bounded
  `filtering_report.json`, independent lexical/vector/object flags, and no
  JSONL output.
- Audited the real current Task04 artifacts. The selected canonical mapping has
  677 keyframes and OCR has 677 matching files. CLIP/Object artifacts were
  produced from a different 643-keyframe TransNetV2 mapping, so incompatible
  rows are reported rather than silently joined. No ASR artifact or confidence
  field exists in the current artifact set.
- Executed the real run into ignored
  `artifacts/processed_real_task04_test/`: 677 documents, 365 lexical-indexable,
  55 vector-indexable, and 55 object-indexable.
- Verified exact set cardinality (677 expected/actual/unique, zero missing,
  extra, or duplicate documents), filtering arithmetic, schema validation,
  UTF-8 decoding, and deterministic rerun content.
- Verified Task04 immutability using aggregate SHA-256 hashes across the
  canonical mapping and the CLIP/Object/OCR source trees; all pre/post hashes
  matched.

Tests:

- Focused Task05A tests: 16 passed.
- All indexing unit tests: 21 passed, 7 skipped.
- Full regression: 246 passed, 13 skipped, 0 failed.
- `python -m compileall -q src/mvr/indexing scripts/indexing`: passed.
- CLI help, `git diff --check`, no-JSONL, artifact staging, and scope audits:
  passed.

Known input limitation:

- The real Task04 corpus is not currently a single identity-compatible feature
  package: CLIP/Object use the 643-row TransNetV2 mapping while OCR/canonical
  ingestion use the 677-row mapping. Task05A correctly isolates those branch
  mismatches; upstream Task04 artifacts must be regenerated against one
  canonical mapping to make all 677 documents vector/object indexable.

Blockers:

- None for Task05A v1 implementation. The upstream branch identity mismatch and
  absent ASR artifact are recorded limitations, not silently repaired.

## Task05B v1 — Elasticsearch/BM25 Lexical Index and Retrieval

Status: COMPLETE

Branch: `feat/lexical`

Completed:

- Added a Task05B-only Elasticsearch projection/index builder under
  `src/mvr/indexing/lexical_index.py`. It reads only Task05A processed
  documents and trusts `indexability.lexical` without recomputing it.
- Projected canonical `keyframe_id`, `video_id`, `frame_idx`, nullable
  `shot_id`, `timestamp`, `ocr.joined -> ocr_text`, and
  `asr.joined -> asr_text`; Elasticsearch `_id` is exactly `keyframe_id`.
- Added an isolated default-BM25 mapping with exactly two text fields,
  `ocr_text` and `asr_text`. No custom analyzer, boost, or combined text field
  was introduced.
- Added create/recreate, bulk failure reporting, one final refresh, exact
  count/ID-set auditing, and a provenance manifest under the generated
  fasttest lexical-index artifact directory.
- Added deterministic Task05B retrieval under
  `src/mvr/retrieval/lexical_bm25.py`: equal-weight `multi_match` over the two
  fields, `_score DESC`, then `keyframe_id ASC`, strict query/top-k validation,
  and canonical result mapping.
- Added reproducible build and verification CLIs under `scripts/indexing/`.
- Reused the existing repository Elasticsearch 8.x client/bulk lifecycle
  conventions while keeping the legacy Lab 6 vector/object/boosted mapping and
  retrieval path unchanged.
- Real Elasticsearch 8.11.4 fasttest acceptance indexed all 5 lexical Task05A
  documents with 0 failed, missing, extra, or duplicate IDs.
- Field projection audit passed for every required identity/text field, and
  `combined_text` was absent.
- OCR-oriented `TIN CHÍNH`, ASR-oriented `đàn bò ở lâm đồng`, and no-match
  `quasarxylophone987654321` retrieval smoke tests passed. Repeated retrieval
  preserved deterministic ordering. No retrieval-quality claim was made.

Tests:

- Focused Task05B suite: 25 passed.
- Indexing/retrieval unit suites during implementation: 99 passed, 8 skipped.
- Full regression: 288 passed, 14 skipped, 0 failed.
- Python compileall, CLI help, `git diff --check`, identity/cardinality,
  manifest, scope, and generated-artifact staging audits: passed.

Known limitations:

- Baseline uses Elasticsearch default text analysis/BM25 only.
- No field-weight or language-analyzer ablation was performed.
- The real acceptance corpus has only five keyframes and demonstrates
  engineering behavior, not retrieval quality.

Blockers:

- None for Task05B v1.

## Multi-Embedding Task 03 — Frozen Five-Keyframe Fasttest Baseline

Status: COMPLETE

Branch: `feat/rebuild-embedding-models`

Completed:

- Added a deterministic structural validator for the existing five-keyframe
  fasttest under `src/mvr/evaluation/fasttest_baseline.py` and a reproducible
  CLI at `scripts/indexing/verify_fasttest_baseline.py`.
- Added `configs/fasttest_baseline.json` with repository-relative paths only;
  generated manifests containing developer-machine paths remain ignored and
  are not committed.
- Frozen the acceptance contract at 5 canonical mapping rows, 5 unique
  `keyframe_id` values, 5 valid Task05A documents, zero identity/schema
  issues, and—when the backend manifest is required—5 Task05B indexed
  documents with zero failed/missing/extra/duplicate IDs.
- Reused the existing Task05A document validator. Identity comparison is set
  based on authoritative `keyframe_id`; the validator never derives identity
  from filenames or positional row joins.
- Kept the structural gate independent of Elasticsearch. The optional
  `--require-lexical` flag turns the Task05B manifest from a documented skip
  into a hard gate.
- Added deterministic, negative, optional-backend, and path-safety tests plus
  `docs/FASTTEST_BASELINE.md` with the exact rerun commands.

Verification:

- Real ignored fasttest:
  `python scripts/indexing/verify_fasttest_baseline.py --require-lexical` —
  PASS; canonical 5/5, Task05A 5 with `invalid_schema=0`, Task05B indexed 5,
  and all failed/missing/extra/duplicate counts 0.
- Focused Task03 plus Task05A/Task05B regression suites: 51 passed.
- Full regression in the verified Python 3.11 Docker environment: 296 passed,
  14 skipped, 0 failed.
- Python compilation and `git diff --check`: passed.

Artifacts:

- `artifacts/fasttest/` remains local/ignored and was neither modified for
  the implementation nor staged. No vectors, model weights, index data, OCR
  runtime output, or generated Task05A documents are part of this task diff.

Blockers:

- None for Task 03. The five-keyframe baseline is a structural regression
  gate, not retrieval-quality evidence.

# Multi-Embedding Task 04 — Provider Artifact Schema

Status: COMPLETE

Branch: `feat/rebuild-embedding-models`

Completed:

- Added a provider-neutral manifest and feature-mapping contract under
  `src/mvr/enrichment/embeddings/artifacts.py`.
- Added deterministic discovery for independent provider directories under
  `features/embeddings/<model_id>/`, with no branches for BEiT-3,
  MetaCLIP-2, or Perception Encoder.
- Validation rejects unsafe IDs/paths, missing fields, unsupported dtypes,
  invalid dimensions, duplicate IDs, directory mismatches, malformed
  mappings, and optional checksum mismatches.
- Added a comma-separated provider-list contract to the unified feature CLI
  and default configuration. Providers that are not implemented yet fail
  explicitly and never fall back to CLIP.
- Documented the artifact tree, identity rules, and validation behavior in
  `docs/MULTI_EMBEDDING_ARTIFACTS.md`.

Verification:

- Tasks 01–03 baseline before edits: 51 passed.
- Provider schema plus Tasks 01–03 regression: 65 passed.
- Task 04 focused suite including the unified runner: 20 passed.
- Full regression: 314 passed, 14 skipped, 0 failed.
- Python compilation and `git diff --check`: passed.

Blockers:

- None for the schema/discovery layer. Provider model specifications and
  inference remain deliberately deferred to Tasks 05–08.

# Multi-Embedding Task 05 — Locked Provider Specifications

Status: COMPLETE

Branch: `feat/rebuild-embedding-models`

Completed:

- Added a runtime-readable, versioned provider matrix for BEiT-3 base ITC,
  distilled MetaCLIP-2 worldwide B/16, and PE Core L/14 336.
- Locked provider-specific image/text entry points, source/checkpoint revisions,
  preprocessing, tokenizers, dimensions, dtype, normalization, similarity,
  licenses, access constraints, and dependency environments.
- Added strict no-download config parsing and Task 04 artifact-compatibility
  validation.
- Documented only authoritative repository/model-card evidence and explicit
  upstream reproducibility risks. No model was substituted or downloaded.

Verification:

- Provider contract plus Task 04/Task05A/Task05B regression: 77 passed.
- Python compilation and `git diff --check`: passed.

Risks:

- Microsoft's BEiT-3 blob has no published immutable revision/SHA256 and its
  official dependency list is partially unpinned. Task 06 must isolate the
  runtime and hash the downloaded file before accepting real artifacts.
- MetaCLIP-2 is CC-BY-NC-4.0; competition/deployment use must comply.
- PE's official source omits a `huggingface-hub` pin. Task 08 must record the
  resolved isolated environment.

# Multi-Embedding Task 06 — BEiT-3 Offline Extraction

Status: COMPLETE (real-weight fasttest unavailable locally)

Branch: `feat/rebuild-embedding-models`

Completed:

- Added a lazy BEiT-3 image adapter for the locked Microsoft UniLM retrieval
  checkpoint, official 224-pixel preprocessing, eval mode, and inference mode.
- Added a provider-neutral atomic offline writer that validates identities and
  images before inference; writes per-video shards plus deterministic mapping;
  validates shape/dtype/finiteness/normalization; and publishes the manifest
  last via an atomic directory rename.
- Added checkpoint/mapping SHA256 and source/checkpoint provenance to emitted
  manifests, plus an identity/vector artifact auditor.
- Wired `beit3` into the unified feature CLI with explicit local checkpoint,
  provider spec, batch size, and device arguments. No fallback or implicit
  model download is allowed.
- Added optional dependencies, unit/negative/deterministic coverage, an
  external-weight five-keyframe target, and an operator guide.

Verification:

- Task 06 focused: 11 passed, 1 skipped (external checkpoint); relevant
  Tasks 01–05 regression: 77 passed.
- Unified runner focused tests: 16 passed.
- Full regression: 336 passed, 15 skipped, 0 failed.
- CLI help/import smoke, Python compilation, and `git diff --check`: passed.

Limitations:

- The local dataset contains no locked BEiT-3 checkpoint or tokenizer, so the
  real five-keyframe target is correctly skipped. Mocked extraction exercises
  the complete artifact writer and audit contract without fake model weights.

# Multi-Embedding Task 07 — MetaCLIP-2 Offline Extraction

Status: COMPLETE (real-weight fasttest unavailable locally)

Branch: `feat/rebuild-embedding-models`

Completed:

- Added a lazy Transformers adapter for the pinned distilled MetaCLIP-2
  worldwide B/16 model and its same-revision processor.
- Kept model access local-only by default, with explicit download opt-in and
  fail-closed errors; no fallback to CLIP, BEiT-3, or another MetaCLIP variant.
- Reused the provider-neutral atomic writer/auditor while retaining the locked
  MetaCLIP-2 dimension, preprocessing, normalization, checkpoint revision, and
  independent output namespace.
- Wired independent runner selection, optional dependencies, provider-specific
  negative/deterministic tests, external-weight fasttest, and operator guide.

Verification:

- Task 07 plus Tasks 01–06 relevant regression: 96 passed, 1 skipped
  (external checkpoint).
- Unified runner/CLI focused suite: 14 passed; CLI help/import smoke passed in
  the project PyTorch image.
- Python compilation and `git diff --check`: passed.

Limitations:

- No pinned MetaCLIP-2 cache exists locally, so real five-keyframe inference is
  skipped rather than downloading weights or fabricating an inference result.

# Multi-Embedding Task 08 — Perception Encoder Offline Extraction

Status: COMPLETE (real-weight fasttest unavailable locally)

Branch: `feat/rebuild-embedding-models`

Completed:

- Added a lazy PE Core L/14-336 adapter using the official paired CLIP class,
  provider-specific transform, `normalize=True`, eval mode, and inference mode.
- Enforced the published checkpoint SHA256 before runner inference and recorded
  checkpoint/source provenance in the independent PE artifact manifest.
- Added a unified PowerShell feature wrapper supporting configurable provider
  lists, all three providers together, and each provider independently.
- Added explicit embedding-only runner mode, optional dependencies,
  provider-specific negative/deterministic tests, external-weight fasttest,
  and operator documentation.

Verification:

- Provider-focused and baseline regression suite: 105 passed, 1 skipped
  (external PE checkpoint fasttest).
- Unified runner/CLI smoke suite: 15 passed; CLI help/import smoke passed in
  the project PyTorch image.
- PowerShell wrapper parsing, Python compilation, and `git diff --check`:
  passed.

Limitations:

- The 2.68 GB PE checkpoint is not present locally, so real five-keyframe
  inference is skipped. No checkpoint, vector, cache, or generated artifact was
  downloaded or committed.

# Multi-Embedding Task 09 — Task05A Multi-Embedding Validation

Status: COMPLETE

Branch: `feat/rebuild-embedding-models`

Completed:

- Migrated Task05A from one legacy CLIP vector flag to deterministic discovery
  and independent validation of every provider under `features/embeddings/`.
- Added provider-specific locators and indexability while retaining the legacy
  aggregate flag as exactly `any(indexability["embeddings"].values())`.
- Validated mapping checksum/identity/cardinality, manifest dimension/dtype/
  normalization, locator bounds, finite vectors, and required/optional policy.
- Kept the pre-Task04 CLIP artifact as an explicit compatibility adapter when
  no provider artifact exists; provider failures never fall back to it.
- Added negative, isolation, deterministic reordered-mapping, aggregate-flag,
  and five-keyframe three-provider coverage. ASR auxiliary JSON exclusion
  remains regression-tested.

Verification:

- Focused Task05A suite: 28 passed.
- Task05A/Task05B/fasttest regression: 61 passed.
- Existing ignored five-keyframe structural/lexical fasttest: PASS with 5
  canonical documents, zero invalid schema, and zero missing/extra/duplicates.
- Python compilation and `git diff --check`: passed.

Blockers:

- None.

# Multi-Embedding Task 10 — Provider-Specific Semantic Indexes

Status: COMPLETE

Branch: `feat/rebuild-embedding-models`

Completed:

- Added a provider-isolated semantic index builder gated by the Task05A
  provider flag, current provider manifest hash, canonical `keyframe_id`,
  dimension, dtype, finite-value, normalization, and locator checks.
- Added environment/dataset/provider collection namespaces and a routing guard
  that rejects cross-provider collections before backend calls.
- Added explicit create/rebuild/append policies, batching, partial-failure
  accounting, provenance manifests, and a persistent exact FAISS backend.
- Added a multi-provider build CLI and operator documentation.

Verification:

- Fake-backend unit suite: 8 passed.
- Task09/Task10 plus real FAISS backend integration: 19 passed.
- Five-vector contracts independently indexed BEiT-3, MetaCLIP-2, and PE with
  zero missing/extra/duplicates/failures in the acceptance fixture.
- CLI help, Python compilation, and `git diff --check`: passed.

Blockers:

- None. Full-corpus index construction remains Task 15 and was not run.

# Multi-Embedding Task 11 — Provider Query Encoders

Status: COMPLETE (real-weight smoke unavailable locally)

Branch: `feat/rebuild-embedding-models`

Completed:

- Added lazy, cached online text adapters for the exact paired BEiT-3,
  MetaCLIP-2, and PE Core checkpoints/tokenizers.
- Added a typed `QueryEmbedding` contract carrying provider identity, vector,
  dimension, dtype, and normalized state.
- Added explicit empty-query policy, disabled/unavailable provider errors,
  dimension checks, and pre-encoding cross-model routing rejection.
- Added adapter-call contract tests, cache/failure isolation tests, and one
  query-to-three-matching-contract acceptance coverage.

Verification:

- Focused query encoder suite: 12 passed.
- Python compilation and `git diff --check`: passed.

Limitations:

- Locked BEiT-3, MetaCLIP-2, and PE weights/tokenizers are not all available
  locally, so real-weight three-provider query inference was not run. No model
  downloads were initiated.

# Multi-Embedding Task 12 — Independent Semantic Retrieval

Status: COMPLETE

Branch: `feat/rebuild-embedding-models`

Completed:

- Added provider-matched query encoding and index routing with one common
  result contract and three independent ranked lists.
- Added canonical metadata projection, top-k caps, 1-based contiguous ranks,
  metric-aware ordering, deterministic keyframe tie-breaks, and within-provider
  duplicate suppression.
- Added explicit fail-closed/degraded policies with provider coverage/errors;
  no provider fallback, raw-score normalization, or fusion was introduced.
- Added a persistent FAISS search adapter and real-backend round-trip coverage.

Verification:

- Focused independent retrieval suite: 10 passed.
- Task10–12 regression plus real FAISS build/search integration: 31 passed.
- Five-document, three-provider ranked-list acceptance fixture: passed with
  complete identities and matching `model_id` values.
- Python compilation and `git diff --check`: passed.

Blockers:

- None. Retrieval-quality evaluation remains out of scope for this task.

# Multi-Embedding Task 13 — Model-Level Semantic RRF

Status: COMPLETE

Branch: `feat/rebuild-embedding-models`

Completed:

- Added pure semantic RRF keyed by `keyframe_id`, using positive 1-based ranks
  and configurable `rrf_k` with default 60.
- Preserved canonical identity metadata and provider-rank provenance; raw
  provider scores do not enter the fusion calculation.
- Added deterministic duplicate warnings, identity conflict rejection,
  missing-provider policy handling, global tie-breaks, and serialization.
- Kept semantic model fusion separate from legacy BM25/object hybrid fusion.

Verification:

- Focused semantic RRF suite: 14 passed.
- Task12/13 plus legacy hybrid RRF regression: 30 passed.
- Hand-computed scores, raw-score invariance, provider-order permutations, and
  byte-equivalent deterministic serialization: passed.
- Python compilation and `git diff --check`: passed.

Blockers:

- None. No quality claim or `rrf_k` tuning was performed.

# Multi-Embedding Task 14 — Fasttest and Seven Ablations

Status: COMPLETE (structural fixture; real weights unavailable)

Branch: `feat/rebuild-embedding-models`

Completed:

- Added the seven required single/pair/triple provider configurations and a
  deterministic JSON/Markdown ablation report framework.
- Added separated warm-up, measured query, provider encoding/search, and fusion
  timing collection with median/p95 summaries and runtime/provenance metadata.
- Added a five-keyframe structural E2E test covering three offline artifacts,
  Task05A, three indexes, three online searches, and semantic RRF.
- Quality fields are explicitly `not evaluated` without representative ground
  truth; the conclusion makes no model-combination improvement claim.

Verification:

- Ablation and structural E2E suite: 6 passed.
- Committed ablation CLI/config: PASS; 7 rows, expected keyframes 5, quality
  not evaluated, bounded conclusion present.
- Python compilation and `git diff --check`: passed.

Limitations:

- Real three-provider artifacts are unavailable locally, so this is structural
  E2E with injected deterministic encoders/backends, not a model-quality run.

# Multi-Embedding Task 15 — Full-Corpus Build Workflow

Status: IMPLEMENTATION COMPLETE; EXPENSIVE EXECUTION BLOCKED BY HARD GATES

Branch: `feat/rebuild-embedding-models`

Completed:

- Added an approval-gated, resumable, hash-bound stage state machine enforcing
  provider extraction followed immediately by its identity audit.
- Added ZIP-only source identity audit, full dry-run plan, resource/artifact/
  index impact report, provider artifact auditor, failed-ID output, and CLIs.
- Added path safety (output outside Git), explicit create/rebuild policy,
  config-hash resume checks, and fail-stop audit criteria.
- Added a dataset-specific config and detailed runbook without changing model,
  schema, index, or ranking contracts.

Verification:

- Full-corpus orchestration suite: 5 passed.
- Real read-only L21 dry-run: PASS; 29 videos, 7,800 expected/unique identities,
  zero missing/extra/duplicates, archive hashes recorded.
- Python compilation and `git diff --check`: passed.

Execution blockers:

- Full canonical mapping/extracted keyframe root and exact commands are not yet
  prepared; embedding model assets are not all local; GPU/resource estimates
  are not measured. No extraction or index construction was launched.

# Multi-Embedding Task 16 — Final Release Gate

Status: CODE/STRUCTURAL GATE PASS; OPERATIONAL RELEASE BLOCKED

Branch: `feat/rebuild-embedding-models`

Completed:

- Added a reproducible architecture and Git-hygiene release audit.
- Verified provider dimensions/routing/indexability, ID-keyed RRF, fusion
  separation, and unchanged Task05B lexical fields.
- Added the final task-to-commit, provider provenance, command, count,
  limitation, approval, and rollback handoff report.
- Confirmed no lint, type-check, formatter, or dependency/security scanner is
  configured; no fabricated substitute result is claimed.

Verification:

- Static release audit: PASS; all 10 architecture and hygiene checks passed.
- Python compilation and `git diff --check`: PASS.
- Full repository pytest in the CI dependency set: PASS, 429 passed and 9
  opt-in/real-runtime tests skipped.
- Historical pre-isolation `pip check` failed when a broad semantic extra was
  installed into `mvr-asr:prepush`. The current unchanged ASR image now passes
  `pip check` with Torch/torchaudio `2.5.1+cpu`; semantic providers remain in
  separate pinned images.

Operational blockers:

- At this historical Task 16 checkpoint, real-weight three-provider inference
  had not yet run; the acceptance follow-up below supersedes this status.
- Full-corpus execution and representative-ground-truth evaluation remain
  approval- and resource-gated.

# Real three-provider five-keyframe acceptance follow-up

Status: PASS FOR REAL FASTTEST; FULL-CORPUS RELEASE REMAINS APPROVAL-GATED

Completed:

- Resolved the exact BEiT-3 base ITC checkpoint and `beit3.spm` through the
  current official Microsoft UniLM README, pinned their observed SHA256 values,
  and built an isolated CPU runtime without substituting another model.
- Ran real image and text inference for BEiT-3, MetaCLIP-2, and Perception
  Encoder on the same five canonical IDs. Each provider produced five valid,
  unique, normalized vectors with zero missing/extra/duplicate/invalid rows.
- Regenerated provider manifests with adapter, device, batch size,
  preprocessing, latency, and per-shard SHA256 evidence.
- Passed Task05A provider-specific indexability, created three isolated FAISS
  indexes, ran three provider-specific text query paths, proved cross-provider
  route rejection, and fused independent ranked lists with rank-only RRF k=60.
- Ran the full-corpus dry run only; no expensive extraction/index build was
  launched.

Verification:

- Focused offline/provider unit tests: 29 passed.
- Real BEiT-3 opt-in integration: 1 passed.
- Full regression: 430 passed, 9 skipped.
- Three artifact audits, Task05A, three indexes, three queries, cross-routing
  guard, RRF, structural release audit, and `git diff --check`: PASS.
- Full details: `REAL_THREE_PROVIDER_FASTTEST_REPORT.md`.

Remaining boundary:

- GPU compatibility, non-dry-run full-corpus execution, and representative
  ground-truth quality evaluation remain unverified and approval-gated.

# Current active phase

Multi-Embedding Tasks 04–16 are implementation-complete on
`feat/rebuild-embedding-models`. The code/structural gate and real three-provider
five-keyframe gate pass; operational competition release remains blocked by the
documented GPU/full-corpus/quality-evaluation gates. Task05A/Task05B remain
complete and regression-verified.

Next action: resolve the full-corpus commands and resource plan, then request
explicit approval before any non-dry-run full build.

# Multi-embedding pre-commit remediation

Status: PASS; FULL-CORPUS RELEASE REMAINS APPROVAL-GATED

- Removed developer-specific absolute paths from committed fasttest config and
  the full-corpus runbook.
- Locked the BEiT-3 tokenizer SHA256 alongside the checkpoint and made both
  image and text runtime paths reject missing or altered assets before model
  loading.
- Hardened recorded-query replay to validate provider identity, revision,
  dimension, dtype, normalization, vector SHA256, provider-specific index
  contract, and source embedding manifest SHA256.
- Pinned all three embedding Docker base images by digest.
- Corrected canonical/feature mapping hashes, recorded the five input image
  hashes, clarified BEiT-3 asset licensing, and reconciled the current ASR
  dependency status.
- The static release audit no longer claims that real inference did not run;
  it explicitly leaves real artifact validation to the runtime evidence audit.

Verification:

- Focused provenance/query/release tests: 54 passed.
- Full regression: 438 passed, 9 intentional opt-in/optional-runtime skips.
- Real BEiT-3, MetaCLIP-2, and Perception Encoder five-keyframe tests: PASS.
- Three pinned-digest Docker builds and `pip check`: PASS.
- Recorded query/index evidence replay, structural release audit, full-corpus
  dry run, and `git diff --check`: PASS.

No full-corpus non-dry-run operation was launched. Full-corpus execution and
representative-ground-truth quality evaluation remain approval-gated.

# Resume instruction

If work is interrupted, read:

```text
AGENTS.md
codex/00_MASTER_PLAN.md
docs/CODEX_PROGRESS.md
docs/AIC_KIS_IMPLEMENTATION_NOTES.md
docs/AIC_KIS_FINAL_REPORT.md
```
