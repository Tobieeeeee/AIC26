# Textual KIS Competition-Readiness Report

## Result

- Status: **`COMPLETE`** (per `00_MASTER_PLAN.md`'s "Competition-Ready Complete"
  definition — see Final Decision below for exact scope and caveats)
- Branch: `feat/kis-competition-ready` (local only as of this report; not yet
  pushed, per the team's own "push once everything is done" convention)
- Commits (in order):
  `9b373dc` Task 01 docs, `37576a6` Task 02 frame registry, `bfcf125` Task 03
  evaluator, `dc297b1` Task 04 search/API, `9e688c0` Task 05 refinement,
  `77c89aa` Task 06 benchmark, `47cd694` Task 07 export, `245bda6` Task 08 E2E.

## Source Contract Audit

- Existing retrieval types reused: `mvr.api.schemas.SearchHit` (outward API
  shape), `mvr.retrieval.semantic.SearchResult`, `mvr.indexing.faiss_index.FaissHit`.
  No parallel/duplicate hit type was invented.
- Existing frame/video resolver reused: `mvr.ingestion.probe.probe_video`,
  `mvr.ingestion.extract.extract_keyframes` (both used unmodified throughout
  Tasks 02/05/08).
- Existing API conventions reused: `create_app()` injectable-service factory
  pattern, `Field(ge=1, le=100)` top_k bounds, `ValueError`→HTTP 400 /
  `RuntimeError`→HTTP 503 error convention, conditional route registration.

## Frame Registry

- Original video source: `E:\AIC\AIC_DATASET_ROOT\videos\` — real AIC L21
  subset, 29 videos.
- Optional organizer support assets used: `keyframes/` (7800 JPEGs) +
  `map_keyframes/*.csv` + `media_info/*.json`. `media_info` kept strictly
  descriptive (never used as a frame-mapping source, per Task 02's explicit
  caution).
- Original frame mapping verified: **yes** — `L21_V001/003.jpg → frame_idx=261`
  verified bit-exact against the team's data-handoff doc; all 307 rows of
  `L21_V001.csv` swept and confirmed exact in Task 02; cross-checked again
  live through the KIS pipeline in Task 04/08.
- Cross-video keyframe ID collision protected: **yes** — composite
  `(video_id, keyframe_ref)` resolution; tested and verified with real data
  (`L21_V002/003` vs `L21_V003/003` resolve to different, correct `frame_idx`).

## Official Evaluator

- Unit tests: 13, all passing (`tests/unit/evaluation/test_aic_kis.py`).
- Boundary tests: inclusive `start_frame`/`end_frame` explicitly covered.
- Empty/fewer-than-k behavior: empty answer list scores 0 across all cutoffs;
  fewer-than-5-answers case explicitly tested and correct.
- Official per-query score implementation: `kis_r_score()` (exact video_id +
  inclusive interval match) → `evaluate_kis()` computes R@1/5/20/50/100 →
  Final Score = mean of those five. Deliberately not Recall@K/MRR.

## KIS Search/API

- Modes enabled: **semantic** (FAISS, always available) and **hybrid**
  (RRF-fused semantic+lexical, auto-used instead of semantic when
  Elasticsearch is configured). The real deployment tested in this report is
  **semantic-only** — no local Elasticsearch/Java was set up (scoped out,
  user-approved in Task 06).
- Top-100: enforced `1 ≤ top_k ≤ 100` both in `search_kis()` and at the
  Pydantic schema layer (`KisSearchRequest`).
- Original frame IDs: resolved via `KeyframeFrameIndexResolver` (this repo's
  own pipeline) or `AicDatasetRegistry` (organizer data) — a candidate that
  cannot be verified is dropped, never guessed.
- API route: `POST /search/kis`, unconditionally registered (works standalone
  off just the FAISS index, unlike the ES-dependent routes).
- Heavy model/index reload per request: **no** — verified both by a unit test
  (`test_kis_search_never_rebuilds_index_or_reloads_model_per_request`) and by
  a real Uvicorn run reusing the same loaded index/model across multiple real
  HTTP requests.

## Exact-Frame Refinement

- Original video decoded: **yes**, real FFmpeg decode, independently
  cross-verified against a separate OpenCV decode of the same frame index
  (mean pixel difference < 5.0).
- top-N refined: configurable (`top_n`); benchmark used `top_n=3`.
- search window: configurable `frame_radius` or `time_radius_seconds`
  (converted via real per-video FPS), clamped to valid video bounds.
- latency: real benchmark measured **mean 205s/query, p95 281s/query**
  (`frame_radius=45`, `top_n=3`) — this is **not production-viable as
  currently implemented**.
- before/after benchmark: **identical** mean Final Score (0.633) with vs.
  without refinement on the internal query set — refinement did not change
  correctness here, only added enormous latency. Root cause (documented
  since Task 05, empirically confirmed in Task 06): refinement reuses the
  existing exact frame-0 linear-scan decoder, whose cost scales with a
  candidate's position in the video rather than the refinement window size.

## Real-Data Benchmark

GT source: **team-curated internal** (not organizer-labeled). Explicitly
**AI-assisted and biased** — queries were written by Claude after directly
viewing the target keyframe image, so the query author knew the answer in
advance. This is sufficient to prove the pipeline is wired correctly
end-to-end; it must **not** be read as an unbiased quality measurement.
6 queries, real L21 corpus (7800 keyframes, 29 videos), real FAISS index,
real OpenCLIP CPU inference, real FFmpeg decode.

| Mode | mean R@1 | mean R@5 | mean R@20 | mean R@50 | mean R@100 | mean Final | mean latency | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic | 0.5 | 0.5 | 0.667 | 0.667 | 0.833 | 0.633 | 0.058s | 0.096s |
| text/OCR | not run — no local Java/Elasticsearch/OCR engine set up (scoped out, user-approved) | | | | | | | |
| hybrid | not run — same reason | | | | | | | |
| semantic + exact-frame refinement | 0.5 | 0.5 | 0.667 | 0.667 | 0.833 | 0.633 | 205s | 281s |

*(The template's "hybrid + exact-frame" row is reported as "semantic +
exact-frame" above since hybrid itself was not exercised in this environment.)*

**Decision recorded**: semantic-only is the current KIS baseline. Exact-frame
refinement is implemented, correct, and verified on real video, but is not
enabled by default given its latency — this benchmark is exactly the
evidence needed to justify a future seek-based-decode optimization before
refinement is turned on in a live setting.

## Prediction / Export Contract

- max predictions: 100 (enforced *after* removing exact duplicate pairs, so
  duplicates don't consume prediction budget).
- pair format: `(video_id: str, frame_id: int)`.
- local debug export: CSV and JSON, both explicitly labeled
  `"local_verification_export"` inside the file content itself.
- official organizer transport known: **no** — not implemented or assumed;
  `SubmissionAdapter` is a thin, unimplemented extension point for it.

## E2E

- Real AIC KIS E2E: **PASS** (`tests/integration/test_aic_kis_e2e.py`,
  opt-in, real run took 60.37s) — exercises the full documented path with no
  mocks: real video → frame registry → real FAISS index → KIS orchestration
  → real exact-frame refinement → official evaluator → export round-trip →
  real API response via the real ASGI app.
- API KIS: **PASS** — real Uvicorn process, real HTTP `POST /search/kis`
  returned 5 correctly-ranked real predictions; server verified cleanly
  stopped afterward (port confirmed free).
- normal pytest before Task 08: 163 passed, 4 skipped, 0 failed.
- normal pytest after Task 08, the final cleanup pass, and the PR #2
  hardening pass (see below): **171 passed, 5 skipped, 0 failed** (working
  environment, `semantic` extras installed).
- **CI-equivalent verification**: in a clean throwaway virtualenv installed
  with only `.[dev,api]` — exactly what `.github/workflows/ci.yml` installs,
  with no torch/faiss/open-clip/elasticsearch present — the suite is
  **163 passed, 13 skipped, 0 failed**. The 8 extra skips versus the working
  environment are the pre-existing faiss-dependent tests, matching the Phase
  06 fresh-clone precedent exactly. No new module introduces a hard
  dependency that would break CI.

## Post-Review Hardening (PR #2)

An independent review of PR #2 (head `50cdc47`) found and required fixing
three real issues before merge; all three were verified as real against the
actual code (not assumed from the review alone) before fixing:

1. **Weak E2E correctness assertion** — the real E2E test only asserted
   `0.0 <= score.final_score <= 1.0`, which a completely wrong retrieval also
   satisfies. Verified the real rank of the known query's correct answer is
   **6** (`L21_V001`, frame 5760, within the tested GT interval), so the test
   now asserts `score.recall_at_k[20] == 1.0` — the strongest official cutoff
   supported by measured evidence — and would fail if retrieval regressed.
2. **Coarse and refinement scores were being mixed** — `refine_top_candidates`
   overwrote a candidate's coarse retrieval `score` with the refinement
   window's local similarity score, two non-comparable score spaces
   (dangerous once hybrid/RRF scores are refined). `KisCandidate` gained a
   separate `refinement_score: float | None` field; `score` is now always
   preserved unchanged by refinement.
3. **Top-100 overfetch was capped at exactly 100** — `search_kis`'s default
   internal fetch pool was `min(top_k*3, 100)`, so at `top_k=100` no
   overfetching happened at all; unmapped/duplicate hits in the first 100 raw
   ranks could starve the final result below 100 even when valid candidates
   existed at raw rank 101+. Confirmed with a test using 200 raw hits (only
   even-indexed ones resolvable) that failed at `50 == 100` under the old
   formula and passes at the new `max(top_k*3, 30)` (uncapped, so 300 at
   `top_k=100`). Only affects the semantic-only path directly; the existing
   Elasticsearch hybrid branch's own internal `TOP_N_EACH=30` per-branch
   candidate limit is pre-existing baseline code and was left untouched.

6 new tests added across `test_kis.py` and `test_frame_refinement.py`. Real
E2E re-verified PASS (72.67s) with the strengthened assertion; real
frame-refinement decode re-verified PASS (47.67s). Full regression: 171
passed / 5 skipped / 0 failed; CI-equivalent: 163 passed / 13 skipped / 0
failed.
- `git diff --check`: clean at every task boundary (01 through 08) and after
  the final cleanup pass.
- private/generated data tracked: **no** — verified via `git status` at every
  task boundary; the AIC dataset lives entirely outside the repository tree
  (`E:\AIC\AIC_DATASET_ROOT`, sibling to `E:\AIC\PROJECT`), and
  `artifacts/verification/`, `agent_task/`, `.claude/` are all gitignored.

## Known Limitations

1. GT is team-curated/internal and AI-assisted (query author knew the
   answer) — not an unbiased or official quality measurement, and small
   (6 queries). Do not cite the R@k numbers above as competition performance.
2. Hybrid and text/OCR retrieval modes were not benchmarked or exercised
   against real data — no local Java/Elasticsearch/OCR engine was set up in
   this environment (explicitly scoped out for Task 06, user-approved).
3. Exact-frame refinement is implemented and verified correct, but is **not
   production-viable at its current latency** (mean 205s/query) — needs a
   seek-based decode optimization before enabling by default.
4. Queries were written in English, not Vietnamese — the base OpenCLIP model
   is English-trained; Vietnamese query translation (NLLB) was already noted
   as deferred/out-of-scope in `docs/LAB_REUSE_MAP.md` from the original
   reuse baseline.
5. The real FAISS index covers only the L21 subset (29 videos, 7800
   keyframes). `map_keyframes`/`media_info` for L22–L30 are present locally
   (shipped bundled in the same organizer archive) but no corresponding
   video/keyframe files were downloaded, so L22–L30 are not indexed.
6. The object-retrieval branch was not wired into `/search/kis` — per Task
   04's explicit instruction not to feed arbitrary natural language into an
   endpoint expecting structured object constraints.
7. Refinement's decode approach (reusing the existing exact frame-0 linear
   scan from the Phase 02 baseline) scales with a candidate's position in the
   video, not the refinement window size — a known, documented, and now
   empirically confirmed trade-off (see Task 05/06 notes above).

## Final Decision

All "Functional Complete" criteria from `00_MASTER_PLAN.md` are met (original
frame mapping is trustworthy and verified against an independent decode;
Top-100 KIS predictions work through both service and real HTTP API
boundaries; the official evaluator is correct per 13 passing tests including
boundary cases; the prediction contract validates with a working export
round-trip). All additional "Competition-Ready Complete" criteria are also
met (exact-frame refinement is implemented and verified on the real original
video; a real-AIC-data benchmark exists; at least one real-data E2E run
passes; the normal regression suite stayed green with zero regressions
across all 8 tasks; no private/generated data is committed).

> Textual KIS is functionally complete for the currently documented AIC 2026
> preliminary-round Textual KIS requirements, with the following limitations
> explicitly noted: the real-data benchmark GT is team-curated/internal and
> AI-assisted rather than organizer-provided or independently verified;
> hybrid/text-OCR retrieval was not exercised against real data in this
> environment; and exact-frame refinement, while correct, is not yet
> latency-viable for production use without a seek-based decode optimization.

STOP — per the task pack's stop gate, Q&A, TRAKE, Frontend, VLM, and
deployment work do not begin as part of this task pack.
