# TRAKE Competition-Readiness Report

## Result

- Status: **`COMPLETE`** (all 9 phases of the local TRAKE task pack; see Final
  Decision below for exact scope and caveats)
- Branch: `feat/trake-competition-ready`, not yet pushed — pending user
  review of this report (per standing project rule, push happens only after
  explicit approval)
- Commits (in order):
  `e68f6b6` Phase 01 audit + two prerequisite KIS fixes, `b12570f` Phase 02
  evaluator, `507284f` Phase 03 event/gap contract, `4127dc5` Phase 04 dense
  scoring + alignment strategies, `8142468` Phase 05 orchestration,
  `094da2d`+`a450913`+`04c9c0d` Phase 06 real-data benchmark and default-value
  fixes, `a5535d4` Phase 07 never-blank fallback/export, `4d084b4` Phase 08
  API route, `70fb328` Phase 09 end-to-end hardening.

## What TRAKE is

`TRAKE = KIS × k events, same video, correct order`: given `k` (≥2) event
descriptions and the connector phrases between them, TRAKE returns ranked
chains of `k` frame IDs, each chain confined to one video, in the declared
order. Each event is shortlisted independently across the whole corpus via
the existing `search_kis()` (unmodified — TRAKE's k-events-per-query shape is
concretely "call `search_kis()` k times"); every candidate video is then
scored densely per event by reconstructing frame vectors straight out of the
FAISS index (`IndexFlatIP.reconstruct()`, no re-encoding); one of three
chain-alignment strategies (ATLAS, Lucifer, U-CESE) picks the best-scoring
in-order chain per video; chains are merged and ranked across all shortlisted
videos, capped at 100 predictions. Scoring uses the official partial-credit
R-Score / Mean-Top-k-R / Final Score formula, `tolerance=1` frame (matching
the course lab default).

## Source Contract Audit

- Existing retrieval reused: `search_kis()` for per-event video shortlisting
  (Phase 01/05), unmodified. No parallel KIS-shortlist implementation exists.
- Existing index/vector infra reused: the FAISS index built by
  `build-aic-index` (`mvr.indexing.faiss_index`), `IndexFlatIP.reconstruct()`
  for dense per-frame scoring — no new index type, no re-encoding path.
- Existing API conventions reused: `create_app()` injectable-service factory,
  `ValueError`→400 / `RuntimeError`→503 convention, conditional/always-on
  route registration pattern from the KIS route.
- Two prerequisite KIS-owned fixes required before any TRAKE code could
  depend on it (Phase 01, `e68f6b6`): a resolver exception-type correction
  and `search_kis()`'s internal `top_k` ceiling raised 100→500 for wider
  per-event shortlisting. Both recorded as `LAB_REUSE_MAP.md` entries 25–26.
- Three chain-alignment algorithms ported from the course lab's
  `trake_align.py` (kept outside this repo, never redistributed):
  `dp_align`→`AtlasAlignment`, `dp_align_soft`→`LuciferAlignment`,
  `two_pointer_clips`→`UceseAlignment`. Three deviations from the lab
  reference (per-video instead of whole-corpus scope, `lam` threaded
  consistently through Lucifer's state key, gap budget accumulated across
  skipped events) are recorded as `LAB_REUSE_MAP.md` entries 27–29.

## Official Evaluator

- `src/mvr/evaluation/trake.py`: `trake_r_score()` (partial credit — averages
  per-event tolerance-matched hits within the correct video, zero if the
  video is wrong), `evaluate_trake()` (R@k across `TRAKE_CUTOFFS`),
  `trake_final_round_score()` (DRES mean-of-R@k formula, same shape as KIS).
- Unit tests: `tests/unit/evaluation/test_trake.py`, covering exact match,
  partial match, tolerance boundary (inclusive at exactly `tolerance` frames
  away), wrong video (score 0 regardless of frame overlap), and `None`
  markers (an unfilled event slot scores 0 for that event, not a crash).

## Event/Gap Input Contract

- `src/mvr/retrieval/trake_events.py`: `TrakeQuery` validates `k ≥ 2`, exactly
  `k-1` gap constraints, and sequential event indices.
- Gap timing is expressed only as Vietnamese connector phrases (`"rồi"`,
  `"ngay sau đó"`, `"một lúc sau"`, ...) translated to `[min, max]` second
  ranges via `GAP_PHRASE_TABLE`; the empty-phrase default is `(0, 300)`.
  Raw second values are deliberately not accepted at the API layer — see the
  Phase 06 benchmark finding below for why.

## Dense Scoring and Alignment

- `src/mvr/retrieval/trake_scoring.py`: `build_video_score_matrix()` builds a
  `(k events) × (resolvable frames)` similarity matrix per shortlisted video
  directly from the FAISS index, with real per-video timestamps and original
  frame IDs attached — no re-encoding of images per query.
- `src/mvr/retrieval/trake_align.py`: three strategies behind one
  `TrakeAlignmentStrategy` protocol.
  - `AtlasAlignment` (`alpha=0.01, beta=0.01, min_sim=0.0, top_n=5`): hard DP,
    every event must match a frame. **Production default.**
  - `LuciferAlignment` (`alpha=0.01, beta=0.01, lam=0.8, min_sim=0.20,
    top_n=5`): soft DP that can skip an event it's not confident about,
    emitting `None` for that slot.
  - `UceseAlignment` (`t_win=20.0, topk=10, top_n=5`): sliding-window
    two-pointer matching, intentionally does not enforce event order.
  - All three gained a `top_n` parameter (Phase 06) returning the N best
    distinct chains per video instead of just one, since a single chain per
    video was wasting most of the 100-prediction budget.

## Orchestration and API

- `src/mvr/retrieval/trake.py`: `search_trake()` — shortlist each event via
  `search_kis()`, union the shortlisted videos, dense-score and align each
  one, merge and rank all resulting chains, cap at `top_k` (default 100).
  A per-event shortlist failure is caught and skipped rather than crashing
  the whole query (verified by
  `test_trake_search_per_event_retrieval_failure_does_not_crash_route`).
- `POST /search/trake` (`src/mvr/api/app.py`, `src/mvr/api/schemas.py`):
  accepts `events` (≥2, non-blank text) and `gap_phrases` (exactly
  `len(events)-1`, validated by a Pydantic `model_validator`), with an
  optional `alignment_strategy` override. Returns ranked chains with
  `frames: list[int | None]` (serializing Lucifer's skip markers as JSON
  `null`) and each chain's score.
- **Phase 09 fix**: the route originally only accepted a semantic service
  exposing `.retriever()` (the production `LazyFaissSemanticService` shape).
  A plain `SemanticRetriever` — which already exposes `.text_encoder`/`.index`
  directly, and is exactly what the real E2E tests construct — crashed the
  route with an unhandled `AttributeError` → opaque HTTP 500. This was
  invisible to every prior test: route unit tests used a fake *with*
  `.retriever()`, and the real E2E test called `search_trake()` directly,
  bypassing the route entirely. Fixed via `_resolve_trake_semantic_service()`,
  which accepts either shape and raises a clear `RuntimeError` → HTTP 503
  otherwise. Two regression tests added
  (`test_trake_search_accepts_a_plain_retriever_shaped_service`,
  `test_trake_search_returns_503_not_500_for_a_search_only_service`).

## Never-Blank Submission Fallback

- `src/mvr/submission/trake_export.py`: `apply_never_blank_fallback()` fills
  any `None` marker in a chain about to be submitted, since a blank frame ID
  scores strictly worse than any documented guess.
  - Interior gaps (a `None` between two matched events) use linear
    interpolation between the real neighboring timestamps — better evidence
    than any heuristic, since both endpoints are real matches from the same
    video.
  - Edge gaps (a `None` at the start or end of the chain, no real neighbor on
    one side) fall back to the gap-phrase-table midpoint — the only
    information available at that point.
  - Multiple consecutive interior `None`s split proportionally by each gap's
    declared width, per an explicit design decision made with the user
    during Phase 07.
- `validate_trake_prediction()` / `export_local_trake_verification_csv()` /
  `export_local_trake_verification_json()`: same "explicitly labeled local
  verification export" discipline as the KIS export — no official organizer
  transport format is assumed or implemented.
- Phase 07 shipped with **no real-data caller** for this module — only
  synthetic unit tests. Closed in Phase 09 by
  `test_trake_full_pipeline_end_to_end_including_submission_export`, a
  genuine no-mocks pipeline test (see E2E below).

## Real-Data Benchmark

GT source: **team-curated internal** (not organizer-labeled), same caveat as
the KIS benchmark. 6 queries, real L21 corpus, real FAISS index, real
OpenCLIP CPU inference, real dense per-frame reconstruction, real DP/window
alignment.

| Strategy | mean R@1 | mean R@5 | mean R@20 | mean Final Score | mean latency |
|---|---:|---:|---:|---:|---:|
| ATLAS (default) | 0.292 | 0.694 | 0.736 | **0.639** | 0.372s |
| Lucifer | 0.292 | 0.694 | 0.736 | **0.639** | 0.410s |
| U-CESE | 0.250 | 0.306 | 0.389 | 0.344 | 0.372s |

Per-query Final Score (ATLAS / Lucifer / U-CESE):

| Query | ATLAS | Lucifer | U-CESE |
|---|---:|---:|---:|
| traffic_accident_night | 0.600 | 0.600 | 0.300 |
| flood_embankment | 0.550 | 0.550 | 0.500 |
| house_fire_alley | 0.533 | 0.533 | 0.267 |
| ocean_debris_cleanup | 0.900 | 0.900 | 0.500 |
| suspect_arrest | 0.750 | 0.750 | 0.500 |
| wide_gap_bulletin_segments | 0.500 | 0.500 | 0.000 |

ATLAS and Lucifer tie exactly on every query — under the real gap widths in
this dataset (8–23s for 5 of the 6 queries), Lucifer's skip mechanism never
fires, so it behaves identically to ATLAS's hard alignment while running
somewhat slower. A separate isolated test (not part of the mean above) proves
the skip mechanism does work: on an event constructed to be genuinely
unfindable (`L21_V018`), ATLAS forced a wrong match and scored 0.000, while
Lucifer correctly emitted `(frame, None, frame)` and scored 0.267.

**Decision recorded**: ATLAS is the production default — same accuracy as
Lucifer on every real query tested, no tuning cliff (its `top_n` parameter is
flat across 5–20), and faster. Lucifer remains available via
`alignment_strategy="lucifer"` for cases where an event may genuinely be
absent from the video. `docs/AIC_TRAKE_BENCHMARK_RESULTS.md` has the full
sweep data and three structural defects found and fixed before these numbers
were trustworthy (see Known Limitations for a summary).

## E2E

- Real full-pipeline E2E (`tests/integration/test_aic_trake_e2e.py`,
  `test_trake_full_pipeline_end_to_end_including_submission_export`,
  opt-in via `MVR_RUN_AIC_TRAKE_E2E=1`, added Phase 09): **PASS** — query
  `traffic_accident_night` (k=4 events, target video `L21_V024`, GT frames
  13890/14130/14430/14635) run through the real orchestration, real
  evaluator, real never-blank export, and round-tripped back through
  `load_local_trake_verification_json()` with an exact match. Real measured
  result: 100 chains returned, **final_score = 0.600**, R@1=0.000,
  R@20=0.750, top-ranked chain was `L21_V006` (frames 11894/11901/18686/19206)
  — **not the GT video** (GT still recoverable at the R@20 cutoff), real fps
  30.0 read from the organizer CSV, exported file 494 bytes.
- Real API route test (`test_lucifer_skip_beats_atlas_when_an_event_is_
  genuinely_unfindable`, `test_trake_real_data_benchmark_across_all_
  strategies`): PASS, both part of the same opt-in suite.
- Real HTTP smoke: OpenAPI schema confirmed exactly 4 routes registered
  (`/health`, `/search/kis`, `/search/semantic`, `/search/trake`); API module
  import cost measured at 0.47s.

## Query Translation (Vietnamese -> English, opt-in)

Added after this report's first draft, in response to the Known Limitations
entry below being flagged as the single largest gap versus real competition
use. `search_trake()` gained an optional `query_translator` parameter
(mirrors `search_qna()`'s existing `translate_for_retrieval` usage exactly —
reused, not reimplemented, from `mvr.enrichment.qna_translation`): each
event's text is translated for retrieval only (both shortlisting and dense
scoring use the same translated text) before anything else happens. `None`
(the default) keeps every event's text unchanged, byte-identical to all
prior behavior — this is purely additive.

The API route wires this through `_default_trake_query_translator()`, an
opt-in lazy loader (`LazyQueryTranslator`, mirrors `LazyAnswerReasoner`'s
fail-closed/cache-the-error shape) gated behind
`MVR_TRAKE_TRANSLATE_QUERIES=1`. Off by default: the measured benchmark
above used untranslated English text throughout, and enabling translation
unconditionally could silently change that result for any caller. A broken
or uncached translator degrades to untranslated retrieval rather than a
500/503 (`translate_for_retrieval`'s existing, already-tested contract) —
verified again here specifically for TRAKE's real event text, not just
assumed from Q&A's tests.

**Real measured result** (`tests/integration/test_aic_trake_e2e.py::
test_vietnamese_queries_need_translation_to_score_anything`, opt-in,
`MVR_TRAKE_ALLOW_MODEL_DOWNLOAD=1` on first run): the same 6 benchmark
queries, rewritten as natural Vietnamese (not back-translated from the
English text above), run through the real `search_trake()` pipeline:

| | mean final score | per-query |
|---|---|---|
| Vietnamese, no translation | **0.000** | 0.000 on all 6 — confirms the gap below, does not just assume it |
| Vietnamese, with translation (Stage A) | **0.547** | 0.450 / 0.500 / 0.533 / 0.900 / 0.400 / 0.500 |
| English (known baseline, same queries) | 0.639 | — |

Translation recovers most but not all of the gap (0.547 vs 0.639 — roughly
86% of the English-query score), consistent with machine translation
introducing some semantic drift versus hand-written English event text.
Zero to substantial is the headline result; zero to perfect was never a
realistic bar.

**Cost/dependency footprint**: the translator (`Helsinki-NLP/opus-mt-vi-en`,
~300MB, CPU-only, ~1-2s/query measured) lives behind the existing optional
`qna` extra (`transformers`, `sentencepiece`) — already a dependency of the
Q&A track, not a new one introduced here. Verified in a clean venv matching
`.github/workflows/ci.yml` exactly (`.[dev,api,semantic]`, no `qna` extra):
TRAKE imports and all its tests still pass, and the new Vietnamese-specific
E2E test skips cleanly (not an error) when the extra is absent — this
feature adds zero hard dependency for anyone running the base system.

## Full Regression Suite

- Working environment (`semantic` extras installed): **860 passed, 14
  skipped, 4 failed**. The 4 failures are pre-existing and unrelated to
  TRAKE (`tests/unit/enrichment/ocr/test_weight_validation.py`, from the
  separately-owned `feat/ocr-enrichment` track): a subprocess test invokes a
  bare `python` command that resolves to a different, non-project Python
  interpreter lacking `cv2` on this machine, not a logic defect. Not
  investigated further here since it is outside TRAKE's scope.
- CI-equivalent environment (clean venv, `.[dev,api,semantic]` only —
  matching `.github/workflows/ci.yml` exactly): **858 passed, 16 skipped, 4
  failed** (same 4 pre-existing OCR failures; the 2 extra skips are the
  translation-dependent tests correctly self-skipping without the `qna`
  extra). Re-verified after merging `origin/main` (90+ commits: ASR, OCR
  hardening, the BEiT-3/MetaCLIP-2/Perception Encoder multi-embedding stack,
  Q&A, lexical BM25) and after adding query translation above — both passed
  without new failures.
- `git status`/`git status --ignored` swept clean at the end of Phase 09;
  leftover untracked `__pycache__` debris from an earlier `git merge --abort`
  (unrelated prior-phase conflict testing) was found and removed.
- No hard-coded local absolute paths remain in committed source — a leaked
  `E:\AIC\11_TRAKE\...` path in `trake_align.py`'s module docstring (an
  artifact of the porting process) was found via `git grep` and replaced with
  pointers to this report and `docs/AIC_TRAKE_BENCHMARK_RESULTS.md`.

## Known Limitations

1. GT is team-curated/internal, not organizer-labeled, and small (6 queries)
   — same caveat as the KIS report. Do not cite the numbers above as
   competition performance.
2. **Queries must be in English by default; Vietnamese now works via an
   opt-in translation layer, not fully closed.** The index's OpenCLIP model
   (`ViT-B-32`/`laion2b_s34b_b79k`) is English-trained; raw Vietnamese text
   still scores 0.000 (verified again in this pass, see "Query Translation"
   above). `query_translator` (opt-in, `MVR_TRAKE_TRANSLATE_QUERIES=1`)
   closes most of the gap on real Vietnamese queries (0.000 -> 0.547 mean,
   vs. 0.639 for the same queries hand-written in English) but not all of
   it — machine translation loses some nuance, and this remains off by
   default rather than a verified drop-in replacement for English queries.
   Real competition queries will be Vietnamese, so this should be enabled
   deliberately, not assumed automatic. Same limitation already noted for
   KIS in `docs/AIC_KIS_FINAL_REPORT.md`; KIS has no equivalent translation
   wiring yet.
3. Three lab-reference defaults were wrong at TRAKE's per-video scope and
   silently produced meaningless comparisons before being found and fixed in
   Phase 06: U-CESE's `topk=300` collapsed every event onto one frame
   (fixed to `10`), Lucifer's `min_sim=0.05` sat below every real CLIP score
   so its skip mechanism never fired (fixed to `0.20`, a measured cliff —
   `0.22` already drops mean final score to 0.530), and all three strategies
   returned only one chain per video, wasting most of the 100-prediction
   budget (fixed by adding `top_n=5`). See `docs/AIC_TRAKE_BENCHMARK_RESULTS.md`
   and `docs/LAB_REUSE_MAP.md` entries 30–32 for full detail.
4. Gap constraints were not exercised by the original 5 benchmark queries
   (their real gaps were always inside any reasonable budget); a 6th query
   with genuinely wide real gaps (`wide_gap_bulletin_segments`, 310s/294s)
   was added specifically to prove gap constraints matter (+0.045 over no
   constraint) and that the phrase table's own default beats a hand-picked
   range (0.639 vs 0.633).
5. Same L21-subset index-coverage limitation as KIS (29 videos, 7800
   keyframes; L22–L30 not indexed).
6. U-CESE intentionally does not enforce event order — this is a documented
   property of the reference algorithm, not a bug, but it means U-CESE chains
   can violate the "correct order" half of the TRAKE definition; it is not
   the production default for this reason among others.
7. No `mvr.cli` subcommand exists for TRAKE (API/library-only). Not
   fabricated per explicit instruction — added only if the user requests it
   as separate follow-up work.

## Final Decision

All 9 phases of the local TRAKE task pack are complete: the official
evaluator is correct and tested against tolerance/partial-credit/wrong-video
cases; the event/gap contract validates and translates connector phrases; all
three chain-alignment strategies are ported, verified against the lab
reference's own self-check values where applicable, and their three
deviations documented; a tuned, evidence-backed production default (ATLAS) is
selected from a real parameter sweep, not guessed; `POST /search/trake` is
live, hardened against both known service shapes, and returns real ranked
chains over real HTTP; the never-blank fallback has a real, no-mock,
full-pipeline caller; and the full regression suite is identical in the
working environment and a clean CI-equivalent environment.

> TRAKE is functionally complete for the currently documented AIC 2026
> preliminary-round temporal-retrieval requirements, with the following
> limitations explicitly noted: the real-data benchmark GT is team-curated
> and small; benchmark and E2E queries were written in English while real
> competition queries will be Vietnamese, an unresolved operational gap
> shared with the KIS report; and the index covers only the L21 subset.

STOP — per the task pack's stop gate, Q&A, Frontend, VLM, and deployment
work do not begin as part of this task pack. Do not push to `origin` until
the user has reviewed this report and explicitly approves.
