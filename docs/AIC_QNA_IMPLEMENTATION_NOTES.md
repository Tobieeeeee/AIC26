# AIC Q&A Implementation Notes — Repository Contract Audit

Task 01 output for the Q&A competition-ready task pack (branch
`feat/qna-competition-ready`). Records the exact current interfaces,
identifiers, contracts, and gaps that all later Q&A task-pack work must reuse
instead of inventing parallel abstractions. No production code was changed to
produce this document.

## 0. Organizer contract (source: `Thong tin vong So tuyen AIC2026.md`)

- Query = a natural-language event description + a question about that event.
- Answer submitted per query, up to 100 ranked: `<video_id>, <frame_id>, <answer>`.
- `answer` may be Vietnamese or English.
- Correctness (R-Score, per submitted answer): `video_id == GT.video_id` AND
  `frame_id ∈ [GT.start_frame, GT.end_frame]` AND `answer` matches
  `GT.answer` **semantically** (not exact string — the organizer's own
  example scores both `"5"` and `"Năm"` as correct against a GT answer of
  `"5"`/`"Năm"`; partial credit is explicitly possible, R-Score ∈ [0,1]).
- `R@k` for k ∈ {1,5,20,50,100} = best R-Score among the first k answers;
  `Final Score` = mean of the five `R@k` values. **Identical aggregation
  mechanics to Textual KIS** — only the R-Score correctness predicate differs
  (adds the answer-match condition).
- Dataset provided: `videos/`, `keyframes/<video_id>/` (ordered filenames,
  real `frame_idx` recorded in metadata, not the filename ordinal),
  `objects/<video_id>/<frame>.json` (Faster R-CNN pretrained on OpenImages
  V4; TensorFlow detection-output shape per the organizer's linked
  reference — fields such as `detection_boxes`, `detection_class_entities`,
  `detection_scores`; **no real sample file has been inspected in this repo
  yet** — treat the exact field names as unconfirmed until Task 02 parses a
  real file), one `.npy` of CLIP ViT-B-32 features per video (index-aligned
  to keyframe order), `metadata/<video_id>.json` (YouTube metadata, may be
  absent per video).

Full spec saved outside Git in project memory
(`project_aic2026_qna_spec` — see the assistant's memory store) since it is
organizer material, not repository source.

## 1. What Q&A can reuse unchanged

- **Retrieval branches**: `mvr.retrieval.semantic` (FAISS) and
  `mvr.retrieval.hybrid` (RRF-fused semantic+lexical) — the event-description
  half of a Q&A query is a KIS-shaped query. `mvr.retrieval.kis.search_kis()`
  is architecturally the right template (overfetch → resolve original frame
  → dedupe → rank), not something to call directly (it returns `KisCandidate`,
  not the richer object a Q&A candidate needs), but its resolution flow
  should be duplicated, not reinvented.
- **Frame registry**: `mvr.ingestion.aic_dataset` — resolves organizer
  keyframes to `(video_id, frame_idx)` via `map_keyframes/<video_id>.csv`
  (`frame_idx` column only, never `n`). Same critical rule applies to Q&A's
  `frame_id` output. Reuse `KeyframeFrameIndexResolver` /
  `resolve_original_frame_id()` from `mvr.retrieval.kis` verbatim — do not
  duplicate the resolution logic.
- **OCR**: `mvr.enrichment.ocr.easyocr_engine` / `paddleocr_engine` (Lab 6
  adapters, confidence threshold `0.3`) — usable as one structured-context
  source for the answer reasoner.
- **Evaluator shape**: `mvr.evaluation.aic_kis` — `KIS_CUTOFFS = (1,5,20,50,100)`,
  the `evaluate_kis()` overfetch/cutoff/mean aggregation. Q&A's evaluator
  must reuse this aggregation function's *shape* (new dataclasses, new
  R-Score predicate with the answer-match condition — see §4) rather than
  parametrize `evaluate_kis()` itself, matching this repo's existing
  KIS-vs-generic-metrics separation (`evaluation/metrics.py` is deliberately
  not reused for KIS either, per `docs/CODEX_PROGRESS.md`).
- **Submission validation/export shape**: `mvr.submission.kis_export` —
  `validate_kis_predictions()` / `export_local_verification_{csv,json}()`
  pattern (dedupe, rank reassignment, max-100 enforcement, explicit
  "LOCAL VERIFICATION EXPORT — not the organizer format" labeling). Q&A needs
  its own `QnaPrediction` (adds `answer: str`) and validator, same shape.
- **API pattern**: `mvr.api.app` — lazy singleton service loading, `Field(ge=1,
  le=100)` top_k bounds, `ValueError`→400 / `RuntimeError`→503 convention,
  conditional route registration (`POST /search/kis` is always registered
  because it only needs the semantic index; the analogous `POST /search/qna`
  route should follow the same rule — always registered, degrades if the new
  answer-reasoner dependency is unavailable rather than never registering).

## 2. What is genuinely new (no existing lab/component to reuse)

1. **Structured-context builder** — for one resolved `(video_id, frame_id)`,
   assemble: organizer objects JSON for that frame (**counts + bounding box
   per label, not just a label list** — the organizer's own example question
   ("bao nhiêu người...") requires counting, and the existing
   `mvr.enrichment.objects.yolo.YoloObjectDetector.detect()` only returns
   `dict[str, int]` counts with no bbox, so if the organizer file has richer
   per-detection records than YOLO's Lab 6 output, the new loader must not
   collapse to that same coarser shape) + OCR text for that frame (reuse Lab
   6 adapters). No existing module in this repo builds a per-frame structured
   context bundle today.
2. **Answer reasoner** — a new LLM-based component that takes
   (question, structured context) and returns a short VI/EN answer. Two
   existing experimental precedents in this repo use the same local-HF-model
   base+backend pattern (`scripts/verification/query_decomposition/qwen.py`,
   `scripts/verification/query_translation/local.py`) but neither is wired
   into `src/mvr/` or declared as a real dependency — **`transformers` is
   imported by those scripts but is not listed anywhere in
   `pyproject.toml`'s optional-dependency groups** (confirmed via repo-wide
   grep). Task 03 must add a proper extras group (or extend an existing one)
   rather than leave it undeclared. Per the reference-repo review from the
   prior conversation turn, the reasoner should (a) fail closed if its model
   weights are not locally cached and no explicit download opt-in is passed
   — mirroring `CrossEncoderReranker` in `HCMAIC-SkillPixel-KIS`'s
   `retrieval/rerank.py` — and (b) tag each answer with a provenance source
   (`context-grounded` / `knowledge-grounded` / `mixed`) since the SOICT 2025
   winner paper (OpenCubee_1 / ConvAgent) documents that competition Q&A
   queries can require the model's own general knowledge, not just visual
   context.
3. **Q&A evaluator** — new `KisGroundTruth`/`KisAnswer`/`KisScore`-shaped
   dataclasses (`QnaGroundTruth` adds `answer: str`; `QnaAnswer` adds
   `answer: str`) and a new R-Score predicate that ANDs the existing
   video+frame-interval check with an answer-match check. The organizer rule
   is "semantic match", which is a human-judge operation in the real
   competition; for internal benchmarking this repo cannot fabricate a human
   judge, so Task 04 must implement and clearly document a **normalized/fuzzy
   match approximation** (case-fold, whitespace/punctuation normalization,
   number-word equivalence for VI/EN, no LLM-as-judge unless explicitly
   scoped) and label it as such everywhere it is reported, the same way
   `mvr.evaluation.aic_kis`'s module docstring warns against mislabeling its
   own aggregate as the organizer's official score.
4. **`POST /search/qna` route and orchestration** — new, modeled on
   `mvr.retrieval.kis.search_kis()` + `mvr.api.app`'s KIS route wiring.
5. **Q&A prediction validation/export** — new, modeled on
   `mvr.submission.kis_export`.

## 3. Known gaps / unconfirmed assumptions to close in later tasks

- No real organizer `objects/<video_id>/<frame>.json` sample has been
  inspected locally in this repo (no AIC dataset is present on this machine
  yet). The Faster-RCNN/OpenImages field-name assumptions in §0 come from the
  organizer doc's prose description and its linked TF-Hub reference, not a
  parsed real file. **Task 02 must obtain and parse a real sample before
  finalizing the loader schema**, or explicitly document that it proceeded
  without one and why.
- `transformers` is an undeclared dependency (see §2.2) — must be added to
  `pyproject.toml` in the task that first makes production code depend on it
  (Task 03), not left implicit.
- No ASR/audio component exists in this repo. The SOICT 2025 paper uses ASR
  as a third structured-context source; this is explicitly **out of scope**
  for this task pack per the reuse-first/no-unnecessary-dependencies rules in
  `AGENTS.md` — noted here so it is a deliberate exclusion, not an oversight.
- No Q&A ground-truth query set exists in this repo (KIS had a small
  team-curated internal set). Task 06 (benchmark) will need one before any
  Final Score can be computed; building it is in scope for that task, not
  this one.

## 3b. Task 02 outcome — structured-context builder

Implemented, no real organizer sample available to validate against (see §3):

- `src/mvr/enrichment/objects/aic_objects.py` — `load_frame_objects()` parses
  the assumed organizer schema (`detection_class_entities`,
  `detection_scores`, `detection_boxes`) into a validated `FrameObjects`
  (per-detection `label`/`confidence`/`bbox`, plus a `.counts` property).
  Raises `InvalidAicObjectsError` on any schema violation (length mismatch,
  out-of-range confidence, malformed bbox, blank label) rather than
  coercing bad data — mirrors the strict `__post_init__` validation style
  used by `ObjectRecord` in the `HCMAIC-SkillPixel-KIS` reference repo.
  **This schema is unconfirmed against real data; revisit when a real
  `objects/<video_id>/<frame>.json` sample becomes available.**
- `src/mvr/enrichment/qna_context.py` — `build_frame_context()` combines
  objects (optional accelerator, same "missing file → None" policy as
  `mvr.ingestion.aic_dataset`'s organizer support data) with OCR text from
  an injected `OcrEngine` Protocol (`.read_text(image) -> str`, matching the
  existing `EasyOcrEngine`/PaddleOCR adapter method signature) so this
  module never imports EasyOCR/PaddleOCR itself. A *missing* objects file
  degrades to `objects=None`; a *malformed* one raises and is not swallowed.
- Tests: `tests/unit/enrichment/objects/test_aic_objects.py` (8 cases) and
  `tests/unit/enrichment/test_qna_context.py` (5 cases), all synthetic
  fixtures/fakes per this repo's "no huge models for lightweight tests"
  rule. Full suite: 211 passed, 13 skipped (no regressions from Task 01's
  198 passed, 13 skipped).
- Not yet done: wiring a *real* EasyOCR/PaddleOCR engine instance through
  this interface end-to-end (deferred — no real keyframe image is present
  on this machine right now; the injected-Protocol design means real
  wiring is a call-site concern for Task 05's orchestration, not a change
  to this module).

## 3c. Task 03 outcome — local-LLM answer reasoner

Upstream update since Task 02: `main` (PR #4, `feat/ingestion-enrichment`)
merged a live torchvision `fasterrcnn_resnet50_fpn` (COCO-pretrained)
detector at `src/mvr/enrichment/objects/faster_rcnn.py`, plus
`src/mvr/enrichment/feature_extraction.py` for per-video CLIP feature
extraction. This is a **different, unrelated component** from Task 02's
`aic_objects.py` (which reads the organizer's own precomputed
OpenImages-labeled JSON, not a live COCO detector) — no file conflicts, no
code changes needed to Task 02's work, but it does mean **two distinct
object-detection label spaces now exist in the system** (organizer
OpenImages labels vs. this new COCO labels vs. existing YOLO COCO labels).
Task 05's orchestration must keep them source-tagged rather than merging
counts across label spaces — see the branch-strategy note recorded after
this rebase (organizer file first, this detector or YOLO as fallback when
the organizer file is absent for a video).

Implemented, mirroring the reference repo's real-provider fail-closed
policy and this repo's own `QwenDecomposer` retry pattern:

- `src/mvr/enrichment/qna_answer.py` — split into `LlmBackend` (Protocol,
  swappable) and `AnswerReasoner` (pure prompt/parse/retry logic, backend
  agnostic, no transformers/torch import at module load time). The prompt
  explicitly allows `context-grounded` / `knowledge-grounded` / `mixed`
  answers per the SOICT 2025 winner-paper finding that competition Q&A can
  require general knowledge, not only visual context; the model self-reports
  which, recorded as-authored (not independently verified) in `QnaAnswer`.
  One deterministic retry on malformed JSON output, then raises
  `MalformedAnswerError` — mirrors `QwenDecomposer`'s existing retry shape.
- `HuggingFaceCausalLmBackend` — real local-HF-model backend, defaults to
  `Qwen/Qwen2.5-1.5B-Instruct`, **fails closed**: `local_files_only=True`
  unless `allow_model_download=True` is passed explicitly, and forces
  `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` in that mode so a missing
  cache fails fast rather than blocking on a network timeout (measured:
  without the offline env vars, a cold local-files-only failure against a
  nonexistent model still attempted network and took ~14s; forcing offline
  mode is a real, verified mitigation, not a full fix — DNS/socket-level
  timeout can still add a few seconds depending on the machine's network
  stack). `manifest_metadata()` records model id, transformers version,
  device, `local_files_only`, and a `provider_evidence` tag, following the
  `HCMAIC-SkillPixel-KIS` reference repo's `CrossEncoderReranker` provenance
  pattern.
- **`transformers` is now a declared dependency**: added a new `qna` extra
  (`transformers>=4.40,<5`, `torch>=2.2,<3`) to `pyproject.toml`, closing the
  undeclared-dependency gap flagged in Task 01 §3. Not added to CI (`.github
  /workflows/ci.yml` still installs `dev,api,semantic` only) since no test
  requires it to run — the one real-backend test uses `pytest.importorskip
  ("transformers")` and self-skips when the extra is absent.
- Tests: `tests/unit/enrichment/test_qna_answer.py` (9 cases: 8 pure-logic
  cases against a scripted fake backend, covering valid parse, OCR/object
  content in the prompt, retry-then-succeed, retry-then-fail, invalid
  `source` value, blank answer text, blank question, and
  `knowledge-grounded` acceptance; 1 real, non-mocked verification that
  `HuggingFaceCausalLmBackend` actually refuses to run against an
  uncached/nonexistent model rather than hanging or downloading). Full
  suite with the `qna` extra installed: 230 passed, 5 skipped (no
  regressions from the post-rebase baseline of 221 passed, 5 skipped).
- Not yet done: choosing/verifying a specific small model actually runs
  real inference end-to-end on this machine (no GPU confirmed, no model
  weights cached) — deferred to whichever later task first needs a real
  generated answer (Task 05 orchestration or Task 06 benchmark), consistent
  with "verify real behavior only when the task requires it," not before.

## 3d. Task 04 outcome — Q&A evaluator

Implemented as a standalone module mirroring `aic_kis.py`'s shape rather
than reusing it directly (per Task 01's plan and the repo's existing
KIS-vs-generic-metrics separation precedent):

- `src/mvr/evaluation/aic_qna.py` — `QnaGroundTruth`/`QnaAnswer` (adds
  `answer: str` to the KIS shapes), `QNA_CUTOFFS = (1,5,20,50,100)`
  (redefined locally, not imported from `aic_kis`, so the two tracks don't
  accidentally couple if either organizer rule changes independently even
  though the values coincide today), `evaluate_qna()` (identical
  overfetch/cutoff/mean aggregation to `evaluate_kis()`).
- `qna_r_score()` ANDs the KIS video+frame-interval check with
  `qna_answer_matches()` — a **documented approximation** of the
  organizer's real "semantic match" human-judge rule: case-fold +
  punctuation/whitespace normalization, plus VI/EN digit↔number-word
  equivalence (0–20) covering the organizer's own worked example
  (`"5"` ≡ `"Năm"` ≡ `"năm"`, matched in both GT-direction and either
  language). Anything else scores 0 — deliberately no embedding/LLM-as-judge
  broader semantic similarity, since a false "match" would be worse than an
  honest miss for internal benchmarking. Every scoring surface must label
  results from this module as an approximate self-evaluation, never the
  organizer's official judged score (same caveat `aic_kis` already carries).
- Tests: `tests/unit/evaluation/test_aic_qna.py` (15 cases) covering the
  organizer's own worked example verbatim, rank-based R@k aggregation
  (rank 3/after-100 cases mirrored from the KIS test suite), case/whitespace
  insensitivity, and both VI and EN number-word equivalence plus explicit
  rejection of unrelated numbers. Full suite: 245 passed, 5 skipped (no
  regressions from Task 03's 230 passed, 5 skipped).
- Not yet done / explicitly deferred: broader fuzzy matching (synonyms,
  paraphrase, non-numeric near-misses) — only add if a real benchmark
  (Task 06) shows the current approximation is too conservative on actual
  queries, not speculatively now.

## 3e. Task 05 outcome — orchestration + `POST /search/qna`

- `src/mvr/retrieval/qna.py` — `search_qna(event_description, question, ...)`
  deliberately duplicates `search_kis()`'s overfetch → resolve original
  frame → dedupe → rank shape (a `QnaCandidate` needs an extra `answer`
  field `KisCandidate` has no room for, so it can't just call `search_kis`),
  but **reuses `resolve_original_frame_id()`/`KeyframeFrameIndexResolver`
  from `mvr.retrieval.kis` verbatim** per the Task 01 plan — no
  frame-resolution logic duplicated. Answer reasoning (Task 03) only runs
  on the final top-`k` resolved candidates, never the overfetch pool, since
  it is an LLM call per candidate and comparatively expensive versus the
  coarse retrieval score.
- `ocr_engine`/`image_loader` are accepted as a matched pair (both or
  neither — mismatched configuration raises `ValueError` immediately rather
  than silently running with half-built context).
  `objects_path_resolver: Callable[[video_id, keyframe_id], Path | None]`
  is injected per-candidate rather than assumed from a fixed directory
  layout, since the organizer's real directory structure is still
  unconfirmed (Task 01 §3).
- **Object-source-tagging decision applied**: only the organizer's
  file-based objects (Task 02's `aic_objects` loader, reached through
  `objects_path_resolver`) are wired into `build_frame_context` here.
  Falling back to a live COCO detector (YOLO, or PR #4's new
  `faster_rcnn.py`) when the organizer file is absent is explicitly
  **not** done in this task — it would require resolving the label-space
  mixing risk flagged after the mid-pack rebase, which needs its own
  design decision, not a silent default. Recorded as deferred follow-up,
  not an oversight.
- `src/mvr/api/app.py` — `LazyAnswerReasoner` (lazy-loads
  `HuggingFaceCausalLmBackend` once, only on first `answer()` call, fails
  closed exactly as Task 03 designed) plus `POST /search/qna`, registered
  **unconditionally** like `/search/kis` (degrades to 503 via
  `RuntimeError` when unconfigured, never silently absent from the
  OpenAPI surface) — per the Task 01 plan. Reuses the *same*
  `kis_pipeline_resolver`/`kis_aic_registry` injection points as `/search/
  kis` rather than adding parallel Q&A-specific ones. New env vars:
  `MVR_QNA_MODEL` (unset ⇒ route always 503s), `MVR_QNA_ALLOW_MODEL_
  DOWNLOAD` (opt-in only). `ocr_engine`/`image_loader`/
  `objects_path_resolver` are **not yet wired to real
  EasyOCR/PaddleOCR/organizer-directory defaults** at the API layer —
  by default `/search/qna` answers from the question and the LLM's own
  knowledge only, with empty OCR/object context. Wiring real defaults is
  deferred to Task 06 (benchmark), once real AIC data is available to
  verify against, rather than guessed now.
- `src/mvr/api/schemas.py` — `QnaSearchRequest`
  (`event_description`/`question`/`top_k`), `QnaHit` (adds
  `answer`/`answer_source` to the KIS hit shape), `QnaSearchResponse`.
- Tests: `tests/unit/retrieval/test_qna.py` (13 cases: validation,
  frame-resolution reuse/dedup/overfetch parity with the KIS suite, OCR/
  image-loader wiring, objects-path-resolver wiring, evaluator-identity
  conversion) and 5 new cases appended to `tests/unit/api/test_app.py`
  (ranked/answered response contract, validation 422s, 503 when reasoner
  or retrieval backend unconfigured, unresolvable candidates dropped).
  Full suite: 263 passed, 5 skipped (no regressions from Task 04's 245
  passed, 5 skipped).

## 3f. Post-Task-05 work on real organizer data

Real AIC data became available (`D:\AIC26_data`, `D:\AIC_query`), which
closed several Task 01 unknowns and changed some priorities.

### Data findings (measured, not assumed)

- **Objects JSON schema confirmed.** The Task 01/02 assumption was correct:
  `detection_class_entities` / `detection_scores` / `detection_boxes`, with
  boxes as `[ymin, xmin, ymax, xmax]` normalized to [0,1] (cross-checked
  against the organizer's own FiftyOne baseline notebook, which converts
  them to `[x, y, w, h]`). Numeric fields are JSON **strings**, which
  `float()` already handled. Parsed 500 random real files: 0 errors.
- **Organizer asset coverage is very uneven** (873 videos total):
  CLIP features 873/873, map-keyframes 873/873, media-info 873/873,
  keyframes 850/873, but **objects only 59/873 (6.8%)** — the organizer
  ships an objects *sample*, not full coverage. Of the 60 videos indexed
  for testing (L21+L22), only 2 have object files. **Object context is
  therefore mostly absent at scale; OCR is the dominant per-frame
  evidence source.**
- **Query modality analysis** (9 real Q&A queries, inferred from query text,
  not yet verified against the videos): ~7/9 need OCR (on-screen text,
  signs, recipe cards, exam questions), ~3/9 may need ASR (spoken-only
  facts), and only 1/9 truly needs visual counting (a VLM). This is
  consistent with the SOICT 2025 winner paper, whose Q&A relied on LLM
  reasoning over OCR/ASR/object evidence rather than a frame-reading VLM.
- **media-info is an unused free signal**: YouTube title/description/
  keywords/publish_date at 100% coverage, not yet wired into `FrameContext`.

### Real end-to-end verification

Built a semantic index over L21+L22 (60 videos, 16,896 keyframes) and ran
the full pipeline on real queries. Retrieval, frame resolution, OCR, and
answer generation all execute correctly; OCR text demonstrably reaches the
prompt. **Answer quality is blocked by machine capacity, not by the
pipeline**: the dev machine has 3.8GB RAM and no GPU, where a 1.5B model
fails to load (Windows paging-file error) and a 0.5B model echoes the
question back instead of answering.

### Threshold ownership (deliberate reversal)

An earlier commit applied the organizer baseline's 0.4 detection threshold
inside the Q&A context builder, measured to cut real files from 100.0 to
4.7 detections/frame (32.1 → 3.2 distinct labels). That policy was then
**removed from Q&A** at the user's direction: choosing a confidence
threshold belongs to the indexing/enrichment layer that produces the
detections and already carries its own (`yolo.py` 0.4, `faster_rcnn.py`
0.5). Q&A now owns no threshold value — `build_frame_context` and
`search_qna` default to `0.0`, and `run_qna_pipeline.py` exposes
`--min-object-confidence` for the operator. The loader's `min_confidence`
parameter remains as a *capability*, not a policy. The indexing owner has
been asked to deliver pre-filtered detections.

### Additions

- `image_path` on `QnaCandidate`/`QnaHit` so the human-review step (and the
  UI being built by a teammate) can display the candidate frame. The API
  maps it to a servable `/keyframes/...` URL and returns `null` rather than
  leaking a local absolute path when it cannot serve the file.
- `src/mvr/submission/qna_export.py` — validation + local export for the
  three-field contract. Dedup deliberately differs from KIS: only exact
  `(video_id, frame_id, answer)` triples are duplicates, because the
  organizer scores MAX over top-k and hedging one answer across frames (or
  several answers on one frame) is a legitimate use of the 100-slot budget.
- `src/mvr/enrichment/qna_batch.py` + `scripts/kaggle_qna_generate.py` +
  `docs/QNA_KAGGLE_RUNBOOK.md` — text-only offline batch answering on
  external GPU (Kaggle), since a live inference server is impractical there
  (~20 min idle timeout) while "Save & Run All" batch mode is not. Only
  prompts leave the machine; no video or keyframes. Retrieval is not re-run
  on merge, so the expensive local OCR pass happens once.

### Explicitly out of Q&A scope (team boundaries)

Confirmed by tracing shared code: the **text/image encoder**
(`enrichment/embeddings/openclip.py`) is used by one shared
`LazyFaissSemanticService` serving `/search/semantic`, `/search/kis` **and**
`/search/qna`, and by a teammate's `feature_extraction.py` (PR #4).
Switching to the organizer's precomputed CLIP features — attractive, since
they cover all 873 videos and would remove hours of CPU encoding — would
change KIS and TRAKE retrieval too, and requires matching the OpenAI CLIP
weights the organizer used (not the `laion2b_s34b_b79k` weights currently
indexed). **That is a team-level decision, not a Q&A one.** Likewise ASR,
running detectors over the full corpus, and multi-embedding RRF.

## 3g. Retrieval work and the first real measurement

### Full-corpus index from organizer features

Re-encoding all 873 videos locally was ~15 CPU-hours; the organizer ships
CLIP ViT-B/32 features for every video, so
`scripts/build_qna_index_from_organizer_features.py` assembles
`artifacts/indexes/qna_organizer_all` (850 videos, **166,550 vectors**) in
about four minutes. Two traps found while doing it, both silent:

- The shipped float16 vectors have norms ~0.9995, outside `FaissIndex`'s
  1e-4 tolerance; they are renormalized to exact unit length.
- **QuickGELU.** `open_clip` accepts `ViT-B-32` + `pretrained="openai"` but
  builds the wrong activation, warning only. Measured 0.3527 vs **0.3642**
  for the correct pairing. The manifest records `ViT-B-32-quickgelu`, and
  every consumer builds its encoder from the manifest, so both this index
  and the older LAION one coexist safely.

### Query translation

`src/mvr/enrichment/qna_translation.py`, opt-in via `--translate-query`.
Only the retrieval query is translated; the question reaches the reasoner
untouched. Measured over the full index:

| query | no translation | with translation |
|---|---|---|
| p1-15 | #7709 | **#1** |
| p1-19 | #5877 | #153 |
| p2-3  | #1483 | #113 |
| p3-13 | #1703 | **#38** |
| median | 3790 | **75** |

The MT model translates proper nouns literally ("Khánh Hòa" → "South
Korea"); it still wins by ~50x because CLIP leans on scene description
over names. A name-preserving translator should do better.

### media_info as a diagnostic

`qna_media_info.py` + `find_qna_source_videos.py`. Covers all 873 videos
(objects cover 59), and found source videos that retrieval could not:
`L24_V011` for p1-19 and, after the user corrected a wrong assumption
about the video genre, `L30_V078` ("Lớp học 0 đồng cho người yêu bếp") for
p1-22 — a title scan beat image search for queries whose distinguishing
content is abstract ("a class", "teaching others").

### First measurement against human-verified ground truth

Three p1 queries verified by a human (`artifacts/qna/ground_truth.json`).
Both pipelines score **Final = 0.000**, but the split matters:

| query | right video | GT keyframes | retrieved | gap |
|---|---|---|---|---|
| p1-15 | **ranks #1, #2** | 34, 35 | 27, 30 | **4 keyframes (~10s)** |
| p1-19 | absent from top-20 (rank 153) | 185–188 | — | — |
| p1-22 | absent from top-20 | 30, 31 | — | — |

**Video-level retrieval already works; frame-level precision is the
failure.** The organizer requires `frame_id` inside the interval, so a
4-keyframe miss scores the same as a completely wrong answer.

### Next steps this points to (highest value first)

1. **Temporal expansion.** When a video ranks highly, submit its
   neighbouring keyframes too. This is what converts p1-15's near-miss
   into a hit. -- **done, see 3h.**
2. **Use the full 100-answer budget.** Runs currently use `--top-k 20`,
   spending the budget on 20 *different* videos while the top video gets
   two slots. -- **done, see 3i.**

Both live in `search_qna`, and the benchmark now makes each change
measurable.

### Verifying ground truth (two mistakes worth not repeating)

`verify_qna_ground_truth.py` now rejects both, because both happened and
each silently scores 0 forever: a mistyped `video_id` (`L030_V072`), and
an interval containing no keyframe (`[186,188]`, nearest keyframes at 153
and 248) — the system can only emit frame indices of existing keyframes.

### Environment (this machine)

```
repo            D:\AIC26 (branch feat/qna-competition-ready)
venv            D:\AIC26\.venv  (py -3.12, extras dev,api,semantic,qna,ocr)
organizer data  D:\AIC26_data          (raw layout: Keyframes_L*/, map-keyframes-aic25-b1/, ...)
registry root   D:\AIC26_test_root     (normalized layout AicDatasetRegistry needs)
queries         D:\AIC_query           (9 Q&A queries across p1/p2/p3)
index           artifacts/indexes/qna_organizer_all   (850 videos, quickgelu+openai)
ground truth    artifacts/qna/ground_truth.json       (3 p1 queries, human-verified)
```

Machine limit: 3.8GB RAM, no GPU — answer generation runs on Kaggle via
`docs/QNA_KAGGLE_RUNBOOK.md`.

## 3h. Temporal expansion (step 1 of the 3g next steps)

`search_qna(temporal_radius=..., neighbour_cost=...)`, opt-in, default
`0` — retrieval's ranking untouched. When a video ranks highly the
neighbouring keyframes are predicted too, resolved through the *same*
frame map as the hits, so a neighbour that does not exist is dropped
rather than guessed.

Neighbour ids come from `shift_keyframe_id()`, which shifts the trailing
number of the id *string* so zero-padding and any prefix survive
(`"0027"` → `"0028"`); resolvers key on the exact string, so a rebuilt
`"28"` would silently find nothing.

A neighbour is a hedge, not a hit: it inherits the parent's `score`
because it has none of its own, and carries `temporal_offset` +
`expanded_from` so that borrowing is explicit. Its `timestamp` is `None` —
the parent's belongs to a different frame and interpolating one would
invent data. Each neighbour is re-answered against its own OCR/object
context; copying the parent's answer would defeat the point.

### Sharing the 100-answer budget

The budget funds two different bets — *more videos* (retrieval picked the
wrong one) and *more frames of the same video* (right video, wrong
moment). Each `(hit index, offset)` pair costs
`hit_index + neighbour_cost * abs(offset)`, and the submission order is
that cost ascending. One knob spans the whole range: high (50) submits
every retrieval hit before any neighbour, low (0.1) exhausts the top
hit's neighbourhood first, `1.0` treats one rank down and one keyframe
aside as equally speculative. Ties break toward the smaller `abs(offset)`,
so a real retrieval hit always beats a hedge landing on the same cost —
and `R@1`, one fifth of the final score, is untouched at any setting.

### Measured (retrieval reachability, top-k 100, translated query)

p1-15 was the whole motivation: right video at ranks #1/#2, answer four
keyframes further on, scoring zero.

| | rank of a correct frame |
|---|---|
| baseline (`radius 0`) | **absent from all 100** |
| `radius 5, cost 1.0` | #28 |
| `radius 5, cost 0.5` | #19 |
| `radius 5, cost 0.2` | #13 |

Radius sweep at `cost 0.5`: radius 2 → absent, 3 → absent, **4 → #18**,
5 → #19, 8 → #19. The first radius that works is exactly the 4-keyframe
gap 3g measured, which is mechanical confirmation rather than a
coincidence — and a radius past the gap costs nothing, because the cost
ordering submits far neighbours late.

**Caveat, and it is a large one: n=1.** Only p1-15 can move — p1-19's
video sits at rank 153 and p1-22's is absent from the top 20, so no
expansion reaches either. The sweep numbers above are therefore a
mechanism check, not a tuned hyperparameter, and the library default
stays at the neutral `1.0`. Fitting `0.2` to a single query would be
overfitting: a low cost also means burning ~11 slots per video, covering
only ~9 videos out of 100 answers, which is a real loss whenever the top
video is wrong. Choosing a value honestly needs more verified ground
truth.

Final Score is still 0.000 across all three, because these runs use
`--export-prompts` (no LLM, placeholder answers). This step fixed the
*retrieval* half of p1-15 only; the answer half still has to land.

### Neighbour images

Neighbours are not retrieval hits, so the index supplied no `image_path`
for them — and OCR is the dominant Q&A signal, so an unresolved neighbour
reaches the reasoner with almost nothing. `search_qna` takes a
`keyframe_path_resolver` (wired to `AicDatasetRegistry.keyframe_path` in
`run_qna_pipeline.py`) which also backfills any primary hit whose backend
returned no image.

## 3i. The 100-answer budget (step 2 of the 3g next steps)

`run_qna_pipeline.py --top-k` now defaults to **100**, the competition
maximum, instead of 5. The final score averages R@1/5/20/50/100, and three
of those five cutoffs are structurally unreachable below 20 answers — a
smaller default discards score that costs nothing but compute to claim.

Raising it is safe in the strict sense, which was verified rather than
assumed: the expansion plan is sorted globally by cost and truncated at
`top_k`, so **the first 20 rows of a 100-answer run are identical to a
20-answer run** (checked on all three ground-truth queries). R@1, R@5 and
R@20 therefore cannot degrade when the budget grows; only R@50 and R@100
can move, and only upward.

### Measured: it bought nothing here, and that is worth recording

| top-k (radius 4, cost 0.5) | p1-15 | p1-19 | p1-22 | Final |
|---|---|---|---|---|
| 20 | frame at #18 | absent | absent | 0.000 |
| 100 | frame at #18 | absent | absent | 0.000 |

Identical. p1-15's correct frame already lands at #18, inside 20; p1-19
and p1-22 have no correct video anywhere near the top, so eighty more
slots of the wrong videos change nothing. **The change is insurance, not
a gain** — five times the LLM calls for zero measured benefit on n=3. It
is still the right default, because "the correct frame happened to land
at #18" is a property of these three queries, not of the system.

### How the budget actually splits

Making `expanded_from` survive the batch round-trip (below) exposed the
split for the first time. At `radius 4, cost 0.5` a 100-answer run is
**13 retrieval hits and 87 temporal hedges** — 100 slots covering only
~13 videos. That is the whole trade-off in one number, so it was worth
measuring the extremes:

| configuration | videos covered | correct frame found |
|---|---|---|
| `radius 0` (pure retrieval, 100 hits) | 73 | none of the three |
| `radius 4, cost 50` (breadth-first) | 53 | none of the three |
| `radius 4, cost 0.5` (depth-leaning) | ~13 | **p1-15 at #18** |

Depth wins decisively, and the reason is visible in the retrieval ranks:
video-level retrieval is either very good (p1-15 at ranks #1–2) or
hopeless (p1-19 at #153). There is little in between, so buying breadth —
73 videos instead of 13 — reaches nothing that 13 did not already reach.
Spending the budget on *more frames of a good video* is what pays.

This is still n=3 and all three are p1. The conclusion to carry forward is
the shape of the trade-off, not the specific constants.

### Expansion provenance survives the GPU round-trip

`PromptItem` gained `temporal_offset` / `expanded_from`, both defaulted so
the batches already sitting in `artifacts/` still load unchanged. Without
this a 100-row batch could not be debugged after the GPU host replaced
every answer — the batch file is the only record, since the GPU host never
sees the retrieval run, and ~87% of the rows are hedges.

`import_qna_answers.py` restores the fields, prints the offset per row,
and closes with a budget line: slots used out of 100, split into retrieval
hits vs temporal hedges, plus any lost to unparseable model output. Blank
or unparseable answers were already dropped by `merge_answer_batch` rather
than fabricated, so the larger budget does not risk failing the export —
`validate_qna_predictions` still refuses blanks by design.

## 3j. First end-to-end score, and the instrument that was hiding it

The whole chain ran for the first time: retrieval → temporal expansion →
keyframe image → EasyOCR → Qwen2.5-7B on Kaggle → import → score.

**Mean Final Score 0.2000** (p1-15 = 0.600, the other two = 0.000), against
0.000 before. Not a good score. But the first non-zero one, and the run
that proved every component works when connected.

### Two bugs had to be fixed before it could run at all

Both were latent because every previous measurement passed `--no-ocr`, so
the OCR path had never once executed against a real engine, and temporal
expansion is what finally made it reachable.

1. `build_frame_context` guarded on `ocr_engine is None` but not `image is
   None`. An expanded neighbour has no image from the retrieval backend, so
   EasyOCR got `None` and raised `UnboundLocalError` from inside its own
   input reformatting, aborting the query.
2. `run_qna_pipeline` resolved neighbour images through `AicDatasetRegistry`,
   whose root can be a smaller normalized tree than the one the index was
   built from. Every neighbour outside that tree silently lost its image —
   and with it nearly all of its Q&A signal. The index's own metadata
   already stores a path for every keyframe it holds, which is exactly the
   set neighbours are drawn from, so it is consulted first now.

### What actually happened on p1-15

```
retrieval        L30_V072 keyframes 027, 030   (right video, ~9s early)
expansion +4     keyframe 034, frame 1745      (start of GT interval)
EasyOCR          "... Xa Giang Ly huyen Khonh Vinh, tinh Khanh Hoa ..."
Qwen2.5-7B       {"answer": "Xa Giang Ly", "source": "context-grounded"}
ground truth     "Giang Ly"
```

Every link was load-bearing: without expansion the frame is never reached,
without the two fixes above it arrives with no image and no text.

### The instrument was reporting 0.000 for a correct answer

`qna_answer_matches` did normalized equality, so `"xa giang ly"` failed
against `"giang ly"` — for a question that literally asks *"what is this
**commune** called?"*. The organizer uses a human judge, who would not call
that wrong.

The module docstring had anticipated exactly this: widen it "explicitly in
a later task rather than silently" if it proved too conservative on an
observed real query. So:

- **strict** now folds diacritics and strips *one leading* Vietnamese
  administrative-unit word (`xa`, `phuong`, `thi tran`, `thi xa`, `huyen`,
  `quan`, `tinh`, `thanh pho`). Leading and one only — `"Xa Giang Ly huyen
  Khanh Vinh"` is more than the answer and still scores 0. Diacritic
  folding matters because answers arrive from OCR, which mangles tone marks
  routinely.
- **lenient** (`qna_answer_matches_lenient`) adds containment as an explicit
  upper bound, and is never the default.
- `evaluate_qna(..., matcher=...)` threads the choice through, and the
  benchmark prints both. One number cannot express "somewhere between", and
  quietly adopting the generous end would have us stop fixing what still
  needs fixing.

On this run both bounds land on 0.2000 — the answer is unambiguously
correct, not a judgment call.

### The other two are retrieval failures, definitively

| query | candidates from the right video | why |
|---|---|---|
| p1-15 | 12 of 100, one inside the GT interval | works |
| p1-19 | **0 of 100** | right video ranks #153 |
| p1-22 | **0 of 100** | right video absent from the top hits |

Temporal expansion cannot help: it only expands around videos retrieval
already found. Nor can budget reallocation — the widest measured setting
covers 53–73 videos, still short of #153. These two need a *different
signal*: p1-19 asks about a **recited poem** (ASR), p1-22 about an
**on-screen title** (OCR + BM25). Both are the lexical branch.

### Dropped answers are not a bug

p1-15 lost 43 of 100 slots, which looked alarming. 40 of those were
`{"answer": ""}` — the model honestly declining on frames whose OCR held
nothing. An empty answer scores 0 anyway, so dropping it costs nothing. The
remaining 3 had an invalid `source`. p1-19 lost 1, p1-22 lost 9.

### n=3 is permanent, and that changes the method

The p2/p3 queries are reference queries whose answers lie **outside** the
data we hold, so they can never be verified. Three queries is the ceiling —
and since only one is currently scorable, the answer half is effectively
n=1.

Consequences, stated plainly so nobody re-derives them later:

- `temporal_radius=4` and `neighbour_cost=0.5` were chosen on **n=1**. They
  are frozen defaults, **not** tuned optima, and further fitting against
  these three queries is overfitting to three queries.
- Improvements must now be argued from **mechanism** and verified by
  **component-level tests**, not by moving a Final Score that only one
  query can move.
- The one way to get real n is a **synthetic set**: start from a known
  frame, OCR it, generate a question whose answer is a span of that text.
  Ground truth is exact by construction — no human verification — because
  the frame is the input, not the search result. It measures answer
  extraction and frame precision well; it measures *retrieval* only as
  realistically as the synthesized query, which must be generated from
  media_info rather than the frame's own OCR or it leaks the answer.

## 3k. Reading Task05A documents, and neighbours by time

Task05A (`src/mvr/indexing/task05a.py`) landed on `main` and emits one
document per keyframe carrying OCR, ASR, object counts, CLIP row and the
competition `frame_idx`. Q&A now reads it instead of assembling context
itself.

### What the reader replaces

Query-time context was: open the keyframe JPEG, run EasyOCR, read the
organizer objects JSON -- **~6.7s per candidate on CPU**, which is why
every measurement before this ran with `--no-ocr`. `frame_context_from_record()`
replaces all of it with a file read.

Taking `frame_idx` from the document also removes `resolve_original_frame_id`
from this path, and with it the failure mode where an unresolvable mapping
silently produced `frame_id=None` for every candidate.

### ASR, which Q&A never had

The document's `asr.joined` reaches the prompt as its **own line**, not
merged into the OCR blob: a question about what someone *said* and one about
what is *written* rest on different evidence, and collapsing them lets the
model answer from the wrong source with nothing marking the swap.

This is aimed at a specific measured failure. Of the three verified queries,
p1-19 asks what two lines of poetry are -- and the poetry is *recited*.

### Objects arrive as counts without detections

Task05A aggregates objects down to `labels` + `counts`, keeping no
per-detection confidence or box. `FrameContext` holds those in
`aggregated_object_counts` rather than back-filling a `FrameObjects`:
inventing detections to satisfy a dataclass would fabricate data the frame
never had.

### image_path was being dropped

`CanonicalKeyframe` reads `image_path` from the mapping CSV (task05a.py:69,
:240) and `_project_document` left it out, so the one field that lets a
consumer *see* the frame was discarded at the last step. Q&A ranks up to 100
candidates and the flow ends in human or VLM review; nobody adjudicates a
frame from `frame_idx: 578`, and a search UI cannot render an integer. Now
emitted and required as a non-blank string by `validate_task05a_document`.

### Neighbours by time, not by arithmetic on an id

`shift_keyframe_id` bumps the trailing number of the id string. That holds
for organizer keyframes, one running sequence per video (`"027"` ->
`"028"`). It breaks completely on the ids this repo's own ingestion
produces:

```
L21_V010_S0007_K01
         ^^^^ ^^^
         shot  index WITHIN that shot
```

`make_keyframes` takes the shot midpoint and samples extra frames only past
`--long-shot-seconds` (default 12s), so ordinary 3-8 second shots yield
exactly `K01`. Asking for `+1` requests `K02`, which does not exist, and
**expansion returns nothing -- silently, for most frames**. The real next
keyframe is `S0008_K01`.

`KeyframeNeighbourIndex` orders each video's keyframes by time and steps
through that ordering. It accepts `frame_idx` or `timestamp` (both increase
monotonically within a video, so they agree) and `keyframe_id` or
`frame_id`, which lets it be built from the persisted FAISS metadata --
that stores `timestamp` and no `frame_idx`, and re-encoding 166,550 vectors
to add one field would be absurd.

Verified against the real 166,550-keyframe index:

| | old `shift_keyframe_id` | `KeyframeNeighbourIndex` |
|---|---|---|
| `L30_V072` `030` `+4` | `034` | `034` |
| `L30_V072` `041` `+1` (last keyframe) | **`042`** -- does not exist | `None` |

Same answer where it worked, and it stops at the end of a video instead of
inventing an id that then fails to resolve.

**Deliberately does not stop at shot boundaries.** It sounds right -- a
neighbour in another shot is another scene -- but the one measured case
argues against it: p1-15's answer sat 4 keyframes past the retrieval hit,
roughly 9 seconds of broadcast news, almost certainly across several shots.
Stopping there would discard the only hit this system has ever scored.

## 3l. Fusing lexical retrieval (Task05B) into Q&A

`feat/lexical` landed on `main` with an Elasticsearch BM25 index over
exactly two fields:

```python
SEARCHABLE_FIELDS = ["ocr_text", "asr_text"]
```

Those are the two signals the measured failures need. Of the three verified
queries, two score zero and both are **retrieval** failures -- neither has a
single candidate from the correct video, so answering and budget changes
cannot reach them, and temporal expansion cannot either since it only
expands around videos retrieval already found:

| query | asks about | signal |
|---|---|---|
| p1-19 | "Hai câu thơ đó là gì?" -- poetry, **recited** | `asr_text` |
| p1-22 | the recipe title, **on screen** | `ocr_text` |

Finding those two videos by hand earlier meant scanning `media_info`
titles: doing BM25's job manually.

### `QnaFusedRetriever`

Semantic (FAISS/CLIP) + lexical (BM25), fused with the existing `rrf_fuse`,
exposed as `search_hybrid` so `search_qna` consumes it unchanged. Opt-in via
`--elasticsearch-url` + `--lexical-index`; without them the pipeline is
semantic-only exactly as before.

**Fusion keys on `(video_id, keyframe_id)`, never the bare id.** The real
index holds **166,550 keyframes with only 632 distinct ids** -- `"001"`
exists in roughly 850 videos. Fusing on the id alone would merge unrelated
frames from unrelated videos into one ranked entry.

Worth flagging to the team: `mvr.api.app.MultimodalService.search_hybrid`
does key RRF on `frame_id` alone. That is safe while ids are this repo's own
globally-unique `<video>_S0007_K01`, and unsafe for organizer-style ids. Not
changed here -- different code path, different owner -- but it is the same
trap.

### `frame_id` vs `frame_idx`

Two names differing by three characters, meaning entirely different things:

```
frame_id   in a search hit  ->  the KEYFRAME id       ("034", "V_S0007_K01")
frame_idx                   ->  the ORIGINAL-VIDEO frame  (1745)
```

`LexicalSearchHit` carries both, so the adapter maps them explicitly rather
than guessing. Swapping them would make every candidate fail to resolve --
silently, since an unresolvable candidate is simply dropped.

Because the lexical document states `frame_idx`, `search_qna` now uses it
directly and skips keyframe->frame resolution for those hits. **Only at
offset 0**: `frame_idx` describes the hit's own keyframe, and a temporally
expanded neighbour is a different frame. Reusing the parent's index would
submit the parent's frame under the neighbour's id -- a wrong answer that
looks perfectly well-formed. When both a hit `frame_idx` and a resolver
mapping exist and disagree, this raises rather than picking a winner.

### Degradation is asymmetric on purpose

A failing **lexical** branch is caught and logged: Elasticsearch being
unreachable should cost recall, not the query. A failing **semantic** branch
propagates: it is the primary path, and swallowing its failure would hide an
outage behind quietly worse results.

### Not yet measured

This cannot be scored until Elasticsearch is running with the Task05B
documents indexed, which needs `build_index_documents` over the corpus
first. The adapter and its 14 tests are ready; the measurement is not.

## 3m. A synthetic set, because n=3 is permanent

The p2/p3 queries are reference queries whose answers lie outside the data
we hold, so three verified queries is the ceiling -- forever. Only one of
the three is currently scorable, which means every decision about *answering*
is being made on **n=1**.

### Inverting the direction

Real ground truth runs query -> search -> frame -> a human confirms, and the
confirmation is the expensive, error-prone step. It already went wrong once:
an early verification picked L26 for p1-22 before the user corrected it.

This runs the other way: **choose the frame, then derive the question.**
`video_id` and `frame_id` are the *input*, so they cannot be wrong. No
search, no human confirmation, nothing to get wrong about identity.

| | guaranteed? |
|---|---|
| `video_id`, `frame_id` | **yes** -- they are the input |
| the answer appears on that frame | **yes** -- enforced in `__post_init__` |
| the answer is what a *human* reads | no -- OCR mangles text |
| the answer is unique to that frame | no -- filtered, not guaranteed |

So it measures **answer extraction**: given context X and question Q, does
the model return span S? OCR errors cancel, because expectation and model
see the same X. It does **not** measure retrieval -- any query derived from
a frame's own text leaks the answer -- and it does not measure human-judged
correctness. Scores are never comparable with `benchmark_qna.py`.

Questions come from deterministic rules, not an LLM: an instrument has to be
reproducible and inspectable, and a question whose provenance nobody can
reconstruct is not a measurement.

### Probing 60 real frames caught a serious flaw

The first run produced three cases and **all three were false positives**:

```
"Hỏi quận này có tên là gì?"  ->  'SỰ CỦA TA TRONG KHÁNG CHlẾN CHỐNG PHÁP'
"Hỏi tỉnh này có tên là gì?"  ->  'Pskov'
```

Cause: the generator folded diacritics away so OCR tone damage would not
defeat the rule. But **Vietnamese diacritics carry meaning** -- `quân`
(army) folds onto `quận` (district), `tính`/`tình` onto `tỉnh` (province).
The rule was firing on unrelated words.

For a ground-truth generator a false positive is far worse than a miss: it
writes a wrong expectation into the benchmark, where a miss only makes the
benchmark smaller. Two fixes, both accepting lower yield for precision:

1. **Match unit words with diacritics intact.** The OCR-damaged p1-15 case
   ("huyên" for "huyện") is now lost. Correct trade.
2. **Reject names containing a Vietnamese function word** (`của`, `và`,
   `trong`, `ta`, ...). Capping the name at four words was not enough on its
   own: it shortened the seven-word headline to "SỰ CỦA TA TRONG", equally
   wrong and merely less obvious.

Re-running the same 60 frames left exactly one case -- and inspecting it
showed the generator was right and the earlier read was wrong:

```
OCR: "... Việt Nam Hàn Quốc Tỉnh Pskov của Nga mời các già gọi"
     -> "Hỏi tỉnh này có tên là gì?" -> "Pskov"
```

Pskov *is* a Russian oblast. A true case, not a false one.

### Yield, and why that is acceptable

Roughly 2% of sampled frames produce a case: most keyframes carry no text
matching a rule. Reaching n≈20-50 needs ~900-2000 frames at ~6.7s of CPU OCR
each. That is slow but it is a one-off, and n=20 against n=1 is the whole
point.

`drop_ambiguous_cases` then removes answers seen on more than a few frames.
A banner like "THỜI SỰ" appears everywhere, and since scoring requires
`frame_id` inside the interval, such a case would mark a *right* answer
wrong -- understating performance, which is the more dangerous direction: it
sends you chasing problems that are not there.

### What it unlocks

Three things that are currently unmeasurable at n=1:

- **Answer formatting.** "Xã Giang Ly" vs "Giang Ly" was a single data point
  that cost a whole query until the matcher was widened. With n=20+ the
  frequency of that class is visible instead of guessed.
- **`qna_answer_matches` calibration** -- where the approximation is too
  strict or too loose, measured rather than argued.
- **Prompt and model changes**, which currently have no way to show whether
  they helped.

## 3n. The mapping adapter, and a file nobody opened

`scripts/build_keyframe_mapping.py` generates `keyframe_mapping.csv` from
the organizer's data in one command. It exists because the team decided the
organizer's 166,550 keyframes are the single canonical identity, and every
consumer already reads identity from that one CSV:

    indexing.task05a.load_canonical_mapping
    enrichment.ocr.docker_ocr.process_dataset_docker_ocr
    enrichment.feature_extraction.sort_keyframes_deterministically
    enrichment.asr.mapping
    indexing.semantic_builder.build_semantic_index

so adopting them is a generation step, not a refactor. No teammate's
pipeline changes.

Measured on the real corpus: 873 videos with a mapping CSV, 850 with
downloaded keyframes, **166,550 rows in 6.3 seconds**. The known ground
truth resolves exactly -- `L30_V072 / 034 -> frame_idx 1745`.

### The bug this uncovered

The 6-column adapter committed in `6e04f4c` produced a file that **not one
downstream module could load**. `shot_id` was left blank, and nine call
sites across `task05a`, `feature_extraction`, `docker_ocr`, `scene_text`
and the ASR mapping all do `int(row["shot_id"])`:

    Task05AValidationError: invalid numeric identity in mapping line 2

The docstring justified the blank by citing `indexing.lexical_index`, which
accepts a null `shot_id`. That is true and irrelevant: a null there is a
*document field*, this is a *CSV column*. The wrong consumer was checked.
`ordinal` was missing outright, so `sort_keyframes_deterministically` would
have raised `KeyError` as well.

Every column assertion in the test file passed throughout. They asserted on
the header and never opened the file with a real reader -- which is why the
tests now end with `test_task05a_can_load_what_this_writes` and
`test_feature_extraction_can_sort_what_this_writes`, and why the script
verifies its own output through both before exiting. **Reading a file's
header is not the same as reading the file.**

### What `shot_id` is without shot detection

Each organizer keyframe becomes its own shot: `shot_id` is the organizer's
own `n`, `ordinal` is 1. This is the honest degradation rather than a guess
at boundaries -- the organizer's keyframes are already shot-aware (measured
spacing 0.04s-7.0s, 30-150 distinct values per video, i.e. cut-driven, not
uniform), so it restates their sampling instead of inventing cuts.

The cost is grouping: `scene_text` and `docker_ocr` dedupe OCR *within* a
shot, and singleton shots mean no dedupe, so every keyframe is read
independently. That is the safe direction. The alternative -- one shot per
video -- would make temporal OCR sample a single anchor for ~190 keyframes.

`--shots-json` takes the team's own `shots.json` and fills real shot ids,
numbered from 1 to match `ingestion.shot_detection`; `ordinal` then counts
position within the shot exactly as `ingestion.keyframes` does. A keyframe
no supplied shot contains gets `UNKNOWN_SHOT_ID = 0`, which cannot collide
because the detectors number from 1. The script refuses a `shots.json`
whose own numbering is not `1..n` in start-frame order rather than
renumbering silently, since renumbering would break every join back to that
file and surface far away as OCR attached to the wrong shot.

### Left for the owner of `enrichment/ocr/validator.py`

`validate_ocr_output` keys its duplicate check on `keyframe_id` alone. The
organizer's 166,550 keyframes carry only **632 distinct ids** -- `"001"`
exists in ~850 videos -- so against the real mapping it stops at row 309:

    ERROR: Duplicate keyframe_id in mapping: 001
    (L21_V001, then L21_V002)

It needs `(video_id, keyframe_id)`, the same fix the Elasticsearch `_id`
and the fusion key already carry (3l). Not changed here: it is outside Q&A
and belongs to whoever owns the OCR pipeline. Nothing else is blocked by it
-- the validator is a post-run check, not a step in the pipeline.

## 4. Branch / task-pack plan (confirmed with the user)

Branch: `feat/qna-competition-ready`. Tasks, one at a time, stop-and-report
after each, mirroring the KIS task pack's discipline recorded in
`docs/CODEX_PROGRESS.md`:

1. Contract audit (this document).
2. Structured-context builder (objects JSON loader with counts+bbox, OCR
   wiring).
3. Local-LLM answer reasoner (fail-closed on missing weights, provenance
   tagging, `transformers` added as a declared dependency).
4. Q&A evaluator (R-Score/R@k/Final Score + documented fuzzy-match
   approximation).
5. Orchestration + Top-100 contract + `POST /search/qna`.
6. Benchmark on real data + prediction export.
7. Final docs/report.
