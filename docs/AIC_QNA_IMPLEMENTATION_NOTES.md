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
