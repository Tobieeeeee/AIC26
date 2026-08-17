# AIC KIS Implementation Notes — Repository Contract Audit

Task 01 output. Records the exact current interfaces, identifiers, and conventions
that all later KIS task-pack work (`agent_task/kis_competition_ready/02`–`09`)
must reuse instead of inventing parallel abstractions. No production code was
changed to produce this document.

## 1. Existing retrieval hit/result types

There is **no single unified hit type** across the codebase yet — three distinct
shapes exist depending on path:

- `mvr.retrieval.semantic.SearchResult` (frozen dataclass) — returned by
  `SemanticRetriever.search()` (FAISS path only):
  `rank, frame_id, video_id, timestamp, score, image_path` + `.to_dict()`.
- `mvr.indexing.faiss_index.FaissHit` (frozen dataclass) — lower-level, internal
  to `FaissIndex.search()`: `score, metadata: dict`. `SearchResult` is built from
  this by `SemanticRetriever`.
- `mvr.api.schemas.SearchHit` (Pydantic model) — the API-facing shape, richer:
  `rank, frame_id, video_id, timestamp, score, image_path, ocr_text, caption,
  objects, highlight, rank_lexical, rank_semantic` (last 6 fields optional).
  Built from raw Elasticsearch hits via the private helper
  `mvr.api.app._hit_payload()`, and from `SearchResult` via
  `mvr.api.app._semantic_payload()`.

**For KIS**: reuse `SearchHit` as the outward-facing shape (it already has every
field KIS needs plus optional metadata); do not invent a `KISHit`.

## 2. Frame/keyframe identifier format

Keyframe ID is built in `mvr.ingestion.keyframes.make_keyframes()`:

```python
keyframe_id = f"{shot.video_id}_S{shot.shot_id:04d}_K{ordinal:02d}"
# e.g. "L01_V001_S0002_K01"
```

This identifies a **shot-selected keyframe**, not a raw competition frame number.
The underlying `mvr.ingestion.models.Keyframe` dataclass separately carries
`frame_idx: int` — the actual 0-based frame index in the source video that this
keyframe was extracted from (see §4).

`video_id` is derived from the filename stem: `Path(video).stem` in
`mvr.ingestion.probe.probe_video()` (e.g. `L01_V001.mp4` → `"L01_V001"`).

## 3. Keyframe ID uniqueness

`keyframe_id` is **globally unique** across the whole corpus (it embeds
`video_id` inside the string), not just unique within one video.

## 4. Existing mapping: keyframe → original video frame

Two separate, non-interchangeable mappings exist / are needed:

1. **This repo's own ingestion pipeline** (`mvr.ingestion.keyframes` +
   `mvr.ingestion.extract`) already anchors every keyframe it generates to a
   real original-video frame index: `Keyframe.frame_idx` is the exact frame
   number chosen by shot-midpoint/long-shot sampling
   (`mvr.ingestion.keyframes.select_frame_indices`), and
   `mvr.ingestion.extract.extract_keyframes()` extracts that literal frame via a
   single-pass FFmpeg decode (counts frames from 0, matches on
   `current_frame_idx == target_frame_idx`). So for keyframes produced by this
   pipeline, `frame_idx` **is already** the KIS-contract frame id — no
   additional mapping needed.
2. **Organizer-provided AIC keyframes** (`Keyframes_L21.zip`, matched against
   `map-keyframes-aic25-b1.zip`) are a **different, currently unhandled** input.
   Per the team's data handoff notes, each video ships a
   `map_keyframes/<video_id>.csv` with columns `n, pts_time, fps, frame_idx`.
   **`frame_idx` is the correct column to use as the KIS `frame_id`; `n` (the
   organizer's sequential keyframe number) must not be used.** No code in this
   repo currently parses this CSV or maps an organizer keyframe filename (e.g.
   `L21_V001/003.jpg`) to its `frame_idx`. **This adapter is net new — it is
   exactly the scope of Task 02** (`FRAME_REGISTRY_AND_AIC_ADAPTER.md`).

## 5. Existing video probe metadata

`mvr.ingestion.probe.probe_video(path: Path) -> VideoInfo` (ffprobe-based).
`VideoInfo` (`mvr.ingestion.models`): `video_id, path, fps, frame_count,
duration`. `Shot` (`mvr.ingestion.models`) additionally exposes computed
`.frame_count` and `.duration` properties.

## 6. Existing original-frame extraction/decoding utility

`mvr.ingestion.extract.extract_keyframes(video_path: Path, frames:
list[Keyframe], output_root: Path) -> None`. Single FFmpeg-decode-pipe +
FFmpeg-encode-pipe (`rgb24` raw frames, no OpenCV), exact frame matching by
counting decoded frames, writes JPEGs to each `Keyframe.image_path`.

**Compatibility note for Task 05** (exact-frame refinement): this function's
signature takes a **batch of already-constructed `Keyframe` objects** (each
needing a `frame_idx` + `image_path`), not an arbitrary single `frame_idx`
lookup. Any KIS exact-frame-refinement utility that needs "give me frame N from
video V" as a standalone call will need a thin wrapper around this — reusing
its exact single-pass-decode technique — rather than calling it as-is with one
synthetic `Keyframe`.

## 7. Existing semantic/text/object/hybrid service interfaces

- **Semantic (FAISS)**: `mvr.retrieval.semantic.SemanticRetriever(text_encoder,
  index).search(query: str, top_k: int) -> list[SearchResult]`.
- **Semantic (Elasticsearch kNN)**:
  `mvr.retrieval.semantic.ElasticsearchSemanticRetriever(text_encoder, client,
  index_name="keyframes", vector_field="clip_vector",
  num_candidates=100).search(query, top_k) -> dict` (raw ES response).
- **Lexical/OCR (BM25)**: `mvr.retrieval.lexical.search_lexical(client, query,
  *, top_k, person_min, car_min, index_name) -> dict` (raw ES response);
  request-only variant `build_lexical_request(...)`.
- **Object**: `mvr.retrieval.object_search.parse_object_conditions(want: str) ->
  list[tuple[str, int]]` (validates against `COCO_CLASSES`, raises `ValueError`
  on bad input) and `search_objects(client, want, *, top_k, index_name) -> dict`.
- **Hybrid fusion** (pure functions, no I/O):
  `mvr.retrieval.hybrid.rrf_fuse(ranked_lists, *, rrf_k=60, top_k=10) ->
  list[tuple[id, score]]`, `weighted_fuse(lexical_scores, semantic_scores, *,
  alpha=0.25, top_k=10) -> list[tuple[id, score]]`, `min_max_normalize(scores)`.
- **API-level orchestration** (what actually wires the above together per
  request): `mvr.api.app.LazyFaissSemanticService` (lazy FAISS+OpenCLIP load,
  `.search(query, top_k)`) and `mvr.api.app.ElasticsearchMultimodalService`
  (`.search_text`, `.search_object`, `.search_hybrid`, `.document_count`).

**For KIS orchestration (Task 04)**: call these existing services/functions
directly; do not reload models or rebuild indexes per query, and do not
reimplement RRF/weighted fusion.

## 8. Existing API request/response conventions

Defined in `mvr.api.schemas`, wired in `mvr.api.app.create_app()`:

- Request models (`SemanticSearchRequest`, `TextSearchRequest`,
  `ObjectSearchRequest`, `HybridSearchRequest`) all bound `top_k` as
  `Field(default=5, ge=1, le=100)` — **the competition's max-100 rule is
  already enforced at the schema level for existing endpoints.** A new KIS
  request schema should follow the same `Field(ge=1, le=100)` pattern.
- Response models: `SearchResponse(query, count, results: list[SearchHit])`,
  `HybridSearchResponse(SearchResponse, rrf_k)`.
- `create_app()` accepts **injectable services** (`semantic_service`,
  `multimodal_service`, `keyframe_dir`) defaulting via `_default_*()` factories
  reading `MVR_*` env vars — this is what keeps `tests/unit/api/test_app.py`
  lightweight. A KIS endpoint should follow the same
  factory-with-injectable-service pattern for testability.
- Error convention: invalid input → `ValueError` → HTTP `400`
  (`parse_object_conditions`); unavailable backend/runtime → `RuntimeError` →
  HTTP `503`.
- Routes are registered conditionally: `/search/text|object|hybrid` only exist
  on the app when `multimodal_service is not None` (i.e. Elasticsearch was
  reachable at startup). `/search/semantic` is always registered but returns
  `503` if the FAISS index isn't available.

## 9. Current config conventions

Single file: `configs/default.yaml`, flat YAML grouped by domain
(`ingestion`, `enrichment`, `retrieval`, `elasticsearch`, `api`).

Relevant existing keys: `api.top_k_default: 20`, `api.top_k_max: 100`.

**Existing inconsistency to be aware of (not fixed here, just recorded)**: these
`api.top_k_*` YAML values are **not actually read** by
`mvr.api.schemas` — the `Field(default=5, ge=1, le=100)` bounds are hard-coded
directly in the Pydantic models instead of being loaded from
`configs/default.yaml`. A KIS config addition should decide explicitly whether
to follow the YAML-value pattern or the current hard-coded-Field pattern;
mixing silently would be inconsistent with either.

## 10. Current artifact-ignore rules relevant to KIS

From `.gitignore`: `data/{raw,interim,processed}/*`,
`artifacts/{keyframes,embeddings,indexes,metrics}/*` (all except `.gitkeep`),
raw video extensions (`*.mp4` etc.), model weight extensions, Elasticsearch
distributions, and `agent_output/` + `agent_task/` (AI-agent working files) are
all ignored. **`docs/` is not ignored** — this file, and any future KIS docs
under `docs/`, are expected to be committed normally.

## Confirmed absent (not partially built — do not assume any scaffold exists)

Verified by direct filesystem check, not just by absence from documentation:

- `src/mvr/submission/` — does not exist.
- `src/mvr/query/` — does not exist.
- `src/mvr/pipeline/` — does not exist as a dedicated module (offline/online
  orchestration currently lives inline in `mvr.ingestion.pipeline` and the
  API/CLI layers).
- `src/mvr/enrichment/captions/` — does not exist.
- `src/mvr/retrieval/temporal.py` or any sequence/temporal retrieval file —
  does not exist.
- `mvr.evaluation.metrics` only has generic `compute_retrieval_metrics`,
  `recall_at_k`, `mean_reciprocal_rank` (set-membership based). **No official
  KIS evaluator exists** implementing the competition's actual correctness rule
  (`video_id == GT_video AND frame_id ∈ [start_frame, end_frame]`), which is
  fundamentally different from simple ID-set membership. This is exactly the
  scope of Task 03.
