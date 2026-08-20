# Task06A Unified Search Contract Specification Report

## Final verdict

**PASS_WITH_OPEN_QUESTIONS**

The future UI-facing `POST /search` contract is specified without changing
production code. Request and response identity, numeric frame/timestamp
semantics, capability-level retrieval modes, top-k behavior, structured errors,
empty results, object/temporal/visual extension points, service ownership, and
current frontend/backend migration gaps are documented. Several architecture
decisions intentionally remain open and are listed explicitly rather than
guessed.

## Branch and HEAD

- Branch: `docs/task06a-unified-search-contract`
- Base branch: `test/frontend-ui-batch1`
- HEAD: `cb81795e45028312ab67713787de1665b72fa212`
- Base commit: `cb81795 test: add frontend UI batch 1 coverage`
- Worktree: `mvr-competition-starter-task06a-spec`
- No merge and no push were performed.

## Files changed

Documentation/specification only:

- `docs/contracts/UNIFIED_SEARCH_API.md`
- `docs/reviews/TASK06A_SPEC_REPORT.md`

No frontend production file, backend production file, test file, dependency
file, generated artifact, or dataset was changed.

## Contract and version decision

- Keep the single relative endpoint `POST /search`.
- Keep deployment prefixes, if any, in the configured API base URL rather than
  changing the frontend route contract.
- Define contract version `"1"` in the response envelope.
- Keep existing specialized routes as internal/debug/service endpoints; this
  task does not remove or redefine them.
- Use lowercase machine enums:
  - `task_mode`: `kis | trake | qa`
  - `retrieval_mode`: `semantic | lexical | object | hybrid`
- Do not expose provider/model/RRF names as required frontend fields.

The contract document is:

[UNIFIED_SEARCH_API.md](../contracts/UNIFIED_SEARCH_API.md)

## Request summary

Required normalized core:

- `task_mode`
- non-blank `query` for supported v1 textual search
- `retrieval_mode`
- `top_k`

`top_k` defaults to `100` during migration normalization, is constrained to
`1..100`, and limits the final response count. Internal provider pools may
over-fetch.

Optional typed extensions:

- `object_constraints` with logic, object labels, color labels, and typed
  `max_objects`;
- reserved `temporal.events[]` rather than hard-coded `scene1`/`scene2`;
- reserved `visual_query` reference using `image_url` or `asset_id`.

The legacy `sim_reorder` flag and `spatial_filter.max_objects_text` are not
canonical v1 fields. A future frontend migration adapter should drop or
translate them explicitly instead of making provider behavior depend on them.

Image-only search, non-null temporal execution, and non-null visual-query
execution are rejected or reserved in v1 with structured errors.

## Response summary

Success responses contain:

- `status: "ok"`;
- `contract_version: "1"`;
- optional `query_id`;
- `total_results` equal to the number of returned hits;
- rank-ordered `results`.

Every canonical hit requires:

- `rank`;
- `keyframe_id`;
- `video_id`;
- integer `frame_id`;
- numeric `timestamp` in seconds;
- finite `score`.

Optional metadata includes `shot_id`, `source`, media references, OCR/ASR text,
structured object counts, and debug-only provenance.

Identity semantics are explicit:

- `keyframe_id`: canonical retrieval/join/UI identity;
- `video_id + frame_id`: competition output identity;
- `shot_id + timestamp`: navigation/temporal metadata.

Formatted timestamp strings belong to frontend adapters. Raw semantic, BM25,
RRF, and hybrid scores are not cross-mode comparable; clients trust `rank`.

## Error and empty-result summary

Errors use:

```json
{
  "status": "error",
  "contract_version": "1",
  "error": {
    "code": "PROVIDER_UNAVAILABLE",
    "message": "Semantic retrieval is currently unavailable.",
    "details": null,
    "retryable": true
  }
}
```

Intended HTTP classes:

- `400`: semantically invalid request combination;
- `422`: schema, enum, blank-query, or range validation;
- `503`: provider or index unavailable;
- `500`: unexpected failure.

An empty successful search is HTTP `200` with `status: "ok"`,
`total_results: 0`, and `results: []`. It is never represented as an error.

## Feature maturity and ownership

`SUPPORTED_V1`:

- KIS textual search;
- semantic capability;
- lexical/BM25 capability;
- object capability with typed constraints;
- hybrid capability with backend-owned fusion.

`RESERVED`:

- TRAKE temporal event execution;
- visual query execution.

`DEFERRED`:

- QA request/answer contract.

`OUT_OF_SCOPE`:

- competition submission transport.

Provider selection, model-level RRF, query processing, score normalization,
candidate pooling, fusion, and canonical hit normalization are backend-owned.
The FastAPI route should validate, delegate to `UnifiedSearchService`, and map
structured failures; it should not implement retrieval itself.

## Frontend compatibility audit

| Current frontend field/behavior | Proposed API field | Compatible? | Adapter needed? | Later action |
|---|---|---:|---:|---|
| `task_mode` (`kis/trake/qa`) | `task_mode` | yes | no enum rename | Reject reserved/deferred modes in v1 |
| `query` | `query` | mostly | trim and validate | Keep user intent semantics |
| `retrieval_mode` | `retrieval_mode` | yes | no | Backend owns providers |
| optional `top_k`, current `100` | `top_k`, default `100` | yes | normalize omission | Enforce `1..100` |
| `sim_reorder` | none in canonical v1 | no | drop/ignore explicitly | Remove from future frontend request |
| `spatial_filter.enabled` | null/non-null `object_constraints` | partial | map disabled to null | Use typed object block |
| `spatial_filter.logic` | `object_constraints.logic` | yes | field relocation | Confirm taxonomy semantics |
| `selectedObjects` | `object_constraints.objects` | yes | field relocation | Validate labels backend-side |
| `selectedColors` | `object_constraints.colors` | partial | field relocation | Freeze color taxonomy later |
| `max_objects_text` | `object_constraints.max_objects` | no | parse typed integer | Reject invalid text; no guessing |
| `visual_query.image_url` | reserved `visual_query.image_url` | shape yes | v1 capability validation | Upload/reference contract later |
| image-only search | no v1 equivalent | no | reject or defer | Enable only with visual retrieval |
| `results` array | `results` array | yes | response envelope validation | Preserve canonical hits |
| `keyframe_id` | `keyframe_id` | yes | none | Required; never synthesize |
| numeric `frame_id` | integer `frame_id` | yes | validate | Backend must stop emitting ambiguous strings |
| formatted `timestamp` | numeric `timestamp` seconds | no | format for display | Keep raw numeric value in adapter |
| `timestamp_sec` | derived from `timestamp` | partial | derive/drop duplicate | Do not make it canonical |
| `score` | `score` | yes | none | Display only with rank semantics |
| `rank`/`source` | `rank`/`source` | partial | optional defaults | Rank is authoritative |
| `thumbnail_url` from `image_path` | `thumbnail_url` | partial | map media reference | Define media serving later |
| `objects: string[]` | structured object-count map | partial | map keys for display | Keep counts for object semantics |
| `ocr_text`/`asr_text` | same optional fields | yes | none | Backend populates when available |

The current tested frontend is therefore structurally ready for the future
endpoint, but it needs a later request/response adapter update for the
canonicalized fields above. This task intentionally does not make that update.

## Backend migration audit

Current `src/mvr/api/app.py` exposes specialized routes only:

- `/search/semantic`;
- `/search/kis`;
- `/search/text`;
- `/search/object`;
- `/search/hybrid`.

Current `src/mvr/api/schemas.py` has separate request models with route-specific
defaults and limits. `SearchHit` currently uses string `frame_id`, numeric
`timestamp`, `image_path`, object-count dictionaries, and no required
`keyframe_id` or `shot_id`. `KisHit` has optional `keyframe_id` and timestamp,
but is a separate response shape. Existing response envelopes use `query`,
`count`, and `results`, not the unified `status`, `contract_version`, and
`total_results` shape.

Later implementation work should add, without doing so in this task:

- `UnifiedSearchRequest`;
- `UnifiedSearchResponse`;
- `UnifiedSearchHit`;
- structured `SearchError`/error response models;
- `UnifiedSearchService`;
- capability service interfaces for semantic, lexical, object, and hybrid
  retrieval;
- canonical hit normalization and integer frame-ID validation;
- thin FastAPI `POST /search` delegation and exception mapping.

Existing service injection in `create_app()` is a useful testing seam. The
unified route should reuse the existing retrieval services rather than
duplicating Lab 6, Task05B, or semantic ranking logic. Specialized routes
remain available for current diagnostics and should not be called by the
frontend as a substitute for orchestration.

## Open questions

The specification deliberately leaves these unresolved:

1. Final hybrid normalization/fusion policy, candidate pool, and partial
   failure behavior.
2. Whether `max_objects` means a total count or a per-label count.
3. Canonical color taxonomy and matching behavior.
4. Safe visual upload/storage/authentication transport.
5. TRAKE event ordering, temporal windows, sequence constraints, and response
   scoring.
6. QA answer input/output and evaluation contract.
7. Submission endpoint and organizer transport.
8. Stable provider/fusion provenance visibility.
9. Translation and language-normalization ownership.
10. Whether a future full-corpus count should be named `total_available`.
11. Deployment prefix consistency (`/search` relative to a configured
    `/api/v1` base).
12. Mapping all source frame indices and organizer frame identifiers to the
    canonical integer `frame_id`.

## Confirmation and final scope

- Only the two required documentation/specification files were added.
- No `POST /search` implementation was added.
- No frontend or backend production source was changed.
- Task05C was not started.
- No merge or push was performed.
- Final verdict: **PASS_WITH_OPEN_QUESTIONS**.
