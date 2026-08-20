# Unified Search API Contract

## Status

**Task06A specification only.**

This document freezes the intended UI-facing search contract. It does not
implement the route, Pydantic models, retrieval orchestration, provider
adapters, or frontend changes.

The contract is intentionally provider-independent:

> Frontend selects retrieval capability; backend selects and orchestrates
> model/provider implementation.

The frontend must not be required to know whether semantic retrieval uses
BEiT-3, MetaCLIP-2, Perception Encoder, another encoder, or model-level RRF.

## 1. Endpoint and versioning

### Canonical endpoint

```text
POST /search
Content-Type: application/json
Accept: application/json
```

`/search` is the single UI-facing orchestration endpoint. Existing specialized
routes such as `/search/semantic`, `/search/kis`, `/search/text`,
`/search/object`, and `/search/hybrid` are not removed or redefined by this
specification. They may remain internal, debug, benchmark, or service-level
routes.

The current frontend's API client uses a configurable base URL and posts to
relative `/search`. A deployment may mount the application below a prefix such
as `/api/v1`; that deployment prefix belongs in the configured base URL and is
not duplicated in the contract path.

### Contract version

This document defines **contract version `1`**. Responses include
`contract_version: "1"` so clients can record the schema version. URL
versioning is not introduced for this first unified endpoint because the
frontend already targets `/search` and the repository's current FastAPI routes
are unversioned.

If a breaking change is needed later, it should be introduced as an explicit
contract version rather than silently changing the meaning of an existing
field.

## 2. Ownership boundaries

```text
Frontend
  -> collects user intent and selects task/capability
  -> sends one normalized request
  -> renders rank-ordered canonical hits

POST /search route
  -> validates the request
  -> calls UnifiedSearchService
  -> maps domain failures to the structured error envelope
  -> returns the normalized response

UnifiedSearchService
  -> applies query-processing policy
  -> routes by retrieval_mode
  -> invokes the selected backend retrieval services
  -> fuses supported branches
  -> normalizes canonical hits

Retrieval services
  -> SemanticSearchService
  -> LexicalSearchService
  -> ObjectSearchService
  -> HybridSearchService

Offline/indexing systems
  -> feature extraction, model loading, index construction, and persistence
```

Task06A does not own:

- offline video/keyframe feature extraction;
- OCR, ASR, object, or embedding generation;
- FAISS/Elasticsearch index construction or rebuilds;
- model training or provider selection policy;
- frontend rendering or submission UI;
- competition submission transport;
- TRAKE sequence execution;
- query translation implementation.

FastAPI handlers should remain thin. They should not load models, build
indexes, duplicate fusion logic, or call five specialized routes to simulate
orchestration.

## 3. Request contract

### 3.1 Canonical v1 shape

```json
{
  "task_mode": "kis",
  "query": "a man riding a motorcycle on a road",
  "retrieval_mode": "hybrid",
  "top_k": 100,
  "object_constraints": null,
  "temporal": null,
  "visual_query": null
}
```

The normalized request has four core fields:

| Field | Type | Required in normalized request | v1 meaning |
|---|---|---:|---|
| `task_mode` | `"kis" \| "trake" \| "qa"` | yes | Competition task family |
| `query` | string | yes | User intent text; trimmed before execution |
| `retrieval_mode` | `"semantic" \| "lexical" \| "object" \| "hybrid"` | yes | Retrieval capability, not provider |
| `top_k` | integer | yes | Requested maximum final result count |

For migration convenience, an omitted `top_k` may be normalized to the v1
default of `100`. A canonical client should send it explicitly. The server
must validate the normalized value before calling retrieval.

Unknown fields should be rejected or explicitly handled by a compatibility
layer; they must not silently change retrieval behavior.

### 3.2 Task mode

Machine values are lowercase and stable:

- `kis`: textual keyframe search and competition-style keyframe results;
- `trake`: reserved temporal/event search;
- `qa`: deferred question-answering task.

Display labels such as `KIS`, `AVS / TRAKE`, and `Q&A` are not API enum values.

For supported v1 textual search, `query` must contain non-whitespace text.
Image-only search is not supported by v1; an empty query with a non-null
`visual_query` returns a validation error until visual retrieval is enabled.

The backend may later apply language detection, normalization, translation,
rewrite, query expansion, or event decomposition. Those are backend policies;
the frontend sends user intent and does not send model-specific prompts.

### 3.3 Retrieval mode

`retrieval_mode` is capability-level vocabulary:

- `semantic`: semantic retrieval service;
- `lexical`: lexical/OCR/ASR retrieval service;
- `object`: object constraints/filter retrieval service;
- `hybrid`: backend fusion of supported branches.

Provider/model names are not required request fields. No request should contain
`model_engine`, `ClipVideo`, `ClipLAION`, `Aladin`, `BEiT-3`, `MetaCLIP-2`,
`Perception Encoder`, or `RRF` settings as a frontend requirement.

### 3.4 `top_k`

Validation rules:

- default: `100` when omitted during migration normalization;
- minimum: `1`;
- maximum: `100`;
- non-integer, zero, negative, or greater-than-100 values: validation error;
- final response count is always `<= top_k`;
- internal candidate pools may be larger than `top_k` for deduplication,
  unmapped-candidate removal, or fusion.

`top_k` controls the final API response, not the number of internal provider
calls or candidates.

### 3.5 Object constraints

The canonical typed block is:

```json
{
  "logic": "AND",
  "objects": ["person", "motorcycle"],
  "colors": ["red"],
  "max_objects": 3
}
```

The block is nullable. Its fields are:

| Field | Type | Meaning |
|---|---|---|
| `logic` | `"AND" \| "OR"` | Combination rule for listed constraints |
| `objects` | array of non-blank strings | Backend object-taxonomy labels |
| `colors` | array of non-blank strings | Backend color-taxonomy labels |
| `max_objects` | integer or null, `>= 0` | Optional upper bound on matching object instances |

Arrays should be deduplicated after trimming. The backend owns the valid object
and color taxonomy; the frontend must not invent provider-specific labels.

`retrieval_mode: "object"` requires a non-empty object constraint block. For
`semantic`, `lexical`, and `hybrid`, the block is optional. A disabled legacy
spatial panel maps to `object_constraints: null`.

The current `max_objects_text` UI field is not part of the canonical contract.
A migration adapter may parse a blank string as null and a non-negative decimal
string as `max_objects`. A non-numeric non-blank value must be rejected rather
than guessed.

Complex scene graphs, relative geometry, bounding-box constraints, and
per-event object relationships are outside v1.

### 3.6 Temporal extension point

The reserved shape is:

```json
{
  "events": [
    {
      "event_id": "e1",
      "query": "an athlete approaches the bar"
    },
    {
      "event_id": "e2",
      "query": "the athlete jumps over the bar"
    }
  ]
}
```

`events` order is significant. `event_id` values must be unique and event
queries must be non-blank. This deliberately avoids hard-coded `scene1`,
`scene2`, or `scene2Query` fields.

Temporal execution is **RESERVED**, not supported by v1. A non-null
`temporal` block or `task_mode: "trake"` must return a structured validation
error (`TEMPORAL_UNSUPPORTED`) until a separate temporal contract is approved.
This task does not define temporal ranking, same-video constraints, time
windows, or sequence scoring.

### 3.7 Visual query extension point

The future reference block may contain exactly one of:

```json
{ "image_url": "https://example.invalid/query.jpg" }
```

or:

```json
{ "asset_id": "uploaded-asset-123" }
```

The reference representation, upload transport, storage lifecycle, access
control, and retrieval behavior are separate concerns. A URL in this block is
a reference, not an instruction for the search API to upload or fetch arbitrary
content.

Visual query execution is **RESERVED** in v1. A non-null `visual_query` returns
`VISUAL_QUERY_UNSUPPORTED` until an upload/reference contract and backend
visual retrieval path are approved. Image-only queries therefore remain
unsupported in v1.

## 4. Retrieval capability semantics

### Semantic

The backend calls `SemanticSearchService`. It may use one or multiple
providers, model-level RRF, provider fallback policy, or a future encoder
without changing the request contract. Raw semantic scores are mode-local.

### Lexical

The backend calls the Task05B lexical/BM25 service. Field analysis, query
construction, and Elasticsearch details remain backend-owned. The unified
contract does not expose analyzer, field-boost, or index settings.

### Object

The backend calls `ObjectSearchService` with typed constraints. Object counts
and taxonomy interpretation remain backend-owned. The response may expose
structured object counts as optional metadata.

### Hybrid

The backend calls `HybridSearchService`, which may combine semantic, lexical,
object, and later visual branches that are available for the request.

Raw score addition across incompatible spaces is forbidden. Fusion must use an
explicit backend-defined normalization/fusion policy. In v1, the returned
`score` remains an opaque ordering signal and `rank` is authoritative.

The v1 contract does not promise partial results when one hybrid branch is
unavailable. The default behavior is a structured `503` provider/index error;
partial-degraded hybrid behavior is an open question.

## 5. Response contract

### 5.1 Success envelope

```json
{
  "status": "ok",
  "contract_version": "1",
  "query_id": "optional-server-generated-id",
  "total_results": 2,
  "results": [
    {
      "rank": 1,
      "keyframe_id": "L21_V010_S0007_K01",
      "video_id": "L21_V010",
      "frame_id": 578,
      "shot_id": "L21_V010_S0007",
      "timestamp": 23.12,
      "score": 0.81,
      "source": "hybrid",
      "thumbnail_url": "/keyframes/L21_V010/L21_V010_S0007_K01.jpg",
      "video_url": null,
      "ocr_text": null,
      "asr_text": null,
      "objects": {
        "person": 1,
        "motorcycle": 1
      },
      "provenance": null
    }
  ]
}
```

| Field | Type | Required | Semantics |
|---|---|---:|---|
| `status` | `"ok"` | yes | Successful request |
| `contract_version` | string | yes | `"1"` |
| `query_id` | string or null | no | Optional server correlation ID |
| `total_results` | integer `>= 0` | yes | Number of hits in `results` |
| `results` | array | yes | Rank-ordered canonical hits |

`total_results` means the number of results returned in this response, not a
promise of a full corpus count. A future `total_available` field would require
an explicit contract decision.

### 5.2 Canonical result hit

| Field | Type | Required | Semantics |
|---|---|---:|---|
| `rank` | integer `>= 1` | yes | 1-based position in this response |
| `keyframe_id` | non-blank string | yes | Canonical retrieval/join identity |
| `video_id` | non-blank string | yes | Competition video identity |
| `frame_id` | integer `>= 0` | yes | Canonical competition frame identity |
| `shot_id` | string or null | no | Shot/navigation identity |
| `timestamp` | finite number `>= 0` | yes | Seconds from video start |
| `score` | finite number | yes | Mode-local relevance score |
| `source` | string or null | no | Capability/fusion source label |
| `thumbnail_url` | string or null | no | UI thumbnail reference |
| `video_url` | string or null | no | UI preview reference |
| `ocr_text` | string or null | no | OCR text metadata |
| `asr_text` | string or null | no | ASR text metadata |
| `objects` | object map or null | no | Object label to count metadata |
| `provenance` | object or null | no | Optional debug/provider metadata |

Identity rules:

1. `keyframe_id` is the canonical internal identity for retrieval joins,
   selection, React keys, deduplication, preview lookup, and drag ordering.
2. `video_id + frame_id` is the competition output identity.
3. `shot_id + timestamp` is navigation/temporal metadata and does not replace
   `keyframe_id`.
4. The API and UI must never invent sequential IDs when `keyframe_id` is
   missing.

`frame_id` is deliberately an integer in the unified contract. Existing
backend `frame_id` strings must be parsed/validated during migration; a
non-numeric or ambiguous source identity is an error, not a generated
fallback. `timestamp` is deliberately numeric seconds. Formatted strings such
as `00:00:23` belong to frontend display adapters.

The response must contain ranks `1..N` in ascending order. The API guarantees
that `N <= top_k`. For equal scores, implementations should use
`keyframe_id` ascending as the deterministic tie-breaker; rank ordering remains
the source of truth for clients.

### 5.3 Score and provenance semantics

`score` is not cross-mode comparable. A semantic cosine score, BM25 score, RRF
score, and hybrid score may have different scales and meanings. Clients must
trust `rank` for ordering and must not compare or threshold scores across
retrieval modes.

`provenance` is optional and debug-oriented. If emitted, it may contain fields
such as selected branch, provider identifier, model identifier, index
identifier, or fusion method. These fields are not stable frontend inputs,
must not be required for rendering, and may be omitted or redacted.

## 6. Errors

### 6.1 Error envelope

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

`details` is nullable. For validation errors it may contain a list such as:

```json
[
  {
    "field": "top_k",
    "issue": "must be between 1 and 100"
  }
]
```

Stable error codes are preferred over client parsing of human messages.

### 6.2 HTTP mapping

| HTTP status | Error class | Examples |
|---:|---|---|
| `400` | Semantically invalid request | Object mode without constraints; reserved feature combination |
| `422` | Schema/field validation | Missing field, invalid enum, blank query, invalid range |
| `503` | Dependency unavailable | `PROVIDER_UNAVAILABLE`, `INDEX_UNAVAILABLE` |
| `500` | Unexpected server failure | `INTERNAL_ERROR` |

Recommended codes:

- `INVALID_REQUEST`
- `QUERY_REQUIRED`
- `TOP_K_INVALID`
- `UNSUPPORTED_TASK_MODE`
- `UNSUPPORTED_RETRIEVAL_MODE`
- `OBJECT_CONSTRAINTS_REQUIRED`
- `TEMPORAL_UNSUPPORTED`
- `VISUAL_QUERY_UNSUPPORTED`
- `PROVIDER_UNAVAILABLE`
- `INDEX_UNAVAILABLE`
- `INTERNAL_ERROR`

Errors must not silently fall back to fixtures, another retrieval capability,
or a different provider unless that fallback is an explicit backend policy and
its result remains contract-valid. A hybrid dependency failure returns `503` by
default; partial results require a future contract decision.

### 6.3 Empty successful results

No-hit search is not an exception:

- HTTP `200`;
- `status: "ok"`;
- `total_results: 0`;
- `results: []`.

## 7. Feature maturity

| Capability/task | v1 status | Contract behavior |
|---|---|---|
| KIS textual search | `SUPPORTED_V1` | Non-blank text query with supported retrieval mode |
| Semantic capability | `SUPPORTED_V1` | Backend-owned provider orchestration |
| Lexical capability | `SUPPORTED_V1` | Task05B/BM25 service |
| Object capability | `SUPPORTED_V1` | Typed object constraints required |
| Hybrid capability | `SUPPORTED_V1` | Backend fusion; raw scores remain opaque |
| TRAKE temporal search | `RESERVED` | `TEMPORAL_UNSUPPORTED` until separately specified |
| QA task | `DEFERRED` | Answer request/response contract not frozen |
| Visual query execution | `RESERVED` | Reference shape reserved; execution rejected in v1 |
| Submission transport | `OUT_OF_SCOPE` | Separate competition submission contract |
| Provider/model/RRF fields | `BACKEND_INTERNAL` | Never required frontend fields |

## 8. Mapping to current specialized backend services

The future orchestration maps conceptually as follows:

```text
POST /search
  ├── semantic -> SemanticSearchService
  ├── lexical  -> Task05B LexicalSearchService
  ├── object   -> ObjectSearchService
  └── hybrid   -> HybridSearchService / fusion policy
```

The frontend must not call specialized backend routes directly to emulate the
unified contract. Specialized routes may remain useful for diagnostics,
benchmarks, and internal service testing.

## 9. Required examples

### 9.1 Semantic KIS request

```http
POST /search
Content-Type: application/json
```

```json
{
  "task_mode": "kis",
  "query": "a person riding a bicycle",
  "retrieval_mode": "semantic",
  "top_k": 100,
  "object_constraints": null,
  "temporal": null,
  "visual_query": null
}
```

### 9.2 Lexical KIS request

```json
{
  "task_mode": "kis",
  "query": "TIN CHINH",
  "retrieval_mode": "lexical",
  "top_k": 20,
  "object_constraints": null,
  "temporal": null,
  "visual_query": null
}
```

### 9.3 Hybrid request with object constraints

```json
{
  "task_mode": "kis",
  "query": "a red motorcycle on a street",
  "retrieval_mode": "hybrid",
  "top_k": 100,
  "object_constraints": {
    "logic": "AND",
    "objects": ["motorcycle"],
    "colors": ["red"],
    "max_objects": 3
  },
  "temporal": null,
  "visual_query": null
}
```

### 9.4 Successful response

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "status": "ok",
  "contract_version": "1",
  "query_id": "q-123",
  "total_results": 1,
  "results": [
    {
      "rank": 1,
      "keyframe_id": "L21_V010_S0007_K01",
      "video_id": "L21_V010",
      "frame_id": 578,
      "shot_id": "L21_V010_S0007",
      "timestamp": 23.12,
      "score": 0.81,
      "source": "semantic",
      "thumbnail_url": "/keyframes/L21_V010/L21_V010_S0007_K01.jpg",
      "video_url": null,
      "ocr_text": null,
      "asr_text": null,
      "objects": null,
      "provenance": null
    }
  ]
}
```

### 9.5 Empty successful response

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "status": "ok",
  "contract_version": "1",
  "query_id": "q-124",
  "total_results": 0,
  "results": []
}
```

### 9.6 Validation error

```http
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/json
```

```json
{
  "status": "error",
  "contract_version": "1",
  "error": {
    "code": "TOP_K_INVALID",
    "message": "top_k must be between 1 and 100.",
    "details": [
      {
        "field": "top_k",
        "issue": "must be between 1 and 100"
      }
    ],
    "retryable": false
  }
}
```

### 9.7 Provider unavailable error

```http
HTTP/1.1 503 Service Unavailable
Content-Type: application/json
```

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

## 10. Open questions

These questions are intentionally not guessed by this specification:

1. What exact hybrid fusion policy, normalization, and candidate pool should
   v1 use, and what is the policy when a branch is unavailable?
2. Are object counts interpreted as a total across selected labels or as a
   per-label bound for `max_objects`?
3. What canonical color taxonomy and matching semantics should the object
   service expose?
4. What upload/storage/authentication transport should produce a safe
   `asset_id` for visual queries?
5. What temporal/TRAKE request, sequence constraints, response grouping, and
   scoring contract should be approved?
6. What QA answer input/output schema and evaluation semantics should be used?
7. What submission endpoint and organizer transport should be standardized?
8. Which provider/fusion provenance fields may be exposed for debugging without
   becoming a stable UI dependency?
9. Where should translation and language normalization occur, and how should
   language failures be surfaced?
10. Should the API later expose `total_available` separately from returned
    `total_results`?
11. Is the deployment prefix consistently `/api/v1` while the route remains
    relative `/search`?
12. How should source `frame_idx`, organizer frame IDs, and canonical
    `frame_id` be reconciled for every corpus?

## 11. Explicit non-goals

This document does not implement:

- backend `POST /search`;
- Pydantic request/response classes;
- retrieval orchestration or fusion;
- Task05C;
- semantic-provider adapters;
- BEiT-3, MetaCLIP-2, or Perception Encoder integration;
- RRF implementation;
- query translation;
- object reranking;
- TRAKE retrieval;
- frontend code;
- submission backend.
