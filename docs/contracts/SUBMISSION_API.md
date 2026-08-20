# Submission and Competition Output Contract

## Status and evidence boundary

**Task06B specification only.**

This document defines the future submission boundary after retrieval and
frontend selection. It does not implement `/submit`, organizer transport,
persistence, frontend changes, or competition delivery.

The repository has verified source evidence for a textual KIS prediction
contract, but it does not contain an independently verified organizer upload
URL, authentication method, final CSV header rule, official JSON schema, or
transport protocol. Those details are deliberately marked
`OPEN_QUESTION`/`DEFERRED` below.

The strongest source-backed KIS facts are:

- the prediction pair is `(video_id: str, frame_id: int)`;
- `frame_id` is the original-video frame index, sourced from `frame_idx`;
- the maximum is 100 unique output pairs per query;
- exact duplicate output pairs are removed while preserving the first input;
- input order is retained and ranks are reassigned contiguously;
- local CSV/JSON exports are explicitly labeled verification exports, not
  official organizer files;
- no organizer transport adapter is implemented or assumed.

KIS maturity is intentionally split at the repository boundary:

```text
KIS validation + local format preview: SUPPORTED_V1
Official organizer formatter: OPEN_QUESTION
Organizer transport: DEFERRED
```

`SUPPORTED_V1` therefore does not mean that an official organizer submission
workflow is supported.

## 1. Ownership boundary

```text
Frontend submission tray
  -> selects canonical results and explicit user order
  -> sends canonical submission items

Submission API
  -> validates task mode and envelope
  -> cross-validates canonical identity
  -> applies duplicate/order/limit policy
  -> returns normalized items and a local format preview

Competition formatter
  -> converts normalized task items to an organizer-specific representation
  -> owns headers, delimiters, encoding, and task-specific row layout

Organizer transport
  -> sends a formatter output through a documented official mechanism
  -> owns credentials, URLs, authentication, retries, and delivery status

Persistence/audit
  -> optional future recording of normalized request/output metadata
```

The submission boundary does not own retrieval ranking, query processing,
semantic/lexical/object/hybrid search, offline indexing, media serving, or the
Task06A `/search` orchestration endpoint.

Formatting and transport are separate. A formatter must be testable without a
network, and a transport must not reconstruct or reinterpret canonical
identities.

## 2. Endpoint and version decision

### Conceptual endpoint

```text
POST /submit
Content-Type: application/json
Accept: application/json
```

This is a future UI-facing boundary only. In v1 it means:

> validate + normalize + format preview

It does **not** persist a submission or send anything to an organizer. The
response must clearly label its formatted output as non-official until an
organizer format is independently verified.

The endpoint uses contract version `"1"` in the request and response envelope.
The route remains relative `/submit`; any deployment prefix belongs in the
configured API base URL, consistent with Task06A.

A separate validate-only route is not required for this first design. A future
transport-enabled workflow may add an explicit mode or endpoint after the
organizer protocol is known. It must not silently change the current
format-preview meaning of `/submit`.

## 3. Canonical request envelope

```json
{
  "contract_version": "1",
  "task_mode": "kis",
  "query_id": "optional-query-id",
  "items": [
    {
      "keyframe_id": "L21_V001_S0001_K01",
      "video_id": "L21_V001",
      "frame_id": 261
    }
  ]
}
```

| Field | Type | Required | v1 meaning |
|---|---|---:|---|
| `contract_version` | `"1"` | yes | Submission contract version |
| `task_mode` | `"kis" \| "qa" \| "trake"` | yes | Machine task enum |
| `query_id` | non-blank string or null | no | Correlation to a search/query |
| `items` | ordered array | yes | User-selected canonical items |

Display labels such as `KIS`, `Q&A`, and `AVS / TRAKE` are not API enum
values. The request array order is meaningful and must not be silently sorted.

For the KIS v1 path, `items` must contain 1 through 100 input items before any
formatting or duplicate normalization. After identity validation and output
pair normalization, the result must still contain 1 through 100 unique
`(video_id, frame_id)` pairs. An empty request is rejected by the submission
API; the evaluator's ability to score an empty answer list as zero is not
evidence that an empty organizer submission is valid.

## 4. KIS item contract

### 4.1 Required identity

```json
{
  "keyframe_id": "L21_V001_S0001_K01",
  "video_id": "L21_V001",
  "frame_id": 261
}
```

All three fields are required at the submission API boundary:

- `keyframe_id`: canonical internal retrieval/join/audit identity;
- `video_id + frame_id`: competition-facing prediction identity;
- `frame_id`: original-video frame identity, not a keyframe ordinal.

The API accepts all three so it can detect stale or corrupted frontend state.
It must cross-validate them and reject a mismatch. It must not silently
overwrite `video_id` or `frame_id` from a conflicting request.

### 4.2 Frame identity rules

`frame_id` must be:

- a JSON integer, not a string or boolean;
- non-negative;
- the original-video zero-based frame index where the source mapping uses
  zero-based indexing;
- finite and within any verified source-video range when that range is
  available.

The repository explicitly distinguishes:

```text
frame_id / frame_idx
  original-video frame index used for KIS output

keyframe ordinal n
  organizer keyframe sequence number, lookup-only

keyframe filename/ordinal
  lookup reference, not the submitted frame ID

shot-relative frame
  not a KIS submission identity
```

For organizer support data, `map_keyframes/<video_id>.csv` provides
`n, pts_time, fps, frame_idx`; only `frame_idx` is used as the KIS `frame_id`.
The `n` value and filename are never substituted as a prediction frame.

### 4.3 Cross-validation sources

When available, the validation service resolves:

```text
keyframe_id -> expected video_id -> expected original frame_id
```

Supported source patterns in the repository are:

- this repository's `keyframe_mapping.csv`, keyed by canonical
  `keyframe_id` and carrying `frame_idx`;
- the AIC registry's composite `(video_id, keyframe_ref)` lookup backed by
  `map_keyframes/<video_id>.csv`.

If no authoritative mapping is available for a request item, the API must not
guess from timestamp, filename, keyframe ordinal, or shot position. It should
return a mapping-unavailable/unverified error rather than emit a potentially
corrupted competition pair. The exact handling of a source whose
`keyframe_id` format cannot be joined to the organizer reference is an open
question.

## 5. KIS ordering and duplicate policy

### 5.1 Ordering

KIS submission order is explicit user intent:

```text
item 1 -> submitted answer rank 1
item 2 -> submitted answer rank 2
...
```

The frontend drag/drop tray order is preserved. The API and formatter must not
sort by score, `frame_id`, `video_id`, timestamp, or retrieval rank unless an
official rule later requires it.

After normalization, ranks are reassigned contiguously from 1. This matches the
verified local KIS validator and evaluator behavior.

### 5.2 Duplicate levels and surfaced normalization

There are two identity levels:

1. **Canonical duplicate**: repeated `keyframe_id`.
2. **Output duplicate**: repeated `(video_id, frame_id)`.

The proposed API policy is:

- repeated `keyframe_id` with the same complete identity is a
  `DUPLICATE_ITEM` request error; the frontend store should prevent this before
  calling the API;
- repeated `keyframe_id` with conflicting `video_id` or `frame_id` is an
  `IDENTITY_MISMATCH` error;
- distinct `keyframe_id` values resolving to the same `(video_id, frame_id)`
  are collapsed at the submission normalization boundary, keeping the first
  input occurrence;
- the normalized output pair is emitted once and consumes one item of the
  100-pair budget;
- same `frame_id` in different videos is allowed because the output identity is
  the pair `(video_id, frame_id)`.

The last two rules preserve verified repository behavior in
`search_kis()` and `validate_kis_predictions()`. The stricter repeated
`keyframe_id` request error protects the canonical internal identity and is a
submission API rule, not an invented organizer requirement.

This collapse must not be silently hidden by a future `SubmissionService`.
The normalized response or debug metadata should conceptually expose the
counts, for example:

```json
{
  "input_items": 12,
  "accepted_items": 11,
  "collapsed_output_pairs": 1
}
```

These fields are a contract/documentation concept for observability; this task
does not implement the response or `SubmissionService`.

Conflicting identities are never repaired silently.

## 6. Maximum item behavior

| Layer | Current/source-backed value | Status |
|---|---:|---|
| Frontend `useSubmissionStore.maxItems` | 100 selected items | Implemented UI behavior |
| Frontend CSV utility | First 100 rows | Implemented convenience behavior |
| Proposed v1 API input limit | Reject more than 100 raw input items before formatting/deduplication | `SOURCE_BACKED_PROVISIONAL` |
| Proposed v1 final output limit | 1..100 unique `(video_id, frame_id)` pairs after normalization | `SOURCE_BACKED_PROVISIONAL` |
| AIC KIS source contract in repository docs/code | 100 unique `(video_id, frame_id)` pairs per query | `SOURCE_BACKED_PROVISIONAL` |
| Backend KIS search/schema | `1 <= top_k <= 100` | Implemented search behavior |
| Backend local KIS export validator | Rejects >100 unique pairs after dedup | Implemented validation |
| Organizer maximum independently verified from organizer transport docs | Not found in repository | `OPEN_QUESTION` |

The value `100` is `SOURCE_BACKED_PROVISIONAL`: it is supported by repository
documentation/code, but it is not independently verified organizer transport
behavior until the official rules/file schema are located.

The v1 sequence is deterministic:

1. Count raw `items` and reject more than 100 input items before formatting or
   duplicate normalization.
2. Validate identity and normalize distinct output pairs, retaining the first
   input occurrence.
3. Reject if the final unique output-pair count exceeds 100.

Therefore, 101 input items that collapse to 99 output pairs are still rejected.
Deduplication cannot bypass the API input limit.

## 7. Submission response

The format-preview response is:

```json
{
  "status": "ok",
  "contract_version": "1",
  "task_mode": "kis",
  "query_id": "q1",
  "input_items": 3,
  "accepted_items": 2,
  "collapsed_output_pairs": 1,
  "formatted_output": {
    "format": "kis_pair_rows_local_preview_v1",
    "official": false,
    "rows": [
      {
        "video_id": "L21_V001",
        "frame_id": 261
      },
      {
        "video_id": "L21_V002",
        "frame_id": 45
      }
    ]
  }
}
```

`input_items` counts the raw request items. `accepted_items` counts normalized
output rows. `collapsed_output_pairs` records how many input rows were removed
because they mapped to an already-emitted output pair. These fields are
conceptual response/debug metadata for the future service and make
normalization observable. The response contains no retrieval score, rank from
search, timestamp, shot, provider, or model metadata.

`formatted_output.official` is `false` because the repository does not verify
the organizer's final file/API format. A future official formatter may replace
the preview format without changing the canonical request or validation
boundary.

No `submission_id`, persistence status, or transport status is promised in v1;
there is no persistence or transport implementation.

## 8. Structured errors

Errors follow the Task06A envelope:

```json
{
  "status": "error",
  "contract_version": "1",
  "error": {
    "code": "IDENTITY_MISMATCH",
    "message": "keyframe_id does not match video_id/frame_id.",
    "details": {
      "item_index": 3
    },
    "retryable": false
  }
}
```

Recommended HTTP classes:

| HTTP status | Codes | Meaning |
|---:|---|---|
| `400` | `INVALID_TASK_MODE`, `EMPTY_SUBMISSION`, `TOO_MANY_ITEMS`, `DUPLICATE_ITEM`, `IDENTITY_MISMATCH`, `INVALID_ORDER`, `UNSUPPORTED_TASK` | Valid JSON but semantically unacceptable |
| `422` | `INVALID_FRAME_ID`, `INVALID_REQUEST` | Invalid field type, range, or envelope shape |
| `503` | `IDENTITY_UNVERIFIED`, `MAPPING_UNAVAILABLE`, `TRANSPORT_UNAVAILABLE` | Required external/source service is unavailable |
| `500` | `FORMAT_ERROR`, `INTERNAL_ERROR` | Unexpected formatter or server failure |

`TOO_MANY_ITEMS` must identify its limit scope in `error.details`:
`"input"` for more than 100 raw request items (rejected before
formatting/deduplication), or `"output"` for more than 100 final unique
`(video_id, frame_id)` pairs after normalization.

`TRANSPORT_UNAVAILABLE` is reserved for a future transport-enabled path and
must not be emitted by the current format-preview-only v1 endpoint.

The API must not silently drop invalid rows, synthesize missing identities,
sort rows, or fall back to frontend CSV output after a validation error.

## 9. Competition formatter boundary

The proposed flow is:

```text
SubmissionRequest
    -> SubmissionService
    -> identity/order/duplicate/limit validation
    -> normalized KIS pair rows
    -> CompetitionFormatter
    -> local preview or future official representation
    -> optional OrganizerTransport
```

The formatter owns:

- task-specific row structure;
- official headers, if required;
- delimiter/quoting;
- encoding and newline policy;
- output filename or payload shape;
- format version and provenance.

The formatter does not own:

- canonical identity lookup;
- retrieval ranking;
- organizer authentication;
- network retries;
- transport delivery.

### Current frontend exports

The current frontend has two convenience paths:

1. `SubmissionTray.tsx` manually emits `video_id,frame_id` lines with no header
   and no robust CSV escaping.
2. `csvExporter.ts` emits current KIS/QA/TRAKE-like strings, strips a video
   extension, slices to 100 rows, and only escapes QA answer quotes.

These are useful local/debug exports, not the canonical organizer formatter.
The backend formatter should become the single source of truth once official
rules are available. No frontend CSV redesign is part of this task.

### Existing backend local export

`src/mvr/submission/kis_export.py` writes a clearly labeled
`local_verification_export` CSV/JSON round-trip. Its CSV header is:

```text
format,query_id,rank,video_id,frame_id
```

That header and label are repository-local verification conventions. They must
not be presented as the official organizer file format.

## 10. QA and TRAKE status

### QA

`qa` is **DEFERRED**. Current source material provides only a frontend
`answer?: string` convenience field and a simple local CSV branch. It does not
establish official answer type, normalization, empty-answer policy, length
limits, multiple-answer semantics, or organizer output format.

No public QA submission request, response, or example is frozen here. Future
QA work must first identify official/source rules and then define answer
validation separately from KIS frame validation.

### TRAKE

`trake` is **DEFERRED / OPEN_QUESTION**. Task06A reserves temporal events and
the repository has no verified temporal retrieval or organizer submission
format. The current frontend's flat `frame_ids` CSV branch is insufficient
evidence for event count, same-video requirements, increasing frame order,
missing events, duplicate frames, or sequence scoring.

No TRAKE submission request, response, or example is frozen here. A future
internal representation may use ordered N-event records, but it must not be
turned into an organizer-facing format without source confirmation.

## 11. Persistence and audit trail

Persistence is **DEFERRED** for v1. The format-preview endpoint need not write a
database or artifact. A later audit record may include:

- request/correlation ID;
- task mode and query ID;
- normalized canonical items;
- formatted output hash;
- created time;
- transport state.

The exact storage location, retention, and idempotency key are open questions.
No model/provider/retrieval internals are required in an audit record.

## 12. Organizer transport boundary

Organizer transport is **DEFERRED**. The repository explicitly documents that
no official organizer transport is known or implemented. The existing
`SubmissionAdapter` protocol is only a thin extension point.

Future transport may be an official HTTP API, a file download/upload workflow,
an official client, or another documented mechanism. This specification does
not guess:

- organizer URL;
- credentials or authentication;
- request headers;
- retry/idempotency behavior;
- rate limits;
- response/status schema.

Only a documented `OrganizerTransport` implementation may cross that boundary.

## 13. Feature maturity

| Feature | Status | Meaning |
|---|---|---|
| KIS validation + local format preview | `SUPPORTED_V1` | Triple-field validation, source-backed frame rules, and explicitly non-official output |
| KIS official organizer formatter | `OPEN_QUESTION` | Exact format not found |
| KIS organizer transport | `DEFERRED` | No verified protocol/credentials |
| QA validate/format | `DEFERRED` | Source material insufficient |
| TRAKE validate/format | `DEFERRED` | Temporal/output rules insufficient |
| Frontend CSV convenience export | `SUPPORTED_V1` | Debug/convenience only |
| Submission persistence/audit | `DEFERRED` | Not required for format preview |
| `/submit` implementation | `OUT_OF_SCOPE` | Future implementation task |

## 14. Required examples

### 14.1 Valid KIS submission request

```http
POST /submit
Content-Type: application/json
```

```json
{
  "contract_version": "1",
  "task_mode": "kis",
  "query_id": "q1",
  "items": [
    {
      "keyframe_id": "L21_V001_S0001_K01",
      "video_id": "L21_V001",
      "frame_id": 261
    },
    {
      "keyframe_id": "L21_V002_S0001_K01",
      "video_id": "L21_V002",
      "frame_id": 45
    }
  ]
}
```

### 14.2 Valid KIS success response

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "status": "ok",
  "contract_version": "1",
  "task_mode": "kis",
  "query_id": "q1",
  "input_items": 2,
  "accepted_items": 2,
  "collapsed_output_pairs": 0,
  "formatted_output": {
    "format": "kis_pair_rows_local_preview_v1",
    "official": false,
    "rows": [
      {"video_id": "L21_V001", "frame_id": 261},
      {"video_id": "L21_V002", "frame_id": 45}
    ]
  }
}
```

### 14.3 Empty submission error

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json
```

```json
{
  "status": "error",
  "contract_version": "1",
  "error": {
    "code": "EMPTY_SUBMISSION",
    "message": "At least one KIS item is required.",
    "details": null,
    "retryable": false
  }
}
```

### 14.4 Too-many-items error

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json
```

```json
{
  "status": "error",
  "contract_version": "1",
  "error": {
    "code": "TOO_MANY_ITEMS",
    "message": "At most 100 KIS input items are allowed.",
    "details": {
      "limit_scope": "input",
      "input_items": 101,
      "max_input_items": 100
    },
    "retryable": false
  }
}
```

### 14.5 Identity mismatch error

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json
```

```json
{
  "status": "error",
  "contract_version": "1",
  "error": {
    "code": "IDENTITY_MISMATCH",
    "message": "keyframe_id does not resolve to the supplied video_id/frame_id.",
    "details": {"item_index": 0},
    "retryable": false
  }
}
```

### 14.6 Duplicate-item error

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json
```

```json
{
  "status": "error",
  "contract_version": "1",
  "error": {
    "code": "DUPLICATE_ITEM",
    "message": "The same keyframe_id was submitted more than once.",
    "details": {"item_index": 2, "keyframe_id": "L21_V001_S0001_K01"},
    "retryable": false
  }
}
```

QA and TRAKE examples are intentionally omitted because current official/source
material does not support a stable contract for them.

## 15. Open questions

1. What official organizer document confirms the maximum answer count and does
   it exactly match the repository's source-backed 100-pair rule?
2. What exact official KIS file/API format, header, delimiter, encoding, and
   newline policy must `CompetitionFormatter` emit?
3. Does organizer scoring use submitted order exactly as the repository
   evaluator does, or are there additional tie/duplicate rules?
4. Should repeated identical `keyframe_id` items be rejected (this proposal) or
   normalized idempotently by a future API?
5. How should canonical internal keyframe IDs join to organizer keyframe
   references across every corpus and mapping source?
6. Is a separate validate/preview endpoint or explicit request mode needed once
   transport exists?
7. What QA answer type, normalization, limits, and output format are official?
8. What TRAKE event count, same-video rule, increasing-frame rule, and output
   format are official?
9. Is submission persistence required, and what audit retention/idempotency
   policy should it use?
10. What organizer transport URL, authentication, retry, rate-limit, and
    response semantics are official?

## 16. Explicit non-goals

This document does not implement:

- `POST /submit`;
- organizer API calls, credentials, or authentication;
- submission persistence;
- Task06A `/search`;
- retrieval or query processing;
- multi-embedding or Task05C work;
- TRAKE retrieval;
- QA answering;
- frontend changes;
- CSV redesign beyond this specification.
