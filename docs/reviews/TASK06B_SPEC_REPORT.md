# Task06B Submission Contract Specification Report

## Final verdict

**PASS_WITH_OPEN_QUESTIONS**

The submission boundary is specified without changing production code. The
contract defines a format-preview-only `/submit` v1 boundary, canonical KIS
identity validation, original-frame semantics, ordering, duplicate handling,
the source-backed provisional 100-item/100-pair limits, structured errors, formatter/transport
separation, frontend/backend compatibility findings, and explicit deferred
QA/TRAKE/organizer transport decisions.

The organizer's final upload protocol and exact file format were not found in
the repository. They remain open rather than being invented.

## Branch and HEAD

- Branch: `docs/task06b-submission-contract`
- Base branch: `docs/task06a-unified-search-contract`
- HEAD: `e61cb65021832ce54fc3431b95bb676e6b1d0426`
- Base commit: `e61cb65 docs: define unified search API contract`
- Worktree: `mvr-competition-starter-task06b-spec`
- No merge and no push were performed.

## Files changed

Documentation/specification only:

- `docs/contracts/SUBMISSION_API.md`
- `docs/reviews/TASK06B_SPEC_REPORT.md`

No frontend production file, backend production file, test file, dependency
file, generated artifact, dataset, or organizer transport code was changed.

## Source documents and modules inspected

Repository competition/source documentation:

- `docs/AIC_KIS_FINAL_REPORT.md`
- `docs/AIC_KIS_IMPLEMENTATION_NOTES.md`
- `docs/CODEX_PROGRESS.md`
- `docs/contracts/UNIFIED_SEARCH_API.md`
- `docs/reviews/TASK06A_SPEC_REPORT.md`
- `docs/reviews/FRONTEND_UI_REVIEW.md`
- `docs/reviews/FRONTEND_UI_HARDENING_REPORT.md`
- `docs/reviews/FRONTEND_UI_TEST_BATCH1_REPORT.md`

Submission, identity, and API source:

- `src/mvr/submission/kis_export.py`
- `src/mvr/retrieval/kis.py`
- `src/mvr/ingestion/aic_dataset.py`
- `src/mvr/api/app.py`
- `src/mvr/api/schemas.py`
- `frontend/src/stores/useSubmissionStore.ts`
- `frontend/src/components/Submission/SubmissionTray.tsx`
- `frontend/src/components/Submission/SortableItem.tsx`
- `frontend/src/utils/csvExporter.ts`
- `frontend/src/types/competition.ts`

Verification/source tests inspected:

- `tests/unit/submission/test_kis_export.py`
- `tests/unit/retrieval/test_kis.py`
- `tests/unit/evaluation/test_aic_kis.py`
- `tests/unit/ingestion/test_aic_dataset.py`
- `tests/integration/test_aic_kis_e2e.py`

No repository file provided a verified organizer URL, credential scheme,
official upload protocol, official CSV header, or official JSON submission
schema.

## Contract/version decision

- Future UI-facing endpoint: `POST /submit`.
- Contract version: `"1"`.
- v1 behavior: validate + normalize + format preview only.
- No persistence and no organizer transport in v1.
- `formatted_output.official` is explicitly `false` until organizer format
  evidence is available.
- Formatter and organizer transport are separate interfaces.
- Existing local verification CSV/JSON outputs remain local/debug formats and
  are not promoted to official organizer output.

## KIS contract status

The KIS maturity split is:

```text
KIS validation + local format preview: SUPPORTED_V1
Official organizer formatter: OPEN_QUESTION
Organizer transport: DEFERRED
```

`SUPPORTED_V1` applies only to validation and local preview; it does not label
official organizer submission as supported.

The KIS request requires:

- `contract_version`;
- `task_mode: "kis"`;
- optional `query_id`;
- ordered `items`;
- every item containing `keyframe_id`, `video_id`, and integer `frame_id`.

Validation:

- `frame_id` is a non-negative integer and must be the original-video
  `frame_idx`;
- organizer `n`, filename ordinal, and shot-relative frame are never used as
  the output frame;
- all three identity fields are cross-validated when an authoritative mapping
  is available;
- conflicting identity is rejected, never repaired;
- input order is preserved and ranks are reassigned contiguously;
- repeated canonical IDs are rejected by the proposed API;
- distinct keyframes resolving to the same output pair are collapsed,
  first-in-order, at the submission normalization boundary;
- a future `SubmissionService` must surface this normalization in response or
  debug metadata rather than hide it, conceptually including
  `input_items`, `accepted_items`, and `collapsed_output_pairs`;
- more than 100 raw input items is rejected before formatting/deduplication;
- after normalization, final unique `(video_id, frame_id)` pairs must also be
  at most 100.

The value `100` is `SOURCE_BACKED_PROVISIONAL`: repository source code, tests,
and schemas support it, but official organizer transport rules were not found.
The deterministic v1 policy rejects 101 input items even when they would
collapse to 99 output pairs; deduplication cannot bypass the API input limit.

The three limit layers are kept distinct:

- frontend maximum: 100 selected tray items;
- API input item limit: reject more than 100 raw request items before
  formatting/deduplication;
- final unique output-pair limit: reject more than 100 distinct
  `(video_id, frame_id)` pairs after validation/normalization.

## QA status

**DEFERRED.**

The repository only has a frontend convenience `answer?: string` field and a
local CSV branch. No source material establishes official answer type,
normalization, empty-answer policy, length limits, multiple-answer semantics,
or organizer output. No QA contract or example was invented.

## TRAKE status

**DEFERRED / OPEN_QUESTION.**

Task06A reserves temporal execution, and the current frontend flat frame list
does not establish event count, same-video requirements, increasing frame
constraints, missing-event policy, or organizer formatting. No TRAKE contract
or example was invented.

## Identity and frame validation summary

The repository evidence establishes:

```text
keyframe_id
  canonical internal retrieval/join/audit identity

video_id + frame_id
  competition-facing output identity

frame_id = original-video frame_idx
  not organizer keyframe ordinal n, filename ordinal, or shot-relative frame
```

The proposed API accepts all three identity fields and cross-validates them.
The current local export validator only receives `video_id`/`frame_id`, so the
future SubmissionService must add the canonical mapping check rather than
silently treating the local exporter as the complete API.

## Error contract summary

The response follows Task06A's structured envelope:

```json
{
  "status": "error",
  "contract_version": "1",
  "error": {
    "code": "IDENTITY_MISMATCH",
    "message": "keyframe_id does not match video_id/frame_id.",
    "details": {"item_index": 0},
    "retryable": false
  }
}
```

Proposed classes:

- `400`: empty submission, input/output limit violations, duplicate canonical
  item, identity mismatch, unsupported task, invalid order;
- `422`: malformed envelope, invalid frame type/range;
- `503`: identity mapping unavailable; future transport unavailable;
- `500`: formatter/internal failure.

Invalid rows are never silently dropped or guessed. Output-pair
deduplication is the one documented normalization step, and its count must be
observable through conceptual normalization metadata.

## Frontend compatibility findings

| Current behavior | Proposed contract | Finding |
|---|---|---|
| Store uses `keyframe_id` for dedup/removal/reorder | Canonical item identity | Compatible |
| Store has `video_id`, `frame_id`, timestamp, score | KIS item requires identity triple but emits only pair | Adapter must retain `keyframe_id` for audit |
| Store cap `maxItems = 100` | 100 selected frontend items | Compatible, but API remains authoritative |
| Store prevents duplicate `keyframe_id` | Proposed `DUPLICATE_ITEM` | Compatible safety behavior |
| Tray preserves drag/drop order | Ordered KIS items | Compatible; do not sort |
| Tray export manually emits `video_id,frame_id` | Formatter-owned pair rows | Convenience only; no official format claim |
| `csvExporter.ts` slices to 100 and supports KIS/QA/TRAKE strings | Backend canonical formatter | Frontend export remains debug/convenience |
| `SubmissionItem.timestamp/score/shot_id` | Not in KIS organizer pair | Keep diagnostic/UI state, exclude from organizer rows |
| QA `answer?: string` | QA contract | Insufficient source; deferred |
| TRAKE `frame_ids: number[]` | N-event submission | Insufficient source; deferred |

No frontend source changes were made.

## Backend migration findings

`src/mvr/api/app.py` currently has no `/submit` route. `src/mvr/api/schemas.py`
contains only search/health models. The existing API follows injectable service
factories, Pydantic validation, and `ValueError`/`RuntimeError` mappings that a
future submission route can reuse.

Later implementation should add, without being part of this task:

- `SubmissionRequest`;
- `KisSubmissionItem`;
- `SubmissionResponse`;
- `SubmissionError`;
- `SubmissionService`;
- a canonical identity/mapping validator;
- `CompetitionFormatter` interface;
- `OrganizerTransport` protocol and documented concrete adapter.

The route should remain thin:

```text
POST /submit
  -> SubmissionService
  -> task-specific validation
  -> identity cross-check
  -> CompetitionFormatter
  -> local preview or future transport boundary
```

The existing `src/mvr/submission/kis_export.py` is a local verification
validator/exporter, not a complete `/submit` implementation. It deliberately
does not invent organizer transport.

## Formatter and organizer transport status

- Local KIS validation/format preview: `SUPPORTED_V1`.
- Official KIS formatter: `OPEN_QUESTION`.
- Organizer transport: `DEFERRED`.
- Credentials, URL, auth, retries, rate limits, and response schema:
  intentionally unspecified.
- Persistence/audit: `DEFERRED`.

The proposed formatter returns explicitly non-official pair rows until source
rules identify the official representation. Transport must accept formatter
output and must never reinterpret canonical IDs.

## Open questions

1. Official maximum answer count and whether it is exactly 100.
2. Exact official KIS CSV/JSON/API format, headers, delimiter, encoding, and
   newline policy.
3. Whether organizer scoring uses submitted order and has extra duplicate
   rules.
4. Whether repeated canonical IDs should be rejected or idempotently dropped.
5. Mapping from every corpus's canonical `keyframe_id` to organizer reference.
6. QA answer schema and semantics.
7. TRAKE event count, same-video/increasing-frame rules, and output format.
8. Persistence, audit retention, and idempotency policy.
9. Organizer transport URL/authentication/retry/rate-limit/status behavior.
10. Whether a separate validate/preview endpoint is needed once transport is
    enabled.

## Confirmation and final scope

- Only the two required documentation/specification files are intended to
  change.
- No `/submit` implementation was added.
- No frontend/backend production source was changed.
- Task06A was not modified or reimplemented.
- Task05C was not started.
- No merge or push was performed.
- Final verdict: **PASS_WITH_OPEN_QUESTIONS**.
