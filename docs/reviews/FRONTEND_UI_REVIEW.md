# Frontend UI Independent Review

## Verdict

**NEEDS_FIXES**

The frontend is a visually substantial operator-console prototype, but it is not merge-ready as a production retrieval UI. The primary blockers are that search remains mock-only, fixture results are shown by default, canonical keyframe identity is not represented end-to-end, and the current frontend request/response contract does not match the current FastAPI backend. The frontend build also cannot run from a clean checkout because its referenced TypeScript project files are missing.

## Review metadata

- Review date: 2026-08-20 (Asia/Saigon)
- Branch: `review/frontend-ui-current-main`
- HEAD: `68d4a7da04fdd67ce2e5c29cd05026f53cbeaed7`
- Base relationship: branch is ahead of `origin/main` by 3 commits; the review merge commit is `68d4a7d`.
- Review scope: read-only audit of the current frontend and current backend/API source. No frontend or backend implementation was changed.

## Scope and repository audit

The application changes under review are confined to the frontend feature work. Existing untracked files are pre-existing runtime, artifact, or note files; they were not modified or staged by this review. No merge, push, generated frontend artifact, lockfile, or dependency change was committed.

The frontend package has no `package-lock.json`. `npm install --no-package-lock` was used only to run the requested checks, so installation is not reproducible from a lockfile yet. `node_modules` created for the check is ignored runtime data and is not part of the review change.

## Verification evidence

| Check | Result |
|---|---|
| Branch check | PASS: exact required branch |
| `git diff --check` | PASS after report creation |
| `npm install --no-package-lock` | PASS; 63 packages installed, no reported vulnerabilities |
| `npm run build` | FAIL: `frontend/tsconfig.json` references missing `tsconfig.app.json` and `tsconfig.node.json` |
| `npm run lint` | Process completed, but scans `node_modules` and emits a very large dependency warning stream; it is not a clean, focused project lint |
| `npx oxlint src --deny-warnings` | PASS |
| Frontend automated tests | **NOT PRESENT**; `package.json` has no test script |

The build failure is independently reproducible from the current checkout:

```text
error TS5083: Cannot read file '.../frontend/tsconfig.app.json'.
error TS5083: Cannot read file '.../frontend/tsconfig.node.json'.
```

## Architecture summary

The frontend is a React/Vite/Zustand application composed of search controls, a result grid, a preview modal, spatial controls, and a submission tray. The UI components are modular enough for continued work, but the central store currently owns fixture data and simulates search rather than orchestrating an API request. The backend is a FastAPI application with specialized semantic, KIS, text, object, and hybrid routes.

## API contract audit

`frontend/src/api/searchApi.ts` posts `SearchPayload` to generic `POST /search` and posts submissions to `POST /submit`. The current backend source in `src/mvr/api/app.py` exposes `POST /search/semantic`, `/search/kis`, `/search/text`, `/search/object`, and `/search/hybrid`; no generic `/search` or `/submit` route exists.

The frontend payload uses `task_mode`, `query`, `model_engine`, `sim_reorder`, `spatial_filter`, `visual_query`, and `top_k`. The current backend route schemas use route-specific request models and do not accept this frontend shape as a generic orchestration request.

The response contract is also unsafe. The frontend casts `response.data.results` directly to `KeyframeResult[]`. The backend `SearchHit` contains `rank`, string `frame_id`, `video_id`, numeric `timestamp`, `score`, `image_path`, OCR/caption/object fields, and ranking metadata. The frontend expects `id`, numeric `frame_id`, formatted `timestamp`, `timestamp_sec`, thumbnail/video URLs, and an object array. `KisHit` optionally carries `keyframe_id`, but the normal `SearchHit` contract does not expose the canonical identity required by the UI.

The missing generic `/search` is a known dependency for **Task06A — Unified Search Orchestration API**. The frontend should not work around this by hard-coding five specialized backend calls in this review.

## Search-store audit

`frontend/src/stores/useSearchStore.ts` initializes `results` from `initialMockResults` and `selectedResultId` to `'1'`. `executeSearch()` sets a loading flag, waits 300 ms, reverses the current result array, and clears loading. It does not call `searchApi.search()`. `runMockSearch()` is also exposed through a visible “Fixture Search” button in `ResultGrid.tsx`.

There is no explicit `error` state, request cancellation, stale-response protection, or validation/error distinction. Because results begin populated and no backend request is made, a failed or unavailable backend cannot be distinguished from a successful search. Selected-result state is not reset based on a new result set.

Required state distinctions are currently missing: idle, loading, successful non-empty, successful empty, validation error, and backend/network error.

## Mock safety

There are two fixture sources: `frontend/src/stores/useSearchStore.ts` contains `initialMockResults`, and `frontend/src/data/mockResults.ts` contains another mock collection. The store uses fixtures as the default runtime state rather than an explicit development-only mode. Fixture video URLs point to `https://www.w3schools.com/html/mov_bbb.mp4`, and the shell explicitly displays `API status: Online (Mock Fixtures)`.

This makes the interface appear operational without a real retrieval round trip and risks silently masking backend failures.

## Canonical identity audit

`frontend/src/types/competition.ts` defines `KeyframeResult` with generic `id`, `video_id`, and `frame_id`, but no explicit `keyframe_id` or `shot_id`. React keys, selection, preview lookup, drag-sort IDs, submission deduplication, and removal all use `id`.

The required identity layers are not preserved:

- `keyframe_id`: canonical internal retrieval/join identity;
- `video_id + frame_id`: competition output identity;
- `shot_id + timestamp`: navigation/temporal metadata.

The generic `id` values in fixture data (`1`, `2`, and `kf-*`) cannot safely substitute for canonical backend identity.

## Model and retrieval coupling

`ModelEngine` is hard-coded to `'use all' | 'ClipVideo' | 'ClipLAION' | 'Aladin'` and is rendered in the query controls. `RetrievalMethod` uses `ocr | asr | clip | fusion | spatial`. This exposes provider/model-era vocabulary in the main UI contract rather than capability-level retrieval modes (`semantic`, `lexical`, `object`, `hybrid`).

This coupling will make the planned BEiT-3, MetaCLIP-2, Perception Encoder, and model-level RRF migration harder. Task06A should own provider orchestration while the UI selects capabilities.

## Task and temporal extensibility

The display label `AVS / TRAKE` is mapped to machine value `TRAKE`, while Q&A uses a separate display spelling. Search state is hard-coded to `scene1`, `scene2`, and `scene2Query`. This is acceptable as a prototype affordance but does not provide an extensible N-event TRAKE model.

## Spatial and visual-query audit

`SpatialPanel.tsx` exposes object/color controls, `maxObjectsText`, scene controls, and a visual-query action. `maxObjectsText` is maintained in UI state but is not represented in a real request because the store never constructs a request. The upload button calls `prompt()` for an image URL; it is not a file input or upload flow. The state shape suggests filename/evaluation metadata, but only an image URL is actually maintained.

Visible controls therefore do not have a verified backend payload mapping. This should be resolved as part of the unified orchestration contract rather than by adding ad hoc frontend fields.

## Grid and preview audit

The result grid groups by `video_id` and renders generic `id`-based selection. Cards use clickable `div` elements in several places, which is weak for keyboard accessibility. Thumbnail URLs can be empty and there is no robust image-error fallback.

`VideoPlayerModal.tsx` uses the supplied `video_url` or an external demo MP4 fallback. It does not seek the video element to `timestamp_sec`/`timestamp`; nearby-frame controls only call `alert()`. The modal has no explicit dialog semantics, focus trap, or focus restoration. Preview metadata and lookup also use generic `id` and frontend-only field assumptions.

## Submission audit

`useSubmissionStore.ts` deduplicates and removes items by generic `id`, stores only frontend-shaped `video_id`, `frame_id`, timestamp, and score, and does not preserve `keyframe_id` or `shot_id`. The 100-item cap is enforced with `alert()`. The tray can therefore submit stale or synthetic identities if results came from fixtures or an unadapted backend response.

No current backend `/submit` route was found, so submission is not end-to-end compatible on this branch.

## CSV/export audit

`SubmissionTray.tsx` manually emits `video_id,frame_id` lines without a header or robust escaping. `frontend/src/utils/csvExporter.ts` supports KIS, QA, and TRAKE-like formats, but the task-mode type uses `Q&A`; this naming mismatch needs an explicit adapter. CSV output has no header, no general comma/quote escaping for IDs, no explicit UTF-8 BOM/encoding policy, and assumes `frame_ids[0]` exists for KIS and Q&A. Empty exports and duplicate policy are not clearly defined. Ordering is based on current tray order, which is useful, but identity is still generic-ID based upstream.

## Task06A readiness

The UI has a natural seam for migration because API calls are isolated in `searchApi.ts` and state is centralized in `useSearchStore.ts`. It is not currently ready for Task06A integration without first:

1. replacing simulated search with a real request path and explicit error/empty states;
2. defining a normalized UI response adapter;
3. preserving `keyframe_id`, `video_id`, `frame_id`, `shot_id`, and timestamp;
4. changing the UI request to capability-level retrieval fields;
5. agreeing the unified `/search` and submission contracts.

Task06A should provide the orchestration endpoint and normalized result contract. The frontend should consume that contract rather than own backend model/provider selection.

## Findings

| ID | Severity | File(s) | Evidence | Impact | Recommendation |
|---|---|---|---|---|---|
| UI-001 | BLOCKER | `frontend/src/stores/useSearchStore.ts` | `executeSearch()` only reverses results after `setTimeout`; no `searchApi.search()` call | No real retrieval E2E exists | Implement the agreed real search path before merge |
| UI-002 | HIGH | `frontend/src/stores/useSearchStore.ts`, `frontend/src/components/Grid/ResultGrid.tsx` | `results: initialMockResults`; visible “Fixture Search” | UI can appear successful without backend | Default to empty state; make fixtures explicit dev-only mode |
| UI-003 | HIGH | `frontend/src/types/competition.ts`, grid/preview/submission consumers | Generic `id`; no `keyframe_id`/`shot_id` | Retrieval/join and submission identity can be corrupted | Normalize and preserve canonical identity across all consumers |
| UI-004 | HIGH | `frontend/src/api/searchApi.ts`, `src/mvr/api/app.py`, `src/mvr/api/schemas.py` | Frontend calls `/search` and `/submit`; backend has neither | Requests cannot reach current backend; submission is unavailable | Resolve through the unified API contract; do not patch around Task06A |
| UI-005 | HIGH | `frontend/src/api/searchApi.ts`, `frontend/src/types/competition.ts` | Direct cast of incompatible payload/response shapes | Runtime fields are missing or mistyped | Add a strict request/response adapter with validation |
| UI-006 | HIGH | `frontend/src/stores/useSearchStore.ts` | No `error`; mock behavior has no rejection path | Backend failures cannot be shown safely | Add explicit error and empty-result states |
| UI-007 | HIGH | `frontend/tsconfig.json` | References missing `tsconfig.app.json` and `tsconfig.node.json` | Clean frontend build fails | Restore or correct the TypeScript project references |
| UI-008 | MEDIUM | `frontend/src/types/competition.ts`, query controls | `ModelEngine` provider names and `clip`/`fusion` retrieval values | Couples UI to obsolete providers | Use capability-level modes; defer provider orchestration to backend |
| UI-009 | MEDIUM | `frontend/src/stores/useSearchStore.ts`, `SpatialPanel.tsx` | `scene1`/`scene2`; `maxObjectsText` not sent | Spatial/temporal controls are incomplete | Define normalized object/temporal request fields in Task06A |
| UI-010 | MEDIUM | `frontend/src/components/Search/SpatialPanel.tsx` | Upload uses `prompt()` URL | No actual image upload | Add an agreed upload/reference flow |
| UI-011 | MEDIUM | `frontend/src/components/Preview/VideoPlayerModal.tsx` | External demo MP4; no seek; alert-only frame stepping | Preview does not represent the selected keyframe in production | Bind preview to backend media and timestamp semantics |
| UI-012 | MEDIUM | `frontend/src/stores/useSubmissionStore.ts`, `SubmissionTray.tsx` | Generic-ID dedupe; alert-only capacity/empty handling; no `/submit` route | Submission correctness and UX are unreliable | Normalize identity and agree submission API |
| UI-013 | MEDIUM | `frontend/src/utils/csvExporter.ts`, `SubmissionTray.tsx` | No headers/robust escaping; QA/Q&A mismatch; unsafe empty frame access | Export may be invalid for real task data | Add task-specific validated CSV serialization |
| UI-014 | MEDIUM | `frontend/src/components/Grid/KeyframeCard.tsx`, `VideoPlayerModal.tsx` | Clickable `div`s; incomplete dialog semantics; alert controls | Accessibility and usability gaps | Add keyboard semantics and accessible modal lifecycle |
| UI-015 | LOW | frontend source/comments | Mojibake appears in comments and labels | Reduced maintainability/readability | Normalize source encoding during routine cleanup |
| UI-016 | LOW | `frontend/package.json` | No lockfile; unscoped lint scans dependencies | Reproducibility and signal quality are weak | Commit a lockfile and scope lint to project sources |
| UI-017 | LOW | `frontend/public/okchokchok-logo.png`, `okchokchok.png` | Approximately 6.5 MB and 2.3 MB | Larger-than-needed frontend payload | Optimize assets when convenient |

## Required fixes before frontend merge

- Replace mock-only `executeSearch()` with the agreed production request path.
- Remove fixture results as the default runtime state; keep mock mode explicit and development-only.
- Add explicit loading, empty, validation, and backend/network error states.
- Define and enforce a normalized response contract that preserves canonical identity.
- Restore the missing TypeScript project files or correct `tsconfig.json` so a clean build passes.
- Do not mark the frontend production-ready until the `/search` and `/submit` contracts are executable or an explicitly documented integration stub is supplied.

## Defer to Task06A — Unified Search Orchestration API

- Provide generic `POST /search` request and response contracts.
- Normalize specialized semantic/lexical/object/hybrid results for UI consumption.
- Move model/provider selection and fusion/RRF ownership to the backend.
- Define object constraints, temporal events, visual-query references, top-K, and error semantics.
- Provide or explicitly replace the submission route/contract.

## Future and non-blocking improvements

- Expand two-scene state into a general event/TRAKE representation.
- Implement real image upload and backend media serving/seek behavior.
- Improve keyboard accessibility, modal focus management, thumbnail fallback, and replace alert-based interactions.
- Add robust task-specific CSV headers, escaping, encoding, and empty/duplicate policy.
- Add a frontend lockfile, focused lint configuration, automated component/store tests, source encoding cleanup, and image optimization.

## Known limitations

- This review did not claim retrieval quality or successful E2E search because the current frontend does not make a real search request and no unified `/search` route exists.
- No frontend automated test suite is present.
- The broad `npm run lint` script is not a clean project-only lint because it traverses installed dependencies; the focused source lint passed.
- Existing untracked artifacts and notes were outside the review scope and were not altered.

## Final conclusion

**NEEDS_FIXES.** The UI can be continued as a prototype, but it should not be merged as a production frontend until real search/error behavior, canonical identity, build reproducibility, and the unified backend contract are resolved. Task06A is the correct place to establish the orchestration API; this review intentionally did not implement that work.
