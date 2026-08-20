# Frontend UI Hardening Report — Batch 1

## Final verdict

**PASS_WITH_DEFERRED_TASK06A_DEPENDENCY**

The frontend hardening scope is complete. The UI now starts empty, sends real search requests through the API layer, surfaces backend failures, preserves canonical `keyframe_id` identity, uses capability-level retrieval modes, and builds reproducibly. The unified `/search` backend endpoint remains intentionally deferred to Task06A.

## Branch and revision

- Branch: `fix/frontend-ui-hardening`
- Starting branch: `review/frontend-ui-current-main`
- HEAD at implementation: `68d4a7da04fdd67ce2e5c29cd05026f53cbeaed7`
- No merge to `main` and no push performed.

## Files changed

Frontend/configuration files changed:

- `.gitignore`
- `frontend/package.json`
- `frontend/package-lock.json`
- `frontend/tsconfig.app.json`
- `frontend/tsconfig.node.json`
- `frontend/src/types/competition.ts`
- `frontend/src/api/searchApi.ts`
- `frontend/src/stores/useSearchStore.ts`
- `frontend/src/stores/useSubmissionStore.ts`
- `frontend/src/data/mockResults.ts`
- `frontend/src/components/Layout/AppShell.tsx`
- `frontend/src/components/Search/SearchWorkspace.tsx`
- `frontend/src/components/Search/QueryBar.tsx`
- `frontend/src/components/Grid/ResultGrid.tsx`
- `frontend/src/components/Grid/KeyframeCard.tsx`
- `frontend/src/components/Preview/VideoPlayerModal.tsx`
- `frontend/src/components/Submission/SortableItem.tsx`
- `frontend/src/components/Submission/SubmissionTray.tsx`
- `frontend/src/utils/csvExporter.ts`

No backend source file under `src/mvr/` was changed.

## Issues fixed from `FRONTEND_UI_REVIEW.md`

### Clean build and package hygiene

- Restored `frontend/tsconfig.app.json` and `frontend/tsconfig.node.json`.
- Added `frontend/package-lock.json`.
- Scoped the lint script to `oxlint src`, preventing dependency-tree scanning.
- Added frontend `node_modules/` and `dist/` ignore rules.

### Search lifecycle and mock safety

- Removed default fixture results; initial state is `results: []`, `selectedResultId: null`, `error: null`, and `status: 'idle'`.
- Replaced the timer/reverse-results behavior with `searchApi.search()`.
- Search start clears stale results, selection, preview state, and previous errors.
- Added `idle`, `loading`, `success`, `success-empty`, and `error` lifecycle states.
- Added request sequencing so an older response cannot overwrite a newer search.
- Backend/network/validation failures clear results and remain visible in the result grid.
- Fixture mode is opt-in through `VITE_USE_MOCK_SEARCH=true`; the development-only control is hidden otherwise.

### Canonical identity

`KeyframeResult` and `SubmissionItem` now explicitly preserve:

- `keyframe_id` for React keys, selection, preview lookup, drag-sort, deduplication, and removal;
- `video_id` and `frame_id` for competition output;
- optional `shot_id` and timestamp metadata for navigation.

Fixtures use stable keyframe-style IDs such as `L01_V001_S0001_K01`; sequential values such as `1`, `2`, or `kf-1` are no longer used as canonical identity.

The API adapter rejects result items that do not contain `keyframe_id`, `video_id`, and a numeric frame identity rather than inventing a fallback ID.

### Retrieval abstraction and task modes

- Removed `ModelEngine`, `model_engine`, `ClipVideo`, `ClipLAION`, and `Aladin` from the primary UI contract and controls.
- Added capability-level `RetrievalMode`: `semantic`, `lexical`, `object`, or `hybrid`.
- Task-mode machine values are now `kis`, `trake`, and `qa`; display labels remain `KIS`, `AVS / TRAKE`, and `Q&A`.
- The frontend does not select embedding providers or implement orchestration.

## Final frontend request/result contract

The frontend sends the future unified endpoint:

```text
POST /search
```

with the frontend-owned request shape:

```json
{
  "task_mode": "kis",
  "query": "…",
  "retrieval_mode": "hybrid",
  "sim_reorder": true,
  "top_k": 100,
  "spatial_filter": {
    "enabled": true,
    "logic": "AND",
    "objects": [],
    "colors": [],
    "max_objects_text": ""
  },
  "visual_query": null
}
```

The normalized result contract requires:

```text
keyframe_id
video_id
frame_id
timestamp
score
```

and supports optional `shot_id`, `timestamp_sec`, thumbnail/video URLs, objects, rank, source, OCR text, and ASR text. The adapter accepts a numeric backend timestamp and formats it for display, but does not synthesize missing canonical identity.

## Mock-mode behavior

Production/default mode is active unless `VITE_USE_MOCK_SEARCH=true` is explicitly provided. In default mode:

- no fixture results are rendered at startup;
- no fixture button is shown;
- `executeSearch()` always calls the API layer;
- a 404/unavailable unified endpoint becomes a visible Task06A dependency error;
- no fake fallback is performed.

With the explicit environment flag, the development-only fixture action loads the normalized fixture collection and labels the footer as development fixture mode.

## Canonical identity behavior

Selection, React keys, modal lookup, submission deduplication, drag-sort IDs, removal, and preview metadata all use `keyframe_id`. Submission items retain `video_id`, `frame_id`, `shot_id`, timestamp, and presentation metadata. No backend identity field is remapped to a sequential frontend ID.

## Verification results

From `frontend/`:

```text
npm ci         PASS
npm run build  PASS
npm run lint   PASS
```

The production build completed successfully with Vite after TypeScript project references were restored. Lint now runs only against `src` and completes without warnings.

Frontend automated tests: **NOT PRESENT**. `frontend/package.json` has no test script, so no frontend test command was available to run.

From repository root:

```text
git diff --check  PASS
```

Repository safety checks confirmed:

- no backend source files changed;
- no `frontend/node_modules/` or `frontend/dist/` files are tracked;
- no Task06A backend route or retrieval orchestration was added;
- no merge or push was performed.

## Intentionally deferred

The following remain outside this batch and belong to Task06A or later tasks:

- implementing backend `POST /search`;
- semantic/lexical/object/hybrid orchestration and fusion;
- provider routing and model-level RRF;
- backend result normalization implementation;
- final `/submit` backend behavior;
- N-event TRAKE redesign;
- real visual-query upload backend;
- media serving and video seek semantics;
- neighbor-frame retrieval;
- CSV format overhaul;
- accessibility overhaul;
- asset optimization;
- new frontend test framework.

## Task06A dependency status

The frontend is structurally ready to consume the future unified orchestration endpoint, but the current backend does not provide `POST /search`. That absence is surfaced as an explicit error and is intentionally not worked around by calling specialized backend routes. Task06A remains unimplemented.

## Remaining known limitations

- Search cannot return live results until Task06A supplies the unified backend endpoint and normalized response.
- Video preview still depends on existing media URL behavior; media serving and timestamp seek were explicitly out of scope.
- The existing UI still contains prototype spatial/temporal controls, but their request fields are now represented without adding backend implementation.
- No automated frontend test suite exists.
