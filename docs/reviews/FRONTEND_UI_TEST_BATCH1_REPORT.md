# Frontend UI Test Batch 1 Report

## Final verdict

**PASS**

The hardened frontend now has focused, deterministic automated coverage for the
search adapter, search lifecycle, stale-request protection, mock safety,
canonical identity, submission behavior, CSV serialization, and critical
error/empty result states. The suite is offline and does not require a live
backend, Elasticsearch, model files, or dataset artifacts.

## Branch and revision

- Branch: `test/frontend-ui-batch1`
- Base branch: `fix/frontend-ui-hardening`
- HEAD: `639babfcd593f80f61819be4d6839f3ff9580105`
- Base commit: `639babf fix: harden frontend for unified search integration`
- Work was performed in an isolated Git worktree because the original checkout
  contained unrelated untracked user files.
- No merge and no push were performed.

## Files changed

Tooling and dependency files:

- `frontend/package.json`
- `frontend/package-lock.json`

Test support and suites:

- `frontend/tests/testData.ts`
- `frontend/tests/searchApi.test.ts`
- `frontend/tests/useSearchStore.test.ts`
- `frontend/tests/mockSafety.test.ts`
- `frontend/tests/useSubmissionStore.test.ts`
- `frontend/tests/csvExporter.test.ts`
- `frontend/tests/ResultGrid.test.tsx`

Documentation:

- `docs/reviews/FRONTEND_UI_TEST_BATCH1_REPORT.md`

No backend source file under `src/mvr/` was changed.

## Test framework and setup

- Added Vitest `4.1.11`.
- Added React Testing Library `16.3.2` and its DOM package only for the minimal
  component coverage.
- Added jsdom `29.1.1`, which is compatible with the workspace's Node
  `24.11.1`. jsdom 30 was not used because its engine requires Node 24.15 or
  newer.
- Added deterministic scripts:
  - `npm test`
  - `npm run test:watch`
- The scripts use one worker and disable file-level parallelism. Vitest's
  default parallel worker mode stalled on this Windows/Node environment before
  reporting test files; the deterministic single-worker runs completed with
  stable results in 7.51 to 44.28 seconds, with the variation coming from jsdom
  environment startup.
- Expanded lint scope from `src` to `src tests`.
- No Playwright or Cypress dependency was added.

## Search API adapter coverage

`frontend/src/api/searchApi.ts` is covered for:

- valid response normalization;
- numeric-string `frame_id` normalization;
- numeric timestamp formatting and `timestamp_sec` preservation;
- already-formatted timestamps;
- supported optional identity, media, ranking, OCR, ASR, and object fields;
- filtering non-string object entries;
- `image_path` to `thumbnail_url` normalization;
- missing `keyframe_id` rejection;
- missing `video_id` rejection;
- missing or non-numeric `frame_id` rejection;
- non-object record rejection;
- missing `results` array rejection;
- confirmation that no sequential or synthetic canonical ID is invented;
- the `/search` API boundary and payload pass-through.

## Search-store lifecycle coverage

`frontend/src/stores/useSearchStore.ts` is covered for:

- initial `idle` status;
- initial empty results, null selection, null error, and non-loading state;
- `loading` to `success`;
- API results replacing existing results;
- prior error, selection, results, and preview state clearing at search start;
- `loading` to `success-empty`;
- explicit API error state with empty results and no fixture fallback;
- local validation failure before an API call;
- current normalized payload construction;
- image-only search;
- canonical selection identity;
- deterministic concurrent-request behavior in which request B resolves before
  request A and the stale A response cannot overwrite B.

The API boundary is mocked directly. No live backend is used.

## Mock-safety coverage

Tests verify:

- fixture mode is disabled by default;
- initial results remain empty in default mode;
- calling the fixture action while disabled produces an explicit error and does
  not load fixtures;
- fixture loading occurs only after
  `VITE_USE_MOCK_SEARCH=true` is explicitly enabled;
- fixture results use unique, non-sequential canonical `keyframe_id` values;
- a production API error leaves results empty instead of silently falling back
  to fixture data;
- the fixture UI control is hidden in default mode.

## Canonical identity coverage

Tests verify:

- search selection stores `keyframe_id`;
- submission deduplication uses `keyframe_id`;
- records with the same `keyframe_id` are treated as one canonical item even if
  other fields differ;
- different `keyframe_id` values remain distinct even when `video_id` and
  `frame_id` are identical;
- removal and reordering use canonical identity;
- missing canonical identity is rejected by the API adapter;
- no sequential fixture-ID assumption remains.

## Submission-store coverage

`frontend/src/stores/useSubmissionStore.ts` is covered for:

- adding a valid result;
- preserving canonical and competition identity fields;
- duplicate prevention;
- distinct-keyframe handling;
- removal;
- reordering;
- the store-owned 100-item capacity rule and alert path.

## CSV export coverage

`frontend/src/utils/csvExporter.ts` is covered for current implemented behavior:

- KIS export;
- QA export;
- TRAKE-like multi-frame export;
- input ordering;
- video-extension stripping;
- quote escaping in QA answers;
- empty input;
- empty/missing frame arrays for KIS and QA;
- first-100-row limiting.

The tests intentionally preserve the existing serializer contract. This task
does not redesign CSV headers, delimiters, encoding policy, or validation.

## Component coverage

`frontend/src/components/Grid/ResultGrid.tsx` is rendered only where store/unit
tests are insufficient. Tests verify:

- critical backend error rendering;
- successful empty-result rendering;
- the development fixture control is absent in default mode.

No snapshot-heavy tests were added.

## Production-code changes for testability

None.

No file under `frontend/src/` or `src/mvr/` was changed. The implementation
changes are limited to frontend test dependencies, deterministic package
scripts, lint scope, test files, and this report.

## Commands run

Dependency/version and lockfile setup:

```powershell
npm view vitest version
npm view @testing-library/react version
npm view @testing-library/dom version
npm view jsdom version
npm view jsdom@29 version engines
npm view jsdom@28 version engines
npm view vitest@4.1.11 engines
npm install --package-lock-only --ignore-scripts
```

Focused diagnosis and verification:

```powershell
npx vitest run tests/csvExporter.test.ts --environment node --reporter verbose --no-file-parallelism --maxWorkers=1
npx vitest run --environment jsdom --reporter verbose --no-file-parallelism --maxWorkers=1
```

Required final verification:

```powershell
cd frontend
npm ci
npm run build
npm run lint
npm test

cd ..
git diff --check
git status --short
```

## Test counts and results

- Test files: **6 passed**
- Tests: **37 passed**
- Failed: **0**
- Skipped: **0**
- Live backend calls: **0**

Coverage distribution:

- Search API adapter: 9 tests
- Search store: 9 tests
- Mock safety: 3 tests
- Submission store: 6 tests
- CSV exporter: 7 tests
- Result grid states: 3 tests

## Build, lint, and install results

- `npm ci`: **PASS**
  - 145 packages installed
  - 0 vulnerabilities reported
- `npm run build`: **PASS**
  - TypeScript project build passed
  - Vite production build passed
- `npm run lint`: **PASS**
  - `oxlint src tests`
- `npm test`: **PASS**
  - 6 files, 37 tests
- `git diff --check`: **PASS**

## Repository safety

- Backend source changes: **0**
- Tracked/staged `frontend/node_modules/` files: **0**
- Tracked/staged `frontend/dist/` files: **0**
- Dataset, model, Elasticsearch, or generated artifact changes: **0**
- Unrelated staged files: **0**
- No files are staged.

## Known deferred limitations

- These are unit/store/component tests, not live-browser end-to-end tests.
- The current CSV format is tested as implemented; a broader CSV contract
  redesign remains deferred.
- Media serving, video seek behavior, visual upload, accessibility hardening,
  and N-event TRAKE behavior remain outside this batch.
- Backend submission transport remains unavailable.
- Search cannot produce live results until the unified backend route exists.

## Task06A dependency statement

Task06A remains intentionally unimplemented. The frontend still targets the
future unified `POST /search` contract, and the current backend does not expose
that route. These tests mock the API boundary and verify that its absence or
failure is surfaced explicitly without fixture fallback. No backend `/search`,
backend `/submit`, retrieval orchestration, or Task06A code was added.
