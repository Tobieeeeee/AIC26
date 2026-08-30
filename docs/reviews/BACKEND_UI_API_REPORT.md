# Backend UI API Report

Version: 1.0
Branch: `feat/backend-ui-api`
Commit: `6e7f1a3` — "feat: add frontend API adapter for retrieval UI"
Author: Golden <24521777@gm.uit.edu.vn>
Reviewed against: `origin/main` @ `712274d`

## Verdict

```
BACKEND_UI_API_UNIT_TESTED
NEVER_RUN_END_TO_END
MERGE_INTO_CURRENT_MAIN_BREAKS_/api/v1/search
LAUNCH_SCRIPT_PRECONDITIONS_UNMET
```

The adapter is implemented and its 52 unit tests pass on this branch. It has
never served a real request: no index it can load exists on this machine, and
the launch script's dataset root is an empty directory. Merging it into
`main` as it stands produces a `TypeError` on the TRAKE path of the new
endpoint, with no Git conflict to warn anyone.

## 1. What was added

Diff against the merge base (`git diff --stat origin/main...origin/feat/backend-ui-api`):
**5 files, 440 insertions, 0 deletions.** Nothing was modified or removed.

| File | Purpose |
|---|---|
| `src/mvr/api/app.py` (+251) | CORS middleware; `_format_timestamp`, `_frame_asset_urls`, `_ui_result` helpers; the `POST /api/v1/search` and `POST /api/v1/submit` endpoints the UI talks to. |
| `src/mvr/api/schemas.py` (+69) | `UiSearchRequest`, `UiSpatialFilter`, `UiVisualQuery`, `UiSearchResponse`, `UiSubmitRequest`/`Response` and friends — 22 Pydantic models in total on the branch. |
| `tests/unit/api/test_app.py` (+92) | Unit coverage for the new endpoints. |
| `scripts/run_backend_full.ps1` (+27) | Sets `MVR_*` env vars and starts uvicorn on `127.0.0.1:8000`. |
| `.gitignore` (+1) | One added entry. |

`create_app()` constructs, and the application exposes 7 routes:

```text
GET   /health
POST  /api/v1/search        <- new
POST  /api/v1/submit        <- new
POST  /search/semantic
POST  /search/kis
POST  /search/trake
POST  /search/qna
```

No documentation was added. This report is the missing deliverable required by
`AGENTS.md` §19 step 11 and §20.

## 2. Verification

Run from a dedicated worktree checked out at `6e7f1a3`, so the branch's own
`src/` is under test and not another branch's:

```text
pytest tests/unit/api/  ->  52 passed in 16.92s
```

That the worktree's source was the one imported was confirmed, not assumed:

```text
mvr resolved to: <worktree>\src\mvr\__init__.py
```

`create_app()` was constructed for real and its route table read back (above).

## 3. Merging this into `main` breaks the new endpoint

`main` commit `712274d` ("unify query preprocessing") moved translation out of
the retrieval functions:

```python
# search_trake()
-    query_translator: QueryTranslator | None = None,
+    lexical_queries: Sequence[str | None] | None = None,
```

`search_qna()` changed the same way (`query_translator` → `lexical_query`), and
`main` updated its own call sites accordingly. This branch predates that change
and adds a *new* `search_trake()` call that still passes `query_translator`.

Git does not object. The trial merge is clean:

```text
git merge-tree --write-tree origin/main origin/feat/backend-ui-api
-> exit 0, tree b0714903f20ae7ff9948c8da9ba7497f3ebdd53b
```

Reading the merged `app.py` and comparing every call's keywords against the
live signatures:

```text
line  992  search_trake()  -> ok
line 1023  search_qna()    -> ok
line  760  search_qna()    -> ok
line  821  search_trake()  -> BROKEN ['query_translator']
```

Line 821 is the TRAKE branch of `POST /api/v1/search`. On merge it raises
`TypeError: search_trake() got an unexpected keyword argument
'query_translator'` the first time the UI issues a TRAKE query — after the
encoder and index have loaded, so the failure looks like a runtime fault
rather than a merge defect.

The same check run against this branch in isolation reports all four call sites
`ok`. The branch is internally consistent; only the merge is broken. The fix is
one call site: build the dense/lexical pair with `QueryPreprocessingService`
and pass `lexical_queries=`, exactly as `main`'s own TRAKE route at line 992
already does.

## 4. The launch script cannot start the server

`scripts/run_backend_full.ps1`, checked path by path on this machine:

| Path the script requires | State |
|---|---|
| `D:\AIC26\data\raw` | exists but is **empty** — 0 entries, 0 bytes |
| `D:\AIC26\artifacts\indexes\aic_full\manifest.json` | missing → script exits 2 |
| `D:\AIC26\artifacts\indexes\aic_smoke` | missing |

Three consequences:

1. The script exits 2 before reaching uvicorn. The backend has never started
   from it.
2. Its fallback message, `"Smoke index exists at: $smokeIndex"`, names a
   directory that does not exist. The operator is pointed at nothing.
3. The `Test-Path` guard on the dataset root passes on an empty directory, so
   `MVR_KEYFRAME_DIR` and `MVR_VIDEO_DIR` would be exported pointing at
   nothing even if an index appeared.

The real corpus on this machine is at `E:/AIC26_data`, and the working index is
`artifacts/indexes/qna_organizer_all` (177,321 vectors, `ViT-B-32-quickgelu` /
`openai`, `image_root: E:/AIC26_data`). The script knows about neither.

Separately, the script hardcodes `D:\AIC26\...` in three places. `main` scrubbed
personal absolute paths out of eight scripts in favour of `<your-dataset-root>`
placeholders and CLI arguments (`637be9d`, "make run_canonical.py portable via
CLI args"). This script reintroduces the pattern.

## 5. Branch state

- Not merged into `main`, nor into any other remote branch.
- 6 commits behind `origin/main`.
- No `docs/` change, no `docs/CODEX_PROGRESS.md` entry.

## 6. What remains

1. Fix the `search_trake()` call at merged-line 821 to the `lexical_queries`
   contract. One call site.
2. Make `run_backend_full.ps1` take the dataset root and index as arguments
   instead of hardcoding `D:\AIC26`, and correct or drop the smoke-index
   message.
3. Build an index the script can actually load, or point it at
   `artifacts/indexes/qna_organizer_all`, then start the server and issue one
   real request per endpoint. Until that happens, no claim about end-to-end
   behaviour is supportable.
4. Rebase onto `origin/main` and re-run `pytest tests/unit/`.

Per `AGENTS.md` §20, this branch must not be described as an end-to-end working
backend: it has not been executed end to end.
