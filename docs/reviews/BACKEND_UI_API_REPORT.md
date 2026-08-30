# Backend UI API Report

Version: 2.0
Branch: `docs/backend-ui-api-report`, rebased onto `origin/main` @ `712274d`
Reviews: `6e7f1a3` — "feat: add frontend API adapter for retrieval UI",
Golden <24521777@gm.uit.edu.vn>

## Verdict

```
BACKEND_UI_API_RUN_END_TO_END_ON_REAL_DATA
MERGE_DEFECT_FOUND_AND_FIXED
LAUNCH_SCRIPT_REWRITTEN_AND_EXERCISED
UI_ADAPTER_TRAKE_MEDIA_FIELDS_STILL_WRONG
```

Version 1.0 of this report said the backend had never served a real request.
It has now. The server was started from the rewritten launch script against
the real 177,321-vector index and the real corpus on `E:`, and answered both a
KIS and a TRAKE query with correct video ids, frame indices and timestamps.
Getting there required fixing a merge defect and three separate faults in the
launch script. Two defects in the UI adapter's TRAKE result mapping remain,
documented below rather than fixed, being outside the scope of this branch.

## 1. What `6e7f1a3` delivered

5 files, 440 insertions, 0 deletions against its merge base. Nothing existing
was modified or removed.

| File | Purpose |
|---|---|
| `src/mvr/api/app.py` (+251) | CORS middleware; `_format_timestamp`, `_frame_asset_urls`, `_ui_result`; the `POST /api/v1/search` and `POST /api/v1/submit` endpoints the UI calls. |
| `src/mvr/api/schemas.py` (+69) | The `Ui*` request/response models. |
| `tests/unit/api/test_app.py` (+92) | Unit coverage for the new endpoints. |
| `scripts/run_backend_full.ps1` (+27) | Launch script. |
| `.gitignore` (+1) | One entry. |

No documentation. This report is the deliverable `AGENTS.md` section 19 step 11
and section 20 require.

## 2. Merging into `main` broke `/api/v1/search`, and Git said nothing

`main` `712274d` moved translation out of the retrieval functions:

```python
# search_trake()
-    query_translator: QueryTranslator | None = None,
+    lexical_queries: Sequence[str | None] | None = None,
```

`search_qna()` changed the same way. `main` updated its own call sites; this
branch predates the change and adds a NEW `search_trake()` call still passing
`query_translator`. The two sides never touch the same line, so:

```text
git merge-tree --write-tree origin/main origin/feat/backend-ui-api
-> exit 0, tree b0714903f20ae7ff9948c8da9ba7497f3ebdd53b
```

A clean merge. Comparing every call's keywords in the merged file against the
live signatures found it:

```text
line  992  search_trake()  -> ok
line 1023  search_qna()    -> ok
line  760  search_qna()    -> ok
line  821  search_trake()  -> BROKEN ['query_translator']
```

The existing 52 API tests did not catch this: nothing exercised the TRAKE
branch of `/api/v1/search`. A test was written first, and failed for the right
reason before any production code was touched:

```text
TypeError: search_trake() got an unexpected keyword argument 'query_translator'
src\mvr\api\app.py:821
```

FIXED by preparing the dense/lexical pair through `QueryPreprocessingService`
and passing `lexical_queries=`, identical to what `main`'s own `/search/trake`
route does thirty lines below. The regression test
(`test_ui_search_trake_mode_reaches_search_trake`) stays.

## 3. The launch script could not start the server

`scripts/run_backend_full.ps1` was rewritten. Three faults, each verified on
this machine before and after.

**a. Hardcoded paths.** It pinned `D:\AIC26\data\raw` and
`D:\AIC26\artifacts\indexes\aic_full`, so it ran on exactly one machine.
`main` `637be9d` had just removed this pattern from eight other scripts.
`-DatasetRoot` and `-Index` are now mandatory parameters, with `-KeyframeDir`,
`-VideoDir`, `-Python`, `-Device`, `-BindAddress` and `-Port` optional.

**b. The dataset guard passed on a directory holding no dataset.**
`D:\AIC26\data\raw` contains exactly one entry: a zero-byte `.gitkeep`. A bare
`Test-Path` is satisfied by that, so the script would export `MVR_KEYFRAME_DIR`
pointing at nothing and start a backend with nothing behind it. Version 1.0 of
this report called that directory empty. It is not, and the distinction
matters: the first replacement guard used `Get-ChildItem -Force` and was
fooled by the same `.gitkeep`. The guard now ignores `.gitkeep`/`.gitignore`:

```text
THROW: Dataset root holds no dataset (placeholder files only): D:\AIC26\data\raw
```

**c. `python` resolved to the wrong interpreter.** The script invoked bare
`python`, which here is the system Python 3.12, not the venv:

```text
C:\...\Python312\python.exe: No module named uvicorn
```

`-Python` now names the interpreter.

Two further corrections: the missing-index message named an `aic_smoke`
directory that does not exist, and is gone; the keyframe/video directory checks
now warn and name the flag to pass instead of failing silently later.

## 4. The environment variable whose absence returned nothing, quietly

With the script fixed the server started, but `/api/v1/search` answered
`200 OK` with `total_results: 0` for every query — the failure mode that reads
as "nothing matched" rather than "misconfigured".

Cause: `_default_kis_pipeline_resolver()` reads `MVR_KEYFRAME_MAPPING_CSV`,
which the script never set. Without it no `KeyframeFrameIndexResolver` is
built, so every candidate fails to resolve to an original frame index and is
dropped. The script now takes `-KeyframeMappingCsv`, and when it is absent says
so in yellow instead of starting a backend that answers nothing.

The mapping was checked against the index rather than assumed:

```text
artifacts/keyframe_mapping.csv   177,322 lines (177,321 rows + header)
sha256                           e57bfdb35b86946a8b6aace5bed5949293a2d8d1f9ab92a95cb9475ba9779f1a
manifest.mapping_sha256          e57bfdb35b86946a8b6aace5bed5949293a2d8d1f9ab92a95cb9475ba9779f1a
manifest.mapping_row_count       177321
```

## 5. Verification

Unit tests, from a worktree checked out on this branch, with the imported
source confirmed to be that worktree's own (`mvr resolved to:
<worktree>\src\mvr\__init__.py`) rather than another branch's:

```text
pytest tests/unit/   ->  1091 passed, 3 skipped in 29.23s
```

Live run, server started via the rewritten script against the real corpus:

```text
run_backend_full.ps1 -DatasetRoot E:\AIC26_data
                     -Index D:\AIC26\artifacts\indexes\qna_organizer_all
                     -KeyframeMappingCsv D:\AIC26\artifacts\keyframe_mapping.csv
                     -Python D:\AIC26\.venv\Scripts\python.exe -Port 8012

GET /health -> 200
{"semantic":{"configured":true,"available":true,"loaded":true}}
```

`loaded: true` is the index actually resident, not merely configured.

`POST /api/v1/search`, `task_mode: KIS`, "a man riding a bicycle on a street",
24.9s including first-query model load:

```json
{"id":"L23_V019:98:1","video_id":"L23_V019","frame_id":98,
 "timestamp":"00:00:03","timestamp_sec":3.92,
 "thumbnail_url":"/keyframes/L23_V019/002.jpg","score":0.3065}
```

`POST /api/v1/search`, `task_mode: TRAKE`, two events, 4.8s — the path fixed in
section 2, which before the fix returned HTTP 500:

```json
{"id":"L27_V015:5674:1","video_id":"L27_V015","frames":[5674,5786],"score":0.674}
```

That chain is independently corroborated: the same two events run through
`scripts/run_trake_pipeline.py` during the live round returned
`L27_V015 (5674, 5786, 7741, 13094)` as chain #3. The API and the pipeline
agree.

## 6. Defects found live and NOT fixed here

**a. TRAKE thumbnails 404.** The TRAKE branch passes `keyframe_id=None,
image_path=None` to `_ui_result`, so the URL is built from the frame index:
`/keyframes/L27_V015/5674.jpg`. Keyframe files are named by keyframe id — the
real directory holds `001.jpg`, `002.jpg`, ... and no `5674.jpg`. The KIS
branch, which passes the keyframe id, produces `/keyframes/L23_V019/002.jpg`,
and that file exists. Both checked on disk.

**b. TRAKE results carry no timestamp.** The branch passes `timestamp=None`,
so every chain reports `"timestamp":"00:00:00"`, `"timestamp_sec":0.0`. KIS
results carry real timestamps.

**c. Media serving assumes the normalized layout.** `MVR_KEYFRAME_DIR` is a
single directory, but the organizer download splits keyframes across 14 roots
(`Keyframes_L21/keyframes` through `Keyframes_L30/keyframes`). No single
`-KeyframeDir` covers the corpus, so `/search/semantic` returns
`503 "semantic result image is outside the configured keyframe directory"`.
Retrieval is unaffected; only media URLs are. This needs a decision — a
multi-root served mount, or a normalization pass — not another script flag.

## 7. Branch state

- `feat/backend-ui-api` is not merged into `main` and was 6 commits behind it.
- This branch rebases it onto `712274d` and adds the fixes above.
- `feat/backend-ui-api` belongs to another author; nothing was pushed to it.
