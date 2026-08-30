# Backend UI API Report

Version: 4.1

Supersedes v1.0, which `main` merged as `d58f70a` (PR #20) before this branch
was rebased. That version states the backend was never run end to end, and
calls `D:\AIC26\data\raw` empty. The first is no longer true; the second was
never true -- see section 3b. Merging this branch replaces it.
Branch: `docs/backend-ui-api-report`, rebased onto `origin/main` @ `712274d`
Reviews: `6e7f1a3` — "feat: add frontend API adapter for retrieval UI",
Golden <24521777@gm.uit.edu.vn>

## Verdict

```
BACKEND_UI_API_RUN_END_TO_END_ON_REAL_DATA
MERGE_DEFECT_FOUND_AND_FIXED
LAUNCH_SCRIPT_REWRITTEN_AND_EXERCISED
UI_ADAPTER_TRAKE_MEDIA_FIELDS_FIXED
R2_REORG_VIDEO_LAYOUT_DIVERGENCE_FIXED_BUT_UNPROVEN_ON_REAL_VIDEO
BORROWS_ONE_UNMERGED_COMMIT_FROM_feat/official-submission-format
```

Version 1.0 of this report said the backend had never served a real request.
It has now. The server was started from the rewritten launch script against
the real 177,321-vector index and the real corpus on `E:`, and answered both a
KIS and a TRAKE query with correct video ids, frame indices and timestamps.
Getting there required fixing a merge defect and three faults in the launch
script. Running it then exposed three more, in the UI adapter's TRAKE result
mapping and in how keyframe images are served; those are fixed too, and the
fixes are shown working against the real corpus rather than only in tests.

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

### That merge has since happened

PR #20 merged this branch's v1.0 commit into `main`. Because the branch was cut
from `feat/backend-ui-api`, that carried Golden's `6e7f1a3` along with it, so
`main` took the endpoint as well as the report warning about it. Checked
against `main`'s own `search_trake` signature, not this branch's:

```text
origin/main app.py line 992: ok
origin/main app.py line 821: BROKEN ['query_translator']
```

`POST /api/v1/search` with `task_mode: TRAKE` raises `TypeError` on `main`
today, and no test on `main` covers that path. The fix is on this branch;
merging it is what closes this.

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

## 5. The backend, running for the first time

Test totals for the finished branch are in section 8, which is the only place
this report quotes them. This section is the first live run -- what turned the
sections above from readings of the code into observations of it.

Server started via the rewritten script against the real corpus:

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

## 6. Three defects found live, and fixed

All three were found by running the server, not by reading it. Each was
reproduced by a failing test before anything was changed.

**a. TRAKE thumbnails 404ed.** The branch passed `keyframe_id=None,
image_path=None` to `_ui_result`, so the URL fell through to the frame index:
`/keyframes/L27_V015/5674.jpg`. Keyframe files are named by keyframe id — that
directory holds `001.jpg` … and no `5674.jpg`.

A chain carries frame indices and nothing else, so the fix reads the keyframe
mapping backwards. `KeyframeFrameIndexResolver.keyframe_id_for(video_id,
frame_idx)` is new, and it does not pretend the reverse is one-to-one:
measured on the real 177,321-row mapping there are only 176,707 distinct
`(video_id, frame_idx)` pairs, so 614 carry two keyframes — adjacent keyframes
about 0.04s apart that round to the same frame (`L24_V003` keyframe `172` at
556.52s and `173` at 556.56s both land on frame 13913). The lowest keyframe id
wins, so one frame always yields one thumbnail.

**b. TRAKE results carried no timestamp.** `timestamp=None` meant every chain
reported `00:00:00`; a reviewer scrubbing to a hit landed at the start of the
video every time. The timestamp now comes from the registry via the recovered
keyframe id.

The id comes from `keyframe_mapping.csv` and the timestamp from the
organizer's `map-keyframes` — two files. If they disagree about which frame an
id sits at, the timestamp belongs to a different moment and nothing on screen
would say so, so the round trip is checked and a disagreeing timestamp is
withheld rather than shown.

**c. Media serving assumed the normalized layout.** `MVR_KEYFRAME_DIR` names
one directory; the organizer download splits the corpus across fourteen
(`Keyframes_L21/keyframes` … `Keyframes_L30/keyframes`). `StaticFiles` mounted
on one of them 404s everything else.

`GET /keyframes/{video_id}/{filename}` is now a route registered ahead of the
mount. It tries the mounted directory first, then asks `AicDatasetRegistry`,
which already globs the batch folders. Path components that are empty, `..`,
or contain a separator are refused.

That registry behaviour comes from commit `1e9a8d4` ("fix: read the dataset
layout the organizer actually ships"), which lives on
`feat/official-submission-format` and is **not yet on `main`**. It is
cherry-picked here as a prerequisite. Both branches now carry the same patch;
whichever merges second should be a no-op for those two files.

Live evidence, same server, after the fixes:

```text
TRAKE  L27_V015  frame=5674  ts=00:03:46 (226.96)  thumb=/keyframes/L27_V015/140.jpg
                             was: ts=00:00:00 (0.0)  thumb=.../5674.jpg

GET /keyframes/L27_V015/140.jpg   200   163623 bytes   (Keyframes_L27)
GET /keyframes/L23_V019/002.jpg   200   158180 bytes   (Keyframes_L23)
GET /keyframes/L21_V001/001.jpg   200   135232 bytes   (Keyframes_L21)
GET /keyframes/L30_V031/010.jpg   200   260761 bytes   (Keyframes_L30)
GET /keyframes/L27_V015/..%2F..%2F..%2Fsecret.txt   404
```

163,623 bytes is the byte-for-byte size of
`E:\AIC26_data\Keyframes_L27\keyframes\L27_V015\140.jpg` on disk. Four
different batch roots answered in one process, which is exactly what a single
mount cannot do.

KIS was re-checked for regression and is unchanged: `L23_V019` frame 98 at
`00:00:03`, thumbnail `/keyframes/L23_V019/002.jpg` serving 158,180 bytes.

## 7. The R2 reorg moved the videos and nothing on the reading side was told

Section 6c blamed the organizer's fourteen keyframe roots for the media
problem. That was half the story. Chasing why no `Videos_*` folder exists on
this machine turned up a second, unrelated divergence.

`main` `712274d` ("finish R2 reorg") rewrote object keys **inside the bucket**:

```text
Videos_L21_a/video/L21_V001.mp4  ->  data/raw/batch1/videos/L21/L21_V001.mp4
```

`src/mvr/operations/r2_layout.py` declares that canonical
(`raw_video_key(batch, collection, video_id)`), and
`scripts/operations/r2_reng.py` applies it. Its `build_plan()` calls only
`plan_videos_to_canonical()`, so **videos moved and nothing else did** —
keyframes, objects and map-keyframes keep the organizer's shape.

Nobody taught the reading side. `AicDatasetRegistry` — the module all three
tracks use to find assets — knew two layouts and not the new one. Probed with
all three built in a temp tree:

```text
organizer  Videos_*/video/          -> found
R2 mirror  <batch>/videos/<coll>/   -> NOT FOUND
normalized videos/                  -> found
```

`mvr/enrichment/ocr/scene_text.py` carried its **own second copy** of the same
rules and had the same gap — and its caller only warns on a miss, so an OCR
run against a mirrored corpus would report success having read zero frames.

This is not why `E:\AIC26_data` has no videos: that download never included
them, and this machine has no `.env`, so nothing was ever pulled from R2. The
divergence is real but latent — it fires the first time anyone mirrors the
bucket.

### Fixed

- `AicDatasetRegistry.RAW_LAYOUT` is now a list of directory globs per asset
  kind rather than a single `(outer, inner)` pair, and videos gained
  `*/videos/*`. Normalized still wins, then the organizer shape, then the
  mirror.
- `resolve_video_path()` in `scene_text.py` keeps only its own flat
  convention and delegates everything else to the registry. The duplicate copy
  of the layout rules is gone, which is what let the two drift apart.
- `GET /videos/{filename}` is now a route ahead of the static mount, mirroring
  the keyframe one: mounted directory first, then the registry.
- `_frame_asset_urls` took a `video_dir` it never used as a directory, only as
  a "can anything serve this?" flag, and withheld the URL whenever
  `MVR_VIDEO_DIR` was unset. It now takes `video_available`, true when either
  the mount or the registry can answer, so a link that would resolve is no
  longer hidden.

## 8. Verification

One total, for the finished branch. Earlier drafts quoted 1091 and then 1110 as
the work progressed; those are gone rather than left where a reader could
mistake one for the current figure.

Unit tests, from a worktree checked out on this branch, with the imported
source confirmed to be that worktree's own (`mvr resolved to:
<worktree>\src\mvr\__init__.py`) rather than another branch's:

```text
pytest tests/unit/   ->  1121 passed, 3 skipped in 47.94s
```

Live, same server and corpus, after every change in this report:

```text
TRAKE  L27_V015 frame=5674 ts=00:03:46 thumb=/keyframes/L27_V015/140.jpg
KIS    L23_V019 frame=98   ts=00:00:03 thumb=/keyframes/L23_V019/002.jpg

/keyframes/L27_V015/140.jpg  200  163623 bytes   Keyframes_L27
/keyframes/L21_V001/001.jpg  200  135232 bytes   Keyframes_L21
/keyframes/L30_V031/010.jpg  200  260761 bytes   Keyframes_L30
/videos/L27_V015.mp4         404
```

Both tracks return exactly what they returned before the refactor, so renaming
the parameter through the KIS, Q&A and TRAKE result paths changed no output.

That `404` is the correct answer, not a failure: there is no video file
anywhere on this machine. **The video route and the two new video layouts are
proven by tests only.** No real `.mp4` has been served, and this report does
not claim otherwise.

## 9. Still open

- `/search/semantic` still resolves images through `_servable_keyframe_url`
  against the single mounted directory. The new routes cover what the UI asks
  for; that older endpoint was not touched.
- `RAW_LAYOUT` on this branch has no `objects` entry. That fix is commit
  `70d117f` on `feat/official-submission-format`, not cherry-picked here, so
  Q&A object detections still resolve only under the normalized layout on this
  branch.
- Whether the R2 bucket has actually been reorganized is unverified: no
  credentials exist on this machine, and none were sought.

## 10. Branch state

- `feat/backend-ui-api` is not merged into `main` and was 6 commits behind it.
- This branch rebases it onto `712274d`, adds the fixes above, and borrows
  `1e9a8d4` from `feat/official-submission-format`.
- `feat/backend-ui-api` belongs to another author; nothing was pushed to it.
