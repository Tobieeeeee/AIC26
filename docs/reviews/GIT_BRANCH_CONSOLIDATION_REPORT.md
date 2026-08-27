# Git Branch Consolidation Report

Date: 2026-08-27

## Scope and target

- In-scope remote branches: `audit/system-full-readiness` and
  `fix/milvus-production-runtime`.
- Reference production descendant: `fix/rfdetr-production-pin`.
- Explicitly excluded: `fix/trake-milvus-dense-scoring`.
- Consolidated production branch: `fix/rfdetr-production-pin`.
- Starting HEAD: `9f9ca400ef08524dff4ad32829dedf821ae25789`.
- Final production-content HEAD before this report-only commit:
  `9f9ca400ef08524dff4ad32829dedf821ae25789`.
- Final branch HEAD: the report-only commit containing this file. Its exact SHA
  is verified after the normal push and recorded in the task handoff because a
  commit cannot embed its own SHA.

No production code, configuration, test, model, index, dataset, or runtime
artifact was changed by this consolidation task.

## Initial safety inspection

`git fetch --all --prune` completed successfully before branch analysis. The
current branch was already `fix/rfdetr-production-pin`, tracking
`origin/fix/rfdetr-production-pin` at `9f9ca40`.

The main worktree contained pre-existing unrelated work:

- modified: `docs/CODEX_PROGRESS.md`;
- untracked: local documents, experiment scripts, benchmark tests, frontend
  files, and ignored/runtime material under `artifacts/` and `data/`.

Those files were not altered, removed, staged, or committed. The worktree list
showed no checkout of either deleted remote branch. The only in-scope checked
out branch was the retained consolidation target.

## Commit and file classification

| Branch / comparison | Unique commits | Classification | Already in descendant? |
| --- | --- | --- | --- |
| `audit/system-full-readiness` vs `origin/main` | `67b4684` | `AUDIT_DOC_ONLY` | Yes |
| `audit/system-full-readiness` vs `fix/rfdetr-production-pin` | none | already subsumed | Yes |
| `fix/milvus-production-runtime` vs audit | `f6bb249`, `a284b00` | production/config/tests/docs plus evidence | Yes |
| `fix/milvus-production-runtime` vs `fix/rfdetr-production-pin` | none | already subsumed | Yes |
| `fix/rfdetr-production-pin` vs Milvus | `9f9ca40` | RF-DETR production/config/tests/docs | target tip |

### `audit/system-full-readiness`

The branch had one commit not in `origin/main`:

- `67b4684799ed0e8ded38c5d0fce7effc5de09510` —
  `docs: audit full system readiness including trake and qa`.

That commit added only:

- `docs/reviews/SYSTEM_FULL_READINESS_AUDIT.md`.

Classification: `AUDIT_DOC_ONLY`. It contained no production code, config, or
tests. The audit report's blob on the audit and Milvus tips was
`ddf708bb989ecd5026f323869a0bf97495a984ab`. The consolidated RF-DETR branch
contains the same report, updated by four lines in `9f9ca40`; its blob is
`778cd263be70929af26ea3a419f11ed164ef3042`. Audit evidence was therefore
already preserved and updated on the chosen production descendant. No merge,
cherry-pick, restore, or additional preservation commit was needed.

Remote cleanup decision: deleted after ancestry, content, worktree, and GitHub
PR/review checks passed. No open, closed, or merged PR used this branch as its
head, and no open PR title/body/comment referenced it.

### `fix/milvus-production-runtime`

The branch added these commits after the audit tip:

- `f6bb2497730f0d4c0862a270d7d639e0275eb6ad` —
  `fix: make Milvus the production semantic backend`;
- `a284b00cb7bdac0014f139f64c0d54c679e74b9d` —
  `docs: record Milvus production runtime evidence`.

`f6bb249` changed production configuration, Milvus Docker documentation and
compose configuration, package metadata, API schemas/runtime wiring, Milvus
client/backend and semantic retrieval code, KIS/Q&A integration, reuse
documentation, and added focused Milvus integration/API tests. The exact files
were:

- `configs/default.yaml`, `configs/production.yaml`;
- `docker/milvus/README.md`, `docker/milvus/docker-compose.milvus.yml`;
- `docs/LAB_REUSE_MAP.md`, `pyproject.toml`;
- `src/mvr/api/app.py`, `src/mvr/api/schemas.py`;
- `src/mvr/indexing/milvus_backend.py`,
  `src/mvr/indexing/milvus_client.py`;
- `src/mvr/retrieval/kis.py`, `src/mvr/retrieval/milvus_semantic.py`,
  `src/mvr/retrieval/qna.py`;
- `tests/integration/test_milvus_production_runtime.py`,
  `tests/unit/api/test_milvus_runtime.py`.

`a284b00` changed `docs/CODEX_PROGRESS.md` and added
`docs/reviews/MILVUS_PRODUCTION_RUNTIME_FIX.md`.

The required command

```text
git merge-base --is-ancestor origin/fix/milvus-production-runtime origin/fix/rfdetr-production-pin
```

returned exit code `0`. The Milvus tip is therefore a direct ancestor of the
RF-DETR tip and is classified `SUBSUMED_BY_FIX_RFDETR_PRODUCTION_PIN`.

Merge/cherry-pick decision: none. `git cherry
origin/fix/rfdetr-production-pin origin/fix/milvus-production-runtime`
returned no entries, so no Milvus patch is missing from the target.

Remote cleanup decision: deleted after reachability, content, worktree, and
GitHub PR/review checks passed. No open, closed, or merged PR used this branch
as its head, and no open PR title/body/comment referenced it.

### `fix/rfdetr-production-pin`

The verified production chain is linear:

```text
67b4684 audit report
  -> f6bb249 Milvus production semantic backend
  -> a284b00 Milvus runtime evidence
  -> 9f9ca40 RF-DETR production pin
```

The range from the audit tip to the RF-DETR tip has no merge commits. Stable
patch IDs for `f6bb249`, `a284b00`, and `9f9ca40` were all distinct. No merge
or cherry-pick was created by this task. The Milvus runtime implementation is
present once in the ancestry, followed by its evidence commit and the RF-DETR
production commit.

### `fix/trake-milvus-dense-scoring`

The exact local and remote refs were absent both before and after cleanup.
Consequently there was no TRAKE branch operation to perform. No checkout,
merge, rebase, commit, push, deletion, or file edit targeted the unfinished
TRAKE work.

## Verification

Focused tracked tests:

```text
py -3.12 -m pytest -q \
  tests/unit/test_milvus_schema.py \
  tests/unit/test_milvus_search.py \
  tests/unit/test_milvus_failure.py \
  tests/unit/test_milvus_parity.py \
  tests/unit/api/test_milvus_runtime.py \
  tests/unit/test_production_defaults.py \
  tests/unit/enrichment/objects/test_rfdetr.py \
  tests/unit/api/test_app.py \
  tests/integration/test_milvus_production_runtime.py \
  tests/integration/test_rfdetr_production_runtime.py
```

Result: `150 passed, 3 skipped in 10.23s`. The skipped cases are opt-in real
Milvus/RF-DETR runtime tests whose service/assets were not enabled for this
Git-only cleanup. Unit, default/config, parity, fail-closed, RF-DETR adapter,
and API health/search coverage passed.

- `git diff --check`: PASS.
- `git diff --cached --check`: PASS.
- No unrelated file was staged.

## Cleanup and push results

Normal, non-force remote deletion result:

```text
- [deleted] audit/system-full-readiness
- [deleted] fix/milvus-production-runtime
```

`git ls-remote --heads` then returned only
`fix/rfdetr-production-pin` among the existing in-scope/reference production
refs, at `9f9ca400ef08524dff4ad32829dedf821ae25789`. Local branch pointers for audit
and Milvus were intentionally retained; this task authorized only optional
remote cleanup.

This report-only commit is pushed normally to
`origin/fix/rfdetr-production-pin` without force, and the post-push remote SHA
is verified in the final task handoff.

Final graph before the report-only commit:

```text
* 9f9ca40 fix: pin production RF-DETR detector
* a284b00 docs: record Milvus production runtime evidence
* f6bb249 fix: make Milvus the production semantic backend
* 67b4684 docs: audit full system readiness including trake and qa
```

## Final verdict

```text
MILVUS_RUNTIME_HISTORY_PRESERVED
AUDIT_EVIDENCE_PRESERVED
NO_DUPLICATE_PRODUCTION_MERGE
TRAKE_BRANCH_LEFT_UNTOUCHED
BRANCH_CLEANUP_COMPLETE
DELETED_REMOTE_BRANCHES=fix/milvus-production-runtime,audit/system-full-readiness
RETAINED_REMOTE_BRANCHES=fix/rfdetr-production-pin
CONSOLIDATED_PRODUCTION_BRANCH=fix/rfdetr-production-pin
```
