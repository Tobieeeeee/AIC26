# Production Chain Merge Report

Date: 2026-08-27

## Result

- Canonical integration target: `main`.
- Selection basis: `origin/HEAD` points to `origin/main`, the repository Git
  workflow defines `main` as the stable shared integration branch, and no
  `origin/integration` branch exists.
- Pre-merge target HEAD: `a779ad1a6581f408fe7347b1e459d52f54af9836`.
- Production-chain tip merged: `68227f96ce2b74f5113bed8ca917b656697964a7`
  (`origin/fix/rfdetr-production-pin`).
- Post-merge HEAD: `987501aabea3a05f095bf772469470879d3a9a24`.
- Final branch HEAD: the report-only commit containing this file. Its exact SHA
  is verified after the normal push and recorded in the task handoff because a
  commit cannot embed its own SHA.

The merge was performed in a separate clean worktree. Pre-existing modified
and untracked files in the original RF-DETR worktree were not altered, staged,
removed, or committed.

## Ancestry verification

The requested remote-ref command returned exit code `128`:

```text
git merge-base --is-ancestor \
  origin/fix/milvus-production-runtime \
  origin/fix/rfdetr-production-pin

fatal: Not a valid object name origin/fix/milvus-production-runtime
```

This was not a history divergence. The Milvus and audit remote branches had
already been deleted after the prior consolidation audit. Their local refs
were deliberately retained at the previously verified immutable tips:

- `fix/milvus-production-runtime`:
  `a284b00cb7bdac0014f139f64c0d54c679e74b9d`;
- `audit/system-full-readiness`:
  `67b4684799ed0e8ded38c5d0fce7effc5de09510`.

Equivalent ancestry checks against those retained refs returned exit code `0`:

```text
git merge-base --is-ancestor \
  fix/milvus-production-runtime \
  origin/fix/rfdetr-production-pin

git merge-base --is-ancestor \
  audit/system-full-readiness \
  origin/fix/rfdetr-production-pin
```

Post-merge ancestry checks also returned exit code `0` for Milvus, RF-DETR,
and audit tips against `main`.

## Merge decision and provenance

`origin/main` and `origin/fix/rfdetr-production-pin` diverged at
`d47ade06939698e4dd01b8ed544602bf0eb0a88b`. The commits unique to the
production descendant were:

```text
67b4684 docs: audit full system readiness including trake and qa
f6bb249 fix: make Milvus the production semantic backend
a284b00 docs: record Milvus production runtime evidence
9f9ca40 fix: pin production RF-DETR detector
68227f9 docs: record git branch consolidation
```

The commits unique to `main` were two existing PR merge commits with no tree
delta absent from the production descendant. `git merge-tree --write-tree`
completed without conflicts. A normal `--no-ff` merge was used; no squash,
rebase, cherry-pick, reset, force-push, or blanket conflict strategy was used.

The resulting merge commit has exactly these parents:

```text
a779ad1a6581f408fe7347b1e459d52f54af9836
68227f96ce2b74f5113bed8ca917b656697964a7
```

The post-merge tree
`a1d12bfba831df9ad9fad7fc6784f8dd05d9e413` is byte-for-byte identical to
the verified RF-DETR production descendant tree. `git cherry HEAD
origin/fix/rfdetr-production-pin` returned no entries. Milvus was not merged or
cherry-picked separately, so no duplicate Milvus production patch exists.

## Preserved production contracts

Critical production files match the verified RF-DETR descendant exactly:

- `configs/default.yaml`;
- `configs/production.yaml`;
- `configs/rfdetr_model.lock.json`;
- `src/mvr/api/app.py`;
- `src/mvr/retrieval/milvus_semantic.py`;
- `src/mvr/enrichment/objects/rfdetr.py`.

Verified contracts:

- production semantic backend remains `milvus`;
- FAISS remains available only as explicit `faiss_reference` for
  parity/benchmark use;
- object provider remains `rfdetr`;
- model class remains `RFDETRLarge`;
- package pin remains `rfdetr==1.9.3`;
- checkpoint remains `rf-detr-large-2026.pth` with the committed lock file;
- default and production YAML configurations remain aligned.

## Audit evidence

The audit report was already contained and updated on the RF-DETR descendant.
The merge preserved all relevant evidence without separately merging the audit
branch:

- `docs/reviews/SYSTEM_FULL_READINESS_AUDIT.md`;
- `docs/reviews/MILVUS_PRODUCTION_RUNTIME_FIX.md`;
- `docs/reviews/RFDETR_PRODUCTION_PIN_REPORT.md`;
- `docs/reviews/GIT_BRANCH_CONSOLIDATION_REPORT.md`.

No audit-only branch merge was created.

## TRAKE boundary

The exact local and remote refs named `fix/trake-milvus-dense-scoring` were
absent before the merge and remained absent afterward. No operation targeted
that unfinished branch: it was not checked out, merged, rebased, deleted, or
pushed. Only the already-reviewed historical TRAKE functionality present in
the existing common ancestry of `main` and RF-DETR remains in the graph.

## Focused verification

The focused suite covered production defaults, Milvus schema/search/failure/
parity/runtime configuration, RF-DETR provider/configuration, API semantic
health/search, and the two opt-in runtime smoke modules.

Result:

```text
150 passed, 3 skipped in 6.22s
```

The three skips are the expected real Milvus/RF-DETR runtime gates because
their external service/assets were not enabled for this Git-only merge. No
focused test failed. Full Batch1 was not run.

- `git diff --check`: PASS.
- `git diff --cached --check`: PASS.
- Merge worktree status before this report: clean, `main` ahead of
  `origin/main` only by the five preserved descendant commits plus one merge
  commit.

## Push and branch cleanup

This merge and report are pushed normally to `origin/main` without force. The
post-push remote SHA is verified in the final task handoff.

Remote branch state:

- already deleted by the prior verified cleanup:
  `fix/milvus-production-runtime`, `audit/system-full-readiness`;
- retained: `main`, `fix/rfdetr-production-pin`;
- `fix/trake-milvus-dense-scoring`: absent before and after this task and not
  touched.

No remote branch was deleted during this merge task.

## Final verdict

```text
PRODUCTION_CHAIN_MERGED
MILVUS_RUNTIME_PRESERVED
RFDETR_PRODUCTION_PIN_PRESERVED
AUDIT_EVIDENCE_PRESERVED
TRAKE_UNFINISHED_BRANCH_NOT_MERGED
READY_TO_RESTART_TASK_FIX_03
```
