# CI Task05A Image Path Import Fix Report

Date: 2026-08-27

## Scope and Git state

- Branch: `fix/ci-task05a-image-path-test`.
- Base commit: `45758f14306a55454ce69815799027a29083e74f`
  (`origin/main` after the production-chain merge).
- Implementation commit: `c8f2c84a2eff5d8f5492729b3b2e609d996c298c`.
- Implementation commit message: `test: remove fragile cross-test import`.
- Remote branch: `origin/fix/ci-task05a-image-path-test`.
- Implementation push: PASS; normal upstream push, no force.

The work was performed in a clean worktree based exactly on the latest fetched
`origin/main`. The unrelated modified and untracked files in the original
RF-DETR worktree were not touched.

## Original failure and root cause

Observed CI error:

```text
ModuleNotFoundError: No module named 'tests.unit'
```

The offending code in
`tests/unit/indexing/test_task05a_image_path.py` was:

```python
from tests.unit.indexing.test_task05a import _make_features
```

This was a fragile package-style import from one test module into another.
The repository does not define `tests` as a supported import package, and its
pytest configuration exposes `src`, not the test tree, on `pythonpath`.
Whether the import worked therefore depended on the runner's import/path
environment.

The exact targeted command happened to pass before the edit on this Windows
machine because its local import environment resolved the test namespace. That
did not invalidate the CI failure or make the cross-test import portable.

## Chosen fix

`_make_features` in `test_task05a.py` is a large module-specific fixture that
creates two mappings plus CLIP, OCR, ASR, and object artifacts for many Task05A
behaviors. The image-path regression needs none of that setup; it needs only:

- one canonical mapping row containing `image_path`;
- an existing empty feature root;
- the real `build_index_documents` call;
- the unchanged assertion against the emitted document.

A small local `_make_image_path_fixture` was therefore added to
`test_task05a_image_path.py`. It writes one canonical CSV row and returns the
empty feature root and mapping path. The test still runs the real Task05A
builder end to end and still asserts exactly:

```python
assert document["image_path"] == "keyframes/V1/V1_S0001_K01.jpg"
```

No assertion, builder call, output path, or image-path value was weakened.

No `tests/__init__.py` package marker, `sys.path` mutation, `PYTHONPATH` hack,
dynamic import, relative test import, skip, xfail, or mock replacement was
introduced. A repository-wide search after the edit found no remaining
`from tests...` or `import tests...` cross-test imports.

## Verification

### Targeted test

Command:

```text
python -m pytest -q tests/unit/indexing/test_task05a_image_path.py
```

Result: `5 passed in 0.17s`, exit code `0`.

The local `python` executable is Anaconda Python 3.9.12. After pytest returned
success it printed an unrelated Windows `atexit` permission traceback while
resolving an old pytest temporary directory. This did not change the command's
exit code or test result and did not occur under Python 3.12.

### Indexing unit suite

Command:

```text
python -m pytest -q tests/unit/indexing
```

Result: `60 passed, 7 skipped in 2.39s`, exit code `0`.

### Broader unit suite

Command:

```text
py -3.12 -m pytest -q tests/unit
```

Result: `1087 passed, 4 skipped in 11.55s`, exit code `0`.

### Full CI test selection on the supported local runtime

Command:

```text
py -3.12 -m pytest -q
```

Result: `1089 passed, 19 skipped in 18.81s`, exit code `0`.

Python 3.11 is not installed locally. Running the same command through the
machine's default Python 3.9 was not CI-equivalent: it stopped on unsupported
runtime annotation evaluation and a missing local `polars` dependency.

### Exact GitHub Actions workflow

The checked-in workflow uses Python 3.11, installs
`.[dev,api,semantic]`, and runs:

```text
python -m pytest -q
```

Push-triggered run:

- run: `33042129624`;
- implementation SHA: `c8f2c84a2eff5d8f5492729b3b2e609d996c298c`;
- URL: `https://github.com/Tobieeeeee/AIC26/actions/runs/33042129624`;
- Task05A import outcome: PASS — the original `tests.unit` collection error is
  absent;
- overall workflow outcome: FAIL due to a separate collection error:
  `ModuleNotFoundError: No module named 'polars'` in
  `tests/integration/test_rfdetr_production_runtime.py`.

The failure occurs after the requested Task05A repair and belongs to the
explicitly excluded RF-DETR/dependency scope. It was not modified or suppressed
in this task.

### Static verification

- `py -3.12 -m compileall -q src tests`: PASS.
- `git diff --check`: PASS.
- Production code changed: NO.
- Test assertion behavior preserved: YES.
- Full Batch1 executed: NO.

## Files changed

Implementation commit:

- `tests/unit/indexing/test_task05a_image_path.py`.

Documentation follow-up:

- `docs/reviews/CI_TASK05A_IMAGE_PATH_FIX_REPORT.md`.

No TRAKE, Milvus, RF-DETR, OCR, ASR, frontend, configuration, dependency, or
production source file was changed.

## Remaining failure

The requested Task05A cross-test import is fixed, targeted/indexing/unit/local
full suites pass, and original intent is preserved. However, the repository's
exact GitHub Actions workflow is not globally green because the newly exposed
RF-DETR integration test imports `polars` before it can be skipped while the CI
extra set does not install `polars`.

That remaining failure requires a separate explicitly scoped task. This report
does not claim `CI_EQUIVALENT_TESTS_PASS` or `READY_FOR_MAIN_CI_RERUN` while the
exact workflow still fails.

## Task-specific verdict

```text
TASK05A_IMAGE_PATH_TEST_IMPORT_FIXED
ORIGINAL_TEST_INTENT_PRESERVED
INDEXING_TESTS_PASS
CI_EQUIVALENT_TESTS_BLOCKED_BY_UNRELATED_RFDETR_POLARS_IMPORT
NO_PRODUCTION_BEHAVIOR_CHANGED
MAIN_CI_RERUN_BLOCKED_BY_SEPARATE_RFDETR_DEPENDENCY_FAILURE
```
