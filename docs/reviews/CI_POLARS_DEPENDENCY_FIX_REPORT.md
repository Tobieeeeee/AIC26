# CI Polars Dependency Contract Fix Report

Date: 2026-08-27

Branch: `fix/ci-polars-dependency-contract`

Base: `origin/main` at `45758f14306a55454ce69815799027a29083e74f`

Implementation commit: `72d5d1a0ccbd3a0770ae6c560e4a1c60305bce13` (`ci: align Polars dependency contract`)

## Scope and prerequisite

The branch is based on the requested `origin/main`. Because the already-fixed Task05A test-import commit had not yet reached `origin/main`, the branch also includes its existing commit `c8f2c84a2eff5d8f5492729b3b2e609d996c298c`. No further Task05A changes were made here.

This task changed only the CI dependency contract and the import boundary of the opt-in RF-DETR integration smoke test. It did not change Milvus, RF-DETR, TRAKE, retrieval, or other production algorithms.

## Root cause

The failure had two independent parts, so the root-cause classification is **both packaging and test import boundary**.

1. **Packaging:** Data-02 operations use Polars for Parquet consolidation, validation/promotion, scans, joins, and Task05 document preparation. However, `pyproject.toml` exposed no installable extra for that canonical runtime dependency, and CI installed only `dev`, `api`, and `semantic` extras. The repository readiness audit records Polars 1.43.2 as the verified Data-02 version.
2. **Test import boundary:** `tests/integration/test_rfdetr_production_runtime.py` is an opt-in CUDA/live-data smoke test, but it imported Polars and Data-02 operations modules while the test module was being collected. Pytest therefore raised `ModuleNotFoundError` before it could apply the existing runtime `skipif` gate.

The integration test genuinely exercises Data-02 when enabled, so merely deferring the imports would have left the documented dependency contract incomplete. Conversely, adding the package alone would have retained a fragile collection boundary for environments that intentionally do not install the optional operations runtime. Both defects were fixed minimally.

## Changes

- Added `operations = ["polars==1.43.2"]` to `project.optional-dependencies`.
- Updated GitHub Actions to install `.[dev,api,semantic,operations]`.
- Moved the Polars and Data-02 operations imports into the opt-in RF-DETR test function, behind its existing environment/runtime gate.
- Preserved the RF-DETR test body and assertions unchanged; no `importorskip`, new skip, xfail, deletion, or test weakening was introduced.
- Made no production-code changes.

## Verification

### Collection boundary without Polars

The repository's default Python 3.9 environment did not have Polars installed. In that environment:

```text
python -m pytest -q tests/integration/test_rfdetr_production_runtime.py
1 skipped in 2.31s
```

This proves the opt-in test now collects and reaches its existing runtime gate without requiring an optional heavy import first.

### Fresh isolated environment

Created a fresh Python 3.12.10 virtual environment outside the repository. Polars was absent before installation. Installed only the documented CI extras from this checkout:

```text
pip install -e .[dev,api,semantic,operations]
```

Results:

- Resolver installed `polars==1.43.2` and `polars-runtime-32==1.43.2`.
- Installed project metadata contains `polars==1.43.2; extra == "operations"`.
- `python -m pip check`: `No broken requirements found.`
- RF-DETR live integration test: `1 skipped in 6.69s` (expected because live checkpoint/mapping variables were not supplied).
- Task05A image-path regression: `5 passed in 0.17s`.
- Data-02/Polars-focused selection (`tests/unit/test_parquet_promote.py`, `tests/unit/operations`, `tests/unit/test_milvus_schema.py`): `56 passed in 2.47s`.
- Unit suite: `1087 passed, 4 skipped in 18.77s`.
- Exact CI pytest selection, `python -m pytest -q`: `1089 passed, 19 skipped in 12.74s`.
- `python -m compileall -q src tests`: PASS.
- `git diff --check`: PASS.

No full Batch1 run or model download was performed.

### GitHub Actions

The exact workflow ran on Ubuntu with Python 3.11, installed `.[dev,api,semantic,operations]`, and executed `python -m pytest -q` successfully:

- Run: [CI 33043668448](https://github.com/Tobieeeeee/AIC26/actions/runs/33043668448)
- Head: `72d5d1a0ccbd3a0770ae6c560e4a1c60305bce13`
- Result: SUCCESS
- Tests: `1089 passed, 19 skipped in 7.45s`

The implementation commit was pushed normally to `origin/fix/ci-polars-dependency-contract`; no force-push was used.

## Verdict

```text
POLARS_DEPENDENCY_CONTRACT_ALIGNED
RFDETR_INTEGRATION_COLLECTION_FIXED
FRESH_ENVIRONMENT_VERIFIED
CI_EQUIVALENT_TESTS_PASS
NO_RETRIEVAL_BEHAVIOR_CHANGED
READY_TO_MERGE_CI_FIX_02
```
