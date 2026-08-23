# Ingestion Canonical Freeze

## Decision

CANONICAL INGESTION V1

Shot detector: TransNetV2 @ 0.5

Keyframe selector: periodic midpoint + 12s/8s

Status: FROZEN FOR BATCH1 PRODUCTION

Full Batch1 execution: deferred

## Scope

Production defaults were aligned in the ingestion library, module CLI,
standalone ingestion script, and default YAML configuration. No applicable
runtime or Docker ingestion-default configuration exists in this repository;
the Dockerfiles present are isolated enrichment workers and do not resolve
shot detector or keyframe selector settings.

PySceneDetect, midpoint-only, `hybrid_temporal_content`, and the Meta/PE/BEiT
selector experiments remain available only when explicitly selected. No
experimental strategy was removed.

## Safety and compatibility

- TransNetV2 is the sole default detector and primary detector.
- TransNetV2 threshold is `0.5` at every runnable default surface.
- Missing TransNetV2 dependencies, weights, or runtime failures propagate; no
  PySceneDetect fallback exists.
- The periodic default returns before the hybrid provider-loading branch, so it
  does not initialize MetaCLIP-2. PE and BEiT-3 are not referenced by the
  production ingestion pipeline.
- Periodic frame selection mathematics and canonical identity/mapping semantics
  were not changed.
- Full Batch1 ingestion was not run.

## Evidence basis

The frozen choice follows the previously measured four-video, 99-GT pilot in
`INGEST_MULTI_VIDEO_KEYFRAME_REEVALUATION.md`: periodic covered 98/99 segments
under frozen PySceneDetect shots and 99/99 under frozen TransNetV2 shots, while
content-aware selectors did not provide a consistent detector-independent
improvement. That pilot supports the lowest-risk production freeze; it is not a
claim that periodic is a corpus-wide theoretical optimum.

## Verification

Default-resolution tests cover the CLI, standalone script, library signature,
and YAML profile. A pipeline test runs the default library path with a fake
TransNetV2 detector, verifies the exact periodic selector manifest, and fails
if a MetaCLIP-2 encoder is constructed. Existing tests continue to cover the
explicit hybrid failure path and legacy periodic selection behavior.

Measured verification results:

- focused ingestion, CLI, feature-runner, and provider-isolation tests:
  `93 passed, 1 skipped`;
- full repository suite: `921 passed, 16 skipped, 1 failed`;
- the sole full-suite failure is the pre-existing unrelated
  `tests/unit/indexing/test_task05a_image_path.py` import of
  `tests.unit.indexing.test_task05a` (`ModuleNotFoundError: tests.unit`);
- both CLI help entrypoints loaded successfully from this worktree with
  `PYTHONPATH=src`;
- direct parser/signature inspection resolved CLI, script, and library behavior
  to TransNetV2 `0.5`, periodic, `12.0`, and `8.0`;
- Python compilation and `git diff --check`: PASS.
