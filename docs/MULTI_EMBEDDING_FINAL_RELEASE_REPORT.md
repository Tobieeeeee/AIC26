# Multi-embedding final release report

Date: 2026-08-21  
Branch: `feat/rebuild-embedding-models`

## Release decision

- Code and structural release gate: **PASS**.
- Operational/competition release gate: **BLOCKED**.
- No provider-quality or three-provider improvement claim is made because
  representative ground truth has not been evaluated.

The operational block is intentional: real-weight five-keyframe inference has
not run for all providers, and the approval-gated full-corpus build has not run.

## Architecture and regression gate

The automated audit verifies the canonical
`features/embeddings/<model_id>/` layout, provider-specific dimensions,
provider-matched query/index routing, provider-specific Task05A/Task05C
indexability, `keyframe_id` joins, metadata preservation, rank-only RRF, fusion
separation, and unchanged Task05B `ocr_text`/`asr_text` fields.

Git hygiene reports no tracked or staged weights, arrays, vector index data,
archives, videos, logs, oversized files, or suspicious secret filenames.

## Provider lock matrix

| Provider | Exact checkpoint | Model revision | Source revision | Dimension | Contract |
|---|---|---|---|---:|---|
| BEiT-3 | `beit3_base_itc_patch16_224.pth` | No immutable upstream checkpoint revision; artifact SHA256 is mandatory | `69961945a2fd283f2beed74d1f3c543265c9930c` | 768 | normalized `float32`, inner product |
| MetaCLIP-2 | `facebook/metaclip-2-worldwide-b16` | `8cc8591f32b43a809f3325192d3b77aa56968ddb` | `f47f7841f6a91cc5676729a3d125519393d87d1e` | 512 | normalized `float32`, inner product |
| Perception Encoder | `facebook/PE-Core-L14-336:PE-Core-L14-336.pt` | `ee1fdb467ab42497fa2e60e94a79e9d7383c1cbd` | `65a4a17a587b6a89eaa0dc08f231c2cb1c285f9f` | 1024 | normalized `float32`, inner product |

MetaCLIP-2 is CC-BY-NC-4.0. Complete access, dependency, preprocessing, and
license locks are in `configs/embedding_providers.json`.

## Task-to-commit mapping

| Task | Commit |
|---|---|
| Task 04 — artifact schema/discovery | `24bcc59` |
| Task 05 — provider specification lock | `652f404` |
| Task 06 — BEiT-3 offline extraction | `0a237c7` |
| Task 07 — MetaCLIP-2 offline extraction | `373a5df` |
| Task 08 — Perception Encoder offline extraction | `bf52fa0` |
| Task 09 — Task05A multi-provider validation | `71c7e57` |
| Task 10 — provider-specific indexes | `cd74eca` |
| Task 11 — matching online text encoders | `6ec784e` |
| Task 12 — independent semantic retrieval | `d8b3a09` |
| Task 13 — deterministic model-level RRF | `064b254` |
| Task 14 — fasttest and seven ablations | `1c0850e` |
| Task 15 — approval-gated full-corpus workflow | `6fdf92b` |
| Task 16 — final release gates | recorded after this report is committed |

No remote push or PR was created.

## Reproducible commands

```powershell
# Five-keyframe structural E2E
python -m pytest -q tests/integration/test_multi_embedding_fasttest_e2e.py

# Seven required ablations
python scripts/verification/run_multi_embedding_ablation.py `
  --config configs/multi_embedding_fasttest.json

# Read-only full-corpus planning gate
python scripts/operations/run_full_corpus.py `
  --config configs/full_corpus_build.json `
  --dry-run

# Final release gates
python scripts/verification/audit_multi_embedding_release.py
python -m pytest -q
git diff --check
```

## Five-keyframe structural counts

| Check | BEiT-3 | MetaCLIP-2 | Perception Encoder |
|---|---:|---:|---:|
| Expected keyframes | 5 | 5 | 5 |
| Valid vectors | 5 | 5 | 5 |
| Unique IDs | 5 | 5 | 5 |
| Missing IDs | 0 | 0 | 0 |
| Extra IDs | 0 | 0 | 0 |
| Duplicate IDs | 0 | 0 | 0 |
| Invalid vectors | 0 | 0 | 0 |
| Indexed vectors | 5 | 5 | 5 |

The fixture produces three independent ranked lists and five canonical fused
identities. These are deterministic injected vectors, not real model outputs.

## Verification results

- Static audit: PASS, all 10 architecture checks true.
- Task 15 read-only L21 dry run: PASS, 29 videos and 7,800 unique expected
  identities, with zero missing/extra/duplicate source identities.
- Full pytest in the CI dependency set: PASS, 429 passed and 9 opt-in or
  real-runtime tests skipped.
- Python compilation and `git diff --check`: PASS.
- Lint, type-check, formatter, and dependency/security scanners: not configured
  in `pyproject.toml` or CI; no substitute result is claimed. The release audit
  supplies static architecture and Git-hygiene checks.
- Extra `pip check` in `mvr-asr:prepush`: FAIL because the image preinstalls
  `torchaudio 2.5.1+cpu`, while the broad semantic extra resolves a newer Torch.
  This mixed image is not the configured CI environment; pytest is separate.

## Remaining skips and blockers

- Real three-provider five-keyframe inference: model assets and isolated
  provider runtimes are not all available locally.
- Full-corpus extraction/indexing/online smoke/benchmark: not launched because
  commands still contain approval-blocking placeholders, the extracted root and
  canonical mapping are absent, and resource estimates are unmeasured.
- Quality evaluation: representative ground truth is unavailable.

Before a full build, resolve every placeholder in
`configs/full_corpus_build.json`, run one real-weight five-keyframe smoke per
provider, record environments and artifact SHA256 values, and obtain explicit
approval with exact targets, resources, artifacts, and affected collections.

## Rollback and rebuild

- Collections are isolated as `<environment>__<dataset>__<model_id>`; never
  reuse one for another model or dimension.
- Default policy is `create`. Existing targets stop the run instead of being
  overwritten. `rebuild` is destructive and requires explicit approval naming
  all affected collections.
- Keep previous manifests/mappings until replacements pass identity audit and
  online smoke. A provider may be disabled independently; it must never fall
  back to another provider's vectors.
- Code commits can be reverted in reverse dependency order without rewriting
  shared Git history.
