# Data-01 Corpus Readiness Audit Report

## Final verdict

**Implementation verdict: `PASS_WITH_ENVIRONMENT_LIMITATIONS`**

The read-only audit implementation, CLI, machine-readable output bundle,
synthetic-fixture tests, metadata exclusion rules, provider discovery, and
status aggregation are complete. The implementation does not require a live
backend, GPU, model weights, OCR/ASR runtimes, Elasticsearch, or a full
corpus.

**Fasttest corpus status: `BLOCKED`**

The available fasttest artifact bundle contains five canonical keyframes and
complete OCR, ASR, object, Task05 processed-document, CLIP, and lexical
manifest branches. It is not a source-video-complete corpus: the canonical
mapping references videos that are not present, and its `image_path` values
point to nested `keyframes/...` paths while the available fasttest JPEGs are
flat under `ingest/`. These are reported as missing/orphan artifacts; the
auditor does not repair or infer the intended path.

**Full-corpus status: `NOT_RUN_DATA_UNAVAILABLE`**

No local directory was found containing a complete canonical mapping,
source-video inventory, keyframe inventory, and enrichment/index roots for a
full-corpus run. The read-only lab sample directory contains videos only and
was not treated as a full corpus.

## Branch and base

- Branch: `feat/data-readiness-audit`
- HEAD: `bd22e273fe0ec44b2faa23b8cb42063b43b27d90`
- Base: `bd22e27 docs: define submission API contract`
- Worktree: `mvr-competition-starter-data-readiness-audit`
- No merge or push was performed.

## Files changed

- `src/mvr/audit/__init__.py`
- `src/mvr/audit/data_readiness.py`
- `scripts/audit/run_data_readiness.py`
- `tests/unit/audit/test_data_readiness.py`
- `.gitignore`
- `docs/reviews/DATA_READINESS_AUDIT_REPORT.md`

Generated runtime output is written under
`reports/data_readiness/<run_id>/` and is ignored by Git. No corpus, model,
embedding, index, or generated fasttest file is part of the change set.

## Audit architecture

The implementation is a stdlib-first, filesystem/manifest-only service:

```text
AuditConfig
  -> explicit/discovered roots
  -> canonical mapping
  -> videos / keyframes / media_info
  -> OCR / ASR / objects
  -> discovered embedding providers
  -> Task05 processed documents
  -> existing index manifests
  -> coverage matrix + issues
  -> summary/report bundle
```

The CLI is:

```powershell
python scripts/audit/run_data_readiness.py `
  --dataset-root "<dataset>" `
  --artifacts-root "<artifacts>" `
  --output-root "reports\data_readiness"
```

Every relevant root can also be supplied explicitly (`--mapping-path`,
`--videos-root`, `--keyframes-root`, `--ocr-root`, `--asr-root`,
`--objects-root`, `--embeddings-root`, `--processed-root`, and
`--indexes-root`). Personal machine paths are not hard-coded.

Each run writes:

```text
manifest.json
summary.json
issues.jsonl
coverage.csv
report.md
```

Exit codes are deterministic:

- `0`: `READY` or `READY_WITH_WARNINGS`;
- `2`: invalid CLI/configuration;
- `3`: `BLOCKED` data invariant;
- `4`: unexpected runtime failure.

## Checks implemented

### Canonical mapping

- captures path, SHA-256, columns, row count, unique `keyframe_id` count, and
  unique `(video_id, frame_idx)` count;
- reports blank identities, invalid/non-finite frame/timestamp values,
  duplicate canonical IDs, and duplicate output pairs;
- preserves the canonical `frame_idx`/timestamp semantics without positional
  joins or repair.

### Videos and keyframes

- inventories video IDs, duplicate stems, extensions, zero-byte/unreadable
  files, size, and missing mapped videos;
- resolves keyframes through the mapping's `image_path`;
- reports expected/present/missing/orphan image files, collisions, zero-byte
  files, and cheap JPEG signature anomalies;
- does not infer a replacement path from a filename when the mapping path is
  absent.

### Media info

- supports observed CSV/JSON media-info records;
- validates duration, FPS, frame count, finite values, duplicate records, and
  canonical frame/timestamp bounds;
- does not fabricate fields absent from the observed schema.

### OCR, ASR, and objects

- validates branch JSON identities against canonical records;
- reports coverage, missing/orphan/duplicate artifacts, malformed JSON/schema,
  identity mismatches, empty results, and non-finite confidence/score values;
- excludes `manifest.json`, `errors.json`, and `shot_summary.json` from
  per-keyframe counts while still validating malformed real keyframe JSON;
- performs no OCR, ASR, detector, NMS, threshold, or second-stage filtering.

### Embedding providers

- discovers provider directories from vector files and vector/embedding
  manifests rather than assuming one global provider;
- records provider/model/checkpoint, manifest hash, mapping path/hash where
  available, count, dtype, normalization, and per-provider dimensions;
- validates finite values, dimensions, mapping/vector counts, duplicate and
  orphan identities, and mapping coverage;
- reads `.npy`/`.npz` through optional NumPy and supports small JSON vector
  records for offline fixtures;
- never hard-codes a global embedding dimension;
- represents absent `beit3`, `metaclip2`, and `perception_encoder` branches as
  `PENDING` unless explicitly listed with `--required-provider`, in which case
  absence is a blocker.

### Task05 processed documents and index manifests

- checks one-document-per-canonical-keyframe identity, duplicates, missing and
  orphan documents, finite timestamps, and indexability flags;
- summarizes lexical/object/vector coverage and provider-independent
  processed-document coverage;
- reads existing index manifests without connecting to or mutating a live
  service;
- records index name/provider/count/source mapping hash/timestamp and reports
  count/hash mismatches where comparable.

### Coverage and disk usage

`coverage.csv` contains one row per canonical keyframe with explicit
`present`/`missing` states for keyframe, OCR, ASR, objects, Task05 documents,
and each discovered/pending provider. Section summaries include filesystem
file counts, byte totals, extension distributions, and pending providers.

## Fasttest audit result

Command executed against the existing read-only bundle:

```powershell
python scripts/audit/run_data_readiness.py `
  --artifacts-root "<repository>\artifacts\fasttest" `
  --output-root "reports\data_readiness" `
  --run-id fasttest-final
```

Result:

- CLI exit code: `3`;
- overall status: `BLOCKED`;
- canonical keyframes: `5`;
- errors: `7`;
- warnings: `8`;
- videos: `BLOCKED` (two mapped source videos absent);
- keyframes: `BLOCKED` (five mapping paths missing; five flat JPEGs orphaned);
- media info: `READY`;
- OCR: `READY`, 100% coverage;
- ASR: `READY`, 100% coverage, metadata files excluded;
- objects: `READY`, 100% coverage;
- discovered CLIP provider: `READY`, 5 vectors, dimension `512`, finite;
- BEiT-3: `PENDING`;
- MetaCLIP-2: `PENDING`;
- Perception Encoder: `PENDING`;
- Task05 processed documents: `READY`, 5 documents;
- lexical index manifest: `READY`, 5 indexed documents.

SHA-256 checks of the fasttest mapping and representative manifests before and
after the audit were unchanged. No generated runtime report was staged.

## Tests and verification

Focused offline audit tests:

```text
6 passed
```

Coverage includes duplicate canonical IDs/output pairs, missing videos and
keyframes, orphan keyframes/OCR, ASR metadata exclusion, malformed ASR JSON,
invalid object schema/counts, embedding NaN/Inf/dimension/count mismatches,
required-provider pending behavior, processed-document duplicates,
indexability, readiness aggregation, and report serialization.

Full lightweight repository suite:

```text
294 passed, 14 skipped
```

The skips are pre-existing optional runtime gates; no model download, GPU,
Elasticsearch, OCR, ASR, retrieval, or index build was started by this task.

`python -m compileall -q src/mvr/audit scripts/audit` passed.

## Read-only and scope confirmation

- No dataset/artifact file was modified, repaired, regenerated, renamed, or
  moved.
- No missing data or model was downloaded.
- No embedding extraction or provider code was changed.
- No OCR, ASR, object detection, retrieval, or index build was started.
- No Elasticsearch service was contacted or mutated.
- No Task06A/Task06B implementation was started.
- Generated reports are ignored under `reports/data_readiness/`.
- Only the six listed source/test/documentation files are intended to change;
  no unrelated files are staged.

## Remaining blockers and warnings

Blockers observed in the available fasttest bundle:

1. canonical mapping references source videos absent from the artifact bundle;
2. canonical keyframe `image_path` values do not match the flat fasttest JPEG
   layout.

Warnings/pending work:

1. BEiT-3, MetaCLIP-2, and Perception Encoder artifacts are not present and are
   intentionally `PENDING`, not corruption;
2. full-corpus readiness is unverified because no complete local corpus roots
   are available.

These findings describe the audited environment, not a claim that the audit
implementation is defective.
