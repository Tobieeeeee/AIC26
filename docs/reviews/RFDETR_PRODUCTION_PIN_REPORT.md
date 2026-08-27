# RF-DETR production pin report

- Date: 2026-08-24
- Branch: `fix/rfdetr-production-pin`
- Base: `a284b00` (`fix/milvus-production-runtime`)
- Commit: `fix: pin production RF-DETR detector` (the commit containing this report)
- Merge status: not merged

## Verdict

RF-DETR Large is the production detector. The runtime pin blocker is cleared;
full Batch1 object extraction remains deferred.

```text
RFDETR_LARGE_PRODUCTION_PINNED
RFDETR_CHECKPOINT_INTEGRITY_VERIFIED
RFDETR_REAL_CUDA_SMOKE_PASS
RFDETR_DATA02_TASK05_COMPATIBLE
RFDETR_RUNTIME_PIN_PENDING_CLEARED
FULL_OBJECT_EXTRACTION_DEFERRED
```

## Upstream and package evidence

The current stable package was resolved from PyPI and installed into the
project environment as `rfdetr==1.9.3` (released 2026-08-17). The corresponding
upstream tag resolves to revision
`3f7748b1783533bc2e62804ca5975fe9b9cfc041`. Package and selected model license:
Apache-2.0.

Official sources:

- package: <https://pypi.org/project/rfdetr/1.9.3/>
- tag: <https://github.com/roboflow/rf-detr/tree/1.9.3>
- Large config: <https://github.com/roboflow/rf-detr/blob/1.9.3/src/rfdetr/config.py>
- checkpoint registry: <https://github.com/roboflow/rf-detr/blob/1.9.3/src/rfdetr/assets/model_weights.py>

The released `RFDETRLarge` config uses 704 x 704 input, 90 internal class
slots, and `rf-detr-large-2026.pth`. The package exposes the actual 80-name MS
COCO 2017 mapping with sparse COCO category IDs. The upstream documentation
marks `rf-detr-large.pth` as the deprecated/legacy Large release superseded by
the 2026 checkpoint.

Large loaded, fit CUDA batch 1, and passed all real gates, so Medium and Small
were not attempted. Base, segmentation, keypoint preview, XLarge/2XLarge Plus,
and preview detection models were never candidates or loaded.

## Exact production lock

| Field | Value |
|---|---|
| Provider | `rfdetr` |
| Variant / class | `large` / `RFDETRLarge` |
| Package | `rfdetr==1.9.3` |
| Revision | `3f7748b1783533bc2e62804ca5975fe9b9cfc041` |
| Checkpoint | `rf-detr-large-2026.pth` |
| Official source | `https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth` |
| Exact bytes | `135954129` |
| SHA-256 | `0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38` |
| Upstream MD5 | `5cb72153541cbcb9aa6efa26222acc75` (matched) |
| License | Apache-2.0 |
| Vocabulary | MS COCO 2017, 80 labels, sparse category IDs in 90 slots |
| Resolution | 704 x 704 |
| Threshold | 0.4 (existing canonical value; not tuned) |
| NMS | none; RF-DETR set prediction |
| Device / batch | CUDA / 1 |

The full lock, including all 80 class names, is in
`configs/rfdetr_model.lock.json`. Checkpoint files and caches remain ignored and
were not staged.

## Runtime implementation

`src/mvr/enrichment/objects/rfdetr.py` is now the only implementation stack.
It:

- accepts only the exact production model identity;
- requires an explicit local path or `MVR_RFDETR_CHECKPOINT`;
- verifies filename, exact byte count, and SHA-256 before package/model load;
- verifies the installed package version;
- loads lazily once and reuses the same model;
- never downloads at runtime and never falls back to another detector/model;
- normalizes and deterministically sorts results;
- preserves `score`/`bbox` for existing Task04 consumers while emitting the
  canonical flat fields.

The former `rf_detr.py` is now a compatibility import only. Faster R-CNN and
YOLO are explicit reference/legacy backends, not available production
fallbacks.

## Canonical schema and identity

Every production detection row has:

```text
keyframe_id, video_id, frame_id, frame_idx,
label, class_id, confidence, x1, y1, x2, y2
```

The real fixture proved `keyframe_id` and `video_id` round-trip exactly and
`frame_id == frame_idx` from the canonical mapping. Identity never derives
from array position.

Per-keyframe Task05A envelopes are written under `objects/`; flat DATA-02 rows
are separately staged under `objects_staging/`, preventing Task05A recursive
loading from seeing duplicate artifacts.

## Real CUDA smoke and repeatability

Input: 20 existing representative keyframes, one each from 20 different
videos in `artifacts/ingest_public_test`; no generated or mocked images.

| Measurement | Result |
|---|---:|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Driver | 572.16 |
| PyTorch / CUDA | `2.5.1+cu124` / 12.4 |
| Batch size | 1 |
| Images / videos | 20 / 20 |
| Detection rows | 60 |
| Cold verified load | 81.005 s |
| First pass | 7.198 s (2.779 images/s) |
| Repeated warm pass | 2.386 s (8.383 images/s) |
| Peak CUDA allocated | 243,598,848 bytes |
| Peak CUDA reserved | 274,726,912 bytes |

The cold measurement includes SHA-256 verification, first import, and model
construction in a fresh pytest process. It is reported separately from
inference. No exact full-Batch runtime is extrapolated.

The same 20 identities were inferred twice. Detection count, `class_id`, and
label matched; confidence matched within absolute `1e-5` and every box
coordinate within absolute `1e-3`. Adapter rounding is confidence 6 decimals
and coordinates 3 decimals. The integration gate passed.

## DATA-02, Task05A, and Elasticsearch compatibility

The real run completed:

```text
RF-DETR rows
  -> per-keyframe staging JSON
  -> canonical object schema validation
  -> 60-row objects Parquet
  -> validated atomic promotion
  -> Polars lazy scan
  -> canonical mapping join by keyframe_id
```

The Parquet columns were exactly
`keyframe_id,video_id,frame_id,frame_idx,label,class_id,confidence,x1,y1,x2,y2`.
All joined `video_id`/frame identities matched. Consolidation and promotion now
use `video_id,keyframe_id,class_id,x1,y1,x2,y2` as object-row identity so
multiple detections from one keyframe and repeated keyframe IDs across videos
are retained.

The same real envelopes then built 20 Task05A documents. Object label/count
projection applied no second threshold and no NMS. Counts populated the
existing Elasticsearch `objects.<COCO label>` range-filter representation and
the legacy SearchDocument object-count field successfully. No object-filter
mathematics changed.

## Verification

- focused RF-DETR/schema/DATA-02/Task05A/Elasticsearch suite: 103 passed;
- API, object, hybrid, CLI regressions: 72 passed;
- real CUDA integration gate: 1 passed (one Polars deprecation warning);
- broader safe unit suite: 1,090 passed, 1 unrelated pre-existing failure;
- Python compile check: passed;
- `git diff --check`: passed;
- forbidden artifact/staging audit: passed.

The broader failure is
`tests/unit/indexing/test_task05a_image_path.py::test_the_pipeline_emits_the_image_path_from_the_mapping`:
it imports `tests.unit.indexing.test_task05a`, but `tests` is not a package.
This same unrelated collection-path issue predates TASK-FIX-02 and is also
recorded in existing progress notes.

Installing upstream `rfdetr==1.9.3` also exposes pre-existing optional-extra
environment conflicts in `pip check` (the large perception stack is partially
installed, OpenCLIP/timm constraints differ, and old `torchmetrics` requires
an older `pyDeprecate`). They did not affect RF-DETR loading, the real CUDA
gate, or 1,090 passing unit tests. This task did not mutate unrelated model
stacks to conceal those conflicts.

## Documentation and Git hygiene

Canonical configs, the model lock, shared architecture, RF-DETR runbook, CLI
status, Q&A object comments, progress, and reuse mapping now name RF-DETR Large
and COCO. Historical DATA-02/readiness reports have explicit superseded-status
notes; their original findings remain audit evidence. OpenImages references
that remain describe organizer-provided historical artifacts, not production.

The final commit is pushed to `origin/fix/rfdetr-production-pin` with upstream
tracking and is not merged. No checkpoint, cache, generated JSON/Parquet,
artifact, credential, Milvus/Elasticsearch volume, `run_canonical.py`, or
`aic26_docker_rules_for_teammates.md` is included.

## Remaining work

- Full object extraction / full Batch1: deferred by task scope.
- Full-corpus runtime and quality estimates: not claimed.
- Resolve the repository's unrelated optional-extra dependency conflicts in a
  separate environment/dependency task if one combined environment is
  required.
