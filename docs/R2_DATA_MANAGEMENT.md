# R2 Data Management Guide

Cloudflare R2 is the canonical object storage for the AIC26 MVR pipeline.
This guide covers the object-key layout, how to sync data between R2 and
local filesystems, and how to run the pipeline against R2-backed data from
both a local workstation and a rented GPU machine.

## Object-key layout

All R2 objects follow the canonical layout defined in
`src/mvr/operations/r2_layout.py`:

```
data/
  raw/
    batch{batch_id}/videos/{collection_id}/{video_id}.mp4
  staging/
    {feature}/{run_id}/{filename}
  versions/
    enrichment/{version}/batch={batch_id}/collection={collection_id}/...
    manifest-backups/{activated_at}/dataset_manifest.json
  corpus/
    dataset_manifest.json
    ingest/
      ingest_manifest.json
      manifests/{batch_id}.json
    keyframes/{video_id}/{filename}.jpg
    metadata/{feature}/batch={batch_id}/collection={collection_id}/part-000.parquet
    features/embeddings/{provider}/{filename}.npy
    indexes/{provider}/{filename}
    reports/{name}.json
```

- **`data/raw/`** — original competition videos (read-only source of truth).
- **`data/staging/`** — per-run extraction outputs before consolidation.
- **`data/versions/`** — immutable, run-scoped enrichment outputs. A version is
  never edited after `_READY.json`; `dataset_manifest.json` holds the pointer
  that decides which one is active. Kept outside `data/corpus` on purpose:
  `r2 pull --corpus` pulls that prefix blindly and would otherwise download
  every unreleased version.
- **`data/corpus/`** — canonical, consolidated artifacts (keyframes, metadata
  Parquets, embedding vectors, indexes, reports).

## Environment setup

Copy `.env.example` to `.env` and fill in your R2 credentials:

```bash
cp .env.example .env
# edit .env with your R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, etc.
```

Required R2 variables:

| Variable | Description |
|---|---|
| `R2_ACCOUNT_ID` | Cloudflare account ID (32-char hex) |
| `R2_ACCESS_KEY_ID` | R2 access key |
| `R2_SECRET_ACCESS_KEY` | R2 secret key |
| `R2_BUCKET` | Bucket name |
| `R2_ENDPOINT` | `https://<account_id>.r2.cloudflarestorage.com` |

## CLI commands

The `r2` command group is available via `python -m mvr.cli r2`:

```bash
# Check connectivity
python -m mvr.cli r2 status

# List objects under a prefix
python -m mvr.cli r2 list data/corpus/keyframes --max-keys 50

# Pull the entire corpus tree
python -m mvr.cli r2 pull --corpus --local-root .

# Pull specific raw videos
python -m mvr.cli r2 pull --videos L21_V001,L21_V029 --batch 1 --collection L21

# Pull only the dataset manifest
python -m mvr.cli r2 pull --manifest

# Push staging artifacts
python -m mvr.cli r2 push --staging artifacts/ocr --feature ocr --run-id run-001

# Push consolidated metadata
python -m mvr.cli r2 push --parquet out/ocr.parquet --feature ocr --batch 1 --collection L21

# Push provider embeddings
python -m mvr.cli r2 push --embeddings out/beit3.npy --provider beit3
```

## Standalone scripts

For use in automation or GPU machines without the full CLI:

```bash
# Pull (with dry-run)
python scripts/operations/r2_pull.py --corpus --dry-run
python scripts/operations/r2_pull.py --corpus --local-root /data

# Push
python scripts/operations/r2_push.py --staging artifacts/ocr --feature ocr --run-id run-001
python scripts/operations/r2_push.py --embeddings out/beit3.npy --provider beit3
```

## Local workstation workflow

1. Pull raw videos from R2:
   ```bash
   python -m mvr.cli r2 pull --videos L21_V001 --batch 1 --collection L21
   ```
2. Ingest (shot detection + keyframe extraction):
   ```bash
   python -m mvr.cli ingest data/raw/batch1/videos/L21/L21_V001.mp4 --output artifacts/ingest_L21_V001
   ```
   The default is the canonical TransNetV2 `0.5` +
   `hybrid_temporal_content` selector and therefore requires the pinned
   MetaCLIP-2 model to be cached locally. Add `--allow-model-downloads` only
   when an explicit first-run download is intended.
3. Run feature extraction:
   ```bash
   python scripts/run_feature_extraction.py \
     --mapping-csv artifacts/ingest_L21_V001/keyframe_mapping.csv \
     --keyframe-root artifacts/ingest_L21_V001/keyframes \
     --output artifacts/features_L21_V001 \
     --device cuda
   ```
4. Push results back to R2:
   ```bash
   python -m mvr.cli r2 push --staging artifacts/features_L21_V001 --feature embeddings --run-id L21_V001
   ```

## GPU machine workflow

See [GPU_DEPLOYMENT_RUNBOOK.md](./GPU_DEPLOYMENT_RUNBOOK.md) for the full
step-by-step guide. The short version:

1. Clone the repo, copy `.env.example` to `.env`, fill in R2 credentials.
   Set the pinned model locations with `MVR_RFDETR_CHECKPOINT`,
   `MVR_BEIT3_CHECKPOINT`, `MVR_METACLIP2_MODEL`, and `MVR_PE_CHECKPOINT`.
2. Run a one-video canonical canary directly from R2:
   `python scripts/operations/run_pipeline_local.py --collections L21 --videos L21_V001 --device cuda`.
3. Expand to selected collections, for example `--collections L21,L22`, or
   omit both filters for L21-L30. The autopilot pulls one video at a time and
   pushes results by default; add `--no-push` for a local-only run. CLI
   checkpoint arguments override `.env`; `.env` overrides the repository-local
   fallback paths.
4. Start the serving stack: `docker compose -f docker-compose.gpu.yml up -d`

## Unattended ingestion-only workflow

Use the operations runner when raw videos already live on R2 and only
canonical ingestion should run. The runner pulls one video at a time, runs
TransNetV2 `0.5` plus `hybrid_temporal_content`, uploads the publishable output
tree, records resume state, and then moves to the next video. The regenerable
`selector_cache/` candidate frames are not uploaded and remain local until any
requested post-success eviction; final `keyframes/`, `selector_embeddings/`,
mappings, metadata, manifests, and the success marker are uploaded. It does
not run RF-DETR, the downstream three-provider embedding stage, consolidation,
or promotion.

One-video canary:

```powershell
.\.venv\Scripts\python.exe scripts\operations\run_pipeline_local.py `
  --ingest-only `
  --collections L21 `
  --videos L21_V001 `
  --device cuda `
  --run-id ingest-batch1-canary
```

Unattended selected-collection run with bounded local disk:

```powershell
.\.venv\Scripts\python.exe scripts\operations\run_pipeline_local.py `
  --ingest-only `
  --collections L21,L22 `
  --device cuda `
  --run-id ingest-batch1-001 `
  --work-root D:\aic26_ingest_work `
  --evict-artifacts-after-push
```

Omit `--collections` to process the runner's default L21-L30 set. Add
`--allow-model-downloads` only when the pinned MetaCLIP-2 first-run download is
intentional. Raw local videos are removed after success unless `--keep-video`
is passed. `--evict-artifacts-after-push` additionally removes the local
ingestion tree only after R2 accepts the final success marker.

Each completed video is staged under:

```text
data/staging/ingestion/<run-id>/batch=1/collection=<Lxx>/<video-id>/
```

The runner uploads `_SUCCESS.json` last. A prefix without that marker is
partial and is not marked complete locally. Resume state and the final report
default to `<work-root>/ingest_state.json` and
`<work-root>/ingest_report.json`. To resume after an interruption, rerun the
same command with the same `--run-id` and `--work-root`; completed videos are
skipped and failed/incomplete videos are attempted again.

Excluding `selector_cache/` affects only new uploads. It does not delete cache
objects already present under an existing R2 run prefix.

These objects intentionally remain in `data/staging`. Validation and promotion
into the single corpus-level `data/corpus/ingest/keyframe_mapping.csv` are a
separate operation; per-video workers must not overwrite that shared mapping.

### Consolidate multiple ingestion runs

Prepare one fail-closed candidate after every assigned collection has finished:

```powershell
.\.venv\Scripts\python.exe scripts\operations\prepare_r2_ingestion_promotion.py `
  --batch-id 1 `
  --source "ingest-batch1-001:L21,L22,L23,L24,L25" `
  --source "ingest-batch1-vm1:L26,L27,L28,L29,L30" `
  --promotion-id "batch1-hybrid-20260901" `
  --work-root "artifacts\verification\r2-ingestion-promote\preparations" `
  --upload
```

The command treats raw MP4 objects as authoritative membership and requires
one valid `_SUCCESS.json` plus the complete metadata/mapping tree for every
video. It validates marker counts, per-video manifests, mapping/image
identity, selector-profile consistency, global keyframe uniqueness, and
Parquet/CSV consolidation. JPEGs are not downloaded; a server-side copy
manifest is generated instead.

`--upload` writes only to:

```text
data/staging/ingestion-consolidated/<promotion-id>/
```

The `_READY.json` marker is uploaded last. This command deliberately does not
mutate `data/corpus`. Its report compares the candidate with the active
canonical mapping and blocks activation when existing embeddings, objects,
ASR, or indexes would become stale. Resolve that lifecycle decision before
copying keyframes or replacing the shared corpus mapping.

### Activate a validated ingestion candidate

Activation is an explicit, resumable single-writer operation. If preparation
reports stale downstream data, acknowledge each affected collection by name:

```powershell
.\.venv\Scripts\python.exe scripts\operations\activate_r2_ingestion_promotion.py `
  --promotion-id "batch1-hybrid-20260901" `
  --bundle-root "artifacts\verification\r2-ingestion-promote\preparations\batch1-hybrid-20260901\bundle" `
  --acknowledge-stale L24 `
  --work-root "artifacts\verification\r2-ingestion-activate\activation" `
  --workers 20
```

The command revalidates local checksums and the remote `_READY.json`, then
copies the current canonical ingest metadata, dataset manifest, and affected
keyframes to:

```text
data/staging/ingestion-promotion-backups/<promotion-id>/
```

`_BACKUP_READY.json` is required before any promoted keyframe is copied. The
operation skips destination objects whose size and ETag already match, so the
same command safely resumes after interruption. It verifies every expected
keyframe before uploading canonical metadata, writes
`data/corpus/dataset_manifest.json` as the final active pointer, and only then
removes exact stale image keys that are present in the backup. Superseded
embedding/object/ASR artifacts remain physically on R2 but are marked stale
and must be rebuilt for the new keyframe identity.

The real Batch1 activation `batch1-hybrid-20260901` completed with 872 videos,
97,712 shots, and 133,265 canonical keyframes. It replaced the older L24
identity, removed 251 stale L24 images after backup, and marked downstream data
`REBUILD_REQUIRED`.

## Optimized full enrichment run from canonical ingestion

After ingestion activation, do **not** use the old full mode of
`run_pipeline_local.py`: that mode downloads raw videos and runs ingestion
again. Use the dedicated enrichment runner. It downloads canonical JPEGs once
for RF-DETR and the three visual providers. It separately downloads each raw
MP4 once and shares that file between production OCR and ASR. It publishes
only to a run-scoped staging prefix.

Before the first run, the VM must have the two pinned OCR images
`mvr-ocr-deepsolo:task04b` and `mvr-ocr-parseq-vi:task04b`, verified OCR
weights under `MVR_OCR_WEIGHTS_ROOT` (default `artifacts/_ocr_models`), and the
pinned ASR dependencies/model cache available in `.venv`. The wrapper checks
the `[operations]` extra (`pip install -e '.[operations]'`), Docker, both OCR
images, and the native ChunkFormer/Silero imports before it starts an expensive
R2 transfer.

One-video canary on Linux/RTX 5060 Ti:

```bash
./scripts/operations/run_enrichment_rtx5060ti.sh \
  L21 enrich-batch1-5060ti-canary /home/ezycloudx-admin/aic26_enrichment \
  --videos L21_V001
```

Full Batch1 after the canary succeeds:

```bash
./scripts/operations/run_enrichment_rtx5060ti.sh \
  L21,L22,L23,L24,L25,L26,L27,L28,L29,L30 \
  enrich-batch1-5060ti-v1 \
  /home/ezycloudx-admin/aic26_enrichment
```

The equivalent direct command is:

```bash
.venv/bin/python scripts/operations/run_enrichment_r2.py \
  --collections L21,L22,L23,L24,L25,L26,L27,L28,L29,L30 \
  --batch-id 1 \
  --run-id enrich-batch1-5060ti-v1 \
  --work-root /home/ezycloudx-admin/aic26_enrichment \
  --device cuda \
  --download-workers 24 \
  --raw-download-workers 4 \
  --upload-workers 12 \
  --embedding-batch-size 8
```

Use the same `--run-id` and `--work-root` to resume. Exact R2 downloads are
atomic and existing local files are skipped. Upload retries list the collection
prefix once and skip remote objects with the same size, avoiding duplicate
PUTs. Model execution is provider-major: RF-DETR, BEiT-3, and PE-Core-L are
each loaded once and reused across all selected collections; only one in-process
GPU model is resident at a time. MetaCLIP-2 is not loaded because the runner
validates and merges the selected-vector artifacts already produced during
ingestion. OCR creates one combined resumable Docker job for every still-
currently pending collection, so DeepSolo and PARSeq each load once. ASR constructs
Silero and ChunkFormer once, keeps the independent +/-3 second keyframe
contract, but decodes a video's 16 kHz mono audio once into a one-video PCM
cache instead of spawning FFmpeg for every keyframe.

The result layout is:

```text
data/versions/enrichment/<run-id>/
  batch=1/collection=<Lxx>/
    embeddings/{beit3,metaclip2,perception_encoder}/
    metadata/{objects,ocr,asr}/part-000.parquet
    task05a/{objects,ocr,asr}.tar.gz
    ocr/manifest.json
    asr/manifest.json
    collection_report.json
    _SUCCESS.json
  enrichment_report.json
  _READY.json
```

The three archives replace one PUT per keyframe JSON envelope. Objects, OCR,
and ASR are consolidated once per collection into typed Parquet. A collection
`_SUCCESS.json` now requires all six branches: objects, BEiT-3, MetaCLIP-2,
PE-Core-L, OCR, and ASR. `_READY.json` is written last only after every selected
collection satisfies that contract. These markers mean the VM output is safely
present on R2. They do **not** mean the artifacts are active: this runner
never overwrites `data/corpus`, and the pointer swap below is a separate,
explicit step. Extraction-status Parquet is also deliberately not fabricated
here because its current shared schema lacks `video_id`; that identity gap must
be resolved before status or Milvus ingest can be safely promoted.

With a complete Batch1 run, raw-video R2 reads are bounded at roughly one GET
per selected video (872 for the current activated corpus), not one GET per OCR
frame or ASR window. `--evict-after-push` removes a collection's JPEGs, raw
MP4s, selector inputs, and local artifacts only after the six-feature success
marker has been uploaded.

The current production OCR images retain their reuse-baseline CUDA/PyTorch
pins. On newer GPUs such as RTX 5060 Ti, image presence alone does not prove
kernel compatibility. The wrapper therefore runs a CUDA matrix-operation
canary inside both OCR images before any R2 input download; a failure stops the
run and requires a separately validated pinned-image rebuild rather than an
implicit dependency upgrade.

### Activate an enrichment version

Publishing a version does not make it active. Activation validates the version
and swaps one pointer:

```powershell
.\.venv\Scripts\python.exe scripts\operations\activate_enrichment_version.py `
  --version "enrich-batch1-5060ti-v1"
```

Validation runs first and is fail-closed. From one prefix listing plus the
version's own markers it requires: `_READY.json` with `status=READY`, a
`_SUCCESS.json` per declared collection, every artifact each `_SUCCESS.json`
declares (`object_metadata`, `object_transport`, the OCR/ASR equivalents, and
the OCR/ASR manifests) present and non-empty, `manifest.json`,
`feature_mapping.csv` plus at least one non-empty `.npy` for every declared
provider, and per-collection `video_count`/`keyframe_count` that sum to the
totals in `_READY.json`. Expected keys are read from the markers rather than
hard-coded, and every problem is reported at once so a broken run can be fixed
in one pass.

Only then does it back up the current `dataset_manifest.json` under
`data/versions/manifest-backups/<activated_at>/` and write the manifest with
`active_enrichment_version` set. No artifact is copied, so activation costs one
listing and two PUTs regardless of corpus size.

To run the check on its own, before building indexes or activating anything:

```powershell
.\.venv\Scripts\python.exe scripts\operations\activate_enrichment_version.py `
  --version "enrich-batch1-5060ti-v1" --validate-only
```

Roll back by running the same command with the previous version id; the report
of every activation names the version it replaced under `previous_version`.

Consumers must resolve `active_enrichment_version` from the manifest rather
than scanning `data/versions/`, because an unreleased or failed version can be
present under that prefix by design.

### Building indexes from a version

Index builders read the version itself, not `extraction_status.parquet`. That
schema has no `video_id` column and no pipeline writes it, while a version
already publishes a stronger source: each collection `_SUCCESS.json` proves all
six features completed, and each provider `feature_mapping.csv` carries
`video_id` next to `keyframe_id` plus `feature_index`, the row number of that
keyframe inside the provider `.npy`.

`mvr.operations.enrichment_catalog.read_enrichment_catalog` validates the
version and returns one `CatalogEntry` per keyframe, holding
`(feature_file, feature_index)` for every declared provider. Identity is always
the `(video_id, keyframe_id)` pair, never `keyframe_id` alone: the organizer's
keyframe ids are per-video ordinals, so their 177,321 keyframes carry only 632
distinct id strings and a keyframe-id-only join attaches one video's vector to
another video's keyframe.

The read is fail-closed. A keyframe any provider did not embed, a vector file
the mapping names but nobody published, a duplicate identity, or an unusable
row index stops the read instead of yielding an index that silently omits or
misaligns entries.

## Caching

R2 downloads are cached locally under `.cache/mvr/` (configurable via
`R2_CACHE_DIR`). The cache is bounded (default 50 GB, configurable via
`R2_CACHE_MAX_GB`) and uses LRU eviction. Re-running a pull skips files
already cached unless `--force` is passed.

## Cost notes

Cloudflare R2 has **zero egress fees**, so pulling data to any machine
(local or cloud GPU) is free. You pay only for storage ($0.015/GB-month)
and operations ($4.50/million Class A, $0.36/million Class B).

A full Batch1 enrichment run over the activated corpus (872 videos, 133,265
keyframes) costs roughly 273,500 Class B operations (~$0.10) and ~3,000 Class A
operations (~$0.014); activating the resulting version adds three more Class A
operations. Both stay inside the free monthly allowance.
