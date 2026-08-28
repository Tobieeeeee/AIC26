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
   python -m mvr.cli ingest data/raw/1/videos/L21/L21_V001.mp4 --output artifacts/ingest_L21_V001
   ```
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
2. Pull data from R2: `python scripts/operations/r2_pull.py --corpus`
3. Run extraction with GPU: `python scripts/run_feature_extraction.py ... --device cuda`
4. Push results: `python scripts/operations/r2_push.py --embeddings ...`
5. Start the serving stack: `docker compose -f docker-compose.gpu.yml up -d`

## Caching

R2 downloads are cached locally under `.cache/mvr/` (configurable via
`R2_CACHE_DIR`). The cache is bounded (default 50 GB, configurable via
`R2_CACHE_MAX_GB`) and uses LRU eviction. Re-running a pull skips files
already cached unless `--force` is passed.

## Cost notes

Cloudflare R2 has **zero egress fees**, so pulling data to any machine
(local or cloud GPU) is free. You pay only for storage ($0.015/GB-month)
and operations ($4.50/million Class A, $0.36/million Class B).