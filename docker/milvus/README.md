# Milvus Local Dev Setup

## Quick Start

```powershell
# Start all services
docker compose -f docker/milvus/docker-compose.milvus.yml up -d

# Verify Milvus is healthy (takes ~60s on first boot)
docker compose -f docker/milvus/docker-compose.milvus.yml ps

# Check CLI connection
$env:MILVUS_HOST = "localhost"; $env:MILVUS_PORT = "19530"
python -m pip install -e ".[api,milvus]"
python -m mvr.cli milvus status
```

## Services

| Service | Port | Purpose |
|---|---|---|
| Milvus standalone | 19530 (gRPC), 9091 (HTTP) | Vector database |
| etcd | 2379 | Metadata store (Milvus internal) |
| MinIO | 9000 (API), 9001 (Console) | Object storage (Milvus internal only) |

> **Important**: This MinIO instance is **Milvus-internal storage only**.
> It is NOT connected to Cloudflare R2. R2 remains the canonical corpus source of truth.

## Environment Variables (no credentials committed)

Set these in the shell or deployment secret/config system:

```
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_TOKEN=
MILVUS_DATABASE=default
MILVUS_COLLECTION=aic26_keyframes
MVR_SEMANTIC_BACKEND=milvus
```

`milvus` is the production default. `faiss_reference` is accepted only when
explicitly selected with `MVR_SEMANTIC_BACKEND=faiss_reference`; the API never
falls back to it when Milvus is unavailable.

## CLI Commands

```powershell
# Check connection
python -m mvr.cli milvus status

# Create collection schema
python -m mvr.cli milvus init-schema

# Ingest from canonical Data-02 Parquet
python -m mvr.cli milvus ingest --status-parquet data/corpus/... --manifest data/milvus_manifest.json

# Build ANN indexes
python -m mvr.cli milvus build-index --index-type FLAT

# Verify collection and manifest
python -m mvr.cli milvus verify

# Show current manifest
python -m mvr.cli milvus show-manifest --manifest data/milvus_manifest.json

# Check stale status
python -m mvr.cli milvus stale-status --manifest data/milvus_manifest.json
```

## Teardown

```powershell
# Stop services
docker compose -f docker/milvus/docker-compose.milvus.yml down

# Remove volumes (WARNING: deletes all indexed data)
docker compose -f docker/milvus/docker-compose.milvus.yml down -v
```

## Recovery

If Milvus is lost, rebuild from canonical R2 artifacts:
1. Start dev stack
2. `python -m mvr.cli milvus init-schema`
3. `python -m mvr.cli milvus ingest ...`

No re-extraction of embeddings is required — R2 is the canonical source.
