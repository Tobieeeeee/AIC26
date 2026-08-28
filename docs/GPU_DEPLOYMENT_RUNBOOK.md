# GPU Machine Deployment Runbook

Step-by-step guide for deploying the AIC26 MVR pipeline on a rented GPU
machine. This runbook covers setup, data access, GPU extraction, serving,
and troubleshooting.

## 1. Machine requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA T4 (16 GB VRAM) | RTX 4090 / A100 |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB free | 200 GB (full corpus + embeddings) |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| CUDA | 12.0+ | 12.4+ |
| Docker | 24.0+ with Compose V2 | Latest |

## 2. Install prerequisites

```bash
# NVIDIA driver + Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Docker Compose V2
docker compose version  # should show v2.x

# Python 3.12
sudo apt-get install -y python3.12 python3.12-venv python3-pip
```

## 3. Clone and configure

```bash
cd /data  # or wherever you have disk space
git clone https://github.com/Tobieeeeee/AIC26.git
cd AIC26
git checkout feat/r2-gpu-runtime

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install the project
pip install -e ".[dev,api,semantic,search,ocr,objects,milvus]"

# Configure environment
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# R2 (copy these from your Cloudflare dashboard)
R2_ACCOUNT_ID=<your-32-char-hex>
R2_ACCESS_KEY_ID=<your-key>
R2_SECRET_ACCESS_KEY=<your-secret>
R2_BUCKET=<your-bucket>
R2_ENDPOINT=https://<your-account-id>.r2.cloudflarestorage.com

# GPU settings
MVR_DEVICE=cuda
```

## 4. Start infrastructure services

```bash
docker compose -f docker-compose.gpu.yml up -d milvus elasticsearch
docker compose -f docker-compose.gpu.yml ps
# Wait for healthy status (30-60s for Milvus, 30s for ES)
```

## 5. Pull data from R2

```bash
# Dry-run to check what's available
python -m mvr.cli r2 pull --corpus --dry-run

# Pull the entire corpus (keyframes, metadata, features)
python -m mvr.cli r2 pull --corpus --local-root .

# Or pull specific raw videos for ingestion
python -m mvr.cli r2 pull --videos L21_V001,L21_V029 --batch 1 --collection L21
```

## 6. Initialize Milvus schema

```bash
python -m mvr.cli milvus init-schema --drop-existing
python -m mvr.cli milvus status
```

## 7. Run extraction (GPU)

### Feature extraction (embeddings + objects)

```bash
python scripts/run_feature_extraction.py \
  --mapping-csv <path-to-keyframe_mapping.csv> \
  --keyframe-root <path-to-keyframes-root> \
  --output artifacts/features \
  --embedding-providers beit3,metaclip2,perception_encoder \
  --device cuda \
  --beit3-checkpoint <path-to-beit3.pth> \
  --perception-encoder-checkpoint <path-to-PE-Core-L14-336.pt>
```

### OCR (via Docker)

```bash
# Start OCR worker
docker run --gpus all --rm -d \
  --name mvr-ocr \
  -v $(pwd)/artifacts/ocr-runtime:/ocr-runtime \
  mvr-ocr-deepsolo:task04b \
  python /workspace/docker/ocr/worker/deepsolo_worker.py \
    --requests-dir /ocr-runtime/requests \
    --output-dir /ocr-runtime/detections \
    --weights /models/deepsolo/tt_res50_finetune_synth-tt-mlt-13-15-textocr.pth \
    --resume
```

### ASR (via Docker)

```bash
docker run --rm -it \
  -v $(pwd)/artifacts/asr:/workspace/output \
  mvr-asr:latest \
  python -m mvr.cli build-asr-features --help
```

## 8. Ingest into Milvus

```bash
python -m mvr.cli milvus ingest \
  --status-parquet artifacts/features/status.parquet \
  --keyframe-mapping artifacts/keyframe_mapping.csv \
  --beit3-vectors artifacts/features/embeddings/beit3/vectors.npy \
  --beit3-mapping artifacts/features/embeddings/beit3/mapping.json \
  --beit3-manifest artifacts/features/embeddings/beit3/manifest.json \
  --metaclip2-vectors artifacts/features/embeddings/metaclip2/vectors.npy \
  --metaclip2-mapping artifacts/features/embeddings/metaclip2/mapping.json \
  --metaclip2-manifest artifacts/features/embeddings/metaclip2/manifest.json \
  --pe-vectors artifacts/features/embeddings/perception_encoder/vectors.npy \
  --pe-mapping artifacts/features/embeddings/perception_encoder/mapping.json \
  --pe-manifest artifacts/features/embeddings/perception_encoder/manifest.json \
  --manifest artifacts/milvus_manifest.json \
  --index-type FLAT
```

## 9. Start the serving layer

```bash
# Option A: Via docker compose (recommended)
docker compose -f docker-compose.gpu.yml up -d api

# Option B: Direct (for debugging)
MVR_SEMANTIC_BACKEND=milvus \
MILVUS_HOST=localhost \
MILVUS_PORT=19530 \
MVR_ES_URL=http://localhost:9200 \
MVR_KEYFRAME_DIR=artifacts/keyframes \
uvicorn mvr.api.app:app --host 0.0.0.0 --port 8015
```

## 10. Verify

```bash
# Health
curl http://localhost:8015/health

# Semantic search
curl -X POST http://localhost:8015/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "a person walking in a park", "top_k": 5}'

# KIS search
curl -X POST http://localhost:8015/search/kis \
  -H "Content-Type: application/json" \
  -d '{"query": "a geyser of water", "top_k": 5}'
```

## 11. Push results back to R2

```bash
# Push embeddings
python -m mvr.cli r2 push \
  --embeddings artifacts/features/embeddings/beit3/vectors.npy \
  --provider beit3

# Push metadata
python -m mvr.cli r2 push \
  --parquet artifacts/features/metadata/ocr.parquet \
  --feature ocr --batch 1 --collection L21

# Push staging
python -m mvr.cli r2 push \
  --staging artifacts/features \
  --feature embeddings --run-id full-batch1
```

## 12. Useful commands

```bash
# Check Milvus status
python -m mvr.cli milvus status

# Verify Milvus schema
python -m mvr.cli milvus verify

# Show Milvus manifest
python -m mvr.cli milvus show-manifest --manifest artifacts/milvus_manifest.json

# Build provider indexes (on Milvus)
python -m mvr.cli milvus build-index --index-type FLAT

# R2 list
python -m mvr.cli r2 list data/corpus/keyframes --max-keys 20

# Run tests
python -m pytest tests/ -x -q
```

## Troubleshooting

### Docker permission denied
```bash
sudo usermod -aG docker $USER
# Log out and back in, or:
newgrp docker
```

### Milvus won't start
```bash
docker compose -f docker-compose.gpu.yml logs milvus
# Check etcd and minio health first
```

### R2 connection refused
```bash
# Verify env vars
env | grep R2_

# Test manually
python -m mvr.cli r2 list data/raw --max-keys 5
```

### Out of GPU memory
Run providers one at a time:
```bash
python scripts/run_feature_extraction.py \
  --embedding-providers beit3 --device cuda ...
# then
python scripts/run_feature_extraction.py \
  --embedding-providers metaclip2 --device cuda ...
```

### Full corpus run
For the full Batch1 corpus (29 videos, 7800 keyframes):
```bash
python scripts/operations/r2_pull.py --corpus --local-root /data/corpus
python scripts/run_all_features.ps1 \
  -MappingCsv /data/corpus/keyframe_mapping.csv \
  -KeyframeRoot /data/corpus/keyframes \
  -OutputRoot /data/output/features \
  -EmbeddingProviders beit3,metaclip2,perception_encoder \
  -Device cuda
python scripts/operations/r2_push.py \
  --staging /data/output/features \
  --feature embeddings --run-id full-batch1
```