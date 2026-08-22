# OCR Docker Pipeline

This directory contains the Dockerfiles and definitions for the pinned production
DeepSolo and Vietnamese-finetuned PARSeq models in isolated containers.

## Images
- `mvr/deepsolo`: Uses Detectron2 v0.6 on CUDA 11.1 / PyTorch 1.9.0.
- `mvr/parseq_vi`: Uses CUDA 11.1 / PyTorch 1.10.0.

## Workflow
1. Fetch and SHA256-verify models: `powershell scripts\ocr\fetch_ocr_models.ps1`
2. Build images: `powershell scripts\ocr\build_ocr_images.ps1`
3. Verify GPU access: `powershell scripts\ocr\verify_ocr_docker.ps1`
4. Run OCR: `powershell scripts\ocr\run_ocr_docker.ps1`

## Communication
The model lock is `docker/ocr/model_sources.lock.json`. Fetching validates existing
files, downloads into temporary files, verifies both distribution and checkpoint
hashes, and atomically publishes only trusted bytes. Host and worker runtime
preflights verify the same SHA256 values before inference. Model mounts are
read-only inside the containers.

The host and containers communicate over JSON files through volume mounts. The
workers import only the small shared integrity verifier from `/workspace/src`;
model inference remains isolated in each pinned Docker environment.
