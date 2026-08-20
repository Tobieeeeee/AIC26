# OCR Docker Pipeline

This directory contains the Dockerfiles and definitions for running the legacy DeepSolo and PARSeq-Vietnamese OCR models in isolated containers.

## Images
- `mvr/deepsolo`: Uses Detectron2 v0.6 on CUDA 11.1 / PyTorch 1.9.0.
- `mvr/parseq_vi`: Uses CUDA 11.1 / PyTorch 1.10.0.

## Workflow
1. Fetch models: `powershell scripts\ocr\fetch_ocr_models.ps1`
2. Build images: `powershell scripts\ocr\build_ocr_images.ps1`
3. Verify GPU access: `powershell scripts\ocr\verify_ocr_docker.ps1`
4. Run OCR: `powershell scripts\ocr\run_ocr_docker.ps1`

## Communication
The host and containers communicate entirely over JSONL files through volume mounts (`/workspace`). No direct API calls or Python imports are shared between the host `mvr` environment and the container model workers.
