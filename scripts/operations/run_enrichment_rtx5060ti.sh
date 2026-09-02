#!/usr/bin/env bash
set -Eeuo pipefail

# One-command RTX 5060 Ti full enrichment for an activated Batch1 corpus.
# Prerequisites: repository cloned, .venv installed, .env configured, and the
# visual/OCR checkpoints cached. OCR uses the two pinned GPU Docker images;
# ASR uses the native pinned Silero + ChunkFormer runtime in .venv.

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COLLECTIONS="${1:-L21,L22,L23,L24,L25,L26,L27,L28,L29,L30}"
RUN_ID="${2:-enrich-batch1-5060ti-v1}"
WORK_ROOT="${3:-${MVR_ENRICH_WORK_ROOT:-${REPOSITORY_ROOT}/../aic26_enrichment}}"

cd "$REPOSITORY_ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv/bin/python. Create/install the project environment first." >&2
  exit 2
fi
if [[ ! -f ".env" ]]; then
  echo "Missing .env with R2 credentials and pinned model paths." >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "Missing Docker; production DeepSolo/PARSeq OCR runs in GPU containers." >&2
  exit 2
fi
for image in mvr-ocr-deepsolo:task04b mvr-ocr-parseq-vi:task04b; do
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Missing OCR image: $image" >&2
    exit 2
  fi
  if ! docker run --rm --gpus all "$image" python -c \
    'import torch; assert torch.cuda.is_available(); value = torch.ones((8, 8), device="cuda"); print(float((value @ value).sum()))' \
    >/dev/null 2>&1; then
    echo "OCR image cannot execute CUDA kernels on this GPU: $image" >&2
    echo "Do not start Batch1; verify/rebuild the pinned OCR runtime for this GPU first." >&2
    exit 2
  fi
done
if ! .venv/bin/python -c 'import chunkformer, silero_vad' >/dev/null 2>&1; then
  echo "Missing native ASR dependencies; install the pinned ASR runtime first." >&2
  exit 2
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

exec .venv/bin/python scripts/operations/run_enrichment_r2.py \
  --collections "$COLLECTIONS" \
  --batch-id 1 \
  --run-id "$RUN_ID" \
  --work-root "$WORK_ROOT" \
  --device cuda \
  --download-workers 24 \
  --raw-download-workers 4 \
  --upload-workers 12 \
  --embedding-batch-size 8 \
  "${@:4}"
