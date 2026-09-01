#!/usr/bin/env bash
set -Eeuo pipefail

# One-command RTX 5060 Ti enrichment for an already-activated Batch1 corpus.
# Prerequisites: repository cloned, .venv installed, .env configured, and the
# three pinned local checkpoint paths exported in .env.

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

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

exec .venv/bin/python scripts/operations/run_enrichment_r2.py \
  --collections "$COLLECTIONS" \
  --batch-id 1 \
  --run-id "$RUN_ID" \
  --work-root "$WORK_ROOT" \
  --device cuda \
  --download-workers 24 \
  --upload-workers 12 \
  --embedding-batch-size 8 \
  "${@:4}"
