#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare a rented GPU box for a full enrichment run.
#
# Every stage is idempotent and separately runnable, because these machines are
# rented by the hour and a run that dies at stage four should not repeat the
# three that already succeeded:
#
#   bash scripts/operations/setup_vm.sh              # every stage, in order
#   bash scripts/operations/setup_vm.sh checkpoints  # just one
#
# Checkpoints are pinned by SHA256 and verified into a temporary file, so a
# truncated or substituted download never reaches the destination path. The
# hashes below are re-verified at load time by rfdetr.py and Beit3ImageEncoder;
# tests/unit/scripts/test_setup_vm.py fails if they drift from the lock files.

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY_ROOT"

# Vendored sources live outside the repository: they are separate upstream
# checkouts, and rule 11 keeps them and the weights out of Git.
SOURCES_ROOT="${MVR_VENDOR_ROOT:-${HOME}/aic26_sources}"
UNILM_REVISION="69961945a2fd283f2beed74d1f3c543265c9930c"
PERCEPTION_MODELS_REVISION="65a4a17a587b6a89eaa0dc08f231c2cb1c285f9f"
TORCHSCALE_VERSION="0.3.0"

# name|url|sha256|destination relative to the repository root
CHECKPOINTS=(
  "rfdetr|https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth|0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38|artifacts/verification/rfdetr-production-pin/models/rf-detr-large-2026.pth"
  "beit3|https://github.com/addf400/files/releases/download/beit3/beit3_base_itc_patch16_224.pth|d483a41d92b9aea7ef87d17551413b8c22faae927110666b01ac3aa7a199a25b|artifacts/_embedding_models/beit3/beit3_base_itc_patch16_224.pth"
  "beit3_tokenizer|https://github.com/addf400/files/releases/download/beit3/beit3.spm|6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4|artifacts/_embedding_models/beit3/beit3.spm"
  "perception_encoder|https://huggingface.co/facebook/PE-Core-L14-336/resolve/ee1fdb467ab42497fa2e60e94a79e9d7383c1cbd/PE-Core-L14-336.pt|0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8|artifacts/_embedding_models/perception_encoder/PE-Core-L14-336.pt"
)

log() {
  printf '\n== %s\n' "$1"
}

digest_of() {
  sha256sum "$1" | awk '{print $1}'
}

fetch_verified() {
  local url="$1" destination="$2" expected="$3" name="$4"

  if [[ -f "$destination" ]] && [[ "$(digest_of "$destination")" == "$expected" ]]; then
    printf '  %s: already verified\n' "$name"
    return 0
  fi

  mkdir -p "$(dirname "$destination")"
  local temporary="${destination}.partial"
  rm -f "$temporary"
  printf '  %s: downloading\n' "$name"
  if ! curl -fL --retry 3 --retry-delay 5 -o "$temporary" "$url"; then
    rm -f "$temporary"
    echo "  ${name}: download failed from ${url}" >&2
    return 1
  fi

  local observed
  observed="$(digest_of "$temporary")"
  if [[ "$observed" != "$expected" ]]; then
    rm -f "$temporary"
    echo "  ${name}: checksum mismatch" >&2
    echo "    expected ${expected}" >&2
    echo "    observed ${observed}" >&2
    echo "    the pinned artifact did not arrive intact; nothing was published" >&2
    return 1
  fi
  mv "$temporary" "$destination"
  printf '  %s: verified\n' "$name"
}

stage_deps() {
  log "Python environment"
  if [[ ! -x ".venv/bin/python" ]]; then
    python3.12 -m venv .venv
  fi
  .venv/bin/pip install -U pip
  .venv/bin/pip install -e '.[operations,objects,asr,beit3,perception-encoder]'
  # torchscale's metadata pins timm==0.6.13, which cannot coexist with the 1.x
  # the Perception Encoder needs. It imports neither timm nor anything else the
  # BEiT-3 path lacks, so it is installed alone; see the beit3 group's comment.
  .venv/bin/pip install --no-deps "torchscale==${TORCHSCALE_VERSION}"
}

fetch_revision() {
  # Only one commit is ever needed, and microsoft/unilm carries dozens of
  # unrelated subprojects across years of history. Cloning it in full would
  # spend gigabytes of transfer to reach a directory worth a few hundred
  # kilobytes, on every machine that gets rented.
  local url="$1" directory="$2" revision="$3"
  mkdir -p "$directory"
  if [[ ! -d "${directory}/.git" ]]; then
    git -C "$directory" init -q
    git -C "$directory" remote add origin "$url"
  fi
  git -C "$directory" fetch -q --depth 1 origin "$revision"
  git -C "$directory" checkout -q --detach FETCH_HEAD
  printf '  %s at %s\n' "$directory" "${revision:0:12}"
}

stage_sources() {
  log "Upstream sources"
  mkdir -p "$SOURCES_ROOT"

  fetch_revision https://github.com/microsoft/unilm.git \
    "${SOURCES_ROOT}/unilm" "$UNILM_REVISION"
  fetch_revision https://github.com/facebookresearch/perception_models.git \
    "${SOURCES_ROOT}/perception_models" "$PERCEPTION_MODELS_REVISION"

  # A .pth file puts both checkouts on sys.path for every process that uses the
  # venv. PYTHONPATH would have to be re-exported in each shell, tmux pane and
  # wrapper; this cannot be forgotten.
  local site_packages
  site_packages="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
  printf '%s\n%s\n' \
    "${SOURCES_ROOT}/unilm/beit3" \
    "${SOURCES_ROOT}/perception_models" \
    > "${site_packages}/_mvr_vendored_sources.pth"
  printf '  sys.path entries written to %s\n' \
    "${site_packages}/_mvr_vendored_sources.pth"
}

stage_checkpoints() {
  log "Pinned checkpoints"
  local entry name url expected destination
  for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r name url expected destination <<< "$entry"
    fetch_verified "$url" "$destination" "$expected" "$name"
  done
}

stage_ocr() {
  log "OCR weights"
  .venv/bin/python -m mvr.enrichment.ocr.model_integrity fetch \
    --lock docker/ocr/model_sources.lock.json \
    --weights-root artifacts/_ocr_models
  .venv/bin/python -m mvr.enrichment.ocr.model_integrity verify \
    --lock docker/ocr/model_sources.lock.json \
    --weights-root artifacts/_ocr_models
}

stage_images() {
  log "OCR Docker images"
  docker build -t mvr-ocr-deepsolo:task04b -f docker/ocr/deepsolo.Dockerfile docker/ocr
  docker build -t mvr-ocr-parseq-vi:task04b -f docker/ocr/parseq_vi.Dockerfile docker/ocr
}

stage_verify() {
  log "Runtime check"
  .venv/bin/python - <<'PYTHON'
import importlib

for module in (
    "boto3", "dotenv", "polars", "torch", "timm", "torchscale",
    "rfdetr", "chunkformer", "silero_vad", "core.vision_encoder.pe",
):
    importlib.import_module(module)
    print(f"  import {module}: OK")

# The pinned unilm checkout still reaches for torch._six, which torch removed
# in 2.x. Beit3ImageEncoder installs a shim before importing modeling_finetune,
# and this check calls that same function rather than repeating it: a bare
# import fails on a machine where BEiT-3 would in fact load.
from mvr.enrichment.embeddings.beit3 import _install_torch_six_inf_compat

_install_torch_six_inf_compat()
importlib.import_module("modeling_finetune")
print("  import modeling_finetune: OK")

import torch

print(f"  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
PYTHON

  cat <<EOF

Add this line to .env; RF-DETR and BEiT-3 already sit on their default paths.

  MVR_PE_CHECKPOINT=${REPOSITORY_ROOT}/artifacts/_embedding_models/perception_encoder/PE-Core-L14-336.pt

Then canary one video:

  ./scripts/operations/run_enrichment_rtx5060ti.sh L21 enrich-canary ~/aic26_enrichment --videos L21_V001
EOF
}

main() {
  local stage="${1:-all}"
  case "$stage" in
    deps) stage_deps ;;
    sources) stage_sources ;;
    checkpoints) stage_checkpoints ;;
    ocr) stage_ocr ;;
    images) stage_images ;;
    verify) stage_verify ;;
    all)
      stage_deps
      stage_sources
      stage_checkpoints
      stage_ocr
      stage_images
      stage_verify
      ;;
    *)
      echo "unknown stage: ${stage}" >&2
      echo "expected one of: deps sources checkpoints ocr images verify all" >&2
      exit 2
      ;;
  esac
}

main "$@"
