#!/usr/bin/env bash
# Bootstrap and run canonical R2 ingestion on an Ubuntu 24.04 RTX 5060 Ti host.

set -Eeuo pipefail

readonly TORCH_VERSION="2.7.1"
readonly TORCHVISION_VERSION="0.22.1"
readonly PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
readonly TRANSNET_VERSION="1.0.5"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON="${VENV_DIR}/bin/python"

COLLECTIONS=""
VIDEOS=""
RUN_ID=""
BATCH_ID="1"
LIMIT="0"
WORK_ROOT=""
ALLOW_MODEL_DOWNLOADS=false
DETACH=false
FORCE_SETUP=false
KEEP_ARTIFACTS=false
SETUP_ONLY=false
SKIP_SYSTEM_PACKAGES=false

usage() {
  cat <<'EOF'
Usage:
  bash scripts/operations/run_ingest_rtx5060ti.sh [options]

Required for an ingestion run:
  --collections L26,L27      Explicit shard owned by this machine
  --run-id ingest-batch1-vm1 Stable R2 staging/resume identity

Options:
  --videos L26_V001          Optional comma-separated video IDs; one collection only
  --batch-id 1               Raw R2 batch (default: 1)
  --limit N                  Process at most N videos (default: all in shard)
  --work-root PATH           Resume/output root (default: ~/aic26_ingest/<run-id>)
  --allow-model-downloads    Permit the pinned MetaCLIP-2 first-run download
  --detach                   Run ingestion under nohup and write runner.log/runner.pid
  --keep-artifacts           Keep local ingestion output after successful R2 upload
  --force-setup              Reinstall the verified RTX 5060 Ti Python runtime
  --skip-system-packages     Do not run apt-get; require Python 3.12 and FFmpeg
  --setup-only               Install and validate the GPU runtime, then exit
  -h, --help                 Show this help

Examples:
  # Install/validate only
  bash scripts/operations/run_ingest_rtx5060ti.sh --setup-only

  # One-video canary
  bash scripts/operations/run_ingest_rtx5060ti.sh \
    --collections L26 --videos L26_V001 \
    --run-id ingest-batch1-vm1-canary --allow-model-downloads

  # Unattended collection shard; safe to rerun with the same run-id/work-root
  bash scripts/operations/run_ingest_rtx5060ti.sh \
    --collections L26,L27,L28,L29,L30 \
    --run-id ingest-batch1-vm1 --detach --allow-model-downloads
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

log() {
  printf '[rtx5060ti] %s\n' "$*"
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "${value}" ]] || die "${option} requires a value"
}

while (($#)); do
  case "$1" in
    --collections)
      require_value "$1" "${2:-}"
      COLLECTIONS="$2"
      shift 2
      ;;
    --videos)
      require_value "$1" "${2:-}"
      VIDEOS="$2"
      shift 2
      ;;
    --run-id)
      require_value "$1" "${2:-}"
      RUN_ID="$2"
      shift 2
      ;;
    --batch-id)
      require_value "$1" "${2:-}"
      BATCH_ID="$2"
      shift 2
      ;;
    --limit)
      require_value "$1" "${2:-}"
      LIMIT="$2"
      shift 2
      ;;
    --work-root)
      require_value "$1" "${2:-}"
      WORK_ROOT="$2"
      shift 2
      ;;
    --allow-model-downloads)
      ALLOW_MODEL_DOWNLOADS=true
      shift
      ;;
    --detach)
      DETACH=true
      shift
      ;;
    --keep-artifacts)
      KEEP_ARTIFACTS=true
      shift
      ;;
    --force-setup)
      FORCE_SETUP=true
      shift
      ;;
    --skip-system-packages)
      SKIP_SYSTEM_PACKAGES=true
      shift
      ;;
    --setup-only)
      SETUP_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

[[ -r /etc/os-release ]] || die "this runner requires Ubuntu 24.04"
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == 24.04* ]] || \
  die "verified only on Ubuntu 24.04; found ${PRETTY_NAME:-unknown OS}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export PYTHONUNBUFFERED=1

cd -- "${REPO_ROOT}"

runtime_is_ready() {
  [[ -x "${PYTHON}" ]] || return 1
  "${PYTHON}" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version

assert version("torch").split("+")[0] == "2.7.1"
assert version("torchvision").split("+")[0] == "0.22.1"
assert version("transnetv2-pytorch") == "1.0.5"

import boto3
import cv2
import torch
import torchvision
import transformers
import transnetv2_pytorch

assert torch.version.cuda is not None and torch.version.cuda.startswith("12.8")
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0)
assert "sm_120" in torch.cuda.get_arch_list()
PY
}

NEEDS_SETUP=false
if [[ "${FORCE_SETUP}" == true ]] || ! runtime_is_ready; then
  NEEDS_SETUP=true
fi

NEEDS_SYSTEM_PACKAGES=false
for command_name in python3 ffmpeg ffprobe; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    NEEDS_SYSTEM_PACKAGES=true
  fi
done

if [[ "${SKIP_SYSTEM_PACKAGES}" == false ]] && \
   [[ "${NEEDS_SETUP}" == true || "${NEEDS_SYSTEM_PACKAGES}" == true ]]; then
  command -v apt-get >/dev/null 2>&1 || die "apt-get is unavailable"
  if ((EUID == 0)); then
    apt-get update
    env DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3 python3-venv python3-pip ffmpeg
  else
    command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages"
    sudo apt-get update
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3 python3-venv python3-pip ffmpeg
  fi
fi

command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg is required (rerun without --skip-system-packages)"
command -v ffprobe >/dev/null 2>&1 || die "ffprobe is required (rerun without --skip-system-packages)"

PYTHON_MINOR="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PYTHON_MINOR}" == "3.12" ]] || die "Python 3.12 is required; found ${PYTHON_MINOR}"
if [[ -x "${PYTHON}" ]]; then
  VENV_PYTHON_MINOR="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "${VENV_PYTHON_MINOR}" == "3.12" ]] || \
    die "existing .venv uses Python ${VENV_PYTHON_MINOR}; move it aside and rerun for Python 3.12"
fi

if [[ "${NEEDS_SETUP}" == true ]]; then
  log "installing verified PyTorch CUDA 12.8 runtime for RTX 5060 Ti"
  if [[ ! -x "${PYTHON}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
  "${PYTHON}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON}" -m pip install --upgrade \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${PYTORCH_INDEX_URL}"
  "${PYTHON}" -m pip install -e ".[operations]"
  "${PYTHON}" -m pip install --upgrade \
    "numpy>=1.26,<3" \
    "Pillow==11.0.0" \
    "sentencepiece==0.2.0" \
    "transformers==4.57.1" \
    "transnetv2-pytorch==${TRANSNET_VERSION}"
else
  log "reusing the already verified .venv runtime"
fi

"${PYTHON}" -m pip check

log "checking CUDA capability, sm_120 kernels, imports and GPU matmul"
"${PYTHON}" - <<'PY'
import boto3
import cv2
import torch
import torchvision
import transformers
import transnetv2_pytorch

expected_capability = (12, 0)
capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
arches = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
if capability != expected_capability:
    raise SystemExit(f"expected RTX 5060 Ti capability {expected_capability}, got {capability}")
if "sm_120" not in arches:
    raise SystemExit(f"PyTorch has no sm_120 kernel support: {arches}")
x = torch.randn((1024, 1024), device="cuda")
y = x @ x
torch.cuda.synchronize()
print(f"GPU_OK: {torch.cuda.get_device_name(0)} capability={capability}")
print(f"PYTORCH_OK: torch={torch.__version__} cuda={torch.version.cuda}")
print(f"CUDA_MATMUL_OK: {float(y[0, 0]):.6f}")
print("IMPORTS_OK")
PY

if [[ "${SETUP_ONLY}" == true ]]; then
  [[ "${DETACH}" == false ]] || die "--detach cannot be combined with --setup-only"
  log "setup and RTX 5060 Ti validation complete"
  exit 0
fi

[[ -n "${COLLECTIONS}" ]] || die "--collections is required to prevent two machines from claiming the same shard"
[[ -n "${RUN_ID}" ]] || die "--run-id is required for stable R2 staging and resume"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid --run-id: ${RUN_ID}"
[[ "${BATCH_ID}" =~ ^[0-9]+$ ]] || die "--batch-id must be numeric"
[[ "${LIMIT}" =~ ^[0-9]+$ ]] || die "--limit must be a non-negative integer"

IFS=',' read -r -a COLLECTION_ARRAY <<< "${COLLECTIONS}"
((${#COLLECTION_ARRAY[@]} > 0)) || die "--collections is empty"
for index in "${!COLLECTION_ARRAY[@]}"; do
  collection="${COLLECTION_ARRAY[$index]//[[:space:]]/}"
  [[ "${collection}" =~ ^L[0-9]{2}$ ]] || die "invalid collection: ${collection}"
  COLLECTION_ARRAY[$index]="${collection}"
done
COLLECTIONS="$(IFS=','; printf '%s' "${COLLECTION_ARRAY[*]}")"

if [[ -n "${VIDEOS}" ]]; then
  ((${#COLLECTION_ARRAY[@]} == 1)) || die "--videos requires exactly one collection"
  IFS=',' read -r -a VIDEO_ARRAY <<< "${VIDEOS}"
  for index in "${!VIDEO_ARRAY[@]}"; do
    video="${VIDEO_ARRAY[$index]//[[:space:]]/}"
    [[ "${video}" =~ ^L[0-9]{2}_V[0-9]{3}$ ]] || die "invalid video id: ${video}"
    [[ "${video%%_*}" == "${COLLECTION_ARRAY[0]}" ]] || \
      die "video ${video} does not belong to ${COLLECTION_ARRAY[0]}"
    VIDEO_ARRAY[$index]="${video}"
  done
  VIDEOS="$(IFS=','; printf '%s' "${VIDEO_ARRAY[*]}")"
fi

if [[ ! -f "${REPO_ROOT}/.env" ]]; then
  for variable in R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET R2_ENDPOINT; do
    [[ -n "${!variable:-}" ]] || \
      die ".env is missing and ${variable} is not exported; copy the VM's R2 configuration first"
  done
fi

log "confirming real R2 read access for every assigned collection"
for collection in "${COLLECTION_ARRAY[@]}"; do
  prefix="data/raw/batch${BATCH_ID}/videos/${collection}"
  r2_json="$("${PYTHON}" -m mvr.cli r2 list "${prefix}" --max-keys 1)" || \
    die "R2 list failed for ${prefix}"
  r2_count="$(printf '%s' "${r2_json}" | "${PYTHON}" -c 'import json, sys; print(json.load(sys.stdin)["count"])')" || \
    die "could not parse R2 list response for ${prefix}"
  ((r2_count > 0)) || die "R2 prefix is empty or wrong: ${prefix}"
  log "R2_OK: ${prefix}"
done

if [[ -z "${WORK_ROOT}" ]]; then
  WORK_ROOT="${HOME}/aic26_ingest/${RUN_ID}"
fi
mkdir -p -- "${WORK_ROOT}"

RUNNER_CMD=(
  "${PYTHON}"
  "${REPO_ROOT}/scripts/operations/run_pipeline_local.py"
  --ingest-only
  --collections "${COLLECTIONS}"
  --batch-id "${BATCH_ID}"
  --device cuda
  --run-id "${RUN_ID}"
  --work-root "${WORK_ROOT}"
)
if [[ -n "${VIDEOS}" ]]; then
  RUNNER_CMD+=(--videos "${VIDEOS}")
fi
if ((LIMIT > 0)); then
  RUNNER_CMD+=(--limit "${LIMIT}")
fi
if [[ "${ALLOW_MODEL_DOWNLOADS}" == true ]]; then
  RUNNER_CMD+=(--allow-model-downloads)
fi
if [[ "${KEEP_ARTIFACTS}" == false ]]; then
  RUNNER_CMD+=(--evict-artifacts-after-push)
fi

printf '[rtx5060ti] launching:'
printf ' %q' "${RUNNER_CMD[@]}"
printf '\n'

if [[ "${DETACH}" == true ]]; then
  log_file="${WORK_ROOT}/runner.log"
  pid_file="${WORK_ROOT}/runner.pid"
  if [[ -s "${pid_file}" ]]; then
    previous_pid="$(<"${pid_file}")"
    if [[ "${previous_pid}" =~ ^[0-9]+$ ]] && kill -0 "${previous_pid}" 2>/dev/null; then
      die "runner is already active with PID ${previous_pid}"
    fi
  fi
  nohup "${RUNNER_CMD[@]}" >>"${log_file}" 2>&1 </dev/null &
  runner_pid=$!
  printf '%s\n' "${runner_pid}" >"${pid_file}"
  disown "${runner_pid}" 2>/dev/null || true
  log "detached PID=${runner_pid} log=${log_file}"
  log "follow progress with: tail -f '${log_file}'"
  exit 0
fi

exec "${RUNNER_CMD[@]}"
