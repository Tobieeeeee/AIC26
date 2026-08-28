# AIC26 MVR — GPU Inference Environment
# ======================================
# Base image for running OCR / embedding / object extraction on a GPU host.
#
# Build:
#   docker build -f docker/gpu/Dockerfile -t mvr-gpu:latest .
#
# Run (example: feature extraction):
#   docker run --gpus all --rm -it \
#     -v $(pwd)/artifacts:/srv/artifacts \
#     -v $(pwd)/configs:/srv/configs:ro \
#     --env-file .env \
#     mvr-gpu:latest \
#     python scripts/run_feature_extraction.py \
#       --mapping-csv /srv/artifacts/keyframe_mapping.csv \
#       --keyframe-root /srv/artifacts/keyframes \
#       --output /srv/artifacts/features \
#       --device cuda
#
# NOTE: This image is intentionally CPU-only for the API service in
# docker-compose.gpu.yml. GPU extraction runs on the host directly to
# avoid NVIDIA Container Toolkit complexity in compose. This Dockerfile
# is provided for users who want a reproducible GPU extraction container.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 12.4
RUN python -m pip install --no-cache-dir \
      torch==2.5.1 \
      torchvision==0.20.1

# Core Python deps (lightweight; full extras installed via pip install -e ".[...]")
RUN python -m pip install --no-cache-dir \
      numpy \
      Pillow \
      tqdm \
      opencv-python-headless \
      pyyaml \
      requests

WORKDIR /repo

# Copy project metadata first for better layer caching
COPY pyproject.toml README.md ./
RUN python -m pip install --no-cache-dir -e "."

# Copy the rest of the source
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY docker ./docker

# Default: show help
CMD ["python", "-m", "mvr.cli", "--help"]