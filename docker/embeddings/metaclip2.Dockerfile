FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN python -m pip install --no-cache-dir \
      --index-url "${TORCH_INDEX_URL}" \
      torch==2.5.1 torchvision==0.20.1 \
    && python -m pip install --no-cache-dir \
      numpy==2.1.2 \
      Pillow==11.0.0 \
      sentencepiece==0.2.0 \
      transformers==4.57.1 \
    && python -m pip check

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /repo
