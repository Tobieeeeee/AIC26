FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
ARG PARSEQ_VI_REVISION=76cc5f3cc6268457aac764653400fdff681f8271
RUN git clone https://github.com/lynguyenminh/vietnamese-scenetext-detection-recognition.git && \
    git -C /opt/vietnamese-scenetext-detection-recognition checkout --detach "$PARSEQ_VI_REVISION"

WORKDIR /opt/vietnamese-scenetext-detection-recognition
# We install common dependencies directly to avoid issues with older requirements files
RUN pip install --no-cache-dir opencv-python-headless==4.5.5.64 pyyaml timm lmdb==1.4.1 pytorch-lightning albumentations nltk

WORKDIR /workspace
