FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA="1"

# Install system dependencies
# Remove expired NVIDIA keys/lists to prevent apt-get update from failing
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y \
    git \
    wget \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    python3.8 \
    python3.8-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Map python to python3
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip and setuptools to fix pyproject.toml and easy_install issues
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch for CUDA 11.1
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install python dependencies
RUN pip install --no-cache-dir opencv-python==4.5.5.64 shapely scipy timm pyclipper numba editdistance scikit-image rapidfuzz Pillow==9.5.0

# Install Detectron2 for CUDA 11.1 and PyTorch 1.9.0
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# Clone the exact source revision paired with the production checkpoint lock.
WORKDIR /opt
ARG DEEPSOLO_REVISION=dbadae995035246bad3376c7a44c015c69e9b313
RUN git clone https://github.com/ViTAE-Transformer/DeepSolo.git && \
    git -C /opt/DeepSolo checkout --detach "$DEEPSOLO_REVISION"

WORKDIR /opt/DeepSolo/DeepSolo
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
RUN python setup.py build develop

WORKDIR /workspace
