FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Python tooling
RUN python3 -m pip install --upgrade pip setuptools wheel


RUN pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install Python deps
COPY requirements.txt .

RUN pip install -r requirements.txt

# HuggingFace CLI
RUN pip install huggingface-hub

# Copy application code
COPY . .

# clone latentsync
RUN git clone https://github.com/bytedance/LatentSync.git

WORKDIR /app/LatentSync

# Pre-download checkpoints
RUN mkdir -p /app/checkpoints/whisper

RUN huggingface-cli download ByteDance/LatentSync-1.6 \
    latentsync_unet.pt \
    --local-dir /app/checkpoints \
    --local-dir-use-symlinks False

RUN huggingface-cli download ByteDance/LatentSync-1.6 \
    whisper/tiny.pt \
    --local-dir /app/checkpoints/whisper \
    --local-dir-use-symlinks False

# Pre-download VAE
RUN python3 - <<EOF
from diffusers import AutoencoderKL
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
EOF

WORKDIR /app

# Runtime command
CMD ["python3", "app.py"]
