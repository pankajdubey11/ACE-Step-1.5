# ACE-Step 1.5 Music Engine — RunPod Serverless image
#
# ⚠️ CUDA 12.8 / Python 3.11 (per the repo's own requirements: torch 2.10.0+cu128).
# This is the third distinct CUDA across the pipeline (Video cu118, Voice cu121,
# Music cu128) — inherent to self-hosting three different model stacks.
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf \
    CHECKPOINT_DIR=/runpod-volume/checkpoints

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git ffmpeg build-essential && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends curl python3.11 python3.11-venv python3.11-dev && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python -m ensurepip --upgrade && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt carries platform-marked torch (linux x86_64 -> 2.10.0+cu128)
# and the cu128 extra-index, so a plain install resolves the right wheels.
# flash-attn is EXCLUDED: it's optional (ACE-Step falls back to SDPA), its build
# needs torch present at build time, and there's no prebuilt wheel for
# torch2.10/cu128 (a source compile would blow RunPod's 30-min build limit).
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    grep -viE '^flash-attn' /app/requirements.txt > /app/req.trimmed.txt && \
    pip install -r /app/req.trimmed.txt && \
    pip install runpod

COPY . /app
# Install the package itself if it exposes one (pyproject.toml present).
RUN pip install -e . || echo "editable install skipped (using sys.path)"

# Optional: bake checkpoints (else mount a RunPod network volume at /runpod-volume).
#   docker build --build-arg BAKE=1 ...
ARG BAKE=""
RUN if [ -n "$BAKE" ]; then \
      python -c "from acestep.model_downloader import get_checkpoints_dir; print(get_checkpoints_dir())" || true; \
    fi

ENV DIT_CONFIG=acestep-v15-turbo \
    LM_MODEL=acestep-5Hz-lm-0.6B \
    LM_BACKEND=hf \
    INFERENCE_STEPS=8 \
    OFFLOAD_TO_CPU=0 \
    PRELOAD=1

CMD ["python", "-u", "handler.py"]
