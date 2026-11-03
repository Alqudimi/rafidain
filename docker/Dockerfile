# syntax=docker/dockerfile:1
# ============================================================================
# Rafidain — realtime Iraqi voice-conversion backend (Seed-VC)
# Multi-stage image: base (light, mock/dev) + full (GPU inference stack)
# ============================================================================

ARG PYTHON_VERSION=3.11

# ----------------------------------------------------------------------------
# Base image: FastAPI service without the ML stack (used for CI & mock mode)
# ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SEED_VC_ROOT=/opt/seed-vc

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/rafidain

COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt

COPY rafidain ./rafidain
COPY finetune ./finetune
COPY evaluation ./evaluation
COPY scripts ./scripts
RUN pip install --no-deps -e . && mkdir -p /srv/data

ENV RAFIDAIN__DATA_DIR=/srv/data \
    RAFIDAIN__ENGINE__CACHE_DIR=/srv/checkpoints

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "rafidain.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ----------------------------------------------------------------------------
# Full image: + Seed-VC repository and the ML stack (GPU inference & training)
# ----------------------------------------------------------------------------
FROM base AS full

# Torch first (large layer, rarely rebuilt). For a CUDA-specific build, swap in:
#   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch torchaudio

RUN pip install transformers>=4.40 librosa>=0.10 soundfile>=0.12 "huggingface-hub>=0.23"

# Seed-VC repository (pinned for reproducibility — override at build time).
ARG SEED_VC_REF=main
RUN git clone --depth 1 https://github.com/Plachtta/seed-vc /opt/seed-vc \
    && cd /opt/seed-vc && git fetch --depth 1 origin ${SEED_VC_REF} \
    && git checkout ${SEED_VC_REF} || true

# Pre-download checkpoints at build time (comment out to fetch at runtime).
# RUN python scripts/download_models.py --cache-dir /srv/checkpoints

VOLUME ["/srv/data", "/srv/checkpoints"]
