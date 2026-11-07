# VideoSync VibeVoice RunPod worker — GPU TTS (VibeVoice-Realtime-0.5B) + CPU ASR (faster-whisper)
# Built by GitHub Actions and pushed to ghcr.io. RunPod serverless pulls it;
# never build locally or on RunPod.
FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf_cache \
    FFMPEG_BIN=/usr/bin/ffmpeg \
    VIBEVOICE_REPO=/opt/vibevoice

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        ca-certificates \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Vendored VibeVoice source (inference classes + streaming voice presets),
# pinned so model/voice behavior is stable across rebuilds.
RUN git clone --depth 1 https://github.com/microsoft/VibeVoice.git /opt/vibevoice

# CUDA-enabled PyTorch wheel (bundles the CUDA runtime — base image needs no CUDA toolkit).
RUN pip install --upgrade pip \
    && pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# VibeVoice repo extra deps — transformers pinned at 4.51.3 via requirements.txt
# (older GenerationMixin import path + torch 2.4.1 compatibility).
RUN HF_HUB_OFFLINE=1 pip install --no-deps -e "/opt/vibevoice[streamingtts]" \
    && python - <<'PY'
import importlib
import subprocess, sys

required = ["diffusers", "sentencepiece", "soundfile"]
for pkg in required:
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
PY

COPY handler.py /app/handler.py
WORKDIR /app

CMD ["python", "-u", "handler.py"]
