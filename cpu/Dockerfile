# CPU VibeVoice stack for the DigitalOcean Droplet (BlenderMCP/YTDLP/HyperFrames already
# live there). Builds a single image with three internal services:
#   gateway uvicorn  0.0.0.0:8015  (this forks to TTS/ASR — the single public port)
#   tts uvicorn      127.0.0.1:8016  persistent VibeVoice CPU inference (~1.5-3.1 GB)
#   asr uvicorn      127.0.0.1:8017  faster-whisper base.en int8
# Model weights + voice presets are baked at build (no HF calls at runtime).
# Runs alongside gpu/ worker deployments of the same repo: RunPod stays the GPU upgrade path.
FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf_cache \
    HF_HUB_OFFLINE=1 \
    FFMPEG_BIN=/usr/bin/ffmpeg \
    VIBEVOICE_REPO_DIR=/opt/vibevoice \
    VIBEVOICE_TTS_MODEL=/opt/models/vibevoice-realtime \
    VIBEVOICE_ASR_MODEL=base.en \
    VIBEVOICE_TTS_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        ca-certificates \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Vendored VibeVoice source (inference classes + demo/voices/streaming_model presets)
RUN git clone --depth 1 https://github.com/microsoft/VibeVoice.git /opt/vibevoice

RUN pip install --upgrade pip \
    && pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# VibeVoice repo extras minus deps already satisfied (torch, transformers pinned at
# 4.51.3 via requirements.txt — newer transformers breaks GenerationMixin import
# and demands torch>=2.6, incompatible with the CPU wheel).
RUN HF_HUB_OFFLINE=1 pip install --no-deps -e /opt/vibevoice[streamingtts] \
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

COPY *.py /app/
COPY supervisord.conf /app/supervisord.conf

# Bake TTS weights + whisper ASR weights into the image (network ON for this step only;
# runtime stays HF_HUB_OFFLINE=1 via the base ENV so the droplet container never calls HF).
RUN HF_HUB_OFFLINE=0 python /app/preload.py

EXPOSE 8015
CMD ["/usr/local/bin/supervisord", "-c", "/app/supervisord.conf"]
