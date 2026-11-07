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
# (older GenerationMixin import path + torch 2.4.1 compatibility). diffusers and
# sentencepiece are in requirements.txt so the hub <1.0 cap protects them.
RUN HF_HUB_OFFLINE=1 pip install --no-deps -e "/opt/vibevoice[streamingtts]"

# Guard: hub must still be <1.0 or transformers 4.51.3 breaks at import.
RUN python -c "import huggingface_hub, transformers; v=huggingface_hub.__version__; assert v[0]=='0', f'huggingface_hub {v} must be <1.0'; print('hub', v, 'transformers', transformers.__version__)"

COPY handler.py /app/handler.py
WORKDIR /app

CMD ["python", "-u", "handler.py"]
