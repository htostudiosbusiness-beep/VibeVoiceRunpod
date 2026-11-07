# Bake all model weights into the image so the droplet container never hits HF
# at runtime (and cold starts stay fast and deterministic).
import os

from huggingface_hub import snapshot_download

TTS_MODEL = os.getenv("VIBEVOICE_TTS_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
LOCAL_DIR = "/opt/models/vibevoice-realtime"

print(f"Downloading {TTS_MODEL} -> {LOCAL_DIR} ...")
snapshot_download(repo_id=TTS_MODEL, local_dir=LOCAL_DIR, max_workers=4)
print("TTS model baked.")

ASR_MODEL = os.getenv("VIBEVOICE_ASR_MODEL", "base.en")
print(f"Baking faster-whisper {ASR_MODEL} into HF cache ...")
from faster_whisper import WhisperModel  # noqa: E402

WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
print("ASR model baked.")
