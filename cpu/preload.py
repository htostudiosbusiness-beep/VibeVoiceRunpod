# Bake all model weights into the image so the droplet container never hits HF
# at runtime (and cold starts stay fast and deterministic).
import os

from huggingface_hub import snapshot_download

# HF repo id for build-time baking (distinct from the runtime path env
# VIBEVOICE_TTS_MODEL, which points at the baked local dir).
TTS_MODEL = os.getenv("VIBEVOICE_TTS_HF_REPO", "microsoft/VibeVoice-Realtime-0.5B")
LOCAL_DIR = "/opt/models/vibevoice-realtime"

print(f"Downloading {TTS_MODEL} -> {LOCAL_DIR} ...")
snapshot_download(repo_id=TTS_MODEL, local_dir=LOCAL_DIR, max_workers=4)
print("TTS model baked.")

# The VibeVoice processor loads its text tokenizer by HF repo id from the model's
# config ("language_model_pretrained_name": "Qwen/Qwen2.5-0.5B"). Runtime runs with
# HF_HUB_OFFLINE=1, so bake that tokenizer into the HF cache at build time.
# Tokenizer files only — deliberately exclude the 1GB+ Qwen weights.
QWEN_TOKENIZER_REPO = os.getenv("VIBEVOICE_TOKENIZER_HF_REPO", "Qwen/Qwen2.5-0.5B")
print(f"Baking tokenizer files of {QWEN_TOKENIZER_REPO} into HF cache ...")
snapshot_download(
    repo_id=QWEN_TOKENIZER_REPO,
    allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                    "special_tokens_map.json", "added_tokens.json", "generation_config.json"],
    max_workers=2,
)
print("Tokenizer baked.")

ASR_MODEL = os.getenv("VIBEVOICE_ASR_MODEL", "base.en")
print(f"Baking faster-whisper {ASR_MODEL} into HF cache ...")
from faster_whisper import WhisperModel  # noqa: E402

WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
print("ASR model baked.")
