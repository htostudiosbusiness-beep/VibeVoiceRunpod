"""RunPod serverless worker for VideoSync VibeVoice (GPU TTS + faster-whisper ASR).

Deployed as a RunPod Serverless endpoint. Scales to zero when idle — GPU cost is
billed per second only while a job runs (plus a short idle window after it).

Replacement for the old `videosync-vibevoice` cloud service. The Rust API
(`vibevoice_client.rs`) detects `https://api.runpod.ai/v2/<endpoint>` URLs and
speaks the queue protocol directly, so both sides agree on this job spec.

Job input (`{"input": {...}}`):

  kind = "tts"
    text       str   narration script (English, single speaker)
    speaker    str   voice preset name (default "Emma")
    format     str   "wav" | "mp3" (default "wav"; mp3 converted via ffmpeg)
    job_id/metadata   pass-through, echoed in the output

  kind = "transcribe"
    audio_url     str   http(s) URL of the audio file
    hotwords      list  optional hint words
    language      str   optional language hint
    context_info  str   optional whisper initial_prompt

Job output:

  tts        -> {"success": True, "provider": "vibevoice", "format": requested,
                 "audio_base64": ..., "duration_seconds": ...}
  transcribe -> {"success": True, "provider": "faster-whisper", "text": ...,
                 "language": ..., "segments": [{"id","start","end","text"}, ...]}

Environment:

  VIBEVOICE_TTS_MODEL   default microsoft/VibeVoice-Realtime-0.5B
  VIBEVOICE_TTS_DTYPE   bfloat16 (default) | float16
  VIBEVOICE_ASR_MODEL   default base.en
  VIBEVOICE_REPO        path of the vendored VibeVoice repo (/opt/vibevoice)
"""

from __future__ import annotations

import base64
import copy
import glob
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx
import torch
import runpod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("videosync.runpod")

VIBEVOICE_REPO = os.getenv("VIBEVOICE_REPO", "/opt/vibevoice")
TTS_MODEL = os.getenv("VIBEVOICE_TTS_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
TTS_DTYPE = os.getenv("VIBEVOICE_TTS_DTYPE", "bfloat16").strip().lower()
ASR_MODEL = os.getenv("VIBEVOICE_ASR_MODEL", "base.en")
ASR_DEVICE = os.getenv("VIBEVOICE_ASR_DEVICE", "cpu")
ASR_CPU_THREADS = int(os.getenv("VIBEVOICE_ASR_CPU_THREADS", "4"))
ASR_COMPUTE_TYPE = os.getenv("VIBEVOICE_ASR_COMPUTE_TYPE", "int8")
MAX_TEXT_CHARS = int(os.getenv("VIBEVOICE_MAX_TEXT_CHARS", "6000"))
MAX_DOWNLOAD_MB = int(os.getenv("VIBEVOICE_MAX_DOWNLOAD_MB", "200"))
DEFAULT_SPEAKER = os.getenv("VIBEVOICE_DEFAULT_SPEAKER", "Emma")
SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# TTS — VibeVoice-Realtime-0.5B on GPU (mirrors the official
# demo/realtime_model_inference_from_file.py flow)
# ---------------------------------------------------------------------------

_tts = {"model": None, "processor": None, "voices": {}}
_tts_lock = threading.Lock()


def _resolve_dtype():
    if TTS_DTYPE == "float16":
        return torch.float16
    return torch.bfloat16


def _tts_init() -> dict:
    if _tts["model"] is not None:
        return _tts

    with _tts_lock:
        if _tts["model"] is not None:
            return _tts

        started = time.monotonic()
        sys.path.insert(0, VIBEVOICE_REPO)
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        processor = VibeVoiceStreamingProcessor.from_pretrained(TTS_MODEL)
        try:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                TTS_MODEL,
                torch_dtype=_resolve_dtype(),
                device_map="cuda",
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
        except Exception:
            logger.warning("vibevoice sdpa load failed; retrying with eager")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                TTS_MODEL,
                torch_dtype=_resolve_dtype(),
                device_map="cuda",
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
        model.eval()
        model.set_ddpm_inference_steps(num_steps=5)

        voices: dict[str, str] = {}
        for pt in sorted(
            glob.glob(f"{VIBEVOICE_REPO}/demo/voices/streaming_model/**/*.pt", recursive=True)
        ):
            voices[Path(pt).stem.lower()] = pt

        _tts.update(model=model, processor=processor, voices=voices)
        logger.info(
            "vibevoice.model.loaded speakers=%d elapsed=%.1fs",
            len(voices),
            time.monotonic() - started,
        )
    return _tts


def _find_voice(state: dict, speaker: str | None) -> str:
    requested = (speaker or DEFAULT_SPEAKER).strip().lower()
    presets = state["voices"]
    if requested in presets:
        return presets[requested]
    matches = [name for name in presets if requested in name or name in requested]
    if len(matches) == 1:
        return presets[matches[0]]
    if matches:
        raise ValueError(
            f"speaker '{speaker}' is ambiguous, matches: {', '.join(sorted(matches))}; be more specific"
        )
    if presets:
        fallback = sorted(presets)[0]
        logger.warning("no voice preset for '%s'; using %s", speaker, fallback)
        return presets[fallback]
    raise FileNotFoundError(f"No VibeVoice voice presets found in {VIBEVOICE_REPO}")


def _convert_format(wav_path: Path, requested: str) -> Path:
    wanted = (requested or "wav").strip().lower() or "wav"
    current = wav_path.suffix.lstrip(".").lower()
    if wanted == current:
        return wav_path
    target = wav_path.with_suffix(f".{wanted}")
    subprocess.run(
        [
            os.getenv("FFMPEG_BIN", "ffmpeg"),
            "-y",
            "-i",
            str(wav_path),
            str(target),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return target


def _run_tts(text: str, speaker: str | None, requested_format: str) -> dict:
    state = _tts_init()
    started = time.monotonic()

    script = text.strip()
    if not script:
        raise ValueError("text is empty")
    script = script.replace("’", "'").replace("“", '"').replace("”", '"')
    if len(script) > MAX_TEXT_CHARS:
        raise ValueError(
            f"text is {len(script)} chars; the streaming model caps ~10 minutes per pass, keep it under {MAX_TEXT_CHARS} chars"
        )

    voice_path = _find_voice(state, speaker)
    all_prefilled_outputs = torch.load(voice_path, map_location="cuda", weights_only=False)

    processor = state["processor"]
    model = state["model"]
    inputs = processor.process_input_with_cached_prompt(
        text=script,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=float(os.getenv("VIBEVOICE_CFG_SCALE", "1.5")),
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
    )

    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        raise RuntimeError("VibeVoice produced no audio output")

    speech = outputs.speech_outputs[0]
    duration = float(speech.shape[-1] if speech.dim() > 0 else len(speech)) / SAMPLE_RATE
    generation_seconds = time.monotonic() - started

    with tempfile.TemporaryDirectory(prefix="vibevoice-tts-") as tmp:
        out_path = Path(tmp) / "generated.wav"
        processor.save_audio(speech, output_path=str(out_path))
        final_path = _convert_format(out_path, requested_format)
        audio_bytes = final_path.read_bytes()

    logger.info(
        "vibevoice.tts.done chars=%d speaker=%s audio=%.1fs gpu=%.1fs",
        len(script),
        speaker,
        duration,
        generation_seconds,
    )
    if duration < 0.4:
        raise RuntimeError("generated audio is suspiciously short (<0.4s)")

    extension = "wav"
    if final_path.suffix.lower().lstrip(".") == "mp3":
        extension = "mp3"

    return {
        "success": True,
        "provider": "vibevoice",
        "format": extension,
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "duration_seconds": round(duration, 3),
    }


# ---------------------------------------------------------------------------
# ASR — faster-whisper (mirrors infra/media_ai/asr_service.py output shape)
# ---------------------------------------------------------------------------

_asr_state: dict = {}
_asr_lock = threading.Lock()


def _asr_init():
    model = _asr_state.get("model")
    if model is None:
        with _asr_lock:
            model = _asr_state.get("model")
            if model is None:
                from faster_whisper import WhisperModel

                started = time.monotonic()
                model = WhisperModel(
                    ASR_MODEL,
                    device=ASR_DEVICE,
                    compute_type=ASR_COMPUTE_TYPE,
                    cpu_threads=ASR_CPU_THREADS,
                    num_workers=1,
                )
                _asr_state["model"] = model
                logger.info(
                    "asr.model.loaded model=%s elapsed=%.1fs", ASR_MODEL, time.monotonic() - started
                )
    return model


def _add_chunk(segments: list, texts: list, words: list) -> None:
    if not words:
        return
    text = "".join(getattr(w, "word", "") or "" for w in words).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    if not text:
        return
    start = getattr(words[0], "start", None)
    end = getattr(words[-1], "end", None)
    if start is None or end is None or float(end) <= float(start):
        return
    segments.append(
        {"id": len(segments), "start": round(float(start), 3), "end": round(float(end), 3), "text": text}
    )
    texts.append(text)


def _transcribe_pass(
    model,
    path: str,
    *,
    language: str | None,
    hotwords: str | None,
    prompt: str | None,
    vad: bool,
) -> tuple[list, list, Any, int]:
    segments_iter, info = model.transcribe(
        path,
        language=language,
        beam_size=5,
        best_of=5,
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": 500} if vad else None,
        word_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        hallucination_silence_threshold=2.0,
        hotwords=hotwords,
        initial_prompt=prompt,
    )

    segments: list = []
    texts: list = []
    raw_segments = 0
    max_words = int(os.getenv("VIBEVOICE_ASR_MAX_WORDS", "5"))

    for segment in segments_iter:
        raw_segments += 1
        if float(getattr(segment, "no_speech_prob", 0.0) or 0.0) > 0.75:
            continue
        words = [w for w in (getattr(segment, "words", None) or []) if (getattr(w, "word", "") or "").strip()]
        if words:
            chunk: list = []
            for word in words:
                chunk.append(word)
                token = (getattr(word, "word", "") or "").strip()
                if len(chunk) >= max_words or token.endswith((".", "!", "?", ",", ";", ":")):
                    _add_chunk(segments, texts, chunk)
                    chunk = []
            if chunk:
                _add_chunk(segments, texts, chunk)
            continue

        text = (getattr(segment, "text", "") or "").strip()
        start = getattr(segment, "start", None)
        end = getattr(segment, "end", None)
        if text and start is not None and end is not None and float(end) > float(start):
            segments.append(
                {"id": len(segments), "start": round(float(start), 3), "end": round(float(end), 3), "text": text}
            )
            texts.append(text)

    return segments, texts, info, raw_segments


def _download_audio(url: str, path: Path) -> None:
    max_bytes = MAX_DOWNLOAD_MB * 1024 * 1024
    total = 0
    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(f"audio exceeds {MAX_DOWNLOAD_MB} MB")
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        raise ValueError(f"audio download failed: {exc}") from exc
    if total == 0:
        raise ValueError("audio file is empty")


def _run_transcribe(audio_url: str, hotwords: list, language: str | None, context_info: str | None) -> dict:
    if not audio_url.startswith(("http://", "https://")):
        raise ValueError("audio_url must be http(s)")

    started = time.monotonic()
    model = _asr_init()
    lang = (language or "").strip() or None
    hw = ", ".join(w.strip() for w in (hotwords or []) if w and w.strip()) or None
    prompt = (context_info or "").strip() or None
    vad = os.getenv("VIBEVOICE_ASR_VAD", "1").lower() not in ("0", "false", "no")

    with tempfile.TemporaryDirectory(prefix="vibevoice-asr-") as tmp:
        audio_path = Path(tmp) / "input_audio"
        _download_audio(audio_url, audio_path)

        try:
            segments, texts, info, raw_segments = _transcribe_pass(
                model, str(audio_path), language=lang, hotwords=hw, prompt=prompt, vad=vad
            )
            vad_fallback = False
        except Exception as exc:
            logger.warning("asr vad pass failed (%s), retrying without VAD", exc)
            segments, texts, info, raw_segments = _transcribe_pass(
                model, str(audio_path), language=lang, hotwords=hw, prompt=prompt, vad=False
            )
            vad_fallback = True

        payload = {
            "success": True,
            "provider": "faster-whisper",
            "model": ASR_MODEL,
            "text": " ".join(texts).strip(),
            "language": getattr(info, "language", None) or lang or "en",
            "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
            "duration": float(getattr(info, "duration", 0.0) or 0.0),
            "segments": segments,
            "vad": vad and not vad_fallback,
            "vad_fallback": vad_fallback,
            "raw_segments": raw_segments,
        }

    logger.info(
        "vibevoice.asr.done segments=%d language=%s audio=%.1fs elapsed=%.1fs",
        len(segments),
        payload.get("language"),
        payload.get("duration", 0.0),
        time.monotonic() - started,
    )
    return payload


# ---------------------------------------------------------------------------
# RunPod worker wiring
# ---------------------------------------------------------------------------


def handler(job: dict) -> dict:
    payload = ((job or {}).get("input") or {}) if isinstance((job or {}).get("input"), dict) else {}
    kind = str(payload.get("kind") or "tts").strip().lower()
    job_id = payload.get("job_id") or str(uuid.uuid4())
    started = time.monotonic()
    logger.info("job.start id=%s kind=%s", job_id, kind)

    if not torch.cuda.is_available():
        raise ValueError("this worker requires a GPU (torch.cuda not available)")
    torch.cuda.empty_cache()

    if kind in ("tts", "tts_base64", "tts_base64_vibevoice"):
        result = _run_tts(
            payload.get("text") or "",
            payload.get("speaker"),
            payload.get("format") or "wav",
        )
    elif kind in ("transcribe", "asr"):
        result = _run_transcribe(
            payload.get("audio_url") or "",
            payload.get("hotwords") or [],
            payload.get("language"),
            payload.get("context_info"),
        )
    else:
        raise ValueError(f"unsupported job kind '{kind}' (expected 'tts' or 'transcribe')")

    result = {**result, "job_id": job_id, "metadata": payload.get("metadata", {})}
    logger.info("job.done id=%s kind=%s elapsed=%.1fs", job_id, kind, time.monotonic() - started)
    return result


def _safe_warmup() -> None:
    """Preload the TTS model after boot so the first real job is warm.

    Runs on a daemon thread so the RunPod heartbeat stays alive while the
    model downloads. Failures are non-fatal (inference happens lazily on
    the first job instead).
    """
    try:
        logger.info("warmup.start model=%s", TTS_MODEL)
        _run_tts("Warm up.", DEFAULT_SPEAKER, "wav")
        logger.info("warmup.done")
    except Exception as exc:
        logger.warning("warmup skipped: %s: %s", type(exc).__name__, exc)


if os.getenv("VIBEVOICE_SKIP_WARMUP", "").lower() not in ("1", "true"):
    threading.Thread(target=_safe_warmup, name="vibevoice-warmup", daemon=True).start()

runpod.serverless.start({"handler": handler})
