# VibeVoice RunPod Worker (GPU, serverless)

Scales-to-zero GPU TTS/ASR for VideoSync. Replaces the old `videosync-vibevoice`
cloud service with the same data contract, executed as RunPod Serverless jobs.

- **TTS**: VibeVoice-Realtime-0.5B (single speaker, English, ~10 min per pass) on
  GPU — near real-time factor instead of CPU batch minutes.
- **ASR**: faster-whisper (same output shape as `infra/media_ai/asr_service.py`,
  i.e. `{text, language, segments[{id,start,end,text}], ...}`).
- The Rust API (`video_editor/src/vibevoice_client.rs`) speaks the RunPod v2
  queue protocol directly when `VIBEVOICE_SERVICE_URL` is set to
  `https://api.runpod.ai/v2/<ENDPOINT_ID>` — no proxy or contract change needed.

## Cost model (why serverless)

2026-11 RunPod serverless pricing, billed **per second while a worker is active**:

| GPU class (serverless tiers) | Rate |
| --- | --- |
| 16 GB class (A4000/A4500/RTX 2000) | ~$0.58/hr |
| 24 GB class (L4/A5000/3090) | ~$0.75/hr |

Idle = 0 workers = $0. Typical narration job (~600 chars) takes ~20–40s GPU
compute → **∼$0.004–$0.01 per generated video narration**. Cold start after
idle: model load into VRAM at worker boot (weights download from HF on first
boot, cached in the warmup thread's HF cache — set `HF_HOME` to a Network
Volume to persist across cold boots).

## Deploy (owner actions)

1. **Image**: the repo workflow `.github/workflows/build-and-push.yml` builds
   and pushes on every push to
   `ghcr.io/htostudiosbusiness-beep/vibevoice-runpod:latest`. Make the GHCR
   package **public** (or give RunPod GHCR read creds when creating the
   endpoint).
2. **RunPod**: create a Serverless Endpoint (`runpod.io/console/serverless`):
   - Image: `ghcr.io/hthuku95/vibevoice-runpod:latest`
   - GPU: **24 GB class recommended** (L4, RTX A5000, 3090). 16 GB class also
     fits (model needs ~3 GB VRAM).
   - Workers: **min 0, max 1** (max 2 if TTS latency during batches matters —
     doubles idle-bill risk, so prefer max 1 for production cost control).
   - Idle timeout: default (5 min).
   - Container disk: 10 GB, Network volume optional (persist `HF_HOME` +
     `VIBEVOICE_REPO` warm caches across boots).
   - Env vars: none required (defaults are correct); optional overrides:
     `VIBEVOICE_TTS_MODEL`, `VIBEVOICE_TTS_DTYPE` (bfloat16 default),
     `VIBEVOICE_ASR_MODEL` (`base.en` default), `VIBEVOICE_SKIP_WARMUP`.
3. **Auth**: create an API key (console > Settings > API Keys).
4. **Render env** (Rust API service):
   - `VIBEVOICE_SERVICE_URL=https://api.runpod.ai/v2/<ENDPOINT_ID>`
   - `VIBEVOICE_SERVICE_API_KEY=<RunPod API key>`

## Local test (after endpoint exists)

```
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/run" \
  -H "Authorization: Bearer <KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"kind": "tts", "text": "Hello from VibeVoice.", "speaker": "Emma", "format": "mp3"}}'
# poll: GET .../status/<job_id> until status == COMPLETED → output.audio_base64
```

## Notes / limitations (from Microsoft)
- English-only (multilingual experimental voices exist but untested here).
- Text with code/formulas/symbols must be pre-cleaned (the Rust agent does
  narration-ready prose, but review QA may still flag odd symbols).
- Sub-3-word inputs can be unstable — fine for narration scripts.
`
