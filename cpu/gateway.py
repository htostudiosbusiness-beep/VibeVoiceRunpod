#!/usr/bin/env python3
"""Single-port gateway for the CPU VibeVoice stack on the DO Droplet.

    :8015 /health            -> internal TTS service 127.0.0.1:8016 (Rust contract)
    :8015 /api/tts/*         -> internal TTS service 127.0.0.1:8016
    :8015 /api/transcribe    -> internal ASR service 127.0.0.1:8017

Same contract the old nginx gateway provided (infra/media_ai/vibevoice-gateway.conf)
so the Rust API keeps a single VIBEVOICE_SERVICE_URL. Replaces nginx with a 60-line
httpx forwarder — one container, no nginx.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response

TTS_INTERNAL = os.getenv("TTS_INTERNAL_URL", "http://127.0.0.1:8016")
ASR_INTERNAL = os.getenv("ASR_INTERNAL_URL", "http://127.0.0.1:8017")
FORWARD_TIMEOUT = float(os.getenv("GATEWAY_TIMEOUT_SECONDS", "660"))

ttsc = httpx.AsyncClient(base_url=TTS_INTERNAL, timeout=FORWARD_TIMEOUT)
asrc = httpx.AsyncClient(base_url=ASR_INTERNAL, timeout=FORWARD_TIMEOUT)

app = FastAPI(title="VideoSync VibeVoice Gateway", version="1.0.0", docs_url=None, redoc_url=None)


@app.get("/health")
async def health() -> Any:
    resp = await ttsc.get("/health")
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.get("/metrics")
async def metrics() -> Any:
    resp = await ttsc.get("/metrics")
    return Response(content=resp.content, status_code=resp.status_code, media_type="text/plain")


@app.get("/health/asr")
async def health_asr() -> Any:
    resp = await asrc.get("/health")
    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


@app.post("/api/tts")
@app.post("/api/tts/base64")
async def tts(req: Request) -> Any:
    body = await req.body()
    resp = await ttsc.post(
        req.url.path,
        content=body,
        headers={"content-type": req.headers.get("content-type", "application/json"),
                 "authorization": req.headers.get("authorization", "")},
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.post("/api/transcribe")
async def transcribe(req: Request) -> Any:
    body = await req.body()
    resp = await asrc.post(
        req.url.path,
        content=body,
        headers={"content-type": req.headers.get("content-type", "application/json"),
                 "authorization": req.headers.get("authorization", "")},
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )
