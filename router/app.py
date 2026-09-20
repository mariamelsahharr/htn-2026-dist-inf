"""
app.py - the HTTP face of the router: validation, endpoints, lifecycle.

    POST /v1/chat/completions   any OpenAI-compatible client
    POST /v1/responses          Codex (Responses API)
    GET  /v1/models  /healthz  /readyz  /stats

Run:  python app.py        (or: uvicorn app:app --host 0.0.0.0 --port 8000)
"""

import asyncio
import contextlib
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx2
import telemetry
from config import load_settings
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from responses import error_body
from routing import models_payload
from service import Router


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: str | list[Any] | None = None


class ChatRequest(BaseModel):
    """The fields the router reads. Everything else passes through to the upstream untouched."""

    model_config = ConfigDict(extra="allow")
    messages: list[ChatMessage] = Field(min_length=1)
    model: str = "auto"
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=32_768)
    tools: list[Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    telemetry.init(settings.sentry_dsn, settings.sentry_environment)
    # http2: cloud tiers multiplex on one connection; the Pi root falls back to 1.1
    client = httpx2.AsyncClient(
        limits=httpx2.Limits(max_connections=64, max_keepalive_connections=16),
        http2=True,
        timeout=httpx2.Timeout(connect=settings.connect_timeout, read=settings.read_timeout, write=10.0, pool=10.0),
    )
    router = Router(settings, client)
    app.state.router = router
    for name, verdict in (await router.up.probe_tiers()).items():
        print(f"[router] tier {name}: {verdict}", flush=True)
    task = asyncio.create_task(router.poll_status())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await client.aclose()


app = FastAPI(title="PiHive router", lifespan=lifespan)
# The dashboard is its own site. The router has no auth, so open CORS costs nothing;
# the routing headers must be readable from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Served-By", "X-Route-Reason", "X-Request-Id"],
)


def _router(request: Request) -> Router:
    return request.app.state.router


def _bad_request(message: str, responses_api: bool = False) -> JSONResponse:
    body = error_body(message, "invalid_request") if responses_api else {"error": {"message": message}}
    return JSONResponse(body, status_code=400)


@app.middleware("http")
async def cap_body_size(request: Request, call_next):
    limit = _router(request).s.max_body_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        return JSONResponse({"error": {"message": f"body over {limit} bytes"}}, status_code=413)
    return await call_next(request)


@app.get("/v1/models")
async def list_models(request: Request):
    return models_payload(_router(request).cfg)


@app.get("/healthz")
async def healthz(request: Request):
    """Liveness: the process is up. /readyz says whether anything can answer."""
    rt = _router(request)
    return {
        "ok": True,
        "cluster_status": rt.st.cluster_status,
        "status_age_s": round(time.time() - rt.st.status_last_ok, 1) if rt.st.status_last_ok else None,
        "cloud_configured": bool(rt.cfg.cloud_tiers()),
        "uptime_s": round(time.time() - rt.st.started, 1),
    }


@app.get("/readyz")
async def readyz(request: Request):
    ok, detail = _router(request).ready()
    return JSONResponse({"ready": ok, **detail}, status_code=200 if ok else 503)


@app.get("/stats")
async def stats(request: Request):
    return _router(request).stats()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return _bad_request("invalid JSON body")
    try:
        body = ChatRequest.model_validate(raw).model_dump(exclude_none=True)
    except ValidationError as e:  # a subclass of ValueError, so it gets its own block
        first = e.errors()[0]
        return _bad_request(f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}")
    return await _router(request).chat(body, dict(request.headers))


@app.post("/v1/responses")
async def responses(request: Request):
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        return _bad_request("invalid JSON body", responses_api=True)
    if not isinstance(body, dict) or "input" not in body:
        return _bad_request("input is required", responses_api=True)
    return await _router(request).responses(body, dict(request.headers))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), timeout_graceful_shutdown=30)
