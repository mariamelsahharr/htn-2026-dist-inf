"""
app.py - the HTTP face of the router: validation, endpoints, lifecycle.

    POST /v1/chat/completions   any OpenAI-compatible client
    POST /v1/responses          Codex (Responses API)
    GET  /v1/models  /healthz  /readyz  /stats

Run:  python app.py        (or: uvicorn app:app --host 0.0.0.0 --port 8000)

ROUTER_API_KEY, when set, is required as `Authorization: Bearer <key>` on the two POST
endpoints; the read-only endpoints stay open for the dashboard and health checks.
"""

import asyncio
import contextlib
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx2
import logs
import orjson
import telemetry
from config import Settings, load_settings
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from responses import error_body
from routing import models_payload
from schemas import ChatRequest, ResponsesRequest
from service import Router

__all__ = ["ChatRequest", "ResponsesRequest", "app", "create_app"]

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logs.configure()
    telemetry.init(settings.sentry_dsn.get_secret_value(), settings.sentry_environment)
    async with contextlib.AsyncExitStack() as stack:
        # http2: cloud tiers multiplex on one connection; the Pi root falls back to 1.1
        client = await stack.enter_async_context(
            httpx2.AsyncClient(
                limits=httpx2.Limits(max_connections=64, max_keepalive_connections=16),
                http2=True,
                timeout=settings.timeout(settings.read_timeout),
                transport=app.state.transport,
            )
        )
        router = Router(settings, client)
        stack.push_async_callback(router.aclose)
        app.state.router = router
        for name, verdict in (await router.up.probe_tiers()).items():
            log.info("tier %s: %s", name, verdict)
        async with asyncio.TaskGroup() as tasks:
            background = [tasks.create_task(router.poll_status(), name="status-poller")]
            if router.attestor:
                background.append(tasks.create_task(router.attestor.run(settings.solana_interval), name="attestor"))
            try:
                yield
            finally:
                for task in background:
                    task.cancel()


def _router(request: Request) -> Router:
    return request.app.state.router


def _settings(request: Request) -> Settings:
    return request.app.state.settings


class PayloadTooLargeError(Exception):
    pass


async def read_body(request: Request, limit: int) -> bytes:
    """The raw body, refused past `limit` bytes whether it was declared up front or arrived chunked."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise PayloadTooLargeError
    parts: list[bytes] = []
    size = 0
    async for part in request.stream():
        size += len(part)
        if size > limit:
            raise PayloadTooLargeError
        parts.append(part)
    return b"".join(parts)


async def parse_body(request: Request, model: type[BaseModel]) -> dict[str, Any] | tuple[int, str]:
    """The validated body as the upstream should see it, or (status, message) to refuse it with."""
    try:
        raw = orjson.loads(await read_body(request, _settings(request).max_body_bytes))
    except PayloadTooLargeError:
        return 413, f"body over {_settings(request).max_body_bytes} bytes"
    except (orjson.JSONDecodeError, UnicodeDecodeError):
        return 400, "invalid JSON body"
    try:
        return model.model_validate(raw).model_dump(exclude_unset=True)
    except ValidationError as e:
        first = e.errors()[0]
        return 400, f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"


async def require_api_key(request: Request) -> None:
    """Bearer auth on the generation endpoints, only when ROUTER_API_KEY is configured."""
    key = _settings(request).router_api_key.get_secret_value()
    if not key:
        return
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token.strip().encode(), key.encode()):
        raise HTTPException(status_code=401, detail="invalid API key", headers={"WWW-Authenticate": "Bearer"})


api = APIRouter()


@api.get("/v1/models")
async def list_models(request: Request):
    return models_payload(_router(request).cfg)


@api.get("/healthz")
async def healthz(request: Request):
    """Liveness: the process is up. /readyz says whether anything can answer."""
    rt = _router(request)
    return {
        "ok": True,
        "cluster_status": rt.st.cluster_status,
        "status_age_s": round(time.monotonic() - rt.st.status_last_ok, 1) if rt.st.status_last_ok else None,
        "cloud_configured": bool(rt.cfg.cloud_tiers()),
        "uptime_s": round(time.monotonic() - rt.st.started, 1),
    }


@api.get("/readyz")
async def readyz(request: Request):
    ok, detail = _router(request).ready()
    return JSONResponse({"ready": ok, **detail}, status_code=200 if ok else 503)


@api.get("/stats")
async def stats(request: Request):
    return _router(request).stats()


@api.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(request: Request):
    body = await parse_body(request, ChatRequest)
    if isinstance(body, tuple):
        return JSONResponse({"error": {"message": body[1]}}, status_code=body[0])
    return await _router(request).chat(body, dict(request.headers))


@api.post("/v1/responses", dependencies=[Depends(require_api_key)])
async def responses(request: Request):
    body = await parse_body(request, ResponsesRequest)
    if isinstance(body, tuple):
        return JSONResponse(error_body(body[1], "invalid_request"), status_code=body[0])
    return await _router(request).responses(body, dict(request.headers))


def create_app(settings: Settings | None = None, transport: httpx2.AsyncBaseTransport | None = None) -> FastAPI:
    """The application. `transport` replaces the network for the upstream client (tests)."""
    settings = settings or load_settings()
    app = FastAPI(title="PiHive router", lifespan=lifespan)
    app.state.settings = settings
    app.state.transport = transport
    # The dashboard is its own site; the routing headers must be readable from the browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Served-By", "X-Route-Reason", "X-Request-Id"],
    )
    app.include_router(api)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        code = "invalid_api_key" if exc.status_code == 401 else "invalid_request"
        body = {"error": {"message": exc.detail, "type": "invalid_request_error", "code": code}}
        return JSONResponse(body, status_code=exc.status_code, headers=exc.headers)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), timeout_graceful_shutdown=30)
