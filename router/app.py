"""
app.py - OpenAI-compatible router in front of the Pi cluster and Baseten.

    LOCAL   http://pi-node-1.local:9990   (distributed-llama root node)
    CLOUD   https://...baseten.../v1      (big model)

One endpoint the clients see, two upstreams behind it, and a hard rule that the
client never sees a failure if the cloud is reachable.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
    # or just: python app.py
"""

import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from routing import (BASETEN, CACHE, CLUSTER, Decision, RouterConfig,
                     continuation_body, models_payload, route)


# --------------------------------------------------------------------- config

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class Settings:
    local_base = _env("LOCAL_BASE_URL", "http://pi-node-1.local:9990").rstrip("/")
    cloud_base = _env("CLOUD_BASE_URL", "https://inference.baseten.co/v1").rstrip("/")
    cloud_key = _env("CLOUD_API_KEY", _env("BASETEN_API_KEY", ""))
    status_url = _env("STATUS_URL", "http://pi-node-1.local:9991/status")

    local_model = _env("LOCAL_MODEL", "llama-3.2-3b-instruct")
    cloud_model = _env("CLOUD_MODEL", "zai-org/GLM-5.3")

    size_threshold = int(_env("SIZE_THRESHOLD", "2048"))
    first_token_timeout = float(_env("FIRST_TOKEN_TIMEOUT", "8"))
    read_timeout = float(_env("READ_TIMEOUT", "60"))
    connect_timeout = float(_env("CONNECT_TIMEOUT", "3"))
    status_interval = float(_env("STATUS_INTERVAL", "2"))

    decision_log = _env("DECISION_LOG", "routing_decisions.jsonl")
    cache_file = _env("CACHE_FILE", "demo_cache.json")
    demo_fallback = _env("DEMO_FALLBACK", "1") not in ("0", "false", "no", "")

    # When no supervisor is reachable yet, assume the cluster is fine rather
    # than shipping every request to the cloud. Set to 0 once the supervisor
    # is live and you want strict behaviour.
    assume_healthy = _env("ASSUME_HEALTHY_IF_NO_STATUS", "1") not in ("0", "false", "no")


S = Settings()


def build_config() -> RouterConfig:
    return RouterConfig(
        size_threshold=S.size_threshold,
        cloud_available=bool(S.cloud_base and S.cloud_key),
        local_model=S.local_model,
        cloud_model=S.cloud_model,
    )


class UpstreamError(RuntimeError):
    pass


# ---------------------------------------------------------------------- state

class State:
    cluster_status: str = "unknown"
    status_detail: Dict[str, Any] = {}
    status_last_ok: float = 0.0
    counts: Counter = Counter()          # (upstream, reason) -> n
    fallbacks: Counter = Counter()       # reason -> n
    served: Counter = Counter()          # upstream -> n
    started: float = time.time()
    cache: Dict[str, str] = {}


ST = State()
CLIENT: Optional[httpx.AsyncClient] = None


def log_decision(record: Dict[str, Any]) -> None:
    record["ts"] = time.time()
    record["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    try:
        with open(S.decision_log, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass  # never let logging take down a request


async def poll_status() -> None:
    """Background: keep ST.cluster_status fresh from Person 1's supervisor."""
    global CLIENT
    while True:
        try:
            r = await CLIENT.get(S.status_url, timeout=2.0)
            data = r.json()
            ST.status_detail = data if isinstance(data, dict) else {}
            ST.cluster_status = str(
                ST.status_detail.get("state")
                or ST.status_detail.get("status")
                or "unknown"
            ).lower()
            ST.status_last_ok = time.time()
        except Exception:
            # Supervisor not up yet (it is Person 1's Saturday afternoon task).
            # Treat as healthy while bootstrapping so the router is usable now.
            ST.cluster_status = "healthy" if S.assume_healthy else "unreachable"
        await asyncio.sleep(S.status_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CLIENT
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=16)
    CLIENT = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(
        connect=S.connect_timeout, read=S.read_timeout, write=10.0, pool=10.0))
    if Path(S.cache_file).exists():
        try:
            ST.cache = json.loads(Path(S.cache_file).read_text())
        except Exception:
            ST.cache = {}
    task = asyncio.create_task(poll_status())
    try:
        yield
    finally:
        task.cancel()
        await CLIENT.aclose()


app = FastAPI(title="pi-cluster router", lifespan=lifespan)


# ------------------------------------------------------------------- upstream

def upstream_target(upstream: str) -> Tuple[str, Dict[str, str]]:
    if upstream == BASETEN:
        url = S.cloud_base + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if S.cloud_key:
            headers["Authorization"] = f"Bearer {S.cloud_key}"
        return url, headers
    return S.local_base + "/v1/chat/completions", {"Content-Type": "application/json"}


def _content_of(line: str) -> Optional[str]:
    """Return the text delta in an SSE line, or None if it carries no content."""
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    for choice in obj.get("choices") or []:
        delta = choice.get("delta") or {}
        piece = delta.get("content")
        if piece:
            return piece
    return None


async def sse_stream(upstream: str, body: Dict[str, Any]) -> AsyncIterator[Tuple[bool, str]]:
    """Yield (is_content, raw_sse_line) from an upstream.

    Raises UpstreamError on a bad status. The first-token timeout applies until
    real content appears; after that the slower read timeout applies, because a
    long generation is not a failure.
    """
    url, headers = upstream_target(upstream)
    payload = dict(body)
    payload["stream"] = True
    async with CLIENT.stream("POST", url, json=payload, headers=headers,
                             timeout=httpx.Timeout(connect=S.connect_timeout,
                                                   read=S.read_timeout,
                                                   write=10.0, pool=10.0)) as r:
        if r.status_code >= 400:
            raw = await r.aread()
            raise UpstreamError(f"HTTP {r.status_code}: {raw[:200].decode('utf-8', 'replace')}")
        it = r.aiter_lines().__aiter__()
        got_content = False
        while True:
            limit = S.read_timeout if got_content else S.first_token_timeout
            try:
                line = await asyncio.wait_for(it.__anext__(), timeout=limit)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                raise UpstreamError(
                    "first-token timeout" if not got_content else "stream stalled")
            if not line.strip():
                continue
            piece = _content_of(line)
            if piece:
                got_content = True
            yield (piece is not None, line)


def sse(line: str) -> bytes:
    return (line + "\n\n").encode()


def sse_chunk(text: str, model: str, finish: Optional[str] = None) -> bytes:
    obj = {
        "id": "chatcmpl-router",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": ({"content": text} if text else {}),
                     "finish_reason": finish}],
    }
    return sse("data: " + json.dumps(obj))


# ---------------------------------------------------------------- demo cache

def cache_key(body: Dict[str, Any]) -> str:
    msgs = body.get("messages") or []
    last = ""
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            last = c if isinstance(c, str) else json.dumps(c)
            break
    norm = " ".join(last.lower().split())
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ endpoints

@app.get("/v1/models")
async def list_models():
    return models_payload(build_config())


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "cluster_status": ST.cluster_status,
        "status_age_s": round(time.time() - ST.status_last_ok, 1) if ST.status_last_ok else None,
        "cloud_configured": bool(S.cloud_base),
        "uptime_s": round(time.time() - ST.started, 1),
    }


@app.get("/stats")
async def stats():
    total = sum(ST.served.values())
    local = ST.served.get(CLUSTER, 0)
    return {
        "total_requests": total,
        "served": dict(ST.served),
        "pct_local": round(100.0 * local / total, 1) if total else None,
        "by_reason": {f"{u}:{r}": n for (u, r), n in ST.counts.items()},
        "fallbacks": dict(ST.fallbacks),
        "cluster_status": ST.cluster_status,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)

    cfg = build_config()
    headers = dict(request.headers)
    decision = route(body, headers, ST.cluster_status, cfg)
    ST.counts[(decision.upstream, decision.reason)] += 1

    wants_stream = bool(body.get("stream", False))
    upstream_body = dict(body)
    upstream_body["model"] = decision.model_sent

    if wants_stream:
        return await handle_stream(body, upstream_body, decision, cfg)
    return await handle_blocking(body, upstream_body, decision, cfg)


# ----------------------------------------------------------- blocking variant

def fallback_plan(body: Dict[str, Any], decision: Decision,
                  cfg: RouterConfig) -> List[Tuple[str, Dict[str, Any]]]:
    """Primary upstream first, then the other one as a safety net.

    Both directions matter. Cluster -> cloud covers a dead or restarting Pi.
    Cloud -> cluster covers a Baseten 429 or outage: an escalated request that
    would otherwise 502 gets served, slowly, by the Pis. A forced upstream is
    never second-guessed, because that is the point of forcing it.
    """
    plan: List[Tuple[str, Dict[str, Any]]] = [(decision.upstream, body)]
    if decision.forced:
        return plan
    if decision.upstream == CLUSTER and cfg.cloud_available:
        alt = dict(body)
        alt["model"] = cfg.cloud_model
        plan.append((BASETEN, alt))
    elif decision.upstream == BASETEN and ST.cluster_status == "healthy":
        alt = dict(body)
        alt["model"] = cfg.local_model
        plan.append((CLUSTER, alt))
    return plan



async def handle_blocking(orig: Dict[str, Any], body: Dict[str, Any],
                          decision: Decision, cfg: RouterConfig):
    t0 = time.time()
    attempts = fallback_plan(body, decision, cfg)

    last_err = ""
    for i, (up, payload) in enumerate(attempts):
        url, hdrs = upstream_target(up)
        p = dict(payload)
        p["stream"] = False
        try:
            r = await CLIENT.post(url, json=p, headers=hdrs)
            if r.status_code >= 400:
                raise UpstreamError(f"HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            fell_back = i > 0
            if fell_back:
                ST.fallbacks[f"blocking:{last_err[:40]}"] += 1
            ST.served[up] += 1
            log_decision({**decision.as_log(), "served_by": up, "stream": False,
                          "latency_ms": int((time.time() - t0) * 1000),
                          "fallback": fell_back, "error": last_err or None})
            return JSONResponse(data, headers={
                "X-Served-By": up,
                "X-Route-Reason": decision.reason + ("+fallback" if fell_back else ""),
            })
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"[:200]

    cached = ST.cache.get(cache_key(orig)) if S.demo_fallback else None
    if cached:
        ST.served[CACHE] += 1
        log_decision({**decision.as_log(), "served_by": CACHE, "stream": False,
                      "latency_ms": int((time.time() - t0) * 1000),
                      "fallback": True, "error": last_err})
        return JSONResponse({
            "id": "chatcmpl-cache", "object": "chat.completion",
            "created": int(time.time()), "model": decision.model_sent,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": cached}}],
        }, headers={"X-Served-By": CACHE, "X-Route-Reason": "demo_cache"})

    log_decision({**decision.as_log(), "served_by": "none", "stream": False,
                  "latency_ms": int((time.time() - t0) * 1000),
                  "fallback": True, "error": last_err})
    return JSONResponse({"error": {"message": f"all upstreams failed: {last_err}"}},
                        status_code=502, headers={"X-Served-By": "none"})


# ---------------------------------------------------------- streaming variant

async def handle_stream(orig: Dict[str, Any], body: Dict[str, Any],
                        decision: Decision, cfg: RouterConfig):
    """Two-phase, and the phases matter.

    Phase 1 (pre-commit): we hold response headers until the chosen upstream has
    produced its first real token. Nothing has reached the client yet, so any
    failure here is invisible - we silently retry on the cloud. This covers the
    demo case: cluster hung, supervisor mid-restart, connection refused.

    Phase 2 (post-commit): bytes are on the wire. We can no longer pretend
    nothing happened, so a mid-stream death is recovered by asking the cloud to
    continue from the partial text. The seam is usually invisible; the log says
    it happened.
    """
    t0 = time.time()
    plan = fallback_plan(body, decision, cfg)

    gen = buffered = served_by = None
    last_err = ""
    ttft = None

    for i, (up, payload) in enumerate(plan):
        candidate = sse_stream(up, payload)
        buf: List[str] = []
        try:
            got = False
            async for is_content, line in candidate:
                buf.append(line)
                if is_content:
                    got = True
                    break
            if not got:
                raise UpstreamError("upstream produced no content")
            gen, buffered, served_by = candidate, buf, up
            ttft = int((time.time() - t0) * 1000)
            if i > 0:
                ST.fallbacks[f"pre_commit:{last_err[:40]}"] += 1
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"[:200]
            try:
                await candidate.aclose()
            except Exception:
                pass

    if gen is None:
        return await stream_cache_or_error(orig, decision, last_err, t0)

    fell_back_pre = served_by != decision.upstream
    ST.served[served_by] += 1

    async def body_iter() -> AsyncIterator[bytes]:
        partial: List[str] = []
        try:
            for line in buffered:
                piece = _content_of(line)
                if piece:
                    partial.append(piece)
                yield sse(line)
            async for _is_content, line in gen:
                piece = _content_of(line)
                if piece:
                    partial.append(piece)
                yield sse(line)
            yield sse("data: [DONE]")
            log_decision({**decision.as_log(), "served_by": served_by, "stream": True,
                          "ttft_ms": ttft, "latency_ms": int((time.time() - t0) * 1000),
                          "gen_chars": sum(len(p) for p in partial),
                          "fallback": fell_back_pre, "error": last_err or None})
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:200]
            text = "".join(partial)
            recovered = False
            if cfg.cloud_available and served_by == CLUSTER:
                ST.fallbacks[f"mid_stream:{err[:40]}"] += 1
                try:
                    cont = continuation_body(orig, text, cfg.cloud_model)
                    async for _c, line in sse_stream(BASETEN, cont):
                        yield sse(line)
                    recovered = True
                except Exception as e2:
                    err += f" | continuation failed: {e2}"[:120]
            yield sse("data: [DONE]")
            log_decision({**decision.as_log(), "served_by": served_by, "stream": True,
                          "ttft_ms": ttft, "latency_ms": int((time.time() - t0) * 1000),
                          "gen_chars": len(text), "fallback": True,
                          "mid_stream_error": err, "recovered": recovered})

    return StreamingResponse(body_iter(), media_type="text/event-stream", headers={
        "X-Served-By": served_by,
        "X-Route-Reason": decision.reason + ("+fallback" if fell_back_pre else ""),
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


async def stream_cache_or_error(orig: Dict[str, Any], decision: Decision,
                                last_err: str, t0: float):
    cached = ST.cache.get(cache_key(orig)) if S.demo_fallback else None
    if not cached:
        log_decision({**decision.as_log(), "served_by": "none", "stream": True,
                      "latency_ms": int((time.time() - t0) * 1000),
                      "fallback": True, "error": last_err})
        return JSONResponse({"error": {"message": f"all upstreams failed: {last_err}"}},
                            status_code=502, headers={"X-Served-By": "none"})

    ST.served[CACHE] += 1
    log_decision({**decision.as_log(), "served_by": CACHE, "stream": True,
                  "latency_ms": int((time.time() - t0) * 1000),
                  "fallback": True, "error": last_err})

    async def cached_iter() -> AsyncIterator[bytes]:
        words = cached.split(" ")
        for i, w in enumerate(words):
            yield sse_chunk((" " if i else "") + w, decision.model_sent)
            await asyncio.sleep(0.02)   # so it looks like generation, not a paste
        yield sse_chunk("", decision.model_sent, finish="stop")
        yield sse("data: [DONE]")

    return StreamingResponse(cached_iter(), media_type="text/event-stream", headers={
        "X-Served-By": CACHE, "X-Route-Reason": "demo_cache", "Cache-Control": "no-cache",
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(_env("PORT", "8000")))
