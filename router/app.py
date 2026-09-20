"""
app.py - OpenAI-compatible router in front of the Pi cluster and cloud tiers.

    LOCAL   http://192.168.50.13:9990     (distributed-llama root, pi-node-3)
    CLOUD   Baseten via CLOUD_*; optional OPENAI_ / GEMINI_ / SNOWFLAKE_ tiers via
            <NAME>_API_KEY, <NAME>_MODEL, <NAME>_BASE_URL. CLOUD_TIER_ORDER sets
            fallback order; TOOL_TIER (off by default) pins tool requests to a tier.

Serves /v1/chat/completions for everything OpenAI-compatible and /v1/responses
for Codex. The client never sees a failure if any cloud is up.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
    # or just: python app.py
"""

import asyncio
import contextlib
import hashlib
import json
import os
import time
from collections import Counter, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from responses import ResponseBuilder, error_body, responses_to_chat
from routing import (
    BASETEN,
    CACHE,
    CLUSTER,
    DEFAULT_CLOUD_ORDER,
    Breaker,
    Decision,
    RouterConfig,
    Tier,
    cluster_state,
    continuation_body,
    estimate_tokens,
    fallback_chain,
    missing_required_tool_call,
    models_payload,
    route,
)

# ------------------------------------------------------------------- settings

# Base URL defaults per tier (Snowflake's is per-account). A tier exists only with key + model.
EXTRA_TIER_DEFAULTS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "snowflake": "",
}


class Settings(BaseSettings):
    """Every knob, read from env and router/.env (env wins). Field name = env var name."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).with_name(".env")), extra="ignore", env_ignore_empty=True
    )

    local_base_url: str = "http://192.168.50.13:9990"  # pi-node-3, wired; mDNS is not trusted here
    local_model: str = "llama-3.2-3b-instruct"
    status_url: str = "http://192.168.50.13:9991/status"

    cloud_base_url: str = "https://inference.baseten.co/v1"
    cloud_api_key: str = Field("", validation_alias=AliasChoices("CLOUD_API_KEY", "BASETEN_API_KEY"))
    cloud_model: str = "zai-org/GLM-5.3"
    openai_base_url: str = EXTRA_TIER_DEFAULTS["openai"]
    openai_api_key: str = ""
    openai_model: str = ""
    openai_reasoning_effort: str = "none"  # gpt-5.6-luna rejects function tools on chat completions otherwise
    openai_tools_model: str = ""
    gemini_base_url: str = EXTRA_TIER_DEFAULTS["gemini"]
    gemini_api_key: str = ""
    gemini_model: str = ""
    gemini_reasoning_effort: str = ""
    gemini_tools_model: str = ""
    snowflake_base_url: str = ""
    snowflake_api_key: str = ""
    snowflake_model: str = ""
    snowflake_tools_model: str = ""  # e.g. claude-haiku-4-5; Cortex's Llama/Mistral reject tools
    cloud_tools_model: str = ""
    cloud_tier_order: str = ",".join(DEFAULT_CLOUD_ORDER)
    tool_tier: str = ""
    stream_usage_tiers: str = "baseten,openai,gemini"  # verified to return usage on the final streamed chunk

    size_threshold: int = 2048
    first_token_timeout: float = 8.0
    read_timeout: float = 60.0
    local_read_timeout: float = 600.0  # a blocking cluster call returns nothing until generation ends
    local_prefill_tps: float = 25.0  # measured prompt-processing rate; scales the first-token wait
    connect_timeout: float = 3.0
    status_interval: float = 2.0
    min_local_nodes: int = 2  # a degraded cluster below this many nodes routes to cloud
    breaker_failures: int = 2  # consecutive cloud-tier errors before it is skipped ...
    breaker_cooldown: float = 30.0  # ... for this many seconds
    decision_log: str = "routing_decisions.jsonl"
    cache_file: str = "demo_cache.json"
    demo_fallback: bool = True

    @property
    def config(self) -> RouterConfig:
        tiers: dict[str, Tier] = {}
        usage_tiers = {n.strip().lower() for n in self.stream_usage_tiers.split(",")}
        if self.cloud_base_url and self.cloud_api_key:
            tiers[BASETEN] = Tier(
                BASETEN,
                self.cloud_model,
                self.cloud_base_url.rstrip("/"),
                self.cloud_api_key,
                tools_model=self.cloud_tools_model or None,
                usage_in_stream=BASETEN in usage_tiers,
            )
        for name in EXTRA_TIER_DEFAULTS:
            key, model, base = (
                getattr(self, f"{name}_api_key"),
                getattr(self, f"{name}_model"),
                getattr(self, f"{name}_base_url").rstrip("/"),
            )
            if key and model and base:
                tiers[name] = Tier(
                    name,
                    model,
                    base,
                    key,
                    reasoning_effort=getattr(self, f"{name}_reasoning_effort", "") or None,
                    tools_model=getattr(self, f"{name}_tools_model", "") or None,
                    usage_in_stream=name in usage_tiers,
                )
        local = self.local_base_url.rstrip("/")
        if not local.endswith("/v1"):
            local += "/v1"
        order = tuple(n.strip().lower() for n in self.cloud_tier_order.split(",") if n.strip())
        return RouterConfig(
            size_threshold=self.size_threshold,
            cloud_available=BASETEN in tiers,
            local_model=self.local_model,
            cloud_model=self.cloud_model,
            tiers=tiers,
            local_base_url=local,
            cloud_order=order,
            tool_tier=self.tool_tier.lower() or None,
        )


def load_settings() -> Settings:
    """ROUTER_NO_DOTENV=1 ignores router/.env (tests, or a box whose env is the whole config)."""
    if os.environ.get("ROUTER_NO_DOTENV"):
        return Settings(_env_file=None)
    return Settings()


@dataclass
class State:
    cluster_status: str = "unknown"
    status_detail: dict[str, Any] = field(default_factory=dict)
    status_last_ok: float = 0.0
    counts: Counter = field(default_factory=Counter)  # (upstream, reason) -> n
    fallbacks: Counter = field(default_factory=Counter)  # reason -> n
    served: Counter = field(default_factory=Counter)  # upstream -> n
    http_versions: dict[str, str] = field(default_factory=dict)
    breaker: Breaker = field(default_factory=Breaker)
    cache: dict[str, str] = field(default_factory=dict)
    recent: deque = field(default_factory=lambda: deque(maxlen=50))  # last served requests, for rates
    inflight: Counter = field(default_factory=Counter)  # upstream -> requests being answered now
    started: float = field(default_factory=time.time)


class UpstreamError(RuntimeError):
    pass


# Failures that mean "try the next upstream". Anything else is a bug, or the client
# hanging up, and must not trigger a cloud call on its behalf.
UPSTREAM_ERRORS = (httpx2.HTTPError, UpstreamError, asyncio.TimeoutError, ValueError)


# ------------------------------------------------------------------- helpers


def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:200]


def _chunk_of(line: str) -> dict[str, Any] | None:
    """The JSON object in a `data:` line, or None for blanks, [DONE] and non-objects."""
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _error_of(line: str) -> str | None:
    """Upstream error message carried mid-stream as {"error": ...}, else None."""
    obj = _chunk_of(line)
    if obj is not None and "error" in obj and "choices" not in obj:
        return str(obj["error"])[:200]
    return None


def _finished(line: str) -> bool:
    obj = _chunk_of(line)
    return bool(obj) and any(c.get("finish_reason") for c in obj.get("choices") or [] if isinstance(c, dict))


def _content_of(line: str) -> str | None:
    """Text delta in an SSE line; "" for a tool-call delta (arrived, no text); None otherwise."""
    obj = _chunk_of(line)
    if obj is None:
        return None
    for choice in obj.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        piece = delta.get("content")
        if piece:
            return piece
        if delta.get("tool_calls"):
            return ""
    return None


def sse(line: str) -> bytes:
    return (line + "\n\n").encode()


def sse_chunk(text: str, model: str, finish: str | None = None) -> bytes:
    obj = {
        "id": "chatcmpl-router",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": ({"content": text} if text else {}), "finish_reason": finish}],
    }
    return sse("data: " + json.dumps(obj))


def chat_completion(text: str, model: str, id_: str = "chatcmpl-router") -> dict[str, Any]:
    return {
        "id": id_,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
    }


def cache_key(body: dict[str, Any]) -> str:
    """Hash of the normalised last user message; a typo on stage misses the cache."""
    last = ""
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            last = c if isinstance(c, str) else json.dumps(c)
            break
    return hashlib.sha256(" ".join(last.lower().split()).encode()).hexdigest()[:16]


# -------------------------------------------------------------------- router

Attempt = Callable[[str, dict[str, Any]], Awaitable[Any]]


RECENT_FIELDS = (
    "request_id",
    "served_by",
    "routed_to",
    "reason",
    "fallback",
    "stream",
    "latency_ms",
    "ttft_ms",
    "prompt_tokens",
    "gen_tokens",
    "tokens_source",
    "prefill_tps",
    "decode_tps",
    "tps",
    "nodes_active",
    "cluster_state",
    "ts",
)


def _percentile(vals: list[float], pct: float) -> float:
    ordered = sorted(vals)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


class TokenMeter:
    """Token counts and rates for one answer. Exact when the upstream reports usage; the Pi API
    sends one token per chunk so its chunk count is exact too; other tiers fall back to chars/4."""

    def __init__(self, tier: Tier, prompt_estimate: int, t_first: float | None = None):
        self.tier, self.prompt_estimate, self.t_first = tier, prompt_estimate, t_first
        self.t_last: float | None = None
        self.chunks = self.chars = 0
        self.usage: dict[str, Any] | None = None

    def see(self, line: str) -> None:
        obj = _chunk_of(line)
        if obj is None:
            return
        usage = obj.get("usage")
        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
            self.usage = usage
        piece = _content_of(line)
        if piece is not None:
            now = time.time()
            self.t_first = self.t_first or now
            self.t_last = now
            self.chunks += 1
            self.chars += len(piece)

    def see_completion(self, data: dict[str, Any]) -> None:
        usage = data.get("usage")
        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
            self.usage = usage
        self.chars += len(_answer_text(data))

    def result(self, t0: float) -> dict[str, Any]:
        if self.usage:
            gen, source = int(self.usage["completion_tokens"]), "usage"
            prompt = int(self.usage.get("prompt_tokens") or self.prompt_estimate)
        elif self.tier.is_local and self.chunks:
            gen, source, prompt = self.chunks, "chunks", self.prompt_estimate
        else:
            gen, source, prompt = round(self.chars / 4), "chars", self.prompt_estimate
        out: dict[str, Any] = {"gen_tokens": gen, "tokens_source": source, "prompt_tokens_actual": prompt}
        if self.t_first and self.t_first > t0:
            out["prefill_tps"] = round(prompt / (self.t_first - t0), 1)
        if self.t_first and self.t_last and self.t_last > self.t_first and gen > 1:
            out["decode_tps"] = round((gen - 1) / (self.t_last - self.t_first), 1)
        elapsed = time.time() - t0
        if elapsed > 0 and gen:
            out["tps"] = round(gen / elapsed, 1)
        return out


def _answer_text(data: dict[str, Any]) -> str:
    """What a blocking completion said: its text, or its tool calls when there is no text."""
    msg = ((data.get("choices") or [{}])[0]).get("message") or {}
    return msg.get("content") or (json.dumps(msg["tool_calls"], sort_keys=True) if msg.get("tool_calls") else "")


class Router:
    """Settings, live state, the HTTP client, and every request path."""

    def __init__(self, settings: Settings, client: httpx2.AsyncClient) -> None:
        self.s = settings
        self.cfg = settings.config
        self.st = State(breaker=Breaker(settings.breaker_failures, settings.breaker_cooldown))
        self.client = client
        if Path(settings.cache_file).exists():
            try:
                self.st.cache = json.loads(Path(settings.cache_file).read_text())
            except (OSError, ValueError):
                self.st.cache = {}

    # ----- bookkeeping ----------------------------------------------------

    def log(
        self,
        decision: Decision,
        served_by: str,
        *,
        stream: bool,
        t0: float,
        fallback: bool = False,
        error: str | None = None,
        **extra: Any,
    ) -> None:
        record = {
            **decision.as_log(),
            "served_by": served_by,
            "stream": stream,
            "latency_ms": _ms(t0),
            "fallback": fallback,
            "error": error or None,
            **extra,
            "nodes_active": self.st.status_detail.get("nodes_active"),
            "cluster_state": self.st.cluster_status,
            "ts": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        if served_by not in ("none", CACHE):
            self.st.recent.append({k: record[k] for k in RECENT_FIELDS if k in record})
        try:
            with Path(self.s.decision_log).open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            pass  # never let logging take down a request

    @staticmethod
    def headers_for(decision: Decision, served_by: str, fallback: bool) -> dict[str, str]:
        reason = "demo_cache" if served_by == CACHE else decision.reason + ("+fallback" if fallback else "")
        return {"X-Served-By": served_by, "X-Route-Reason": reason, "Cache-Control": "no-cache"}

    async def poll_status(self) -> None:
        """Keep st.cluster_status fresh from the supervisor; without one, ask the root itself."""
        while True:
            try:
                r = await self.client.get(self.s.status_url, timeout=2.0)
                r.raise_for_status()
                data = r.json()
                self.st.status_detail = data if isinstance(data, dict) else {}
                self.st.cluster_status = cluster_state(self.st.status_detail, self.s.min_local_nodes)
                self.st.status_last_ok = time.time()
            except Exception:
                try:
                    self.st.cluster_status = await self.probe_root()
                except Exception:
                    self.st.cluster_status = "unreachable"
            await asyncio.sleep(self.s.status_interval)

    async def probe_root(self) -> str:
        """healthy if the root API answers /v1/models. A timeout keeps the previous verdict:
        dllama-api is single-threaded and simply queues the GET during a generation."""
        try:
            r = await self.client.get(self.cfg.local_tier.base_url + "/models", timeout=2.0)
            return "healthy" if r.status_code == 200 else "unreachable"
        except httpx2.TimeoutException:
            return self.st.cluster_status if self.st.cluster_status != "unknown" else "healthy"
        except httpx2.HTTPError:
            return "unreachable"

    # ----- upstream calls -------------------------------------------------

    def fallback_plan(self, body: dict[str, Any], decision: Decision) -> list[tuple[str, dict[str, Any]]]:
        """(upstream, body) per attempt, in routing.fallback_chain order."""
        return [
            (up, {**body, "model": self.cfg.model_for(up, bool(body.get("tools")))})
            for up in fallback_chain(decision, self.st.cluster_status, self.cfg)
        ]

    async def first_success(
        self, plan: list[tuple[str, dict[str, Any]]], attempt: Attempt
    ) -> tuple[int | None, str | None, Any, str]:
        """Run attempt() down the plan, skipping cloud tiers whose breaker is open (a
        forced upstream, or the only candidate, is always tried).
        Returns (index, upstream, result, last_error)."""
        last_err = ""
        now = time.time()
        for i, (up, payload) in enumerate(plan):
            if up != CLUSTER and len(plan) > 1 and self.st.breaker.is_open(up, now):
                last_err = f"{up}: breaker open"
                continue
            try:
                result = await attempt(up, payload)
            except UPSTREAM_ERRORS as e:
                last_err = _err(e)
                if up != CLUSTER and self.st.breaker.record_failure(up, now):
                    self.st.fallbacks[f"breaker_open:{up}"] += 1
                continue
            if up != CLUSTER:
                self.st.breaker.record_success(up)
            return i, up, result, last_err
        return None, None, None, last_err

    async def post_blocking(self, upstream: str, payload: dict[str, Any]) -> dict[str, Any]:
        tier = self.cfg.tier(upstream)
        read = self.s.local_read_timeout if tier.is_local else self.s.read_timeout
        timeout = httpx2.Timeout(connect=self.s.connect_timeout, read=read, write=10.0, pool=10.0)
        self.st.inflight[upstream] += 1
        try:
            r = await self.client.post(
                tier.chat_url, json=tier.payload(payload, stream=False), headers=tier.headers(), timeout=timeout
            )
        finally:
            self.st.inflight[upstream] -= 1
        self.st.http_versions[upstream] = r.http_version
        if r.status_code >= 400:
            raise UpstreamError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        if tier.is_local and payload.get("tools"):
            msg = ((data.get("choices") or [{}])[0]).get("message") or {}
            if missing_required_tool_call(payload, msg):
                raise UpstreamError("local model answered in prose to a required tool call")
        return data

    def first_token_limit(self, tier: Tier, body: dict[str, Any]) -> float:
        """The cluster prefills at local_prefill_tps, so its first-token wait scales with the prompt."""
        if not tier.is_local:
            return self.s.first_token_timeout
        prompt = estimate_tokens(body.get("messages") or [])
        return max(self.s.first_token_timeout, prompt / max(1.0, self.s.local_prefill_tps) + 5.0)

    async def sse_stream(self, upstream: str, body: dict[str, Any]) -> AsyncIterator[tuple[bool, str]]:
        """Yield (is_content, sse_line). First-token timeout until content, then read timeout."""
        tier = self.cfg.tier(upstream)
        timeout = httpx2.Timeout(connect=self.s.connect_timeout, read=self.s.read_timeout, write=10.0, pool=10.0)
        self.st.inflight[upstream] += 1
        try:
            async for item in self._sse_events(tier, upstream, body, timeout):
                yield item
        finally:
            self.st.inflight[upstream] -= 1

    async def _sse_events(
        self, tier: Tier, upstream: str, body: dict[str, Any], timeout: httpx2.Timeout
    ) -> AsyncIterator[tuple[bool, str]]:
        async with self.client.sse(
            tier.chat_url, method="POST", json=tier.payload(body, stream=True), headers=tier.headers(), timeout=timeout
        ) as source:
            r = source.response
            self.st.http_versions[upstream] = r.http_version
            if r.status_code >= 400:
                raw = await r.aread()
                raise UpstreamError(f"HTTP {r.status_code}: {raw[:200].decode('utf-8', 'replace')}")
            it = source.__aiter__()
            got_content = False
            while True:
                limit = self.s.read_timeout if got_content else self.first_token_limit(tier, body)
                try:
                    event = await asyncio.wait_for(it.__anext__(), timeout=limit)
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    raise UpstreamError("first-token timeout" if not got_content else "stream stalled") from None
                if event.data.strip() == "[DONE]":
                    continue  # relay appends its own terminator
                line = "data: " + event.data
                piece = _content_of(line)
                if piece is not None:
                    got_content = True
                yield (piece is not None, line)

    async def blocking_as_stream(self, upstream: str, payload: dict[str, Any]) -> AsyncIterator[tuple[bool, str]]:
        """Blocking call re-emitted as chat chunks. dllama-api only parses tool calls
        when not streaming, so tool requests to the cluster go this way."""
        data = await self.post_blocking(upstream, payload)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        if missing_required_tool_call(payload, msg):
            raise UpstreamError("local model answered in prose to a required tool call")
        base = {
            "id": data.get("id", "chatcmpl-router"),
            "object": "chat.completion.chunk",
            "created": data.get("created", int(time.time())),
            "model": data.get("model", ""),
        }

        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            return "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

        yield False, chunk({"role": "assistant"})
        if msg.get("content"):
            yield True, chunk({"content": msg["content"]})
        if msg.get("tool_calls"):
            calls = [{**tc, "index": i} for i, tc in enumerate(msg["tool_calls"])]
            yield True, chunk({"tool_calls": calls})
        yield False, chunk({}, choice.get("finish_reason") or "stop")
        if data.get("usage"):
            yield False, "data: " + json.dumps({**base, "choices": [], "usage": data["usage"]})

    async def acquire(self, upstream: str, payload: dict[str, Any]) -> tuple[AsyncIterator, list[str]]:
        """Open a stream and read until the first content line. Nothing has reached
        the client yet, so a failure here can be retried invisibly."""
        if self.cfg.tier(upstream).is_local and payload.get("tools"):
            gen = self.blocking_as_stream(upstream, payload)
        else:
            gen = self.sse_stream(upstream, payload)
        buf: list[str] = []
        try:
            async for is_content, line in gen:
                buf.append(line)
                if is_content:
                    return gen, buf
            raise UpstreamError("upstream produced no content")
        except BaseException:
            await gen.aclose()
            raise

    # ----- request paths --------------------------------------------------

    async def chat(self, body: dict[str, Any], headers: dict[str, str]):
        decision = route(body, headers, self.st.cluster_status, self.cfg)
        self.st.counts[(decision.upstream, decision.reason)] += 1
        t0 = time.time()
        if body.get("stream"):
            return await self.handle_stream(body, decision, t0)
        return await self.handle_blocking(body, decision, t0)

    async def handle_blocking(self, body: dict[str, Any], decision: Decision, t0: float):
        i, up, data, last_err = await self.first_success(self.fallback_plan(body, decision), self.post_blocking)
        if data is None:
            return self.cached_or_error(body, decision, stream=False, last_err=last_err, t0=t0)
        fell_back = i > 0
        if fell_back:
            self.st.fallbacks[f"blocking:{last_err[:40]}"] += 1
        self.st.served[up] += 1
        meter = TokenMeter(self.cfg.tier(up), decision.prompt_tokens)
        meter.see_completion(data)
        self.log(decision, up, stream=False, t0=t0, fallback=fell_back, error=last_err, **meter.result(t0))
        return JSONResponse(data, headers=self.headers_for(decision, up, fell_back))

    async def handle_stream(self, body: dict[str, Any], decision: Decision, t0: float):
        """Pre-commit: hold headers until an upstream produces its first token, retrying
        invisibly. Post-commit: bytes are on the wire, so a mid-stream death is recovered
        by asking another tier to continue from the partial text."""
        i, up, acquired, last_err = await self.first_success(self.fallback_plan(body, decision), self.acquire)
        if acquired is None:
            return self.cached_or_error(body, decision, stream=True, last_err=last_err, t0=t0)
        gen, buffered = acquired
        fell_back = i > 0
        if fell_back:
            self.st.fallbacks[f"pre_commit:{last_err[:40]}"] += 1
        self.st.served[up] += 1
        ttft = _ms(t0)
        relay = self.relay(body, decision, gen, buffered, up, ttft=ttft, t0=t0, fallback=fell_back, last_err=last_err)
        return StreamingResponse(
            relay,
            media_type="text/event-stream",
            headers={**self.headers_for(decision, up, fell_back), "X-Accel-Buffering": "no"},
        )

    async def relay(
        self,
        orig: dict[str, Any],
        decision: Decision,
        gen: AsyncIterator,
        buffered: list[str],
        served_by: str,
        *,
        ttft: int,
        t0: float,
        fallback: bool,
        last_err: str,
    ) -> AsyncIterator[bytes]:
        partial: list[str] = []
        finished = False
        meter = TokenMeter(self.cfg.tier(served_by), decision.prompt_tokens, t_first=t0 + ttft / 1000.0)

        def forward(line: str) -> bytes:
            nonlocal finished
            err = _error_of(line)
            if err:
                raise UpstreamError(err)
            meter.see(line)
            piece = _content_of(line)
            if piece:
                partial.append(piece)
            finished = finished or _finished(line)
            return sse(line)

        try:
            for line in buffered:
                yield forward(line)
            async for _is_content, line in gen:
                yield forward(line)
            yield sse("data: [DONE]")
            self.log(
                decision,
                served_by,
                stream=True,
                t0=t0,
                fallback=fallback,
                error=last_err,
                ttft_ms=ttft,
                gen_chars=sum(len(p) for p in partial),
                **meter.result(t0),
            )
        except UPSTREAM_ERRORS as e:
            err, text, recovered = _err(e), "".join(partial), False
            # no continuation onto a forced upstream, or onto an answer that already finished
            cont_tier = (
                None
                if (decision.forced or finished)
                else next((n for n in self.cfg.cloud_tiers() if n != served_by), None)
            )
            if cont_tier:
                self.st.fallbacks[f"mid_stream:{err[:40]}"] += 1
                try:
                    cont = continuation_body(orig, text, self.cfg.model_for(cont_tier))
                    async for _c, line in self.sse_stream(cont_tier, cont):
                        yield sse(line)
                    recovered = True
                except UPSTREAM_ERRORS as e2:
                    err += f" | continuation failed: {e2}"[:120]
            yield sse("data: [DONE]")
            self.log(
                decision,
                served_by,
                stream=True,
                t0=t0,
                fallback=True,
                ttft_ms=ttft,
                gen_chars=len(text),
                mid_stream_error=err,
                recovered=recovered,
                **meter.result(t0),
            )
        finally:
            with contextlib.suppress(Exception):
                await gen.aclose()  # client hung up: release the upstream connection now

    # ----- Responses API (Codex) -------------------------------------------

    async def responses(self, body: dict[str, Any], headers: dict[str, str]):
        chat, custom = responses_to_chat(body)
        decision = route(chat, headers, self.st.cluster_status, self.cfg)
        self.st.counts[(decision.upstream, decision.reason)] += 1
        t0 = time.time()
        builder = ResponseBuilder(decision.model_sent, custom)
        plan = self.fallback_plan(chat, decision)

        if not body.get("stream"):
            i, up, data, last_err = await self.first_success(plan, self.post_blocking)
            if data is None:
                self.log(decision, "none", stream=False, t0=t0, fallback=True, error=last_err, api="responses")
                return JSONResponse(
                    error_body(f"all upstreams failed: {last_err}"), status_code=502, headers={"X-Served-By": "none"}
                )
            self.st.served[up] += 1
            meter = TokenMeter(self.cfg.tier(up), decision.prompt_tokens)
            meter.see_completion(data)
            self.log(
                decision, up, stream=False, t0=t0, fallback=i > 0, error=last_err, api="responses", **meter.result(t0)
            )
            for _ in builder.feed(data):
                pass
            return JSONResponse(builder.response_object(), headers=self.headers_for(decision, up, i > 0))

        i, up, acquired, last_err = await self.first_success(plan, self.acquire)
        if acquired is None:
            self.log(decision, "none", stream=True, t0=t0, fallback=True, error=last_err, api="responses")
            return JSONResponse(
                error_body(f"all upstreams failed: {last_err}"), status_code=502, headers={"X-Served-By": "none"}
            )
        gen, buffered = acquired
        fell_back = i > 0
        if fell_back:
            self.st.fallbacks[f"pre_commit:{last_err[:40]}"] += 1
        self.st.served[up] += 1
        ttft = _ms(t0)

        meter = TokenMeter(self.cfg.tier(up), decision.prompt_tokens, t_first=t0 + ttft / 1000.0)

        async def events() -> AsyncIterator[bytes]:
            for ev in builder.start():
                yield ev.encode()
            try:
                for line in buffered:
                    meter.see(line)
                    for ev in self._feed_line(builder, line):
                        yield ev.encode()
                async for _is_content, line in gen:
                    meter.see(line)
                    for ev in self._feed_line(builder, line):
                        yield ev.encode()
                for ev in builder.finish():
                    yield ev.encode()
                self.log(
                    decision,
                    up,
                    stream=True,
                    t0=t0,
                    fallback=fell_back,
                    error=last_err,
                    api="responses",
                    ttft_ms=ttft,
                    gen_chars=len("".join(builder.text)),
                    **meter.result(t0),
                )
            except UPSTREAM_ERRORS as e:
                for ev in builder.finish(error=_err(e)):
                    yield ev.encode()
                self.log(
                    decision,
                    up,
                    stream=True,
                    t0=t0,
                    fallback=True,
                    api="responses",
                    ttft_ms=ttft,
                    mid_stream_error=_err(e),
                    recovered=False,
                    **meter.result(t0),
                )
            finally:
                with contextlib.suppress(Exception):
                    await gen.aclose()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={**self.headers_for(decision, up, fell_back), "X-Accel-Buffering": "no"},
        )

    @staticmethod
    def _feed_line(builder: ResponseBuilder, line: str) -> Iterator[str]:
        err = _error_of(line)
        if err:
            raise UpstreamError(err)
        chunk = _chunk_of(line)
        return builder.feed(chunk) if chunk is not None else iter(())

    def cached_or_error(self, orig: dict[str, Any], decision: Decision, *, stream: bool, last_err: str, t0: float):
        cached = self.st.cache.get(cache_key(orig)) if self.s.demo_fallback else None
        if not cached:
            self.log(decision, "none", stream=stream, t0=t0, fallback=True, error=last_err)
            return JSONResponse(
                {"error": {"message": f"all upstreams failed: {last_err}"}},
                status_code=502,
                headers={"X-Served-By": "none"},
            )
        self.st.served[CACHE] += 1
        self.log(decision, CACHE, stream=stream, t0=t0, fallback=True, error=last_err)
        headers = self.headers_for(decision, CACHE, True)
        if not stream:
            return JSONResponse(chat_completion(cached, decision.model_sent, "chatcmpl-cache"), headers=headers)

        async def typed() -> AsyncIterator[bytes]:
            for i, w in enumerate(cached.split(" ")):
                yield sse_chunk((" " if i else "") + w, decision.model_sent)
                await asyncio.sleep(0.02)  # looks like generation, not a paste
            yield sse_chunk("", decision.model_sent, finish="stop")
            yield sse("data: [DONE]")

        return StreamingResponse(typed(), media_type="text/event-stream", headers=headers)

    def stats(self) -> dict[str, Any]:
        total = sum(self.st.served.values())
        local = self.st.served.get(CLUSTER, 0)
        return {
            "total_requests": total,
            "served": dict(self.st.served),
            "pct_local": round(100.0 * local / total, 1) if total else None,
            "pct_by_upstream": {u: round(100.0 * n / total, 1) for u, n in self.st.served.items()} if total else {},
            "tiers": self.cfg.cloud_tiers(),
            "http_versions": dict(self.st.http_versions),
            "by_reason": {f"{u}:{r}": n for (u, r), n in self.st.counts.items()},
            "fallbacks": dict(self.st.fallbacks),
            "cluster_status": self.st.cluster_status,
            "breakers_open_s": self.st.breaker.snapshot(time.time()),
            "rates": self.rates(),
            "inflight": dict(self.st.inflight),
            "recent": list(self.st.recent),
        }

    def rates(self) -> dict[str, dict[str, Any]]:
        """Per-upstream means over the last served requests: what the dashboard shows as tok/s."""
        out: dict[str, dict[str, Any]] = {}
        for up in {r["served_by"] for r in self.st.recent}:
            rows = [r for r in self.st.recent if r["served_by"] == up]
            summary: dict[str, Any] = {"n": len(rows), "inflight": self.st.inflight.get(up, 0)}
            for key in ("decode_tps", "prefill_tps", "tps", "ttft_ms", "latency_ms"):
                vals = [r[key] for r in rows if r.get(key) is not None]
                summary[key] = round(sum(vals) / len(vals), 1) if vals else None
                summary[f"{key}_p50"] = _percentile(vals, 50) if vals else None
                summary[f"{key}_p95"] = _percentile(vals, 95) if vals else None
            out[up] = summary
        return out


# ----------------------------------------------------------------------- app


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    # http2: cloud tiers multiplex on one connection; the Pi root falls back to 1.1
    client = httpx2.AsyncClient(
        limits=httpx2.Limits(max_connections=64, max_keepalive_connections=16),
        http2=True,
        timeout=httpx2.Timeout(connect=settings.connect_timeout, read=settings.read_timeout, write=10.0, pool=10.0),
    )
    router = Router(settings, client)
    app.state.router = router
    task = asyncio.create_task(router.poll_status())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await client.aclose()


app = FastAPI(title="pi-cluster router", lifespan=lifespan)


def _router(request: Request) -> Router:
    return request.app.state.router


@app.get("/v1/models")
async def list_models(request: Request):
    return models_payload(_router(request).cfg)


@app.get("/healthz")
async def healthz(request: Request):
    rt = _router(request)
    return {
        "ok": True,
        "cluster_status": rt.st.cluster_status,
        "status_age_s": round(time.time() - rt.st.status_last_ok, 1) if rt.st.status_last_ok else None,
        "cloud_configured": bool(rt.cfg.cloud_tiers()),
        "uptime_s": round(time.time() - rt.st.started, 1),
    }


@app.get("/stats")
async def stats(request: Request):
    return _router(request).stats()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
    return await _router(request).chat(body, dict(request.headers))


@app.post("/v1/responses")
async def responses(request: Request):
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(error_body("invalid JSON body", "invalid_request"), status_code=400)
    return await _router(request).responses(body, dict(request.headers))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
