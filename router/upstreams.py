"""
upstreams.py - talking to the tiers: blocking and streamed calls, the circuit breaker,
in-flight accounting, the cluster's concurrency cap, and the boot-time probe that
also fetches each provider's model catalog.
"""

import asyncio
import json
import time
from collections import Counter
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx2
import telemetry
from config import Settings
from metering import PrefillEstimate
from routing import CLUSTER, Breaker, RouterConfig, Tier, chat_models, estimate_tokens, missing_required_tool_call
from wire import UPSTREAM_ERRORS, UpstreamError, content_of, err_text

Attempt = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class Won:
    """The attempt that answered: its position in the plan, the tier, and what came back."""

    index: int
    upstream: str
    result: Any

    @property
    def fell_back(self) -> bool:
        return self.index > 0


class Upstreams:
    def __init__(self, settings: Settings, cfg: RouterConfig, client: httpx2.AsyncClient, fallbacks: Counter) -> None:
        self.s, self.cfg, self.client = settings, cfg, client
        self.fallbacks = fallbacks  # shared with the router's stats
        self.breaker = Breaker(settings.breaker_failures, settings.breaker_cooldown)
        self.inflight: Counter = Counter()
        self.http_versions: dict[str, str] = {}
        self.prefill = PrefillEstimate(settings.local_prefill_tps)
        self.cluster_slots = asyncio.Semaphore(max(1, settings.local_concurrency))
        self.cluster_waiting = 0
        self.tier_health: dict[str, str] = {}

    # ----- boot -------------------------------------------------------------

    async def probe_tiers(self) -> dict[str, str]:
        """Ask every cloud tier for its model list. A dead key or host opens its breaker now
        instead of on the first user request; a list fills the tier's catalog."""
        for name in self.cfg.cloud_tiers():
            tier = self.cfg.tiers[name]
            try:
                r = await self.client.get(tier.base_url + "/models", headers=tier.headers(), timeout=10.0)
            except httpx2.HTTPError as e:
                self._mark_down(name, err_text(e))
                continue
            if r.status_code in (401, 403):
                self._mark_down(name, f"HTTP {r.status_code}: key rejected")
                continue
            ids: list[str] = []
            if r.status_code == 200:
                try:
                    ids = [m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id")]
                except (ValueError, AttributeError):
                    ids = []
            if ids:
                self.cfg.catalogs[name] = chat_models(ids)
                self.tier_health[name] = f"ok, {len(self.cfg.catalogs[name])} chat models"
            else:
                self.tier_health[name] = f"ok (HTTP {r.status_code}, no catalog)"
        return dict(self.tier_health)

    def _mark_down(self, name: str, why: str) -> None:
        now = time.time()
        for _ in range(self.s.breaker_failures):
            self.breaker.record_failure(name, now)
        self.tier_health[name] = f"down: {why}"

    # ----- the cluster's one lane -------------------------------------------

    @asynccontextmanager
    async def cluster_slot(self) -> AsyncIterator[None]:
        """The Pi API is single-threaded. A few requests may wait for it; past that the
        request spills to the next tier instead of queueing behind everyone."""
        if self.cluster_slots.locked() and self.cluster_waiting >= self.s.local_queue_max:
            raise UpstreamError(f"cluster busy: {self.cluster_waiting} already waiting")
        self.cluster_waiting += 1
        try:
            await self.cluster_slots.acquire()
        finally:
            self.cluster_waiting -= 1
        try:
            yield
        finally:
            self.cluster_slots.release()

    # ----- calls ------------------------------------------------------------

    async def first_success(self, plan: list[tuple[str, dict[str, Any]]], attempt: Attempt) -> tuple[Won | None, str]:
        """Run attempt() down the plan, skipping cloud tiers whose breaker is open (a forced
        upstream, or the only candidate, is always tried) and stopping once the attempt
        budget is spent. Returns (the winning attempt or None, the last error)."""
        last_err = ""
        started = time.time()
        for i, (up, payload) in enumerate(plan):
            now = time.time()
            if i and now - started > self.s.attempt_budget_s:
                last_err = f"attempt budget of {self.s.attempt_budget_s:.0f}s spent; last: {last_err}"
                break
            if up != CLUSTER and len(plan) > 1 and self.breaker.is_open(up, now):
                last_err = f"{up}: breaker open"
                continue
            with telemetry.chat_span(up, payload.get("model", ""), i) as span:
                try:
                    result = await attempt(up, payload)
                except UPSTREAM_ERRORS as e:
                    last_err = err_text(e)
                    telemetry.mark_failed(span, last_err)
                    if up != CLUSTER and self.breaker.record_failure(up, now):
                        self.fallbacks[f"breaker_open:{up}"] += 1
                    continue
                if isinstance(result, dict):
                    telemetry.record_usage(span, result.get("usage"))
                if up != CLUSTER:
                    self.breaker.record_success(up)
                return Won(i, up, result), last_err
        return None, last_err

    async def post_blocking(self, upstream: str, payload: dict[str, Any]) -> dict[str, Any]:
        tier = self.cfg.tier(upstream)
        if tier.is_local:
            async with self.cluster_slot():
                return await self._post(tier, upstream, payload)
        return await self._post(tier, upstream, payload)

    async def _post(self, tier: Tier, upstream: str, payload: dict[str, Any]) -> dict[str, Any]:
        read = self.s.local_read_timeout if tier.is_local else self.s.read_timeout
        timeout = httpx2.Timeout(connect=self.s.connect_timeout, read=read, write=10.0, pool=10.0)
        self.inflight[upstream] += 1
        try:
            r = await self.client.post(
                tier.chat_url, json=tier.payload(payload, stream=False), headers=tier.headers(), timeout=timeout
            )
        finally:
            self.inflight[upstream] -= 1
        self.http_versions[upstream] = r.http_version
        if r.status_code >= 400:
            raise UpstreamError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        if tier.is_local and payload.get("tools"):
            msg = ((data.get("choices") or [{}])[0]).get("message") or {}
            if missing_required_tool_call(payload, msg):
                raise UpstreamError("local model answered in prose to a required tool call")
        return data

    def first_token_limit(self, tier: Tier, body: dict[str, Any]) -> float:
        """The cluster's first-token wait scales with the prompt at the learned prefill rate."""
        if not tier.is_local:
            return self.s.first_token_timeout
        prompt = estimate_tokens(body.get("messages") or [])
        return max(self.s.first_token_timeout, prompt / self.prefill.value + 5.0)

    async def sse_stream(self, upstream: str, body: dict[str, Any]) -> AsyncGenerator[tuple[bool, str], None]:
        """Yield (is_content, sse_line). First-token timeout until content, then read timeout."""
        tier = self.cfg.tier(upstream)
        timeout = httpx2.Timeout(connect=self.s.connect_timeout, read=self.s.read_timeout, write=10.0, pool=10.0)
        if tier.is_local:
            async with self.cluster_slot():
                async for item in self._counted(tier, upstream, body, timeout):
                    yield item
            return
        async for item in self._counted(tier, upstream, body, timeout):
            yield item

    async def _counted(
        self, tier: Tier, upstream: str, body: dict[str, Any], timeout: httpx2.Timeout
    ) -> AsyncIterator[tuple[bool, str]]:
        self.inflight[upstream] += 1
        try:
            async for item in self._sse_events(tier, upstream, body, timeout):
                yield item
        finally:
            self.inflight[upstream] -= 1

    async def _sse_events(
        self, tier: Tier, upstream: str, body: dict[str, Any], timeout: httpx2.Timeout
    ) -> AsyncIterator[tuple[bool, str]]:
        async with self.client.sse(
            tier.chat_url, method="POST", json=tier.payload(body, stream=True), headers=tier.headers(), timeout=timeout
        ) as source:
            r = source.response
            self.http_versions[upstream] = r.http_version
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
                    continue  # the relay appends its own terminator
                line = "data: " + event.data
                piece = content_of(line)
                if piece is not None:
                    got_content = True
                yield (piece is not None, line)

    async def blocking_as_stream(
        self, upstream: str, payload: dict[str, Any]
    ) -> AsyncGenerator[tuple[bool, str], None]:
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

    async def acquire(
        self, upstream: str, payload: dict[str, Any]
    ) -> tuple[AsyncGenerator[tuple[bool, str], None], list[str]]:
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
