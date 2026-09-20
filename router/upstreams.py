"""
upstreams.py - talking to the tiers: blocking and streamed calls, the circuit breaker,
in-flight accounting, the cluster's concurrency cap, and the boot-time probe that
also fetches each provider's model catalog.
"""

import asyncio
import contextlib
import logging
import time
from collections import Counter
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar

import httpx2
import telemetry
from config import Settings
from metering import PrefillEstimate
from routing import CLUSTER, Breaker, RouterConfig, Tier, chat_models, estimate_tokens, missing_required_tool_call
from wire import UPSTREAM_ERRORS, Chunk, UpstreamError, err_text

log = logging.getLogger(__name__)

T = TypeVar("T")

# An attempt answers with its result and the time.monotonic() it really started: after the
# cluster's queue, so first-token and prefill rates measure the upstream, not the wait.
Attempt: TypeAlias = Callable[[str, dict[str, Any]], Awaitable[tuple[T, float]]]
Stream: TypeAlias = tuple[AsyncGenerator[Chunk, None], list[Chunk]]  # the open stream and what was read so far


@dataclass(frozen=True)
class Won(Generic[T]):
    """The attempt that answered: its position in the plan, the tier, what came back, and when it started."""

    index: int
    upstream: str
    result: T
    started: float

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
        self.cluster_slots = asyncio.Semaphore(settings.local_concurrency)
        self.cluster_waiting = 0
        self.tier_health: dict[str, str] = {}

    # ----- boot -------------------------------------------------------------

    async def probe_tiers(self) -> dict[str, str]:
        """Ask every cloud tier for its model list, all at once. A dead key or host opens its
        breaker now instead of on the first user request; a list fills the tier's catalog."""
        async with asyncio.TaskGroup() as tg:
            for name in self.cfg.cloud_tiers():
                tg.create_task(self._probe(name))
        return dict(self.tier_health)

    async def _probe(self, name: str) -> None:
        tier = self.cfg.tiers[name]
        try:
            r = await self.client.get(tier.base_url + "/models", headers=tier.headers(), timeout=10.0)
        except httpx2.HTTPError as e:
            self._mark_down(name, err_text(e))
            return
        if r.status_code in (401, 403):
            self._mark_down(name, f"HTTP {r.status_code}: key rejected")
            return
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

    def _mark_down(self, name: str, why: str) -> None:
        now = time.monotonic()
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

    async def hold(self, tier: Tier) -> AsyncExitStack:
        """A stack holding the cluster slot for a local tier (nothing for a cloud one)."""
        held = AsyncExitStack()
        if tier.is_local:
            await held.enter_async_context(self.cluster_slot())
        return held

    # ----- calls ------------------------------------------------------------

    async def first_success(
        self, plan: list[tuple[str, dict[str, Any]]], attempt: Attempt[T]
    ) -> tuple[Won[T] | None, str]:
        """Run attempt() down the plan, skipping cloud tiers whose breaker is open (a forced
        upstream, or the only candidate, is always tried) and stopping once the attempt
        budget is spent. Returns (the winning attempt or None, the last error)."""
        last_err = ""
        began = time.monotonic()
        for i, (up, payload) in enumerate(plan):
            now = time.monotonic()
            if i and now - began > self.s.attempt_budget_s:
                last_err = f"attempt budget of {self.s.attempt_budget_s:.0f}s spent; last: {last_err}"
                break
            if up != CLUSTER and len(plan) > 1 and self.breaker.is_open(up, now):
                last_err = f"{up}: breaker open"
                continue
            with telemetry.chat_span(up, payload.get("model", ""), i) as span:
                try:
                    result, started = await attempt(up, payload)
                except UPSTREAM_ERRORS as e:
                    last_err = err_text(e)
                    telemetry.mark_failed(span, last_err)
                    # stamped now, not before the attempt: a slow failure must open the breaker for the full cooldown
                    if up != CLUSTER and self.breaker.record_failure(up, time.monotonic()):
                        self.fallbacks[f"breaker_open:{up}"] += 1
                    continue
                if isinstance(result, dict):
                    telemetry.record_usage(span, result.get("usage"))
                if up != CLUSTER:
                    self.breaker.record_success(up)
                return Won(i, up, result, started), last_err
        return None, last_err

    async def post_blocking(self, upstream: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        tier = self.cfg.tier(upstream)
        async with await self.hold(tier):
            started = time.monotonic()
            return await self._post(tier, upstream, payload), started

    async def _post(self, tier: Tier, upstream: str, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = self.s.timeout(self.s.local_read_timeout if tier.is_local else self.s.read_timeout)
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
        try:
            data = r.json()
        except ValueError as e:
            raise UpstreamError(f"non-JSON body: {r.text[:120]!r}") from e
        if not isinstance(data, dict):
            raise UpstreamError(f"non-object body: {r.text[:120]!r}")
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

    async def open_stream(self, upstream: str, payload: dict[str, Any]) -> tuple[Stream, float]:
        """Take the cluster slot if needed, open the stream and read until the first content chunk.
        Nothing has reached the client yet, so a failure here can be retried invisibly."""
        tier = self.cfg.tier(upstream)
        held = await self.hold(tier)
        gen = self._stream(tier, upstream, payload, held)  # owns `held` once it runs, which is right now
        buf: list[Chunk] = []
        try:
            started = time.monotonic()
            async for chunk in gen:
                buf.append(chunk)
                if chunk.is_content:
                    return (gen, buf), started
            raise UpstreamError("upstream produced no content")
        except BaseException:
            await gen.aclose()
            await held.aclose()  # a no-op when the generator already released it
            raise

    async def sse_stream(self, upstream: str, body: dict[str, Any]) -> AsyncGenerator[Chunk, None]:
        """A whole stream, slot and accounting included: what a mid-stream continuation reads."""
        tier = self.cfg.tier(upstream)
        held = await self.hold(tier)
        async with held, contextlib.aclosing(self._stream(tier, upstream, body, held)) as chunks:
            async for chunk in chunks:
                yield chunk

    async def _stream(
        self, tier: Tier, upstream: str, body: dict[str, Any], held: AsyncExitStack
    ) -> AsyncGenerator[Chunk, None]:
        """One owned stream: closing it closes the upstream response, releases the slot and
        decrements inflight, in that order, whether it finished, failed or the client hung up."""
        source = (
            self._blocking_as_stream(tier, upstream, body)
            if tier.is_local and body.get("tools")
            else self._sse_events(tier, upstream, body)
        )
        self.inflight[upstream] += 1
        try:
            async with held, contextlib.aclosing(source) as chunks:
                async for chunk in chunks:
                    yield chunk
        finally:
            self.inflight[upstream] -= 1

    async def _sse_events(self, tier: Tier, upstream: str, body: dict[str, Any]) -> AsyncGenerator[Chunk, None]:
        """First-token timeout until content, then read timeout."""
        first_limit = self.first_token_limit(tier, body)
        async with self.client.sse(
            tier.chat_url,
            method="POST",
            json=tier.payload(body, stream=True),
            headers=tier.headers(),
            timeout=self.s.timeout(self.s.read_timeout),
        ) as source:
            r = source.response
            self.http_versions[upstream] = r.http_version
            if r.status_code >= 400:
                raw = await r.aread()
                raise UpstreamError(f"HTTP {r.status_code}: {raw[:200].decode('utf-8', 'replace')}")
            it = source.__aiter__()
            got_content = False
            while True:
                limit = self.s.read_timeout if got_content else first_limit
                try:
                    event = await asyncio.wait_for(it.__anext__(), timeout=limit)
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    raise UpstreamError("first-token timeout" if not got_content else "stream stalled") from None
                if event.data.strip() == "[DONE]":
                    continue  # the relay appends its own terminator
                chunk = Chunk.parse("data: " + event.data)
                got_content = got_content or chunk.is_content
                yield chunk

    async def _blocking_as_stream(
        self, tier: Tier, upstream: str, payload: dict[str, Any]
    ) -> AsyncGenerator[Chunk, None]:
        """Blocking call re-emitted as chat chunks. dllama-api only parses tool calls
        when not streaming, so tool requests to the cluster go this way. _post already
        rejected prose to a required tool call."""
        data = await self._post(tier, upstream, payload)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        base = {
            "id": data.get("id", "chatcmpl-router"),
            "object": "chat.completion.chunk",
            "created": data.get("created", int(time.time())),
            "model": data.get("model", ""),
        }

        def chunk(delta: dict[str, Any], finish: str | None = None) -> Chunk:
            return Chunk.of({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

        yield chunk({"role": "assistant"})
        if msg.get("content"):
            yield chunk({"content": msg["content"]})
        if msg.get("tool_calls"):
            calls = [{**tc, "index": i} for i, tc in enumerate(msg["tool_calls"])]
            yield chunk({"tool_calls": calls})
        yield chunk({}, choice.get("finish_reason") or "stop")
        if data.get("usage"):
            yield Chunk.of({**base, "choices": [], "usage": data["usage"]})
