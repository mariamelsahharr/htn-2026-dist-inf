"""
service.py - the Router: takes a request, decides, calls the tiers through Upstreams,
relays the answer, and keeps the numbers the dashboard reads.
"""

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import attest
import httpx2
import logs
import orjson
import telemetry
from cachetools import TTLCache
from config import Settings
from fastapi.responses import JSONResponse, StreamingResponse
from metering import RECENT_FIELDS, TokenMeter, percentile
from responses import ResponseBuilder, responses_to_chat
from routing import CACHE, CLUSTER, SERVING_STATES, Decision, cluster_state, continuation_body, fallback_chain, route
from state import DecisionLog, State
from streaming import Api, ChatApi, ResponsesApi
from upstreams import Stream, Upstreams, Won
from wire import (
    UPSTREAM_ERRORS,
    answer_cache_key,
    answer_text,
    chat_completion,
    demo_cache_key,
    err_text,
    handover_chunk,
    ms_since,
    sse,
    sse_chunk,
)

log = logging.getLogger(__name__)


class Router:
    def __init__(self, settings: Settings, client: httpx2.AsyncClient) -> None:
        self.s = settings
        self.cfg = settings.config
        self.st = State()
        self.st.answers = TTLCache(maxsize=1024, ttl=settings.answer_cache_ttl or 1.0)  # unused when ttl is 0
        self.up = Upstreams(settings, self.cfg, client, self.st.fallbacks)
        self.client = client
        self.decisions = DecisionLog(settings.decision_log, settings.decision_log_max_bytes)
        self.records = attest.RecordQueue()
        self.attestor = self.build_attestor()
        if Path(settings.cache_file).exists():
            try:
                self.st.cache = orjson.loads(Path(settings.cache_file).read_bytes())
            except (OSError, ValueError):
                self.st.cache = {}

    async def aclose(self) -> None:
        """Release what the router owns: the attestor's RPC client and the decision-log writer thread."""
        if self.attestor:
            await self.attestor.chain.close()
        self.decisions.close()

    # ----- bookkeeping ----------------------------------------------------

    def build_attestor(self) -> attest.Attestor | None:
        """On-chain attestation is on when a keypair is configured; it never touches the request path."""
        if not self.s.solana_keypair:
            return None
        program_id = self.s.solana_program_id or attest.default_program_id()
        if not program_id:
            raise ValueError("SOLANA_KEYPAIR is set but no program id: set SOLANA_PROGRAM_ID or build solana/program")
        payer = attest.Keypair.from_json(Path(self.s.solana_keypair).expanduser().read_text())
        chain = attest.Chain(self.s.solana_rpc_url, payer, attest.Pubkey.from_string(program_id))
        return attest.Attestor(chain, self.fresh_status, self.records.read, self.s.attestations_file)

    async def fresh_status(self) -> dict[str, Any] | None:
        """The last supervisor document, or None once it is older than a few polls."""
        fresh = time.monotonic() - self.st.status_last_ok < 3 * self.s.status_interval
        return self.st.status_detail if fresh and self.st.status_detail else None

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
            "latency_ms": ms_since(t0),
            "fallback": fallback,
            "error": error or None,
            **extra,
            "nodes_active": self.st.status_detail.get("nodes_active"),
            "cluster_state": self.st.cluster_status,
            "ts": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        logs.bind(served_by=served_by, reason=record["reason"])
        if served_by not in ("none", CACHE):
            self.st.recent.append({k: record[k] for k in RECENT_FIELDS if k in record})
        if served_by == CLUSTER and extra.get("prefill_tps"):
            self.up.prefill.update(float(extra["prefill_tps"]))
        self.decisions.write(record)
        telemetry.log_decision(record)
        if self.attestor:
            self.records.push(record)
        log.info("served_by=%s latency_ms=%s fallback=%s error=%s", served_by, record["latency_ms"], fallback, error)

    @staticmethod
    def headers_for(decision: Decision, served_by: str, fallback: bool, reason: str | None = None) -> dict[str, str]:
        why = reason or decision.reason + ("+fallback" if fallback else "")
        return {
            "X-Served-By": served_by,
            "X-Route-Reason": why,
            "X-Request-Id": decision.request_id,
            "Cache-Control": "no-cache",
        }

    async def poll_status(self) -> None:
        """Keep st.cluster_status fresh from the supervisor; without one, ask the root itself."""
        while True:
            await self.refresh_status()
            await asyncio.sleep(self.s.status_interval)

    async def refresh_status(self) -> None:
        try:
            r = await self.client.get(self.s.status_url, timeout=2.0)
            r.raise_for_status()
            data = r.json()
            self.st.status_detail = data if isinstance(data, dict) else {}
            self.st.cluster_status = cluster_state(self.st.status_detail, self.s.min_local_nodes)
            self.st.status_last_ok = time.monotonic()
            self.st.status_failures = 0
            self.adopt_local_model(self.st.status_detail)
        except httpx2.TimeoutException:
            # A timeout is what a lossy link looks like, not what a dead root looks like: keep
            # the last verdict for status_grace_s so one dropped poll does not send every
            # request to the cloud. A refused connection falls through to the probe at once.
            self.st.status_failures += 1
            held = self.st.status_last_ok and time.monotonic() - self.st.status_last_ok < self.s.status_grace_s
            if held:
                log.warning(
                    "status poll timed out (%d in a row); keeping %s", self.st.status_failures, self.st.cluster_status
                )
                return
            await self._status_lost(after_grace=True)
        except Exception:
            self.st.status_failures += 1
            await self._status_lost()

    async def _status_lost(self, after_grace: bool = False) -> None:
        """No status document: ask the root itself, and call it unreachable if that fails too.
        Once the grace period has run out, a probe that times out is unreachable as well: the
        link has been dropping for that long, this is no longer a busy root."""
        try:
            self.st.cluster_status = await self.probe_root(timeout_is_busy=not after_grace)
        except Exception:
            self.st.cluster_status = "unreachable"

    def adopt_local_model(self, status: dict[str, Any]) -> None:
        """The cluster's model name comes from what the supervisor actually loaded, not a setting:
        /home/pi/.../dllama_model_qwen3_30b_a3b_q40.m is advertised as qwen3_30b_a3b_q40."""
        path = (status.get("root") or {}).get("model")
        if isinstance(path, str) and path:
            name = Path(path).stem.removeprefix("dllama_model_")
            if name and name != self.cfg.local_model:
                self.cfg.local_model = name

    async def probe_root(self, timeout_is_busy: bool = True) -> str:
        """healthy if the root API answers /v1/models. A read timeout keeps the previous verdict:
        dllama-api is single-threaded and simply queues the GET during a generation."""
        try:
            r = await self.client.get(self.cfg.local_tier.base_url + "/models", timeout=2.0)
            return "healthy" if r.status_code == 200 else "unreachable"
        except httpx2.ConnectTimeout:
            return "unreachable"  # no host there at all (a laptop off the Pi subnet), not a busy root
        except httpx2.TimeoutException:
            if not timeout_is_busy:
                return "unreachable"
            return self.st.cluster_status if self.st.cluster_status != "unknown" else "healthy"
        except httpx2.HTTPError:
            return "unreachable"

    def ready(self) -> tuple[bool, dict[str, Any]]:
        """Can anything answer right now: the cluster in a serving state, or a cloud tier
        whose breaker is closed."""
        now = time.monotonic()
        cloud = [n for n in self.cfg.cloud_tiers() if not self.up.breaker.is_open(n, now)]
        cluster = self.st.cluster_status in SERVING_STATES
        return bool(cloud or cluster), {"cluster": cluster, "cloud_ready": cloud, "tiers": self.up.tier_health}

    # ----- the answer cache -------------------------------------------------

    def cache_get(self, body: dict[str, Any], decision: Decision) -> str | None:
        if self.s.answer_cache_ttl <= 0 or decision.forced or body.get("tools"):
            return None
        return self.st.answers.get(answer_cache_key(body, decision.model_sent))

    def cache_put(self, body: dict[str, Any], decision: Decision, text: str) -> None:
        if self.s.answer_cache_ttl <= 0 or decision.forced or body.get("tools") or not text:
            return
        self.st.answers[answer_cache_key(body, decision.model_sent)] = text

    # ----- request paths --------------------------------------------------

    def fallback_plan(self, body: dict[str, Any], decision: Decision) -> list[tuple[str, dict[str, Any]]]:
        """(upstream, body) per attempt, in routing.fallback_chain order."""
        with_tools = bool(body.get("tools"))
        return [
            (up, {**body, "model": self.cfg.model_for(up, with_tools, decision.model_requested)})
            for up in fallback_chain(decision, self.st.cluster_status, self.cfg)
        ]

    def decide(self, body: dict[str, Any], headers: dict[str, str]) -> Decision:
        decision = replace(route(body, headers, self.st.cluster_status, self.cfg), request_id=uuid.uuid4().hex)
        self.st.counts[(decision.upstream, decision.reason)] += 1
        logs.bind(request_id=decision.request_id, reason=decision.reason, served_by="-")
        telemetry.tag_request(decision, self.st.cluster_status)
        return decision

    async def chat(self, body: dict[str, Any], headers: dict[str, str]):
        return await self.serve(body, headers, ChatApi())

    async def responses(self, body: dict[str, Any], headers: dict[str, str]):
        chat, custom = responses_to_chat(body)
        decision = self.decide(chat, headers)
        return await self.serve(chat, headers, ResponsesApi(ResponseBuilder(decision.model_sent, custom)), decision)

    async def serve(self, body: dict[str, Any], headers: dict[str, str], api: Api, decision: Decision | None = None):
        """The one path every request takes: decide, maybe answer from memory, walk the fallback
        plan blocking or streaming, and shape the outcome the way this API expects."""
        decision, t0 = decision or self.decide(body, headers), time.monotonic()
        stream = bool(body.get("stream"))
        if api.cached:
            cached = self.cache_get(body, decision)
            if cached is not None:
                return self.serve_cached(cached, decision, stream=stream, reason="answer_cache", t0=t0)
        plan = self.fallback_plan(body, decision)
        if not stream:
            won, last_err = await self.answer_blocking(plan, decision, t0, api)
            if won is None:
                return self.failed(body, decision, api, stream=False, last_err=last_err, t0=t0)
            headers = self.headers_for(decision, won.upstream, won.fell_back)
            return JSONResponse(api.blocking(won.result), headers=headers)
        with telemetry.agent_span(decision, self.st.cluster_status):
            opened, last_err = await self.up.first_success(plan, self.up.open_stream)
        if opened is None:
            return self.failed(body, decision, api, stream=True, last_err=last_err, t0=t0)
        return self.stream_response(body, decision, opened, api, t0=t0, last_err=last_err)

    async def answer_blocking(
        self, plan: list[tuple[str, dict[str, Any]]], decision: Decision, t0: float, api: Api
    ) -> tuple[Won[dict[str, Any]] | None, str]:
        """One blocking answer down the plan, metered and logged; the caller shapes the response."""
        with telemetry.agent_span(decision, self.st.cluster_status):
            won, last_err = await self.up.first_success(plan, self.up.post_blocking)
        if won is None:
            self.log(decision, "none", stream=False, t0=t0, fallback=True, error=last_err, api=api.name)
            return None, last_err
        data = won.result
        if won.fell_back:
            self.st.fallbacks[f"blocking:{last_err[:40]}"] += 1
        self.st.served[won.upstream] += 1
        meter = TokenMeter(self.cfg.tier(won.upstream), decision.prompt_tokens)
        meter.see_completion(data)
        self.log(
            decision,
            won.upstream,
            stream=False,
            t0=t0,
            fallback=won.fell_back,
            error=last_err,
            result_sha256=sha256(answer_text(data)),
            api=api.name,
            **meter.result(won.started),
        )
        message = ((data.get("choices") or [{}])[0]).get("message") or {}
        if api.cached:
            self.cache_put(plan[0][1], decision, "" if message.get("tool_calls") else answer_text(data))
        return won, last_err

    def stream_response(
        self, orig: dict[str, Any], decision: Decision, won: Won[Stream], api: Api, *, t0: float, last_err: str
    ) -> StreamingResponse:
        if won.fell_back:
            self.st.fallbacks[f"pre_commit:{last_err[:40]}"] += 1
        self.st.served[won.upstream] += 1
        _gen, buffered = won.result
        first = next((c.at for c in buffered if c.is_content), time.monotonic())
        ttft = int((first - won.started) * 1000)
        telemetry.set_ttft(ttft)
        body = self.run_stream(orig, decision, won, api, ttft=ttft, t0=t0, last_err=last_err)
        return StreamingResponse(
            body,
            media_type="text/event-stream",
            headers={**self.headers_for(decision, won.upstream, won.fell_back), "X-Accel-Buffering": "no"},
        )

    async def run_stream(
        self,
        orig: dict[str, Any],
        decision: Decision,
        won: Won[Stream],
        api: Api,
        *,
        ttft: int,
        t0: float,
        last_err: str,
    ) -> AsyncIterator[bytes]:
        """Pre-commit is done: bytes go on the wire from here. A mid-stream death is recovered,
        when allowed, by asking another tier to continue from the partial text."""
        gen, buffered = won.result
        served_by = won.upstream
        sink = api.sink()
        meter = TokenMeter(self.cfg.tier(served_by), decision.prompt_tokens)
        try:
            for chunk in sink.start():
                yield chunk
            for line in buffered:
                meter.see(line)
                for chunk in sink.line(line):
                    yield chunk
            async for line in gen:
                meter.see(line)
                for chunk in sink.line(line):
                    yield chunk
            self.log(  # before the terminator, so the record exists once the client sees the end
                decision,
                served_by,
                stream=True,
                t0=t0,
                fallback=won.fell_back,
                error=last_err,
                ttft_ms=ttft,
                gen_chars=len(sink.text),
                result_sha256=sha256(sink.text),
                api=api.name,
                **meter.result(won.started),
            )
            if api.cached:
                self.cache_put(orig, decision, sink.text)
            for chunk in sink.finish():
                yield chunk
        except asyncio.CancelledError:
            # the client hung up (Starlette cancels the response task): say so in the log, then let it through
            self.log(
                decision,
                served_by,
                stream=True,
                t0=t0,
                fallback=won.fell_back,
                error="client_disconnected",
                ttft_ms=ttft,
                gen_chars=len(sink.text),
                api=api.name,
                **meter.result(won.started),
            )
            raise
        except UPSTREAM_ERRORS as e:
            err, recovered = err_text(e), False
            # no continuation onto a forced upstream, or onto an answer that already finished
            cont_tier = (
                next((n for n in self.cfg.cloud_tiers() if n != served_by), None)
                if api.continuation and not (decision.forced or sink.done)
                else None
            )
            if cont_tier:
                self.st.fallbacks[f"mid_stream:{err[:40]}"] += 1
                try:
                    cont = continuation_body(orig, sink.text, self.cfg.model_for(cont_tier))
                    for chunk in sink.line(handover_chunk(served_by, cont_tier)):
                        yield chunk
                    async for line in self.up.sse_stream(cont_tier, cont):
                        for chunk in sink.line(line):
                            yield chunk
                    recovered = True
                except UPSTREAM_ERRORS as e2:
                    err += f" | continuation failed: {e2}"[:120]
            self.log(
                decision,
                served_by,
                stream=True,
                t0=t0,
                fallback=True,
                ttft_ms=ttft,
                gen_chars=len(sink.text),
                mid_stream_error=err,
                recovered=recovered,
                api=api.name,
                **meter.result(won.started),
            )
            for chunk in sink.finish() if recovered else sink.fail(err):
                yield chunk
        finally:
            with contextlib.suppress(Exception):
                await gen.aclose()  # returns the upstream connection and the cluster slot now, not at GC

    # ----- cached answers and failures --------------------------------------

    def serve_cached(self, text: str, decision: Decision, *, stream: bool, reason: str, t0: float):
        self.st.served[CACHE] += 1
        self.log(decision, CACHE, stream=stream, t0=t0, fallback=reason == "demo_cache", cache=reason)
        headers = self.headers_for(decision, CACHE, False, reason=reason)
        if not stream:
            return JSONResponse(chat_completion(text, decision.model_sent, "chatcmpl-cache"), headers=headers)

        async def typed() -> AsyncIterator[bytes]:
            for i, w in enumerate(text.split(" ")):
                yield sse_chunk((" " if i else "") + w, decision.model_sent)
                await asyncio.sleep(0.02)  # looks like generation, not a paste
            yield sse_chunk("", decision.model_sent, finish="stop")
            yield sse("data: [DONE]")

        return StreamingResponse(typed(), media_type="text/event-stream", headers=headers)

    def failed(self, orig: dict[str, Any], decision: Decision, api: Api, *, stream: bool, last_err: str, t0: float):
        """Every upstream failed: the demo cache if this API may use it, else a 502 in its error shape."""
        cached = self.st.cache.get(demo_cache_key(orig)) if api.cached and self.s.demo_fallback else None
        if cached:
            return self.serve_cached(cached, decision, stream=stream, reason="demo_cache", t0=t0)
        self.log(decision, "none", stream=stream, t0=t0, fallback=True, error=last_err, api=api.name)
        return JSONResponse(
            api.error(f"all upstreams failed: {last_err}"), status_code=502, headers={"X-Served-By": "none"}
        )

    # ----- numbers ---------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        total = sum(self.st.served.values())
        local = self.st.served.get(CLUSTER, 0)
        return {
            "total_requests": total,
            "served": dict(self.st.served),
            "pct_local": round(100.0 * local / total, 1) if total else None,
            "pct_by_upstream": {u: round(100.0 * n / total, 1) for u, n in self.st.served.items()} if total else {},
            "tiers": self.cfg.cloud_tiers(),
            "tier_health": dict(self.up.tier_health),
            "http_versions": dict(self.up.http_versions),
            "by_reason": {f"{u}:{r}": n for (u, r), n in self.st.counts.items()},
            "fallbacks": dict(self.st.fallbacks),
            "cluster_status": self.st.cluster_status,
            "breakers_open_s": self.up.breaker.snapshot(time.monotonic()),
            "rates": self.rates(),
            "inflight": dict(self.up.inflight),
            "cluster_waiting": self.up.cluster_waiting,
            "local_prefill_tps_estimate": round(self.up.prefill.value, 1),
            "recent": list(self.st.recent),
            "status_age_s": round(time.monotonic() - self.st.status_last_ok, 1) if self.st.status_last_ok else None,
            "status_failures": self.st.status_failures,
            "decision_log_queue_depth": self.decisions.depth,
            "attestor_alive": self.attestor.alive if self.attestor else None,
            "cluster": self.st.status_detail,  # the supervisor document as last seen, for the dashboard
            "solana": self.attestor.summary() if self.attestor else None,
        }

    def rates(self) -> dict[str, dict[str, Any]]:
        """Per-upstream means and percentiles over the last served requests."""
        out: dict[str, dict[str, Any]] = {}
        for up in {r["served_by"] for r in self.st.recent}:
            rows = [r for r in self.st.recent if r["served_by"] == up]
            summary: dict[str, Any] = {"n": len(rows), "inflight": self.up.inflight.get(up, 0)}
            for key in ("decode_tps", "prefill_tps", "tps", "ttft_ms", "latency_ms"):
                vals = [r[key] for r in rows if r.get(key) is not None]
                summary[key] = round(sum(vals) / len(vals), 1) if vals else None
                summary[f"{key}_p50"] = percentile(vals, 50) if vals else None
                summary[f"{key}_p95"] = percentile(vals, 95) if vals else None
            out[up] = summary
        return out


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
