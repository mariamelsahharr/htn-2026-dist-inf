"""
service.py - the Router: takes a request, decides, calls the tiers through Upstreams,
relays the answer, and keeps the numbers the dashboard reads.
"""

import asyncio
import contextlib
import hashlib
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import attest
import httpx2
import telemetry
from config import Settings
from fastapi.responses import JSONResponse, StreamingResponse
from metering import RECENT_FIELDS, TokenMeter, percentile
from responses import ResponseBuilder, error_body, responses_to_chat
from routing import CACHE, CLUSTER, SERVING_STATES, Decision, cluster_state, continuation_body, fallback_chain, route
from state import DecisionLog, State
from streaming import ChatSink, ResponsesSink, Sink
from upstreams import Upstreams, Won
from wire import (
    UPSTREAM_ERRORS,
    answer_text,
    cache_key,
    chat_completion,
    err_text,
    handover_line,
    ms_since,
    sse,
    sse_chunk,
)


class Router:
    def __init__(self, settings: Settings, client: httpx2.AsyncClient) -> None:
        self.s = settings
        self.cfg = settings.config
        self.st = State()
        self.up = Upstreams(settings, self.cfg, client, self.st.fallbacks)
        self.client = client
        self.decisions = DecisionLog(settings.decision_log, settings.decision_log_max_bytes)
        self.records = attest.RecordQueue()
        self.attestor = self.build_attestor()
        if Path(settings.cache_file).exists():
            try:
                self.st.cache = json.loads(Path(settings.cache_file).read_text())
            except (OSError, ValueError):
                self.st.cache = {}

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

    def fresh_status(self) -> dict[str, Any] | None:
        """The last supervisor document, or None once it is older than a few polls."""
        fresh = time.time() - self.st.status_last_ok < 3 * self.s.status_interval
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
        if served_by not in ("none", CACHE):
            self.st.recent.append({k: record[k] for k in RECENT_FIELDS if k in record})
        if served_by == CLUSTER and extra.get("prefill_tps"):
            self.up.prefill.update(float(extra["prefill_tps"]))
        self.decisions.write(record)
        telemetry.log_decision(record)
        if self.attestor:
            self.records.push(record)

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
            self.st.status_last_ok = time.time()
            self.adopt_local_model(self.st.status_detail)
        except Exception:
            try:
                self.st.cluster_status = await self.probe_root()
            except Exception:
                self.st.cluster_status = "unreachable"

    def adopt_local_model(self, status: dict[str, Any]) -> None:
        """The cluster's model name comes from what the supervisor actually loaded, not a setting:
        /home/pi/.../dllama_model_qwen3_0.6b_q40.m is advertised as qwen3_0.6b_q40."""
        path = (status.get("root") or {}).get("model")
        if isinstance(path, str) and path:
            name = Path(path).stem.removeprefix("dllama_model_")
            if name and name != self.cfg.local_model:
                self.cfg.local_model = name

    async def probe_root(self) -> str:
        """healthy if the root API answers /v1/models. A read timeout keeps the previous verdict:
        dllama-api is single-threaded and simply queues the GET during a generation."""
        try:
            r = await self.client.get(self.cfg.local_tier.base_url + "/models", timeout=2.0)
            return "healthy" if r.status_code == 200 else "unreachable"
        except httpx2.ConnectTimeout:
            return "unreachable"  # no host there at all (a laptop off the Pi subnet), not a busy root
        except httpx2.TimeoutException:
            return self.st.cluster_status if self.st.cluster_status != "unknown" else "healthy"
        except httpx2.HTTPError:
            return "unreachable"

    def ready(self) -> tuple[bool, dict[str, Any]]:
        """Can anything answer right now: the cluster in a serving state, or a cloud tier
        whose breaker is closed."""
        now = time.time()
        cloud = [n for n in self.cfg.cloud_tiers() if not self.up.breaker.is_open(n, now)]
        cluster = self.st.cluster_status in SERVING_STATES
        return bool(cloud or cluster), {"cluster": cluster, "cloud_ready": cloud, "tiers": self.up.tier_health}

    # ----- the answer cache -------------------------------------------------

    def cache_get(self, body: dict[str, Any], decision: Decision) -> str | None:
        if self.s.answer_cache_ttl <= 0 or decision.forced or body.get("tools"):
            return None
        hit = self.st.answers.get(f"{decision.model_sent}:{cache_key(body)}")
        return hit[0] if hit and hit[1] > time.time() else None

    def cache_put(self, body: dict[str, Any], decision: Decision, text: str) -> None:
        if self.s.answer_cache_ttl <= 0 or decision.forced or body.get("tools") or not text:
            return
        now = time.time()
        self.st.answers = {k: v for k, v in self.st.answers.items() if v[1] > now}
        self.st.answers[f"{decision.model_sent}:{cache_key(body)}"] = (text, now + self.s.answer_cache_ttl)

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
        telemetry.tag_request(decision, self.st.cluster_status)
        return decision

    async def chat(self, body: dict[str, Any], headers: dict[str, str]):
        decision, t0 = self.decide(body, headers), time.time()
        stream = bool(body.get("stream"))
        cached = self.cache_get(body, decision)
        if cached is not None:
            return self.serve_cached(cached, decision, stream=stream, reason="answer_cache", t0=t0)
        plan = self.fallback_plan(body, decision)
        if not stream:
            won, last_err = await self.answer_blocking(plan, decision, t0)
            if won is None:
                return self.cached_or_error(body, decision, stream=False, last_err=last_err, t0=t0)
            return JSONResponse(won.result, headers=self.headers_for(decision, won.upstream, won.fell_back))
        with telemetry.agent_span(decision, self.st.cluster_status):
            won, last_err = await self.up.first_success(plan, self.up.acquire)
        if won is None:
            return self.cached_or_error(body, decision, stream=True, last_err=last_err, t0=t0)
        return self.stream_response(body, decision, won, ChatSink(), t0=t0, last_err=last_err, continuation=True)

    async def responses(self, body: dict[str, Any], headers: dict[str, str]):
        chat, custom = responses_to_chat(body)
        decision, t0 = self.decide(chat, headers), time.time()
        builder = ResponseBuilder(decision.model_sent, custom)
        plan = self.fallback_plan(chat, decision)
        if not body.get("stream"):
            won, last_err = await self.answer_blocking(plan, decision, t0, api="responses")
            if won is None:
                return self.failed_response(decision, last_err, stream=False, t0=t0)
            for _ in builder.feed(won.result):
                pass
            return JSONResponse(
                builder.response_object(), headers=self.headers_for(decision, won.upstream, won.fell_back)
            )
        with telemetry.agent_span(decision, self.st.cluster_status):
            won, last_err = await self.up.first_success(plan, self.up.acquire)
        if won is None:
            return self.failed_response(decision, last_err, stream=True, t0=t0)
        sink = ResponsesSink(builder)
        return self.stream_response(chat, decision, won, sink, t0=t0, last_err=last_err, api="responses")

    async def answer_blocking(
        self, plan: list[tuple[str, dict[str, Any]]], decision: Decision, t0: float, api: str | None = None
    ) -> tuple[Won | None, str]:
        """One blocking answer down the plan, metered and logged; the caller shapes the response."""
        with telemetry.agent_span(decision, self.st.cluster_status):
            won, last_err = await self.up.first_success(plan, self.up.post_blocking)
        extra = {"api": api} if api else {}
        if won is None:
            self.log(decision, "none", stream=False, t0=t0, fallback=True, error=last_err, **extra)
            return None, last_err
        data: dict[str, Any] = won.result
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
            **extra,
            **meter.result(t0),
        )
        message = ((data.get("choices") or [{}])[0]).get("message") or {}
        self.cache_put(plan[0][1], decision, "" if message.get("tool_calls") else answer_text(data))
        return won, last_err

    def stream_response(
        self,
        orig: dict[str, Any],
        decision: Decision,
        won: Won,
        sink: Sink,
        *,
        t0: float,
        last_err: str,
        api: str | None = None,
        continuation: bool = False,
    ) -> StreamingResponse:
        if won.fell_back:
            self.st.fallbacks[f"pre_commit:{last_err[:40]}"] += 1
        self.st.served[won.upstream] += 1
        telemetry.set_ttft(ms_since(t0))
        body = self.run_stream(
            orig, decision, won, sink, ttft=ms_since(t0), t0=t0, last_err=last_err, api=api, continuation=continuation
        )
        return StreamingResponse(
            body,
            media_type="text/event-stream",
            headers={**self.headers_for(decision, won.upstream, won.fell_back), "X-Accel-Buffering": "no"},
        )

    async def run_stream(
        self,
        orig: dict[str, Any],
        decision: Decision,
        won: Won,
        sink: Sink,
        *,
        ttft: int,
        t0: float,
        last_err: str,
        api: str | None,
        continuation: bool,
    ) -> AsyncIterator[bytes]:
        """Pre-commit is done: bytes go on the wire from here. A mid-stream death is recovered,
        when allowed, by asking another tier to continue from the partial text."""
        gen: AsyncGenerator[tuple[bool, str], None]
        gen, buffered = won.result
        served_by = won.upstream
        extra: dict[str, Any] = {"api": api} if api else {}
        meter = TokenMeter(self.cfg.tier(served_by), decision.prompt_tokens, t_first=t0 + ttft / 1000.0)
        try:
            for chunk in sink.start():
                yield chunk
            for line in buffered:
                meter.see(line)
                for chunk in sink.line(line):
                    yield chunk
            async for _is_content, line in gen:
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
                **extra,
                **meter.result(t0),
            )
            self.cache_put(orig, decision, sink.text)
            for chunk in sink.finish():
                yield chunk
        except UPSTREAM_ERRORS as e:
            err, recovered = err_text(e), False
            # no continuation onto a forced upstream, or onto an answer that already finished
            cont_tier = (
                next((n for n in self.cfg.cloud_tiers() if n != served_by), None)
                if continuation and not (decision.forced or sink.done)
                else None
            )
            if cont_tier:
                self.st.fallbacks[f"mid_stream:{err[:40]}"] += 1
                try:
                    cont = continuation_body(orig, sink.text, self.cfg.model_for(cont_tier))
                    for chunk in sink.line(handover_line(served_by, cont_tier)):
                        yield chunk
                    async for _c, line in self.up.sse_stream(cont_tier, cont):
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
                **extra,
                **meter.result(t0),
            )
            for chunk in sink.finish() if recovered else sink.fail(err):
                yield chunk
        finally:
            with contextlib.suppress(Exception):
                await gen.aclose()  # client hung up: release the upstream connection now

    def failed_response(self, decision: Decision, last_err: str, *, stream: bool, t0: float) -> JSONResponse:
        self.log(decision, "none", stream=stream, t0=t0, fallback=True, error=last_err, api="responses")
        return JSONResponse(
            error_body(f"all upstreams failed: {last_err}"), status_code=502, headers={"X-Served-By": "none"}
        )

    # ----- cached answers ---------------------------------------------------

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

    def cached_or_error(self, orig: dict[str, Any], decision: Decision, *, stream: bool, last_err: str, t0: float):
        cached = self.st.cache.get(cache_key(orig)) if self.s.demo_fallback else None
        if cached:
            return self.serve_cached(cached, decision, stream=stream, reason="demo_cache", t0=t0)
        self.log(decision, "none", stream=stream, t0=t0, fallback=True, error=last_err)
        return JSONResponse(
            {"error": {"message": f"all upstreams failed: {last_err}"}},
            status_code=502,
            headers={"X-Served-By": "none"},
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
            "breakers_open_s": self.up.breaker.snapshot(time.time()),
            "rates": self.rates(),
            "inflight": dict(self.up.inflight),
            "cluster_waiting": self.up.cluster_waiting,
            "local_prefill_tps_estimate": round(self.up.prefill.value, 1),
            "recent": list(self.st.recent),
            "status_age_s": round(time.time() - self.st.status_last_ok, 1) if self.st.status_last_ok else None,
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
