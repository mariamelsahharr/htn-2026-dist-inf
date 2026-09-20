"""
test_integration.py - end-to-end proof that the cascade actually cascades.

Fake LOCAL, CLOUD and STATUS endpoints live on one httpx2 MockTransport whose behaviour
a test switches at will; the router runs in-process behind an ASGITransport with its
lifespan, so nothing listens on a port. Every failure path is exercised: refused
connection, hung upstream (first-token timeout), a stream that dies halfway through,
the breaker, the demo cache, both APIs, tool calls, metering and attestation wiring.

Run: pytest -q test_integration.py
"""

import asyncio
import hashlib
import json
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import attest
import httpx2
import pytest_asyncio
from app import create_app
from asgi_lifespan import LifespanManager
from config import Settings
from fastapi import FastAPI
from service import Router
from test_attest import FakeChain
from wire import demo_cache_key

BASE = "http://fake"


# ----------------------------------------------------------------- fake stack


@dataclass
class Mode:
    local: str = "ok"
    cloud: str = "ok"
    status: str = "healthy"
    nodes: int = 4


async def _sse(words, model, die_after=None, stall=False, usage=False) -> AsyncIterator[bytes]:
    if stall:
        await asyncio.sleep(30)
    for i, w in enumerate(words):
        if die_after is not None and i >= die_after:
            raise httpx2.RemoteProtocolError("simulated upstream death")
        chunk = {"choices": [{"index": 0, "delta": {"content": w + " "}, "finish_reason": None}], "model": model}
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        await asyncio.sleep(0.01)
    yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
    if usage:  # what OpenAI/Baseten/Gemini send for stream_options.include_usage
        yield f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 11, 'completion_tokens': 40}})}\n\n".encode()
    yield b"data: [DONE]\n\n"


def _tool_call_completion(which: str, name: str) -> httpx2.Response:
    args = json.dumps({"input": "*** patch ***"} if name == "apply_patch" else {"command": ["ls"]})
    return httpx2.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "model": which,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_fake", "type": "function", "function": {"name": name, "arguments": args}}
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        },
    )


def fake_stack(mode: Mode):
    async def chat(request: httpx2.Request, which: str) -> httpx2.Response:
        body = json.loads(request.content)
        state = getattr(mode, which)
        if state == "refuse":
            return httpx2.Response(503, json={"error": "upstream is sad"})
        words = [f"{which}{i}" for i in range(8)]
        if not body.get("stream"):
            if body.get("tools") and state != "prose":  # "prose": ignore the tools like a weak local model
                return _tool_call_completion(which, body["tools"][0]["function"]["name"])  # blocking path only
            return httpx2.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "model": which,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": " ".join(words)},
                        }
                    ],
                },
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                words,
                which,
                die_after=3 if state == "die_midstream" else None,
                stall=state == "hang",
                usage=bool((body.get("stream_options") or {}).get("include_usage")),
            ),
        )

    async def handler(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path == "/status":
            if mode.status == "gone":  # no supervisor at all
                return httpx2.Response(500, json={"error": "no supervisor"})
            return httpx2.Response(200, json={"state": mode.status, "nodes_active": mode.nodes, "nodes_total": 4})
        if path == "/cloud/models":
            return httpx2.Response(
                200, json={"data": [{"id": "cloud-default"}, {"id": "cloud-fancy"}, {"id": "text-embedding-3"}]}
            )
        if path == "/local/v1/models":
            if mode.local == "refuse":
                return httpx2.Response(503, json={"error": "down"})
            return httpx2.Response(200, json={"data": [{"id": "local-3b", "object": "model"}]})
        if path == "/local/v1/chat/completions":
            return await chat(request, "local")
        if path == "/cloud/chat/completions":
            return await chat(request, "cloud")
        return httpx2.Response(404, json={"error": f"no fake for {path}"})

    return handler


# ------------------------------------------------------------------- harness


def settings_for(tmp_path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        breaker_failures=2,
        breaker_cooldown=1.0,
        local_base_url=f"{BASE}/local",
        cloud_base_url=f"{BASE}/cloud",
        cloud_api_key="test-key",
        status_url=f"{BASE}/status",
        first_token_timeout=2,
        status_interval=0.5,
        size_threshold=2048,
        decision_log=str(tmp_path / "decisions.jsonl"),
        answer_cache_ttl=0,  # identical prompts below must hit the upstreams every time
        max_body_bytes=60000,
        cache_file=str(tmp_path / "cache.json"),
        local_model="local-3b",
        cloud_model="cloud-70b",
        **overrides,
    )


@dataclass
class Stack:
    mode: Mode
    client: httpx2.AsyncClient
    app: FastAPI
    decisions: Path

    @property
    def router(self) -> Router:
        return self.app.state.router

    async def set_mode(self, **kw) -> None:
        """Switch the fakes and take a fresh status reading, as the poller would."""
        for k, v in kw.items():
            setattr(self.mode, k, v)
        await self.router.refresh_status()

    async def stream_text(self, body, headers=None):
        """(served_by, reason, concatenated text) of a streamed chat completion."""
        r = await self.client.post("/v1/chat/completions", json={**body, "stream": True}, headers=headers or {})
        out = []
        for line in r.text.splitlines():
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                obj = json.loads(p)
            except ValueError:
                continue
            for c in obj.get("choices") or []:
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    out.append(piece)
        return r.headers.get("X-Served-By"), r.headers.get("X-Route-Reason"), "".join(out)

    async def responses_events(self, body, headers=None):
        r = await self.client.post("/v1/responses", json={**body, "stream": True}, headers=headers or {})
        evs = [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]
        return r.headers.get("X-Served-By"), evs

    async def records(self) -> list[dict]:
        await asyncio.to_thread(self.router.decisions.flush)
        return [json.loads(line) for line in self.decisions.read_text().strip().splitlines()]

    async def last_record(self) -> dict:
        return (await self.records())[-1]

    async def stats(self) -> dict:
        return (await self.client.get("/stats")).json()


@pytest_asyncio.fixture
async def stack(tmp_path):
    mode = Mode()
    settings = settings_for(tmp_path)
    app = create_app(settings, transport=httpx2.MockTransport(fake_stack(mode)))
    async with (
        LifespanManager(app) as manager,
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app=manager.app), base_url="http://router") as c,
    ):
        s = Stack(mode, c, app, tmp_path / "decisions.jsonl")
        await s.router.refresh_status()
        yield s


# ------------------------------------------------------------------- routing


async def test_routing_rules(stack: Stack):
    served, reason, text = await stack.stream_text({"messages": [{"role": "user", "content": "hi"}]})
    assert served == "cluster", f"small prompt, healthy cluster -> local ({served}/{reason})"
    assert "local0" in text, "local content actually streamed"
    raw = (
        await stack.client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
    ).text
    assert raw.count("data: [DONE]") == 1, "exactly one [DONE] terminator per stream"
    assert "cloud" not in raw, "no cloud continuation after a finished local answer"

    served, reason, _ = await stack.stream_text({"messages": [{"role": "user", "content": "x" * 20000}]})
    assert served == "baseten" and reason == "over_size_threshold", "huge prompt -> baseten"

    served, reason, _ = await stack.stream_text(
        {"messages": [{"role": "user", "content": "hi"}], "model": "local-3b-heavy"}
    )
    assert served == "baseten" and reason == "heavy_model_tag", "heavy model tag -> baseten"

    served, _, _ = await stack.stream_text(
        {"messages": [{"role": "user", "content": "hi"}]}, headers={"X-Escalate": "true"}
    )
    assert served == "baseten", "escalate header -> baseten"

    served, _, _ = await stack.stream_text(
        {"messages": [{"role": "user", "content": "x" * 20000}]}, headers={"X-Force-Upstream": "cluster"}
    )
    assert served == "cluster", "force header overrides size rule"


async def test_cluster_health(stack: Stack):
    await stack.set_mode(status="restarting")
    served, reason, _ = await stack.stream_text({"messages": [{"role": "user", "content": "hi"}]})
    assert served == "baseten" and reason == "cluster_restarting", "supervisor says restarting -> baseten"
    await stack.set_mode(status="degraded", nodes=2)
    served, reason, _ = await stack.stream_text({"messages": [{"role": "user", "content": "hi"}]})
    assert served == "cluster", f"degraded on 2 nodes -> still served by the cluster ({reason})"
    await stack.set_mode(status="degraded", nodes=1)
    served, reason, _ = await stack.stream_text({"messages": [{"role": "user", "content": "hi"}]})
    assert served == "baseten" and reason == "cluster_degraded_below_min", "degraded to root alone -> baseten"
    await stack.set_mode(status="healthy", nodes=4)

    await stack.set_mode(status="gone", local="ok")
    served, reason, _ = await stack.stream_text({"messages": [{"role": "user", "content": "hi"}]})
    assert served == "cluster", f"no supervisor but root answers -> still local ({reason})"
    await stack.set_mode(status="gone", local="refuse")
    served, reason, _ = await stack.stream_text({"messages": [{"role": "user", "content": "hi"}]})
    assert served == "baseten" and reason == "cluster_unreachable", "no supervisor and root dead -> cloud, no wait"


async def test_pre_commit_fallback_is_invisible_to_the_client(stack: Stack):
    await stack.set_mode(local="refuse")
    served, reason, text = await stack.stream_text({"messages": [{"role": "user", "content": "hi"}]})
    assert served == "baseten" and "fallback" in (reason or ""), "local 503 -> transparent baseten retry"
    assert "cloud0" in text and "cloud7" in text, "fallback response is complete"

    await stack.set_mode(local="hang")
    t0 = time.monotonic()
    served, reason, text = await stack.stream_text({"messages": [{"role": "user", "content": "hi"}]})
    el = time.monotonic() - t0
    assert served == "baseten", f"local hangs -> first-token timeout -> baseten ({reason})"
    assert 1.5 < el < 6, f"timeout fired near FIRST_TOKEN_TIMEOUT ({el:.1f}s)"


async def test_mid_stream_death_is_continued_by_the_cloud(stack: Stack):
    await stack.set_mode(local="die_midstream")
    served, _, text = await stack.stream_text({"messages": [{"role": "user", "content": "hi"}]})
    assert served == "cluster", "stream still committed to cluster"
    assert "local0" in text, "partial local output preserved"
    assert "cloud" in text, "cloud continuation appended"

    raw = (
        await stack.client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "keep going"}], "stream": True}
        )
    ).text
    chunks = [json.loads(line[5:]) for line in raw.splitlines() if line.startswith("data:") and "[DONE]" not in line]
    markers = [i for i, c in enumerate(chunks) if c.get("pihive", {}).get("continued_by") == "baseten"]
    first_cloud = next(i for i, c in enumerate(chunks) if "cloud0" in json.dumps(c))
    assert markers and markers[0] < first_cloud, "a mid-answer handover is announced in the stream, before cloud text"
    assert chunks[markers[0]]["pihive"]["after"] == "cluster" and chunks[markers[0]]["choices"] == []


async def test_required_tool_call_answered_in_prose(stack: Stack):
    await stack.set_mode(local="prose", cloud="ok", status="healthy")
    req = {
        "messages": [{"role": "user", "content": "list files"}],
        "tool_choice": "required",
        "tools": [
            {"type": "function", "function": {"name": "shell", "parameters": {"type": "object", "properties": {}}}}
        ],
    }
    r = await stack.client.post("/v1/chat/completions", json=req)
    assert (
        r.headers.get("X-Served-By") == "baseten"
        and "+fallback" in r.headers.get("X-Route-Reason", "")
        and r.json()["choices"][0]["message"].get("tool_calls")
    ), "tool_choice=required + local prose -> falls through to the cloud's tool call"
    r = await stack.client.post("/v1/chat/completions", json={**req, "tool_choice": "auto"})
    assert r.headers.get("X-Served-By") == "cluster" and r.json()["choices"][0]["message"].get("content"), (
        "tool_choice=auto + local prose -> prose is a valid answer, stays local"
    )


async def test_circuit_breaker_on_a_dead_cloud_tier(stack: Stack):
    await stack.set_mode(local="refuse", cloud="refuse")
    for _ in range(2):  # two failures open the breaker (BREAKER_FAILURES=2)
        await stack.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "unique-1"}]})
    await stack.set_mode(cloud="ok")
    t0 = time.monotonic()
    r = await stack.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "unique-2"}]})
    st = await stack.stats()
    assert r.status_code == 502 and "baseten" in st.get("breakers_open_s", {}) and time.monotonic() - t0 < 2, (
        f"open breaker skips the cloud tier without waiting on it ({r.status_code} {st.get('breakers_open_s')})"
    )
    await asyncio.sleep(1.2)  # BREAKER_COOLDOWN=1
    r = await stack.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "unique-3"}]})
    assert r.headers.get("X-Served-By") == "baseten", "after the cooldown the tier is retried and serves"


async def test_everything_down(stack: Stack):
    await stack.set_mode(local="refuse", cloud="refuse")
    r = await stack.client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True}
    )
    assert r.status_code == 502, "both upstreams down -> 502, not a hang"

    stack.router.st.cache[demo_cache_key({"messages": [{"role": "user", "content": "demo prompt"}]})] = (
        "cached demo answer here"
    )
    served, _reason, text = await stack.stream_text({"messages": [{"role": "user", "content": "Demo   Prompt"}]})
    assert served == "cache" and "cached demo answer" in text, "demo cache serves when all else fails"


async def test_non_streaming_clients(stack: Stack):
    r = await stack.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.headers.get("X-Served-By") == "cluster", "blocking request works"
    assert r.json()["choices"][0]["message"]["content"].startswith("local"), "blocking body is OpenAI-shaped"
    rid = r.headers.get("X-Request-Id", "")
    last = await stack.last_record()
    assert len(rid) == 32 and last.get("request_id") == rid, (
        "every answer gets a request id that the decision log carries"
    )
    assert (
        last.get("result_sha256") == hashlib.sha256(r.json()["choices"][0]["message"]["content"].encode()).hexdigest()
    ), "decision log hashes the finished answer for attestation"

    await stack.set_mode(local="refuse")
    r = await stack.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.headers.get("X-Served-By") == "baseten", "blocking falls back to cloud"


async def test_tool_calls_on_the_chat_endpoint_both_modes(stack: Stack):
    tool_body = {
        "messages": [{"role": "user", "content": "list files"}],
        "tools": [
            {"type": "function", "function": {"name": "shell", "parameters": {"type": "object", "properties": {}}}}
        ],
    }
    r = await stack.client.post("/v1/chat/completions", json=tool_body)
    msg = r.json()["choices"][0]["message"]
    assert (
        r.headers.get("X-Served-By") == "cluster"
        and msg.get("tool_calls", [{}])[0].get("function", {}).get("name") == "shell"
    ), "blocking chat with tools -> tool_calls in the message, served locally"
    raw = await stack.client.post("/v1/chat/completions", json={**tool_body, "stream": True})
    assert (
        raw.headers.get("X-Served-By") == "cluster"
        and '"tool_calls"' in raw.text
        and raw.text.count("data: [DONE]") == 1
    ), "streaming chat with tools -> tool_calls delta streamed from the cluster's blocking path"


async def test_responses_api(stack: Stack):
    served, evs = await stack.responses_events({"model": "auto", "instructions": "be brief", "input": "hi"})
    kinds = [e["type"] for e in evs]
    assert served == "cluster", "responses: text request served locally"
    assert (
        kinds[0] == "response.created"
        and "response.output_text.delta" in kinds
        and "response.output_item.done" in kinds
        and kinds[-1] == "response.completed"
    ), "responses: created -> deltas -> item.done -> completed"
    done = next((e for e in evs if e["type"] == "response.output_item.done"), {})
    assert done.get("item", {}).get("content", [{}])[0].get("text", "").startswith("local0"), (
        "responses: message item carries the local text"
    )

    served, evs = await stack.responses_events(
        {
            "model": "auto",
            "input": "list files",
            "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {}}}],
        }
    )
    items = [e["item"] for e in evs if e["type"] == "response.output_item.done"]
    assert served == "cluster", "responses: tool request still served by the cluster (blocking path)"
    assert (
        items
        and items[0]["type"] == "function_call"
        and items[0]["call_id"] == "call_fake"
        and json.loads(items[0]["arguments"]) == {"command": ["ls"]}
    ), "responses: function_call item with call_id/name/arguments"

    served, evs = await stack.responses_events(
        {
            "model": "auto",
            "input": "patch it",
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "p",
                    "format": {"type": "grammar", "syntax": "lark", "definition": "x"},
                }
            ],
        }
    )
    items = [e["item"] for e in evs if e["type"] == "response.output_item.done"]
    assert items and items[0]["type"] == "custom_tool_call" and items[0]["input"] == "*** patch ***", (
        "responses: custom tool comes back as custom_tool_call with raw input"
    )

    r = await stack.client.post("/v1/responses", json={"model": "auto", "input": "hi"})
    assert r.status_code == 200 and r.json()["object"] == "response" and r.json()["output"][0]["type"] == "message", (
        "responses: non-streaming returns a response object"
    )

    await stack.set_mode(local="refuse")
    served, evs = await stack.responses_events({"model": "auto", "input": "hi"})
    assert served == "baseten" and evs[-1]["type"] == "response.completed", "responses: local down -> cloud fallback"


async def test_responses_stream_that_dies_ends_in_response_failed(stack: Stack):
    """The Responses API has no handover marker, so a dead stream fails honestly instead of continuing."""
    await stack.set_mode(local="die_midstream")
    served, evs = await stack.responses_events({"model": "auto", "input": "hi"})
    kinds = [e["type"] for e in evs]
    assert served == "cluster" and "response.output_text.delta" in kinds
    assert kinds[-1] == "response.failed" and "response.completed" not in kinds
    assert "RemoteProtocolError" in evs[-1]["response"]["error"]["message"]
    last = await stack.last_record()
    assert (
        last["served_by"] == "cluster"
        and last["recovered"] is False
        and "RemoteProtocolError" in last["mid_stream_error"]
    )
    r = await stack.client.post("/v1/responses", json={"model": "auto"})
    assert r.status_code == 400 and "input" in r.json()["error"]["message"]


async def test_metadata_endpoints(stack: Stack):
    await stack.set_mode(local="refuse")  # one fallback, so /stats has something to count
    await stack.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    await stack.set_mode(local="ok")
    models = (await stack.client.get("/v1/models")).json()
    ids = [m["id"] for m in models["data"]]
    assert "local-3b" in ids and "local-3b-heavy" in ids, "/v1/models lists local + heavy"
    st = await stack.stats()
    assert st.get("pct_local") is not None, "/stats reports pct_local"
    assert sum(st.get("fallbacks", {}).values()) > 0, "/stats counts fallbacks"


async def test_on_chain_attestation_wiring(stack: Stack):
    st = await stack.stats()
    assert st.get("solana") is None, "/stats reports solana off without a keypair"
    rt = stack.router
    chain = FakeChain()
    rt.attestor = attest.Attestor(chain, rt.fresh_status, rt.records.read, os.devnull)
    r = await stack.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "attest me"}]})
    await rt.attestor.step()
    rid_bytes = bytes.fromhex(r.headers["X-Request-Id"])
    assert (
        chain.sent[-1][0] == 3
        and rid_bytes in chain.sent[-1]
        and hashlib.sha256(r.json()["choices"][0]["message"]["content"].encode()).digest() in chain.sent[-1]
    ), "a served request is committed on chain with its request id and answer hash"
    st = await stack.stats()
    assert st["solana"]["sent"] >= 2, "/stats shows the attestation summary"
    assert st["solana"]["alive"] is False and st["attestor_alive"] is False and st["decision_log_queue_depth"] == 0


async def test_production_edges(stack: Stack):
    r = await stack.client.get("/readyz")
    assert r.status_code == 200 and r.json()["ready"], "/readyz is 200 while something can serve"
    r = await stack.client.post("/v1/chat/completions", json={"model": "auto"})
    assert r.status_code == 400 and "messages" in r.text, "a body without messages is a 400 with a reason"
    huge = {"messages": [{"role": "user", "content": "x" * 100_000}]}  # MAX_BODY_BYTES is 60000 in this run
    r = await stack.client.post("/v1/chat/completions", json=huge)
    assert r.status_code == 413, "an oversized body is refused up front"
    st = await stack.stats()
    assert st["tier_health"].get("baseten", "").startswith("ok, 2 chat"), "boot probe filled the cloud catalog"
    ids = [m["id"] for m in (await stack.client.get("/v1/models")).json()["data"]]
    assert "cloud-fancy" in ids and "text-embedding-3" not in ids, "/v1/models lists every catalog model"
    r = await stack.client.post(
        "/v1/chat/completions", json={"model": "cloud-fancy", "messages": [{"role": "user", "content": "hi"}]}
    )
    sent = await stack.last_record()
    assert (
        r.headers.get("X-Served-By") == "baseten"
        and r.headers.get("X-Route-Reason") == "model_pinned"
        and sent.get("model_sent") == "cloud-fancy"
    ), "asking for a catalog model pins its tier and sends that exact model"


async def test_chunked_body_over_the_cap_is_refused(stack: Stack):
    async def drip() -> AsyncIterator[bytes]:
        yield b'{"messages": [{"role": "user", "content": "'
        for _ in range(70):
            yield b"x" * 1000
            await asyncio.sleep(0)
        yield b'"}]}'

    r = await stack.client.post("/v1/chat/completions", content=drip())
    assert r.status_code == 413 and "over 60000 bytes" in r.json()["error"]["message"]
    assert all(v == 0 for v in (await stack.stats())["inflight"].values()), "nothing reached an upstream"


async def test_token_rates(stack: Stack):
    for _ in range(6):  # enough recent answers for the rates to mean something
        await stack.stream_text({"messages": [{"role": "user", "content": "rate me"}]})
    rec = await stack.last_record()
    assert rec["served_by"] == "cluster" and rec["gen_tokens"] == 8 and rec["tokens_source"] == "chunks", (
        "cluster stream counts one token per chunk"
    )
    assert rec.get("decode_tps", 0) > 0 and rec.get("prefill_tps", 0) > 0, "cluster stream has decode and prefill rates"
    await stack.stream_text(
        {"messages": [{"role": "user", "content": "rate me"}]}, headers={"X-Force-Upstream": "baseten"}
    )
    rec = await stack.last_record()
    assert (
        rec["served_by"] == "baseten"
        and rec["gen_tokens"] == 40
        and rec["tokens_source"] == "usage"
        and rec["prompt_tokens_actual"] == 11
    ), "cloud stream asks for usage and takes the exact count"
    st = await stack.stats()
    assert st["rates"]["cluster"]["decode_tps"] > 0 and st["rates"]["baseten"]["n"] >= 1 and len(st["recent"]) > 5, (
        "/stats has per-upstream rates and the recent answers"
    )
    assert st["rates"]["cluster"]["latency_ms_p95"] >= st["rates"]["cluster"]["latency_ms_p50"] and all(
        v == 0 for v in st["inflight"].values()
    ), "rates carry p50/p95 and nothing is in flight between requests"
    assert rec.get("cluster_state") == "healthy" and "nodes_active" in rec, "every answer records the cluster size"

    lines = await stack.records()
    assert len(lines) > 6, "decision log written as JSONL"
    assert all(k in lines[0] for k in ("routed_to", "reason", "prompt_tokens", "latency_ms")), (
        "decision log has the pitch fields"
    )


# ------------------------------------------------------------------- auth


async def test_router_api_key_guards_the_generation_endpoints(tmp_path):
    mode = Mode()
    app = create_app(settings_for(tmp_path, router_api_key="s3cret"), transport=httpx2.MockTransport(fake_stack(mode)))
    async with (
        LifespanManager(app) as manager,
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app=manager.app), base_url="http://router") as c,
    ):
        await app.state.router.refresh_status()
        body = {"messages": [{"role": "user", "content": "hi"}]}
        r = await c.post("/v1/chat/completions", json=body)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert r.json()["error"]["code"] == "invalid_api_key"
        r = await c.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
        r = await c.post("/v1/responses", json={"input": "hi"}, headers={"Authorization": "Basic s3cret"})
        assert r.status_code == 401
        r = await c.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200 and r.headers["X-Served-By"] == "cluster"
        r = await c.post("/v1/responses", json={"input": "hi"}, headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200
        for path in ("/v1/models", "/stats", "/readyz", "/healthz"):
            assert (await c.get(path)).status_code == 200, f"{path} stays open"
        assert "s3cret" not in repr(app.state.settings), "the key never appears in a repr"
