"""
test_service.py - the Router against a fake upstream: answer cache, readiness, the
cluster's concurrency cap, the boot probe, and what happens when a client hangs up.
Run: pytest -q
"""

import asyncio
import json
import time
from collections import Counter
from collections.abc import AsyncIterator

import httpx2
import orjson
import pytest
from config import Settings
from schemas import ChatRequest
from service import Router
from wire import UpstreamError, answer_cache_key


def completion(body: dict, text: str, model_field: bool = True) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "model": body.get("model") if model_field else "x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )


async def sse_body(words: list[str], gap: float, then_hang: bool = False) -> AsyncIterator[bytes]:
    for w in words:
        chunk = {"choices": [{"index": 0, "delta": {"content": w}, "finish_reason": None}]}
        yield f"data: {orjson.dumps(chunk).decode()}\n\n".encode()
        await asyncio.sleep(gap)
    if then_hang:
        await asyncio.sleep(30)
    yield b'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n'
    yield b"data: [DONE]\n\n"


def fake_upstream(delay: float = 0.0, model_field: bool = True, hang_after: int | None = None):
    """A supervisor, a cluster root and a cloud tier on one MockTransport. `hang_after` makes
    the streamed answer stall after that many words; `delay` slows the blocking answer."""
    calls: list[dict] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/status"):
            return httpx2.Response(
                200,
                json={
                    "state": "healthy",
                    "nodes_active": 4,
                    "root": {"model": "/home/pi/m/dllama_model_qwen3_30b_a3b_q40.m"},
                },
            )
        if request.url.path.endswith("/models"):
            if "pi" in request.url.host:
                return httpx2.Response(200, json={"data": []})
            return httpx2.Response(200, json={"data": [{"id": "big-1"}, {"id": "whisper-1"}]})
        body = json.loads(request.content)
        calls.append(body)
        if body.get("stream"):
            words = ["one ", "two ", "three "]
            hang = hang_after is not None
            return httpx2.Response(
                200,
                content=sse_body(words[:hang_after], 0.01, then_hang=hang),
                headers={"content-type": "text/event-stream"},
            )
        if delay:
            await asyncio.sleep(delay)
        return completion(body, f"answer {len(calls)}", model_field)

    return handler, calls


def make_router(tmp_path, transport_kw: dict | None = None, **overrides) -> tuple[Router, list[dict]]:
    handler, calls = fake_upstream(**(transport_kw or {}))
    settings = Settings(
        _env_file=None,
        cloud_api_key="k",
        cloud_base_url="http://cloud/v1",
        local_base_url="http://pi:9990",
        status_url="http://pi:9991/status",
        decision_log=str(tmp_path / "decisions.jsonl"),
        cache_file=str(tmp_path / "demo_cache.json"),
        **overrides,
    )
    return Router(settings, httpx2.AsyncClient(transport=httpx2.MockTransport(handler))), calls


async def ask(router: Router, text: str = "hello", headers: dict | None = None, **body):
    return await router.chat({"messages": [{"role": "user", "content": text}], **body}, headers or {})


async def test_identical_conversation_is_answered_from_memory_within_the_ttl(tmp_path):
    router, calls = make_router(tmp_path, answer_cache_ttl=30)
    router.st.cluster_status = "unreachable"
    first = await ask(router)
    second = await ask(router)
    assert first.headers["x-served-by"] == "baseten" and second.headers["x-served-by"] == "cache"
    assert second.headers["x-route-reason"] == "answer_cache" and len(calls) == 1
    assert json.loads(second.body)["choices"][0]["message"]["content"] == "answer 1"
    assert (await ask(router, "hello", model="auto")).headers["x-served-by"] == "cache"
    assert (await ask(router, "different")).headers["x-served-by"] == "baseten"


async def test_cache_key_covers_the_whole_conversation_and_the_generation_params(tmp_path):
    router, calls = make_router(tmp_path, answer_cache_ttl=30)
    router.st.cluster_status = "unreachable"
    turn = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "hi"},
    ]
    await router.chat({"messages": turn}, {})
    assert (await ask(router, "hi")).headers["x-served-by"] == "baseten", "same last message, different history"
    assert (await ask(router, "hi")).headers["x-served-by"] == "cache"
    assert (await ask(router, "hi", temperature=0.9)).headers["x-served-by"] == "baseten", "temperature is part of it"
    assert (await ask(router, "hi", max_tokens=5)).headers["x-served-by"] == "baseten", "so is max_tokens"
    assert len(calls) == 4
    a = answer_cache_key({"messages": [{"role": "user", "content": "Hi   there"}]}, "m")
    assert a == answer_cache_key({"messages": [{"role": "user", "content": "hi there"}]}, "m")
    assert a != answer_cache_key({"messages": [{"role": "user", "content": "hi there"}]}, "other-model")


async def test_cached_answers_expire(tmp_path):
    router, calls = make_router(tmp_path, answer_cache_ttl=0.2)
    router.st.cluster_status = "unreachable"
    await ask(router, "same")
    assert (await ask(router, "same")).headers["x-served-by"] == "cache"
    await asyncio.sleep(0.25)
    assert (await ask(router, "same")).headers["x-served-by"] == "baseten" and len(calls) == 2


async def test_cache_is_off_for_forced_requests_tools_and_ttl_zero(tmp_path):
    router, calls = make_router(tmp_path, answer_cache_ttl=30)
    router.st.cluster_status = "unreachable"
    await ask(router, "same")
    forced = await ask(router, "same", headers={"X-Force-Upstream": "baseten"})
    assert forced.headers["x-served-by"] == "baseten" and len(calls) == 2
    router2, calls2 = make_router(tmp_path, answer_cache_ttl=0)
    router2.st.cluster_status = "unreachable"
    await ask(router2, "same")
    await ask(router2, "same")
    assert len(calls2) == 2


async def test_ready_needs_a_serving_cluster_or_a_closed_breaker(tmp_path):
    router, _ = make_router(tmp_path)
    router.st.cluster_status = "unreachable"
    assert router.ready()[0] is True  # baseten's breaker is closed
    now = time.monotonic()
    for _ in range(router.s.breaker_failures):
        router.up.breaker.record_failure("baseten", now)
    assert router.ready()[0] is False
    router.st.cluster_status = "degraded"
    assert router.ready()[0] is True


async def test_boot_probe_fills_the_catalog_and_opens_a_dead_tier(tmp_path):
    router, _ = make_router(tmp_path)
    verdict = await router.up.probe_tiers()
    assert verdict["baseten"].startswith("ok, 1 chat") and router.cfg.catalogs["baseten"] == ("big-1",)
    assert router.cfg.tier_for_model("big-1") == "baseten" and router.cfg.tier_for_model("whisper-1") is None

    def refuse(request):
        raise httpx2.ConnectError("no route")

    settings = Settings(
        _env_file=None, cloud_api_key="k", cloud_base_url="http://dead/v1", decision_log=str(tmp_path / "d")
    )
    dead = Router(settings, httpx2.AsyncClient(transport=httpx2.MockTransport(refuse)))
    verdict = await dead.up.probe_tiers()
    assert verdict["baseten"].startswith("down") and dead.up.breaker.is_open("baseten", time.monotonic())


async def test_boot_probe_runs_the_tiers_concurrently(tmp_path):
    started: list[float] = []

    async def slow_models(request):
        started.append(time.monotonic())
        await asyncio.sleep(0.2)
        return httpx2.Response(200, json={"data": []})

    settings = Settings(
        _env_file=None,
        cloud_api_key="k",
        openai_api_key="k",
        openai_model="gpt-x",
        gemini_api_key="k",
        gemini_model="g",
        decision_log=str(tmp_path / "d"),
    )
    router = Router(settings, httpx2.AsyncClient(transport=httpx2.MockTransport(slow_models)))
    t0 = time.monotonic()
    verdict = await router.up.probe_tiers()
    assert set(verdict) == {"baseten", "openai", "gemini"} and time.monotonic() - t0 < 0.5
    assert max(started) - min(started) < 0.1


async def test_cluster_queue_spills_to_cloud_when_full(tmp_path):
    router, calls = make_router(tmp_path, local_queue_max=0)  # nobody may wait: the slot is take-it-or-leave-it
    router.st.cluster_status = "healthy"
    r = await ask(router)
    assert r.headers["x-served-by"] == "cluster" and calls[-1]["model"] == router.s.local_model

    async with router.up.cluster_slot():  # a request is already on the Pis
        r = await router.chat({"messages": [{"role": "user", "content": "second"}]}, {})
    assert r.headers["x-served-by"] == "baseten" and "fallback" in r.headers["x-route-reason"]
    assert any(k.startswith("blocking:UpstreamError: cluster busy") for k in router.st.fallbacks)


async def test_concurrent_cluster_requests_respect_concurrency_and_queue_limits(tmp_path):
    router, _ = make_router(tmp_path, {"delay": 0.15}, local_concurrency=1, local_queue_max=1, answer_cache_ttl=0)
    router.st.cluster_status = "healthy"
    responses = await asyncio.gather(*(ask(router, f"q{i}") for i in range(4)))
    served = Counter(r.headers["x-served-by"] for r in responses)
    assert served == {"cluster": 2, "baseten": 2}, "one runs, one waits, the rest spill to the cloud"
    assert all(r.status_code == 200 for r in responses)
    assert router.up.cluster_waiting == 0 and not router.up.cluster_slots.locked()
    assert all(v == 0 for v in router.up.inflight.values())


async def test_client_disconnect_mid_stream_releases_everything(tmp_path):
    router, _ = make_router(tmp_path, {"hang_after": 1})
    router.st.cluster_status = "healthy"
    resp = await ask(router, "stream me", stream=True)
    assert resp.headers["x-served-by"] == "cluster"
    assert router.up.inflight["cluster"] == 1 and router.up.cluster_slots.locked()

    got: list[bytes] = []

    async def consume():
        async for piece in resp.body_iterator:
            got.append(piece)  # noqa: PERF401 - must fill as chunks arrive: the task is cancelled midway

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)  # the first chunk is out; the upstream is now stalling
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert got and b"one" in got[0]
    assert router.up.inflight["cluster"] == 0, "the upstream response was closed with the client"
    assert not router.up.cluster_slots.locked(), "the cluster slot went back"
    router.decisions.close()
    last = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[-1])
    assert last["error"] == "client_disconnected" and last["served_by"] == "cluster" and last["stream"] is True


async def test_breaker_opens_for_the_full_cooldown_after_a_slow_failure(tmp_path):
    router, _ = make_router(tmp_path, breaker_failures=1, breaker_cooldown=10)

    async def slow_failure(upstream, payload):
        await asyncio.sleep(0.2)
        raise UpstreamError("upstream died slowly")

    won, err = await router.up.first_success([("baseten", {"model": "m"})], slow_failure)
    assert won is None and "died slowly" in err
    now = time.monotonic()
    assert router.up.breaker.is_open("baseten", now)
    assert router.up.breaker.snapshot(now)["baseten"] > 9.9, "the cooldown counts from the failure, not the start"


async def test_attempt_budget_stops_the_walk(tmp_path):
    router, _ = make_router(tmp_path, attempt_budget_s=0.05)
    tried: list[str] = []

    async def slow_failure(upstream, payload):
        tried.append(upstream)
        await asyncio.sleep(0.1)
        raise UpstreamError("nope")

    won, err = await router.up.first_success([("cluster", {}), ("baseten", {})], slow_failure)
    assert won is None and tried == ["cluster"] and err.startswith("attempt budget of 0s spent; last: UpstreamError")


async def test_requested_catalog_model_is_sent_verbatim(tmp_path):
    router, calls = make_router(tmp_path)
    await router.up.probe_tiers()
    router.st.cluster_status = "unreachable"
    r = await ask(router, model="big-1")
    assert r.headers["x-route-reason"] == "model_pinned" and calls[-1]["model"] == "big-1"
    r = await ask(router, model="auto")
    assert calls[-1]["model"] == router.s.cloud_model


@pytest.mark.parametrize("bad", [{"messages": []}, {"messages": [{"role": "nobody", "content": "x"}]}])
def test_validation_rejects_bad_bodies(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatRequest.model_validate(bad)


def test_validated_body_keeps_an_explicit_null_and_drops_what_was_not_sent():
    raw = {
        "messages": [
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
        "stream": True,
    }
    body = ChatRequest.model_validate(raw).model_dump(exclude_unset=True)
    assistant = body["messages"][1]
    assert "content" in assistant and assistant["content"] is None and assistant["tool_calls"]
    assert "model" not in body and "max_tokens" not in body and body["stream"] is True


async def test_cluster_model_name_comes_from_the_supervisor(tmp_path):
    router, calls = make_router(tmp_path)
    assert router.cfg.local_model == "qwen3-30b-a3b"
    await router.refresh_status()
    assert router.st.cluster_status == "healthy" and router.cfg.local_model == "qwen3_30b_a3b_q40"
    from routing import models_payload

    assert models_payload(router.cfg)["data"][0]["id"] == "qwen3_30b_a3b_q40"
    await ask(router)
    assert calls[-1]["model"] == "qwen3_30b_a3b_q40"


async def test_a_long_adopted_model_id_still_routes_escalates_and_lists(tmp_path):
    """The adopted id is a distributed-llama file stem: it must behave like any other local id."""
    from routing import models_payload

    router, calls = make_router(tmp_path)
    await router.refresh_status()
    local = router.cfg.local_model
    assert local == "qwen3_30b_a3b_q40"
    ids = {m["id"]: m["owned_by"] for m in models_payload(router.cfg)["data"]}
    assert ids[local] == "cluster" and ids[local + "-heavy"] == "baseten"
    assert (await ask(router, "small", model=local)).headers["x-served-by"] == "cluster"
    r = await ask(router, "small", model=local + "-heavy")
    assert r.headers["x-served-by"] == "baseten" and r.headers["x-route-reason"] == "heavy_model_tag"
    r = await ask(router, "x" * 20000, model=local)
    assert r.headers["x-route-reason"] == "over_size_threshold" and calls[-1]["model"] == router.s.cloud_model


async def test_stats_report_the_writer_queue_and_the_attestor(tmp_path):
    router, _ = make_router(tmp_path)
    st = router.stats()
    assert st["decision_log_queue_depth"] == 0 and st["attestor_alive"] is None and st["solana"] is None
