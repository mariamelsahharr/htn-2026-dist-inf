"""
test_service.py - the Router against a fake upstream: answer cache, readiness, the
cluster's concurrency cap and the boot probe. Run: pytest -q
"""

import asyncio
import json

import httpx2
import pytest
from config import Settings
from service import Router


def fake_upstream(model_field: bool = True):
    calls: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/status"):
            return httpx2.Response(
                200,
                json={
                    "state": "healthy",
                    "nodes_active": 4,
                    "root": {"model": "/home/pi/m/dllama_model_qwen3_0.6b_q40.m"},
                },
            )
        if request.url.path.endswith("/models"):
            if "pi" in request.url.host:
                return httpx2.Response(200, json={"data": []})
            return httpx2.Response(200, json={"data": [{"id": "big-1"}, {"id": "whisper-1"}]})
        body = json.loads(request.content)
        calls.append(body)
        text = f"answer {len(calls)}"
        return httpx2.Response(
            200,
            json={
                "model": body.get("model") if model_field else "x",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    return handler, calls


def make_router(**overrides) -> tuple[Router, list[dict]]:
    handler, calls = fake_upstream()
    settings = Settings(
        _env_file=None,
        cloud_api_key="k",
        cloud_base_url="http://cloud/v1",
        local_base_url="http://pi:9990",
        status_url="http://pi:9991/status",
        decision_log="/tmp/test_service_decisions.jsonl",
        **overrides,
    )
    return Router(settings, httpx2.AsyncClient(transport=httpx2.MockTransport(handler))), calls


def ask(router: Router, text: str = "hello", **body):
    return asyncio.run(router.chat({"messages": [{"role": "user", "content": text}], **body}, {}))


def test_identical_prompt_is_answered_from_memory_within_the_ttl():
    router, calls = make_router(answer_cache_ttl=30)
    router.st.cluster_status = "unreachable"
    first = ask(router)
    second = ask(router)
    assert first.headers["x-served-by"] == "baseten" and second.headers["x-served-by"] == "cache"
    assert second.headers["x-route-reason"] == "answer_cache" and len(calls) == 1
    assert json.loads(second.body)["choices"][0]["message"]["content"] == "answer 1"
    assert ask(router, "hello", model="auto").headers["x-served-by"] == "cache"
    assert ask(router, "different").headers["x-served-by"] == "baseten"


def test_cache_is_off_for_forced_requests_tools_and_ttl_zero():
    router, calls = make_router(answer_cache_ttl=30)
    router.st.cluster_status = "unreachable"
    ask(router, "same")
    forced = asyncio.run(
        router.chat({"messages": [{"role": "user", "content": "same"}]}, {"X-Force-Upstream": "baseten"})
    )
    assert forced.headers["x-served-by"] == "baseten" and len(calls) == 2
    router2, calls2 = make_router(answer_cache_ttl=0)
    router2.st.cluster_status = "unreachable"
    ask(router2, "same")
    ask(router2, "same")
    assert len(calls2) == 2


def test_ready_needs_a_serving_cluster_or_a_closed_breaker():
    router, _ = make_router()
    router.st.cluster_status = "unreachable"
    assert router.ready()[0] is True  # baseten's breaker is closed
    now = __import__("time").time()
    for _ in range(router.s.breaker_failures):
        router.up.breaker.record_failure("baseten", now)
    assert router.ready()[0] is False
    router.st.cluster_status = "degraded"
    assert router.ready()[0] is True


def test_boot_probe_fills_the_catalog_and_opens_a_dead_tier():
    router, _ = make_router()
    verdict = asyncio.run(router.up.probe_tiers())
    assert verdict["baseten"].startswith("ok, 1 chat") and router.cfg.catalogs["baseten"] == ("big-1",)
    assert router.cfg.tier_for_model("big-1") == "baseten" and router.cfg.tier_for_model("whisper-1") is None

    def refuse(request):
        raise httpx2.ConnectError("no route")

    settings = Settings(_env_file=None, cloud_api_key="k", cloud_base_url="http://dead/v1")
    dead = Router(settings, httpx2.AsyncClient(transport=httpx2.MockTransport(refuse)))
    verdict = asyncio.run(dead.up.probe_tiers())
    assert verdict["baseten"].startswith("down") and dead.up.breaker.is_open("baseten", __import__("time").time())


def test_cluster_queue_spills_to_cloud_when_full():
    router, calls = make_router(local_queue_max=0)  # nobody may wait: the slot is take-it-or-leave-it
    router.st.cluster_status = "healthy"
    r = ask(router)
    assert r.headers["x-served-by"] == "cluster" and calls[-1]["model"] == router.s.local_model

    async def hold_then_ask():
        async with router.up.cluster_slot():  # a request is already on the Pis
            return await router.chat({"messages": [{"role": "user", "content": "second"}]}, {})

    r = asyncio.run(hold_then_ask())
    assert r.headers["x-served-by"] == "baseten" and "fallback" in r.headers["x-route-reason"]
    assert any(k.startswith("blocking:UpstreamError: cluster busy") for k in router.st.fallbacks)


def test_requested_catalog_model_is_sent_verbatim():
    router, calls = make_router()
    asyncio.run(router.up.probe_tiers())
    router.st.cluster_status = "unreachable"
    r = ask(router, model="big-1")
    assert r.headers["x-route-reason"] == "model_pinned" and calls[-1]["model"] == "big-1"
    r = ask(router, model="auto")
    assert calls[-1]["model"] == router.s.cloud_model


@pytest.mark.parametrize("bad", [{"messages": []}, {"messages": [{"role": "nobody", "content": "x"}]}])
def test_validation_rejects_bad_bodies(bad):
    from app import ChatRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatRequest.model_validate(bad)


def test_cluster_model_name_comes_from_the_supervisor():
    router, calls = make_router()
    assert router.cfg.local_model == "llama-3.2-3b-instruct"
    asyncio.run(router.refresh_status())
    assert router.st.cluster_status == "healthy" and router.cfg.local_model == "qwen3_0.6b_q40"
    from routing import models_payload

    assert models_payload(router.cfg)["data"][0]["id"] == "qwen3_0.6b_q40"
    ask(router)
    assert calls[-1]["model"] == "qwen3_0.6b_q40"
