"""
test_routing.py - one test per policy branch. Run: pytest -q

These are pure-function tests, no server and no network, so they run in well
under a second and you can put them in the pre-demo checklist.
"""

import json
from pathlib import Path

import pytest
from routing import (
    BASETEN,
    CLUSTER,
    Breaker,
    RouterConfig,
    Tier,
    cluster_state,
    continuation_body,
    estimate_tokens,
    fallback_chain,
    is_heavy_model,
    missing_required_tool_call,
    models_payload,
    route,
    strip_heavy,
)

CFG = RouterConfig(
    size_threshold=2048, cloud_available=True, local_model="llama-3.2-3b-instruct", cloud_model="big-70b"
)
NO_CLOUD = RouterConfig(
    size_threshold=2048, cloud_available=False, local_model="llama-3.2-3b-instruct", cloud_model="big-70b"
)


def tiered(**kw):
    """Baseten plus openai (tool tier) plus gemini, in the default order."""
    return RouterConfig(
        size_threshold=2048,
        cloud_available=True,
        local_model="llama-3.2-3b-instruct",
        cloud_model="big-70b",
        tiers={"openai": Tier("openai", "gpt-x", handles_tools=True), "gemini": Tier("gemini", "gemini-x")},
        tool_tier="openai",
        **kw,
    )


TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]


def body(text="hello", model=None, max_tokens=None):
    b = {"messages": [{"role": "user", "content": text}]}
    if model:
        b["model"] = model
    if max_tokens:
        b["max_tokens"] = max_tokens
    return b


# ------------------------------------------------------------ branch 0: force


def test_force_header_wins_over_everything():
    d = route(body("x" * 100000, max_tokens=4000), {"X-Force-Upstream": "cluster"}, "degraded", CFG)
    assert d.upstream == CLUSTER
    assert d.reason == "forced_by_header"
    assert d.forced is True


def test_force_cloud_on_a_tiny_healthy_request():
    d = route(body("hi"), {"x-force-upstream": "baseten"}, "healthy", CFG)
    assert d.upstream == BASETEN and d.forced is True


def test_force_header_garbage_is_ignored():
    d = route(body("hi"), {"X-Force-Upstream": "banana"}, "healthy", CFG)
    assert d.upstream == CLUSTER and d.reason == "default_local"


# ----------------------------------------------------- branch 1: cluster health


@pytest.mark.parametrize("status", ["restarting", "degraded_below_min", "down", "unknown", "unreachable"])
def test_unhealthy_cluster_goes_to_cloud(status):
    d = route(body("hi"), {}, status, CFG)
    assert d.upstream == BASETEN
    assert d.reason == f"cluster_{status}"


STATUS_EXAMPLE = json.loads(
    (Path(__file__).resolve().parents[1] / "cluster" / "supervisor" / "status.example.json").read_text()
)


def test_supervisor_status_contract_parses():
    """The example document every consumer is tested against: degraded on 2 of 4 nodes."""
    assert cluster_state(STATUS_EXAMPLE, min_local_nodes=2) == "degraded"
    assert cluster_state(STATUS_EXAMPLE, min_local_nodes=3) == "degraded_below_min"
    assert cluster_state({**STATUS_EXAMPLE, "state": "healthy"}, 2) == "healthy"
    assert cluster_state({**STATUS_EXAMPLE, "state": "restarting"}, 2) == "restarting"
    assert cluster_state({}, 2) == "unknown"


def test_degraded_cluster_still_serves_locally():
    """The supervisor's reduced node set is there to be used, not routed around."""
    d = route(body("hi"), {}, "degraded", CFG)
    assert d.upstream == CLUSTER and d.reason == "default_local"


def test_degraded_cluster_is_a_valid_fallback_target():
    d = route(body("hi"), {"X-Escalate": "1"}, "degraded", CFG)
    assert fallback_chain(d, "degraded", CFG) == [BASETEN, CLUSTER]


def test_health_check_is_case_insensitive():
    assert route(body("hi"), {}, "HEALTHY", CFG).upstream == CLUSTER


def test_unhealthy_cluster_without_cloud_still_tries_local():
    """A missing Baseten key must degrade the demo, not 500 it."""
    d = route(body("hi"), {}, "restarting", NO_CLOUD)
    assert d.upstream == CLUSTER
    assert d.reason == "cluster_restarting_no_cloud"


# -------------------------------------------------------- branch 2: size budget


def test_long_prompt_escalates():
    d = route(body("word " * 3000), {}, "healthy", CFG)
    assert d.upstream == BASETEN and d.reason == "over_size_threshold"


def test_large_max_tokens_escalates_even_with_a_short_prompt():
    d = route(body("hi", max_tokens=4096), {}, "healthy", CFG)
    assert d.upstream == BASETEN and d.reason == "over_size_threshold"


def test_just_under_threshold_stays_local():
    # ~4 chars/token, so 1000 chars is ~250 tokens; +512 default max_tokens
    d = route(body("x" * 1000), {}, "healthy", CFG)
    assert d.upstream == CLUSTER


def test_threshold_is_prompt_plus_max_tokens_not_either_alone():
    small = RouterConfig(size_threshold=300, cloud_available=True)
    d = route(body("x" * 400, max_tokens=200), {}, "healthy", small)
    assert d.upstream == BASETEN


# ---------------------------------------------------- branch: task complexity


def test_keyword_only_counts_in_the_last_user_turn():
    b = {
        "messages": [
            {"role": "user", "content": "please refactor everything"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "thanks, what time is it"},
        ]
    }
    assert route(b, {}, "healthy", CFG).upstream == CLUSTER


def test_large_pasted_code_escalates():
    code = "```python\n" + "x = 1\n" * 200 + "```"
    d = route(body("fix this: " + code), {}, "healthy", CFG)
    assert d.upstream == BASETEN and d.reason == "complex_task_code"


def test_small_snippet_stays_local():
    code = "```python\n" + "x = 1\n" * 10 + "```"
    assert route(body("fix this: " + code), {}, "healthy", CFG).upstream == CLUSTER


def test_long_agent_session_escalates():
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": "ok"} for i in range(14)]
    d = route({"messages": msgs}, {}, "healthy", CFG)
    assert d.upstream == BASETEN and d.reason == "complex_task_turns"


def test_complexity_thresholds_are_configurable():
    lax = RouterConfig(
        cloud_available=True,
        cloud_model="big-70b",
        code_lines_threshold=10_000,
        max_local_turns=100,
    )
    b = body("refactor the entire repo from scratch")
    assert route(b, {}, "healthy", lax).upstream == CLUSTER


def test_size_beats_complexity():
    d = route(body("refactor " + "word " * 3000), {}, "healthy", CFG)
    assert d.reason == "over_size_threshold"


# --------------------------------------------------- branch 3: explicit heavy


def test_heavy_model_suffix_escalates():
    d = route(body("hi", model="llama-3.2-3b-instruct-heavy"), {}, "healthy", CFG)
    assert d.upstream == BASETEN and d.reason == "heavy_model_tag"


def test_escalate_header_escalates():
    d = route(body("hi"), {"X-Escalate": "true"}, "healthy", CFG)
    assert d.upstream == BASETEN and d.reason == "escalate_header"


@pytest.mark.parametrize("v", ["1", "true", "TRUE", "yes"])
def test_escalate_header_accepts_common_truthy_values(v):
    assert route(body("hi"), {"X-Escalate": v}, "healthy", CFG).upstream == BASETEN


def test_escalate_header_false_stays_local():
    assert route(body("hi"), {"X-Escalate": "false"}, "healthy", CFG).upstream == CLUSTER


# ------------------------------------------------------------ branch 4: local


def test_default_is_local():
    d = route(body("write a bash one-liner"), {}, "healthy", CFG)
    assert d.upstream == CLUSTER and d.reason == "default_local"


def test_local_decision_sends_the_local_model_name():
    d = route(body("hi", model="whatever-the-client-asked-for"), {}, "healthy", CFG)
    assert d.model_sent == CFG.local_model
    assert d.model_requested == "whatever-the-client-asked-for"


def test_cloud_decision_sends_the_cloud_model_name():
    d = route(body("hi"), {"X-Escalate": "1"}, "healthy", CFG)
    assert d.model_sent == CFG.cloud_model


# --------------------------------------------------------------- priority order


def test_health_beats_size():
    d = route(body("word " * 3000), {}, "restarting", CFG)
    assert d.reason == "cluster_restarting"  # not over_size_threshold


def test_size_beats_heavy_tag():
    d = route(body("word " * 3000, model="foo-heavy"), {}, "healthy", CFG)
    assert d.reason == "over_size_threshold"


# ------------------------------------------------------------------- estimation


def test_estimate_tokens_scales_with_length():
    short = estimate_tokens([{"role": "user", "content": "hi"}])
    long = estimate_tokens([{"role": "user", "content": "x" * 4000}])
    assert long > short * 10


def test_estimate_tokens_handles_multipart_content():
    n = estimate_tokens(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "x" * 400},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
    )
    assert n >= 100


def test_estimate_tokens_survives_empty_and_none():
    assert estimate_tokens([]) == 0
    assert estimate_tokens([{"role": "user"}]) >= 0


def test_heavy_detection_and_stripping():
    assert is_heavy_model("llama-heavy")
    assert not is_heavy_model("llama-3.2-3b-instruct")
    assert "heavy" not in strip_heavy("llama-3.2-3b-instruct-heavy").lower()


# ----------------------------------------------------- mid-stream continuation


def test_continuation_appends_partial_and_instruction():
    orig = body("explain systemd")
    cont = continuation_body(orig, "Systemd is an init", "big-70b")
    assert cont["model"] == "big-70b"
    assert len(cont["messages"]) == 3
    assert cont["messages"][1]["role"] == "assistant"
    assert cont["messages"][1]["content"] == "Systemd is an init"
    assert cont["messages"][2]["role"] == "user"


def test_continuation_with_no_partial_text_is_a_plain_retry():
    orig = body("explain systemd")
    cont = continuation_body(orig, "", "big-70b")
    assert len(cont["messages"]) == 1


def test_continuation_does_not_mutate_the_original():
    orig = body("explain systemd")
    continuation_body(orig, "partial", "big-70b")
    assert len(orig["messages"]) == 1


# ------------------------------------------------------------------- /v1/models


def test_models_payload_lists_local_and_heavy():
    ids = [m["id"] for m in models_payload(CFG)["data"]]
    assert CFG.local_model in ids
    assert any(i.endswith("-heavy") for i in ids)


def test_models_payload_omits_cloud_when_unconfigured():
    ids = [m["id"] for m in models_payload(NO_CLOUD)["data"]]
    assert NO_CLOUD.cloud_model not in ids


# --------------------------------------------------------- breaker + tool miss


def test_breaker_opens_after_n_failures_and_closes_after_cooldown():
    b = Breaker(failures=2, cooldown=10)
    assert b.record_failure("baseten", 100.0) is False and not b.is_open("baseten", 100.0)
    assert b.record_failure("baseten", 101.0) is True and b.is_open("baseten", 105.0)
    assert not b.is_open("baseten", 111.5)  # cooldown elapsed
    assert b.snapshot(105.0) == {"baseten": 6.0}


def test_breaker_success_resets_the_count_and_clears_an_open_circuit():
    b = Breaker(failures=2, cooldown=10)
    b.record_failure("gemini", 0.0)
    b.record_success("gemini")
    assert b.record_failure("gemini", 1.0) is False  # count restarted
    b.record_failure("gemini", 2.0)
    b.record_success("gemini")
    assert not b.is_open("gemini", 3.0)


def test_required_tool_call_answered_in_prose_is_a_miss():
    prose = {"role": "assistant", "content": "I would run ls."}
    called = {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function"}]}
    assert missing_required_tool_call({"tools": TOOLS, "tool_choice": "required"}, prose)
    assert missing_required_tool_call(
        {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "shell"}}}, prose
    )
    assert not missing_required_tool_call({"tools": TOOLS, "tool_choice": "required"}, called)
    assert not missing_required_tool_call({"tools": TOOLS, "tool_choice": "auto"}, prose)
    assert not missing_required_tool_call({"tools": TOOLS}, prose)


def test_tools_model_is_used_only_when_the_request_carries_tools():
    cfg = RouterConfig(
        cloud_available=True,
        cloud_model="big-70b",
        tiers={"snowflake": Tier("snowflake", "llama3.1-8b", "https://x/v1", "k", tools_model="claude-haiku-4-5")},
    )
    plain = route(body("hi"), {"X-Force-Upstream": "snowflake"}, "healthy", cfg)
    with_tools = route({**body("hi"), "tools": TOOLS}, {"X-Force-Upstream": "snowflake"}, "healthy", cfg)
    assert plain.model_sent == "llama3.1-8b" and with_tools.model_sent == "claude-haiku-4-5"
    assert cfg.model_for("snowflake", True) == "claude-haiku-4-5" and cfg.model_for(BASETEN, True) == "big-70b"
    ids = [m["id"] for m in models_payload(cfg)["data"]]
    assert "llama3.1-8b" in ids and "claude-haiku-4-5" in ids


def test_tier_reasoning_effort_is_added_only_when_set():
    t = Tier("openai", "gpt-5.6-luna", "https://x/v1", "k", reasoning_effort="none")
    out = t.payload({"model": "gpt-5.6-luna", "messages": []}, stream=True)
    assert out["reasoning_effort"] == "none" and out["stream"] is True and out["messages"] == []
    assert "reasoning_effort" not in Tier("gemini", "g", "https://x/v1", "k").payload({"messages": []})


def test_models_payload_lists_every_configured_tier():
    ids = [m["id"] for m in models_payload(tiered())["data"]]
    assert "big-70b" in ids and "gpt-x" in ids and "gemini-x" in ids


# ------------------------------------------------------------ tiers: tool tier


def test_tools_go_to_the_tool_tier():
    d = route({**body("hi"), "tools": TOOLS}, {}, "healthy", tiered())
    assert d.upstream == "openai" and d.reason == "tools_attached"
    assert d.model_sent == "gpt-x"


def test_tools_outrank_cluster_health():
    d = route({**body("hi"), "tools": TOOLS}, {}, "restarting", tiered())
    assert d.upstream == "openai"


def test_tools_without_a_tool_tier_follow_the_normal_policy():
    d = route({**body("hi"), "tools": TOOLS}, {}, "healthy", CFG)
    assert d.upstream == CLUSTER and d.reason == "default_local"


def test_tool_tier_that_is_not_configured_is_dropped():
    cfg = RouterConfig(cloud_available=True, cloud_model="big-70b", tool_tier="openai")
    assert cfg.tool_tier is None


def test_empty_tools_list_is_not_a_tool_request():
    d = route({**body("hi"), "tools": []}, {}, "healthy", tiered())
    assert d.upstream == CLUSTER


# ---------------------------------------------------------- tiers: force + order


def test_force_header_can_name_any_configured_tier():
    d = route(body("hi"), {"X-Force-Upstream": "gemini"}, "healthy", tiered())
    assert d.upstream == "gemini" and d.forced is True


def test_force_header_for_unconfigured_tier_is_ignored():
    d = route(body("hi"), {"X-Force-Upstream": "gemini"}, "healthy", CFG)
    assert d.upstream == CLUSTER and d.reason == "default_local"


def test_cloud_tiers_follow_configured_order_then_extras():
    cfg = tiered(cloud_order=("gemini", "baseten"))
    assert cfg.cloud_tiers() == ["gemini", "baseten", "openai"]


def test_escalation_uses_the_heavy_tier_even_when_it_is_not_baseten():
    cfg = tiered(heavy_tier="gemini")
    d = route(body("word " * 3000), {}, "healthy", cfg)
    assert d.upstream == "gemini" and d.reason == "over_size_threshold"


# --------------------------------------------------------------- fallback chain


def test_cluster_primary_falls_back_through_every_cloud_tier_in_order():
    d = route(body("hi"), {}, "healthy", tiered())
    assert fallback_chain(d, "healthy", tiered()) == [CLUSTER, BASETEN, "gemini", "openai"]


def test_cloud_primary_falls_back_to_other_clouds_then_cluster():
    d = route(body("hi"), {"X-Escalate": "1"}, "healthy", tiered())
    assert fallback_chain(d, "healthy", tiered()) == [BASETEN, "gemini", "openai", CLUSTER]


def test_oversized_request_never_falls_back_to_the_cluster():
    d = route(body("word " * 3000), {}, "healthy", tiered())
    chain = fallback_chain(d, "healthy", tiered())
    assert chain[0] == BASETEN and CLUSTER not in chain


def test_tool_request_never_falls_back_to_the_cluster():
    d = route({**body("hi"), "tools": TOOLS}, {}, "healthy", tiered())
    assert CLUSTER not in fallback_chain(d, "healthy", tiered())


def test_unhealthy_cluster_is_not_a_fallback_target():
    d = route(body("hi"), {}, "restarting", tiered())
    assert CLUSTER not in fallback_chain(d, "restarting", tiered())


def test_forced_upstream_has_no_fallback():
    d = route(body("hi"), {"X-Force-Upstream": "cluster"}, "healthy", tiered())
    assert fallback_chain(d, "healthy", tiered()) == [CLUSTER]


def test_two_upstream_config_keeps_the_old_chain():
    d = route(body("hi"), {}, "healthy", CFG)
    assert fallback_chain(d, "healthy", CFG) == [CLUSTER, BASETEN]


def test_tier_asks_for_usage_only_when_it_supports_it():
    t = Tier("baseten", "glm", "https://x/v1", "k", usage_in_stream=True)
    assert t.payload({"messages": []}, stream=True)["stream_options"] == {"include_usage": True}
    assert "stream_options" not in t.payload({"messages": []}, stream=False)
    assert "stream_options" not in Tier("snowflake", "m", "https://x/v1", "k").payload({"messages": []}, stream=True)
    kept = t.payload({"messages": [], "stream_options": {"other": 1}}, stream=True)["stream_options"]
    assert kept == {"other": 1, "include_usage": True}


def test_asking_for_a_tiers_model_by_name_pins_that_tier():
    cfg = tiered()
    d = route({"model": cfg.tiers["gemini"].model, "messages": [{"role": "user", "content": "hi"}]}, {}, "healthy", cfg)
    assert (d.upstream, d.reason, d.forced) == ("gemini", "model_pinned", True)
    d = route({"model": "auto", "messages": [{"role": "user", "content": "hi"}]}, {}, "healthy", cfg)
    assert d.upstream == CLUSTER and d.reason != "model_pinned"
    d = route({"model": cfg.local_model, "messages": [{"role": "user", "content": "hi"}]}, {}, "healthy", cfg)
    assert d.reason != "model_pinned"
    header = route({"model": cfg.tiers["gemini"].model, "messages": []}, {"X-Force-Upstream": "openai"}, "healthy", cfg)
    assert header.upstream == "openai", "the header still wins over the model name"


def test_models_payload_says_which_tier_owns_each_model():
    owners = {m["id"]: m["owned_by"] for m in models_payload(tiered())["data"]}
    cfg = tiered()
    assert owners[cfg.local_model] == CLUSTER
    assert owners[cfg.tiers["gemini"].model] == "gemini"
    assert owners[cfg.local_model + "-heavy"] == cfg.heavy_tier
