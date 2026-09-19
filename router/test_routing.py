"""
test_routing.py - one test per policy branch. Run: pytest -q

These are pure-function tests, no server and no network, so they run in well
under a second and you can put them in the pre-demo checklist.
"""

import pytest

from routing import (BASETEN, CLUSTER, RouterConfig, continuation_body,
                     estimate_tokens, is_heavy_model, models_payload, route,
                     strip_heavy)

CFG = RouterConfig(size_threshold=2048, cloud_available=True,
                   local_model="llama-3.2-3b-instruct",
                   cloud_model="big-70b")
NO_CLOUD = RouterConfig(size_threshold=2048, cloud_available=False,
                        local_model="llama-3.2-3b-instruct", cloud_model="big-70b")


def body(text="hello", model=None, max_tokens=None):
    b = {"messages": [{"role": "user", "content": text}]}
    if model:
        b["model"] = model
    if max_tokens:
        b["max_tokens"] = max_tokens
    return b


# ------------------------------------------------------------ branch 0: force

def test_force_header_wins_over_everything():
    d = route(body("x" * 100000, max_tokens=4000),
              {"X-Force-Upstream": "cluster"}, "degraded", CFG)
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

@pytest.mark.parametrize("status", ["restarting", "degraded", "down", "unknown", "unreachable"])
def test_unhealthy_cluster_goes_to_cloud(status):
    d = route(body("hi"), {}, status, CFG)
    assert d.upstream == BASETEN
    assert d.reason == f"cluster_{status}"


def test_health_check_is_case_insensitive():
    assert route(body("hi"), {}, "HEALTHY", CFG).upstream == CLUSTER


def test_unhealthy_cluster_without_cloud_still_tries_local():
    """A missing Baseten key must degrade the demo, not 500 it."""
    d = route(body("hi"), {}, "degraded", NO_CLOUD)
    assert d.upstream == CLUSTER
    assert d.reason == "cluster_degraded_no_cloud"


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
    assert d.reason == "cluster_restarting"   # not over_size_threshold


def test_size_beats_heavy_tag():
    d = route(body("word " * 3000, model="foo-heavy"), {}, "healthy", CFG)
    assert d.reason == "over_size_threshold"


# ------------------------------------------------------------------- estimation

def test_estimate_tokens_scales_with_length():
    short = estimate_tokens([{"role": "user", "content": "hi"}])
    long = estimate_tokens([{"role": "user", "content": "x" * 4000}])
    assert long > short * 10


def test_estimate_tokens_handles_multipart_content():
    n = estimate_tokens([{"role": "user", "content": [
        {"type": "text", "text": "x" * 400},
        {"type": "image_url", "image_url": {"url": "..."}},
    ]}])
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
