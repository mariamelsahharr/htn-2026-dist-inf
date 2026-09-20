"""
test_routing_properties.py - invariants of the routing policy over generated requests.
Run: pytest -q
"""

from hypothesis import given, settings
from hypothesis import strategies as st
from routing import CLUSTER, RouterConfig, Tier, fallback_chain, route

REASONS = {
    "forced_by_header",
    "model_pinned",
    "tools_attached",
    "over_size_threshold",
    "complex_task_code",
    "complex_task_turns",
    "heavy_model_tag",
    "escalate_header",
    "default_local",
}
STATES = ["healthy", "degraded", "degraded_below_min", "restarting", "down", "unreachable", "unknown"]


def config() -> RouterConfig:
    return RouterConfig(
        cloud_available=True,
        tiers={"openai": Tier("openai", "gpt-x", handles_tools=True), "gemini": Tier("gemini", "gem-x")},
        catalogs={"gemini": ("gem-big",)},
        tool_tier="openai",
    )


CFG = config()
MODELS = ["auto", CFG.local_model, CFG.local_model + "-heavy", "big-70b", "gpt-x", "gem-x", "gem-big", "nope"]

message = st.fixed_dictionaries(
    {"role": st.sampled_from(["user", "assistant", "system"]), "content": st.text(max_size=400)}
)
body = st.fixed_dictionaries(
    {
        "messages": st.lists(message, min_size=1, max_size=20),
        "model": st.sampled_from(MODELS),
    },
    optional={
        "max_tokens": st.integers(min_value=1, max_value=8000),
        "tools": st.just([{"type": "function", "function": {"name": "f"}}]),
    },
)
headers = st.fixed_dictionaries(
    {},
    optional={
        "X-Force-Upstream": st.sampled_from(["cluster", "openai", "gemini", "baseten", "bogus"]),
        "X-Escalate": st.sampled_from(["1", "0"]),
    },
)


@settings(max_examples=300, deadline=None)
@given(body=body, hdrs=headers, state=st.sampled_from(STATES))
def test_every_decision_is_servable_and_explained(body, hdrs, state):
    d = route(body, hdrs, state, CFG)
    assert d.upstream == CLUSTER or d.upstream in CFG.tiers
    assert d.reason.removesuffix("_no_cloud").removeprefix("cluster_") in REASONS | set(STATES)
    assert d.model_sent
    chain = fallback_chain(d, state, CFG)
    assert chain[0] == d.upstream and len(chain) == len(set(chain))


@settings(max_examples=200, deadline=None)
@given(body=body, state=st.sampled_from(STATES))
def test_a_valid_force_header_always_wins(body, state):
    for forced in ("cluster", "openai", "gemini"):
        d = route(body, {"X-Force-Upstream": forced}, state, CFG)
        assert d.upstream == forced and d.forced and d.reason == "forced_by_header"


@settings(max_examples=200, deadline=None)
@given(body=body, state=st.sampled_from(STATES))
def test_a_named_cloud_model_goes_to_its_tier_verbatim(body, state):
    for model, tier in (("gpt-x", "openai"), ("gem-big", "gemini")):
        d = route({**body, "model": model}, {}, state, CFG)
        assert d.upstream == tier and d.model_sent == model and d.reason == "model_pinned"


@settings(max_examples=200, deadline=None)
@given(body=body, state=st.sampled_from(["healthy", "degraded"]))
def test_nothing_reaches_the_cluster_when_it_cannot_serve(body, state):
    for down in ("down", "unreachable", "restarting", "degraded_below_min"):
        d = route({**body, "model": "auto"}, {}, down, CFG)
        assert d.upstream != CLUSTER
