"""
routing.py - the routing policy, as pure functions.

Deliberately has no I/O and no FastAPI import, so the whole policy is unit
testable in milliseconds and you can reason about it at 2am without running a
server. app.py does all the networking and calls in here to decide.
"""

import re
from dataclasses import dataclass, field
from typing import Any

HEAVY_MARKERS = ("-heavy", ":heavy", "-big", "-cloud")

# Words in the last user turn that mean "this is a whole-codebase or design task",
# which a 3B model answers confidently and wrong. Matched case-insensitively.
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)

# Upstream names. These are also the values of the X-Served-By response header.
CLUSTER = "cluster"
BASETEN = "baseten"
CACHE = "cache"

# Fallback order between cloud tiers.
DEFAULT_CLOUD_ORDER = ("baseten", "gemini", "openai", "snowflake")

# Supervisor states in which the cluster takes traffic.
SERVING_STATES = ("healthy", "degraded")


def cluster_state(status: dict[str, Any], min_local_nodes: int) -> str:
    """Routing verdict from the supervisor's /status document (cluster/supervisor/status.example.json)."""
    state = str(status.get("state") or status.get("status") or "unknown").lower()
    nodes = status.get("nodes_active")
    if state == "degraded" and isinstance(nodes, int) and nodes < min_local_nodes:
        return "degraded_below_min"
    return state


@dataclass
class Tier:
    """One OpenAI-compatible upstream. The policy reads name/model; app.py uses the rest."""

    name: str
    model: str
    base_url: str = ""  # ends in /v1 for every tier, the cluster included
    api_key: str = field(default="", repr=False)  # never in a repr, a log line or a Sentry event
    handles_tools: bool = False
    is_local: bool = False
    reasoning_effort: str | None = None  # OpenAI/Gemini knob; some models need "none" to accept tools
    tools_model: str | None = None  # used instead of `model` when the request carries tools
    usage_in_stream: bool = False  # tier honours stream_options.include_usage (exact token counts)

    def model_for_request(self, with_tools: bool) -> str:
        return self.tools_model if (with_tools and self.tools_model) else self.model

    @property
    def chat_url(self) -> str:
        return self.base_url + "/chat/completions"

    def payload(self, body: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        out = {**body, **overrides}
        if self.reasoning_effort:
            out["reasoning_effort"] = self.reasoning_effort
        if out.get("stream") and self.usage_in_stream:
            out["stream_options"] = {**(out.get("stream_options") or {}), "include_usage": True}
        return out

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h


@dataclass
class RouterConfig:
    """Everything the policy needs to know. app.py builds this from env."""

    # prompt_tokens + max_tokens above this -> cloud. The cluster runs dllama with
    # --max-seq-len 4096, so this is "will it fit the context", with room for the estimate
    # being ~15% off; anything that fits stays local, however long it reads.
    size_threshold: int = 3584
    cloud_available: bool = True  # False when no Baseten URL/key is configured
    local_model: str = "qwen3-30b-a3b"
    cloud_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    heavy_markers: tuple[str, ...] = HEAVY_MARKERS
    default_max_tokens: int = 512  # assumed when the client doesn't say
    tiers: dict[str, Tier] = field(default_factory=dict)  # baseten is synthesised if absent
    local_base_url: str = ""
    cloud_order: tuple[str, ...] = DEFAULT_CLOUD_ORDER
    heavy_tier: str | None = None  # size/health/heavy-tag escalations go here; default: the first cloud tier
    tool_tier: str | None = None  # requests with tool definitions go here
    # "too complex for a small model" is meant to catch the outliers only: a few hundred
    # lines of pasted code or a very long back-and-forth, not an ordinary paste or chat.
    code_lines_threshold: int = 400  # fenced code lines in the conversation
    max_local_turns: int = 40  # messages
    catalogs: dict[str, tuple[str, ...]] = field(default_factory=dict)  # tier -> every model it offers

    def __post_init__(self) -> None:
        if self.cloud_available and not self.tiers:
            self.tiers[BASETEN] = Tier(BASETEN, self.cloud_model)
        if not self.cloud_available:
            self.tiers.clear()
        if self.tool_tier and self.tool_tier not in self.tiers:
            self.tool_tier = None
        if self.heavy_tier not in self.tiers:
            self.heavy_tier = next(iter(self.cloud_tiers()), None)

    def cloud_tiers(self) -> list[str]:
        """Configured cloud tier names in fallback order."""
        ordered = [n for n in self.cloud_order if n in self.tiers]
        ordered += [n for n in self.tiers if n not in ordered]
        return ordered

    @property
    def local_tier(self) -> Tier:
        return Tier(CLUSTER, self.local_model, self.local_base_url, is_local=True)

    def tier(self, upstream: str) -> Tier:
        if upstream == CLUSTER:
            return self.local_tier
        return self.tiers[upstream]

    def known_models(self, name: str) -> set[str]:
        """The tier's configured models plus its live catalog."""
        tier = self.tiers[name]
        return {m for m in (tier.model, tier.tools_model) if m} | set(self.catalogs.get(name, ()))

    def tier_for_model(self, model: str) -> str | None:
        """The tier that offers this model by name; None for `auto` and the local ids."""
        wanted = normalize_model_id(model)
        for name in self.cloud_tiers():
            if wanted in self.known_models(name):
                return name
        return None

    def model_for(self, upstream: str, with_tools: bool = False, requested: str | None = None) -> str:
        """What to send: the model the client named if this tier offers it, else the tier's default."""
        if upstream != CLUSTER and requested:
            wanted = normalize_model_id(requested)
            if wanted in self.known_models(upstream):
                return wanted
        return self.tier(upstream).model_for_request(with_tools)


@dataclass
class Decision:
    upstream: str
    reason: str
    prompt_tokens: int
    max_tokens: int
    model_requested: str = ""
    model_sent: str = ""
    forced: bool = False
    request_id: str = ""

    def as_log(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "routed_to": self.upstream,
            "reason": self.reason,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "model_requested": self.model_requested,
            "model_sent": self.model_sent,
            "forced": self.forced,
        }


# --------------------------------------------------------------------- tokens


def _text_of(content: Any) -> str:
    """Messages may carry a plain string or a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(part.get("text") or "")
            elif isinstance(part, str):
                out.append(part)
        return " ".join(out)
    return str(content)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """~4 chars per token, plus a few tokens of role/framing overhead per message.

    A real tokenizer would be more accurate but would mean shipping a tokenizer
    to the router and keeping it in sync with whatever model is loaded. The
    threshold is a fuzzy policy knob, not an accounting figure, so an estimate
    that is consistently within ~15% is good enough. Tune the threshold, not this.
    """
    total = 0
    for m in messages or []:
        total += len(_text_of(m.get("content"))) // 4 + 4
    return total


NOT_CHAT = (
    "embedding",
    "whisper",
    "tts",
    "dall-e",
    "moderation",
    "realtime",
    "transcribe",
    "image",
    "audio",
    "search",
    "babbage",
    "davinci",
    "instruct-0",
    "computer-use",
    "veo",
    "imagen",
    "aqa",
    "embed",
)


def normalize_model_id(model: str) -> str:
    """Gemini's compatibility endpoint lists `models/x`; clients and the router say `x`."""
    return model.removeprefix("models/")


def chat_models(ids: list[str]) -> tuple[str, ...]:
    """A provider's list, trimmed to what can answer a chat: no embeddings, speech, image or search models."""
    out: list[str] = []
    for raw in ids:
        m = normalize_model_id(str(raw))
        if m and not any(tag in m.lower() for tag in NOT_CHAT) and m not in out:
            out.append(m)
    return tuple(out)


def classify_complexity(messages: list[dict[str, Any]], cfg: "RouterConfig") -> str | None:
    """Reason string when the task is too complex for the local model, else None."""
    code_lines = 0
    for m in messages or []:
        for block in _FENCE_RE.findall(_text_of(m.get("content"))):
            code_lines += block.count("\n") + 1
    if code_lines > cfg.code_lines_threshold:
        return "complex_task_code"
    if len(messages or []) > cfg.max_local_turns:
        return "complex_task_turns"
    return None


def is_heavy_model(model: str, markers: tuple[str, ...] = HEAVY_MARKERS) -> bool:
    m = (model or "").lower()
    return any(marker in m for marker in markers)


def strip_heavy(model: str, markers: tuple[str, ...] = HEAVY_MARKERS) -> str:
    m = model or ""
    for marker in markers:
        if marker in m.lower():
            idx = m.lower().index(marker)
            return (m[:idx] + m[idx + len(marker) :]) or m
    return m


# ---------------------------------------------------------------- the policy


def route(body: dict[str, Any], headers: dict[str, str], cluster_status: str, cfg: RouterConfig) -> Decision:
    """Pick the upstream. Priority: force header, tools, cluster health, size,
    complexity, heavy tag / X-Escalate, else cluster. Unconfigured tiers degrade
    to heavy tier, then cluster with `_no_cloud` appended."""
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    messages = body.get("messages") or []
    model_req = body.get("model") or cfg.local_model
    prompt_tokens = estimate_tokens(messages)
    max_tokens = int(body.get("max_tokens") or cfg.default_max_tokens)
    heavy = cfg.heavy_tier

    def finish(upstream: str, reason: str, forced: bool = False) -> Decision:
        if upstream != CLUSTER and upstream not in cfg.tiers:
            if heavy and upstream != heavy:
                upstream = heavy
            else:
                upstream, reason = CLUSTER, reason + "_no_cloud"
        return Decision(
            upstream=upstream,
            reason=reason,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            model_requested=model_req,
            model_sent=cfg.model_for(upstream, bool(body.get("tools")), model_req),
            forced=forced,
        )

    # 0. explicit override: the X-Force-Upstream header, or asking for a tier's model by name
    #    (what any OpenAI client can do; `auto` and the local ids leave the choice to the router)
    forced = headers.get("x-force-upstream", "").strip().lower()
    if forced == CLUSTER or forced in cfg.tiers:
        return finish(forced, "forced_by_header", forced=True)
    pinned = cfg.tier_for_model(model_req)
    if pinned:
        return finish(pinned, "model_pinned", forced=True)

    def escalate(reason: str) -> Decision:
        """To the heavy tier; without any cloud, the cluster takes it and the reason says so."""
        return finish(heavy, reason) if heavy else finish(CLUSTER, reason + "_no_cloud")

    # 1. tool calls: the local model is not tool-tuned
    if body.get("tools") and cfg.tool_tier:
        return finish(cfg.tool_tier, "tools_attached")

    # 2. cluster health: healthy and degraded both serve; the supervisor's reduced
    #    set is the whole point of degrading instead of dying
    status = (cluster_status or "unknown").lower()
    if status not in SERVING_STATES:
        return escalate(f"cluster_{status}")

    # 3. size budget
    if prompt_tokens + max_tokens > cfg.size_threshold:
        return escalate("over_size_threshold")

    # 4. task complexity
    complexity = classify_complexity(messages, cfg)
    if complexity:
        return escalate(complexity)

    # 5. explicit escalation
    if is_heavy_model(model_req, cfg.heavy_markers):
        return escalate("heavy_model_tag")
    if headers.get("x-escalate", "").strip().lower() in ("1", "true", "yes"):
        return escalate("escalate_header")

    # 6. default
    return finish(CLUSTER, "default_local")


# ------------------------------------------------------------- fallback chain


class Breaker:
    """Per-upstream circuit breaker: after `failures` consecutive errors an upstream
    is skipped for `cooldown` seconds, so a dead cloud does not cost a connect
    timeout on every request. The cluster is exempt; the supervisor owns its health."""

    def __init__(self, failures: int = 2, cooldown: float = 30.0) -> None:
        self.failures = max(1, failures)
        self.cooldown = cooldown
        self.fails: dict[str, int] = {}
        self.open_until: dict[str, float] = {}

    def is_open(self, upstream: str, now: float) -> bool:
        return self.open_until.get(upstream, 0.0) > now

    def record_failure(self, upstream: str, now: float) -> bool:
        """Returns True when this failure opened the breaker."""
        n = self.fails.get(upstream, 0) + 1
        self.fails[upstream] = n
        if n >= self.failures:
            self.open_until[upstream] = now + self.cooldown
            self.fails[upstream] = 0
            return True
        return False

    def record_success(self, upstream: str) -> None:
        self.fails.pop(upstream, None)
        self.open_until.pop(upstream, None)

    def snapshot(self, now: float) -> dict[str, float]:
        return {u: round(t - now, 1) for u, t in self.open_until.items() if t > now}


def missing_required_tool_call(request: dict[str, Any], message: dict[str, Any]) -> bool:
    """True when the client demanded a tool call (tool_choice required / named) and
    the model answered in prose instead: for the local model that is a miss worth
    falling through on, not an answer."""
    choice = request.get("tool_choice")
    demanded = choice == "required" or isinstance(choice, dict)
    return demanded and not message.get("tool_calls")


# Escalation reasons that make the cluster an invalid fallback.
CLUSTER_UNFIT_REASONS = ("over_size_threshold", "tools_attached")


def fallback_chain(decision: Decision, cluster_status: str, cfg: RouterConfig) -> list[str]:
    """Upstreams to try in order: primary, other clouds, then a healthy cluster
    unless the request was escalated for size or tools. Forced = no fallback."""
    chain = [decision.upstream]
    if decision.forced:
        return chain
    for name in cfg.cloud_tiers():
        if name not in chain:
            chain.append(name)
    cluster_ok = (cluster_status or "").lower() in SERVING_STATES
    if CLUSTER not in chain and cluster_ok and not decision.reason.startswith(CLUSTER_UNFIT_REASONS):
        chain.append(CLUSTER)
    return chain


# ------------------------------------------------------- mid-stream recovery

CONTINUATION_INSTRUCTION = (
    "Continue your previous answer from exactly where it stopped. "
    "Do not repeat any text you already wrote, and do not add a preamble."
)


def continuation_body(
    original: dict[str, Any], partial_text: str, model: str, instruction: str = CONTINUATION_INSTRUCTION
) -> dict[str, Any]:
    """Build the cloud request that resumes a stream the cluster dropped halfway.

    Once bytes have gone downstream we cannot silently re-run the request: the
    client has already rendered part of an answer. So we hand the cloud model
    what was produced so far and ask it to carry on. The seam is usually
    invisible, and it is honest - the router logs that it happened.
    """
    msgs = list(original.get("messages") or [])
    if partial_text:
        msgs.append({"role": "assistant", "content": partial_text})
        msgs.append({"role": "user", "content": instruction})
    body = dict(original)
    body["messages"] = msgs
    body["model"] = model
    return body


def models_payload(cfg: RouterConfig) -> dict[str, Any]:
    """/v1/models. Clients use this to populate model pickers and to sanity
    check the endpoint before sending real traffic, so it has to be right."""
    entries = [(cfg.local_model, CLUSTER)]
    if cfg.heavy_tier:  # no "-heavy" alias when nothing heavy exists to take it
        entries.append((cfg.local_model + "-heavy", cfg.heavy_tier))
    seen = {m for m, _ in entries}
    for name in cfg.cloud_tiers():
        tier = cfg.tiers[name]
        for model in (tier.model, tier.tools_model, *cfg.catalogs.get(name, ())):
            if model and model not in seen:
                entries.append((model, name))
                seen.add(model)
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": 0, "owned_by": owner} for m, owner in entries],
    }
