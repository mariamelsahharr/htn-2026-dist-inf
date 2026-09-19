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
COMPLEX_KEYWORDS = (
    "refactor", "rewrite", "migrate", "architecture", "redesign",
    "across the codebase", "entire repo", "whole repo", "all files",
    "from scratch",
)
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)

# Upstream names. These are also the values of the X-Served-By response header.
CLUSTER = "cluster"
BASETEN = "baseten"
CACHE = "cache"

# Fallback order between cloud tiers.
DEFAULT_CLOUD_ORDER = ("baseten", "gemini", "openai", "snowflake")


@dataclass
class Tier:
    """One OpenAI-compatible upstream. The policy reads name/model; app.py uses the rest."""
    name: str
    model: str
    base_url: str = ""       # ends in /v1 for every tier, the cluster included
    api_key: str = ""
    handles_tools: bool = False
    is_local: bool = False

    @property
    def chat_url(self) -> str:
        return self.base_url + "/chat/completions"

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h


@dataclass
class RouterConfig:
    """Everything the policy needs to know. app.py builds this from env."""
    size_threshold: int = 2048          # prompt_tokens + max_tokens above this -> cloud
    cloud_available: bool = True        # False when no Baseten URL/key is configured
    local_model: str = "llama-3.2-3b-instruct"
    cloud_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    heavy_markers: tuple[str, ...] = HEAVY_MARKERS
    default_max_tokens: int = 512       # assumed when the client doesn't say
    tiers: dict[str, Tier] = field(default_factory=dict)  # baseten is synthesised if absent
    local_base_url: str = ""
    cloud_order: tuple[str, ...] = DEFAULT_CLOUD_ORDER
    heavy_tier: str = BASETEN           # size/health/heavy-tag escalations go here
    tool_tier: str | None = None     # requests with tool definitions go here
    complex_keywords: tuple[str, ...] = COMPLEX_KEYWORDS
    code_lines_threshold: int = 120     # fenced code lines in the conversation
    max_local_turns: int = 12           # messages

    def __post_init__(self) -> None:
        if self.cloud_available and BASETEN not in self.tiers:
            self.tiers[BASETEN] = Tier(BASETEN, self.cloud_model)
        if not self.cloud_available:
            self.tiers.pop(BASETEN, None)
        if self.tool_tier and self.tool_tier not in self.tiers:
            self.tool_tier = None

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

    def model_for(self, upstream: str) -> str:
        return self.tier(upstream).model


@dataclass
class Decision:
    upstream: str
    reason: str
    prompt_tokens: int
    max_tokens: int
    model_requested: str = ""
    model_sent: str = ""
    forced: bool = False

    def as_log(self) -> dict[str, Any]:
        return {
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


def classify_complexity(messages: list[dict[str, Any]], cfg: "RouterConfig") -> str | None:
    """Reason string when the task is too complex for the local model, else None."""
    code_lines = 0
    last_user = ""
    for m in messages or []:
        text = _text_of(m.get("content"))
        for block in _FENCE_RE.findall(text):
            code_lines += block.count("\n") + 1
        if m.get("role") == "user":
            last_user = text
    if code_lines > cfg.code_lines_threshold:
        return "complex_task_code"
    lowered = last_user.lower()
    if any(kw in lowered for kw in cfg.complex_keywords):
        return "complex_task_keyword"
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
            return (m[:idx] + m[idx + len(marker):]) or m
    return m


# ---------------------------------------------------------------- the policy

def route(body: dict[str, Any],
          headers: dict[str, str],
          cluster_status: str,
          cfg: RouterConfig) -> Decision:
    """Pick the upstream. Priority: force header, tools, cluster health, size,
    complexity, heavy tag / X-Escalate, else cluster. Unconfigured tiers degrade
    to heavy tier, then cluster with `_no_cloud` appended."""
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    messages = body.get("messages") or []
    model_req = body.get("model") or cfg.local_model
    prompt_tokens = estimate_tokens(messages)
    max_tokens = int(body.get("max_tokens") or cfg.default_max_tokens)
    heavy = cfg.heavy_tier if cfg.heavy_tier in cfg.tiers else None

    def finish(upstream: str, reason: str, forced: bool = False) -> Decision:
        if upstream != CLUSTER and upstream not in cfg.tiers:
            if heavy and upstream != heavy:
                upstream = heavy
            else:
                upstream, reason = CLUSTER, reason + "_no_cloud"
        return Decision(upstream=upstream, reason=reason,
                        prompt_tokens=prompt_tokens, max_tokens=max_tokens,
                        model_requested=model_req, model_sent=cfg.model_for(upstream),
                        forced=forced)

    # 0. explicit override (demo control)
    forced = headers.get("x-force-upstream", "").strip().lower()
    if forced == CLUSTER or forced in cfg.tiers:
        return finish(forced, "forced_by_header", forced=True)

    # 1. tool calls: the local model is not tool-tuned
    if body.get("tools") and cfg.tool_tier:
        return finish(cfg.tool_tier, "tools_attached")

    # 2. cluster health
    status = (cluster_status or "unknown").lower()
    if status != "healthy":
        return finish(heavy or BASETEN, f"cluster_{status}")

    # 3. size budget
    if prompt_tokens + max_tokens > cfg.size_threshold:
        return finish(heavy or BASETEN, "over_size_threshold")

    # 4. task complexity
    complexity = classify_complexity(messages, cfg)
    if complexity:
        return finish(heavy or BASETEN, complexity)

    # 5. explicit escalation
    if is_heavy_model(model_req, cfg.heavy_markers):
        return finish(heavy or BASETEN, "heavy_model_tag")
    if headers.get("x-escalate", "").strip().lower() in ("1", "true", "yes"):
        return finish(heavy or BASETEN, "escalate_header")

    # 6. default
    return finish(CLUSTER, "default_local")


# ------------------------------------------------------------- fallback chain

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
    cluster_ok = (cluster_status or "").lower() == "healthy"
    if (CLUSTER not in chain and cluster_ok
            and not decision.reason.startswith(CLUSTER_UNFIT_REASONS)):
        chain.append(CLUSTER)
    return chain


# ------------------------------------------------------- mid-stream recovery

CONTINUATION_INSTRUCTION = (
    "Continue your previous answer from exactly where it stopped. "
    "Do not repeat any text you already wrote, and do not add a preamble."
)


def continuation_body(original: dict[str, Any],
                      partial_text: str,
                      model: str,
                      instruction: str = CONTINUATION_INSTRUCTION) -> dict[str, Any]:
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
    ids = [cfg.local_model, cfg.local_model + "-heavy"]
    for name in cfg.cloud_tiers():
        model = cfg.tiers[name].model
        if model and model not in ids:
            ids.append(model)
    return {
        "object": "list",
        "data": [
            {"id": i, "object": "model", "created": 0, "owned_by": "pi-cluster"}
            for i in ids
        ],
    }
