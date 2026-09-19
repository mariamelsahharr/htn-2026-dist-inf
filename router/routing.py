"""
routing.py - the routing policy, as pure functions.

Deliberately has no I/O and no FastAPI import, so the whole policy is unit
testable in milliseconds and you can reason about it at 2am without running a
server. app.py does all the networking and calls in here to decide.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

HEAVY_MARKERS = ("-heavy", ":heavy", "-big", "-cloud")

# Upstream names. These are also the values of the X-Served-By response header.
CLUSTER = "cluster"
BASETEN = "baseten"
CACHE = "cache"


@dataclass
class RouterConfig:
    """Everything the policy needs to know. app.py builds this from env."""
    size_threshold: int = 2048          # prompt_tokens + max_tokens above this -> cloud
    cloud_available: bool = True        # False when no Baseten URL/key is configured
    local_model: str = "llama-3.2-3b-instruct"
    cloud_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    heavy_markers: Tuple[str, ...] = HEAVY_MARKERS
    default_max_tokens: int = 512       # assumed when the client doesn't say


@dataclass
class Decision:
    upstream: str
    reason: str
    prompt_tokens: int
    max_tokens: int
    model_requested: str = ""
    model_sent: str = ""
    forced: bool = False

    def as_log(self) -> Dict[str, Any]:
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


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
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


def is_heavy_model(model: str, markers: Tuple[str, ...] = HEAVY_MARKERS) -> bool:
    m = (model or "").lower()
    return any(marker in m for marker in markers)


def strip_heavy(model: str, markers: Tuple[str, ...] = HEAVY_MARKERS) -> str:
    m = model or ""
    for marker in markers:
        if marker in m.lower():
            idx = m.lower().index(marker)
            return (m[:idx] + m[idx + len(marker):]) or m
    return m


# ---------------------------------------------------------------- the policy

def route(body: Dict[str, Any],
          headers: Dict[str, str],
          cluster_status: str,
          cfg: RouterConfig) -> Decision:
    """Decide which upstream serves this request.

    Priority order, highest first:
      0. X-Force-Upstream header          (demo control / debugging escape hatch)
      1. cluster status is not healthy    -> cloud
      2. prompt + max_tokens over budget  -> cloud
      3. model tagged heavy, or X-Escalate -> cloud
      4. otherwise                        -> cluster

    If no cloud upstream is configured, every cloud verdict falls back to the
    cluster with the reason suffixed `_no_cloud`, so a missing Baseten key
    degrades the demo instead of 500ing it.
    """
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    messages = body.get("messages") or []
    model_req = body.get("model") or cfg.local_model
    prompt_tokens = estimate_tokens(messages)
    max_tokens = int(body.get("max_tokens") or cfg.default_max_tokens)

    def finish(upstream: str, reason: str, forced: bool = False) -> Decision:
        if upstream == BASETEN and not cfg.cloud_available:
            upstream, reason = CLUSTER, reason + "_no_cloud"
        model_sent = cfg.cloud_model if upstream == BASETEN else cfg.local_model
        return Decision(upstream=upstream, reason=reason,
                        prompt_tokens=prompt_tokens, max_tokens=max_tokens,
                        model_requested=model_req, model_sent=model_sent,
                        forced=forced)

    # 0. explicit override. Lets Person 4 make the escalation beat deterministic
    #    on stage instead of hoping the prompt is long enough.
    forced = headers.get("x-force-upstream", "").strip().lower()
    if forced in (CLUSTER, BASETEN):
        return finish(forced, "forced_by_header", forced=True)

    # 1. cluster health wins over everything else
    status = (cluster_status or "unknown").lower()
    if status != "healthy":
        return finish(BASETEN, f"cluster_{status}")

    # 2. size budget
    if prompt_tokens + max_tokens > cfg.size_threshold:
        return finish(BASETEN, "over_size_threshold")

    # 3. explicit escalation
    if is_heavy_model(model_req, cfg.heavy_markers):
        return finish(BASETEN, "heavy_model_tag")
    if headers.get("x-escalate", "").strip().lower() in ("1", "true", "yes"):
        return finish(BASETEN, "escalate_header")

    # 4. default
    return finish(CLUSTER, "default_local")


# ------------------------------------------------------- mid-stream recovery

CONTINUATION_INSTRUCTION = (
    "Continue your previous answer from exactly where it stopped. "
    "Do not repeat any text you already wrote, and do not add a preamble."
)


def continuation_body(original: Dict[str, Any],
                      partial_text: str,
                      model: str,
                      instruction: str = CONTINUATION_INSTRUCTION) -> Dict[str, Any]:
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


def models_payload(cfg: RouterConfig) -> Dict[str, Any]:
    """/v1/models. Clients use this to populate model pickers and to sanity
    check the endpoint before sending real traffic, so it has to be right."""
    ids = [cfg.local_model, cfg.local_model + "-heavy"]
    if cfg.cloud_available and cfg.cloud_model not in ids:
        ids.append(cfg.cloud_model)
    return {
        "object": "list",
        "data": [
            {"id": i, "object": "model", "created": 0, "owned_by": "pi-cluster"}
            for i in ids
        ],
    }
