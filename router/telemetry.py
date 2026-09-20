"""
telemetry.py - Sentry wiring for the router: Tracing + AI Agent Monitoring + Logs.

Design rule (per CLAUDE.md): nothing here may break the Sunday demo.
- No SENTRY_DSN in .env  -> every function is a no-op.
- sentry-sdk not installed -> every function is a no-op.
- All Sentry calls are wrapped so a telemetry failure never takes down a request.

What lands in Sentry when enabled:
- One transaction per HTTP request (FastAPI auto-instrumentation).
- An `gen_ai.invoke_agent` span ("pi-router") wrapping the fallback chain,
  so requests appear in Sentry's AI Agents dashboard.
- One `gen_ai.chat` child span per upstream attempt (cluster / baseten / ...),
  tagged with tier, attempt index, model, and token usage. Failed attempts are
  marked failed with the upstream error -- an escalation is visible as a red
  cluster span followed by a green cloud span in the same trace.
- Structured logs (Sentry Logs) mirroring routing_decisions.jsonl and TTFT.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger("router")

try:
    import sentry_sdk
except ImportError:  # sdk not installed: stay silent, stay no-op
    sentry_sdk = None  # type: ignore[assignment]

_enabled = False

# gen_ai.system values shown in the AI dashboard, per upstream name
_SYSTEM = {
    "cluster": "distributed-llama",
    "baseten": "baseten",
    "openai": "openai",
    "gemini": "gemini",
    "snowflake": "snowflake",
}


def init(dsn: str, environment: str = "demo") -> bool:
    """Call once at startup. Returns True if Sentry is live."""
    global _enabled
    if not (sentry_sdk and dsn):
        return False
    kwargs: dict[str, Any] = {
        "dsn": dsn,
        "environment": environment,
        "traces_sample_rate": 1.0,  # hackathon traffic is tiny; keep every trace
        "send_default_pii": True,  # demo prompts/outputs are fine to show in the AI dashboard
        "include_local_variables": False,  # frames hold Tier objects and headers: no API keys in an event
        "enable_logs": True,  # Sentry Logs product
    }
    try:
        sentry_sdk.init(**kwargs)
    except TypeError:  # older SDK without enable_logs
        kwargs.pop("enable_logs", None)
        sentry_sdk.init(**kwargs)
    _enabled = True
    log.info("sentry telemetry enabled (env=%s)", environment)
    return True


def enabled() -> bool:
    return _enabled


def _safe(fn, *a, **kw) -> None:
    """Telemetry must never take down a request."""
    with contextlib.suppress(Exception):
        fn(*a, **kw)


# ------------------------------------------------------------------ request tags


def tag_request(decision: Any, cluster_status: str) -> None:
    """Tag the active transaction with the routing decision (searchable in Sentry)."""
    if not _enabled or sentry_sdk is None:
        return
    sdk = sentry_sdk  # narrowed here; the closure below cannot see the guard

    def _do() -> None:
        scope = sdk.get_current_scope()
        scope.set_tag("router.decision", getattr(decision, "upstream", "?"))
        scope.set_tag("router.reason", getattr(decision, "reason", "?"))
        scope.set_tag("router.cluster_status", cluster_status)

    _safe(_do)


# ------------------------------------------------------------------ AI agent spans


@contextmanager
def agent_span(decision: Any, cluster_status: str) -> Iterator[Any]:
    """Wraps the whole fallback chain; makes the request show up as an AI agent run."""
    if not _enabled or sentry_sdk is None:
        yield None
        return
    try:
        cm = sentry_sdk.start_span(op="gen_ai.invoke_agent", name="invoke_agent pi-router")
    except Exception:
        yield None
        return
    with cm as span:
        _safe(span.set_data, "gen_ai.operation.name", "invoke_agent")
        _safe(span.set_data, "gen_ai.agent.name", "pi-router")
        _safe(span.set_data, "router.reason", getattr(decision, "reason", "?"))
        _safe(span.set_data, "router.planned_upstream", getattr(decision, "upstream", "?"))
        _safe(span.set_data, "router.cluster_status", cluster_status)
        yield span


@contextmanager
def chat_span(upstream: str, model: str, attempt_index: int) -> Iterator[Any]:
    """One span per upstream attempt. attempt_index > 0 means we're escalating."""
    if not _enabled or sentry_sdk is None:
        yield None
        return
    try:
        cm = sentry_sdk.start_span(op="gen_ai.chat", name=f"chat {model or upstream}")
    except Exception:
        yield None
        return
    with cm as span:
        _safe(span.set_data, "gen_ai.operation.name", "chat")
        _safe(span.set_data, "gen_ai.system", _SYSTEM.get(upstream, upstream))
        _safe(span.set_data, "gen_ai.request.model", model)
        _safe(span.set_data, "router.tier", upstream)
        _safe(span.set_data, "router.attempt", attempt_index)
        _safe(span.set_data, "router.fallback", attempt_index > 0)
        yield span


def mark_failed(span: Any, error: str) -> None:
    if span is None:
        return
    _safe(span.set_status, "internal_error")
    _safe(span.set_data, "router.error", error[:200])
    if _enabled:
        _safe(log.warning, "upstream attempt failed: %s", error[:200])


def record_usage(span: Any, usage: Any) -> None:
    """Attach token usage from an OpenAI-style `usage` object to the chat span."""
    if span is None or not isinstance(usage, dict):
        return
    pairs = (
        ("prompt_tokens", "gen_ai.usage.input_tokens"),
        ("completion_tokens", "gen_ai.usage.output_tokens"),
        ("total_tokens", "gen_ai.usage.total_tokens"),
    )
    for src, dst in pairs:
        if isinstance(usage.get(src), int):
            _safe(span.set_data, dst, usage[src])


# ------------------------------------------------------------------ logs


def log_decision(record: dict[str, Any]) -> None:
    """Mirror routing_decisions.jsonl into Sentry Logs (structured attributes)."""
    if not _enabled or sentry_sdk is None:
        return

    def _do() -> None:
        extra = {
            f"router.{k}": v
            for k, v in record.items()
            if isinstance(v, (str, int, float, bool)) and k not in ("ts", "ts_iso")
        }
        log.info(
            "served_by=%s reason=%s latency_ms=%s fallback=%s",
            record.get("served_by"),
            record.get("reason"),
            record.get("latency_ms"),
            record.get("fallback"),
            extra=extra,
        )

    _safe(_do)


def set_ttft(ttft_ms: int) -> None:
    """Record time-to-first-token on the active transaction (queryable in traces)."""
    if not _enabled or sentry_sdk is None:
        return
    sdk = sentry_sdk

    def _do() -> None:
        span = sdk.get_current_span()
        if span is not None:
            span.set_data("router.ttft_ms", ttft_ms)

    _safe(_do)
