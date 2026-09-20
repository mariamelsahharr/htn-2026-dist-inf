"""
wire.py - the OpenAI chat wire format in both directions. httpx2 carries the SSE
stream; this reads what each event means (text delta, finish, mid-stream error,
usage) and builds the chunks and completions the router emits itself.
"""

import asyncio
import hashlib
import json
import time
from typing import Any

import httpx2


class UpstreamError(RuntimeError):
    pass


# Failures that mean "try the next upstream". Anything else is a bug, or the client
# hanging up, and must not trigger a cloud call on its behalf.
UPSTREAM_ERRORS = (httpx2.HTTPError, UpstreamError, asyncio.TimeoutError, ValueError)


def ms_since(t0: float) -> int:
    return int((time.time() - t0) * 1000)


def err_text(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:200]


def chunk_of(line: str) -> dict[str, Any] | None:
    """The JSON object in a `data:` line, or None for blanks, [DONE] and non-objects."""
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def error_of(line: str) -> str | None:
    """Upstream error message carried mid-stream as {"error": ...}, else None."""
    obj = chunk_of(line)
    if obj is not None and "error" in obj and "choices" not in obj:
        return str(obj["error"])[:200]
    return None


def finished(line: str) -> bool:
    obj = chunk_of(line)
    return bool(obj) and any(c.get("finish_reason") for c in obj.get("choices") or [] if isinstance(c, dict))


def content_of(line: str) -> str | None:
    """Text delta in an SSE line; "" for a tool-call delta (arrived, no text); None otherwise."""
    obj = chunk_of(line)
    if obj is None:
        return None
    for choice in obj.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        piece = delta.get("content")
        if piece:
            return piece
        if delta.get("tool_calls"):
            return ""
    return None


def sse(line: str) -> bytes:
    return (line + "\n\n").encode()


def handover_line(from_tier: str, to_tier: str) -> str:
    """A chunk with no choices that says the rest of the answer comes from another tier.
    OpenAI clients skip it; the dashboard switches the badge on it."""
    marker = {"object": "chat.completion.chunk", "choices": [], "pihive": {"continued_by": to_tier, "after": from_tier}}
    return "data: " + json.dumps(marker)


def sse_chunk(text: str, model: str, finish: str | None = None) -> bytes:
    obj = {
        "id": "chatcmpl-router",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": ({"content": text} if text else {}), "finish_reason": finish}],
    }
    return sse("data: " + json.dumps(obj))


def chat_completion(text: str, model: str, id_: str = "chatcmpl-router") -> dict[str, Any]:
    return {
        "id": id_,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
    }


def answer_text(data: dict[str, Any]) -> str:
    """What a blocking completion said: its text, or its tool calls when there is no text."""
    msg = ((data.get("choices") or [{}])[0]).get("message") or {}
    return msg.get("content") or (json.dumps(msg["tool_calls"], sort_keys=True) if msg.get("tool_calls") else "")


def cache_key(body: dict[str, Any]) -> str:
    """Hash of the normalised last user message; a typo on stage misses the cache."""
    last = ""
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            last = c if isinstance(c, str) else json.dumps(c)
            break
    return hashlib.sha256(" ".join(last.lower().split()).encode()).hexdigest()[:16]
