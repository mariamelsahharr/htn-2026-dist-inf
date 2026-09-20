"""
wire.py - the OpenAI chat wire format in both directions. httpx2 carries the SSE
stream; this reads what each event means (text delta, finish, mid-stream error,
usage) once, into a Chunk, and builds the chunks and completions the router emits itself.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any

import httpx2
import orjson


class UpstreamError(RuntimeError):
    pass


# Failures that mean "try the next upstream". Anything else is a bug, or the client
# hanging up, and must not trigger a cloud call on its behalf.
UPSTREAM_ERRORS = (httpx2.HTTPError, UpstreamError, asyncio.TimeoutError)

# Generation parameters that change what a model answers, so they are part of the answer-cache key.
CACHE_PARAMS = ("temperature", "max_tokens", "tools")


def ms_since(t0: float) -> int:
    """Milliseconds since a time.monotonic() stamp."""
    return int((time.monotonic() - t0) * 1000)


def err_text(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:200]


def dumps(obj: Any) -> str:
    return orjson.dumps(obj).decode()


@dataclass(frozen=True, slots=True)
class Chunk:
    """One SSE `data:` line, read once. `raw` is the line as relayed; the rest is what it meant."""

    raw: str
    obj: dict[str, Any] | None  # the JSON object, None for [DONE], blanks and non-objects
    content: str | None  # text delta; "" for a tool-call delta (arrived, no text); None otherwise
    finished: bool  # a choice carried a finish_reason
    error: str | None  # an {"error": ...} object with no choices
    usage: dict[str, Any] | None  # a usage object with completion_tokens
    at: float  # time.monotonic() when it arrived

    @classmethod
    def parse(cls, line: str, at: float | None = None) -> "Chunk":
        obj = _object_of(line)
        content: str | None = None
        finished = False
        for choice in (obj.get("choices") if obj else None) or []:
            if not isinstance(choice, dict):
                continue
            finished = finished or bool(choice.get("finish_reason"))
            if content is None:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content = piece
                elif delta.get("tool_calls"):
                    content = ""
        error = str(obj["error"])[:200] if obj is not None and "error" in obj and "choices" not in obj else None
        usage = obj.get("usage") if obj else None
        if not (isinstance(usage, dict) and usage.get("completion_tokens") is not None):
            usage = None
        return cls(line, obj, content, finished, error, usage, time.monotonic() if at is None else at)

    @classmethod
    def of(cls, obj: dict[str, Any]) -> "Chunk":
        """A chunk the router builds itself."""
        return cls.parse("data: " + dumps(obj))

    @property
    def is_content(self) -> bool:
        return self.content is not None


def _object_of(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = orjson.loads(payload)
    except orjson.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def sse(line: str) -> bytes:
    return (line + "\n\n").encode()


def handover_chunk(from_tier: str, to_tier: str) -> Chunk:
    """A chunk with no choices that says the rest of the answer comes from another tier.
    OpenAI clients skip it; the dashboard switches the badge on it."""
    return Chunk.of(
        {"object": "chat.completion.chunk", "choices": [], "pihive": {"continued_by": to_tier, "after": from_tier}}
    )


def sse_chunk(text: str, model: str, finish: str | None = None) -> bytes:
    obj = {
        "id": "chatcmpl-router",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": ({"content": text} if text else {}), "finish_reason": finish}],
    }
    return sse("data: " + dumps(obj))


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
    if msg.get("content"):
        return msg["content"]
    return orjson.dumps(msg["tool_calls"], option=orjson.OPT_SORT_KEYS).decode() if msg.get("tool_calls") else ""


def _normalized(content: Any) -> str:
    text = content if isinstance(content, str) else dumps(content)
    return " ".join(text.lower().split())


def demo_cache_key(body: dict[str, Any]) -> str:
    """Hash of the normalised last user message: the key warm_cache.py wrote the demo answers under."""
    last = ""
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            last = _normalized(m.get("content"))
            break
    return hashlib.sha256(last.encode()).hexdigest()[:16]


def answer_cache_key(body: dict[str, Any], model: str) -> str:
    """Hash of the whole conversation, normalised, plus the model and the generation parameters."""
    messages = body.get("messages") or []
    doc = {
        "model": model,
        "messages": [{"role": m.get("role"), "content": _normalized(m.get("content"))} for m in messages],
        **{k: body.get(k) for k in CACHE_PARAMS},
    }
    return hashlib.sha256(orjson.dumps(doc, option=orjson.OPT_SORT_KEYS)).hexdigest()
