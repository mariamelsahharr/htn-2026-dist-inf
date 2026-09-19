"""
responses.py - OpenAI Responses API <-> Chat Completions translation, pure functions.

Codex speaks only the Responses API; every upstream speaks chat completions.
Shapes follow the OpenAI reference and the subset Codex's parser reads:
response.created, output_item.added, output_text.delta, output_item.done,
response.completed (with usage), response.failed.
"""

import json
import time
import uuid
from typing import Any
from collections.abc import Iterator

_TEXT_TYPES = ("input_text", "output_text", "text")


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _content_to_chat(content: Any) -> Any:
    """Responses content parts -> chat content (string, or parts when images are present)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        t = part.get("type")
        if t in _TEXT_TYPES:
            texts.append(part.get("text") or "")
            parts.append({"type": "text", "text": part.get("text") or ""})
        elif t == "input_image" and part.get("image_url"):
            parts.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
    if any(p["type"] == "image_url" for p in parts):
        return parts
    return "\n".join(texts)


def _output_to_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(p.get("text", "") for p in output if isinstance(p, dict))
    return json.dumps(output)


def responses_to_chat(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return (chat completions body, custom tools by name).

    Freeform "custom" tools have no chat equivalent, so they become function
    tools with a single string `input`; the name map lets the response side turn
    the call back into a custom_tool_call item."""
    messages: list[dict[str, Any]] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})

    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]

    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            messages.append({"role": "assistant", "content": None, "tool_calls": list(pending)})
            pending.clear()

    for item in items or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type") or "message"
        if kind == "message":
            flush()
            role = item.get("role") or "user"
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": _content_to_chat(item.get("content"))})
        elif kind == "function_call":
            pending.append({"id": item.get("call_id") or _uid("call"), "type": "function",
                            "function": {"name": item.get("name", ""),
                                         "arguments": item.get("arguments") or "{}"}})
        elif kind == "custom_tool_call":
            pending.append({"id": item.get("call_id") or _uid("call"), "type": "function",
                            "function": {"name": item.get("name", ""),
                                         "arguments": json.dumps({"input": item.get("input") or ""})}})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            flush()
            messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                             "content": _output_to_text(item.get("output"))})
        # reasoning, web_search_call and friends have no chat equivalent
    flush()

    tools: list[dict[str, Any]] = []
    custom: dict[str, dict[str, Any]] = {}
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        kind = tool.get("type")
        if kind == "function":
            fn: dict[str, Any] = {"name": tool.get("name", ""),
                                  "parameters": tool.get("parameters") or {"type": "object", "properties": {}}}
            if tool.get("description"):
                fn["description"] = tool["description"]
            tools.append({"type": "function", "function": fn})
        elif kind == "custom":
            custom[tool.get("name", "")] = tool
            desc = tool.get("description") or ""
            fmt = tool.get("format") or {}
            if fmt.get("definition"):
                desc += f"\nInput grammar ({fmt.get('syntax', '')}):\n{fmt['definition']}"
            tools.append({"type": "function", "function": {
                "name": tool.get("name", ""), "description": desc,
                "parameters": {"type": "object", "required": ["input"],
                               "properties": {"input": {"type": "string", "description": "Raw tool input"}}}}})

    chat: dict[str, Any] = {"model": body.get("model"), "messages": messages, "stream": bool(body.get("stream"))}
    if tools:
        chat["tools"] = tools
        choice = body.get("tool_choice")
        if isinstance(choice, str) and choice in ("auto", "none", "required"):
            chat["tool_choice"] = choice
        if body.get("parallel_tool_calls") is not None:
            chat["parallel_tool_calls"] = bool(body["parallel_tool_calls"])
    for src, dst in (("max_output_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p")):
        if body.get(src) is not None:
            chat[dst] = body[src]
    return chat, custom


class ResponseBuilder:
    """Feed chat-completion chunks (or one blocking response); emit Responses events."""

    def __init__(self, model: str, custom_tools: dict[str, Any] | None = None) -> None:
        self.id = _uid("resp")
        self.model = model or ""
        self.custom = custom_tools or {}
        self.created_at = int(time.time())
        self.seq = 0
        self.text: list[str] = []
        self.msg_id = _uid("msg")
        self.msg_open = False
        self.calls: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] | None = None
        self.output: list[dict[str, Any]] = []

    # ----- events ---------------------------------------------------------

    def _event(self, kind: str, **fields: Any) -> str:
        self.seq += 1
        obj = {"type": kind, "sequence_number": self.seq, **fields}
        return f"event: {kind}\ndata: {json.dumps(obj)}\n\n"

    def _response(self, status: str, error: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "id": self.id, "object": "response", "created_at": self.created_at, "status": status,
            "model": self.model, "output": list(self.output), "error": error,
            "incomplete_details": None, "usage": self.usage_object() if status == "completed" else None,
        }

    def usage_object(self) -> dict[str, Any]:
        u = self.usage or {}
        inp = int(u.get("prompt_tokens") or 0)
        out = int(u.get("completion_tokens") or 0)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0}}

    def start(self) -> Iterator[str]:
        yield self._event("response.created", response=self._response("in_progress"))
        yield self._event("response.in_progress", response=self._response("in_progress"))

    def feed(self, chunk: dict[str, Any]) -> Iterator[str]:
        """One streamed chunk or one blocking completion object."""
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            if content:
                if not self.msg_open:
                    self.msg_open = True
                    yield self._event("response.output_item.added", output_index=0, item={
                        "id": self.msg_id, "type": "message", "status": "in_progress",
                        "role": "assistant", "content": []})
                    yield self._event("response.content_part.added", item_id=self.msg_id,
                                      output_index=0, content_index=0,
                                      part={"type": "output_text", "text": "", "annotations": []})
                self.text.append(content)
                yield self._event("response.output_text.delta", item_id=self.msg_id,
                                  output_index=0, content_index=0, delta=content)
            for i, tc in enumerate(delta.get("tool_calls") or []):
                slot = self.calls.setdefault(tc.get("index", i), {"id": None, "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"] if not slot["name"] else slot["name"] + fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]

    def finish(self, error: str | None = None) -> Iterator[str]:
        idx = 0
        if self.msg_open:
            text = "".join(self.text)
            yield self._event("response.output_text.done", item_id=self.msg_id, output_index=0,
                              content_index=0, text=text)
            part = {"type": "output_text", "text": text, "annotations": []}
            yield self._event("response.content_part.done", item_id=self.msg_id, output_index=0,
                              content_index=0, part=part)
            item = {"id": self.msg_id, "type": "message", "status": "completed",
                    "role": "assistant", "content": [part]}
            self.output.append(item)
            yield self._event("response.output_item.done", output_index=0, item=item)
            idx = 1
        for key in sorted(self.calls):
            call = self.calls[key]
            call_id = call["id"] or _uid("call")
            if call["name"] in self.custom:
                try:
                    inp = json.loads(call["arguments"] or "{}").get("input", "")
                except (ValueError, AttributeError):
                    inp = call["arguments"]
                item = {"id": _uid("ctc"), "type": "custom_tool_call", "status": "completed",
                        "call_id": call_id, "name": call["name"], "input": inp}
            else:
                item = {"id": _uid("fc"), "type": "function_call", "status": "completed",
                        "call_id": call_id, "name": call["name"], "arguments": call["arguments"] or "{}"}
            yield self._event("response.output_item.added", output_index=idx,
                              item={**item, "status": "in_progress"})
            self.output.append(item)
            yield self._event("response.output_item.done", output_index=idx, item=item)
            idx += 1
        if error:
            yield self._event("response.failed", response=self._response(
                "failed", error={"code": "upstream_error", "message": error}))
            return
        yield self._event("response.completed", response=self._response("completed"))

    def response_object(self) -> dict[str, Any]:
        """Non-streaming result after feed(); finalises output items."""
        for _ in self.finish():
            pass
        return self._response("completed")


def error_body(message: str, code: str = "upstream_error") -> dict[str, Any]:
    return {"error": {"type": "server_error", "code": code, "message": message}}
