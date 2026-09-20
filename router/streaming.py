"""
streaming.py - how one upstream answer turns into bytes for a client. The router runs
the request once, the same way for every API; an Api says what each chunk becomes and
how a blocking answer or an error is shaped.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from responses import ResponseBuilder, error_body
from wire import Chunk, UpstreamError, sse


class Sink(Protocol):
    """What the client gets at each moment of the stream."""

    @property
    def text(self) -> str: ...  # visible answer so far

    @property
    def done(self) -> bool: ...  # the upstream already sent a finish reason

    def start(self) -> Iterable[bytes]: ...

    def line(self, chunk: Chunk) -> Iterable[bytes]: ...  # raises UpstreamError on a mid-stream error

    def finish(self) -> Iterable[bytes]: ...

    def fail(self, error: str) -> Iterable[bytes]: ...


class ChatSink:
    """Chat completions: upstream chunks pass through untouched, ending in [DONE]."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.done = False

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def start(self) -> Iterable[bytes]:
        return ()

    def line(self, chunk: Chunk) -> Iterable[bytes]:
        if chunk.error:
            raise UpstreamError(chunk.error)
        if chunk.content:
            self.parts.append(chunk.content)
        self.done = self.done or chunk.finished
        return (sse(chunk.raw),)

    def finish(self) -> Iterable[bytes]:
        return (sse("data: [DONE]"),)

    def fail(self, error: str) -> Iterable[bytes]:
        return (sse("data: [DONE]"),)  # the client already has the partial text; the error is logged


class ResponsesSink:
    """Codex's Responses API: chunks become response.* events through the builder."""

    def __init__(self, builder: ResponseBuilder) -> None:
        self.builder = builder
        self.done = False

    @property
    def text(self) -> str:
        return "".join(self.builder.text)

    def start(self) -> Iterable[bytes]:
        return [ev.encode() for ev in self.builder.start()]

    def line(self, chunk: Chunk) -> Iterable[bytes]:
        if chunk.error:
            raise UpstreamError(chunk.error)
        self.done = self.done or chunk.finished
        return [ev.encode() for ev in self.builder.feed(chunk.obj)] if chunk.obj is not None else ()

    def finish(self) -> Iterable[bytes]:
        return [ev.encode() for ev in self.builder.finish()]

    def fail(self, error: str) -> Iterable[bytes]:
        return [ev.encode() for ev in self.builder.finish(error=error)]


class Api(Protocol):
    """The face one request wears: which sink streams it, how a blocking answer and an error look,
    and whether the answer caches and a dead stream may be continued by another tier."""

    name: str
    cached: bool
    continuation: bool

    def sink(self) -> Sink: ...

    def blocking(self, data: dict[str, Any]) -> dict[str, Any]: ...

    def error(self, message: str) -> dict[str, Any]: ...


@dataclass
class ChatApi:
    name: str = "chat"
    cached: bool = True
    continuation: bool = True

    def sink(self) -> Sink:
        return ChatSink()

    def blocking(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def error(self, message: str) -> dict[str, Any]:
        return {"error": {"message": message}}


@dataclass
class ResponsesApi:
    builder: ResponseBuilder
    name: str = "responses"
    cached: bool = field(default=False, init=False)
    continuation: bool = field(default=False, init=False)

    def sink(self) -> Sink:
        return ResponsesSink(self.builder)

    def blocking(self, data: dict[str, Any]) -> dict[str, Any]:
        for _ in self.builder.feed(data):
            pass
        return self.builder.response_object()

    def error(self, message: str) -> dict[str, Any]:
        return error_body(message)
