"""
streaming.py - how one upstream stream turns into bytes for a client. The router runs
the stream once, the same way for every API; a sink only says what each line becomes.
"""

from collections.abc import Iterable
from typing import Protocol

from responses import ResponseBuilder
from wire import UpstreamError, chunk_of, content_of, error_of, finished, sse


class Sink(Protocol):
    """What the client gets at each moment of the stream."""

    @property
    def text(self) -> str: ...  # visible answer so far

    @property
    def done(self) -> bool: ...  # the upstream already sent a finish reason

    def start(self) -> Iterable[bytes]: ...

    def line(self, line: str) -> Iterable[bytes]: ...  # raises UpstreamError on a mid-stream error

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

    def line(self, line: str) -> Iterable[bytes]:
        err = error_of(line)
        if err:
            raise UpstreamError(err)
        piece = content_of(line)
        if piece:
            self.parts.append(piece)
        self.done = self.done or finished(line)
        return (sse(line),)

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

    def line(self, line: str) -> Iterable[bytes]:
        err = error_of(line)
        if err:
            raise UpstreamError(err)
        self.done = self.done or finished(line)
        chunk = chunk_of(line)
        return [ev.encode() for ev in self.builder.feed(chunk)] if chunk is not None else ()

    def finish(self) -> Iterable[bytes]:
        return [ev.encode() for ev in self.builder.finish()]

    def fail(self, error: str) -> Iterable[bytes]:
        return [ev.encode() for ev in self.builder.finish(error=error)]
