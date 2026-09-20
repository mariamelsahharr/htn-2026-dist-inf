"""
logs.py - stdlib logging for the router, with the request in every line.

bind() puts request_id / served_by / reason into a contextvar; ContextFilter copies
them onto each record, so any logger in the process (including uvicorn's, once
configured) prints which request it was talking about.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any

FIELDS = ("request_id", "served_by", "reason")
FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s %(served_by)s %(reason)s] %(message)s"

_context: ContextVar[dict[str, Any] | None] = ContextVar("router_log_context", default=None)


def bind(**fields: Any) -> None:
    """Merge fields into the current task's log context (a copy: tasks do not share it)."""
    _context.set({**context(), **fields})


def clear() -> None:
    _context.set(None)


def context() -> dict[str, Any]:
    return _context.get() or {}


class ContextFilter(logging.Filter):
    """Stamps the bound fields onto every record so the format string can always name them."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = context()
        for name in FIELDS:
            setattr(record, name, ctx.get(name, "-"))
        return True


def configure(level: int = logging.INFO) -> None:
    """Root handler to stderr with the context fields. Idempotent: uvicorn and tests may call it twice."""
    root = logging.getLogger()
    if any(isinstance(f, ContextFilter) for h in root.handlers for f in h.filters):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(FORMAT))
    handler.addFilter(ContextFilter())
    root.addHandler(handler)
    root.setLevel(level)
