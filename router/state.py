"""
state.py - what the router remembers between requests, and the decision log.
"""

import contextlib
import json
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from typing import Any


@dataclass
class State:
    cluster_status: str = "unknown"
    status_detail: dict[str, Any] = field(default_factory=dict)
    status_last_ok: float = 0.0
    counts: Counter = field(default_factory=Counter)  # (upstream, reason) -> n
    fallbacks: Counter = field(default_factory=Counter)  # reason -> n
    served: Counter = field(default_factory=Counter)  # upstream -> n
    cache: dict[str, str] = field(default_factory=dict)  # demo answers for when everything is down
    answers: dict[str, tuple[str, float]] = field(default_factory=dict)  # recent answers by prompt: (text, expires)
    recent: deque = field(default_factory=lambda: deque(maxlen=50))  # last served requests, for rates
    started: float = field(default_factory=time.time)


class DecisionLog:
    """One JSON line per answer, rotated so a long-running router does not fill the disk."""

    def __init__(self, path: str, max_bytes: int) -> None:
        self.logger = logging.getLogger(f"decisions.{path}")
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=3)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def write(self, record: dict[str, Any]) -> None:
        with contextlib.suppress(OSError, ValueError):  # never let logging take down a request
            self.logger.info(json.dumps(record))
