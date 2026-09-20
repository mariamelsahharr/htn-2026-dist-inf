"""
state.py - what the router remembers between requests, and the decision log.
"""

import logging
import queue
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Any

from cachetools import TTLCache
from wire import dumps

log = logging.getLogger(__name__)


@dataclass
class State:
    cluster_status: str = "unknown"
    status_detail: dict[str, Any] = field(default_factory=dict)
    status_last_ok: float = 0.0
    counts: Counter = field(default_factory=Counter)  # (upstream, reason) -> n
    fallbacks: Counter = field(default_factory=Counter)  # reason -> n
    served: Counter = field(default_factory=Counter)  # upstream -> n
    cache: dict[str, str] = field(default_factory=dict)  # demo answers for when everything is down
    answers: TTLCache = field(default_factory=lambda: TTLCache(maxsize=1024, ttl=30.0))  # recent answers by request
    recent: deque = field(default_factory=lambda: deque(maxlen=50))  # last served requests, for rates
    started: float = field(default_factory=time.monotonic)


class DecisionLog:
    """One JSON line per answer, rotated so a long-running router does not fill the disk.
    Records go through a queue; the file write happens on the listener's thread, never on the loop."""

    def __init__(self, path: str, max_bytes: int) -> None:
        self.queue: queue.Queue = queue.Queue()  # joinable: the listener marks each record done once written
        self.file = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=3)
        self.file.setFormatter(logging.Formatter("%(message)s"))
        self.listener = QueueListener(self.queue, self.file)
        self.logger = logging.Logger(f"decisions.{path}")  # private: not in the registry, never propagates
        self.logger.propagate = False
        self.logger.addHandler(QueueHandler(self.queue))
        self.listener.start()

    @property
    def depth(self) -> int:
        """Records waiting for the writer thread."""
        return self.queue.qsize()

    def write(self, record: dict[str, Any]) -> None:
        try:
            self.logger.info(dumps(record))
        except (TypeError, ValueError):  # never let logging take down a request
            log.exception("decision record is not serialisable")

    def flush(self) -> None:
        """Block until every queued record is on disk (call off the event loop)."""
        self.queue.join()

    def close(self) -> None:
        self.listener.stop()
        self.file.close()
