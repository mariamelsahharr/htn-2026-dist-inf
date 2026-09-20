#!/usr/bin/env python3
"""
supervisor.py - runs dllama-api on the root and relaunches it on the largest node
set the model allows when workers die or return. State on :9991/status.

Rules (from distributed-llama source, see README.md):
  never TCP-probe worker port 9998; never bare-connect to root port 9990;
  reset surviving workers over SSH before relaunch.

Stdlib only, Python 3.11.
"""

from __future__ import annotations

import argparse
import contextlib
import hmac
import json
import logging
import logging.handlers
import os
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HEALTHY = "healthy"
DEGRADED = "degraded"
RESTARTING = "restarting"
DOWN = "down"

DEFAULT_MODEL_DIR = "/home/pi/distributed-llama/models/qwen3_0.6b_q40"
DEFAULT_PROBE_CMD = "ping -c 1 -W 1 {host}"
DEFAULT_RESET_CMD = (
    "ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "
    "pi@{host} sudo systemctl restart dllama-worker"
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
except ImportError:  # Pis without the SDK (apt python3-sentry-sdk): telemetry is a no-op
    sentry_sdk = None
    LoggingIntegration = None

_pylog = logging.getLogger("supervisor")
_SENTRY_ON = False
_JOURNAL_PRIORITY = {logging.DEBUG: 7, logging.INFO: 6, logging.WARNING: 4, logging.ERROR: 3, logging.CRITICAL: 2}


class _JournalFormatter(logging.Formatter):
    """ISO-8601 UTC timestamps; under systemd a <priority> prefix so journald keeps the level."""

    def __init__(self) -> None:
        super().__init__("%(message)s")
        self.journal = "JOURNAL_STREAM" in os.environ

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")
        line = f"{ts} {record.getMessage()}"
        return f"<{_JOURNAL_PRIORITY.get(record.levelno, 6)}>{line}" if self.journal else line


def configure_logging() -> None:
    """One stdout handler (journald under systemd); Sentry hooks the same records. Idempotent."""
    if _pylog.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JournalFormatter())
    _pylog.addHandler(handler)
    _pylog.setLevel(logging.INFO)
    _pylog.propagate = False


def log(msg: str, level: int = logging.INFO) -> None:
    configure_logging()
    _pylog.log(level, msg)


def sentry_init_from_env() -> None:
    """Enable Sentry Logs + error capture if SENTRY_DSN is set. Never raises."""
    global _SENTRY_ON
    dsn = os.environ.get("SENTRY_DSN", "")
    if not (sentry_sdk and dsn):
        return
    env = os.environ.get("SENTRY_ENVIRONMENT", "demo")
    kwargs: dict = {"dsn": dsn, "environment": env, "traces_sample_rate": 0.0}
    if LoggingIntegration is not None:
        # every log() record becomes a Sentry log line; issues are raised explicitly by sentry_note
        kwargs["integrations"] = [LoggingIntegration(level=logging.INFO, event_level=None)]
    try:
        try:
            sentry_sdk.init(enable_logs=True, **kwargs)
        except TypeError:  # older SDK without enable_logs
            sentry_sdk.init(**kwargs)
        _SENTRY_ON = True
        log("sentry telemetry enabled")
    except Exception:
        pass


def sentry_note(msg: str, level: str = "info") -> None:
    """Raise a Sentry issue for a warning/error; the log line itself already went out via log(). Never raises."""
    if not _SENTRY_ON or sentry_sdk is None or level not in ("warning", "error"):
        return
    with contextlib.suppress(Exception):
        sentry_sdk.capture_message(msg, level=level)


def sd_notify(state: str) -> bool:
    """systemd sd_notify(3) over $NOTIFY_SOCKET (AF_UNIX datagram). No-op, False, outside systemd."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):  # abstract namespace
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.send(state.encode())
        return True
    except OSError:
        return False


# ------------------------------------------------------------------ pure logic


def powers_of_two(max_nodes: int) -> list[int]:
    out, p = [], 1
    while p <= max_nodes:
        out.append(p)
        p *= 2
    return out


# .m header: int32 magic, int32 size (incl. these two), then int32 key/value pairs.
MODEL_MAGIC = 0xA00ABCD
HEADER_KEYS = {
    0: "version",
    1: "arch_type",
    2: "dim",
    3: "hidden_dim",
    4: "n_layers",
    5: "n_heads",
    6: "n_kv_heads",
    7: "n_experts",
    8: "n_active_experts",
    9: "vocab_size",
    10: "seq_len",
    11: "hidden_act",
    12: "rope_theta",
    13: "weight_float_type",
    14: "rope_scaling_factor",
    15: "rope_scaling_low_freq_factor",
    16: "rope_scaling_high_freq_factor",
    17: "rope_scaling_orig_max_seq_len",
    18: "rope_type",
    19: "head_dim",
    20: "norm_epsilon",
    21: "moe_hidden_dim",
}


def read_model_header(path: str) -> dict:
    """Parse a distributed-llama .m header; raises OSError/ValueError if it is not one."""
    with Path(path).open("rb") as f:
        magic, header_size = struct.unpack("<ii", f.read(8))
        if magic != MODEL_MAGIC:
            raise ValueError(f"{path}: not a distributed-llama model (magic 0x{magic:x})")
        raw = f.read(header_size - 8)
    n = len(raw) // 8
    pairs = struct.unpack(f"<{2 * n}i", raw[: 8 * n])
    header = {HEADER_KEYS.get(k, f"key_{k}"): v for k, v in zip(pairs[0::2], pairs[1::2], strict=True)}
    head_dim = header.get("head_dim") or (header["dim"] // header["n_heads"])
    header["head_dim"] = head_dim
    header["q_dim"] = head_dim * header["n_heads"]
    header["kv_dim"] = head_dim * header["n_kv_heads"]
    return header


def valid_node_counts(header: dict, max_nodes: int) -> list[int]:
    """Counts that divide every sliced dim (nn-core.cpp asserts), up to max_nodes."""
    dims = [header["n_heads"], header["kv_dim"], header["hidden_dim"], header["vocab_size"]]
    if header.get("moe_hidden_dim"):
        dims.append(header["moe_hidden_dim"])
    return [n for n in range(1, max_nodes + 1) if all(d % n == 0 for d in dims)]


def choose_workers(alive_in_priority: list, valid_counts: list[int] | None = None, min_nodes: int = 1) -> list | None:
    """Largest valid set the alive workers fill, in priority order.
    None when no valid count fits between min_nodes and the survivors: stand down
    rather than launch a set the model or the RAM cannot take."""
    n_alive = len(alive_in_priority)
    counts = powers_of_two(n_alive + 1) if valid_counts is None else valid_counts
    fits = [c for c in counts if min_nodes <= c <= n_alive + 1]
    if not fits:
        return None
    return list(alive_in_priority[: max(fits) - 1])


def parse_workers(spec: str, default_port: int) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.partition(":")
        out.append((host, int(port) if port else default_port))
    return out


def worker_process_present(doc: dict | None) -> bool:
    """False only when the node agent positively reports no dllama worker socket on its port:
    neither listening for a root nor holding a root's connection (the worker closes its
    listen socket once the root connects). No telemetry or an older agent: trust the ping."""
    if not doc:
        return True
    listening, connections = doc.get("worker_listening"), doc.get("worker_connections")
    if listening is None and connections is None:
        return True
    return bool(listening) or bool(connections)


def post_allowed(client_ip: str, headers: Mapping[str, str] | Message, token: str | None) -> bool:
    """POST is for the operator on the root itself, or for a caller presenting SUPERVISOR_TOKEN."""
    if client_ip in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        return True
    if not token:
        return False
    auth = headers.get("Authorization") or ""
    presented = auth[len("Bearer ") :] if auth.startswith("Bearer ") else (headers.get("X-Supervisor-Token") or "")
    return bool(presented) and hmac.compare_digest(presented, token)


# dllama-api (src/dllama-api.cpp) prints these every 3 s, forever, while a worker is unreachable
CONNECT_RETRY_RE = re.compile(r"Connection error|Cannot connect", re.IGNORECASE)
RETRY_FILLER_RE = re.compile(r"Retrying in")
# what a root that could not hold its share of the weights leaves in the log (or none: SIGKILL)
OOM_RE = re.compile(r"bad_alloc|Cannot allocate memory|out of memory|Killed process|mmap.*failed", re.IGNORECASE)


def tail_lines(path: Path, n: int = 20, nbytes: int = 8192) -> list[str]:
    """The last n non-empty lines of a file, reading at most nbytes."""
    try:
        with path.open("rb") as fh:
            end = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, end - nbytes))
            raw = fh.read().split(b"\n")
    except OSError:
        return []
    return [ln.decode(errors="replace").strip() for ln in raw if ln.strip()][-n:]


class LogTail:
    """Lines a file gained since the last call; survives truncation and rotation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.pos = path.stat().st_size if path.exists() else 0
        self.partial = b""

    def read_new(self) -> list[str]:
        try:
            size = self.path.stat().st_size
            if size < self.pos:
                self.pos, self.partial = 0, b""
            if size == self.pos:
                return []
            with self.path.open("rb") as fh:
                fh.seek(self.pos)
                data = self.partial + fh.read(size - self.pos)
            self.pos = size
        except OSError:
            return []
        lines = data.split(b"\n")
        self.partial = lines.pop()
        return [ln.decode(errors="replace").strip() for ln in lines]


def cap_file(path: Path, max_bytes: int) -> bool:
    """Keep an append-only file that a child process writes to under max_bytes: truncate it in
    place and write the newest half back (O_APPEND writers land after it). True when capped."""
    try:
        if path.stat().st_size <= max_bytes:
            return False
        keep = max_bytes // 2
        with path.open("r+b") as fh:
            fh.seek(-keep, os.SEEK_END)
            tail = fh.read()
            fh.seek(0)
            fh.truncate(0)
            fh.write(f"=== {now_iso()} capped at {max_bytes} bytes; older lines dropped\n".encode() + tail)
        return True
    except OSError:
        return False


class Worker:
    """Worker host with probe hysteresis: first probe decides, then fail_after / ok_after."""

    def __init__(self, host: str, port: int, fail_after: int = 2, ok_after: int = 2) -> None:
        self.host = host
        self.port = port
        self.fail_after = max(1, fail_after)
        self.ok_after = max(1, ok_after)
        self.alive: bool | None = None
        self.fails = 0
        self.oks = 0
        self.last_seen: float | None = None
        self.last_probe: float | None = None
        self.telemetry: dict | None = None

    @property
    def addr(self) -> str:
        return f"{self.host}:{self.port}"

    def record(self, ok: bool, now: float) -> bool | None:
        """Returns the new alive value when it flips, else None."""
        before = self.alive
        self.last_probe = now
        if ok:
            self.oks += 1
            self.fails = 0
            self.last_seen = now
            if self.alive is None or self.oks >= self.ok_after:
                self.alive = True
        else:
            self.fails += 1
            self.oks = 0
            if self.alive is None or self.fails >= self.fail_after:
                self.alive = False
        return self.alive if self.alive != before else None

    def as_dict(self, in_set: bool, reset_failed: bool = False) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "alive": self.alive,
            "in_set": in_set,
            "reset_failed": reset_failed,
            "consecutive_fails": self.fails,
            "last_seen": self.last_seen,
            "last_probe": self.last_probe,
            "telemetry": self.telemetry,
        }


@dataclass
class Config:
    workers: list[tuple[str, int]]
    dllama_bin: str = "/home/pi/distributed-llama/dllama-api"
    model: str = f"{DEFAULT_MODEL_DIR}/dllama_model_qwen3_0.6b_q40.m"
    tokenizer: str = f"{DEFAULT_MODEL_DIR}/dllama_tokenizer_qwen3_0.6b_q40.t"
    buffer_float_type: str = "q80"
    nthreads: int = 4
    api_host: str = "0.0.0.0"
    api_port: int = 9990
    extra_args: str = ""
    status_host: str = "0.0.0.0"
    status_port: int = 9991
    status_file: str = "/tmp/dllama-supervisor-status.json"
    log_dir: str = "/home/pi/logs"
    interval: float = 2.0
    probe_cmd: str = DEFAULT_PROBE_CMD
    probe_timeout: float = 3.0
    telemetry_port: int = 9997  # node_agent.py on every Pi; 0 disables
    telemetry_timeout: float = 1.0
    fail_after: int = 2
    ok_after: int = 2
    rejoin_grace: float = 10.0
    auto_rejoin: bool = True
    reset_cmd: str = DEFAULT_RESET_CMD
    reset_workers: bool = True
    settle: float = 3.0
    ready_timeout: float = 600.0
    api_check_interval: float = 10.0
    # 0 = off. dllama-api is single-threaded, so back-to-back generations look like a
    # stall to a GET probe; process exit and worker loss are the reliable signals.
    api_stall_timeout: float = 0.0
    # Retry delay after a failed launch: min(max_launch_backoff, launch_backoff * 2**launch_failures).
    launch_backoff: float = 5.0
    max_launch_backoff: float = 300.0
    # A root that dies sooner than this after becoming ready counts as a launch failure (crash loop),
    # and one that has stayed up this long clears the failure counter.
    stable_after: float = 60.0
    # A root that dies sooner than this after its spawn never reached the workers: no SSH reset.
    quick_exit: float = 2.0
    # Abandon a launch once the dllama-api log shows this many consecutive worker-connect retries
    # (one every 3 s), instead of waiting out ready_timeout.
    max_connect_retries: int = 10
    # A worker whose SSH reset failed twice is left out of the set; retry its reset this often.
    reset_retry_interval: float = 60.0
    # SD card protection: dllama-api.log is capped in place, supervisor-events.jsonl rotates.
    root_log_max_bytes: int = 20 * 1024 * 1024
    events_max_bytes: int = 2 * 1024 * 1024
    # Explicit list of allowed node counts; None derives them from the model header.
    node_counts: list[int] | None = None
    # Below this many nodes the supervisor reports `down` instead of launching (RAM floor).
    min_nodes: int = 1


# ------------------------------------------------------------------ supervisor


class Supervisor:
    """Owns the dllama-api process. probe/spawn/reset_worker/api_check are injectable."""

    def __init__(
        self,
        cfg: Config,
        *,
        probe: Callable[[str], bool] | None = None,
        spawn: Callable[[list[str], str], subprocess.Popen] | None = None,
        reset_worker: Callable[[str], bool] | None = None,
        api_check: Callable[[], bool] | None = None,
        telemetry: Callable[[str], dict | None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.workers = [Worker(h, p, cfg.fail_after, cfg.ok_after) for h, p in cfg.workers]
        self.state = DOWN
        self.state_since = time.time()
        self.state_reason = "not started"
        self.active: list[Worker] = []
        self.proc: subprocess.Popen | None = None
        self._proc_log = None
        self.generation = 0
        self.restarts = 0
        self.launch_failures = 0
        self.launched_at: float | None = None
        self.ready_at: float | None = None
        self.load_seconds: float | None = None
        self.last_api_ok: float | None = None
        self.last_api_check = 0.0
        self.events: deque = deque(maxlen=50)
        self.started_at = time.time()
        self._stop = threading.Event()
        self._restart_requested: str | None = None
        self._grow_since: float | None = None
        self._next_launch_at = 0.0
        self._probe = probe or self._ping
        self._spawn = spawn or self._spawn_root
        self._reset_worker = reset_worker or self._ssh_reset
        self._api_check = api_check or self._models_ok
        self._telemetry = telemetry or self._fetch_telemetry
        self.root_telemetry: dict | None = None
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.workers)))
        self.model_header: dict | None = None
        self.valid_counts, self.node_counts_source = self._resolve_node_counts()
        self._snapshot: dict = {}
        self._reset_failed: set[str] = set()
        self._last_reset_retry = 0.0
        self._log_tail: LogTail | None = None
        self._last_cap_check = 0.0
        self._events_log = self._open_events_log()

    @property
    def root_log_path(self) -> Path:
        return Path(self.cfg.log_dir) / "dllama-api.log"

    def _open_events_log(self) -> logging.Logger | None:
        """supervisor-events.jsonl, rotated by size: the failover history outlives a supervisor restart."""
        try:
            Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                Path(self.cfg.log_dir) / "supervisor-events.jsonl", maxBytes=self.cfg.events_max_bytes, backupCount=2
            )
        except OSError as e:
            log(f"no events log in {self.cfg.log_dir}: {e}", logging.WARNING)
            return None
        handler.setFormatter(logging.Formatter("%(message)s"))
        events = logging.getLogger("supervisor.events")
        for old in list(events.handlers):  # one supervisor per process; tests build several
            events.removeHandler(old)
            old.close()
        events.addHandler(handler)
        events.setLevel(logging.INFO)
        events.propagate = False
        return events

    def _resolve_node_counts(self) -> tuple[list[int], str]:
        max_nodes = len(self.workers) + 1
        if self.cfg.node_counts:
            counts = sorted({c for c in self.cfg.node_counts if 1 <= c <= max_nodes})
            if not counts:
                log(
                    f"--node-counts {self.cfg.node_counts} has no entry <= {max_nodes} nodes; "
                    "the supervisor will stay down until that changes",
                    logging.WARNING,
                )
            return counts, "override"
        try:
            self.model_header = read_model_header(self.cfg.model)
            counts = valid_node_counts(self.model_header, max_nodes)
            return counts, "model header"
        except (OSError, ValueError, KeyError, struct.error) as e:
            log(f"cannot read model header ({e}); assuming powers of two", logging.WARNING)
            return powers_of_two(max_nodes), "powers of two (header unreadable)"

    def choose(self, alive: list[Worker]) -> list[Worker] | None:
        return choose_workers(alive, self.valid_counts, self.cfg.min_nodes)

    def launch_or_stand_down(self, reason: str) -> None:
        """Launch on the best set the survivors allow, or report down and wait."""
        self.valid_counts, self.node_counts_source = self._resolve_node_counts()  # the model file may have changed
        desired = self.choose(self.eligible_workers())
        if desired is None:
            msg = (
                f"only {len(self.eligible_workers()) + 1} node(s) usable; need a valid count "
                f"in {self.valid_counts} of at least {self.cfg.min_nodes}"
            )
            if self._reset_failed:
                msg += "; excluded after a failed reset: " + ", ".join(sorted(self._reset_failed))
            self.active = []
            if not (self.state == DOWN and self.state_reason == msg):
                self.set_state(DOWN, msg)
            self._next_launch_at = time.time() + self.cfg.interval
            return
        self.launch(desired, reason)

    # ----- bookkeeping ---------------------------------------------------------

    def event(self, msg: str, level: int = logging.INFO) -> None:
        log(msg, level)
        entry = {"t": time.time(), "msg": msg}
        self.events.append(entry)
        if self._events_log is not None:
            self._events_log.info(json.dumps(entry))

    def set_state(self, state: str, reason: str) -> None:
        changed = state != self.state
        if changed:
            self.state_since = time.time()
        self.state = state
        self.state_reason = reason
        level = {DEGRADED: logging.WARNING, RESTARTING: logging.WARNING, DOWN: logging.ERROR}.get(state, logging.INFO)
        self.event(f"state={state}: {reason}", level)
        if changed:
            sentry_note(
                f"cluster state -> {state}: {reason} (active={len(self.active)}, restarts={self.restarts})",
                logging.getLevelName(level).lower(),
            )
        self.publish()  # /status must not say healthy for another tick after the root died

    @property
    def serving(self) -> bool:
        return self.state in (HEALTHY, DEGRADED)

    def root_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def request_restart(self, reason: str) -> None:
        self._restart_requested = reason

    def stop(self) -> None:
        self._stop.set()

    # ----- probing -------------------------------------------------------------

    def probe_all(self, now: float | None = None) -> None:
        if not self.workers:
            return
        now = now or time.time()
        pings = list(self._pool.map(self._probe, [w.host for w in self.workers]))
        docs: dict[str, dict | None] = {}
        if self.cfg.telemetry_port:
            hosts = [w.host for w, ok in zip(self.workers, pings, strict=True) if ok] + ["127.0.0.1"]
            docs = dict(zip(hosts, self._pool.map(self._telemetry, hosts), strict=True))
            self.root_telemetry = docs.get("127.0.0.1")
        for w, ping_ok in zip(self.workers, pings, strict=True):
            doc = docs.get(w.host) if ping_ok else None
            # a Pi that answers ping but has no dllama worker process is not a worker
            change = w.record(ping_ok and worker_process_present(doc), now)
            w.telemetry = doc if w.alive else None
            if change is None:
                continue
            if change:
                self._reset_failed.discard(w.host)  # it came back (rebooted or restarted): reset it afresh
                self.event(f"worker {w.host} reachable")
            elif ping_ok:
                unit = (doc or {}).get("worker_unit")
                self.event(f"worker {w.host} has no dllama worker process (ping ok, unit {unit})", logging.WARNING)
            else:
                self.event(f"worker {w.host} unreachable", logging.WARNING)

    def alive_workers(self) -> list[Worker]:
        return [w for w in self.workers if w.alive]

    def eligible_workers(self) -> list[Worker]:
        """Alive workers the root can actually use: a failed SSH reset benches one until it resets or reboots."""
        return [w for w in self.workers if w.alive and w.host not in self._reset_failed]

    def _fetch_telemetry(self, host: str) -> dict | None:
        url = f"http://{host}:{self.cfg.telemetry_port}/telemetry"
        try:
            with urllib.request.urlopen(url, timeout=self.cfg.telemetry_timeout) as r:
                doc = json.loads(r.read().decode())
                return doc if isinstance(doc, dict) else None
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _ping(self, host: str) -> bool:
        cmd = shlex.split(self.cfg.probe_cmd.format(host=shlex.quote(host)))
        try:
            r = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=self.cfg.probe_timeout
            )
            return r.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    # ----- root process --------------------------------------------------------

    def build_command(self, workers: list[Worker]) -> list[str]:
        cmd = [
            self.cfg.dllama_bin,
            "--host",
            self.cfg.api_host,
            "--port",
            str(self.cfg.api_port),
            "--model",
            self.cfg.model,
            "--tokenizer",
            self.cfg.tokenizer,
            "--buffer-float-type",
            self.cfg.buffer_float_type,
            "--nthreads",
            str(self.cfg.nthreads),
        ]
        cmd += shlex.split(self.cfg.extra_args)
        if workers:  # must be last: dllama-api swallows argv until the next '-'
            cmd += ["--workers", *[w.addr for w in workers]]
        return cmd

    def _spawn_root(self, cmd: list[str], reason: str) -> subprocess.Popen:
        Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)
        path = self.root_log_path
        cap_file(path, self.cfg.root_log_max_bytes)
        f = path.open("ab")  # handed to the child process; closed with it
        f.write(f"\n=== {now_iso()} gen={self.generation} {reason}\n$ {shlex.join(cmd)}\n".encode())
        f.flush()
        try:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError:
            f.close()
            raise
        self._proc_log = f
        return proc

    def _models_ok(self) -> bool:
        url = f"http://127.0.0.1:{self.cfg.api_port}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                return r.status == 200
        except Exception:
            return False

    def kill_root(self, reason: str) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            self.event(f"stopping root pid {proc.pid}: {reason}")
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                proc.wait()
        if self._proc_log is not None:
            self._proc_log.close()
            self._proc_log = None
        self.ready_at = None
        self.last_api_ok = None
        self.publish()

    def reset_workers(self, workers: list[Worker]) -> list[Worker]:
        """Restart dllama-worker over SSH on each host, one retry. A host that still fails is
        benched (see eligible_workers): it may be wedged on the dead root's session and would
        stall the new root's connect. Returns the workers that could not be reset."""
        if not self.cfg.reset_workers or not workers:
            return []
        self.event("resetting workers: " + ", ".join(w.host for w in workers))
        failed = self._reset_round(workers)
        if failed:
            self.publish()  # feeds the watchdog before a second round of SSH timeouts
            failed = self._reset_round(failed)
        failed_ids = {id(w) for w in failed}
        for w in workers:
            if id(w) in failed_ids:
                self._reset_failed.add(w.host)
            else:
                self._reset_failed.discard(w.host)
        if failed:
            self.event(
                "reset failed twice, leaving out of the set: " + ", ".join(w.host for w in failed), logging.WARNING
            )
        return failed

    def _reset_round(self, workers: list[Worker]) -> list[Worker]:
        results = list(self._pool.map(self._reset_worker, [w.host for w in workers]))
        return [w for w, ok in zip(workers, results, strict=True) if not ok]

    def _retry_failed_resets(self, now: float) -> None:
        """A benched worker gets another reset every reset_retry_interval; success makes it eligible again."""
        if not self._reset_failed or now - self._last_reset_retry < self.cfg.reset_retry_interval:
            return
        self._last_reset_retry = now
        benched = [w for w in self.workers if w.host in self._reset_failed and w.alive]
        if not benched:
            return
        failed_ids = {id(w) for w in self._reset_round(benched)}
        for w in benched:
            if id(w) not in failed_ids:
                self._reset_failed.discard(w.host)
                self.event(f"worker {w.host} reset succeeded; eligible again")

    def _ssh_reset(self, host: str) -> bool:
        cmd = shlex.split(self.cfg.reset_cmd.format(host=shlex.quote(host)))
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=20)
            if r.returncode != 0:
                err = r.stderr.decode(errors="replace").strip()[:200]
                log(f"reset of {host} failed (rc={r.returncode}): {err}", logging.WARNING)
            return r.returncode == 0
        except (subprocess.SubprocessError, OSError) as e:
            log(f"reset of {host} failed: {e}", logging.WARNING)
            return False

    # ----- launch / restart ----------------------------------------------------

    def _launch_failed(self) -> float:
        """Count a failed launch; returns the exponential backoff before the next attempt."""
        self.launch_failures += 1
        return min(self.cfg.max_launch_backoff, self.cfg.launch_backoff * 2**self.launch_failures)

    def _exit_diagnosis(self, rc: int | None) -> str:
        """Why the root died, from its exit code and the end of dllama-api.log. An out-of-memory
        exit is the usual failure of a bigger model and must read as such, not as a silent loop."""
        recent = [ln for ln in tail_lines(self.root_log_path) if not ln.startswith("===")]
        last = recent[-1][:160] if recent else ""
        oom = rc in (-signal.SIGKILL, 128 + signal.SIGKILL) or any(OOM_RE.search(ln) for ln in recent)
        out = f"last log line: {last!r}" if last else "no output in dllama-api.log"
        if oom:
            out = (
                "likely out of memory: this node's share of the model does not fit its RAM "
                "(check mem_available_mb in /status telemetry; raise --min-nodes or pick a smaller model); " + out
            )
        return out

    def _root_reached_workers(self, now: float | None = None) -> bool:
        """A root that died within quick_exit of its spawn (bad flag, missing model) never
        connected to the workers, so they are not wedged and need no SSH reset."""
        now = now or time.time()
        return self.launched_at is not None and now - self.launched_at >= self.cfg.quick_exit

    def launch(self, workers: list[Worker], reason: str) -> bool:
        workers = list(workers)
        self.generation += 1
        self.active = workers
        self.launched_at = time.time()
        self.ready_at = None
        self.load_seconds = None
        n = len(workers) + 1
        self.set_state(RESTARTING, f"launching generation {self.generation} on {n} node(s): {reason}")
        cmd = self.build_command(workers)
        try:
            self.proc = self._spawn(cmd, reason)
        except OSError as e:
            self._next_launch_at = time.time() + self._launch_failed()
            self.set_state(DOWN, f"cannot start {self.cfg.dllama_bin}: {e}")
            return False
        self._log_tail = LogTail(self.root_log_path)
        problem = self.wait_ready()
        if problem is None:
            self.ready_at = time.time()
            self.load_seconds = round(self.ready_at - self.launched_at, 1)
            self.last_api_ok = self.ready_at
            self.last_api_check = self.ready_at
            missing = [w.host for w in self.workers if w not in workers]
            if missing:
                self.set_state(
                    DEGRADED, f"serving on {n} node(s) after {self.load_seconds}s; missing {', '.join(missing)}"
                )
            else:
                self.set_state(HEALTHY, f"serving on all {n} node(s) after {self.load_seconds}s")
            return True
        touched = self._root_reached_workers()
        if self.proc is not None and self.proc.poll() is not None:
            problem += f"; {self._exit_diagnosis(self.proc.returncode)}"
        self.kill_root(problem)
        if touched:
            self.reset_workers([w for w in workers if w.alive is not False])
        self.active = []
        delay = self._launch_failed()
        self._next_launch_at = time.time() + delay
        self.set_state(
            DOWN if self.launch_failures >= 3 else RESTARTING,
            f"{problem}; launch failure {self.launch_failures}, next attempt in {delay:.0f}s",
        )
        return False

    def wait_ready(self) -> str | None:
        """Block until /v1/models answers. None when ready, else the reason it will not."""
        deadline = time.time() + self.cfg.ready_timeout
        retries = 0
        while not self._stop.is_set():
            rc = self.proc.poll() if self.proc else -1
            if rc is not None:
                return f"root exited with code {rc} during load"
            try:
                ready = self._api_check()
            except Exception:
                ready = False
            if ready:
                return None
            self.probe_all()
            dead = [w.host for w in self.active if w.alive is False]
            if dead:
                return "worker lost during load: " + ", ".join(dead)
            retries = self._count_connect_retries(retries)
            if retries >= self.cfg.max_connect_retries:
                return f"root cannot reach its workers ({retries} consecutive connect retries)"
            self.publish()
            if time.time() > deadline:
                return f"root not ready after {self.cfg.ready_timeout:.0f}s"
            self._stop.wait(min(1.0, self.cfg.interval))
        return "supervisor stopping"

    def _count_connect_retries(self, retries: int) -> int:
        """Consecutive worker-connect retries in the dllama-api log; any other output resets the count."""
        if self._log_tail is None:
            return retries
        for line in self._log_tail.read_new():
            if CONNECT_RETRY_RE.search(line):
                retries += 1
            elif line and not RETRY_FILLER_RE.search(line):
                retries = 0
        return retries

    def restart(self, reason: str) -> None:
        self.restarts += 1
        self.set_state(RESTARTING, reason)
        old = list(self.active)
        self.kill_root(reason)
        self.probe_all()
        # reset survivors and any worker about to rejoin: a returning worker may still be
        # attached to the dead root's session and would stall the new root's connect
        desired = self.choose(self.alive_workers()) or []
        to_reset = {id(w): w for w in old if w.alive is not False}
        to_reset.update({id(w): w for w in desired})
        self.reset_workers(list(to_reset.values()))
        self._stop.wait(self.cfg.settle)
        self.probe_all()
        self.launch_or_stand_down(reason)

    def _api_stalled(self, now: float) -> bool:
        if self.cfg.api_stall_timeout <= 0:
            return False
        if now - self.last_api_check < self.cfg.api_check_interval:
            return False
        self.last_api_check = now
        if self._api_check():
            self.last_api_ok = now
            return False
        # single-threaded server: a miss during generation is normal, only a long silence counts
        return self.last_api_ok is not None and now - self.last_api_ok > self.cfg.api_stall_timeout

    # ----- main loop -----------------------------------------------------------

    def tick(self) -> None:
        now = time.time()
        self.probe_all(now)
        if not self.root_running():
            if self.proc is not None:
                rc = self.proc.returncode
                # a crash soon after ready is a launch failure too, else a crash loop relaunches every settle
                crashed_early = self.ready_at is not None and now - self.ready_at < self.cfg.stable_after
                touched = self._root_reached_workers(now)
                self.kill_root(f"exit code {rc}")
                self.restarts += 1
                delay = self.cfg.settle
                reason = f"root exited with code {rc}; {self._exit_diagnosis(rc)}"
                if crashed_early:
                    delay = max(delay, self._launch_failed())
                    reason += f" within {self.cfg.stable_after:.0f}s of ready; launch failure {self.launch_failures}"
                    reason += f", next attempt in {delay:.0f}s"
                self.set_state(DOWN if self.launch_failures >= 3 else RESTARTING, reason)
                if touched:
                    self.reset_workers([w for w in self.active if w.alive is not False])
                self._next_launch_at = now + delay
            if now < self._next_launch_at:
                return
            self.launch_or_stand_down("startup" if self.generation == 0 else "relaunch")
            return
        if self.launch_failures and self.ready_at is not None and now - self.ready_at >= self.cfg.stable_after:
            self.launch_failures = 0  # stable: the crash loop is over
        if now - self._last_cap_check >= 60:
            self._last_cap_check = now
            if cap_file(self.root_log_path, self.cfg.root_log_max_bytes):
                self.event(f"{self.root_log_path} capped at {self.cfg.root_log_max_bytes} bytes")
        dead = [w.host for w in self.active if w.alive is False]
        if dead:
            self.restart("worker lost: " + ", ".join(dead))
            return
        if self._restart_requested:
            reason, self._restart_requested = self._restart_requested, None
            self.restart(reason)
            return
        if self._api_stalled(now):
            self.restart(f"root API unresponsive for {self.cfg.api_stall_timeout:.0f}s")
            return
        self._retry_failed_resets(now)
        if self.cfg.auto_rejoin:
            desired = self.choose(self.eligible_workers())
            if desired is not None and len(desired) > len(self.active):
                if self._grow_since is None:
                    self._grow_since = now
                    gained = [w.host for w in desired if w not in self.active]
                    self.event(
                        f"can grow to {len(desired) + 1} node(s) ({', '.join(gained)}); "
                        f"waiting {self.cfg.rejoin_grace:.0f}s for them to settle"
                    )
                elif now - self._grow_since >= self.cfg.rejoin_grace:
                    self._grow_since = None
                    self.restart(f"rejoin: growing to {len(desired) + 1} node(s)")
                return
        self._grow_since = None

    def run(self) -> None:
        self.event(
            f"supervisor starting: {len(self.workers)} worker(s) configured, "
            f"api :{self.cfg.api_port}, status :{self.cfg.status_port}"
        )
        self.event(f"valid node counts for this model: {self.valid_counts} ({self.node_counts_source})")
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick()
            except Exception as e:
                self.event(f"tick failed: {type(e).__name__}: {e}")
            self.publish()
            self._stop.wait(max(0.1, self.cfg.interval - (time.monotonic() - t0)))
        self.kill_root("supervisor stopping")
        self.set_state(DOWN, "supervisor stopped")
        self.publish()
        self._pool.shutdown(wait=False)

    # ----- status --------------------------------------------------------------

    def snapshot(self) -> dict:
        now = time.time()
        active_ids = {id(w) for w in self.active}
        return {
            "state": self.state,
            "reason": self.state_reason,
            "since": self.state_since,
            "state_age_s": round(now - self.state_since, 1),
            "nodes_total": len(self.workers) + 1,
            "nodes_active": len(self.active) + 1 if self.root_running() else 0,
            "min_nodes": self.cfg.min_nodes,
            "valid_node_counts": self.valid_counts,
            "node_counts_source": self.node_counts_source,
            "model_header": {
                k: self.model_header[k]
                for k in (
                    "arch_type",
                    "dim",
                    "hidden_dim",
                    "n_layers",
                    "n_heads",
                    "n_kv_heads",
                    "head_dim",
                    "vocab_size",
                    "seq_len",
                )
                if k in self.model_header
            }
            if self.model_header
            else None,
            "active_workers": [w.addr for w in self.active],
            "workers": [w.as_dict(id(w) in active_ids, w.host in self._reset_failed) for w in self.workers],
            "root": {
                "pid": self.proc.pid if self.proc and self.root_running() else None,
                "api": f"http://{self.cfg.api_host}:{self.cfg.api_port}/v1",
                "model": self.cfg.model,
                "nthreads": self.cfg.nthreads,
                "launched_at": self.launched_at,
                "ready_at": self.ready_at,
                "load_seconds": self.load_seconds,
                "last_api_ok": self.last_api_ok,
                "telemetry": self.root_telemetry,
            },
            "generation": self.generation,
            "restarts": self.restarts,
            "launch_failures": self.launch_failures,
            "auto_rejoin": self.cfg.auto_rejoin,
            "events": list(self.events)[-20:],
            "uptime_s": round(now - self.started_at, 1),
            "updated_at": now,
        }

    def publish(self) -> None:
        """Refresh the cached snapshot the HTTP thread serves, the status file, and the systemd watchdog."""
        self._snapshot = self.snapshot()
        sd_notify(f"WATCHDOG=1\nSTATUS={self.state}: {self.state_reason[:120]}")
        if not self.cfg.status_file:
            return
        tmp = self.cfg.status_file + ".tmp"
        try:
            with Path(tmp).open("w") as f:
                json.dump(self._snapshot, f)
            Path(tmp).replace(self.cfg.status_file)
        except OSError as e:
            log(f"cannot write {self.cfg.status_file}: {e}")

    def last_snapshot(self) -> dict:
        """Safe to call from any thread; the main loop is the only writer."""
        return self._snapshot or self.snapshot()


# ----------------------------------------------------------------- status HTTP


def make_handler(sup: Supervisor):
    class Handler(BaseHTTPRequestHandler):
        server_version = "dllama-supervisor/1.0"

        def log_message(self, format: str, *args: object) -> None:
            pass

        def _json(self, code: int, obj, cors: bool = True) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            if cors:  # the dashboard reads GET /status from a browser; POST is never cross-origin
                self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/status", "/status.json"):
                self._json(200, sup.last_snapshot())
            elif path == "/healthz":
                self._json(200 if sup.serving else 503, {"ok": sup.serving, "state": sup.state})
            elif path == "/events":
                self._json(200, sup.last_snapshot().get("events", []))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if not post_allowed(self.client_address[0], self.headers, os.environ.get("SUPERVISOR_TOKEN")):
                msg = "POST needs a loopback client or Authorization: Bearer $SUPERVISOR_TOKEN"
                self._json(403, {"error": msg}, cors=False)
            elif path == "/restart":
                sup.request_restart(f"manual restart via HTTP from {self.client_address[0]}")
                self._json(202, {"ok": True, "state": sup.state}, cors=False)
            else:
                self._json(404, {"error": "not found"}, cors=False)

    return Handler


def serve_status(sup: Supervisor, host: str, port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(sup))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True, name="status-http").start()
    return srv


# ------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    d = Config(workers=[])
    ap = argparse.ArgumentParser(description="distributed-llama root supervisor with power-of-two failover")
    ap.add_argument(
        "--workers",
        default="192.168.50.11,192.168.50.12,192.168.50.14",
        help="comma-separated host[:port] in priority order; the first ones are kept "
        "when the set shrinks, so list the best-cooled / biggest-RAM nodes first",
    )
    ap.add_argument("--worker-port", type=int, default=9998)
    ap.add_argument("--dllama-bin", default=d.dllama_bin)
    ap.add_argument("--model", default=d.model)
    ap.add_argument("--tokenizer", default=d.tokenizer)
    ap.add_argument("--buffer-float-type", default=d.buffer_float_type)
    ap.add_argument("--nthreads", type=int, default=d.nthreads)
    ap.add_argument("--api-host", default=d.api_host)
    ap.add_argument("--api-port", type=int, default=d.api_port)
    ap.add_argument("--extra-args", default="", help="extra dllama-api flags, quoted as one string")
    ap.add_argument("--status-host", default=d.status_host)
    ap.add_argument("--status-port", type=int, default=d.status_port)
    ap.add_argument("--status-file", default=d.status_file)
    ap.add_argument("--log-dir", default=d.log_dir)
    ap.add_argument("--interval", type=float, default=d.interval, help="probe period in seconds")
    ap.add_argument("--probe-cmd", default=d.probe_cmd, help="liveness command; {host} is substituted")
    ap.add_argument("--probe-timeout", type=float, default=d.probe_timeout)
    ap.add_argument(
        "--telemetry-port",
        type=int,
        default=d.telemetry_port,
        help="node_agent.py port on every Pi, folded into /status; 0 disables",
    )
    ap.add_argument(
        "--fail-after", type=int, default=d.fail_after, help="consecutive probe misses before a worker counts as dead"
    )
    ap.add_argument(
        "--ok-after", type=int, default=d.ok_after, help="consecutive probe hits before a dead worker counts as back"
    )
    ap.add_argument(
        "--rejoin-grace",
        type=float,
        default=d.rejoin_grace,
        help="seconds a returned worker must stay up before the set grows",
    )
    ap.add_argument("--no-auto-rejoin", action="store_true")
    ap.add_argument(
        "--reset-cmd", default=d.reset_cmd, help="run on each surviving worker before a relaunch; {host} is substituted"
    )
    ap.add_argument("--no-reset-workers", action="store_true")
    ap.add_argument(
        "--settle", type=float, default=d.settle, help="seconds between killing the root and relaunching it"
    )
    ap.add_argument("--ready-timeout", type=float, default=d.ready_timeout)
    ap.add_argument("--api-check-interval", type=float, default=d.api_check_interval)
    ap.add_argument(
        "--api-stall-timeout",
        type=float,
        default=d.api_stall_timeout,
        help="restart if /v1/models has not answered for this long; 0 disables",
    )
    ap.add_argument(
        "--launch-backoff",
        type=float,
        default=d.launch_backoff,
        help="base of the exponential retry delay after a failed launch (doubles per failure, capped)",
    )
    ap.add_argument("--max-launch-backoff", type=float, default=d.max_launch_backoff)
    ap.add_argument(
        "--max-connect-retries",
        type=int,
        default=d.max_connect_retries,
        help="give up a launch after this many consecutive worker-connect retries in the dllama-api log",
    )
    ap.add_argument(
        "--stable-after",
        type=float,
        default=d.stable_after,
        help="a root crash sooner than this after ready counts as a launch failure",
    )
    ap.add_argument(
        "--min-nodes",
        type=int,
        default=1,
        help="below this many nodes report down instead of launching (set it to the "
        "smallest count whose per-node share of the model fits in RAM)",
    )
    ap.add_argument(
        "--node-counts",
        default="",
        help="comma-separated node counts to allow, e.g. 1,2,4,8; default derives "
        "them from the model header (divisibility of heads/dims/vocab)",
    )
    ap.add_argument(
        "--print-command", action="store_true", help="print the dllama-api command for the full set and exit"
    )
    ap.add_argument("--print-node-counts", action="store_true", help="print the node counts the model allows and exit")
    return ap


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        workers=parse_workers(args.workers, args.worker_port),
        dllama_bin=str(Path(args.dllama_bin).expanduser()),
        model=str(Path(args.model).expanduser()),
        tokenizer=str(Path(args.tokenizer).expanduser()),
        buffer_float_type=args.buffer_float_type,
        nthreads=args.nthreads,
        api_host=args.api_host,
        api_port=args.api_port,
        extra_args=args.extra_args,
        status_host=args.status_host,
        status_port=args.status_port,
        status_file=args.status_file,
        log_dir=str(Path(args.log_dir).expanduser()),
        interval=args.interval,
        probe_cmd=args.probe_cmd,
        probe_timeout=args.probe_timeout,
        telemetry_port=args.telemetry_port,
        fail_after=args.fail_after,
        ok_after=args.ok_after,
        rejoin_grace=args.rejoin_grace,
        auto_rejoin=not args.no_auto_rejoin,
        reset_cmd=args.reset_cmd,
        reset_workers=not args.no_reset_workers,
        settle=args.settle,
        ready_timeout=args.ready_timeout,
        api_check_interval=args.api_check_interval,
        api_stall_timeout=args.api_stall_timeout,
        launch_backoff=args.launch_backoff,
        max_launch_backoff=args.max_launch_backoff,
        max_connect_retries=args.max_connect_retries,
        stable_after=args.stable_after,
        node_counts=[int(c) for c in args.node_counts.split(",") if c.strip()] or None,
        min_nodes=max(1, args.min_nodes),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    sentry_init_from_env()
    cfg = config_from_args(args)
    sup = Supervisor(cfg)
    if args.print_command:
        print(shlex.join(sup.build_command(sup.workers)))
        return 0
    if args.print_node_counts:
        print(f"{sup.valid_counts} ({sup.node_counts_source})")
        if sup.model_header:
            h = sup.model_header
            print(
                f"n_heads={h['n_heads']} kv_dim={h['kv_dim']} hidden_dim={h['hidden_dim']} "
                f"vocab_size={h['vocab_size']} n_layers={h.get('n_layers')}"
            )
        return 0
    srv = serve_status(sup, cfg.status_host, cfg.status_port)
    sd_notify("READY=1")  # Type=notify: the unit is up once /status answers; WATCHDOG=1 rides on publish()

    def on_signal(signum, _frame) -> None:
        log(f"signal {signum}, shutting down")
        sd_notify("STOPPING=1")
        sup.stop()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        sup.run()
    finally:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
