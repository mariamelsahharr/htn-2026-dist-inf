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
import json
import os
import shlex
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, UTC
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections.abc import Callable

HEALTHY = "healthy"
DEGRADED = "degraded"
RESTARTING = "restarting"
DOWN = "down"

DEFAULT_MODEL_DIR = "/home/pi/distributed-llama/models/qwen3_0.6b_q40"
DEFAULT_PROBE_CMD = "ping -c 1 -W 1 {host}"
DEFAULT_RESET_CMD = ("ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "
                     "pi@{host} sudo systemctl restart dllama-worker")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def log(msg: str) -> None:
    print(f"{now_iso()} {msg}", flush=True)


# ------------------------------------------------------------------ pure logic

def largest_power_of_two_at_most(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def powers_of_two(max_nodes: int) -> list[int]:
    out, p = [], 1
    while p <= max_nodes:
        out.append(p)
        p *= 2
    return out


# .m header: int32 magic, int32 size (incl. these two), then int32 key/value pairs.
MODEL_MAGIC = 0xA00ABCD
HEADER_KEYS = {
    0: "version", 1: "arch_type", 2: "dim", 3: "hidden_dim", 4: "n_layers", 5: "n_heads",
    6: "n_kv_heads", 7: "n_experts", 8: "n_active_experts", 9: "vocab_size", 10: "seq_len",
    11: "hidden_act", 12: "rope_theta", 13: "weight_float_type", 14: "rope_scaling_factor",
    15: "rope_scaling_low_freq_factor", 16: "rope_scaling_high_freq_factor",
    17: "rope_scaling_orig_max_seq_len", 18: "rope_type", 19: "head_dim", 20: "norm_epsilon",
    21: "moe_hidden_dim",
}


def read_model_header(path: str) -> dict:
    """Parse a distributed-llama .m header; raises OSError/ValueError if it is not one."""
    with open(path, "rb") as f:
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


def choose_workers(alive_in_priority: list, valid_counts: list[int] | None = None) -> list:
    """Largest valid set the alive workers fill, in priority order."""
    n_alive = len(alive_in_priority)
    counts = valid_counts if valid_counts else powers_of_two(n_alive + 1)
    nodes = max((c for c in counts if c <= n_alive + 1), default=1)
    return list(alive_in_priority[: nodes - 1])


def parse_workers(spec: str, default_port: int) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.partition(":")
        out.append((host, int(port) if port else default_port))
    return out


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

    def as_dict(self, in_set: bool) -> dict:
        return {
            "host": self.host, "port": self.port, "alive": self.alive, "in_set": in_set,
            "consecutive_fails": self.fails,
            "last_seen": self.last_seen, "last_probe": self.last_probe,
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
    fail_after: int = 2
    ok_after: int = 2
    rejoin_grace: float = 10.0
    auto_rejoin: bool = True
    reset_cmd: str = DEFAULT_RESET_CMD
    reset_workers: bool = True
    settle: float = 3.0
    ready_timeout: float = 600.0
    api_check_interval: float = 10.0
    api_stall_timeout: float = 180.0
    launch_backoff: float = 5.0
    # Explicit list of allowed node counts; None derives them from the model header.
    node_counts: list[int] | None = None


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
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.workers)))
        self.model_header: dict | None = None
        self.valid_counts, self.node_counts_source = self._resolve_node_counts()

    def _resolve_node_counts(self) -> tuple[list[int], str]:
        max_nodes = len(self.workers) + 1
        if self.cfg.node_counts:
            counts = sorted({c for c in self.cfg.node_counts if 1 <= c <= max_nodes})
            return (counts or [1]), "override"
        try:
            self.model_header = read_model_header(self.cfg.model)
            counts = valid_node_counts(self.model_header, max_nodes)
            return counts, "model header"
        except (OSError, ValueError, KeyError, struct.error) as e:
            log(f"cannot read model header ({e}); assuming powers of two")
            return powers_of_two(max_nodes), "powers of two (header unreadable)"

    def choose(self, alive: list[Worker]) -> list[Worker]:
        return choose_workers(alive, self.valid_counts)

    # ----- bookkeeping ---------------------------------------------------------

    def event(self, msg: str) -> None:
        log(msg)
        self.events.append({"t": time.time(), "msg": msg})

    def set_state(self, state: str, reason: str) -> None:
        if state != self.state:
            self.state_since = time.time()
        self.state = state
        self.state_reason = reason
        self.event(f"state={state}: {reason}")

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
        results = list(self._pool.map(self._probe, [w.host for w in self.workers]))
        for w, ok in zip(self.workers, results, strict=True):
            change = w.record(ok, now)
            if change is not None:
                self.event(f"worker {w.host} {'reachable' if change else 'unreachable'}")

    def alive_workers(self) -> list[Worker]:
        return [w for w in self.workers if w.alive]

    def _ping(self, host: str) -> bool:
        cmd = shlex.split(self.cfg.probe_cmd.format(host=shlex.quote(host)))
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=self.cfg.probe_timeout)
            return r.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    # ----- root process --------------------------------------------------------

    def build_command(self, workers: list[Worker]) -> list[str]:
        cmd = [
            self.cfg.dllama_bin,
            "--host", self.cfg.api_host,
            "--port", str(self.cfg.api_port),
            "--model", self.cfg.model,
            "--tokenizer", self.cfg.tokenizer,
            "--buffer-float-type", self.cfg.buffer_float_type,
            "--nthreads", str(self.cfg.nthreads),
        ]
        cmd += shlex.split(self.cfg.extra_args)
        if workers:  # must be last: dllama-api swallows argv until the next '-'
            cmd += ["--workers", *[w.addr for w in workers]]
        return cmd

    def _spawn_root(self, cmd: list[str], reason: str) -> subprocess.Popen:
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        path = os.path.join(self.cfg.log_dir, "dllama-api.log")
        f = open(path, "ab")
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
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def kill_root(self, reason: str) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            self.event(f"stopping root pid {proc.pid}: {reason}")
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                proc.wait()
        if self._proc_log is not None:
            self._proc_log.close()
            self._proc_log = None
        self.ready_at = None
        self.last_api_ok = None

    def reset_workers(self, workers: list[Worker]) -> None:
        if not self.cfg.reset_workers or not workers:
            return
        self.event("resetting workers: " + ", ".join(w.host for w in workers))
        list(self._pool.map(self._reset_worker, [w.host for w in workers]))

    def _ssh_reset(self, host: str) -> bool:
        cmd = shlex.split(self.cfg.reset_cmd.format(host=shlex.quote(host)))
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=20)
            if r.returncode != 0:
                err = r.stderr.decode(errors="replace").strip()[:200]
                log(f"reset of {host} failed (rc={r.returncode}): {err}")
            return r.returncode == 0
        except (subprocess.SubprocessError, OSError) as e:
            log(f"reset of {host} failed: {e}")
            return False

    # ----- launch / restart ----------------------------------------------------

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
            self.launch_failures += 1
            self._next_launch_at = time.time() + self.cfg.launch_backoff
            self.set_state(DOWN, f"cannot start {self.cfg.dllama_bin}: {e}")
            return False
        problem = self.wait_ready()
        if problem is None:
            self.ready_at = time.time()
            self.load_seconds = round(self.ready_at - self.launched_at, 1)
            self.last_api_ok = self.ready_at
            self.last_api_check = self.ready_at
            self.launch_failures = 0
            missing = [w.host for w in self.workers if w not in workers]
            if missing:
                self.set_state(DEGRADED, f"serving on {n} node(s) after {self.load_seconds}s; "
                                         f"missing {', '.join(missing)}")
            else:
                self.set_state(HEALTHY, f"serving on all {n} node(s) after {self.load_seconds}s")
            return True
        self.launch_failures += 1
        self.kill_root(problem)
        self.reset_workers([w for w in workers if w.alive is not False])
        self._next_launch_at = time.time() + self.cfg.launch_backoff
        self.set_state(DOWN if self.launch_failures >= 3 else RESTARTING, problem)
        return False

    def wait_ready(self) -> str | None:
        """Block until /v1/models answers. None when ready, else the reason it will not."""
        deadline = time.time() + self.cfg.ready_timeout
        while not self._stop.is_set():
            rc = self.proc.poll() if self.proc else -1
            if rc is not None:
                return f"root exited with code {rc} during load"
            if self._api_check():
                return None
            self.probe_all()
            dead = [w.host for w in self.active if w.alive is False]
            if dead:
                return "worker lost during load: " + ", ".join(dead)
            self.publish()
            if time.time() > deadline:
                return f"root not ready after {self.cfg.ready_timeout:.0f}s"
            self._stop.wait(1.0)
        return "supervisor stopping"

    def restart(self, reason: str) -> None:
        self.restarts += 1
        self.set_state(RESTARTING, reason)
        old = list(self.active)
        self.kill_root(reason)
        self.reset_workers([w for w in old if w.alive is not False])
        self._stop.wait(self.cfg.settle)
        self.probe_all()
        self.launch(self.choose(self.alive_workers()), reason)

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
                self.kill_root(f"exit code {rc}")
                self.restarts += 1
                self.set_state(RESTARTING, f"root exited with code {rc}")
                self.reset_workers([w for w in self.active if w.alive is not False])
                self._next_launch_at = now + self.cfg.settle
            if now < self._next_launch_at:
                return
            self.launch(self.choose(self.alive_workers()),
                        "startup" if self.generation == 0 else "relaunch")
            return
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
        if self.cfg.auto_rejoin:
            desired = self.choose(self.alive_workers())
            if len(desired) > len(self.active):
                if self._grow_since is None:
                    self._grow_since = now
                    gained = [w.host for w in desired if w not in self.active]
                    self.event(f"can grow to {len(desired) + 1} node(s) ({', '.join(gained)}); "
                               f"waiting {self.cfg.rejoin_grace:.0f}s for them to settle")
                elif now - self._grow_since >= self.cfg.rejoin_grace:
                    self._grow_since = None
                    self.restart(f"rejoin: growing to {len(desired) + 1} node(s)")
                return
        self._grow_since = None

    def run(self) -> None:
        self.event(f"supervisor starting: {len(self.workers)} worker(s) configured, "
                   f"api :{self.cfg.api_port}, status :{self.cfg.status_port}")
        self.event(f"valid node counts for this model: {self.valid_counts} ({self.node_counts_source})")
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001
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
            "nodes_active": len(self.active) + 1,
            "valid_node_counts": self.valid_counts,
            "node_counts_source": self.node_counts_source,
            "model_header": {k: self.model_header[k] for k in
                             ("arch_type", "dim", "hidden_dim", "n_layers", "n_heads",
                              "n_kv_heads", "head_dim", "vocab_size", "seq_len")
                             if k in self.model_header} if self.model_header else None,
            "active_workers": [w.addr for w in self.active],
            "workers": [w.as_dict(id(w) in active_ids) for w in self.workers],
            "root": {
                "pid": self.proc.pid if self.root_running() else None,
                "api": f"http://{self.cfg.api_host}:{self.cfg.api_port}/v1",
                "model": self.cfg.model,
                "nthreads": self.cfg.nthreads,
                "launched_at": self.launched_at,
                "ready_at": self.ready_at,
                "load_seconds": self.load_seconds,
                "last_api_ok": self.last_api_ok,
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
        if not self.cfg.status_file:
            return
        tmp = self.cfg.status_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.snapshot(), f)
            os.replace(tmp, self.cfg.status_file)
        except OSError as e:
            log(f"cannot write {self.cfg.status_file}: {e}")


# ----------------------------------------------------------------- status HTTP

def make_handler(sup: Supervisor):
    class Handler(BaseHTTPRequestHandler):
        server_version = "dllama-supervisor/1.0"

        def log_message(self, *args) -> None:  # noqa: D102 - quiet
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/status", "/status.json"):
                self._json(200, sup.snapshot())
            elif path == "/healthz":
                self._json(200 if sup.serving else 503, {"ok": sup.serving, "state": sup.state})
            elif path == "/events":
                self._json(200, list(sup.events))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/restart":
                sup.request_restart("manual restart via HTTP")
                self._json(202, {"ok": True, "state": sup.state})
            else:
                self._json(404, {"error": "not found"})

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
    ap.add_argument("--workers", default="pi-node-1.local,pi-node-2.local,pi-node-4.local",
                    help="comma-separated host[:port] in priority order; the first ones are kept "
                         "when the set shrinks, so list the best-cooled / biggest-RAM nodes first")
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
    ap.add_argument("--fail-after", type=int, default=d.fail_after,
                    help="consecutive probe misses before a worker counts as dead")
    ap.add_argument("--ok-after", type=int, default=d.ok_after,
                    help="consecutive probe hits before a dead worker counts as back")
    ap.add_argument("--rejoin-grace", type=float, default=d.rejoin_grace,
                    help="seconds a returned worker must stay up before the set grows")
    ap.add_argument("--no-auto-rejoin", action="store_true")
    ap.add_argument("--reset-cmd", default=d.reset_cmd,
                    help="run on each surviving worker before a relaunch; {host} is substituted")
    ap.add_argument("--no-reset-workers", action="store_true")
    ap.add_argument("--settle", type=float, default=d.settle,
                    help="seconds between killing the root and relaunching it")
    ap.add_argument("--ready-timeout", type=float, default=d.ready_timeout)
    ap.add_argument("--api-check-interval", type=float, default=d.api_check_interval)
    ap.add_argument("--api-stall-timeout", type=float, default=d.api_stall_timeout,
                    help="restart if /v1/models has not answered for this long; 0 disables")
    ap.add_argument("--launch-backoff", type=float, default=d.launch_backoff)
    ap.add_argument("--node-counts", default="",
                    help="comma-separated node counts to allow, e.g. 1,2,4,8; default derives "
                         "them from the model header (divisibility of heads/dims/vocab)")
    ap.add_argument("--print-command", action="store_true",
                    help="print the dllama-api command for the full set and exit")
    ap.add_argument("--print-node-counts", action="store_true",
                    help="print the node counts the model allows and exit")
    return ap


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        workers=parse_workers(args.workers, args.worker_port),
        dllama_bin=os.path.expanduser(args.dllama_bin),
        model=os.path.expanduser(args.model),
        tokenizer=os.path.expanduser(args.tokenizer),
        buffer_float_type=args.buffer_float_type,
        nthreads=args.nthreads,
        api_host=args.api_host,
        api_port=args.api_port,
        extra_args=args.extra_args,
        status_host=args.status_host,
        status_port=args.status_port,
        status_file=args.status_file,
        log_dir=os.path.expanduser(args.log_dir),
        interval=args.interval,
        probe_cmd=args.probe_cmd,
        probe_timeout=args.probe_timeout,
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
        node_counts=[int(c) for c in args.node_counts.split(",") if c.strip()] or None,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    sup = Supervisor(cfg)
    if args.print_command:
        print(shlex.join(sup.build_command(sup.workers)))
        return 0
    if args.print_node_counts:
        print(f"{sup.valid_counts} ({sup.node_counts_source})")
        if sup.model_header:
            h = sup.model_header
            print(f"n_heads={h['n_heads']} kv_dim={h['kv_dim']} hidden_dim={h['hidden_dim']} "
                  f"vocab_size={h['vocab_size']} n_layers={h.get('n_layers')}")
        return 0
    srv = serve_status(sup, cfg.status_host, cfg.status_port)

    def on_signal(signum, _frame) -> None:
        log(f"signal {signum}, shutting down")
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
