#!/usr/bin/env python3
"""
metrics.py - live per-node telemetry for a Raspberry Pi distributed-llama cluster.

Design notes:
  * One PERSISTENT ssh connection per node, running a tiny bash loop that emits a
    stats line every INTERVAL seconds. No per-poll connection setup (which costs
    200-500ms and would wreck your timing), and a node that loses power shows up
    as DOWN within a few seconds instead of hanging a poll.
  * Zero third-party dependencies. Stdlib only. This matters at 2am.
  * Everything printed is also appended to CSV so you can line up the unplug drill
    timeline with loadtest.py output afterwards.

Usage:
    ./metrics.py --nodes 192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13 \
                 --user pi --csv run1.csv \
                 --status-url http://192.168.1.10:9991/status \
                 --tps-file /tmp/dllama_tps.json

    Press ENTER at any time to drop a marker row into the CSV (use this the
    instant you pull a power cable).
"""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

# ---------------------------------------------------------------- remote probe

REMOTE_PROBE = r"""
while true; do
  TS=$(date +%s.%N)
  CPU=$(grep '^cpu ' /proc/stat)
  if command -v vcgencmd >/dev/null 2>&1; then
    TEMP=$(vcgencmd measure_temp 2>/dev/null)
    THR=$(vcgencmd get_throttled 2>/dev/null)
    CLK=$(vcgencmd measure_clock arm 2>/dev/null)
  else
    TEMP="temp=$(awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp)'C"
    THR="throttled=0x0"
    CLK="frequency(48)=$(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0) * 1000 ))"
  fi
  MEMA=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
  LOAD=$(awk '{print $1}' /proc/loadavg)
  NPROC=$(nproc)
  echo "STAT|$TS|$CPU|$TEMP|$THR|$CLK|$MEMA|$LOAD|$NPROC"
  sleep %INTERVAL%
done
"""

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=5",
    # Keepalives tuned low so a yanked power cable is detected in ~6s, not 2min.
    "-o", "ServerAliveInterval=2",
    "-o", "ServerAliveCountMax=3",
    "-o", "ExitOnForwardFailure=yes",
]


def ssh_prefix(ssh_key=None, password=None):
    """Return (argv_prefix, env) for an ssh invocation.

    With a password we shell out through sshpass in -e mode, so the password
    travels in the child's environment and never appears in `ps` output.
    Without one we use BatchMode so a missing key fails fast instead of
    hanging on an interactive prompt.
    """
    env = dict(os.environ)
    cmd = []
    if password:
        cmd += ["sshpass", "-e"]
        env["SSHPASS"] = password
    cmd += ["ssh"] + SSH_OPTS
    if password:
        cmd += ["-o", "PubkeyAuthentication=no",
                "-o", "PreferredAuthentications=password,keyboard-interactive",
                "-o", "NumberOfPasswordPrompts=1"]
    else:
        cmd += ["-o", "BatchMode=yes"]
    if ssh_key:
        cmd += ["-i", ssh_key]
    return cmd, env

# Bit meanings from `vcgencmd get_throttled`.
# NOTE: bits 0-3 are LIVE state. Bits 16-19 are STICKY "has happened since boot"
# flags that never clear until reboot. 0x50000 == bits 16+18 == "under-voltage
# HAS occurred" + "throttling HAS occurred", i.e. history, not right now.
# The live "currently throttled" bit is bit 2 (0x4).
THROTTLE_BITS = [
    (0,  "UNDERVOLT!",   True),
    (1,  "FREQCAP!",     True),
    (2,  "THROTTLED!",   True),
    (3,  "SOFTTEMP!",    True),
    (16, "uv-past",      False),
    (17, "cap-past",     False),
    (18, "thr-past",     False),
    (19, "soft-past",    False),
]


def decode_throttle(value):
    live, past = [], []
    for bit, label, is_live in THROTTLE_BITS:
        if value & (1 << bit):
            (live if is_live else past).append(label)
    return live, past


# ---------------------------------------------------------------- ANSI helpers

class C:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    YEL = "\033[33m"
    GRN = "\033[32m"
    CYA = "\033[36m"
    MAG = "\033[35m"


def color(s, c, enabled=True):
    return f"{c}{s}{C.RESET}" if enabled else s


# ---------------------------------------------------------------- node monitor

class Node:
    """One Pi. Owns a persistent ssh process and a reader thread."""

    def __init__(self, host, user, interval, ssh_key=None, name=None, password=None):
        self.host = host
        self.name = name or short_name(host)
        self.user = user
        self.interval = interval
        self.ssh_key = ssh_key
        self.password = password

        self.lock = threading.Lock()
        self.state = "connecting"      # connecting | ok | stale | down
        self.sample = {}               # latest decoded sample
        self.last_seen = 0.0
        self.prev_cpu = None           # (busy, total)
        self.reconnects = 0
        self.last_error = ""

        self._proc = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._kill_proc()

    def _kill_proc(self):
        p = self._proc
        if p and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

    def _ssh_cmd(self):
        cmd, env = ssh_prefix(self.ssh_key, self.password)
        cmd = cmd + [f"{self.user}@{self.host}", "bash -s"]
        return cmd, env

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            try:
                with self.lock:
                    if self.state != "connecting":
                        self.reconnects += 1
                    self.state = "connecting"
                argv, env = self._ssh_cmd()
                self._proc = subprocess.Popen(
                    argv,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                script = REMOTE_PROBE.replace("%INTERVAL%", str(self.interval))
                self._proc.stdin.write(script)
                self._proc.stdin.close()

                for line in self._proc.stdout:
                    if self._stop.is_set():
                        break
                    if line.startswith("STAT|"):
                        self._ingest(line.rstrip("\n"))
                        backoff = 1.0

                err = ""
                try:
                    err = (self._proc.stderr.read() or "").strip().splitlines()[-1]
                except Exception:
                    pass
                with self.lock:
                    self.state = "down"
                    self.last_error = err[:60]
            except Exception as e:  # ssh binary missing, DNS, etc.
                with self.lock:
                    self.state = "down"
                    self.last_error = str(e)[:60]
            finally:
                self._kill_proc()

            # Reconnect with capped backoff so a rebooting Pi rejoins on its own.
            self._stop.wait(backoff)
            backoff = min(backoff * 1.7, 8.0)

    # -- parsing -----------------------------------------------------------

    def _ingest(self, line):
        try:
            parts = line.split("|")
            _, ts, cpu_line, temp_s, thr_s, clk_s, mem_s, load_s, nproc_s = parts[:9]

            f = [int(x) for x in cpu_line.split()[1:]]
            idle_all = f[3] + (f[4] if len(f) > 4 else 0)
            total = sum(f)
            busy = total - idle_all

            cpu_pct = None
            if self.prev_cpu:
                d_busy = busy - self.prev_cpu[0]
                d_total = total - self.prev_cpu[1]
                if d_total > 0:
                    cpu_pct = 100.0 * d_busy / d_total
            self.prev_cpu = (busy, total)

            m = re.search(r"temp=([\d.]+)", temp_s)
            temp_c = float(m.group(1)) if m else None

            m = re.search(r"0x([0-9a-fA-F]+)", thr_s)
            thr = int(m.group(1), 16) if m else 0

            m = re.search(r"=(\d+)", clk_s)
            arm_mhz = int(m.group(1)) // 1_000_000 if m else None
            if not arm_mhz:          # sysfs fallback can report 0; show as unknown
                arm_mhz = None

            live, past = decode_throttle(thr)

            with self.lock:
                self.sample = {
                    "remote_ts": float(ts),
                    "cpu_pct": cpu_pct,
                    "temp_c": temp_c,
                    "arm_mhz": arm_mhz,
                    "throttled": thr,
                    "live_flags": live,
                    "past_flags": past,
                    "mem_avail_mb": int(mem_s) // 1024 if mem_s.strip() else None,
                    "load1": float(load_s),
                    "nproc": int(nproc_s),
                }
                self.last_seen = time.time()
                self.state = "ok"
        except Exception as e:
            with self.lock:
                self.last_error = f"parse: {e}"[:60]

    def snapshot(self, stale_after):
        with self.lock:
            st = self.state
            if st == "ok" and (time.time() - self.last_seen) > stale_after:
                st = "stale"
            return st, dict(self.sample), self.reconnects, self.last_error


# ---------------------------------------------------------------- side pollers

def short_name(host):
    """pi-node-1.local -> pi-node-1; 192.168.50.11 stays 192.168.50.11."""
    return host[:-6] if host.endswith(".local") else host


CSV_HEADER = [
    "ts_iso", "ts_unix", "node", "state", "cpu_pct", "temp_c",
    "arm_mhz", "throttled_hex", "live_flags", "past_flags",
    "mem_avail_mb", "load1", "reconnects", "cluster_status",
    "tps", "tps_source", "event", "cluster_nodes", "load_s",
]


def parse_status(raw):
    """(state, 'active/total', 'load seconds') from the supervisor's /status document
    (cluster/supervisor/status.example.json). Missing fields become '' rather than errors."""
    state = str(raw.get("state", raw.get("status", "?")))
    a, t = raw.get("nodes_active"), raw.get("nodes_total")
    nodes = f"{a}/{t}" if a is not None and t is not None else ""
    ls = (raw.get("root") or {}).get("load_seconds")
    load_s = f"{ls:.1f}" if isinstance(ls, (int, float)) else ""
    return state, nodes, load_s


class StatusPoller(threading.Thread):
    """Polls Person 1's supervisor status JSON over HTTP."""

    def __init__(self, url, interval):
        super().__init__(daemon=True)
        self.url = url
        self.interval = interval
        self.value = "n/a"
        self.nodes = ""          # "2/4" active/total from the supervisor
        self.load_s = ""         # seconds the last launch took to become ready
        self.raw = {}
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=2) as r:
                    data = json.loads(r.read().decode())
                self.raw = data if isinstance(data, dict) else {}
                self.value, self.nodes, self.load_s = parse_status(self.raw)
            except Exception:
                self.value, self.nodes, self.load_s = "unreachable", "", ""
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


class TpsFilePoller(threading.Thread):
    """Reads the sidecar JSON that loadtest.py writes: {"tps":..,"inflight":..}."""

    def __init__(self, path, interval):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self.tps = None
        self.inflight = None
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                age = time.time() - os.path.getmtime(self.path)
                if age < 15:
                    with open(self.path) as fh:
                        d = json.load(fh)
                    self.tps = d.get("tps")
                    self.inflight = d.get("inflight")
                else:
                    self.tps, self.inflight = None, None
            except Exception:
                self.tps, self.inflight = None, None
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


class LogTailer(threading.Thread):
    """Tails the root node's log over ssh and scrapes a tokens/sec number."""

    def __init__(self, host, user, path, pattern, unit, ssh_key=None, password=None):
        super().__init__(daemon=True)
        self.host, self.user, self.path = host, user, path
        self.re = re.compile(pattern)
        self.unit = unit  # "tps" or "ms_per_token"
        self.ssh_key = ssh_key
        self.password = password
        self.tps = None
        self.last_line = ""
        self._stop = threading.Event()
        self._proc = None

    def run(self):
        while not self._stop.is_set():
            cmd, env = ssh_prefix(self.ssh_key, self.password)
            cmd = cmd + [f"{self.user}@{self.host}", f"tail -n 0 -F {self.path}"]
            try:
                self._proc = subprocess.Popen(
                    cmd, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, text=True, bufsize=1)
                for line in self._proc.stdout:
                    if self._stop.is_set():
                        break
                    m = self.re.search(line)
                    if m:
                        try:
                            v = float(m.group(1))
                            self.tps = (1000.0 / v) if (self.unit == "ms_per_token" and v > 0) else v
                            self.last_line = line.strip()[:100]
                        except Exception:
                            pass
            except Exception:
                pass
            self._stop.wait(3)

    def stop(self):
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.kill()


# ---------------------------------------------------------------- rendering

def fmt(v, spec, default="-"):
    return default if v is None else format(v, spec)


def temp_color(t):
    if t is None:
        return C.DIM
    if t >= 80:
        return C.RED
    if t >= 70:
        return C.YEL
    return C.GRN


def state_color(s):
    return {"ok": C.GRN, "connecting": C.YEL, "stale": C.YEL, "down": C.RED}.get(s, C.DIM)


def render(nodes, status, tps_src, tps_val, marks, started, stale_after, use_color,
           status_nodes="", status_load=""):
    cols = shutil.get_terminal_size((100, 30)).columns
    out = []
    elapsed = time.time() - started
    hdr = (f"{C.BOLD}dllama cluster{C.RESET}  "
           f"t+{int(elapsed)//60:02d}:{int(elapsed)%60:02d}  "
           f"{datetime.now().strftime('%H:%M:%S')}")
    bits = [hdr]
    if status is not None:
        sc = C.GRN if status == "healthy" else (C.RED if status in ("down", "unreachable") else C.YEL)
        label = status + (f" {status_nodes}" if status_nodes else "") + (f" load={status_load}s" if status_load else "")
        bits.append("supervisor=" + color(label, sc, use_color))
    if tps_val is not None:
        bits.append(f"tok/s={color(f'{tps_val:.1f}', C.CYA, use_color)} ({tps_src})")
    else:
        bits.append(f"tok/s=- ({tps_src})")
    bits.append(f"marks={marks}")
    out.append("  ".join(bits))

    out.append(color(
        f"{'NODE':<16}{'STATE':<12}{'CPU%':>6}  {'TEMP':>7}  {'ARM':>8}  "
        f"{'LOAD':>6}  {'MEMfree':>8}  {'RC':>3}  FLAGS", C.DIM, use_color))

    hot = []
    for n in nodes:
        st, s, rc, err = n.snapshot(stale_after)
        temp = s.get("temp_c")
        if temp is not None:
            hot.append((temp, n.name))
        live = s.get("live_flags") or []
        past = s.get("past_flags") or []
        flags = ""
        if live:
            flags += color(",".join(live), C.RED + C.BOLD, use_color)
        if past:
            flags += (" " if flags else "") + color(",".join(past), C.DIM, use_color)
        if st == "down" and err:
            flags = color(err, C.RED, use_color)

        row = (f"{n.name:<16}"
               f"{color(f'{st:<12}', state_color(st), use_color)}"
               f"{fmt(s.get('cpu_pct'), '6.1f')}  "
               f"{color((fmt(temp, '6.1f') + 'C') if temp is not None else '      -', temp_color(temp), use_color)}  "
               f"{fmt(s.get('arm_mhz'), '5d') + 'MHz' if s.get('arm_mhz') else '       -'}  "
               f"{fmt(s.get('load1'), '6.2f')}  "
               f"{(str(s.get('mem_avail_mb')) + 'MB').rjust(8) if s.get('mem_avail_mb') else '       -'}  "
               f"{rc:>3}  {flags}")
        out.append(row[:cols + 200])

    if len(hot) > 1:
        hot.sort(reverse=True)
        spread = hot[0][0] - hot[-1][0]
        note = (f"straggler watch: hottest {hot[0][1]} {hot[0][0]:.1f}C, "
                f"coolest {hot[-1][1]} {hot[-1][0]:.1f}C, spread {spread:.1f}C")
        out.append(color(note, C.MAG if spread >= 6 else C.DIM, use_color))

    out.append(color("ENTER = mark event   Ctrl-C = stop", C.DIM, use_color))
    return "\n".join(out)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Live telemetry for a Pi llama cluster.")
    ap.add_argument("--nodes",
                    default="pi-node-1.local,pi-node-2.local,pi-node-3.local,pi-node-4.local",
                    help="comma-separated hosts; first one is treated as the root node")
    ap.add_argument("--names", default="",
                    help="display names, same order as --nodes (default: derived from hostname)")
    ap.add_argument("--user", default="pi")
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--password", default=None,
                    help="SSH password (needs sshpass installed). Prefer --password-env.")
    ap.add_argument("--password-env", default=None, metavar="VAR",
                    help="read the SSH password from this environment variable")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--csv", default=None, help="append samples here (recommended)")
    ap.add_argument("--status-url", default=None,
                    help="supervisor status JSON URL, e.g. http://192.168.1.10:9991/status")
    ap.add_argument("--tps-file", default=None,
                    help="sidecar JSON written by loadtest.py, e.g. /tmp/dllama_tps.json")
    ap.add_argument("--log-path", default=None,
                    help="root-node log to tail for tokens/sec, e.g. /var/log/dllama-root.log")
    ap.add_argument("--tps-regex", default=r"(?i)(\d+(?:\.\d+)?)\s*(?:tok(?:ens)?/s|tps)")
    ap.add_argument("--tps-unit", choices=["tps", "ms_per_token"], default="tps")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    password = args.password
    if args.password_env:
        password = os.environ.get(args.password_env)
        if not password:
            sys.exit(f"error: ${args.password_env} is not set")
    if password and not shutil.which("sshpass"):
        sys.exit("error: --password needs sshpass.\n"
                 "  Debian/Ubuntu/WSL : sudo apt install sshpass\n"
                 "  macOS             : brew install hudochenkov/sshpass/sshpass\n"
                 "  or drop the flag and use an SSH key instead.")

    use_color = not args.no_color and sys.stdout.isatty()
    hosts = [h.strip() for h in args.nodes.split(",") if h.strip()]
    names = [n.strip() for n in args.names.split(",")] if args.names.strip() else []
    # default display name: strip the .local suffix off the hostname; IPs stay whole
    names += [short_name(hosts[i]) for i in range(len(names), len(hosts))]

    nodes = [Node(h, args.user, args.interval, args.ssh_key, names[i], password)
             for i, h in enumerate(hosts)]
    for n in nodes:
        n.start()

    status_poller = None
    if args.status_url:
        status_poller = StatusPoller(args.status_url, args.interval)
        status_poller.start()

    tps_poller = None
    if args.tps_file:
        tps_poller = TpsFilePoller(args.tps_file, args.interval)
        tps_poller.start()

    tailer = None
    if args.log_path:
        tailer = LogTailer(hosts[0], args.user, args.log_path,
                           args.tps_regex, args.tps_unit, args.ssh_key, password)
        tailer.start()

    # CSV
    writer = fh = None
    csv_lock = threading.Lock()   # marker rows come from the ENTER thread
    if args.csv:
        new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        if not new:
            with open(args.csv, newline="") as check:
                existing = next(csv.reader(check), [])
            if existing != CSV_HEADER:
                sys.exit(f"{args.csv} has a different column layout ({len(existing)} columns, "
                         f"expected {len(CSV_HEADER)}); use a new filename")
        fh = open(args.csv, "a", newline="")
        writer = csv.writer(fh)
        if new:
            writer.writerow(CSV_HEADER)

    marks = {"n": 0}
    stop = threading.Event()

    def mark_reader():
        if not sys.stdin.isatty():
            return
        while not stop.is_set():
            try:
                label = sys.stdin.readline()
            except Exception:
                return
            if not label:
                return
            marks["n"] += 1
            label = label.strip() or f"mark-{marks['n']}"
            if writer:
                now = time.time()
                with csv_lock:
                    writer.writerow([
                        datetime.now(timezone.utc).isoformat(), f"{now:.3f}", "-", "-",
                        "", "", "", "", "", "", "", "", "", "", "", "", label, "", "",
                    ])
                    fh.flush()

    threading.Thread(target=mark_reader, daemon=True).start()

    def handle_sigint(*_):
        stop.set()
    signal.signal(signal.SIGINT, handle_sigint)

    started = time.time()
    stale_after = max(3 * args.interval, 5.0)
    sys.stdout.write("\033[2J")
    try:
        while not stop.is_set():
            status = status_poller.value if status_poller else None
            if tps_poller and tps_poller.tps is not None:
                tps_val, tps_src = tps_poller.tps, "client"
            elif tailer and tailer.tps is not None:
                tps_val, tps_src = tailer.tps, "rootlog"
            else:
                tps_val, tps_src = None, ("client" if tps_poller else
                                          "rootlog" if tailer else "none")

            frame = render(nodes, status, tps_src, tps_val, marks["n"],
                           started, stale_after, use_color,
                           status_poller.nodes if status_poller else "",
                           status_poller.load_s if status_poller else "")
            sys.stdout.write("\033[H" + frame + "\033[J")
            sys.stdout.flush()

            if writer:
                now = time.time()
                iso = datetime.now(timezone.utc).isoformat()
                with csv_lock:
                    for n in nodes:
                        st, s, rc, _ = n.snapshot(stale_after)
                        writer.writerow([
                            iso, f"{now:.3f}", n.name, st,
                            fmt(s.get("cpu_pct"), ".2f", ""),
                            fmt(s.get("temp_c"), ".1f", ""),
                            s.get("arm_mhz", "") if s.get("arm_mhz") is not None else "",
                            hex(s.get("throttled", 0)) if s else "",
                            ";".join(s.get("live_flags") or []),
                            ";".join(s.get("past_flags") or []),
                            s.get("mem_avail_mb", "") if s.get("mem_avail_mb") is not None else "",
                            fmt(s.get("load1"), ".2f", ""),
                            rc,
                            status or "",
                            f"{tps_val:.2f}" if tps_val is not None else "",
                            tps_src, "",
                            status_poller.nodes if status_poller else "",
                            status_poller.load_s if status_poller else "",
                        ])
                    fh.flush()

            stop.wait(args.interval)
    finally:
        for n in nodes:
            n.stop()
        for p in (status_poller, tps_poller, tailer):
            if p:
                p.stop()
        if fh:
            fh.close()
        sys.stdout.write("\033[?25h\n")
        print(f"stopped. {'csv: ' + args.csv if args.csv else 'no csv written'}")


if __name__ == "__main__":
    main()
