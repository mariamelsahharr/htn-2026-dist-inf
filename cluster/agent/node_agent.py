#!/usr/bin/env python3
"""
node_agent.py - one JSON document about this Pi's health, over HTTP. Stdlib only;
psutil (apt python3-psutil) sharpens cpu_percent / dllama_rss_mb / net_io when present.

    GET /telemetry  -> {"temp_c": 61.2, "throttled": "0x50000", "flags": [...], "mem_available_mb": 3120,
                        "mem_total_mb": 8048, "load1": 3.9, "cpu_mhz": 2400, "cpu_percent": 71.5,
                        "dllama_rss_mb": 412, "net_io": {"rx_bytes": ..., "tx_bytes": ...},
                        "worker_listening": true, "worker_connections": 0, "worker_unit": "active",
                        "uptime_s": 8812, "ts": ...}
    GET /healthz    -> ok

The supervisor fetches this for every worker (and itself) and folds it into /status,
so the router, dashboard and anything else read hardware state without SSH.

worker_listening / worker_connections come from /proc/net/tcp: a dllama worker either
listens on 9998 for a root or holds the root's connection on that port (it closes the
listen socket once the root connects). The supervisor uses them to tell a dead worker
process from a dead Pi without ever connecting to 9998 itself, which would kill the worker.
"""

import argparse
import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import psutil  # ty: ignore[unresolved-import]
except ImportError:  # stdlib fallbacks below
    psutil = None

TEMP = "/sys/class/thermal/thermal_zone0/temp"
THROTTLED_DEFAULT = "/sys/devices/platform/soc/soc:firmware/get_throttled"
CPU_FREQ = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
MEMINFO = "/proc/meminfo"
LOADAVG = "/proc/loadavg"
UPTIME = "/proc/uptime"
PROC_STAT = "/proc/stat"
NET_DEV = "/proc/net/dev"
NET_TCP = ("/proc/net/tcp", "/proc/net/tcp6")
PROC = "/proc"
WORKER_PORT = 9998
WORKER_UNIT = "dllama-worker"
DLLAMA_NAMES = ("dllama", "dllama-api")
TCP_ESTABLISHED, TCP_LISTEN = "01", "0A"

# vcgencmd get_throttled bits; 16-19 are the same conditions "since boot"
FLAG_BITS = {
    0: "under_voltage",
    1: "freq_capped",
    2: "throttled",
    3: "soft_temp_limit",
    16: "under_voltage_since_boot",
    17: "freq_capped_since_boot",
    18: "throttled_since_boot",
    19: "soft_temp_limit_since_boot",
}


def _read(path: str) -> str | None:
    try:
        with Path(path).open() as fh:
            return fh.read().strip()
    except OSError:
        return None


def find_throttled_path(root: str = "/sys/devices/platform") -> str | None:
    """The firmware's get_throttled node moved between Pi generations; glob for it once at startup."""
    try:
        return next((str(p) for p in sorted(Path(root).glob("**/get_throttled"))), None)
    except OSError:
        return None


THROTTLED = find_throttled_path() or THROTTLED_DEFAULT


def throttle_flags(value: int) -> list[str]:
    return [name for bit, name in FLAG_BITS.items() if value & (1 << bit)]


_vcgencmd_missing = False


def vcgencmd_throttled() -> str | None:
    """`vcgencmd get_throttled` -> "0x50000"; remembered as absent after the first ENOENT so a
    non-Pi (or a Pi without the userland tools) never shells out per request."""
    global _vcgencmd_missing
    if _vcgencmd_missing:
        return None
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
    except FileNotFoundError:
        _vcgencmd_missing = True
        return None
    except (OSError, subprocess.SubprocessError):
        return None
    return out.split("=", 1)[1].strip() if "=" in out else None


def parse_hex(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw, 16)
    except ValueError:
        return None


def read_throttled(path: str | None = None) -> int | None:
    """The firmware sysfs file (world-readable on Pi OS) or vcgencmd as the fallback."""
    raw = _read(path or THROTTLED)
    if raw is None:
        raw = vcgencmd_throttled()
    return parse_hex(raw)


def read_meminfo(path: str = MEMINFO) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (_read(path) or "").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key] = int(parts[0])
    return out


def parse_proc_net_tcp(text: str, port: int) -> dict[str, int]:
    """Socket states on a local port from /proc/net/tcp{,6}: {"0A": n_listening, "01": n_established, ...}."""
    want = f":{port:04X}"
    states: dict[str, int] = {}
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4 and parts[1].endswith(want):
            states[parts[3]] = states.get(parts[3], 0) + 1
    return states


def worker_sockets(port: int = WORKER_PORT, paths: tuple[str, ...] = NET_TCP) -> tuple[bool | None, int | None]:
    """(listening, established connections) on the worker port; (None, None) when /proc/net/tcp is unreadable."""
    listening = established = 0
    readable = False
    for path in paths:
        text = _read(path)
        if text is None:
            continue
        readable = True
        states = parse_proc_net_tcp(text, port)
        listening += states.get(TCP_LISTEN, 0)
        established += states.get(TCP_ESTABLISHED, 0)
    if not readable:
        return None, None
    return listening > 0, established


def unit_active(unit: str = WORKER_UNIT) -> str | None:
    """`systemctl is-active` word (active / inactive / failed / activating); None without systemd."""
    try:
        r = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None


_cpu_last: dict[str, tuple[int, int]] = {}


def cpu_percent_from_stat(text: str | None, key: str = PROC_STAT) -> float | None:
    """Busy share of all CPUs since the previous call on the same key (/proc/stat "cpu" line);
    None on the first call."""
    if not text:
        return None
    fields = text.split("\n", 1)[0].split()
    if len(fields) < 5 or fields[0] != "cpu" or not all(f.isdigit() for f in fields[1:]):
        return None
    nums = [int(f) for f in fields[1:]]
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)  # idle + iowait
    total = sum(nums)
    prev = _cpu_last.get(key)
    _cpu_last[key] = (total, idle)
    if prev is None or total <= prev[0]:
        return None
    return round(100.0 * (1 - (idle - prev[1]) / (total - prev[0])), 1)


def parse_net_dev(text: str | None) -> dict[str, int] | None:
    """rx/tx byte totals over every interface but lo, from /proc/net/dev."""
    if not text:
        return None
    rx = tx = 0
    found = False
    for line in text.splitlines()[2:]:
        name, _, rest = line.partition(":")
        fields = rest.split()
        if name.strip() == "lo" or len(fields) < 9 or not (fields[0].isdigit() and fields[8].isdigit()):
            continue
        rx += int(fields[0])
        tx += int(fields[8])
        found = True
    return {"rx_bytes": rx, "tx_bytes": tx} if found else None


def dllama_rss_mb_from_proc(proc: str = PROC, names: tuple[str, ...] = DLLAMA_NAMES) -> int | None:
    """Resident set of the dllama / dllama-api process(es) from /proc/<pid>/{comm,status}; None when none runs."""
    try:
        pids = [p for p in Path(proc).iterdir() if p.name.isdigit()]
    except OSError:
        return None
    total = None
    for p in pids:
        if _read(str(p / "comm")) not in names:
            continue
        for line in (_read(str(p / "status")) or "").splitlines():
            if line.startswith("VmRSS:"):
                kb = line.split()[1:2]
                if kb and kb[0].isdigit():
                    total = (total or 0) + int(kb[0]) // 1024
    return total


def cpu_percent_psutil() -> float | None:
    if psutil is None:
        return None
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def net_io_psutil() -> dict[str, int] | None:
    if psutil is None:
        return None
    try:
        c = psutil.net_io_counters()
        return {"rx_bytes": c.bytes_recv, "tx_bytes": c.bytes_sent}
    except Exception:
        return None


def dllama_rss_mb_psutil(names: tuple[str, ...] = DLLAMA_NAMES) -> int | None:
    if psutil is None:
        return None
    total = None
    try:
        for p in psutil.process_iter(["name", "memory_info"]):
            if p.info["name"] in names and p.info["memory_info"]:
                total = (total or 0) + p.info["memory_info"].rss // (1024 * 1024)
    except Exception:
        return total
    return total


def telemetry(
    temp_path: str = TEMP,
    throttled_path: str | None = None,
    meminfo_path: str = MEMINFO,
    loadavg_path: str = LOADAVG,
    uptime_path: str = UPTIME,
    cpu_freq_path: str = CPU_FREQ,
    *,
    stat_path: str = PROC_STAT,
    net_dev_path: str = NET_DEV,
    net_tcp_paths: tuple[str, ...] = NET_TCP,
    proc: str = PROC,
    worker_port: int = WORKER_PORT,
    worker_unit: str | None = WORKER_UNIT,
) -> dict:
    temp = _read(temp_path)
    throttled = read_throttled(throttled_path)
    mem = read_meminfo(meminfo_path)
    load = (_read(loadavg_path) or "").split()
    uptime = (_read(uptime_path) or "").split()
    freq = _read(cpu_freq_path)
    listening, connections = worker_sockets(worker_port, net_tcp_paths)
    return {
        "temp_c": round(int(temp) / 1000, 1) if temp and temp.lstrip("-").isdigit() else None,
        "throttled": f"0x{throttled:x}" if throttled is not None else None,
        "flags": throttle_flags(throttled) if throttled is not None else [],
        "mem_available_mb": mem["MemAvailable"] // 1024 if "MemAvailable" in mem else None,
        "mem_total_mb": mem["MemTotal"] // 1024 if "MemTotal" in mem else None,
        "load1": float(load[0]) if load else None,
        "cpu_mhz": int(freq) // 1000 if freq and freq.isdigit() else None,
        "cpu_percent": cpu_percent_psutil() if psutil else cpu_percent_from_stat(_read(stat_path), stat_path),
        "dllama_rss_mb": dllama_rss_mb_psutil() if psutil else dllama_rss_mb_from_proc(proc),
        "net_io": net_io_psutil() if psutil else parse_net_dev(_read(net_dev_path)),
        "worker_listening": listening,
        "worker_connections": connections,
        "worker_unit": unit_active(worker_unit) if worker_unit else None,
        "uptime_s": int(float(uptime[0])) if uptime else None,
        "ts": time.time(),
    }


class Handler(BaseHTTPRequestHandler):
    timeout = 5  # a client that stalls mid-request must not pin a thread forever

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/telemetry":
            body = json.dumps(telemetry()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        elif self.path == "/healthz":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
        else:
            body = b"not found"
            self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9997)
    ap.add_argument("--once", action="store_true", help="print one document and exit")
    args = ap.parse_args()
    if psutil:
        psutil.cpu_percent(interval=None)  # prime: the first sample is always 0.0
    if args.once:
        print(json.dumps(telemetry(), indent=2))
        return
    print(f"node_agent: throttle source {THROTTLED}, psutil {'yes' if psutil else 'no'}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
