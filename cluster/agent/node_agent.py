#!/usr/bin/env python3
"""
node_agent.py - one JSON document about this Pi's health, over HTTP. Stdlib only.

    GET /telemetry  -> {"temp_c": 61.2, "throttled": "0x50000", "flags": [...], "mem_available_mb": 3120,
                        "mem_total_mb": 8048, "load1": 3.9, "cpu_mhz": 2400, "uptime_s": 8812, "ts": ...}
    GET /healthz    -> ok

The supervisor fetches this for every worker (and itself) and folds it into /status,
so the router, dashboard and anything else read hardware state without SSH.
"""

import argparse
import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TEMP = "/sys/class/thermal/thermal_zone0/temp"
THROTTLED = "/sys/devices/platform/soc/soc:firmware/get_throttled"
CPU_FREQ = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
MEMINFO = "/proc/meminfo"
LOADAVG = "/proc/loadavg"
UPTIME = "/proc/uptime"

# vcgencmd get_throttled bits; 16-19 are the same conditions "since boot"
FLAG_BITS = {0: "under_voltage", 1: "freq_capped", 2: "throttled", 3: "soft_temp_limit",
             16: "under_voltage_since_boot", 17: "freq_capped_since_boot",
             18: "throttled_since_boot", 19: "soft_temp_limit_since_boot"}


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def throttle_flags(value: int) -> list[str]:
    return [name for bit, name in FLAG_BITS.items() if value & (1 << bit)]


def read_throttled(path: str = THROTTLED) -> int | None:
    """The firmware sysfs file (world-readable on Pi OS) or vcgencmd as the fallback."""
    raw = _read(path)
    if raw is None:
        try:
            out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
            raw = out.split("=", 1)[1].strip() if "=" in out else None
        except (OSError, subprocess.SubprocessError):
            raw = None
    if raw is None:
        return None
    try:
        return int(raw, 16)
    except ValueError:
        return None


def read_meminfo(path: str = MEMINFO) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (_read(path) or "").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key] = int(parts[0])
    return out


def telemetry(temp_path: str = TEMP, throttled_path: str = THROTTLED, meminfo_path: str = MEMINFO,
              loadavg_path: str = LOADAVG, uptime_path: str = UPTIME, cpu_freq_path: str = CPU_FREQ) -> dict:
    temp = _read(temp_path)
    throttled = read_throttled(throttled_path)
    mem = read_meminfo(meminfo_path)
    load = (_read(loadavg_path) or "").split()
    uptime = (_read(uptime_path) or "").split()
    freq = _read(cpu_freq_path)
    return {
        "temp_c": round(int(temp) / 1000, 1) if temp and temp.lstrip("-").isdigit() else None,
        "throttled": f"0x{throttled:x}" if throttled is not None else None,
        "flags": throttle_flags(throttled) if throttled is not None else [],
        "mem_available_mb": mem["MemAvailable"] // 1024 if "MemAvailable" in mem else None,
        "mem_total_mb": mem["MemTotal"] // 1024 if "MemTotal" in mem else None,
        "load1": float(load[0]) if load else None,
        "cpu_mhz": int(freq) // 1000 if freq and freq.isdigit() else None,
        "uptime_s": int(float(uptime[0])) if uptime else None,
        "ts": time.time(),
    }


class Handler(BaseHTTPRequestHandler):
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

    def log_message(self, *_args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9997)
    ap.add_argument("--once", action="store_true", help="print one document and exit")
    args = ap.parse_args()
    if args.once:
        print(json.dumps(telemetry(), indent=2))
        return
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
