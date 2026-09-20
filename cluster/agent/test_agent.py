"""
test_agent.py - the telemetry document from fake sysfs/proc files. Run: pytest -q
"""

import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import node_agent


def fake_fs(tmp_path, throttled="50000"):
    files = {
        "temp": "61234\n",
        "throttled": throttled + "\n",
        "meminfo": "MemTotal:        8241000 kB\nMemFree:          500000 kB\nMemAvailable:    3195000 kB\n",
        "loadavg": "3.91 2.10 1.05 2/410 12345\n",
        "uptime": "8812.55 30000.1\n",
        "freq": "2400000\n",
    }
    paths = {}
    for name, content in files.items():
        p = tmp_path / name
        p.write_text(content)
        paths[name] = str(p)
    return paths


def test_document_from_sysfs(tmp_path):
    p = fake_fs(tmp_path)
    doc = node_agent.telemetry(p["temp"], p["throttled"], p["meminfo"], p["loadavg"], p["uptime"], p["freq"])
    assert doc["temp_c"] == 61.2 and doc["throttled"] == "0x50000"
    assert doc["flags"] == ["under_voltage_since_boot", "throttled_since_boot"]
    assert (doc["mem_available_mb"], doc["mem_total_mb"]) == (3120, 8047)
    assert (doc["load1"], doc["cpu_mhz"], doc["uptime_s"]) == (3.91, 2400, 8812)
    assert doc["ts"] > 0


def test_current_throttle_bits_are_named():
    assert node_agent.throttle_flags(0x50005) == [
        "under_voltage",
        "throttled",
        "under_voltage_since_boot",
        "throttled_since_boot",
    ]
    assert node_agent.throttle_flags(0) == []


def test_missing_files_give_nulls_not_errors(tmp_path):
    doc = node_agent.telemetry(*(str(tmp_path / f"missing{i}") for i in range(6)))
    assert doc["temp_c"] is None and doc["throttled"] is None and doc["flags"] == []
    assert doc["mem_available_mb"] is None and doc["load1"] is None and doc["uptime_s"] is None


def test_http_endpoint_serves_json_and_404s():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), node_agent.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    doc = json.load(urllib.request.urlopen(f"{base}/telemetry", timeout=2))
    assert set(doc) >= {"temp_c", "throttled", "flags", "mem_available_mb", "load1", "uptime_s", "ts"}
    assert urllib.request.urlopen(f"{base}/healthz", timeout=2).read() == b"ok"
    try:
        urllib.request.urlopen(f"{base}/nope", timeout=2)
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    srv.shutdown()
