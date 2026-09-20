"""
test_agent.py - the telemetry document from fake sysfs/proc files. Run: pytest -q
"""

import json
import sys
import threading
import urllib.error
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
    assert set(doc) >= {
        "cpu_percent",
        "dllama_rss_mb",
        "net_io",
        "worker_listening",
        "worker_connections",
        "worker_unit",
    }


# A worker waiting for its root (LISTEN on 9998 = 0x270E), an ssh session, and a stale TIME_WAIT on 9998.
PROC_NET_TCP_LISTENING = """\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 00000000:270E 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 12345 1 0000000000000000 100 0 0 10 0
   1: 0B32A8C0:0016 6432A8C0:E1A2 01 00000000:00000000 00:00000000 00000000     0        0 11111 1 0000000000000000 20 4 30 10 -1
   2: 0B32A8C0:270E 0A32A8C0:B8F2 06 00000000:00000000 03:00000A1B 00000000     0        0 0 3 0000000000000000
"""
# The same worker after the root connected: listen socket closed, one ESTABLISHED connection on 9998.
PROC_NET_TCP_ATTACHED = """\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0B32A8C0:270E 0A32A8C0:B8F4 01 00000000:00000000 00:00000000 00000000  1000        0 12346 1 0000000000000000 20 4 30 10 -1
   1: 0B32A8C0:0016 6432A8C0:E1A2 01 00000000:00000000 00:00000000 00000000     0        0 11111 1 0000000000000000 20 4 30 10 -1
"""


def test_proc_net_tcp_counts_states_on_the_worker_port():
    assert node_agent.parse_proc_net_tcp(PROC_NET_TCP_LISTENING, 9998) == {"0A": 1, "06": 1}
    assert node_agent.parse_proc_net_tcp(PROC_NET_TCP_ATTACHED, 9998) == {"01": 1}
    assert node_agent.parse_proc_net_tcp(PROC_NET_TCP_ATTACHED, 22) == {"01": 1}
    assert node_agent.parse_proc_net_tcp("header only\n", 9998) == {}


def test_worker_sockets_listening_attached_and_gone(tmp_path):
    tcp = tmp_path / "tcp"
    tcp6 = tmp_path / "tcp6"
    tcp6.write_text("  sl  local_address rem_address st\n")
    tcp.write_text(PROC_NET_TCP_LISTENING)
    assert node_agent.worker_sockets(9998, (str(tcp), str(tcp6))) == (True, 0)
    tcp.write_text(PROC_NET_TCP_ATTACHED)
    assert node_agent.worker_sockets(9998, (str(tcp), str(tcp6))) == (False, 1)
    tcp.write_text(PROC_NET_TCP_ATTACHED.splitlines()[0] + "\n")  # process gone: only TIME_WAIT-free header
    assert node_agent.worker_sockets(9998, (str(tcp), str(tcp6))) == (False, 0)
    assert node_agent.worker_sockets(9998, (str(tmp_path / "nope"),)) == (None, None), "unreadable = unknown, not dead"


def test_telemetry_reports_worker_sockets(tmp_path):
    p = fake_fs(tmp_path)
    tcp = tmp_path / "tcp"
    tcp.write_text(PROC_NET_TCP_LISTENING)
    doc = node_agent.telemetry(
        p["temp"], p["throttled"], p["meminfo"], p["loadavg"], p["uptime"], p["freq"], net_tcp_paths=(str(tcp),)
    )
    assert doc["worker_listening"] is True and doc["worker_connections"] == 0


class FakeRun:
    def __init__(self, stdout: str):
        self.stdout = stdout

    def __call__(self, *args, **kwargs):
        return self


def test_vcgencmd_fallback_when_sysfs_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(node_agent, "_vcgencmd_missing", False)
    monkeypatch.setattr(node_agent.subprocess, "run", FakeRun("throttled=0x50005\n"))
    assert node_agent.read_throttled(str(tmp_path / "missing")) == 0x50005
    monkeypatch.setattr(node_agent.subprocess, "run", FakeRun("throttled=0xzz\n"))
    assert node_agent.read_throttled(str(tmp_path / "missing")) is None, "malformed hex is null, not a crash"
    monkeypatch.setattr(node_agent.subprocess, "run", FakeRun("garbage"))
    assert node_agent.read_throttled(str(tmp_path / "missing")) is None


def test_vcgencmd_is_not_retried_once_known_missing(monkeypatch, tmp_path):
    calls = []

    def missing(*args, **kwargs):
        calls.append(args)
        raise FileNotFoundError("vcgencmd")

    monkeypatch.setattr(node_agent, "_vcgencmd_missing", False)
    monkeypatch.setattr(node_agent.subprocess, "run", missing)
    assert node_agent.read_throttled(str(tmp_path / "missing")) is None
    assert node_agent.read_throttled(str(tmp_path / "missing")) is None
    assert len(calls) == 1


def test_malformed_sysfs_hex_is_null(tmp_path):
    p = fake_fs(tmp_path, throttled="0xnotahex")
    assert node_agent.read_throttled(p["throttled"]) is None
    assert node_agent.parse_hex("50000") == 0x50000 and node_agent.parse_hex(None) is None


def test_throttled_path_is_resolved_by_glob(tmp_path):
    assert node_agent.find_throttled_path(str(tmp_path)) is None
    node = tmp_path / "soc" / "soc:firmware"
    node.mkdir(parents=True)
    (node / "get_throttled").write_text("0\n")
    assert node_agent.find_throttled_path(str(tmp_path)) == str(node / "get_throttled")


def test_cpu_percent_from_two_proc_stat_samples():
    key = "test_cpu"
    node_agent._cpu_last.pop(key, None)
    first = "cpu  1000 0 500 8000 100 0 0 0 0 0\ncpu0 1 2 3 4 5 6 7 8 9 10\n"
    second = "cpu  1600 0 800 8600 100 0 0 0 0 0\n"
    assert node_agent.cpu_percent_from_stat(first, key) is None, "no baseline on the first call"
    assert node_agent.cpu_percent_from_stat(second, key) == 60.0  # 900 busy of 1500 elapsed
    assert node_agent.cpu_percent_from_stat("nonsense", key) is None
    assert node_agent.cpu_percent_from_stat(None, key) is None


def test_net_dev_sums_every_interface_but_lo():
    text = (
        "Inter-|   Receive                                                |  Transmit\n"
        " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n"
        "    lo:   99999    1000    0    0    0     0          0         0    99999    1000    0    0    0     0       0          0\n"
        "  eth0: 5000000   40000    0    0    0     0          0         0  3000000   30000    0    0    0     0       0          0\n"
        " wlan0: 1000000   10000    0    0    0     0          0         0   500000    5000    0    0    0     0       0          0\n"
    )
    assert node_agent.parse_net_dev(text) == {"rx_bytes": 6000000, "tx_bytes": 3500000}
    assert node_agent.parse_net_dev("") is None


def test_dllama_rss_from_a_fake_proc_tree(tmp_path):
    for pid, comm, rss_kb in ((100, "dllama", 300 * 1024), (101, "sshd", 9000), (102, "dllama-api", 112 * 1024)):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "comm").write_text(comm + "\n")
        (d / "status").write_text(f"Name:\t{comm}\nVmRSS:\t{rss_kb} kB\nThreads:\t4\n")
    (tmp_path / "self").mkdir()
    assert node_agent.dllama_rss_mb_from_proc(str(tmp_path)) == 412
    assert node_agent.dllama_rss_mb_from_proc(str(tmp_path / "self")) is None
    assert node_agent.dllama_rss_mb_from_proc(str(tmp_path / "missing")) is None


def test_handler_has_a_socket_timeout():
    assert node_agent.Handler.timeout == 5


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
