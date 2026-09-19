"""
test_supervisor.py - the failover ladder, exercised on a laptop. Run: pytest -q

Pure-logic tests for set selection and probe hysteresis, plus an end-to-end run
with fake_dllama_api.py standing in for the real root: workers are "unplugged"
by writing their names to a file the probe and the fake both read.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from supervisor import (DEGRADED, DOWN, HEALTHY, MODEL_MAGIC, RESTARTING,  # noqa: E402
                        Config, Supervisor, Worker, build_parser, choose_workers,
                        config_from_args, largest_power_of_two_at_most,
                        parse_workers, powers_of_two, read_model_header,
                        serve_status, valid_node_counts)

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake_dllama_api.py")

# Header key ids from distributed-llama src/llm.hpp
KEY = {"dim": 2, "hidden_dim": 3, "n_layers": 4, "n_heads": 5, "n_kv_heads": 6,
       "vocab_size": 9, "seq_len": 10, "weight_float_type": 13, "head_dim": 19,
       "moe_hidden_dim": 21}

LLAMA_3_2_3B = dict(dim=3072, hidden_dim=8192, n_layers=28, n_heads=24, n_kv_heads=8,
                    vocab_size=128256, seq_len=8192, weight_float_type=2, head_dim=128)
QWEN3_0_6B = dict(dim=1024, hidden_dim=3072, n_layers=28, n_heads=16, n_kv_heads=8,
                  vocab_size=151936, seq_len=4096, weight_float_type=2, head_dim=128)
# Synthetic model whose dims all divide by 3: valid counts are not powers of two.
THREESY = dict(dim=3072, hidden_dim=6144, n_layers=4, n_heads=24, n_kv_heads=6,
               vocab_size=98304, seq_len=1024, weight_float_type=2, head_dim=128)


def write_model(path, params):
    """Write a .m file with only the header, the way converter/writer.py does."""
    import struct
    data = b"".join(struct.pack("<ii", KEY[k], v) for k, v in params.items())
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", MODEL_MAGIC, 8 + len(data)))
        f.write(data)
        f.write(b"\0" * 64)   # a few bytes of "weights"
    return str(path)


# --------------------------------------------------------------- set selection

@pytest.mark.parametrize("n,expected", [(1, 1), (2, 2), (3, 2), (4, 4), (5, 4), (7, 4), (8, 8), (9, 8)])
def test_largest_power_of_two(n, expected):
    assert largest_power_of_two_at_most(n) == expected


def test_three_alive_workers_make_a_four_node_set():
    assert choose_workers(["a", "b", "c"]) == ["a", "b", "c"]


def test_two_alive_workers_drop_to_two_nodes_keeping_priority():
    assert choose_workers(["a", "b"]) == ["a"]


def test_one_alive_worker_is_a_two_node_set():
    assert choose_workers(["c"]) == ["c"]


def test_no_alive_workers_means_root_alone():
    assert choose_workers([]) == []


def test_seven_workers_make_eight_nodes_and_six_make_four():
    assert len(choose_workers(list("abcdefg"))) == 7
    assert choose_workers(list("abcdef")) == ["a", "b", "c"]


def test_valid_counts_drive_the_choice_not_powers_of_two():
    counts = [1, 2, 3, 4, 6, 8]
    assert choose_workers(list("ab"), counts) == ["a", "b"]          # 3 nodes
    assert choose_workers(list("abcde"), counts) == list("abcde")    # 6 nodes
    assert choose_workers(list("abcd"), counts) == list("abc")       # 5 not valid -> 4
    assert choose_workers(list("abcdef"), counts) == list("abcde")   # 7 not valid -> 6


def test_powers_of_two_helper():
    assert powers_of_two(1) == [1] and powers_of_two(5) == [1, 2, 4] and powers_of_two(8) == [1, 2, 4, 8]


# ---------------------------------------------------------------- model header

def test_read_header_round_trips_the_converter_format(tmp_path):
    h = read_model_header(write_model(tmp_path / "m.m", LLAMA_3_2_3B))
    assert h["n_heads"] == 24 and h["n_kv_heads"] == 8 and h["vocab_size"] == 128256
    assert h["q_dim"] == 3072 and h["kv_dim"] == 1024


def test_head_dim_is_derived_when_absent(tmp_path):
    params = {k: v for k, v in LLAMA_3_2_3B.items() if k != "head_dim"}
    h = read_model_header(write_model(tmp_path / "m.m", params))
    assert h["head_dim"] == 128 and h["kv_dim"] == 1024


def test_bad_magic_is_rejected(tmp_path):
    p = tmp_path / "x.m"
    p.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError):
        read_model_header(str(p))


def test_llama_3_2_3b_allows_exactly_1_2_4_8(tmp_path):
    h = read_model_header(write_model(tmp_path / "m.m", LLAMA_3_2_3B))
    assert valid_node_counts(h, 32) == [1, 2, 4, 8]      # 16 fails on 24 heads


def test_qwen3_0_6b_allows_up_to_16(tmp_path):
    h = read_model_header(write_model(tmp_path / "m.m", QWEN3_0_6B))
    assert valid_node_counts(h, 32) == [1, 2, 4, 8, 16]


def test_a_model_divisible_by_three_allows_non_powers_of_two(tmp_path):
    h = read_model_header(write_model(tmp_path / "m.m", THREESY))
    assert valid_node_counts(h, 8) == [1, 2, 3, 4, 6, 8]


def test_supervisor_derives_counts_from_the_model(tmp_path):
    m = write_model(tmp_path / "m.m", LLAMA_3_2_3B)
    sup = Supervisor(Config(workers=[("w", 9998)] * 7, model=m))
    assert sup.valid_counts == [1, 2, 4, 8] and sup.node_counts_source == "model header"


def test_supervisor_falls_back_to_powers_of_two_without_a_header(tmp_path):
    sup = Supervisor(Config(workers=[("w", 9998)] * 5, model=str(tmp_path / "missing.m")))
    assert sup.valid_counts == [1, 2, 4] and "powers of two" in sup.node_counts_source


def test_node_counts_override_wins(tmp_path):
    m = write_model(tmp_path / "m.m", THREESY)
    sup = Supervisor(Config(workers=[("w", 9998)] * 7, model=m, node_counts=[8, 4, 2, 1, 99]))
    assert sup.valid_counts == [1, 2, 4, 8] and sup.node_counts_source == "override"


# ------------------------------------------------------------------ hysteresis

def test_first_probe_decides_outright():
    w = Worker("h", 9998, fail_after=2, ok_after=2)
    assert w.record(True, 1.0) is True and w.alive is True
    w2 = Worker("h", 9998, fail_after=2, ok_after=2)
    assert w2.record(False, 1.0) is False and w2.alive is False


def test_one_miss_is_not_death():
    w = Worker("h", 9998, fail_after=2, ok_after=2)
    w.record(True, 1.0)
    assert w.record(False, 2.0) is None and w.alive is True
    assert w.record(False, 3.0) is False and w.alive is False


def test_one_hit_is_not_recovery():
    w = Worker("h", 9998, fail_after=2, ok_after=2)
    w.record(False, 1.0)
    assert w.record(True, 2.0) is None and w.alive is False
    assert w.record(True, 3.0) is True and w.alive is True


def test_hit_resets_the_miss_counter():
    w = Worker("h", 9998, fail_after=3, ok_after=1)
    w.record(True, 1.0)
    w.record(False, 2.0)
    w.record(False, 3.0)
    w.record(True, 4.0)
    w.record(False, 5.0)
    assert w.alive is True


# --------------------------------------------------------------------- command

def test_workers_flag_is_last_and_space_separated():
    cfg = Config(workers=[("a", 9998), ("b", 9998)], extra_args="--max-seq-len 4096")
    sup = Supervisor(cfg)
    cmd = sup.build_command(sup.workers)
    assert cmd.index("--workers") == len(cmd) - 3
    assert cmd[-2:] == ["a:9998", "b:9998"]
    assert "--max-seq-len" in cmd and cmd.index("--max-seq-len") < cmd.index("--workers")


def test_root_alone_has_no_workers_flag():
    sup = Supervisor(Config(workers=[("a", 9998)]))
    assert "--workers" not in sup.build_command([])


def test_parse_workers_accepts_optional_ports():
    assert parse_workers("a, b:5000 ,,c", 9998) == [("a", 9998), ("b", 5000), ("c", 9998)]


def test_cli_defaults_round_trip():
    cfg = config_from_args(build_parser().parse_args(["--workers", "x,y", "--no-auto-rejoin"]))
    assert cfg.workers == [("x", 9998), ("y", 9998)]
    assert cfg.auto_rejoin is False and cfg.reset_workers is True


# ------------------------------------------------------------------ end to end

def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(pred, timeout=20.0, step=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(step)
    return False


class Lab:
    """A supervisor wired to fake_dllama_api.py and a file-driven probe."""

    def __init__(self, tmp_path, workers=("w1", "w2", "w3"), model="m.m"):
        self.dead_file = str(tmp_path / "dead")
        self.resets = []
        self.port = free_port()
        self.cfg = Config(
            workers=[(w, 9998) for w in workers],
            dllama_bin=FAKE, model=model, tokenizer="t.t",
            api_port=self.port, status_file=str(tmp_path / "status.json"),
            log_dir=str(tmp_path / "logs"),
            interval=0.2, fail_after=2, ok_after=1, rejoin_grace=0.6,
            settle=0.1, ready_timeout=15.0, launch_backoff=0.3,
            api_check_interval=0.5, api_stall_timeout=0,
        )
        self.sup = Supervisor(self.cfg, probe=self.probe, spawn=self.spawn,
                              reset_worker=self.reset)
        self.thread = threading.Thread(target=self.sup.run, daemon=True)

    def dead(self) -> set:
        if not os.path.exists(self.dead_file):
            return set()
        with open(self.dead_file) as f:
            return {l.strip() for l in f if l.strip()}

    def set_dead(self, *hosts) -> None:
        with open(self.dead_file, "w") as f:
            f.write("\n".join(hosts))

    def probe(self, host: str) -> bool:
        return host not in self.dead()

    def spawn(self, cmd, reason):
        env = {**os.environ, "FAKE_DEAD_FILE": self.dead_file,
               "FAKE_LOAD_SECONDS": "0.3", "FAKE_RETRY_SECONDS": "0.2"}
        return subprocess.Popen([sys.executable, FAKE, *cmd[1:]], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def reset(self, host: str) -> bool:
        self.resets.append(host)
        return True

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.sup.stop()
        self.thread.join(timeout=10)

    def active(self):
        return [w.host for w in self.sup.active]

    def status_file(self):
        with open(self.cfg.status_file) as f:
            return json.load(f)


@pytest.fixture
def lab(tmp_path):
    lab = Lab(tmp_path).start()
    yield lab
    lab.stop()


def test_full_ladder_down_and_back_up(lab):
    sup = lab.sup
    assert wait_for(lambda: sup.state == HEALTHY), sup.state_reason
    assert lab.active() == ["w1", "w2", "w3"]
    assert lab.status_file()["state"] == HEALTHY
    assert lab.status_file()["nodes_active"] == 4

    # pull w3: 4 nodes -> 2, keeping w1 by priority; survivors get reset
    gen = sup.generation
    lab.set_dead("w3")
    assert wait_for(lambda: sup.state == DEGRADED and sup.generation > gen), sup.state_reason
    assert lab.active() == ["w1"]
    assert sup.restarts >= 1
    assert {"w1", "w2"} <= set(lab.resets)
    assert any("worker lost: w3" in e["msg"] for e in sup.events)

    # plug w3 back in: after the grace period the set grows to 4 again
    gen = sup.generation
    lab.set_dead()
    assert wait_for(lambda: sup.state == HEALTHY and sup.generation > gen), sup.state_reason
    assert lab.active() == ["w1", "w2", "w3"]

    # lose two: 2 nodes on the one survivor, even though it is last in priority
    lab.set_dead("w1", "w2")
    assert wait_for(lambda: sup.state == DEGRADED and lab.active() == ["w3"]), lab.active()

    # lose everything: the root serves alone
    lab.set_dead("w1", "w2", "w3")
    assert wait_for(lambda: sup.state == DEGRADED and lab.active() == []), lab.active()
    assert lab.status_file()["nodes_active"] == 1

    # everything back
    lab.set_dead()
    assert wait_for(lambda: sup.state == HEALTHY and len(lab.active()) == 3), sup.state_reason


def test_n_workers_with_model_derived_counts(tmp_path):
    """Five workers on a model that allows 6 nodes: all five serve. Lose one and
    the model does not allow 5, so it drops to 4 nodes (three workers)."""
    m = write_model(tmp_path / "m.m", THREESY)
    lab = Lab(tmp_path, workers=("w1", "w2", "w3", "w4", "w5"), model=m).start()
    try:
        sup = lab.sup
        assert sup.valid_counts == [1, 2, 3, 4, 6]
        assert wait_for(lambda: sup.state == HEALTHY), sup.state_reason
        assert lab.active() == ["w1", "w2", "w3", "w4", "w5"]
        assert lab.status_file()["nodes_active"] == 6

        lab.set_dead("w4")
        assert wait_for(lambda: sup.state == DEGRADED and lab.active() == ["w1", "w2", "w3"]), lab.active()

        lab.set_dead("w4", "w5", "w1")   # two alive -> 3 nodes, a set powers of two could not use
        assert wait_for(lambda: lab.active() == ["w2", "w3"]), lab.active()
        assert lab.status_file()["nodes_active"] == 3
    finally:
        lab.stop()


def test_root_crash_is_relaunched(lab):
    sup = lab.sup
    assert wait_for(lambda: sup.state == HEALTHY)
    gen = sup.generation
    sup.proc.kill()
    assert wait_for(lambda: sup.generation > gen and sup.state == HEALTHY), sup.state_reason
    assert any("root exited" in e["msg"] for e in sup.events)


def test_worker_dying_during_load_shrinks_the_set(tmp_path):
    lab = Lab(tmp_path)
    lab.set_dead("w2")          # dead before the first launch is even attempted
    lab.start()
    try:
        # w2 dead -> 3 alive+root = 3 -> largest 2^n = 2 -> one worker, w1
        assert wait_for(lambda: lab.sup.state == DEGRADED), lab.sup.state_reason
        assert lab.active() == ["w1"]
    finally:
        lab.stop()


def test_status_http_and_manual_restart(lab):
    sup = lab.sup
    assert wait_for(lambda: sup.state == HEALTHY)
    srv = serve_status(sup, "127.0.0.1", 0)
    try:
        port = srv.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as r:
            snap = json.loads(r.read())
        assert snap["state"] == HEALTHY and snap["active_workers"] == ["w1:9998", "w2:9998", "w3:9998"]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as r:
            assert r.status == 200

        gen = sup.generation
        req = urllib.request.Request(f"http://127.0.0.1:{port}/restart", method="POST")
        with urllib.request.urlopen(req, timeout=2) as r:
            assert r.status == 202
        assert wait_for(lambda: sup.generation > gen and sup.state == HEALTHY), sup.state_reason
    finally:
        srv.shutdown()


def test_stop_kills_the_root(tmp_path):
    lab = Lab(tmp_path).start()
    assert wait_for(lambda: lab.sup.state == HEALTHY)
    proc = lab.sup.proc
    lab.stop()
    assert proc.poll() is not None
    assert lab.sup.state == DOWN


def test_state_names_are_the_router_contract():
    assert {HEALTHY, DEGRADED, RESTARTING, DOWN} == {"healthy", "degraded", "restarting", "down"}
