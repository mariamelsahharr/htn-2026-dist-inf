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
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from supervisor import (
    DEGRADED,
    DOWN,
    HEALTHY,
    MODEL_MAGIC,
    RESTARTING,
    Config,
    LogTail,
    Supervisor,
    Worker,
    build_parser,
    cap_file,
    choose_workers,
    config_from_args,
    parse_workers,
    post_allowed,
    powers_of_two,
    read_model_header,
    sd_notify,
    serve_status,
    valid_node_counts,
    worker_process_present,
)

HERE = Path(__file__).resolve().parent
FAKE = str(HERE / "fake_dllama_api.py")

# Header key ids from distributed-llama src/llm.hpp
KEY = {
    "dim": 2,
    "hidden_dim": 3,
    "n_layers": 4,
    "n_heads": 5,
    "n_kv_heads": 6,
    "vocab_size": 9,
    "seq_len": 10,
    "weight_float_type": 13,
    "head_dim": 19,
    "moe_hidden_dim": 21,
}

LLAMA_3_2_3B = {
    "dim": 3072,
    "hidden_dim": 8192,
    "n_layers": 28,
    "n_heads": 24,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "seq_len": 8192,
    "weight_float_type": 2,
    "head_dim": 128,
}
QWEN3_0_6B = {
    "dim": 1024,
    "hidden_dim": 3072,
    "n_layers": 28,
    "n_heads": 16,
    "n_kv_heads": 8,
    "vocab_size": 151936,
    "seq_len": 4096,
    "weight_float_type": 2,
    "head_dim": 128,
}
# Synthetic model whose dims all divide by 3: valid counts are not powers of two.
THREESY = {
    "dim": 3072,
    "hidden_dim": 6144,
    "n_layers": 4,
    "n_heads": 24,
    "n_kv_heads": 6,
    "vocab_size": 98304,
    "seq_len": 1024,
    "weight_float_type": 2,
    "head_dim": 128,
}


def write_model(path, params):
    """Write a .m file with only the header, the way converter/writer.py does."""
    import struct

    data = b"".join(struct.pack("<ii", KEY[k], v) for k, v in params.items())
    with Path(path).open("wb") as f:
        f.write(struct.pack("<ii", MODEL_MAGIC, 8 + len(data)))
        f.write(data)
        f.write(b"\0" * 64)  # a few bytes of "weights"
    return str(path)


# --------------------------------------------------------------- set selection


def test_three_alive_workers_make_a_four_node_set():
    assert choose_workers(["a", "b", "c"]) == ["a", "b", "c"]


def test_two_alive_workers_drop_to_two_nodes_keeping_priority():
    assert choose_workers(["a", "b"]) == ["a"]


def test_one_alive_worker_is_a_two_node_set():
    assert choose_workers(["c"]) == ["c"]


def test_no_alive_workers_means_root_alone():
    assert choose_workers([]) == []


def test_seven_workers_make_eight_nodes_and_six_make_four():
    seven = choose_workers(list("abcdefg"))
    assert seven is not None and len(seven) == 7
    assert choose_workers(list("abcdef")) == ["a", "b", "c"]


def test_valid_counts_drive_the_choice_not_powers_of_two():
    counts = [1, 2, 3, 4, 6, 8]
    assert choose_workers(list("ab"), counts) == ["a", "b"]  # 3 nodes
    assert choose_workers(list("abcde"), counts) == list("abcde")  # 6 nodes
    assert choose_workers(list("abcd"), counts) == list("abc")  # 5 not valid -> 4
    assert choose_workers(list("abcdef"), counts) == list("abcde")  # 7 not valid -> 6


def test_min_nodes_floor_stands_down_instead_of_shrinking():
    assert choose_workers(list("abc"), [1, 2, 4, 8], min_nodes=4) == ["a", "b", "c"]
    assert choose_workers(list("ab"), [1, 2, 4, 8], min_nodes=4) is None  # 3 nodes < floor
    assert choose_workers(list("ab"), [1, 2, 4, 8], min_nodes=2) == ["a"]


def test_explicit_counts_never_fall_through_to_one_node():
    """--node-counts 4,8 with three survivors must not launch on a single Pi."""
    assert choose_workers(list("ab"), [4, 8]) is None
    assert choose_workers(list("abc"), [4, 8]) == ["a", "b", "c"]
    assert choose_workers([], [4, 8]) is None


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
    assert valid_node_counts(h, 32) == [1, 2, 4, 8]  # 16 fails on 24 heads


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


def test_override_with_no_reachable_count_never_collapses_to_one():
    sup = Supervisor(Config(workers=[("w", 9998)] * 3, model="missing.m", node_counts=[8]))
    assert sup.valid_counts == [] and sup.choose(sup.workers) is None


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

    def __init__(
        self, tmp_path, workers=("w1", "w2", "w3"), model="m.m", min_nodes=1, node_counts=None, load_seconds=0.3
    ):
        self.load_seconds = load_seconds
        self.dead_file = str(tmp_path / "dead")
        self.resets = []
        self.port = free_port()
        self.cfg = Config(
            workers=[(w, 9998) for w in workers],
            dllama_bin=FAKE,
            model=model,
            tokenizer="t.t",
            min_nodes=min_nodes,
            node_counts=node_counts,
            api_port=self.port,
            status_file=str(tmp_path / "status.json"),
            log_dir=str(tmp_path / "logs"),
            interval=0.2,
            fail_after=2,
            ok_after=1,
            rejoin_grace=0.6,
            settle=0.1,
            ready_timeout=15.0,
            launch_backoff=0.3,
            api_check_interval=0.5,
            api_stall_timeout=0,
        )
        self.sup = Supervisor(
            self.cfg, probe=self.probe, spawn=self.spawn, reset_worker=self.reset, telemetry=self.telemetry
        )
        self.thread = threading.Thread(target=self.sup.run, daemon=True)

    def dead(self) -> set:
        if not Path(self.dead_file).exists():
            return set()
        with Path(self.dead_file).open() as f:
            return {line.strip() for line in f if line.strip()}

    def set_dead(self, *hosts) -> None:
        with Path(self.dead_file).open("w") as f:
            f.write("\n".join(hosts))

    def probe(self, host: str) -> bool:
        return host not in self.dead()

    def telemetry(self, host: str) -> dict | None:
        return None if host in self.dead() else {"temp_c": 50.0 + len(host), "throttled": "0x0", "flags": []}

    def spawn(self, cmd, reason):
        env = {
            **os.environ,
            "FAKE_DEAD_FILE": self.dead_file,
            "FAKE_LOAD_SECONDS": str(self.load_seconds),
            "FAKE_RETRY_SECONDS": "0.2",
        }
        return subprocess.Popen(
            [sys.executable, FAKE, *cmd[1:]], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

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
        with Path(self.cfg.status_file).open() as f:
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

        lab.set_dead("w4", "w5", "w1")  # two alive -> 3 nodes, a set powers of two could not use
        assert wait_for(lambda: lab.active() == ["w2", "w3"]), lab.active()
        assert lab.status_file()["nodes_active"] == 3
    finally:
        lab.stop()


def test_below_min_nodes_goes_down_and_recovers(tmp_path):
    """RAM floor: with min_nodes=4 a lost worker means down, not a 2-node launch."""
    lab = Lab(tmp_path, min_nodes=4).start()
    try:
        sup = lab.sup
        assert wait_for(lambda: sup.state == HEALTHY), sup.state_reason
        gen = sup.generation
        lab.set_dead("w2")
        assert wait_for(lambda: sup.state == DOWN), sup.state_reason
        assert sup.proc is None and lab.active() == []
        assert "at least 4" in sup.state_reason
        assert lab.status_file()["nodes_active"] == 0
        assert sup.generation == gen  # nothing was launched on a smaller set
        lab.set_dead()
        assert wait_for(lambda: sup.state == HEALTHY and sup.generation > gen), sup.state_reason
    finally:
        lab.stop()


def test_explicit_counts_without_a_fit_stand_down(tmp_path):
    lab = Lab(tmp_path, node_counts=[4, 8])
    lab.set_dead("w3")  # three nodes reachable, only 4 and 8 allowed
    lab.start()
    try:
        assert wait_for(lambda: lab.sup.state == DOWN), lab.sup.state_reason
        assert lab.sup.generation == 0 and lab.sup.proc is None
    finally:
        lab.stop()


def test_root_crash_is_relaunched(lab):
    sup = lab.sup
    assert wait_for(lambda: sup.state == HEALTHY)
    gen = sup.generation
    sup.proc.kill()
    assert wait_for(lambda: sup.generation > gen and sup.state == HEALTHY), sup.state_reason
    assert any("root exited" in e["msg"] for e in sup.events)


def test_worker_dead_before_first_launch_shrinks_the_set(tmp_path):
    lab = Lab(tmp_path)
    lab.set_dead("w2")  # dead before the first launch is even attempted
    lab.start()
    try:
        # w2 dead -> 3 alive+root = 3 -> largest 2^n = 2 -> one worker, w1
        assert wait_for(lambda: lab.sup.state == DEGRADED), lab.sup.state_reason
        assert lab.active() == ["w1"]
    finally:
        lab.stop()


def test_worker_dying_during_load_is_detected_by_wait_ready(tmp_path):
    """The root is mid-load (fake sleeps 3s) when a set member dies: wait_ready must
    notice, abandon that launch, and relaunch on the survivors."""
    lab = Lab(tmp_path, load_seconds=3.0).start()
    try:
        sup = lab.sup
        assert wait_for(lambda: sup.state == RESTARTING and sup.proc is not None), sup.state_reason
        lab.set_dead("w2")
        assert wait_for(lambda: sup.state == DEGRADED and lab.active() == ["w1"], timeout=30), sup.state_reason
        assert any("worker lost during load: w2" in e["msg"] for e in sup.events)
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


def test_snapshot_matches_the_published_example(tmp_path):
    """status.example.json is what the router and metrics tests parse; keep it honest."""
    example = json.loads((HERE / "status.example.json").read_text())
    snap = Supervisor(Config(workers=[("a", 9998), ("b", 9998), ("c", 9998)], model="missing.m")).snapshot()
    assert set(snap) == set(example)
    assert set(snap["root"]) == set(example["root"])
    assert set(snap["workers"][0]) == set(example["workers"][0])


def test_telemetry_rides_along_with_liveness(lab):
    wait_for(lambda: lab.sup.state == HEALTHY, 20)
    doc = lab.sup.last_snapshot()
    by_host = {w["host"]: w for w in doc["workers"]}
    assert by_host["w1"]["telemetry"]["temp_c"] == 52.0 and doc["root"]["telemetry"]["temp_c"] == 59.0
    lab.set_dead("w3")
    wait_for(lambda: lab.sup.last_snapshot()["workers"][2]["alive"] is False, 20)
    assert lab.sup.last_snapshot()["workers"][2]["telemetry"] is None, (
        "a dead worker has no telemetry, not stale telemetry"
    )


def test_telemetry_off_leaves_nulls(tmp_path):
    cfg = Config(workers=[("w1", 9998)], telemetry_port=0, model="m.m", tokenizer="t.t")
    sup = Supervisor(cfg, probe=lambda h: True, telemetry=lambda h: {"temp_c": 1})
    sup.probe_all()
    assert sup.workers[0].telemetry is None and sup.root_telemetry is None


# ------------------------------------------------------ crash loop, resets, liveness (no fake root)


class FakeProc:
    """A Popen stand-in: alive until returncode is set."""

    def __init__(self, returncode=None):
        self.returncode = returncode
        self.pid = 4242

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class Bench:
    """A supervisor with injected process, probes and resets; nothing sleeps for long."""

    def __init__(self, tmp_path, spawn=None, reset_ok=None, telemetry=None, api_ok=True, **cfg):
        self.resets = []
        self.procs = []
        self.reset_ok = reset_ok or (lambda host: True)
        self.telemetry = telemetry or (lambda host: None)
        self.cfg = Config(
            workers=[("w1", 9998), ("w2", 9998), ("w3", 9998)],
            model="missing.m",
            tokenizer="t.t",
            log_dir=str(tmp_path / "logs"),
            status_file=str(tmp_path / "status.json"),
            interval=0.05,
            settle=0.0,
            launch_backoff=5.0,
            fail_after=2,
            ok_after=1,
            **cfg,
        )
        self.sup = Supervisor(
            self.cfg,
            probe=lambda h: True,
            spawn=spawn or self.spawn,
            reset_worker=self.reset,
            api_check=lambda: api_ok,
            telemetry=self.telemetry,
        )

    def spawn(self, cmd, reason):
        proc = FakeProc()
        self.procs.append(proc)
        return proc

    def reset(self, host):
        self.resets.append(host)
        return self.reset_ok(host)

    def status_file(self):
        return json.loads(Path(self.cfg.status_file).read_text())


def crash_at_spawn(cmd, reason):
    return FakeProc(returncode=1)


def test_launch_backoff_doubles_per_failure_and_caps(tmp_path):
    b = Bench(tmp_path, spawn=crash_at_spawn, max_launch_backoff=30.0)
    sup = b.sup
    for failures, delay in ((1, 10.0), (2, 20.0), (3, 30.0), (4, 30.0)):
        t0 = time.time()
        assert sup.launch(sup.workers, "test") is False
        assert sup.launch_failures == failures
        assert abs(sup._next_launch_at - t0 - delay) < 0.5, f"failure {failures}: expected {delay}s backoff"
        assert f"next attempt in {delay:.0f}s" in sup.state_reason
    assert b.resets == [], "a root that died within quick_exit of its spawn never reached the workers: no reset"


def test_root_killed_for_memory_is_named_in_the_reason(tmp_path):
    b = Bench(tmp_path, spawn=lambda cmd, reason: FakeProc(returncode=-9))  # the OOM killer's SIGKILL
    sup = b.sup
    sup.root_log_path.write_text("=== gen=1 startup\n💡 Loading weights from part 3\n")
    assert sup.launch(sup.workers, "test") is False
    assert "likely out of memory" in sup.state_reason and "Loading weights from part 3" in sup.state_reason
    assert sup.state == RESTARTING and "next attempt in" in sup.state_reason


def test_bad_alloc_in_the_log_is_named_out_of_memory(tmp_path):
    b = Bench(tmp_path, spawn=lambda cmd, reason: FakeProc(returncode=-6))
    sup = b.sup
    sup.root_log_path.write_text("terminate called after throwing an instance of 'std::bad_alloc'\n")
    sup.launch(sup.workers, "test")
    assert "likely out of memory" in sup.state_reason


def test_a_plain_exit_is_not_called_out_of_memory(tmp_path):
    b = Bench(tmp_path, spawn=crash_at_spawn)
    sup = b.sup
    sup.root_log_path.write_text("Unknown option --frobnicate\n")
    sup.launch(sup.workers, "test")
    assert "out of memory" not in sup.state_reason
    assert "last log line: 'Unknown option --frobnicate'" in sup.state_reason
    sup.root_log_path.unlink()
    sup.launch(sup.workers, "test")
    assert "no output in dllama-api.log" in sup.state_reason


def test_three_launch_failures_report_down(tmp_path):
    sup = Bench(tmp_path, spawn=crash_at_spawn).sup
    sup.launch(sup.workers, "one")
    sup.launch(sup.workers, "two")
    assert sup.state == RESTARTING
    sup.launch(sup.workers, "three")
    assert sup.state == DOWN and sup.launch_failures == 3
    assert Path(sup.cfg.status_file).exists() and json.loads(Path(sup.cfg.status_file).read_text())["state"] == DOWN


def test_crash_soon_after_ready_counts_as_a_launch_failure(tmp_path):
    b = Bench(tmp_path, stable_after=60.0, quick_exit=2.0)
    sup = b.sup
    sup.tick()
    assert sup.state == HEALTHY and sup.launch_failures == 0
    now = time.time()
    sup.launched_at, sup.ready_at = now - 10, now - 5  # ran long enough to have connected the workers
    b.procs[-1].returncode = 139
    sup.tick()
    assert sup.launch_failures == 1 and sup.state == RESTARTING
    assert sup._next_launch_at - now > 9.0, "exponential backoff, not the 0 s settle"
    assert b.resets == ["w1", "w2", "w3"], "the workers were attached to that root: reset them"
    assert b.status_file()["state"] == RESTARTING, "kill_root/set_state publish immediately"
    assert "within 60s of ready" in sup.state_reason

    sup._next_launch_at = 0.0
    sup.tick()  # relaunch
    assert sup.state == HEALTHY and sup.launch_failures == 1, "a fresh ready does not forgive the crash yet"
    sup.ready_at = time.time() - 61
    sup.tick()
    assert sup.launch_failures == 0, "stable for stable_after: the crash loop is over"


def test_instant_exit_after_ready_skips_the_worker_reset(tmp_path):
    b = Bench(tmp_path)
    sup = b.sup
    sup.tick()
    assert sup.state == HEALTHY
    b.procs[-1].returncode = 1  # launched_at is "now": the root never got to the workers
    sup.tick()
    assert b.resets == [] and sup.launch_failures == 1


def test_failed_reset_benches_the_worker_after_one_retry(tmp_path):
    b = Bench(tmp_path, reset_ok=lambda host: host != "w2")
    sup = b.sup
    sup.tick()
    assert sup.state == HEALTHY and [w.host for w in sup.active] == ["w1", "w2", "w3"]
    sup.restart("drill")
    assert b.resets.count("w2") == 2 and b.resets.count("w1") == 1, "one retry for the failure, none for successes"
    assert sup._reset_failed == {"w2"}
    assert [w.host for w in sup.active] == ["w1"], "w2 is left out: 2 usable workers -> a 2-node set on w1"
    assert sup.state == DEGRADED
    by_host = {w["host"]: w for w in sup.snapshot()["workers"]}
    assert by_host["w2"]["reset_failed"] is True and by_host["w1"]["reset_failed"] is False
    assert any("reset failed twice" in e["msg"] and "w2" in e["msg"] for e in sup.events)

    # while benched it must not trigger a rejoin restart loop
    gen = sup.generation
    for _ in range(3):
        sup.tick()
    assert sup.generation == gen and sup._grow_since is None

    # the periodic retry succeeds: eligible again, and the set grows after the grace period
    b.reset_ok = lambda host: True
    sup._retry_failed_resets(time.time() + sup.cfg.reset_retry_interval + 1)
    assert sup._reset_failed == set()
    sup.tick()
    assert sup._grow_since is not None
    sup._grow_since = time.time() - sup.cfg.rejoin_grace - 1
    sup.tick()
    assert [w.host for w in sup.active] == ["w1", "w2", "w3"] and sup.state == HEALTHY


def test_a_reachable_pi_without_a_worker_process_is_dead(tmp_path):
    docs = {}
    b = Bench(tmp_path, telemetry=lambda host: docs.get(host, {"worker_listening": True, "worker_connections": 0}))
    sup = b.sup
    sup.probe_all()
    assert [w.alive for w in sup.workers] == [True, True, True]
    docs["w2"] = {"temp_c": 50.0, "worker_listening": False, "worker_connections": 0, "worker_unit": "failed"}
    sup.probe_all()
    assert sup.workers[1].alive is True, "first miss: hysteresis (fail_after=2)"
    sup.probe_all()
    assert sup.workers[1].alive is False and sup.workers[0].alive is True
    assert sup.workers[1].telemetry is None
    assert any("w2 has no dllama worker process" in e["msg"] and "failed" in e["msg"] for e in sup.events)
    docs["w2"] = {"worker_listening": False, "worker_connections": 1}  # attached to a root: alive
    sup.probe_all()
    assert sup.workers[1].alive is True


def test_worker_process_present_semantics():
    assert worker_process_present(None) is True, "no telemetry: the ping decides"
    assert worker_process_present({"temp_c": 1}) is True, "older agent without the fields"
    assert worker_process_present({"worker_listening": None, "worker_connections": None}) is True
    assert worker_process_present({"worker_listening": True, "worker_connections": 0}) is True
    assert worker_process_present({"worker_listening": False, "worker_connections": 2}) is True
    assert worker_process_present({"worker_listening": False, "worker_connections": 0}) is False


def test_set_state_publishes_immediately(tmp_path):
    sup = Bench(tmp_path).sup
    sup.set_state(DOWN, "unit test")
    assert json.loads(Path(sup.cfg.status_file).read_text())["state"] == DOWN
    assert sup.last_snapshot()["reason"] == "unit test"


def test_wait_ready_gives_up_after_consecutive_connect_retries(tmp_path):
    b = Bench(tmp_path, api_ok=False, max_connect_retries=3)
    sup = b.sup
    log = sup.root_log_path
    log.write_text("=== old launch\n🚨 Connection error: Cannot connect to w2:9998\n")
    sup.proc = FakeProc()  # ty: ignore[invalid-assignment]
    sup._log_tail = LogTail(log)  # what launch() does right after the spawn: old lines do not count
    with log.open("a") as fh:
        for _ in range(3):
            fh.write(
                "🚨 Connection error: Cannot connect to w2:9998 (Connection refused)\n🔄 Retrying in 3 seconds...\n"
            )
    assert sup.wait_ready() == "root cannot reach its workers (3 consecutive connect retries)"


def test_connect_retry_count_resets_on_other_output(tmp_path):
    sup = Bench(tmp_path).sup
    log = sup.root_log_path
    log.write_text("")
    sup._log_tail = LogTail(log)
    with log.open("a") as fh:
        fh.write("🚨 Connection error: Cannot connect to w2:9998\n🔄 Retrying in 3 seconds...\n")
        fh.write("🚨 Connection error: Cannot connect to w2:9998\n")
    assert sup._count_connect_retries(0) == 2
    with log.open("a") as fh:
        fh.write("🔄 Retrying in 3 seconds...\n💡 Loaded 1024 kB\n🚨 Connection error: x\npartial line without newline")
    assert sup._count_connect_retries(2) == 1, "a real log line resets the run; the partial line waits"


def test_log_tail_survives_truncation(tmp_path):
    p = tmp_path / "log"
    p.write_text("a\nb\n")
    tail = LogTail(p)
    assert tail.read_new() == []
    p.write_text("a\nb\nc\n")
    assert tail.read_new() == ["c"]
    p.write_text("x\n")
    assert tail.read_new() == ["x"], "shorter file = truncated: start over"
    p.unlink()
    assert tail.read_new() == []


def test_cap_file_keeps_the_newest_half(tmp_path):
    p = tmp_path / "dllama-api.log"
    p.write_bytes(b"".join(f"line {i:05d}\n".encode() for i in range(1000)))
    assert cap_file(p, 100_000) is False
    assert cap_file(p, 2_000) is True
    text = p.read_text()
    assert text.startswith("=== ") and "capped at 2000 bytes" in text
    assert "line 00999" in text and "line 00010" not in text and len(text) < 2_200
    with p.open("ab") as fh:  # the child's O_APPEND handle keeps appending after the cap
        fh.write(b"after\n")
    assert p.read_text().endswith("line 00999\nafter\n")
    assert cap_file(tmp_path / "missing", 10) is False


def test_events_file_rotates_by_size(tmp_path):
    sup = Bench(tmp_path, events_max_bytes=2_000).sup
    for i in range(200):
        sup.event(f"event number {i} with some padding to fill the file quickly")
    logs = tmp_path / "logs"
    assert (logs / "supervisor-events.jsonl").exists() and (logs / "supervisor-events.jsonl.1").exists()
    assert not (logs / "supervisor-events.jsonl.3").exists()
    line = (logs / "supervisor-events.jsonl.1").read_text().splitlines()[0]
    assert set(json.loads(line)) == {"t", "msg"}


def test_sd_notify_is_a_noop_without_systemd(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert sd_notify("READY=1") is False
    monkeypatch.setenv("NOTIFY_SOCKET", "/nonexistent/notify")
    assert sd_notify("READY=1") is False


def test_sd_notify_sends_a_datagram_to_notify_socket(monkeypatch):
    import socket
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "notify")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        srv.bind(path)
        srv.settimeout(2)
        monkeypatch.setenv("NOTIFY_SOCKET", path)
        assert sd_notify("READY=1") is True
        assert srv.recv(64) == b"READY=1"
        srv.close()


def test_post_restart_needs_loopback_or_token():
    assert post_allowed("127.0.0.1", {}, None) is True
    assert post_allowed("::1", {}, None) is True
    assert post_allowed("192.168.50.20", {}, None) is False, "no token configured: remote POST is refused"
    assert post_allowed("192.168.50.20", {"Authorization": "Bearer s3cret"}, "s3cret") is True
    assert post_allowed("192.168.50.20", {"X-Supervisor-Token": "s3cret"}, "s3cret") is True
    assert post_allowed("192.168.50.20", {"Authorization": "Bearer wrong"}, "s3cret") is False
    assert post_allowed("192.168.50.20", {"Authorization": "Bearer "}, "s3cret") is False


def test_post_responses_carry_no_cors_header(lab):
    sup = lab.sup
    assert wait_for(lambda: sup.state == HEALTHY)
    srv = serve_status(sup, "127.0.0.1", 0)
    try:
        port = srv.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as r:
            assert r.headers.get("Access-Control-Allow-Origin") == "*"
        req = urllib.request.Request(f"http://127.0.0.1:{port}/restart", method="POST")
        with urllib.request.urlopen(req, timeout=2) as r:
            assert r.status == 202 and r.headers.get("Access-Control-Allow-Origin") is None
    finally:
        srv.shutdown()
