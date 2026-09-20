"""
test_attest.py - encoders, status/log parsing and the sidecar loop against a fake chain. Run: pytest -q
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from solders.keypair import Keypair
from solders.pubkey import Pubkey

sys.path.insert(0, str(Path(__file__).parent))
import attest

EXAMPLE = json.loads((Path(__file__).parent.parent / "cluster" / "supervisor" / "status.example.json").read_text())
PROGRAM = Pubkey.new_unique()


# ----- borsh layout: must match the Rust crate (see instruction_layout_is_stable there) -----


def ix(name, **fields):
    return attest.ClusterInstruction.build(getattr(attest.ClusterInstruction.enum, name)(**fields))


def test_instruction_encoding_matches_rust_layout():
    assert ix("SetWorkerSet", state=2, active=["a"]) == bytes([2, 2, 1, 0, 0, 0, 1, 0, 0, 0, ord("a")])
    assert ix("Initialize", model_hash=b"\x07" * 32) == b"\x00" + b"\x07" * 32
    assert ix("RegisterNode", host="192.168.50.11") == b"\x01" + (13).to_bytes(4, "little") + b"192.168.50.11"
    job = ix("CommitJob", job_id=b"\x01" * 16, served_by="cluster", result_hash=b"\x02" * 32)
    assert job == b"\x03" + b"\x01" * 16 + (7).to_bytes(4, "little") + b"cluster" + b"\x02" * 32


def test_cluster_account_round_trip():
    authority = Pubkey.new_unique()
    raw = (
        bytes(authority)
        + b"\x09" * 32
        + (5).to_bytes(8, "little")
        + b"\x02"
        + (2).to_bytes(4, "little")
        + (13).to_bytes(4, "little")
        + b"192.168.50.11"
        + b"\x01"
        + (13).to_bytes(4, "little")
        + b"192.168.50.14"
        + b"\x00"
        + (10).to_bytes(8, "little")
        + (7).to_bytes(8, "little")
        + b"\xaa" * 16
    )
    padded = raw + bytes(attest_space := 1024 - len(raw))
    assert attest_space > 0
    c = attest.decode_cluster(padded)
    assert c["authority"] == str(authority)
    assert c["epoch"] == 5 and c["state"] == "degraded"
    assert c["nodes"] == [{"host": "192.168.50.11", "active": True}, {"host": "192.168.50.14", "active": False}]
    assert (c["jobs_total"], c["jobs_local"], c["last_job"]) == (10, 7, "aa" * 16)


def test_pdas_are_deterministic_and_distinct():
    auth = Pubkey.new_unique()
    cluster = attest.cluster_pda(PROGRAM, auth)
    assert cluster == attest.cluster_pda(PROGRAM, auth)
    assert attest.job_pda(PROGRAM, cluster, b"\x01" * 16) != attest.job_pda(PROGRAM, cluster, b"\x02" * 16)


# ----- what gets derived from the supervisor and router --------------------------------


def test_worker_set_from_status_contract():
    state, hosts, active = attest.worker_set(EXAMPLE)
    assert state == attest.STATE_CODES["degraded"]
    assert hosts == ("192.168.50.11", "192.168.50.12", "192.168.50.14")
    assert active == ("192.168.50.11",)  # port stripped from active_workers
    assert attest.worker_set({}) == (0, (), ())


def test_model_hash_depends_on_header_and_file_name():
    h = attest.model_hash(EXAMPLE)
    assert len(h) == 32 and h == attest.model_hash(json.loads(json.dumps(EXAMPLE)))
    other = {**EXAMPLE, "root": {**EXAMPLE["root"], "model": "/x/qwen3.m"}}
    assert attest.model_hash(other) != h


def test_only_finished_answers_become_jobs():
    digest = hashlib.sha256(b"hi").hexdigest()
    rid = "0123456789abcdef0123456789abcdef"
    job = attest.job_from_record({"request_id": rid, "served_by": "baseten", "result_sha256": digest})
    assert job == (bytes.fromhex(rid), "baseten", bytes.fromhex(digest))
    assert attest.job_from_record({"request_id": rid, "served_by": "none", "error": "all failed"}) is None
    assert attest.job_from_record({"request_id": rid, "served_by": "cluster"}) is None  # died mid-stream
    assert attest.job_from_record({"request_id": "zz", "served_by": "cluster", "result_sha256": digest}) is None


def test_log_tail_survives_partial_lines_and_truncation(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"a":1}\n')
    tail = attest.LogTail(str(p))  # starts at the end: old records are not replayed
    assert tail.read() == []
    with p.open("a") as fh:
        fh.write('{"a":2}\n{"a":3')
    assert tail.read() == [{"a": 2}]
    with p.open("a") as fh:
        fh.write("}\nnot json\n")
    assert tail.read() == [{"a": 3}]
    p.write_text('{"a":9}\n')  # truncated and rewritten
    assert tail.read() == [{"a": 9}]
    assert attest.LogTail(str(p), from_start=True).read() == [{"a": 9}]


def test_explorer_links_follow_the_rpc():
    assert attest.explorer_url(attest.DEVNET, "tx", "sig") == "https://explorer.solana.com/tx/sig?cluster=devnet"
    assert "customUrl=http://127.0.0.1:8899" in attest.explorer_url("http://127.0.0.1:8899", "address", "x")


# ----- the loop, against a fake chain -----------------------------------------------------


class FakeStatus:
    def __init__(self, doc):
        self.doc = doc

    def __call__(self):
        return self.doc


class FakeChain:
    """Records instruction bytes; account_data reflects whether Initialize ran."""

    def __init__(self):
        self.payer, self.program_id, self.rpc_url = Keypair(), PROGRAM, attest.DEVNET
        self.sent: list[bytes] = []
        self.initialized = False
        self.fail_next: str | None = None

    def account_data(self, pubkey):
        return b"\x00" * 1024 if self.initialized else None

    def send(self, data, accounts, timeout=0):
        if self.fail_next:
            err, self.fail_next = self.fail_next, None
            raise attest.ChainError(err)
        self.sent.append(data)
        if data[0] == 0:
            self.initialized = True
        return f"sig{len(self.sent)}"


@pytest.fixture
def att(tmp_path):
    log = tmp_path / "decisions.jsonl"
    log.write_text("")
    chain = FakeChain()
    src = FakeStatus(EXAMPLE)  # what the fake supervisor says right now
    a = attest.Attestor(chain, src, attest.LogTail(str(log)).read, str(tmp_path / "out.jsonl"))
    return a, chain, log, tmp_path / "out.jsonl", src


def test_first_step_initializes_registers_and_sets_the_worker_set(att):
    a, chain, _, out, _src = att
    a.step()
    kinds = [d[0] for d in chain.sent]
    assert kinds == [0, 1, 1, 1, 2]  # initialize, three registers, one worker set
    assert chain.sent[-1] == ix("SetWorkerSet", state=2, active=["192.168.50.11"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["kind"] for r in rows] == [
        "initialize",
        "register_node",
        "register_node",
        "register_node",
        "set_worker_set",
    ]
    assert rows[-1]["explorer"].startswith("https://explorer.solana.com/tx/sig5?cluster=devnet")


def test_unchanged_status_sends_nothing(att):
    a, chain, _, _, _src = att
    a.step()
    n = len(chain.sent)
    a.step()
    assert len(chain.sent) == n


def test_worker_returning_bumps_the_set_once(att):
    a, chain, _, _, src = att
    a.step()
    src.doc = {
        **EXAMPLE,
        "state": "healthy",
        "active_workers": ["192.168.50.11:9998", "192.168.50.12:9998", "192.168.50.14:9998"],
    }
    a.step()
    assert chain.sent[-1] == ix("SetWorkerSet", state=1, active=["192.168.50.11", "192.168.50.12", "192.168.50.14"])
    a.step()
    assert chain.sent[-1][0] == 2 and len(chain.sent) == 6


def test_jobs_are_committed_and_retried_after_rpc_errors(att):
    a, chain, log, out, _src = att
    a.step()
    digest = hashlib.sha256(b"answer").hexdigest()
    with log.open("a") as fh:
        fh.write(json.dumps({"request_id": "ab" * 16, "served_by": "cluster", "result_sha256": digest}) + "\n")
        fh.write(json.dumps({"request_id": "cd" * 16, "served_by": "none", "error": "x"}) + "\n")
    chain.fail_next = "getLatestBlockhash: timeout"
    with pytest.raises(attest.ChainError):
        a.step()
    assert a.pending_jobs and chain.sent[-1][0] == 2  # nothing committed yet, job kept
    a.step()
    assert chain.sent[-1] == ix(
        "CommitJob", job_id=bytes.fromhex("ab" * 16), served_by="cluster", result_hash=bytes.fromhex(digest)
    )
    assert not a.pending_jobs
    assert json.loads(out.read_text().splitlines()[-1])["kind"] == "commit_job"


def test_duplicate_job_is_dropped_not_retried(att):
    a, chain, log, _, _src = att
    a.step()
    with log.open("a") as fh:
        fh.write(json.dumps({"request_id": "ef" * 16, "served_by": "openai", "result_sha256": "00" * 32}) + "\n")
    chain.fail_next = "transaction failed: {'InstructionError': [0, {'Custom': 0}]} custom program error"
    a.step()
    assert not a.pending_jobs and chain.sent[-1][0] == 2


def test_status_outage_still_commits_jobs(att):
    a, chain, log, _, src = att
    a.step()
    src.doc = None
    with log.open("a") as fh:
        fh.write(json.dumps({"request_id": "12" * 16, "served_by": "gemini", "result_sha256": "11" * 32}) + "\n")
    a.step()
    assert chain.sent[-1][0] == 3


def test_record_queue_feeds_jobs_in_process():
    q = attest.RecordQueue()
    chain = FakeChain()
    a = attest.Attestor(chain, lambda: EXAMPLE, q.read, os.devnull)
    a.step()
    q.push({"request_id": "ab" * 16, "served_by": "cluster", "result_sha256": "00" * 32})
    q.push({"request_id": "cd" * 16, "served_by": "none"})
    a.step()
    assert chain.sent[-1][0] == 3 and q.read() == []
    assert a.summary()["sent"] == 6 and a.summary()["last"]["kind"] == "commit_job"


def test_run_stops_on_event():
    a = attest.Attestor(FakeChain(), lambda: None, list, os.devnull)
    t = attest.start_thread(a, 0.05)
    a.stop.set()
    t.join(2)
    assert not t.is_alive()


def test_rpc_backs_off_on_429(monkeypatch):
    import httpx2

    calls, sleeps = [], []
    monkeypatch.setattr(attest.time, "sleep", sleeps.append)

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx2.Response(429, headers={"retry-after": "0.1"})
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"value": 7}})

    chain = attest.Chain("http://rpc", Keypair(), PROGRAM, httpx2.Client(transport=httpx2.MockTransport(handler)))
    assert chain.rpc("getBalance", "x") == {"value": 7}
    assert len(calls) == 3 and sleeps == [0.1, 0.1]
