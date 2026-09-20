"""
test_attest.py - encoders, status/log parsing and the sidecar loop against a fake chain. Run: pytest -q
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import attest
import pytest
from solana.exceptions import SolanaRpcException
from solana.rpc.async_api import AsyncClient
from solana.rpc.core import RPCException, TransactionExpiredBlockheightExceededError, UnconfirmedTxError
from solders.keypair import Keypair
from solders.pubkey import Pubkey

EXAMPLE = json.loads((Path(__file__).parent.parent / "cluster" / "supervisor" / "status.example.json").read_text())
PROGRAM = Pubkey.new_unique()


# ----- borsh layout: must match the Rust crate (see instruction_layout_is_stable there) -----


def ix(name, **fields):
    return attest.instruction(name, **fields)


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
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if isinstance(self.doc, Exception):
            raise self.doc
        return self.doc


class FakeChain:
    """Applies instructions to an in-memory Cluster account and serves it back borsh-encoded,
    so a re-read after a failure sees exactly what landed."""

    def __init__(self):
        self.payer, self.program_id, self.rpc_url = Keypair(), PROGRAM, attest.DEVNET
        self.sent: list[bytes] = []
        self.initialized = False
        self.fail_next: str | None = None
        self.state = 0
        self.nodes: dict[str, bool] = {}
        self.jobs = 0
        self.closed = False

    async def account_data(self, pubkey):
        if not self.initialized:
            return None
        doc = {
            "authority": bytes(self.payer.pubkey()),
            "model_hash": bytes(32),
            "epoch": 1,
            "state": self.state,
            "nodes": [{"host": h, "active": a} for h, a in self.nodes.items()],
            "jobs_total": self.jobs,
            "jobs_local": 0,
            "last_job": bytes(16),
        }
        return attest.Cluster.build(doc).ljust(1024, b"\0")

    async def send(self, data, accounts):
        if self.fail_next:
            err, self.fail_next = self.fail_next, None
            raise attest.ChainError(err)
        self.sent.append(data)
        parsed = attest.ClusterInstruction.parse(data)
        if parsed.tag == 0:
            self.initialized = True
        elif parsed.tag == 1:
            self.nodes[parsed.body.host] = False
        elif parsed.tag == 2:
            self.state = parsed.body.state
            for host in self.nodes:
                self.nodes[host] = host in parsed.body.active
        elif parsed.tag == 3:
            self.jobs += 1
        return f"sig{len(self.sent)}"

    async def close(self):
        self.closed = True


@pytest.fixture
def att(tmp_path):
    log = tmp_path / "decisions.jsonl"
    log.write_text("")
    chain = FakeChain()
    src = FakeStatus(EXAMPLE)  # what the fake supervisor says right now
    a = attest.Attestor(chain, src, attest.LogTail(str(log)).read, str(tmp_path / "out.jsonl"))
    return a, chain, log, tmp_path / "out.jsonl", src


async def test_first_step_initializes_registers_and_sets_the_worker_set(att):
    a, chain, _, out, _src = att
    await a.step()
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


async def test_unchanged_status_sends_nothing(att):
    a, chain, _, _, _src = att
    await a.step()
    n = len(chain.sent)
    await a.step()
    assert len(chain.sent) == n


async def test_worker_returning_bumps_the_set_once(att):
    a, chain, _, _, src = att
    await a.step()
    src.doc = {
        **EXAMPLE,
        "state": "healthy",
        "active_workers": ["192.168.50.11:9998", "192.168.50.12:9998", "192.168.50.14:9998"],
    }
    await a.step()
    assert chain.sent[-1] == ix("SetWorkerSet", state=1, active=["192.168.50.11", "192.168.50.12", "192.168.50.14"])
    await a.step()
    assert chain.sent[-1][0] == 2 and len(chain.sent) == 6


async def test_jobs_are_committed_and_retried_after_rpc_errors(att):
    a, chain, log, out, _src = att
    await a.step()
    digest = hashlib.sha256(b"answer").hexdigest()
    with log.open("a") as fh:
        fh.write(json.dumps({"request_id": "ab" * 16, "served_by": "cluster", "result_sha256": digest}) + "\n")
        fh.write(json.dumps({"request_id": "cd" * 16, "served_by": "none", "error": "x"}) + "\n")
    chain.fail_next = "getLatestBlockhash: timeout"
    with pytest.raises(attest.ChainError):
        await a.step()
    assert a.pending_jobs and chain.sent[-1][0] == 2  # nothing committed yet, job kept
    assert not a.initialized, "a chain error forgets the account view"
    await a.step()
    assert chain.sent[-1] == ix(
        "CommitJob", job_id=bytes.fromhex("ab" * 16), served_by="cluster", result_hash=bytes.fromhex(digest)
    )
    assert not a.pending_jobs and a.initialized
    assert [d[0] for d in chain.sent] == [0, 1, 1, 1, 2, 3], "the re-read found everything registered: no resend"
    assert json.loads(out.read_text().splitlines()[-1])["kind"] == "commit_job"


async def test_partial_worker_set_failure_resyncs_from_the_chain(att):
    """RegisterNode for the second host fails: what the attestor remembers must come back from
    the account, not from the half-applied local sets."""
    a, chain, _, _, _src = att
    failed = False

    async def flaky_send(data, accounts, _real=chain.send):
        nonlocal failed
        if data[0] == 1 and len(chain.sent) == 2 and not failed:  # the second RegisterNode, once
            failed = True
            raise attest.ChainError("SolanaRpcException: ReadTimeout")
        return await _real(data, accounts)

    chain.send = flaky_send  # type: ignore[method-assign]
    with pytest.raises(attest.ChainError):
        await a.step()
    assert [d[0] for d in chain.sent] == [0, 1] and a.registered == {"192.168.50.11"} and not a.initialized
    await a.step()
    assert [d[0] for d in chain.sent] == [0, 1, 1, 1, 2]
    assert a.registered == {"192.168.50.11", "192.168.50.12", "192.168.50.14"}
    assert a.last_set == (2, ("192.168.50.11",))
    await a.step()
    assert len(chain.sent) == 5, "and the chain's worker set is trusted: nothing is sent again"


async def test_duplicate_job_is_dropped_not_retried(att):
    a, chain, log, _, _src = att
    await a.step()
    with log.open("a") as fh:
        fh.write(json.dumps({"request_id": "ef" * 16, "served_by": "openai", "result_sha256": "00" * 32}) + "\n")
    chain.fail_next = "transaction failed: {'InstructionError': [0, {'Custom': 0}]} custom program error"
    await a.step()
    assert not a.pending_jobs and chain.sent[-1][0] == 2 and a.dropped_jobs == 1
    with log.open("a") as fh:
        fh.write(json.dumps({"request_id": "0f" * 16, "served_by": "openai", "result_sha256": "00" * 32}) + "\n")
    chain.fail_next = (
        'RPCException: SendTransactionPreflightFailureMessage { message: "Transaction simulation failed: '
        'Error processing Instruction 0: custom program error: 0x0", data: RpcSimulateTransactionResult('
        'logs: Some(["Program log: Allocate: account Address { address: X, base: None } already in use"]) }'
    )
    await a.step()
    assert not a.pending_jobs and a.dropped_jobs == 2, "a preflight failure carrying the marker is final too"


def test_pending_jobs_are_bounded_and_the_oldest_is_dropped():
    records = [{"request_id": f"{i:02x}" * 16, "served_by": "openai", "result_sha256": "00" * 32} for i in range(5)]
    a = attest.Attestor(FakeChain(), FakeStatus(None), lambda: records, os.devnull, max_pending=3)
    a.queue_jobs()
    assert len(a.pending_jobs) == 3 and a.dropped_jobs == 2
    assert a.pending_jobs[0][0] == bytes.fromhex("02" * 16), "the two oldest went"


async def test_status_outage_still_commits_jobs(att):
    a, chain, log, _, src = att
    await a.step()
    src.doc = None
    with log.open("a") as fh:
        fh.write(json.dumps({"request_id": "12" * 16, "served_by": "gemini", "result_sha256": "11" * 32}) + "\n")
    await a.step()
    assert chain.sent[-1][0] == 3


async def test_record_queue_feeds_jobs_in_process():
    q = attest.RecordQueue()
    chain = FakeChain()
    a = attest.Attestor(chain, FakeStatus(EXAMPLE), q.read, os.devnull)
    await a.step()
    q.push({"request_id": "ab" * 16, "served_by": "cluster", "result_sha256": "00" * 32})
    q.push({"request_id": "cd" * 16, "served_by": "none"})
    await a.step()
    assert chain.sent[-1][0] == 3 and q.read() == []
    assert a.summary()["sent"] == 6 and a.summary()["last"]["kind"] == "commit_job"


def test_record_queue_drops_when_full_instead_of_growing():
    q = attest.RecordQueue(maxsize=2)
    for i in range(3):
        q.push({"request_id": str(i)})
    assert q.dropped == 1 and [r["request_id"] for r in q.read()] == ["0", "1"]


async def test_run_stops_on_event_and_reports_alive():
    a = attest.Attestor(FakeChain(), FakeStatus(None), list, os.devnull)
    assert a.summary()["alive"] is False
    task = asyncio.create_task(a.run(0.05))
    await asyncio.sleep(0.01)
    assert a.alive is True
    a.stop.set()
    await asyncio.wait_for(task, 2)
    assert a.alive is False and a.summary()["alive"] is False


async def test_run_survives_an_unexpected_exception():
    """A bug in one step must not kill the loop: it is logged, counted, and the next step runs."""
    src = FakeStatus(RuntimeError("status parser bug"))
    a = attest.Attestor(FakeChain(), src, list, os.devnull)
    task = asyncio.create_task(a.run(0.01))
    await asyncio.sleep(0.08)
    assert a.alive is True and a.errors >= 2 and src.calls >= 2
    src.doc = EXAMPLE  # the bug goes away: the loop picks up where it should
    await asyncio.sleep(0.05)
    assert a.initialized and a.summary()["sent"] >= 5
    a.stop.set()
    await asyncio.wait_for(task, 2)


async def test_run_survives_chain_errors_and_cancels_cleanly():
    chain = FakeChain()
    a = attest.Attestor(chain, FakeStatus(EXAMPLE), list, os.devnull)
    chain.fail_next = "SolanaRpcException: ConnectError"
    task = asyncio.create_task(a.run(0.01))
    await asyncio.sleep(0.08)
    assert a.errors == 1 and a.initialized and chain.sent, "one retry, then it went through"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert a.alive is False


# ----- Chain: solana-py's failures all become ChainError ---------------------------------


class FakeRpc:
    """Stands in for solana.rpc.async_api.AsyncClient: each method raises what it is told to."""

    def __init__(self, **raises: Exception):
        self.raises = raises
        self.closed = False

    async def close(self):
        self.closed = True

    def __getattr__(self, name: str):
        exc = self.raises.get(name)

        async def call(*args: Any, **kwargs: Any):
            if exc is not None:
                raise exc
            raise AssertionError(f"unexpected RPC call {name}")

        return call


@pytest.mark.parametrize(
    "exc",
    [
        RPCException("preflight: custom program error: 0x1"),
        UnconfirmedTxError("Unable to confirm transaction"),
        TransactionExpiredBlockheightExceededError("expired: block height exceeded"),
        TimeoutError(),
        SolanaRpcException(ConnectionError("refused"), lambda: None, None, object()),
    ],
)
async def test_every_rpc_failure_is_a_chain_error(exc):
    chain = attest.Chain(attest.DEVNET, Keypair(), PROGRAM)
    await chain.close()
    chain.client = cast(AsyncClient, FakeRpc(get_latest_blockhash=exc, get_account_info=exc))
    with pytest.raises(attest.ChainError) as info:
        await chain.send(ix("RegisterNode", host="h"), [])
    assert type(exc).__name__ in str(info.value)
    with pytest.raises(attest.ChainError):
        await chain.account_data(chain.payer.pubkey())


async def test_chain_send_confirms_against_the_blockhash_expiry():
    """The confirmation waits until the blockhash expires, not a fixed 60 s, and a failed
    status is reported as a ChainError."""
    from types import SimpleNamespace

    from solders.hash import Hash
    from solders.signature import Signature

    seen: dict[str, Any] = {}

    class Rpc(FakeRpc):
        async def get_latest_blockhash(self):
            return SimpleNamespace(value=SimpleNamespace(blockhash=Hash.default(), last_valid_block_height=1234))

        async def send_raw_transaction(self, raw, opts=None):
            seen["opts"] = opts
            return SimpleNamespace(value=Signature.default())

        async def confirm_transaction(self, sig, commitment=None, sleep_seconds=0.5, last_valid_block_height=None):
            seen["last_valid_block_height"] = last_valid_block_height
            return SimpleNamespace(value=[SimpleNamespace(err={"InstructionError": [0, {"Custom": 1}]})])

    chain = attest.Chain(attest.DEVNET, Keypair(), PROGRAM)
    await chain.close()
    chain.client = cast(AsyncClient, Rpc())
    with pytest.raises(attest.ChainError, match=r"Custom.*1"):
        await chain.send(ix("RegisterNode", host="h"), [])
    assert seen["last_valid_block_height"] == 1234 and seen["opts"].skip_confirmation is True
