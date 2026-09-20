#!/usr/bin/env python3
"""
attest.py - mirrors cluster control-plane events onto the cluster_attest program.

The router runs an Attestor in a background thread when SOLANA_KEYPAIR is set: every
worker-set change seen in the supervisor's status becomes a SetWorkerSet transaction
and every finished answer a CommitJob. Signatures land in attestations.jsonl with
Explorer links. Standalone, for debugging or a second box:

    ./attest.py --show                       # decode the on-chain Cluster account
    ./attest.py --status-url ... --decision-log ...   # follow the files instead of the router
"""

import argparse
import asyncio
import hashlib
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx2
from construct import Bytes, Flag, Int8ul, Int32ul, Int64ul, PascalString, PrefixedArray, Struct, Switch, this
from solana.exceptions import SolanaRpcException
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.models import TxOpts
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import ID as SYSTEM_PROGRAM
from solders.transaction import Transaction

DEVNET = "https://api.devnet.solana.com"
STATE_CODES = {"down": 0, "healthy": 1, "degraded": 2, "restarting": 3}
STATE_NAMES = {v: k for k, v in STATE_CODES.items()}
HERE = Path(__file__).resolve().parent
PROGRAM_KEYPAIR = HERE.parent / "solana" / "program" / "target" / "deploy" / "cluster_attest-keypair.json"

# ----- wire format: the borsh layouts declared in solana/program/src/lib.rs ----------
# borsh is little-endian; strings and vectors carry a u32 length; an enum is a u8 tag.

BorshString = PascalString(Int32ul, "utf8")
INSTRUCTION_TAGS = {"Initialize": 0, "RegisterNode": 1, "SetWorkerSet": 2, "CommitJob": 3}
ClusterInstruction = Struct(
    "tag" / Int8ul,
    "body"
    / Switch(
        this.tag,
        {
            0: Struct("model_hash" / Bytes(32)),
            1: Struct("host" / BorshString),
            2: Struct("state" / Int8ul, "active" / PrefixedArray(Int32ul, BorshString)),
            3: Struct("job_id" / Bytes(16), "served_by" / BorshString, "result_hash" / Bytes(32)),
        },
    ),
)
Node = Struct("host" / BorshString, "active" / Flag)
Cluster = Struct(
    "authority" / Bytes(32),
    "model_hash" / Bytes(32),
    "epoch" / Int64ul,
    "state" / Int8ul,
    "nodes" / PrefixedArray(Int32ul, Node),
    "jobs_total" / Int64ul,
    "jobs_local" / Int64ul,
    "last_job" / Bytes(16),
)


def instruction(kind: str, **fields: Any) -> bytes:
    """Instruction data for the program: instruction("SetWorkerSet", state=2, active=["a"])."""
    return ClusterInstruction.build({"tag": INSTRUCTION_TAGS[kind], "body": fields})


def decode_cluster(data: bytes) -> dict[str, Any]:
    c = Cluster.parse(data)
    return {
        "authority": str(Pubkey.from_bytes(c.authority)),
        "model_hash": c.model_hash.hex(),
        "epoch": c.epoch,
        "state": STATE_NAMES.get(c.state, c.state),
        "nodes": [{"host": n.host, "active": n.active} for n in c.nodes],
        "jobs_total": c.jobs_total,
        "jobs_local": c.jobs_local,
        "last_job": c.last_job.hex(),
    }


def cluster_pda(program_id: Pubkey, authority: Pubkey) -> Pubkey:
    return Pubkey.find_program_address([b"cluster", bytes(authority)], program_id)[0]


def job_pda(program_id: Pubkey, cluster: Pubkey, job_id: bytes) -> Pubkey:
    return Pubkey.find_program_address([b"job", bytes(cluster), job_id], program_id)[0]


# ----- what gets derived from supervisor status and router decisions -----------------


def model_hash(status: dict[str, Any]) -> bytes:
    """sha256 over the model file name and the header the supervisor read from it."""
    root = status.get("root") or {}
    doc = {"model": Path(root.get("model") or "").name, "header": status.get("model_header") or {}}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).digest()


def worker_set(status: dict[str, Any]) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    """(state code, all known worker hosts, active worker hosts) from a /status document."""
    state = STATE_CODES.get(str(status.get("state")), 0)
    hosts = tuple(sorted({str(w["host"]) for w in status.get("workers") or [] if w.get("host")}))
    active = tuple(sorted({str(a).split(":")[0] for a in status.get("active_workers") or []}))
    return state, hosts, active


def job_from_record(rec: dict[str, Any]) -> tuple[bytes, str, bytes] | None:
    """A decision record becomes a job once an answer finished; anything else is skipped."""
    rid, served, digest = rec.get("request_id"), rec.get("served_by"), rec.get("result_sha256")
    if not (rid and served and served != "none" and digest):
        return None
    try:
        return bytes.fromhex(rid)[:16].ljust(16, b"\0"), str(served)[:16], bytes.fromhex(digest)
    except ValueError:
        return None


class LogTail:
    """New JSON lines from a file, surviving truncation and partial writes."""

    def __init__(self, path: str, from_start: bool = False):
        self.path, self.pos = Path(path), 0
        if not from_start and self.path.exists():
            self.pos = self.path.stat().st_size

    def read(self) -> list[dict[str, Any]]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.pos:
            self.pos = 0
        with self.path.open("rb") as fh:
            fh.seek(self.pos)
            chunk = fh.read()
        if not chunk.endswith(b"\n"):
            chunk = chunk[: chunk.rfind(b"\n") + 1]
        self.pos += len(chunk)
        out = []
        for line in chunk.decode(errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


class RecordQueue:
    """In-process source: the router's log() pushes records, the attestor thread drains them."""

    def __init__(self) -> None:
        self.q: queue.SimpleQueue = queue.SimpleQueue()

    def push(self, rec: dict[str, Any]) -> None:
        self.q.put(rec)

    def read(self) -> list[dict[str, Any]]:
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out


def http_status_source(url: str, http: httpx2.Client | None = None) -> Callable[[], dict[str, Any] | None]:
    client = http or httpx2.Client(timeout=3.0)

    def fetch() -> dict[str, Any] | None:
        try:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    return fetch


# ----- chain access -----------------------------------------------------------------


class ChainError(RuntimeError):
    pass


class ChainLike(Protocol):
    """What the attestor needs from a chain; Chain implements it, tests fake it."""

    rpc_url: str
    payer: Keypair
    program_id: Pubkey

    def account_data(self, pubkey: Pubkey) -> bytes | None: ...

    def send(self, data: bytes, accounts: list[AccountMeta], timeout: float = 60.0) -> str: ...


class Chain:
    """solana-py's AsyncClient behind a synchronous face, because the attestor runs on a
    plain thread. One private event loop, one client, a request-rate cap for the public
    Devnet endpoint."""

    def __init__(self, rpc_url: str, payer: Keypair, program_id: Pubkey, rate_limit: float = 4.0) -> None:
        self.rpc_url, self.payer, self.program_id = rpc_url, payer, program_id
        self.rate_limit = rate_limit
        self._loop = asyncio.new_event_loop()
        self._client: AsyncClient | None = None

    async def _connected(self) -> AsyncClient:
        if self._client is None:
            self._client = AsyncClient(
                self.rpc_url, commitment=Confirmed, rate_limit=self.rate_limit, max_transport_retries=3, timeout=20.0
            )
        return self._client

    def _run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        try:
            return self._loop.run_until_complete(coro)
        except SolanaRpcException as e:
            raise ChainError(str(e)) from e
        except TimeoutError as e:
            raise ChainError("transaction not confirmed in time") from e

    def account_data(self, pubkey: Pubkey) -> bytes | None:
        async def go() -> bytes | None:
            resp = await (await self._connected()).get_account_info(pubkey, encoding="base64")
            return bytes(resp.value.data) if resp.value else None

        return self._run(go())

    def send(self, data: bytes, accounts: list[AccountMeta], timeout: float = 60.0) -> str:
        async def go() -> str:
            client = await self._connected()
            blockhash = (await client.get_latest_blockhash()).value.blockhash
            ix = Instruction(self.program_id, data, accounts)
            tx = Transaction.new_signed_with_payer([ix], self.payer.pubkey(), [self.payer], blockhash)
            sig = (await client.send_raw_transaction(bytes(tx), opts=TxOpts(preflight_commitment=Confirmed))).value
            resp = await asyncio.wait_for(client.confirm_transaction(sig, Confirmed), timeout=timeout)
            status = resp.value[0] if resp.value else None
            if status is not None and status.err is not None:
                raise ChainError(f"transaction failed: {status.err}")
            return str(sig)

        return self._run(go())

    def airdrop(self, sol: float) -> str:
        async def go() -> str:
            resp = await (await self._connected()).request_airdrop(self.payer.pubkey(), int(sol * 1_000_000_000))
            return str(resp.value)

        return self._run(go())

    def balance_sol(self) -> float:
        async def go() -> float:
            return (await (await self._connected()).get_balance(self.payer.pubkey())).value / 1_000_000_000

        return self._run(go())


def explorer_url(rpc_url: str, kind: str, ident: str) -> str:
    cluster = "devnet" if "devnet" in rpc_url else f"custom&customUrl={rpc_url}"
    return f"https://explorer.solana.com/{kind}/{ident}?cluster={cluster}"


def default_program_id() -> str:
    if os.environ.get("ATTEST_PROGRAM_ID"):
        return os.environ["ATTEST_PROGRAM_ID"]
    return str(Keypair.from_json(PROGRAM_KEYPAIR.read_text()).pubkey()) if PROGRAM_KEYPAIR.exists() else ""


# ----- the attestor -------------------------------------------------------------------


@dataclass
class Attestor:
    chain: ChainLike
    status_source: Callable[[], dict[str, Any] | None]
    records_source: Callable[[], list[dict[str, Any]]]
    out_path: str
    initialized: bool = False
    registered: set[str] = field(default_factory=set)
    last_set: tuple[int, tuple[str, ...]] | None = None
    pending_jobs: list[tuple[bytes, str, bytes]] = field(default_factory=list)
    max_pending: int = 200
    sent: int = 0
    errors: int = 0
    last: dict[str, Any] | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=20))
    stop: threading.Event = field(default_factory=threading.Event)

    @property
    def cluster(self) -> Pubkey:
        return cluster_pda(self.chain.program_id, self.chain.payer.pubkey())

    def _authority_accounts(self, writable_payer: bool) -> list[AccountMeta]:
        return [
            AccountMeta(self.chain.payer.pubkey(), is_signer=True, is_writable=writable_payer),
            AccountMeta(self.cluster, is_signer=False, is_writable=True),
        ]

    def record(self, kind: str, sig: str, **detail: Any) -> None:
        row = {
            "t": time.time(),
            "kind": kind,
            "signature": sig,
            "explorer": explorer_url(self.chain.rpc_url, "tx", sig),
            **detail,
        }
        self.sent += 1
        self.last = row
        self.recent.append(row)
        try:
            with Path(self.out_path).open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            pass
        print(f"[attest] {kind} {sig[:16]}... {json.dumps(detail)}", flush=True)

    def ensure_initialized(self, status: dict[str, Any] | None) -> bool:
        """True once the Cluster account exists; creates it when a status document is at hand."""
        if self.initialized:
            return True
        data = self.chain.account_data(self.cluster)
        if data:
            self.registered = {n["host"] for n in decode_cluster(data)["nodes"]}
            self.initialized = True
            return True
        if status is None:
            return False
        accounts = [*self._authority_accounts(True), AccountMeta(SYSTEM_PROGRAM, is_signer=False, is_writable=False)]
        sig = self.chain.send(instruction("Initialize", model_hash=model_hash(status)), accounts)
        self.initialized = True
        self.record("initialize", sig, cluster=str(self.cluster), model_hash=model_hash(status).hex())
        return True

    def sync_worker_set(self, status: dict[str, Any]) -> bool:
        state, hosts, active = worker_set(status)
        for host in hosts:
            if host not in self.registered:
                sig = self.chain.send(instruction("RegisterNode", host=host), self._authority_accounts(False))
                self.registered.add(host)
                self.record("register_node", sig, host=host)
        if self.last_set == (state, active):
            return False
        data = instruction("SetWorkerSet", state=state, active=list(active))
        sig = self.chain.send(data, self._authority_accounts(False))
        self.last_set = (state, active)
        self.record("set_worker_set", sig, state=STATE_NAMES[state], active=list(active), reason=status.get("reason"))
        return True

    def sync_jobs(self) -> int:
        for rec in self.records_source():
            job = job_from_record(rec)
            if job:
                self.pending_jobs.append(job)
        self.pending_jobs = self.pending_jobs[-self.max_pending :]
        sent = 0
        while self.pending_jobs:
            job_id, served_by, digest = self.pending_jobs[0]
            accounts = [
                *self._authority_accounts(True),
                AccountMeta(job_pda(self.chain.program_id, self.cluster, job_id), is_signer=False, is_writable=True),
                AccountMeta(SYSTEM_PROGRAM, is_signer=False, is_writable=False),
            ]
            data = instruction("CommitJob", job_id=job_id, served_by=served_by, result_hash=digest)
            try:
                sig = self.chain.send(data, accounts)
            except ChainError as e:
                if "already in use" in str(e) or "custom program error" in str(e):
                    self.pending_jobs.pop(0)  # committed before, or rejected for good: do not retry
                    print(f"[attest] job {job_id.hex()} dropped: {e}", file=sys.stderr)
                    continue
                raise
            self.pending_jobs.pop(0)
            sent += 1
            self.record("commit_job", sig, job_id=job_id.hex(), served_by=served_by, result_sha256=digest.hex())
        return sent

    def step(self) -> None:
        status = self.status_source()
        if not self.ensure_initialized(status):
            return
        if status is not None:
            self.sync_worker_set(status)
        self.sync_jobs()

    def run(self, interval: float) -> None:
        while not self.stop.is_set():
            try:
                self.step()
            except (ChainError, httpx2.HTTPError) as e:
                self.errors += 1
                print(f"[attest] retrying: {e}", file=sys.stderr, flush=True)
            self.stop.wait(interval)

    def summary(self) -> dict[str, Any]:
        return {
            "cluster": str(self.cluster),
            "explorer": explorer_url(self.chain.rpc_url, "address", str(self.cluster)),
            "initialized": self.initialized,
            "sent": self.sent,
            "errors": self.errors,
            "pending_jobs": len(self.pending_jobs),
            "last": self.last,
            "recent": list(self.recent),
            "payer": str(self.chain.payer.pubkey()),
        }


def start_thread(att: Attestor, interval: float) -> threading.Thread:
    t = threading.Thread(target=att.run, args=(interval,), name="attest", daemon=True)
    t.start()
    return t


# ----- CLI ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpc-url", default=os.environ.get("SOLANA_RPC_URL", DEVNET))
    ap.add_argument("--keypair", default=os.environ.get("SOLANA_KEYPAIR", str(Path.home() / ".config/solana/id.json")))
    ap.add_argument("--program-id", default=default_program_id())
    ap.add_argument("--status-url", default=os.environ.get("STATUS_URL", "http://192.168.50.10:9991/status"))
    ap.add_argument("--decision-log", default=os.environ.get("DECISION_LOG", str(HERE / "routing_decisions.jsonl")))
    ap.add_argument("--out", default=str(HERE / "attestations.jsonl"))
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--from-start", action="store_true", help="attest every existing decision-log record too")
    ap.add_argument("--show", action="store_true", help="print the on-chain Cluster account and exit")
    ap.add_argument("--once", action="store_true", help="one sync pass, then exit")
    args = ap.parse_args()
    if not args.program_id:
        sys.exit("no program id: pass --program-id, set ATTEST_PROGRAM_ID, or build the program first")

    payer = Keypair.from_json(Path(args.keypair).read_text())
    chain = Chain(args.rpc_url, payer, Pubkey.from_string(args.program_id))
    att = Attestor(
        chain, http_status_source(args.status_url), LogTail(args.decision_log, args.from_start).read, args.out
    )

    if args.show:
        data = chain.account_data(att.cluster)
        doc = {"cluster": str(att.cluster), "explorer": explorer_url(args.rpc_url, "address", str(att.cluster))}
        print(json.dumps({**doc, **(decode_cluster(data) if data else {"initialized": False})}, indent=2))
        return
    print(
        f"[attest] payer {payer.pubkey()} balance {chain.balance_sol():.3f} SOL, cluster account {att.cluster}",
        flush=True,
    )
    if args.once:
        att.step()
        return
    att.run(args.interval)


if __name__ == "__main__":
    main()
