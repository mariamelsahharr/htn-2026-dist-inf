#!/usr/bin/env python3
"""
attest.py - mirrors cluster control-plane events onto the cluster_attest program.

The router runs an Attestor as a task on its event loop when SOLANA_KEYPAIR is set:
every worker-set change seen in the supervisor's status becomes a SetWorkerSet
transaction and every finished answer a CommitJob. Signatures land in
attestations.jsonl with Explorer links. Standalone, for debugging or a second box:

    ./attest.py --show                       # decode the on-chain Cluster account
    ./attest.py --status-url ... --decision-log ...   # follow the files instead of the router
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sys
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx2
import logs
from construct import Bytes, Flag, Int8ul, Int32ul, Int64ul, PascalString, PrefixedArray, Struct, Switch, this
from solana.exceptions import SolanaRpcException
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.core import (
    RPCException,
    RPCNoResultException,
    TransactionExpiredBlockheightExceededError,
    UnconfirmedTxError,
)
from solana.rpc.models import TxOpts
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import ID as SYSTEM_PROGRAM
from solders.transaction import Transaction

log = logging.getLogger(__name__)

DEVNET = "https://api.devnet.solana.com"
STATE_CODES = {"down": 0, "healthy": 1, "degraded": 2, "restarting": 3}
STATE_NAMES = {v: k for k, v in STATE_CODES.items()}
HERE = Path(__file__).resolve().parent
PROGRAM_KEYPAIR = HERE.parent / "solana" / "program" / "target" / "deploy" / "cluster_attest-keypair.json"

# Everything solana-py raises for a failed call: transport (SolanaRpcException), an RPC error
# result such as a preflight failure (RPCException), no result, and the two confirmation timeouts.
RPC_ERRORS = (
    SolanaRpcException,
    RPCException,
    RPCNoResultException,
    UnconfirmedTxError,
    TransactionExpiredBlockheightExceededError,
    TimeoutError,
)
# A transaction the program rejected for good, or one already landed: retrying cannot help.
FINAL_REJECTION_MARKERS = ("already in use", "custom program error")

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
    """In-process source: the router's log() pushes records on the loop, the attestor drains them."""

    def __init__(self, maxsize: int = 1000) -> None:
        self.q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def push(self, rec: dict[str, Any]) -> None:
        try:
            self.q.put_nowait(rec)
        except asyncio.QueueFull:
            self.dropped += 1
            log.warning("attestation queue full (%d): dropping record %s", self.q.maxsize, rec.get("request_id"))

    def read(self) -> list[dict[str, Any]]:
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except asyncio.QueueEmpty:
                return out


StatusSource = Callable[[], Awaitable[dict[str, Any] | None]]


def http_status_source(url: str, http: httpx2.AsyncClient | None = None) -> StatusSource:
    client = http or httpx2.AsyncClient(timeout=3.0)

    async def fetch() -> dict[str, Any] | None:
        try:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, dict) else None
        except (httpx2.HTTPError, ValueError):
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

    async def account_data(self, pubkey: Pubkey) -> bytes | None: ...

    async def send(self, data: bytes, accounts: list[AccountMeta]) -> str: ...

    async def close(self) -> None: ...


class Chain:
    """solana-py's AsyncClient on the caller's event loop, every failure a ChainError,
    with a request-rate cap for the public Devnet endpoint."""

    def __init__(self, rpc_url: str, payer: Keypair, program_id: Pubkey, rate_limit: float = 4.0) -> None:
        self.rpc_url, self.payer, self.program_id = rpc_url, payer, program_id
        self.client = AsyncClient(
            rpc_url, commitment=Confirmed, rate_limit=rate_limit, max_transport_retries=3, timeout=20.0
        )

    async def close(self) -> None:
        await self.client.close()

    @contextlib.asynccontextmanager
    async def _translated(self) -> AsyncIterator[None]:
        try:
            yield
        except RPC_ERRORS as e:
            raise ChainError(f"{type(e).__name__}: {e}") from e

    async def account_data(self, pubkey: Pubkey) -> bytes | None:
        async with self._translated():
            resp = await self.client.get_account_info(pubkey, encoding="base64")
            return bytes(resp.value.data) if resp.value else None

    async def send(self, data: bytes, accounts: list[AccountMeta]) -> str:
        """Sign, send with preflight, and wait for confirmation until the blockhash expires."""
        async with self._translated():
            latest = (await self.client.get_latest_blockhash()).value
            ix = Instruction(self.program_id, data, accounts)
            tx = Transaction.new_signed_with_payer([ix], self.payer.pubkey(), [self.payer], latest.blockhash)
            opts = TxOpts(skip_confirmation=True, preflight_commitment=Confirmed)
            sig = (await self.client.send_raw_transaction(bytes(tx), opts=opts)).value
            resp = await self.client.confirm_transaction(
                sig, Confirmed, last_valid_block_height=latest.last_valid_block_height
            )
            status = resp.value[0] if resp.value else None
            if status is not None and status.err is not None:
                raise ChainError(f"transaction failed: {status.err}")
            return str(sig)

    async def airdrop(self, sol: float) -> str:
        async with self._translated():
            resp = await self.client.request_airdrop(self.payer.pubkey(), int(sol * 1_000_000_000))
            return str(resp.value)

    async def balance_sol(self) -> float:
        async with self._translated():
            return (await self.client.get_balance(self.payer.pubkey())).value / 1_000_000_000


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
    status_source: StatusSource
    records_source: Callable[[], list[dict[str, Any]]]
    out_path: str
    max_pending: int = 200
    initialized: bool = False
    registered: set[str] = field(default_factory=set)
    last_set: tuple[int, tuple[str, ...]] | None = None
    pending_jobs: deque[tuple[bytes, str, bytes]] = field(init=False)
    sent: int = 0
    errors: int = 0
    dropped_jobs: int = 0
    alive: bool = False
    last: dict[str, Any] | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=20))
    stop: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.pending_jobs = deque(maxlen=self.max_pending)

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
            log.exception("could not append to %s", self.out_path)
        log.info("%s %s... %s", kind, sig[:16], json.dumps(detail))

    async def ensure_initialized(self, status: dict[str, Any] | None) -> bool:
        """True once the Cluster account exists; creates it when a status document is at hand.
        Reading the account also adopts its node list and worker set, so after any failure
        the attestor continues from what the chain holds, not from what it remembers."""
        if self.initialized:
            return True
        data = await self.chain.account_data(self.cluster)
        if data:
            onchain = decode_cluster(data)
            self.registered = {n["host"] for n in onchain["nodes"]}
            state = STATE_CODES.get(onchain["state"], 0)
            self.last_set = (state, tuple(sorted(n["host"] for n in onchain["nodes"] if n["active"])))
            self.initialized = True
            return True
        if status is None:
            return False
        accounts = [*self._authority_accounts(True), AccountMeta(SYSTEM_PROGRAM, is_signer=False, is_writable=False)]
        sig = await self.chain.send(instruction("Initialize", model_hash=model_hash(status)), accounts)
        self.initialized = True
        self.record("initialize", sig, cluster=str(self.cluster), model_hash=model_hash(status).hex())
        return True

    async def sync_worker_set(self, status: dict[str, Any]) -> bool:
        state, hosts, active = worker_set(status)
        for host in hosts:
            if host not in self.registered:
                sig = await self.chain.send(instruction("RegisterNode", host=host), self._authority_accounts(False))
                self.registered.add(host)
                self.record("register_node", sig, host=host)
        if self.last_set == (state, active):
            return False
        data = instruction("SetWorkerSet", state=state, active=list(active))
        sig = await self.chain.send(data, self._authority_accounts(False))
        self.last_set = (state, active)
        self.record("set_worker_set", sig, state=STATE_NAMES[state], active=list(active), reason=status.get("reason"))
        return True

    def queue_jobs(self) -> None:
        for rec in self.records_source():
            job = job_from_record(rec)
            if job is None:
                continue
            if len(self.pending_jobs) == self.pending_jobs.maxlen:
                oldest = self.pending_jobs[0]
                self.dropped_jobs += 1
                log.warning("pending jobs at %d: dropping oldest job %s", self.max_pending, oldest[0].hex())
            self.pending_jobs.append(job)

    async def sync_jobs(self) -> int:
        self.queue_jobs()
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
                sig = await self.chain.send(data, accounts)
            except ChainError as e:
                if any(marker in str(e) for marker in FINAL_REJECTION_MARKERS):
                    self.pending_jobs.popleft()  # committed before, or rejected for good: do not retry
                    self.dropped_jobs += 1
                    log.warning("job %s dropped: %.300s", job_id.hex(), e)
                    continue
                raise
            self.pending_jobs.popleft()
            sent += 1
            self.record("commit_job", sig, job_id=job_id.hex(), served_by=served_by, result_sha256=digest.hex())
        return sent

    async def step(self) -> None:
        """One sync pass. A chain error forgets the account view so the next pass re-reads it."""
        try:
            status = await self.status_source()
            if not await self.ensure_initialized(status):
                return
            if status is not None:
                await self.sync_worker_set(status)
            await self.sync_jobs()
        except ChainError:
            self.initialized = False
            raise

    async def run(self, interval: float) -> None:
        """Sync forever, `interval` seconds apart, until stop is set or the task is cancelled.
        No exception ends the loop: expected ones are counted, unexpected ones logged with their trace."""
        self.alive = True
        try:
            while not self.stop.is_set():
                try:
                    await self.step()
                except (ChainError, httpx2.HTTPError) as e:
                    self.errors += 1
                    log.warning("retrying: %.300s", e)
                except Exception:
                    self.errors += 1
                    log.exception("attestor step failed; continuing")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.stop.wait(), timeout=interval)
        finally:
            self.alive = False

    def summary(self) -> dict[str, Any]:
        return {
            "cluster": str(self.cluster),
            "explorer": explorer_url(self.chain.rpc_url, "address", str(self.cluster)),
            "initialized": self.initialized,
            "alive": self.alive,
            "sent": self.sent,
            "errors": self.errors,
            "pending_jobs": len(self.pending_jobs),
            "dropped_jobs": self.dropped_jobs,
            "last": self.last,
            "recent": list(self.recent),
            "payer": str(self.chain.payer.pubkey()),
        }


# ----- CLI ---------------------------------------------------------------------------


async def amain(args: argparse.Namespace) -> None:
    payer = Keypair.from_json(Path(args.keypair).read_text())
    chain = Chain(args.rpc_url, payer, Pubkey.from_string(args.program_id))
    async with httpx2.AsyncClient(timeout=3.0) as http:
        att = Attestor(
            chain, http_status_source(args.status_url, http), LogTail(args.decision_log, args.from_start).read, args.out
        )
        try:
            if args.show:
                data = await chain.account_data(att.cluster)
                doc = {"cluster": str(att.cluster), "explorer": explorer_url(args.rpc_url, "address", str(att.cluster))}
                print(json.dumps({**doc, **(decode_cluster(data) if data else {"initialized": False})}, indent=2))
                return
            log.info(
                "payer %s balance %.3f SOL, cluster account %s", payer.pubkey(), await chain.balance_sol(), att.cluster
            )
            if args.once:
                await att.step()
                return
            await att.run(args.interval)
        finally:
            await chain.close()


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
    logs.configure()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
