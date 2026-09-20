"""
test_attest_e2e.py - the sidecar against the real program on a validator. Skipped unless
ATTEST_RPC_URL points at one (solana-test-validator, or devnet with a funded keypair):

    ATTEST_RPC_URL=http://127.0.0.1:8899 pytest -q test_attest_e2e.py
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import attest
import httpx2
import pytest
from solders.keypair import Keypair
from solders.pubkey import Pubkey

RPC = os.environ.get("ATTEST_RPC_URL", "")
pytestmark = pytest.mark.skipif(not RPC, reason="ATTEST_RPC_URL not set")
EXAMPLE = json.loads((Path(__file__).parent.parent / "cluster" / "supervisor" / "status.example.json").read_text())


class StatusServer:
    """What the supervisor would serve, as an httpx2 MockTransport instead of a port."""

    def __init__(self):
        self.doc = dict(EXAMPLE)
        self.url = "http://supervisor.test/status"
        self.http = httpx2.AsyncClient(transport=httpx2.MockTransport(lambda req: httpx2.Response(200, json=self.doc)))


async def funded_chain(program_id: Pubkey) -> attest.Chain:
    c = attest.Chain(RPC, Keypair(), program_id)  # fresh authority each time so the Cluster PDA starts empty
    sig = await c.airdrop(2)
    for _ in range(60):
        if await c.balance_sol() >= 1:
            break
        await asyncio.sleep(1)
    assert await c.balance_sol() >= 1, f"airdrop {sig} did not land"
    return c


async def onchain(chain: attest.Chain, att: attest.Attestor) -> dict:
    data = await chain.account_data(att.cluster)
    assert data, "the Cluster account exists"
    return attest.decode_cluster(data)


async def test_events_land_on_chain_and_rules_hold(tmp_path):
    chain = await funded_chain(Pubkey.from_string(attest.default_program_id()))
    status = StatusServer()
    log = tmp_path / "decisions.jsonl"
    log.write_text("")
    out = tmp_path / "attestations.jsonl"
    att = attest.Attestor(
        chain, attest.http_status_source(status.url, status.http), attest.LogTail(str(log)).read, str(out)
    )

    await att.step()
    c = await onchain(chain, att)
    assert c["authority"] == str(chain.payer.pubkey())
    assert c["model_hash"] == attest.model_hash(EXAMPLE).hex()
    assert (c["epoch"], c["state"]) == (1, "degraded")
    assert [n["host"] for n in c["nodes"]] == ["192.168.50.11", "192.168.50.12", "192.168.50.14"]
    assert [n["active"] for n in c["nodes"]] == [True, False, False]

    # worker comes back: epoch bumps once, set grows
    status.doc = {
        **EXAMPLE,
        "state": "healthy",
        "active_workers": ["192.168.50.11:9998", "192.168.50.12:9998", "192.168.50.14:9998"],
    }
    await att.step()
    await att.step()
    c = await onchain(chain, att)
    assert (c["epoch"], c["state"]) == (2, "healthy") and all(n["active"] for n in c["nodes"])

    # two finished answers, one failure record
    digest = hashlib.sha256(b"the answer").hexdigest()
    rid1, rid2 = "11" * 16, "22" * 16
    with log.open("a") as fh:
        fh.write(json.dumps({"request_id": rid1, "served_by": "cluster", "result_sha256": digest}) + "\n")
        fh.write(json.dumps({"request_id": rid2, "served_by": "baseten", "result_sha256": digest}) + "\n")
        fh.write(json.dumps({"request_id": "33" * 16, "served_by": "none", "error": "all failed"}) + "\n")
    await att.step()
    c = await onchain(chain, att)
    assert (c["jobs_total"], c["jobs_local"], c["last_job"]) == (2, 1, rid2)
    job = await chain.account_data(attest.job_pda(chain.program_id, att.cluster, bytes.fromhex(rid1)))
    assert job and digest in job.hex() and b"cluster" in job

    # the program refuses a second commit of the same job and an unknown host
    with pytest.raises(attest.ChainError):
        await chain.send(
            attest.instruction("CommitJob", job_id=bytes.fromhex(rid1), served_by="cluster", result_hash=bytes(32)),
            [
                *att._authority_accounts(True),
                attest.AccountMeta(attest.job_pda(chain.program_id, att.cluster, bytes.fromhex(rid1)), False, True),
                attest.AccountMeta(attest.SYSTEM_PROGRAM, False, False),
            ],
        )
    with pytest.raises(attest.ChainError, match=r"Custom.*1"):
        await chain.send(
            attest.instruction("SetWorkerSet", state=1, active=["10.9.9.9"]), att._authority_accounts(False)
        )
    # ... and a stranger cannot touch the cluster account at all
    stranger = await funded_chain(chain.program_id)
    with pytest.raises(attest.ChainError, match=r"Custom.*0"):
        await stranger.send(
            attest.instruction("SetWorkerSet", state=0, active=[]),
            [attest.AccountMeta(stranger.payer.pubkey(), True, False), attest.AccountMeta(att.cluster, False, True)],
        )

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["kind"] for r in rows] == [
        "initialize",
        "register_node",
        "register_node",
        "register_node",
        "set_worker_set",
        "set_worker_set",
        "commit_job",
        "commit_job",
    ]
    assert all(r["signature"] and r["explorer"] for r in rows)
    await stranger.close()
    await chain.close()
    await status.http.aclose()
