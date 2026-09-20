"""
test_e2e.py - the sidecar against the real program on a validator. Skipped unless
ATTEST_RPC_URL points at one (solana-test-validator, or devnet with a funded keypair):

    ATTEST_RPC_URL=http://127.0.0.1:8899 pytest -q test_attest_e2e.py
"""

import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from solders.keypair import Keypair
from solders.pubkey import Pubkey

sys.path.insert(0, os.path.dirname(__file__))
import attest  # noqa: E402

RPC = os.environ.get("ATTEST_RPC_URL")
pytestmark = pytest.mark.skipif(not RPC, reason="ATTEST_RPC_URL not set")
EXAMPLE = json.load(open(Path(__file__).parent.parent / "cluster" / "supervisor" / "status.example.json"))


class StatusServer:
    def __init__(self):
        self.doc = dict(EXAMPLE)
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(outer.doc).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/status"


@pytest.fixture(scope="module")
def chain():
    program_id = Pubkey.from_string(attest.default_program_id())
    payer = Keypair()   # fresh authority each run so the Cluster PDA starts empty
    c = attest.Chain(RPC, payer, program_id)
    sig = c.airdrop(2)
    deadline = 60
    while c.balance_sol() < 1 and deadline:
        time.sleep(1)
        deadline -= 1
    assert c.balance_sol() >= 1, f"airdrop {sig} did not land"
    return c


def test_events_land_on_chain_and_rules_hold(chain, tmp_path):
    status = StatusServer()
    log = tmp_path / "decisions.jsonl"
    log.write_text("")
    out = tmp_path / "attestations.jsonl"
    att = attest.Attestor(chain, attest.http_status_source(status.url), attest.LogTail(str(log)).read, str(out))

    att.step()
    c = attest.decode_cluster(chain.account_data(att.cluster))
    assert c["authority"] == str(chain.payer.pubkey())
    assert c["model_hash"] == attest.model_hash(EXAMPLE).hex()
    assert (c["epoch"], c["state"]) == (1, "degraded")
    assert [n["host"] for n in c["nodes"]] == ["192.168.50.11", "192.168.50.12", "192.168.50.14"]
    assert [n["active"] for n in c["nodes"]] == [True, False, False]

    # worker comes back: epoch bumps once, set grows
    status.doc = {**EXAMPLE, "state": "healthy",
                  "active_workers": ["192.168.50.11:9998", "192.168.50.12:9998", "192.168.50.14:9998"]}
    att.step()
    att.step()
    c = attest.decode_cluster(chain.account_data(att.cluster))
    assert (c["epoch"], c["state"]) == (2, "healthy") and all(n["active"] for n in c["nodes"])

    # two finished answers, one failure record
    digest = hashlib.sha256(b"the answer").hexdigest()
    rid1, rid2 = "11" * 16, "22" * 16
    with open(log, "a") as fh:
        fh.write(json.dumps({"request_id": rid1, "served_by": "cluster", "result_sha256": digest}) + "\n")
        fh.write(json.dumps({"request_id": rid2, "served_by": "baseten", "result_sha256": digest}) + "\n")
        fh.write(json.dumps({"request_id": "33" * 16, "served_by": "none", "error": "all failed"}) + "\n")
    att.step()
    c = attest.decode_cluster(chain.account_data(att.cluster))
    assert (c["jobs_total"], c["jobs_local"], c["last_job"]) == (2, 1, rid2)
    job = chain.account_data(attest.job_pda(chain.program_id, att.cluster, bytes.fromhex(rid1)))
    assert job and digest in job.hex() and b"cluster" in job

    # the program refuses a second commit of the same job and an unknown host
    with pytest.raises(attest.ChainError):
        chain.send(attest.ClusterInstruction.build(attest.ClusterInstruction.enum.CommitJob(
            job_id=bytes.fromhex(rid1), served_by="cluster", result_hash=bytes(32))),
            att._authority_accounts(True) + [
                attest.AccountMeta(attest.job_pda(chain.program_id, att.cluster, bytes.fromhex(rid1)), False, True),
                attest.AccountMeta(attest.SYSTEM_PROGRAM, False, False)])
    with pytest.raises(attest.ChainError, match="Custom.*1"):
        chain.send(attest.ClusterInstruction.build(attest.ClusterInstruction.enum.SetWorkerSet(state=1, active=["10.9.9.9"])),
                   att._authority_accounts(False))
    # ... and a stranger cannot touch the cluster account at all
    stranger = attest.Chain(RPC, Keypair(), chain.program_id)
    stranger.airdrop(1)
    for _ in range(30):
        if stranger.balance_sol() > 0:
            break
        time.sleep(1)
    with pytest.raises(attest.ChainError, match="Custom.*0"):
        stranger.send(attest.ClusterInstruction.build(attest.ClusterInstruction.enum.SetWorkerSet(state=0, active=[])),
                      [attest.AccountMeta(stranger.payer.pubkey(), True, False),
                       attest.AccountMeta(att.cluster, False, True)])

    rows = [json.loads(line) for line in open(out)]
    assert [r["kind"] for r in rows] == ["initialize", "register_node", "register_node", "register_node",
                                         "set_worker_set", "set_worker_set", "commit_job", "commit_job"]
    assert all(r["signature"] and r["explorer"] for r in rows)
