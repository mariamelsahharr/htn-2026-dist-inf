# Solana control plane

The Pis do inference over Ethernet; nothing here touches that. This directory puts
the cluster's *control plane* on Solana Devnet so a judge can verify, without
trusting us, which nodes were serving, when the supervisor failed over, and who
answered each request.

```
program/            cluster_attest: native Rust program (no Anchor), one Cluster PDA + one Job PDA per request
../router/attest.py the attestor: runs inside the router when SOLANA_KEYPAIR is set
```

## What the program enforces

| Rule | Where |
|---|---|
| Only the authority that initialized the Cluster account can change it | `load_cluster` |
| A host must be registered before it can be in the worker set | `set_worker_set` → `UnknownNode` |
| Every worker-set change bumps the epoch; failed transitions do not | `set_worker_set` |
| A job can be committed once (its PDA must not exist yet → `AlreadyExists`) | `CommitJob` |
| Only the authority can close a Job PDA; its rent goes back to the authority | `CloseJob` |
| Pre-funding a PDA address does not block its creation (top-up + allocate + assign) | `create_pda` |
| At most 16 nodes, host ≤ 40 bytes, served_by ≤ 16 bytes | constants in `lib.rs` |

On-chain state (`Cluster`): authority, `model_hash` (sha256 of the model file name
and the header the supervisor read from it), `epoch`, `state`
(down/healthy/degraded/restarting), `nodes[] {host, active}`, `jobs_total`,
`jobs_local`, `last_job`. Each `Job`: cluster, 16-byte id (the router's
`X-Request-Id`), epoch, `served_by`, `result_hash` (sha256 of the answer text,
the `result_sha256` field the router logs).

## Build and deploy

```bash
sh -c "$(curl -sSfL https://release.anza.xyz/stable/install)"    # solana + cargo-build-sbf
cd program && cargo test && cargo build-sbf --arch v3            # target/deploy/cluster_attest.so
solana config set --url devnet && solana-keygen new && solana airdrop 2
solana program deploy target/deploy/cluster_attest.so            # prints the program id
```

Devnet has SBPF v3 enabled; a local `solana-test-validator` refuses v0 builds, so
`--arch v3` works on both.

## Run it

Set in `router/.env` and start the router as usual:

```
SOLANA_KEYPAIR=~/.config/solana/id.json     # Devnet keypair, the cluster's authority and fee payer
SOLANA_PROGRAM_ID=<id printed by solana program deploy>   # optional: defaults to program/target/deploy
SOLANA_RPC_URL=https://api.devnet.solana.com              # optional
```

The router then runs the attestor in a background thread: the supervisor status it
already polls becomes SetWorkerSet on every change, and every finished answer it logs
becomes CommitJob. It never sits in the request path; RPC failures are retried on the
next tick. `/stats` has a `solana` block with the Cluster account, its Explorer link,
counts and the last signature, and `router/attestations.jsonl` has every one.

```bash
cd ../router && ./attest.py --show        # decode the on-chain Cluster account
```

`attest.py` also runs standalone (`--status-url`, `--decision-log`) for a box that is
not the router.

## Tests

```bash
cd program && cargo build-sbf --arch v3 && cargo test          # unit: transitions, layout pins; mollusk: the .so in the Agave runtime
cd program && cargo test-sbf --arch v3                         # same, one command (builds the .so first)
cd ../router && pytest -q test_attest.py                       # encoders, parsing, loop against a fake chain
solana-test-validator -r &  solana program deploy ../solana/program/target/deploy/cluster_attest.so
ATTEST_RPC_URL=http://127.0.0.1:8899 pytest -q test_attest_e2e.py   # real program: rules and events on chain
```
