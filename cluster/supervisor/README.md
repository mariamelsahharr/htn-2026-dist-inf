# Supervisor: N-node failover for the distributed-llama root

`supervisor.py` runs on the root Pi instead of `dllama-root.service`. It launches
`dllama-api` itself, pings the workers, and when one dies it relaunches the root on
the largest node set the model still accepts. When the worker comes back it grows
the set again. Cluster state is published for the router.

## Which node counts are valid

Not hardcoded. distributed-llama slices every tensor evenly across nodes, so the
node count must divide the model's head count, KV dim, feed-forward dim and
vocabulary (the asserts in `nn-core.cpp`; the upstream README's "powers of two" is
just the common case). At startup the supervisor reads the `.m` header and computes
the list, logs it, and exposes it as `valid_node_counts` in `/status`:

| Model | Valid node counts |
|---|---|
| Llama 3.2 3B (24 heads) | 1, 2, 4, 8 |
| Qwen3 0.6B (16 heads) | 1, 2, 4, 8, 16 |

With 4 nodes that plays out as 4 → 2 → 1; with 8 as 8 → 4 → 2 → 1; with a model
whose dims divide by 3 it would use 3- and 6-node sets too. Check any new model with:

```bash
./supervisor.py --model path/to/dllama_model_x.m --workers a,b,c,d,e,f,g --print-node-counts
```

`--node-counts 1,2,4,8` pins the list if you ever want to be conservative, and if
the header cannot be read the supervisor falls back to powers of two and says so.

**RAM floor.** Node count does not know about memory. A 17 GB model on 8 nodes is
fine; the same model on the 2 nodes left after a cascade is not. `--min-nodes N`
makes the supervisor report `down` (and the router send everything to the cloud)
instead of launching a set that cannot hold the weights. Set it to the smallest
valid count whose per-node share fits the smallest Pi; it comes back up on its own
when enough nodes return. With no fit at all the supervisor never falls through
to a single node.

## llama.cpp backend

`--backend llamacpp` runs `llama-server` (pass it as `--dllama-bin`, a `.gguf` as
`--model`) over `ggml-rpc-server` workers on port 50052. What changes:

- every node count from 1 to N is valid (layers, not tensor slices), so losing one
  of eight leaves seven; `--node-counts` and `--min-nodes` still apply
- readiness is `GET /health` (503 until the weights are placed), not `/v1/models`
- survivors are reset with `systemctl restart llama-rpc`
- `--root-rpc` (default `127.0.0.1:50052`) puts the root's own RPC server first in
  `--rpc`; with `-ngl 99` the root would otherwise hold no layers at all
- `--alias` is the GGUF file stem, which is also the name the router derives from
  `root.model`

`--print-command` shows the exact `llama-server` line. Setup is in `../README.md`.

## The contract the router reads

```
GET http://<root>:9991/status
{
  "state": "healthy" | "restarting" | "degraded" | "down",
  "reason": "worker lost: pi-node-3.local",
  "nodes_total": 4, "nodes_active": 2,
  "valid_node_counts": [1, 2, 4, 8], "node_counts_source": "model header",
  "active_workers": ["pi-node-1.local:9998"],
  "workers": [{"host": "...", "alive": true, "in_set": true, ...}, ...],
  "root": {"pid": 1234, "load_seconds": 18.2, ...},
  "generation": 3, "restarts": 2, "events": [...]
}
GET  /healthz    200 while healthy or degraded, 503 otherwise
GET  /events     recent events
POST /restart    force a relaunch (demo control)
```

`status.example.json` next to this file is the full document. The supervisor,
router and metrics test suites all parse that same file, so a field rename fails a
test before it fails a demo.

The same JSON is written to `--status-file` (default `/tmp/dllama-supervisor-status.json`).
`healthy` means every configured worker is in the set. `degraded` means serving on
fewer nodes. The router sends everything except `healthy` to the cloud.

## Install

`cluster/bootstrap.yml` does all of it: builds distributed-llama on every node,
generates an SSH key on the root and authorizes it on the workers (the supervisor
restarts survivors over SSH before every relaunch), installs the worker units, and
installs `supervisor.py` as `/home/pi/supervisor.py` plus the supervisor unit on the
root with the plain root unit left disabled as the rollback.

```bash
cd cluster && ansible-playbook -i inventory.ini bootstrap.yml -u pi
ssh pi@<root> journalctl -u dllama-supervisor -f
```

Edit `--workers` in `systemd/dllama-supervisor.service` first: comma-separated
hosts in **priority order**. When the set shrinks the first ones are kept, so list
the 8 GB, best-cooled nodes first. `ping` to each worker must work from the root;
the wired IPs are safer than mDNS names.

Rollback to the plain root is one command: `ROOT_UNIT=dllama-root ~/cluster up`
(after `~/cluster down`). The old unit stays installed, just disabled.

## Rehearse on a laptop, no Pis

```bash
cd cluster/supervisor
python3 -m pytest -q             # ~20 s, covers the whole ladder on both backends
```

Or drive it by hand with the fake root:

```bash
FAKE_DEAD_FILE=/tmp/dead FAKE_LOAD_SECONDS=2 \
./supervisor.py --workers w1,w2,w3 --dllama-bin ./fake_dllama_api.py \
  --model m --tokenizer t --probe-cmd 'sh -c "! grep -qx {host} /tmp/dead 2>/dev/null"' \
  --no-reset-workers --status-port 9991 --api-port 9990 --interval 1 --rejoin-grace 3
# another terminal:
curl -s localhost:9991/status | python3 -m json.tool
echo w3 > /tmp/dead          # "pull the cable": watch it go restarting -> degraded on 2 nodes
: > /tmp/dead                # plug it back: healthy on 4 after the grace period
```

## Tuning

| Flag | Default | What it controls |
|---|---|---|
| `--interval` | 2 s | probe period |
| `--fail-after` / `--ok-after` | 2 / 2 | consecutive misses / hits before a worker flips dead / alive |
| `--rejoin-grace` | 10 s | how long a returned worker must stay up before the set grows |
| `--settle` | 3 s | pause between killing the root and relaunching |
| `--ready-timeout` | 600 s | how long a launch may take before it is declared failed |
| `--api-stall-timeout` | 180 s | restart if `/v1/models` has not answered this long (0 = off) |
| `--no-reset-workers` | | skip the SSH restart of survivors (only for the laptop fake) |

Recovery time = detection (`fail-after × interval`, ~4 s) + kill + reset (~2 s) +
settle + model load + weight transfer. The last two dominate and scale with model
size. Measure it with the unplug drill and put the median on the sticky note.

## Do not do these, they kill processes

- **Never TCP-connect to a worker's port 9998.** A worker only listens while waiting
  for a root and treats the first accepted connection as the root; a probe that hangs
  up makes it throw and exit. Liveness is ICMP ping for this reason.
- **Never bare-connect to the root's port 9990 without sending an HTTP request.** The
  root's HTTP reader raises an uncaught exception on an empty read and the process
  dies. The supervisor's health check sends a real `GET /v1/models`.
- Consequently: **never run nmap or a port scanner against the cluster.**

Both behaviours were confirmed in distributed-llama's source (`nn-network.cpp` and
`dllama-api.cpp`), not guessed.
