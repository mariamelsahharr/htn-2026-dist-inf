# Raspberry Pi Distributed Inference Cluster

## Hardware

| Host | Role | RAM |
|---|---|---:|
| pi-node-3 | Root/API | 8 GB |
| pi-node-1 | Worker | 8 GB |
| pi-node-2 | Worker | 4 GB |
| pi-node-4 | Worker | 4 GB |

`pi-node-3` is the root because it has 8 GB RAM and active cooling.

A separate Raspberry Pi is used by Person 2 as the API routing/failover node.

## Current status

Working:

- Raspberry Pi OS Lite 64-bit on all inference Pis
- SSH between nodes
- distributed-llama builds successfully
- Qwen3 0.6B Q40 downloaded on pi-node-3
- OpenAI-compatible `/v1/models` endpoint works
- OpenAI-compatible `/v1/chat/completions` works
- 2-node distributed inference works over Wi-Fi
- systemd worker/root services configured
- cluster up/down/status wrapper working
- Ansible bootstrap created

Current limitation:

4-node distributed inference is unreliable over the temporary Wi-Fi/hotspot network. Individual 2-node configurations work, suggesting the software/model setup is correct. Retest 4-node mode after moving cluster traffic to gigabit Ethernet.

## Bootstrap

```bash
cp inventory.example.ini inventory.ini
ansible-playbook -i inventory.ini bootstrap.yml -u pi     # build, SSH key, units, supervisor
ansible-playbook -i inventory.ini root-model.yml -u pi    # model on the root
```

The model is one variable, `dllama_model` in `group_vars/all.yml` (a distributed-llama
launcher key; currently `qwen3_30b_a3b_q40`). Both root units and `root-model.yml` derive
`--model` / `--tokenizer` from it, so switching is one line or a flag:

```bash
ansible-playbook -i inventory.ini root-model.yml bootstrap.yml -u pi -e dllama_model=qwen3_0.6b_q40
```

Whether the model fits is decided at launch, not here: the supervisor's header math picks the
node counts and each Pi's RAM (8 GB on pi-node-1 and the root, 4 GB on pi-node-2 / pi-node-4)
decides whether its share loads. An OOM exit is reported as "likely out of memory" in
`/status` and the journal and retried with backoff; `dllama_min_nodes` is the RAM floor.

Rerunning `bootstrap.yml` on a reflashed card brings that node back into the cluster.
Adding nodes is an inventory edit (`[cluster]` and `[workers]`, see the comment at the top
of `bootstrap.yml`): the worker lists in the root units and `~/cluster` are rendered from
it. `cp supervisor.env.example supervisor.env` and fill in `SENTRY_DSN` /
`SUPERVISOR_TOKEN` to have them installed on the root (gitignored).

## Ports

Workers: TCP 9998 (never probe or scan this port, see `supervisor/README.md`)

Root OpenAI-compatible API: TCP 9990

Supervisor status JSON for the router: TCP 9991 (`/status`, `/healthz`, `POST /restart`;
POST needs a loopback client or `Authorization: Bearer $SUPERVISOR_TOKEN`)

Telemetry agent on every Pi: TCP 9997 (`/telemetry`)

## Failover

`supervisor/supervisor.py` replaces `dllama-root.service` on the root. It pings the
workers, relaunches `dllama-api` on the largest node set the model allows when one
dies (derived from the model header; 4 → 2 → 1 for four nodes, 8 → 4 → 2 → 1 for
eight), folds a returning worker back in, and publishes cluster state on port 9991. Install steps, the status contract, and a laptop rehearsal are in
`supervisor/README.md`. `~/cluster up|down|status` drives the supervisor unit by
default; `ROOT_UNIT=dllama-root ~/cluster up` is the rollback.

## Current known-good topology

```text
pi-node-3
    |
pi-node-1
```

## Expected wired topology

```text
             pi-node-3
                root
           /      |      \
pi-node-1     pi-node-2    pi-node-4
 worker        worker       worker
```

## Cluster control

Run on pi-node-3:

```bash
~/cluster up
~/cluster down
~/cluster restart
~/cluster status
```

## API test

```bash
curl http://pi-node-3.local:9990/v1/models
```

## Next work

1. Move inference nodes to gigabit Ethernet.
2. Verify 4-node inference.
3. Install the failover supervisor on the root (`supervisor/README.md`) and rehearse the unplug.
4. Benchmark throughput and thermals.
6. Replace Qwen3 0.6B with the intended larger demo model.

## Telemetry agent

Every Pi runs `agent/node_agent.py` (unit `dllama-agent.service`, port 9997): one JSON
document with CPU temperature, the firmware throttle flags, memory, load, clock,
`cpu_percent`, the dllama process RSS and network byte counters (psutil when
`python3-psutil` is installed, /proc otherwise), plus `worker_listening`,
`worker_connections` (port 9998 state from `/proc/net/tcp`, never a probe) and
`worker_unit` (`systemctl is-active dllama-worker`). The supervisor fetches it for
each pinging worker and for itself in the same pass as the liveness ping, uses the
socket fields to spot a dead worker process on a live Pi, and puts the document under
`workers[].telemetry` and `root.telemetry` in `/status`, so the router and dashboard
see hardware state without SSH. A dead worker shows `telemetry: null`.
`--telemetry-port 0` turns it off.

```bash
curl -s http://192.168.50.11:9997/telemetry | python3 -m json.tool
python3 agent/node_agent.py --once          # on a Pi: print the document and exit
```
