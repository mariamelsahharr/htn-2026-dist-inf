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

Rerunning `bootstrap.yml` on a reflashed card brings that node back into the cluster.

## Ports

Workers: TCP 9998 (never probe or scan this port, see `supervisor/README.md`)

Root OpenAI-compatible API: TCP 9990

Supervisor status JSON for the router: TCP 9991 (`/status`, `/healthz`, `POST /restart`)

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

## Gemma 4 on llama.cpp RPC

distributed-llama cannot run Gemma 4: its converter only accepts `llama`, `mistral`,
`qwen3` and `qwen3_moe`, and the runtime has no GELU, sliding-window attention,
per-layer head sizes, post-attention/FFN norms or logit softcapping. llama.cpp can,
and its RPC backend spreads the layers over the Pis. Same supervisor, same ports
(9990 API, 9991 status), same `/status` contract, so the router and dashboard do not
change; `/status` gains `"backend": "llamacpp"`.

| | distributed-llama | llama.cpp RPC |
|---|---|---|
| Root | `dllama-api` | `llama-server --rpc ...` |
| Workers | `dllama worker` :9998 | `ggml-rpc-server` :50052 (the root runs one too) |
| Split | every tensor across nodes (tensor parallel) | whole layers per node (pipeline) |
| More nodes buy | speed and RAM | RAM only: tokens walk the nodes in turn |
| Valid node counts | must divide heads/dims (4 → 2 → 1) | any (8 → 7 → 6 ...) |
| Units | `dllama-worker`, `dllama-supervisor` | `llama-rpc`, `llama-supervisor` |

```bash
ansible-playbook -i inventory.ini bootstrap.yml -u pi            # once: agent, SSH key
ansible-playbook -i inventory.ini bootstrap-llamacpp.yml -u pi   # build llama.cpp, swap the units
ansible-playbook -i inventory.ini root-model-gemma4.yml -u pi    # 26B-A4B Q4_0 (14.6 GB) + start
ssh pi@192.168.50.10 journalctl -u llama-supervisor -f
curl -s http://192.168.50.10:9991/status | python3 -m json.tool
curl -s http://192.168.50.10:9990/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in five words."}],"max_tokens":32}'
```

Smoke-test with the 4.6 GB E4B first (fits the root alone, so it isolates build
problems from network ones); the command is at the top of `root-model-gemma4.yml`.

- **First load is slow.** The root streams every worker its share of the weights
  (14.6 GB over gigabit is minutes; over Wi-Fi do not try). `ggml-rpc-server -c`
  caches them under `~/.cache/llama.cpp/rpc` on each Pi, so a failover relaunch reads
  from the local card. Each card needs free space for its share; the root needs the
  whole file plus its share.
- **RAM floor.** `min_nodes` in `root-model-gemma4.yml` (default 4) lands in
  `/home/pi/llama-supervisor.env`. llama.cpp places layers in proportion to each
  node's free memory, so mixed 4/8 GB Pis are fine.
- **No auth on 50052.** Keep it on the wired cluster subnet only.
- `~/cluster up|down|status` follows whichever supervisor is enabled;
  `BACKEND=dllama` / `BACKEND=llamacpp` forces one.
- Back to dllama: `ansible-playbook -i inventory.ini bootstrap-llamacpp.yml -u pi -e llamacpp_state=off`

## Telemetry agent

Every Pi runs `agent/node_agent.py` (unit `dllama-agent.service`, port 9997): one JSON
document with CPU temperature, the firmware throttle flags, memory, load and clock,
read from sysfs and /proc. The supervisor fetches it for each alive worker and for
itself in the same pass as the liveness ping and puts it under `workers[].telemetry`
and `root.telemetry` in `/status`, so the router and dashboard see hardware state
without SSH. A dead worker shows `telemetry: null`. `--telemetry-port 0` turns it off.

```bash
curl -s http://192.168.50.11:9997/telemetry | python3 -m json.tool
python3 agent/node_agent.py --once          # on a Pi: print the document and exit
```
