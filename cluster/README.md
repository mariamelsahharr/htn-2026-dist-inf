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
ansible-playbook -i inventory.ini bootstrap.yml -u pi
ansible-playbook -i inventory.ini root-model.yml -u pi
```

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
