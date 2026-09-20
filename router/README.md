# Router

One OpenAI-compatible endpoint in front of the Pi cluster and the cloud tiers.

```
POST /v1/chat/completions   any OpenAI-compatible client (Warp, Aider, openai SDK, ...)
POST /v1/responses          Codex (it only speaks the Responses API since Feb 2026)
GET  /v1/models  /healthz  /stats
```

Every response carries `X-Served-By: cluster|baseten|openai|gemini|snowflake|cache`,
`X-Route-Reason` and `X-Request-Id`. Every decision is appended to
`routing_decisions.jsonl` with that id and, once an answer finished, `result_sha256`
of its text; the on-chain attestor (`attest.py`) commits those.

## Run

```bash
pip install -r requirements.txt
cat > .env <<'EOF'                    # gitignored; read automatically at startup, env vars override it
BASETEN_API_KEY=...
LOCAL_BASE_URL=http://192.168.50.13:9990
STATUS_URL=http://192.168.50.13:9991/status
CLOUD_MODEL=zai-org/GLM-5.3-Fast
EOF
python app.py                         # :8000
```

Every setting is a field on `Settings` in `config.py`; the env var is the field name in caps.

```
app.py        HTTP: validation, endpoints, lifecycle      service.py    the Router: decide, relay, log, stats
upstreams.py  calls to the tiers, breaker, cluster cap    routing.py    the policy (pure, tested by property)
metering.py   token counts and rates                      wire.py       the OpenAI chunk format both ways
responses.py  Codex's Responses API translation           config.py     Settings
```

Extra tiers exist when key and model are set: `OPENAI_API_KEY` + `OPENAI_MODEL`,
`GEMINI_API_KEY` + `GEMINI_MODEL`, `SNOWFLAKE_API_KEY` + `SNOWFLAKE_MODEL` +
`SNOWFLAKE_BASE_URL=https://<account>.snowflakecomputing.com/api/v2/cortex/v1`.
`CLOUD_TIER_ORDER` sets fallback order. `TOOL_TIER=openai` pins tool requests to a
tier; leave it unset and the cluster tries tool calls itself.
`OPENAI_REASONING_EFFORT` (default `none`, which `gpt-5.6-luna` requires to accept
function tools on chat completions) and `GEMINI_REASONING_EFFORT` set that field on
every request to the tier; blank sends nothing. `<NAME>_TOOLS_MODEL` (and
`CLOUD_TOOLS_MODEL` for Baseten) is used instead of the tier's model when the
request carries tool definitions, so a cheap model can take plain chat and a
tool-capable one the agent turns; on Snowflake, `SNOWFLAKE_MODEL=llama3.1-8b`
with `SNOWFLAKE_TOOLS_MODEL=claude-haiku-4-5`.

At boot the router asks every cloud tier for its model list: a rejected key or dead
host opens that tier's breaker immediately (`/stats` → `tier_health`), and the list
becomes the tier's catalog, so `/v1/models` offers every chat model each provider
has and asking for one by name (`"model": "gemini-2.5-pro"`) pins that tier and
sends that exact model (`X-Route-Reason: model_pinned`). `/readyz` is 503 when
neither the cluster nor any cloud tier can answer.

The Pi API is single-threaded, so `LOCAL_CONCURRENCY` (1) requests run on the
cluster and up to `LOCAL_QUEUE_MAX` (2) wait; beyond that a request spills to the
cloud instead of queueing. `ATTEMPT_BUDGET_S` (90) caps the total time spent
walking the fallback chain. An identical prompt to the same model within
`ANSWER_CACHE_TTL` seconds (30; 0 disables) is answered from memory with
`X-Served-By: cache`, never for forced or tool requests. Bodies over
`MAX_BODY_BYTES` (1 MiB) get 413; a body without `messages` gets 400 with the reason.
The cluster's first-token budget scales the prompt by a prefill rate the router
learns from its own answers (`local_prefill_tps_estimate` in `/stats`).

A cloud tier that fails `BREAKER_FAILURES` times in a row (default 2) is skipped for
`BREAKER_COOLDOWN` seconds (default 30) so a dead Baseten on venue Wi-Fi does not
cost a connect timeout on every request; `/stats` shows `breakers_open_s`. A local
answer in prose to a `tool_choice: required` request counts as a miss and falls
through to the next tier.

Cluster health comes from the supervisor's status URL. `healthy` and `degraded`
both take traffic; `restarting`, `down` and a degraded cluster below
`MIN_LOCAL_NODES` (default 2, so root-alone goes to cloud) do not. When the status
URL is unreachable the router asks the root API directly, so it works before the
supervisor exists and still notices a dead root afterwards.

## Dashboard

`../dashboard` is a separate site that reads `/stats` and `/v1` from this router
(CORS is open and the routing headers are exposed). See its README.

## Token rates

Every logged answer carries `gen_tokens`, `tokens_source` and the rates behind the
dashboard: `prefill_tps` (prompt tokens over the first-token wait), `decode_tps`
(tokens after the first over the time to the last) and `tps` overall. Counts are
exact when the upstream reports usage (`STREAM_USAGE_TIERS`, default
`baseten,openai,gemini`, asks for `stream_options.include_usage`) and for the
cluster, whose API sends one token per chunk; anything else is chars/4 and says
so. `/stats` has `rates` (per-upstream means over the last 50 answers) and
`recent` (those answers).

## Solana

With `SOLANA_KEYPAIR=~/.config/solana/id.json` in `.env` the router also runs the
on-chain attestor (`attest.py`, program in `../solana`): worker-set changes and
finished answers are committed to the `cluster_attest` program on Devnet, `/stats`
has a `solana` block with the account, counts and recent signatures. Without the
keypair nothing changes.

## Codex

`~/.codex/config.toml` (user-level; Codex ignores provider blocks in project configs):

```toml
model = "auto"
model_provider = "picluster"

[model_providers.picluster]
name = "Pi cluster router"
base_url = "http://localhost:8000/v1"   # or the router Pi's address
env_key = "PICLUSTER_API_KEY"           # any value; the router does not check it
wire_api = "responses"
```

```bash
export PICLUSTER_API_KEY=x
codex "add a docstring to app.py"
```

The router translates Codex's Responses requests to chat completions and the
stream back into the events Codex parses (`responses.py`). Function tools pass
through; Codex's freeform `apply_patch` tool becomes a function with one string
input and comes back as a `custom_tool_call`. The Pi API only parses tool calls
on its non-streaming path, so tool requests routed to the cluster go blocking
and are re-emitted as a stream.

## Warp

Settings → AI → custom inference endpoint: URL `http://<public-host>/v1`,
any API key, model `auto`. Warp calls the endpoint from its backend, so the
router must be reachable from the internet (Cloudflare Tunnel or ngrok on the
router Pi).

## Any OpenAI SDK client

```python
from openai import OpenAI

c = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
for chunk in c.chat.completions.create(model="auto", stream=True, messages=[{"role": "user", "content": "say hi"}]):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Force an upstream on stage with `X-Force-Upstream: baseten` (or any tier name).

## Tests

```bash
uv run pytest router                                    # policy, properties, Codex translation, service
python test_integration.py                              # fake cluster/cloud/supervisor, every failure path
```
