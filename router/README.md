# Router

One OpenAI-compatible endpoint in front of the Pi cluster and the cloud tiers.

```
POST /v1/chat/completions   any OpenAI-compatible client (Warp, Aider, openai SDK, ...)
POST /v1/responses          Codex (it only speaks the Responses API since Feb 2026)
GET  /v1/models  /healthz  /stats
```

Every response carries `X-Served-By: cluster|baseten|openai|gemini|snowflake|cache`
and `X-Route-Reason`. Every decision is appended to `routing_decisions.jsonl`.

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

Every setting is a field on `Settings` in `app.py`; the env var is the field name in caps.

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
for chunk in c.chat.completions.create(model="auto", stream=True,
        messages=[{"role": "user", "content": "say hi"}]):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Force an upstream on stage with `X-Force-Upstream: baseten` (or any tier name).

## Tests

```bash
python -m pytest -q test_routing.py test_responses.py   # policy + Codex translation
python test_integration.py                              # fake cluster/cloud/supervisor, every failure path
```
