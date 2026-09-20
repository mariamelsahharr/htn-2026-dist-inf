PiHive

Local LLM inference on Raspberry Pis, with a router that escalates to the cloud
when it has to.

```
router/      OpenAI-compatible router: Pis first, Baseten/OpenAI/Gemini/Snowflake when needed
dashboard/   the live front panel (separate site, talks to the router)
cluster/     what runs on the Pis: bootstrap playbook, supervisor, telemetry agent, tools
```

```bash
uv sync                                  # one environment for every Python component
uv run pytest                            # every suite (router, supervisor, tools, agent)
uv run ruff check . && uv run ty check router cluster/supervisor cluster/agent cluster/tools
```
