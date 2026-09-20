"""
config.py - every router knob, read from the environment and router/.env (env wins).

Field name in caps is the variable name. A cloud tier exists when its key and model
are set; Baseten also accepts CLOUD_* names.
"""

import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from routing import BASETEN, DEFAULT_CLOUD_ORDER, RouterConfig, Tier

# Base URL defaults per tier (Snowflake's is per-account).
EXTRA_TIER_DEFAULTS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "snowflake": "",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).with_name(".env")), extra="ignore", env_ignore_empty=True
    )

    local_base_url: str = "http://192.168.50.13:9990"  # pi-node-3, wired; mDNS is not trusted here
    local_model: str = "llama-3.2-3b-instruct"
    status_url: str = "http://192.168.50.13:9991/status"

    cloud_base_url: str = "https://inference.baseten.co/v1"
    cloud_api_key: str = Field("", validation_alias=AliasChoices("CLOUD_API_KEY", "BASETEN_API_KEY"))
    cloud_model: str = "zai-org/GLM-5.3"
    openai_base_url: str = EXTRA_TIER_DEFAULTS["openai"]
    openai_api_key: str = ""
    openai_model: str = ""
    openai_reasoning_effort: str = "none"  # gpt-5.6-luna rejects function tools on chat completions otherwise
    openai_tools_model: str = ""
    gemini_base_url: str = EXTRA_TIER_DEFAULTS["gemini"]
    gemini_api_key: str = ""
    gemini_model: str = ""
    gemini_reasoning_effort: str = ""
    gemini_tools_model: str = ""
    snowflake_base_url: str = ""
    snowflake_api_key: str = ""
    snowflake_model: str = ""
    snowflake_tools_model: str = ""  # e.g. claude-haiku-4-5; Cortex's Llama/Mistral reject tools
    cloud_tools_model: str = ""
    cloud_tier_order: str = ",".join(DEFAULT_CLOUD_ORDER)
    tool_tier: str = ""
    sentry_dsn: str = ""  # empty = telemetry fully off
    sentry_environment: str = "demo"
    stream_usage_tiers: str = "baseten,openai,gemini"  # verified to return usage on the final streamed chunk

    size_threshold: int = 2048
    first_token_timeout: float = 8.0
    read_timeout: float = 60.0
    local_read_timeout: float = 600.0  # a blocking cluster call returns nothing until generation ends
    local_prefill_tps: float = 25.0  # starting estimate; the router learns the real rate from answers
    connect_timeout: float = 3.0
    status_interval: float = 2.0
    min_local_nodes: int = 2  # a degraded cluster below this many nodes routes to cloud
    local_concurrency: int = 1  # the Pi API is single-threaded; more than this only queues
    local_queue_max: int = 2  # requests allowed to wait for the cluster before spilling to cloud
    attempt_budget_s: float = 90.0  # total time across fallback attempts before giving up
    breaker_failures: int = 2  # consecutive cloud-tier errors before it is skipped ...
    breaker_cooldown: float = 30.0  # ... for this many seconds
    answer_cache_ttl: float = 30.0  # identical prompt within this window is answered from memory; 0 disables
    max_body_bytes: int = 1_048_576
    decision_log: str = "routing_decisions.jsonl"
    decision_log_max_bytes: int = 20_000_000
    cache_file: str = "demo_cache.json"
    demo_fallback: bool = True

    @property
    def config(self) -> RouterConfig:
        tiers: dict[str, Tier] = {}
        usage_tiers = {n.strip().lower() for n in self.stream_usage_tiers.split(",")}
        if self.cloud_base_url and self.cloud_api_key:
            tiers[BASETEN] = Tier(
                BASETEN,
                self.cloud_model,
                self.cloud_base_url.rstrip("/"),
                self.cloud_api_key,
                tools_model=self.cloud_tools_model or None,
                usage_in_stream=BASETEN in usage_tiers,
            )
        for name in EXTRA_TIER_DEFAULTS:
            key, model, base = (
                getattr(self, f"{name}_api_key"),
                getattr(self, f"{name}_model"),
                getattr(self, f"{name}_base_url").rstrip("/"),
            )
            if key and model and base:
                tiers[name] = Tier(
                    name,
                    model,
                    base,
                    key,
                    reasoning_effort=getattr(self, f"{name}_reasoning_effort", "") or None,
                    tools_model=getattr(self, f"{name}_tools_model", "") or None,
                    usage_in_stream=name in usage_tiers,
                )
        local = self.local_base_url.rstrip("/")
        if not local.endswith("/v1"):
            local += "/v1"
        order = tuple(n.strip().lower() for n in self.cloud_tier_order.split(",") if n.strip())
        return RouterConfig(
            size_threshold=self.size_threshold,
            cloud_available=BASETEN in tiers,
            local_model=self.local_model,
            cloud_model=self.cloud_model,
            tiers=tiers,
            local_base_url=local,
            cloud_order=order,
            tool_tier=self.tool_tier.lower() or None,
        )


def load_settings() -> Settings:
    """ROUTER_NO_DOTENV=1 ignores router/.env (tests, or a box whose env is the whole config)."""
    if os.environ.get("ROUTER_NO_DOTENV"):
        return Settings(_env_file=None)
    return Settings()
