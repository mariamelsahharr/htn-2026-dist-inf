"""
config.py - every router knob, read from the environment and router/.env (env wins).

Field name in caps is the variable name. A cloud tier exists when its key and model
are set; Baseten also accepts CLOUD_* names. Comma lists (CLOUD_TIER_ORDER,
STREAM_USAGE_TIERS, CORS_ORIGINS) are split here, not JSON-decoded.
"""

import os
from pathlib import Path
from typing import Annotated, Literal

import httpx2
from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from routing import BASETEN, DEFAULT_CLOUD_ORDER, RouterConfig, Tier

# Base URL defaults per tier (Snowflake's is per-account).
EXTRA_TIER_DEFAULTS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "snowflake": "",
}

Effort = Literal["", "none", "minimal", "low", "medium", "high", "xhigh"]  # "" sends nothing
CommaList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).with_name(".env")), extra="ignore", env_ignore_empty=True
    )

    local_base_url: str = "http://192.168.50.13:9990"  # pi-node-3, wired; mDNS is not trusted here
    local_model: str = "qwen3-30b-a3b"  # until the supervisor says what it loaded (adopt_local_model)
    status_url: str = "http://192.168.50.13:9991/status"

    cloud_base_url: str = "https://inference.baseten.co/v1"
    cloud_api_key: SecretStr = Field(SecretStr(""), validation_alias=AliasChoices("CLOUD_API_KEY", "BASETEN_API_KEY"))
    cloud_model: str = "zai-org/GLM-5.3"
    openai_base_url: str = EXTRA_TIER_DEFAULTS["openai"]
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = ""
    openai_reasoning_effort: Effort = "none"  # gpt-5.6-luna rejects function tools on chat completions otherwise
    openai_tools_model: str = ""
    gemini_base_url: str = EXTRA_TIER_DEFAULTS["gemini"]
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = ""
    gemini_reasoning_effort: Effort = ""
    gemini_tools_model: str = ""
    snowflake_base_url: str = ""
    snowflake_api_key: SecretStr = SecretStr("")
    snowflake_model: str = ""
    snowflake_tools_model: str = ""  # e.g. claude-haiku-4-5; Cortex's Llama/Mistral reject tools
    cloud_tools_model: str = ""
    cloud_tier_order: CommaList = list(DEFAULT_CLOUD_ORDER)
    tool_tier: str = ""
    sentry_dsn: SecretStr = SecretStr("")  # empty = telemetry fully off
    sentry_environment: str = "demo"
    stream_usage_tiers: CommaList = ["baseten", "openai", "gemini"]  # return usage on the final streamed chunk

    router_api_key: SecretStr = SecretStr("")  # when set, /v1/chat/completions and /v1/responses need it as a Bearer
    cors_origins: CommaList = ["*"]

    size_threshold: int = Field(3584, gt=0)  # prompt + max_tokens that still fits the cluster's 4096 context
    code_lines_threshold: int = Field(400, gt=0)  # fenced code lines before a request counts as complex
    max_local_turns: int = Field(40, gt=0)  # messages in the conversation before it counts as complex
    first_token_timeout: float = Field(8.0, gt=0)
    read_timeout: float = Field(60.0, gt=0)
    local_read_timeout: float = Field(600.0, gt=0)  # a blocking cluster call returns nothing until generation ends
    local_prefill_tps: float = Field(25.0, gt=0)  # starting estimate; the router learns the real rate from answers
    connect_timeout: float = Field(3.0, gt=0)
    status_interval: float = Field(2.0, gt=0)
    min_local_nodes: int = Field(2, ge=1)  # a degraded cluster below this many nodes routes to cloud
    local_concurrency: int = Field(1, ge=1)  # the Pi API is single-threaded; more than this only queues
    local_queue_max: int = Field(2, ge=0)  # requests allowed to wait for the cluster before spilling to cloud
    attempt_budget_s: float = Field(90.0, gt=0)  # total time across fallback attempts before giving up
    breaker_failures: int = Field(2, ge=1)  # consecutive cloud-tier errors before it is skipped ...
    breaker_cooldown: float = Field(30.0, gt=0)  # ... for this many seconds
    answer_cache_ttl: float = Field(30.0, ge=0)  # identical request within this window is answered from memory; 0 off
    max_body_bytes: int = Field(1_048_576, gt=0)
    solana_keypair: str = ""  # path to a Devnet keypair; empty = no on-chain attestation
    solana_rpc_url: str = "https://api.devnet.solana.com"
    solana_program_id: str = ""  # default: the id cargo build-sbf wrote under solana/program
    solana_interval: float = Field(2.0, gt=0)
    attestations_file: str = "attestations.jsonl"
    decision_log: str = "routing_decisions.jsonl"
    decision_log_max_bytes: int = Field(20_000_000, gt=0)
    cache_file: str = "demo_cache.json"
    demo_fallback: bool = True

    @field_validator("cloud_tier_order", "stream_usage_tiers", mode="before")
    @classmethod
    def _tier_list(cls, v: object) -> object:
        return [n.strip().lower() for n in v.split(",") if n.strip()] if isinstance(v, str) else v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _origin_list(cls, v: object) -> object:
        return [n.strip() for n in v.split(",") if n.strip()] if isinstance(v, str) else v

    def timeout(self, read: float) -> httpx2.Timeout:
        """The one httpx2 timeout shape every upstream call uses; only the read budget varies."""
        return httpx2.Timeout(connect=self.connect_timeout, read=read, write=10.0, pool=10.0)

    @property
    def config(self) -> RouterConfig:
        tiers: dict[str, Tier] = {}
        usage_tiers = set(self.stream_usage_tiers)
        if self.cloud_base_url and self.cloud_api_key.get_secret_value():
            tiers[BASETEN] = Tier(
                BASETEN,
                self.cloud_model,
                self.cloud_base_url.rstrip("/"),
                self.cloud_api_key.get_secret_value(),
                tools_model=self.cloud_tools_model or None,
                usage_in_stream=BASETEN in usage_tiers,
            )
        for name in EXTRA_TIER_DEFAULTS:
            key: SecretStr = getattr(self, f"{name}_api_key")
            model: str = getattr(self, f"{name}_model")
            base: str = getattr(self, f"{name}_base_url").rstrip("/")
            if key.get_secret_value() and model and base:
                tiers[name] = Tier(
                    name,
                    model,
                    base,
                    key.get_secret_value(),
                    reasoning_effort=getattr(self, f"{name}_reasoning_effort", "") or None,
                    tools_model=getattr(self, f"{name}_tools_model", "") or None,
                    usage_in_stream=name in usage_tiers,
                )
        local = self.local_base_url.rstrip("/")
        if not local.endswith("/v1"):
            local += "/v1"
        return RouterConfig(
            size_threshold=self.size_threshold,
            code_lines_threshold=self.code_lines_threshold,
            max_local_turns=self.max_local_turns,
            cloud_available=bool(tiers),
            local_model=self.local_model,
            cloud_model=self.cloud_model,
            tiers=tiers,
            local_base_url=local,
            cloud_order=tuple(self.cloud_tier_order),
            tool_tier=self.tool_tier.lower() or None,
        )


def load_settings() -> Settings:
    """ROUTER_NO_DOTENV=1 ignores router/.env (tests, or a box whose env is the whole config)."""
    if os.environ.get("ROUTER_NO_DOTENV"):
        return Settings(_env_file=None)
    return Settings()
