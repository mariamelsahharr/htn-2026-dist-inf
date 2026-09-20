"""
schemas.py - the request bodies the router validates. Only the fields the router reads
are typed; everything else passes through to the upstream untouched. Dump with
exclude_unset=True: an explicit null (an assistant turn with tool_calls and
`content: null`) must reach the upstream as sent.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: str | list[Any] | None = None


class ChatRequest(BaseModel):
    """POST /v1/chat/completions."""

    model_config = ConfigDict(extra="allow")
    messages: list[ChatMessage] = Field(min_length=1)
    model: str = "auto"
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=32_768)
    temperature: float | None = Field(default=None, ge=0, le=2)
    tools: list[Any] | None = None


class ResponsesRequest(BaseModel):
    """POST /v1/responses: the subset Codex sends that responses.responses_to_chat reads."""

    model_config = ConfigDict(extra="allow")
    input: str | list[Any]
    model: str = "auto"
    stream: bool = False
    instructions: str | None = None
    tools: list[Any] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    max_output_tokens: int | None = Field(default=None, ge=1, le=32_768)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
