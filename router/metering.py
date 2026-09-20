"""
metering.py - what one answer cost and how fast it came, plus the running prefill
estimate the first-token budget is scaled by. All stamps are time.monotonic().
"""

import time
from typing import Any

from routing import Tier
from wire import Chunk, answer_text

RECENT_FIELDS = (
    "request_id",
    "served_by",
    "routed_to",
    "reason",
    "fallback",
    "stream",
    "latency_ms",
    "ttft_ms",
    "prompt_tokens",
    "gen_tokens",
    "tokens_source",
    "prefill_tps",
    "decode_tps",
    "tps",
    "nodes_active",
    "cluster_state",
    "ts",
)


def percentile(vals: list[float], pct: float) -> float:
    ordered = sorted(vals)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


class PrefillEstimate:
    """Exponentially weighted prompt-processing rate of the cluster, tok/s. Starts from
    the configured guess and follows measured answers, so the first-token budget tracks
    the real hardware instead of a constant."""

    def __init__(self, initial: float, alpha: float = 0.3) -> None:
        self.value = max(1.0, initial)
        self.alpha = alpha
        self.samples = 0

    def update(self, tps: float) -> None:
        if tps <= 0:
            return
        self.value = tps if self.samples == 0 else (1 - self.alpha) * self.value + self.alpha * tps
        self.samples += 1


class TokenMeter:
    """Token counts and rates for one answer. Exact when the upstream reports usage; the Pi API
    sends one token per chunk so its chunk count is exact too; other tiers fall back to chars/4."""

    def __init__(self, tier: Tier, prompt_estimate: int, t_first: float | None = None):
        self.tier, self.prompt_estimate, self.t_first = tier, prompt_estimate, t_first
        self.t_last: float | None = None
        self.chunks = self.chars = 0
        self.usage: dict[str, Any] | None = None

    def see(self, chunk: Chunk) -> None:
        if chunk.usage is not None:
            self.usage = chunk.usage
        if chunk.content is not None:
            self.t_first = self.t_first or chunk.at
            self.t_last = chunk.at
            self.chunks += 1
            self.chars += len(chunk.content)

    def see_completion(self, data: dict[str, Any]) -> None:
        usage = data.get("usage")
        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
            self.usage = usage
        self.chars += len(answer_text(data))

    def result(self, started: float) -> dict[str, Any]:
        """Rates count what the client saw, from the moment the attempt started (after any queue wait).
        A reasoning model's usage includes thinking tokens that never stream, so those are removed
        (reported, or inferred from the text length)."""
        out: dict[str, Any] = {}
        if self.usage:
            billed = int(self.usage["completion_tokens"])
            details = self.usage.get("completion_tokens_details") or {}
            hidden = int(details.get("reasoning_tokens") or 0)
            visible_estimate = round(self.chars / 4)
            if not hidden and self.chars and billed > 3 * visible_estimate + 16:  # chars/4 is crude; be sure
                hidden = billed - visible_estimate  # usage hides the reasoning; the text length is the honest count
            gen, source = billed - hidden, "usage"
            prompt = int(self.usage.get("prompt_tokens") or self.prompt_estimate)
            out["completion_tokens"] = billed
            if hidden:
                out["reasoning_tokens"] = hidden
        elif self.tier.is_local and self.chunks:
            gen, source, prompt = self.chunks, "chunks", self.prompt_estimate
        else:
            gen, source, prompt = round(self.chars / 4), "chars", self.prompt_estimate
        out.update({"gen_tokens": gen, "tokens_source": source, "prompt_tokens_actual": prompt})
        if self.t_first and self.t_first >= started:  # a sub-ms first token still rates
            out["prefill_tps"] = round(prompt / max(self.t_first - started, 0.001), 1)
        # a decode rate needs an interval between tokens: at least two chunks, spanning ≥ 50 ms;
        # a whole answer in one burst (some cloud tiers after a long wait) only gets the overall tps
        if self.t_first and self.t_last and self.chunks >= 2 and self.t_last - self.t_first >= 0.05 and gen > 1:
            out["decode_tps"] = round((gen - 1) / (self.t_last - self.t_first), 1)
        elapsed = time.monotonic() - started
        if elapsed > 0 and gen:
            out["tps"] = round(gen / elapsed, 1)
        return out
