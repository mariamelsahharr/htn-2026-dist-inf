"""
test_meter.py - token counting and rates for one answer. Run: pytest -q
"""

import json
import time

from metering import TokenMeter, percentile
from routing import Tier
from wire import Chunk

LOCAL = Tier("cluster", "llama", "http://pi/v1", is_local=True)
CLOUD = Tier("baseten", "glm", "https://x/v1", "k")


def chunk(text=None, usage=None, finish=None, at=None) -> Chunk:
    obj = {"choices": [{"index": 0, "delta": {"content": text} if text is not None else {}, "finish_reason": finish}]}
    if usage:
        obj = {"choices": [], "usage": usage}
    return Chunk.parse("data: " + json.dumps(obj), at=at)


def test_pi_stream_counts_one_token_per_chunk():
    t0 = time.monotonic() - 1.0
    m = TokenMeter(LOCAL, prompt_estimate=20, t_first=t0 + 0.5)
    for i, w in enumerate(["The", " cat", " sat", " down", "."]):
        m.see(chunk(w, at=t0 + 0.5 + 0.1 * i))
    r = m.result(t0)
    assert (r["gen_tokens"], r["tokens_source"], r["prompt_tokens_actual"]) == (5, "chunks", 20)
    assert r["prefill_tps"] == 40.0  # 20 prompt tokens over the 0.5 s first-token wait
    assert r["decode_tps"] == 10.0 and r["tps"] > 0  # 4 intervals of 100 ms between 5 tokens


def test_usage_chunk_wins_over_counting():
    m = TokenMeter(CLOUD, prompt_estimate=99)
    m.see(chunk("hello world, a long chunk with many tokens"))
    m.see(chunk(finish="stop"))
    m.see(chunk(usage={"prompt_tokens": 12, "completion_tokens": 9}))
    r = m.result(time.monotonic() - 1)
    assert (r["gen_tokens"], r["tokens_source"], r["prompt_tokens_actual"]) == (9, "usage", 12)


def test_cloud_without_usage_estimates_from_chars():
    m = TokenMeter(CLOUD, prompt_estimate=5)
    m.see(chunk("x" * 40))
    r = m.result(time.monotonic() - 1)
    assert (r["gen_tokens"], r["tokens_source"]) == (10, "chars")
    assert "decode_tps" not in r  # a single chunk has no inter-token interval


def test_blocking_completion_uses_usage_then_text():
    m = TokenMeter(CLOUD, prompt_estimate=5)
    m.see_completion(
        {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    )
    assert m.result(time.monotonic() - 0.5)["gen_tokens"] == 2
    m2 = TokenMeter(LOCAL, prompt_estimate=5)
    m2.see_completion({"choices": [{"message": {"content": "x" * 8, "tool_calls": None}}]})
    r = m2.result(time.monotonic() - 0.5)
    assert (r["gen_tokens"], r["tokens_source"]) == (2, "chars")


def test_tool_call_deltas_count_as_tokens_without_text():
    m = TokenMeter(LOCAL, prompt_estimate=5)
    m.see(Chunk.of({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0}]}, "finish_reason": None}]}))
    m.see(chunk("ok"))
    assert m.result(time.monotonic() - 1)["gen_tokens"] == 2


def test_non_content_lines_are_ignored():
    m = TokenMeter(LOCAL, prompt_estimate=5)
    for line in ("data: [DONE]", ": keepalive", "data: not json", "data: [1, 2]"):
        parsed = Chunk.parse(line)
        assert parsed.obj is None and not parsed.is_content and parsed.error is None
        m.see(parsed)
    r = m.result(time.monotonic() - 1)
    assert r["gen_tokens"] == 0 and "tps" not in r


def test_chunk_reads_everything_once():
    c = Chunk.parse('data: {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]}')
    assert c.content == "hi" and c.finished and c.error is None and c.usage is None and c.obj
    err = Chunk.parse('data: {"error": {"message": "boom"}}')
    assert err.error and "boom" in err.error and not err.is_content
    usage = Chunk.parse('data: {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}')
    assert usage.usage == {"prompt_tokens": 1, "completion_tokens": 2} and not usage.is_content
    assert Chunk.parse('data: {"choices": [], "usage": {"prompt_tokens": 1}}').usage is None


def test_percentiles_are_nearest_rank():
    assert percentile([5, 1, 3], 50) == 3 and percentile([5, 1, 3], 95) == 5 and percentile([7], 95) == 7


def test_reasoning_tokens_do_not_count_as_generated():
    # reported by the upstream (OpenAI shape)
    m = TokenMeter(CLOUD, prompt_estimate=5)
    m.see(chunk("a short answer"))
    m.see(
        chunk(
            usage={"prompt_tokens": 5, "completion_tokens": 300, "completion_tokens_details": {"reasoning_tokens": 290}}
        )
    )
    r = m.result(time.monotonic() - 1)
    assert (r["gen_tokens"], r["completion_tokens"], r["reasoning_tokens"]) == (10, 300, 290)
    # hidden: billed far more than the text could hold, so the text length wins
    m = TokenMeter(CLOUD, prompt_estimate=5)
    m.see(chunk("x" * 120))
    m.see(chunk(usage={"prompt_tokens": 5, "completion_tokens": 211}))
    r = m.result(time.monotonic() - 1)
    assert r["gen_tokens"] == 30 and r["reasoning_tokens"] == 181 and r["tokens_source"] == "usage"
    # a plain model's usage is taken as is
    m = TokenMeter(CLOUD, prompt_estimate=5)
    m.see(chunk("x" * 120))
    m.see(chunk(usage={"prompt_tokens": 5, "completion_tokens": 33}))
    assert "reasoning_tokens" not in m.result(time.monotonic() - 1)


def test_one_burst_answer_has_no_decode_rate():
    m = TokenMeter(CLOUD, prompt_estimate=5, t_first=time.monotonic() - 0.001)
    m.see(chunk("the whole answer arrived at once in a single chunk"))
    m.see(chunk(usage={"prompt_tokens": 5, "completion_tokens": 200}))
    r = m.result(time.monotonic() - 10)
    assert "decode_tps" not in r and r["tps"] > 0


def test_rates_start_at_the_attempt_not_the_request():
    """Queue wait and failed earlier attempts are not the upstream's fault: rates count from `started`."""
    started = time.monotonic() - 2.0
    m = TokenMeter(LOCAL, prompt_estimate=100)
    m.see(chunk("a", at=started + 1.0))
    r = m.result(started)
    assert r["prefill_tps"] == 100.0  # 100 tokens over the 1 s between the attempt start and the first token
