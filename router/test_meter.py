"""
test_meter.py - token counting and rates for one answer. Run: pytest -q
"""

import json
import time

from app import TokenMeter, _percentile
from routing import Tier

LOCAL = Tier("cluster", "llama", "http://pi/v1", is_local=True)
CLOUD = Tier("baseten", "glm", "https://x/v1", "k")


def chunk(text=None, usage=None, finish=None):
    obj = {"choices": [{"index": 0, "delta": {"content": text} if text is not None else {}, "finish_reason": finish}]}
    if usage:
        obj = {"choices": [], "usage": usage}
    return "data: " + json.dumps(obj)


def test_pi_stream_counts_one_token_per_chunk():
    t0 = time.time() - 1.0
    m = TokenMeter(LOCAL, prompt_estimate=20, t_first=t0 + 0.5)
    for w in ["The", " cat", " sat", " down", "."]:
        m.see(chunk(w))
        time.sleep(0.01)
    r = m.result(t0)
    assert (r["gen_tokens"], r["tokens_source"], r["prompt_tokens_actual"]) == (5, "chunks", 20)
    assert r["prefill_tps"] == 40.0           # 20 prompt tokens over the 0.5 s first-token wait
    assert r["decode_tps"] > 0 and r["tps"] > 0


def test_usage_chunk_wins_over_counting():
    m = TokenMeter(CLOUD, prompt_estimate=99)
    m.see(chunk("hello world, a long chunk with many tokens"))
    m.see(chunk(finish="stop"))
    m.see(chunk(usage={"prompt_tokens": 12, "completion_tokens": 9}))
    r = m.result(time.time() - 1)
    assert (r["gen_tokens"], r["tokens_source"], r["prompt_tokens_actual"]) == (9, "usage", 12)


def test_cloud_without_usage_estimates_from_chars():
    m = TokenMeter(CLOUD, prompt_estimate=5)
    m.see(chunk("x" * 40))
    r = m.result(time.time() - 1)
    assert (r["gen_tokens"], r["tokens_source"]) == (10, "chars")
    assert "decode_tps" not in r                      # a single chunk has no inter-token interval


def test_blocking_completion_uses_usage_then_text():
    m = TokenMeter(CLOUD, prompt_estimate=5)
    m.see_completion({"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    assert m.result(time.time() - 0.5)["gen_tokens"] == 2
    m2 = TokenMeter(LOCAL, prompt_estimate=5)
    m2.see_completion({"choices": [{"message": {"content": "x" * 8, "tool_calls": None}}]})
    assert (m2.result(time.time() - 0.5)["gen_tokens"], m2.result(time.time() - 0.5)["tokens_source"]) == (2, "chars")


def test_tool_call_deltas_count_as_tokens_without_text():
    m = TokenMeter(LOCAL, prompt_estimate=5)
    m.see("data: " + json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0}]}, "finish_reason": None}]}))
    m.see(chunk("ok"))
    assert m.result(time.time() - 1)["gen_tokens"] == 2


def test_non_content_lines_are_ignored():
    m = TokenMeter(LOCAL, prompt_estimate=5)
    m.see("data: [DONE]")
    m.see(": keepalive")
    m.see("data: not json")
    assert m.result(time.time() - 1)["gen_tokens"] == 0 and "tps" not in m.result(time.time() - 1)


def test_percentiles_are_nearest_rank():
    assert _percentile([5, 1, 3], 50) == 3 and _percentile([5, 1, 3], 95) == 5 and _percentile([7], 95) == 7
