"""
test_integration.py - end-to-end proof that the cascade actually cascades.

Spins up fake LOCAL, CLOUD and STATUS endpoints whose behaviour you can switch
at runtime, points the router at them, and exercises every failure path:
refused connection, hung upstream (first-token timeout), and a stream that dies
halfway through.

Run:  python test_integration.py
No pytest needed. Exits non-zero if anything fails.
"""

import asyncio
import json
import os
import sys
import threading
import time

import httpx2 as httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

FAKE_PORT = 9100
ROUTER_PORT = 9101

# ----------------------------------------------------------------- fake stack

fake = FastAPI()
MODE = {"local": "ok", "cloud": "ok", "status": "healthy", "nodes": 4}


@fake.get("/control")
async def control(local: str = None, cloud: str = None, status: str = None, nodes: int = None):
    if local:
        MODE["local"] = local
    if cloud:
        MODE["cloud"] = cloud
    if status:
        MODE["status"] = status
    if nodes is not None:
        MODE["nodes"] = nodes
    return MODE


@fake.get("/status")
async def status():
    if MODE["status"] == "gone":   # no supervisor at all
        return JSONResponse({"error": "no supervisor"}, status_code=500)
    return {"state": MODE["status"], "nodes_active": MODE["nodes"], "nodes_total": 4}


@fake.get("/local/v1/models")
async def local_models():
    if MODE["local"] == "refuse":
        return JSONResponse({"error": "down"}, status_code=503)
    return {"object": "list", "data": [{"id": "local-3b", "object": "model"}]}


async def _sse(words, model, die_after=None, stall=False, usage=False):
    if stall:
        await asyncio.sleep(30)
    for i, w in enumerate(words):
        if die_after is not None and i >= die_after:
            raise RuntimeError("simulated upstream death")
        chunk = {"choices": [{"index": 0, "delta": {"content": w + " "},
                              "finish_reason": None}], "model": model}
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        await asyncio.sleep(0.01)
    yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
    if usage:   # what OpenAI/Baseten/Gemini send for stream_options.include_usage
        yield f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 11, 'completion_tokens': 40}})}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _handle(request: Request, which: str):
    body = await request.json()
    mode = MODE[which]
    if mode == "refuse":
        return JSONResponse({"error": "upstream is sad"}, status_code=503)
    words = [f"{which}{i}" for i in range(8)]
    if not body.get("stream"):
        if body.get("tools") and mode != "prose":   # "prose": ignore the tools like a weak local model
            # like dllama-api: tool calls only come back on the blocking path
            name = body["tools"][0]["function"]["name"]
            args = json.dumps({"input": "*** patch ***"} if name == "apply_patch" else {"command": ["ls"]})
            return JSONResponse({
                "id": "x", "object": "chat.completion", "model": which,
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "call_fake", "type": "function",
                                    "function": {"name": name, "arguments": args}}]}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            })
        return JSONResponse({
            "id": "x", "object": "chat.completion", "model": which,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": " ".join(words)}}],
        })
    return StreamingResponse(
        _sse(words, which,
             die_after=3 if mode == "die_midstream" else None,
             stall=(mode == "hang"),
             usage=bool((body.get("stream_options") or {}).get("include_usage"))),
        media_type="text/event-stream")


@fake.post("/local/v1/chat/completions")
async def local_ep(request: Request):
    return await _handle(request, "local")


@fake.post("/cloud/chat/completions")
async def cloud_ep(request: Request):
    return await _handle(request, "cloud")


# ------------------------------------------------------------------- harness

def serve(app, port):
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvicorn.Server(cfg).run()


def wait_for(url, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            httpx.get(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


def set_mode(**kw):
    httpx.get(f"http://127.0.0.1:{FAKE_PORT}/control", params=kw, timeout=5)
    time.sleep(0.1)


def stream_text(client, body, headers=None):
    """Returns (served_by, reason, concatenated_text)."""
    with client.stream("POST", f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions",
                       json={**body, "stream": True}, headers=headers or {},
                       timeout=60) as r:
        served = r.headers.get("X-Served-By")
        reason = r.headers.get("X-Route-Reason")
        out = []
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                obj = json.loads(p)
            except Exception:
                continue
            for c in obj.get("choices") or []:
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    out.append(piece)
        return served, reason, "".join(out)


def main():
    base = f"http://127.0.0.1:{FAKE_PORT}"
    os.environ.update({
        "ROUTER_NO_DOTENV": "1",          # never let a real router/.env leak real tiers into this run
        "BREAKER_FAILURES": "2",
        "BREAKER_COOLDOWN": "3",
        "LOCAL_BASE_URL": f"{base}/local",
        "CLOUD_BASE_URL": f"{base}/cloud",
        "CLOUD_API_KEY": "test-key",
        "STATUS_URL": f"{base}/status",
        "FIRST_TOKEN_TIMEOUT": "2",
        "STATUS_INTERVAL": "0.5",
        "SIZE_THRESHOLD": "2048",
        "DECISION_LOG": "/tmp/itest_decisions.jsonl",
        "CACHE_FILE": "/tmp/itest_cache.json",
        "LOCAL_MODEL": "local-3b",
        "CLOUD_MODEL": "cloud-70b",
    })
    with open("/tmp/itest_cache.json", "w") as fh:
        json.dump({}, fh)
    open("/tmp/itest_decisions.jsonl", "w").close()   # this run's decisions only

    threading.Thread(target=serve, args=(fake, FAKE_PORT), daemon=True).start()
    if not wait_for(f"{base}/status"):
        sys.exit("fake stack did not start")

    import app as router_app
    threading.Thread(target=serve, args=(router_app.app, ROUTER_PORT), daemon=True).start()
    if not wait_for(f"http://127.0.0.1:{ROUTER_PORT}/healthz"):
        sys.exit("router did not start")
    time.sleep(1.2)  # let the status poller take a reading

    c = httpx.Client()
    print("\n--- routing ---")

    set_mode(local="ok", cloud="ok", status="healthy")
    served, reason, text = stream_text(c, {"messages": [{"role": "user", "content": "hi"}]})
    check("small prompt, healthy cluster -> local", served == "cluster", f"{served}/{reason}")
    check("local content actually streamed", "local0" in text, text[:60])
    raw = c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions",
                 json={"messages": [{"role": "user", "content": "hi"}], "stream": True}, timeout=30).text
    check("exactly one [DONE] terminator per stream", raw.count("data: [DONE]") == 1, str(raw.count("data: [DONE]")))
    check("no cloud continuation after a finished local answer", "cloud" not in raw, raw[-120:])

    served, reason, _ = stream_text(c, {"messages": [{"role": "user", "content": "x" * 20000}]})
    check("huge prompt -> baseten", served == "baseten" and reason == "over_size_threshold",
          f"{served}/{reason}")

    served, reason, _ = stream_text(c, {"messages": [{"role": "user", "content": "hi"}],
                                        "model": "local-3b-heavy"})
    check("heavy model tag -> baseten", served == "baseten" and reason == "heavy_model_tag",
          f"{served}/{reason}")

    served, _, _ = stream_text(c, {"messages": [{"role": "user", "content": "hi"}]},
                               headers={"X-Escalate": "true"})
    check("escalate header -> baseten", served == "baseten", served)

    served, _, _ = stream_text(c, {"messages": [{"role": "user", "content": "x" * 20000}]},
                               headers={"X-Force-Upstream": "cluster"})
    check("force header overrides size rule", served == "cluster", served)

    print("\n--- cluster health ---")
    set_mode(status="restarting")
    time.sleep(1.2)
    served, reason, _ = stream_text(c, {"messages": [{"role": "user", "content": "hi"}]})
    check("supervisor says restarting -> baseten",
          served == "baseten" and reason == "cluster_restarting", f"{served}/{reason}")
    set_mode(status="degraded", nodes=2)
    time.sleep(1.2)
    served, reason, _ = stream_text(c, {"messages": [{"role": "user", "content": "hi"}]})
    check("degraded on 2 nodes -> still served by the cluster", served == "cluster", f"{served}/{reason}")
    set_mode(status="degraded", nodes=1)
    time.sleep(1.2)
    served, reason, _ = stream_text(c, {"messages": [{"role": "user", "content": "hi"}]})
    check("degraded to root alone (below MIN_LOCAL_NODES) -> baseten",
          served == "baseten" and reason == "cluster_degraded_below_min", f"{served}/{reason}")
    set_mode(status="healthy", nodes=4)
    time.sleep(1.2)

    set_mode(status="gone", local="ok")
    time.sleep(1.2)
    served, reason, _ = stream_text(c, {"messages": [{"role": "user", "content": "hi"}]})
    check("no supervisor but root answers -> still local", served == "cluster", f"{served}/{reason}")
    set_mode(status="gone", local="refuse")
    time.sleep(1.2)
    served, reason, _ = stream_text(c, {"messages": [{"role": "user", "content": "hi"}]})
    check("no supervisor and root dead -> cloud, no timeout wait",
          served == "baseten" and reason == "cluster_unreachable", f"{served}/{reason}")
    set_mode(status="healthy", local="ok")
    time.sleep(1.2)

    print("\n--- pre-commit fallback (client sees nothing) ---")
    set_mode(local="refuse")
    served, reason, text = stream_text(c, {"messages": [{"role": "user", "content": "hi"}]})
    check("local 503 -> transparent baseten retry",
          served == "baseten" and "fallback" in (reason or ""), f"{served}/{reason}")
    check("fallback response is complete", "cloud0" in text and "cloud7" in text, text[:60])

    set_mode(local="hang")
    t0 = time.time()
    served, reason, text = stream_text(c, {"messages": [{"role": "user", "content": "hi"}]})
    el = time.time() - t0
    check("local hangs -> first-token timeout -> baseten", served == "baseten", f"{served}/{reason}")
    check("timeout fired near FIRST_TOKEN_TIMEOUT", 1.5 < el < 6, f"{el:.1f}s")

    print("\n--- mid-stream death (bytes already sent) ---")
    set_mode(local="die_midstream")
    served, reason, text = stream_text(c, {"messages": [{"role": "user", "content": "hi"}]})
    check("stream still committed to cluster", served == "cluster", served)
    check("partial local output preserved", "local0" in text, text[:80])
    check("cloud continuation appended", "cloud" in text, text[:120])

    print("\n--- required tool call answered in prose ---")
    set_mode(local="prose", cloud="ok", status="healthy")
    time.sleep(1.2)
    req = {"messages": [{"role": "user", "content": "list files"}], "tool_choice": "required",
           "tools": [{"type": "function", "function": {"name": "shell", "parameters": {"type": "object", "properties": {}}}}]}
    r = c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions", json=req, timeout=30)
    check("tool_choice=required + local prose -> falls through to the cloud's tool call",
          r.headers.get("X-Served-By") == "baseten" and "+fallback" in r.headers.get("X-Route-Reason", "")
          and r.json()["choices"][0]["message"].get("tool_calls"), f"{r.headers.get('X-Served-By')} {r.headers.get('X-Route-Reason')}")
    r = c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions", json={**req, "tool_choice": "auto"}, timeout=30)
    check("tool_choice=auto + local prose -> prose is a valid answer, stays local",
          r.headers.get("X-Served-By") == "cluster" and r.json()["choices"][0]["message"].get("content"),
          r.headers.get("X-Served-By"))
    set_mode(local="ok")

    print("\n--- circuit breaker on a dead cloud tier ---")
    set_mode(local="refuse", cloud="refuse")
    for _ in range(2):   # two failures open the breaker (BREAKER_FAILURES=2)
        c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "unique-no-cache-1"}]}, timeout=30)
    set_mode(cloud="ok")
    t0 = time.time()
    r = c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "unique-no-cache-2"}]}, timeout=30)
    st = c.get(f"http://127.0.0.1:{ROUTER_PORT}/stats").json()
    check("open breaker skips the cloud tier without waiting on it",
          r.status_code == 502 and "baseten" in st.get("breakers_open_s", {}) and time.time() - t0 < 2,
          f"{r.status_code} {st.get('breakers_open_s')}")
    time.sleep(3.5)   # BREAKER_COOLDOWN=3
    r = c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "unique-no-cache-3"}]}, timeout=30)
    check("after the cooldown the tier is retried and serves", r.headers.get("X-Served-By") == "baseten",
          r.headers.get("X-Served-By"))
    set_mode(local="ok", cloud="ok")

    print("\n--- everything down ---")
    set_mode(local="refuse", cloud="refuse")
    r = c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "hi"}], "stream": True}, timeout=30)
    check("both upstreams down -> 502, not a hang", r.status_code == 502, str(r.status_code))

    router_app.app.state.router.st.cache[
        router_app.cache_key({"messages": [{"role": "user", "content": "demo prompt"}]})] = \
        "cached demo answer here"
    served, reason, text = stream_text(c, {"messages": [{"role": "user", "content": "Demo   Prompt"}]})
    check("demo cache serves when all else fails",
          served == "cache" and "cached demo answer" in text, f"{served}/{text[:40]}")

    print("\n--- non-streaming clients ---")
    set_mode(local="ok", cloud="ok")
    time.sleep(3.5)   # "everything down" above tripped the cloud breaker; let BREAKER_COOLDOWN pass
    r = c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "hi"}]}, timeout=30)
    check("blocking request works", r.status_code == 200 and r.headers.get("X-Served-By") == "cluster",
          r.headers.get("X-Served-By"))
    check("blocking body is OpenAI-shaped",
          r.json()["choices"][0]["message"]["content"].startswith("local"))

    set_mode(local="refuse")
    r = c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "hi"}]}, timeout=30)
    check("blocking falls back to cloud", r.headers.get("X-Served-By") == "baseten",
          r.headers.get("X-Served-By"))
    set_mode(local="ok")

    print("\n--- tool calls on the chat endpoint, both modes ---")
    set_mode(local="ok", cloud="ok", status="healthy")
    time.sleep(1.2)
    tool_body = {"messages": [{"role": "user", "content": "list files"}],
                 "tools": [{"type": "function", "function": {"name": "shell", "parameters": {"type": "object", "properties": {}}}}]}
    r = c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions", json=tool_body, timeout=30)
    msg = r.json()["choices"][0]["message"]
    check("blocking chat with tools -> tool_calls in the message, served locally",
          r.headers.get("X-Served-By") == "cluster" and msg.get("tool_calls", [{}])[0].get("function", {}).get("name") == "shell",
          f"{r.headers.get('X-Served-By')} {str(msg)[:80]}")
    raw = c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions", json={**tool_body, "stream": True}, timeout=30)
    check("streaming chat with tools -> tool_calls delta streamed from the cluster's blocking path",
          raw.headers.get("X-Served-By") == "cluster" and '"tool_calls"' in raw.text and raw.text.count("data: [DONE]") == 1,
          raw.text[:120])

    print("\n--- Responses API (Codex) ---")
    set_mode(local="ok", cloud="ok", status="healthy")
    time.sleep(1.2)

    def responses_events(body, headers=None):
        with c.stream("POST", f"http://127.0.0.1:{ROUTER_PORT}/v1/responses",
                      json={**body, "stream": True}, headers=headers or {}, timeout=60) as r:
            served = r.headers.get("X-Served-By")
            evs = []
            for line in r.iter_lines():
                if line.startswith("data:"):
                    evs.append(json.loads(line[5:]))
            return served, evs

    served, evs = responses_events({"model": "auto", "instructions": "be brief", "input": "hi"})
    kinds = [e["type"] for e in evs]
    check("responses: text request served locally", served == "cluster", served)
    check("responses: created -> deltas -> item.done -> completed",
          kinds[0] == "response.created" and "response.output_text.delta" in kinds
          and "response.output_item.done" in kinds and kinds[-1] == "response.completed", str(kinds[:4]))
    done = next((e for e in evs if e["type"] == "response.output_item.done"), {})
    check("responses: message item carries the local text",
          done.get("item", {}).get("content", [{}])[0].get("text", "").startswith("local0"), str(done)[:100])

    served, evs = responses_events({"model": "auto", "input": "list files", "tools": [
        {"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {}}}]})
    items = [e["item"] for e in evs if e["type"] == "response.output_item.done"]
    check("responses: tool request still served by the cluster (blocking path)", served == "cluster", served)
    check("responses: function_call item with call_id/name/arguments",
          items and items[0]["type"] == "function_call" and items[0]["call_id"] == "call_fake"
          and json.loads(items[0]["arguments"]) == {"command": ["ls"]}, str(items)[:120])

    served, evs = responses_events({"model": "auto", "input": "patch it", "tools": [
        {"type": "custom", "name": "apply_patch", "description": "p", "format": {"type": "grammar", "syntax": "lark", "definition": "x"}}]})
    items = [e["item"] for e in evs if e["type"] == "response.output_item.done"]
    check("responses: custom tool comes back as custom_tool_call with raw input",
          items and items[0]["type"] == "custom_tool_call" and items[0]["input"] == "*** patch ***", str(items)[:120])

    r = c.post(f"http://127.0.0.1:{ROUTER_PORT}/v1/responses",
               json={"model": "auto", "input": "hi"}, timeout=30)
    check("responses: non-streaming returns a response object",
          r.status_code == 200 and r.json()["object"] == "response" and r.json()["output"][0]["type"] == "message",
          str(r.json())[:100])

    set_mode(local="refuse")
    served, evs = responses_events({"model": "auto", "input": "hi"})
    check("responses: local down -> transparent cloud fallback",
          served == "baseten" and evs[-1]["type"] == "response.completed", f"{served}/{evs[-1]['type'] if evs else None}")
    set_mode(local="ok")

    print("\n--- metadata endpoints ---")
    models = c.get(f"http://127.0.0.1:{ROUTER_PORT}/v1/models").json()
    ids = [m["id"] for m in models["data"]]
    check("/v1/models lists local + heavy", "local-3b" in ids and "local-3b-heavy" in ids, str(ids))
    st = c.get(f"http://127.0.0.1:{ROUTER_PORT}/stats").json()
    check("/stats reports pct_local", st.get("pct_local") is not None, json.dumps(st)[:120])
    check("/stats counts fallbacks", sum(st.get("fallbacks", {}).values()) > 0, str(st.get("fallbacks")))

    print("\n--- token rates ---")
    set_mode(local="ok", cloud="ok", status="healthy")
    stream_text(c, {"messages": [{"role": "user", "content": "rate me"}]})
    recs = [json.loads(line) for line in open("/tmp/itest_decisions.jsonl").read().strip().split("\n")]
    rec = recs[-1]
    check("cluster stream counts one token per chunk",
          rec["served_by"] == "cluster" and rec["gen_tokens"] == 8 and rec["tokens_source"] == "chunks", str(rec)[:160])
    check("cluster stream has decode and prefill rates", rec.get("decode_tps", 0) > 0 and rec.get("prefill_tps", 0) > 0,
          f"{rec.get('decode_tps')} {rec.get('prefill_tps')}")
    stream_text(c, {"messages": [{"role": "user", "content": "rate me"}]}, headers={"X-Force-Upstream": "baseten"})
    rec = json.loads(open("/tmp/itest_decisions.jsonl").read().strip().split("\n")[-1])
    check("cloud stream asks for usage and takes the exact count",
          rec["served_by"] == "baseten" and rec["gen_tokens"] == 40 and rec["tokens_source"] == "usage"
          and rec["prompt_tokens_actual"] == 11, str(rec)[:160])
    st = c.get(f"http://127.0.0.1:{ROUTER_PORT}/stats").json()
    check("/stats has per-upstream rates and the recent answers",
          st["rates"]["cluster"]["decode_tps"] > 0 and st["rates"]["baseten"]["n"] >= 1 and len(st["recent"]) > 5,
          json.dumps(st.get("rates"))[:160])

    lines = open("/tmp/itest_decisions.jsonl").read().strip().split("\n")
    check("decision log written as JSONL", len(lines) > 10, f"{len(lines)} lines")
    rec = json.loads(lines[0])
    check("decision log has the pitch fields",
          all(k in rec for k in ("routed_to", "reason", "prompt_tokens", "latency_ms")), str(rec)[:100])

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
