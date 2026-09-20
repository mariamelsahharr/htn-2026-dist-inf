#!/usr/bin/env python3
"""
loadtest.py - concurrency / latency / throughput harness for any OpenAI-compatible
chat-completions endpoint (the Pi cluster directly, or the router in front of it).

Measures, per request: time-to-first-token, generation tokens/sec, total latency,
and which upstream served it (reads the X-Served-By header the router sets).
Prints p50/p95 and writes every request to JSONL so you can diff runs.

Zero third-party dependencies.

Typical uses:
    # quick sanity check, 4 at a time, 20 requests
    ./loadtest.py --url http://192.168.1.10:9990/v1/chat/completions -c 4 -n 20

    # the 15-minute sustained run for the straggler hunt
    ./loadtest.py --url http://192.168.1.10:9990/v1/chat/completions \
        -c 4 --duration 900 --prompt-tokens 400 --max-tokens 200 \
        --out sustained1.jsonl --tps-file /tmp/dllama_tps.json

    # through the router, to prove escalation
    ./loadtest.py --url http://localhost:8000/v1/chat/completions \
        -c 2 -n 10 --prompt-tokens 4000
"""

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------- prompt corpus

# Shaped like what a coding agent actually sends, so the numbers mean something.
TASKS = [
    "Write a bash one-liner that finds every file over 100MB under the current directory and prints size and path, "
    "sorted.",
    "Explain what this systemd unit does and why it might fail to restart after a reboot.",
    "Refactor this function so the error handling is not duplicated. Keep the signature.",
    "Give me a Python snippet that retries an HTTP call with exponential backoff and a jitter.",
    "What is the difference between a bind mount and a volume, in two sentences?",
    "Write a regex that matches an IPv4 address in a log line and capture the octets.",
    "Summarise what changed in this diff in three bullet points.",
    "Convert this curl command into a Python requests call.",
    "Why would a process show 100% CPU on one core but the load average stays under 1?",
    "Write a Makefile target that builds, tests, and fails loudly on a non-zero exit.",
]

FILLER = (
    "The service reads a configuration file at startup and validates each field before "
    "binding to the listen address. Workers are spawned per core and share a queue. "
    "Metrics are flushed every ten seconds to a local collector. When a worker exits "
    "unexpectedly the supervisor logs the exit code and restarts it after a short delay. "
)


def build_prompt(target_tokens, rng):
    """Approximate: ~0.75 words per token for English."""
    task = rng.choice(TASKS)
    words_needed = int(target_tokens * 0.75) - len(task.split())
    if words_needed <= 0:
        return task
    filler_words = FILLER.split()
    chunk = []
    while len(chunk) < words_needed:
        chunk.extend(filler_words)
    context = " ".join(chunk[:words_needed])
    return f"Context:\n{context}\n\nTask: {task}"


# ---------------------------------------------------------------- stats helpers


def pct(values, p):
    if not values:
        return None
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    k = (len(vs) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)


class Live:
    """Thread-safe rolling counters + the sidecar file metrics.py reads."""

    def __init__(self, tps_file=None):
        self.lock = threading.Lock()
        self.completed = 0
        self.errors = 0
        self.inflight = 0
        self.tokens_window = []  # (timestamp, tokens)
        self.tps_file = tps_file

    def add_tokens(self, n):
        with self.lock:
            self.tokens_window.append((time.time(), n))

    def rolling_tps(self, window=10.0):
        now = time.time()
        with self.lock:
            self.tokens_window = [(t, n) for (t, n) in self.tokens_window if now - t <= window]
            total = sum(n for _, n in self.tokens_window)
        return total / window

    def write_sidecar(self):
        if not self.tps_file:
            return
        try:
            tmp = self.tps_file + ".tmp"
            with Path(tmp).open("w") as fh:
                json.dump(
                    {
                        "tps": round(self.rolling_tps(), 2),
                        "inflight": self.inflight,
                        "completed": self.completed,
                        "errors": self.errors,
                        "ts": time.time(),
                    },
                    fh,
                )
            Path(tmp).replace(self.tps_file)
        except Exception:
            pass


# ---------------------------------------------------------------- one request


def one_request(url, api_key, model, prompt, max_tokens, temperature, timeout, live):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    rec = {
        "t_start": time.time(),
        "start_iso": datetime.now(UTC).isoformat(),
        "prompt_chars": len(prompt),
        "max_tokens": max_tokens,
        "ok": False,
        "ttft_s": None,
        "total_s": None,
        "gen_tokens": 0,
        "gen_tps": None,
        "served_by": None,
        "error": None,
        "finish_reason": None,
    }

    t0 = time.time()
    first = None
    chunks = 0
    usage_tokens = None
    text_len = 0

    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rec["served_by"] = resp.headers.get("X-Served-By")
            rec["http_status"] = resp.status
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens", usage_tokens)
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if choices[0].get("finish_reason"):
                    rec["finish_reason"] = choices[0]["finish_reason"]
                if piece:
                    if first is None:
                        first = time.time()
                        rec["ttft_s"] = first - t0
                    chunks += 1
                    text_len += len(piece)
                    live.add_tokens(1)
        rec["ok"] = True
    except urllib.error.HTTPError as e:
        rec["served_by"] = e.headers.get("X-Served-By")  # the router labels failures too
        try:
            rec["error"] = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
        except Exception:
            rec["error"] = f"HTTP {e.code}"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:200]

    end = time.time()
    rec["total_s"] = end - t0
    rec["gen_tokens"] = usage_tokens if usage_tokens is not None else chunks
    rec["chunks"] = chunks
    rec["text_chars"] = text_len
    if rec["ok"] and first and rec["gen_tokens"] > 1:
        gen_window = end - first
        if gen_window > 0:
            rec["gen_tps"] = (rec["gen_tokens"] - 1) / gen_window
    return rec


# ---------------------------------------------------------------- runner


def main():
    ap = argparse.ArgumentParser(description="Load test an OpenAI-compatible endpoint.")
    ap.add_argument("--url", required=True, help="full chat-completions URL")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", default="llama-3.2-3b-instruct")
    ap.add_argument("-c", "--concurrency", type=int, default=4)
    ap.add_argument("-n", "--requests", type=int, default=None, help="total requests (burst mode)")
    ap.add_argument("--duration", type=float, default=None, help="seconds to keep load on (sustained mode)")
    ap.add_argument("--prompt-tokens", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--think-time", type=float, default=0.0, help="seconds a worker pauses between requests")
    ap.add_argument("--out", default=None, help="JSONL of per-request records")
    ap.add_argument("--tps-file", default=None, help="sidecar JSON for metrics.py, e.g. /tmp/dllama_tps.json")
    ap.add_argument("--label", default="", help="free-text tag stored in each record")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if not args.requests and not args.duration:
        args.requests = args.concurrency * 5

    live = Live(args.tps_file)
    records = []
    rec_lock = threading.Lock()
    out_fh = Path(args.out).open("a") if args.out else None  # noqa: SIM115 - closed at exit below
    stop = threading.Event()

    deadline = time.time() + args.duration if args.duration else None
    remaining = {"n": args.requests if args.requests else None}

    def take_job():
        if stop.is_set():
            return False
        if deadline is not None:
            return time.time() < deadline
        with rec_lock:
            if remaining["n"] is None or remaining["n"] <= 0:
                return False
            remaining["n"] -= 1
            return True

    def worker(wid):
        wrng = random.Random((args.seed or 0) + wid)
        while take_job():
            prompt = build_prompt(args.prompt_tokens, wrng)
            with live.lock:
                live.inflight += 1
            rec = one_request(
                args.url, args.api_key, args.model, prompt, args.max_tokens, args.temperature, args.timeout, live
            )
            rec["worker"] = wid
            rec["label"] = args.label
            rec["target_prompt_tokens"] = args.prompt_tokens
            with live.lock:
                live.inflight -= 1
                live.completed += 1
                if not rec["ok"]:
                    live.errors += 1
            with rec_lock:
                records.append(rec)
                if out_fh:
                    out_fh.write(json.dumps(rec) + "\n")
                    out_fh.flush()
            if args.think_time:
                time.sleep(args.think_time)

    def reporter():
        t0 = time.time()
        while not stop.is_set():
            live.write_sidecar()
            el = time.time() - t0
            total = args.requests if args.requests else "-"
            done = live.completed
            ttfts = [r["ttft_s"] for r in records if r.get("ttft_s")]
            p50 = pct(ttfts, 50)
            msg = (
                f"\r[{int(el) // 60:02d}:{int(el) % 60:02d}] done={done}/{total} "
                f"inflight={live.inflight} err={live.errors} "
                f"tok/s(10s)={live.rolling_tps():5.1f} "
                f"ttft_p50={p50:.2f}s"
                if p50
                else f"\r[{int(el) // 60:02d}:{int(el) % 60:02d}] done={done}/{total} "
                f"inflight={live.inflight} err={live.errors} ..."
            )
            sys.stderr.write(msg.ljust(90))
            sys.stderr.flush()
            stop.wait(2)

    print(f"target : {args.url}")
    print(
        f"mode   : {'sustained ' + str(args.duration) + 's' if args.duration else str(args.requests) + ' requests'}"
        f"  concurrency={args.concurrency}  prompt~{args.prompt_tokens}tok  max_tokens={args.max_tokens}"
    )
    print()

    rep = threading.Thread(target=reporter, daemon=True)
    rep.start()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.concurrency)]
    wall0 = time.time()
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\ninterrupted, draining in-flight requests...", file=sys.stderr)
        stop.set()
        for t in threads:
            t.join(timeout=args.timeout)
    wall = time.time() - wall0
    stop.set()
    live.write_sidecar()  # final inflight=0 so metrics.py stops showing a stale rate
    time.sleep(0.1)
    sys.stderr.write("\r".ljust(95) + "\r")
    if out_fh:
        out_fh.close()

    # ------------------------------------------------------------ summary
    ok = [r for r in records if r["ok"]]
    bad = [r for r in records if not r["ok"]]
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s")]
    tpss = [r["gen_tps"] for r in ok if r.get("gen_tps")]
    totals = [r["total_s"] for r in ok]
    gen_total = sum(r["gen_tokens"] for r in ok)

    def line(k, v):
        print(f"  {k:<26} {v}")

    print("=" * 62)
    print(f"RESULTS  {args.label or ''}".rstrip())
    print("=" * 62)
    line("requests ok / failed", f"{len(ok)} / {len(bad)}")
    line("wall clock", f"{wall:.1f}s")
    line("aggregate throughput", f"{gen_total / wall:.1f} tok/s across {args.concurrency} streams")
    line("requests/min", f"{len(ok) / wall * 60:.1f}")
    if ttfts:
        line("TTFT p50 / p95 / max", f"{pct(ttfts, 50):.2f}s / {pct(ttfts, 95):.2f}s / {max(ttfts):.2f}s")
    if tpss:
        line("per-stream tok/s p50/p95", f"{pct(tpss, 50):.1f} / {pct(tpss, 95):.1f}")
        line("per-stream tok/s min", f"{min(tpss):.1f}")
    if totals:
        line("total latency p50 / p95", f"{pct(totals, 50):.1f}s / {pct(totals, 95):.1f}s")
    if ok:
        line("mean gen tokens", f"{statistics.mean(r['gen_tokens'] for r in ok):.0f}")

    served = Counter(r.get("served_by") or "unset" for r in records)
    if len(served) > 1 or "unset" not in served:
        print()
        print("  served by (X-Served-By):")
        for k, v in served.most_common():
            print(f"    {k:<20} {v:>4}  ({100 * v / max(len(records), 1):.0f}%)")

    if bad:
        print()
        print("  failures:")
        for k, v in Counter(r["error"][:60] for r in bad).most_common(5):
            print(f"    {v:>4}x {k}")

    if args.out:
        print(f"\n  per-request records -> {args.out}")
    print()


if __name__ == "__main__":
    main()
