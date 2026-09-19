#!/usr/bin/env python3
"""
fake_dllama_api.py - laptop stand-in for dllama-api, same flags. Sleeps
FAKE_LOAD_SECONDS, retries forever while any worker is listed in FAKE_DEAD_FILE,
then serves /v1/models and streaming /v1/chat/completions single-threaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

WORDS = "a hash map stores key value pairs in buckets chosen by hashing the key".split()


def dead_hosts() -> set:
    path = os.environ.get("FAKE_DEAD_FILE", "")
    if not path or not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


class Handler(BaseHTTPRequestHandler):
    model_path = "fake"

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [
                {"id": os.path.basename(self.model_path), "object": "model", "owned_by": "fake"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        tokens = min(int(body.get("max_tokens") or 32), 200)
        tps = float(os.environ.get("FAKE_TPS", "8"))
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(tokens):
                chunk = {"id": "fake", "object": "chat.completion.chunk", "model": "fake",
                         "choices": [{"index": 0, "delta": {"content": WORDS[i % len(WORDS)] + " "},
                                      "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(1.0 / tps)
            self.wfile.write(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            return
        text = " ".join(WORDS[i % len(WORDS)] for i in range(tokens))
        time.sleep(tokens / tps)
        resp = json.dumps({"id": "fake", "object": "chat.completion", "model": "fake",
                           "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": text}}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": tokens}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9990)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--buffer-float-type", default="q80")
    ap.add_argument("--nthreads", type=int, default=4)
    ap.add_argument("--workers", nargs="*", default=[])
    args = ap.parse_args()

    load = float(os.environ.get("FAKE_LOAD_SECONDS", "1.5"))
    retry = float(os.environ.get("FAKE_RETRY_SECONDS", "3"))
    hosts = [w.split(":")[0] for w in args.workers]
    print(f"fake dllama-api: loading {args.model} for {load}s, workers={args.workers}", flush=True)
    time.sleep(load)
    while True:
        dead = [h for h in hosts if h in dead_hosts()]
        if not dead:
            break
        print(f"🚨 Connection error: Cannot connect to {dead[0]}", flush=True)
        print(f"🔄 Retrying in {retry} seconds...", flush=True)
        time.sleep(retry)
    Handler.model_path = args.model
    srv = HTTPServer((args.host, args.port), Handler)
    print(f"Server URL: http://localhost:{args.port}/v1/", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
