#!/usr/bin/env python3
"""
warm_cache.py - pre-generate answers for the demo prompts.

Venue Wi-Fi will betray you. Run this while the network is good, and the router
can serve those exact prompts from disk if both upstreams are unreachable on
stage. Cached responses are served with `X-Served-By: cache` and logged as such,
so you can be straight with judges if it ever fires.

    ./warm_cache.py --prompts demo_prompts.txt --url http://pi-node-5.local:8000/v1/chat/completions

demo_prompts.txt is one prompt per line, blank lines and # comments ignored.
Use the EXACT prompts from Person 4's shot list - matching is on the normalised
text of the last user message, so a typo on stage misses the cache.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import httpx2 as httpx
import logs
from wire import demo_cache_key

log = logging.getLogger(__name__)


def cache_key(prompt: str) -> str:
    """The key the router looks demo answers up under: the normalised last user message."""
    return demo_cache_key({"messages": [{"role": "user", "content": prompt}]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--url", default="http://pi-node-5.local:8000/v1/chat/completions")
    ap.add_argument("--model", default="qwen3-30b-a3b")
    ap.add_argument("--out", default="demo_cache.json")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument(
        "--force-upstream", default="", help="cluster|baseten - pin which upstream generates the cached answer"
    )
    args = ap.parse_args()
    logs.configure()

    lines = [line.strip() for line in Path(args.prompts).read_text().splitlines()]
    prompts = [line for line in lines if line and not line.startswith("#")]
    if not prompts:
        sys.exit("no prompts found")

    cache = {}
    if Path(args.out).exists():
        try:
            cache = json.loads(Path(args.out).read_text())
        except Exception:
            cache = {}

    headers = {}
    if args.force_upstream:
        headers["X-Force-Upstream"] = args.force_upstream

    with httpx.Client(timeout=180) as c:
        for i, p in enumerate(prompts, 1):
            log.info("[%d/%d] %s", i, len(prompts), p[:70])
            try:
                r = c.post(
                    args.url,
                    headers=headers,
                    json={
                        "model": args.model,
                        "messages": [{"role": "user", "content": p}],
                        "max_tokens": args.max_tokens,
                        "stream": False,
                    },
                )
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                cache[cache_key(p)] = text
                log.info("-> %d chars via %s", len(text), r.headers.get("X-Served-By"))
            except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
                log.warning("%s: %s", type(e).__name__, e)

    Path(args.out).write_text(json.dumps(cache, indent=2))
    log.info("wrote %d entries to %s; restart the router to load it", len(cache), args.out)


if __name__ == "__main__":
    main()
