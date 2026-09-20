#!/usr/bin/env python3
"""
gen_distill_data.py - build the SFT dataset for distilling the cloud model
into the local Pi model.

Black-box distillation: we only need the teacher's OUTPUTS, not its weights.
So this is just API calls - no 60GB model download needed for this step.

Two phases:
  1. ask the teacher to invent realistic coding-agent prompts (diverse, short)
  2. ask the teacher to answer them

Output is JSONL in the format TRL's SFTTrainer expects:
  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Resumable: re-running skips prompts already answered in --out.

    export BASETEN_API_KEY=...
    ./gen_distill_data.py --n 3000 --out distill_data.jsonl

Expect ~45 min for 3000 pairs at --concurrency 16.
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

import httpx

BASE_URL = os.environ.get("CLOUD_BASE_URL", "https://inference.baseten.co/v1")
API_KEY = os.environ.get("BASETEN_API_KEY") or os.environ.get("CLOUD_API_KEY", "")

# The traffic shape we actually serve: short coding-agent queries. Keep this
# aligned with loadtest.py's prompt mix and demo_prompts.txt - the student only
# needs to be good at the slice the router will send it.
TOPICS = [
    "shell one-liners (find, grep, awk, sed, xargs, tar, rsync)",
    "git operations and recovering from mistakes",
    "small Python refactors and bug fixes",
    "reading and explaining error messages and stack traces",
    "systemd units, cron, and process management",
    "Docker and container troubleshooting",
    "regex construction and explanation",
    "SQL queries and query debugging",
    "JavaScript/TypeScript snippets and Node tooling",
    "HTTP, curl, and API debugging",
    "file and text processing (csv, json, jq)",
    "networking diagnostics (ss, netstat, dig, iperf3)",
    "short 'explain this concept in two sentences' questions",
    "writing small test cases",
    "build tooling: make, npm, pip, virtualenvs",
]

PROMPT_GEN_TEMPLATE = """Write {k} distinct questions a developer would type into a coding assistant about: {topic}

Rules:
- one question per line, no numbering, no bullets, no quotes
- each under 25 words
- phrased the way a tired engineer actually types, not like documentation
- vary the form: some imperative ("write a..."), some questions ("why does...")
- no preamble, output only the lines"""

ANSWER_SYSTEM = (
    "You are a concise coding assistant. Answer directly and correctly. "
    "Use a short code block when code is the answer. No preamble, no filler, "
    "no restating the question. Keep it under 200 words unless the question "
    "genuinely needs more."
)


class Teacher:
    def __init__(self, client, model, concurrency):
        self.client = client
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)

    async def chat(self, messages, max_tokens, temperature, retries=4):
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature}
        async with self.sem:
            for attempt in range(retries):
                try:
                    r = await self.client.post(f"{BASE_URL}/chat/completions",
                                               json=body, headers=headers, timeout=120)
                    if r.status_code == 429 or r.status_code >= 500:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"  !! giving up: {type(e).__name__}: {e}", file=sys.stderr)
                        return None
                    await asyncio.sleep(2 ** attempt + random.random())
        return None


def clean_prompt(line):
    line = line.strip().strip('"').strip("'")
    # strip leading "1. ", "- ", "* " if the model ignored instructions
    for prefix in ("- ", "* "):
        if line.startswith(prefix):
            line = line[len(prefix):]
    if line[:3].rstrip(". ").isdigit():
        line = line.split(".", 1)[-1].strip()
    return line.strip()


async def make_prompts(teacher, target):
    """Phase 1: have the teacher invent the prompt distribution."""
    per_topic = max(4, target // len(TOPICS) + 4)
    batches = [(t, min(25, per_topic)) for t in TOPICS]
    # repeat topics until we have enough requested
    while sum(k for _, k in batches) < target * 1.3:
        batches += [(t, min(25, per_topic)) for t in TOPICS]

    async def one(topic, k):
        out = await teacher.chat(
            [{"role": "user", "content": PROMPT_GEN_TEMPLATE.format(k=k, topic=topic)}],
            max_tokens=900, temperature=1.0)
        if not out:
            return []
        return [clean_prompt(l) for l in out.splitlines() if l.strip()]

    print(f"phase 1: generating prompts across {len(TOPICS)} topics...")
    results = await asyncio.gather(*(one(t, k) for t, k in batches))

    seen, prompts = set(), []
    for group in results:
        for p in group:
            key = " ".join(p.lower().split())
            if len(p) < 10 or key in seen:
                continue
            seen.add(key)
            prompts.append(p)
    random.shuffle(prompts)
    print(f"phase 1: {len(prompts)} unique prompts")
    return prompts[:target]


async def answer_all(teacher, prompts, out_path, max_tokens):
    """Phase 2: teacher answers. Appends as it goes so a crash loses nothing."""
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done.add(" ".join(rec["messages"][0]["content"].lower().split()))
            except Exception:
                continue
        print(f"phase 2: {len(done)} already done, skipping those")

    todo = [p for p in prompts if " ".join(p.lower().split()) not in done]
    print(f"phase 2: answering {len(todo)} prompts...")

    lock = asyncio.Lock()
    written = 0

    async def one(i, p):
        nonlocal written
        ans = await teacher.chat(
            [{"role": "system", "content": ANSWER_SYSTEM},
             {"role": "user", "content": p}],
            max_tokens=max_tokens, temperature=0.3)
        if not ans:
            return
        rec = {"messages": [{"role": "user", "content": p},
                            {"role": "assistant", "content": ans}]}
        async with lock:
            with out_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            written += 1
            if written % 25 == 0:
                print(f"  {written}/{len(todo)}")

    await asyncio.gather(*(one(i, p) for i, p in enumerate(todo)))
    print(f"phase 2: wrote {written} pairs")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="target number of pairs")
    ap.add_argument("--out", default="distill_data.jsonl")
    ap.add_argument("--model", default=os.environ.get("CLOUD_MODEL", "zai-org/GLM-5.3-Fast"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--prompts-only", action="store_true",
                    help="stop after phase 1 and dump prompts to <out>.prompts.txt")
    args = ap.parse_args()

    if not API_KEY:
        sys.exit("set BASETEN_API_KEY (or CLOUD_API_KEY)")

    out_path = Path(args.out)
    limits = httpx.Limits(max_connections=args.concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        teacher = Teacher(client, args.model, args.concurrency)
        print(f"teacher: {args.model} @ {BASE_URL}")

        prompt_file = Path(str(out_path) + ".prompts.txt")
        if prompt_file.exists():
            prompts = [l for l in prompt_file.read_text().splitlines() if l.strip()]
            print(f"phase 1: reusing {len(prompts)} prompts from {prompt_file}")
        else:
            prompts = await make_prompts(teacher, args.n)
            prompt_file.write_text("\n".join(prompts) + "\n")
            print(f"phase 1: saved to {prompt_file}")

        if args.prompts_only:
            return

        await answer_all(teacher, prompts, out_path, args.max_tokens)

    n = sum(1 for _ in out_path.open()) if out_path.exists() else 0
    print(f"\ndone: {n} pairs in {out_path}")
    print("sanity check a few:  head -3 " + str(out_path))


if __name__ == "__main__":
    asyncio.run(main())
