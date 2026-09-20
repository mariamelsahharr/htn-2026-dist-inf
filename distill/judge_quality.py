#!/usr/bin/env python3
"""
judge_quality.py - the number Option 1 produces: does the tuned model give
better answers than the stock one?

Runs a held-out prompt set through two endpoints, then has the cloud model
judge the pairs blind. Each pair is judged TWICE with the order swapped,
because LLM judges have a strong position bias and a one-shot verdict is
close to worthless.

    export BASETEN_API_KEY=...
    ./judge_quality.py \
        --prompts heldout.txt \
        --a http://192.168.50.13:9990/v1 --a-name stock \
        --b http://192.168.50.13:9990/v1 --b-name tuned \
        --out judge_results.json

Point --a and --b at whatever serves each model (two ports, or the same port
before/after a restart using --a-file / --b-file to reuse saved answers).

IMPORTANT: the prompts must NOT appear in distill_data.jsonl, or you are
measuring memorisation. Generate them with:
    ./gen_distill_data.py --prompts-only --n 60 --out heldout
"""

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

import httpx

CLOUD = os.environ.get("CLOUD_BASE_URL", "https://inference.baseten.co/v1")
KEY = os.environ.get("BASETEN_API_KEY") or os.environ.get("CLOUD_API_KEY", "")
JUDGE_MODEL = os.environ.get("CLOUD_MODEL", "zai-org/GLM-5.3-Fast")

JUDGE_PROMPT = """You are grading two answers to the same developer question.

Question:
{q}

Answer 1:
{a1}

Answer 2:
{a2}

Which answer is better? Judge on correctness first, then concision and
usefulness to a working engineer. Ignore length unless it hurts clarity.
Ignore formatting differences that do not affect understanding.

Reply with exactly one word: 1, 2, or TIE."""


async def post(client, base, model, messages, max_tokens, key=None, temp=0.0):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    for attempt in range(3):
        try:
            r = await client.post(f"{base.rstrip('/')}/chat/completions",
                                  json={"model": model, "messages": messages,
                                        "max_tokens": max_tokens, "temperature": temp},
                                  headers=headers, timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == 2:
                print(f"  !! {type(e).__name__}: {e}")
                return None
            await asyncio.sleep(2 ** attempt)
    return None


async def collect(client, base, model, prompts, max_tokens, label):
    print(f"generating {label} answers ({len(prompts)} prompts)...")
    out = []
    for i, p in enumerate(prompts, 1):
        ans = await post(client, base, model, [{"role": "user", "content": p}], max_tokens)
        out.append(ans)
        if i % 10 == 0:
            print(f"  {label}: {i}/{len(prompts)}")
    return out


def parse_verdict(text):
    if not text:
        return None
    t = text.strip().upper()
    if t.startswith("TIE"):
        return "tie"
    m = re.search(r"\b([12])\b", t)
    return m.group(1) if m else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--a", help="base url for model A")
    ap.add_argument("--b", help="base url for model B")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--a-model", default="qwen3-30b-a3b")
    ap.add_argument("--b-model", default="qwen3-30b-a3b")
    ap.add_argument("--a-file", help="reuse saved answers instead of generating")
    ap.add_argument("--b-file", help="reuse saved answers instead of generating")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--out", default="judge_results.json")
    args = ap.parse_args()

    if not KEY:
        raise SystemExit("set BASETEN_API_KEY for the judge")

    prompts = [l.strip() for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    print(f"{len(prompts)} held-out prompts\n")

    async with httpx.AsyncClient() as client:
        if args.a_file:
            ans_a = json.loads(Path(args.a_file).read_text())
        else:
            ans_a = await collect(client, args.a, args.a_model, prompts, args.max_tokens, args.a_name)
            Path(f"answers_{args.a_name}.json").write_text(json.dumps(ans_a, indent=2))

        if args.b_file:
            ans_b = json.loads(Path(args.b_file).read_text())
        else:
            ans_b = await collect(client, args.b, args.b_model, prompts, args.max_tokens, args.b_name)
            Path(f"answers_{args.b_name}.json").write_text(json.dumps(ans_b, indent=2))

        print("\njudging (each pair twice, orders swapped)...")
        wins_a = wins_b = ties = skipped = 0
        records = []

        for i, (q, a, b) in enumerate(zip(prompts, ans_a, ans_b), 1):
            if not a or not b:
                skipped += 1
                continue
            votes = []
            for first_is_a in (True, False):
                a1, a2 = (a, b) if first_is_a else (b, a)
                v = parse_verdict(await post(
                    client, CLOUD, JUDGE_MODEL,
                    [{"role": "user", "content": JUDGE_PROMPT.format(q=q, a1=a1, a2=a2)}],
                    16, key=KEY))
                if v == "tie" or v is None:
                    votes.append("tie")
                elif (v == "1") == first_is_a:
                    votes.append("a")
                else:
                    votes.append("b")

            if votes[0] == votes[1] == "a":
                wins_a += 1; verdict = args.a_name
            elif votes[0] == votes[1] == "b":
                wins_b += 1; verdict = args.b_name
            else:
                ties += 1; verdict = "tie/inconsistent"

            records.append({"prompt": q, "verdict": verdict, "votes": votes})
            if i % 10 == 0:
                print(f"  {i}/{len(prompts)}")

    decided = wins_a + wins_b
    print("\n" + "=" * 50)
    print(f"{args.a_name}: {wins_a}    {args.b_name}: {wins_b}    tie/inconsistent: {ties}")
    if decided:
        print(f"{args.b_name} win rate (of decided pairs): "
              f"{100.0 * wins_b / decided:.0f}%")
    if skipped:
        print(f"skipped {skipped} (a model failed to answer)")
    print("=" * 50)

    Path(args.out).write_text(json.dumps(
        {"a_name": args.a_name, "b_name": args.b_name,
         "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
         "skipped": skipped, "records": records}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
