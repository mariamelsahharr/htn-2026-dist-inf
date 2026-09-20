#!/usr/bin/env python3
"""
train_distill.py - QLoRA fine-tune Qwen3-30B-A3B on the cloud model's answers,
then merge the adapter back to bf16 so distributed-llama's converter can read it.

Runs on one H100 (80GB). Two modes:

    # 1. train (4-bit base + LoRA adapter, ~1-2h for 3k samples)
    ./train_distill.py --data distill_data.jsonl \
                       --base ~/qwen3-30b-a3b \
                       --out ~/a3b-lora

    # 2. merge into full bf16 weights (CPU, slow but needs no GPU memory)
    ./train_distill.py --merge-only --base ~/qwen3-30b-a3b \
                       --adapter ~/a3b-lora \
                       --merged ~/qwen3-30b-a3b-tuned

DISK: base 61GB + merged 61GB + q40 17GB = ~140GB. Check `df -h` FIRST.

LoRA targets attention projections only, not the expert MLPs. Experts are 95%
of the params; adapting them is slower, uses far more memory, and merging them
is where MoE checkpoints tend to go wrong. Attention-only is enough to move
style, formatting and instruction-following, which is what this is for.
"""

import argparse
import json
import os
from pathlib import Path

ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    print(f"{len(rows)} training pairs")
    ds = Dataset.from_list(rows)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # NOTE: do NOT use bitsandbytes 4-bit here. Qwen3-MoE keeps its expert
    # weights as fused parameter tensors, not nn.Linear modules, and bnb only
    # swaps nn.Linear - so the experts (95% of the model) stay in bf16 and you
    # OOM at ~72GB. Load bf16 and shard across GPUs instead.
    print("loading base in bf16, sharded across available GPUs...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    print(f"model footprint: {model.get_memory_footprint()/1e9:.1f} GB")

    # prepare_model_for_kbit_training upcasts every non-quantized param to fp32,
    # which is what actually blew up. We only need these two things from it.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=ATTN_TARGETS,
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=args.max_len,
        packing=False,
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\nadapter saved to {args.out}")
    print(f"next:  ./train_distill.py --merge-only --base {args.base} "
          f"--adapter {args.out} --merged <path>")


def merge(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading base in bf16 on CPU (needs ~61GB RAM, takes a while)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)

    print("applying adapter...")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()

    print(f"saving to {args.merged} ...")
    merged.save_pretrained(args.merged, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.merged)

    # the converter reads config.json; make sure it came along
    assert (Path(args.merged) / "config.json").exists(), "config.json missing from merged dir"
    print("\nmerged. now convert:")
    print(f"  cd distributed-llama/converter")
    print(f"  python convert-hf.py {args.merged} q40 qwen3-30b-a3b-tuned")
    print(f"  python convert-tokenizer-hf.py {args.merged} qwen3-30b-a3b-tuned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF dir of Qwen/Qwen3-30B-A3B")
    ap.add_argument("--data", help="distill_data.jsonl from gen_distill_data.py")
    ap.add_argument("--out", default="a3b-lora", help="where to write the adapter")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--adapter", help="adapter dir, for --merge-only")
    ap.add_argument("--merged", help="output dir for merged bf16, for --merge-only")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    if args.merge_only:
        if not (args.adapter and args.merged):
            raise SystemExit("--merge-only needs --adapter and --merged")
        merge(args)
    else:
        if not args.data:
            raise SystemExit("training needs --data")
        train(args)


if __name__ == "__main__":
    main()
