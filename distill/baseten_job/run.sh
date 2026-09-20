#!/usr/bin/env bash
# Runs inside the Baseten training container.
# train -> merge -> convert to distributed-llama q40, all in one job so the
# only thing that ever leaves the container is the finished ~17GB .m file.
set -euxo pipefail

# ---------------------------------------------------------------- 0. sanity
nvidia-smi
df -h            # need ~140GB: 61 base + 61 merged + 17 q40. CHECK THIS FIRST.
free -g          # merge step wants ~61GB RAM

# Baseten exposes the synced checkpoint directory as an env var. Confirm the
# exact name against the ml-cookbook recipes; this falls back to a local dir
# so the job still completes and you can pull files over sftp if it differs.
# /mnt/ckpts is the mount Baseten syncs as checkpoints (seen in the job logs).
OUT_DIR="${BT_CHECKPOINT_DIR:-/mnt/ckpts}"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------- 1. deps
pip install -q \
  "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0" \
  datasets accelerate bitsandbytes safetensors \
  huggingface_hub hf_transfer

# ---------------------------------------------------------------- 2. weights
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/root/qwen3-30b-a3b
if [ ! -f "$BASE/config.json" ]; then
  hf download Qwen/Qwen3-30B-A3B --local-dir "$BASE"
fi

# ---------------------------------------------------------------- 3. train
python train_distill.py \
  --data distill_data.jsonl \
  --base "$BASE" \
  --out "$OUT_DIR" \
  --epochs 2 --rank 16 --max-len 1024 --max-samples 32

# keep the adapter regardless - it is small and it is the thing worth saving
# adapter already written straight into $OUT_DIR by the trainer

# ---------------------------------------------------------------- 4. merge
python train_distill.py --merge-only \
  --base "$BASE" \
  --adapter "$OUT_DIR" \
  --merged /root/merged

# reclaim space before the conversion writes another 17GB
rm -rf "$BASE"

# ---------------------------------------------------------------- 5. convert
git clone --depth 1 https://github.com/b4rtaz/distributed-llama.git /root/dllama
cd /root/dllama/converter
python convert-hf.py /root/merged q40 qwen3-30b-a3b-tuned
python convert-tokenizer-hf.py /root/merged qwen3-30b-a3b-tuned

# ---------------------------------------------------------------- 6. ship
# converter writes into its own dir; grab whatever it produced
find /root/dllama/converter -maxdepth 2 \( -name '*.m' -o -name '*.t' \) \
     -exec cp -v {} "$OUT_DIR"/ \;

ls -lh "$OUT_DIR"
echo "DONE - pull the .m and .t from checkpoints, rsync to pi-node-3"
