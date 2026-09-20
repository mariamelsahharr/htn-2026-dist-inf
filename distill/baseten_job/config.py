"""
Baseten training job: QLoRA fine-tune Qwen3-30B-A3B on cloud-model answers,
merge, and convert straight to distributed-llama q40 format.

Everything happens in the container so nothing large crosses your laptop.
Push with:

    baseten train push --config config.py
    baseten train job logs --job-id <job_id> --tail

Files that must sit next to this config (they get uploaded with the job):
    run.sh
    train_distill.py
    distill_data.jsonl
"""

from truss_train import (
    CacheConfig,
    CheckpointingConfig,
    Compute,
    Image,
    Runtime,
    TrainingJob,
    TrainingProject,
)
from truss.base.truss_config import AcceleratorSpec

# CUDA 12.8 + torch 2.7. bitsandbytes needs a CUDA runtime, so don't slim this.
BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    # caches the HF download between runs - matters a lot if you have to retry
    cache_config=CacheConfig(enabled=True),
    checkpointing_config=CheckpointingConfig(enabled=True),
)

# ONE H100. 4-bit does not work on Qwen3-MoE (fused expert tensors are
# invisible to bitsandbytes), so the base loads in bf16 at ~61GB. That fits an
# 80GB card with ~19GB spare once prepare_model_for_kbit_training's fp32 upcast
# is gone; gradient checkpointing keeps activations small. Two cards queued for
# over an hour waiting for capacity, one schedules immediately.
training_compute = Compute(
    accelerator=AcceleratorSpec(accelerator="H100", count=1),
)

training_job = TrainingJob(
    image=Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = TrainingProject(
    name="a3b-coding-distill",
    job=training_job,
)
