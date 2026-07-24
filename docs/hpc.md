# Scheduler-Neutral HPC Guide

This guide assumes one node with homogeneous NVIDIA GPUs. Adapt resource
requests and container setup to the local scheduler; keep the training command
portable.

## Environment

Install a PyTorch build matching the node's driver and CUDA runtime, then:

```bash
python -m pip install -e .
python scripts/gpu_smoke.py --dtype bfloat16
```

Confirm:

- `torch.cuda.is_available()` is true;
- every allocated rank sees a distinct GPU;
- the selected dtype is supported;
- the dataset and tokenizer are visible on the node; and
- the run directory is on persistent storage.

Avoid writing required artifacts only to an ephemeral container filesystem.

## One GPU

```bash
export NANODLLM_DATA_DIR=/shared/data/climbmix-400b-shuffle
export NANODLLM_TOKENIZER_DIR=/shared/tokenizers/climbmix-32k

nanodllm-pretrain \
  --config configs/climbmix_mdlm_small.json \
  --run-dir /shared/runs/mdlm-small-probe \
  --set train.num_steps=1 \
  --set train.total_batch_size=1024 \
  --set train.device_batch_size=1 \
  --set runtime.compile=false \
  --set eval.enabled=false
```

Begin with compilation disabled and a conservative device batch. After a
successful real-data step, enable compilation and increase the device batch
while leaving several GiB of memory headroom for evaluation and checkpointing.

## Multi-GPU

```bash
export NANODLLM_DATA_DIR=/shared/data/climbmix-400b-shuffle
export NANODLLM_TOKENIZER_DIR=/shared/tokenizers/climbmix-32k

torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/climbmix_mdlm_d20.json \
  --run-dir /shared/runs/mdlm-d20-probe \
  --set train.num_steps=1 \
  --set train.total_batch_size=16384 \
  --set train.device_batch_size=1 \
  --set runtime.compile=false \
  --set eval.enabled=false
```

Do not use `--standalone` for a multi-node job. Supply the scheduler's rendezvous
endpoint, node rank, and total node count according to the installed PyTorch
version.

The current release is exercised primarily on single-node DDP. Multi-node
operation should be treated as an integration task requiring explicit network,
filesystem, and resume testing.

## Resource Calibration

For each GPU type:

1. run one eager step with `train.device_batch_size=1`;
2. measure peak allocated and reserved memory;
3. run enough compiled steps to pass compilation warmup;
4. increase the device batch conservatively;
5. verify the resolved global batch and accumulation;
6. save an intermediate checkpoint; and
7. resume it with the same world size.

Record GPU product names, memory, driver, CUDA, PyTorch, world size, CPU
allocation, and compilation settings with the run.

## Job Skeleton

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /path/to/nano-dllm
source /path/to/venv/bin/activate

export NANODLLM_DATA_DIR=/shared/data/climbmix-400b-shuffle
export NANODLLM_TOKENIZER_DIR=/shared/tokenizers/climbmix-32k
export OMP_NUM_THREADS="${CPUS_PER_TASK:-8}"

torchrun --standalone --nproc_per_node="${GPUS_PER_NODE:-8}" pretrain.py \
  --config configs/climbmix_mdlm_d20.json \
  --run-dir "/shared/runs/${RUN_ID:?set RUN_ID}"
```

Keep credentials out of job environments and logs. Dataset acquisition and
code publication are separate operations from training.

## Comparison Hygiene

When comparing runs, hold constant:

- code snapshot;
- tokenizer bytes;
- train and validation membership;
- model and initialization seed;
- total token budget and global batch;
- GPU world size and gradient accumulation;
- dtype and compilation settings; and
- evaluation plans and seeds.

Throughput measurements from different GPU types are operational data, not
model-quality evidence.
