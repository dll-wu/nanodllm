#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket

import torch
import torch.distributed as dist

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Small CUDA and optional NCCL correctness smoke test.")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    dtype = DTYPES[args.dtype]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("bfloat16 requested but unsupported by this GPU/runtime")

    torch.manual_seed(1234 + local_rank)
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(local_rank)
    model = torch.nn.Sequential(
        torch.nn.Linear(args.width, args.width * 4),
        torch.nn.GELU(),
        torch.nn.Linear(args.width * 4, args.width),
    ).to(device=device, dtype=dtype)
    inputs = torch.randn(args.batch, args.width, device=device, dtype=dtype, requires_grad=True)
    loss = model(inputs).float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(loss) or inputs.grad is None or not torch.isfinite(inputs.grad).all():
        raise SystemExit("non-finite CUDA forward/backward result")

    collective = torch.tensor(float(local_rank + 1), device=device)
    if world_size > 1:
        dist.all_reduce(collective)
    expected_collective = world_size * (world_size + 1) / 2
    if float(collective) != expected_collective:
        raise SystemExit(f"collective mismatch: {float(collective)} != {expected_collective}")

    payload = {
        "status": "ok",
        "host": socket.gethostname(),
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": world_size,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "dtype": args.dtype,
        "loss": float(loss.detach()),
        "collective": float(collective),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
    }
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(payload, sort_keys=True))
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
