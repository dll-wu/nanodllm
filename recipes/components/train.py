"""Single-rank training for the component objectives on the native model and source stream."""

from __future__ import annotations

import copy
import math
import os
import random
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import torch
from nanodllm.runtime.checkpoint_load_binding import load_checkpoint_with_optional_binding
from nanodllm.runtime.data import build_tokenizer, build_train_loader
from nanodllm.runtime.data_identity import build_data_identity
from nanodllm.runtime.model import DiffusionConfig, DiffusionTransformer
from nanodllm.runtime.telemetry import append_jsonl, write_json_atomic
from nanodllm.runtime.training_optim import _set_optimizer_lrs, _zero_grad, make_optimizer
from nanodllm.runtime.training_plan import (
    _resolve_diffusion_config,
    _resolved_warmup_steps,
    _training_budget,
    get_lr_multiplier,
)
from nanodllm.runtime.training_rng import _restore_rng_state, _rng_state
from nanodllm.runtime.training_runtime import _configure_numeric_runtime, dtype_from_name

from . import block, foundation


def read_ar_parent(path, sha256, *, seed, expected_tokens=115343360, expected_steps=1760):
    """Admit the caller's explicitly trusted native AR endpoint, bound to its opened bytes."""
    parent = load_checkpoint_with_optional_binding(
        path,
        expected_sha256=sha256,
        expected_size_bytes=Path(path).stat().st_size,
        map_location="cpu",
        weights_only=False,
    )
    cfg = parent["train_config"]
    if not parent["config"]["causal"] or (parent["step"], parent["train_tokens"]) != (expected_steps, expected_tokens):
        raise ValueError("expected a completed AR10 parent with the reference source budget")
    if (cfg["runtime"]["model_seed"], cfg["runtime"]["training_seed"]) != (seed, seed):
        raise ValueError("parent and conversion seeds differ")
    return {k: parent[k] for k in ("model_state_dict", "config", "train_config")}


def _seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def _frozen_prefix(cfg, tokenizer, count):
    rng = _rng_state(cfg)
    try:
        batches = build_train_loader(cfg, tokenizer, rank=0, world_size=1, device=torch.device("cpu"))
        frozen = []
        for _ in range(count):
            batch = next(batches)
            # Packing can reuse buffers. Every calibration interval sees the original examples.
            frozen.append(
                replace(
                    batch,
                    **{
                        f.name: value.detach().clone()
                        for f in fields(batch)
                        if isinstance(value := getattr(batch, f.name), torch.Tensor)
                    },
                )
            )
        return frozen
    finally:
        _restore_rng_state(rng, cfg)


def dense_child_flops(positions, projected):
    """20M two-stream dense matmuls for equal-token / matched-compute BL32 runs."""
    if any(type(x) is not int or x < 0 for x in (positions, projected)) or projected > positions:
        raise ValueError("expected physical position and selected projection counts")
    return 3 * ((2 * 3145728 + 4 * 4 * 256 * 2048) * positions + 2 * 256 * 32768 * projected)


def train_stage(cfg, arm, policy, output, *, parent=None, parent_sha256=None):
    """Run one complete component schedule and save an exportable endpoint, with fresh optimizer and RNG."""
    if arm not in {"F1", "F2", "F3", "F4", "B1", "B2", "B3"} or (parent is not None) != arm.startswith("B"):
        raise ValueError("conversion takes a model-only parent and scratch training takes none")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1 or int(os.environ.get("RANK", 0)) != 0:
        raise ValueError("component updates use one training process")
    device = torch.device(cfg.runtime.device_type)
    dtype = dtype_from_name(cfg.runtime.dtype)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    _configure_numeric_runtime(cfg)
    tokenizer = build_tokenizer(cfg)
    _seed_all(cfg.runtime.model_seed, device)
    architecture = _resolve_diffusion_config(cfg, tokenizer)
    model = DiffusionTransformer(
        architecture, rope_base=cfg.model.rope_base, compute_dtype=dtype, pad_vocab_to=cfg.model.pad_vocab_to
    ).to(device)
    if parent is not None:
        expected = asdict(architecture)
        source = asdict(DiffusionConfig(**parent["config"]))
        source["causal"] = expected["causal"]
        if source != expected or parent["train_config"]["model"]["rope_base"] != cfg.model.rope_base:
            raise ValueError("parent and child parameter architectures differ")
        model.load_state_dict(parent["model_state_dict"], strict=True)
    counts = model.parameter_counts()
    batch_tokens, accumulation, steps, lr_scale = _training_budget(cfg, counts, 1, raw_model=model)
    if steps * batch_tokens != cfg.train.target_tokens or steps != cfg.train.num_steps:
        raise ValueError("resolved update count differs from the declared source budget")
    optimizers = make_optimizer(model, cfg, lr_scale)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])
    tuning_policy = copy.deepcopy(policy)
    warmup = _resolved_warmup_steps(cfg, steps)
    seed = cfg.runtime.training_seed
    _seed_all(seed, device)
    generator = torch.Generator(device=device).manual_seed(seed + 100003)
    tuning_generator = torch.Generator(device=device).manual_seed(seed + 200003)
    frozen = _frozen_prefix(cfg, tokenizer, policy["calibration_batch_count"]) if arm == "F4" else []
    if frozen:
        foundation.initialize_policy(tuning_policy)
    loader = iter(build_train_loader(cfg, tokenizer, rank=0, world_size=1, device=device))
    objective = foundation.backward_microbatch if arm.startswith("F") else block.backward_microbatch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        str(output / "resolved_config.json"),
        {"config": asdict(cfg), "objective": arm, "policy": policy, "parent_sha256": parent_sha256},
    )
    write_json_atomic(str(output / "data_identity.v1.json"), build_data_identity(cfg))
    valid_tokens = 0
    totals = Counter()
    for update in range(steps):
        multiplier = get_lr_multiplier(update, steps, warmup, cfg.optim.warmdown_ratio, cfg.optim.final_lr_frac)
        _set_optimizer_lrs(optimizers, multiplier)
        _zero_grad(optimizers)
        telemetry = Counter()
        with torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else nullcontext():
            if frozen:
                telemetry.update(
                    foundation.prepare_update(
                        model,
                        step=update,
                        num_steps=steps,
                        generator=tuning_generator,
                        policy=tuning_policy,
                        calibration_batches=frozen,
                    )
                )
        for _ in range(accumulation):
            batch = next(loader)
            valid_tokens += int(batch.attention_mask.sum())
            if batch.x_0.numel() * accumulation != batch_tokens:
                raise ValueError("packed source shape differs from the allocated update budget")
            with torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else nullcontext():
                telemetry.update(
                    objective(
                        model,
                        batch,
                        arm=arm,
                        step=update,
                        num_steps=steps,
                        generator=generator,
                        accumulation=accumulation,
                        policy=tuning_policy,
                    )
                )
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.optim.grad_clip or math.inf, error_if_nonfinite=True
        )
        for optimizer in optimizers:
            optimizer.step()
        totals.update(telemetry)
        if update == 0 or (update + 1) % cfg.train.log_interval == 0 or update + 1 == steps or telemetry["noise_tuned"]:
            append_jsonl(
                str(output / "train_log.jsonl"),
                {
                    "step": update + 1,
                    "lr_multiplier": multiplier,
                    "grad_norm": float(norm),
                    "telemetry_sum": dict(telemetry),
                },
            )
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError("nonfinite parameters at endpoint")
    path = output / "model.pt"
    payload = {
        "config": asdict(architecture),
        "train_config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "step": steps,
        "train_tokens": batch_tokens * steps,
        "train_valid_tokens": valid_tokens,
        "objective": arm,
        "policy": tuning_policy,
        "parent_sha256": parent_sha256,
    }
    torch.save(payload, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    report = {
        "step": steps,
        "source_positions": batch_tokens * steps,
        "parameter_counts": counts,
        "parent_sha256": parent_sha256,
        "method_state": tuning_policy.get("method_state"),
        "telemetry_totals": dict(totals),
    }
    if (
        arm in {"B1", "B3"}
        and policy["block_size"] == 32
        and architecture.sequence_len == 1024
        and counts["transformer_body_parameters"] == 3145728
        and counts["output_head"] == 8388608
    ):
        report["estimated_child_dense_matmul_flops"] = dense_child_flops(
            int(totals["forward_token_positions"]), int(totals["active_tokens"])
        )
    write_json_atomic(str(output / "training_report.json"), report)
    return path
