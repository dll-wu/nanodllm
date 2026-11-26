"""Fixed-block, curriculum and shifted-target conversion objectives.

The paper curriculum uses block-causal clean attention in both arms. The
10/80/10 phase fractions and equal ramp plateaus adapt LLaDA 2.0 section 4.1
(arXiv:2512.15745v2) to the 20M experiment horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .paper_components_layout import TrainingView, batch_layout, block_times, validate_model, weighted_target_loss
from .paper_runtime.data import DLMDataBatch
from .paper_runtime.training_bd3lm import bd3lm_concat_inputs


@dataclass(frozen=True)
class CurriculumStage:
    start: int
    stop: int
    block_size: int
    phase: int


def curriculum_stages(num_steps: int, *, context_length: int, policy: dict) -> tuple[CurriculumStage, ...]:
    """Integer, half-open, restart-stable 10/80/10 BL warmup/stable/decay.

    Floor cumulative equal shares within each ramp: plateaus differ by at most
    one update. Reject horizons too short to visit every nominated plateau.
    No legacy monotonic-only config validator is changed or bypassed globally.
    """
    if num_steps <= 0 or context_length <= 0:
        raise ValueError("positive num_steps and context_length are required")
    warm = tuple(policy.get("warmup_blocks", [1, 4, 32, 64, context_length]))
    decay = tuple(policy.get("decay_blocks", [context_length, 512, 256, 128, 64, 32]))
    terminal = int(policy.get("block_size", 32))
    for values in (warm, decay):
        if not values or any(type(n) is not int or n <= 0 or context_length % n for n in values):
            raise ValueError("curriculum block lengths must be positive integer divisors of context")
    if warm[0] != 1 or warm[-1] != context_length or decay[0] != context_length or decay[-1] != terminal:
        raise ValueError("curriculum must grow BL1 to full context and end decay at block_size")
    if any(a >= b for a, b in zip(warm, warm[1:], strict=False)) or any(
        a <= b for a, b in zip(decay, decay[1:], strict=False)
    ):
        raise ValueError("warmup must increase and decay must decrease strictly")
    wf, df = float(policy.get("warmup_fraction", 0.1)), float(policy.get("decay_fraction", 0.1))
    if not (math.isfinite(wf) and math.isfinite(df) and wf > 0 and df > 0 and wf + df < 1):
        raise ValueError("positive finite ramp fractions must leave a nonempty stable phase")
    nw, nd = math.floor(num_steps * wf), math.floor(num_steps * df)
    if nw < len(warm) or nd < len(decay):
        raise ValueError("horizon cannot allocate at least one update to every BL plateau")
    stages = []
    for offset, count, values, phase in ((0, nw, warm, 0), (num_steps - nd, nd, decay, 2)):
        for i, size in enumerate(values):
            stages.append(
                CurriculumStage(offset + i * count // len(values), offset + (i + 1) * count // len(values), size, phase)
            )
    stages.append(CurriculumStage(nw, num_steps - nd, context_length, 1))
    return tuple(sorted(stages, key=lambda stage: stage.start))


def resolve_block_size(*, arm: str, step: int, num_steps: int, context_length: int, policy: dict) -> tuple[int, int]:
    if arm not in {"B1", "B2", "B3"}:
        raise ValueError(f"unknown conversion arm: {arm}")
    if type(step) is not int or num_steps <= 0 or not 0 <= step < num_steps:
        raise ValueError("step must be a zero-based optimizer update inside the horizon")
    size = policy.get("block_size", 32)
    if type(size) is not int or size <= 0 or context_length % size:
        raise ValueError("block_size must be a positive integer divisor of context")
    if arm != "B2":
        return size, 0
    for stage in curriculum_stages(num_steps, context_length=context_length, policy=policy):
        if stage.start <= step < stage.stop:
            return stage.block_size, stage.phase
    raise AssertionError("curriculum failed to cover an update")


def build_block_view(
    batch: DLMDataBatch,
    noisy: torch.Tensor,
    active: torch.Tensor,
    times: torch.Tensor,
    *,
    block_size: int,
    clean_stream_mode: str = "causal",
) -> TrainingView:
    """Build the declared clean-stream view; preserve historical token-causal default.

    LLaDA 2.0 Eq.3 uses block_causal clean attention. The correction is opt-in
    so retained B1/B2 contracts do not silently change meaning on replay.
    """
    if clean_stream_mode not in {"causal", "block_causal"}:
        raise ValueError("unknown clean-stream attention")
    positions, visibility = batch_layout(batch)
    x = batch.x_0
    if noisy.shape != x.shape or active.shape != x.shape or times.shape != x.shape:
        raise ValueError("corruption tensors must match source shape")
    if bool((active & ~batch.loss_mask).any()) or bool((times <= 0).any()):
        raise ValueError("active corruption must be lossable with positive times")
    if block_size == x.shape[1]:
        tokens = noisy
    else:
        tokens, visibility, positions = bd3lm_concat_inputs(
            noisy, x, visibility, block_size=block_size, clean_stream_mode=clean_stream_mode, position_ids=positions
        )
    rows, cols = active.nonzero(as_tuple=True)
    return TrainingView(
        tokens, positions, visibility, rows * tokens.shape[1] + cols, x[active], times[active].float().reciprocal()
    )


def backward_microbatch(
    model,
    batch: DLMDataBatch,
    *,
    arm: str,
    step: int,
    num_steps: int,
    generator: torch.Generator,
    accumulation: int = 1,
    policy: dict,
) -> dict[str, float | int]:
    """Accumulate gradients only; caller owns autocast, optimizer and RNG state."""
    validate_model(model)
    if type(accumulation) is not int or accumulation <= 0:
        raise ValueError("accumulation must be a positive integer")
    size, phase = resolve_block_size(
        arm=arm, step=step, num_steps=num_steps, context_length=batch.x_0.shape[1], policy=policy
    )
    if arm == "B3":
        from .paper_components_fastv2 import backward_fastv2

        return backward_fastv2(
            model, batch, block_size=size, generator=generator, accumulation=accumulation, policy=policy
        )
    positions, _ = batch_layout(batch)
    denominator = int(batch.loss_mask.sum())
    if denominator <= 0:
        raise ValueError("conversion microbatch must contain at least one lossable source token")
    times = block_times(positions, block_size=size, generator=generator, policy=policy)
    active = (torch.rand(times.shape, device=times.device, generator=generator) < times) & batch.loss_mask
    noisy = batch.x_0.masked_fill(active, model.config.mask_token_id)
    view = build_block_view(
        batch, noisy, active, times, block_size=size, clean_stream_mode=policy.get("clean_stream_mode", "causal")
    )
    logits = model(
        view.tokens, logits_indices=view.logit_indices, attention_mask=view.visibility, position_ids=view.positions
    )
    loss, raw_sum = weighted_target_loss(logits, view.targets, view.weights, denominator=denominator)
    (loss / accumulation).backward()
    count = view.targets.numel()
    return {
        "loss": float(loss.detach()),
        "raw_ce": float(raw_sum.detach()) / max(count, 1),
        "source_positions": batch.x_0.numel(),
        "lossable_tokens": denominator,
        "active_tokens": count,
        "forward_calls": 1,
        "backward_calls": 1,
        "forward_token_positions": view.tokens.numel(),
        "backward_token_positions": view.tokens.numel(),
        "block_size": size,
        "curriculum_phase_index": phase,
        "view_count": 1,
        "padding_positions": 0,
        "full_context_single_stream": int(size == batch.x_0.shape[1]),
    }
