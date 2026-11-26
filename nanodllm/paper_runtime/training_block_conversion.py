"""Causal-clean-stream block conversion corruption and loss contracts."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .noise import MDLMNoise, UniformNoise
from .training_bd3lm import sample_bd3lm_block_times

BLOCK_CONVERSION_OBJECTIVE_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class BlockConversionCorruption:
    branch: str
    z_t: torch.Tensor
    active: torch.Tensor
    block_times: torch.Tensor
    decoder_weights: torch.Tensor
    changed_token_count: int


@dataclass(frozen=True)
class BlockConversionLogitPlan:
    logits_indices: torch.Tensor
    decoder_targets: torch.Tensor
    ar_targets: torch.Tensor

    @property
    def decoder_count(self) -> int:
        return int(self.decoder_targets.numel())

    @property
    def ar_count(self) -> int:
        return int(self.ar_targets.numel())


@dataclass(frozen=True)
class BlockConversionLosses:
    loss: torch.Tensor
    decoder_loss: torch.Tensor
    decoder_raw_ce: torch.Tensor
    ar_loss: torch.Tensor
    decoder_loss_normalization_scale: float


def block_conversion_training_block_size_phase_index(config: ExperimentConfig, optimizer_step: int) -> int:
    """Resolve the zero-based BL-schedule phase for one optimizer step."""

    step = int(optimizer_step)
    if step < 0:
        raise ValueError("block-conversion optimizer_step must be non-negative")
    if config.train.bd3lm_block_size_schedule == "fixed":
        return 0
    boundaries = config.train.bd3lm_block_size_schedule_start_step_indices
    index = bisect_right(boundaries, step) - 1
    if index < 0 or index >= len(config.train.bd3lm_block_size_schedule_values):
        raise ValueError(f"block-conversion schedule does not cover optimizer step {step}")
    return index


def block_conversion_training_block_size(config: ExperimentConfig, optimizer_step: int) -> int:
    """Resolve the active training block length for a zero-based optimizer step."""

    if config.train.bd3lm_block_size_schedule == "fixed":
        if int(optimizer_step) < 0:
            raise ValueError("block-conversion optimizer_step must be non-negative")
        return int(config.model.bd3lm_block_size)
    phase_index = block_conversion_training_block_size_phase_index(config, optimizer_step)
    return int(config.train.bd3lm_block_size_schedule_values[phase_index])


def block_conversion_block_size_telemetry(config: ExperimentConfig) -> dict[str, object]:
    """Return machine-readable telemetry for the configured BL schedule."""

    schedule = config.train.bd3lm_block_size_schedule
    steps = config.train.bd3lm_block_size_schedule_start_step_indices
    values = config.train.bd3lm_block_size_schedule_values
    return {
        **(
            {
                "bd3lm_partition_mode": config.train.bd3lm_partition_mode,
                "bd3lm_partition_lengths": list(config.train.bd3lm_partition_lengths),
                "bd3lm_partition_seed": config.train.bd3lm_partition_seed,
                "bd3lm_scalar_block_size_is_reference_only": True,
            }
            if config.train.bd3lm_partition_mode != "fixed"
            else {}
        ),
        "bd3lm_block_size_schedule": schedule,
        "bd3lm_block_size_schedule_start_step_indices": list(steps),
        "bd3lm_block_size_schedule_values": list(values),
        "bd3lm_block_size_schedule_step_unit": "zero_based_inclusive_optimizer_step_index",
        "bd3lm_initial_training_block_size": (
            int(values[0]) if schedule == "stepwise" else int(config.model.bd3lm_block_size)
        ),
        "bd3lm_target_block_size": int(config.model.bd3lm_block_size),
        "bd3lm_block_size_curriculum_enabled": schedule == "stepwise",
    }


def block_conversion_decoder_branch(config: ExperimentConfig, microbatch_index: int) -> str:
    """Return the deterministic whole-microbatch decoder objective."""

    if config.train.bd3lm_recipe_mechanism.startswith("staged_"):
        rows = config.train.device_batch_size * config.model.max_seq_len
        accumulation = config.train.total_batch_size // rows
        if microbatch_index < config.train.bd3lm_recipe_start_step * accumulation:
            return "m2t"
    fraction = float(config.train.bd3lm_uniform_t2t_batch_fraction)
    if fraction <= 0.0:
        return "m2t"
    if fraction >= 1.0:
        return "t2t"
    index = int(microbatch_index)
    selected = math.floor((index + 1) * fraction) > math.floor(index * fraction)
    return "t2t" if selected else "m2t"


def full_mask_microbatch(config: ExperimentConfig, microbatch_index: int) -> bool:
    """Stateless cadence; GA4 and fraction1/4 selects the fourth microbatch."""
    if microbatch_index < 0:
        raise ValueError("microbatch index must be nonnegative")
    fraction = float(config.train.bd3lm_full_mask_batch_fraction)
    return math.floor((microbatch_index + 1) * fraction) > math.floor(microbatch_index * fraction)


def build_block_conversion_corruption(
    config: ExperimentConfig,
    x_0: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    microbatch_index: int,
    position_ids: torch.Tensor | None = None,
    block_size: int | None = None,
    block_ids: torch.Tensor | None = None,
) -> BlockConversionCorruption:
    """Corrupt one block-conversion microbatch and bind decoder weights.

    M2T uses absorbing masks and inverse block-time weights on masked targets.
    T2T uses uniform vocabulary replacement and unit weights on every lossable
    canvas position, including positions whose visible token remains correct.
    """

    branch = block_conversion_decoder_branch(config, microbatch_index)
    resolved_block_size = int(config.model.bd3lm_block_size) if block_size is None else int(block_size)
    block_times = sample_bd3lm_block_times(
        x_0.shape,
        block_size=resolved_block_size,
        t_min=float(config.noise.t_min),
        t_max=float(config.noise.t_max),
        device=x_0.device,
        position_ids=position_ids,
        block_ids=block_ids,
    )
    if branch == "t2t":
        noise = UniformNoise(mask_token_id=mask_token_id, vocab_size=vocab_size)
        z_t, _ = noise.corrupt(x_0, block_times, maskable=loss_mask)
        active = loss_mask.to(dtype=torch.bool)
        decoder_weights = torch.ones(int(active.sum().item()), device=x_0.device, dtype=torch.float32)
    else:
        noise = MDLMNoise(mask_token_id=mask_token_id)
        z_t, weights = noise.corrupt(x_0, block_times, maskable=loss_mask)
        active = (weights > 0) & loss_mask
        target_times = block_times.reshape(-1).index_select(0, active.reshape(-1).nonzero(as_tuple=True)[0])
        decoder_weights = 1.0 / target_times.to(dtype=torch.float32).clamp_min(1e-8)
        if full_mask_microbatch(config, microbatch_index):
            # Consume the ordinary draw first so later ordinary microbatches
            # retain paired corruption RNG. Protected prefix/BOS/tails stay put.
            active = loss_mask.to(dtype=torch.bool)
            z_t = x_0.masked_fill(active, mask_token_id)
            block_times = torch.ones_like(block_times)
            decoder_weights = torch.ones(int(active.sum()), device=x_0.device, dtype=torch.float32)
    return BlockConversionCorruption(
        branch=branch,
        z_t=z_t,
        active=active,
        block_times=block_times,
        decoder_weights=decoder_weights,
        changed_token_count=int(((z_t != x_0) & active).sum().item()),
    )


def _resolved_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if position_ids is None:
        return None
    resolved = position_ids.to(device=device, dtype=torch.long)
    if resolved.dim() == 1:
        if resolved.shape != (seq_len,):
            raise ValueError(f"position_ids must have shape ({seq_len},), got {tuple(resolved.shape)}")
        return resolved.view(1, seq_len).expand(batch_size, -1)
    if resolved.shape != (batch_size, seq_len):
        raise ValueError(f"position_ids must have shape {(batch_size, seq_len)}, got {tuple(resolved.shape)}")
    return resolved


def build_block_conversion_logit_plan(
    x_0: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    active: torch.Tensor,
    *,
    include_ar: bool,
    position_ids: torch.Tensor | None = None,
) -> BlockConversionLogitPlan:
    """Select decoder and shifted causal-clean logits from one doubled input."""

    if x_0.ndim != 2:
        raise ValueError(f"x_0 must have shape [batch, seq], got {tuple(x_0.shape)}")
    if loss_mask.shape != x_0.shape or attention_mask.shape != x_0.shape or active.shape != x_0.shape:
        raise ValueError("block-conversion token and mask tensors must share shape [batch, seq]")
    batch_size, seq_len = x_0.shape
    decoder_rows, decoder_cols = active.nonzero(as_tuple=True)
    decoder_indices = decoder_rows.to(torch.long) * (2 * seq_len) + decoder_cols.to(torch.long)
    decoder_targets = x_0[active]

    if not include_ar:
        empty = torch.empty(0, dtype=torch.long, device=x_0.device)
        return BlockConversionLogitPlan(
            logits_indices=decoder_indices,
            decoder_targets=decoder_targets,
            ar_targets=empty,
        )

    ar_target_mask = (
        loss_mask[:, 1:].to(torch.bool) & attention_mask[:, 1:].to(torch.bool) & attention_mask[:, :-1].to(torch.bool)
    )
    positions = _resolved_position_ids(
        position_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        device=x_0.device,
    )
    if positions is not None:
        ar_target_mask &= positions[:, 1:] == positions[:, :-1] + 1
    ar_rows, ar_query_cols = ar_target_mask.nonzero(as_tuple=True)
    ar_indices = ar_rows.to(torch.long) * (2 * seq_len) + seq_len + ar_query_cols.to(torch.long)
    ar_targets = x_0[:, 1:][ar_target_mask]
    return BlockConversionLogitPlan(
        logits_indices=torch.cat((decoder_indices, ar_indices), dim=0),
        decoder_targets=decoder_targets,
        ar_targets=ar_targets,
    )


def block_conversion_losses(
    logits: torch.Tensor,
    plan: BlockConversionLogitPlan,
    decoder_weights: torch.Tensor,
    *,
    decoder_lossable_tokens: int,
    ar_loss_weight: float,
) -> BlockConversionLosses:
    """Reduce decoder and AR terms with independent token normalization."""

    expected = plan.decoder_count + plan.ar_count
    if logits.ndim != 2 or logits.shape[0] != expected:
        raise ValueError(f"block-conversion logits must have shape [{expected}, vocab], got {tuple(logits.shape)}")
    if decoder_weights.shape != (plan.decoder_count,):
        raise ValueError(f"decoder_weights must have shape ({plan.decoder_count},), got {tuple(decoder_weights.shape)}")
    if decoder_lossable_tokens <= 0:
        raise ValueError("decoder_lossable_tokens must be positive")
    decoder_logits = logits[: plan.decoder_count]
    ar_logits = logits[plan.decoder_count :]
    if plan.decoder_count:
        decoder_ce = F.cross_entropy(decoder_logits, plan.decoder_targets, reduction="none")
        decoder_raw_ce = decoder_ce.mean()
        decoder_loss = (decoder_ce.float() * decoder_weights.float()).sum() / int(decoder_lossable_tokens)
    else:
        decoder_raw_ce = logits.sum() * 0.0
        decoder_loss = decoder_raw_ce
    ar_loss = F.cross_entropy(ar_logits, plan.ar_targets, reduction="mean") if plan.ar_count else logits.sum() * 0.0
    loss = decoder_loss + float(ar_loss_weight) * ar_loss
    return BlockConversionLosses(
        loss=loss,
        decoder_loss=decoder_loss,
        decoder_raw_ce=decoder_raw_ce,
        ar_loss=ar_loss,
        decoder_loss_normalization_scale=float(decoder_weights.detach().sum().item()) / int(decoder_lossable_tokens),
    )


def block_conversion_objective_label(config: ExperimentConfig) -> str:
    if config.train.bd3lm_recipe_mechanism != "none":
        return "causal_clean_block_recipe_" + config.train.bd3lm_recipe_mechanism
    fraction = float(config.train.bd3lm_uniform_t2t_batch_fraction)
    ar_weight = float(config.train.bd3lm_causal_ar_loss_weight)
    decoder = "m2t" if fraction == 0.0 else "uniform_t2t" if fraction == 1.0 else "mixed_m2t_uniform_t2t"
    if config.train.bd3lm_full_mask_batch_fraction:
        decoder += "_full_mask_emphasis"
    schedule = "_bl_curriculum" if config.train.bd3lm_block_size_schedule == "stepwise" else ""
    return f"causal_clean_block_{decoder}" + ("_plus_ar" if ar_weight > 0.0 else "") + schedule


def block_conversion_objective_telemetry(config: ExperimentConfig) -> dict:
    """Objective metadata stays with its loss owner; disabled fields preserve old reports."""
    uniform_fraction = float(config.train.bd3lm_uniform_t2t_batch_fraction)
    full_mask = config.train.bd3lm_full_mask_batch_fraction
    if full_mask:
        weighting = "cadenced_m2t_inverse_time_and_full_mask_ce"
    elif uniform_fraction == 0.0:
        weighting = "m2t_inverse_block_time_over_clean_lossable_tokens"
    elif uniform_fraction == 1.0:
        weighting = "uniform_t2t_all_position_ce_over_clean_lossable_tokens"
    else:
        weighting = "cadenced_m2t_inverse_time_and_uniform_t2t_all_position_ce"
    return {
        "objective": "bd3lm_block_conversion",
        "objective_label": block_conversion_objective_label(config),
        "bd3lm_enabled": True,
        "bd3lm_block_conversion_enabled": True,
        "bd3lm_block_conversion_contract_version": BLOCK_CONVERSION_OBJECTIVE_CONTRACT_VERSION,
        "loss_normalization": "decoder_and_ar_separately_token_normalized",
        "bd3lm_block_size": int(config.model.bd3lm_block_size),
        "bd3lm_clean_seq_len": int(config.model.max_seq_len),
        "bd3lm_concat_seq_len": int(config.model.max_seq_len) * 2,
        "bd3lm_clean_stream_mode": config.model.bd3lm_clean_stream_mode,
        "bd3lm_corruption_time_sampling": "cadenced_full_mask_or_independent_per_segment_block"
        if full_mask
        else "independent-per-segment-block",
        "bd3lm_uniform_t2t_batch_fraction": uniform_fraction,
        "bd3lm_uniform_t2t_support": "uniform_non_mask_vocabulary_with_accidental_clean_matches_allowed",
        "bd3lm_uniform_t2t_target_scope": "all_lossable_canvas_positions",
        "bd3lm_causal_ar_loss_weight": float(config.train.bd3lm_causal_ar_loss_weight),
        **(
            {
                "bd3lm_full_mask_batch_fraction": full_mask,
                "bd3lm_full_mask_objective": "cadenced_all_lossable_ce_not_unbiased_nelbo",
            }
            if full_mask
            else {}
        ),
        "bd3lm_causal_ar_source": "same_forward_causal_clean_half_shifted_next_token",
        "bd3lm_loss_weighting": weighting,
        "bd3lm_document_mask_policy": "clean-visibility-intersection-in-every-concat-quadrant",
        "bd3lm_position_policy": "duplicate-clean-segment-reset-positions",
        **block_conversion_block_size_telemetry(config),
    }
