"""Corruption-kernel and block-visibility validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config_validation_block_conversion import validate_bd3lm_and_block_conversion_config

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_noise_config(config: ExperimentConfig) -> None:
    if config.noise.type not in ("mdlm", "uniform", "mask_uniform", "bd3lm"):
        raise ValueError("noise.type must be one of 'mdlm', 'uniform', 'mask_uniform', or 'bd3lm'")
    if config.noise.t_min < 0 or config.noise.t_max > 1 or config.noise.t_min > config.noise.t_max:
        raise ValueError("noise.t_min and noise.t_max must satisfy 0 <= t_min <= t_max <= 1")
    if config.noise.type == "mdlm" and config.noise.schedule_weight and config.noise.t_max == 0:
        raise ValueError("schedule-weighted MDLM requires noise.t_max > 0")
    from .mdlm_objective_controls import validate_mdlm_objective_controls

    validate_mdlm_objective_controls(config)
    if config.noise.type == "bd3lm" and config.noise.t_max == 0:
        raise ValueError("BD3-LM requires noise.t_max > 0")
    if config.noise.mask_uniform_mix_prob < 0 or config.noise.mask_uniform_mix_prob > 1:
        raise ValueError("noise.mask_uniform_mix_prob must be in [0, 1]")
    if config.noise.type == "mdlm" and config.noise.time_conditioning:
        raise ValueError("noise.time_conditioning is only allowed for non-MDLM corruption kernels")
    if config.noise.suffix_masking and config.noise.type != "mdlm":
        raise ValueError("noise.suffix_masking is only supported for noise.type='mdlm'")
    validate_bd3lm_and_block_conversion_config(config)
    if config.model.block_causal_pilot_size < 0:
        raise ValueError("model.block_causal_pilot_size must be non-negative")
    if config.model.block_causal_pilot_size > 0:
        if config.noise.type != "mdlm":
            raise ValueError("model.block_causal_pilot_size > 0 requires noise.type='mdlm'")
        if config.data.boundary_aware_packing or config.data.content_budget_segmented_packing:
            raise ValueError("model.block_causal_pilot_size > 0 requires row-global packing without document masks")
        if config.model.causal:
            raise ValueError("model.block_causal_pilot_size > 0 requires model.causal=false")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("model.block_causal_pilot_size cannot be combined with model.bd3lm_block_size")
        if config.noise.suffix_masking:
            raise ValueError("model.block_causal_pilot_size > 0 requires noise.suffix_masking=false")
        if config.noise.complement_masking:
            raise ValueError("model.block_causal_pilot_size > 0 requires noise.complement_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("model.block_causal_pilot_size > 0 requires noise.time_conditioning=false")
        if config.model.block_causal_pilot_size > config.model.max_seq_len:
            raise ValueError("model.block_causal_pilot_size must not exceed model.max_seq_len")
    if config.noise.uniform_t_block_size <= 0:
        raise ValueError("noise.uniform_t_block_size must be positive")
    if config.noise.uniform_t_coarse_block_size <= 0:
        raise ValueError("noise.uniform_t_coarse_block_size must be positive")
    if not isinstance(config.noise.soft_input_curriculum, bool):
        raise ValueError("noise.soft_input_curriculum must be a boolean")
    if (
        config.noise.soft_input_curriculum_start_mix < 0
        or config.noise.soft_input_curriculum_start_mix > 1
        or config.noise.soft_input_curriculum_end_mix < 0
        or config.noise.soft_input_curriculum_end_mix > 1
    ):
        raise ValueError("noise.soft_input_curriculum mix values must be in [0, 1]")
    if config.noise.soft_input_curriculum_start_mix < config.noise.soft_input_curriculum_end_mix:
        raise ValueError("noise.soft_input_curriculum_start_mix must be >= end_mix for soft-to-hard annealing")
    if (
        config.noise.soft_input_curriculum_anneal_fraction <= 0
        or config.noise.soft_input_curriculum_anneal_fraction > 1
    ):
        raise ValueError("noise.soft_input_curriculum_anneal_fraction must be in (0, 1]")
    if config.noise.soft_input_curriculum:
        if config.noise.type != "uniform":
            raise ValueError("noise.soft_input_curriculum requires noise.type='uniform'")
        if not config.noise.time_conditioning:
            raise ValueError("noise.soft_input_curriculum requires noise.time_conditioning=true")
        if config.model.causal:
            raise ValueError("noise.soft_input_curriculum requires model.causal=false")
