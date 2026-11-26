"""Validation boundary for the causal-clean BD3 block-conversion route."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .block_partitions import validate_partition_config
from .config_validation_block_recipe import validate_block_recipe

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_bd3lm_and_block_conversion_config(config: ExperimentConfig) -> None:
    """Validate BD3 topology plus the opt-in block-conversion isolation contract."""

    validate_block_recipe(config)
    validate_partition_config(config)
    if config.noise.type == "bd3lm":
        if config.noise.time_conditioning:
            raise ValueError("noise.time_conditioning is not supported for BD3LM")
        if config.model.causal:
            raise ValueError("noise.type='bd3lm' requires model.causal=false")
        if config.model.local_attention_lower_layers != 0 or config.model.local_attention_window != 0:
            raise ValueError(
                "noise.type='bd3lm' requires local attention to be disabled "
                "(model.local_attention_lower_layers=0 and model.local_attention_window=0)"
            )
        if not config.noise.schedule_weight:
            raise ValueError("noise.type='bd3lm' requires noise.schedule_weight=true")
        if config.noise.complement_masking:
            raise ValueError("noise.type='bd3lm' requires noise.complement_masking=false")
        if config.model.bd3lm_block_size <= 0:
            raise ValueError("model.bd3lm_block_size must be positive for noise.type='bd3lm'")
        if config.model.bd3lm_block_size > config.model.max_seq_len:
            raise ValueError("model.bd3lm_block_size must not exceed model.max_seq_len")
        if config.model.max_seq_len % config.model.bd3lm_block_size != 0:
            raise ValueError("model.bd3lm_block_size must divide model.max_seq_len")
    elif config.model.bd3lm_block_size < 0:
        raise ValueError("model.bd3lm_block_size must be non-negative")

    uniform_t2t_fraction = config.train.bd3lm_uniform_t2t_batch_fraction
    if (
        isinstance(uniform_t2t_fraction, bool)
        or not isinstance(uniform_t2t_fraction, (int, float))
        or not math.isfinite(float(uniform_t2t_fraction))
        or not 0.0 <= float(uniform_t2t_fraction) <= 1.0
    ):
        raise ValueError("train.bd3lm_uniform_t2t_batch_fraction must be a finite number in [0, 1]")
    ar_loss_weight = config.train.bd3lm_causal_ar_loss_weight
    if (
        isinstance(ar_loss_weight, bool)
        or not isinstance(ar_loss_weight, (int, float))
        or not math.isfinite(float(ar_loss_weight))
        or float(ar_loss_weight) < 0.0
    ):
        raise ValueError("train.bd3lm_causal_ar_loss_weight must be a finite non-negative number")
    causal_clean_stream = config.model.bd3lm_clean_stream_mode == "causal"
    full_mask = config.train.bd3lm_full_mask_batch_fraction
    if (
        isinstance(full_mask, bool)
        or not isinstance(full_mask, (int, float))
        or not math.isfinite(full_mask)
        or not 0 <= full_mask <= 1
    ):
        raise ValueError("bd3lm_full_mask_batch_fraction must be finite in [0, 1]")
    if full_mask and (
        not causal_clean_stream
        or config.noise.type != "bd3lm"
        or uniform_t2t_fraction
        or ar_loss_weight
        or config.train.bd3lm_recipe_mechanism != "none"
    ):
        raise ValueError("full-mask emphasis requires isolated causal-clean block M2T")
    block_size_schedule = config.train.bd3lm_block_size_schedule
    schedule_steps = config.train.bd3lm_block_size_schedule_start_step_indices
    schedule_values = config.train.bd3lm_block_size_schedule_values
    if block_size_schedule not in {"fixed", "stepwise"}:
        raise ValueError("train.bd3lm_block_size_schedule must be 'fixed' or 'stepwise'")
    if not isinstance(schedule_steps, list) or not isinstance(schedule_values, list):
        raise ValueError("BD3 block-size schedule steps and values must be lists")
    if block_size_schedule == "fixed":
        if schedule_steps or schedule_values:
            raise ValueError("fixed BD3 block-size schedule requires empty step and value lists")
    else:
        if not causal_clean_stream:
            raise ValueError("stepwise BD3 block-size schedule requires causal-clean block conversion")
        if config.train.num_steps <= 0:
            raise ValueError("stepwise BD3 block-size schedule requires an explicit positive train.num_steps")
        if not schedule_steps or len(schedule_steps) != len(schedule_values):
            raise ValueError("stepwise BD3 block-size schedule requires equally sized nonempty step and value lists")
        if any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in schedule_steps):
            raise ValueError("BD3 block-size schedule steps must be non-negative integers")
        if schedule_steps[0] != 0 or any(
            right <= left for left, right in zip(schedule_steps, schedule_steps[1:], strict=False)
        ):
            raise ValueError("BD3 block-size schedule steps must start at zero and increase strictly")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in schedule_values):
            raise ValueError("BD3 block-size schedule values must be positive integers")
        if any(right <= left for left, right in zip(schedule_values, schedule_values[1:], strict=False)):
            raise ValueError("BD3 block-size schedule values must increase strictly")
        if any(config.model.max_seq_len % value != 0 for value in schedule_values):
            raise ValueError("every BD3 block-size schedule value must divide model.max_seq_len")
        if schedule_values[-1] != config.model.bd3lm_block_size:
            raise ValueError("the final BD3 block-size schedule value must equal model.bd3lm_block_size")
        if schedule_steps[-1] >= config.train.num_steps:
            raise ValueError("the final BD3 block-size schedule step must precede train.num_steps")
    if causal_clean_stream and config.noise.type != "bd3lm":
        raise ValueError("model.bd3lm_clean_stream_mode='causal' requires noise.type='bd3lm'")
    if not causal_clean_stream and (float(uniform_t2t_fraction) > 0.0 or float(ar_loss_weight) > 0.0):
        raise ValueError("BD3 uniform T2T and causal AR losses require model.bd3lm_clean_stream_mode='causal'")
    if not causal_clean_stream:
        return

    failures: list[str] = []
    if not config.data.content_budget_segmented_packing:
        failures.append("data.content_budget_segmented_packing must be true")
    if not config.data.prepend_bos or config.data.loss_on_bos:
        failures.append("segmented AR boundaries require prepend_bos=true and loss_on_bos=false")
    if config.model.output_factorization != "flat" or config.model.auxiliary_head_layers:
        failures.append("flat output without auxiliary heads is required")
    if config.model.length_head or config.model.target_length_conditioning or config.model.mask_position_embedding:
        failures.append("length and mask-position extensions must be disabled")
    if (
        config.model.token_edit_adapter_bottleneck_dim
        or config.model.token_edit_adapter_top_layers
        or config.model.token_edit_provenance_embedding
    ):
        failures.append("token-edit model extensions must be disabled")
    if config.train.token_edit_corruption_batch_fraction != 0 or config.train.token_edit_corruption_ratio != 0:
        failures.append("the separate token-edit objective must be disabled")
    if (
        config.train.auxiliary_loss_weight != 0
        or config.train.entropy_bonus_beta != 0
        or config.train.confidence_penalty_weight != 0
        or config.train.z_loss_weight != 0
        or config.train.eos_hazard_loss_weight != 0
        or config.train.length_head_loss_weight != 0
        or config.train.position_decile_balanced_loss
        or config.train.uniform_loss_estimator != "single_sample"
        or config.train.label_smoothing_epsilon != 0
    ):
        failures.append("unrelated auxiliary training losses and estimators must be disabled")
    if config.eval.enabled:
        failures.append("generic evaluation must be disabled; use the block-conversion evaluator")
    if failures:
        raise ValueError("causal-clean BD3 block conversion requires: " + "; ".join(failures))
