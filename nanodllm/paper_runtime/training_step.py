from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import training_joint_two_view as joint_two_view
from .mdlm_objective_controls import target_weights as mdlm_target_weights
from .nanochat_muon_schedule import apply as _set_optimizer_schedules
from .parameter_state_hash import append_parameter_state_hash_record_once, parameter_state_hash_record
from .parameterization import (
    parameterization_gradient_rms_metrics,
    parameterization_parameter_update_rms_metrics,
)
from .telemetry import append_jsonl, flush_tensorboard_writer
from .telemetry import write_tensorboard_scalars as _write_tensorboard_scalars
from .training_adapter_gradients import _isolate_token_edit_adapter_gradients as _isolate_token_edit_adapter_gradients
from .training_block_conversion import (
    block_conversion_training_block_size,
    block_conversion_training_block_size_phase_index,
)
from .training_block_conversion_step import run_block_conversion_microbatch
from .training_checkpoint import (
    _all_reduce_numeric_accumulator,
    _collect_resume_contract,
    _merge_numeric_accumulator,
    _packed_all_reduce_float_counters,
    _packed_all_reduce_int_counters,
    alias_checkpoint,
    periodic_path,
    save_checkpoint,
)
from .training_compute import BD3LM_DENSE_CONCAT_FLOPS_ACCOUNTING_METHOD, training_compute_estimate
from .training_empty_target import backward_empty_ddp_microbatch
from .training_evaluation import _evaluate_fixed_noise_ce_trace, _run_eval_with_runtime
from .training_evaluation_runtime import run_periodic_evaluation
from .training_features import (
    _is_causal,
    _self_generated_draft_device_batch_size,
    _token_edit_detaches_backbone,
    _token_edit_probability_bounds,
    _token_edit_support_policy,
    _token_edit_uses_residual_adapter,
    _uses_bd3lm,
    _uses_bd3lm_causal_ar_auxiliary,
    _uses_bd3lm_causal_clean_stream,
    _uses_block_causal_pilot,
    _uses_differentiable_self_generated_draft,
    _uses_fixed_row_token_edit_layout,
    _uses_length_head,
    _uses_matched_token_edit_schedule,
    _uses_mixed_token_edit_canvas,
    _uses_per_token_noise_levels,
    _uses_self_generated_token_edit,
    _uses_token_edit_corruption,
    _uses_token_edit_prefix,
    _uses_trajectory_token_edit_canvas,
    _uses_uniform_low_variance_estimator,
)
from .training_mdlm_auxiliary import run_mdlm_auxiliary_microbatch
from .training_metrics import (
    LOGIT_SCALE_HIST_BINS,
    PER_TOKEN_NOISE_LEVEL_HIST_BINS,
    TOP1_PROB_HIST_BINS,
    _accumulate_probability_histogram,
    _accumulate_value_histogram,
    _component_update_grad_ratio_metrics,
    _component_update_norm_metrics,
    _eos_hazard_accumulator_metrics,
    _eos_hazard_calibration_loss,
    _grad_norm,
    _gradient_component_norm_metrics,
    _layerwise_gradient_norm_metrics,
    _length_head_accumulator_metrics,
    _length_head_loss_and_stats,
    _lm_head_row_norm_metrics,
    _logit_confidence_statistics,
    _logit_scale_statistics,
    _merge_token_edit_reconstruction_accumulator,
    _new_eos_hazard_accumulator,
    _new_length_head_accumulator,
    _new_position_decile_ce_accumulator,
    _new_token_edit_reconstruction_accumulator,
    _parameter_norm_metrics,
    _per_target_t_values,
    _per_token_effective_masked_fraction_metrics,
    _position_decile_balanced_ce_loss,
    _position_decile_ce_accumulator_metrics,
    _prefixed_metric_keys,
    _probability_histogram_metrics,
    _residual_branch_rms_metrics,
    _residual_projection_init_metrics,
    _second_moment_norm,
    _token_edit_reconstruction_metrics,
    _update_eos_hazard_accumulator,
    _update_length_head_accumulator,
    _update_norm,
    _update_position_decile_ce_accumulator,
    _update_token_edit_reconstruction_accumulator,
    _value_histogram_metrics,
    _weighted_token_mean,
)
from .training_mtp import causal_training_loss, mtp_metrics, mtp_training_enabled
from .training_objective import (
    _bd3lm_noised_half_indices,
    _build_uniform_soft_input_embeddings,
    _causal_ar_label_smoothing,
    _corrupt_token_edit_training_batch,
    _corrupt_training_batch,
    _fast_dllm_loss_normalization_scale,
    _forward_training_model,
    _soft_input_curriculum_mix,
    _soft_input_curriculum_phase,
    _target_lengths_for_conditioning,
    _token_edit_branch_enabled_for_microbatch,
    _token_edit_loss_normalization_scale,
    _token_edit_ordinary_eligible_mask,
    _training_targets_from_indices,
    _uniform_two_sample_mean_loss,
    _update_eos_supervision_batch_counts,
)
from .training_optim import (
    _optimizer_lr_metrics,
    _primary_optimizer_lr,
    _set_optimizer_lrs,
    _step_lr_multiplier,
    _step_optimizers,
    _unscale_optimizers,
    _zero_grad,
)
from .training_rng import _restore_rng_state, _rng_state
from .training_row_mixed_edit import (
    accumulate_fixed_row_step_branches,
    assemble_fixed_row_correction_loss,
    fixed_row_lossable_partition,
    fixed_row_t2t_observations,
    prepare_fixed_row_draft,
)
from .training_runtime import (
    _apply_block_causal_pilot_training,
    _batch_model_attention_mask,
    print0,
)
from .training_self_generated_edit import (
    accumulate_self_generated_correction_step,
    build_self_generated_correction_objective_for_session,
    coerce_self_generated_edit_accumulator,
    merge_self_generated_edit_accumulator,
    new_self_generated_edit_accumulator,
    self_generated_edit_metrics,
    self_generated_objective_configuration_telemetry,
    self_generated_t2t_target_weights,
)
from .training_state import TrainingSession, _content_byte_counts, _current_trainer_state, _record_ar_boundary_batch
from .training_token_edit_prefix import (
    apply_token_edit_prefix_loss_weights,
    prepare_self_generated_edit_draft_with_prefix,
    prepare_token_edit_prefix_microbatch,
    token_edit_prefix_target_mean,
    token_edit_prefix_target_means,
)
from .training_token_edit_trajectory import apply_trajectory_loss_scope
from .training_variable_depth import metrics as variable_depth_metrics
from .training_variable_depth import microbatch_kwargs as variable_depth_kwargs


def _parameterization_role_telemetry_enabled(session: TrainingSession) -> bool:
    """Keep detailed O(P) role reductions on explicitly bound campaigns only."""

    return (
        session.record_total_update_norm
        and session.rank == 0
        and session.config.parameterization.contract_binding == "required"
    )


def _reduce_step_accumulators(session: TrainingSession) -> None:
    session.step_counters = _packed_all_reduce_int_counters(
        {
            "processed_tokens": session.step_processed_tokens,
            "valid_tokens": session.step_valid_tokens,
            "lossable_tokens": session.step_lossable_tokens,
            "active_tokens": session.step_active_tokens,
            "content_tokens": session.step_content_tokens,
            "content_bytes": session.step_content_bytes,
            "m2t_microbatches": session.step_m2t_microbatches,
            "m2t_active_tokens": session.step_m2t_active_tokens,
            "m2t_lossable_tokens": session.step_m2t_lossable_tokens,
            "t2t_microbatches": session.step_t2t_microbatches,
            "t2t_active_tokens": session.step_t2t_active_tokens,
            "t2t_lossable_tokens": session.step_t2t_lossable_tokens,
            "t2t_ordinary_eligible_tokens": session.step_t2t_ordinary_eligible_tokens,
            "t2t_eligible_tokens": session.step_t2t_eligible_tokens,
            "visible_prefix_count": session.step_visible_prefix_count,
            "loss_count": session.step_loss_count,
            "m2t_loss_count": session.step_m2t_loss_count,
            "t2t_loss_count": session.step_t2t_loss_count,
            "self_generated_correction_target_count": session.step_self_generated_correction_target_count,
            **joint_two_view.joint_two_view_step_int_counters(session),
            "token_edit_prefix_correction_loss_targets": session.step_token_edit_prefix_correction_loss_targets,
            "t_count": session.step_t_count,
            "soft_input_stat_count": session.step_soft_input_stat_count,
            "uniform_estimator_stat_count": session.step_uniform_estimator_stat_count,
            "per_token_effective_masked_fraction_count": session.step_per_token_effective_masked_fraction_count,
        },
        device=session.device,
        is_ddp=session.is_ddp,
    )
    session.step_processed_tokens = session.step_counters["processed_tokens"]
    session.step_valid_tokens = session.step_counters["valid_tokens"]
    session.step_lossable_tokens = session.step_counters["lossable_tokens"]
    session.step_active_tokens = session.step_counters["active_tokens"]
    session.step_content_tokens = session.step_counters["content_tokens"]
    session.step_content_bytes = session.step_counters["content_bytes"]
    session.step_m2t_microbatches = session.step_counters["m2t_microbatches"]
    session.step_m2t_active_tokens = session.step_counters["m2t_active_tokens"]
    session.step_m2t_lossable_tokens = session.step_counters["m2t_lossable_tokens"]
    session.step_t2t_microbatches = session.step_counters["t2t_microbatches"]
    session.step_t2t_active_tokens = session.step_counters["t2t_active_tokens"]
    session.step_t2t_lossable_tokens = session.step_counters["t2t_lossable_tokens"]
    session.step_t2t_ordinary_eligible_tokens = session.step_counters["t2t_ordinary_eligible_tokens"]
    session.step_t2t_eligible_tokens = session.step_counters["t2t_eligible_tokens"]
    session.step_visible_prefix_count = session.step_counters["visible_prefix_count"]
    session.step_loss_count = session.step_counters["loss_count"]
    session.step_m2t_loss_count = session.step_counters["m2t_loss_count"]
    session.step_t2t_loss_count = session.step_counters["t2t_loss_count"]
    session.step_self_generated_correction_target_count = session.step_counters[
        "self_generated_correction_target_count"
    ]
    joint_two_view.apply_reduced_joint_two_view_step_int_counters(session, session.step_counters)
    session.step_token_edit_prefix_correction_loss_targets = session.step_counters[
        "token_edit_prefix_correction_loss_targets"
    ]
    session.step_t_count = session.step_counters["t_count"]
    session.step_soft_input_stat_count = session.step_counters["soft_input_stat_count"]
    session.step_uniform_estimator_stat_count = session.step_counters["uniform_estimator_stat_count"]
    session.step_per_token_effective_masked_fraction_count = session.step_counters[
        "per_token_effective_masked_fraction_count"
    ]
    session.step_float_sums = _packed_all_reduce_float_counters(
        {
            "loss_sum": session.step_loss_sum,
            "main_ce_sum": session.step_main_ce_sum,
            "main_loss_sum": session.step_main_loss_sum,
            "confidence_penalty_sum": session.step_confidence_penalty_sum,
            "confidence_penalty_loss_sum": session.step_confidence_penalty_loss_sum,
            "z_loss_sum": session.step_z_loss_sum,
            "z_loss_loss_sum": session.step_z_loss_loss_sum,
            "eos_hazard_calibration_loss_sum": session.step_eos_hazard_calibration_loss_sum,
            "eos_hazard_loss_sum": session.step_eos_hazard_loss_sum,
            "class_ce_sum": session.step_class_ce_sum,
            "within_class_ce_sum": session.step_within_class_ce_sum,
            "ar_unsmoothed_ce_sum": session.step_ar_unsmoothed_ce_sum,
            "m2t_ce_sum": session.step_m2t_ce_sum,
            "m2t_loss_sum": session.step_m2t_loss_sum,
            "t2t_ce_sum": session.step_t2t_ce_sum,
            "t2t_loss_sum": session.step_t2t_loss_sum,
            "self_generated_expected_error_sum": session.step_self_generated_expected_error_sum,
            "self_generated_draft_preference_sum": session.step_self_generated_draft_preference_sum,
            **joint_two_view.joint_two_view_step_float_sums(session),
            "t_sum": session.step_t_sum,
            "schedule_multiplier_sum": session.step_schedule_multiplier_sum,
            "loss_normalization_scale_sum": session.step_loss_normalization_scale_sum,
            "entropy_sum": session.step_entropy_sum,
            "top1_prob_sum": session.step_top1_prob_sum,
            "argmax_mode_sum": session.step_argmax_mode_sum,
            "logit_norm_sum": session.step_logit_norm_sum,
            "logit_rms_sum": session.step_logit_rms_sum,
            "logsumexp_sum": session.step_logsumexp_sum,
            "visible_prefix_frac_sum": session.step_visible_prefix_frac_sum,
            "per_token_effective_masked_fraction_sum": session.step_per_token_effective_masked_fraction_sum,
        },
        device=session.device,
        is_ddp=session.is_ddp,
    )
    session.step_loss_sum = session.step_float_sums["loss_sum"]
    session.step_main_ce_sum = session.step_float_sums["main_ce_sum"]
    session.step_main_loss_sum = session.step_float_sums["main_loss_sum"]
    session.step_confidence_penalty_sum = session.step_float_sums["confidence_penalty_sum"]
    session.step_confidence_penalty_loss_sum = session.step_float_sums["confidence_penalty_loss_sum"]
    session.step_z_loss_sum = session.step_float_sums["z_loss_sum"]
    session.step_z_loss_loss_sum = session.step_float_sums["z_loss_loss_sum"]
    session.step_eos_hazard_calibration_loss_sum = session.step_float_sums["eos_hazard_calibration_loss_sum"]
    session.step_eos_hazard_loss_sum = session.step_float_sums["eos_hazard_loss_sum"]
    session.step_class_ce_sum = session.step_float_sums["class_ce_sum"]
    session.step_within_class_ce_sum = session.step_float_sums["within_class_ce_sum"]
    session.step_ar_unsmoothed_ce_sum = session.step_float_sums["ar_unsmoothed_ce_sum"]
    session.step_m2t_ce_sum = session.step_float_sums["m2t_ce_sum"]
    session.step_m2t_loss_sum = session.step_float_sums["m2t_loss_sum"]
    session.step_t2t_ce_sum = session.step_float_sums["t2t_ce_sum"]
    session.step_t2t_loss_sum = session.step_float_sums["t2t_loss_sum"]
    session.step_self_generated_expected_error_sum = session.step_float_sums["self_generated_expected_error_sum"]
    session.step_self_generated_draft_preference_sum = session.step_float_sums["self_generated_draft_preference_sum"]
    joint_two_view.apply_reduced_joint_two_view_step_float_sums(session, session.step_float_sums)
    session.step_t_sum = session.step_float_sums["t_sum"]
    session.step_schedule_multiplier_sum = session.step_float_sums["schedule_multiplier_sum"]
    session.step_loss_normalization_scale_sum = session.step_float_sums["loss_normalization_scale_sum"]
    session.step_entropy_sum = session.step_float_sums["entropy_sum"]
    session.step_top1_prob_sum = session.step_float_sums["top1_prob_sum"]
    session.step_argmax_mode_sum = session.step_float_sums["argmax_mode_sum"]
    session.step_logit_norm_sum = session.step_float_sums["logit_norm_sum"]
    session.step_logit_rms_sum = session.step_float_sums["logit_rms_sum"]
    session.step_logsumexp_sum = session.step_float_sums["logsumexp_sum"]
    session.step_visible_prefix_frac_sum = session.step_float_sums["visible_prefix_frac_sum"]
    session.step_per_token_effective_masked_fraction_sum = session.step_float_sums[
        "per_token_effective_masked_fraction_sum"
    ]
    session.step_aux_ce_sums = list(
        _all_reduce_numeric_accumulator(
            {"values": session.step_aux_ce_sums}, device=session.device, is_ddp=session.is_ddp
        )["values"]
    )
    session.step_soft_input_stat_sums = dict(
        _all_reduce_numeric_accumulator(session.step_soft_input_stat_sums, device=session.device, is_ddp=session.is_ddp)
    )
    session.step_uniform_estimator_stat_sums = dict(
        _all_reduce_numeric_accumulator(
            session.step_uniform_estimator_stat_sums, device=session.device, is_ddp=session.is_ddp
        )
    )
    session.reduced_histograms = _all_reduce_numeric_accumulator(
        {
            "top1": session.step_top1_prob_hist_counts,
            "entropy": session.step_entropy_hist_counts,
            "logsumexp": session.step_logsumexp_hist_counts,
            "logit_norm": session.step_logit_norm_hist_counts,
            "per_token_noise": session.step_per_token_noise_level_hist_counts,
        },
        device=session.device,
        is_ddp=session.is_ddp,
    )
    session.step_top1_prob_hist_counts = list(session.reduced_histograms["top1"])
    session.step_entropy_hist_counts = list(session.reduced_histograms["entropy"])
    session.step_logsumexp_hist_counts = list(session.reduced_histograms["logsumexp"])
    session.step_logit_norm_hist_counts = list(session.reduced_histograms["logit_norm"])
    session.step_per_token_noise_level_hist_counts = list(session.reduced_histograms["per_token_noise"])
    session.step_eos_hazard_accumulator = _all_reduce_numeric_accumulator(
        session.step_eos_hazard_accumulator, device=session.device, is_ddp=session.is_ddp
    )
    session.step_length_head_accumulator = _all_reduce_numeric_accumulator(
        session.step_length_head_accumulator, device=session.device, is_ddp=session.is_ddp
    )
    session.step_position_decile_ce_accumulator = _all_reduce_numeric_accumulator(
        session.step_position_decile_ce_accumulator, device=session.device, is_ddp=session.is_ddp
    )
    session.step_token_edit_reconstruction_accumulator = _all_reduce_numeric_accumulator(
        session.step_token_edit_reconstruction_accumulator, device=session.device, is_ddp=session.is_ddp
    )
    session.step_self_generated_edit_accumulator = coerce_self_generated_edit_accumulator(
        _all_reduce_numeric_accumulator(
            session.step_self_generated_edit_accumulator,
            device=session.device,
            is_ddp=session.is_ddp,
        ),
        field_name="step_self_generated_edit_accumulator",
    )
    session.step_mins = _packed_all_reduce_float_counters(
        {
            "t": session.step_t_min,
            "logit_norm": session.step_logit_norm_min,
            "logsumexp": session.step_logsumexp_min,
            "per_token_effective_masked_fraction": session.step_per_token_effective_masked_fraction_min,
            **joint_two_view.joint_two_view_step_minima(session),
        },
        device=session.device,
        is_ddp=session.is_ddp,
        op=dist.ReduceOp.MIN,
    )
    session.step_maxes = _packed_all_reduce_float_counters(
        {
            "t": session.step_t_max,
            "logit_norm": session.step_logit_norm_max,
            "logsumexp": session.step_logsumexp_max,
            "per_token_effective_masked_fraction": session.step_per_token_effective_masked_fraction_max,
            **joint_two_view.joint_two_view_step_maxima(session),
        },
        device=session.device,
        is_ddp=session.is_ddp,
        op=dist.ReduceOp.MAX,
    )
    session.step_t_min = session.step_mins["t"]
    session.step_logit_norm_min = session.step_mins["logit_norm"]
    session.step_logsumexp_min = session.step_mins["logsumexp"]
    session.step_per_token_effective_masked_fraction_min = session.step_mins["per_token_effective_masked_fraction"]
    session.step_t_max = session.step_maxes["t"]
    session.step_logit_norm_max = session.step_maxes["logit_norm"]
    session.step_logsumexp_max = session.step_maxes["logsumexp"]
    session.step_per_token_effective_masked_fraction_max = session.step_maxes["per_token_effective_masked_fraction"]
    joint_two_view.apply_reduced_joint_two_view_step_extrema(session, session.step_mins, session.step_maxes)


def _prepare_training_step(session: TrainingSession, step: int) -> None:
    """Apply the step schedule, clear gradients, and reset step-local state.

    This is the sole initialization boundary for values accumulated by one
    optimizer step. Keep schedule application and gradient clearing first, and
    keep every value here free of RNG consumption so training data and noise
    order remain independent of lifecycle bookkeeping.
    """

    session.lrm = _step_lr_multiplier(session.config, step, session.total_steps, session.warmup_steps)
    if _uses_bd3lm_causal_clean_stream(session.config):
        session.block_conversion_training_block_size = block_conversion_training_block_size(session.config, step)
        session.block_conversion_training_block_size_phase_index = block_conversion_training_block_size_phase_index(
            session.config, step
        )
        if session.config.train.bd3lm_recipe_mechanism != "none":
            from .training_block_recipe import prepare_recipe_step

            prepare_recipe_step(session, step)
    _set_optimizer_lrs(session.optimizers, session.lrm)
    _set_optimizer_schedules(session.optimizers, session.config, step, session.total_steps)
    _zero_grad(session.optimizers)

    # Optional first-step diagnostics capture their input during the microbatch
    # phase, after all normal corruption and attention-mask construction.
    session.record_early_residual_diagnostics = (
        step == 0 and session.config.model.residual_output_projection_init == "scaled_normal"
    )
    session.residual_diagnostic_batch: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None

    # Objective sums and structured diagnostic accumulators.
    session.step_loss_sum = 0.0
    session.step_main_ce_sum = 0.0
    session.step_main_loss_sum = 0.0
    session.step_confidence_penalty_sum = 0.0
    session.step_confidence_penalty_loss_sum = 0.0
    session.step_z_loss_sum = 0.0
    session.step_z_loss_loss_sum = 0.0
    session.step_eos_hazard_calibration_loss_sum = 0.0
    session.step_eos_hazard_loss_sum = 0.0
    session.step_eos_hazard_accumulator = _new_eos_hazard_accumulator()
    session.step_length_head_accumulator = _new_length_head_accumulator(int(session.config.model.max_seq_len))
    session.step_position_decile_ce_accumulator = _new_position_decile_ce_accumulator()
    session.step_token_edit_reconstruction_accumulator = _new_token_edit_reconstruction_accumulator()
    session.step_self_generated_edit_accumulator = new_self_generated_edit_accumulator()
    session.step_self_generated_correction_target_count = 0
    session.step_token_edit_prefix_correction_loss_targets = 0
    session.step_self_generated_expected_error_sum = 0.0
    session.step_self_generated_draft_preference_sum = 0.0
    joint_two_view.reset_joint_two_view_step_metrics(session)
    session.step_class_ce_sum = 0.0
    session.step_within_class_ce_sum = 0.0
    session.step_ar_unsmoothed_ce_sum = 0.0
    session.step_aux_ce_sums = (
        [0.0 for _ in session.model_config.auxiliary_head_layers] if session.return_aux_logits else []
    )
    session.step_loss_count = 0

    # Branch-specific objective and token counts.
    session.step_m2t_ce_sum = 0.0
    session.step_m2t_loss_sum = 0.0
    session.step_m2t_loss_count = 0
    session.step_m2t_microbatches = 0
    session.step_m2t_active_tokens = 0
    session.step_m2t_lossable_tokens = 0
    session.step_t2t_ce_sum = 0.0
    session.step_t2t_loss_sum = 0.0
    session.step_t2t_loss_count = 0
    session.step_t2t_microbatches = 0
    session.step_t2t_active_tokens = 0
    session.step_t2t_lossable_tokens = 0
    session.step_t2t_ordinary_eligible_tokens = 0
    session.step_t2t_eligible_tokens = 0
    session.step_processed_tokens = 0
    session.step_active_tokens = 0
    session.step_lossable_tokens = 0
    session.step_valid_tokens = 0
    session.step_content_tokens = 0
    session.step_content_bytes = 0

    # Noise, schedule, and confidence telemetry.
    session.step_t_sum = 0.0
    session.step_t_min = float("inf")
    session.step_t_max = float("-inf")
    session.step_t_count = 0
    session.step_schedule_multiplier_sum = 0.0
    session.step_loss_normalization_scale_sum = 0.0
    session.step_entropy_sum = 0.0
    session.step_top1_prob_sum = 0.0
    session.step_argmax_mode_sum = 0.0
    session.step_top1_prob_hist_counts = [0 for _ in range(TOP1_PROB_HIST_BINS)]
    session.step_entropy_hist_counts = [0 for _ in range(LOGIT_SCALE_HIST_BINS)]
    session.step_logsumexp_hist_counts = [0 for _ in range(LOGIT_SCALE_HIST_BINS)]
    session.step_logit_norm_hist_counts = [0 for _ in range(LOGIT_SCALE_HIST_BINS)]
    session.step_logit_norm_sum = 0.0
    session.step_logit_norm_min = float("inf")
    session.step_logit_norm_max = float("-inf")
    session.step_logit_rms_sum = 0.0
    session.step_logsumexp_sum = 0.0
    session.step_logsumexp_min = float("inf")
    session.step_logsumexp_max = float("-inf")
    session.step_visible_prefix_frac_sum = 0.0
    session.step_visible_prefix_count = 0

    # Optional feature accumulators retain their existing dynamic key sets.
    session.step_soft_input_stat_sums: dict[str, float] = {}
    session.step_soft_input_stat_count = 0
    session.step_uniform_estimator_stat_sums: dict[str, float] = {}
    session.step_uniform_estimator_stat_count = 0
    session.step_uniform_estimator_label = session.config.train.uniform_loss_estimator
    session.step_per_token_noise_level_hist_counts = [0 for _ in range(PER_TOKEN_NOISE_LEVEL_HIST_BINS)]
    session.step_per_token_effective_masked_fraction_sum = 0.0
    session.step_per_token_effective_masked_fraction_count = 0
    session.step_per_token_effective_masked_fraction_min = float("inf")
    session.step_per_token_effective_masked_fraction_max = float("-inf")
    session.step_soft_input_phase = _soft_input_curriculum_phase(
        session.config, _soft_input_curriculum_mix(session.config, step, session.total_steps)
    )


def _unscale_step_gradients(session: TrainingSession) -> bool:
    """Unscale real step gradients and report whether an update may run.

    ``step_loss_count`` is globally reduced before this boundary. DDP ranks that
    had no local targets still hold collective-parity gradients when a peer had
    targets, so they must participate in the normal update. If every rank and
    microbatch was empty, retaining the scaled zero gradients while skipping
    unscale/step/update keeps optimizer, weight-decay, and GradScaler state from
    advancing; the next step's normal ``zero_grad`` clears those dummy gradients.
    """

    if session.step_loss_count == 0:
        return False
    _unscale_optimizers(session.scaler, session.optimizers)
    return True


def _advance_step_optimizers(session: TrainingSession, *, enabled: bool) -> None:
    """Advance optimizer and GradScaler state only for a globally nonempty step."""

    if not enabled:
        return
    _step_optimizers(session.scaler, session.optimizers)
    session.scaler.update()


def _run_training_steps(session: TrainingSession) -> None:
    for step in range(session.start_step, session.total_steps):
        _prepare_training_step(session, step)
        for micro in range(session.grad_accum_steps):
            depth_kwargs = variable_depth_kwargs(session.config, step, micro, session.grad_accum_steps)
            session.step_processed_tokens += int(session.x_0.numel())
            if _is_causal(session.config):
                if session.record_early_residual_diagnostics and session.residual_diagnostic_batch is None:
                    session.diagnostic_attention_mask = _batch_model_attention_mask(
                        session.batch, session.attention_mask
                    )
                    session.diagnostic_position_ids = (
                        session.batch.position_ids.detach()
                        if getattr(session.batch, "position_ids", None) is not None
                        else None
                    )
                    session.residual_diagnostic_batch = (
                        session.x_0.detach(),
                        session.diagnostic_attention_mask.detach()
                        if session.diagnostic_attention_mask is not None
                        else None,
                        session.diagnostic_position_ids,
                    )
                _record_ar_boundary_batch(
                    session,
                    session.x_0,
                    session.loss_mask,
                    session.attention_mask,
                    getattr(session.batch, "position_ids", None),
                    getattr(session.batch, "document_boundary_mask", None),
                    getattr(session.batch, "document_start_mask", None),
                )
                session.valid_tokens = int(session.attention_mask.sum().item())
                session.lossable_tokens = int(session.loss_mask.sum().item())
                session.content_tokens, session.content_bytes = _content_byte_counts(
                    session, session.x_0, session.attention_mask
                )
                session.step_valid_tokens += session.valid_tokens
                session.step_lossable_tokens += session.lossable_tokens
                session.step_content_tokens += session.content_tokens
                session.step_content_bytes += session.content_bytes
                session.active_mask = session.loss_mask & session.attention_mask
                session.step_active_tokens += int(session.active_mask.sum().item())
                session.sync_context = (
                    session.model.no_sync if session.is_ddp and micro < session.grad_accum_steps - 1 else nullcontext
                )
                with session.sync_context():
                    with session.autocast_ctx:
                        session.model_kwargs = dict(depth_kwargs)
                        if getattr(session.batch, "position_ids", None) is not None:
                            session.model_kwargs["position_ids"] = session.batch.position_ids
                        session.model_kwargs["attention_mask"] = _batch_model_attention_mask(
                            session.batch, session.attention_mask
                        )
                        causal_training_loss(session, label_smoothing=_causal_ar_label_smoothing)
                    session.loss_value = float(session.loss.detach().item())
                    if not math.isfinite(session.loss_value):
                        raise RuntimeError(f"non-finite loss at step {step}, micro {micro}: {session.loss_value}")
                    session.step_loss_sum += session.loss_value
                    session.step_main_ce_sum += float(session.raw_main_ce.detach().item())
                    session.step_main_loss_sum += float(session.main_loss_for_log.detach().item())
                    session.step_ar_unsmoothed_ce_sum += float(session.raw_unsmoothed_ce.detach().item())
                    session.step_entropy_sum += float(session.raw_entropy.detach().item())
                    session.step_top1_prob_sum += float(session.raw_max_prob.detach().item())
                    session.step_argmax_mode_sum += float(session.raw_argmax_mode_frac.detach().item())
                    session.step_loss_normalization_scale_sum += 1.0
                    session.step_loss_count += (
                        int(session.target_mask.any()) if session.config.model.mtp_mode != "none" else 1
                    )
                    session.scaler.scale(session.loss / session.grad_accum_steps).backward()
                session.batch = next(session.train_loader)
                session.x_0 = session.batch.x_0
                session.loss_mask = session.batch.loss_mask
                session.attention_mask = session.batch.attention_mask
                session.epoch = session.batch.epoch
                continue
            if _uses_bd3lm_causal_clean_stream(session.config):
                run_block_conversion_microbatch(
                    session,
                    step=step,
                    micro=micro,
                    backward_empty_ddp_microbatch=backward_empty_ddp_microbatch,
                )
                session.batch = next(session.train_loader)
                session.x_0 = session.batch.x_0
                session.loss_mask = session.batch.loss_mask
                session.attention_mask = session.batch.attention_mask
                session.epoch = session.batch.epoch
                continue
            if run_mdlm_auxiliary_microbatch(session, step=step, micro=micro):
                session.batch = next(session.train_loader)
                session.x_0 = session.batch.x_0
                session.loss_mask = session.batch.loss_mask
                session.attention_mask = session.batch.attention_mask
                session.epoch = session.batch.epoch
                continue
            session.global_microbatch_idx = step * session.grad_accum_steps + micro
            session.row_mixed_token_edit = _uses_fixed_row_token_edit_layout(session.config)
            session.token_edit_branch = (
                False
                if session.row_mixed_token_edit
                else _token_edit_branch_enabled_for_microbatch(session.config, session.global_microbatch_idx)
            )
            session.self_generated_edit_branch = session.token_edit_branch and _uses_self_generated_token_edit(
                session.config
            )
            session.self_generated_draft_correct = None
            session.self_generated_drafted = None
            session.self_generated_draft_m2t_logits = None
            session.self_generated_draft_target_probabilities = None
            session.token_edit_prefix_plan = None
            session.token_edit_prefix_inputs = None
            session.token_edit_correction_loss_target_mask = None
            session.token_edit_loss_denominator = None
            session.token_edit_trajectory_model_forwards = 0
            session.token_edit_trajectory_committed_targets = 0
            session.token_edit_trajectory_remaining_targets = 0
            session.row_mixed_t2t_target_mask = None
            session.fixed_row_t2t_rows = None
            session.token_edit_input_provenance = None
            if session.token_edit_branch:
                if not session.self_generated_edit_branch and session.token_edit_class_ids is None:
                    raise RuntimeError("token-edit corruption requires frequency decile token classes")
                if session.self_generated_edit_branch:
                    session.token_edit_eligible_tokens = int(session.loss_mask.sum().item())
                    session.token_edit_ordinary_eligible_tokens = 0
                    session.z_t, session.w, session.t_val, session.noise_level = _corrupt_training_batch(
                        session.config,
                        session.noise,
                        session.x_0,
                        session.loss_mask,
                        position_ids=getattr(session.batch, "position_ids", None),
                    )
                elif _uses_matched_token_edit_schedule(session.config):
                    if session.token_edit_candidate_ids is None:
                        raise RuntimeError("matched token-edit corruption requires ordinary-token candidates")
                    session.token_edit_ordinary_eligible_tokens = int(
                        _token_edit_ordinary_eligible_mask(
                            session.x_0,
                            session.loss_mask,
                            session.token_edit_candidate_ids,
                        )
                        .sum()
                        .item()
                    )
                    session.token_edit_eligible_tokens = session.token_edit_ordinary_eligible_tokens
                else:
                    session.token_edit_ordinary_eligible_tokens = int(session.loss_mask.sum().item())
                    session.token_edit_eligible_tokens = session.token_edit_ordinary_eligible_tokens
                if not session.self_generated_edit_branch:
                    session.z_t, session.w, session.t_val, session.noise_level = _corrupt_token_edit_training_batch(
                        session.config,
                        session.x_0,
                        session.loss_mask,
                        session.token_edit_candidate_ids,
                        vocab_size=session.model_config.vocab_size,
                        mask_token_id=session.model_config.mask_token_id,
                    )
            else:
                session.z_t, session.w, session.t_val, session.noise_level = _corrupt_training_batch(
                    session.config,
                    session.noise,
                    session.x_0,
                    session.loss_mask,
                    position_ids=getattr(session.batch, "position_ids", None),
                )
                session.token_edit_ordinary_eligible_tokens = 0
                session.token_edit_eligible_tokens = 0
            session.soft_input_embeddings = None
            if session.config.noise.soft_input_curriculum:
                session.soft_input_embeddings, session.soft_input_stats = _build_uniform_soft_input_embeddings(
                    session.raw_model,
                    session.config,
                    session.x_0,
                    session.z_t,
                    session.loss_mask,
                    session.t_val,
                    step,
                    session.total_steps,
                )
                session.step_soft_input_phase = str(session.soft_input_stats["soft_input_curriculum_phase"])
                for key, value in session.soft_input_stats.items():
                    if isinstance(value, int | float) and (not isinstance(value, bool)):
                        session.step_soft_input_stat_sums[key] = session.step_soft_input_stat_sums.get(
                            key, 0.0
                        ) + float(value)
                session.step_soft_input_stat_count += 1
            if session.record_early_residual_diagnostics and session.residual_diagnostic_batch is None:
                session.residual_diagnostic_batch = (
                    session.z_t.detach(),
                    _batch_model_attention_mask(session.batch, session.attention_mask).detach(),
                    session.batch.position_ids.detach()
                    if getattr(session.batch, "position_ids", None) is not None
                    else None,
                )
            session.forward_attention_mask = _batch_model_attention_mask(session.batch, session.attention_mask)
            if _uses_block_causal_pilot(session.config):
                session.block_size = int(session.config.model.block_causal_pilot_size)
                (
                    session.z_t,
                    session.w,
                    session.forward_attention_mask,
                    session._in_current_block,
                    session.visible_prefix_frac,
                ) = _apply_block_causal_pilot_training(
                    session.config,
                    session.x_0,
                    session.z_t,
                    session.w,
                    session.attention_mask,
                    session.loss_mask,
                    session.block_size,
                )
                session.step_visible_prefix_frac_sum += float(session.visible_prefix_frac.sum().item())
                session.step_visible_prefix_count += session.x_0.shape[0]
            _record_ar_boundary_batch(
                session,
                session.x_0,
                session.loss_mask,
                session.attention_mask,
                getattr(session.batch, "position_ids", None),
                getattr(session.batch, "document_boundary_mask", None),
                getattr(session.batch, "document_start_mask", None),
            )
            session.active = (session.w > 0) & session.loss_mask
            session.valid_tokens = int(session.attention_mask.sum().item())
            session.lossable_tokens = int(session.loss_mask.sum().item())
            if session.self_generated_edit_branch and _uses_token_edit_prefix(session.config):
                prepare_token_edit_prefix_microbatch(session)
            session.content_tokens, session.content_bytes = _content_byte_counts(
                session, session.x_0, session.attention_mask
            )
            session.step_valid_tokens += session.valid_tokens
            session.step_lossable_tokens += session.lossable_tokens
            if session.row_mixed_token_edit:
                (
                    session.fixed_row_t2t_rows,
                    session.m2t_lossable_tokens,
                    session.t2t_lossable_tokens,
                ) = fixed_row_lossable_partition(session.loss_mask)
                session.step_m2t_lossable_tokens += session.m2t_lossable_tokens
                session.step_t2t_lossable_tokens += session.t2t_lossable_tokens
                session.token_edit_eligible_tokens = session.t2t_lossable_tokens
                session.token_edit_ordinary_eligible_tokens = 0
                session.step_t2t_eligible_tokens += session.token_edit_eligible_tokens
            elif session.token_edit_branch:
                session.step_t2t_lossable_tokens += session.lossable_tokens
                session.step_t2t_ordinary_eligible_tokens += session.token_edit_ordinary_eligible_tokens
                session.step_t2t_eligible_tokens += session.token_edit_eligible_tokens
            else:
                session.step_m2t_lossable_tokens += session.lossable_tokens
            session.step_content_tokens += session.content_tokens
            session.step_content_bytes += session.content_bytes
            if session.eos_supervision_audit_enabled:
                _update_eos_supervision_batch_counts(
                    session.progress.lineage.eos_supervision_train_counts,
                    session.x_0,
                    session.loss_mask,
                    session.attention_mask,
                    session.special_token_ids,
                    session.eos_token_id,
                    session.active,
                )
            if _uses_per_token_noise_levels(session.config):
                if session.token_edit_branch:
                    raise ValueError("per-token noise-level telemetry is only defined for MDLM M2T corruption")
                if not isinstance(session.t_val, torch.Tensor) or session.t_val.shape != session.x_0.shape:
                    raise ValueError(
                        "noise.per_token_noise_levels requires token-shaped t values from training corruption"
                    )
                session.lossable_t_values = session.t_val[session.loss_mask].detach().float()
                _accumulate_value_histogram(
                    session.step_per_token_noise_level_hist_counts, session.lossable_t_values, 0.0, 1.0
                )
                session.effective_masked_fraction = int(session.active.sum().item()) / max(1, session.lossable_tokens)
                session.step_per_token_effective_masked_fraction_sum += session.effective_masked_fraction
                session.step_per_token_effective_masked_fraction_count += 1
                session.step_per_token_effective_masked_fraction_min = min(
                    session.step_per_token_effective_masked_fraction_min, session.effective_masked_fraction
                )
                session.step_per_token_effective_masked_fraction_max = max(
                    session.step_per_token_effective_masked_fraction_max, session.effective_masked_fraction
                )
            if bool(session.active.all().item()):
                session.target_indices = session.all_logits_indices
            else:
                session.target_indices = session.active.reshape(-1).nonzero(as_tuple=True)[0]
            session.logits_indices = (
                _bd3lm_noised_half_indices(session.active) if _uses_bd3lm(session.config) else session.target_indices
            )
            session.sync_context = (
                session.model.no_sync if session.is_ddp and micro < session.grad_accum_steps - 1 else nullcontext
            )
            if session.target_indices.numel() == 0:
                if session.is_ddp:
                    backward_empty_ddp_microbatch(session)
                session.batch = next(session.train_loader)
                session.x_0 = session.batch.x_0
                session.loss_mask = session.batch.loss_mask
                session.attention_mask = session.batch.attention_mask
                session.epoch = session.batch.epoch
                continue
            session.targets = _training_targets_from_indices(session.x_0, session.target_indices)
            session.step_active_tokens += int(session.target_indices.numel())
            if session.row_mixed_token_edit:
                with session.autocast_ctx:
                    prepare_fixed_row_draft(session)
            elif session.self_generated_edit_branch:
                prepare_self_generated_edit_draft_with_prefix(session, step=step, microbatch=micro)
            if _uses_uniform_low_variance_estimator(session.config):
                if isinstance(session.t_val, torch.Tensor) or session.t_val is None:
                    raise ValueError("two-sample uniform estimator requires scalar uniform t")
                if session.use_hierarchical_output or session.return_aux_logits:
                    raise ValueError("two-sample uniform estimator requires flat logits without auxiliary heads")
                with session.sync_context():
                    with session.autocast_ctx:
                        (
                            session.loss,
                            session.raw_main_ce,
                            session.main_loss_for_log,
                            session.loss_normalization_scale,
                            session.uniform_estimator_stats,
                        ) = _uniform_two_sample_mean_loss(
                            session.model,
                            session.config,
                            session.noise,
                            session.x_0,
                            session.z_t,
                            session.loss_mask,
                            session.forward_attention_mask,
                            session.target_indices,
                            float(session.t_val),
                            session.noise_level,
                            session.lossable_tokens,
                        )
                    session.loss_value = float(session.loss.detach().item())
                    if not math.isfinite(session.loss_value):
                        raise RuntimeError(f"non-finite loss at step {step}, micro {micro}: {session.loss_value}")
                    session.step_loss_sum += session.loss_value
                    session.step_main_ce_sum += float(session.raw_main_ce.detach().item())
                    session.step_main_loss_sum += float(session.main_loss_for_log.detach().item())
                    session.step_loss_normalization_scale_sum += float(session.loss_normalization_scale)
                    for key, value in session.uniform_estimator_stats.items():
                        if isinstance(value, int | float) and (not isinstance(value, bool)):
                            session.step_uniform_estimator_stat_sums[key] = (
                                session.step_uniform_estimator_stat_sums.get(key, 0.0) + float(value)
                            )
                        elif key == "uniform_loss_estimator":
                            session.step_uniform_estimator_label = str(value)
                    session.step_uniform_estimator_stat_count += 1
                    if session.t_val is not None:
                        session.step_t_sum += float(session.t_val)
                        session.step_t_min = min(session.step_t_min, float(session.t_val))
                        session.step_t_max = max(session.step_t_max, float(session.t_val))
                        session.step_t_count += 1
                        if session.config.noise.schedule_weight:
                            session.step_schedule_multiplier_sum += 1.0 / max(float(session.t_val), 1e-08)
                    session.step_m2t_ce_sum += float(session.raw_main_ce.detach().item())
                    session.step_m2t_loss_sum += float(session.main_loss_for_log.detach().item())
                    session.step_m2t_loss_count += 1
                    session.step_m2t_microbatches += 1
                    session.step_m2t_active_tokens += int(session.target_indices.numel())
                    session.step_loss_count += 1
                    session.scaler.scale(session.loss / session.grad_accum_steps).backward()
                session.batch = next(session.train_loader)
                session.x_0 = session.batch.x_0
                session.loss_mask = session.batch.loss_mask
                session.attention_mask = session.batch.attention_mask
                session.epoch = session.batch.epoch
                continue
            with session.sync_context():
                with session.autocast_ctx:
                    session.target_lengths = _target_lengths_for_conditioning(
                        session.config, session.x_0, session.attention_mask, session.eos_token_id
                    )
                    session.model_out = _forward_training_model(
                        session.model,
                        session.config,
                        session.z_t,
                        session.x_0,
                        session.forward_attention_mask,
                        logits_indices=session.logits_indices,
                        return_aux_logits=session.return_aux_logits,
                        noise_level=session.noise_level,
                        return_hierarchical_logits=session.use_hierarchical_output,
                        return_length_logits=_uses_length_head(session.config),
                        input_embeddings=session.soft_input_embeddings,
                        input_provenance=session.token_edit_input_provenance,
                        position_ids=getattr(session.batch, "position_ids", None),
                        target_lengths=session.target_lengths,
                        detach_final_hidden=(
                            session.self_generated_edit_branch and _token_edit_detaches_backbone(session.config)
                        ),
                        use_token_edit_adapter=(
                            session.self_generated_edit_branch and _token_edit_uses_residual_adapter(session.config)
                        ),
                        **depth_kwargs,
                    )
                    session.entropy_values = None
                    session.raw_entropy = torch.tensor(0.0, device=session.device)
                    session.raw_max_prob = torch.tensor(0.0, device=session.device)
                    session.raw_argmax_mode_frac = torch.tensor(0.0, device=session.device)
                    session.z_loss_values = None
                    session.logsumexp_values = None
                    session.logit_norms = None
                    session.logit_rms = None
                    session.length_logits = None
                    if session.use_hierarchical_output:
                        if session.return_aux_logits and _uses_length_head(session.config):
                            session.class_logits, session.token_logits, session.aux_logits, session.length_logits = (
                                session.model_out
                            )
                        elif session.return_aux_logits:
                            session.class_logits, session.token_logits, session.aux_logits = session.model_out
                        elif _uses_length_head(session.config):
                            session.class_logits, session.token_logits, session.length_logits = session.model_out
                            session.aux_logits = []
                        else:
                            session.class_logits, session.token_logits = session.model_out
                            session.aux_logits = []
                        session.main_ce_values, session.class_ce_values, session.within_class_ce_values = (
                            session.raw_model.hierarchical_ce(
                                session.class_logits, session.token_logits, session.targets
                            )
                        )
                    else:
                        if session.return_aux_logits and _uses_length_head(session.config):
                            session.logits, session.aux_logits, session.length_logits = session.model_out
                        elif session.return_aux_logits:
                            session.logits, session.aux_logits = session.model_out
                        elif _uses_length_head(session.config):
                            session.logits, session.length_logits = session.model_out
                            session.aux_logits = []
                        else:
                            session.logits, session.aux_logits = (session.model_out, [])
                        session.main_ce_values = F.cross_entropy(session.logits, session.targets, reduction="none")
                        session.class_ce_values = None
                        session.within_class_ce_values = None
                        (
                            session.entropy_values,
                            session.max_probs,
                            session.raw_entropy,
                            session.raw_max_prob,
                            session.raw_argmax_mode_frac,
                        ) = _logit_confidence_statistics(session.logits)
                        _accumulate_probability_histogram(session.step_top1_prob_hist_counts, session.max_probs)
                        _accumulate_value_histogram(session.step_entropy_hist_counts, session.entropy_values, 0.0, 16.0)
                        if session.config.train.z_loss_weight > 0 or session.record_embedding_head_init_diagnostics:
                            session.z_loss_values, session.logsumexp_values, session.logit_norms, session.logit_rms = (
                                _logit_scale_statistics(session.logits)
                            )
                            _accumulate_value_histogram(
                                session.step_logsumexp_hist_counts, session.logsumexp_values, 0.0, 64.0
                            )
                            _accumulate_value_histogram(
                                session.step_logit_norm_hist_counts, session.logit_norms, 0.0, 512.0
                            )
                        else:
                            session.z_loss_values = None
                            session.logsumexp_values = None
                            session.logit_norms = None
                            session.logit_rms = None
                    session.raw_confidence_penalty = (
                        -session.raw_entropy
                        if session.config.train.confidence_penalty_weight > 0
                        else torch.tensor(0.0, device=session.device)
                    )
                    session.confidence_penalty_loss_for_log = torch.tensor(0.0, device=session.device)
                    session.raw_z_loss = torch.tensor(0.0, device=session.device)
                    session.z_loss_for_log = torch.tensor(0.0, device=session.device)
                    if session.config.train.z_loss_weight > 0:
                        if session.z_loss_values is None:
                            raise RuntimeError("z-loss requires flat logits")
                        session.raw_z_loss = session.z_loss_values.mean()
                    session.raw_eos_hazard_calibration_loss = torch.tensor(0.0, device=session.device)
                    session.eos_hazard_loss_for_log = torch.tensor(0.0, device=session.device)
                    session.eos_hazard_stats = None
                    if session.config.train.eos_hazard_loss_weight > 0:
                        if session.use_hierarchical_output:
                            raise RuntimeError("EOS hazard calibration requires flat logits")
                        session.raw_eos_hazard_calibration_loss, session.eos_hazard_stats = (
                            _eos_hazard_calibration_loss(
                                session.logits,
                                session.targets,
                                session.target_indices,
                                session.attention_mask,
                                session.eos_token_id,
                            )
                        )
                    session.raw_length_head_ce = torch.tensor(0.0, device=session.device)
                    session.length_head_loss_for_log = torch.tensor(0.0, device=session.device)
                    session.length_head_stats = None
                    if session.config.train.length_head_loss_weight > 0:
                        if session.length_logits is None:
                            raise RuntimeError("length-head training requires length logits")
                        session.raw_length_head_ce, session.length_head_stats = _length_head_loss_and_stats(
                            session.length_logits, session.x_0, session.attention_mask, session.eos_token_id
                        )
                        session.length_head_loss_for_log = (
                            float(session.config.train.length_head_loss_weight) * session.raw_length_head_ce
                        )
                    session.aux_ce_values = [
                        F.cross_entropy(aux, session.targets, reduction="none") for aux in session.aux_logits
                    ]
                    session.raw_main_ce = token_edit_prefix_target_mean(
                        session.main_ce_values,
                        session.token_edit_correction_loss_target_mask,
                    )
                    session.main_objective_values = session.main_ce_values
                    session.self_generated_correction_objective = None
                    if session.self_generated_edit_branch:
                        if session.self_generated_draft_predictions is None:
                            raise RuntimeError("self-generated edit branch is missing detached draft predictions")
                        if session.self_generated_draft_correct is None:
                            raise RuntimeError("self-generated edit branch is missing detached draft labels")
                        session.self_generated_correction_objective = (
                            build_self_generated_correction_objective_for_session(session)
                        )
                        session.main_objective_values = session.self_generated_correction_objective.loss_values
                    session.raw_class_ce = (
                        session.class_ce_values.mean() if session.class_ce_values is not None else None
                    )
                    session.raw_within_class_ce = (
                        session.within_class_ce_values.mean() if session.within_class_ce_values is not None else None
                    )
                    session.raw_aux_ces = token_edit_prefix_target_means(
                        session.aux_ce_values,
                        session.token_edit_correction_loss_target_mask,
                    )
                    session.position_decile_stats = None
                    if isinstance(session.t_val, torch.Tensor):
                        if session.config.noise.type not in {"mdlm", "bd3lm"}:
                            raise ValueError("tensor t loss path is only supported for MDLM-style objectives")
                        if session.row_mixed_token_edit:
                            assemble_fixed_row_correction_loss(session)
                        elif session.self_generated_edit_branch:
                            if session.self_generated_draft_correct is None:
                                raise RuntimeError("self-generated edit branch is missing detached draft labels")
                            if session.self_generated_drafted is None:
                                raise RuntimeError("self-generated edit branch is missing drafted-position labels")
                            session.token_weights = self_generated_t2t_target_weights(
                                _per_target_t_values(session.t_val, session.x_0, session.target_indices),
                                session.self_generated_draft_correct,
                                correct_weight=float(session.config.train.token_edit_correct_draft_loss_weight),
                                wrong_weight=float(session.config.train.token_edit_wrong_draft_loss_weight),
                                drafted=session.self_generated_drafted,
                            )
                            apply_token_edit_prefix_loss_weights(session)
                            apply_trajectory_loss_scope(session)
                        else:
                            session.token_weights = mdlm_target_weights(
                                _per_target_t_values(session.t_val, session.x_0, session.target_indices),
                                schedule_weight=session.config.noise.schedule_weight,
                                constant_scale=float(session.config.noise.mdlm_unweighted_loss_scale),
                            )
                        session.loss_denominator = (
                            session.token_edit_loss_denominator
                            if session.token_edit_loss_denominator is not None
                            else max(1, session.lossable_tokens)
                        )
                        if session.config.train.position_decile_balanced_loss:
                            session.main_loss, session.position_decile_stats = _position_decile_balanced_ce_loss(
                                session.main_objective_values,
                                session.target_indices,
                                session.attention_mask,
                                session.loss_mask,
                                session.token_weights,
                            )
                            session.aux_losses = [
                                _position_decile_balanced_ce_loss(
                                    aux_ce,
                                    session.target_indices,
                                    session.attention_mask,
                                    session.loss_mask,
                                    session.token_weights,
                                )[0]
                                for aux_ce in session.aux_ce_values
                            ]
                            session.loss_normalization_scale = float(session.token_weights.detach().mean().item())
                        else:
                            session.main_loss = _weighted_token_mean(
                                session.main_objective_values,
                                session.token_weights,
                                session.loss_denominator,
                            )
                            session.aux_losses = [
                                _weighted_token_mean(aux_ce, session.token_weights, session.loss_denominator)
                                for aux_ce in session.aux_ce_values
                            ]
                            session.loss_normalization_scale = (
                                float(session.token_weights.detach().sum().item()) / session.loss_denominator
                            )
                        session.loss = session.main_loss
                        session.main_loss_for_log = session.main_loss
                        for aux_loss in session.aux_losses:
                            session.loss = session.loss + session.config.train.auxiliary_loss_weight * aux_loss
                        if session.config.train.entropy_bonus_beta > 0 and session.entropy_values is not None:
                            session.entropy_loss = (
                                -session.config.train.entropy_bonus_beta
                                * session.entropy_values.float()
                                * session.token_weights
                            ).sum() / session.loss_denominator
                            session.loss = session.loss + session.entropy_loss
                        if session.config.train.confidence_penalty_weight > 0:
                            if session.entropy_values is None:
                                raise RuntimeError("confidence penalty requires flat logits entropy values")
                            session.confidence_penalty_loss = (
                                session.config.train.confidence_penalty_weight
                                * -session.entropy_values.float()
                                * session.token_weights
                            ).sum() / session.loss_denominator
                            session.confidence_penalty_loss_for_log = session.confidence_penalty_loss
                            session.loss = session.loss + session.confidence_penalty_loss
                        if session.config.train.z_loss_weight > 0:
                            if session.z_loss_values is None:
                                raise RuntimeError("z-loss requires flat logits")
                            session.z_loss = (
                                session.config.train.z_loss_weight
                                * session.z_loss_values.float()
                                * session.token_weights
                            ).sum() / session.loss_denominator
                            session.z_loss_for_log = session.z_loss
                            session.loss = session.loss + session.z_loss
                        if session.config.train.eos_hazard_loss_weight > 0:
                            session.eos_hazard_loss = (
                                session.config.train.eos_hazard_loss_weight * session.raw_eos_hazard_calibration_loss
                            )
                            session.eos_hazard_loss_for_log = session.eos_hazard_loss
                            session.loss = session.loss + session.eos_hazard_loss
                    elif session.config.train.position_decile_balanced_loss:
                        session.schedule_multiplier = (
                            1.0 / max(float(session.t_val), 1e-08)
                            if session.config.noise.schedule_weight and session.t_val is not None
                            else 1.0
                        )
                        session.token_weights = torch.full_like(
                            session.main_ce_values, float(session.schedule_multiplier), dtype=torch.float32
                        )
                        session.main_loss, session.position_decile_stats = _position_decile_balanced_ce_loss(
                            session.main_ce_values,
                            session.target_indices,
                            session.attention_mask,
                            session.loss_mask,
                            session.token_weights,
                        )
                        session.aux_losses = [
                            _position_decile_balanced_ce_loss(
                                aux_ce,
                                session.target_indices,
                                session.attention_mask,
                                session.loss_mask,
                                session.token_weights,
                            )[0]
                            for aux_ce in session.aux_ce_values
                        ]
                        session.loss = session.main_loss
                        for aux_loss in session.aux_losses:
                            session.loss = session.loss + session.config.train.auxiliary_loss_weight * aux_loss
                        session.loss_normalization_scale = float(session.schedule_multiplier)
                        session.main_loss_for_log = session.main_loss
                    else:
                        session.main_loss = session.raw_main_ce
                        session.aux_losses = session.raw_aux_ces
                        session.loss = session.main_loss
                        for aux_loss in session.aux_losses:
                            session.loss = session.loss + session.config.train.auxiliary_loss_weight * aux_loss
                        if session.config.train.entropy_bonus_beta > 0 and session.entropy_values is not None:
                            session.entropy_loss = -session.config.train.entropy_bonus_beta * session.raw_entropy
                            session.loss = session.loss + session.entropy_loss
                        if session.config.train.confidence_penalty_weight > 0:
                            session.confidence_penalty_loss = (
                                session.config.train.confidence_penalty_weight * session.raw_confidence_penalty
                            )
                            session.loss = session.loss + session.confidence_penalty_loss
                        if session.config.train.z_loss_weight > 0:
                            session.z_loss = session.config.train.z_loss_weight * session.raw_z_loss
                            session.loss = session.loss + session.z_loss
                        if session.config.train.eos_hazard_loss_weight > 0:
                            session.eos_hazard_loss = (
                                session.config.train.eos_hazard_loss_weight * session.raw_eos_hazard_calibration_loss
                            )
                            session.loss = session.loss + session.eos_hazard_loss
                        session.uses_fast_dllm_scale = session.config.noise.type in {"mdlm", "bd3lm"} or (
                            session.config.noise.type == "uniform" and session.config.noise.time_conditioning
                        )
                        if session.token_edit_branch:
                            session.loss_normalization_scale = _token_edit_loss_normalization_scale(
                                int(session.target_indices.numel()),
                                session.lossable_tokens,
                                float(session.config.train.token_edit_corruption_ratio),
                            )
                        elif session.uses_fast_dllm_scale:
                            session.loss_normalization_scale = _fast_dllm_loss_normalization_scale(
                                int(session.target_indices.numel()),
                                session.lossable_tokens,
                                session.t_val,
                                bool(session.config.noise.schedule_weight),
                            )
                        else:
                            session.loss_normalization_scale = 1.0
                        session.loss = session.loss * session.loss_normalization_scale
                        session.main_loss_for_log = session.main_loss * session.loss_normalization_scale
                        if session.config.train.confidence_penalty_weight > 0:
                            session.confidence_penalty_loss_for_log = (
                                session.confidence_penalty_loss * session.loss_normalization_scale
                            )
                        if session.config.train.z_loss_weight > 0:
                            session.z_loss_for_log = session.z_loss * session.loss_normalization_scale
                        if session.config.train.eos_hazard_loss_weight > 0:
                            session.eos_hazard_loss_for_log = session.eos_hazard_loss * session.loss_normalization_scale
                    if session.config.train.length_head_loss_weight > 0:
                        session.loss = session.loss + session.length_head_loss_for_log
                    joint_two_view.apply_joint_two_view_loss(session)
                session.loss_value = float(session.loss.detach().item())
                if not math.isfinite(session.loss_value):
                    raise RuntimeError(f"non-finite loss at step {step}, micro {micro}: {session.loss_value}")
                session.step_loss_sum += session.loss_value
                session.step_main_ce_sum += float(session.raw_main_ce.detach().item())
                session.step_main_loss_sum += float(session.main_loss_for_log.detach().item())
                session.step_confidence_penalty_sum += float(session.raw_confidence_penalty.detach().item())
                session.step_confidence_penalty_loss_sum += float(
                    session.confidence_penalty_loss_for_log.detach().item()
                )
                session.step_z_loss_sum += float(session.raw_z_loss.detach().item())
                session.step_z_loss_loss_sum += float(session.z_loss_for_log.detach().item())
                session.step_eos_hazard_calibration_loss_sum += float(
                    session.raw_eos_hazard_calibration_loss.detach().item()
                )
                session.step_eos_hazard_loss_sum += float(session.eos_hazard_loss_for_log.detach().item())
                if session.eos_hazard_stats is not None:
                    _update_eos_hazard_accumulator(session.step_eos_hazard_accumulator, session.eos_hazard_stats, 1)
                if session.length_head_stats is not None:
                    _update_length_head_accumulator(session.step_length_head_accumulator, session.length_head_stats, 1)
                if session.position_decile_stats is not None:
                    _update_position_decile_ce_accumulator(
                        session.step_position_decile_ce_accumulator, session.position_decile_stats, 1
                    )
                if session.token_edit_branch and not session.self_generated_edit_branch:
                    if session.token_edit_class_ids is None:
                        raise RuntimeError("token-edit corruption requires frequency decile token classes")
                    _update_token_edit_reconstruction_accumulator(
                        session.step_token_edit_reconstruction_accumulator,
                        session.logits,
                        session.targets,
                        session.token_edit_class_ids,
                        1,
                    )
                    session.step_token_edit_reconstruction_accumulator["microbatches"] = (
                        int(session.step_token_edit_reconstruction_accumulator["microbatches"]) + 1
                    )
                if session.self_generated_edit_branch or session.row_mixed_token_edit:
                    if session.row_mixed_token_edit:
                        if session.row_mixed_t2t_target_mask is None:
                            raise RuntimeError("fixed-row token editing is missing its T2T target partition")
                        correction_logits, correction_targets, correction_ce_values = fixed_row_t2t_observations(
                            session.logits,
                            session.targets,
                            session.main_ce_values,
                            session.row_mixed_t2t_target_mask,
                        )
                    else:
                        correction_logits = session.logits
                        correction_targets = session.targets
                        correction_ce_values = session.main_ce_values
                    accumulate_self_generated_correction_step(
                        session,
                        correction_logits,
                        correction_targets,
                        correction_ce_values,
                        target_mask=session.token_edit_correction_loss_target_mask,
                    )
                if session.raw_class_ce is not None and session.raw_within_class_ce is not None:
                    session.step_class_ce_sum += float(session.raw_class_ce.detach().item())
                    session.step_within_class_ce_sum += float(session.raw_within_class_ce.detach().item())
                session.step_entropy_sum += float(session.raw_entropy.detach().item())
                session.step_top1_prob_sum += float(session.raw_max_prob.detach().item())
                session.step_argmax_mode_sum += float(session.raw_argmax_mode_frac.detach().item())
                if (
                    session.logsumexp_values is not None
                    and session.logit_norms is not None
                    and (session.logit_rms is not None)
                ):
                    session.logit_norms_f = session.logit_norms.detach().float()
                    session.logsumexp_values_f = session.logsumexp_values.detach().float()
                    session.logit_rms_f = session.logit_rms.detach().float()
                    session.step_logit_norm_sum += float(session.logit_norms_f.mean().item())
                    session.step_logit_norm_min = min(
                        session.step_logit_norm_min, float(session.logit_norms_f.min().item())
                    )
                    session.step_logit_norm_max = max(
                        session.step_logit_norm_max, float(session.logit_norms_f.max().item())
                    )
                    session.step_logit_rms_sum += float(session.logit_rms_f.mean().item())
                    session.step_logsumexp_sum += float(session.logsumexp_values_f.mean().item())
                    session.step_logsumexp_min = min(
                        session.step_logsumexp_min, float(session.logsumexp_values_f.min().item())
                    )
                    session.step_logsumexp_max = max(
                        session.step_logsumexp_max, float(session.logsumexp_values_f.max().item())
                    )
                session.step_loss_normalization_scale_sum += float(session.loss_normalization_scale)
                if session.row_mixed_token_edit:
                    accumulate_fixed_row_step_branches(session)
                elif session.token_edit_branch:
                    session.step_t2t_ce_sum += float(session.raw_main_ce.detach().item())
                    session.step_t2t_loss_sum += float(session.main_loss_for_log.detach().item())
                    session.step_t2t_loss_count += 1
                    session.step_t2t_microbatches += 1
                    session.step_t2t_active_tokens += int(session.target_indices.numel())
                else:
                    session.step_m2t_ce_sum += float(session.raw_main_ce.detach().item())
                    session.step_m2t_loss_sum += float(session.main_loss_for_log.detach().item())
                    session.step_m2t_loss_count += 1
                    session.step_m2t_microbatches += 1
                    session.step_m2t_active_tokens += int(session.target_indices.numel())
                for aux_idx, aux_ce in enumerate(session.raw_aux_ces):
                    session.step_aux_ce_sums[aux_idx] += float(aux_ce.detach().item())
                if session.t_val is not None:
                    if isinstance(session.t_val, torch.Tensor):
                        if session.t_val.shape == session.x_0.shape:
                            session.t_values = session.t_val[session.loss_mask].detach().float().flatten()
                        else:
                            session.t_values = session.t_val.detach().float().flatten()
                        if session.t_values.numel() > 0:
                            session.step_t_sum += float(session.t_values.sum().item())
                            session.step_t_min = min(session.step_t_min, float(session.t_values.min().item()))
                            session.step_t_max = max(session.step_t_max, float(session.t_values.max().item()))
                            session.step_t_count += int(session.t_values.numel())
                            if session.config.noise.schedule_weight:
                                session.step_schedule_multiplier_sum += float(
                                    (1.0 / session.t_values.clamp_min(1e-08)).sum().item()
                                )
                    else:
                        session.step_t_sum += float(session.t_val)
                        session.step_t_min = min(session.step_t_min, float(session.t_val))
                        session.step_t_max = max(session.step_t_max, float(session.t_val))
                        session.step_t_count += 1
                        if session.config.noise.schedule_weight:
                            session.step_schedule_multiplier_sum += 1.0 / max(float(session.t_val), 1e-08)
                session.step_loss_count += 1
                adapter_gradient_context = (
                    _isolate_token_edit_adapter_gradients(session.raw_model)
                    if session.self_generated_edit_branch and _token_edit_uses_residual_adapter(session.config)
                    else nullcontext()
                )
                with adapter_gradient_context:
                    session.scaler.scale(session.loss / session.grad_accum_steps).backward()
            session.batch = next(session.train_loader)
            session.x_0 = session.batch.x_0
            session.loss_mask = session.batch.loss_mask
            session.attention_mask = session.batch.attention_mask
            session.epoch = session.batch.epoch
        _reduce_step_accumulators(session)
        if session.step_processed_tokens != session.total_batch_size:
            raise RuntimeError(
                f"globally reduced processed-token count does not match the configured effective batch: observed={session.step_processed_tokens}, expected={session.total_batch_size}"
            )
        if not _is_causal(session.config):
            if session.step_m2t_lossable_tokens + session.step_t2t_lossable_tokens != session.step_lossable_tokens:
                raise RuntimeError("M2T/T2T lossable-token denominators do not partition the noncausal step")
            if session.step_m2t_active_tokens + session.step_t2t_active_tokens != session.step_active_tokens:
                raise RuntimeError("M2T/T2T active-token counts do not partition the noncausal step")
            if session.step_m2t_active_tokens > session.step_m2t_lossable_tokens:
                raise RuntimeError("M2T active tokens exceed the M2T lossable-token denominator")
            if session.step_t2t_ordinary_eligible_tokens > session.step_t2t_lossable_tokens:
                raise RuntimeError("T2T ordinary-token eligibility exceeds the T2T lossable-token denominator")
            if session.step_t2t_eligible_tokens > session.step_t2t_lossable_tokens:
                raise RuntimeError("T2T eligibility exceeds the T2T lossable-token denominator")
            if session.step_t2t_active_tokens > session.step_t2t_eligible_tokens:
                raise RuntimeError("T2T active edits exceed the source-specific eligibility denominator")
            if (
                _uses_token_edit_corruption(session.config)
                and _uses_matched_token_edit_schedule(session.config)
                and not _uses_self_generated_token_edit(session.config)
                and session.step_t2t_active_tokens > session.step_t2t_ordinary_eligible_tokens
            ):
                raise RuntimeError("random-replacement T2T active edits exceed ordinary-token eligibility")
            if _uses_self_generated_token_edit(session.config):
                self_generated_counts = session.step_self_generated_edit_accumulator
                if int(self_generated_counts["microbatches"]) != session.step_t2t_microbatches:
                    raise RuntimeError("self-generated edit microbatches do not match the T2T branch count")
                draft_tokens = int(self_generated_counts["draft_tokens"])
                if _uses_mixed_token_edit_canvas(session.config):
                    if session.step_t2t_microbatches == 0 and draft_tokens != 0:
                        raise RuntimeError("mixed-canvas draft tokens exist without a T2T microbatch")
                    if session.step_t2t_microbatches > 0 and (
                        draft_tokens <= 0 or draft_tokens > session.step_t2t_active_tokens
                    ):
                        raise RuntimeError("mixed-canvas drafted tokens must be a nonempty subset of hybrid targets")
                elif _uses_trajectory_token_edit_canvas(session.config):
                    if draft_tokens != session.step_self_generated_correction_target_count:
                        raise RuntimeError("trajectory proposal counts do not match remaining correction targets")
                elif draft_tokens != (
                    session.step_token_edit_prefix_correction_loss_targets
                    if _uses_token_edit_prefix(session.config)
                    else session.step_t2t_active_tokens
                ):
                    raise RuntimeError("self-generated draft-token counts do not match T2T active tokens")
        session.progress.lineage.train_m2t_ce_sum += session.step_m2t_ce_sum
        session.progress.lineage.train_m2t_loss_sum += session.step_m2t_loss_sum
        session.progress.lineage.train_m2t_loss_count += session.step_m2t_loss_count
        session.progress.lineage.train_m2t_microbatches += session.step_m2t_microbatches
        session.progress.lineage.train_m2t_active_tokens += session.step_m2t_active_tokens
        session.progress.lineage.train_m2t_lossable_tokens += session.step_m2t_lossable_tokens
        session.progress.interval.m2t_ce_sum += session.step_m2t_ce_sum
        session.progress.interval.m2t_loss_sum += session.step_m2t_loss_sum
        session.progress.interval.m2t_loss_count += session.step_m2t_loss_count
        session.progress.interval.m2t_microbatches += session.step_m2t_microbatches
        session.progress.interval.m2t_active_tokens += session.step_m2t_active_tokens
        session.progress.interval.m2t_lossable_tokens += session.step_m2t_lossable_tokens
        session.progress.lineage.train_t2t_ce_sum += session.step_t2t_ce_sum
        session.progress.lineage.train_t2t_loss_sum += session.step_t2t_loss_sum
        session.progress.lineage.train_t2t_loss_count += session.step_t2t_loss_count
        session.progress.lineage.train_t2t_microbatches += session.step_t2t_microbatches
        session.progress.lineage.train_t2t_active_tokens += session.step_t2t_active_tokens
        session.progress.lineage.train_t2t_lossable_tokens += session.step_t2t_lossable_tokens
        session.progress.lineage.train_t2t_ordinary_eligible_tokens += session.step_t2t_ordinary_eligible_tokens
        session.progress.lineage.train_t2t_eligible_tokens += session.step_t2t_eligible_tokens
        session.progress.interval.t2t_ce_sum += session.step_t2t_ce_sum
        session.progress.interval.t2t_loss_sum += session.step_t2t_loss_sum
        session.progress.interval.t2t_loss_count += session.step_t2t_loss_count
        session.progress.interval.t2t_microbatches += session.step_t2t_microbatches
        session.progress.interval.t2t_active_tokens += session.step_t2t_active_tokens
        session.progress.interval.t2t_lossable_tokens += session.step_t2t_lossable_tokens
        session.progress.interval.t2t_ordinary_eligible_tokens += session.step_t2t_ordinary_eligible_tokens
        session.progress.interval.t2t_eligible_tokens += session.step_t2t_eligible_tokens
        merge_self_generated_edit_accumulator(
            session.progress.interval.self_generated_edit_accumulator,
            session.step_self_generated_edit_accumulator,
        )
        session.progress.interval.lossable_tokens += session.step_lossable_tokens
        _merge_numeric_accumulator(
            session.progress.lineage.train_length_head_accumulator, session.step_length_head_accumulator
        )
        for index, value in enumerate(session.step_per_token_noise_level_hist_counts):
            session.progress.lineage.train_per_token_noise_level_hist_counts[index] += int(value)
        session.progress.lineage.train_per_token_effective_masked_fraction_sum += (
            session.step_per_token_effective_masked_fraction_sum
        )
        session.progress.lineage.train_per_token_effective_masked_fraction_count += (
            session.step_per_token_effective_masked_fraction_count
        )
        session.progress.lineage.train_per_token_effective_masked_fraction_min = min(
            session.progress.lineage.train_per_token_effective_masked_fraction_min,
            session.step_per_token_effective_masked_fraction_min,
        )
        session.progress.lineage.train_per_token_effective_masked_fraction_max = max(
            session.progress.lineage.train_per_token_effective_masked_fraction_max,
            session.step_per_token_effective_masked_fraction_max,
        )
        session.record_total_update_norm = (
            step + 1
        ) % session.config.train.log_interval == 0 or step + 1 == session.total_steps
        optimizer_step_enabled = _unscale_step_gradients(session)
        session.step_gradient_component_norms = _gradient_component_norm_metrics(session.raw_model)
        session.parameterization_gradient_rms_metrics = (
            parameterization_gradient_rms_metrics(session.raw_model, session.parameterization_contract)
            if _parameterization_role_telemetry_enabled(session)
            else {}
        )
        if not session.progress.diagnostics.initial_gradient_metrics:
            session.progress.diagnostics.initial_gradient_metrics = {"initial_gradient_step": step + 1}
            session.progress.diagnostics.initial_gradient_metrics.update(
                {f"initial_{key}": value for key, value in session.step_gradient_component_norms.items()}
            )
        if (
            session.record_early_residual_diagnostics
            and session.rank == 0
            and (session.residual_diagnostic_batch is not None)
        ):
            session.diagnostic_tokens, session.diagnostic_attention_mask, session.diagnostic_position_ids = (
                session.residual_diagnostic_batch
            )
            session.residual_record: dict[str, float | int | str] = {
                "step": step + 1,
                **_residual_projection_init_metrics(session.config),
                **_residual_branch_rms_metrics(
                    session.raw_model,
                    session.diagnostic_tokens,
                    session.diagnostic_attention_mask,
                    session.diagnostic_position_ids,
                ),
                **_layerwise_gradient_norm_metrics(session.raw_model),
            }
            os.makedirs(os.path.dirname(session.residual_diagnostics_path), exist_ok=True)
            append_jsonl(session.residual_diagnostics_path, session.residual_record)
            session.progress.diagnostics.early_residual_diagnostics = {
                "early_residual_diagnostics_step": step + 1,
                "early_residual_diagnostics_path": session.residual_diagnostics_path,
            }
            session.progress.diagnostics.early_residual_diagnostics.update(
                {
                    f"early_{key}": value
                    for key, value in session.residual_record.items()
                    if key != "step" and isinstance(value, int | float)
                }
            )
        session.record_component_update_norm = session.record_total_update_norm or (
            session.record_embedding_head_init_diagnostics and step == session.start_step
        )
        session.pre_step_params = None
        session.pre_step_components = None
        if session.record_total_update_norm:
            session.pre_step_params = [p.detach().clone() for p in session.raw_model.parameters() if p.requires_grad]
        if session.record_component_update_norm:
            session.pre_step_components = {
                "embedding": [p.detach().clone() for p in session.raw_model.wte.parameters()],
                "lm_head": [p.detach().clone() for p in session.raw_model.lm_head.parameters()],
            }
            if len(session.raw_model.token_edit_adapters) > 0:
                session.pre_step_components["token_edit_adapter"] = [
                    p.detach().clone() for p in session.raw_model.token_edit_adapters.parameters()
                ]
            if session.raw_model.token_edit_provenance_embedding is not None:
                session.pre_step_components["token_edit_provenance_embedding"] = [
                    p.detach().clone() for p in session.raw_model.token_edit_provenance_embedding.parameters()
                ]
        if session.grad_clip_enabled:
            session.grad_norm = float(
                torch.nn.utils.clip_grad_norm_(session.raw_model.parameters(), session.grad_clip_threshold)
            )
            session.progress.last.grad_clipped = session.grad_norm > session.grad_clip_threshold
            session.progress.last.grad_clip_ratio = min(
                1.0, session.grad_clip_threshold / max(session.grad_norm, 1e-12)
            )
            session.progress.lineage.grad_clip_event_count += int(session.progress.last.grad_clipped)
            session.progress.lineage.grad_clip_observed_steps += 1
        else:
            session.grad_norm = _grad_norm(session.raw_model.parameters())
            session.progress.last.grad_clipped = False
            session.progress.last.grad_clip_ratio = 1.0
        _advance_step_optimizers(session, enabled=optimizer_step_enabled)
        session.second_moment_norm = 0.0
        session.update_norm = 0.0
        if session.pre_step_params is not None:
            session.second_moment_norm = _second_moment_norm(session.optimizers)
            session.post_step_params = [p for p in session.raw_model.parameters() if p.requires_grad]
            session.update_norm = _update_norm(session.pre_step_params, session.post_step_params)
        session.update_component_norms = _component_update_norm_metrics(session.pre_step_components, session.raw_model)
        session.update_component_norms.update(
            _component_update_grad_ratio_metrics(session.update_component_norms, session.step_gradient_component_norms)
        )
        session.parameterization_parameter_update_rms_metrics = (
            parameterization_parameter_update_rms_metrics(
                session.raw_model,
                session.parameterization_contract,
                session.pre_step_params,
            )
            if session.pre_step_params is not None and _parameterization_role_telemetry_enabled(session)
            else {}
        )
        if session.pre_step_components is not None:
            session.progress.last.update_component_norms = session.update_component_norms
            if not session.progress.diagnostics.initial_update_metrics:
                session.progress.diagnostics.initial_update_metrics = {"initial_update_step": step + 1}
                session.progress.diagnostics.initial_update_metrics.update(
                    {f"initial_{key}": value for key, value in session.update_component_norms.items()}
                )
        session.train_tokens += session.step_processed_tokens
        session.train_valid_tokens += session.step_valid_tokens
        if session.rank == 0 and step + 1 in session.parameter_state_hash_steps:
            append_parameter_state_hash_record_once(
                session.parameter_state_hash_path,
                parameter_state_hash_record(
                    session.raw_model,
                    step=step + 1,
                    train_tokens=session.train_tokens,
                    architecture_fingerprint_sha256=str(session.architecture_fingerprint["fingerprint_sha256"]),
                    parameterization_contract_sha256=session.config.parameterization.contract_sha256,
                ),
                configured_steps=sorted(session.parameter_state_hash_steps),
            )
        session.train_content_tokens += session.step_content_tokens
        session.train_content_bytes += session.step_content_bytes
        if session.monitor is not None:
            session.monitor.progress(step + 1, session.train_tokens, session.train_content_tokens)
        if _uses_block_causal_pilot(session.config):
            session.progress.lineage.block_causal_visible_prefix_frac_sum += session.step_visible_prefix_frac_sum
            session.progress.lineage.block_causal_visible_prefix_count += session.step_visible_prefix_count
        session.progress.last.loss_value = session.step_loss_sum / max(1, session.step_loss_count)
        session.progress.last.main_ce = session.step_main_ce_sum / max(1, session.step_loss_count)
        session.progress.last.main_loss = session.step_main_loss_sum / max(1, session.step_loss_count)
        session.progress.last.confidence_penalty = session.step_confidence_penalty_sum / max(1, session.step_loss_count)
        session.progress.last.confidence_penalty_loss = session.step_confidence_penalty_loss_sum / max(
            1, session.step_loss_count
        )
        session.progress.last.z_loss = session.step_z_loss_sum / max(1, session.step_loss_count)
        session.progress.last.z_loss_loss = session.step_z_loss_loss_sum / max(1, session.step_loss_count)
        session.progress.last.eos_hazard_calibration_loss = session.step_eos_hazard_calibration_loss_sum / max(
            1, session.step_loss_count
        )
        session.progress.last.eos_hazard_loss = session.step_eos_hazard_loss_sum / max(1, session.step_loss_count)
        session.progress.last.eos_hazard_accumulator = session.step_eos_hazard_accumulator
        session.progress.last.length_head_accumulator = session.step_length_head_accumulator
        session.progress.last.position_decile_ce_accumulator = session.step_position_decile_ce_accumulator
        session.progress.last.token_edit_reconstruction_accumulator = session.step_token_edit_reconstruction_accumulator
        _merge_token_edit_reconstruction_accumulator(
            session.progress.lineage.train_token_edit_reconstruction_accumulator,
            session.step_token_edit_reconstruction_accumulator,
        )
        session.progress.last.self_generated_edit_accumulator = session.step_self_generated_edit_accumulator
        merge_self_generated_edit_accumulator(
            session.progress.lineage.train_self_generated_edit_accumulator,
            session.step_self_generated_edit_accumulator,
        )
        session.progress.last.m2t_ce = session.step_m2t_ce_sum / max(1, session.step_m2t_loss_count)
        session.progress.last.m2t_loss = session.step_m2t_loss_sum / max(1, session.step_m2t_loss_count)
        session.progress.last.m2t_microbatches = session.step_m2t_microbatches
        session.progress.last.m2t_active_tokens = session.step_m2t_active_tokens
        session.progress.last.m2t_lossable_tokens = session.step_m2t_lossable_tokens
        session.progress.last.t2t_ce = session.step_t2t_ce_sum / max(1, session.step_t2t_loss_count)
        session.progress.last.t2t_loss = session.step_t2t_loss_sum / max(1, session.step_t2t_loss_count)
        session.progress.last.t2t_microbatches = session.step_t2t_microbatches
        session.progress.last.t2t_active_tokens = session.step_t2t_active_tokens
        session.progress.last.t2t_lossable_tokens = session.step_t2t_lossable_tokens
        session.progress.last.t2t_ordinary_eligible_tokens = session.step_t2t_ordinary_eligible_tokens
        session.progress.last.t2t_eligible_tokens = session.step_t2t_eligible_tokens
        session.progress.last.grad_norm = session.grad_norm
        session.progress.last.gradient_component_norms = session.step_gradient_component_norms
        session.progress.last.second_moment_norm = session.second_moment_norm
        session.progress.last.update_norm = session.update_norm
        session.progress.last.update_grad_ratio = session.update_norm / max(session.grad_norm, 1e-08)
        session.progress.last.active_tokens = session.step_active_tokens
        session.progress.last.lossable_tokens = session.step_lossable_tokens
        session.progress.last.valid_tokens = session.step_valid_tokens
        session.progress.last.content_tokens = session.step_content_tokens
        session.progress.last.content_bytes = session.step_content_bytes
        if session.use_hierarchical_output:
            session.progress.last.class_ce = session.step_class_ce_sum / max(1, session.step_loss_count)
            session.progress.last.within_class_ce = session.step_within_class_ce_sum / max(1, session.step_loss_count)
        if _is_causal(session.config) or _uses_bd3lm_causal_ar_auxiliary(session.config):
            session.progress.last.ar_unsmoothed_ce = session.step_ar_unsmoothed_ce_sum / max(1, session.step_loss_count)
        if session.step_loss_count:
            session.progress.last.entropy_mean = session.step_entropy_sum / session.step_loss_count
            session.progress.last.top1_prob_mean = session.step_top1_prob_sum / session.step_loss_count
            session.progress.last.argmax_mode_frac = session.step_argmax_mode_sum / session.step_loss_count
            session.progress.last.top1_prob_hist_counts = list(session.step_top1_prob_hist_counts)
            session.progress.last.entropy_hist_counts = list(session.step_entropy_hist_counts)
            session.progress.last.logsumexp_hist_counts = list(session.step_logsumexp_hist_counts)
            session.progress.last.logit_norm_hist_counts = list(session.step_logit_norm_hist_counts)
            session.progress.last.logit_norm_mean = session.step_logit_norm_sum / session.step_loss_count
            session.progress.last.logit_norm_min = (
                session.step_logit_norm_min if math.isfinite(session.step_logit_norm_min) else 0.0
            )
            session.progress.last.logit_norm_max = (
                session.step_logit_norm_max if math.isfinite(session.step_logit_norm_max) else 0.0
            )
            session.progress.last.logit_rms_mean = session.step_logit_rms_sum / session.step_loss_count
            session.progress.last.logsumexp_mean = session.step_logsumexp_sum / session.step_loss_count
            session.progress.last.logsumexp_min = (
                session.step_logsumexp_min if math.isfinite(session.step_logsumexp_min) else 0.0
            )
            session.progress.last.logsumexp_max = (
                session.step_logsumexp_max if math.isfinite(session.step_logsumexp_max) else 0.0
            )
        if session.step_soft_input_stat_count:
            session.progress.last.soft_input_stats = {
                key: value / session.step_soft_input_stat_count
                for key, value in session.step_soft_input_stat_sums.items()
            }
            session.progress.last.soft_input_stats["soft_input_curriculum_phase"] = session.step_soft_input_phase
        if session.step_uniform_estimator_stat_count:
            session.progress.last.uniform_estimator_stats = {
                key: value / session.step_uniform_estimator_stat_count
                for key, value in session.step_uniform_estimator_stat_sums.items()
            }
            session.progress.last.uniform_estimator_stats["uniform_loss_estimator"] = (
                session.step_uniform_estimator_label
            )
        session.progress.last.per_token_noise_level_hist_counts = list(session.step_per_token_noise_level_hist_counts)
        session.progress.last.per_token_effective_masked_fraction_sum = (
            session.step_per_token_effective_masked_fraction_sum
        )
        session.progress.last.per_token_effective_masked_fraction_count = (
            session.step_per_token_effective_masked_fraction_count
        )
        session.progress.last.per_token_effective_masked_fraction_min = (
            session.step_per_token_effective_masked_fraction_min
        )
        session.progress.last.per_token_effective_masked_fraction_max = (
            session.step_per_token_effective_masked_fraction_max
        )
        if session.record_embedding_head_init_diagnostics and (
            not session.progress.diagnostics.early_training_diagnostics
        ):
            session.progress.diagnostics.early_training_diagnostics = {
                "early_train_step": step + 1,
                "early_train_entropy_mean": session.progress.last.entropy_mean,
                "early_train_top1_prob_mean": session.progress.last.top1_prob_mean,
                "early_train_argmax_mode_frac": session.progress.last.argmax_mode_frac,
                "early_train_masked_logit_norm_mean": session.progress.last.logit_norm_mean,
                "early_train_masked_logit_norm_min": session.progress.last.logit_norm_min,
                "early_train_masked_logit_norm_max": session.progress.last.logit_norm_max,
                "early_train_masked_logit_rms_mean": session.progress.last.logit_rms_mean,
                "early_train_masked_logsumexp_mean": session.progress.last.logsumexp_mean,
                "early_train_masked_logsumexp_min": session.progress.last.logsumexp_min,
                "early_train_masked_logsumexp_max": session.progress.last.logsumexp_max,
            }
            session.progress.diagnostics.early_training_diagnostics.update(
                _probability_histogram_metrics(
                    "early_train_masked_top1_prob_hist", session.progress.last.top1_prob_hist_counts
                )
            )
            session.progress.diagnostics.early_training_diagnostics.update(
                _value_histogram_metrics(
                    "early_train_masked_entropy_hist", session.progress.last.entropy_hist_counts, 0.0, 16.0
                )
            )
            session.progress.diagnostics.early_training_diagnostics.update(
                _value_histogram_metrics(
                    "early_train_masked_logsumexp_hist", session.progress.last.logsumexp_hist_counts, 0.0, 64.0
                )
            )
            session.progress.diagnostics.early_training_diagnostics.update(
                _value_histogram_metrics(
                    "early_train_masked_logit_norm_hist", session.progress.last.logit_norm_hist_counts, 0.0, 512.0
                )
            )
            if session.rank == 0:
                os.makedirs(os.path.dirname(session.embedding_head_init_diagnostics_path), exist_ok=True)
                append_jsonl(
                    session.embedding_head_init_diagnostics_path,
                    {
                        "event": "early_training_confidence",
                        **session.progress.diagnostics.early_training_diagnostics,
                        **{
                            f"early_{key}": value for key, value in session.progress.last.update_component_norms.items()
                        },
                        **{f"early_{key}": value for key, value in session.step_gradient_component_norms.items()},
                    },
                )
        if (
            session.record_embedding_head_init_diagnostics
            and session.rank == 0
            and (not session.progress.diagnostics.early_lm_head_row_norm_metrics)
        ):
            session.saved_state = _rng_state(session.config)
            try:
                session.progress.diagnostics.early_lm_head_row_norm_metrics = {
                    "early_lm_head_row_norm_step": step + 1,
                    **_prefixed_metric_keys(
                        "early_",
                        _lm_head_row_norm_metrics(
                            session.raw_model,
                            session.tokenizer,
                            session.config.data,
                            session.device,
                            int(session.config.train.device_batch_size),
                            int(session.config.model.max_seq_len),
                            batches=10,
                        ),
                    ),
                }
                os.makedirs(os.path.dirname(session.embedding_head_init_diagnostics_path), exist_ok=True)
                append_jsonl(
                    session.embedding_head_init_diagnostics_path,
                    {"event": "early_lm_head_row_norms", **session.progress.diagnostics.early_lm_head_row_norm_metrics},
                )
            finally:
                _restore_rng_state(session.saved_state, session.config)
        if session.config.train.divergence_check_interval > 0 and (
            (step + 1) % session.config.train.divergence_check_interval == 0 or step + 1 == session.total_steps
        ):
            session.failed = (
                not math.isfinite(session.progress.last.loss_value)
                or session.progress.last.loss_value > session.config.train.divergence_loss_threshold
            )
            if session.is_ddp:
                session.fail_tensor = torch.tensor(int(session.failed), device=session.device)
                dist.all_reduce(session.fail_tensor, op=dist.ReduceOp.MAX)
                session.failed = bool(session.fail_tensor.item())
            if session.failed:
                raise RuntimeError(f"training diverged at step {step + 1}: loss={session.progress.last.loss_value}")
        if (step + 1) % session.config.train.log_interval == 0 or step + 1 == session.total_steps:
            if session.config.runtime.device_type == "cuda":
                torch.cuda.synchronize()
            session.progress.interval.smooth_count += 1
            session.progress.interval.smooth_loss = (
                0.9 * session.progress.interval.smooth_loss + 0.1 * session.progress.last.loss_value
            )
            session.debiased = session.progress.interval.smooth_loss / (1 - 0.9**session.progress.interval.smooth_count)
            session.now = time.time()
            session.interval_steps = step + 1 - session.last_log_step
            session.tok_sec = int(
                session.total_batch_size * session.interval_steps / max(1e-08, session.now - session.last_log_t)
            )
            session.runtime_interval_self_generated_draft_forwards = int(
                session.progress.interval.self_generated_edit_accumulator["draft_forwards"]
            ) - int(session.runtime_interval_start_self_generated_draft_forwards)
            if session.runtime_interval_self_generated_draft_forwards < 0:
                raise RuntimeError("runtime self-generated draft-forward interval counter moved backwards")
            session.interval_compute_estimate = training_compute_estimate(
                training_flops_per_token=int(session.flops_per_token),
                training_tokens=int(session.total_batch_size * session.interval_steps),
                self_generated_draft_forwards=session.runtime_interval_self_generated_draft_forwards,
                device_batch_size=_self_generated_draft_device_batch_size(session.config),
                max_seq_len=int(session.config.model.max_seq_len),
                base_accounting_method=(
                    BD3LM_DENSE_CONCAT_FLOPS_ACCOUNTING_METHOD
                    if _uses_bd3lm(session.config)
                    else "main_plus_mtp_dense_training_estimate"
                    if mtp_training_enabled(session.config)
                    else "base_training_estimate"
                ),
                token_edit_backbone_gradient_mode=session.config.train.token_edit_backbone_gradient_mode,
                differentiable_draft=_uses_differentiable_self_generated_draft(session.config),
            )
            session.mfu = int(session.interval_compute_estimate["total_flops"]) / max(
                1e-08, session.now - session.last_log_t
            )
            session.log_record = {
                "step": step + 1,
                "total_steps": session.total_steps,
                "train_loss": session.debiased,
                "raw_train_loss": session.progress.last.loss_value,
                "raw_main_ce": session.step_main_ce_sum / max(1, session.step_loss_count),
                "raw_total_loss": session.progress.last.loss_value,
                "train_main_ce": session.progress.last.main_ce,
                "train_main_loss": session.progress.last.main_loss,
                "last_step_self_generated_expected_error_mean": session.step_self_generated_expected_error_sum
                / max(1, session.step_self_generated_correction_target_count),
                "last_step_self_generated_draft_preference_mean": session.step_self_generated_draft_preference_sum
                / max(1, session.step_self_generated_correction_target_count),
                "last_step_self_generated_correction_target_count": session.step_self_generated_correction_target_count,
                "token_edit_corruption_enabled": _uses_token_edit_corruption(session.config),
                "token_edit_corruption_batch_fraction": float(
                    session.config.train.token_edit_corruption_batch_fraction
                ),
                "token_edit_batch_layout": session.config.train.token_edit_batch_layout,
                "token_edit_corruption_schedule": session.config.train.token_edit_corruption_schedule,
                "token_edit_corruption_ratio": float(session.config.train.token_edit_corruption_ratio),
                "token_edit_corruption_probability_min": _token_edit_probability_bounds(session.config)[0],
                "token_edit_corruption_probability_max": _token_edit_probability_bounds(session.config)[1],
                "token_edit_corruption_support_policy": _token_edit_support_policy(session.config),
                "train_m2t_ce": session.progress.interval.m2t_ce_sum / max(1, session.progress.interval.m2t_loss_count),
                "train_m2t_loss": session.progress.interval.m2t_loss_sum
                / max(1, session.progress.interval.m2t_loss_count),
                "train_m2t_microbatches": session.progress.interval.m2t_microbatches,
                "train_m2t_active_tokens": session.progress.interval.m2t_active_tokens,
                "train_interval_m2t_lossable_tokens": session.progress.interval.m2t_lossable_tokens,
                "train_m2t_active_token_frac": session.progress.interval.m2t_active_tokens
                / max(1, session.progress.interval.lossable_tokens),
                "train_m2t_active_token_combined_lossable_share": session.progress.interval.m2t_active_tokens
                / max(1, session.progress.interval.lossable_tokens),
                "train_m2t_active_token_frac_within_branch": session.progress.interval.m2t_active_tokens
                / max(1, session.progress.interval.m2t_lossable_tokens),
                "train_t2t_ce": session.progress.interval.t2t_ce_sum / max(1, session.progress.interval.t2t_loss_count),
                "train_t2t_loss": session.progress.interval.t2t_loss_sum
                / max(1, session.progress.interval.t2t_loss_count),
                "train_t2t_microbatches": session.progress.interval.t2t_microbatches,
                "train_t2t_active_tokens": session.progress.interval.t2t_active_tokens,
                "train_interval_t2t_lossable_tokens": session.progress.interval.t2t_lossable_tokens,
                "train_interval_t2t_eligible_tokens": session.progress.interval.t2t_eligible_tokens,
                "train_interval_t2t_ordinary_eligible_tokens": session.progress.interval.t2t_ordinary_eligible_tokens,
                "train_t2t_active_token_frac": session.progress.interval.t2t_active_tokens
                / max(1, session.progress.interval.lossable_tokens),
                "train_t2t_active_token_combined_lossable_share": session.progress.interval.t2t_active_tokens
                / max(1, session.progress.interval.lossable_tokens),
                "train_t2t_active_token_frac_within_branch": session.progress.interval.t2t_active_tokens
                / max(1, session.progress.interval.t2t_lossable_tokens),
                "train_t2t_active_token_frac_within_eligible": session.progress.interval.t2t_active_tokens
                / max(1, session.progress.interval.t2t_eligible_tokens),
                "train_t2t_active_token_frac_within_ordinary_eligible": session.progress.interval.t2t_active_tokens
                / session.progress.interval.t2t_ordinary_eligible_tokens
                if session.progress.interval.t2t_ordinary_eligible_tokens
                else 0.0,
                "train_interval_lossable_tokens": session.progress.interval.lossable_tokens,
                "last_step_m2t_ce": session.progress.last.m2t_ce,
                "last_step_m2t_loss": session.progress.last.m2t_loss,
                "last_step_m2t_microbatches": session.progress.last.m2t_microbatches,
                "last_step_m2t_active_tokens": session.progress.last.m2t_active_tokens,
                "last_step_m2t_lossable_tokens": session.progress.last.m2t_lossable_tokens,
                "last_step_m2t_active_token_frac_within_branch": session.progress.last.m2t_active_tokens
                / max(1, session.progress.last.m2t_lossable_tokens),
                "last_step_t2t_ce": session.progress.last.t2t_ce,
                "last_step_t2t_loss": session.progress.last.t2t_loss,
                "last_step_t2t_microbatches": session.progress.last.t2t_microbatches,
                "last_step_t2t_active_tokens": session.progress.last.t2t_active_tokens,
                "last_step_t2t_lossable_tokens": session.progress.last.t2t_lossable_tokens,
                "last_step_t2t_eligible_tokens": session.progress.last.t2t_eligible_tokens,
                "last_step_t2t_ordinary_eligible_tokens": session.progress.last.t2t_ordinary_eligible_tokens,
                "last_step_t2t_active_token_frac_within_branch": session.progress.last.t2t_active_tokens
                / max(1, session.progress.last.t2t_lossable_tokens),
                "last_step_t2t_active_token_frac_within_eligible": session.progress.last.t2t_active_tokens
                / max(1, session.progress.last.t2t_eligible_tokens),
                "last_step_t2t_active_token_frac_within_ordinary_eligible": session.progress.last.t2t_active_tokens
                / session.progress.last.t2t_ordinary_eligible_tokens
                if session.progress.last.t2t_ordinary_eligible_tokens
                else 0.0,
                "train_confidence_penalty": session.progress.last.confidence_penalty,
                "train_confidence_penalty_loss": session.progress.last.confidence_penalty_loss,
                "confidence_penalty_weight": float(session.config.train.confidence_penalty_weight),
                "train_z_loss": session.progress.last.z_loss,
                "train_z_loss_loss": session.progress.last.z_loss_loss,
                "z_loss_weight": float(session.config.train.z_loss_weight),
                "train_eos_hazard_calibration_loss": session.progress.last.eos_hazard_calibration_loss,
                "train_eos_hazard_loss": session.progress.last.eos_hazard_loss,
                "eos_hazard_loss_weight": float(session.config.train.eos_hazard_loss_weight),
                "lr_multiplier": session.lrm,
                "lr": _primary_optimizer_lr(session.optimizers),
                "lr_schedule_warmup_steps": session.warmup_steps,
                "lr_schedule_warmup_fraction": session.lr_schedule["warmup_fraction"],
                "lr_schedule_warmup_ratio_config": session.lr_schedule["warmup_ratio_config"],
                "tok_sec": session.tok_sec,
                "flops_sec": session.mfu,
                "flops_accounting_method": session.interval_compute_estimate["accounting_method"],
                "base_training_flops_sec": int(session.interval_compute_estimate["base_training_flops"])
                / max(1e-08, session.now - session.last_log_t),
                "self_generated_draft_forward_flops_sec": int(session.interval_compute_estimate["draft_forward_flops"])
                / max(1e-08, session.now - session.last_log_t),
                "self_generated_draft_backward_flops_sec": int(
                    session.interval_compute_estimate["draft_backward_flops"]
                )
                / max(1e-08, session.now - session.last_log_t),
                "runtime_interval_self_generated_draft_forwards": (
                    session.runtime_interval_self_generated_draft_forwards
                ),
                "grad_norm": session.progress.last.grad_norm,
                "grad_clip_enabled": session.grad_clip_enabled,
                "grad_clip_threshold": session.grad_clip_threshold,
                "grad_clipped": session.progress.last.grad_clipped,
                "grad_clip_ratio": session.progress.last.grad_clip_ratio,
                "grad_clip_event_count": session.progress.lineage.grad_clip_event_count,
                "grad_clip_fraction": session.progress.lineage.grad_clip_event_count
                / max(1, session.progress.lineage.grad_clip_observed_steps),
                "second_moment_norm": session.progress.last.second_moment_norm,
                "update_norm": session.progress.last.update_norm,
                "update_grad_ratio": session.progress.last.update_grad_ratio,
                "valid_tokens": session.progress.last.valid_tokens,
                "lossable_tokens": session.progress.last.lossable_tokens,
                "active_tokens": session.progress.last.active_tokens,
                "active_token_frac": session.progress.last.active_tokens
                / max(1, session.progress.last.lossable_tokens),
                "content_tokens": session.progress.last.content_tokens,
                "content_bytes": session.progress.last.content_bytes,
                "content_token_byte_fertility": session.progress.last.content_tokens
                / session.progress.last.content_bytes
                if session.progress.last.content_bytes
                else 0.0,
                "epoch": session.epoch,
                "train_tokens": session.train_tokens,
                "train_valid_tokens": session.train_valid_tokens,
                "train_content_tokens": session.train_content_tokens,
                "train_content_bytes": session.train_content_bytes,
                "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
                if session.config.runtime.device_type == "cuda"
                else 0.0,
                "train_entropy_mean": session.progress.last.entropy_mean,
                "train_top1_prob_mean": session.progress.last.top1_prob_mean,
                "train_argmax_mode_frac": session.progress.last.argmax_mode_frac,
                "train_masked_logit_norm_mean": session.progress.last.logit_norm_mean,
                "train_masked_logit_norm_min": session.progress.last.logit_norm_min,
                "train_masked_logit_norm_max": session.progress.last.logit_norm_max,
                "train_masked_logit_rms_mean": session.progress.last.logit_rms_mean,
                "train_masked_logsumexp_mean": session.progress.last.logsumexp_mean,
                "train_masked_logsumexp_min": session.progress.last.logsumexp_min,
                "train_masked_logsumexp_max": session.progress.last.logsumexp_max,
            }
            session.log_record.update(self_generated_objective_configuration_telemetry(session.config))
            session.log_record.update(
                _probability_histogram_metrics("train_masked_top1_prob_hist", session.step_top1_prob_hist_counts)
            )
            session.log_record.update(
                _value_histogram_metrics("train_masked_entropy_hist", session.step_entropy_hist_counts, 0.0, 16.0)
            )
            session.log_record.update(
                _value_histogram_metrics("train_masked_logsumexp_hist", session.step_logsumexp_hist_counts, 0.0, 64.0)
            )
            session.log_record.update(
                _value_histogram_metrics(
                    "train_masked_logit_norm_hist", session.step_logit_norm_hist_counts, 0.0, 512.0
                )
            )
            session.log_record.update(
                _eos_hazard_accumulator_metrics(
                    "train_",
                    session.step_eos_hazard_accumulator,
                    session.progress.last.eos_hazard_calibration_loss,
                    session.progress.last.eos_hazard_loss,
                    float(session.config.train.eos_hazard_loss_weight),
                )
            )
            if session.config.train.length_head_loss_weight > 0:
                session.log_record.update(
                    _length_head_accumulator_metrics(
                        "train_",
                        session.step_length_head_accumulator,
                        float(session.config.train.length_head_loss_weight),
                    )
                )
            if session.config.train.position_decile_balanced_loss:
                session.log_record.update(
                    _position_decile_ce_accumulator_metrics("train_", session.step_position_decile_ce_accumulator, True)
                )
            if _uses_token_edit_corruption(session.config) and not _uses_self_generated_token_edit(session.config):
                session.log_record.update(
                    _token_edit_reconstruction_metrics(
                        "train_", session.step_token_edit_reconstruction_accumulator, True
                    )
                )
            if _uses_self_generated_token_edit(session.config):
                session.log_record.update(
                    self_generated_edit_metrics(
                        "train_self_generated_edit_",
                        session.progress.interval.self_generated_edit_accumulator,
                    )
                )
            session.log_record.update(joint_two_view.joint_two_view_log_metrics(session))
            session.log_record.update(variable_depth_metrics(session.config, step + 1))
            if _is_causal(session.config):
                session.log_record.update(
                    mtp_metrics(
                        session.config,
                        main_loss=session.step_main_loss_sum / max(1, session.step_loss_count),
                        total_loss=session.step_loss_sum / max(1, session.step_loss_count),
                    )
                )
                session.log_record["ar_train_ce"] = session.debiased
                if session.config.model.mtp_mode != "none":
                    session.log_record["ar_train_ce"] = session.progress.last.main_ce
                session.log_record["ar_train_label_smoothed_ce"] = session.step_main_ce_sum / max(
                    1, session.step_loss_count
                )
                session.log_record["ar_train_unsmoothed_ce"] = session.step_ar_unsmoothed_ce_sum / max(
                    1, session.step_loss_count
                )
                session.log_record["ar_label_smoothing_epsilon"] = _causal_ar_label_smoothing(session.config)
            elif _uses_bd3lm_causal_ar_auxiliary(session.config):
                session.log_record["ar_train_ce"] = session.progress.last.ar_unsmoothed_ce
                session.log_record["ar_train_label_smoothed_ce"] = session.progress.last.ar_unsmoothed_ce
                session.log_record["ar_train_unsmoothed_ce"] = session.progress.last.ar_unsmoothed_ce
                session.log_record["ar_label_smoothing_epsilon"] = 0.0
                session.log_record["block_conversion_ar_loss_weight"] = float(
                    session.config.train.bd3lm_causal_ar_loss_weight
                )
            if session.use_hierarchical_output:
                session.log_record["raw_class_ce"] = session.step_class_ce_sum / max(1, session.step_loss_count)
                session.log_record["raw_within_class_ce"] = session.step_within_class_ce_sum / max(
                    1, session.step_loss_count
                )
            if _uses_bd3lm(session.config):
                training_block_size = int(
                    getattr(
                        session,
                        "block_conversion_training_block_size",
                        session.config.model.bd3lm_block_size,
                    )
                )
                session.log_record.update(
                    {
                        "bd3lm_block_size": int(session.config.model.bd3lm_block_size),
                        "bd3lm_training_block_size": training_block_size,
                        "bd3lm_training_block_size_phase_index": int(
                            getattr(session, "block_conversion_training_block_size_phase_index", 0)
                        ),
                        "bd3lm_concat_seq_len": int(session.config.model.max_seq_len) * 2,
                        "bd3lm_loss_target_tokens": session.progress.last.active_tokens,
                        "bd3lm_lossable_noised_half_tokens": session.progress.last.lossable_tokens,
                    }
                )
            if _uses_block_causal_pilot(session.config):
                session.log_record["block_causal_pilot_size"] = int(session.config.model.block_causal_pilot_size)
                session.log_record["block_causal_pilot_num_blocks"] = (
                    int(session.config.model.max_seq_len) + int(session.config.model.block_causal_pilot_size) - 1
                ) // int(session.config.model.block_causal_pilot_size)
                if session.step_visible_prefix_count:
                    session.log_record["block_causal_pilot_visible_prefix_frac_mean"] = (
                        session.step_visible_prefix_frac_sum / session.step_visible_prefix_count
                    )
            if session.step_soft_input_stat_count:
                session.log_record.update(session.progress.last.soft_input_stats)
            if session.step_uniform_estimator_stat_count:
                session.log_record.update(session.progress.last.uniform_estimator_stats)
            if _uses_per_token_noise_levels(session.config):
                session.log_record.update(
                    _value_histogram_metrics(
                        "per_token_noise_level_hist", session.step_per_token_noise_level_hist_counts, 0.0, 1.0
                    )
                )
                session.log_record.update(
                    _per_token_effective_masked_fraction_metrics(
                        "",
                        session.step_per_token_effective_masked_fraction_sum,
                        session.step_per_token_effective_masked_fraction_count,
                        session.step_per_token_effective_masked_fraction_min,
                        session.step_per_token_effective_masked_fraction_max,
                    )
                )
            for aux_idx, aux_ce_sum in enumerate(session.step_aux_ce_sums):
                session.log_record[f"raw_aux_ce_{aux_idx}"] = aux_ce_sum / max(1, session.step_loss_count)
            if session.step_t_count:
                session.log_record["sampled_t_mean"] = session.step_t_sum / session.step_t_count
                session.log_record["sampled_t_min"] = session.step_t_min
                session.log_record["sampled_t_max"] = session.step_t_max
                if session.config.noise.schedule_weight:
                    session.log_record["schedule_multiplier_mean"] = (
                        session.step_schedule_multiplier_sum / session.step_t_count
                    )
                session.log_record["loss_normalization_scale_mean"] = session.step_loss_normalization_scale_sum / max(
                    1, session.step_loss_count
                )
            if session.monitor is not None:
                session.log_record.update(session.monitor.log_fields())
            session.log_record.update(_optimizer_lr_metrics(session.optimizers))
            session.log_record.update(session.progress.last.gradient_component_norms)
            session.log_record.update(session.progress.last.update_component_norms)
            from .training_block_recipe import recipe_step_metrics

            session.log_record.update(recipe_step_metrics(session))
            session.log_record.update(_parameter_norm_metrics(session.raw_model))
            session.log_record.update(session.parameterization_gradient_rms_metrics)
            session.log_record.update(session.parameterization_parameter_update_rms_metrics)
            print0(
                session.rank,
                f"step {step + 1:05d}/{session.total_steps:05d} loss {session.debiased:.4f} lrm {session.lrm:.3f} grad {session.progress.last.grad_norm:.2f} {session.tok_sec:,} tok/s epoch {session.epoch}",
                flush=True,
            )
            if session.rank == 0:
                append_jsonl(session.train_log_path, session.log_record)
                _write_tensorboard_scalars(session.tensorboard_writer, "train", session.log_record, step + 1)
                flush_tensorboard_writer(session.tensorboard_writer)
            session.progress.interval.m2t_ce_sum = 0.0
            session.progress.interval.m2t_loss_sum = 0.0
            session.progress.interval.m2t_loss_count = 0
            session.progress.interval.m2t_microbatches = 0
            session.progress.interval.m2t_active_tokens = 0
            session.progress.interval.m2t_lossable_tokens = 0
            session.progress.interval.t2t_ce_sum = 0.0
            session.progress.interval.t2t_loss_sum = 0.0
            session.progress.interval.t2t_loss_count = 0
            session.progress.interval.t2t_microbatches = 0
            session.progress.interval.t2t_active_tokens = 0
            session.progress.interval.t2t_lossable_tokens = 0
            session.progress.interval.t2t_ordinary_eligible_tokens = 0
            session.progress.interval.t2t_eligible_tokens = 0
            session.progress.interval.self_generated_edit_accumulator = new_self_generated_edit_accumulator()
            session.runtime_interval_start_self_generated_draft_forwards = 0
            session.progress.interval.lossable_tokens = 0
            session.last_log_t = session.now
            session.last_log_step = step + 1
        if session.rank == 0 and step + 1 in session.fixed_ce_trace_steps:
            session.trace_record = _evaluate_fixed_noise_ce_trace(
                session.raw_model,
                session.config,
                session.tokenizer,
                session.noise,
                session.autocast_ctx,
                session.device,
                step + 1,
                session.total_steps,
                session.train_tokens,
                session.train_valid_tokens,
            )
            session.trace_record["fixed_ce_trace_checkpoint_index"] = (
                session.progress.diagnostics.fixed_ce_trace_records + 1
            )
            os.makedirs(os.path.dirname(session.fixed_ce_trace_path), exist_ok=True)
            append_jsonl(session.fixed_ce_trace_path, session.trace_record)
            _write_tensorboard_scalars(session.tensorboard_writer, "fixed_ce_trace", session.trace_record, step + 1)
            flush_tensorboard_writer(session.tensorboard_writer)
            session.progress.diagnostics.fixed_ce_trace_records += 1
            session.progress.diagnostics.last_fixed_ce_trace_record = session.trace_record
        session.periodic_resume_contract = None
        if session.config.train.save_every > 0 and (step + 1) % session.config.train.save_every == 0:
            if session.monitor is not None:
                session.monitor.switch("checkpoint")
            session.periodic_resume_contract = _collect_resume_contract(
                session.config,
                session.batch,
                device=session.device,
                rank=session.rank,
                world_size=session.world_size,
                is_ddp=session.is_ddp,
                grad_accum_steps=session.grad_accum_steps,
                consumed_microbatches=(step + 1) * session.grad_accum_steps,
                trajectory_contract=session.trajectory_contract,
                trainer_state=_current_trainer_state(session),
            )
        if session.periodic_resume_contract is not None and session.rank == 0:
            session.checkpoint_metrics = {
                "train_loss": session.progress.last.loss_value,
                "raw_main_ce": session.progress.last.main_ce,
                "train_main_loss": session.progress.last.main_loss,
                "last_step_self_generated_expected_error_mean": session.step_self_generated_expected_error_sum
                / max(1, session.step_self_generated_correction_target_count),
                "last_step_self_generated_draft_preference_mean": session.step_self_generated_draft_preference_sum
                / max(1, session.step_self_generated_correction_target_count),
                "last_step_self_generated_correction_target_count": session.step_self_generated_correction_target_count,
                "raw_total_loss": session.progress.last.loss_value,
                "token_edit_corruption_enabled": _uses_token_edit_corruption(session.config),
                "last_train_m2t_ce": session.progress.last.m2t_ce,
                "last_train_m2t_loss": session.progress.last.m2t_loss,
                "last_train_m2t_microbatches": session.progress.last.m2t_microbatches,
                "last_train_m2t_active_tokens": session.progress.last.m2t_active_tokens,
                "last_train_m2t_lossable_tokens": session.progress.last.m2t_lossable_tokens,
                "last_train_m2t_active_token_frac_within_branch": session.progress.last.m2t_active_tokens
                / max(1, session.progress.last.m2t_lossable_tokens),
                "last_train_t2t_ce": session.progress.last.t2t_ce,
                "last_train_t2t_loss": session.progress.last.t2t_loss,
                "last_train_t2t_microbatches": session.progress.last.t2t_microbatches,
                "last_train_t2t_active_tokens": session.progress.last.t2t_active_tokens,
                "last_train_t2t_lossable_tokens": session.progress.last.t2t_lossable_tokens,
                "last_train_t2t_eligible_tokens": session.progress.last.t2t_eligible_tokens,
                "last_train_t2t_ordinary_eligible_tokens": session.progress.last.t2t_ordinary_eligible_tokens,
                "last_train_t2t_active_token_frac_within_branch": session.progress.last.t2t_active_tokens
                / max(1, session.progress.last.t2t_lossable_tokens),
                "last_train_t2t_active_token_frac_within_eligible": session.progress.last.t2t_active_tokens
                / max(1, session.progress.last.t2t_eligible_tokens),
                "last_train_t2t_active_token_frac_within_ordinary_eligible": session.progress.last.t2t_active_tokens
                / session.progress.last.t2t_ordinary_eligible_tokens
                if session.progress.last.t2t_ordinary_eligible_tokens
                else 0.0,
                "train_confidence_penalty": session.progress.last.confidence_penalty,
                "train_confidence_penalty_loss": session.progress.last.confidence_penalty_loss,
                "confidence_penalty_weight": float(session.config.train.confidence_penalty_weight),
                "last_train_z_loss": session.progress.last.z_loss,
                "last_train_z_loss_loss": session.progress.last.z_loss_loss,
                "z_loss_weight": float(session.config.train.z_loss_weight),
                "last_train_eos_hazard_calibration_loss": session.progress.last.eos_hazard_calibration_loss,
                "last_train_eos_hazard_loss": session.progress.last.eos_hazard_loss,
                "eos_hazard_loss_weight": float(session.config.train.eos_hazard_loss_weight),
                "ar_unsmoothed_ce": session.progress.last.ar_unsmoothed_ce,
                "grad_norm": session.progress.last.grad_norm,
                "last_grad_clipped": session.progress.last.grad_clipped,
                "last_grad_clip_ratio": session.progress.last.grad_clip_ratio,
                "grad_clip_event_count": session.progress.lineage.grad_clip_event_count,
                "grad_clip_observed_steps": session.progress.lineage.grad_clip_observed_steps,
                **session.progress.last.gradient_component_norms,
                **session.progress.last.update_component_norms,
                "valid_tokens": session.progress.last.valid_tokens,
                "lossable_tokens": session.progress.last.lossable_tokens,
                "active_tokens": session.progress.last.active_tokens,
                "train_valid_tokens": session.train_valid_tokens,
                "train_content_tokens": session.train_content_tokens,
                "train_content_bytes": session.train_content_bytes,
                "pre_update_parameter_state_sha256": session.initial_state_artifact[
                    "pre_update_parameter_state_sha256"
                ],
                "initial_state_sha256": session.initial_state_artifact["initial_state_sha256"],
                "initialization_contract_sha256": session.initial_state_artifact["initialization_contract_sha256"],
                **session.source_checkpoint_load_receipt,
                "train_entropy_mean": session.progress.last.entropy_mean,
                "train_top1_prob_mean": session.progress.last.top1_prob_mean,
                "train_argmax_mode_frac": session.progress.last.argmax_mode_frac,
                "last_train_masked_logit_norm_mean": session.progress.last.logit_norm_mean,
                "last_train_masked_logit_rms_mean": session.progress.last.logit_rms_mean,
                "last_train_masked_logsumexp_mean": session.progress.last.logsumexp_mean,
                **{f"last_{key}": value for key, value in session.progress.last.soft_input_stats.items()},
                **{f"last_{key}": value for key, value in session.progress.last.uniform_estimator_stats.items()},
                **_probability_histogram_metrics(
                    "train_masked_top1_prob_hist", session.progress.last.top1_prob_hist_counts
                ),
                **_value_histogram_metrics(
                    "train_masked_entropy_hist", session.progress.last.entropy_hist_counts, 0.0, 16.0
                ),
                **_value_histogram_metrics(
                    "train_masked_logsumexp_hist", session.progress.last.logsumexp_hist_counts, 0.0, 64.0
                ),
                **_value_histogram_metrics(
                    "train_masked_logit_norm_hist", session.progress.last.logit_norm_hist_counts, 0.0, 512.0
                ),
            }
            if _uses_per_token_noise_levels(session.config):
                session.checkpoint_metrics.update(
                    _value_histogram_metrics(
                        "last_train_per_token_noise_level_hist",
                        session.progress.last.per_token_noise_level_hist_counts,
                        0.0,
                        1.0,
                    )
                )
                session.checkpoint_metrics.update(
                    _per_token_effective_masked_fraction_metrics(
                        "last_train_",
                        session.progress.last.per_token_effective_masked_fraction_sum,
                        session.progress.last.per_token_effective_masked_fraction_count,
                        session.progress.last.per_token_effective_masked_fraction_min,
                        session.progress.last.per_token_effective_masked_fraction_max,
                    )
                )
            if _uses_token_edit_corruption(session.config) and not _uses_self_generated_token_edit(session.config):
                session.checkpoint_metrics.update(
                    _token_edit_reconstruction_metrics(
                        "last_train_", session.progress.last.token_edit_reconstruction_accumulator, True
                    )
                )
            if _uses_self_generated_token_edit(session.config):
                session.checkpoint_metrics.update(
                    self_generated_edit_metrics(
                        "last_train_self_generated_edit_",
                        session.progress.last.self_generated_edit_accumulator,
                    )
                )
            session.checkpoint_metrics.update(joint_two_view.joint_two_view_checkpoint_metrics(session))
            if session.config.train.length_head_loss_weight > 0:
                session.checkpoint_metrics.update(
                    _length_head_accumulator_metrics(
                        "last_train_",
                        session.progress.last.length_head_accumulator,
                        float(session.config.train.length_head_loss_weight),
                    )
                )
            if session.config.train.position_decile_balanced_loss:
                session.checkpoint_metrics.update(
                    _position_decile_ce_accumulator_metrics(
                        "last_train_", session.progress.last.position_decile_ce_accumulator, True
                    )
                )
            if session.use_hierarchical_output:
                session.checkpoint_metrics["raw_class_ce"] = session.progress.last.class_ce
                session.checkpoint_metrics["raw_within_class_ce"] = session.progress.last.within_class_ce
            if _uses_bd3lm(session.config):
                session.checkpoint_metrics["bd3lm_training_block_size"] = int(
                    getattr(
                        session,
                        "block_conversion_training_block_size",
                        session.config.model.bd3lm_block_size,
                    )
                )
                session.checkpoint_metrics["bd3lm_training_block_size_phase_index"] = int(
                    getattr(session, "block_conversion_training_block_size_phase_index", 0)
                )
            session.periodic_checkpoint_path = periodic_path(session.config, step + 1)
            save_checkpoint(
                session.periodic_checkpoint_path,
                session.raw_model,
                session.optimizers,
                session.scaler,
                session.config,
                session.model_config,
                step + 1,
                session.train_tokens,
                session.train_valid_tokens,
                session.checkpoint_metrics,
                resume_contract=session.periodic_resume_contract,
            )
            alias_checkpoint(
                session.periodic_checkpoint_path, os.path.join(session.config.train.run_dir, "model_latest.pt")
            )
        if session.periodic_resume_contract is not None and session.monitor is not None:
            session.monitor.switch("train")
        if (
            session.config.eval.enabled
            and session.config.train.eval_every > 0
            and (session.rank == 0)
            and ((step + 1) % session.config.train.eval_every == 0)
            and (step + 1 < session.total_steps)
        ):
            run_periodic_evaluation(session, step, _run_eval_with_runtime)
