"""Initialize and reduce per-update accumulators across microbatches and ranks."""

from __future__ import annotations

import torch
import torch.distributed as dist

from . import training_joint_two_view as joint_two_view
from .nanochat_muon_schedule import apply as _set_optimizer_schedules
from .training_block_conversion import (
    block_conversion_training_block_size,
    block_conversion_training_block_size_phase_index,
)
from .training_checkpoint import (
    _all_reduce_numeric_accumulator,
    _packed_all_reduce_float_counters,
    _packed_all_reduce_int_counters,
)
from .training_features import _uses_bd3lm_causal_clean_stream
from .training_metrics import (
    LOGIT_SCALE_HIST_BINS,
    PER_TOKEN_NOISE_LEVEL_HIST_BINS,
    TOP1_PROB_HIST_BINS,
    _new_eos_hazard_accumulator,
    _new_length_head_accumulator,
    _new_position_decile_ce_accumulator,
    _new_token_edit_reconstruction_accumulator,
)
from .training_objective import _soft_input_curriculum_mix, _soft_input_curriculum_phase
from .training_optim import _set_optimizer_lrs, _step_lr_multiplier, _zero_grad
from .training_self_generated_edit import coerce_self_generated_edit_accumulator, new_self_generated_edit_accumulator
from .training_state import TrainingSession


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
