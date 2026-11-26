"""Write interval measurements and reset the interval after publication."""

from __future__ import annotations

import time

import torch

from . import training_joint_two_view as joint_two_view
from .telemetry import append_jsonl, flush_tensorboard_writer
from .telemetry import write_tensorboard_scalars as _write_tensorboard_scalars
from .training_compute import BD3LM_DENSE_CONCAT_FLOPS_ACCOUNTING_METHOD, training_compute_estimate
from .training_features import (
    _is_causal,
    _self_generated_draft_device_batch_size,
    _token_edit_probability_bounds,
    _token_edit_support_policy,
    _uses_bd3lm,
    _uses_bd3lm_causal_ar_auxiliary,
    _uses_block_causal_pilot,
    _uses_differentiable_self_generated_draft,
    _uses_per_token_noise_levels,
    _uses_self_generated_token_edit,
    _uses_token_edit_corruption,
)
from .training_metrics import (
    _eos_hazard_accumulator_metrics,
    _length_head_accumulator_metrics,
    _parameter_norm_metrics,
    _per_token_effective_masked_fraction_metrics,
    _position_decile_ce_accumulator_metrics,
    _probability_histogram_metrics,
    _token_edit_reconstruction_metrics,
    _value_histogram_metrics,
)
from .training_mtp import mtp_metrics, mtp_training_enabled
from .training_objective import _causal_ar_label_smoothing
from .training_optim import _optimizer_lr_metrics, _primary_optimizer_lr
from .training_runtime import print0
from .training_self_generated_edit import (
    new_self_generated_edit_accumulator,
    self_generated_edit_metrics,
    self_generated_objective_configuration_telemetry,
)
from .training_state import TrainingSession
from .training_variable_depth import metrics as variable_depth_metrics


def _build_interval_record(session: TrainingSession, step: int) -> None:
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
    session.mfu = int(session.interval_compute_estimate["total_flops"]) / max(1e-08, session.now - session.last_log_t)
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
        "token_edit_corruption_batch_fraction": float(session.config.train.token_edit_corruption_batch_fraction),
        "token_edit_batch_layout": session.config.train.token_edit_batch_layout,
        "token_edit_corruption_schedule": session.config.train.token_edit_corruption_schedule,
        "token_edit_corruption_ratio": float(session.config.train.token_edit_corruption_ratio),
        "token_edit_corruption_probability_min": _token_edit_probability_bounds(session.config)[0],
        "token_edit_corruption_probability_max": _token_edit_probability_bounds(session.config)[1],
        "token_edit_corruption_support_policy": _token_edit_support_policy(session.config),
        "train_m2t_ce": session.progress.interval.m2t_ce_sum / max(1, session.progress.interval.m2t_loss_count),
        "train_m2t_loss": session.progress.interval.m2t_loss_sum / max(1, session.progress.interval.m2t_loss_count),
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
        "train_t2t_loss": session.progress.interval.t2t_loss_sum / max(1, session.progress.interval.t2t_loss_count),
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
        "self_generated_draft_backward_flops_sec": int(session.interval_compute_estimate["draft_backward_flops"])
        / max(1e-08, session.now - session.last_log_t),
        "runtime_interval_self_generated_draft_forwards": (session.runtime_interval_self_generated_draft_forwards),
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
        "active_token_frac": session.progress.last.active_tokens / max(1, session.progress.last.lossable_tokens),
        "content_tokens": session.progress.last.content_tokens,
        "content_bytes": session.progress.last.content_bytes,
        "content_token_byte_fertility": session.progress.last.content_tokens / session.progress.last.content_bytes
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


def _append_objective_metrics(session: TrainingSession, step: int) -> None:
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
        _value_histogram_metrics("train_masked_logit_norm_hist", session.step_logit_norm_hist_counts, 0.0, 512.0)
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
            _token_edit_reconstruction_metrics("train_", session.step_token_edit_reconstruction_accumulator, True)
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
        session.log_record["ar_train_label_smoothed_ce"] = session.step_main_ce_sum / max(1, session.step_loss_count)
        session.log_record["ar_train_unsmoothed_ce"] = session.step_ar_unsmoothed_ce_sum / max(
            1, session.step_loss_count
        )
        session.log_record["ar_label_smoothing_epsilon"] = _causal_ar_label_smoothing(session.config)
    elif _uses_bd3lm_causal_ar_auxiliary(session.config):
        session.log_record["ar_train_ce"] = session.progress.last.ar_unsmoothed_ce
        session.log_record["ar_train_label_smoothed_ce"] = session.progress.last.ar_unsmoothed_ce
        session.log_record["ar_train_unsmoothed_ce"] = session.progress.last.ar_unsmoothed_ce
        session.log_record["ar_label_smoothing_epsilon"] = 0.0
        session.log_record["block_conversion_ar_loss_weight"] = float(session.config.train.bd3lm_causal_ar_loss_weight)
    if session.use_hierarchical_output:
        session.log_record["raw_class_ce"] = session.step_class_ce_sum / max(1, session.step_loss_count)
        session.log_record["raw_within_class_ce"] = session.step_within_class_ce_sum / max(1, session.step_loss_count)
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
            session.log_record["schedule_multiplier_mean"] = session.step_schedule_multiplier_sum / session.step_t_count
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


def log_training_interval(session: TrainingSession, step: int) -> None:
    _build_interval_record(session, step)
    _append_objective_metrics(session, step)
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
