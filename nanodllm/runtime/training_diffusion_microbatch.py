"""Accumulate one diffusion microbatch under its configured corruption and loss."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from .training_adapter_gradients import _isolate_token_edit_adapter_gradients
from .training_diffusion_batch import prepare_diffusion_batch
from .training_diffusion_loss import compute_diffusion_loss
from .training_empty_target import backward_empty_ddp_microbatch
from .training_features import _token_edit_uses_residual_adapter, _uses_uniform_low_variance_estimator
from .training_metrics import (
    _update_eos_hazard_accumulator,
    _update_length_head_accumulator,
    _update_position_decile_ce_accumulator,
    _update_token_edit_reconstruction_accumulator,
)
from .training_objective import _training_targets_from_indices, _uniform_two_sample_mean_loss
from .training_row_mixed_edit import (
    accumulate_fixed_row_step_branches,
    fixed_row_t2t_observations,
    prepare_fixed_row_draft,
)
from .training_self_generated_edit import accumulate_self_generated_correction_step
from .training_state import TrainingSession
from .training_token_edit_prefix import prepare_self_generated_edit_draft_with_prefix


def _backward_uniform_estimate(session: TrainingSession, step: int, micro: int) -> None:
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
                session.step_uniform_estimator_stat_sums[key] = session.step_uniform_estimator_stat_sums.get(
                    key, 0.0
                ) + float(value)
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


def _record_objective_statistics(session: TrainingSession, step: int, micro: int) -> None:
    session.loss_value = float(session.loss.detach().item())
    if not math.isfinite(session.loss_value):
        raise RuntimeError(f"non-finite loss at step {step}, micro {micro}: {session.loss_value}")
    session.step_loss_sum += session.loss_value
    session.step_main_ce_sum += float(session.raw_main_ce.detach().item())
    session.step_main_loss_sum += float(session.main_loss_for_log.detach().item())
    session.step_confidence_penalty_sum += float(session.raw_confidence_penalty.detach().item())
    session.step_confidence_penalty_loss_sum += float(session.confidence_penalty_loss_for_log.detach().item())
    session.step_z_loss_sum += float(session.raw_z_loss.detach().item())
    session.step_z_loss_loss_sum += float(session.z_loss_for_log.detach().item())
    session.step_eos_hazard_calibration_loss_sum += float(session.raw_eos_hazard_calibration_loss.detach().item())
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
    if session.logsumexp_values is not None and session.logit_norms is not None and (session.logit_rms is not None):
        session.logit_norms_f = session.logit_norms.detach().float()
        session.logsumexp_values_f = session.logsumexp_values.detach().float()
        session.logit_rms_f = session.logit_rms.detach().float()
        session.step_logit_norm_sum += float(session.logit_norms_f.mean().item())
        session.step_logit_norm_min = min(session.step_logit_norm_min, float(session.logit_norms_f.min().item()))
        session.step_logit_norm_max = max(session.step_logit_norm_max, float(session.logit_norms_f.max().item()))
        session.step_logit_rms_sum += float(session.logit_rms_f.mean().item())
        session.step_logsumexp_sum += float(session.logsumexp_values_f.mean().item())
        session.step_logsumexp_min = min(session.step_logsumexp_min, float(session.logsumexp_values_f.min().item()))
        session.step_logsumexp_max = max(session.step_logsumexp_max, float(session.logsumexp_values_f.max().item()))
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


def run_diffusion_microbatch(session: TrainingSession, step: int, micro: int, depth_kwargs: dict[str, int]) -> None:
    prepare_diffusion_batch(session, step, micro)
    if session.target_indices.numel() == 0:
        if session.is_ddp:
            backward_empty_ddp_microbatch(session)
        return
    session.targets = _training_targets_from_indices(session.x_0, session.target_indices)
    session.step_active_tokens += int(session.target_indices.numel())
    if session.row_mixed_token_edit:
        with session.autocast_ctx:
            prepare_fixed_row_draft(session)
    elif session.self_generated_edit_branch:
        prepare_self_generated_edit_draft_with_prefix(session, step=step, microbatch=micro)
    if _uses_uniform_low_variance_estimator(session.config):
        _backward_uniform_estimate(session, step, micro)
        return
    with session.sync_context():
        with session.autocast_ctx:
            compute_diffusion_loss(session, depth_kwargs)
        _record_objective_statistics(session, step, micro)
        adapter_gradient_context = (
            _isolate_token_edit_adapter_gradients(session.raw_model)
            if session.self_generated_edit_branch and _token_edit_uses_residual_adapter(session.config)
            else nullcontext()
        )
        with adapter_gradient_context:
            session.scaler.scale(session.loss / session.grad_accum_steps).backward()
