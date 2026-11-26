"""Track token counts, losses, and model diagnostics over completed updates."""

from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist

from .parameter_state_hash import append_parameter_state_hash_record_once, parameter_state_hash_record
from .telemetry import append_jsonl
from .training_checkpoint import _merge_numeric_accumulator
from .training_features import (
    _is_causal,
    _uses_bd3lm_causal_ar_auxiliary,
    _uses_block_causal_pilot,
    _uses_matched_token_edit_schedule,
    _uses_mixed_token_edit_canvas,
    _uses_self_generated_token_edit,
    _uses_token_edit_corruption,
    _uses_token_edit_prefix,
    _uses_trajectory_token_edit_canvas,
)
from .training_metrics import (
    _lm_head_row_norm_metrics,
    _merge_token_edit_reconstruction_accumulator,
    _prefixed_metric_keys,
    _probability_histogram_metrics,
    _value_histogram_metrics,
)
from .training_rng import _restore_rng_state, _rng_state
from .training_self_generated_edit import merge_self_generated_edit_accumulator
from .training_state import TrainingSession


def accumulate_objective_totals(session: TrainingSession) -> None:
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


def record_step_outcome(session: TrainingSession, step: int) -> None:
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
            key: value / session.step_soft_input_stat_count for key, value in session.step_soft_input_stat_sums.items()
        }
        session.progress.last.soft_input_stats["soft_input_curriculum_phase"] = session.step_soft_input_phase
    if session.step_uniform_estimator_stat_count:
        session.progress.last.uniform_estimator_stats = {
            key: value / session.step_uniform_estimator_stat_count
            for key, value in session.step_uniform_estimator_stat_sums.items()
        }
        session.progress.last.uniform_estimator_stats["uniform_loss_estimator"] = session.step_uniform_estimator_label
    session.progress.last.per_token_noise_level_hist_counts = list(session.step_per_token_noise_level_hist_counts)
    session.progress.last.per_token_effective_masked_fraction_sum = session.step_per_token_effective_masked_fraction_sum
    session.progress.last.per_token_effective_masked_fraction_count = (
        session.step_per_token_effective_masked_fraction_count
    )
    session.progress.last.per_token_effective_masked_fraction_min = session.step_per_token_effective_masked_fraction_min
    session.progress.last.per_token_effective_masked_fraction_max = session.step_per_token_effective_masked_fraction_max


def check_training_diagnostics(session: TrainingSession, step: int) -> None:
    if session.record_embedding_head_init_diagnostics and (not session.progress.diagnostics.early_training_diagnostics):
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
                    **{f"early_{key}": value for key, value in session.progress.last.update_component_norms.items()},
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
