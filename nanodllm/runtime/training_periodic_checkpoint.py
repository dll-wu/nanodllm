"""Checkpoint the complete trainer state at the configured step boundary."""

from __future__ import annotations

import os

from . import training_joint_two_view as joint_two_view
from .training_checkpoint import _collect_resume_contract, alias_checkpoint, periodic_path, save_checkpoint
from .training_features import (
    _uses_bd3lm,
    _uses_per_token_noise_levels,
    _uses_self_generated_token_edit,
    _uses_token_edit_corruption,
)
from .training_metrics import (
    _length_head_accumulator_metrics,
    _per_token_effective_masked_fraction_metrics,
    _position_decile_ce_accumulator_metrics,
    _probability_histogram_metrics,
    _token_edit_reconstruction_metrics,
    _value_histogram_metrics,
)
from .training_self_generated_edit import self_generated_edit_metrics
from .training_state import TrainingSession, _current_trainer_state


def _checkpoint_measurements(session: TrainingSession) -> None:
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
        "pre_update_parameter_state_sha256": session.initial_state_artifact["pre_update_parameter_state_sha256"],
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
        **_probability_histogram_metrics("train_masked_top1_prob_hist", session.progress.last.top1_prob_hist_counts),
        **_value_histogram_metrics("train_masked_entropy_hist", session.progress.last.entropy_hist_counts, 0.0, 16.0),
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


def checkpoint_if_due(session: TrainingSession, step: int) -> None:
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
        _checkpoint_measurements(session)
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
