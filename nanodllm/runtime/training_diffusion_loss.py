"""Compute token losses and apply diffusion, editing, and auxiliary objective weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import training_joint_two_view as joint_two_view
from .mdlm_objective_controls import target_weights as mdlm_target_weights
from .training_features import _token_edit_detaches_backbone, _token_edit_uses_residual_adapter, _uses_length_head
from .training_metrics import (
    _accumulate_probability_histogram,
    _accumulate_value_histogram,
    _eos_hazard_calibration_loss,
    _length_head_loss_and_stats,
    _logit_confidence_statistics,
    _logit_scale_statistics,
    _per_target_t_values,
    _position_decile_balanced_ce_loss,
    _weighted_token_mean,
)
from .training_objective import (
    _fast_dllm_loss_normalization_scale,
    _forward_training_model,
    _target_lengths_for_conditioning,
    _token_edit_loss_normalization_scale,
)
from .training_row_mixed_edit import assemble_fixed_row_correction_loss
from .training_self_generated_edit import (
    build_self_generated_correction_objective_for_session,
    self_generated_t2t_target_weights,
)
from .training_state import TrainingSession
from .training_token_edit_prefix import (
    apply_token_edit_prefix_loss_weights,
    token_edit_prefix_target_mean,
    token_edit_prefix_target_means,
)
from .training_token_edit_trajectory import apply_trajectory_loss_scope


def _forward_and_score_tokens(session: TrainingSession, depth_kwargs: dict[str, int]) -> None:
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
        detach_final_hidden=(session.self_generated_edit_branch and _token_edit_detaches_backbone(session.config)),
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
            session.class_logits, session.token_logits, session.aux_logits, session.length_logits = session.model_out
        elif session.return_aux_logits:
            session.class_logits, session.token_logits, session.aux_logits = session.model_out
        elif _uses_length_head(session.config):
            session.class_logits, session.token_logits, session.length_logits = session.model_out
            session.aux_logits = []
        else:
            session.class_logits, session.token_logits = session.model_out
            session.aux_logits = []
        session.main_ce_values, session.class_ce_values, session.within_class_ce_values = (
            session.raw_model.hierarchical_ce(session.class_logits, session.token_logits, session.targets)
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
            _accumulate_value_histogram(session.step_logsumexp_hist_counts, session.logsumexp_values, 0.0, 64.0)
            _accumulate_value_histogram(session.step_logit_norm_hist_counts, session.logit_norms, 0.0, 512.0)
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
        session.raw_eos_hazard_calibration_loss, session.eos_hazard_stats = _eos_hazard_calibration_loss(
            session.logits,
            session.targets,
            session.target_indices,
            session.attention_mask,
            session.eos_token_id,
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
    session.aux_ce_values = [F.cross_entropy(aux, session.targets, reduction="none") for aux in session.aux_logits]
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
        session.self_generated_correction_objective = build_self_generated_correction_objective_for_session(session)
        session.main_objective_values = session.self_generated_correction_objective.loss_values
    session.raw_class_ce = session.class_ce_values.mean() if session.class_ce_values is not None else None
    session.raw_within_class_ce = (
        session.within_class_ce_values.mean() if session.within_class_ce_values is not None else None
    )
    session.raw_aux_ces = token_edit_prefix_target_means(
        session.aux_ce_values,
        session.token_edit_correction_loss_target_mask,
    )
    session.position_decile_stats = None


def _apply_tokenwise_loss(session: TrainingSession) -> None:
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
        session.loss_normalization_scale = float(session.token_weights.detach().sum().item()) / session.loss_denominator
    session.loss = session.main_loss
    session.main_loss_for_log = session.main_loss
    for aux_loss in session.aux_losses:
        session.loss = session.loss + session.config.train.auxiliary_loss_weight * aux_loss
    if session.config.train.entropy_bonus_beta > 0 and session.entropy_values is not None:
        session.entropy_loss = (
            -session.config.train.entropy_bonus_beta * session.entropy_values.float() * session.token_weights
        ).sum() / session.loss_denominator
        session.loss = session.loss + session.entropy_loss
    if session.config.train.confidence_penalty_weight > 0:
        if session.entropy_values is None:
            raise RuntimeError("confidence penalty requires flat logits entropy values")
        session.confidence_penalty_loss = (
            session.config.train.confidence_penalty_weight * -session.entropy_values.float() * session.token_weights
        ).sum() / session.loss_denominator
        session.confidence_penalty_loss_for_log = session.confidence_penalty_loss
        session.loss = session.loss + session.confidence_penalty_loss
    if session.config.train.z_loss_weight > 0:
        if session.z_loss_values is None:
            raise RuntimeError("z-loss requires flat logits")
        session.z_loss = (
            session.config.train.z_loss_weight * session.z_loss_values.float() * session.token_weights
        ).sum() / session.loss_denominator
        session.z_loss_for_log = session.z_loss
        session.loss = session.loss + session.z_loss
    if session.config.train.eos_hazard_loss_weight > 0:
        session.eos_hazard_loss = session.config.train.eos_hazard_loss_weight * session.raw_eos_hazard_calibration_loss
        session.eos_hazard_loss_for_log = session.eos_hazard_loss
        session.loss = session.loss + session.eos_hazard_loss


def _apply_position_balanced_loss(session: TrainingSession) -> None:
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


def _apply_scalar_loss(session: TrainingSession) -> None:
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
        session.eos_hazard_loss = session.config.train.eos_hazard_loss_weight * session.raw_eos_hazard_calibration_loss
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
        session.confidence_penalty_loss_for_log = session.confidence_penalty_loss * session.loss_normalization_scale
    if session.config.train.z_loss_weight > 0:
        session.z_loss_for_log = session.z_loss * session.loss_normalization_scale
    if session.config.train.eos_hazard_loss_weight > 0:
        session.eos_hazard_loss_for_log = session.eos_hazard_loss * session.loss_normalization_scale


def compute_diffusion_loss(session: TrainingSession, depth_kwargs: dict[str, int]) -> None:
    _forward_and_score_tokens(session, depth_kwargs)
    if isinstance(session.t_val, torch.Tensor):
        _apply_tokenwise_loss(session)
    elif session.config.train.position_decile_balanced_loss:
        _apply_position_balanced_loss(session)
    else:
        _apply_scalar_loss(session)
    if session.config.train.length_head_loss_weight > 0:
        session.loss = session.loss + session.length_head_loss_for_log
    joint_two_view.apply_joint_two_view_loss(session)
