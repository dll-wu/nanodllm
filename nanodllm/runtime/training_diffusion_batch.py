"""Corrupt a diffusion microbatch and select loss-bearing positions."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from .training_features import (
    _uses_bd3lm,
    _uses_block_causal_pilot,
    _uses_fixed_row_token_edit_layout,
    _uses_matched_token_edit_schedule,
    _uses_per_token_noise_levels,
    _uses_self_generated_token_edit,
    _uses_token_edit_prefix,
)
from .training_metrics import _accumulate_value_histogram
from .training_objective import (
    _bd3lm_noised_half_indices,
    _build_uniform_soft_input_embeddings,
    _corrupt_token_edit_training_batch,
    _corrupt_training_batch,
    _token_edit_branch_enabled_for_microbatch,
    _token_edit_ordinary_eligible_mask,
    _update_eos_supervision_batch_counts,
)
from .training_row_mixed_edit import fixed_row_lossable_partition
from .training_runtime import _apply_block_causal_pilot_training, _batch_model_attention_mask
from .training_state import TrainingSession, _content_byte_counts, _record_ar_boundary_batch
from .training_token_edit_prefix import prepare_token_edit_prefix_microbatch


def _corrupt_diffusion_inputs(session: TrainingSession, step: int, micro: int) -> None:
    session.global_microbatch_idx = step * session.grad_accum_steps + micro
    session.row_mixed_token_edit = _uses_fixed_row_token_edit_layout(session.config)
    session.token_edit_branch = (
        False
        if session.row_mixed_token_edit
        else _token_edit_branch_enabled_for_microbatch(session.config, session.global_microbatch_idx)
    )
    session.self_generated_edit_branch = session.token_edit_branch and _uses_self_generated_token_edit(session.config)
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
                session.step_soft_input_stat_sums[key] = session.step_soft_input_stat_sums.get(key, 0.0) + float(value)
        session.step_soft_input_stat_count += 1
    if session.record_early_residual_diagnostics and session.residual_diagnostic_batch is None:
        session.residual_diagnostic_batch = (
            session.z_t.detach(),
            _batch_model_attention_mask(session.batch, session.attention_mask).detach(),
            session.batch.position_ids.detach() if getattr(session.batch, "position_ids", None) is not None else None,
        )


def prepare_diffusion_batch(session: TrainingSession, step: int, micro: int) -> None:
    _corrupt_diffusion_inputs(session, step, micro)
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
    session.content_tokens, session.content_bytes = _content_byte_counts(session, session.x_0, session.attention_mask)
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
            raise ValueError("noise.per_token_noise_levels requires token-shaped t values from training corruption")
        session.lossable_t_values = session.t_val[session.loss_mask].detach().float()
        _accumulate_value_histogram(session.step_per_token_noise_level_hist_counts, session.lossable_t_values, 0.0, 1.0)
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
