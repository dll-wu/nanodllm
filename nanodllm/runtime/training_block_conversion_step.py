"""One-microbatch orchestration for causal-clean BD3 block conversion."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext

import torch

from .block_partitions import training_partition_ids
from .training_block_conversion import (
    block_conversion_losses,
    build_block_conversion_corruption,
    build_block_conversion_logit_plan,
)
from .training_block_recipe_views import anchor_loss, backward_ar_teacher, recipe_corruption, record_recipe_microbatch
from .training_features import _uses_bd3lm_causal_ar_auxiliary
from .training_objective import _forward_training_model, _update_eos_supervision_batch_counts
from .training_runtime import _batch_model_attention_mask
from .training_state import TrainingSession, _content_byte_counts, _record_ar_boundary_batch


def run_block_conversion_microbatch(
    session: TrainingSession,
    *,
    step: int,
    micro: int,
    backward_empty_ddp_microbatch: Callable[[TrainingSession], None],
) -> None:
    """Run one causal-clean doubled-stream decoder/AR microbatch."""

    session.global_microbatch_idx = step * session.grad_accum_steps + micro
    block_ids = training_partition_ids(session.config, session.x_0, getattr(session.batch, "position_ids", None), step)
    session.forward_attention_mask = _batch_model_attention_mask(session.batch, session.attention_mask)
    session.block_conversion_corruption = build_block_conversion_corruption(
        session.config,
        session.x_0,
        session.loss_mask,
        mask_token_id=session.model_config.mask_token_id,
        vocab_size=session.model_config.vocab_size,
        microbatch_index=session.global_microbatch_idx,
        position_ids=getattr(session.batch, "position_ids", None),
        block_size=session.block_conversion_training_block_size,
        block_ids=block_ids,
    )
    session.block_conversion_corruption = recipe_corruption(session, session.block_conversion_corruption)
    session.z_t = session.block_conversion_corruption.z_t
    session.w = session.block_conversion_corruption.active.to(dtype=torch.float32)
    session.active = session.block_conversion_corruption.active
    session.t_val = session.block_conversion_corruption.block_times
    session.noise_level = None
    session.soft_input_embeddings = None
    session.target_indices = session.active.reshape(-1).nonzero(as_tuple=True)[0]
    session.block_conversion_plan = build_block_conversion_logit_plan(
        session.x_0,
        session.loss_mask,
        session.attention_mask,
        session.active,
        include_ar=_uses_bd3lm_causal_ar_auxiliary(session.config),
        position_ids=getattr(session.batch, "position_ids", None),
    )
    session.logits_indices = session.block_conversion_plan.logits_indices
    session.sync_context = (
        session.model.no_sync if session.is_ddp and micro < session.grad_accum_steps - 1 else nullcontext
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
        session,
        session.x_0,
        session.attention_mask,
    )
    session.step_valid_tokens += session.valid_tokens
    session.step_lossable_tokens += session.lossable_tokens
    session.step_content_tokens += session.content_tokens
    session.step_content_bytes += session.content_bytes
    session.step_active_tokens += session.block_conversion_plan.decoder_count
    if session.block_conversion_corruption.branch == "t2t":
        session.step_t2t_lossable_tokens += session.lossable_tokens
        session.step_t2t_eligible_tokens += session.lossable_tokens
    else:
        session.step_m2t_lossable_tokens += session.lossable_tokens

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
    if session.record_early_residual_diagnostics and session.residual_diagnostic_batch is None:
        session.residual_diagnostic_batch = (
            session.z_t.detach(),
            session.forward_attention_mask.detach(),
            session.batch.position_ids.detach() if getattr(session.batch, "position_ids", None) is not None else None,
        )

    if session.block_conversion_plan.decoder_count + session.block_conversion_plan.ar_count == 0:
        if session.is_ddp:
            backward_empty_ddp_microbatch(session)
        return

    with session.sync_context():
        with session.autocast_ctx:
            session.logits = _forward_training_model(
                session.model,
                session.config,
                session.z_t,
                session.x_0,
                session.forward_attention_mask,
                logits_indices=session.block_conversion_plan.logits_indices,
                return_aux_logits=False,
                noise_level=None,
                return_hierarchical_logits=False,
                return_length_logits=False,
                input_embeddings=None,
                position_ids=getattr(session.batch, "position_ids", None),
                bd3lm_block_size=session.block_conversion_training_block_size,
                bd3lm_block_ids=block_ids,
            )
            session.block_conversion_losses = block_conversion_losses(
                session.logits,
                session.block_conversion_plan,
                session.block_conversion_corruption.decoder_weights,
                decoder_lossable_tokens=session.lossable_tokens,
                ar_loss_weight=float(session.config.train.bd3lm_causal_ar_loss_weight),
            )
            session.loss = session.block_conversion_losses.loss
            session.loss = session.loss + anchor_loss(session)
        session.loss_value = float(session.loss.detach().item())
        if not math.isfinite(session.loss_value):
            raise RuntimeError(f"non-finite block-conversion loss at step {step}, micro {micro}: {session.loss_value}")
        session.step_loss_sum += session.loss_value
        session.step_main_ce_sum += float(session.block_conversion_losses.decoder_raw_ce.detach().item())
        session.step_main_loss_sum += float(session.block_conversion_losses.decoder_loss.detach().item())
        session.step_loss_normalization_scale_sum += session.block_conversion_losses.decoder_loss_normalization_scale
        if _uses_bd3lm_causal_ar_auxiliary(session.config):
            session.step_ar_unsmoothed_ce_sum += float(session.block_conversion_losses.ar_loss.detach().item())
        if session.block_conversion_corruption.branch == "t2t":
            session.step_t2t_ce_sum += float(session.block_conversion_losses.decoder_raw_ce.detach().item())
            session.step_t2t_loss_sum += float(session.block_conversion_losses.decoder_loss.detach().item())
            session.step_t2t_loss_count += 1
            session.step_t2t_microbatches += 1
            session.step_t2t_active_tokens += session.block_conversion_plan.decoder_count
        else:
            session.step_m2t_ce_sum += float(session.block_conversion_losses.decoder_raw_ce.detach().item())
            session.step_m2t_loss_sum += float(session.block_conversion_losses.decoder_loss.detach().item())
            session.step_m2t_loss_count += 1
            session.step_m2t_microbatches += 1
            session.step_m2t_active_tokens += session.block_conversion_plan.decoder_count
        session.step_loss_count += 1
        session.scaler.scale(session.loss / session.grad_accum_steps).backward()
        backward_ar_teacher(session)
        record_recipe_microbatch(session, step, micro)

    session.t_values = session.t_val[session.loss_mask].detach().float().flatten()
    if session.t_values.numel() > 0:
        session.step_t_sum += float(session.t_values.sum().item())
        session.step_t_min = min(session.step_t_min, float(session.t_values.min().item()))
        session.step_t_max = max(session.step_t_max, float(session.t_values.max().item()))
        session.step_t_count += int(session.t_values.numel())
        if session.block_conversion_corruption.branch == "m2t":
            session.step_schedule_multiplier_sum += float((1.0 / session.t_values.clamp_min(1e-8)).sum().item())
        else:
            session.step_schedule_multiplier_sum += float(session.t_values.numel())
