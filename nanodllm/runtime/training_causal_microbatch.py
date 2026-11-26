"""Accumulate gradients and next-token statistics for a causal microbatch."""

from __future__ import annotations

import math
from contextlib import nullcontext

from .training_mtp import causal_training_loss
from .training_objective import _causal_ar_label_smoothing
from .training_runtime import _batch_model_attention_mask
from .training_state import TrainingSession, _content_byte_counts, _record_ar_boundary_batch


def run_causal_microbatch(session: TrainingSession, step: int, micro: int, depth_kwargs: dict[str, int]) -> None:
    if session.record_early_residual_diagnostics and session.residual_diagnostic_batch is None:
        session.diagnostic_attention_mask = _batch_model_attention_mask(session.batch, session.attention_mask)
        session.diagnostic_position_ids = (
            session.batch.position_ids.detach() if getattr(session.batch, "position_ids", None) is not None else None
        )
        session.residual_diagnostic_batch = (
            session.x_0.detach(),
            session.diagnostic_attention_mask.detach() if session.diagnostic_attention_mask is not None else None,
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
    session.content_tokens, session.content_bytes = _content_byte_counts(session, session.x_0, session.attention_mask)
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
            session.model_kwargs["attention_mask"] = _batch_model_attention_mask(session.batch, session.attention_mask)
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
        session.step_loss_count += int(session.target_mask.any()) if session.config.model.mtp_mode != "none" else 1
        session.scaler.scale(session.loss / session.grad_accum_steps).backward()
