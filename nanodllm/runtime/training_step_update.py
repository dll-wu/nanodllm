"""Gradient diagnostics, clipping, and optimizer updates."""

from __future__ import annotations

import os

import torch

from .parameterization import parameterization_gradient_rms_metrics, parameterization_parameter_update_rms_metrics
from .telemetry import append_jsonl
from .training_metrics import (
    _component_update_grad_ratio_metrics,
    _component_update_norm_metrics,
    _grad_norm,
    _gradient_component_norm_metrics,
    _layerwise_gradient_norm_metrics,
    _residual_branch_rms_metrics,
    _residual_projection_init_metrics,
    _second_moment_norm,
    _update_norm,
)
from .training_optim import _step_optimizers, _unscale_optimizers
from .training_state import TrainingSession


def _parameterization_role_telemetry_enabled(session: TrainingSession) -> bool:
    """Keep detailed O(P) role reductions on explicitly bound campaigns only."""

    return (
        session.record_total_update_norm
        and session.rank == 0
        and session.config.parameterization.contract_binding == "required"
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


def apply_optimizer_step(session: TrainingSession, step: int) -> None:
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
        session.progress.last.grad_clip_ratio = min(1.0, session.grad_clip_threshold / max(session.grad_norm, 1e-12))
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
