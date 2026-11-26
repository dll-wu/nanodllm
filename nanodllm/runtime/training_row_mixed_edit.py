"""Loss assembly for one combined M2T/T2T correction forward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExperimentConfig
from .training_metrics import _per_target_t_values, _weighted_token_mean
from .training_self_generated_edit import (
    SelfGeneratedCorrectionObjective,
    build_fixed_row_self_generated_argmax_draft,
    fixed_12_m2t_4_t2t_row_mask,
    self_generated_t2t_correction_objective,
    self_generated_t2t_target_weights,
)


@dataclass(frozen=True)
class FixedRowCorrectionLoss:
    """Combined loss vectors and branch-local telemetry for one DB16 update."""

    objective_values: torch.Tensor
    token_weights: torch.Tensor
    correction_objective: SelfGeneratedCorrectionObjective
    m2t_target_mask: torch.Tensor
    m2t_ce: torch.Tensor
    t2t_ce: torch.Tensor
    m2t_loss: torch.Tensor
    t2t_loss: torch.Tensor


def fixed_row_lossable_partition(loss_mask: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Partition one DB16 loss denominator into its twelve and four rows."""

    row_mask = fixed_12_m2t_4_t2t_row_mask(loss_mask.shape[0], loss_mask.device)
    return row_mask, int(loss_mask[~row_mask].sum().item()), int(loss_mask[row_mask].sum().item())


def prepare_fixed_row_draft(session) -> None:
    """Populate one training session with its detached four-row draft."""

    if not isinstance(session.t_val, torch.Tensor):
        raise RuntimeError("fixed-row token editing requires per-row MDLM noise times")
    result = build_fixed_row_self_generated_argmax_draft(
        session.self_generated_draft_model,
        session.config,
        session.x_0,
        session.z_t,
        session.active,
        session.forward_attention_mask,
        session.target_indices,
        session.generation_blocked_token_ids,
        session.noise_level,
        position_ids=getattr(session.batch, "position_ids", None),
    )
    if not bool(result.target_mask.any().item()):
        raise RuntimeError("fixed-row token editing requires at least one active T2T target")
    session.z_t = result.tokens
    session.fixed_row_t2t_rows = result.row_mask
    session.row_mixed_t2t_target_mask = result.target_mask
    session.self_generated_draft_predictions = result.predictions
    session.self_generated_draft_correct = result.correct
    session.self_generated_drafted = result.drafted


def assemble_fixed_row_correction_loss(session) -> None:
    """Populate one training session with its combined objective and weights."""

    if session.row_mixed_t2t_target_mask is None:
        raise RuntimeError("fixed-row token editing is missing its T2T target partition")
    if session.self_generated_draft_predictions is None:
        raise RuntimeError("fixed-row token editing is missing detached draft predictions")
    if session.self_generated_draft_correct is None or session.self_generated_drafted is None:
        raise RuntimeError("fixed-row token editing is missing detached draft labels")
    result = fixed_row_correction_loss(
        session.config,
        logits=session.logits,
        targets=session.targets,
        ce_values=session.main_ce_values,
        t_val=session.t_val,
        x_0=session.x_0,
        target_indices=session.target_indices,
        t2t_target_mask=session.row_mixed_t2t_target_mask,
        draft_predictions=session.self_generated_draft_predictions,
        draft_correct=session.self_generated_draft_correct,
        drafted=session.self_generated_drafted,
        m2t_lossable_tokens=session.m2t_lossable_tokens,
        t2t_lossable_tokens=session.t2t_lossable_tokens,
    )
    session.main_objective_values = result.objective_values
    session.token_weights = result.token_weights
    session.self_generated_correction_objective = result.correction_objective
    session.row_mixed_m2t_target_mask = result.m2t_target_mask
    session.row_mixed_m2t_ce = result.m2t_ce
    session.row_mixed_t2t_ce = result.t2t_ce
    session.row_mixed_m2t_loss = result.m2t_loss
    session.row_mixed_t2t_loss = result.t2t_loss


def accumulate_fixed_row_step_branches(session) -> None:
    """Record branch-local means and counts for one mixed correction forward."""

    session.step_m2t_ce_sum += float(session.row_mixed_m2t_ce.detach().item())
    session.step_m2t_loss_sum += float(session.row_mixed_m2t_loss.detach().item())
    session.step_m2t_loss_count += 1
    session.step_m2t_microbatches += 1
    session.step_m2t_active_tokens += int(session.row_mixed_m2t_target_mask.sum().item())
    session.step_t2t_ce_sum += float(session.row_mixed_t2t_ce.detach().item())
    session.step_t2t_loss_sum += float(session.row_mixed_t2t_loss.detach().item())
    session.step_t2t_loss_count += 1
    session.step_t2t_microbatches += 1
    session.step_t2t_active_tokens += int(session.row_mixed_t2t_target_mask.sum().item())


def fixed_row_t2t_observations(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ce_values: torch.Tensor,
    t2t_target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select correction observations aligned to the four T2T rows."""

    if t2t_target_mask.dtype != torch.bool or t2t_target_mask.ndim != 1:
        raise ValueError("fixed-row T2T target mask must be one-dimensional and boolean")
    if not (logits.shape[0] == targets.shape[0] == ce_values.shape[0] == t2t_target_mask.shape[0]):
        raise ValueError("fixed-row correction observations must share one target dimension")
    return logits[t2t_target_mask], targets[t2t_target_mask], ce_values[t2t_target_mask]


def fixed_row_correction_loss(
    config: ExperimentConfig,
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    ce_values: torch.Tensor,
    t_val: torch.Tensor,
    x_0: torch.Tensor,
    target_indices: torch.Tensor,
    t2t_target_mask: torch.Tensor,
    draft_predictions: torch.Tensor,
    draft_correct: torch.Tensor,
    drafted: torch.Tensor,
    m2t_lossable_tokens: int,
    t2t_lossable_tokens: int,
) -> FixedRowCorrectionLoss:
    """Assemble the expected-accuracy T2T rows with CE M2T rows."""

    correction_logits, correction_targets, correction_ce = fixed_row_t2t_observations(
        logits, targets, ce_values, t2t_target_mask
    )
    correction_objective = self_generated_t2t_correction_objective(
        correction_logits,
        correction_targets,
        correction_ce,
        draft_predictions,
        draft_correct,
        objective=config.train.token_edit_correction_objective,
        auxiliary_weight=float(config.train.token_edit_correction_aux_weight),
    )
    t2t_objective_values = torch.zeros_like(ce_values).masked_scatter(t2t_target_mask, correction_objective.loss_values)
    objective_values = torch.where(t2t_target_mask, t2t_objective_values, ce_values)
    per_target_t = _per_target_t_values(t_val, x_0, target_indices)
    token_weights = 1.0 / per_target_t.clamp_min(1e-08)
    token_weights = token_weights.clone()
    token_weights[t2t_target_mask] = self_generated_t2t_target_weights(
        per_target_t[t2t_target_mask],
        draft_correct,
        correct_weight=float(config.train.token_edit_correct_draft_loss_weight),
        wrong_weight=float(config.train.token_edit_wrong_draft_loss_weight),
        drafted=drafted,
    )
    m2t_target_mask = ~t2t_target_mask
    if not bool(m2t_target_mask.any().item()):
        raise RuntimeError("fixed-row token editing requires at least one active M2T target")
    return FixedRowCorrectionLoss(
        objective_values=objective_values,
        token_weights=token_weights,
        correction_objective=correction_objective,
        m2t_target_mask=m2t_target_mask,
        m2t_ce=ce_values[m2t_target_mask].mean(),
        t2t_ce=ce_values[t2t_target_mask].mean(),
        m2t_loss=_weighted_token_mean(
            objective_values[m2t_target_mask], token_weights[m2t_target_mask], m2t_lossable_tokens
        ),
        t2t_loss=_weighted_token_mean(
            objective_values[t2t_target_mask], token_weights[t2t_target_mask], t2t_lossable_tokens
        ),
    )
