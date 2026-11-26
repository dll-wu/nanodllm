"""Stopped-gradient partial decoding trajectories for T2T training canvases."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import ExperimentConfig
from .training_self_generated_edit import generation_blocked_argmax

TRAJECTORY_CANVAS_MODES = ("trajectory_gold_context", "trajectory_model_context")


@dataclass(frozen=True)
class TrajectoryDraftResult:
    """One partial rollout aligned to the original active target ordering."""

    tokens: torch.Tensor
    predictions: torch.Tensor
    correct: torch.Tensor
    drafted: torch.Tensor
    correction_loss_target_mask: torch.Tensor
    model_forwards: int
    committed_targets: int
    remaining_targets: int


def uses_trajectory_canvas(config: ExperimentConfig) -> bool:
    """Return whether training requests one of the partial rollout canvases."""

    return config.train.token_edit_canvas_mode in TRAJECTORY_CANVAS_MODES


def trajectory_draft_forwards_per_microbatch(config: ExperimentConfig) -> int:
    """Return the configured number of stopped-gradient rollout forwards."""

    if not uses_trajectory_canvas(config):
        return 1
    steps = int(config.train.token_edit_trajectory_steps)
    completion_fraction = float(config.train.token_edit_trajectory_completion_fraction)
    return max(1, min(steps - 1, int(math.floor(steps * completion_fraction + 0.5))))


def _allowed_top2_margin(
    logits: torch.Tensor,
    blocked_token_ids: set[int] | frozenset[int] | list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Return the allowed-vocabulary top-one/top-two logit margin."""

    filtered = logits.detach().float().clone()
    blocked = sorted(set(int(token_id) for token_id in blocked_token_ids))
    if blocked:
        blocked_tensor = torch.tensor(blocked, dtype=torch.long, device=filtered.device)
        if int(blocked_tensor.min().item()) < 0 or int(blocked_tensor.max().item()) >= filtered.shape[-1]:
            raise ValueError("trajectory blocked token IDs are outside the logits vocabulary")
        filtered.index_fill_(-1, blocked_tensor, -torch.inf)
    if filtered.shape[-1] - len(blocked) < 2:
        raise ValueError("trajectory selection requires at least two generation-allowed tokens")
    top2 = filtered.topk(k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    if not bool(torch.isfinite(margin).all().item()):
        raise RuntimeError("trajectory selection produced a non-finite allowed-vocabulary margin")
    return margin


def _commit_quota(target_count: int, completion_fraction: float) -> int:
    """Resolve a per-row context quota while retaining correction supervision."""

    if target_count <= 1:
        return 0
    quota = int(math.floor(target_count * completion_fraction + 0.5))
    return min(target_count - 1, max(1, quota))


@torch.no_grad()
def build_trajectory_argmax_draft(
    model,
    config: ExperimentConfig,
    x_0: torch.Tensor,
    z_t: torch.Tensor,
    attention_mask: torch.Tensor,
    target_indices: torch.Tensor,
    generation_blocked_token_ids: set[int] | frozenset[int] | list[int] | tuple[int, ...],
    noise_level,
    *,
    position_ids: torch.Tensor | None = None,
) -> TrajectoryDraftResult:
    """Build a margin-ranked partial rollout and retain only unresolved-target loss.

    Both trajectory modes select positions from the same stopped-gradient model
    predictions. The model-context mode inserts those predictions; the gold-context
    control inserts the corresponding clean tokens. Thus gold context never chooses
    positions or changes the commit schedule within a microbatch.
    """

    mode = config.train.token_edit_canvas_mode
    if mode not in TRAJECTORY_CANVAS_MODES:
        raise ValueError(f"unsupported trajectory canvas mode: {mode}")
    if x_0.shape != z_t.shape or x_0.dim() != 2:
        raise ValueError("trajectory clean and corrupted inputs must share one two-dimensional shape")
    if target_indices.dim() != 1 or target_indices.dtype != torch.long:
        raise ValueError("trajectory target_indices must be one-dimensional int64")
    if target_indices.device != z_t.device:
        raise ValueError("trajectory target_indices must share the token device")
    completion_fraction = float(config.train.token_edit_trajectory_completion_fraction)
    steps = int(config.train.token_edit_trajectory_steps)
    if not 0.0 < completion_fraction < 1.0:
        raise ValueError("trajectory completion fraction must lie in (0, 1)")
    if steps < 2:
        raise ValueError("trajectory steps must be at least two")

    canvas = z_t.detach().clone()
    target_count = int(target_indices.numel())
    if target_count == 0:
        empty_bool = torch.empty(0, dtype=torch.bool, device=z_t.device)
        return TrajectoryDraftResult(
            tokens=canvas,
            predictions=torch.empty(0, dtype=torch.long, device=z_t.device),
            correct=empty_bool,
            drafted=empty_bool,
            correction_loss_target_mask=empty_bool,
            model_forwards=0,
            committed_targets=0,
            remaining_targets=0,
        )

    batch_size, sequence_length = x_0.shape
    target_rows = torch.div(target_indices, sequence_length, rounding_mode="floor")
    if bool(((target_rows < 0) | (target_rows >= batch_size)).any().item()):
        raise ValueError("trajectory target indices are outside the token canvas")
    initial_counts = torch.bincount(target_rows, minlength=batch_size)
    quotas = torch.tensor(
        [_commit_quota(int(count), completion_fraction) for count in initial_counts.tolist()],
        dtype=torch.long,
        device=z_t.device,
    )
    drafted = torch.zeros(target_count, dtype=torch.bool, device=z_t.device)
    predictions = torch.empty(target_count, dtype=torch.long, device=z_t.device)
    exposure_steps = trajectory_draft_forwards_per_microbatch(config)
    model_forwards = 0

    for step_index in range(1, exposure_steps + 1):
        remaining_local = (~drafted).nonzero(as_tuple=True)[0]
        if remaining_local.numel() == 0:
            break
        remaining_indices = target_indices.index_select(0, remaining_local)
        model_kwargs = {
            "logits_indices": remaining_indices,
            "return_aux_logits": False,
            "return_hierarchical_logits": False,
            "return_length_logits": False,
            "attention_mask": attention_mask,
            "noise_level": noise_level if config.noise.time_conditioning else None,
            "input_embeddings": None,
        }
        if position_ids is not None:
            model_kwargs["position_ids"] = position_ids
        logits = model(canvas, **model_kwargs)
        if not torch.is_tensor(logits) or logits.dim() != 2 or logits.shape[0] != remaining_local.numel():
            raise ValueError("trajectory model output must contain one [vocab] row per remaining target")
        local_predictions = generation_blocked_argmax(
            logits,
            generation_blocked_token_ids,
            field_name="token-edit trajectory",
        )
        predictions.index_copy_(0, remaining_local, local_predictions)
        margins = _allowed_top2_margin(logits, generation_blocked_token_ids)
        model_forwards += 1

        committed_by_row = torch.bincount(target_rows[drafted], minlength=batch_size)
        cumulative = torch.div(quotas * step_index, exposure_steps, rounding_mode="floor")
        if step_index == exposure_steps:
            cumulative = quotas
        needed = (cumulative - committed_by_row).clamp_min(0)
        commit_local_parts: list[torch.Tensor] = []
        remaining_rows = target_rows.index_select(0, remaining_local)
        for row, count in enumerate(needed.tolist()):
            if count <= 0:
                continue
            row_positions = (remaining_rows == row).nonzero(as_tuple=True)[0]
            if row_positions.numel() < count:
                raise RuntimeError("trajectory commit quota exceeds remaining row targets")
            selected = row_positions.index_select(0, torch.topk(margins.index_select(0, row_positions), count).indices)
            commit_local_parts.append(selected)
        if not commit_local_parts:
            continue
        selected_remaining = torch.cat(commit_local_parts)
        selected_original = remaining_local.index_select(0, selected_remaining)
        selected_indices = target_indices.index_select(0, selected_original)
        inserted = (
            x_0.reshape(-1).index_select(0, selected_indices)
            if mode == "trajectory_gold_context"
            else predictions.index_select(0, selected_original)
        )
        canvas.reshape(-1).index_copy_(0, selected_indices, inserted.to(dtype=canvas.dtype))
        drafted[selected_original] = True

    correction_mask = ~drafted
    if not bool(correction_mask.any().item()):
        raise RuntimeError("trajectory canvas retained no correction targets")
    clean_targets = x_0.reshape(-1).index_select(0, target_indices)
    return TrajectoryDraftResult(
        tokens=canvas.detach(),
        predictions=predictions.detach(),
        correct=predictions.eq(clean_targets).detach(),
        drafted=drafted.detach(),
        correction_loss_target_mask=correction_mask.detach(),
        model_forwards=model_forwards,
        committed_targets=int(drafted.sum().item()),
        remaining_targets=int(correction_mask.sum().item()),
    )


def apply_trajectory_loss_scope(session) -> None:
    """Zero committed-target weights and preserve the intended loss scale."""

    target_mask = session.token_edit_correction_loss_target_mask
    if target_mask is None:
        return
    if target_mask.dtype != torch.bool or target_mask.shape != session.token_weights.shape:
        raise ValueError("trajectory correction mask must be boolean and match target weights")
    session.token_weights = session.token_weights * target_mask.to(dtype=session.token_weights.dtype)
    session.token_edit_loss_denominator = max(1, int(target_mask.sum().item()))
