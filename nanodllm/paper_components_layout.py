"""Shared source layout, corruption-time and loss boundaries for B1--B3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .paper_runtime.data import DLMDataBatch
from .paper_runtime.training_bd3lm import bd3lm_block_times_from_candidates


@dataclass(frozen=True)
class TrainingView:
    tokens: torch.Tensor
    positions: torch.Tensor
    visibility: torch.Tensor
    logit_indices: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor


def batch_layout(batch: DLMDataBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Bind valid tokens, reset positions and document visibility together."""
    x = batch.x_0
    if x.ndim != 2 or x.dtype != torch.long or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("batch tokens must be nonempty int64 [batch, sequence]")
    if any(
        t.shape != x.shape or t.dtype != torch.bool or t.device != x.device
        for t in (batch.loss_mask, batch.attention_mask)
    ):
        raise ValueError("batch masks must be Boolean tensors matching tokens")
    if bool((batch.loss_mask & ~batch.attention_mask).any()):
        raise ValueError("loss targets cannot lie outside valid source tokens")
    positions = batch.position_ids
    if positions is None:
        positions = torch.arange(x.shape[1], device=x.device).expand_as(x)
    if positions.shape != x.shape or positions.dtype != torch.long or positions.device != x.device:
        raise ValueError("position_ids must be int64 and match tokens")
    if bool(((positions < 0) | (positions >= x.shape[1])).any()):
        raise ValueError("segment-reset positions must lie within source context")
    valid = batch.attention_mask
    resets = positions == 0
    resets = resets.clone()
    resets[:, 0] = True
    segments = resets.long().cumsum(1)
    visibility = (segments[:, :, None] == segments[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    if batch.document_boundary_mask is not None:
        boundary = batch.document_boundary_mask
        if boundary.shape != visibility.shape or boundary.dtype != torch.bool or boundary.device != x.device:
            raise ValueError("document_boundary_mask must match batch visibility")
        visibility &= boundary
    return positions, visibility


def block_times(
    positions: torch.Tensor, *, block_size: int, generator: torch.Generator, policy: dict, default_min: float = 0.05
) -> torch.Tensor:
    low, high = float(policy.get("t_min", default_min)), float(policy.get("t_max", 1.0))
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high <= 1):
        raise ValueError("noise support must satisfy 0 < t_min <= t_max <= 1")
    candidates = torch.rand(positions.shape, device=positions.device, generator=generator) * (high - low) + low
    return bd3lm_block_times_from_candidates(candidates, block_size=block_size, position_ids=positions)


def weighted_target_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, *, denominator: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize by original lossable source tokens, not realized mask count."""
    if denominator <= 0 or logits.ndim != 2 or logits.shape[0] != targets.numel() or weights.shape != targets.shape:
        raise ValueError("invalid target reduction shapes or denominator")
    if targets.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, zero
    ce = F.cross_entropy(logits.float(), targets, reduction="none")
    return (ce * weights.float()).sum() / denominator, ce.sum()


def validate_model(model) -> None:
    config = model.config
    if config.causal or config.time_conditioning or config.output_factorization != "flat":
        raise ValueError("conversion requires plain noncausal, time-agnostic, flat-logit backbone")
