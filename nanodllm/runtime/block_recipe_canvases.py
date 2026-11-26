"""Packed-segment-safe canvas transforms for block mechanism screens."""

import torch

from .training_block_conversion import BlockConversionCorruption


def block_groups(tokens: torch.Tensor, position_ids: torch.Tensor | None, block_size: int) -> torch.Tensor:
    batch, length = tokens.shape
    positions = (
        torch.arange(length, device=tokens.device).expand(batch, -1)
        if position_ids is None
        else position_ids.expand(batch, -1)
    )
    starts = torch.ones_like(tokens, dtype=torch.bool)
    starts[:, 1:] = positions[:, 1:] != positions[:, :-1] + 1
    segments = starts.long().cumsum(1) - 1
    row = torch.arange(batch, device=tokens.device)[:, None]
    return (row * length + segments) * (length + 1) + positions // block_size


def lowest_per_group(
    scores: torch.Tensor, eligible: torch.Tensor, groups: torch.Tensor, counts_from: torch.Tensor
) -> torch.Tensor:
    """Select the requested count in each group, with stable position tie-breaking."""
    indices = eligible.flatten().nonzero(as_tuple=True)[0]
    result = torch.zeros_like(eligible).flatten()
    if indices.numel() == 0:
        return result.view_as(eligible)
    keys = groups.flatten()[indices]
    _, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    counts = torch.zeros(int(inverse.max().item()) + 1, device=scores.device, dtype=torch.long)
    counts.scatter_add_(0, inverse, counts_from.flatten()[indices].long())
    order = torch.argsort(scores.flatten()[indices], stable=True)
    order = order[torch.argsort(inverse[order], stable=True)]
    sorted_groups = inverse[order]
    first = torch.ones_like(sorted_groups, dtype=torch.bool)
    first[1:] = sorted_groups[1:] != sorted_groups[:-1]
    ranks = torch.arange(order.numel(), device=scores.device)
    group_starts = torch.where(first, ranks, 0).cummax(0).values
    keep = ranks - group_starts < counts[sorted_groups]
    result[indices[order[keep]]] = True
    return result.view_as(eligible)


def confidence_margin(logits: torch.Tensor) -> torch.Tensor:
    top = logits.float().topk(2, dim=-1).values
    return top[:, 0] - top[:, 1]


def confidence_masked_canvas(x_0, loss_mask, corruption, scores, groups, mask_token_id):
    active = lowest_per_group(scores, loss_mask, groups, corruption.active)
    canvas = x_0.masked_fill(active, mask_token_id)
    weights = 1.0 / corruption.block_times[active].float().clamp_min(1e-8)
    return BlockConversionCorruption("m2t", canvas, active, corruption.block_times, weights, int(active.sum().item()))


def generated_canvas(x_0, loss_mask, masked, draft_tokens, *, mask_token_id, block_times):
    """Commit a random half of masked positions; supervise the entire canvas."""
    commit = (torch.rand(x_0.shape, device=x_0.device) < 0.5) & masked.eq(mask_token_id) & loss_mask
    canvas = torch.where(commit, draft_tokens, masked).detach()
    active = loss_mask.bool()
    return BlockConversionCorruption(
        "t2t",
        canvas,
        active,
        block_times,
        torch.ones(int(active.sum().item()), device=x_0.device),
        int(((canvas != x_0) & active).sum().item()),
    )
