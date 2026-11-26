"""Fast-dLLM-v2 B3 operators and uncached shifted-logit readout.

Sources: arXiv:2509.26328v1 sections 3.2/A.1--A.3; official HF
Efficient-Large-Model/Fast_dLLM_v2_7B modeling.py at
0661abf5f9f0ee338970d091052a26c8efa51974; NVlabs/Fast-dLLM generation_functions.py
at a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b. Independently implemented operators.

BL32, per-block U(.001, 1) masks, complementary views, unweighted source-target
CE, block-causal clean states and preceding-position logits follow those sources.
Ordinary-text source fragments, document isolation/reset positions, no dropped
packing remainders and sequential view execution are explicit local adaptations.
Padding MASKs are visible context but never direct loss targets; contextual
gradients through them are not suppressed. This is not an NELBO estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from nanodllm.runtime.data import DLMDataBatch
from nanodllm.runtime.training_bd3lm import bd3lm_concat_inputs

from .layout import TrainingView, batch_layout, block_times, validate_model, weighted_target_loss


@dataclass(frozen=True)
class AlignedBatch:
    batch: DLMDataBatch
    source_indices: torch.Tensor
    padding: torch.Tensor


def align_mask_padding(
    batch: DLMDataBatch, *, block_size: int, mask_token_id: int, exclude_unpredictable_first_targets: bool = False
) -> AlignedBatch:
    """Repack complete source-row segments with visible MASK tails, without loss.

    Segments already clipped by the shared source loader remain clipped: this
    routine does not invent missing document context. Add rows when alignment
    would overflow, rather than dropping source tokens or changing source EB.
    The caller chunks expanded rows for bounded peak memory. A segment's first
    token must be protected unless the explicit first-target exclusion policy
    is enabled. Clipped source-row fragments can start with ordinary lossable
    tokens, for which no same-document preceding-position readout exists. The
    exclusion is applied to readout later, not to source or corruption support.
    """
    positions, visibility = batch_layout(batch)
    if type(exclude_unpredictable_first_targets) is not bool:
        raise ValueError("exclude_unpredictable_first_targets must be Boolean")
    x = batch.x_0
    length = x.shape[1]
    if type(block_size) is not int or block_size <= 0 or length % block_size:
        raise ValueError("Fast-v2 block size must divide source context")
    valid_rows = batch.attention_mask.detach().cpu().tolist()
    pos_rows = positions.detach().cpu().tolist()
    loss_rows = batch.loss_mask.detach().cpu().tolist()
    placements = []
    output_row, cursor = 0, 0
    for row, valid in enumerate(valid_rows):
        col = 0
        while col < length:
            if not valid[col]:
                col += 1
                continue
            start = col
            if pos_rows[row][start] != 0 or (loss_rows[row][start] and not exclude_unpredictable_first_targets):
                raise ValueError("Fast-v2 segments require reset positions and a protected first token")
            col += 1
            while col < length and valid[col] and pos_rows[row][col] != 0:
                if pos_rows[row][col] != col - start:
                    raise ValueError("Fast-v2 requires consecutive segment-reset positions")
                col += 1
            count = col - start
            padded = (count + block_size - 1) // block_size * block_size
            if cursor + padded > length:
                output_row, cursor = output_row + 1, 0
            placements.append((row, start, count, padded, output_row, cursor))
            cursor += padded
    if not placements:
        raise ValueError("Fast-v2 requires nonempty valid source data")
    # Do not silently replace a specialized source visibility contract.
    segments = (positions == 0).long().cumsum(1)
    expected = (
        (segments[:, :, None] == segments[:, None, :])
        & batch.attention_mask[:, :, None]
        & batch.attention_mask[:, None, :]
    )
    if not torch.equal(visibility, expected):
        raise ValueError("Fast-v2 alignment supports ordinary document-isolated visibility only")
    shape = (output_row + 1, length)
    # Construct compact metadata on CPU, then upload once per tensor instead of
    # launching several CUDA kernels for every short packed document.
    index_rows = [[-1] * length for _ in range(shape[0])]
    position_rows = [[0] * length for _ in range(shape[0])]
    visibility_rows = [[False] * length for _ in range(shape[0])]
    padding_rows = [[False] * length for _ in range(shape[0])]
    for row, start, count, padded, dest_row, dest_start in placements:
        index_rows[dest_row][dest_start : dest_start + count] = range(
            row * length + start, row * length + start + count
        )
        position_rows[dest_row][dest_start : dest_start + padded] = range(padded)
        visibility_rows[dest_row][dest_start : dest_start + padded] = [True] * padded
        padding_rows[dest_row][dest_start + count : dest_start + padded] = [True] * (padded - count)
    source_indices = torch.tensor(index_rows, dtype=torch.long, device=x.device)
    new_positions = torch.tensor(position_rows, dtype=torch.long, device=x.device)
    visible = torch.tensor(visibility_rows, dtype=torch.bool, device=x.device)
    padding = torch.tensor(padding_rows, dtype=torch.bool, device=x.device)
    gathered = source_indices.clamp_min(0).reshape(-1)
    source = source_indices >= 0
    tokens = x.reshape(-1).index_select(0, gathered).view(shape).masked_fill(~source, mask_token_id)
    loss_mask = batch.loss_mask.reshape(-1).index_select(0, gathered).view(shape) & source
    aligned = DLMDataBatch(tokens, loss_mask, visible, epoch=batch.epoch, position_ids=new_positions)
    return AlignedBatch(aligned, source_indices, padding)


def complementary_active_masks(
    loss_mask: torch.Tensor, times: torch.Tensor, *, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    if times.shape != loss_mask.shape or loss_mask.dtype != torch.bool:
        raise ValueError("complementary masks require matching time and Boolean support tensors")
    if bool(((times <= 0) | (times > 1)).any()):
        raise ValueError("corruption probabilities must lie in (0, 1]")
    first = (torch.rand(times.shape, device=times.device, generator=generator) < times) & loss_mask
    return first, loss_mask & ~first


def build_fastv2_view(
    batch: DLMDataBatch,
    active: torch.Tensor,
    *,
    block_size: int,
    mask_token_id: int,
    exclude_unpredictable_first_targets: bool = False,
) -> TrainingView:
    """Target i reads noised-half logit i-1, including block boundaries.

    The released trainer shifts globally within a document; it does not move a
    boundary target to clean-half logits. Preserve that training operator even
    though native inference seeds a new block from the completed clean prefix.
    """
    positions, visibility = batch_layout(batch)
    if active.shape != batch.x_0.shape or active.dtype != torch.bool or bool((active & ~batch.loss_mask).any()):
        raise ValueError("Fast-v2 active targets must be lossable source tokens")
    if type(exclude_unpredictable_first_targets) is not bool:
        raise ValueError("exclude_unpredictable_first_targets must be Boolean")
    if bool((active & (positions == 0)).any()) and not exclude_unpredictable_first_targets:
        raise ValueError("a shifted target cannot cross a document boundary")
    # Preserve the upstream corruption draw even for an unscorable first token.
    # It must not silently become guaranteed-gold context for the next target.
    noisy = batch.x_0.masked_fill(active, mask_token_id)
    tokens, visibility, concat_positions = bd3lm_concat_inputs(
        noisy, batch.x_0, visibility, block_size=block_size, clean_stream_mode="block_causal", position_ids=positions
    )
    target_mask = active & (positions > 0)
    rows, cols = target_mask.nonzero(as_tuple=True)
    if bool((cols == 0).any()):
        raise ValueError("a shifted target needs a preceding query position")
    return TrainingView(
        tokens,
        concat_positions,
        visibility,
        rows * tokens.shape[1] + cols - 1,
        batch.x_0[target_mask],
        torch.ones(rows.numel(), device=tokens.device),
    )


def backward_fastv2(
    model, batch: DLMDataBatch, *, block_size: int, generator: torch.Generator, accumulation: int, policy: dict
) -> dict[str, float | int]:
    exclude_first = policy.get("exclude_unpredictable_first_targets", False)
    aligned = align_mask_padding(
        batch,
        block_size=block_size,
        mask_token_id=model.config.mask_token_id,
        exclude_unpredictable_first_targets=exclude_first,
    )
    packed = aligned.batch
    source_lossable = int(batch.loss_mask.sum())
    excluded = int((packed.loss_mask & (packed.position_ids == 0)).sum())
    denominator = source_lossable - excluded
    if denominator <= 0:
        raise ValueError("Fast-v2 requires at least one shift-predictable source target")
    if bool((packed.loss_mask & (packed.x_0 == model.config.mask_token_id)).any()):
        raise ValueError("MASK cannot be an eligible clean source target")
    times = block_times(
        packed.position_ids, block_size=block_size, generator=generator, policy=policy, default_min=0.001
    )
    masks = complementary_active_masks(packed.loss_mask, times, generator=generator)
    max_rows = batch.x_0.shape[0]
    loss_total = 0.0
    raw_total = 0.0
    calls = work = targets = 0
    for active in masks:
        for start in range(0, packed.x_0.shape[0], max_rows):
            end = start + max_rows
            chunk = DLMDataBatch(
                packed.x_0[start:end],
                packed.loss_mask[start:end],
                packed.attention_mask[start:end],
                position_ids=packed.position_ids[start:end],
            )
            view = build_fastv2_view(
                chunk,
                active[start:end],
                block_size=block_size,
                mask_token_id=model.config.mask_token_id,
                exclude_unpredictable_first_targets=exclude_first,
            )
            logits = model(
                view.tokens,
                logits_indices=view.logit_indices,
                attention_mask=view.visibility,
                position_ids=view.positions,
            )
            loss, raw_sum = weighted_target_loss(logits, view.targets, view.weights, denominator=denominator)
            (loss / accumulation).backward()
            loss_total += float(loss.detach())
            raw_total += float(raw_sum.detach())
            calls += 1
            work += view.tokens.numel()
            targets += view.targets.numel()
            del logits, loss, raw_sum, view
    if targets != denominator:
        raise AssertionError("complementary views did not cover each eligible source token once")
    return {
        "loss": loss_total,
        "raw_ce": raw_total / denominator,
        "source_positions": batch.x_0.numel(),
        "lossable_tokens": source_lossable,
        "objective_target_tokens": denominator,
        "excluded_unpredictable_first_targets": excluded,
        "corrupted_tokens": source_lossable,
        "active_tokens": targets,
        "forward_calls": calls,
        "backward_calls": calls,
        "forward_token_positions": work,
        "backward_token_positions": work,
        "block_size": block_size,
        "curriculum_phase_index": 0,
        "view_count": 2,
        "padding_positions": int(aligned.padding.sum()),
        "aligned_rows": packed.x_0.shape[0],
        "full_context_single_stream": 0,
    }


def forward_block_logits(model, prefix: torch.Tensor, active: torch.Tensor, *, policy: dict) -> torch.Tensor:
    """Differentiable uncached Fast-v2 logits for one active position-defined block.

    Returns [batch, active_length, vocabulary]. Prefix must be nonempty; its
    length sets the absolute block alignment, not a new block origin. Short
    active canvases are MASK-padded to the true block end without gold length
    conditioning. Completed prefix states use block-causal visibility, not the
    B1/B2 token-causal prefix. At an aligned boundary the first target reads the
    last clean-prefix state; within a block every target reads position i-1.

    No cache equivalence/speed claim is made. Caller owns sampling, transfer
    thresholds, EOS handling and clean-block commitment forward accounting.
    """
    validate_model(model)
    if prefix.ndim != 2 or active.ndim != 2 or prefix.shape[0] != active.shape[0] or prefix.shape[0] == 0:
        raise ValueError("prefix and active must be nonempty matching [batch, length] tensors")
    if prefix.dtype != torch.long or active.dtype != torch.long or prefix.device != active.device:
        raise ValueError("prefix and active must share int64 dtype and device")
    p, length = prefix.shape[1], active.shape[1]
    size = policy.get("block_size", 32)
    if type(size) is not int or size <= 0 or p <= 0 or length <= 0:
        raise ValueError("positive block size and nonempty prefix/active canvas are required")
    block_end = (p // size + 1) * size
    if p + length > block_end:
        raise ValueError("active canvas crosses a Fast-v2 block boundary; decode separate blocks")
    if block_end > model.config.sequence_len:
        raise ValueError("padded decoding block exceeds model RoPE context")
    tail = torch.full(
        (prefix.shape[0], block_end - p - length), model.config.mask_token_id, dtype=torch.long, device=prefix.device
    )
    tokens = torch.cat((prefix, active, tail), dim=1)
    positions = torch.arange(block_end, device=tokens.device)
    blocks = positions // size
    visibility = (blocks[:, None] >= blocks[None, :]).expand(tokens.shape[0], -1, -1)
    queries = torch.arange(p - 1, p + length - 1, device=tokens.device)
    indices = (torch.arange(tokens.shape[0], device=tokens.device)[:, None] * block_end + queries[None, :]).reshape(-1)
    logits = model(tokens, logits_indices=indices, attention_mask=visibility, position_ids=positions)
    return logits.view(tokens.shape[0], length, -1)
