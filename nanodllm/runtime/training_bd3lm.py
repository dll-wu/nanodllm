"""BD3-LM training layout, corruption-time, and compute contracts."""

from __future__ import annotations

import torch

from .block_partitions import checked_block_ids
from .noise import sample_positive_uniform_tensor


def _resolved_clean_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if position_ids is None:
        return torch.arange(seq_len, dtype=torch.long, device=device).view(1, seq_len).expand(batch_size, -1)
    resolved = position_ids.to(device=device, dtype=torch.long)
    if resolved.dim() == 1:
        if resolved.shape != (seq_len,):
            raise ValueError(f"BD3-LM position_ids must have shape ({seq_len},), got {tuple(resolved.shape)}")
        return resolved.view(1, seq_len).expand(batch_size, -1)
    if resolved.shape != (batch_size, seq_len):
        raise ValueError(f"BD3-LM position_ids must have shape {(batch_size, seq_len)}, got {tuple(resolved.shape)}")
    return resolved


def duplicated_position_ids(seq_len: int, device=None) -> torch.Tensor:
    """Return global RoPE positions for an unsegmented ``x_t || x_0`` input."""

    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    return torch.cat((positions, positions), dim=0)


def bd3lm_concat_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Duplicate clean positions while preserving per-document resets."""

    if position_ids is None:
        return duplicated_position_ids(seq_len, device=device)
    clean_positions = _resolved_clean_position_ids(
        position_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    return torch.cat((clean_positions, clean_positions), dim=1)


def bd3lm_attention_mask(
    attention_mask: torch.Tensor,
    block_size: int,
    *,
    position_ids: torch.Tensor | None = None,
    clean_stream_mode: str = "block_causal",
    block_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the document-aware BD3-LM mask for an ``x_t || x_0`` input.

    A 2-D input is interpreted as a valid-token mask. A 3-D input is an
    explicit clean-sequence query/key visibility mask, such as the
    same-document mask emitted by content-budget segmented packing. In both
    cases the clean visibility relation is applied independently to every
    allowed BD3-LM quadrant. ``block_causal`` preserves the historical BD3
    clean half. ``causal`` makes clean states strictly left-to-right so the same
    doubled forward can retain an AR stream without within-block label leakage.
    """

    if block_size <= 0:
        raise ValueError("bd3lm block_size must be positive")
    if clean_stream_mode not in {"block_causal", "causal"}:
        raise ValueError("bd3lm clean_stream_mode must be 'block_causal' or 'causal'")
    if attention_mask.dim() == 2:
        batch_size, seq_len = attention_mask.shape
        valid = attention_mask.to(torch.bool)
        clean_visibility = valid[:, :, None] & valid[:, None, :]
    elif attention_mask.dim() == 3:
        batch_size, query_len, key_len = attention_mask.shape
        if query_len != key_len:
            raise ValueError(
                f"BD3-LM 3-D attention_mask must be square over the clean sequence, got {tuple(attention_mask.shape)}"
            )
        seq_len = query_len
        clean_visibility = attention_mask.to(torch.bool)
    else:
        raise ValueError(
            "BD3-LM attention_mask must have shape [batch, seq] or [batch, seq, seq], "
            f"got {tuple(attention_mask.shape)}"
        )

    positions = _resolved_clean_position_ids(
        position_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        device=attention_mask.device,
    )
    block_ids = (
        torch.div(positions, block_size, rounding_mode="floor")
        if block_ids is None
        else checked_block_ids(block_ids, positions)
    )
    query_blocks = block_ids[:, :, None]
    key_blocks = block_ids[:, None, :]
    same_block = (query_blocks == key_blocks) & clean_visibility
    prior_block = (query_blocks > key_blocks) & clean_visibility
    current_or_prior_block = (query_blocks >= key_blocks) & clean_visibility
    causal_clean = (positions[:, :, None] >= positions[:, None, :]) & clean_visibility
    no_visibility = torch.zeros_like(clean_visibility)

    noised_queries = torch.cat((same_block, prior_block), dim=2)
    clean_visibility_mask = causal_clean if clean_stream_mode == "causal" else current_or_prior_block
    clean_queries = torch.cat((no_visibility, clean_visibility_mask), dim=2)
    return torch.cat((noised_queries, clean_queries), dim=1)


def sample_bd3lm_block_times(
    token_shape: tuple[int, int] | torch.Size,
    *,
    block_size: int,
    t_min: float,
    t_max: float,
    device: torch.device,
    position_ids: torch.Tensor | None = None,
    block_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample one independent positive corruption time per row-local block.

    Segment-reset positions restart block numbering. Sampling one candidate per
    token and gathering from each block's first position keeps the operation
    vectorized while making the retained block times independent.
    """

    if len(token_shape) != 2:
        raise ValueError(f"BD3-LM token shape must be [batch, seq], got {tuple(token_shape)}")
    batch_size, seq_len = int(token_shape[0]), int(token_shape[1])
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("BD3-LM token shape entries must be positive")
    if block_size <= 0:
        raise ValueError("bd3lm block_size must be positive")
    candidate_times = sample_positive_uniform_tensor(
        (batch_size, seq_len),
        device=device,
        t_min=float(t_min),
        t_max=float(t_max),
    )
    return bd3lm_block_times_from_candidates(
        candidate_times,
        block_size=block_size,
        position_ids=position_ids,
        block_ids=block_ids,
    )


def bd3lm_block_times_from_candidates(
    candidate_times: torch.Tensor,
    *,
    block_size: int,
    position_ids: torch.Tensor | None = None,
    block_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand independent candidate draws to segment-local block times.

    Training and native evaluation share this transformation so a segment reset
    always starts a fresh block and only the candidate at that block's first
    position is retained.
    """

    if candidate_times.ndim != 2 or candidate_times.shape[0] <= 0 or candidate_times.shape[1] <= 0:
        raise ValueError(
            f"BD3-LM candidate_times must have positive shape [batch, seq], got {tuple(candidate_times.shape)}"
        )
    if not candidate_times.is_floating_point():
        raise TypeError("BD3-LM candidate_times must be floating point")
    if block_size <= 0:
        raise ValueError("bd3lm block_size must be positive")
    batch_size, seq_len = candidate_times.shape
    positions = _resolved_clean_position_ids(
        position_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        device=candidate_times.device,
    )
    token_indices = (
        torch.arange(seq_len, dtype=torch.long, device=candidate_times.device).view(1, seq_len).expand(batch_size, -1)
    )
    if block_ids is None:
        block_starts = positions.remainder(block_size) == 0
    else:
        ids = checked_block_ids(block_ids, positions)
        block_starts = positions == 0
        block_starts[:, 0] = True
        block_starts[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    owner_indices = torch.where(block_starts, token_indices, torch.zeros_like(token_indices)).cummax(dim=1).values
    return candidate_times.gather(1, owner_indices)


def bd3lm_training_flops_per_source_token(model) -> int:
    """Estimate dense ``x_t || x_0`` training FLOPs per original source token."""

    clean_seq_len = int(model.config.sequence_len)
    return 2 * int(model.estimate_flops(sequence_len=2 * clean_seq_len))


def bd3lm_concat_inputs(
    z_t: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    block_size: int,
    clean_stream_mode: str,
    position_ids: torch.Tensor | None = None,
    block_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bind doubled tokens, segment positions and one common block partition."""
    if z_t.shape != x_0.shape:
        raise ValueError(f"BD3LM requires z_t and x_0 to have the same shape, got {z_t.shape} and {x_0.shape}")
    concat = torch.cat((z_t, x_0), dim=1)
    mask = bd3lm_attention_mask(
        attention_mask,
        block_size,
        position_ids=position_ids,
        clean_stream_mode=clean_stream_mode,
        block_ids=block_ids,
    )
    concat_position_ids = bd3lm_concat_position_ids(
        position_ids,
        batch_size=z_t.shape[0],
        seq_len=z_t.shape[1],
        device=z_t.device,
    )
    return concat, mask, concat_position_ids
