"""Collective parity for empty training batches and rank-zero probes."""

from contextlib import contextmanager

import torch
import torch.distributed as dist


def _graph_connected_zero(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Return an exact zero whose autograd graph reaches every input tensor."""

    if not tensors:
        raise ValueError("graph-connected zero requires at least one tensor")
    zero = tensors[0].reshape(-1)[:0].sum()
    for tensor in tensors[1:]:
        zero = zero + tensor.reshape(-1)[:0].sum()
    return zero


def _empty_target_model_output_zero(
    model_out,
    *,
    use_hierarchical_output: bool,
    return_aux_logits: bool,
    has_length_output: bool,
    include_length_loss: bool,
) -> torch.Tensor:
    """Mirror the configured loss heads while assigning them zero gradient mass."""

    length_logits = None
    if use_hierarchical_output:
        if return_aux_logits and has_length_output:
            class_logits, token_logits, aux_logits, length_logits = model_out
        elif return_aux_logits:
            class_logits, token_logits, aux_logits = model_out
        elif has_length_output:
            class_logits, token_logits, length_logits = model_out
            aux_logits = []
        else:
            class_logits, token_logits = model_out
            aux_logits = []
        loss_tensors = [class_logits, token_logits, *aux_logits]
    else:
        if return_aux_logits and has_length_output:
            logits, aux_logits, length_logits = model_out
        elif return_aux_logits:
            logits, aux_logits = model_out
        elif has_length_output:
            logits, length_logits = model_out
            aux_logits = []
        else:
            logits, aux_logits = model_out, []
        loss_tensors = [logits, *aux_logits]
    if include_length_loss:
        if length_logits is None:
            raise RuntimeError("length-head loss is enabled but the empty-target forward returned no length logits")
        loss_tensors.append(length_logits)
    return _graph_connected_zero(tuple(loss_tensors))


@contextmanager
def synchronize_probe_failure(distributed, world_size, rank):
    """Every rank leaves finalization with the same success or failure."""
    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    if distributed:
        reports = [None] * world_size
        local = f"rank {rank}: {type(error).__name__}: {error}" if error is not None else ""
        dist.all_gather_object(reports, local)
        if any(reports):
            raise RuntimeError("final LM-head probe failed: " + "; ".join(filter(None, reports))) from error
    if error is not None:
        raise error
