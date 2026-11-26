"""Reference helpers for the absorbing-MDLM schedule-weighted objective.

The standard estimator multiplies the negative log-likelihood at every
corrupted target by ``1 / t``.  The shape of ``t`` is part of the objective:
a scalar or ``[batch]`` tensor represents one shared time per sequence, while
``[batch, seq]`` represents independently sampled token times.  These are not
generally interchangeable when predictions depend on the joint corruption
pattern.
"""

from __future__ import annotations

from numbers import Real

import torch


def expand_mdlm_time(
    t: Real | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Expand scalar, per-sequence, or per-token ``t`` to ``reference.shape``.

    ``reference`` must be a ``[batch, seq]`` tensor.  Accepted time shapes are
    scalar, ``[batch]``, and ``[batch, seq]``.  In particular, a ``[batch]``
    tensor is reshaped explicitly rather than relying on PyTorch broadcasting,
    which would otherwise align it with the sequence dimension.
    """

    if reference.ndim != 2:
        raise ValueError(f"reference must have shape [batch, seq], got {tuple(reference.shape)}")

    dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
    t_tensor = torch.as_tensor(t, device=reference.device, dtype=dtype)
    batch_size, seq_len = reference.shape

    if t_tensor.ndim == 0:
        return t_tensor.expand(batch_size, seq_len)
    if t_tensor.ndim == 1 and t_tensor.shape == (batch_size,):
        return t_tensor[:, None].expand(batch_size, seq_len)
    if t_tensor.ndim == 2 and t_tensor.shape == reference.shape:
        return t_tensor
    raise ValueError(
        "t must be a scalar, per-sequence tensor with shape "
        f"({batch_size},), or per-token tensor with shape {tuple(reference.shape)}; "
        f"got {tuple(t_tensor.shape)}"
    )


def schedule_weighted_mdlm_terms(
    token_nll: torch.Tensor,
    corruption_weight: torch.Tensor,
    t: Real | torch.Tensor,
    *,
    lossable_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-token ``corruption_weight * token_nll / t`` terms.

    ``lossable_mask`` excludes padding, prompt, or special-token positions.  A
    non-positive or non-finite time is rejected only when its corresponding
    objective weight is nonzero; this avoids undefined ``0 / 0`` terms without
    silently clipping an active sample's importance weight.
    """

    if token_nll.ndim != 2:
        raise ValueError(f"token_nll must have shape [batch, seq], got {tuple(token_nll.shape)}")
    if corruption_weight.shape != token_nll.shape:
        raise ValueError(
            "corruption_weight must match token_nll shape, got "
            f"{tuple(corruption_weight.shape)} and {tuple(token_nll.shape)}"
        )
    if lossable_mask is not None and lossable_mask.shape != token_nll.shape:
        raise ValueError(
            f"lossable_mask must match token_nll shape {tuple(token_nll.shape)}, got {tuple(lossable_mask.shape)}"
        )

    nll = token_nll if token_nll.is_floating_point() else token_nll.float()
    objective_weight = corruption_weight.to(device=nll.device, dtype=nll.dtype)
    if lossable_mask is not None:
        objective_weight = objective_weight * lossable_mask.to(device=nll.device, dtype=nll.dtype)
    if bool((objective_weight < 0).any().item()):
        raise ValueError("corruption_weight must be non-negative on lossable positions")

    t_expanded = expand_mdlm_time(t, nll)
    active = objective_weight != 0
    invalid_active_t = active & ((t_expanded <= 0) | ~torch.isfinite(t_expanded))
    if bool(invalid_active_t.any().item()):
        raise ValueError("t must be finite and strictly positive wherever the objective weight is nonzero")

    safe_t = torch.where(active, t_expanded, torch.ones_like(t_expanded))
    return nll * objective_weight / safe_t


def schedule_weighted_mdlm_objective(
    token_nll: torch.Tensor,
    corruption_weight: torch.Tensor,
    t: Real | torch.Tensor,
    *,
    lossable_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the sample estimator normalized by the lossable-token count."""

    normalizer = token_nll.numel() if lossable_mask is None else int(lossable_mask.to(dtype=torch.bool).sum().item())
    if normalizer <= 0:
        raise ValueError("the MDLM objective requires at least one lossable token")
    terms = schedule_weighted_mdlm_terms(
        token_nll,
        corruption_weight,
        t,
        lossable_mask=lossable_mask,
    )
    return terms.sum() / normalizer
