"""Validated categorical temperature, top-k and nucleus filtering."""

import math

import torch
import torch.nn.functional as F


def sample_filtered_multinomial(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per row; an explicit generator leaves global RNG unchanged."""
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a number")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise ValueError("top_p must be a number")
    if not math.isfinite(float(top_p)) or top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be in (0, 1]")
    if logits.dim() != 2:
        raise ValueError(f"filtered multinomial logits must be 2D, got shape {tuple(logits.shape)}")
    filtered_logits = logits.float() / float(temperature)
    vocab_size = int(filtered_logits.shape[-1])
    if top_k > 0 and top_k < vocab_size:
        top_values, top_indices = torch.topk(filtered_logits, k=int(top_k), dim=-1)
        top_k_logits = torch.full_like(filtered_logits, -torch.inf)
        filtered_logits = top_k_logits.scatter(-1, top_indices, top_values)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        nucleus_logits = torch.full_like(filtered_logits, -torch.inf)
        filtered_logits = nucleus_logits.scatter(-1, sorted_indices, sorted_logits)
    probs = F.softmax(filtered_logits, dim=-1)
    if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
        raise RuntimeError("top-k/top-p filtering produced an invalid sampling distribution")
    kwargs = {} if generator is None else {"generator": generator}
    return torch.multinomial(probs, 1, **kwargs).squeeze(-1)
