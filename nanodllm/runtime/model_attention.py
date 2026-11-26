"""Attention projections, masks, diagnostics, and projected-attention API."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ar_width_mup


class AttentionConfig(Protocol):
    n_embd: int
    n_head: int
    attention_scale: float
    parameterization_mode: str
    parameterization_attention_scale: float


def rms_norm(x: torch.Tensor, *, epsilon: float | None = None) -> torch.Tensor:
    if epsilon is not None:
        # Preserve FP64 reference arithmetic; BF16 activations use FP32
        # reductions and return to their original dtype before projections.
        value = x if x.dtype == torch.float64 else x.float()
        normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + epsilon)
        return normalized.to(dtype=x.dtype)
    return F.rms_norm(x, (x.size(-1),))


def precompute_rotary(seq_len: int, head_dim: int, base: float, dtype: torch.dtype, device=None):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype)[:, None, :], freqs.sin().to(dtype)[:, None, :]


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((x1 * cos + x2 * sin, -x1 * sin + x2 * cos), dim=-1)


class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


class BidirectionalAttention(nn.Module):
    def __init__(self, config: AttentionConfig, local_window: int | None = None, causal: bool = False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        configured_scale = (
            float(config.parameterization_attention_scale)
            if config.parameterization_mode != "ordinary"
            else float(config.attention_scale)
        )
        self.attention_scale = None if configured_scale == -1.0 else configured_scale
        self.local_window = local_window
        self.causal = causal
        self.ar_width_scope = ar_width_mup.runtime_active(config)
        self.norm_epsilon = ar_width_mup.NORM_EPSILON if self.ar_width_scope else None
        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

    def _build_attention_mask(
        self,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        attention_mask: torch.Tensor | None,
        query_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Resolve model masks, optionally for only a subset of query rows."""
        key_positions = torch.arange(seq_len, device=device)
        query_positions = key_positions if query_indices is None else query_indices
        query_count = int(query_positions.numel())

        attn_mask = None
        if self.causal:
            attn_mask = query_positions[:, None] >= key_positions[None, :]
            attn_mask = attn_mask.view(1, 1, query_count, seq_len)
        elif self.local_window is not None:
            attn_mask = (query_positions[:, None] - key_positions[None, :]).abs() <= self.local_window
            attn_mask = attn_mask.view(1, 1, query_count, seq_len)

        if attention_mask is None:
            return attn_mask

        if attention_mask.dim() == 2 and attention_mask.shape == (batch_size, seq_len):
            external_mask = attention_mask[:, None, None, :].to(device=device, dtype=torch.bool)
        elif attention_mask.dim() == 2 and attention_mask.shape == (seq_len, seq_len):
            external_mask = attention_mask.to(device=device, dtype=torch.bool)
            if query_indices is not None:
                external_mask = external_mask.index_select(0, query_indices)
            external_mask = external_mask[None, None, :, :]
        elif attention_mask.dim() == 3 and attention_mask.shape[-2:] == (seq_len, seq_len):
            external_mask = attention_mask.to(device=device, dtype=torch.bool)
            if query_indices is not None:
                external_mask = external_mask.index_select(1, query_indices)
            external_mask = external_mask[:, None, :, :]
        elif attention_mask.dim() == 4 and attention_mask.shape[-2:] == (seq_len, seq_len):
            external_mask = attention_mask.to(device=device, dtype=torch.bool)
            if query_indices is not None:
                external_mask = external_mask.index_select(2, query_indices)
        else:
            raise ValueError(
                "attention_mask must be [batch, seq], [seq, seq], [batch, seq, seq], "
                f"or [batch, heads, seq, seq]; got {tuple(attention_mask.shape)} "
                f"for batch={batch_size}, seq={seq_len}"
            )
        return external_mask if attn_mask is None else (attn_mask & external_mask)

    @torch.no_grad()
    def sampled_attention_diagnostics(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        query_indices: torch.Tensor | list[int] | tuple[int, ...],
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, float | int | list[int]]:
        """Measure exact attention distributions for selected query positions.

        Q/K projections, post-RoPE RMSNorm, scale, and mask semantics match
        :meth:`forward`. Only ``len(query_indices) * seq_len`` attention logits
        are materialized per batch/head rather than the full square matrix.
        Entropy and KL divergence use natural logarithms, with the uniform
        reference restricted to keys visible from each selected query.
        """
        if x.dim() != 3:
            raise ValueError(f"x must have shape [batch, seq, embedding], got {tuple(x.shape)}")
        batch_size, seq_len, channels = x.shape
        if channels != self.n_head * self.head_dim:
            raise ValueError(f"x embedding dimension must be {self.n_head * self.head_dim}, got {channels}")

        selected_queries = torch.as_tensor(query_indices, dtype=torch.long, device=x.device)
        if selected_queries.dim() != 1 or selected_queries.numel() == 0:
            raise ValueError("query_indices must be a non-empty one-dimensional sequence")
        if bool(((selected_queries < 0) | (selected_queries >= seq_len)).any()):
            raise ValueError(f"query_indices must be in [0, {seq_len})")
        if int(torch.unique(selected_queries).numel()) != int(selected_queries.numel()):
            raise ValueError("query_indices must not contain duplicates")

        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        q, k = rms_norm(q, epsilon=self.norm_epsilon), rms_norm(k, epsilon=self.norm_epsilon)
        q = q.index_select(1, selected_queries).transpose(1, 2).float()
        k = k.transpose(1, 2).float()

        scale = self.head_dim**-0.5 if self.attention_scale is None else self.attention_scale
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        query_count = int(selected_queries.numel())
        resolved_mask = self._build_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
            attention_mask=attention_mask,
            query_indices=selected_queries,
        )
        if resolved_mask is None:
            visible = torch.ones(
                (batch_size, self.n_head, query_count, seq_len),
                dtype=torch.bool,
                device=x.device,
            )
        else:
            try:
                visible = resolved_mask.expand(batch_size, self.n_head, query_count, seq_len)
            except RuntimeError as exc:
                raise ValueError(
                    "attention_mask batch/head dimensions are not broadcastable to "
                    f"[{batch_size}, {self.n_head}, {query_count}, {seq_len}]"
                ) from exc

        visible_key_count = visible.sum(dim=-1)
        valid_rows = visible_key_count > 0
        valid_row_count = int(valid_rows.sum().item())
        if valid_row_count == 0:
            raise ValueError("selected queries have no visible keys")

        masked_logits = logits.masked_fill(~visible, float("-inf"))
        safe_logits = torch.where(valid_rows[..., None], masked_logits, torch.zeros_like(masked_logits))
        probabilities = torch.softmax(safe_logits, dim=-1)
        probabilities = torch.where(visible & valid_rows[..., None], probabilities, 0.0)

        log_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        max_probability = probabilities.max(dim=-1).values
        kl_from_uniform = (visible_key_count.float().log() - entropy).clamp_min(0.0)

        q_rms = q.square().mean(dim=-1).sqrt()
        k_rms = k.square().mean(dim=-1).sqrt()
        visible_keys = visible.any(dim=2)

        def valid_mean(values: torch.Tensor) -> float:
            return float(values.masked_select(valid_rows).mean().item())

        return {
            "selected_query_indices": [int(index) for index in selected_queries.cpu().tolist()],
            "selected_query_count": query_count,
            "valid_attention_row_count": valid_row_count,
            "all_masked_attention_row_count": int(valid_rows.numel()) - valid_row_count,
            "resolved_attention_scale": float(scale),
            "q_rms_mean": valid_mean(q_rms),
            "k_rms_mean": float(k_rms.masked_select(visible_keys).mean().item()),
            "visible_key_count_mean": valid_mean(visible_key_count.float()),
            "visible_key_count_min": int(visible_key_count.masked_select(valid_rows).min().item()),
            "visible_key_count_max": int(visible_key_count.masked_select(valid_rows).max().item()),
            "attention_entropy_mean": valid_mean(entropy),
            "attention_max_probability_mean": valid_mean(max_probability),
            "attention_kl_from_uniform_mean": valid_mean(kl_from_uniform),
        }

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        batch_size, seq_len, channels = x.shape
        q, k, v = self.project_qkv(x, cos, sin)
        attn_mask = self._build_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
            attention_mask=attention_mask,
        )
        return self.attend_projected(q, k, v, attention_mask=attn_mask).reshape(batch_size, seq_len, channels)

    def project_qkv(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states into RoPE/QK-normalized attention tensors.

        Returned tensors use ``[batch, heads, sequence, head_dim]``. The BD3
        completed-prefix cache stores only K/V results from this boundary;
        active-block projections are never retained.
        """

        if x.dim() != 3:
            raise ValueError(f"attention input must have shape [batch, seq, embedding], got {tuple(x.shape)}")
        batch_size, seq_len, channels = x.shape
        expected_channels = self.n_head * self.head_dim
        if channels != expected_channels:
            raise ValueError(f"attention embedding dimension must be {expected_channels}, got {channels}")
        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.c_v(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        q, k = rms_norm(q, epsilon=self.norm_epsilon), rms_norm(k, epsilon=self.norm_epsilon)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def attend_projected(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over already projected Q/K/V tensors and apply ``c_proj``."""

        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError("projected attention tensors must have shape [batch, heads, sequence, head_dim]")
        if k.shape != v.shape:
            raise ValueError(f"projected attention K/V shapes must match, got {tuple(k.shape)} and {tuple(v.shape)}")
        if q.shape[:2] != k.shape[:2] or q.shape[3] != k.shape[3]:
            raise ValueError("projected attention Q/K batch, head, and head-dimension shapes must match")
        batch_size, _, query_len, _ = q.shape
        if self.ar_width_scope and q.is_cuda:
            ar_width_mup.assert_backend()
        context = (
            torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if self.ar_width_scope else nullcontext()
        )
        with context:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                is_causal=False,
                scale=self.attention_scale,
            )
        channels = self.n_head * self.head_dim
        return self.c_proj(y.transpose(1, 2).contiguous().view(batch_size, query_len, channels))
