"""Clean causal-prefix KV reuse for variable-length block denoising.

Unlike the legacy BD3 cache, completed tokens are strictly causal, including
within an appended chunk. Chunk boundaries are therefore not model state.
"""

from __future__ import annotations

import torch

from .model import DiffusionTransformer
from .model_attention import rms_norm
from .model_profiles import NANOLLADA_PROFILE


def _validate_model(model: DiffusionTransformer) -> None:
    if not isinstance(model, DiffusionTransformer):
        raise TypeError("causal-prefix caching requires an unwrapped DiffusionTransformer")
    if model.training or any(block.training for block in model.blocks):
        raise ValueError("causal-prefix caching requires model.eval()")
    if model.recurrence.enabled:
        raise ValueError("causal-prefix caching is not qualified for recurrent block execution")
    if model.config.causal or any(block.attn.causal or block.attn.local_window is not None for block in model.blocks):
        raise ValueError("causal-prefix caching requires non-AR, full-attention blocks")
    if (
        model.noise_mlp is not None
        or model.target_length_embedding is not None
        or model.target_distance_proj is not None
        or model.block_self_conditioning is not None
    ):
        raise ValueError("causal-prefix caching requires prefix-stable inputs without time/length/self conditioning")
    if model.uses_hierarchical_output():
        raise ValueError("causal-prefix caching currently supports flat token outputs only")


class CausalPrefixKVCache:
    """Request-local cache for immutable eval-mode model weights.

    ``append`` recomputes finalized tokens causally; ``logits`` recomputes the
    entire bidirectional active block without changing any cached state. One
    rectangular batch shares a prefix length and absolute RoPE coordinates.
    Create a new cache after model/weight/dtype/device changes. No padding,
    context eviction, past-token edits, or legacy blockwise-prefix semantics.
    """

    def __init__(self, model: DiffusionTransformer, *, batch_size: int, capacity: int):
        _validate_model(model)
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("cache batch_size must be a positive integer")
        if type(capacity) is not int or not 0 < capacity <= model.config.sequence_len:
            raise ValueError("cache capacity must be a positive integer within the trained context")
        self._model = model
        self._batch_size = batch_size
        self._capacity = capacity
        self._device = model.wte.weight.device
        self._dtype = model.compute_dtype
        self._length = 0
        shape = (batch_size, model.config.n_head, capacity, model.config.n_embd // model.config.n_head)
        self._keys = tuple(torch.empty(shape, device=self._device, dtype=self._dtype) for _ in model.blocks)
        self._values = tuple(torch.empty_like(key) for key in self._keys)
        self._tokens = torch.full((batch_size, capacity), -1, device=self._device, dtype=torch.long)

    @property
    def length(self) -> int:
        return self._length

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def prefix_tokens(self) -> torch.Tensor:
        return self._tokens[:, : self._length].clone()

    @property
    def allocated_kv_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self._keys, *self._values))

    @property
    def live_kv_bytes(self) -> int:
        return self.allocated_kv_bytes * self.length // self.capacity

    def reset(self) -> None:
        """Forget the prefix; unused storage is never visible to attention."""
        self._length = 0

    def _validate_tokens(self, tokens: torch.Tensor, *, clean: bool) -> None:
        _validate_model(self._model)
        if self._model.wte.weight.device != self._device or self._model.compute_dtype != self._dtype:
            raise ValueError("cache model device/dtype changed; create a new cache")
        if tokens.ndim != 2 or tokens.shape[0] != self._batch_size or tokens.shape[1] == 0:
            raise ValueError("cache tokens must be a nonempty [batch, sequence] with the cache batch size")
        if tokens.dtype != torch.long or tokens.device != self._device:
            raise ValueError("cache tokens must use torch.long on the model device")
        if self.length + tokens.shape[1] > self.capacity:
            raise ValueError("cache operation exceeds context capacity")
        if bool(((tokens < 0) | (tokens >= self._model.config.vocab_size)).any()):
            raise ValueError("cache tokens must be in the model vocabulary")
        if clean and bool(tokens.eq(self._model.config.mask_token_id).any()):
            raise ValueError("clean causal-prefix append cannot contain MASK")

    def _forward(self, tokens: torch.Tensor, *, causal: bool):
        model = self._model
        if model.nanochat_extras is not None:
            from .nanochat_backbone_cache import forward

            return forward(self, tokens, causal=causal)
        start, width = self.length, tokens.shape[1]
        positions = torch.arange(start, start + width, device=self._device)
        cos, sin = model.rope_cos[start : start + width], model.rope_sin[start : start + width]
        hidden = model.wte(tokens).to(dtype=self._dtype)
        if model.config.backbone_profile == NANOLLADA_PROFILE:
            hidden = rms_norm(hidden)
        if model.mask_position_embedding is not None:
            values = model.mask_position_embedding(positions).to(dtype=self._dtype)
            hidden = hidden + values[None] * tokens.eq(model.config.mask_token_id).unsqueeze(-1).to(self._dtype)
        # Offset causal mask: each new clean query can see all prior cached
        # tokens and only its own/past chunk tokens. is_causal=True alone would
        # use the wrong alignment for a non-square query/key matrix.
        attention_mask = None
        if causal:
            attention_mask = positions[:, None] >= torch.arange(start + width, device=self._device)[None, :]
        pending = []
        for index, block in enumerate(model.blocks):
            query, key, value = block.attn.project_qkv(rms_norm(hidden), cos, sin)
            keys = torch.cat((self._keys[index][:, :, :start], key), dim=2) if start else key
            values = torch.cat((self._values[index][:, :, :start], value), dim=2) if start else value
            update = block.attn.attend_projected(query, keys, values, attention_mask=attention_mask)
            multiplier = block.residual_multiplier
            hidden = hidden + (update if multiplier == 1.0 else multiplier * update)
            update = block.mlp(rms_norm(hidden))
            hidden = hidden + (update if multiplier == 1.0 else multiplier * update)
            if causal:
                pending.append((key, value))
        return hidden, pending

    @torch.inference_mode()
    def append(self, tokens: torch.Tensor) -> None:
        """Prefill or atomically append any nonempty clean chunk, without logits."""
        self._validate_tokens(tokens, clean=True)
        hidden, pending = self._forward(tokens, causal=True)
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError("nonfinite clean-prefix hidden states")
        end = self.length + tokens.shape[1]
        for index, (key, value) in enumerate(pending):
            self._keys[index][:, :, self.length : end].copy_(key)
            self._values[index][:, :, self.length : end].copy_(value)
        self._tokens[:, self.length : end].copy_(tokens)
        self._length = end

    @torch.inference_mode()
    def logits(self, active_tokens: torch.Tensor) -> torch.Tensor:
        """Score only active positions; prefix and active states stay separate."""
        self._validate_tokens(active_tokens, clean=False)
        hidden, _ = self._forward(active_tokens, causal=False)
        model = self._model
        logits = model.lm_head(model.final_norm(hidden))[..., : model.config.vocab_size]
        multiplier = float(model.config.parameterization_output_multiplier)
        if multiplier != 1.0:
            logits = logits * multiplier
        logits = logits.float()
        if model.nanochat_extras is not None:
            from .nanochat_backbone import softcap

            logits = softcap(logits)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite block-generation logits")
        return logits
