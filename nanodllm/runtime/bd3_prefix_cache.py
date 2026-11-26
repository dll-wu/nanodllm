"""Topology-equivalent completed-prefix KV reuse for BD3-LM inference.

The cache contains only clean states from finalized prefix blocks. Active-block
states are recomputed on every denoising call and are never written here. This
preserves the dependency graph, not the legacy full-shape finite-precision
trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import DiffusionTransformer, rms_norm
from .model_profiles import NANOLLADA_PROFILE


@dataclass(frozen=True, slots=True)
class CacheExecutionStats:
    """One cache-operation accounting record."""

    model_calls: int
    processed_query_tokens: int


class CompletedPrefixKVCache:
    """Preallocated per-layer K/V storage for clean finalized prefix tokens."""

    __slots__ = (
        "_batch_size",
        "_block_size",
        "_capacity",
        "_device",
        "_dtype",
        "_keys",
        "_length",
        "_model_identity",
        "_token_ids",
        "_values",
    )

    def __init__(
        self,
        *,
        model: DiffusionTransformer,
        batch_size: int,
        capacity: int,
        block_size: int,
    ) -> None:
        _validate_cache_model(model, block_size=block_size)
        if int(batch_size) <= 0:
            raise ValueError("completed-prefix cache batch_size must be positive")
        if int(capacity) <= 0:
            raise ValueError("completed-prefix cache capacity must be positive")
        if int(capacity) > int(model.config.sequence_len):
            raise ValueError(
                f"completed-prefix cache capacity {int(capacity)} exceeds model sequence capacity "
                f"{int(model.config.sequence_len)}"
            )
        device = model.wte.weight.device
        dtype = model.compute_dtype
        layer_shape = (
            int(batch_size),
            int(model.config.n_head),
            int(capacity),
            int(model.config.n_embd) // int(model.config.n_head),
        )
        self._keys = tuple(torch.empty(layer_shape, dtype=dtype, device=device) for _ in model.blocks)
        self._values = tuple(torch.empty(layer_shape, dtype=dtype, device=device) for _ in model.blocks)
        self._token_ids = torch.full(
            (int(batch_size), int(capacity)),
            -1,
            dtype=torch.long,
            device=device,
        )
        self._batch_size = int(batch_size)
        self._capacity = int(capacity)
        self._block_size = int(block_size)
        self._dtype = dtype
        self._device = device
        self._model_identity = id(model)
        self._length = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def length(self) -> int:
        return self._length

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def prefix_tokens(self) -> torch.Tensor:
        """Return a detached copy so callers cannot mutate cache provenance."""

        return self._token_ids[:, : self._length].detach().clone()

    @property
    def allocated_kv_bytes(self) -> int:
        """Return allocated K/V tensor bytes, excluding the token ledger."""

        return sum(tensor.numel() * tensor.element_size() for tensor in (*self._keys, *self._values))

    @property
    def live_kv_bytes(self) -> int:
        """Return bytes in the logically live prefix slices."""

        if self._capacity == 0:
            return 0
        return self.allocated_kv_bytes * self._length // self._capacity

    def reset(self) -> None:
        """Return the cache to its empty lifecycle state without reallocating."""

        self._length = 0
        self._token_ids.fill_(-1)

    def _validate_for(self, model: DiffusionTransformer) -> None:
        if id(model) != self._model_identity:
            raise ValueError("completed-prefix cache belongs to a different model instance")
        _validate_cache_model(model, block_size=self._block_size)
        if model.wte.weight.device != self._device:
            raise ValueError("completed-prefix cache device no longer matches the model device")
        if model.compute_dtype != self._dtype:
            raise ValueError("completed-prefix cache dtype no longer matches the model compute dtype")
        if len(self._keys) != len(model.blocks) or len(self._values) != len(model.blocks):
            raise ValueError("completed-prefix cache layer count no longer matches the model")
        expected_shape = (
            self._batch_size,
            int(model.config.n_head),
            self._capacity,
            int(model.config.n_embd) // int(model.config.n_head),
        )
        for layer_idx, (key, value) in enumerate(zip(self._keys, self._values, strict=True)):
            if key.shape != expected_shape or value.shape != expected_shape:
                raise ValueError(f"completed-prefix cache layer {layer_idx} has an invalid K/V shape")
            if key.dtype != self._dtype or value.dtype != self._dtype:
                raise ValueError(f"completed-prefix cache layer {layer_idx} has an invalid K/V dtype")
            if key.device != self._device or value.device != self._device:
                raise ValueError(f"completed-prefix cache layer {layer_idx} has an invalid K/V device")
        if self._token_ids.shape != (self._batch_size, self._capacity):
            raise ValueError("completed-prefix cache token ledger has an invalid shape")
        if self._token_ids.dtype != torch.long or self._token_ids.device != self._device:
            raise ValueError("completed-prefix cache token ledger has an invalid dtype or device")
        if self._length < 0 or self._length > self._capacity:
            raise ValueError("completed-prefix cache length is outside its capacity")


def _validate_cache_model(model: DiffusionTransformer, *, block_size: int) -> None:
    if not isinstance(model, DiffusionTransformer):
        raise TypeError("completed-prefix caching requires an unwrapped DiffusionTransformer")
    if model.nanochat_extras is not None:
        raise ValueError("nanochat_block_v1 requires the causal-prefix cache, not the legacy blockwise cache")
    if model.training:
        raise ValueError("completed-prefix caching requires model.eval()")
    if int(block_size) <= 0:
        raise ValueError("completed-prefix cache block_size must be positive")
    trained_block_size = int(model.config.bd3lm_block_size)
    if trained_block_size <= 0:
        raise ValueError("completed-prefix caching requires a BD3-LM model configuration")
    if int(block_size) != trained_block_size:
        raise ValueError(
            f"completed-prefix cache block_size {int(block_size)} does not match trained block size "
            f"{trained_block_size}"
        )
    if bool(model.config.causal):
        raise ValueError("completed-prefix caching does not support an additional causal model mask")
    if int(model.config.local_attention_lower_layers) or int(model.config.local_attention_window):
        raise ValueError("completed-prefix caching does not support local-attention layers")
    if model.noise_mlp is not None:
        raise ValueError("completed-prefix caching requires prefix-stable models without global time conditioning")
    if model.target_length_embedding is not None or model.target_distance_proj is not None:
        raise ValueError("completed-prefix caching does not support target-length conditioning")


def new_completed_prefix_cache(
    model: DiffusionTransformer,
    *,
    batch_size: int,
    capacity: int,
    block_size: int,
) -> CompletedPrefixKVCache:
    """Allocate an empty topology-equivalent cache bound to one model instance."""

    return CompletedPrefixKVCache(
        model=model,
        batch_size=int(batch_size),
        capacity=int(capacity),
        block_size=int(block_size),
    )


def _validate_token_chunk(
    model: DiffusionTransformer,
    cache: CompletedPrefixKVCache,
    token_ids: torch.Tensor,
    *,
    operation: str,
    validate_token_domain: bool,
) -> tuple[int, int]:
    cache._validate_for(model)
    if token_ids.dim() != 2:
        raise ValueError(f"{operation} tokens must have shape [batch, sequence], got {tuple(token_ids.shape)}")
    batch_size, token_count = token_ids.shape
    if int(batch_size) != cache.batch_size:
        raise ValueError(f"{operation} batch size {int(batch_size)} does not match cache batch {cache.batch_size}")
    if int(token_count) <= 0:
        raise ValueError(f"{operation} requires at least one token")
    if token_ids.dtype != torch.long:
        raise TypeError(f"{operation} tokens must use torch.long dtype")
    if token_ids.device != cache.device:
        raise ValueError(f"{operation} token device {token_ids.device} does not match cache device {cache.device}")
    if cache.length % cache.block_size != 0:
        raise ValueError(f"{operation} requires the cached prefix to end on a block boundary")
    if int(token_count) > cache.block_size:
        raise ValueError(f"{operation} cannot process more than one block")
    if cache.length + int(token_count) > cache.capacity:
        raise ValueError(f"{operation} would exceed completed-prefix cache capacity {cache.capacity}")
    if int(token_count) < cache.block_size and cache.length + int(token_count) != cache.capacity:
        raise ValueError(f"{operation} partial block must exactly fill terminal cache capacity")
    if (
        validate_token_domain
        and token_ids.numel()
        and (int(token_ids.min().item()) < 0 or int(token_ids.max().item()) >= int(model.config.vocab_size))
    ):
        raise ValueError(f"{operation} token IDs must be in the model vocabulary")
    return int(batch_size), int(token_count)


def _rope_for_positions(
    model: DiffusionTransformer,
    *,
    start: int,
    token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(start, start + token_count, dtype=torch.long, device=model.wte.weight.device)
    return model.rope_cos.index_select(0, positions), model.rope_sin.index_select(0, positions)


def _input_hidden(model: DiffusionTransformer, token_ids: torch.Tensor, *, start: int) -> torch.Tensor:
    hidden = model.wte(token_ids).to(dtype=model.compute_dtype)
    if model.config.backbone_profile == NANOLLADA_PROFILE:
        hidden = rms_norm(hidden)
    if model.mask_position_embedding is not None:
        token_count = int(token_ids.shape[1])
        positions = torch.arange(start, start + token_count, dtype=torch.long, device=token_ids.device)
        position_values = model.mask_position_embedding(positions).to(dtype=model.compute_dtype)[None, :, :]
        mask_selector = (token_ids == int(model.config.mask_token_id)).unsqueeze(-1).to(dtype=model.compute_dtype)
        hidden = hidden + position_values * mask_selector
    return hidden


@torch.no_grad()
def append_completed_block(
    model: DiffusionTransformer,
    cache: CompletedPrefixKVCache,
    token_ids: torch.Tensor,
) -> CacheExecutionStats:
    """Compute and atomically append one clean finalized prefix block."""

    return _append_completed_block(model, cache, token_ids, validate_token_domain=True)


@torch.no_grad()
def _append_completed_block_trusted(
    model: DiffusionTransformer,
    cache: CompletedPrefixKVCache,
    token_ids: torch.Tensor,
) -> CacheExecutionStats:
    """Sampler-only append after its canvas boundary validated token IDs."""

    return _append_completed_block(model, cache, token_ids, validate_token_domain=False)


def _append_completed_block(
    model: DiffusionTransformer,
    cache: CompletedPrefixKVCache,
    token_ids: torch.Tensor,
    *,
    validate_token_domain: bool,
) -> CacheExecutionStats:

    batch_size, token_count = _validate_token_chunk(
        model,
        cache,
        token_ids,
        operation="completed-prefix append",
        validate_token_domain=validate_token_domain,
    )
    if bool((token_ids == int(model.config.mask_token_id)).any()):
        raise ValueError("completed-prefix append tokens must be finalized and cannot contain MASK")

    start = cache.length
    cos, sin = _rope_for_positions(model, start=start, token_count=token_count)
    hidden = _input_hidden(model, token_ids, start=start)
    pending_keys: list[torch.Tensor] = []
    pending_values: list[torch.Tensor] = []
    for layer_idx, block in enumerate(model.blocks):
        normalized = rms_norm(hidden)
        q, current_k, current_v = block.attn.project_qkv(normalized, cos, sin)
        prefix_k = cache._keys[layer_idx][:, :, :start, :]
        prefix_v = cache._values[layer_idx][:, :, :start, :]
        keys = torch.cat((prefix_k, current_k), dim=2) if start else current_k
        values = torch.cat((prefix_v, current_v), dim=2) if start else current_v
        attention_output = block.attn.attend_projected(q, keys, values)
        hidden = hidden + (
            attention_output if block.residual_multiplier == 1.0 else block.residual_multiplier * attention_output
        )
        mlp_output = block.mlp(rms_norm(hidden))
        hidden = hidden + (mlp_output if block.residual_multiplier == 1.0 else block.residual_multiplier * mlp_output)
        pending_keys.append(current_k)
        pending_values.append(current_v)

    end = start + token_count
    for layer_idx, (current_k, current_v) in enumerate(zip(pending_keys, pending_values, strict=True)):
        cache._keys[layer_idx][:, :, start:end, :].copy_(current_k)
        cache._values[layer_idx][:, :, start:end, :].copy_(current_v)
    cache._token_ids[:, start:end].copy_(token_ids)
    cache._length = end
    return CacheExecutionStats(model_calls=1, processed_query_tokens=batch_size * token_count)


@torch.no_grad()
def prefill_completed_prefix(
    model: DiffusionTransformer,
    cache: CompletedPrefixKVCache,
    prefix_tokens: torch.Tensor,
) -> CacheExecutionStats:
    """Populate an empty cache by clean blockwise execution of a prefix."""

    cache._validate_for(model)
    if cache.length != 0:
        raise ValueError("completed-prefix prefill requires an empty cache")
    if prefix_tokens.dim() != 2:
        raise ValueError(f"prefix tokens must have shape [batch, sequence], got {tuple(prefix_tokens.shape)}")
    if int(prefix_tokens.shape[0]) != cache.batch_size:
        raise ValueError("prefix token batch size does not match cache batch")
    prefix_len = int(prefix_tokens.shape[1])
    if prefix_len == 0:
        if prefix_tokens.dtype != torch.long:
            raise TypeError("prefix tokens must use torch.long dtype")
        if prefix_tokens.device != cache.device:
            raise ValueError("prefix token device does not match cache device")
        return CacheExecutionStats(model_calls=0, processed_query_tokens=0)
    if prefix_len % cache.block_size != 0:
        raise ValueError("completed-prefix prefill length must be block-aligned")
    if prefix_len > cache.capacity:
        raise ValueError("completed-prefix prefill exceeds cache capacity")
    if prefix_tokens.dtype != torch.long:
        raise TypeError("prefix tokens must use torch.long dtype")
    if prefix_tokens.device != cache.device:
        raise ValueError("prefix token device does not match cache device")
    if prefix_tokens.numel() and (
        int(prefix_tokens.min().item()) < 0 or int(prefix_tokens.max().item()) >= int(model.config.vocab_size)
    ):
        raise ValueError("prefix token IDs must be in the model vocabulary")
    if bool((prefix_tokens == int(model.config.mask_token_id)).any()):
        raise ValueError("completed-prefix prefill tokens must be finalized and cannot contain MASK")

    model_calls = 0
    processed_query_tokens = 0
    try:
        for block_start in range(0, prefix_len, cache.block_size):
            stats = _append_completed_block_trusted(
                model,
                cache,
                prefix_tokens[:, block_start : block_start + cache.block_size],
            )
            model_calls += stats.model_calls
            processed_query_tokens += stats.processed_query_tokens
    except Exception:
        cache.reset()
        raise
    return CacheExecutionStats(model_calls=model_calls, processed_query_tokens=processed_query_tokens)


@torch.no_grad()
def forward_active_block(
    model: DiffusionTransformer,
    cache: CompletedPrefixKVCache,
    active_tokens: torch.Tensor,
) -> tuple[torch.Tensor, CacheExecutionStats]:
    """Score one active block without mutating or retaining its hidden state."""

    return _forward_active_block(model, cache, active_tokens, validate_token_domain=True)


@torch.no_grad()
def _forward_active_block_trusted(
    model: DiffusionTransformer,
    cache: CompletedPrefixKVCache,
    active_tokens: torch.Tensor,
) -> tuple[torch.Tensor, CacheExecutionStats]:
    """Sampler-only forward after its canvas boundary validated token IDs."""

    return _forward_active_block(model, cache, active_tokens, validate_token_domain=False)


def _forward_active_block(
    model: DiffusionTransformer,
    cache: CompletedPrefixKVCache,
    active_tokens: torch.Tensor,
    *,
    validate_token_domain: bool,
) -> tuple[torch.Tensor, CacheExecutionStats]:

    batch_size, token_count = _validate_token_chunk(
        model,
        cache,
        active_tokens,
        operation="active-block forward",
        validate_token_domain=validate_token_domain,
    )
    start = cache.length
    cos, sin = _rope_for_positions(model, start=start, token_count=token_count)
    hidden = _input_hidden(model, active_tokens, start=start)
    for layer_idx, block in enumerate(model.blocks):
        normalized = rms_norm(hidden)
        q, active_k, active_v = block.attn.project_qkv(normalized, cos, sin)
        prefix_k = cache._keys[layer_idx][:, :, :start, :]
        prefix_v = cache._values[layer_idx][:, :, :start, :]
        keys = torch.cat((active_k, prefix_k), dim=2) if start else active_k
        values = torch.cat((active_v, prefix_v), dim=2) if start else active_v
        attention_output = block.attn.attend_projected(q, keys, values)
        hidden = hidden + (
            attention_output if block.residual_multiplier == 1.0 else block.residual_multiplier * attention_output
        )
        mlp_output = block.mlp(rms_norm(hidden))
        hidden = hidden + (mlp_output if block.residual_multiplier == 1.0 else block.residual_multiplier * mlp_output)

    hidden = model.final_norm(hidden)
    token_logits = model.lm_head(hidden)[..., : model.config.vocab_size]
    if model.config.parameterization_output_multiplier != 1.0:
        token_logits = token_logits * model.config.parameterization_output_multiplier
    token_logits = token_logits.float()
    if model.uses_hierarchical_output():
        if model.class_head is None:
            raise RuntimeError("hierarchical output is enabled but class_head is missing")
        class_logits = model.class_head(hidden).float()
        logits = model.hierarchical_token_log_probs(class_logits, token_logits)
    else:
        logits = token_logits
    return logits, CacheExecutionStats(model_calls=1, processed_query_tokens=batch_size * token_count)
