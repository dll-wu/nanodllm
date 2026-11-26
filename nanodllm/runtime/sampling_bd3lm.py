"""Block-diffusion sampling over unconditional or prefix-conditioned canvases."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TypeVar

import torch
import torch.nn.functional as F

from .bd3_prefix_cache import (
    _append_completed_block_trusted,
    _forward_active_block_trusted,
    new_completed_prefix_cache,
    prefill_completed_prefix,
)
from .model import bd3lm_attention_mask, duplicated_position_ids
from .noise import MDLMNoise

SamplerTrace = dict[str, float | int]
SamplerOutput = torch.Tensor | tuple[torch.Tensor, SamplerTrace]
ApplyLogitBias = Callable[[torch.Tensor], torch.Tensor]
BlockSpecialLogits = Callable[[torch.Tensor, object], None]
NewTrace = Callable[[], SamplerTrace]
HierarchicalClassProbs = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor | None]
UpdateTrace = Callable[..., None]
CallResult = TypeVar("CallResult")

FULL_RECOMPUTE = "full_recompute"
TOPOLOGY_EQUIVALENT_COMPLETED_PREFIX_CACHE = "topology_equivalent_completed_prefix_cache"
BD3_EXECUTION_MODES = (FULL_RECOMPUTE, TOPOLOGY_EQUIVALENT_COMPLETED_PREFIX_CACHE)


class _ModelPhaseTimer:
    """Measure model-only phase time without synchronizing every CUDA call."""

    def __init__(self, device: torch.device, *, enabled: bool) -> None:
        self._device = device
        self._enabled = bool(enabled)
        self._cpu_seconds = {"denoise": 0.0, "prefill": 0.0, "append": 0.0}
        self._cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            "denoise": [],
            "prefill": [],
            "append": [],
        }

    def call(self, phase: str, operation: Callable[[], CallResult]) -> CallResult:
        if not self._enabled:
            return operation()
        if phase not in self._cpu_seconds:
            raise ValueError(f"unknown BD3-LM timing phase: {phase}")
        if self._device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = operation()
            end.record()
            self._cuda_events[phase].append((start, end))
            return result
        start_time = time.perf_counter()
        result = operation()
        self._cpu_seconds[phase] += time.perf_counter() - start_time
        return result

    def seconds(self) -> dict[str, float]:
        if not self._enabled:
            return {"denoise": 0.0, "prefill": 0.0, "append": 0.0}
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            return {
                phase: sum(float(start.elapsed_time(end)) for start, end in events) / 1000.0
                for phase, events in self._cuda_events.items()
            }
        return dict(self._cpu_seconds)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def _sequence_capacity(model: torch.nn.Module) -> int | None:
    config = getattr(_unwrap_model(model), "config", None)
    capacity = getattr(config, "sequence_len", None)
    return None if capacity is None else int(capacity)


def _model_logits(
    model: torch.nn.Module,
    z_t: torch.Tensor,
    x0_context: torch.Tensor,
    attention_mask: torch.Tensor,
    block_size: int,
    *,
    apply_logit_bias: ApplyLogitBias,
) -> torch.Tensor:
    concat = torch.cat((z_t, x0_context), dim=1)
    composite_mask = bd3lm_attention_mask(attention_mask, block_size)
    position_ids = duplicated_position_ids(z_t.shape[1], device=z_t.device)
    logits = model(concat, attention_mask=composite_mask, position_ids=position_ids)
    return apply_logit_bias(logits[:, : z_t.shape[1]])


def _sample_canvas(
    model: torch.nn.Module,
    noise: MDLMNoise,
    tokenizer: object,
    initial_canvas: torch.Tensor,
    *,
    prefix_len: int,
    n_steps: int,
    block_size: int,
    scoring_mode: str,
    token_selection: str,
    temperature: float,
    jitter_scale: float,
    execution_mode: str,
    measure_model_phase_times: bool,
    return_trace: bool,
    require_complete_suffix_blocks: bool,
    apply_logit_bias: ApplyLogitBias,
    block_special_logits: BlockSpecialLogits,
    new_trace: NewTrace,
    hierarchical_class_probs: HierarchicalClassProbs,
    update_trace: UpdateTrace,
) -> SamplerOutput:
    """Fill the suffix of one BD3-LM canvas under one explicit execution mode.

    ``prefix_len`` must end on a block boundary. The clean half starts with an
    immutable copy of that prefix; only completed suffix blocks are copied into
    it. Thus an active noised block never receives a clean future token.
    """

    if scoring_mode not in ("random", "maxprob", "margin", "entropy", "margin_jitter"):
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    if execution_mode not in BD3_EXECUTION_MODES:
        raise ValueError(f"unknown BD3-LM execution mode: {execution_mode}")
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection}")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(jitter_scale)) or float(jitter_scale) < 0:
        raise ValueError("jitter_scale must be finite and non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if initial_canvas.dim() != 2:
        raise ValueError(f"BD3-LM canvas must be 2D, got shape {tuple(initial_canvas.shape)}")
    batch_size, length = initial_canvas.shape
    if batch_size <= 0:
        raise ValueError("BD3-LM sampling requires a non-empty batch")
    if prefix_len < 0 or prefix_len >= length:
        raise ValueError("BD3-LM sampling requires at least one suffix token")
    if prefix_len % int(block_size) != 0:
        raise ValueError("BD3-LM prefix length must be block-aligned")
    continuation_len = int(length) - int(prefix_len)
    if require_complete_suffix_blocks and continuation_len % int(block_size) != 0:
        raise ValueError("BD3-LM continuation length must be a whole number of blocks")
    if require_complete_suffix_blocks and int(n_steps) <= 0:
        raise ValueError("n_steps must be positive")
    resolved_steps = max(1, int(n_steps))

    device = next(model.parameters()).device
    mask_id = int(noise.mask_token_id)
    z = initial_canvas.to(device=device, dtype=torch.long).clone()
    if not bool((z[:, prefix_len:] == mask_id).all()):
        raise ValueError("BD3-LM suffix canvas must contain only MASK tokens")
    capacity = _sequence_capacity(model)
    if capacity is not None and int(length) > capacity:
        raise ValueError(f"BD3-LM total length {int(length)} exceeds model sequence capacity {capacity}")

    prefix_snapshot = z[:, :prefix_len].clone()
    if prefix_len > 0 and bool((prefix_snapshot == mask_id).any()):
        raise ValueError("BD3-LM prefixes must be clean and cannot contain MASK tokens")
    x0_context = torch.full_like(z, mask_id)
    x0_context[:, :prefix_len] = prefix_snapshot
    attention_mask = torch.ones((batch_size, length), dtype=torch.bool, device=device)
    trace = new_trace() if return_trace else None
    phase_timer = _ModelPhaseTimer(device, enabled=return_trace and measure_model_phase_times)
    denoise_model_calls = 0
    prefill_model_calls = 0
    append_model_calls = 0
    denoise_query_tokens = 0
    prefill_query_tokens = 0
    append_query_tokens = 0
    denoise_cached_kv_token_positions = 0
    denoise_active_kv_token_positions = 0
    cache_allocated_kv_bytes = 0
    cache_high_water_tokens = 0
    cache_high_water_live_kv_bytes = 0
    generated_blocks = 0
    cache = None
    cache_model = _unwrap_model(model)
    if execution_mode == TOPOLOGY_EQUIVALENT_COMPLETED_PREFIX_CACHE:
        model_mask_id = int(cache_model.config.mask_token_id)
        if mask_id != model_mask_id:
            raise ValueError(
                f"BD3-LM cache requires noise mask_token_id {mask_id} to match model mask_token_id {model_mask_id}"
            )
        vocab_size = int(cache_model.config.vocab_size)
        if z.numel() and (int(z.min().item()) < 0 or int(z.max().item()) >= vocab_size):
            raise ValueError("BD3-LM canvas token IDs must be in the model vocabulary")
        cache = new_completed_prefix_cache(
            cache_model,
            batch_size=batch_size,
            capacity=length,
            block_size=int(block_size),
        )
        prefill_stats = phase_timer.call(
            "prefill",
            lambda: prefill_completed_prefix(cache_model, cache, prefix_snapshot),
        )
        prefill_model_calls += prefill_stats.model_calls
        prefill_query_tokens += prefill_stats.processed_query_tokens
        cache_allocated_kv_bytes = cache.allocated_kv_bytes
        cache_high_water_tokens = cache.length
        cache_high_water_live_kv_bytes = cache.live_kv_bytes

    def active_logits(block_slice: slice) -> torch.Tensor:
        nonlocal denoise_active_kv_token_positions, denoise_cached_kv_token_positions
        nonlocal denoise_model_calls, denoise_query_tokens
        if execution_mode == FULL_RECOMPUTE:
            logits = phase_timer.call(
                "denoise",
                lambda: _model_logits(
                    model,
                    z,
                    x0_context,
                    attention_mask,
                    int(block_size),
                    apply_logit_bias=apply_logit_bias,
                )[:, block_slice],
            )
            query_tokens = batch_size * 2 * length
        else:
            if cache is None:
                raise RuntimeError("topology-equivalent completed-prefix cache was not initialized")
            logits, stats = phase_timer.call(
                "denoise",
                lambda: _forward_active_block_trusted(cache_model, cache, z[:, block_slice]),
            )
            query_tokens = stats.processed_query_tokens
            block_token_count = int(block_slice.stop) - int(block_slice.start)
            denoise_cached_kv_token_positions += batch_size * cache.length
            denoise_active_kv_token_positions += batch_size * block_token_count
            logits = apply_logit_bias(logits)
        denoise_model_calls += 1
        denoise_query_tokens += query_tokens
        return logits

    for block_start in range(int(prefix_len), int(length), int(block_size)):
        block_end = min(length, block_start + int(block_size))
        block_slice = slice(block_start, block_end)
        generated_blocks += 1
        for step in range(resolved_steps, 0, -1):
            still_masked = z[:, block_slice] == mask_id
            if not still_masked.any():
                break
            block_logits = active_logits(block_slice)
            block_special_logits(block_logits, tokenizer)
            probs = F.softmax(block_logits / temperature, dim=-1)
            if token_selection == "argmax":
                x_pred = block_logits.argmax(dim=-1)
            else:
                x_pred = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view(
                    batch_size,
                    block_end - block_start,
                )

            p_unmask = 1.0 / step
            if scoring_mode == "random":
                commit = still_masked & (torch.rand_like(still_masked.float()) < p_unmask)
            else:
                k_per_seq = (p_unmask * still_masked.sum(dim=-1)).long().clamp(min=1)
                if scoring_mode == "maxprob":
                    confidence = probs.max(dim=-1).values
                elif scoring_mode in ("margin", "margin_jitter"):
                    top2 = probs.topk(2, dim=-1).values
                    confidence = top2[..., 0] - top2[..., 1]
                    if scoring_mode == "margin_jitter" and jitter_scale > 0:
                        confidence = (
                            confidence
                            + (-torch.empty_like(confidence).exponential_().log() - 0.5772156649) * jitter_scale
                        )
                else:
                    safe_probs = probs.clamp_min(1e-10)
                    confidence = (safe_probs * safe_probs.log()).sum(dim=-1)
                scores = confidence.masked_fill(~still_masked, -1e9)
                commit = torch.zeros_like(still_masked)
                for row_idx in range(batch_size):
                    remaining = int(still_masked[row_idx].sum().item())
                    if remaining == 0:
                        continue
                    k = min(remaining, int(k_per_seq[row_idx].item()))
                    _, indices = torch.topk(scores[row_idx], k)
                    commit[row_idx, indices] = True

            update_trace(
                trace,
                still_masked,
                probs,
                x_pred,
                commit,
                class_probs=hierarchical_class_probs(model, probs),
            )
            block_tokens = z[:, block_slice]
            block_tokens[commit] = x_pred[commit]

        remaining = z[:, block_slice] == mask_id
        if remaining.any():
            block_logits = active_logits(block_slice)
            block_special_logits(block_logits, tokenizer)
            block_tokens = z[:, block_slice]
            block_tokens[remaining] = block_logits.argmax(dim=-1)[remaining]

        x0_context[:, block_slice] = z[:, block_slice]
        if cache is not None:
            prior_cache_length = cache.length
            append_stats = phase_timer.call(
                "append",
                lambda finalized_slice=block_slice: _append_completed_block_trusted(
                    cache_model,
                    cache,
                    z[:, finalized_slice],
                ),
            )
            append_model_calls += append_stats.model_calls
            append_query_tokens += append_stats.processed_query_tokens
            if cache.length <= prior_cache_length:
                raise RuntimeError("completed-prefix append did not advance the cache")
            cache_high_water_tokens = max(cache_high_water_tokens, cache.length)
            cache_high_water_live_kv_bytes = max(cache_high_water_live_kv_bytes, cache.live_kv_bytes)

    if cache is not None and cache.length != int(length):
        raise RuntimeError(
            f"completed-prefix cache finalized length {cache.length} does not match canvas length {int(length)}"
        )

    prefix_preserved = z[:, :prefix_len] == prefix_snapshot
    if prefix_len > 0 and not bool(prefix_preserved.all()):
        raise RuntimeError("BD3-LM sampling mutated the immutable prefix")
    if trace is not None:
        phase_seconds = phase_timer.seconds()
        model_forward_calls = denoise_model_calls + prefill_model_calls + append_model_calls
        processed_query_tokens = denoise_query_tokens + prefill_query_tokens + append_query_tokens
        prefix_rows_preserved = int(prefix_preserved.all(dim=1).sum().item()) if prefix_len > 0 else int(batch_size)
        trace.update(
            {
                "final_mask_count": int((z[:, prefix_len:] == mask_id).sum().item()),
                "num_blocks": int(generated_blocks),
                "nfe_per_block": denoise_model_calls / generated_blocks if generated_blocks else 0.0,
                "bd3lm_actual_nfe": int(denoise_model_calls),
                "bd3lm_model_forward_calls": int(model_forward_calls),
                "bd3lm_denoise_model_calls": int(denoise_model_calls),
                "bd3lm_prefill_model_calls": int(prefill_model_calls),
                "bd3lm_finalized_append_model_calls": int(append_model_calls),
                "bd3lm_processed_query_tokens": int(processed_query_tokens),
                "bd3lm_denoise_processed_query_tokens": int(denoise_query_tokens),
                "bd3lm_prefill_processed_query_tokens": int(prefill_query_tokens),
                "bd3lm_finalized_append_processed_query_tokens": int(append_query_tokens),
                "bd3lm_denoise_cached_kv_token_positions": int(denoise_cached_kv_token_positions),
                "bd3lm_denoise_active_kv_token_positions": int(denoise_active_kv_token_positions),
                "bd3lm_cache_capacity_tokens_per_sequence": int(length if cache is not None else 0),
                "bd3lm_cache_high_water_tokens_per_sequence": int(cache_high_water_tokens),
                "bd3lm_cache_allocated_kv_bytes": int(cache_allocated_kv_bytes),
                "bd3lm_cache_high_water_live_kv_bytes": int(cache_high_water_live_kv_bytes),
                "bd3lm_generated_blocks": int(generated_blocks),
                "bd3lm_prefix_len": int(prefix_len),
                "bd3lm_continuation_len": int(continuation_len),
                "bd3lm_total_len": int(length),
                "bd3lm_block_size": int(block_size),
                "bd3lm_prefix_sample_count": int(batch_size),
                "bd3lm_prefix_preserved_rows": int(prefix_rows_preserved),
                "bd3lm_prefix_token_count": int(batch_size * prefix_len),
                "bd3lm_prefix_preserved_tokens": int(prefix_preserved.sum().item()),
                "bd3lm_full_recompute_no_cache": int(execution_mode == FULL_RECOMPUTE),
                "bd3lm_topology_equivalent_completed_prefix_cache": int(
                    execution_mode == TOPOLOGY_EQUIVALENT_COMPLETED_PREFIX_CACHE
                ),
                "bd3lm_completed_prefix_dependency_topology_equivalent": int(
                    execution_mode == TOPOLOGY_EQUIVALENT_COMPLETED_PREFIX_CACHE
                ),
                "bd3lm_legacy_full_shape_trajectory_equivalence_guaranteed": int(execution_mode == FULL_RECOMPUTE),
                "bd3lm_model_phase_timing_measured": int(measure_model_phase_times),
            }
        )
        if measure_model_phase_times:
            trace.update(
                {
                    "bd3lm_model_forward_seconds": float(sum(phase_seconds.values())),
                    "bd3lm_denoise_model_seconds": float(phase_seconds["denoise"]),
                    "bd3lm_prefill_model_seconds": float(phase_seconds["prefill"]),
                    "bd3lm_finalized_append_model_seconds": float(phase_seconds["append"]),
                }
            )
        return z, trace
    return z


@torch.no_grad()
def sample_bd3lm(
    model: torch.nn.Module,
    noise: MDLMNoise,
    tokenizer: object,
    n: int,
    length: int,
    n_steps: int,
    block_size: int,
    *,
    scoring_mode: str,
    token_selection: str,
    temperature: float,
    jitter_scale: float,
    execution_mode: str,
    measure_model_phase_times: bool,
    return_trace: bool,
    apply_logit_bias: ApplyLogitBias,
    block_special_logits: BlockSpecialLogits,
    new_trace: NewTrace,
    hierarchical_class_probs: HierarchicalClassProbs,
    update_trace: UpdateTrace,
) -> SamplerOutput:
    """Generate an unconditional BD3-LM canvas, preserving partial-block compatibility."""

    if int(n) <= 0:
        raise ValueError("n must be positive")
    if int(length) <= 0:
        raise ValueError("length must be positive")
    device = next(model.parameters()).device
    canvas = torch.full(
        (int(n), int(length)),
        int(noise.mask_token_id),
        dtype=torch.long,
        device=device,
    )
    return _sample_canvas(
        model,
        noise,
        tokenizer,
        canvas,
        prefix_len=0,
        n_steps=int(n_steps),
        block_size=int(block_size),
        scoring_mode=scoring_mode,
        token_selection=token_selection,
        temperature=float(temperature),
        jitter_scale=float(jitter_scale),
        execution_mode=str(execution_mode),
        measure_model_phase_times=bool(measure_model_phase_times),
        return_trace=bool(return_trace),
        require_complete_suffix_blocks=False,
        apply_logit_bias=apply_logit_bias,
        block_special_logits=block_special_logits,
        new_trace=new_trace,
        hierarchical_class_probs=hierarchical_class_probs,
        update_trace=update_trace,
    )


@torch.no_grad()
def sample_bd3lm_from_prefixes(
    model: torch.nn.Module,
    noise: MDLMNoise,
    tokenizer: object,
    prefixes: torch.Tensor,
    continuation_len: int,
    n_steps: int,
    block_size: int,
    *,
    scoring_mode: str,
    token_selection: str,
    temperature: float,
    jitter_scale: float,
    execution_mode: str,
    measure_model_phase_times: bool,
    return_trace: bool,
    apply_logit_bias: ApplyLogitBias,
    block_special_logits: BlockSpecialLogits,
    new_trace: NewTrace,
    hierarchical_class_probs: HierarchicalClassProbs,
    update_trace: UpdateTrace,
) -> SamplerOutput:
    """Generate whole suffix blocks after an immutable, block-aligned prefix."""

    if prefixes.dim() != 2:
        raise ValueError(f"prefixes must be 2D, got shape {tuple(prefixes.shape)}")
    if int(continuation_len) <= 0:
        raise ValueError("continuation_len must be positive")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    batch_size, prefix_len = prefixes.shape
    if int(batch_size) <= 0:
        raise ValueError("prefixes must contain at least one row")
    if int(prefix_len) % int(block_size) != 0:
        raise ValueError("BD3-LM prefix length must be block-aligned")
    if int(continuation_len) % int(block_size) != 0:
        raise ValueError("BD3-LM continuation length must be a whole number of blocks")

    device = next(model.parameters()).device
    canvas = torch.full(
        (int(batch_size), int(prefix_len) + int(continuation_len)),
        int(noise.mask_token_id),
        dtype=torch.long,
        device=device,
    )
    canvas[:, : int(prefix_len)] = prefixes.to(device=device, dtype=torch.long)
    return _sample_canvas(
        model,
        noise,
        tokenizer,
        canvas,
        prefix_len=int(prefix_len),
        n_steps=int(n_steps),
        block_size=int(block_size),
        scoring_mode=scoring_mode,
        token_selection=token_selection,
        temperature=float(temperature),
        jitter_scale=float(jitter_scale),
        execution_mode=str(execution_mode),
        measure_model_phase_times=bool(measure_model_phase_times),
        return_trace=bool(return_trace),
        require_complete_suffix_blocks=True,
        apply_logit_bias=apply_logit_bias,
        block_special_logits=block_special_logits,
        new_trace=new_trace,
        hierarchical_class_probs=hierarchical_class_probs,
        update_trace=update_trace,
    )
