"""Paired one-pass draft/correction transition evaluation for masked models."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .data import DLMDataBatch, build_token_bytes
from .eval_token_edit_counts import EXACT_EDIT_DIAGNOSTIC_SCHEMA
from .eval_token_edit_counts import group_metrics as _group_metrics
from .eval_token_edit_counts import merge_group_accumulator as _merge_group_accumulator
from .eval_token_edit_counts import new_group_accumulator as _new_group_accumulator
from .eval_token_edit_counts import ratio as _ratio
from .eval_token_edit_counts import row_group_record as _row_group_record
from .eval_token_edit_counts import row_transition_counts as _row_transition_counts
from .eval_token_edit_counts import transition_count_view as _transition_count_view
from .eval_token_edit_counts import update_group_accumulator as _update_group_accumulator
from .eval_token_edit_strata import (
    flatten_transition_strata_metrics,
    merge_transition_strata,
    new_transition_strata,
    relative_position_decile_ids,
    transition_strata_definitions,
    update_transition_strata,
    validate_transition_strata_conservation,
)

TOKEN_EDIT_TRANSITION_RECORD_SCHEMA_VERSION = 1
TOKEN_EDIT_TRANSITION_RECORD_SCHEMA = (
    f"nanodllm.token_edit_transition_sequence.v{TOKEN_EDIT_TRANSITION_RECORD_SCHEMA_VERSION}"
)
TOKEN_EDIT_TRANSITION_METRIC_SCHEMA = "nanodllm.token_edit_transition_metrics.v1"

_GROUPS = ("all", "content", "special")


def _validate_mask_fractions(mask_fractions: Sequence[float]) -> tuple[tuple[float, str], ...]:
    if not mask_fractions:
        raise ValueError("mask_fractions must be non-empty")
    resolved: list[tuple[float, str]] = []
    labels: set[str] = set()
    for raw_fraction in mask_fractions:
        if isinstance(raw_fraction, bool) or not isinstance(raw_fraction, (int, float)):
            raise ValueError("mask fractions must be finite numbers in [0, 1]")
        fraction = float(raw_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("mask fractions must be finite numbers in [0, 1]")
        label = f"{int(round(100.0 * fraction)):03d}"
        if label in labels:
            raise ValueError(f"mask fractions collide at metric label mask_{label}")
        labels.add(label)
        resolved.append((fraction, label))
    return tuple(resolved)


def _validate_batches(clean_batches: Sequence[DLMDataBatch]) -> tuple[int, int]:
    if not clean_batches:
        raise ValueError("clean_batches must be non-empty")
    sequence_length: int | None = None
    sequence_count = 0
    for batch_index, batch in enumerate(clean_batches):
        if batch.x_0.dim() != 2:
            raise ValueError(f"clean batch {batch_index} x_0 must be [batch, sequence]")
        batch_sequence_length = int(batch.x_0.shape[1])
        if sequence_length is None:
            sequence_length = batch_sequence_length
        elif batch_sequence_length != sequence_length:
            raise ValueError("all clean batches must have the same sequence length")
        if batch.x_0.shape != batch.loss_mask.shape or batch.x_0.shape != batch.attention_mask.shape:
            raise ValueError(f"clean batch {batch_index} token and mask shapes must match")
        sequence_count += int(batch.x_0.shape[0])
    if sequence_count <= 0:
        raise ValueError("clean_batches must contain at least one sequence")
    assert sequence_length is not None
    if sequence_length <= 0:
        raise ValueError("clean batches must have a positive sequence length")
    return sequence_count, sequence_length


def _sequence_major_uniform_plan(sequence_count: int, sequence_length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand(
        (sequence_count, sequence_length),
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    )


def _mixed_canvas_drafted_mask(
    initial_mask: torch.Tensor,
    draft_uniforms: torch.Tensor,
    draft_fraction: float,
) -> torch.Tensor:
    """Choose an exact deterministic drafted subset independently in each row."""

    if initial_mask.device.type != "cpu" or draft_uniforms.device.type != "cpu":
        raise ValueError("mixed-canvas subset planning requires CPU tensors")
    if initial_mask.dtype != torch.bool or initial_mask.shape != draft_uniforms.shape:
        raise ValueError("mixed-canvas mask and uniform plan must have matching shapes")
    drafted = torch.zeros_like(initial_mask)
    for row_index in range(int(initial_mask.shape[0])):
        target_indices = initial_mask[row_index].nonzero(as_tuple=False).flatten()
        target_count = int(target_indices.numel())
        if target_count == 0:
            continue
        drafted_count = 1 if target_count == 1 else min(target_count - 1, max(1, round(target_count * draft_fraction)))
        ranked = torch.argsort(draft_uniforms[row_index, target_indices], stable=True)
        drafted[row_index, target_indices[ranked[:drafted_count]]] = True
    return drafted


def _tensor_payload(tensor: torch.Tensor) -> tuple[bytes, bytes]:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bool:
        array = cpu.to(dtype=torch.uint8).numpy()
        return b"bool-packbits-little", np.packbits(array, bitorder="little").tobytes(order="C")
    if cpu.dtype == torch.long:
        return b"i64-little", cpu.numpy().astype("<i8", copy=False).tobytes(order="C")
    if cpu.dtype == torch.float32:
        return b"f32-little", cpu.numpy().astype("<f4", copy=False).tobytes(order="C")
    raise ValueError(f"unsupported identity tensor dtype: {cpu.dtype}")


def _tensor_sha256(domain: bytes, tensor: torch.Tensor) -> str:
    kind, payload = _tensor_payload(tensor)
    digest = hashlib.sha256(domain + b"\0")
    digest.update(struct.pack("<I", tensor.dim()))
    for dimension in tensor.shape:
        digest.update(struct.pack("<Q", int(dimension)))
    digest.update(struct.pack("<I", len(kind)))
    digest.update(kind)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)
    return digest.hexdigest()


def _optional_tensor_sha256(domain: bytes, tensor: torch.Tensor | None) -> str | None:
    return None if tensor is None else _tensor_sha256(domain, tensor)


def _input_identity_sha256(
    clean_row: torch.Tensor,
    loss_mask_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    model_attention_mask_row: torch.Tensor,
    position_ids_row: torch.Tensor | None,
) -> str:
    digest = hashlib.sha256(b"nanodllm-token-edit-transition-input-v1\0")
    fields = (
        _tensor_sha256(b"clean", clean_row),
        _tensor_sha256(b"loss-mask", loss_mask_row),
        _tensor_sha256(b"attention-mask", attention_mask_row),
        _tensor_sha256(b"model-attention-mask", model_attention_mask_row),
        _optional_tensor_sha256(b"position-ids", position_ids_row),
    )
    for value in fields:
        if value is None:
            digest.update(b"\0")
        else:
            digest.update(b"\1")
            digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _evaluation_sample_sha256(
    input_identity_sha256: str,
    mask_fraction: float,
    uniform_sha256: str,
    initial_mask_sha256: str,
    draft_sha256: str,
    corrected_sha256: str,
) -> str:
    digest = hashlib.sha256(b"nanodllm-token-edit-transition-sample-v1\0")
    digest.update(bytes.fromhex(input_identity_sha256))
    digest.update(struct.pack("<d", mask_fraction))
    for value in (uniform_sha256, initial_mask_sha256, draft_sha256, corrected_sha256):
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _canonical_record_sha256(record: dict[str, object]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_root(domain: bytes, hashes: Sequence[str]) -> str:
    digest = hashlib.sha256(domain + b"\0")
    digest.update(struct.pack("<Q", len(hashes)))
    for value in hashes:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _blocked_argmax_with_confidence(
    logits: torch.Tensor,
    tokenizer: Any,
    confidence_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(logits) or logits.dim() != 3 or not logits.is_floating_point():
        shape = tuple(logits.shape) if torch.is_tensor(logits) else type(logits)
        raise ValueError(f"model logits must be a floating [batch, sequence, vocab] tensor, got {shape}")
    vocab_size = int(tokenizer.get_vocab_size())
    if logits.shape[-1] < vocab_size:
        raise ValueError(
            f"model logits vocabulary {logits.shape[-1]} is smaller than tokenizer vocabulary {vocab_size}"
        )
    filtered = logits[..., :vocab_size].clone()
    blocked_ids = {
        int(token_id) for token_id in tokenizer.get_generation_blocked_token_ids() if 0 <= int(token_id) < vocab_size
    }
    if len(blocked_ids) >= vocab_size:
        raise ValueError("tokenizer blocks every vocabulary token during generation")
    if blocked_ids:
        blocked = torch.tensor(sorted(blocked_ids), dtype=torch.long, device=filtered.device)
        filtered.index_fill_(-1, blocked, -torch.inf)
    predictions = filtered.argmax(dim=-1)
    if confidence_mask.dtype != torch.bool or confidence_mask.shape != predictions.shape:
        raise ValueError("confidence_mask must be boolean and match the logits batch and sequence dimensions")
    confidence = torch.zeros_like(predictions, dtype=torch.float32)
    if bool(confidence_mask.any().item()):
        selected_logits = filtered[confidence_mask].float()
        selected_max = selected_logits.max(dim=-1).values
        selected_confidence = torch.exp(selected_max - torch.logsumexp(selected_logits, dim=-1)).clamp(0.0, 1.0)
        confidence[confidence_mask] = selected_confidence
    return predictions, confidence


def _call_model_logits(
    model: Any,
    z_t: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    model_logits_fn: Callable[..., torch.Tensor] | None,
    model_kwargs: Mapping[str, object] | None = None,
) -> torch.Tensor:
    extra_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    if model_logits_fn is not None:
        if position_ids is None:
            return model_logits_fn(model, z_t, x_0, attention_mask, **extra_kwargs)
        return model_logits_fn(
            model,
            z_t,
            x_0,
            attention_mask,
            position_ids=position_ids,
            **extra_kwargs,
        )
    kwargs: dict[str, object] = {"attention_mask": attention_mask, **extra_kwargs}
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return model(z_t, **kwargs)


def _synchronize_for_forward_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _flatten_group_metrics(prefix: str, group: str, values: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}{group}_{key}": value for key, value in values.items()}


def _special_target_mask(x_0: torch.Tensor, special_ids: set[int]) -> torch.Tensor:
    if not special_ids:
        return torch.zeros_like(x_0, dtype=torch.bool)
    ids = torch.tensor(sorted(special_ids), dtype=torch.long, device=x_0.device)
    return torch.isin(x_0, ids)


def _decode_row(tokenizer: Any, tokens: torch.Tensor, attention_mask: torch.Tensor) -> str:
    return tokenizer.decode(tokens[attention_mask].detach().cpu().tolist())


@torch.no_grad()
def evaluate_token_edit_transitions(
    model: Any,
    tokenizer: Any,
    clean_batches: Sequence[DLMDataBatch],
    *,
    mask_fractions: Sequence[float],
    seed: int,
    device: torch.device | str,
    canvas_mode: str = "full_draft",
    mixed_canvas_draft_fraction: float = 1.0,
    draft_subset_seed: int | None = None,
    correction_use_token_edit_adapter: bool = False,
    correction_use_token_edit_provenance: bool = False,
    correction_token_edit_provenance_confidence_bins: int = 0,
    model_logits_fn: Callable[..., torch.Tensor] | None = None,
    attention_mask_fn: Callable[[DLMDataBatch], torch.Tensor] | None = None,
    token_frequency_class_ids: torch.Tensor | None = None,
    source_document_sha256s: Sequence[str] | None = None,
    panel_item_sha256s: Sequence[str] | None = None,
    record_context: dict[str, object] | None = None,
    log_fn: Callable[[str], None] | None = None,
    log_samples: int = 0,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Measure one simultaneous token-edit correction after an M2T argmax draft.

    The clean batches are consumed as a fixed paired panel across all mask
    fractions. A dedicated CPU generator creates one sequence-major uniform
    plan, and each fraction thresholds the same rows. Empty masks remain empty.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if seed >= 2**63:
        raise ValueError("seed must be smaller than 2**63")
    if canvas_mode not in ("full_draft", "mixed_mask_draft"):
        raise ValueError("canvas_mode must be 'full_draft' or 'mixed_mask_draft'")
    if isinstance(mixed_canvas_draft_fraction, bool) or not isinstance(mixed_canvas_draft_fraction, int | float):
        raise ValueError("mixed_canvas_draft_fraction must be finite")
    resolved_draft_fraction = float(mixed_canvas_draft_fraction)
    if not math.isfinite(resolved_draft_fraction):
        raise ValueError("mixed_canvas_draft_fraction must be finite")
    if canvas_mode == "full_draft":
        if resolved_draft_fraction != 1.0:
            raise ValueError("full-draft evaluation requires mixed_canvas_draft_fraction=1")
        if draft_subset_seed is not None:
            raise ValueError("full-draft evaluation does not accept draft_subset_seed")
    else:
        if not 0.0 < resolved_draft_fraction < 1.0:
            raise ValueError("mixed MASK/draft evaluation requires mixed_canvas_draft_fraction in (0, 1)")
        if isinstance(draft_subset_seed, bool) or not isinstance(draft_subset_seed, int):
            raise ValueError("mixed MASK/draft evaluation requires a nonnegative draft_subset_seed")
        if not 0 <= draft_subset_seed < 2**63:
            raise ValueError("draft_subset_seed must lie in [0, 2**63)")
    if not isinstance(correction_use_token_edit_adapter, bool):
        raise ValueError("correction_use_token_edit_adapter must be boolean")
    if not isinstance(correction_use_token_edit_provenance, bool):
        raise ValueError("correction_use_token_edit_provenance must be boolean")
    if (
        not isinstance(correction_token_edit_provenance_confidence_bins, int)
        or isinstance(correction_token_edit_provenance_confidence_bins, bool)
        or not 0 <= correction_token_edit_provenance_confidence_bins <= 64
    ):
        raise ValueError("correction_token_edit_provenance_confidence_bins must be an integer in [0, 64]")
    if correction_token_edit_provenance_confidence_bins and not correction_use_token_edit_provenance:
        raise ValueError("confidence-binned correction requires token-edit provenance")
    if isinstance(log_samples, bool) or not isinstance(log_samples, int) or log_samples < 0:
        raise ValueError("log_samples must be a nonnegative integer")
    resolved_fractions = _validate_mask_fractions(mask_fractions)
    sequence_count, sequence_length = _validate_batches(clean_batches)
    if (source_document_sha256s is None) != (panel_item_sha256s is None):
        raise ValueError("source_document_sha256s and panel_item_sha256s must be provided together")
    resolved_source_document_sha256s: tuple[str, ...] | None = None
    resolved_panel_item_sha256s: tuple[str, ...] | None = None
    if source_document_sha256s is not None and panel_item_sha256s is not None:
        if any(not isinstance(value, str) for value in source_document_sha256s):
            raise ValueError("source_document_sha256s must contain strings")
        if any(not isinstance(value, str) for value in panel_item_sha256s):
            raise ValueError("panel_item_sha256s must contain strings")
        resolved_source_document_sha256s = tuple(source_document_sha256s)
        resolved_panel_item_sha256s = tuple(panel_item_sha256s)
        if len(resolved_source_document_sha256s) != sequence_count:
            raise ValueError("source_document_sha256s must contain one identity per sequence")
        if len(resolved_panel_item_sha256s) != sequence_count:
            raise ValueError("panel_item_sha256s must contain one identity per sequence")
        for label, values in (
            ("source document", resolved_source_document_sha256s),
            ("panel item", resolved_panel_item_sha256s),
        ):
            if any(
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in values
            ):
                raise ValueError(f"every {label} identity must be a lowercase SHA-256")
        if len(set(resolved_source_document_sha256s)) != sequence_count:
            raise ValueError("source_document_sha256s must be unique within a scientific panel")
        if len(set(resolved_panel_item_sha256s)) != sequence_count:
            raise ValueError("panel_item_sha256s must be unique within a scientific panel")
    target_device = torch.device(device)
    uniform_plan = _sequence_major_uniform_plan(sequence_count, sequence_length, seed)
    draft_uniform_plan = (
        None
        if draft_subset_seed is None
        else _sequence_major_uniform_plan(sequence_count, sequence_length, draft_subset_seed)
    )
    mask_token_id = int(tokenizer.get_mask_token_id())
    vocab_size = int(tokenizer.get_vocab_size())
    if not 0 <= mask_token_id < vocab_size:
        raise ValueError("mask token id must lie within the tokenizer vocabulary")
    generation_blocked_ids = {int(value) for value in tokenizer.get_generation_blocked_token_ids()}
    special_ids = {int(value) for value in tokenizer.get_special_token_ids()}
    token_bytes = build_token_bytes(tokenizer).to(device=target_device)
    frequency_class_ids = None
    frequency_band_count = 0
    frequency_map_sha256 = None
    if token_frequency_class_ids is not None:
        if token_frequency_class_ids.dim() != 1 or int(token_frequency_class_ids.numel()) != vocab_size:
            raise ValueError("token_frequency_class_ids must have one entry per tokenizer vocabulary token")
        frequency_class_ids = token_frequency_class_ids.to(device=target_device, dtype=torch.long)
        if bool((frequency_class_ids < 0).any().item()):
            raise ValueError("token_frequency_class_ids must be nonnegative")
        frequency_band_count = int(frequency_class_ids.max().item()) + 1
        if frequency_band_count != 8:
            raise ValueError("token_frequency_class_ids must use class 0 plus seven ordinary-token frequency bands")
        frequency_map_sha256 = _tensor_sha256(b"token-frequency-class-map", frequency_class_ids)
    resolved_record_context = {} if record_context is None else dict(record_context)
    json.dumps(resolved_record_context, sort_keys=True, separators=(",", ":"), allow_nan=False)

    metrics: dict[str, object] = {
        "token_edit_transition_schema": TOKEN_EDIT_TRANSITION_METRIC_SCHEMA,
        "token_edit_transition_record_schema": TOKEN_EDIT_TRANSITION_RECORD_SCHEMA,
        "token_edit_transition_exact_edit_diagnostics_schema": EXACT_EDIT_DIAGNOSTIC_SCHEMA,
        "token_edit_transition_exact_edit_zero_denominator_policy": "zero_with_defined_flag",
        "token_edit_transition_seed": seed,
        "token_edit_transition_batch_count": len(clean_batches),
        "token_edit_transition_sequence_count": sequence_count,
        "token_edit_transition_sequence_length": sequence_length,
        "token_edit_transition_mask_fraction_count": len(resolved_fractions),
        "token_edit_transition_mask_fractions": ",".join(f"{fraction:.6g}" for fraction, _ in resolved_fractions),
        "token_edit_transition_draft_policy": "blocked_argmax",
        "token_edit_transition_correction_policy": "blocked_argmax",
        "token_edit_transition_correction_scope": "initially_masked_lossable_positions",
        "token_edit_transition_draft_nfe": 1,
        "token_edit_transition_correction_nfe": 1,
        "token_edit_transition_total_nfe": 2,
        "token_edit_transition_generation_blocked_token_count": len(generation_blocked_ids),
        "token_edit_transition_special_token_count": len(special_ids),
        "token_edit_transition_uniform_plan_sha256": _tensor_sha256(b"uniform-plan", uniform_plan),
        "token_edit_transition_stratification": "marginal_not_cartesian",
        "token_edit_transition_strata_definitions": transition_strata_definitions(),
        "token_edit_transition_token_frequency_strata_enabled": frequency_class_ids is not None,
        "token_edit_transition_token_frequency_band_count": frequency_band_count,
        "token_edit_transition_token_frequency_map_sha256": frequency_map_sha256,
        "token_edit_transition_record_context": resolved_record_context,
        "token_edit_transition_correction_use_token_edit_provenance": correction_use_token_edit_provenance,
        "token_edit_transition_correction_token_edit_provenance_confidence_bins": (
            correction_token_edit_provenance_confidence_bins
        ),
    }
    if canvas_mode == "mixed_mask_draft":
        assert draft_uniform_plan is not None and draft_subset_seed is not None
        metrics.update(
            {
                "token_edit_transition_canvas_mode": canvas_mode,
                "token_edit_transition_mixed_canvas_draft_fraction": resolved_draft_fraction,
                "token_edit_transition_draft_subset_seed": draft_subset_seed,
                "token_edit_transition_draft_subset_uniform_plan_sha256": _tensor_sha256(
                    b"draft-subset-uniform-plan", draft_uniform_plan
                ),
                "token_edit_transition_correction_use_token_edit_adapter": correction_use_token_edit_adapter,
                "token_edit_transition_correction_use_token_edit_provenance": correction_use_token_edit_provenance,
                "token_edit_transition_correction_scope": "inserted_model_draft_positions",
            }
        )
    all_records: list[dict[str, object]] = []
    pooled_accumulators = {group: _new_group_accumulator() for group in _GROUPS}
    pooled_strata = new_transition_strata(frequency_band_count)
    model_forward_seconds = 0.0
    model_forward_calls = 0

    for mask_fraction, mask_label in resolved_fractions:
        prefix = f"token_edit_transition_mask_{mask_label}_"
        accumulators = {group: _new_group_accumulator() for group in _GROUPS}
        strata = new_transition_strata(frequency_band_count)
        fraction_records: list[dict[str, object]] = []
        lossable_token_count = 0
        initially_masked_token_count = 0
        drafted_token_count = 0
        logged_rows = 0
        sequence_offset = 0

        for batch in clean_batches:
            batch_size = int(batch.x_0.shape[0])
            next_offset = sequence_offset + batch_size
            x_0 = batch.x_0.to(device=target_device)
            raw_loss_mask = batch.loss_mask.to(device=target_device)
            base_attention = batch.attention_mask.to(device=target_device)
            lossable = raw_loss_mask & base_attention
            model_attention = (batch.attention_mask if attention_mask_fn is None else attention_mask_fn(batch)).to(
                device=target_device
            )
            expected_attention_shapes = {(batch_size, sequence_length), (batch_size, sequence_length, sequence_length)}
            if model_attention.dtype != torch.bool or tuple(model_attention.shape) not in expected_attention_shapes:
                raise ValueError("attention_mask_fn must return a [batch, sequence] or [batch, query, key] tensor")
            position_ids = None if batch.position_ids is None else batch.position_ids.to(device=target_device)
            uniforms_cpu = uniform_plan[sequence_offset:next_offset]
            initial_mask = (uniforms_cpu.to(device=target_device) < mask_fraction) & lossable
            initially_masked_token_count += int(initial_mask.sum().item())
            if draft_uniform_plan is None:
                drafted_mask = initial_mask
                drafted_mask_cpu = initial_mask.detach().cpu()
            else:
                drafted_mask_cpu = _mixed_canvas_drafted_mask(
                    initial_mask.detach().cpu(),
                    draft_uniform_plan[sequence_offset:next_offset],
                    resolved_draft_fraction,
                )
                drafted_mask = drafted_mask_cpu.to(device=target_device)
            drafted_token_count += int(drafted_mask.sum().item())
            lossable_token_count += int(lossable.sum().item())
            position_deciles = relative_position_decile_ids(base_attention)
            batch_frequency_band_ids = None if frequency_class_ids is None else frequency_class_ids[x_0]

            masked = x_0.clone()
            masked[initial_mask] = mask_token_id
            _synchronize_for_forward_timing(target_device)
            forward_start = time.perf_counter()
            draft_logits = _call_model_logits(
                model,
                masked,
                x_0,
                model_attention,
                position_ids,
                model_logits_fn,
            )
            _synchronize_for_forward_timing(target_device)
            model_forward_seconds += time.perf_counter() - forward_start
            model_forward_calls += 1
            if tuple(draft_logits.shape[:2]) != tuple(x_0.shape):
                raise ValueError("draft logits batch and sequence dimensions must match the clean tokens")
            draft_predictions, draft_confidence = _blocked_argmax_with_confidence(
                draft_logits,
                tokenizer,
                initial_mask,
            )
            del draft_logits
            draft = masked.clone()
            draft[drafted_mask] = draft_predictions[drafted_mask]

            correction_kwargs: dict[str, object] = {}
            if correction_use_token_edit_adapter:
                correction_kwargs["use_token_edit_adapter"] = True
            if correction_use_token_edit_provenance:
                input_provenance = torch.zeros_like(x_0, dtype=torch.long)
                input_provenance[initial_mask] = 1
                if correction_token_edit_provenance_confidence_bins:
                    confidence_bins = (
                        torch.floor(draft_confidence.float() * correction_token_edit_provenance_confidence_bins)
                        .long()
                        .clamp_max(correction_token_edit_provenance_confidence_bins - 1)
                    )
                    input_provenance[drafted_mask] = 3 + confidence_bins[drafted_mask]
                else:
                    input_provenance[drafted_mask] = 2
                correction_kwargs["input_provenance"] = input_provenance

            _synchronize_for_forward_timing(target_device)
            forward_start = time.perf_counter()
            correction_logits = _call_model_logits(
                model,
                draft,
                x_0,
                model_attention,
                position_ids,
                model_logits_fn,
                correction_kwargs,
            )
            _synchronize_for_forward_timing(target_device)
            model_forward_seconds += time.perf_counter() - forward_start
            model_forward_calls += 1
            if tuple(correction_logits.shape[:2]) != tuple(x_0.shape):
                raise ValueError("correction logits batch and sequence dimensions must match the clean tokens")
            correction_predictions, correction_confidence = _blocked_argmax_with_confidence(
                correction_logits,
                tokenizer,
                drafted_mask,
            )
            del correction_logits
            corrected = draft.clone()
            corrected[initial_mask] = correction_predictions[initial_mask]

            draft_correct = draft == x_0
            final_correct = corrected == x_0
            changed = corrected != draft
            special_targets = _special_target_mask(x_0, special_ids)
            content_targets = token_bytes[x_0] > 0
            group_masks = {
                "all": drafted_mask,
                "content": drafted_mask & content_targets,
                "special": drafted_mask & special_targets,
            }

            for row_index in range(batch_size):
                global_sequence_index = sequence_offset + row_index
                row_groups: dict[str, dict[str, object]] = {}
                for group in _GROUPS:
                    row_counts = _row_transition_counts(
                        draft_correct[row_index],
                        final_correct[row_index],
                        group_masks[group][row_index],
                        changed[row_index],
                    )
                    _update_group_accumulator(accumulators[group], row_counts)
                    row_groups[group] = _row_group_record(row_counts)

                row_frequency_band_ids = (
                    None if batch_frequency_band_ids is None else batch_frequency_band_ids[row_index]
                )
                row_strata = update_transition_strata(
                    strata,
                    draft_correct=draft_correct[row_index],
                    final_correct=final_correct[row_index],
                    eligible=drafted_mask[row_index],
                    position_deciles=position_deciles[row_index],
                    draft_confidence=draft_confidence[row_index],
                    correction_confidence=correction_confidence[row_index],
                    frequency_band_ids=row_frequency_band_ids,
                )
                row_draft_error_bands = list(row_strata["draft_error_band"])
                if len(row_draft_error_bands) > 1:
                    raise RuntimeError("one sequence was assigned to multiple draft-error bands")

                model_attention_row = model_attention[row_index]
                position_ids_row = None if position_ids is None else position_ids[row_index]
                input_identity = _input_identity_sha256(
                    x_0[row_index],
                    raw_loss_mask[row_index],
                    base_attention[row_index],
                    model_attention_row,
                    position_ids_row,
                )
                uniform_sha256 = _tensor_sha256(b"uniform-row", uniforms_cpu[row_index])
                initial_mask_sha256 = _tensor_sha256(b"initial-mask-row", initial_mask[row_index])
                draft_sha256 = _tensor_sha256(b"draft-row", draft[row_index])
                corrected_sha256 = _tensor_sha256(b"corrected-row", corrected[row_index])
                sample_sha256 = _evaluation_sample_sha256(
                    input_identity,
                    mask_fraction,
                    uniform_sha256,
                    initial_mask_sha256,
                    draft_sha256,
                    corrected_sha256,
                )
                record: dict[str, object] = {
                    "schema": TOKEN_EDIT_TRANSITION_RECORD_SCHEMA,
                    "exact_edit_diagnostics_schema": EXACT_EDIT_DIAGNOSTIC_SCHEMA,
                    "sequence_index": global_sequence_index,
                    "mask_fraction": mask_fraction,
                    "mask_label": mask_label,
                    "lossable_token_count": int(lossable[row_index].sum().item()),
                    "transition_groups": row_groups,
                    "transition_strata": row_strata,
                    "draft_error_band": row_draft_error_bands[0] if row_draft_error_bands else None,
                    "evaluation_context": resolved_record_context,
                    "clean_input_sha256": _tensor_sha256(b"clean-row", x_0[row_index]),
                    "input_identity_sha256": input_identity,
                    "uniform_sha256": uniform_sha256,
                    "initial_mask_sha256": initial_mask_sha256,
                    "draft_sha256": draft_sha256,
                    "corrected_sha256": corrected_sha256,
                    "evaluation_sample_sha256": sample_sha256,
                }
                if canvas_mode == "mixed_mask_draft":
                    assert draft_uniform_plan is not None
                    record["draft_subset_uniform_sha256"] = _tensor_sha256(
                        b"draft-subset-uniform-row",
                        draft_uniform_plan[global_sequence_index],
                    )
                    record["drafted_mask_sha256"] = _tensor_sha256(
                        b"drafted-mask-row",
                        drafted_mask_cpu[row_index],
                    )
                if resolved_source_document_sha256s is not None and resolved_panel_item_sha256s is not None:
                    record["source_document_sha256"] = resolved_source_document_sha256s[global_sequence_index]
                    record["panel_item_sha256"] = resolved_panel_item_sha256s[global_sequence_index]
                record["record_sha256"] = _canonical_record_sha256(record)
                fraction_records.append(record)

                if log_fn is not None and logged_rows < log_samples and int(row_groups["all"]["eligible_count"]) > 0:
                    counts = row_groups["all"]
                    log_fn(
                        f"[token-edit-transition] mask={mask_fraction:.3f} sequence={global_sequence_index} "
                        f"C->C={counts['c_to_c_count']} C->W={counts['c_to_w_count']} "
                        f"W->C={counts['w_to_c_count']} W->W={counts['w_to_w_count']}"
                    )
                    log_fn(
                        f"[token-edit-transition] target={_decode_row(tokenizer, x_0[row_index], base_attention[row_index])!r}"
                    )
                    log_fn(
                        f"[token-edit-transition] draft={_decode_row(tokenizer, draft[row_index], base_attention[row_index])!r}"
                    )
                    log_fn(
                        f"[token-edit-transition] corrected="
                        f"{_decode_row(tokenizer, corrected[row_index], base_attention[row_index])!r}"
                    )
                    logged_rows += 1

            sequence_offset = next_offset

        if sequence_offset != sequence_count:
            raise RuntimeError("transition evaluator did not consume its complete sequence-major uniform plan")
        metrics[f"{prefix}mask_fraction"] = mask_fraction
        metrics[f"{prefix}sequence_count"] = sequence_count
        metrics[f"{prefix}lossable_token_count"] = lossable_token_count
        metrics[f"{prefix}initially_masked_token_count"] = initially_masked_token_count
        metrics[f"{prefix}realized_mask_fraction"] = _ratio(initially_masked_token_count, lossable_token_count)
        if canvas_mode == "mixed_mask_draft":
            metrics[f"{prefix}drafted_token_count"] = drafted_token_count
            metrics[f"{prefix}realized_draft_fraction"] = _ratio(drafted_token_count, initially_masked_token_count)
        metrics[f"{prefix}logged_sequence_count"] = logged_rows
        for group in _GROUPS:
            metrics.update(_flatten_group_metrics(prefix, group, _group_metrics(accumulators[group])))
            _merge_group_accumulator(pooled_accumulators[group], accumulators[group])
        validate_transition_strata_conservation(strata, _transition_count_view(accumulators["all"]))
        metrics.update(flatten_transition_strata_metrics(prefix, strata))
        merge_transition_strata(pooled_strata, strata)
        record_hashes = [str(record["record_sha256"]) for record in fraction_records]
        metrics[f"{prefix}record_count"] = len(fraction_records)
        metrics[f"{prefix}record_root_sha256"] = _ordered_root(
            b"nanodllm-token-edit-transition-record-root-v1", record_hashes
        )
        all_records.extend(fraction_records)

    all_record_hashes = [str(record["record_sha256"]) for record in all_records]
    metrics["token_edit_transition_record_count"] = len(all_records)
    metrics["token_edit_transition_record_root_sha256"] = _ordered_root(
        b"nanodllm-token-edit-transition-all-record-root-v1", all_record_hashes
    )
    metrics["token_edit_transition_model_forward_calls"] = model_forward_calls
    metrics["token_edit_transition_model_forward_seconds"] = model_forward_seconds
    metrics["token_edit_transition_model_forward_timing"] = (
        "wall_clock_with_pre_and_post_cuda_synchronize" if target_device.type == "cuda" else "wall_clock_cpu"
    )
    pooled_prefix = "token_edit_transition_pooled_"
    metrics[f"{pooled_prefix}semantics"] = "sum_of_panel_exposures_across_mask_fractions_with_paired_sequences"
    for group in _GROUPS:
        metrics.update(_flatten_group_metrics(pooled_prefix, group, _group_metrics(pooled_accumulators[group])))
    validate_transition_strata_conservation(pooled_strata, _transition_count_view(pooled_accumulators["all"]))
    metrics.update(flatten_transition_strata_metrics(pooled_prefix, pooled_strata))
    return metrics, all_records
