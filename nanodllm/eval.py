import hashlib
import json
import math
import os
import struct
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import DataConfig
from .data import MAX_SEQ_LEN, build_token_bytes, make_dataloader
from .mdlm_objective import expand_mdlm_time, schedule_weighted_mdlm_terms
from .sampling import sampling_logit_bias

NELBO_SEQUENCE_RECORD_SCHEMA_VERSION = 2


def _update_identity_digest(digest, label: bytes, payload: bytes | None) -> None:
    """Add one unambiguous, length-prefixed field to a row identity digest."""

    digest.update(struct.pack("<I", len(label)))
    digest.update(label)
    if payload is None:
        digest.update(struct.pack("<Q", 2**64 - 1))
        return
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _little_endian_row_bytes(row: torch.Tensor, dtype: str) -> bytes:
    array = row.detach().contiguous().numpy()
    return array.astype(dtype, copy=False).tobytes(order="C")


def _packed_bool_row_bytes(row: torch.Tensor) -> bytes:
    array = row.detach().to(dtype=torch.uint8).contiguous().numpy()
    return np.packbits(array, bitorder="little").tobytes(order="C")


def _nelbo_input_identity_sha256(
    x_0_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    loss_mask_row: torch.Tensor,
    position_ids_row: torch.Tensor | None,
    document_start_mask_row: torch.Tensor | None,
) -> str:
    digest = hashlib.sha256(b"nanodllm-nelbo-input-identity-v2\0")
    _update_identity_digest(digest, b"x_0_i64_le", _little_endian_row_bytes(x_0_row, "<i8"))
    _update_identity_digest(digest, b"attention_mask_packbits_le", _packed_bool_row_bytes(attention_mask_row))
    _update_identity_digest(digest, b"loss_mask_packbits_le", _packed_bool_row_bytes(loss_mask_row))
    _update_identity_digest(
        digest,
        b"position_ids_i64_le",
        None if position_ids_row is None else _little_endian_row_bytes(position_ids_row, "<i8"),
    )
    _update_identity_digest(
        digest,
        b"document_start_mask_packbits_le",
        None if document_start_mask_row is None else _packed_bool_row_bytes(document_start_mask_row),
    )
    return digest.hexdigest()


def _nelbo_uniform_sha256(uniform_row: torch.Tensor) -> str:
    digest = hashlib.sha256(b"nanodllm-nelbo-corruption-uniform-v1\0")
    _update_identity_digest(digest, b"uniform_f32_le", _little_endian_row_bytes(uniform_row, "<f4"))
    return digest.hexdigest()


def _nelbo_corruption_mask_sha256(active_mask_row: torch.Tensor) -> str:
    digest = hashlib.sha256(b"nanodllm-nelbo-corruption-mask-v2\0")
    _update_identity_digest(digest, b"active_mask_packbits_le", _packed_bool_row_bytes(active_mask_row))
    return digest.hexdigest()


def _nelbo_evaluation_sample_sha256(
    input_identity_sha256: str,
    sampled_t: float,
    corruption_uniform_sha256: str,
    corruption_mask_sha256: str,
) -> str:
    digest = hashlib.sha256(b"nanodllm-nelbo-evaluation-sample-v2\0")
    digest.update(bytes.fromhex(input_identity_sha256))
    digest.update(struct.pack("<f", float(sampled_t)))
    digest.update(bytes.fromhex(corruption_uniform_sha256))
    digest.update(bytes.fromhex(corruption_mask_sha256))
    return digest.hexdigest()


def _ordered_sha256_root(domain: bytes, row_digests: list[str]) -> str:
    digest = hashlib.sha256(domain + b"\0")
    digest.update(struct.pack("<Q", len(row_digests)))
    for row_digest in row_digests:
        digest.update(bytes.fromhex(row_digest))
    return digest.hexdigest()


class JudgeModel:
    def __init__(self, device, model_id: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.model_id = model_id
        print(f"Loading judge model: {model_id}")
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._tok.pad_token = self._tok.eos_token
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()

    @torch.no_grad()
    def score(self, texts: list[str]) -> float:
        enc = self._tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        labels = ids.clone()
        labels[mask == 0] = -100
        out = self._model(input_ids=ids, attention_mask=mask, labels=labels)
        return out.loss.item()


_judge_singleton: JudgeModel | None = None
_judge_device: torch.device | None = None
_judge_model_id: str | None = None


def _get_judge(device, model_id: str) -> JudgeModel:
    global _judge_singleton, _judge_device, _judge_model_id
    requested = torch.device(device)
    if _judge_singleton is None or _judge_device != requested or _judge_model_id != model_id:
        _judge_singleton = JudgeModel(requested, model_id)
        _judge_device = requested
        _judge_model_id = model_id
    return _judge_singleton


def _model_uses_noise_level(model) -> bool:
    config = getattr(model, "config", None)
    return bool(getattr(config, "time_conditioning", False))


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def model_uses_target_length_conditioning(model) -> bool:
    config = getattr(_unwrap_model(model), "config", None)
    return bool(getattr(config, "target_length_conditioning", False))


def target_eos_lengths(
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None,
) -> torch.Tensor:
    if x_0.shape != attention_mask.shape:
        raise ValueError(f"x_0 and attention_mask shapes must match, got {x_0.shape} and {attention_mask.shape}")
    batch_size, seq_len = x_0.shape
    valid_lengths = attention_mask.to(torch.long).sum(dim=1).clamp_min(1)
    if eos_token_id is None:
        return valid_lengths
    positions = torch.arange(seq_len, device=x_0.device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
    eos_hits = (x_0 == int(eos_token_id)) & attention_mask
    first_eos = torch.where(eos_hits, positions, torch.full_like(positions, seq_len)).min(dim=1).values
    return torch.where(first_eos < seq_len, first_eos + 1, valid_lengths).to(torch.long)


def _call_model_logits(
    model,
    z_t: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    model_logits_fn=None,
    noise_level=None,
    position_ids: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if model_logits_fn is not None:
        if target_lengths is not None:
            if position_ids is not None:
                if noise_level is None:
                    return model_logits_fn(
                        model,
                        z_t,
                        x_0,
                        attention_mask,
                        position_ids=position_ids,
                        target_lengths=target_lengths,
                    )
                return model_logits_fn(
                    model,
                    z_t,
                    x_0,
                    attention_mask,
                    noise_level=noise_level,
                    position_ids=position_ids,
                    target_lengths=target_lengths,
                )
            if noise_level is None:
                return model_logits_fn(model, z_t, x_0, attention_mask, target_lengths=target_lengths)
            return model_logits_fn(
                model,
                z_t,
                x_0,
                attention_mask,
                noise_level=noise_level,
                target_lengths=target_lengths,
            )
        if position_ids is not None:
            if noise_level is None:
                return model_logits_fn(model, z_t, x_0, attention_mask, position_ids=position_ids)
            return model_logits_fn(
                model,
                z_t,
                x_0,
                attention_mask,
                noise_level=noise_level,
                position_ids=position_ids,
            )
        if noise_level is None:
            return model_logits_fn(model, z_t, x_0, attention_mask)
        return model_logits_fn(model, z_t, x_0, attention_mask, noise_level=noise_level)
    model_kwargs = {"attention_mask": attention_mask}
    if noise_level is not None and _model_uses_noise_level(model):
        model_kwargs["noise_level"] = noise_level
    if position_ids is not None:
        model_kwargs["position_ids"] = position_ids
    if target_lengths is not None:
        model_kwargs["target_lengths"] = target_lengths
    return model(z_t, **model_kwargs)


def _batch_model_attention_mask(
    batch,
    attention_mask: torch.Tensor | None = None,
    query_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    base_mask = batch.attention_mask if attention_mask is None else attention_mask
    boundary_mask = getattr(batch, "document_boundary_mask", None)
    if boundary_mask is None:
        return base_mask
    if base_mask.dim() != 2:
        raise ValueError(f"base attention mask must be [batch, seq], got {tuple(base_mask.shape)}")
    b, t = base_mask.shape
    if boundary_mask.shape != (b, t, t):
        raise ValueError(f"document boundary mask must have shape {(b, t, t)}, got {tuple(boundary_mask.shape)}")
    query_mask = batch.attention_mask if query_attention_mask is None else query_attention_mask
    if query_mask.shape != base_mask.shape:
        raise ValueError(
            f"query attention mask must have shape {tuple(base_mask.shape)}, got {tuple(query_mask.shape)}"
        )
    return boundary_mask & query_mask[:, :, None] & base_mask[:, None, :]


def _delete_one_sequence_ratio_se(
    numerators: list[float],
    denominators: list[float],
    *,
    scale: float = 1.0,
) -> float | None:
    """Return a delete-one-sequence jackknife diagnostic for a ratio of totals."""

    if len(numerators) != len(denominators):
        raise ValueError("numerator and denominator sequence counts must match")
    if len(numerators) < 2:
        return None
    total_numerator = math.fsum(numerators)
    total_denominator = math.fsum(denominators)
    leave_one_out = []
    for numerator, denominator in zip(numerators, denominators, strict=True):
        remaining_denominator = total_denominator - denominator
        if remaining_denominator <= 0:
            return None
        leave_one_out.append((total_numerator - numerator) / (scale * remaining_denominator))
    mean = math.fsum(leave_one_out) / len(leave_one_out)
    return math.sqrt(
        (len(leave_one_out) - 1) / len(leave_one_out) * sum((estimate - mean) ** 2 for estimate in leave_one_out)
    )


@torch.no_grad()
def evaluate_nelbo(
    model,
    data_config,
    tokenizer,
    noise_fn,
    batch_size: int,
    device,
    seq_len: int = MAX_SEQ_LEN,
    eval_tokens: int = 40 * 524288,
    model_logits_fn=None,
    schedule_weight: bool = False,
    schedule_weight_scale: float = 1.0,
    return_diagnostics: bool = False,
) -> tuple[float, float, float, bool] | tuple[float, float, float, bool, dict[str, object]]:
    """Estimate the per-byte MDLM diffusion loss.

    Returns ``(nelbo_bpb, masked_ce_bpb, weight_mean, schedule_weighted)``.
    With ``return_diagnostics=True``, an additional diagnostics dictionary
    contains delete-one-sequence ratio-jackknife diagnostics and sample counts.

    - ``masked_ce_bpb`` is the unweighted mean masked-token CE per byte (the
      historical ``val_nelbo_bpb`` value). It is NOT a valid likelihood bound.
    - When ``schedule_weight`` is true, ``nelbo_bpb`` reweights each masked CE
      by ``schedule_weight_scale / t``. The scale is the integration width when
      times are drawn uniformly from a subinterval. A scalar or ``[batch]``
      time is the shared-time absorbing-MDLM
      estimator. A ``[batch, seq]`` time is an independent-token-time objective
      and is generally different when logits depend on the joint mask pattern.
      The caller remains responsible for matching the time proposal, integration
      range, and importance weight before interpreting the result as a bound.
      When ``schedule_weight`` is false, ``nelbo_bpb`` falls back to
      ``masked_ce_bpb`` for compatibility with non-absorbing recipes.
    """
    if not math.isfinite(float(schedule_weight_scale)) or float(schedule_weight_scale) <= 0:
        raise ValueError("schedule_weight_scale must be finite and positive")
    tokens_per_batch = int(batch_size) * int(seq_len)
    if int(batch_size) <= 0 or int(seq_len) <= 0 or int(eval_tokens) <= 0 or int(eval_tokens) % tokens_per_batch != 0:
        raise ValueError("eval_tokens must be a positive exact multiple of batch_size * seq_len")
    token_bytes = build_token_bytes(tokenizer).to(device)
    val_loader = make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)
    steps = int(eval_tokens) // tokens_per_batch
    sequence_nelbo_nats: list[float] = []
    sequence_masked_nats: list[float] = []
    sequence_bytes: list[float] = []
    sequence_records: list[dict[str, float | int | None]] = []
    sequence_weight_sums: list[float] = []
    sequence_weight_counts: list[int] = []
    input_identity_digests: list[str] = []
    corruption_uniform_digests: list[str] = []
    corruption_mask_digests: list[str] = []
    evaluation_sample_digests: list[str] = []

    for _ in range(steps):
        batch = next(val_loader)
        x_0 = batch.x_0
        noise_out = noise_fn(x_0, batch.loss_mask)
        noise_audit: dict[str, object] = {}
        if len(noise_out) == 3:
            z_t, _t, loss_weight = noise_out
            noise_level = None
        elif len(noise_out) == 4:
            z_t, _t, loss_weight, noise_level = noise_out
        elif len(noise_out) == 5:
            z_t, _t, loss_weight, noise_level, raw_noise_audit = noise_out
            if not isinstance(raw_noise_audit, dict):
                raise TypeError("the fifth noise_fn result must be an audit dictionary")
            noise_audit = raw_noise_audit
        else:
            raise ValueError(f"noise_fn must return 3, 4, or 5 values, got {len(noise_out)}")
        model_attention_mask = _batch_model_attention_mask(batch)
        logits = _call_model_logits(
            model,
            z_t,
            x_0,
            model_attention_mask,
            model_logits_fn=model_logits_fn,
            noise_level=noise_level,
            position_ids=getattr(batch, "position_ids", None),
        )
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), x_0.view(-1), reduction="none").view(x_0.shape)
        byte_lens = token_bytes[x_0]
        content_mask = (byte_lens > 0) & batch.attention_mask & batch.loss_mask
        masked_term = ce.float() * loss_weight * content_mask
        row_masked = masked_term.sum(dim=1)
        if schedule_weight and _t is not None:
            weighted_terms = schedule_weighted_mdlm_terms(
                ce.float(),
                loss_weight,
                _t.detach() if isinstance(_t, torch.Tensor) else _t,
                lossable_mask=content_mask,
            ) * float(schedule_weight_scale)
            row_nelbo = weighted_terms.sum(dim=1)
        else:
            row_nelbo = row_masked
        row_bytes = (byte_lens * batch.attention_mask).sum(dim=1)

        active = (loss_weight > 0) & content_mask
        row_lossable_content_count = content_mask.sum(dim=1)
        row_masked_target_count = active.sum(dim=1)
        expanded_times = (
            expand_mdlm_time(_t.detach() if isinstance(_t, torch.Tensor) else _t, masked_term)
            if _t is not None
            else None
        )
        if schedule_weight and _t is not None:
            assert expanded_times is not None
            safe_times = torch.where(active, expanded_times, torch.ones_like(expanded_times))
            row_weight_sum = torch.where(
                active,
                float(schedule_weight_scale) / safe_times,
                torch.zeros_like(safe_times),
            ).sum(dim=1)
            row_weight_count = row_masked_target_count
        else:
            row_weight_sum = torch.zeros_like(row_nelbo)
            row_weight_count = torch.zeros_like(row_masked_target_count)

        if expanded_times is None:
            row_sampled_t = torch.full_like(row_nelbo, torch.nan)
            row_sampled_t_min = torch.full_like(row_nelbo, torch.nan)
            row_sampled_t_max = torch.full_like(row_nelbo, torch.nan)
        else:
            has_content = content_mask.any(dim=1)
            positive_inf = torch.full_like(expanded_times, torch.inf)
            negative_inf = torch.full_like(expanded_times, -torch.inf)
            content_t_min = torch.where(content_mask, expanded_times, positive_inf).min(dim=1).values
            content_t_max = torch.where(content_mask, expanded_times, negative_inf).max(dim=1).values
            row_sampled_t_min = torch.where(has_content, content_t_min, expanded_times.min(dim=1).values)
            row_sampled_t_max = torch.where(has_content, content_t_max, expanded_times.max(dim=1).values)
            row_sampled_t = torch.where(
                row_sampled_t_min == row_sampled_t_max,
                row_sampled_t_min,
                torch.full_like(row_sampled_t_min, torch.nan),
            )

        row_stats = (
            torch.stack(
                (
                    row_nelbo,
                    row_masked,
                    row_bytes.to(dtype=row_nelbo.dtype),
                    row_lossable_content_count.to(dtype=row_nelbo.dtype),
                    row_masked_target_count.to(dtype=row_nelbo.dtype),
                    row_weight_sum,
                    row_weight_count.to(dtype=row_nelbo.dtype),
                    row_sampled_t,
                    row_sampled_t_min,
                    row_sampled_t_max,
                ),
                dim=1,
            )
            .detach()
            .to(device="cpu", dtype=torch.float64)
        )

        identity_integer_rows: torch.Tensor | None = None
        identity_mask_rows: torch.Tensor | None = None
        uniform_rows: torch.Tensor | None = None
        has_position_ids = getattr(batch, "position_ids", None) is not None
        has_document_start_mask = getattr(batch, "document_start_mask", None) is not None
        if return_diagnostics:
            if (
                getattr(batch, "document_boundary_mask", None) is not None
                and not has_position_ids
                and not has_document_start_mask
            ):
                raise ValueError(
                    "NELBO identity telemetry cannot compactly identify a document boundary mask "
                    "without position_ids or document_start_mask"
                )
            integer_components = [x_0]
            if has_position_ids:
                integer_components.append(batch.position_ids)
            identity_integer_rows = torch.stack(integer_components, dim=1).detach().to(device="cpu", dtype=torch.int64)
            mask_components = [batch.attention_mask, batch.loss_mask, content_mask, active]
            if has_document_start_mask:
                mask_components.append(batch.document_start_mask)
            identity_mask_rows = torch.stack(mask_components, dim=1).detach().to(device="cpu", dtype=torch.uint8)
            raw_uniform_rows = noise_audit.get("corruption_uniforms")
            if raw_uniform_rows is not None:
                if not isinstance(raw_uniform_rows, torch.Tensor):
                    raise TypeError("noise audit corruption_uniforms must be a tensor")
                if raw_uniform_rows.shape != x_0.shape:
                    raise ValueError(
                        "noise audit corruption_uniforms must match x_0 shape, "
                        f"got {tuple(raw_uniform_rows.shape)} and {tuple(x_0.shape)}"
                    )
                uniform_rows = raw_uniform_rows.detach().to(device="cpu", dtype=torch.float32)

        for row_offset, stats in enumerate(row_stats.tolist()):
            (
                row_nelbo_numerator,
                row_masked_numerator,
                row_denominator,
                row_lossable_count,
                row_active_count,
                row_importance_weight_sum,
                row_schedule_weight_count,
                sampled_t_value,
                sampled_t_min_value,
                sampled_t_max_value,
            ) = stats
            sequence_nelbo_nats.append(row_nelbo_numerator)
            sequence_masked_nats.append(row_masked_numerator)
            sequence_bytes.append(row_denominator)
            sequence_weight_sums.append(row_importance_weight_sum)
            sequence_weight_counts.append(int(row_schedule_weight_count))

            input_identity_sha256: str | None = None
            corruption_uniform_sha256: str | None = None
            corruption_mask_sha256: str | None = None
            evaluation_sample_sha256: str | None = None
            if return_diagnostics:
                assert identity_integer_rows is not None
                assert identity_mask_rows is not None
                input_identity_sha256 = _nelbo_input_identity_sha256(
                    identity_integer_rows[row_offset, 0],
                    identity_mask_rows[row_offset, 0],
                    identity_mask_rows[row_offset, 1],
                    identity_integer_rows[row_offset, 1] if has_position_ids else None,
                    identity_mask_rows[row_offset, 4] if has_document_start_mask else None,
                )
                if uniform_rows is not None:
                    expected_active = (uniform_rows[row_offset] < float(sampled_t_value)) & identity_mask_rows[
                        row_offset, 2
                    ].bool()
                    if not torch.equal(expected_active, identity_mask_rows[row_offset, 3].bool()):
                        raise ValueError(
                            f"NELBO corruption audit does not reconstruct active mask for sequence "
                            f"{len(sequence_records)}"
                        )
                corruption_mask_sha256 = _nelbo_corruption_mask_sha256(identity_mask_rows[row_offset, 3])
                input_identity_digests.append(input_identity_sha256)
                corruption_mask_digests.append(corruption_mask_sha256)
                if uniform_rows is not None:
                    corruption_uniform_sha256 = _nelbo_uniform_sha256(uniform_rows[row_offset])
                    corruption_uniform_digests.append(corruption_uniform_sha256)
                if math.isfinite(sampled_t_value) and corruption_uniform_sha256 is not None:
                    evaluation_sample_sha256 = _nelbo_evaluation_sample_sha256(
                        input_identity_sha256,
                        sampled_t_value,
                        corruption_uniform_sha256,
                        corruption_mask_sha256,
                    )
                    evaluation_sample_digests.append(evaluation_sample_sha256)

            sequence_records.append(
                {
                    "schema_version": NELBO_SEQUENCE_RECORD_SCHEMA_VERSION,
                    "sequence_index": len(sequence_records),
                    "sampled_t": sampled_t_value if math.isfinite(sampled_t_value) else None,
                    "sampled_t_min": sampled_t_min_value if math.isfinite(sampled_t_min_value) else None,
                    "sampled_t_max": sampled_t_max_value if math.isfinite(sampled_t_max_value) else None,
                    "nelbo_nats_numerator": row_nelbo_numerator,
                    "masked_nats_numerator": row_masked_numerator,
                    "byte_denominator": row_denominator,
                    "lossable_content_target_count": int(row_lossable_count),
                    "masked_target_count": int(row_active_count),
                    "schedule_weight_sum": row_importance_weight_sum,
                    "schedule_weight_count": int(row_schedule_weight_count),
                    "input_identity_sha256": input_identity_sha256,
                    "corruption_uniform_sha256": corruption_uniform_sha256,
                    "corruption_mask_sha256": corruption_mask_sha256,
                    "evaluation_sample_sha256": evaluation_sample_sha256,
                }
            )

    total_nelbo_nats = math.fsum(sequence_nelbo_nats)
    total_masked_nats = math.fsum(sequence_masked_nats)
    total_bytes = math.fsum(sequence_bytes)
    masked_target_count = sum(int(record["masked_target_count"]) for record in sequence_records)
    weight_sum = math.fsum(sequence_weight_sums)
    weight_count = sum(sequence_weight_counts)
    denom = math.log(2) * total_bytes
    nelbo_bpb = total_nelbo_nats / denom if total_bytes else 0.0
    masked_ce_bpb = total_masked_nats / denom if total_bytes else 0.0
    weight_mean = weight_sum / weight_count if weight_count else 0.0
    schedule_weighted = bool(schedule_weight and weight_count > 0)
    result = (nelbo_bpb, masked_ce_bpb, weight_mean, schedule_weighted)
    if not return_diagnostics:
        return result
    diagnostics: dict[str, object] = {
        "nelbo_bpb_se": _delete_one_sequence_ratio_se(sequence_nelbo_nats, sequence_bytes, scale=math.log(2)),
        "masked_ce_bpb_se": _delete_one_sequence_ratio_se(sequence_masked_nats, sequence_bytes, scale=math.log(2)),
        "se_method": "delete_one_sequence_jackknife_ratio_conditional_on_fixed_time_grid",
        "se_unit": "sequence",
        "block_count": len(sequence_bytes),
        "sequence_count": len(sequence_bytes),
        "total_bytes": int(total_bytes),
        "masked_target_count": masked_target_count,
        "schedule_weight_count": weight_count,
        "schedule_weight_scale": float(schedule_weight_scale),
        "input_identity_root_sha256": _ordered_sha256_root(
            b"nanodllm-nelbo-input-identity-root-v2", input_identity_digests
        ),
        "corruption_uniform_root_sha256": _ordered_sha256_root(
            b"nanodllm-nelbo-corruption-uniform-root-v1", corruption_uniform_digests
        )
        if len(corruption_uniform_digests) == len(sequence_records)
        else None,
        "corruption_mask_root_sha256": _ordered_sha256_root(
            b"nanodllm-nelbo-corruption-mask-root-v2", corruption_mask_digests
        ),
        "evaluation_sample_root_sha256": _ordered_sha256_root(
            b"nanodllm-nelbo-evaluation-sample-root-v2", evaluation_sample_digests
        )
        if len(evaluation_sample_digests) == len(sequence_records)
        else None,
        "sequence_records": sequence_records,
    }
    return (*result, diagnostics)


@torch.no_grad()
def evaluate_gen_ppl(
    model,
    tokenizer,
    sample_fn,
    batch_size: int,
    run_dir: str | None = None,
    device=None,
    judge_model_id: str = "Qwen/Qwen3-0.6B-Base",
    gen_samples: int = 512,
    gen_len: int = 512,
) -> float:
    device = next(model.parameters()).device if device is None else device
    judge = _get_judge(device, judge_model_id)
    n_batches = max(1, gen_samples // batch_size)
    total_ce = 0.0

    with ExitStack() as stack:
        log_file = None
        if run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            log_file = stack.enter_context(open(os.path.join(run_dir, "gen_samples.log"), "w"))

        def log(msg: str):
            if log_file is not None:
                log_file.write(msg + "\n")
                log_file.flush()
            else:
                print(msg)

        logged_samples = 0
        for batch_idx in range(n_batches):
            tokens = sample_fn(model, batch_size, gen_len)
            texts = [tokenizer.decode(tokens[i].tolist()) for i in range(tokens.shape[0])]
            if logged_samples < 5:
                for i in range(min(batch_size, 5 - logged_samples)):
                    text = texts[i]
                    judge_len = judge._tok(text, return_tensors="pt", truncation=True, max_length=512)[
                        "input_ids"
                    ].shape[1]
                    log(f"[gen] batch={batch_idx} item={i} judge_len={judge_len}")
                    log(f"[gen] text={text[:300]!r}")
                    logged_samples += 1

            empty = sum(1 for txt in texts if len(txt.strip()) == 0)
            if empty:
                log(f"[gen] WARNING: batch {batch_idx} has {empty}/{batch_size} empty decoded strings")
            batch_ce = judge.score(texts)
            log(f"[gen] batch={batch_idx} batch_ce={batch_ce:.4f} batch_ppl={math.exp(batch_ce):.2f}")
            total_ce += batch_ce

    return math.exp(total_ce / n_batches)


def _truncate_at_eos(ids: list[int], eos_id: int | None) -> tuple[list[int], bool]:
    if eos_id is None or eos_id not in ids:
        return ids, False
    eos_pos = ids.index(eos_id)
    return ids[: eos_pos + 1], True


def _content_ids(ids: list[int], special_ids: set[int], eos_id: int | None) -> list[int]:
    truncated, _ = _truncate_at_eos(ids, eos_id)
    return [int(i) for i in truncated if int(i) not in special_ids]


def _token_byte_total(ids: list[int], token_bytes: torch.Tensor) -> int:
    vocab_size = int(token_bytes.numel())
    total = 0
    for token_id in ids:
        token_id = int(token_id)
        if 0 <= token_id < vocab_size:
            total += int(token_bytes[token_id].item())
    return total


def _distinct_ngram_ratio(sequences: list[list[int]], n: int) -> float:
    total = 0
    unique: set[tuple[int, ...]] = set()
    for ids in sequences:
        if len(ids) < n:
            continue
        for idx in range(len(ids) - n + 1):
            unique.add(tuple(ids[idx : idx + n]))
            total += 1
    return len(unique) / total if total else 0.0


def _repeat_ngram_rate(sequences: list[list[int]], n: int) -> float:
    rates = []
    for ids in sequences:
        if len(ids) < n:
            rates.append(0.0)
            continue
        seen: set[tuple[int, ...]] = set()
        repeats = 0
        total = 0
        for idx in range(len(ids) - n + 1):
            ngram = tuple(ids[idx : idx + n])
            repeats += int(ngram in seen)
            seen.add(ngram)
            total += 1
        rates.append(repeats / total if total else 0.0)
    return sum(rates) / len(rates) if rates else 0.0


def _max_token_fraction(ids: list[int]) -> float:
    if not ids:
        return 0.0
    return max(Counter(ids).values()) / len(ids)


def _token_entropy(ids: list[int]) -> float:
    if not ids:
        return 0.0
    counts = Counter(ids)
    denom = len(ids)
    return -sum((count / denom) * math.log(count / denom) for count in counts.values())


def _entropy_std(entropies: list[float]) -> float:
    if not entropies:
        return 0.0
    mean = sum(entropies) / len(entropies)
    return math.sqrt(sum((value - mean) ** 2 for value in entropies) / len(entropies))


def _eos_position_metrics(
    prefix: str,
    eos_position_counts: Counter[int],
    sample_count: int,
    max_len: int,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        f"{prefix}eos_position_observed": int(sum(eos_position_counts.values())),
        f"{prefix}eos_position_max_len": int(max_len),
    }
    for pos in range(max(0, int(max_len))):
        count = int(eos_position_counts.get(pos, 0))
        metrics[f"{prefix}eos_position_{pos:04d}_count"] = count
        metrics[f"{prefix}eos_position_{pos:04d}_rate"] = count / sample_count if sample_count else 0.0
    return metrics


def _eos_block_metrics(
    prefix: str,
    eos_block_counts: Counter[int],
    sample_count: int,
    num_blocks: int,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        f"{prefix}eos_block_observed": int(sum(eos_block_counts.values())),
        f"{prefix}eos_block_count": int(num_blocks),
    }
    for block_idx in range(max(0, int(num_blocks))):
        count = int(eos_block_counts.get(block_idx, 0))
        metrics[f"{prefix}eos_block_{block_idx}_count"] = count
        metrics[f"{prefix}eos_block_{block_idx}_rate"] = count / sample_count if sample_count else 0.0
    return metrics


def _new_eos_suffix_decile_stats(num_deciles: int = 7) -> dict[str, list[float]]:
    return {
        "counts": [0.0 for _ in range(int(num_deciles))],
        "correct": [0.0 for _ in range(int(num_deciles))],
        "rank_sums": [0.0 for _ in range(int(num_deciles))],
        "prob_sums": [0.0 for _ in range(int(num_deciles))],
    }


def _update_eos_suffix_decile_stats(
    stats: dict[str, list[float]],
    logits: torch.Tensor,
    pred: torch.Tensor,
    x_0: torch.Tensor,
    active: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_id: int | None,
) -> None:
    if eos_id is None:
        return
    eos_token_id = int(eos_id)
    if eos_token_id < 0 or eos_token_id >= logits.shape[-1]:
        return
    eos_active = active & (x_0 == eos_token_id)
    if not eos_active.any():
        return
    rows, cols = eos_active.nonzero(as_tuple=True)
    valid_lengths = attention_mask.to(torch.long).sum(dim=1).clamp_min(1)
    denom = valid_lengths.index_select(0, rows).sub(1).clamp_min(1)
    num_deciles = len(stats["counts"])
    deciles = torch.clamp((cols.float() / denom.float() * num_deciles).to(torch.long), max=num_deciles - 1)
    selected_logits = logits[rows, cols].float()
    eos_logits = selected_logits[:, eos_token_id]
    ranks = (selected_logits > eos_logits[:, None]).sum(dim=-1).to(torch.float32) + 1.0
    eos_probs = F.softmax(selected_logits, dim=-1)[:, eos_token_id]
    correct = (pred[rows, cols] == eos_token_id).to(torch.float32)
    for decile_idx in range(num_deciles):
        decile_mask = deciles == decile_idx
        if not decile_mask.any():
            continue
        count = float(decile_mask.sum().item())
        stats["counts"][decile_idx] += count
        stats["correct"][decile_idx] += float(correct[decile_mask].sum().item())
        stats["rank_sums"][decile_idx] += float(ranks[decile_mask].sum().item())
        stats["prob_sums"][decile_idx] += float(eos_probs[decile_mask].sum().item())


def _eos_suffix_decile_metrics(prefix: str, stats: dict[str, list[float]]) -> dict[str, float | int]:
    counts = stats["counts"]
    correct = stats["correct"]
    rank_sums = stats["rank_sums"]
    prob_sums = stats["prob_sums"]
    total_count = float(sum(counts))
    metrics: dict[str, float | int] = {
        f"{prefix}eos_active_count": int(total_count),
        f"{prefix}eos_token_acc": sum(correct) / total_count if total_count else 0.0,
        f"{prefix}eos_rank_mean": sum(rank_sums) / total_count if total_count else 0.0,
        f"{prefix}eos_prob_mean": sum(prob_sums) / total_count if total_count else 0.0,
    }
    for decile_idx, count in enumerate(counts, start=1):
        metrics[f"{prefix}eos_suffix_decile_{decile_idx}_count"] = int(count)
        metrics[f"{prefix}eos_suffix_decile_{decile_idx}_token_acc"] = correct[decile_idx - 1] / count if count else 0.0
        metrics[f"{prefix}eos_suffix_decile_{decile_idx}_rank_mean"] = (
            rank_sums[decile_idx - 1] / count if count else 0.0
        )
        metrics[f"{prefix}eos_suffix_decile_{decile_idx}_prob_mean"] = (
            prob_sums[decile_idx - 1] / count if count else 0.0
        )
    return metrics


def _ngram_entropy(sequences: list[list[int]], n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    counts: Counter[tuple[int, ...]] = Counter()
    total = 0
    for ids in sequences:
        if len(ids) < n:
            continue
        for idx in range(len(ids) - n + 1):
            counts[tuple(ids[idx : idx + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


_PRIOR_CORRECTION_CACHE: dict[tuple, tuple[torch.Tensor, dict[str, float | int | str]]] = {}


def _single_token_text(tokenizer, token_id: int) -> str:
    if token_id < 0:
        return ""
    return tokenizer.decode([int(token_id)])


def _load_generation_prior_settings(run_dir: str | None) -> dict | None:
    if run_dir is None:
        return None
    config_path = os.path.join(run_dir, "resolved_config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    eval_cfg = payload.get("eval", {})
    train_cfg = payload.get("train", {})
    data_cfg = payload.get("data", {})
    if not isinstance(eval_cfg, dict) or not isinstance(train_cfg, dict) or not isinstance(data_cfg, dict):
        return None
    strength = float(eval_cfg.get("gen_prior_correction_strength", 0.0))
    batches = int(eval_cfg.get("gen_prior_correction_batches", 0))
    smoothing = float(eval_cfg.get("gen_prior_correction_smoothing", 1.0))
    if strength <= 0 or not bool(train_cfg.get("eval_only", False)):
        return None
    if batches <= 0:
        raise ValueError("eval.gen_prior_correction_batches must be positive when prior correction is enabled")
    return {
        "config_path": config_path,
        "data_config": DataConfig(**data_cfg),
        "strength": strength,
        "batches": batches,
        "smoothing": smoothing,
    }


@torch.no_grad()
def _estimate_unigram_prior_logit_bias(
    data_config,
    tokenizer,
    batch_size: int,
    device,
    seq_len: int,
    batches: int,
    strength: float,
    smoothing: float,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    vocab_size = tokenizer.get_vocab_size()
    counts = torch.full((vocab_size,), float(smoothing), dtype=torch.float64)
    token_bytes = build_token_bytes(tokenizer).to(device)
    loader = make_dataloader(data_config, tokenizer, batch_size, seq_len, "train", device=device)
    special_ids = tokenizer.get_special_token_ids()
    observed_tokens = 0

    for _ in range(int(batches)):
        batch = next(loader)
        content = batch.attention_mask & batch.loss_mask & (token_bytes[batch.x_0] > 0)
        for token_id in special_ids:
            if 0 <= int(token_id) < vocab_size:
                content &= batch.x_0 != int(token_id)
        if not content.any():
            continue
        ids = batch.x_0[content].detach().cpu()
        counts += torch.bincount(ids, minlength=vocab_size).to(dtype=torch.float64)
        observed_tokens += int(ids.numel())

    log_prior = (counts / counts.sum()).log().to(device=device, dtype=torch.float32)
    logit_bias = -float(strength) * log_prior
    zero_bias_count = 0
    zero_byte_ids = (token_bytes.detach().cpu() <= 0).nonzero(as_tuple=True)[0].tolist()
    for token_id in zero_byte_ids:
        if 0 <= int(token_id) < vocab_size:
            logit_bias[int(token_id)] = 0.0
            zero_bias_count += 1

    observed_counts = counts - float(smoothing)
    top_count, top_id = torch.max(observed_counts, dim=0)
    top_token_id = int(top_id.item()) if observed_tokens else -1
    top_token_count = int(top_count.item()) if observed_tokens else 0
    metrics: dict[str, float | int | str] = {
        "gen_prior_correction_strength": float(strength),
        "gen_prior_correction_batches": int(batches),
        "gen_prior_correction_smoothing": float(smoothing),
        "gen_prior_correction_token_count": int(observed_tokens),
        "gen_prior_correction_nonzero_token_types": int((observed_counts > 0).sum().item()),
        "gen_prior_correction_zero_bias_token_types": int(zero_bias_count),
        "gen_prior_correction_top_token_id": top_token_id,
        "gen_prior_correction_top_token_text": _single_token_text(tokenizer, top_token_id),
        "gen_prior_correction_top_token_count": top_token_count,
        "gen_prior_correction_top_token_fraction": top_token_count / observed_tokens if observed_tokens else 0.0,
    }
    return logit_bias, metrics


def _generation_prior_correction(
    run_dir: str | None,
    tokenizer,
    batch_size: int,
    device,
    seq_len: int,
) -> tuple[torch.Tensor | None, dict[str, float | int | str]]:
    settings = _load_generation_prior_settings(run_dir)
    if settings is None:
        return None, {}

    config_path = str(settings["config_path"])
    strength = float(settings["strength"])
    batches = int(settings["batches"])
    smoothing = float(settings["smoothing"])
    key = (
        config_path,
        os.path.getmtime(config_path),
        str(torch.device(device)),
        tokenizer.get_vocab_size(),
        int(batch_size),
        int(seq_len),
        strength,
        batches,
        smoothing,
    )
    cached = _PRIOR_CORRECTION_CACHE.get(key)
    if cached is None:
        cached = _estimate_unigram_prior_logit_bias(
            settings["data_config"],
            tokenizer,
            batch_size,
            device,
            seq_len,
            batches,
            strength,
            smoothing,
        )
        _PRIOR_CORRECTION_CACHE[key] = cached
    logit_bias, metrics = cached
    return logit_bias, dict(metrics)


def _t_key(t_val: float) -> str:
    return f"t{int(round(float(t_val) * 10)):02d}"


def _attention_row_ids(ids: torch.Tensor, attention_mask: torch.Tensor) -> list[int]:
    row = ids.detach().cpu().tolist()
    keep = attention_mask.detach().cpu().tolist()
    return [int(token_id) for token_id, is_valid in zip(row, keep, strict=True) if bool(is_valid)]


def _decode_attention_row(tokenizer, ids: torch.Tensor, attention_mask: torch.Tensor, limit: int = 500) -> str:
    return tokenizer.decode(_attention_row_ids(ids, attention_mask))[:limit]


def _decode_masked_attention_row(
    tokenizer,
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_id: int,
    eos_id: int | None,
    limit: int = 500,
) -> str:
    pieces: list[str] = []
    segment: list[int] = []

    def flush_segment() -> None:
        nonlocal segment
        if segment:
            pieces.append(tokenizer.decode(segment))
            segment = []

    for token_id in _attention_row_ids(ids, attention_mask):
        if eos_id is not None and token_id == eos_id:
            break
        if token_id == mask_id:
            flush_segment()
            if not pieces or pieces[-1] != "<mask>":
                pieces.append("<mask>")
        else:
            segment.append(token_id)
    flush_segment()
    return "".join(pieces)[:limit]


def _merge_sample_trace(totals: dict[str, float | int | None], trace: dict[str, float | int]) -> None:
    totals["batches"] = int(totals["batches"] or 0) + 1
    totals["steps"] = int(totals["steps"] or 0) + int(trace.get("steps", 0))
    totals["candidate_count"] = int(totals["candidate_count"] or 0) + int(trace.get("candidate_count", 0))
    totals["commit_count"] = int(totals["commit_count"] or 0) + int(trace.get("commit_count", 0))
    totals["candidate_entropy_sum"] = float(totals["candidate_entropy_sum"] or 0.0) + float(
        trace.get("candidate_entropy_sum", 0.0)
    )
    totals["candidate_top1_prob_sum"] = float(totals["candidate_top1_prob_sum"] or 0.0) + float(
        trace.get("candidate_top1_prob_sum", 0.0)
    )
    totals["pred_max_token_frac_sum"] = float(totals["pred_max_token_frac_sum"] or 0.0) + float(
        trace.get("pred_max_token_frac_sum", 0.0)
    )
    totals["pred_max_token_frac_steps"] = int(totals["pred_max_token_frac_steps"] or 0) + int(
        trace.get("pred_max_token_frac_steps", 0)
    )
    for key in (
        "class_entropy_sum",
        "class_max_prob_sum",
        "class_prob_eos_special_sum",
        "class_prob_low_freq_sum",
        "class_prob_high_freq_sum",
    ):
        totals[key] = float(totals.get(key) or 0.0) + float(trace.get(key, 0.0))
    totals["final_mask_count"] = int(totals["final_mask_count"] or 0) + int(trace.get("final_mask_count", 0))
    totals["remask_count"] = int(totals.get("remask_count") or 0) + int(trace.get("remask_count", 0))
    totals["num_blocks"] = int(totals.get("num_blocks") or 0) + int(trace.get("num_blocks", 0))
    totals["nfe_per_block"] = int(totals.get("nfe_per_block") or 0) + int(trace.get("nfe_per_block", 0))
    for key, value in trace.items():
        if not key.startswith("length_policy_") or not isinstance(value, (int, float)):
            continue
        if key.endswith("_position_max_len"):
            totals[key] = max(int(totals.get(key) or 0), int(value))
        else:
            totals[key] = float(totals.get(key) or 0.0) + float(value)

    commit_min = int(trace.get("commit_count_min", 0))
    commit_max = int(trace.get("commit_count_max", 0))
    totals["commit_count_min"] = (
        commit_min if totals["commit_count_min"] is None else min(int(totals["commit_count_min"]), commit_min)
    )
    totals["commit_count_max"] = (
        commit_max if totals["commit_count_max"] is None else max(int(totals["commit_count_max"]), commit_max)
    )


def _new_sample_trace_totals() -> dict[str, float | int | None]:
    return {
        "batches": 0,
        "steps": 0,
        "candidate_count": 0,
        "commit_count": 0,
        "commit_count_min": None,
        "commit_count_max": None,
        "candidate_entropy_sum": 0.0,
        "candidate_top1_prob_sum": 0.0,
        "pred_max_token_frac_sum": 0.0,
        "pred_max_token_frac_steps": 0,
        "class_entropy_sum": 0.0,
        "class_max_prob_sum": 0.0,
        "class_prob_eos_special_sum": 0.0,
        "class_prob_low_freq_sum": 0.0,
        "class_prob_high_freq_sum": 0.0,
        "final_mask_count": 0,
        "remask_count": 0,
        "num_blocks": 0,
        "nfe_per_block": 0,
    }


def _target_length_generation_metrics(
    target_lengths: list[int],
    pre_eos_positions: list[int],
    post_eos_positions: list[int],
    forced_eos_count: int,
    gen_len: int,
) -> dict[str, float | int]:
    if not target_lengths:
        return {
            "gen_target_length_conditioning_applied": 0,
            "gen_target_length_sample_count": 0,
            "gen_target_length_pre_policy_eos_count": 0,
            "gen_target_length_post_policy_eos_count": 0,
            "gen_target_length_forced_eos_count": 0,
            "gen_target_length_no_forced_eos_success_count": 0,
        }
    sample_count = len(target_lengths)
    pre_errors = [
        abs(int(pos) - (int(target_length) - 1))
        for target_length, pos in zip(target_lengths, pre_eos_positions, strict=True)
        if int(pos) >= 0
    ]
    post_errors = [
        abs(int(pos) - (int(target_length) - 1))
        for target_length, pos in zip(target_lengths, post_eos_positions, strict=True)
        if int(pos) >= 0
    ]

    def median(values: list[int]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[mid])
        return 0.5 * float(ordered[mid - 1] + ordered[mid])

    pre_eos_count = sum(1 for pos in pre_eos_positions if int(pos) >= 0)
    post_eos_count = sum(1 for pos in post_eos_positions if int(pos) >= 0)
    target_counter = Counter(int(v) for v in target_lengths)
    metrics: dict[str, float | int] = {
        "gen_target_length_conditioning_applied": 1,
        "gen_target_length_sample_count": int(sample_count),
        "gen_target_length_mean": sum(target_lengths) / sample_count,
        "gen_target_length_min": int(min(target_lengths)),
        "gen_target_length_max": int(max(target_lengths)),
        "gen_target_length_hist_nonzero_bins": int(len(target_counter)),
        "gen_target_length_pre_policy_eos_count": int(pre_eos_count),
        "gen_target_length_pre_policy_eos_rate": pre_eos_count / sample_count,
        "gen_target_length_post_policy_eos_count": int(post_eos_count),
        "gen_target_length_post_policy_eos_rate": post_eos_count / sample_count,
        "gen_target_length_forced_eos_count": int(forced_eos_count),
        "gen_target_length_forced_eos_rate": forced_eos_count / sample_count,
        "gen_target_length_no_forced_eos_success_count": int(pre_eos_count),
        "gen_target_length_pre_policy_eos_abs_error_mean": sum(pre_errors) / len(pre_errors) if pre_errors else 0.0,
        "gen_target_length_pre_policy_eos_abs_error_median": median(pre_errors),
        "gen_target_length_post_policy_eos_abs_error_mean": sum(post_errors) / len(post_errors) if post_errors else 0.0,
        "gen_target_length_post_policy_eos_abs_error_median": median(post_errors),
        "gen_target_length_position_max_len": int(gen_len),
    }
    for target_length, count in sorted(target_counter.items()):
        metrics[f"gen_target_length_length_{target_length:04d}_count"] = int(count)
        metrics[f"gen_target_length_length_{target_length:04d}_rate"] = count / sample_count
    return metrics


def _sample_trace_metrics(totals: dict[str, float | int | None]) -> dict[str, float | int]:
    batches = int(totals["batches"] or 0)
    steps = int(totals["steps"] or 0)
    candidate_count = int(totals["candidate_count"] or 0)
    pred_frac_steps = int(totals["pred_max_token_frac_steps"] or 0)
    num_blocks = int(totals.get("num_blocks") or 0)
    nfe_per_block = int(totals.get("nfe_per_block") or 0)
    metrics: dict[str, float | int] = {
        "gen_trace_batches": batches,
        "gen_trace_steps_mean": steps / batches if batches else 0.0,
        "gen_trace_candidate_tokens": candidate_count,
        "gen_trace_commit_tokens": int(totals["commit_count"] or 0),
        "gen_trace_commit_tokens_mean": int(totals["commit_count"] or 0) / steps if steps else 0.0,
        "gen_trace_commit_tokens_min": int(totals["commit_count_min"] or 0),
        "gen_trace_commit_tokens_max": int(totals["commit_count_max"] or 0),
        "gen_trace_candidate_entropy_mean": float(totals["candidate_entropy_sum"] or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_candidate_top1_prob_mean": float(totals["candidate_top1_prob_sum"] or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_pred_max_token_frac_mean": float(totals["pred_max_token_frac_sum"] or 0.0) / pred_frac_steps
        if pred_frac_steps
        else 0.0,
        "gen_trace_class_entropy_mean": float(totals.get("class_entropy_sum") or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_class_max_prob_mean": float(totals.get("class_max_prob_sum") or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_class_prob_eos_special_mean": float(totals.get("class_prob_eos_special_sum") or 0.0)
        / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_class_prob_low_freq_mean": float(totals.get("class_prob_low_freq_sum") or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_class_prob_high_freq_mean": float(totals.get("class_prob_high_freq_sum") or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_final_mask_count": int(totals["final_mask_count"] or 0),
        "gen_trace_final_mask_count_mean": int(totals["final_mask_count"] or 0) / batches if batches else 0.0,
        "gen_trace_remask_count": int(totals.get("remask_count") or 0),
        "gen_trace_remask_tokens_mean": int(totals.get("remask_count") or 0) / steps if steps else 0.0,
        "gen_trace_num_blocks": num_blocks,
        "gen_trace_num_blocks_mean": num_blocks / batches if batches else 0.0,
        "gen_trace_nfe_per_block": nfe_per_block / batches if batches else 0.0,
    }
    for key, value in totals.items():
        if not key.startswith("length_policy_") or value is None:
            continue
        metric_key = f"gen_{key}"
        if isinstance(value, int) or float(value).is_integer():
            metrics[metric_key] = int(value)
        else:
            metrics[metric_key] = float(value)
    count = int(metrics.get("gen_length_policy_predicted_position_count", 0))
    if count:
        metrics["gen_length_policy_predicted_position_mean"] = (
            float(metrics.get("gen_length_policy_predicted_position_sum", 0)) / count
        )
        metrics["gen_length_policy_predicted_length_mean"] = (
            float(metrics.get("gen_length_policy_predicted_length_sum", 0)) / count
        )
    return metrics


@torch.no_grad()
def _self_reconstruction_ce(
    model,
    tokenizer,
    noise,
    tokens: torch.Tensor,
    device,
    t_val: float,
    model_logits_fn=None,
) -> tuple[float, int]:
    eos_id = tokenizer.get_eos_token_id()
    special_ids = tokenizer.get_special_token_ids()
    token_bytes = build_token_bytes(tokenizer).to(device)
    attention_mask = torch.ones_like(tokens, dtype=torch.bool, device=device)
    if eos_id is not None:
        eos_hits = tokens == int(eos_id)
        for row_idx in range(tokens.shape[0]):
            positions = eos_hits[row_idx].nonzero(as_tuple=True)[0]
            if positions.numel() > 0:
                attention_mask[row_idx, int(positions[0].item()) + 1 :] = False

    score_mask = attention_mask & (token_bytes[tokens] > 0)
    for token_id in special_ids:
        if 0 <= int(token_id) < token_bytes.numel():
            score_mask &= tokens != int(token_id)
    if not score_mask.any():
        return 0.0, 0

    if hasattr(noise, "corrupt"):
        z_t, loss_weight = noise.corrupt(tokens, float(t_val), maskable=score_mask)
    else:
        return 0.0, 0
    active = (loss_weight > 0) & score_mask
    if not active.any():
        return 0.0, 0

    logits = _call_model_logits(
        model,
        z_t,
        tokens,
        attention_mask,
        model_logits_fn=model_logits_fn,
        noise_level=float(t_val),
    )
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), tokens.view(-1), reduction="none").view_as(tokens)
    total = float(ce[active].sum().item())
    count = int(active.sum().item())
    return total, count


@torch.no_grad()
def _self_reconstruction_ce_with_score_mask(
    model,
    tokenizer,
    noise,
    tokens: torch.Tensor,
    score_mask: torch.Tensor,
    device,
    t_val: float,
    model_logits_fn=None,
) -> tuple[float, int]:
    eos_id = tokenizer.get_eos_token_id()
    special_ids = tokenizer.get_special_token_ids()
    token_bytes = build_token_bytes(tokenizer).to(device)
    tokens = tokens.to(device=device)
    score_mask = score_mask.to(device=device, dtype=torch.bool)
    attention_mask = torch.ones_like(tokens, dtype=torch.bool, device=device)
    if eos_id is not None:
        eos_hits = tokens == int(eos_id)
        for row_idx in range(tokens.shape[0]):
            positions = eos_hits[row_idx].nonzero(as_tuple=True)[0]
            if positions.numel() > 0:
                attention_mask[row_idx, int(positions[0].item()) + 1 :] = False

    score_mask &= attention_mask & (token_bytes[tokens] > 0)
    for token_id in special_ids:
        if 0 <= int(token_id) < token_bytes.numel():
            score_mask &= tokens != int(token_id)
    if not score_mask.any() or not hasattr(noise, "corrupt"):
        return 0.0, 0

    z_t, loss_weight = noise.corrupt(tokens, float(t_val), maskable=score_mask)
    active = (loss_weight > 0) & score_mask
    if not active.any():
        return 0.0, 0

    logits = _call_model_logits(
        model,
        z_t,
        tokens,
        attention_mask,
        model_logits_fn=model_logits_fn,
        noise_level=float(t_val),
    )
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), tokens.view(-1), reduction="none").view_as(tokens)
    total = float(ce[active].sum().item())
    count = int(active.sum().item())
    return total, count


def _new_target_length_recon_stats(num_deciles: int = 10) -> dict[str, list[float]]:
    return {
        "active_counts": [0.0 for _ in range(int(num_deciles))],
        "active_ce_sums": [0.0 for _ in range(int(num_deciles))],
        "active_correct": [0.0 for _ in range(int(num_deciles))],
        "content_counts": [0.0 for _ in range(int(num_deciles))],
        "content_ce_sums": [0.0 for _ in range(int(num_deciles))],
        "content_correct": [0.0 for _ in range(int(num_deciles))],
    }


def _target_length_decile_ids(target_lengths: torch.Tensor, seq_len: int, num_deciles: int = 10) -> torch.Tensor:
    scaled = (target_lengths.to(torch.long).clamp_min(1) - 1) * int(num_deciles)
    return torch.clamp(torch.div(scaled, max(1, int(seq_len)), rounding_mode="floor"), min=0, max=int(num_deciles) - 1)


def _update_target_length_recon_stats(
    stats: dict[str, list[float]],
    ce: torch.Tensor,
    correct: torch.Tensor,
    active: torch.Tensor,
    content: torch.Tensor,
    target_lengths: torch.Tensor,
) -> None:
    if not active.any():
        return
    num_deciles = len(stats["active_counts"])
    row_deciles = _target_length_decile_ids(target_lengths, ce.shape[1], num_deciles)
    for label, mask, count_key, ce_key, correct_key in (
        ("active", active, "active_counts", "active_ce_sums", "active_correct"),
        ("content", content, "content_counts", "content_ce_sums", "content_correct"),
    ):
        del label
        if not mask.any():
            continue
        rows, _cols = mask.nonzero(as_tuple=True)
        deciles = row_deciles.index_select(0, rows)
        values = ce[mask].detach().float()
        correct_values = correct[mask].detach().float()
        for decile_idx in range(num_deciles):
            decile_mask = deciles == decile_idx
            if not decile_mask.any():
                continue
            count = int(decile_mask.sum().item())
            stats[count_key][decile_idx] += float(count)
            stats[ce_key][decile_idx] += float(values[decile_mask].sum().item())
            stats[correct_key][decile_idx] += float(correct_values[decile_mask].sum().item())


def _target_length_recon_metrics(prefix: str, stats: dict[str, list[float]]) -> dict[str, float | int]:
    num_deciles = len(stats["active_counts"])
    metrics: dict[str, float | int] = {f"{prefix}target_length_decile_count": int(num_deciles)}
    for decile_idx in range(num_deciles):
        label = f"{decile_idx:02d}"
        active_count = float(stats["active_counts"][decile_idx])
        content_count = float(stats["content_counts"][decile_idx])
        metrics[f"{prefix}target_length_decile_{label}_active_tokens"] = int(active_count)
        metrics[f"{prefix}target_length_decile_{label}_ce"] = (
            stats["active_ce_sums"][decile_idx] / active_count if active_count else 0.0
        )
        metrics[f"{prefix}target_length_decile_{label}_token_acc"] = (
            stats["active_correct"][decile_idx] / active_count if active_count else 0.0
        )
        metrics[f"{prefix}target_length_decile_{label}_content_tokens"] = int(content_count)
        metrics[f"{prefix}target_length_decile_{label}_content_ce"] = (
            stats["content_ce_sums"][decile_idx] / content_count if content_count else 0.0
        )
        metrics[f"{prefix}target_length_decile_{label}_content_token_acc"] = (
            stats["content_correct"][decile_idx] / content_count if content_count else 0.0
        )
    return metrics


@torch.no_grad()
def evaluate_reconstruction_metrics(
    model,
    data_config,
    tokenizer,
    noise,
    batch_size: int,
    device,
    seq_len: int = MAX_SEQ_LEN,
    t_values: list[float] | tuple[float, ...] = (0.7, 0.9),
    batches: int = 2,
    run_dir: str | None = None,
    log_samples: int = 3,
    step: int | None = None,
    model_logits_fn=None,
    corrupt_fn=None,
    corruption_label: str = "independent-mdlm",
) -> dict[str, float | int | str]:
    if batches <= 0 or not t_values:
        return {}

    val_loader = make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)
    token_bytes = build_token_bytes(tokenizer).to(device)
    mask_id = int(tokenizer.get_mask_token_id())
    eos_id = tokenizer.get_eos_token_id()
    metrics: dict[str, float | int | str] = {
        "recon_batches": int(batches),
        "recon_log_samples": int(log_samples),
        "recon_corruption": corruption_label,
    }

    with ExitStack() as stack:
        log_file = None
        if run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            log_file = stack.enter_context(open(os.path.join(run_dir, "reconstruction_samples.log"), "a"))

        def log(msg: str) -> None:
            if log_file is not None:
                log_file.write(msg + "\n")
                log_file.flush()
            else:
                print(msg)

        log(
            f"[recon] step={step if step is not None else '-'} "
            f"batches={int(batches)} t_values={','.join(f'{float(t):.2f}' for t in t_values)} "
            f"corruption={corruption_label}"
        )

        for t_val in t_values:
            ce_sum = 0.0
            content_ce_sum = 0.0
            active_count = 0
            content_count = 0
            correct_count = 0
            content_correct_count = 0
            lossable_count = 0
            rows_with_content = 0
            exact_content_rows = 0
            logged = 0
            eos_suffix_stats = _new_eos_suffix_decile_stats()
            target_length_stats = _new_target_length_recon_stats()
            use_target_length_conditioning = model_uses_target_length_conditioning(model)

            for batch_idx in range(int(batches)):
                batch = next(val_loader)
                x_eval = batch.x_0
                target_lengths = (
                    target_eos_lengths(x_eval, batch.attention_mask, eos_id) if use_target_length_conditioning else None
                )
                noise_level = float(t_val)
                if corrupt_fn is None:
                    z_eval, w_eval = noise.corrupt(x_eval, float(t_val), maskable=batch.loss_mask)
                else:
                    corrupt_out = corrupt_fn(x_eval, batch.loss_mask, float(t_val))
                    if len(corrupt_out) == 2:
                        z_eval, w_eval = corrupt_out
                        noise_level = None
                    elif len(corrupt_out) == 3:
                        z_eval, w_eval, noise_level = corrupt_out
                    else:
                        raise ValueError(f"corrupt_fn must return 2 or 3 values, got {len(corrupt_out)}")
                logits = _call_model_logits(
                    model,
                    z_eval,
                    x_eval,
                    _batch_model_attention_mask(batch),
                    model_logits_fn=model_logits_fn,
                    noise_level=noise_level,
                    position_ids=getattr(batch, "position_ids", None),
                    target_lengths=target_lengths,
                )
                pred = logits.argmax(dim=-1)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    x_eval.reshape(-1),
                    reduction="none",
                ).view_as(x_eval)

                active = (w_eval > 0) & batch.loss_mask & batch.attention_mask
                content = active & (token_bytes[x_eval] > 0)
                correct = pred == x_eval
                if target_lengths is not None:
                    _update_target_length_recon_stats(
                        target_length_stats,
                        ce,
                        correct,
                        active,
                        content,
                        target_lengths,
                    )
                _update_eos_suffix_decile_stats(
                    eos_suffix_stats,
                    logits,
                    pred,
                    x_eval,
                    active,
                    batch.attention_mask,
                    eos_id,
                )
                lossable_count += int((batch.loss_mask & batch.attention_mask).sum().item())
                if active.any():
                    ce_sum += float(ce[active].sum().item())
                    active_count += int(active.sum().item())
                    correct_count += int((correct & active).sum().item())
                if content.any():
                    content_ce_sum += float(ce[content].sum().item())
                    content_count += int(content.sum().item())
                    content_correct_count += int((correct & content).sum().item())
                    row_content_counts = content.sum(dim=-1)
                    row_content_correct = (correct & content).sum(dim=-1)
                    has_content = row_content_counts > 0
                    rows_with_content += int(has_content.sum().item())
                    exact_content_rows += int(((row_content_correct == row_content_counts) & has_content).sum().item())

                if logged < log_samples:
                    reconstructed = x_eval.clone()
                    reconstructed[active] = pred[active]
                    for row_idx in range(x_eval.shape[0]):
                        if logged >= log_samples:
                            break
                        if not bool(content[row_idx].any().item()):
                            continue
                        row_content = int(content[row_idx].sum().item())
                        row_correct = int((correct[row_idx] & content[row_idx]).sum().item())
                        log(
                            f"[recon] step={step if step is not None else '-'} t={float(t_val):.2f} "
                            f"batch={batch_idx} item={row_idx} content_correct={row_correct}/{row_content}"
                        )
                        log(
                            "[recon] target="
                            f"{_decode_attention_row(tokenizer, x_eval[row_idx], batch.attention_mask[row_idx])!r}"
                        )
                        log(
                            "[recon] masked="
                            f"{_decode_masked_attention_row(tokenizer, z_eval[row_idx], batch.attention_mask[row_idx], mask_id, eos_id)!r}"
                        )
                        log(
                            "[recon] pred="
                            f"{_decode_attention_row(tokenizer, reconstructed[row_idx], batch.attention_mask[row_idx])!r}"
                        )
                        logged += 1

            key = _t_key(float(t_val))
            metrics[f"recon_{key}_ce"] = ce_sum / active_count if active_count else 0.0
            metrics[f"recon_{key}_content_ce"] = content_ce_sum / content_count if content_count else 0.0
            metrics[f"recon_{key}_token_acc"] = correct_count / active_count if active_count else 0.0
            metrics[f"recon_{key}_content_token_acc"] = content_correct_count / content_count if content_count else 0.0
            metrics[f"recon_{key}_content_exact_row_rate"] = (
                exact_content_rows / rows_with_content if rows_with_content else 0.0
            )
            metrics[f"recon_{key}_active_frac"] = active_count / lossable_count if lossable_count else 0.0
            metrics[f"recon_{key}_content_tokens"] = content_count
            metrics.update(_eos_suffix_decile_metrics(f"recon_{key}_", eos_suffix_stats))
            if use_target_length_conditioning:
                metrics.update(_target_length_recon_metrics(f"recon_{key}_", target_length_stats))

    return metrics


@torch.no_grad()
def evaluate_generation_metrics(
    model,
    tokenizer,
    noise,
    sample_fn,
    batch_size: int,
    run_dir: str | None = None,
    device=None,
    gen_samples: int = 512,
    gen_len: int = 512,
    log_samples: int = 5,
    score_mode: str = "self",
    self_score_t: float = 0.5,
    judge_model_id: str = "Qwen/Qwen3-0.6B-Base",
    step: int | None = None,
    sample_label: str | None = None,
    model_logits_fn=None,
    block_size: int = 0,
) -> dict[str, float | int | str]:
    if score_mode not in ("none", "self", "judge"):
        raise ValueError(f"unknown generation score mode: {score_mode!r}")

    device = next(model.parameters()).device if device is None else device
    n_batches = max(1, math.ceil(gen_samples / batch_size))
    eos_id = tokenizer.get_eos_token_id()
    special_ids = tokenizer.get_special_token_ids()
    blocked_ids = tokenizer.get_generation_blocked_token_ids()
    token_bytes_cpu = build_token_bytes(tokenizer).cpu()
    prior_logit_bias, prior_metrics = _generation_prior_correction(
        run_dir,
        tokenizer,
        batch_size,
        device,
        gen_len,
    )

    texts: list[str] = []
    content_sequences: list[list[int]] = []
    content_token_counts: Counter[int] = Counter()
    decoded_token_lengths: list[int] = []
    char_lengths: list[int] = []
    content_bytes_per_sample: list[int] = []
    eos_count = 0
    eos_position_counts: Counter[int] = Counter()
    eos_block_counts: Counter[int] = Counter()
    blocked_token_count = 0
    token_count = 0
    content_byte_total = 0
    self_ce_total = 0.0
    self_ce_count = 0
    judge_ce_total = 0.0
    target_trace_lengths: list[int] = []
    target_trace_pre_eos_positions: list[int] = []
    target_trace_post_eos_positions: list[int] = []
    target_trace_forced_eos_count = 0
    trace_totals: dict[str, float | int | None] = {
        "batches": 0,
        "steps": 0,
        "candidate_count": 0,
        "commit_count": 0,
        "commit_count_min": None,
        "commit_count_max": None,
        "candidate_entropy_sum": 0.0,
        "candidate_top1_prob_sum": 0.0,
        "pred_max_token_frac_sum": 0.0,
        "pred_max_token_frac_steps": 0,
        "class_entropy_sum": 0.0,
        "class_max_prob_sum": 0.0,
        "class_prob_eos_special_sum": 0.0,
        "class_prob_low_freq_sum": 0.0,
        "class_prob_high_freq_sum": 0.0,
        "final_mask_count": 0,
        "remask_count": 0,
        "num_blocks": 0,
        "nfe_per_block": 0,
    }

    judge = _get_judge(device, judge_model_id) if score_mode == "judge" else None

    with ExitStack() as stack:
        log_file = None
        if run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            log_file = stack.enter_context(open(os.path.join(run_dir, "gen_samples.log"), "a"))

        def log(msg: str):
            if log_file is not None:
                log_file.write(msg + "\n")
                log_file.flush()
            else:
                print(msg)

        log(
            f"[gen] step={step if step is not None else '-'} "
            f"score_mode={score_mode} samples={gen_samples} len={gen_len}"
            f"{f' label={sample_label}' if sample_label else ''}"
            f"{' prior_correction=true' if prior_logit_bias is not None else ''}"
        )
        logged = 0
        remaining = int(gen_samples)
        for batch_idx in range(n_batches):
            current_batch = min(int(batch_size), remaining)
            remaining -= current_batch
            with sampling_logit_bias(prior_logit_bias):
                sample_out = sample_fn(model, current_batch, gen_len)
            if isinstance(sample_out, tuple):
                tokens, sample_trace = sample_out
                _merge_sample_trace(trace_totals, sample_trace)
                trace_lengths = sample_trace.get("target_length_values")
                trace_pre = sample_trace.get("target_length_pre_policy_eos_positions")
                trace_post = sample_trace.get("target_length_post_policy_eos_positions")
                if isinstance(trace_lengths, list) and isinstance(trace_pre, list) and isinstance(trace_post, list):
                    target_trace_lengths.extend(int(v) for v in trace_lengths)
                    target_trace_pre_eos_positions.extend(int(v) for v in trace_pre)
                    target_trace_post_eos_positions.extend(int(v) for v in trace_post)
                    target_trace_forced_eos_count += int(sample_trace.get("target_length_forced_eos_count", 0))
                log(
                    f"[gen-trace] step={step if step is not None else '-'} batch={batch_idx} "
                    f"{f'label={sample_label} ' if sample_label else ''}"
                    f"{json.dumps(sample_trace, sort_keys=True)}"
                )
            else:
                tokens = sample_out
            batch_texts = []
            for row_idx in range(tokens.shape[0]):
                row = [int(i) for i in tokens[row_idx].detach().cpu().tolist()]
                truncated, has_eos = _truncate_at_eos(row, eos_id)
                content = _content_ids(row, special_ids, eos_id)
                text = tokenizer.decode(row)
                content_byte_count = _token_byte_total(content, token_bytes_cpu)
                texts.append(text)
                batch_texts.append(text)
                content_sequences.append(content)
                content_token_counts.update(content)
                decoded_token_lengths.append(len(truncated))
                char_lengths.append(len(text))
                content_bytes_per_sample.append(content_byte_count)
                eos_count += int(has_eos)
                if has_eos:
                    eos_position_counts[len(truncated) - 1] += 1
                    if block_size > 0:
                        eos_block_counts[(len(truncated) - 1) // block_size] += 1
                token_count += len(row)
                blocked_token_count += sum(1 for token_id in row if int(token_id) in blocked_ids)
                content_byte_total += content_byte_count

                if logged < log_samples:
                    log(
                        f"[gen] step={step if step is not None else '-'} batch={batch_idx} item={row_idx} "
                        f"tokens={len(truncated)} chars={len(text)} eos={has_eos} empty={not text.strip()}"
                        f"{f' label={sample_label}' if sample_label else ''}"
                    )
                    log(f"[gen] text={text[:300]!r}")
                    logged += 1

            empty = sum(1 for txt in batch_texts if len(txt.strip()) == 0)
            if empty:
                log(
                    f"[gen] WARNING: step={step if step is not None else '-'} "
                    f"batch={batch_idx} empty={empty}/{current_batch}"
                )

            if score_mode == "self":
                ce_sum, ce_count = _self_reconstruction_ce(
                    model,
                    tokenizer,
                    noise,
                    tokens.to(device),
                    device,
                    float(self_score_t),
                    model_logits_fn=model_logits_fn,
                )
                self_ce_total += ce_sum
                self_ce_count += ce_count
                if ce_count:
                    log(
                        f"[gen] step={step if step is not None else '-'} batch={batch_idx} "
                        f"self_ce_t={float(self_score_t):.3f} self_ce={ce_sum / ce_count:.4f}"
                    )
            elif score_mode == "judge" and judge is not None:
                batch_ce = judge.score(batch_texts)
                log(
                    f"[gen] step={step if step is not None else '-'} batch={batch_idx} "
                    f"judge_ce={batch_ce:.4f} judge_ppl={math.exp(batch_ce):.2f}"
                )
                judge_ce_total += batch_ce

    sample_count = len(texts)
    unique_texts = len(set(texts))
    empty_count = sum(1 for txt in texts if len(txt.strip()) == 0)
    max_token_fracs = [_max_token_fraction(ids) for ids in content_sequences]
    token_entropies = [_token_entropy(ids) for ids in content_sequences]
    token_entropy_mean = sum(token_entropies) / sample_count if sample_count else 0.0
    if content_token_counts:
        max_token_id, max_token_count = content_token_counts.most_common(1)[0]
        content_token_total = sum(content_token_counts.values())
    else:
        max_token_id, max_token_count, content_token_total = -1, 0, 0

    eos_position_mean = sum(decoded_token_lengths) / sample_count if sample_count else 0.0

    metrics: dict[str, float | int | str] = {
        "gen_score_mode": score_mode,
        "gen_samples": sample_count,
        "gen_empty_rate": empty_count / sample_count if sample_count else 0.0,
        "gen_eos_rate": eos_count / sample_count if sample_count else 0.0,
        "gen_eos_count": int(eos_count),
        "gen_eos_token_id": int(eos_id) if eos_id is not None else -1,
        "gen_avg_decoded_tokens": sum(decoded_token_lengths) / sample_count if sample_count else 0.0,
        "gen_avg_chars": sum(char_lengths) / sample_count if sample_count else 0.0,
        "gen_avg_content_bytes": sum(content_bytes_per_sample) / sample_count if sample_count else 0.0,
        "gen_content_tokens": int(content_token_total),
        "gen_content_bytes": int(content_byte_total),
        "gen_content_token_byte_fertility": content_token_total / content_byte_total if content_byte_total else 0.0,
        "gen_exact_duplicate_rate": 1.0 - (unique_texts / sample_count) if sample_count else 0.0,
        "gen_blocked_token_rate": blocked_token_count / token_count if token_count else 0.0,
        "gen_distinct_1": _distinct_ngram_ratio(content_sequences, 1),
        "gen_distinct_2": _distinct_ngram_ratio(content_sequences, 2),
        "gen_distinct_3": _distinct_ngram_ratio(content_sequences, 3),
        "gen_repeat_4gram_rate": _repeat_ngram_rate(content_sequences, 4),
        "gen_max_token_frac_mean": sum(max_token_fracs) / sample_count if sample_count else 0.0,
        "gen_max_token_id": int(max_token_id),
        "gen_max_token_text": _single_token_text(tokenizer, int(max_token_id)),
        "gen_max_token_count": int(max_token_count),
        "gen_max_token_fraction": max_token_count / content_token_total if content_token_total else 0.0,
        "gen_token_entropy_mean": token_entropy_mean,
        "gen_token_entropy_std": _entropy_std(token_entropies),
        "gen_unigram_entropy": _ngram_entropy(content_sequences, 1),
        "gen_bigram_entropy": _ngram_entropy(content_sequences, 2),
        "gen_token_entropy": token_entropy_mean,
        "gen_eos_position_mean": eos_position_mean,
    }
    for frac in (0.25, 0.5, 0.75):
        threshold = int(gen_len * frac)
        count = sum(1 for length in decoded_token_lengths if length < threshold)
        metrics[f"gen_eos_before_{int(frac * 100)}pct"] = count / sample_count if sample_count else 0.0
    metrics.update(_eos_position_metrics("gen_", eos_position_counts, sample_count, int(gen_len)))
    if block_size > 0:
        metrics.update(
            _eos_block_metrics("gen_", eos_block_counts, sample_count, (gen_len + block_size - 1) // block_size)
        )
    metrics.update(prior_metrics)
    if score_mode == "self":
        metrics["gen_self_score_t"] = float(self_score_t)
        metrics["gen_self_ce"] = self_ce_total / self_ce_count if self_ce_count else 0.0
        metrics["gen_self_ce_tokens"] = self_ce_count
    elif score_mode == "judge":
        metrics["gen_ppl"] = math.exp(judge_ce_total / n_batches)
    if int(trace_totals["batches"] or 0) > 0:
        trace_metrics = _sample_trace_metrics(trace_totals)
        metrics.update(trace_metrics)
        if "gen_trace_commit_tokens" in trace_metrics:
            metrics["sampler_commit_count"] = trace_metrics["gen_trace_commit_tokens"]
        if "gen_trace_candidate_entropy_mean" in trace_metrics:
            metrics["sampler_candidate_entropy"] = trace_metrics["gen_trace_candidate_entropy_mean"]
        if "gen_trace_pred_max_token_frac_mean" in trace_metrics:
            metrics["sampler_pred_max_token_frac"] = trace_metrics["gen_trace_pred_max_token_frac_mean"]
    if target_trace_lengths or model_uses_target_length_conditioning(model):
        metrics.update(
            _target_length_generation_metrics(
                target_trace_lengths,
                target_trace_pre_eos_positions,
                target_trace_post_eos_positions,
                target_trace_forced_eos_count,
                int(gen_len),
            )
        )

    return metrics


@torch.no_grad()
def evaluate_conditional_generation_metrics(
    model,
    tokenizer,
    data_config: DataConfig,
    noise,
    sample_fn,
    batch_size: int,
    run_dir: str | None = None,
    device=None,
    seq_len: int = MAX_SEQ_LEN,
    prefix_fracs: list[float] | tuple[float, ...] = (0.25, 0.5, 0.75),
    gen_samples: int = 64,
    continuation_len: int = 256,
    log_samples: int = 8,
    score_mode: str = "self",
    self_score_t: float = 0.5,
    step: int | None = None,
    sample_label: str | None = None,
    model_logits_fn=None,
) -> dict[str, float | int | str]:
    if score_mode not in ("none", "self"):
        raise ValueError(f"conditional generation score mode must be 'none' or 'self', got {score_mode!r}")
    device = next(model.parameters()).device if device is None else device
    batch_size = max(1, int(batch_size))
    target_samples = max(1, int(gen_samples))
    continuation_len = int(continuation_len)
    seq_len = int(seq_len)
    if continuation_len <= 0:
        raise ValueError("continuation_len must be positive")
    if continuation_len >= seq_len:
        raise ValueError("continuation_len must be smaller than seq_len")

    eos_id = tokenizer.get_eos_token_id()
    special_ids = tokenizer.get_special_token_ids()
    blocked_ids = tokenizer.get_generation_blocked_token_ids()
    token_bytes_cpu = build_token_bytes(tokenizer).cpu()
    metrics: dict[str, float | int | str] = {
        "conditional_gen_samples_requested": int(target_samples),
        "conditional_gen_continuation_len": int(continuation_len),
        "conditional_gen_score_mode": score_mode,
    }

    prefix_records_path = None
    samples_path = None
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        prefix_records_path = os.path.join(run_dir, "prefix_records.jsonl")
        samples_path = os.path.join(run_dir, "conditional_generation_samples.jsonl")
        metrics["conditional_gen_prefix_records_path"] = prefix_records_path
        metrics["conditional_gen_samples_path"] = samples_path

    for raw_frac in prefix_fracs:
        prefix_frac = float(raw_frac)
        label = f"p{int(round(prefix_frac * 100)):02d}"
        prefix = f"conditional_gen_{label}_"
        prefix_len = max(1, min(int(round(continuation_len * prefix_frac)), seq_len - continuation_len))
        total_len = prefix_len + continuation_len
        val_loader = make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)

        prefixes: list[torch.Tensor] = []
        prefix_records: list[dict[str, object]] = []
        attempts = 0
        max_attempts = max(8, target_samples * 8)
        while len(prefixes) < target_samples and attempts < max_attempts:
            batch = next(val_loader)
            batch_idx = attempts
            attempts += 1
            for row_idx in range(batch.x_0.shape[0]):
                if len(prefixes) >= target_samples:
                    break
                valid_len = int(batch.attention_mask[row_idx].sum().item())
                if valid_len < total_len:
                    continue
                prefix_tokens = batch.x_0[row_idx, :prefix_len].detach().clone()
                if eos_id is not None and bool((prefix_tokens == int(eos_id)).any().item()):
                    continue
                prefixes.append(prefix_tokens)
                prefix_ids = [int(i) for i in prefix_tokens.detach().cpu().tolist()]
                gold_ids = [int(i) for i in batch.x_0[row_idx, prefix_len:total_len].detach().cpu().tolist()]
                prefix_records.append(
                    {
                        "step": step,
                        "sample_label": sample_label or "",
                        "source_split": "val",
                        "source_batch_index": int(batch_idx),
                        "source_row_index": int(row_idx),
                        "source_valid_len": int(valid_len),
                        "prefix_fraction": float(prefix_frac),
                        "prefix_len": int(prefix_len),
                        "continuation_len": int(continuation_len),
                        "prefix_token_ids": prefix_ids,
                        "prefix_text": tokenizer.decode(prefix_ids),
                        "gold_continuation_token_ids": gold_ids,
                        "gold_continuation_text": tokenizer.decode(gold_ids),
                    }
                )

        if prefix_records_path is not None:
            with open(prefix_records_path, "a", encoding="utf-8") as f:
                for record in prefix_records:
                    f.write(json.dumps(record, sort_keys=True) + "\n")

        if not prefixes:
            metrics.update(
                {
                    f"{prefix}samples": 0,
                    f"{prefix}prefix_fraction": float(prefix_frac),
                    f"{prefix}prefix_len": int(prefix_len),
                    f"{prefix}continuation_len": int(continuation_len),
                    f"{prefix}collection_attempts": int(attempts),
                }
            )
            continue

        generated_texts: list[str] = []
        content_sequences: list[list[int]] = []
        content_token_counts: Counter[int] = Counter()
        decoded_token_lengths: list[int] = []
        char_lengths: list[int] = []
        content_bytes_per_sample: list[int] = []
        eos_count = 0
        eos_position_counts: Counter[int] = Counter()
        blocked_token_count = 0
        token_count = 0
        content_byte_total = 0
        prefix_copy_hits = 0
        prefix_copy_total = 0
        prefix_preserved = 0
        self_ce_total = 0.0
        self_ce_count = 0
        trace_totals = _new_sample_trace_totals()
        logged = 0

        for start in range(0, len(prefixes), batch_size):
            end = min(len(prefixes), start + batch_size)
            prefix_tensor = torch.stack(prefixes[start:end], dim=0).to(device=device)
            sample_out = sample_fn(model, prefix_tensor, continuation_len)
            if isinstance(sample_out, tuple):
                tokens, sample_trace = sample_out
                _merge_sample_trace(trace_totals, sample_trace)
            else:
                tokens = sample_out
            if tokens.shape[1] != total_len:
                raise RuntimeError(f"conditional sampler returned length {tokens.shape[1]}, expected {total_len}")
            prefix_preserved += int((tokens[:, :prefix_len] == prefix_tensor).all(dim=1).sum().item())

            score_mask = torch.zeros_like(tokens, dtype=torch.bool, device=device)
            score_mask[:, prefix_len:] = True
            if score_mode == "self":
                ce_sum, ce_count = _self_reconstruction_ce_with_score_mask(
                    model,
                    tokenizer,
                    noise,
                    tokens,
                    score_mask,
                    device,
                    float(self_score_t),
                    model_logits_fn=model_logits_fn,
                )
                self_ce_total += ce_sum
                self_ce_count += ce_count

            for local_idx in range(tokens.shape[0]):
                global_idx = start + local_idx
                row = [int(i) for i in tokens[local_idx].detach().cpu().tolist()]
                prefix_ids = row[:prefix_len]
                continuation = row[prefix_len:]
                truncated, has_eos = _truncate_at_eos(continuation, eos_id)
                content = _content_ids(continuation, special_ids, eos_id)
                prefix_content = set(_content_ids(prefix_ids, special_ids, eos_id))
                text = tokenizer.decode(continuation)
                content_byte_count = _token_byte_total(content, token_bytes_cpu)
                generated_texts.append(text)
                content_sequences.append(content)
                content_token_counts.update(content)
                decoded_token_lengths.append(len(truncated))
                char_lengths.append(len(text))
                content_bytes_per_sample.append(content_byte_count)
                eos_count += int(has_eos)
                if has_eos:
                    eos_position_counts[len(truncated) - 1] += 1
                token_count += len(continuation)
                blocked_token_count += sum(1 for token_id in continuation if int(token_id) in blocked_ids)
                content_byte_total += content_byte_count
                prefix_copy_hits += sum(1 for token_id in content if int(token_id) in prefix_content)
                prefix_copy_total += len(content)

                if samples_path is not None and logged < int(log_samples):
                    record = dict(prefix_records[global_idx])
                    record.update(
                        {
                            "sample_index": int(global_idx),
                            "prefix_preserved": bool(prefix_ids == record["prefix_token_ids"]),
                            "continuation_token_ids": continuation,
                            "continuation_text": text,
                            "full_text": tokenizer.decode(row),
                            "has_eos": bool(has_eos),
                        }
                    )
                    with open(samples_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, sort_keys=True) + "\n")
                    logged += 1

        sample_count = len(generated_texts)
        unique_texts = len(set(generated_texts))
        empty_count = sum(1 for txt in generated_texts if len(txt.strip()) == 0)
        max_token_fracs = [_max_token_fraction(ids) for ids in content_sequences]
        token_entropies = [_token_entropy(ids) for ids in content_sequences]
        token_entropy_mean = sum(token_entropies) / sample_count if sample_count else 0.0
        if content_token_counts:
            max_token_id, max_token_count = content_token_counts.most_common(1)[0]
            content_token_total = sum(content_token_counts.values())
        else:
            max_token_id, max_token_count, content_token_total = -1, 0, 0

        frac_metrics: dict[str, float | int | str] = {
            f"{prefix}samples": int(sample_count),
            f"{prefix}prefix_fraction": float(prefix_frac),
            f"{prefix}prefix_len": int(prefix_len),
            f"{prefix}continuation_len": int(continuation_len),
            f"{prefix}total_len": int(total_len),
            f"{prefix}collection_attempts": int(attempts),
            f"{prefix}logged_samples": int(logged),
            f"{prefix}prefix_preserved_rate": prefix_preserved / sample_count if sample_count else 0.0,
            f"{prefix}empty_rate": empty_count / sample_count if sample_count else 0.0,
            f"{prefix}eos_rate": eos_count / sample_count if sample_count else 0.0,
            f"{prefix}eos_count": int(eos_count),
            f"{prefix}eos_token_id": int(eos_id) if eos_id is not None else -1,
            f"{prefix}mean_continuation_len": sum(decoded_token_lengths) / sample_count if sample_count else 0.0,
            f"{prefix}avg_chars": sum(char_lengths) / sample_count if sample_count else 0.0,
            f"{prefix}avg_content_bytes": sum(content_bytes_per_sample) / sample_count if sample_count else 0.0,
            f"{prefix}content_tokens": int(content_token_total),
            f"{prefix}content_bytes": int(content_byte_total),
            f"{prefix}content_token_byte_fertility": content_token_total / content_byte_total
            if content_byte_total
            else 0.0,
            f"{prefix}exact_duplicate_rate": 1.0 - (unique_texts / sample_count) if sample_count else 0.0,
            f"{prefix}blocked_token_rate": blocked_token_count / token_count if token_count else 0.0,
            f"{prefix}distinct_1": _distinct_ngram_ratio(content_sequences, 1),
            f"{prefix}distinct_2": _distinct_ngram_ratio(content_sequences, 2),
            f"{prefix}distinct_3": _distinct_ngram_ratio(content_sequences, 3),
            f"{prefix}repeat_4gram_rate": _repeat_ngram_rate(content_sequences, 4),
            f"{prefix}max_token_frac_mean": sum(max_token_fracs) / sample_count if sample_count else 0.0,
            f"{prefix}max_token_id": int(max_token_id),
            f"{prefix}max_token_text": _single_token_text(tokenizer, int(max_token_id)),
            f"{prefix}max_token_count": int(max_token_count),
            f"{prefix}max_token_fraction": max_token_count / content_token_total if content_token_total else 0.0,
            f"{prefix}token_entropy_mean": token_entropy_mean,
            f"{prefix}token_entropy_std": _entropy_std(token_entropies),
            f"{prefix}unigram_entropy": _ngram_entropy(content_sequences, 1),
            f"{prefix}bigram_entropy": _ngram_entropy(content_sequences, 2),
            f"{prefix}prefix_copy_rate": prefix_copy_hits / prefix_copy_total if prefix_copy_total else 0.0,
        }
        if score_mode == "self":
            frac_metrics[f"{prefix}self_score_t"] = float(self_score_t)
            frac_metrics[f"{prefix}self_ce"] = self_ce_total / self_ce_count if self_ce_count else 0.0
            frac_metrics[f"{prefix}self_ce_tokens"] = int(self_ce_count)
        if int(trace_totals["batches"] or 0) > 0:
            for key, value in _sample_trace_metrics(trace_totals).items():
                suffix = key[len("gen_trace_") :] if key.startswith("gen_trace_") else key
                frac_metrics[f"{prefix}trace_{suffix}"] = value
            frac_metrics[f"{prefix}trace_pred_max_token_frac_mean"] = frac_metrics.get(
                f"{prefix}trace_pred_max_token_frac_mean",
                0.0,
            )
        for key, value in _eos_position_metrics("", eos_position_counts, sample_count, int(continuation_len)).items():
            frac_metrics[f"{prefix}{key}"] = value
        metrics.update(frac_metrics)

    return metrics


@torch.no_grad()
def evaluate_bos_only_first_token(model, tokenizer, device) -> dict[str, float | int]:
    """Evaluate the first-token distribution when the model is fed only BOS."""
    bos_id = tokenizer.get_bos_token_id()
    if bos_id is None:
        return {}
    tokens = torch.full((1, 1), int(bos_id), dtype=torch.long, device=device)
    logits = model(tokens)[:, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = float((-probs * log_probs).sum().item())
    top_prob, top_id = probs.max(dim=-1)
    return {
        "bos_only_first_token_entropy": entropy,
        "bos_only_first_token_top_id": int(top_id.item()),
        "bos_only_first_token_top_prob": float(top_prob.item()),
        "bos_only_first_token_top_text": _single_token_text(tokenizer, int(top_id.item())),
    }


def _ar_block_special_logits(logits: torch.Tensor, tokenizer) -> None:
    for token_id in tokenizer.get_generation_blocked_token_ids():
        if 0 <= int(token_id) < logits.size(-1):
            logits[..., int(token_id)] = -1e9


@torch.no_grad()
def _sample_ar_from_prefixes(
    model,
    tokenizer,
    prefixes: torch.Tensor,
    total_len: int,
    temperature: float,
    token_selection: str,
) -> torch.Tensor:
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection!r}")
    if prefixes.dim() != 2:
        raise ValueError(f"prefixes must be 2D, got shape {tuple(prefixes.shape)}")
    if prefixes.shape[1] >= total_len:
        raise ValueError("prefix length must be smaller than total generation length")

    device = next(model.parameters()).device
    pad_id = tokenizer.get_pad_token_id()
    tokens = torch.full(
        (prefixes.shape[0], int(total_len)),
        int(pad_id) if pad_id is not None else 0,
        dtype=torch.long,
        device=device,
    )
    prefixes = prefixes.to(device=device, dtype=torch.long)
    prefix_len = prefixes.shape[1]
    tokens[:, :prefix_len] = prefixes
    for pos in range(prefix_len, int(total_len)):
        logits = model(tokens[:, :pos])[:, -1, :]
        _ar_block_special_logits(logits, tokenizer)
        if token_selection == "argmax":
            next_token = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / float(temperature), dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
        tokens[:, pos] = next_token
    return tokens


def _nucleus_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    probs = F.softmax(logits / float(temperature), dim=-1)
    if float(top_p) >= 1.0:
        return torch.multinomial(probs, 1).squeeze(-1)

    sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = sorted_probs.masked_fill(remove, 0.0)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sampled = torch.multinomial(filtered, 1)
    return sorted_ids.gather(-1, sampled).squeeze(-1)


def _basin_next_logits(
    model,
    tokenizer,
    tokens: torch.Tensor,
    pos: int,
    is_causal: bool,
    noise_level=None,
    model_logits_fn=None,
) -> torch.Tensor:
    if is_causal:
        return model(tokens[:, : int(pos)])[:, -1, :]

    attention_mask = torch.ones_like(tokens, dtype=torch.bool, device=tokens.device)
    if noise_level is None and _model_uses_noise_level(model):
        mask_id = int(tokenizer.get_mask_token_id())
        noise_level = (tokens == mask_id).float().mean(dim=1)
    logits = _call_model_logits(
        model,
        tokens,
        tokens,
        attention_mask,
        model_logits_fn=model_logits_fn,
        noise_level=noise_level,
    )
    return logits[:, int(pos), :]


def _basin_initial_tokens(
    tokenizer,
    prefixes: torch.Tensor,
    total_len: int,
    is_causal: bool,
    noise=None,
) -> torch.Tensor:
    device = prefixes.device
    pad_id = tokenizer.get_pad_token_id()
    fill_id = pad_id if pad_id is not None else 0
    if is_causal:
        tokens = torch.full(
            (prefixes.shape[0], int(total_len)),
            int(fill_id),
            dtype=torch.long,
            device=device,
        )
    elif noise is not None and hasattr(noise, "sample_terminal_tokens"):
        tokens = noise.sample_terminal_tokens((prefixes.shape[0], int(total_len)), device=device)
    elif noise is not None and hasattr(noise, "sample_uniform_tokens"):
        tokens = noise.sample_uniform_tokens((prefixes.shape[0], int(total_len)), device=device)
    else:
        tokens = torch.full(
            (prefixes.shape[0], int(total_len)),
            int(tokenizer.get_mask_token_id()),
            dtype=torch.long,
            device=device,
        )
    tokens[:, : prefixes.shape[1]] = prefixes
    return tokens


@torch.no_grad()
def _sample_basin_from_prefixes(
    model,
    tokenizer,
    prefixes: torch.Tensor,
    total_len: int,
    forced_argmax_depth: int,
    continuation_policy: str,
    temperature: float,
    nucleus_top_p: float,
    is_causal: bool,
    noise=None,
    model_logits_fn=None,
) -> tuple[torch.Tensor, list[dict[str, float | int | str]]]:
    if continuation_policy not in ("argmax", "nucleus"):
        raise ValueError(f"unknown continuation_policy: {continuation_policy!r}")
    if prefixes.dim() != 2:
        raise ValueError(f"prefixes must be 2D, got shape {tuple(prefixes.shape)}")
    if prefixes.shape[1] >= total_len:
        raise ValueError("prefix length must be smaller than total generation length")

    device = next(model.parameters()).device
    prefix_len = int(prefixes.shape[1])
    prefixes = prefixes.to(device=device, dtype=torch.long)
    tokens = _basin_initial_tokens(
        tokenizer,
        prefixes,
        int(total_len),
        is_causal=bool(is_causal),
        noise=noise,
    )

    traces: list[dict[str, float | int | str]] = []
    max_forced = min(max(0, int(forced_argmax_depth)), int(total_len) - prefix_len)
    for step_idx in range(max_forced):
        pos = prefix_len + step_idx
        step_noise_level = None if is_causal else (int(total_len) - int(pos)) / max(1, int(total_len))
        logits = _basin_next_logits(
            model,
            tokenizer,
            tokens,
            pos,
            is_causal=is_causal,
            noise_level=step_noise_level,
            model_logits_fn=model_logits_fn,
        ).float()
        _ar_block_special_logits(logits, tokenizer)
        probs = F.softmax(logits / float(temperature), dim=-1)
        safe_probs = probs.clamp_min(1e-10)
        entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)
        top1_prob, top1_id = probs.max(dim=-1)
        next_token = logits.argmax(dim=-1)
        tokens[:, pos] = next_token
        counts = torch.bincount(next_token.detach().cpu(), minlength=tokenizer.get_vocab_size())
        max_count = int(counts.max().item()) if counts.numel() else 0
        most_common_id = int(counts.argmax().item()) if counts.numel() else -1
        traces.append(
            {
                "forced_step": int(step_idx + 1),
                "position": int(pos),
                "entropy_mean": float(entropy.mean().item()),
                "top1_prob_mean": float(top1_prob.mean().item()),
                "top1_prob_min": float(top1_prob.min().item()),
                "top1_prob_max": float(top1_prob.max().item()),
                "top1_token_id": most_common_id,
                "top1_token_text": _single_token_text(tokenizer, most_common_id),
                "top1_token_count": max_count,
                "top1_token_fraction": max_count / max(1, int(next_token.numel())),
            }
        )

    for pos in range(prefix_len + max_forced, int(total_len)):
        step_noise_level = None if is_causal else (int(total_len) - int(pos)) / max(1, int(total_len))
        logits = _basin_next_logits(
            model,
            tokenizer,
            tokens,
            pos,
            is_causal=is_causal,
            noise_level=step_noise_level,
            model_logits_fn=model_logits_fn,
        ).float()
        _ar_block_special_logits(logits, tokenizer)
        if continuation_policy == "argmax":
            next_token = logits.argmax(dim=-1)
        else:
            next_token = _nucleus_sample(logits, temperature=float(temperature), top_p=float(nucleus_top_p))
        tokens[:, pos] = next_token
    return tokens, traces


def _basin_continuation_metrics(
    tokenizer,
    tokens: torch.Tensor,
    prefix_len: int,
    total_len: int,
) -> dict[str, float | int | str]:
    eos_id = tokenizer.get_eos_token_id()
    special_ids = tokenizer.get_special_token_ids()
    blocked_ids = tokenizer.get_generation_blocked_token_ids()
    token_bytes_cpu = build_token_bytes(tokenizer).cpu()
    generated_texts: list[str] = []
    content_sequences: list[list[int]] = []
    content_token_counts: Counter[int] = Counter()
    decoded_token_lengths: list[int] = []
    char_lengths: list[int] = []
    content_bytes_per_sample: list[int] = []
    eos_count = 0
    eos_position_counts: Counter[int] = Counter()
    blocked_token_count = 0
    token_count = 0
    content_byte_total = 0

    for row_idx in range(tokens.shape[0]):
        row = [int(i) for i in tokens[row_idx].detach().cpu().tolist()]
        generated = row[int(prefix_len) :]
        truncated, has_eos = _truncate_at_eos(generated, eos_id)
        content = _content_ids(generated, special_ids, eos_id)
        text = tokenizer.decode(generated)
        content_byte_count = _token_byte_total(content, token_bytes_cpu)
        generated_texts.append(text)
        content_sequences.append(content)
        content_token_counts.update(content)
        decoded_token_lengths.append(len(truncated))
        char_lengths.append(len(text))
        content_bytes_per_sample.append(content_byte_count)
        eos_count += int(has_eos)
        if has_eos:
            eos_position_counts[len(truncated) - 1] += 1
        token_count += len(generated)
        blocked_token_count += sum(1 for token_id in generated if int(token_id) in blocked_ids)
        content_byte_total += content_byte_count

    sample_count = len(generated_texts)
    unique_texts = len(set(generated_texts))
    empty_count = sum(1 for txt in generated_texts if len(txt.strip()) == 0)
    max_token_fracs = [_max_token_fraction(ids) for ids in content_sequences]
    token_entropies = [_token_entropy(ids) for ids in content_sequences]
    token_entropy_mean = sum(token_entropies) / sample_count if sample_count else 0.0
    if content_token_counts:
        max_token_id, max_token_count = content_token_counts.most_common(1)[0]
        content_token_total = sum(content_token_counts.values())
    else:
        max_token_id, max_token_count, content_token_total = -1, 0, 0

    generated_len = int(total_len) - int(prefix_len)
    return {
        "samples": sample_count,
        "generated_len": generated_len,
        "empty_rate": empty_count / sample_count if sample_count else 0.0,
        "eos_rate": eos_count / sample_count if sample_count else 0.0,
        "eos_count": int(eos_count),
        "eos_token_id": int(eos_id) if eos_id is not None else -1,
        "avg_decoded_tokens": sum(decoded_token_lengths) / sample_count if sample_count else 0.0,
        "avg_chars": sum(char_lengths) / sample_count if sample_count else 0.0,
        "avg_content_bytes": sum(content_bytes_per_sample) / sample_count if sample_count else 0.0,
        "content_tokens": int(content_token_total),
        "content_bytes": int(content_byte_total),
        "content_token_byte_fertility": content_token_total / content_byte_total if content_byte_total else 0.0,
        "exact_duplicate_rate": 1.0 - (unique_texts / sample_count) if sample_count else 0.0,
        "blocked_token_rate": blocked_token_count / token_count if token_count else 0.0,
        "distinct_1": _distinct_ngram_ratio(content_sequences, 1),
        "distinct_2": _distinct_ngram_ratio(content_sequences, 2),
        "distinct_3": _distinct_ngram_ratio(content_sequences, 3),
        "repeat_4gram_rate": _repeat_ngram_rate(content_sequences, 4),
        "max_token_frac_mean": sum(max_token_fracs) / sample_count if sample_count else 0.0,
        "max_token_id": int(max_token_id),
        "max_token_text": _single_token_text(tokenizer, int(max_token_id)),
        "max_token_count": int(max_token_count),
        "max_token_fraction": max_token_count / content_token_total if content_token_total else 0.0,
        "token_entropy_mean": token_entropy_mean,
        "token_entropy_std": _entropy_std(token_entropies),
        "unigram_entropy": _ngram_entropy(content_sequences, 1),
        "bigram_entropy": _ngram_entropy(content_sequences, 2),
        **_eos_position_metrics("", eos_position_counts, sample_count, generated_len),
    }


@torch.no_grad()
def evaluate_argmax_basin_diagnostic(
    model,
    tokenizer,
    data_config,
    batch_size: int,
    device,
    seq_len: int = MAX_SEQ_LEN,
    gen_samples: int = 16,
    gen_len: int = 256,
    prefix_len: int = 16,
    forced_argmax_depths: list[int] | tuple[int, ...] = (0, 1, 2, 4),
    temperature: float = 1.0,
    nucleus_top_p: float = 0.95,
    is_causal: bool = False,
    run_dir: str | None = None,
    log_samples: int = 5,
    source_run_id: str = "",
    noise=None,
    model_logits_fn=None,
) -> dict[str, float | int | str]:
    total_len = max(2, int(gen_len))
    prefix_len = max(1, min(int(prefix_len), total_len - 1, int(seq_len) - 1))
    target_samples = max(1, int(gen_samples))
    batch_size = max(1, int(batch_size))
    eos_id = tokenizer.get_eos_token_id()
    bos_id = tokenizer.get_bos_token_id()
    val_loader = make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)

    prefixes: list[torch.Tensor] = []
    gold_rows: list[list[int]] = []
    attempts = 0
    max_attempts = max(8, target_samples * 8)
    while len(prefixes) < target_samples and attempts < max_attempts:
        attempts += 1
        batch = next(val_loader)
        for row_idx in range(batch.x_0.shape[0]):
            if len(prefixes) >= target_samples:
                break
            start = _ar_prefix_start(batch.x_0[row_idx], batch.attention_mask[row_idx], bos_id, prefix_len)
            if start < 0:
                continue
            prefix = batch.x_0[row_idx, start : start + prefix_len].detach().clone()
            if eos_id is not None and bool((prefix == int(eos_id)).any().item()):
                continue
            valid_len = int(batch.attention_mask[row_idx].sum().item())
            gold_end = min(valid_len, start + total_len)
            prefixes.append(prefix)
            gold_rows.append([int(i) for i in batch.x_0[row_idx, start:gold_end].detach().cpu().tolist()])

    metrics: dict[str, float | int | str] = {
        "basin_samples": int(len(prefixes)),
        "basin_collection_attempts": int(attempts),
        "basin_prefix_len": int(prefix_len),
        "basin_gen_len": int(total_len),
        "basin_generated_len": int(total_len - prefix_len),
        "basin_nucleus_top_p": float(nucleus_top_p),
        "basin_temperature": float(temperature),
        "basin_model_kind": "causal_ar" if is_causal else "masked_prefix",
        "basin_stochastic_policy": "nucleus_multinomial",
    }
    if not prefixes:
        return metrics

    prefix_tensor = torch.stack(prefixes, dim=0).to(device=device)
    examples_path = os.path.join(run_dir, "basin_diagnostic_examples.jsonl") if run_dir is not None else None
    traces_path = os.path.join(run_dir, "basin_diagnostic_traces.jsonl") if run_dir is not None else None
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)

    per_policy: dict[tuple[int, str], dict[str, float | int | str]] = {}
    for raw_depth in forced_argmax_depths:
        depth = max(0, int(raw_depth))
        trace_written = False
        for policy in ("argmax", "nucleus"):
            tokens, traces = _sample_basin_from_prefixes(
                model,
                tokenizer,
                prefix_tensor,
                total_len,
                forced_argmax_depth=depth,
                continuation_policy=policy,
                temperature=float(temperature),
                nucleus_top_p=float(nucleus_top_p),
                is_causal=bool(is_causal),
                noise=noise,
                model_logits_fn=model_logits_fn,
            )
            depth_key = f"basin_depth_{depth:02d}_{policy}_"
            policy_metrics = _basin_continuation_metrics(tokenizer, tokens, prefix_len, total_len)
            per_policy[(depth, policy)] = policy_metrics
            for key, value in policy_metrics.items():
                metrics[f"{depth_key}{key}"] = value
            metrics[f"{depth_key}forced_argmax_depth"] = int(depth)
            metrics[f"{depth_key}continuation_policy"] = "nucleus_multinomial" if policy == "nucleus" else policy

            if not trace_written:
                for trace in traces:
                    trace_key = f"basin_depth_{depth:02d}_forced_step_{int(trace['forced_step']):02d}_"
                    for key, value in trace.items():
                        if key == "forced_step":
                            continue
                        metrics[f"{trace_key}{key}"] = value
                    if traces_path is not None:
                        with open(traces_path, "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "source_run_id": source_run_id,
                                        "forced_argmax_depth": int(depth),
                                        **trace,
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                trace_written = True

            if examples_path is not None and log_samples > 0:
                with open(examples_path, "a", encoding="utf-8") as f:
                    for row_idx in range(min(int(log_samples), tokens.shape[0])):
                        row = [int(i) for i in tokens[row_idx].detach().cpu().tolist()]
                        forced_end = min(prefix_len + depth, total_len)
                        f.write(
                            json.dumps(
                                {
                                    "source_run_id": source_run_id,
                                    "forced_argmax_depth": int(depth),
                                    "continuation_policy": policy,
                                    "sample_index": int(row_idx),
                                    "prefix_text": tokenizer.decode(row[:prefix_len]),
                                    "forced_text": tokenizer.decode(row[prefix_len:forced_end]),
                                    "continuation_text": tokenizer.decode(row[forced_end:]),
                                    "generated_text": tokenizer.decode(row[prefix_len:]),
                                    "gold_text": tokenizer.decode(gold_rows[row_idx]),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )

        pair_keys = (
            "eos_rate",
            "exact_duplicate_rate",
            "repeat_4gram_rate",
            "distinct_1",
            "distinct_2",
            "distinct_3",
            "max_token_frac_mean",
            "max_token_fraction",
            "token_entropy_mean",
        )
        argmax_metrics = per_policy.get((depth, "argmax"), {})
        nucleus_metrics = per_policy.get((depth, "nucleus"), {})
        for key in pair_keys:
            argmax_value = argmax_metrics.get(key)
            nucleus_value = nucleus_metrics.get(key)
            if isinstance(argmax_value, (int, float)) and isinstance(nucleus_value, (int, float)):
                metrics[f"basin_depth_{depth:02d}_argmax_minus_nucleus_{key}"] = float(argmax_value) - float(
                    nucleus_value
                )

    return metrics


def _ar_prefix_start(row: torch.Tensor, attention_mask: torch.Tensor, bos_id: int | None, prefix_len: int) -> int:
    valid_len = int(attention_mask.sum().item())
    if valid_len <= prefix_len:
        return -1
    if bos_id is not None:
        candidates = ((row == int(bos_id)) & attention_mask).nonzero(as_tuple=True)[0]
        for candidate in candidates.detach().cpu().tolist():
            if int(candidate) + prefix_len < valid_len:
                return int(candidate)
    return 0


@torch.no_grad()
def evaluate_ar_validation_prefix_continuations(
    model,
    tokenizer,
    data_config,
    batch_size: int,
    device,
    seq_len: int = MAX_SEQ_LEN,
    gen_samples: int = 16,
    gen_len: int = 256,
    prefix_len: int = 16,
    temperature: float = 1.0,
    token_selection: str = "argmax",
    run_dir: str | None = None,
    log_samples: int = 5,
    step: int | None = None,
) -> dict[str, float | int | str]:
    total_len = max(2, int(gen_len))
    prefix_len = max(1, min(int(prefix_len), total_len - 1, int(seq_len) - 1))
    target_samples = max(1, int(gen_samples))
    batch_size = max(1, int(batch_size))
    eos_id = tokenizer.get_eos_token_id()
    bos_id = tokenizer.get_bos_token_id()
    special_ids = tokenizer.get_special_token_ids()
    blocked_ids = tokenizer.get_generation_blocked_token_ids()
    token_bytes_cpu = build_token_bytes(tokenizer).cpu()
    val_loader = make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)

    prefixes: list[torch.Tensor] = []
    gold_rows: list[list[int]] = []
    attempts = 0
    max_attempts = max(8, target_samples * 8)
    while len(prefixes) < target_samples and attempts < max_attempts:
        attempts += 1
        batch = next(val_loader)
        for row_idx in range(batch.x_0.shape[0]):
            if len(prefixes) >= target_samples:
                break
            start = _ar_prefix_start(batch.x_0[row_idx], batch.attention_mask[row_idx], bos_id, prefix_len)
            if start < 0:
                continue
            prefix = batch.x_0[row_idx, start : start + prefix_len].detach().clone()
            if eos_id is not None and bool((prefix == int(eos_id)).any().item()):
                continue
            valid_len = int(batch.attention_mask[row_idx].sum().item())
            gold_end = min(valid_len, start + total_len)
            prefixes.append(prefix)
            gold_rows.append([int(i) for i in batch.x_0[row_idx, start:gold_end].detach().cpu().tolist()])

    if not prefixes:
        return {
            "samples": 0,
            "prefix_len": int(prefix_len),
            "gen_len": int(total_len),
            "token_selection": token_selection,
            "collection_attempts": int(attempts),
        }

    prefix_tensor = torch.stack(prefixes, dim=0).to(device=device)
    tokens = _sample_ar_from_prefixes(
        model,
        tokenizer,
        prefix_tensor,
        total_len,
        temperature=float(temperature),
        token_selection=token_selection,
    )

    generated_texts: list[str] = []
    content_sequences: list[list[int]] = []
    content_token_counts: Counter[int] = Counter()
    decoded_token_lengths: list[int] = []
    char_lengths: list[int] = []
    content_bytes_per_sample: list[int] = []
    eos_count = 0
    eos_position_counts: Counter[int] = Counter()
    blocked_token_count = 0
    token_count = 0
    content_byte_total = 0

    for row_idx in range(tokens.shape[0]):
        row = [int(i) for i in tokens[row_idx].detach().cpu().tolist()]
        generated = row[prefix_len:]
        truncated, has_eos = _truncate_at_eos(generated, eos_id)
        content = _content_ids(generated, special_ids, eos_id)
        text = tokenizer.decode(generated)
        content_byte_count = _token_byte_total(content, token_bytes_cpu)
        generated_texts.append(text)
        content_sequences.append(content)
        content_token_counts.update(content)
        decoded_token_lengths.append(len(truncated))
        char_lengths.append(len(text))
        content_bytes_per_sample.append(content_byte_count)
        eos_count += int(has_eos)
        if has_eos:
            eos_position_counts[len(truncated) - 1] += 1
        token_count += len(generated)
        blocked_token_count += sum(1 for token_id in generated if int(token_id) in blocked_ids)
        content_byte_total += content_byte_count

    if run_dir is not None and log_samples > 0:
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "ar_validation_prefix_samples.log")
        with open(log_path, "a") as log_file:
            log_file.write(
                f"[ar-val-prefix] step={step if step is not None else '-'} "
                f"token_selection={token_selection} samples={len(prefixes)} "
                f"prefix_len={prefix_len} gen_len={total_len}\n"
            )
            for row_idx in range(min(int(log_samples), tokens.shape[0])):
                row = [int(i) for i in tokens[row_idx].detach().cpu().tolist()]
                log_file.write(
                    f"[ar-val-prefix] item={row_idx} "
                    f"generated_tokens={len(row) - prefix_len} "
                    f"eos={eos_id in row[prefix_len:] if eos_id is not None else False}\n"
                )
                log_file.write(f"[ar-val-prefix] prefix={tokenizer.decode(row[:prefix_len])!r}\n")
                log_file.write(f"[ar-val-prefix] gold={tokenizer.decode(gold_rows[row_idx])!r}\n")
                log_file.write(f"[ar-val-prefix] generated={tokenizer.decode(row)!r}\n")
                log_file.write(f"[ar-val-prefix] generated_suffix={generated_texts[row_idx]!r}\n")

    sample_count = len(generated_texts)
    unique_texts = len(set(generated_texts))
    empty_count = sum(1 for txt in generated_texts if len(txt.strip()) == 0)
    max_token_fracs = [_max_token_fraction(ids) for ids in content_sequences]
    token_entropies = [_token_entropy(ids) for ids in content_sequences]
    token_entropy_mean = sum(token_entropies) / sample_count if sample_count else 0.0
    if content_token_counts:
        max_token_id, max_token_count = content_token_counts.most_common(1)[0]
        content_token_total = sum(content_token_counts.values())
    else:
        max_token_id, max_token_count, content_token_total = -1, 0, 0

    return {
        "samples": sample_count,
        "prefix_len": int(prefix_len),
        "gen_len": int(total_len),
        "generated_len": int(total_len - prefix_len),
        "token_selection": token_selection,
        "temperature": float(temperature),
        "collection_attempts": int(attempts),
        "empty_rate": empty_count / sample_count if sample_count else 0.0,
        "eos_rate": eos_count / sample_count if sample_count else 0.0,
        "eos_count": int(eos_count),
        "eos_token_id": int(eos_id) if eos_id is not None else -1,
        "avg_decoded_tokens": sum(decoded_token_lengths) / sample_count if sample_count else 0.0,
        "avg_chars": sum(char_lengths) / sample_count if sample_count else 0.0,
        "avg_content_bytes": sum(content_bytes_per_sample) / sample_count if sample_count else 0.0,
        "content_tokens": int(content_token_total),
        "content_bytes": int(content_byte_total),
        "content_token_byte_fertility": content_token_total / content_byte_total if content_byte_total else 0.0,
        "exact_duplicate_rate": 1.0 - (unique_texts / sample_count) if sample_count else 0.0,
        "blocked_token_rate": blocked_token_count / token_count if token_count else 0.0,
        "distinct_1": _distinct_ngram_ratio(content_sequences, 1),
        "distinct_2": _distinct_ngram_ratio(content_sequences, 2),
        "distinct_3": _distinct_ngram_ratio(content_sequences, 3),
        "repeat_4gram_rate": _repeat_ngram_rate(content_sequences, 4),
        "max_token_frac_mean": sum(max_token_fracs) / sample_count if sample_count else 0.0,
        "max_token_id": int(max_token_id),
        "max_token_text": _single_token_text(tokenizer, int(max_token_id)),
        "max_token_count": int(max_token_count),
        "max_token_fraction": max_token_count / content_token_total if content_token_total else 0.0,
        "token_entropy_mean": token_entropy_mean,
        "token_entropy_std": _entropy_std(token_entropies),
        "unigram_entropy": _ngram_entropy(content_sequences, 1),
        "bigram_entropy": _ngram_entropy(content_sequences, 2),
        **_eos_position_metrics("", eos_position_counts, sample_count, int(total_len - prefix_len)),
    }


def _new_conditional_rank_stats() -> dict:
    return {
        "content_token_count": 0,
        "gold_ranks": [],
        "gold_probs": [],
        "top1_hits": 0,
        "top5_hits": 0,
        "top10_hits": 0,
        "top20_hits": 0,
        "top1_probs": [],
        "predicted_max_ids": [],
        "entropies": [],
        "margins": [],
    }


def _new_conditional_rank_decile_stats() -> dict[int, dict]:
    return {d: _new_conditional_rank_stats() for d in range(1, 8)}


def _content_position_mask(
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    token_bytes: torch.Tensor,
    special_ids: set[int],
) -> torch.Tensor:
    content = attention_mask & loss_mask & (token_bytes[x_0] > 0)
    for sid in special_ids:
        content &= x_0 != int(sid)
    return content


def _conditional_rank_frequency_deciles(
    data_config,
    tokenizer,
    batch_size: int,
    seq_len: int,
    device,
    batches: int,
    token_bytes: torch.Tensor,
    special_ids: set[int],
) -> tuple[dict[int, int], int]:
    train_loader = make_dataloader(data_config, tokenizer, batch_size, seq_len, "train", device=device)
    freq_counts = Counter()
    freq_batches = max(batches * 2, 10)
    for _ in range(freq_batches):
        batch = next(train_loader)
        content = _content_position_mask(batch.x_0, batch.attention_mask, batch.loss_mask, token_bytes, special_ids)
        if content.any():
            ids = batch.x_0[content].detach().cpu().tolist()
            freq_counts.update(ids)

    vocab_size = tokenizer.get_vocab_size()
    all_freqs = [(freq_counts.get(tid, 0), tid) for tid in range(vocab_size)]
    all_freqs.sort(key=lambda x: x[0])
    decile_size = vocab_size // 7
    remainder = vocab_size % 7
    token_to_decile: dict[int, int] = {}
    idx = 0
    for decile in range(1, 8):
        size = decile_size + (1 if decile <= remainder else 0)
        for _ in range(size):
            token_to_decile[all_freqs[idx][1]] = decile
            idx += 1
    return token_to_decile, freq_batches


def _score_conditional_rank_position(
    pos_probs: torch.Tensor,
    token_id: int,
    vocab_size: int,
    topk_k: int,
) -> dict[str, float | int | bool]:
    gold_prob = float(pos_probs[token_id].item())
    rank = int((pos_probs > pos_probs[token_id]).sum().item()) + 1
    topk_vals, topk_ids = torch.topk(pos_probs, topk_k)
    top1_prob = float(topk_vals[0].item())
    predicted_max_id = int(topk_ids[0].item())
    top1_hit = bool((topk_ids[0] == token_id).item())
    top5_hit = bool((topk_ids[:5] == token_id).any().item()) if topk_k >= 5 else top1_hit
    top10_hit = bool((topk_ids[:10] == token_id).any().item()) if topk_k >= 10 else top5_hit
    top20_hit = bool((topk_ids[:20] == token_id).any().item()) if topk_k >= 20 else top10_hit
    entropy = float((-pos_probs * torch.log(pos_probs + 1e-10)).sum().item())
    margin = float((topk_vals[0] - topk_vals[1]).item()) if vocab_size > 1 else 0.0
    return {
        "rank": rank,
        "gold_prob": gold_prob,
        "top1_hit": top1_hit,
        "top5_hit": top5_hit,
        "top10_hit": top10_hit,
        "top20_hit": top20_hit,
        "top1_prob": top1_prob,
        "predicted_max_id": predicted_max_id,
        "entropy": entropy,
        "margin": margin,
    }


def _update_conditional_rank_stats(stats: dict, scored: dict[str, float | int | bool]) -> None:
    stats["content_token_count"] += 1
    stats["gold_ranks"].append(int(scored["rank"]))
    stats["gold_probs"].append(float(scored["gold_prob"]))
    stats["top1_hits"] += int(bool(scored["top1_hit"]))
    stats["top5_hits"] += int(bool(scored["top5_hit"]))
    stats["top10_hits"] += int(bool(scored["top10_hit"]))
    stats["top20_hits"] += int(bool(scored["top20_hit"]))
    stats["top1_probs"].append(float(scored["top1_prob"]))
    stats["predicted_max_ids"].append(int(scored["predicted_max_id"]))
    stats["entropies"].append(float(scored["entropy"]))
    stats["margins"].append(float(scored["margin"]))


def _record_conditional_rank_position(
    decile_stats: dict[int, dict],
    suffix_stats: dict,
    token_to_decile: dict[int, int],
    token_id: int,
    scored: dict[str, float | int | bool],
) -> None:
    decile = token_to_decile.get(token_id, 1)
    _update_conditional_rank_stats(decile_stats[decile], scored)
    _update_conditional_rank_stats(suffix_stats, scored)


def _fill_conditional_rank_metrics(stats: dict, prefix: str, metrics_out: dict, tokenizer) -> None:
    count = stats["content_token_count"]
    metrics_out[f"{prefix}content_token_count"] = count
    if count == 0:
        metrics_out[f"{prefix}gold_rank_mean"] = 0.0
        metrics_out[f"{prefix}gold_prob_mean"] = 0.0
        metrics_out[f"{prefix}gold_top1_hit"] = 0.0
        metrics_out[f"{prefix}gold_top5_hit"] = 0.0
        metrics_out[f"{prefix}gold_top10_hit"] = 0.0
        metrics_out[f"{prefix}gold_top20_hit"] = 0.0
        metrics_out[f"{prefix}top1_prob_mean"] = 0.0
        metrics_out[f"{prefix}predicted_max_token_id"] = -1
        metrics_out[f"{prefix}predicted_max_token_text"] = ""
        metrics_out[f"{prefix}predicted_max_token_count"] = 0
        metrics_out[f"{prefix}predicted_max_token_fraction"] = 0.0
        metrics_out[f"{prefix}entropy_mean"] = 0.0
        metrics_out[f"{prefix}margin_mean"] = 0.0
        return
    metrics_out[f"{prefix}gold_rank_mean"] = sum(stats["gold_ranks"]) / count
    metrics_out[f"{prefix}gold_prob_mean"] = sum(stats["gold_probs"]) / count
    metrics_out[f"{prefix}gold_top1_hit"] = stats["top1_hits"] / count
    metrics_out[f"{prefix}gold_top5_hit"] = stats["top5_hits"] / count
    metrics_out[f"{prefix}gold_top10_hit"] = stats["top10_hits"] / count
    metrics_out[f"{prefix}gold_top20_hit"] = stats["top20_hits"] / count
    metrics_out[f"{prefix}top1_prob_mean"] = sum(stats["top1_probs"]) / count
    max_counter = Counter(stats["predicted_max_ids"])
    most_common_id, most_common_count = max_counter.most_common(1)[0]
    metrics_out[f"{prefix}predicted_max_token_id"] = most_common_id
    metrics_out[f"{prefix}predicted_max_token_text"] = _single_token_text(tokenizer, most_common_id)
    metrics_out[f"{prefix}predicted_max_token_count"] = most_common_count
    metrics_out[f"{prefix}predicted_max_token_fraction"] = most_common_count / count
    metrics_out[f"{prefix}entropy_mean"] = sum(stats["entropies"]) / count
    metrics_out[f"{prefix}margin_mean"] = sum(stats["margins"]) / count


@torch.no_grad()
def evaluate_ar_next_token_conditional_rank_suffix(
    model,
    tokenizer,
    data_config,
    device,
    batches=5,
    batch_size: int = 8,
    seq_len: int = MAX_SEQ_LEN,
) -> dict[str, float | int | str]:
    if batches <= 0:
        return {}

    special_ids = tokenizer.get_special_token_ids()
    token_bytes = build_token_bytes(tokenizer).to(device)
    token_to_decile, freq_batches = _conditional_rank_frequency_deciles(
        data_config,
        tokenizer,
        batch_size,
        seq_len,
        device,
        batches,
        token_bytes,
        special_ids,
    )

    val_loader = make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)
    short_document = bool(getattr(data_config, "short_document_curriculum", False))
    prefix_len = seq_len // 2
    if short_document:
        prefix_len = max(1, min(prefix_len, int(getattr(data_config, "short_document_max_tokens", seq_len)) // 2))
    vocab_size = tokenizer.get_vocab_size()
    topk_k = min(20, vocab_size)
    decile_stats = _new_conditional_rank_decile_stats()
    suffix_stats = _new_conditional_rank_stats()
    prefix_len_sum = 0
    prefix_len_min = 0
    prefix_len_max = 0
    prefix_len_count = 0

    for _ in range(int(batches)):
        batch = next(val_loader)
        x_0 = batch.x_0
        content = _content_position_mask(x_0, batch.attention_mask, batch.loss_mask, token_bytes, special_ids)
        if x_0.shape[1] < 2:
            continue
        target_content = content[:, 1:].clone()
        if short_document:
            valid_lens = batch.attention_mask.sum(dim=1)
            row_prefix_lens = torch.clamp(valid_lens // 2, min=1)
            row_prefix_lens = torch.clamp(row_prefix_lens, max=max(1, int(prefix_len)))
            target_positions = torch.arange(1, x_0.shape[1], device=x_0.device).view(1, -1)
            target_content &= target_positions >= row_prefix_lens.view(-1, 1)
            row_prefix_cpu = row_prefix_lens.detach().cpu()
            prefix_len_sum += int(row_prefix_cpu.sum().item())
            prefix_len_count += int(row_prefix_cpu.numel())
            row_min = int(row_prefix_cpu.min().item()) if row_prefix_cpu.numel() else 0
            row_max = int(row_prefix_cpu.max().item()) if row_prefix_cpu.numel() else 0
            prefix_len_min = (
                row_min if prefix_len_count == int(row_prefix_cpu.numel()) else min(prefix_len_min, row_min)
            )
            prefix_len_max = max(prefix_len_max, row_max)
        else:
            target_content[:, : max(0, prefix_len - 1)] = False
        if not target_content.any():
            continue

        model_kwargs = {}
        if getattr(batch, "position_ids", None) is not None:
            model_kwargs["position_ids"] = batch.position_ids
        if getattr(batch, "document_boundary_mask", None) is not None:
            model_kwargs["attention_mask"] = batch.document_boundary_mask & batch.attention_mask[:, None, :]
        else:
            model_kwargs["attention_mask"] = batch.attention_mask
        logits = model(x_0, **model_kwargs)[:, :-1, :].float()
        probs = F.softmax(logits, dim=-1)
        rows, target_offsets = target_content.nonzero(as_tuple=True)
        for row_idx, target_offset in zip(
            rows.detach().cpu().tolist(),
            target_offsets.detach().cpu().tolist(),
            strict=True,
        ):
            target_pos = int(target_offset) + 1
            token_id = int(x_0[int(row_idx), target_pos].item())
            scored = _score_conditional_rank_position(
                probs[int(row_idx), int(target_offset)],
                token_id,
                vocab_size,
                topk_k,
            )
            _record_conditional_rank_position(decile_stats, suffix_stats, token_to_decile, token_id, scored)

    metrics: dict[str, float | int | str] = {
        "ar_next_token_conditional_rank_batches": int(batches),
        "ar_next_token_conditional_rank_freq_batches": int(freq_batches),
        "ar_next_token_conditional_rank_prefix_len": int(prefix_len),
        "ar_next_token_conditional_rank_short_document": int(short_document),
    }
    if short_document:
        metrics["ar_next_token_conditional_rank_prefix_len_mean"] = (
            prefix_len_sum / prefix_len_count if prefix_len_count else 0.0
        )
        metrics["ar_next_token_conditional_rank_prefix_len_min"] = int(prefix_len_min)
        metrics["ar_next_token_conditional_rank_prefix_len_max"] = int(prefix_len_max)
    for decile in range(1, 8):
        _fill_conditional_rank_metrics(
            decile_stats[decile],
            f"ar_next_token_conditional_rank_decile_{decile}_",
            metrics,
            tokenizer,
        )
    _fill_conditional_rank_metrics(
        suffix_stats,
        "ar_next_token_conditional_rank_suffix_",
        metrics,
        tokenizer,
    )
    return metrics


def _commit_fraction_label(fraction: float) -> str:
    return f"commit{int(round(float(fraction) * 100)):02d}"


def _block_generation_special_logits(logits: torch.Tensor, tokenizer) -> None:
    for token_id in tokenizer.get_generation_blocked_token_ids():
        if 0 <= int(token_id) < logits.size(-1):
            logits[..., int(token_id)] = -1e9


def _new_rollout_trace() -> dict[str, float | int | None]:
    return {
        "batches": 0,
        "steps": 0,
        "candidate_count": 0,
        "commit_count": 0,
        "commit_count_min": None,
        "commit_count_max": None,
        "candidate_entropy_sum": 0.0,
        "candidate_top1_prob_sum": 0.0,
        "pred_max_token_frac_sum": 0.0,
        "pred_max_token_frac_steps": 0,
        "final_mask_count": 0,
    }


def _update_rollout_trace(
    trace: dict[str, float | int | None],
    candidate_mask: torch.Tensor,
    probs: torch.Tensor,
    x_pred: torch.Tensor,
    commit_mask: torch.Tensor,
) -> None:
    candidate_count = int(candidate_mask.sum().item())
    commit_count = int(commit_mask.sum().item())
    trace["steps"] = int(trace["steps"] or 0) + 1
    trace["candidate_count"] = int(trace["candidate_count"] or 0) + candidate_count
    trace["commit_count"] = int(trace["commit_count"] or 0) + commit_count
    trace["commit_count_min"] = (
        commit_count if trace["commit_count_min"] is None else min(int(trace["commit_count_min"]), commit_count)
    )
    trace["commit_count_max"] = (
        commit_count if trace["commit_count_max"] is None else max(int(trace["commit_count_max"]), commit_count)
    )
    if candidate_count == 0:
        return

    top1_prob = probs.max(dim=-1).values[candidate_mask]
    safe_probs = probs.clamp_min(1e-10)
    entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)[candidate_mask]
    trace["candidate_top1_prob_sum"] = float(trace["candidate_top1_prob_sum"] or 0.0) + float(top1_prob.sum().item())
    trace["candidate_entropy_sum"] = float(trace["candidate_entropy_sum"] or 0.0) + float(entropy.sum().item())

    pred_ids = x_pred[candidate_mask]
    if pred_ids.numel() > 0:
        counts = torch.unique(pred_ids, return_counts=True)[1]
        trace["pred_max_token_frac_sum"] = float(trace["pred_max_token_frac_sum"] or 0.0) + float(
            counts.max().item() / max(1, pred_ids.numel())
        )
        trace["pred_max_token_frac_steps"] = int(trace["pred_max_token_frac_steps"] or 0) + 1


def _fill_rollout_trace_metrics(trace: dict[str, float | int | None], metrics: dict) -> None:
    batches = int(trace["batches"] or 0)
    steps = int(trace["steps"] or 0)
    candidate_count = int(trace["candidate_count"] or 0)
    pred_frac_steps = int(trace["pred_max_token_frac_steps"] or 0)
    prefix = "conditional_rank_rollout_sampler_"
    metrics[f"{prefix}batches"] = batches
    metrics[f"{prefix}steps_mean"] = steps / batches if batches else 0.0
    metrics[f"{prefix}candidate_tokens"] = candidate_count
    metrics[f"{prefix}commit_tokens"] = int(trace["commit_count"] or 0)
    metrics[f"{prefix}commit_tokens_mean"] = int(trace["commit_count"] or 0) / steps if steps else 0.0
    metrics[f"{prefix}commit_tokens_min"] = int(trace["commit_count_min"] or 0)
    metrics[f"{prefix}commit_tokens_max"] = int(trace["commit_count_max"] or 0)
    metrics[f"{prefix}candidate_entropy_mean"] = (
        float(trace["candidate_entropy_sum"] or 0.0) / candidate_count if candidate_count else 0.0
    )
    metrics[f"{prefix}candidate_top1_prob_mean"] = (
        float(trace["candidate_top1_prob_sum"] or 0.0) / candidate_count if candidate_count else 0.0
    )
    metrics[f"{prefix}pred_max_token_frac_mean"] = (
        float(trace["pred_max_token_frac_sum"] or 0.0) / pred_frac_steps if pred_frac_steps else 0.0
    )
    metrics[f"{prefix}final_mask_count"] = int(trace["final_mask_count"] or 0)
    metrics[f"{prefix}final_mask_count_mean"] = int(trace["final_mask_count"] or 0) / batches if batches else 0.0


def _rollout_confidence(
    probs: torch.Tensor,
    scoring_mode: str,
    jitter_scale: float,
) -> torch.Tensor:
    if scoring_mode == "maxprob":
        return probs.max(dim=-1).values
    if scoring_mode in ("margin", "margin_jitter"):
        top2 = probs.topk(2, dim=-1).values
        conf = top2[..., 0] - top2[..., 1]
        if scoring_mode == "margin_jitter" and jitter_scale > 0:
            conf = conf + (-torch.empty_like(conf).exponential_().log() - 0.5772156649) * float(jitter_scale)
        return conf
    if scoring_mode == "entropy":
        safe_probs = probs.clamp_min(1e-10)
        return (safe_probs * safe_probs.log()).sum(dim=-1)
    if scoring_mode == "random":
        return torch.rand(probs.shape[:-1], device=probs.device)
    raise ValueError(f"unknown scoring mode: {scoring_mode}")


def _rollout_commit_mask(
    still_masked: torch.Tensor,
    probs: torch.Tensor,
    x_pred: torch.Tensor,
    step: int,
    sampler: str,
    scoring_mode: str,
    jitter_scale: float,
    fhs_tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampler == "fhs":
        selected_prob = probs.gather(-1, x_pred.unsqueeze(-1)).squeeze(-1)
        commit = still_masked & (selected_prob >= float(fhs_tau))
        priority = selected_prob
        if commit.any():
            return commit, priority

        p_unmask = 1.0 / max(1, int(step))
        k_per_seq = (p_unmask * still_masked.sum(dim=-1)).long().clamp(min=1)
        scores = selected_prob.masked_fill(~still_masked, -1e9)
        commit = torch.zeros_like(still_masked)
        for row_idx in range(still_masked.shape[0]):
            remaining = int(still_masked[row_idx].sum().item())
            if remaining == 0:
                continue
            k = min(remaining, int(k_per_seq[row_idx].item()))
            _, idx = torch.topk(scores[row_idx], k)
            commit[row_idx, idx] = True
        return commit, priority

    priority = _rollout_confidence(probs, scoring_mode, jitter_scale)
    p_unmask = 1.0 / max(1, int(step))
    if scoring_mode == "random":
        commit = still_masked & (priority < p_unmask)
        return commit, priority

    k_per_seq = (p_unmask * still_masked.sum(dim=-1)).long().clamp(min=1)
    scores = priority.masked_fill(~still_masked, -1e9)
    commit = torch.zeros_like(still_masked)
    for row_idx in range(still_masked.shape[0]):
        remaining = int(still_masked[row_idx].sum().item())
        if remaining == 0:
            continue
        k = min(remaining, int(k_per_seq[row_idx].item()))
        _, idx = torch.topk(scores[row_idx], k)
        commit[row_idx, idx] = True
    return commit, priority


def _assign_rollout_commit_ranks(
    commit_rank: torch.Tensor,
    model_tokens: torch.Tensor,
    row_commit_counts: torch.Tensor,
    commit: torch.Tensor,
    x_pred: torch.Tensor,
    priority: torch.Tensor,
) -> None:
    for row_idx in range(commit.shape[0]):
        positions = torch.nonzero(commit[row_idx], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        order = torch.argsort(priority[row_idx, positions].float(), descending=True)
        positions = positions[order]
        start = int(row_commit_counts[row_idx].item())
        ranks = torch.arange(
            start + 1,
            start + 1 + int(positions.numel()),
            device=commit.device,
            dtype=commit_rank.dtype,
        )
        commit_rank[row_idx, positions] = ranks
        model_tokens[row_idx, positions] = x_pred[row_idx, positions]
        row_commit_counts[row_idx] += int(positions.numel())


def _rollout_suffix_commit_order(
    model,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    candidate: torch.Tensor,
    suffix_mask: torch.Tensor,
    tokenizer,
    model_logits_fn,
    sample_steps: int,
    sampler: str,
    scoring_mode: str,
    token_selection: str,
    temperature: float,
    jitter_scale: float,
    fhs_tau: float,
    trace: dict[str, float | int | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_id = int(tokenizer.get_mask_token_id())
    z_model = x_0.clone()
    z_model[suffix_mask] = mask_id
    commit_rank = torch.full_like(x_0, -1, dtype=torch.long)
    model_tokens = torch.full_like(x_0, mask_id)
    row_commit_counts = torch.zeros(x_0.shape[0], device=x_0.device, dtype=torch.long)
    trace["batches"] = int(trace["batches"] or 0) + 1

    for step in range(max(1, int(sample_steps)), 0, -1):
        still_masked = candidate & (commit_rank < 0)
        if not still_masked.any():
            break
        logits = _call_model_logits(
            model,
            z_model,
            x_0,
            attention_mask,
            model_logits_fn=model_logits_fn,
            position_ids=position_ids,
        ).float()
        _block_generation_special_logits(logits, tokenizer)
        probs = F.softmax(logits / float(temperature), dim=-1)
        if token_selection == "argmax":
            x_pred = logits.argmax(dim=-1)
        elif token_selection == "multinomial":
            x_pred = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view_as(x_0)
        else:
            raise ValueError(f"unknown token selection: {token_selection}")

        commit, priority = _rollout_commit_mask(
            still_masked,
            probs,
            x_pred,
            step,
            sampler,
            scoring_mode,
            jitter_scale,
            fhs_tau,
        )
        commit &= still_masked
        _update_rollout_trace(trace, still_masked, probs, x_pred, commit)
        if commit.any():
            _assign_rollout_commit_ranks(commit_rank, model_tokens, row_commit_counts, commit, x_pred, priority)
            z_model[commit] = x_pred[commit]

    remaining = candidate & (commit_rank < 0)
    trace["final_mask_count"] = int(trace["final_mask_count"] or 0) + int(remaining.sum().item())
    if remaining.any():
        logits = _call_model_logits(
            model,
            z_model,
            x_0,
            attention_mask,
            model_logits_fn=model_logits_fn,
            position_ids=position_ids,
        ).float()
        _block_generation_special_logits(logits, tokenizer)
        probs = F.softmax(logits / float(temperature), dim=-1)
        x_pred = logits.argmax(dim=-1)
        priority = probs.gather(-1, x_pred.unsqueeze(-1)).squeeze(-1)
        _assign_rollout_commit_ranks(
            commit_rank,
            model_tokens,
            row_commit_counts,
            remaining,
            x_pred,
            priority,
        )
    return commit_rank, model_tokens


def _rollout_target_counts(candidate: torch.Tensor, fraction: float) -> torch.Tensor:
    totals = candidate.sum(dim=-1)
    raw = torch.ceil(totals.float() * float(fraction)).to(dtype=torch.long)
    bounded = torch.minimum(raw.clamp(min=1), (totals - 1).clamp(min=0))
    return torch.where(totals > 1, bounded, torch.zeros_like(raw))


def _score_rollout_context(
    model,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    context: torch.Tensor,
    score_mask: torch.Tensor,
    tokenizer,
    token_to_decile: dict[int, int],
    model_logits_fn,
    vocab_size: int,
    topk_k: int,
) -> tuple[dict[int, dict], dict]:
    decile_stats = _new_conditional_rank_decile_stats()
    suffix_stats = _new_conditional_rank_stats()
    if not score_mask.any():
        return decile_stats, suffix_stats
    logits = _call_model_logits(
        model,
        context,
        x_0,
        attention_mask,
        model_logits_fn=model_logits_fn,
        position_ids=position_ids,
    )
    probs = F.softmax(logits.float(), dim=-1)
    rows, positions = score_mask.nonzero(as_tuple=True)
    for row_idx, pos in zip(rows.detach().cpu().tolist(), positions.detach().cpu().tolist(), strict=True):
        token_id = int(x_0[row_idx, pos].item())
        scored = _score_conditional_rank_position(probs[row_idx, pos], token_id, vocab_size, topk_k)
        _record_conditional_rank_position(decile_stats, suffix_stats, token_to_decile, token_id, scored)
    return decile_stats, suffix_stats


def _update_rollout_commit_metrics(
    metrics: dict,
    label: str,
    committed: torch.Tensor,
    model_tokens: torch.Tensor,
    x_0: torch.Tensor,
    tokenizer,
    eos_id: int | None,
    token_counter: Counter[int],
) -> None:
    prefix = f"conditional_rank_rollout_{label}_"
    committed_count = int(committed.sum().item())
    if committed_count:
        committed_model = model_tokens[committed]
        committed_gold = x_0[committed]
        exact = int((committed_model == committed_gold).sum().item())
        ids = [int(i) for i in committed_model.detach().cpu().tolist()]
        token_counter.update(ids)
        max_token_id, max_token_count = Counter(ids).most_common(1)[0]
        eos_count = sum(1 for token_id in ids if eos_id is not None and token_id == int(eos_id))
    else:
        exact = 0
        max_token_id, max_token_count = -1, 0
        eos_count = 0
    total_committed = int(metrics.get(f"{prefix}committed_content_count", 0)) + committed_count
    total_exact = int(metrics.get(f"{prefix}commit_exact_match_count", 0)) + exact
    total_eos = int(metrics.get(f"{prefix}eos_commit_count", 0)) + eos_count
    if token_counter:
        max_token_id, max_token_count = token_counter.most_common(1)[0]
    metrics[f"{prefix}committed_content_count"] = total_committed
    metrics[f"{prefix}commit_exact_match_count"] = total_exact
    metrics[f"{prefix}commit_exact_match_rate"] = total_exact / total_committed if total_committed else 0.0
    metrics[f"{prefix}model_committed_max_token_id"] = int(max_token_id)
    metrics[f"{prefix}model_committed_max_token_text"] = _single_token_text(tokenizer, int(max_token_id))
    metrics[f"{prefix}model_committed_max_token_count"] = int(max_token_count)
    metrics[f"{prefix}model_committed_max_token_fraction"] = (
        max_token_count / total_committed if total_committed else 0.0
    )
    metrics[f"{prefix}eos_commit_count"] = int(total_eos)
    metrics[f"{prefix}eos_commit_rate"] = total_eos / total_committed if total_committed else 0.0


def _write_rollout_token_counts(
    run_dir: str | None,
    token_counts: dict[str, Counter[int]],
    tokenizer,
    top_k: int = 50,
) -> tuple[str, int]:
    if run_dir is None:
        return "", 0
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "conditional_rank_rollout_committed_tokens.jsonl")
    records = 0
    with open(path, "w", encoding="utf-8") as f:
        for label in sorted(token_counts):
            total = sum(token_counts[label].values())
            for token_id, count in token_counts[label].most_common(int(top_k)):
                f.write(
                    json.dumps(
                        {
                            "commit_label": label,
                            "token_id": int(token_id),
                            "token_text": _single_token_text(tokenizer, int(token_id)),
                            "count": int(count),
                            "fraction": count / total if total else 0.0,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                records += 1
    return path, records


def _log_rollout_examples(
    log_file,
    tokenizer,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    model_context: torch.Tensor,
    gold_context: torch.Tensor,
    committed: torch.Tensor,
    label: str,
    logged: int,
    log_samples: int,
    step: int | None,
) -> int:
    if log_file is None or logged >= log_samples:
        return logged
    mask_id = int(tokenizer.get_mask_token_id())
    eos_id = tokenizer.get_eos_token_id()
    for row_idx in range(x_0.shape[0]):
        if logged >= log_samples:
            break
        if not bool(committed[row_idx].any().item()):
            continue
        commit_count = int(committed[row_idx].sum().item())
        exact_count = int((model_context[row_idx, committed[row_idx]] == x_0[row_idx, committed[row_idx]]).sum().item())
        log_file.write(
            f"[conditional-rank-rollout] step={step if step is not None else '-'} "
            f"item={logged} {label} row={row_idx} commit_exact={exact_count}/{commit_count}\n"
        )
        log_file.write(
            "[conditional-rank-rollout] target="
            f"{_decode_attention_row(tokenizer, x_0[row_idx], attention_mask[row_idx])!r}\n"
        )
        model_text = _decode_masked_attention_row(
            tokenizer,
            model_context[row_idx],
            attention_mask[row_idx],
            mask_id,
            eos_id,
        )
        log_file.write(f"[conditional-rank-rollout] model_context={model_text!r}\n")
        gold_text = _decode_masked_attention_row(
            tokenizer,
            gold_context[row_idx],
            attention_mask[row_idx],
            mask_id,
            eos_id,
        )
        log_file.write(f"[conditional-rank-rollout] gold_context={gold_text!r}\n")
        logged += 1
    return logged


@torch.no_grad()
def evaluate_conditional_rank_rollout(
    model,
    tokenizer,
    data_config,
    device,
    batches=5,
    batch_size: int = 8,
    seq_len: int = MAX_SEQ_LEN,
    model_logits_fn=None,
    commit_fractions: list[float] | tuple[float, ...] = (0.25, 0.5, 0.75),
    sample_steps: int = 64,
    sampler: str = "default",
    scoring_mode: str = "margin_jitter",
    token_selection: str = "argmax",
    temperature: float = 1.0,
    jitter_scale: float = 0.02,
    fhs_tau: float = 0.9,
    run_dir: str | None = None,
    log_samples: int = 8,
    step: int | None = None,
) -> dict[str, float | int | str]:
    """Score paired model-committed versus gold-committed suffix contexts."""
    if batches <= 0:
        return {}

    start_time = time.time()
    special_ids = tokenizer.get_special_token_ids()
    token_bytes = build_token_bytes(tokenizer).to(device)
    token_to_decile, freq_batches = _conditional_rank_frequency_deciles(
        data_config,
        tokenizer,
        batch_size,
        seq_len,
        device,
        batches,
        token_bytes,
        special_ids,
    )
    val_loader = make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)
    mask_id = int(tokenizer.get_mask_token_id())
    eos_id = tokenizer.get_eos_token_id()
    prefix_len = seq_len // 2
    vocab_size = tokenizer.get_vocab_size()
    topk_k = min(20, vocab_size)
    fractions = [float(fraction) for fraction in commit_fractions]
    labels = [_commit_fraction_label(fraction) for fraction in fractions]

    decile_stats: dict[tuple[str, str], dict[int, dict]] = {}
    suffix_stats: dict[tuple[str, str], dict] = {}
    for label in labels:
        for source in ("model", "gold"):
            decile_stats[(source, label)] = _new_conditional_rank_decile_stats()
            suffix_stats[(source, label)] = _new_conditional_rank_stats()

    trace = _new_rollout_trace()
    token_counts: dict[str, Counter[int]] = {label: Counter() for label in labels}
    metrics: dict[str, float | int | str] = {
        "conditional_rank_rollout_batches": int(batches),
        "conditional_rank_rollout_freq_batches": int(freq_batches),
        "conditional_rank_rollout_sample_steps": int(sample_steps),
        "conditional_rank_rollout_sampler": sampler,
        "conditional_rank_rollout_scoring_mode": scoring_mode,
        "conditional_rank_rollout_token_selection": token_selection,
        "conditional_rank_rollout_temperature": float(temperature),
        "conditional_rank_rollout_jitter_scale": float(jitter_scale),
        "conditional_rank_rollout_fhs_tau": float(fhs_tau),
        "conditional_rank_rollout_commit_fraction_count": len(fractions),
    }
    for label, fraction in zip(labels, fractions, strict=True):
        metrics[f"conditional_rank_rollout_{label}_target_fraction"] = float(fraction)

    logged = 0
    samples_path = ""
    log_file = None
    log_stack = ExitStack()
    if run_dir is not None and log_samples > 0:
        os.makedirs(run_dir, exist_ok=True)
        samples_path = os.path.join(run_dir, "conditional_rank_rollout_samples.log")
        log_file = log_stack.enter_context(Path(samples_path).open("a", encoding="utf-8"))  # noqa: SIM115
        log_file.write(
            f"[conditional-rank-rollout] step={step if step is not None else '-'} "
            f"batches={int(batches)} fractions={','.join(labels)} sampler={sampler} "
            f"scoring={scoring_mode} token_selection={token_selection}\n"
        )

    try:
        for _ in range(int(batches)):
            batch = next(val_loader)
            x_0 = batch.x_0
            attention_mask = batch.attention_mask
            model_attention_mask = _batch_model_attention_mask(batch)
            position_ids = getattr(batch, "position_ids", None)
            loss_mask = batch.loss_mask
            content = _content_position_mask(x_0, attention_mask, loss_mask, token_bytes, special_ids)
            candidate = content.clone()
            candidate[:, :prefix_len] = False
            suffix_mask = attention_mask.clone()
            suffix_mask[:, :prefix_len] = False
            commit_rank, model_tokens = _rollout_suffix_commit_order(
                model,
                x_0,
                model_attention_mask,
                position_ids,
                candidate,
                suffix_mask,
                tokenizer,
                model_logits_fn,
                int(sample_steps),
                sampler,
                scoring_mode,
                token_selection,
                float(temperature),
                float(jitter_scale),
                float(fhs_tau),
                trace,
            )

            for label, fraction in zip(labels, fractions, strict=True):
                target_counts = _rollout_target_counts(candidate, fraction)
                committed = torch.zeros_like(candidate)
                for row_idx in range(candidate.shape[0]):
                    target_count = int(target_counts[row_idx].item())
                    if target_count <= 0:
                        continue
                    committed[row_idx] = (
                        candidate[row_idx] & (commit_rank[row_idx] > 0) & (commit_rank[row_idx] <= target_count)
                    )
                remaining = candidate & ~committed
                model_context = x_0.clone()
                gold_context = x_0.clone()
                model_context[suffix_mask] = mask_id
                gold_context[suffix_mask] = mask_id
                model_context[committed] = model_tokens[committed]
                gold_context[committed] = x_0[committed]

                candidate_count = int(candidate.sum().item())
                remaining_count = int(remaining.sum().item())
                metrics[f"conditional_rank_rollout_{label}_candidate_content_count"] = (
                    int(metrics.get(f"conditional_rank_rollout_{label}_candidate_content_count", 0)) + candidate_count
                )
                metrics[f"conditional_rank_rollout_{label}_remaining_masked_content_count"] = (
                    int(metrics.get(f"conditional_rank_rollout_{label}_remaining_masked_content_count", 0))
                    + remaining_count
                )
                metrics[f"conditional_rank_rollout_{label}_remaining_mask_count"] = int(
                    metrics.get(f"conditional_rank_rollout_{label}_remaining_mask_count", 0)
                ) + int(((model_context == mask_id) & attention_mask).sum().item())
                _update_rollout_commit_metrics(
                    metrics,
                    label,
                    committed,
                    model_tokens,
                    x_0,
                    tokenizer,
                    eos_id,
                    token_counts[label],
                )
                if candidate_count:
                    running_committed = int(metrics[f"conditional_rank_rollout_{label}_committed_content_count"])
                    running_candidate = int(metrics[f"conditional_rank_rollout_{label}_candidate_content_count"])
                    metrics[f"conditional_rank_rollout_{label}_actual_fraction"] = (
                        running_committed / running_candidate if running_candidate else 0.0
                    )

                model_deciles, model_suffix = _score_rollout_context(
                    model,
                    x_0,
                    model_attention_mask,
                    position_ids,
                    model_context,
                    remaining,
                    tokenizer,
                    token_to_decile,
                    model_logits_fn,
                    vocab_size,
                    topk_k,
                )
                gold_deciles, gold_suffix = _score_rollout_context(
                    model,
                    x_0,
                    model_attention_mask,
                    position_ids,
                    gold_context,
                    remaining,
                    tokenizer,
                    token_to_decile,
                    model_logits_fn,
                    vocab_size,
                    topk_k,
                )
                for decile in range(1, 8):
                    for key in model_deciles[decile]:
                        if isinstance(model_deciles[decile][key], list):
                            decile_stats[("model", label)][decile][key].extend(model_deciles[decile][key])
                        else:
                            decile_stats[("model", label)][decile][key] += model_deciles[decile][key]
                    for key in gold_deciles[decile]:
                        if isinstance(gold_deciles[decile][key], list):
                            decile_stats[("gold", label)][decile][key].extend(gold_deciles[decile][key])
                        else:
                            decile_stats[("gold", label)][decile][key] += gold_deciles[decile][key]
                for key in model_suffix:
                    if isinstance(model_suffix[key], list):
                        suffix_stats[("model", label)][key].extend(model_suffix[key])
                    else:
                        suffix_stats[("model", label)][key] += model_suffix[key]
                for key in gold_suffix:
                    if isinstance(gold_suffix[key], list):
                        suffix_stats[("gold", label)][key].extend(gold_suffix[key])
                    else:
                        suffix_stats[("gold", label)][key] += gold_suffix[key]

                logged = _log_rollout_examples(
                    log_file,
                    tokenizer,
                    x_0,
                    attention_mask,
                    model_context,
                    gold_context,
                    committed,
                    label,
                    logged,
                    int(log_samples),
                    step,
                )
    finally:
        log_stack.close()

    for label in labels:
        for source in ("model", "gold"):
            for decile in range(1, 8):
                _fill_conditional_rank_metrics(
                    decile_stats[(source, label)][decile],
                    f"conditional_rank_rollout_{source}_{label}_decile_{decile}_",
                    metrics,
                    tokenizer,
                )
            _fill_conditional_rank_metrics(
                suffix_stats[(source, label)],
                f"conditional_rank_rollout_{source}_{label}_suffix_",
                metrics,
                tokenizer,
            )
    for label in labels:
        candidate_count = int(metrics.get(f"conditional_rank_rollout_{label}_candidate_content_count", 0))
        committed_count = int(metrics.get(f"conditional_rank_rollout_{label}_committed_content_count", 0))
        metrics[f"conditional_rank_rollout_{label}_actual_fraction"] = (
            committed_count / candidate_count if candidate_count else 0.0
        )

    token_counts_path, token_count_records = _write_rollout_token_counts(run_dir, token_counts, tokenizer)
    _fill_rollout_trace_metrics(trace, metrics)
    metrics["conditional_rank_rollout_logged_samples"] = int(logged)
    metrics["conditional_rank_rollout_samples_path"] = samples_path
    metrics["conditional_rank_rollout_committed_token_counts_path"] = token_counts_path
    metrics["conditional_rank_rollout_committed_token_count_records"] = int(token_count_records)
    metrics["conditional_rank_rollout_seconds"] = time.time() - start_time
    metrics["conditional_rank_rollout_peak_vram_mb"] = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    )
    return metrics


def _selected_suffix_positions(content_positions: torch.Tensor, limit: int) -> list[int]:
    if content_positions.numel() <= limit:
        return [int(pos) for pos in content_positions.detach().cpu().tolist()]
    sample_offsets = torch.linspace(
        0,
        content_positions.numel() - 1,
        steps=limit,
        device=content_positions.device,
    ).round()
    return [int(pos) for pos in content_positions[sample_offsets.to(torch.long)].detach().cpu().tolist()]


def _evaluate_loo_conditional_rank_batch(
    model,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    model_attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    loss_mask: torch.Tensor,
    tokenizer,
    model_logits_fn,
    token_bytes: torch.Tensor,
    special_ids: set[int],
    token_to_decile: dict[int, int],
    decile_stats: dict[int, dict],
    suffix_stats: dict,
    prefix_len: int,
    positions_per_sequence: int,
    expanded_batch_size: int,
    vocab_size: int,
    topk_k: int,
) -> int:
    content = _content_position_mask(x_0, attention_mask, loss_mask, token_bytes, special_ids)
    content[:, :prefix_len] = False
    selected: list[tuple[int, int]] = []
    for row_idx in range(x_0.shape[0]):
        row_positions = torch.nonzero(content[row_idx], as_tuple=False).flatten()
        for pos in _selected_suffix_positions(row_positions, positions_per_sequence):
            selected.append((row_idx, pos))

    mask_id = int(tokenizer.get_mask_token_id())
    selected_count = len(selected)
    for chunk_start in range(0, selected_count, expanded_batch_size):
        chunk = selected[chunk_start : chunk_start + expanded_batch_size]
        source_rows = torch.tensor([row_idx for row_idx, _pos in chunk], device=x_0.device, dtype=torch.long)
        chunk_positions = torch.tensor([pos for _row_idx, pos in chunk], device=x_0.device, dtype=torch.long)
        z_chunk = x_0.index_select(0, source_rows).clone()
        x_chunk = x_0.index_select(0, source_rows)
        attention_source = attention_mask if model_attention_mask is None else model_attention_mask
        attention_chunk = attention_source.index_select(0, source_rows)
        position_chunk = position_ids.index_select(0, source_rows) if position_ids is not None else None
        z_chunk[torch.arange(len(chunk), device=x_0.device), chunk_positions] = mask_id

        logits = _call_model_logits(
            model,
            z_chunk,
            x_chunk,
            attention_chunk,
            model_logits_fn=model_logits_fn,
            position_ids=position_chunk,
        )
        selected_logits = logits[torch.arange(len(chunk), device=x_0.device), chunk_positions].float()
        selected_probs = F.softmax(selected_logits, dim=-1)
        for local_idx, (source_row, pos) in enumerate(chunk):
            token_id = int(x_0[source_row, pos].item())
            scored = _score_conditional_rank_position(selected_probs[local_idx], token_id, vocab_size, topk_k)
            _record_conditional_rank_position(decile_stats, suffix_stats, token_to_decile, token_id, scored)
    return selected_count


def _conditional_rank_decile(decile: int) -> int:
    return min(7, max(1, int(decile)))


def _collect_stratified_conditional_rank_positions(
    batch_records: list[dict[str, torch.Tensor]],
    token_to_decile: dict[int, int],
    target_per_decile: int,
) -> tuple[list[dict[str, int]], dict[int, int]]:
    selected_by_decile: dict[int, list[dict[str, int]]] = {decile: [] for decile in range(1, 8)}
    for batch_idx, record in enumerate(batch_records):
        x_0 = record["x_0"]
        suffix_content = record["suffix_content"]
        for row_idx in range(x_0.shape[0]):
            row_positions = torch.nonzero(suffix_content[row_idx], as_tuple=False).flatten()
            for pos_tensor in row_positions:
                pos = int(pos_tensor.item())
                token_id = int(x_0[row_idx, pos].item())
                decile = _conditional_rank_decile(token_to_decile.get(token_id, 1))
                if len(selected_by_decile[decile]) >= target_per_decile:
                    continue
                selected_by_decile[decile].append(
                    {
                        "batch_idx": int(batch_idx),
                        "row_idx": int(row_idx),
                        "pos": int(pos),
                        "token_id": int(token_id),
                        "decile": int(decile),
                    }
                )
    selected: list[dict[str, int]] = []
    selected_counts: dict[int, int] = {}
    for decile in range(1, 8):
        decile_selected = selected_by_decile[decile]
        selected.extend(decile_selected)
        selected_counts[decile] = len(decile_selected)
    return selected, selected_counts


def _group_conditional_rank_records_by_batch(
    selected: list[dict[str, int]],
) -> dict[int, list[dict[str, int]]]:
    grouped: dict[int, list[dict[str, int]]] = {}
    for record in selected:
        grouped.setdefault(int(record["batch_idx"]), []).append(record)
    return grouped


def _record_stratified_conditional_rank_position(
    decile_stats: dict[int, dict],
    suffix_stats: dict,
    record: dict[str, int],
    scored: dict[str, float | int | bool],
) -> None:
    decile = _conditional_rank_decile(record["decile"])
    _update_conditional_rank_stats(decile_stats[decile], scored)
    _update_conditional_rank_stats(suffix_stats, scored)


def _score_stratified_prefix_context(
    model,
    batch_records: list[dict[str, torch.Tensor]],
    selected: list[dict[str, int]],
    model_logits_fn,
    prefix_len: int,
    mask_id: int,
    vocab_size: int,
    topk_k: int,
) -> tuple[dict[int, dict], dict]:
    decile_stats = _new_conditional_rank_decile_stats()
    suffix_stats = _new_conditional_rank_stats()
    grouped = _group_conditional_rank_records_by_batch(selected)
    for batch_idx, records in grouped.items():
        batch_record = batch_records[batch_idx]
        x_0 = batch_record["x_0"]
        attention_mask = batch_record["attention_mask"]
        position_ids = batch_record.get("position_ids")
        z_eval = x_0.clone()
        z_eval[:, prefix_len:] = mask_id
        eval_attention_mask = attention_mask.clone()
        eval_attention_mask[:, prefix_len:] = False
        boundary_mask = batch_record.get("document_boundary_mask")
        if boundary_mask is not None:
            eval_model_attention_mask = boundary_mask & attention_mask[:, :, None] & eval_attention_mask[:, None, :]
        else:
            eval_model_attention_mask = eval_attention_mask
        logits = _call_model_logits(
            model,
            z_eval,
            x_0,
            eval_model_attention_mask,
            model_logits_fn=model_logits_fn,
            position_ids=position_ids,
        )
        probs = F.softmax(logits.float(), dim=-1)
        for record in records:
            row_idx = int(record["row_idx"])
            pos = int(record["pos"])
            token_id = int(record["token_id"])
            scored = _score_conditional_rank_position(probs[row_idx, pos], token_id, vocab_size, topk_k)
            _record_stratified_conditional_rank_position(decile_stats, suffix_stats, record, scored)
    return decile_stats, suffix_stats


def _partial_suffix_mask_positions(
    suffix_content: torch.Tensor,
    row_idx: int,
    selected_pos: int,
    partial_mask_fraction: float,
) -> list[int]:
    if partial_mask_fraction <= 0:
        return []
    row_positions = torch.nonzero(suffix_content[row_idx], as_tuple=False).flatten()
    other_positions = row_positions[row_positions != int(selected_pos)]
    if other_positions.numel() == 0:
        return []
    other_count = int(round(float(partial_mask_fraction) * int(other_positions.numel())))
    other_count = min(int(other_positions.numel()), max(1, other_count))
    return _selected_suffix_positions(other_positions, other_count)


def _score_stratified_masked_context(
    model,
    batch_records: list[dict[str, torch.Tensor]],
    selected: list[dict[str, int]],
    tokenizer,
    model_logits_fn,
    expanded_batch_size: int,
    vocab_size: int,
    topk_k: int,
    partial_mask_fraction: float | None,
) -> tuple[dict[int, dict], dict]:
    decile_stats = _new_conditional_rank_decile_stats()
    suffix_stats = _new_conditional_rank_stats()
    grouped = _group_conditional_rank_records_by_batch(selected)
    mask_id = int(tokenizer.get_mask_token_id())
    for batch_idx, records in grouped.items():
        batch_record = batch_records[batch_idx]
        x_0 = batch_record["x_0"]
        attention_mask = batch_record["attention_mask"]
        model_attention_mask = batch_record.get("model_attention_mask", attention_mask)
        position_source = batch_record.get("position_ids")
        suffix_content = batch_record["suffix_content"]
        for chunk_start in range(0, len(records), expanded_batch_size):
            chunk = records[chunk_start : chunk_start + expanded_batch_size]
            source_rows = torch.tensor(
                [int(record["row_idx"]) for record in chunk],
                device=x_0.device,
                dtype=torch.long,
            )
            chunk_positions = torch.tensor(
                [int(record["pos"]) for record in chunk],
                device=x_0.device,
                dtype=torch.long,
            )
            z_chunk = x_0.index_select(0, source_rows).clone()
            x_chunk = x_0.index_select(0, source_rows)
            attention_chunk = model_attention_mask.index_select(0, source_rows)
            position_chunk = position_source.index_select(0, source_rows) if position_source is not None else None
            for local_idx, record in enumerate(chunk):
                selected_pos = int(record["pos"])
                z_chunk[local_idx, selected_pos] = mask_id
                if partial_mask_fraction is None:
                    continue
                for other_pos in _partial_suffix_mask_positions(
                    suffix_content,
                    int(record["row_idx"]),
                    selected_pos,
                    partial_mask_fraction,
                ):
                    z_chunk[local_idx, int(other_pos)] = mask_id

            logits = _call_model_logits(
                model,
                z_chunk,
                x_chunk,
                attention_chunk,
                model_logits_fn=model_logits_fn,
                position_ids=position_chunk,
            )
            selected_logits = logits[torch.arange(len(chunk), device=x_0.device), chunk_positions].float()
            selected_probs = F.softmax(selected_logits, dim=-1)
            for local_idx, record in enumerate(chunk):
                token_id = int(record["token_id"])
                scored = _score_conditional_rank_position(selected_probs[local_idx], token_id, vocab_size, topk_k)
                _record_stratified_conditional_rank_position(decile_stats, suffix_stats, record, scored)
    return decile_stats, suffix_stats


def _fill_stratified_conditional_rank_metrics(
    decile_stats: dict[int, dict],
    suffix_stats: dict,
    base_prefix: str,
    metrics_out: dict,
    tokenizer,
    selected_counts: dict[int, int],
    target_per_decile: int,
) -> None:
    total_target = int(target_per_decile) * 7
    total_selected = sum(int(selected_counts.get(decile, 0)) for decile in range(1, 8))
    for decile in range(1, 8):
        prefix = f"{base_prefix}decile_{decile}_"
        selected = int(selected_counts.get(decile, 0))
        _fill_conditional_rank_metrics(decile_stats[decile], prefix, metrics_out, tokenizer)
        metrics_out[f"{prefix}target_count"] = int(target_per_decile)
        metrics_out[f"{prefix}selected_count"] = selected
        metrics_out[f"{prefix}shortfall"] = max(0, int(target_per_decile) - selected)
    suffix_prefix = f"{base_prefix}suffix_"
    _fill_conditional_rank_metrics(suffix_stats, suffix_prefix, metrics_out, tokenizer)
    metrics_out[f"{suffix_prefix}target_count"] = total_target
    metrics_out[f"{suffix_prefix}selected_count"] = total_selected
    metrics_out[f"{suffix_prefix}shortfall"] = max(0, total_target - total_selected)


def _log_stratified_conditional_rank_examples(
    batch_records: list[dict[str, torch.Tensor]],
    selected: list[dict[str, int]],
    tokenizer,
    run_dir: str | None,
    log_samples: int,
    step: int | None,
    target_per_decile: int,
    partial_mask_fraction: float,
    prefix_len: int,
) -> tuple[str, int]:
    if run_dir is None or log_samples <= 0 or not selected:
        return "", 0
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "conditional_rank_stratified_samples.log")
    logged = 0
    with open(path, "a") as log_file:
        log_file.write(
            f"[conditional-rank-stratified] step={step if step is not None else '-'} "
            f"selected_positions={len(selected)} target_per_decile={int(target_per_decile)} "
            f"partial_mask_fraction={float(partial_mask_fraction):.3f}\n"
        )
        for record in selected:
            if logged >= log_samples:
                break
            batch_record = batch_records[int(record["batch_idx"])]
            x_0 = batch_record["x_0"]
            attention_mask = batch_record["attention_mask"]
            row_idx = int(record["row_idx"])
            token_id = int(record["token_id"])
            prefix_attention = attention_mask[row_idx].clone()
            prefix_attention[prefix_len:] = False
            suffix_attention = attention_mask[row_idx].clone()
            suffix_attention[:prefix_len] = False
            log_file.write(
                f"[conditional-rank-stratified] item={logged} decile={int(record['decile'])} "
                f"batch={int(record['batch_idx'])} row={row_idx} pos={int(record['pos'])} "
                f"token_id={token_id} token_text={_single_token_text(tokenizer, token_id)!r}\n"
            )
            log_file.write(
                "[conditional-rank-stratified] prefix="
                f"{_decode_attention_row(tokenizer, x_0[row_idx], prefix_attention)!r}\n"
            )
            log_file.write(
                "[conditional-rank-stratified] suffix="
                f"{_decode_attention_row(tokenizer, x_0[row_idx], suffix_attention)!r}\n"
            )
            logged += 1
    return path, logged


def _evaluate_stratified_conditional_rank(
    model,
    tokenizer,
    batch_records: list[dict[str, torch.Tensor]],
    token_to_decile: dict[int, int],
    model_logits_fn,
    prefix_len: int,
    target_per_decile: int,
    partial_mask_fraction: float,
    expanded_batch_size: int,
    vocab_size: int,
    topk_k: int,
    run_dir: str | None,
    log_samples: int,
    step: int | None,
) -> dict[str, float | int | str]:
    selected, selected_counts = _collect_stratified_conditional_rank_positions(
        batch_records,
        token_to_decile,
        int(target_per_decile),
    )
    prefix_decile_stats, prefix_suffix_stats = _score_stratified_prefix_context(
        model,
        batch_records,
        selected,
        model_logits_fn,
        prefix_len,
        int(tokenizer.get_mask_token_id()),
        vocab_size,
        topk_k,
    )
    partial_decile_stats, partial_suffix_stats = _score_stratified_masked_context(
        model,
        batch_records,
        selected,
        tokenizer,
        model_logits_fn,
        expanded_batch_size,
        vocab_size,
        topk_k,
        partial_mask_fraction=float(partial_mask_fraction),
    )
    loo_decile_stats, loo_suffix_stats = _score_stratified_masked_context(
        model,
        batch_records,
        selected,
        tokenizer,
        model_logits_fn,
        expanded_batch_size,
        vocab_size,
        topk_k,
        partial_mask_fraction=None,
    )
    samples_path, logged_samples = _log_stratified_conditional_rank_examples(
        batch_records,
        selected,
        tokenizer,
        run_dir,
        int(log_samples),
        step,
        int(target_per_decile),
        float(partial_mask_fraction),
        prefix_len,
    )

    selected_total = sum(selected_counts.values())
    target_total = int(target_per_decile) * 7
    metrics: dict[str, float | int | str] = {
        "conditional_rank_stratified_batches": len(batch_records),
        "conditional_rank_stratified_target_per_decile": int(target_per_decile),
        "conditional_rank_stratified_partial_mask_fraction": float(partial_mask_fraction),
        "conditional_rank_stratified_selected_positions": int(selected_total),
        "conditional_rank_stratified_target_positions": int(target_total),
        "conditional_rank_stratified_shortfall": max(0, int(target_total) - int(selected_total)),
        "conditional_rank_stratified_logged_samples": int(logged_samples),
        "conditional_rank_stratified_samples_path": samples_path,
    }
    for decile in range(1, 8):
        selected_count = int(selected_counts.get(decile, 0))
        metrics[f"conditional_rank_stratified_decile_{decile}_selected_count"] = selected_count
        metrics[f"conditional_rank_stratified_decile_{decile}_target_count"] = int(target_per_decile)
        metrics[f"conditional_rank_stratified_decile_{decile}_shortfall"] = max(
            0,
            int(target_per_decile) - selected_count,
        )

    _fill_stratified_conditional_rank_metrics(
        prefix_decile_stats,
        prefix_suffix_stats,
        "conditional_rank_stratified_prefix_",
        metrics,
        tokenizer,
        selected_counts,
        int(target_per_decile),
    )
    _fill_stratified_conditional_rank_metrics(
        partial_decile_stats,
        partial_suffix_stats,
        "conditional_rank_stratified_partial25_",
        metrics,
        tokenizer,
        selected_counts,
        int(target_per_decile),
    )
    _fill_stratified_conditional_rank_metrics(
        loo_decile_stats,
        loo_suffix_stats,
        "conditional_rank_stratified_loo_",
        metrics,
        tokenizer,
        selected_counts,
        int(target_per_decile),
    )
    return metrics


@torch.no_grad()
def evaluate_conditional_rank_suffix(
    model,
    tokenizer,
    data_config,
    device,
    batches=5,
    batch_size: int = 8,
    seq_len: int = MAX_SEQ_LEN,
    model_logits_fn=None,
    include_leave_one_out: bool = False,
    loo_positions_per_sequence: int = 8,
    include_stratified_matched: bool = False,
    stratified_target_per_decile: int = 64,
    stratified_partial_mask_fraction: float = 0.25,
    run_dir: str | None = None,
    stratified_log_samples: int = 8,
    step: int | None = None,
) -> dict[str, float | int | str]:
    """Evaluate suffix conditional rank, optionally adding matched suffix contexts.

    The standard metric masks the full suffix and scores every content suffix
    token from prefix-only context. The leave-one-out variant keeps the suffix
    clean, masks only a bounded sampled set of content suffix positions, and
    scores those positions in expanded minibatches. The stratified matched
    variant chooses the same bounded frequency-decile-balanced suffix positions
    and scores them under prefix-only, partial-suffix, and leave-one-out context.
    """
    if batches <= 0:
        return {}

    special_ids = tokenizer.get_special_token_ids()
    token_bytes = build_token_bytes(tokenizer).to(device)
    token_to_decile, freq_batches = _conditional_rank_frequency_deciles(
        data_config,
        tokenizer,
        batch_size,
        seq_len,
        device,
        batches,
        token_bytes,
        special_ids,
    )

    val_loader = make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)
    mask_id = int(tokenizer.get_mask_token_id())
    prefix_len = seq_len // 2
    vocab_size = tokenizer.get_vocab_size()
    topk_k = min(20, vocab_size)
    expanded_batch_size = max(1, int(batch_size))

    decile_stats = _new_conditional_rank_decile_stats()
    suffix_stats = _new_conditional_rank_stats()
    loo_decile_stats = _new_conditional_rank_decile_stats()
    loo_suffix_stats = _new_conditional_rank_stats()
    loo_selected_positions = 0
    stratified_batch_records: list[dict[str, torch.Tensor]] = []

    for _ in range(batches):
        batch = next(val_loader)
        x_0 = batch.x_0
        attention_mask = batch.attention_mask
        model_attention_mask = _batch_model_attention_mask(batch)
        loss_mask = batch.loss_mask
        target_lengths = (
            target_eos_lengths(x_0, attention_mask, tokenizer.get_eos_token_id())
            if model_uses_target_length_conditioning(model)
            else None
        )
        content = _content_position_mask(x_0, attention_mask, loss_mask, token_bytes, special_ids)
        if include_stratified_matched:
            suffix_content = content.clone()
            suffix_content[:, :prefix_len] = False
            stratified_batch_records.append(
                {
                    "x_0": x_0,
                    "attention_mask": attention_mask,
                    "model_attention_mask": model_attention_mask,
                    "position_ids": getattr(batch, "position_ids", None),
                    "document_boundary_mask": getattr(batch, "document_boundary_mask", None),
                    "suffix_content": suffix_content,
                }
            )

        z_eval = x_0.clone()
        z_eval[:, prefix_len:] = mask_id
        eval_attention_mask = attention_mask.clone()
        eval_attention_mask[:, prefix_len:] = False
        eval_model_attention_mask = _batch_model_attention_mask(
            batch,
            eval_attention_mask,
            query_attention_mask=attention_mask,
        )

        logits = _call_model_logits(
            model,
            z_eval,
            x_0,
            eval_model_attention_mask,
            model_logits_fn=model_logits_fn,
            position_ids=getattr(batch, "position_ids", None),
            target_lengths=target_lengths,
        )
        probs = F.softmax(logits.float(), dim=-1)

        for row_idx in range(x_0.shape[0]):
            for pos in range(prefix_len, seq_len):
                if not bool(content[row_idx, pos].item()):
                    continue
                token_id = int(x_0[row_idx, pos].item())
                scored = _score_conditional_rank_position(probs[row_idx, pos], token_id, vocab_size, topk_k)
                _record_conditional_rank_position(decile_stats, suffix_stats, token_to_decile, token_id, scored)

        if include_leave_one_out:
            loo_selected_positions += _evaluate_loo_conditional_rank_batch(
                model,
                x_0,
                attention_mask,
                model_attention_mask,
                getattr(batch, "position_ids", None),
                loss_mask,
                tokenizer,
                model_logits_fn,
                token_bytes,
                special_ids,
                token_to_decile,
                loo_decile_stats,
                loo_suffix_stats,
                prefix_len,
                loo_positions_per_sequence,
                expanded_batch_size,
                vocab_size,
                topk_k,
            )

    metrics: dict[str, float | int | str] = {
        "conditional_rank_batches": batches,
        "conditional_rank_freq_batches": freq_batches,
    }
    for decile in range(1, 8):
        _fill_conditional_rank_metrics(decile_stats[decile], f"conditional_rank_decile_{decile}_", metrics, tokenizer)
    _fill_conditional_rank_metrics(suffix_stats, "conditional_rank_suffix_", metrics, tokenizer)

    if include_leave_one_out:
        metrics.update(
            {
                "conditional_rank_loo_batches": batches,
                "conditional_rank_loo_freq_batches": freq_batches,
                "conditional_rank_loo_positions_per_sequence": int(loo_positions_per_sequence),
                "conditional_rank_loo_selected_positions": int(loo_selected_positions),
            }
        )
        for decile in range(1, 8):
            _fill_conditional_rank_metrics(
                loo_decile_stats[decile],
                f"conditional_rank_loo_decile_{decile}_",
                metrics,
                tokenizer,
            )
        _fill_conditional_rank_metrics(loo_suffix_stats, "conditional_rank_loo_suffix_", metrics, tokenizer)
    if include_stratified_matched:
        metrics.update(
            _evaluate_stratified_conditional_rank(
                model,
                tokenizer,
                stratified_batch_records,
                token_to_decile,
                model_logits_fn,
                prefix_len,
                int(stratified_target_per_decile),
                float(stratified_partial_mask_fraction),
                expanded_batch_size,
                vocab_size,
                topk_k,
                run_dir,
                int(stratified_log_samples),
                step,
            )
        )
    return metrics
