"""Per-sequence MDLM NELBO scoring and byte-normalized reduction."""

import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F

from . import data
from .eval_model import _batch_model_attention_mask, _call_model_logits
from .eval_nelbo_records import (
    _delete_one_sequence_ratio_se,
    _nelbo_corruption_mask_sha256,
    _nelbo_evaluation_sample_sha256,
    _nelbo_input_identity_sha256,
    _nelbo_uniform_sha256,
    _ordered_sha256_root,
)
from .mdlm_objective import expand_mdlm_time, schedule_weighted_mdlm_terms

NELBO_SEQUENCE_RECORD_SCHEMA_VERSION = 2


@torch.no_grad()
def evaluate_nelbo(
    model,
    data_config,
    tokenizer,
    noise_fn,
    batch_size: int,
    device,
    seq_len: int = data.MAX_SEQ_LEN,
    eval_tokens: int = 40 * 524288,
    model_logits_fn=None,
    schedule_weight: bool = False,
    schedule_weight_scale: float = 1.0,
    return_diagnostics: bool = False,
    *,
    batches: Iterator[data.DLMDataBatch] | None = None,
) -> tuple[float, float, float, bool] | tuple[float, float, float, bool, dict[str, object]]:
    """Estimate the per-byte MDLM diffusion loss.

    Returns ``(nelbo_bpb, masked_ce_bpb, weight_mean, schedule_weighted)``.
    With ``return_diagnostics=True``, an additional diagnostics dictionary
    contains delete-one-sequence ratio-jackknife diagnostics and sample counts.
    ``batches`` optionally supplies already frozen, device-resident batches;
    the ordinary validation loader remains the default. Exactly the requested
    token budget is consumed and every supplied batch must have the declared shape.

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
    token_bytes = data.build_token_bytes(tokenizer).to(device)
    val_loader = (
        data.make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)
        if batches is None
        else iter(batches)
    )
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
        if batch.x_0.shape != (batch_size, seq_len):
            raise ValueError("NELBO batch shape differs from the declared batch_size and seq_len")
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
