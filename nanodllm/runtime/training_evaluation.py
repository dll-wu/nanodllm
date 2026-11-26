from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import build_token_bytes, build_val_loader
from .eval import (
    NELBO_SEQUENCE_RECORD_SCHEMA_VERSION,
    evaluate_ar_next_token_conditional_rank_suffix,
    evaluate_ar_validation_prefix_continuations,
    evaluate_bos_only_first_token,
    evaluate_conditional_generation_metrics,
    evaluate_conditional_rank_rollout,
    evaluate_conditional_rank_suffix,
    evaluate_generation_metrics,
    evaluate_nelbo,
    evaluate_reconstruction_metrics,
)
from .nelbo_time_sampling import MIDPOINT_TIME_SAMPLING, build_nelbo_time_plan
from .sampling import (
    sample_ar,
    sample_bd3lm,
    sample_bd3lm_from_prefixes,
    sample_block_causal_pilot,
    sample_fhs,
    sample_mask_uniform,
    sample_mdlm,
    sample_mdlm_from_prefixes,
    sample_mdlm_left_to_right,
    sample_mdlm_left_to_right_from_prefixes,
    sample_remdm,
    sample_uniform,
)
from .training_artifacts import (
    _ordered_nelbo_digest_root,
    _validated_nelbo_sequence_records,
    write_jsonl_atomic,
)
from .training_evaluation_runtime import EvaluationRuntime as EvaluationRuntime
from .training_evaluation_runtime import run_isolated_evaluation
from .training_features import (
    _effective_eval_sampler,
    _is_causal,
    _uses_bd3lm,
    _uses_block_causal_pilot,
    _uses_per_token_noise_levels,
    _uses_schedule_weighted_nelbo,
    _uses_target_length_conditioning,
)
from .training_metrics import (
    _eos_hazard_accumulator_metrics,
    _eos_hazard_calibration_loss,
    _new_eos_hazard_accumulator,
    _update_eos_hazard_accumulator,
)
from .training_objective import (
    NELBO_CORRUPTION_RNG_SCHEME,
    _corrupt_eval_batch,
    _estimate_target_length_prior,
    _forward_eval_hierarchical_components,
    _forward_eval_logits,
    _population_variance,
    _sample_target_lengths_from_prior,
    _sequence_major_nelbo_corruption_plan,
    _target_lengths_for_conditioning,
    _training_corruption_label,
)
from .training_rng import _restore_rng_state, _rng_state
from .training_runtime import _batch_model_attention_mask
from .training_transition_evaluation import run_token_edit_transition_evaluation

BD3LM_SHARED_SCALAR_TIME_EVAL_LABEL = "bd3lm-shared-scalar-time-diagnostic"
BD3LM_SHARED_SCALAR_TIME_EVAL_SAMPLING = "shared-scalar-time-per-eval-batch-diagnostic"


def _eval_sampler_provenance(config: ExperimentConfig) -> dict[str, str]:
    return {
        "eval_sampler_requested": config.eval.sampler,
        "eval_sampler_effective": _effective_eval_sampler(config),
    }


def _evaluation_corruption_provenance(
    config: ExperimentConfig,
    *,
    training_corruption: str,
    schedule_weighted_nelbo: bool,
) -> dict[str, str | bool]:
    if _uses_bd3lm(config):
        return {
            "eval_corruption": BD3LM_SHARED_SCALAR_TIME_EVAL_LABEL,
            "val_nelbo_corruption": BD3LM_SHARED_SCALAR_TIME_EVAL_LABEL,
            "val_fixed_ce_corruption": BD3LM_SHARED_SCALAR_TIME_EVAL_LABEL,
            "val_nelbo_eval_noise": BD3LM_SHARED_SCALAR_TIME_EVAL_LABEL,
            "val_nelbo_time_sampling": BD3LM_SHARED_SCALAR_TIME_EVAL_SAMPLING,
            "val_fixed_ce_eval_noise": BD3LM_SHARED_SCALAR_TIME_EVAL_LABEL,
            "bd3lm_eval_matches_training_time_sampling": False,
        }
    nelbo_eval_noise = (
        "shared_sequence_time_stratified_schedule_weighted_mdlm"
        if schedule_weighted_nelbo
        else ("standard_scalar_t_mdlm" if _uses_per_token_noise_levels(config) else training_corruption)
    )
    if schedule_weighted_nelbo and config.eval.nelbo_time_sampling != MIDPOINT_TIME_SAMPLING:
        nelbo_eval_noise = f"shared_sequence_time_{config.eval.nelbo_time_sampling}_schedule_weighted_mdlm"
    return {
        "eval_corruption": training_corruption,
        "val_nelbo_corruption": nelbo_eval_noise,
        "val_fixed_ce_corruption": training_corruption,
        "val_nelbo_eval_noise": nelbo_eval_noise,
        "val_nelbo_time_sampling": config.eval.nelbo_time_sampling
        if schedule_weighted_nelbo
        else "training_corruption_sampler",
        "val_fixed_ce_eval_noise": "standard_scalar_t_mdlm"
        if _uses_per_token_noise_levels(config)
        else training_corruption,
    }


def _sample_conditional_generation(
    config: ExperimentConfig,
    model: nn.Module,
    noise,
    tokenizer,
    prefixes: torch.Tensor,
    continuation_len: int,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    """Route one conditional-generation batch without changing MDLM semantics."""

    if _uses_bd3lm(config):
        return sample_bd3lm_from_prefixes(
            model,
            noise,
            tokenizer,
            prefixes,
            int(continuation_len),
            config.eval.sample_steps,
            block_size=int(config.model.bd3lm_block_size),
            scoring_mode=config.eval.sample_scoring_mode,
            token_selection=config.eval.sample_token_selection,
            temperature=config.eval.gen_temperature,
            jitter_scale=config.eval.sample_jitter_scale,
            return_trace=config.eval.sample_trace,
        )

    target_lengths = None
    if _uses_target_length_conditioning(config):
        total_len = int(prefixes.shape[1]) + int(continuation_len)
        target_lengths = torch.full(
            (int(prefixes.shape[0]),),
            total_len,
            dtype=torch.long,
            device=prefixes.device,
        )
    if config.eval.sampler == "left_to_right":
        return sample_mdlm_left_to_right_from_prefixes(
            model,
            noise,
            tokenizer,
            prefixes,
            int(continuation_len),
            config.eval.sample_steps,
            token_selection=config.eval.sample_token_selection,
            temperature=config.eval.gen_temperature,
            top_k=config.eval.sample_top_k,
            top_p=config.eval.sample_top_p,
            return_trace=config.eval.sample_trace,
            target_lengths=target_lengths,
        )
    return sample_mdlm_from_prefixes(
        model,
        noise,
        tokenizer,
        prefixes,
        int(continuation_len),
        config.eval.sample_steps,
        scoring_mode=config.eval.sample_scoring_mode,
        token_selection=config.eval.sample_token_selection,
        temperature=config.eval.gen_temperature,
        jitter_scale=config.eval.sample_jitter_scale,
        return_trace=config.eval.sample_trace,
        target_lengths=target_lengths,
    )


def _run_eval_causal_ar(
    raw_model: nn.Module,
    config: ExperimentConfig,
    tokenizer,
    autocast_ctx,
    device,
    step: int,
    include_gen: bool,
) -> dict:
    token_bytes = build_token_bytes(tokenizer).to(device)
    val_loader = build_val_loader(config, tokenizer, device)
    bos_id = tokenizer.get_bos_token_id()
    eos_id = tokenizer.get_eos_token_id()
    total_ce = 0.0
    total_content_ce = 0.0
    total_tokens = 0
    total_content_tokens = 0
    total_content_bytes = 0
    eos_entropy_sum = 0.0
    eos_top1_hits = 0
    eos_entropy_count = 0
    bucket_sums = {
        "first_generated_token": 0.0,
        "post_bos": 0.0,
        "interior": 0.0,
        "pre_eos": 0.0,
        "post_eos": 0.0,
    }
    bucket_counts = {key: 0 for key in bucket_sums}

    def add_bucket(name: str, ce_values: torch.Tensor, mask: torch.Tensor) -> None:
        if not mask.any():
            return
        bucket_sums[name] += float(ce_values[mask].sum().item())
        bucket_counts[name] += int(mask.sum().item())

    steps = max(1, config.eval.nelbo_tokens // (int(config.train.device_batch_size) * config.model.max_seq_len))
    with torch.no_grad():
        for _ in range(steps):
            batch = next(val_loader)
            x_eval = batch.x_0
            with autocast_ctx:
                model_kwargs = {}
                if getattr(batch, "position_ids", None) is not None:
                    model_kwargs["position_ids"] = batch.position_ids
                model_kwargs["attention_mask"] = _batch_model_attention_mask(batch)
                logits = raw_model(x_eval, **model_kwargs)
            logits_pred = logits[:, :-1, :].contiguous()
            targets = x_eval[:, 1:].contiguous()
            target_mask = batch.loss_mask[:, 1:] & batch.attention_mask[:, 1:]
            ce = F.cross_entropy(
                logits_pred.view(-1, logits_pred.size(-1)),
                targets.view(-1),
                reduction="none",
            ).view_as(targets)
            if target_mask.any():
                total_ce += float(ce[target_mask].sum().item())
                total_tokens += int(target_mask.sum().item())
                content_mask = target_mask & (token_bytes[targets] > 0)
                if content_mask.any():
                    total_content_ce += float(ce[content_mask].sum().item())
                    total_content_tokens += int(content_mask.sum().item())
                    total_content_bytes += int(token_bytes[targets][content_mask].sum().item())
                target_positions = torch.arange(1, x_eval.shape[1], device=x_eval.device).view(1, -1)
                prev_tokens = x_eval[:, :-1].contiguous()
                prev_bos = (
                    prev_tokens == int(bos_id)
                    if bos_id is not None
                    else torch.zeros_like(target_mask, dtype=torch.bool)
                )
                prev_eos = (
                    prev_tokens == int(eos_id)
                    if eos_id is not None
                    else torch.zeros_like(target_mask, dtype=torch.bool)
                )
                target_eos = (
                    targets == int(eos_id) if eos_id is not None else torch.zeros_like(target_mask, dtype=torch.bool)
                )
                eos_eval_mask = target_mask & target_eos
                if eos_eval_mask.any():
                    eos_logits = logits_pred[eos_eval_mask].float()
                    eos_probs = F.softmax(eos_logits, dim=-1)
                    safe_eos_probs = eos_probs.clamp_min(1e-10)
                    eos_entropy_sum += float((-(safe_eos_probs * safe_eos_probs.log()).sum(dim=-1)).sum().item())
                    eos_top1_hits += int((eos_probs.argmax(dim=-1) == int(eos_id)).sum().item())
                    eos_entropy_count += int(eos_eval_mask.sum().item())
                add_bucket("first_generated_token", ce, target_mask & (target_positions == 1))
                add_bucket("post_bos", ce, target_mask & prev_bos)
                add_bucket("interior", ce, content_mask & ~prev_bos & ~prev_eos & ~target_eos)
                add_bucket("pre_eos", ce, target_mask & target_eos)
                add_bucket("post_eos", ce, target_mask & prev_eos)
    metrics: dict[str, float | int | str] = {
        "step": step,
        "eval_ar_batches": steps,
        "eval_ar_tokens": total_tokens,
        "eval_ar_content_tokens": total_content_tokens,
        "eval_ar_content_bytes": total_content_bytes,
        "eval_ar_content_token_byte_fertility": total_content_tokens / total_content_bytes
        if total_content_bytes
        else 0.0,
        "val_ce": total_ce / total_tokens if total_tokens else 0.0,
        "val_content_ce": total_content_ce / total_content_tokens if total_content_tokens else 0.0,
        "ar_gold_context_eos_token_entropy_mean": eos_entropy_sum / eos_entropy_count if eos_entropy_count else 0.0,
        "ar_gold_context_eos_token_top1_hit": eos_top1_hits / eos_entropy_count if eos_entropy_count else 0.0,
        "ar_gold_context_eos_token_count": eos_entropy_count,
        **_eval_sampler_provenance(config),
    }
    for name, ce_sum in bucket_sums.items():
        count = bucket_counts[name]
        metrics[f"val_ar_ce_{name}"] = ce_sum / count if count else 0.0
        metrics[f"val_ar_{name}_tokens"] = count
    ar_bpb = total_content_ce / (math.log(2.0) * total_content_bytes) if total_content_bytes else 0.0
    metrics.update(
        {
            "val_bpb": ar_bpb,
            "val_ar_bpb": ar_bpb,
            "val_content_bpb": ar_bpb,
            "val_bpb_metric": "causal_ar_content_ce_bits_per_byte",
        }
    )
    with autocast_ctx:
        metrics.update(evaluate_bos_only_first_token(raw_model, tokenizer, device))
    if config.eval.conditional_rank_eval:
        with autocast_ctx:
            metrics.update(
                evaluate_ar_next_token_conditional_rank_suffix(
                    raw_model,
                    tokenizer,
                    config.data,
                    device,
                    batches=config.eval.conditional_rank_batches,
                    batch_size=int(config.train.device_batch_size),
                    seq_len=config.model.max_seq_len,
                )
            )
        metrics["ar_gold_context_suffix_top1_hit"] = metrics.get(
            "ar_next_token_conditional_rank_suffix_gold_top1_hit",
            0.0,
        )
        metrics["ar_gold_context_suffix_entropy_mean"] = metrics.get(
            "ar_next_token_conditional_rank_suffix_entropy_mean",
            0.0,
        )
        metrics["ar_gold_context_rare_token_top1_hit"] = metrics.get(
            "ar_next_token_conditional_rank_decile_1_gold_top1_hit",
            0.0,
        )
        metrics["ar_gold_context_rare_token_entropy_mean"] = metrics.get(
            "ar_next_token_conditional_rank_decile_1_entropy_mean",
            0.0,
        )
        metrics["ar_gold_context_common_token_top1_hit"] = metrics.get(
            "ar_next_token_conditional_rank_decile_7_gold_top1_hit",
            0.0,
        )
        metrics["ar_gold_context_common_token_entropy_mean"] = metrics.get(
            "ar_next_token_conditional_rank_decile_7_entropy_mean",
            0.0,
        )
    if include_gen:

        def copy_prefixed(prefix: str, source: dict) -> None:
            for key, value in source.items():
                suffix = key[4:] if key.startswith("gen_") else key
                metrics[f"{prefix}{suffix}"] = value

        def build_sample_fn(token_selection: str):
            def sample_fn(m, n, length):
                return sample_ar(
                    m,
                    tokenizer,
                    n,
                    length,
                    temperature=config.eval.gen_temperature,
                    token_selection=token_selection,
                )

            return sample_fn

        primary_token_selection = config.eval.sample_token_selection
        secondary_token_selection = "multinomial" if primary_token_selection == "argmax" else "argmax"
        metrics["gen_primary_token_selection"] = primary_token_selection
        metrics["gen_secondary_token_selection"] = secondary_token_selection
        for token_selection in (primary_token_selection, secondary_token_selection):
            with autocast_ctx:
                gen_metrics = evaluate_generation_metrics(
                    raw_model,
                    tokenizer,
                    None,
                    build_sample_fn(token_selection),
                    int(config.train.device_batch_size),
                    config.train.run_dir,
                    device=device,
                    judge_model_id=config.eval.judge_model_id,
                    gen_samples=config.eval.gen_samples,
                    gen_len=config.eval.gen_len,
                    log_samples=config.eval.gen_log_samples,
                    score_mode=config.eval.gen_score_mode,
                    self_score_t=config.eval.gen_self_score_t,
                    step=step,
                    sample_label=f"free_run_{token_selection}",
                    model_logits_fn=None,
                )
            if token_selection == primary_token_selection:
                metrics.update(gen_metrics)
            copy_prefixed(f"ar_free_run_{token_selection}_", gen_metrics)

        prefix_len = max(1, min(16, int(config.eval.gen_len) - 1, int(config.model.max_seq_len) - 1))
        for token_selection in (primary_token_selection, secondary_token_selection):
            with autocast_ctx:
                prefix_metrics = evaluate_ar_validation_prefix_continuations(
                    raw_model,
                    tokenizer,
                    config.data,
                    int(config.train.device_batch_size),
                    device=device,
                    seq_len=config.model.max_seq_len,
                    gen_samples=config.eval.gen_samples,
                    gen_len=config.eval.gen_len,
                    prefix_len=prefix_len,
                    temperature=config.eval.gen_temperature,
                    token_selection=token_selection,
                    run_dir=config.train.run_dir,
                    log_samples=config.eval.gen_log_samples,
                    step=step,
                )
            for key, value in prefix_metrics.items():
                metrics[f"ar_val_prefix_{token_selection}_{key}"] = value
    return metrics


def _run_eval_unisolated(
    raw_model: nn.Module,
    config: ExperimentConfig,
    tokenizer,
    noise,
    autocast_ctx,
    device,
    step: int,
    include_gen: bool,
) -> dict:
    was_training = raw_model.training
    raw_model.eval()
    if _is_causal(config):
        metrics = _run_eval_causal_ar(
            raw_model,
            config,
            tokenizer,
            autocast_ctx,
            device,
            step,
            include_gen,
        )
        if was_training:
            raw_model.train()
        return metrics
    corruption_label = _training_corruption_label(config)
    use_hierarchical_output = hasattr(raw_model, "uses_hierarchical_output") and raw_model.uses_hierarchical_output()
    schedule_weighted_nelbo = _uses_schedule_weighted_nelbo(config)
    nelbo_t_min = float(config.eval.nelbo_t_min)
    nelbo_t_max = float(config.eval.nelbo_t_max)
    nelbo_integration_width = nelbo_t_max - nelbo_t_min
    nelbo_full_interval = bool(nelbo_t_min == 0.0 and nelbo_t_max == 1.0)
    eval_batch_size = int(config.train.device_batch_size)
    nelbo_steps = max(1, int(config.eval.nelbo_tokens) // (eval_batch_size * int(config.model.max_seq_len)))
    nelbo_sequence_count = nelbo_steps * eval_batch_size
    nelbo_time_plan = (
        build_nelbo_time_plan(
            nelbo_sequence_count,
            nelbo_t_min,
            nelbo_t_max,
            int(config.eval.nelbo_seed),
            config.eval.nelbo_time_sampling,
        )
        if schedule_weighted_nelbo
        else None
    )
    nelbo_times = nelbo_time_plan.times if nelbo_time_plan is not None else None
    nelbo_corruption_plan: torch.Tensor | None = None
    nelbo_corruption_derived_seed: int | None = None
    if schedule_weighted_nelbo:
        nelbo_corruption_plan, nelbo_corruption_derived_seed = _sequence_major_nelbo_corruption_plan(
            nelbo_sequence_count,
            int(config.model.max_seq_len),
            int(config.eval.nelbo_seed),
        )
    nelbo_time_offset = 0
    metrics: dict[str, float | int | str | bool | None] = {
        "step": step,
        "eval_nelbo_tokens": config.eval.nelbo_tokens,
        "eval_fixed_ce_batches": config.eval.fixed_ce_batches,
        **_eval_sampler_provenance(config),
        **_evaluation_corruption_provenance(
            config,
            training_corruption=corruption_label,
            schedule_weighted_nelbo=schedule_weighted_nelbo,
        ),
        "val_nelbo_seed": int(config.eval.nelbo_seed) if schedule_weighted_nelbo else None,
        "val_nelbo_t_min": nelbo_t_min if schedule_weighted_nelbo else None,
        "val_nelbo_t_max": nelbo_t_max if schedule_weighted_nelbo else None,
        "val_nelbo_integration_width": nelbo_integration_width if schedule_weighted_nelbo else None,
        "val_nelbo_full_interval": bool(schedule_weighted_nelbo and nelbo_full_interval),
        "val_nelbo_sequence_count": nelbo_sequence_count if schedule_weighted_nelbo else 0,
        "val_nelbo_corruption_rng_scheme": NELBO_CORRUPTION_RNG_SCHEME if schedule_weighted_nelbo else None,
        "val_nelbo_corruption_rng_base_seed": int(config.eval.nelbo_seed) if schedule_weighted_nelbo else None,
        "val_nelbo_corruption_rng_derived_seed": nelbo_corruption_derived_seed,
        "val_nelbo_corruption_plan_shape": [nelbo_sequence_count, int(config.model.max_seq_len)]
        if schedule_weighted_nelbo
        else None,
        **(nelbo_time_plan.metadata() if nelbo_time_plan is not None else {}),
    }
    target_length_prior_counts: torch.Tensor | None = None
    if _uses_target_length_conditioning(config):
        saved_state = _rng_state(config)
        try:
            target_length_prior_counts, target_length_prior_metrics = _estimate_target_length_prior(
                config,
                tokenizer,
                int(config.train.device_batch_size),
                int(config.model.max_seq_len),
                device,
                int(config.eval.target_length_distribution_batches),
            )
        finally:
            _restore_rng_state(saved_state, config)
        metrics.update(target_length_prior_metrics)
        total_prior = float(target_length_prior_counts.float().sum().item())
        visible_prior = float(target_length_prior_counts[: int(config.eval.gen_len)].float().sum().item())
        metrics["target_length_prior_visible_count_at_gen_len"] = int(visible_prior)
        metrics["target_length_prior_visible_fraction_at_gen_len"] = visible_prior / total_prior if total_prior else 0.0

    def noise_fn(x_eval, loss_mask):
        nonlocal nelbo_time_offset
        if schedule_weighted_nelbo:
            assert nelbo_times is not None
            assert nelbo_corruption_plan is not None
            start_offset = nelbo_time_offset
            next_offset = nelbo_time_offset + int(x_eval.shape[0])
            if next_offset > int(nelbo_times.numel()):
                raise RuntimeError("NELBO evaluator consumed more sequences than its fixed time design")
            t_rows = nelbo_times[start_offset:next_offset].to(device=x_eval.device)
            uniform_rows_cpu = nelbo_corruption_plan[start_offset:next_offset]
            nelbo_time_offset = next_offset
            z_eval, w_eval = noise.corrupt(
                x_eval,
                t_rows,
                maskable=loss_mask,
                uniforms=uniform_rows_cpu.to(device=x_eval.device),
            )
            noise_level = t_rows if config.noise.time_conditioning else None
            return z_eval, t_rows, w_eval, noise_level, {"corruption_uniforms": uniform_rows_cpu}
        z_eval, w_eval, t_val, noise_level = _corrupt_eval_batch(config, noise, x_eval, loss_mask)
        return z_eval, t_val, w_eval, noise_level

    def model_logits_fn(
        m,
        z_eval,
        x_eval,
        attention_mask,
        noise_level=None,
        position_ids=None,
        target_lengths=None,
    ):
        if target_lengths is None:
            target_lengths = _target_lengths_for_conditioning(
                config,
                x_eval,
                attention_mask,
                tokenizer.get_eos_token_id(),
            )
        return _forward_eval_logits(
            m,
            config,
            z_eval,
            x_eval,
            attention_mask,
            noise_level=noise_level,
            position_ids=position_ids,
            target_lengths=target_lengths,
        )

    with autocast_ctx:
        nelbo_result = evaluate_nelbo(
            raw_model,
            config.data,
            tokenizer,
            noise_fn,
            int(config.train.device_batch_size),
            device=device,
            seq_len=config.model.max_seq_len,
            eval_tokens=config.eval.nelbo_tokens,
            model_logits_fn=model_logits_fn,
            schedule_weight=schedule_weighted_nelbo,
            schedule_weight_scale=nelbo_integration_width,
            return_diagnostics=True,
        )
        has_nelbo_diagnostics = len(nelbo_result) == 5
        if has_nelbo_diagnostics:
            if schedule_weighted_nelbo and nelbo_time_offset != nelbo_sequence_count:
                raise RuntimeError(
                    "NELBO evaluator did not consume its complete fixed corruption plan: "
                    f"{nelbo_time_offset} != {nelbo_sequence_count}"
                )
            nelbo, masked_ce_bpb, weight_mean, schedule_weighted, nelbo_diagnostics = nelbo_result
        else:  # Backward-compatible with small monkeypatched evaluator tests.
            nelbo, masked_ce_bpb, weight_mean, schedule_weighted = nelbo_result
            nelbo_diagnostics = {}
        if has_nelbo_diagnostics:
            sequence_records = _validated_nelbo_sequence_records(
                nelbo_diagnostics.pop("sequence_records", None),
                expected_count=nelbo_sequence_count,
                require_shared_time=schedule_weighted_nelbo,
            )
            diagnostic_sequence_count = nelbo_diagnostics.get("sequence_count")
            if diagnostic_sequence_count != len(sequence_records):
                raise ValueError(
                    "NELBO diagnostic sequence_count does not match the persisted sequence record count: "
                    f"{diagnostic_sequence_count!r} != {len(sequence_records)}"
                )
            total_record_bytes = math.fsum(float(record["byte_denominator"]) for record in sequence_records)
            reconstructed_nelbo = math.fsum(float(record["nelbo_nats_numerator"]) for record in sequence_records) / (
                math.log(2) * total_record_bytes
            )
            reconstructed_masked_ce = math.fsum(
                float(record["masked_nats_numerator"]) for record in sequence_records
            ) / (math.log(2) * total_record_bytes)
            if not math.isclose(reconstructed_nelbo, float(nelbo), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("NELBO sequence records do not reconstruct val_nelbo_bpb")
            if not math.isclose(reconstructed_masked_ce, float(masked_ce_bpb), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("NELBO sequence records do not reconstruct val_masked_ce_bpb")
            expected_totals = {
                "total_bytes": total_record_bytes,
                "masked_target_count": sum(int(record["masked_target_count"]) for record in sequence_records),
                "schedule_weight_count": sum(int(record["schedule_weight_count"]) for record in sequence_records),
            }
            for diagnostic_key, expected_value in expected_totals.items():
                if nelbo_diagnostics.get(diagnostic_key) != expected_value:
                    raise ValueError(
                        f"NELBO diagnostic {diagnostic_key} does not match sequence records: "
                        f"{nelbo_diagnostics.get(diagnostic_key)!r} != {expected_value!r}"
                    )
            expected_identity_roots = {
                "input_identity_root_sha256": _ordered_nelbo_digest_root(
                    b"nanodllm-nelbo-input-identity-root-v2", sequence_records, "input_identity_sha256"
                ),
                "corruption_mask_root_sha256": _ordered_nelbo_digest_root(
                    b"nanodllm-nelbo-corruption-mask-root-v2", sequence_records, "corruption_mask_sha256"
                ),
            }
            if schedule_weighted_nelbo:
                expected_identity_roots.update(
                    {
                        "corruption_uniform_root_sha256": _ordered_nelbo_digest_root(
                            b"nanodllm-nelbo-corruption-uniform-root-v1",
                            sequence_records,
                            "corruption_uniform_sha256",
                        ),
                        "evaluation_sample_root_sha256": _ordered_nelbo_digest_root(
                            b"nanodllm-nelbo-evaluation-sample-root-v2",
                            sequence_records,
                            "evaluation_sample_sha256",
                        ),
                    }
                )
            for diagnostic_key, expected_root in expected_identity_roots.items():
                if nelbo_diagnostics.get(diagnostic_key) != expected_root:
                    raise ValueError(
                        f"NELBO diagnostic {diagnostic_key} does not match sequence records: "
                        f"{nelbo_diagnostics.get(diagnostic_key)!r} != {expected_root!r}"
                    )
                metrics[f"val_nelbo_{diagnostic_key}"] = expected_root
            sequence_artifact_relpath = (
                f"telemetry/nelbo_sequences_step{step}.v{NELBO_SEQUENCE_RECORD_SCHEMA_VERSION}.jsonl"
            )
            sequence_artifact_path = os.path.join(config.train.run_dir, sequence_artifact_relpath)
            enriched_sequence_records = [
                {
                    **record,
                    "evaluation_step": int(step),
                    "nelbo_seed": metrics["val_nelbo_seed"],
                    "time_sampling": metrics["val_nelbo_time_sampling"],
                    "corruption_rng_scheme": metrics["val_nelbo_corruption_rng_scheme"],
                }
                for record in sequence_records
            ]
            metrics["val_nelbo_sequence_records_artifact"] = sequence_artifact_relpath
            metrics["val_nelbo_sequence_records_schema_version"] = NELBO_SEQUENCE_RECORD_SCHEMA_VERSION
            metrics["val_nelbo_sequence_records_count"] = len(enriched_sequence_records)
            metrics["val_nelbo_sequence_records_sha256"] = write_jsonl_atomic(
                sequence_artifact_path,
                enriched_sequence_records,
            )
        metrics["val_nelbo_bpb"] = nelbo
        metrics["val_masked_ce_bpb"] = masked_ce_bpb
        metrics["val_nelbo_is_schedule_weighted_bound"] = bool(schedule_weighted and nelbo_full_interval)
        metrics["val_nelbo_is_truncated_reference_integral"] = bool(schedule_weighted and not nelbo_full_interval)
        if use_hierarchical_output:
            metrics["hierarchical_val_nelbo_bpb"] = nelbo
        metrics["val_nelbo_schedule_weighted"] = schedule_weighted
        metrics["val_nelbo_weight_mean"] = weight_mean
        metrics["val_nelbo_effective_importance_weight_mean_masked_content_targets"] = weight_mean
        metrics["val_nelbo_weight_mean_semantics"] = (
            "integration_width_over_t_mean_on_masked_content_targets" if schedule_weighted else "not_applicable"
        )
        for source_key, metric_key in (
            ("nelbo_bpb_se", "val_nelbo_bpb_se"),
            ("masked_ce_bpb_se", "val_masked_ce_bpb_se"),
            ("se_method", "val_nelbo_se_method"),
            ("se_unit", "val_nelbo_jackknife_unit"),
            ("block_count", "val_nelbo_block_count"),
            ("sequence_count", "val_nelbo_jackknife_sequence_count"),
            ("total_bytes", "val_nelbo_total_bytes"),
            ("masked_target_count", "val_nelbo_masked_target_count"),
            ("schedule_weight_count", "val_nelbo_schedule_weight_count"),
        ):
            if source_key in nelbo_diagnostics:
                metrics[metric_key] = nelbo_diagnostics[source_key]
        if schedule_weighted_nelbo and config.eval.nelbo_time_sampling != MIDPOINT_TIME_SAMPLING:
            metrics["val_nelbo_se_method"] = "delete_one_sequence_jackknife_ratio_conditional_on_realized_time_plan"

    token_bytes = build_token_bytes(tokenizer).to(device)
    eos_token_id = tokenizer.get_eos_token_id()
    val_loader = build_val_loader(config, tokenizer, device)
    with torch.no_grad():
        for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
            all_ce, content_ce = [], []
            class_ce, within_class_ce = [], []
            content_class_ce, content_within_class_ce = [], []
            class_entropy, class_max_prob = [], []
            class_prob_eos_special, class_prob_low_freq, class_prob_high_freq = [], [], []
            eos_hazard_losses = []
            eos_hazard_accumulator = _new_eos_hazard_accumulator()
            for _ in range(config.eval.fixed_ce_batches):
                batch = next(val_loader)
                eval_attention_mask = _batch_model_attention_mask(batch)
                x_eval = batch.x_0
                z_eval, w_eval, _, noise_level = _corrupt_eval_batch(
                    config,
                    noise,
                    x_eval,
                    batch.loss_mask,
                    t_val=t_val,
                )
                with autocast_ctx:
                    if use_hierarchical_output:
                        class_logits, token_logits = _forward_eval_hierarchical_components(
                            raw_model,
                            config,
                            z_eval,
                            x_eval,
                            eval_attention_mask,
                            noise_level=noise_level,
                            position_ids=getattr(batch, "position_ids", None),
                        )
                        ce, class_ce_values, within_class_ce_values = raw_model.hierarchical_ce(
                            class_logits,
                            token_logits,
                            x_eval,
                        )
                    else:
                        logits = _forward_eval_logits(
                            raw_model,
                            config,
                            z_eval,
                            x_eval,
                            eval_attention_mask,
                            noise_level=noise_level,
                            position_ids=getattr(batch, "position_ids", None),
                            target_lengths=_target_lengths_for_conditioning(
                                config,
                                x_eval,
                                batch.attention_mask,
                                eos_token_id,
                            ),
                        )
                        ce = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), x_eval.view(-1), reduction="none"
                        ).view_as(x_eval)
                        class_ce_values = None
                        within_class_ce_values = None
                active = (w_eval > 0) & batch.loss_mask
                content = active & (token_bytes[x_eval] > 0)
                all_ce.append(float(ce[active].mean().item()))
                content_ce.append(float(ce[content].mean().item()))
                if (
                    config.train.eos_hazard_loss_weight > 0
                    and not use_hierarchical_output
                    and eos_token_id is not None
                    and active.any()
                ):
                    active_indices = active.reshape(-1).nonzero(as_tuple=True)[0]
                    active_targets = x_eval.reshape(-1).index_select(0, active_indices)
                    active_logits = logits.reshape(-1, logits.size(-1)).index_select(0, active_indices)
                    eos_hazard_loss, eos_hazard_stats = _eos_hazard_calibration_loss(
                        active_logits,
                        active_targets,
                        active_indices,
                        batch.attention_mask,
                        eos_token_id,
                    )
                    eos_hazard_losses.append(float(eos_hazard_loss.detach().item()))
                    _update_eos_hazard_accumulator(eos_hazard_accumulator, eos_hazard_stats)
                if use_hierarchical_output and class_ce_values is not None and within_class_ce_values is not None:
                    class_ce.append(float(class_ce_values[active].mean().item()))
                    within_class_ce.append(float(within_class_ce_values[active].mean().item()))
                    content_class_ce.append(float(class_ce_values[content].mean().item()))
                    content_within_class_ce.append(float(within_class_ce_values[content].mean().item()))
                    probs = F.softmax(class_logits.float(), dim=-1)
                    active_probs = probs[active]
                    safe_probs = active_probs.clamp_min(1e-10)
                    class_entropy.append(float((-(safe_probs * safe_probs.log()).sum(dim=-1)).mean().item()))
                    class_max_prob.append(float(active_probs.max(dim=-1).values.mean().item()))
                    class_prob_eos_special.append(float(active_probs[:, 0].mean().item()))
                    if active_probs.shape[-1] > 1:
                        class_prob_low_freq.append(float(active_probs[:, 1].mean().item()))
                    if active_probs.shape[-1] > 7:
                        class_prob_high_freq.append(float(active_probs[:, 7].mean().item()))
            key = f"t{int(t_val * 10):02d}"
            metrics[f"val_ce_{key}"] = sum(all_ce) / len(all_ce)
            metrics[f"val_ce_{key}_batch_variance"] = _population_variance(all_ce)
            metrics[f"val_content_ce_{key}"] = sum(content_ce) / len(content_ce)
            metrics[f"val_content_ce_{key}_batch_variance"] = _population_variance(content_ce)
            if config.train.eos_hazard_loss_weight > 0:
                eos_hazard_loss = sum(eos_hazard_losses) / len(eos_hazard_losses) if eos_hazard_losses else 0.0
                metrics.update(
                    _eos_hazard_accumulator_metrics(
                        f"val_{key}_",
                        eos_hazard_accumulator,
                        eos_hazard_loss,
                        eos_hazard_loss * float(config.train.eos_hazard_loss_weight),
                        float(config.train.eos_hazard_loss_weight),
                    )
                )
            if use_hierarchical_output:
                metrics[f"val_class_ce_{key}"] = sum(class_ce) / len(class_ce) if class_ce else 0.0
                metrics[f"val_within_class_ce_{key}"] = (
                    sum(within_class_ce) / len(within_class_ce) if within_class_ce else 0.0
                )
                metrics[f"val_content_class_ce_{key}"] = (
                    sum(content_class_ce) / len(content_class_ce) if content_class_ce else 0.0
                )
                metrics[f"val_content_within_class_ce_{key}"] = (
                    sum(content_within_class_ce) / len(content_within_class_ce) if content_within_class_ce else 0.0
                )
                metrics[f"val_class_entropy_{key}"] = sum(class_entropy) / len(class_entropy) if class_entropy else 0.0
                metrics[f"val_class_max_prob_{key}"] = (
                    sum(class_max_prob) / len(class_max_prob) if class_max_prob else 0.0
                )
                metrics[f"val_class_prob_eos_special_{key}"] = (
                    sum(class_prob_eos_special) / len(class_prob_eos_special) if class_prob_eos_special else 0.0
                )
                metrics[f"val_class_prob_low_freq_{key}"] = (
                    sum(class_prob_low_freq) / len(class_prob_low_freq) if class_prob_low_freq else 0.0
                )
                metrics[f"val_class_prob_high_freq_{key}"] = (
                    sum(class_prob_high_freq) / len(class_prob_high_freq) if class_prob_high_freq else 0.0
                )

    if config.noise.type == "mdlm" and config.noise.suffix_masking:
        metrics["val_indep_fixed_ce_corruption"] = "independent-mdlm-cross-objective"
        with torch.no_grad():
            for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
                all_ce, content_ce = [], []
                class_ce, within_class_ce = [], []
                content_class_ce, content_within_class_ce = [], []
                for _ in range(config.eval.fixed_ce_batches):
                    batch = next(val_loader)
                    eval_attention_mask = _batch_model_attention_mask(batch)
                    x_eval = batch.x_0
                    z_eval, w_eval = noise.corrupt(x_eval, t_val, maskable=batch.loss_mask)
                    with autocast_ctx:
                        if use_hierarchical_output:
                            class_logits, token_logits = _forward_eval_hierarchical_components(
                                raw_model,
                                config,
                                z_eval,
                                x_eval,
                                eval_attention_mask,
                                noise_level=None,
                                position_ids=getattr(batch, "position_ids", None),
                            )
                            ce, class_ce_values, within_class_ce_values = raw_model.hierarchical_ce(
                                class_logits,
                                token_logits,
                                x_eval,
                            )
                        else:
                            logits = _forward_eval_logits(
                                raw_model,
                                config,
                                z_eval,
                                x_eval,
                                eval_attention_mask,
                                noise_level=None,
                                position_ids=getattr(batch, "position_ids", None),
                                target_lengths=_target_lengths_for_conditioning(
                                    config,
                                    x_eval,
                                    batch.attention_mask,
                                    eos_token_id,
                                ),
                            )
                            ce = F.cross_entropy(
                                logits.view(-1, logits.size(-1)), x_eval.view(-1), reduction="none"
                            ).view_as(x_eval)
                            class_ce_values = None
                            within_class_ce_values = None
                    active = (w_eval > 0) & batch.loss_mask
                    content = active & (token_bytes[x_eval] > 0)
                    all_ce.append(float(ce[active].mean().item()))
                    content_ce.append(float(ce[content].mean().item()))
                    if use_hierarchical_output and class_ce_values is not None and within_class_ce_values is not None:
                        class_ce.append(float(class_ce_values[active].mean().item()))
                        within_class_ce.append(float(within_class_ce_values[active].mean().item()))
                        content_class_ce.append(float(class_ce_values[content].mean().item()))
                        content_within_class_ce.append(float(within_class_ce_values[content].mean().item()))
                key = f"t{int(t_val * 10):02d}"
                metrics[f"val_indep_ce_{key}"] = sum(all_ce) / len(all_ce)
                metrics[f"val_indep_content_ce_{key}"] = sum(content_ce) / len(content_ce)
                if use_hierarchical_output:
                    metrics[f"val_indep_class_ce_{key}"] = sum(class_ce) / len(class_ce) if class_ce else 0.0
                    metrics[f"val_indep_within_class_ce_{key}"] = (
                        sum(within_class_ce) / len(within_class_ce) if within_class_ce else 0.0
                    )
                    metrics[f"val_indep_content_class_ce_{key}"] = (
                        sum(content_class_ce) / len(content_class_ce) if content_class_ce else 0.0
                    )
                    metrics[f"val_indep_content_within_class_ce_{key}"] = (
                        sum(content_within_class_ce) / len(content_within_class_ce) if content_within_class_ce else 0.0
                    )

    if config.eval.reconstruction_eval:
        reconstruction_corrupt_fn = None
        reconstruction_corruption_label = (
            BD3LM_SHARED_SCALAR_TIME_EVAL_LABEL if _uses_bd3lm(config) else "independent-mdlm"
        )
        if config.noise.type == "mdlm" and config.noise.suffix_masking:
            reconstruction_corruption_label = corruption_label

            def reconstruction_corrupt_fn(x_eval, loss_mask, t_val):
                z_eval, w_eval, _t, noise_level = _corrupt_eval_batch(
                    config,
                    noise,
                    x_eval,
                    loss_mask,
                    t_val=t_val,
                )
                return z_eval, w_eval, noise_level

        with autocast_ctx:
            metrics.update(
                evaluate_reconstruction_metrics(
                    raw_model,
                    config.data,
                    tokenizer,
                    noise,
                    int(config.train.device_batch_size),
                    device=device,
                    seq_len=config.model.max_seq_len,
                    t_values=config.eval.reconstruction_t_values,
                    batches=config.eval.reconstruction_batches,
                    run_dir=config.train.run_dir,
                    log_samples=config.eval.reconstruction_log_samples,
                    step=step,
                    model_logits_fn=model_logits_fn,
                    corrupt_fn=reconstruction_corrupt_fn,
                    corruption_label=reconstruction_corruption_label,
                )
            )

    if config.eval.token_edit_transition_eval:
        metrics.update(
            run_token_edit_transition_evaluation(
                raw_model,
                config,
                tokenizer,
                autocast_ctx,
                device,
                step,
                model_logits_fn,
            )
        )

    if config.eval.conditional_rank_eval:
        with autocast_ctx:
            metrics.update(
                evaluate_conditional_rank_suffix(
                    raw_model,
                    tokenizer,
                    config.data,
                    device,
                    batches=config.eval.conditional_rank_batches,
                    batch_size=int(config.train.device_batch_size),
                    seq_len=config.model.max_seq_len,
                    model_logits_fn=model_logits_fn,
                    include_leave_one_out=config.eval.conditional_rank_loo_eval,
                    loo_positions_per_sequence=config.eval.conditional_rank_loo_positions_per_sequence,
                    include_stratified_matched=config.eval.conditional_rank_stratified_eval,
                    stratified_target_per_decile=config.eval.conditional_rank_stratified_target_per_decile,
                    stratified_partial_mask_fraction=config.eval.conditional_rank_stratified_partial_mask_fraction,
                    run_dir=config.train.run_dir,
                    stratified_log_samples=config.eval.conditional_rank_stratified_log_samples,
                    step=step,
                )
            )

    if config.eval.conditional_rank_rollout_eval:
        with autocast_ctx:
            metrics.update(
                evaluate_conditional_rank_rollout(
                    raw_model,
                    tokenizer,
                    config.data,
                    device,
                    batches=config.eval.conditional_rank_rollout_batches,
                    batch_size=int(config.train.device_batch_size),
                    seq_len=config.model.max_seq_len,
                    model_logits_fn=model_logits_fn,
                    commit_fractions=config.eval.conditional_rank_rollout_commit_fractions,
                    sample_steps=config.eval.sample_steps,
                    sampler=config.eval.sampler,
                    scoring_mode=config.eval.sample_scoring_mode,
                    token_selection=config.eval.sample_token_selection,
                    temperature=config.eval.gen_temperature,
                    jitter_scale=config.eval.sample_jitter_scale,
                    fhs_tau=config.eval.fhs_tau,
                    run_dir=config.train.run_dir,
                    log_samples=config.eval.conditional_rank_rollout_log_samples,
                    step=step,
                )
            )

    if config.eval.conditional_gen_eval:

        def conditional_sample_fn(m, prefixes, continuation_len):
            return _sample_conditional_generation(
                config,
                m,
                noise,
                tokenizer,
                prefixes,
                int(continuation_len),
            )

        with autocast_ctx:
            metrics.update(
                evaluate_conditional_generation_metrics(
                    raw_model,
                    tokenizer,
                    config.data,
                    noise,
                    conditional_sample_fn,
                    int(config.train.device_batch_size),
                    config.train.run_dir,
                    device=device,
                    seq_len=config.model.max_seq_len,
                    prefix_fracs=config.eval.conditional_gen_prefix_fracs,
                    gen_samples=config.eval.conditional_gen_samples,
                    continuation_len=config.eval.conditional_gen_len,
                    log_samples=config.eval.conditional_gen_log_samples,
                    score_mode=config.eval.gen_score_mode,
                    self_score_t=config.eval.gen_self_score_t,
                    step=step,
                    sample_label=config.eval.sample_token_selection,
                    model_logits_fn=model_logits_fn,
                )
            )

    if include_gen:

        def build_sample_fn(token_selection: str):
            def sample_fn(m, n, length):
                target_lengths = _sample_target_lengths_from_prior(
                    target_length_prior_counts,
                    n,
                    length,
                    device,
                )
                if _uses_bd3lm(config):
                    return sample_bd3lm(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        block_size=int(config.model.bd3lm_block_size),
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=token_selection,
                        temperature=config.eval.gen_temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        return_trace=config.eval.sample_trace,
                    )
                if _uses_block_causal_pilot(config):
                    return sample_block_causal_pilot(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        block_size=int(config.model.block_causal_pilot_size),
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=token_selection,
                        temperature=config.eval.gen_temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        tau=config.eval.fhs_tau,
                        length_policy=config.eval.length_policy,
                        return_trace=config.eval.sample_trace,
                    )
                if config.eval.sampler == "left_to_right":
                    return sample_mdlm_left_to_right(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        token_selection=token_selection,
                        temperature=config.eval.gen_temperature,
                        top_k=config.eval.sample_top_k,
                        top_p=config.eval.sample_top_p,
                        return_trace=config.eval.sample_trace,
                        target_lengths=target_lengths,
                    )
                if config.eval.sampler == "remdm":
                    return sample_remdm(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=token_selection,
                        temperature=config.eval.gen_temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        tau_edit=config.eval.remdm_tau_edit,
                        return_trace=config.eval.sample_trace,
                        target_lengths=target_lengths,
                    )
                if config.eval.sampler == "fhs":
                    return sample_fhs(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=token_selection,
                        temperature=config.eval.gen_temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        tau=config.eval.fhs_tau,
                        length_policy=config.eval.length_policy,
                        return_trace=config.eval.sample_trace,
                        target_lengths=target_lengths,
                    )
                if config.noise.type == "mdlm":
                    return sample_mdlm(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=token_selection,
                        temperature=config.eval.gen_temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        return_trace=config.eval.sample_trace,
                        target_lengths=target_lengths,
                    )
                sample_impl = {
                    "uniform": sample_uniform,
                    "mask_uniform": sample_mask_uniform,
                }[config.noise.type]
                return sample_impl(
                    m,
                    noise,
                    tokenizer,
                    n,
                    length,
                    config.eval.sample_steps,
                    scoring_mode=config.eval.sample_scoring_mode,
                    token_selection=token_selection,
                    temperature=config.eval.gen_temperature,
                    jitter_scale=config.eval.sample_jitter_scale,
                    return_trace=config.eval.sample_trace,
                )

            return sample_fn

        primary_token_selection = config.eval.sample_token_selection
        metrics["gen_primary_token_selection"] = primary_token_selection
        with autocast_ctx:
            metrics.update(
                evaluate_generation_metrics(
                    raw_model,
                    tokenizer,
                    noise,
                    build_sample_fn(primary_token_selection),
                    int(config.train.device_batch_size),
                    config.train.run_dir,
                    device=device,
                    judge_model_id=config.eval.judge_model_id,
                    gen_samples=config.eval.gen_samples,
                    gen_len=config.eval.gen_len,
                    log_samples=config.eval.gen_log_samples,
                    score_mode=config.eval.gen_score_mode,
                    self_score_t=config.eval.gen_self_score_t,
                    step=step,
                    sample_label=primary_token_selection,
                    model_logits_fn=model_logits_fn,
                    block_size=int(config.model.block_causal_pilot_size) if _uses_block_causal_pilot(config) else 0,
                )
            )
        if primary_token_selection == "argmax":
            metrics["gen_argmax_eos_rate"] = float(metrics.get("gen_eos_rate", 0.0))
        elif primary_token_selection == "multinomial":
            metrics["gen_stochastic_eos_rate"] = float(metrics.get("gen_eos_rate", 0.0))

        if config.noise.type == "mdlm" and (config.noise.suffix_masking or config.train.eos_hazard_loss_weight > 0):
            secondary_token_selection = "multinomial" if primary_token_selection == "argmax" else "argmax"
            secondary_label = "stochastic" if secondary_token_selection == "multinomial" else "argmax"
            metrics["gen_secondary_token_selection"] = secondary_token_selection
            with autocast_ctx:
                secondary_metrics = evaluate_generation_metrics(
                    raw_model,
                    tokenizer,
                    noise,
                    build_sample_fn(secondary_token_selection),
                    int(config.train.device_batch_size),
                    config.train.run_dir,
                    device=device,
                    judge_model_id=config.eval.judge_model_id,
                    gen_samples=config.eval.gen_samples,
                    gen_len=config.eval.gen_len,
                    log_samples=config.eval.gen_log_samples,
                    score_mode=config.eval.gen_score_mode,
                    self_score_t=config.eval.gen_self_score_t,
                    step=step,
                    sample_label=secondary_label,
                    model_logits_fn=model_logits_fn,
                )
            for key, value in secondary_metrics.items():
                suffix = key[4:] if key.startswith("gen_") else key
                metrics[f"gen_{secondary_label}_{suffix}"] = value
            if secondary_token_selection == "argmax":
                metrics["gen_argmax_eos_rate"] = float(secondary_metrics.get("gen_eos_rate", 0.0))
            elif secondary_token_selection == "multinomial":
                metrics["gen_stochastic_eos_rate"] = float(secondary_metrics.get("gen_eos_rate", 0.0))

    if was_training:
        raw_model.train()
    return metrics


def _run_eval_with_runtime(runtime: EvaluationRuntime, step: int, include_gen: bool) -> dict:
    return run_isolated_evaluation(runtime, step, include_gen, _run_eval_unisolated)


def _run_eval(
    raw_model: nn.Module,
    config: ExperimentConfig,
    tokenizer,
    noise,
    autocast_ctx,
    device,
    step: int,
    include_gen: bool,
) -> dict:
    """Compatibility wrapper for repository-private callers using the legacy signature."""

    return _run_eval_with_runtime(
        EvaluationRuntime(
            raw_model=raw_model,
            config=config,
            tokenizer=tokenizer,
            noise=noise,
            autocast_ctx=autocast_ctx,
            device=device,
        ),
        step,
        include_gen,
    )


def _evaluate_fixed_noise_ce_trace(
    raw_model: nn.Module,
    config: ExperimentConfig,
    tokenizer,
    noise,
    autocast_ctx,
    device,
    step: int,
    total_steps: int,
    train_tokens: int,
    train_valid_tokens: int,
) -> dict[str, bool | float | int | list[float] | str]:
    """Evaluate fixed validation examples and corruption masks for CE traces."""
    saved_state = _rng_state(config)
    was_training = raw_model.training
    raw_model.eval()
    token_bytes = build_token_bytes(tokenizer).to(device)
    eos_token_id = tokenizer.get_eos_token_id()
    use_hierarchical_output = hasattr(raw_model, "uses_hierarchical_output") and raw_model.uses_hierarchical_output()
    t_values = (0.1, 0.3, 0.5, 0.7, 0.9)
    record: dict[str, bool | float | int | list[float] | str] = {
        "step": int(step),
        "checkpoint_step": int(step),
        "total_steps": int(total_steps),
        "train_tokens": int(train_tokens),
        "train_valid_tokens": int(train_valid_tokens),
        "fixed_ce_trace_checkpoint_step": int(step),
        "fixed_ce_trace_batches": int(config.eval.fixed_ce_batches),
        "fixed_ce_trace_corruption": BD3LM_SHARED_SCALAR_TIME_EVAL_LABEL
        if _uses_bd3lm(config)
        else _training_corruption_label(config),
        "fixed_ce_trace_eval_noise": BD3LM_SHARED_SCALAR_TIME_EVAL_LABEL
        if _uses_bd3lm(config)
        else "standard_scalar_t_mdlm"
        if _uses_per_token_noise_levels(config)
        else _training_corruption_label(config),
        "fixed_ce_trace_t_values": list(t_values),
    }
    seed_base = config.runtime.training_seed if config.runtime.training_seed >= 0 else config.runtime.seed
    try:
        for t_idx, t_val in enumerate(t_values):
            fixed_seed = int(seed_base) + 1_048_583 + t_idx * 8_191
            torch.manual_seed(fixed_seed)
            np.random.seed(fixed_seed % (2**32 - 1))
            random.seed(fixed_seed)
            if config.runtime.device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed(fixed_seed)
            val_loader = build_val_loader(config, tokenizer, device)
            ce_sum = 0.0
            content_ce_sum = 0.0
            active_count = 0
            content_count = 0
            lossable_count = 0
            for _ in range(int(config.eval.fixed_ce_batches)):
                batch = next(val_loader)
                x_eval = batch.x_0
                eval_attention_mask = _batch_model_attention_mask(batch)
                z_eval, w_eval, _, noise_level = _corrupt_eval_batch(
                    config,
                    noise,
                    x_eval,
                    batch.loss_mask,
                    t_val=t_val,
                )
                with autocast_ctx:
                    if use_hierarchical_output:
                        class_logits, token_logits = _forward_eval_hierarchical_components(
                            raw_model,
                            config,
                            z_eval,
                            x_eval,
                            eval_attention_mask,
                            noise_level=noise_level,
                            position_ids=getattr(batch, "position_ids", None),
                        )
                        ce, _class_ce_values, _within_class_ce_values = raw_model.hierarchical_ce(
                            class_logits,
                            token_logits,
                            x_eval,
                        )
                    else:
                        logits = _forward_eval_logits(
                            raw_model,
                            config,
                            z_eval,
                            x_eval,
                            eval_attention_mask,
                            noise_level=noise_level,
                            position_ids=getattr(batch, "position_ids", None),
                            target_lengths=_target_lengths_for_conditioning(
                                config,
                                x_eval,
                                batch.attention_mask,
                                eos_token_id,
                            ),
                        )
                        ce = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            x_eval.view(-1),
                            reduction="none",
                        ).view_as(x_eval)
                active = (w_eval > 0) & batch.loss_mask
                content = active & (token_bytes[x_eval] > 0)
                active_values = ce[active].detach().float()
                content_values = ce[content].detach().float()
                ce_sum += float(active_values.sum().item()) if active_values.numel() else 0.0
                content_ce_sum += float(content_values.sum().item()) if content_values.numel() else 0.0
                active_count += int(active.sum().item())
                content_count += int(content.sum().item())
                lossable_count += int(batch.loss_mask.sum().item())
            key = f"t{int(round(t_val * 10)):02d}"
            record[f"fixed_ce_trace_ce_{key}"] = ce_sum / max(1, active_count)
            record[f"fixed_ce_trace_content_ce_{key}"] = content_ce_sum / max(1, content_count)
            record[f"fixed_ce_trace_active_tokens_{key}"] = int(active_count)
            record[f"fixed_ce_trace_content_tokens_{key}"] = int(content_count)
            record[f"fixed_ce_trace_active_fraction_{key}"] = active_count / max(1, lossable_count)
            record[f"fixed_ce_trace_seed_{key}"] = int(fixed_seed)
    finally:
        _restore_rng_state(saved_state, config)
        if was_training:
            raw_model.train()
    return record
