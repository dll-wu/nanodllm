from __future__ import annotations

import copy
import gc
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .checkpoint_load_binding import load_checkpoint_with_optional_binding
from .config import ExperimentConfig, config_from_dict, write_config
from .data import build_frequency_token_classes, build_tokenizer
from .eval import (
    evaluate_argmax_basin_diagnostic,
    evaluate_conditional_generation_metrics,
    evaluate_generation_metrics,
)
from .model import DiffusionTransformer
from .noise import build_noise
from .parameterization import (
    bind_parameterization_contract,
    resolve_parameterization_contract,
    validate_checkpoint_eval_architecture_fingerprint,
    validate_checkpoint_parameterization_contract,
)
from .sampling import (
    sample_ar,
    sample_bd3lm,
    sample_fhs,
    sample_mask_uniform,
    sample_mdlm,
    sample_mdlm_from_prefixes,
    sample_mdlm_left_to_right,
    sample_mdlm_left_to_right_from_prefixes,
    sample_remdm,
    sample_uniform,
)
from .telemetry import write_json_atomic
from .training_artifacts import (
    _prepare_data_identity,
    autoresearch_config_summary,
    source_promotion_metrics,
)
from .training_checkpoint import (
    _load_json_if_exists,
    _resolve_comparison_sources,
    _resolve_eval_only_resume,
)
from .training_evaluation import EvaluationRuntime, _run_eval_with_runtime
from .training_features import (
    _effective_eval_sampler,
    _is_causal,
    _uses_bd3lm,
    _uses_hierarchical_output,
    _uses_target_length_conditioning,
)
from .training_metrics import (
    _embedding_head_init_metrics,
    _residual_projection_init_metrics,
    _token_ratio_metrics,
)
from .training_objective import (
    _bd3lm_telemetry,
    _forward_eval_logits,
    _hierarchical_output_telemetry,
    _noise_telemetry,
    _target_lengths_for_conditioning,
    _training_corruption_label,
)
from .training_plan import (
    _parameter_accounting_summary,
    _resolve_diffusion_config,
    _selected_target_param_count,
    _training_budget,
)
from .training_rng import _restore_rng_state, _rng_state
from .training_runtime import (
    _check_gpu_preflight,
    _configure_numeric_runtime,
    _cpu_threading_telemetry,
    dtype_from_name,
    init_distributed,
    print0,
)


def _metric_label(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_")


def _load_source_train_config_dict(source_info: dict[str, str | list[str] | dict]) -> dict:
    source_run_dir = Path(str(source_info.get("source_run_dir", "")))
    resolved_config_path = source_run_dir / "resolved_config.json"
    raw_config = _load_json_if_exists(resolved_config_path)
    if raw_config:
        return raw_config

    checkpoint_path = str(source_info.get("source_checkpoint_path", source_info.get("source_checkpoint", "")))
    if not checkpoint_path:
        raise FileNotFoundError("comparison source has no checkpoint path")
    checkpoint = load_checkpoint_with_optional_binding(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    raw_config = checkpoint.get("train_config", {})
    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError(f"checkpoint {checkpoint_path} does not contain a train_config for replay")
    return raw_config


def _source_replay_config(
    active_config: ExperimentConfig,
    source_info: dict[str, str | list[str] | dict],
) -> tuple[ExperimentConfig, list[str]]:
    raw_config = copy.deepcopy(_load_source_train_config_dict(source_info))
    normalizations: list[str] = []
    model_section = raw_config.setdefault("model", {})
    if isinstance(model_section, dict) and model_section.get("output_factorization") == "eos_hazard":
        model_section["output_factorization"] = "flat"
        normalizations.append("legacy_eos_hazard_output_factorization_to_flat")

    source_config = config_from_dict(raw_config)
    source_run_id = str(source_info.get("source_run_id", ""))
    source_config.runtime = copy.deepcopy(active_config.runtime)
    source_config.eval = copy.deepcopy(active_config.eval)
    source_config.train.run_dir = active_config.train.run_dir
    source_config.train.eval_only = True
    source_config.train.resume_from = "comparison"
    source_config.train.comparison_run_ids = [source_run_id] if source_run_id else []
    source_config.train.device_batch_size = int(active_config.train.device_batch_size)
    source_config.train.tensorboard = False
    source_config.train.tensorboard_dir = ""
    return source_config, normalizations


def _load_model_checkpoint_for_eval(
    path: str,
    raw_model: nn.Module,
    device,
    *,
    expected_parameterization_contract_sha256: str,
    restore_rng_config: ExperimentConfig | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_size_bytes: int | None = None,
    checkpoint_load_receipt: dict[str, object] | None = None,
) -> tuple[int, int, int, dict, list[str], str]:
    # Campaign checkpoints use PyTorch's zip serialization. Memory mapping lets
    # eval-only workers fault in the model tensors they copy to the model while
    # leaving optimizer tensor pages untouched.
    checkpoint = load_checkpoint_with_optional_binding(
        path,
        map_location="cpu",
        weights_only=False,
        mmap_unbound=True,
        expected_sha256=expected_checkpoint_sha256,
        expected_size_bytes=expected_checkpoint_size_bytes,
        receipt=checkpoint_load_receipt,
    )
    state_dict = checkpoint["model_state_dict"]
    ignored_legacy_keys: list[str] = []
    model_keys = set(raw_model.state_dict().keys())
    legacy_eval_only_keys = ("stop_head.weight", "stop_head.bias")
    removable_keys = [key for key in legacy_eval_only_keys if key in state_dict and key not in model_keys]
    if removable_keys:
        state_dict = dict(state_dict)
        for key in removable_keys:
            state_dict.pop(key)
        ignored_legacy_keys = removable_keys
    checkpoint_parameterization = validate_checkpoint_parameterization_contract(checkpoint, raw_model)
    validate_checkpoint_eval_architecture_fingerprint(checkpoint, raw_model)
    observed_parameterization_sha256 = (
        str(checkpoint_parameterization["contract_sha256"]) if checkpoint_parameterization is not None else ""
    )
    if observed_parameterization_sha256 != expected_parameterization_contract_sha256:
        raise ValueError(
            "checkpoint parameterization contract does not match the active eval contract: "
            f"checkpoint={observed_parameterization_sha256 or '<unbound>'}, "
            f"active={expected_parameterization_contract_sha256 or '<unbound>'}"
        )
    raw_model.load_state_dict(state_dict)
    if restore_rng_config is not None:
        _restore_rng_state(checkpoint.get("rng_state", {}), restore_rng_config)
    checkpoint_metrics = dict(checkpoint.get("metrics", {})) if isinstance(checkpoint.get("metrics", {}), dict) else {}
    checkpoint_step = int(checkpoint.get("step", 0))
    checkpoint_train_tokens = int(checkpoint.get("train_tokens", 0))
    checkpoint_train_valid_tokens = int(
        checkpoint.get("train_valid_tokens", checkpoint_metrics.get("train_valid_tokens", 0))
    )
    del state_dict
    del checkpoint
    return (
        checkpoint_step,
        checkpoint_train_tokens,
        checkpoint_train_valid_tokens,
        checkpoint_metrics,
        ignored_legacy_keys,
        observed_parameterization_sha256,
    )


def _selected_basin_aggregate_key(key: str) -> bool:
    if not key.startswith("basin_depth_"):
        return False
    suffixes = (
        "_samples",
        "_eos_rate",
        "_exact_duplicate_rate",
        "_repeat_4gram_rate",
        "_distinct_1",
        "_distinct_2",
        "_distinct_3",
        "_max_token_frac_mean",
        "_max_token_fraction",
        "_token_entropy_mean",
        "_argmax_minus_nucleus_eos_rate",
        "_argmax_minus_nucleus_exact_duplicate_rate",
        "_argmax_minus_nucleus_repeat_4gram_rate",
        "_argmax_minus_nucleus_distinct_1",
        "_argmax_minus_nucleus_distinct_2",
        "_argmax_minus_nucleus_distinct_3",
        "_argmax_minus_nucleus_max_token_frac_mean",
        "_argmax_minus_nucleus_token_entropy_mean",
    )
    return any(key.endswith(suffix) for suffix in suffixes)


def eval_only_basin_replay(config: ExperimentConfig) -> dict:
    """Eval-only source replay for prefix-conditioned argmax-attractor diagnostics."""
    os.makedirs(config.train.run_dir, exist_ok=True)
    _configure_numeric_runtime(config)
    is_ddp, rank, _local_rank, world_size, device = init_distributed(
        config.runtime.device_type,
        config.train.run_dir,
        init_single_rank_process_group=config.runtime.init_single_rank_process_group,
    )
    _check_gpu_preflight(config, rank)
    cpu_threading = _cpu_threading_telemetry()
    compute_dtype = dtype_from_name(config.runtime.dtype)
    autocast_ctx = (
        torch.autocast(device_type=config.runtime.device_type, dtype=compute_dtype)
        if config.runtime.device_type == "cuda" and compute_dtype != torch.float32
        else nullcontext()
    )

    training_seed = config.runtime.seed if config.runtime.training_seed < 0 else config.runtime.training_seed
    torch.manual_seed(training_seed + rank)
    np.random.seed(training_seed + rank)
    random.seed(training_seed + rank)
    if config.runtime.device_type == "cuda":
        torch.cuda.manual_seed(training_seed + rank)

    if rank == 0:
        write_config(os.path.join(config.train.run_dir, "resolved_config.json"), config)

    sources = _resolve_comparison_sources(config)
    if not sources:
        raise ValueError("basin diagnostic requires train.comparison_run_ids")

    metrics: dict[str, object] = {
        **source_promotion_metrics(),
        "scale": "eval-only",
        "experiment_type": "eval-only-basin-diagnostic",
        "eval_only": True,
        "train_eval_only": bool(config.train.eval_only),
        "replay_metric_scope": "checkpoint-replay-argmax-basin-diagnostic",
        **autoresearch_config_summary(config, os.environ.get("NANODLLM_ENTRYPOINT_MODE", "eval-only")),
        "source_run_ids": [str(source.get("source_run_id", "")) for source in sources],
        "comparison_source_count": int(len(sources)),
        "basin_diagnostic": True,
        "basin_forced_argmax_depths": [int(d) for d in config.eval.basin_forced_argmax_depths],
        "basin_nucleus_top_p": float(config.eval.basin_nucleus_top_p),
        "basin_stochastic_policy": "nucleus_multinomial",
        "basin_prefix_len": int(config.eval.basin_prefix_len),
        "basin_gen_samples": int(config.eval.gen_samples),
        "basin_gen_len": int(config.eval.gen_len),
        "training_seconds": 0.0,
        "world_size": world_size,
        "device_batch_size": int(config.train.device_batch_size),
        "cpu_threading": cpu_threading,
    }
    if metrics.get("status") != "ok":
        if rank == 0:
            write_json_atomic(os.path.join(config.train.run_dir, "metrics.json"), metrics)
        if is_ddp:
            dist.barrier()
            dist.destroy_process_group()
        return metrics

    aggregate_values: dict[str, list[float]] = {}
    source_val_nelbo_values: list[float] = []
    source_val_ce_values: list[float] = []

    for source_idx, source_info in enumerate(sources):
        source_run_id = str(source_info.get("source_run_id", ""))
        source_label = f"source_{source_idx:02d}_{_metric_label(source_run_id)}"
        source_config, normalizations = _source_replay_config(config, source_info)
        tokenizer = build_tokenizer(source_config)
        token_class_ids = None
        token_class_metadata: dict | None = None
        if _uses_hierarchical_output(source_config):
            saved_state = _rng_state(source_config)
            token_class_ids, token_class_metadata = build_frequency_token_classes(
                source_config.data,
                tokenizer,
                int(source_config.train.device_batch_size),
                int(source_config.model.max_seq_len),
                device,
                int(source_config.model.hierarchical_frequency_batches),
            )
            _restore_rng_state(saved_state, source_config)

        model_config = _resolve_diffusion_config(source_config, tokenizer)
        raw_model = DiffusionTransformer(
            model_config,
            rope_base=source_config.model.rope_base,
            compute_dtype=compute_dtype,
            pad_vocab_to=source_config.model.pad_vocab_to,
            token_class_ids=token_class_ids,
        ).to(device)
        source_parameterization_sha256 = str(source_config.parameterization.contract_sha256)
        checkpoint_path = str(source_info.get("source_checkpoint_path", source_info.get("source_checkpoint", "")))
        (
            checkpoint_step,
            checkpoint_train_tokens,
            checkpoint_train_valid_tokens,
            checkpoint_metrics,
            ignored_checkpoint_keys,
            checkpoint_parameterization_sha256,
        ) = _load_model_checkpoint_for_eval(
            checkpoint_path,
            raw_model,
            device,
            expected_parameterization_contract_sha256=source_parameterization_sha256,
        )
        raw_model.eval()
        noise = build_noise(source_config.noise, model_config.mask_token_id, model_config.vocab_size)
        source_eos_token_id = int(tokenizer.get_eos_token_id())

        def model_logits_fn(
            m,
            z_eval,
            x_eval,
            attention_mask,
            noise_level=None,
            position_ids=None,
            target_lengths=None,
            source_config=source_config,
            eos_token_id=source_eos_token_id,
        ):
            if target_lengths is None:
                target_lengths = _target_lengths_for_conditioning(
                    source_config,
                    x_eval,
                    attention_mask,
                    eos_token_id,
                )
            return _forward_eval_logits(
                m,
                source_config,
                z_eval,
                x_eval,
                attention_mask,
                noise_level=noise_level,
                position_ids=position_ids,
                target_lengths=target_lengths,
            )

        source_metrics = source_info.get("source_metrics", {})
        if not isinstance(source_metrics, dict):
            source_metrics = {}
        source_val_nelbo = source_metrics.get("val_nelbo_bpb", checkpoint_metrics.get("val_nelbo_bpb"))
        source_val_ce = source_metrics.get("val_ce", checkpoint_metrics.get("val_ce"))
        if isinstance(source_val_nelbo, (int, float)):
            source_val_nelbo_values.append(float(source_val_nelbo))
        if isinstance(source_val_ce, (int, float)):
            source_val_ce_values.append(float(source_val_ce))

        with autocast_ctx:
            diag_start = time.time()
            basin_metrics = evaluate_argmax_basin_diagnostic(
                raw_model,
                tokenizer,
                source_config.data,
                int(source_config.train.device_batch_size),
                device=device,
                seq_len=int(source_config.model.max_seq_len),
                gen_samples=int(config.eval.gen_samples),
                gen_len=int(config.eval.gen_len),
                prefix_len=int(config.eval.basin_prefix_len),
                forced_argmax_depths=config.eval.basin_forced_argmax_depths,
                temperature=float(config.eval.gen_temperature),
                nucleus_top_p=float(config.eval.basin_nucleus_top_p),
                is_causal=_is_causal(source_config),
                run_dir=config.train.run_dir if rank == 0 else None,
                log_samples=int(config.eval.gen_log_samples),
                source_run_id=source_run_id,
                noise=noise,
                model_logits_fn=model_logits_fn,
            )
            basin_seconds = time.time() - diag_start

        source_prefix = f"{source_label}_"
        source_metadata: dict[str, object] = {
            f"{source_prefix}run_id": source_run_id,
            f"{source_prefix}checkpoint": checkpoint_path,
            f"{source_prefix}artifacts_manifest": str(source_info.get("source_artifacts_manifest", "")),
            f"{source_prefix}metrics_path": str(source_info.get("source_metrics_path", "")),
            f"{source_prefix}checkpoint_step": int(checkpoint_step),
            f"{source_prefix}checkpoint_train_tokens": int(checkpoint_train_tokens),
            f"{source_prefix}checkpoint_train_valid_tokens": int(checkpoint_train_valid_tokens),
            f"{source_prefix}source_val_nelbo_bpb": float(source_val_nelbo)
            if isinstance(source_val_nelbo, (int, float))
            else 0.0,
            f"{source_prefix}source_val_ce": float(source_val_ce) if isinstance(source_val_ce, (int, float)) else 0.0,
            f"{source_prefix}source_training_seconds": float(source_metrics.get("training_seconds", 0.0))
            if isinstance(source_metrics.get("training_seconds", 0.0), (int, float))
            else 0.0,
            f"{source_prefix}training_seconds": 0.0,
            f"{source_prefix}diagnostic_seconds": float(basin_seconds),
            f"{source_prefix}model_depth": int(source_config.model.depth),
            f"{source_prefix}model_n_embd": int(source_config.model.n_embd),
            f"{source_prefix}model_n_head": int(source_config.model.n_head),
            f"{source_prefix}model_causal": bool(source_config.model.causal),
            f"{source_prefix}training_corruption": _training_corruption_label(source_config),
            f"{source_prefix}noise_type": source_config.noise.type,
            f"{source_prefix}noise_time_conditioning": bool(source_config.noise.time_conditioning),
            f"{source_prefix}noise_complement_masking": bool(source_config.noise.complement_masking),
            f"{source_prefix}tokenizer_path": source_config.data.tokenizer_path
            if source_config.data.tokenizer == "hf"
            else "",
            f"{source_prefix}config_normalizations": normalizations,
            f"{source_prefix}ignored_checkpoint_keys": ignored_checkpoint_keys,
            f"{source_prefix}active_parameterization_contract_sha256": source_parameterization_sha256 or None,
            f"{source_prefix}checkpoint_parameterization_contract_sha256": (checkpoint_parameterization_sha256 or None),
            f"{source_prefix}output_factorization": source_config.model.output_factorization,
            f"{source_prefix}output_class_count": int(
                token_class_metadata.get("output_class_count", 0) if token_class_metadata else 0
            ),
        }
        metrics.update(source_metadata)
        for key, value in basin_metrics.items():
            metrics[f"{source_prefix}{key}"] = value
            if isinstance(value, (int, float)) and _selected_basin_aggregate_key(key):
                aggregate_values.setdefault(key, []).append(float(value))

        del raw_model, noise, tokenizer
        gc.collect()
        if config.runtime.device_type == "cuda":
            torch.cuda.empty_cache()

    for key, values in aggregate_values.items():
        metrics[f"source_mean_{key}"] = sum(values) / len(values) if values else 0.0
    if source_val_nelbo_values:
        metrics["source_val_nelbo_bpb_mean"] = sum(source_val_nelbo_values) / len(source_val_nelbo_values)
        metrics["val_nelbo_bpb"] = metrics["source_val_nelbo_bpb_mean"]
        metrics["val_nelbo_bpb_is_source_copy"] = True
    else:
        metrics["val_nelbo_bpb"] = 0.0
        metrics["val_nelbo_bpb_is_source_copy"] = False
    if source_val_ce_values:
        metrics["source_val_ce_mean"] = sum(source_val_ce_values) / len(source_val_ce_values)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if config.runtime.device_type == "cuda" else 0.0
    metrics["peak_vram_mb"] = peak_vram_mb

    if rank == 0:
        write_json_atomic(os.path.join(config.train.run_dir, "metrics.json"), metrics)
        print0(rank, f"metrics saved to {os.path.join(config.train.run_dir, 'metrics.json')}")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()

    return metrics


def _merge_eval_only_replay_nelbo_metrics(
    metrics: dict,
    replay_metrics: dict,
    source_val_nelbo_bpb: float,
) -> None:
    """Keep promotion-facing source metrics separate from newly replayed NELBO semantics."""

    metrics.update(replay_metrics)
    replay_metric_keys = [
        key
        for key in replay_metrics
        if key.startswith("val_nelbo") or key in {"val_masked_ce_bpb", "hierarchical_val_nelbo_bpb"}
    ]
    for key in replay_metric_keys:
        metrics.pop(key, None)
        metrics[f"eval_only_replay_{key}"] = replay_metrics[key]
    metrics["source_val_nelbo_bpb"] = float(source_val_nelbo_bpb)
    metrics["val_nelbo_bpb"] = float(source_val_nelbo_bpb)
    metrics["val_nelbo_metric_origin"] = "source_checkpoint_metrics"


def eval_only_replay(config: ExperimentConfig) -> dict:
    """Eval-only checkpoint replay: load a checkpoint and run the configured eval bundle."""
    if config.eval.basin_diagnostic:
        return eval_only_basin_replay(config)

    os.makedirs(config.train.run_dir, exist_ok=True)
    _configure_numeric_runtime(config)
    is_ddp, rank, local_rank, world_size, device = init_distributed(
        config.runtime.device_type,
        config.train.run_dir,
        init_single_rank_process_group=config.runtime.init_single_rank_process_group,
    )
    _check_gpu_preflight(config, rank)
    cpu_threading = _cpu_threading_telemetry()
    compute_dtype = dtype_from_name(config.runtime.dtype)
    autocast_ctx = (
        torch.autocast(device_type=config.runtime.device_type, dtype=compute_dtype)
        if config.runtime.device_type == "cuda" and compute_dtype != torch.float32
        else nullcontext()
    )

    training_seed = config.runtime.seed if config.runtime.training_seed < 0 else config.runtime.training_seed
    torch.manual_seed(training_seed + rank)
    np.random.seed(training_seed + rank)
    random.seed(training_seed + rank)
    if config.runtime.device_type == "cuda":
        torch.cuda.manual_seed(training_seed + rank)

    tokenizer = build_tokenizer(config)
    data_identity_summary = _prepare_data_identity(
        config,
        rank=rank,
        world_size=world_size,
        is_ddp=is_ddp,
    )
    token_class_ids = None
    token_class_metadata: dict | None = None
    if _uses_hierarchical_output(config):
        saved_state = _rng_state(config)
        token_class_ids, token_class_metadata = build_frequency_token_classes(
            config.data,
            tokenizer,
            int(config.train.device_batch_size),
            int(config.model.max_seq_len),
            device,
            int(config.model.hierarchical_frequency_batches),
        )
        _restore_rng_state(saved_state, config)
    model_config = _resolve_diffusion_config(config, tokenizer)

    saved_state = _rng_state(config)
    model_seed = config.runtime.seed if config.runtime.model_seed < 0 else config.runtime.model_seed
    torch.manual_seed(model_seed)
    if config.runtime.device_type == "cuda":
        torch.cuda.manual_seed(model_seed)
    raw_model = DiffusionTransformer(
        model_config,
        rope_base=config.model.rope_base,
        compute_dtype=compute_dtype,
        pad_vocab_to=config.model.pad_vocab_to,
        token_class_ids=token_class_ids,
    ).to(device)
    _restore_rng_state(saved_state, config)

    counts = raw_model.parameter_counts()
    flops_per_token = raw_model.estimate_flops()
    parameter_accounting = _parameter_accounting_summary(config, counts, raw_model.parameter_inventory())
    total_batch_size, grad_accum_steps, total_steps, batch_lr_scale = _training_budget(
        config, counts, world_size, raw_model=raw_model
    )
    parameterization_contract = resolve_parameterization_contract(
        raw_model,
        config,
        batch_lr_scale=batch_lr_scale,
    )
    bind_parameterization_contract(config, parameterization_contract)

    print0(
        rank,
        json.dumps(
            {
                "model": asdict(model_config),
                "param_counts": counts,
                "parameter_accounting": parameter_accounting,
                "parameterization_contract_sha256": parameterization_contract["contract_sha256"],
                "flops_per_token": flops_per_token,
                "total_batch_size": total_batch_size,
                "effective_batch_size_tokens": total_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "total_steps": total_steps,
                "target_tokens": config.train.target_tokens,
                "world_size": world_size,
                "compute_dtype": str(compute_dtype),
                "eval_only": True,
            },
            indent=2,
        ),
    )
    if rank == 0:
        write_config(os.path.join(config.train.run_dir, "resolved_config.json"), config)
    resume_metrics: dict = {}
    resume_path, source_info = _resolve_eval_only_resume(config)
    if resume_path is None:
        raise ValueError(
            "eval_only_replay requires train.resume_from or train.comparison_run_ids to point to a checkpoint"
        )
    checkpoint_load_receipt: dict[str, object] = {}
    checkpoint_identity_seconds = 0.0
    checkpoint_load_start = time.time()
    (
        checkpoint_step,
        checkpoint_train_tokens,
        checkpoint_train_valid_tokens,
        resume_metrics,
        ignored_checkpoint_keys,
        checkpoint_parameterization_sha256,
    ) = _load_model_checkpoint_for_eval(
        resume_path,
        raw_model,
        device,
        expected_parameterization_contract_sha256=config.parameterization.contract_sha256,
        restore_rng_config=config,
        expected_checkpoint_sha256=(
            config.eval.source_checkpoint_expected_sha256
            if config.eval.source_checkpoint_identity_mode == "sha256"
            else None
        ),
        expected_checkpoint_size_bytes=(
            config.eval.source_checkpoint_expected_size_bytes
            if config.eval.source_checkpoint_identity_mode == "sha256"
            else None
        ),
        checkpoint_load_receipt=checkpoint_load_receipt,
    )
    checkpoint_load_seconds = time.time() - checkpoint_load_start
    if checkpoint_load_receipt:
        checkpoint_identity_seconds = checkpoint_load_seconds
    checkpoint_load_peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if config.runtime.device_type == "cuda" else 0.0
    )
    if config.runtime.device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print0(rank, f"eval_only_replay loaded checkpoint {resume_path}")

    noise = build_noise(config.noise, model_config.mask_token_id, model_config.vocab_size)
    evaluation_runtime = EvaluationRuntime(
        raw_model=raw_model,
        config=config,
        tokenizer=tokenizer,
        noise=noise,
        autocast_ctx=autocast_ctx,
        device=device,
    )

    # Bind artifact names and panel records to the loaded state, not the eval-only process's zero-update cursor.
    base_metrics: dict[str, float | int | str] = {}
    if config.eval.enabled:
        eval_start = time.time()
        base_metrics = _run_eval_with_runtime(
            evaluation_runtime,
            step=checkpoint_step,
            include_gen=False,
        )
        base_metrics["evaluation_seconds"] = time.time() - eval_start

    source_metrics = source_info.get("source_metrics", {})
    if not isinstance(source_metrics, dict):
        source_metrics = {}
    source_val_nelbo_bpb = float(
        source_metrics.get("val_nelbo_bpb", resume_metrics.get("val_nelbo_bpb", base_metrics.get("val_nelbo_bpb", 0.0)))
    )
    source_val_ce = source_metrics.get("val_ce", resume_metrics.get("val_ce"))
    replay_val_ce = base_metrics.get("val_ce")
    source_train_tokens = int(
        source_metrics.get("train_tokens", source_metrics.get("total_tokens", checkpoint_train_tokens))
    )

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

    if config.eval.conditional_gen_eval and not any(k.startswith("conditional_gen_p") for k in base_metrics):
        if config.noise.type != "mdlm":
            raise ValueError("eval.conditional_gen_eval currently supports noise.type='mdlm'")

        def conditional_sample_fn(m, prefixes, continuation_len):
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
                    m,
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
                m,
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

        with autocast_ctx:
            conditional_start = time.time()
            conditional_metrics = evaluate_conditional_generation_metrics(
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
                step=0,
                sample_label=config.eval.sample_token_selection,
                model_logits_fn=model_logits_fn,
            )
        conditional_metrics["conditional_gen_seconds"] = time.time() - conditional_start
        base_metrics.update(conditional_metrics)

    if config.eval.gen_eval:
        temperatures = [0.5, 0.8, 1.0] if config.eval.eval_only_temperature_sweep else [config.eval.gen_temperature]

        def _make_sample_fn(sampler_name: str, fixed_temp: float, tau_edit: float | None = None):
            def sample_fn(m, n, length, temperature=fixed_temp):
                if _is_causal(config):
                    return sample_ar(
                        m,
                        tokenizer,
                        n,
                        length,
                        temperature=temperature,
                        token_selection=config.eval.sample_token_selection,
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
                        token_selection=config.eval.sample_token_selection,
                        temperature=temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        return_trace=config.eval.sample_trace,
                    )
                if sampler_name == "left_to_right":
                    return sample_mdlm_left_to_right(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        token_selection=config.eval.sample_token_selection,
                        temperature=temperature,
                        top_k=config.eval.sample_top_k,
                        top_p=config.eval.sample_top_p,
                        return_trace=config.eval.sample_trace,
                    )
                if sampler_name == "remdm":
                    return sample_remdm(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=config.eval.sample_token_selection,
                        temperature=temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        tau_edit=tau_edit if tau_edit is not None else config.eval.remdm_tau_edit,
                        return_trace=config.eval.sample_trace,
                    )
                if sampler_name == "fhs":
                    return sample_fhs(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=config.eval.sample_token_selection,
                        temperature=temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        tau=config.eval.fhs_tau,
                        length_policy=config.eval.length_policy,
                        return_trace=config.eval.sample_trace,
                    )
                sample_impl = {
                    "mdlm": sample_mdlm,
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
                    token_selection=config.eval.sample_token_selection,
                    temperature=temperature,
                    jitter_scale=config.eval.sample_jitter_scale,
                    return_trace=config.eval.sample_trace,
                )

            return sample_fn

        def _run_generation_pass(sampler_name: str, label_prefix: str, tau_edit: float | None = None):
            for temp in temperatures:
                sample_fn = _make_sample_fn(sampler_name, fixed_temp=temp, tau_edit=tau_edit)
                with autocast_ctx:
                    gen_start = time.time()
                    gen_metrics = evaluate_generation_metrics(
                        raw_model,
                        tokenizer,
                        noise,
                        sample_fn,
                        int(config.train.device_batch_size),
                        config.train.run_dir,
                        device=device,
                        judge_model_id=config.eval.judge_model_id,
                        gen_samples=config.eval.gen_samples,
                        gen_len=config.eval.gen_len,
                        log_samples=config.eval.gen_log_samples,
                        score_mode=config.eval.gen_score_mode,
                        self_score_t=config.eval.gen_self_score_t,
                        step=0,
                        sample_label=f"{label_prefix}temp{int(temp * 10):02d}",
                        model_logits_fn=model_logits_fn,
                    )
                    gen_seconds = time.time() - gen_start

                temp_key = f"{label_prefix}temp{int(temp * 10):02d}"
                if temp == 1.0 and label_prefix == "":
                    base_metrics.update(gen_metrics)
                elif temp == 1.0:
                    for k, v in gen_metrics.items():
                        if isinstance(v, (int, float)):
                            base_metrics[f"gen_{label_prefix}{k[4:] if k.startswith('gen_') else k}"] = v
                for k, v in gen_metrics.items():
                    if isinstance(v, (int, float)):
                        base_metrics[f"gen_{temp_key}_{k[4:] if k.startswith('gen_') else k}"] = v
                base_metrics[f"gen_{temp_key}_seconds"] = gen_seconds

        _run_generation_pass(config.eval.sampler, "")
        if config.eval.remasking_comparison:
            _run_generation_pass("remdm", "remask_", tau_edit=config.eval.remdm_tau_edit)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if config.runtime.device_type == "cuda" else 0.0

    metrics = {
        **source_promotion_metrics(),
        "scale": "eval-only",
        "experiment_type": "eval-only",
        "eval_only": True,
        "train_eval_only": bool(config.train.eval_only),
        "replay_metric_scope": "checkpoint-replay-diagnostic",
        **autoresearch_config_summary(config, os.environ.get("NANODLLM_ENTRYPOINT_MODE", "eval-only")),
        "source_run_ids": source_info.get("source_run_ids", config.train.comparison_run_ids),
        "source_run_id": str((source_info.get("source_run_ids", config.train.comparison_run_ids) or [""])[0]),
        "source_checkpoint": source_info.get("source_checkpoint", resume_path),
        "source_checkpoint_path": source_info.get(
            "source_checkpoint_path", source_info.get("source_checkpoint", resume_path)
        ),
        "source_checkpoint_identity_mode": config.eval.source_checkpoint_identity_mode,
        "source_checkpoint_sha256": checkpoint_load_receipt.get("source_checkpoint_sha256"),
        "source_checkpoint_size_bytes": checkpoint_load_receipt.get("source_checkpoint_size_bytes"),
        "source_checkpoint_identity_seconds": float(checkpoint_identity_seconds),
        "eval_only_checkpoint_load_mode": (
            "exact_private_descriptor" if checkpoint_load_receipt else "model_state_only_cpu_mmap"
        ),
        "eval_only_checkpoint_load_seconds": float(checkpoint_load_seconds),
        "eval_only_checkpoint_load_peak_vram_mb": float(checkpoint_load_peak_vram_mb),
        "eval_only_checkpoint_rng_restored": True,
        "eval_only_ignored_checkpoint_keys": ignored_checkpoint_keys,
        "active_parameterization_contract_sha256": config.parameterization.contract_sha256 or None,
        "source_parameterization_contract_sha256": checkpoint_parameterization_sha256 or None,
        **checkpoint_load_receipt,
        "source_artifacts_manifest": source_info.get("source_artifacts_manifest", ""),
        "source_metrics_path": source_info.get("source_metrics_path", ""),
        "source_val_nelbo_bpb": source_val_nelbo_bpb,
        "source_val_ce": float(source_val_ce) if isinstance(source_val_ce, (int, float)) else 0.0,
        "val_nelbo_bpb": source_val_nelbo_bpb,
        "val_nelbo_metric_origin": "source_checkpoint_metrics",
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_train_tokens": int(checkpoint_train_tokens),
        "checkpoint_train_valid_tokens": int(checkpoint_train_valid_tokens),
        "source_train_tokens": source_train_tokens,
        "replay_train_tokens_start": int(checkpoint_train_tokens),
        "replay_train_tokens_end": int(checkpoint_train_tokens),
        "replay_train_tokens_delta": 0,
        "sampler": config.eval.sampler,
        "eval_sampler_requested": config.eval.sampler,
        "eval_sampler_effective": _effective_eval_sampler(config),
        "sample_steps": config.eval.sample_steps,
        "sample_scoring_mode": config.eval.sample_scoring_mode,
        "sample_token_selection": config.eval.sample_token_selection,
        "sample_top_k": config.eval.sample_top_k,
        "sample_top_p": config.eval.sample_top_p,
        "sample_jitter_scale": config.eval.sample_jitter_scale,
        "remdm_tau_edit": config.eval.remdm_tau_edit,
        "remasking_comparison": config.eval.remasking_comparison,
        "model_depth": config.model.depth,
        "model_n_embd": config.model.n_embd,
        "model_n_head": config.model.n_head,
        **_noise_telemetry(config),
        **_bd3lm_telemetry(config),
        **_hierarchical_output_telemetry(token_class_metadata),
        **_residual_projection_init_metrics(config),
        **_embedding_head_init_metrics(config),
        "training_corruption": _training_corruption_label(config),
        "tokenizer": config.data.tokenizer,
        "data_identity_enabled": bool(data_identity_summary["enabled"]),
        "data_identity_mode": str(data_identity_summary["mode"]),
        "data_identity_artifact": str(data_identity_summary["artifact"]),
        "data_identity_sha256": str(data_identity_summary["identity_sha256"]),
        "data_identity_sha256_strength": str(data_identity_summary["identity_sha256_strength"]),
        "target_tokens": config.train.target_tokens,
        "target_param_count_basis": config.train.target_param_count_basis,
        "target_param_count": _selected_target_param_count(config, counts),
        "tokens_per_target_param": checkpoint_train_tokens / max(1, _selected_target_param_count(config, counts)),
        "parameter_accounting": parameter_accounting,
        "vocab_size": model_config.vocab_size,
        "mask_token_id": model_config.mask_token_id,
        "eos_token_id": tokenizer.get_eos_token_id(),
        "num_params_M": counts["total"] / 1e6,
        "num_non_embedding_params_M": counts["non_embedding"] / 1e6,
        "num_scaling_params_M": counts["scaling"] / 1e6,
        "num_wte_params_M": counts["wte"] / 1e6,
        "num_lm_head_params_M": counts["lm_head"] / 1e6,
        "num_class_head_params_M": counts["class_head"] / 1e6,
        "num_length_head_params_M": counts["length_head"] / 1e6,
        "num_auxiliary_head_params_M": counts["auxiliary_heads"] / 1e6,
        "num_transformer_params_M": counts["transformer"] / 1e6,
        **_token_ratio_metrics(checkpoint_train_tokens, counts),
        "flops_per_token": flops_per_token,
        "training_seconds": 0.0,
        "world_size": world_size,
        "total_batch_size": total_batch_size,
        "device_batch_size": config.train.device_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "peak_vram_mb": peak_vram_mb,
        "cpu_threading": cpu_threading,
    }
    _merge_eval_only_replay_nelbo_metrics(metrics, base_metrics, source_val_nelbo_bpb)
    if isinstance(replay_val_ce, int | float):
        metrics["eval_only_replay_val_ce"] = float(replay_val_ce)

    if rank == 0:
        write_json_atomic(os.path.join(config.train.run_dir, "metrics.json"), metrics)
        print0(rank, f"metrics saved to {os.path.join(config.train.run_dir, 'metrics.json')}")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()

    return metrics
