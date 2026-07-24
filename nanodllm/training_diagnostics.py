"""The opt-in analytic Bayes audit for training data and corruption.

The trainer imports this module lazily so normal training does not load the
audit implementation or create a top-level import cycle.
"""

import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .config import ExperimentConfig, write_config
from .data import (
    EOS_OVERFIT_CANARY_DOCUMENTS,
    DLMDataBatch,
    build_synthetic_eos_overfit_canary_batch,
    encode_document_batch,
)
from .model import DiffusionConfig
from .noise import build_noise
from .telemetry import append_jsonl, write_json_atomic
from .training import (
    _comparison_manifest_path,
    _fast_dllm_loss_normalization_scale,
    _find_manifest_artifact_path,
    _is_causal,
    _load_json_if_exists,
    _noise_telemetry,
    _per_target_t_values,
    _register_run_artifacts,
    _restore_rng_state,
    _rng_state,
    _run_artifact_manifest_relative_path,
    _tokenizer_summary,
    _training_corruption_label,
    _uses_bd3lm,
    _uses_block_causal_pilot,
    _uses_per_token_noise_levels,
    _uses_strict_mdlm_per_row_t,
    _uses_token_edit_corruption,
    run_config_summary,
    source_provenance_metrics,
)


def run_configured_diagnostic(
    config: ExperimentConfig,
    tokenizer,
    *,
    rank: int,
    world_size: int,
    device,
    is_ddp: bool,
) -> dict | None:
    """Run the public analytic audit, or return ``None`` for training."""

    if not _strict_mdlm_all_token_bayes_audit_enabled(config):
        return None
    if rank == 0:
        write_config(Path(config.train.run_dir) / "resolved_config.json", config)
    noise = build_noise(
        config.noise,
        int(tokenizer.get_mask_token_id()),
        int(tokenizer.get_vocab_size()),
    )
    return _run_strict_mdlm_all_token_bayes_audit_only(
        config,
        tokenizer,
        noise,
        rank,
        world_size,
        device,
        is_ddp,
        time.time(),
    )


def _strict_mdlm_static_invariant_failures(config: ExperimentConfig) -> list[str]:
    failures: list[str] = []
    if _is_causal(config):
        failures.append("model.causal must be false")
    if config.noise.type != "mdlm":
        failures.append("noise.type must be mdlm")
    if not bool(config.noise.schedule_weight):
        failures.append("noise.schedule_weight must be true")
    if bool(config.noise.complement_masking):
        failures.append("noise.complement_masking must be false")
    if bool(config.noise.suffix_masking):
        failures.append("noise.suffix_masking must be false")
    if bool(config.noise.time_conditioning):
        failures.append("noise.time_conditioning must be false")
    if _uses_per_token_noise_levels(config):
        failures.append("noise.per_token_noise_levels must be false")
    if _uses_block_causal_pilot(config):
        failures.append("model.block_causal_pilot_size must be 0")
    if _uses_bd3lm(config):
        failures.append("noise.type must not use BD3LM")
    if _uses_token_edit_corruption(config):
        failures.append("token-edit corruption must be disabled")
    if config.model.output_factorization != "flat":
        failures.append("model.output_factorization must be flat")
    if config.model.auxiliary_head_layers:
        failures.append("auxiliary heads must be disabled")
    if bool(config.model.length_head):
        failures.append("length head must be disabled")
    if bool(config.model.target_length_conditioning):
        failures.append("target length conditioning must be disabled")
    if float(config.train.auxiliary_loss_weight) != 0.0:
        failures.append("auxiliary loss must be disabled")
    if float(config.train.entropy_bonus_beta) != 0.0:
        failures.append("entropy bonus must be disabled")
    if float(config.train.confidence_penalty_weight) != 0.0:
        failures.append("confidence penalty must be disabled")
    if float(config.train.z_loss_weight) != 0.0:
        failures.append("z-loss must be disabled")
    if float(config.train.eos_hazard_loss_weight) != 0.0:
        failures.append("EOS hazard loss must be disabled")
    if float(config.train.length_head_loss_weight) != 0.0:
        failures.append("length-head loss must be disabled")
    if bool(config.train.position_decile_balanced_loss):
        failures.append("position decile balancing must be disabled")
    if config.train.uniform_loss_estimator != "single_sample":
        failures.append("uniform loss estimator must be single_sample")
    return failures


def _json_float_list(values: torch.Tensor) -> list[float]:
    return [float(v) for v in values.detach().to(device="cpu", dtype=torch.float32).flatten().tolist()]


def _json_int_list(values: torch.Tensor) -> list[int]:
    return [int(v) for v in values.detach().to(device="cpu", dtype=torch.long).flatten().tolist()]


def _strict_mdlm_t_summary(
    t_val: float | torch.Tensor | None,
    x_0: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[dict[str, bool | float | int | list[float] | list[int] | str], torch.Tensor | None, list[str]]:
    failures: list[str] = []
    batch_size = int(x_0.shape[0])
    if t_val is None:
        return (
            {
                "strict_mdlm_audit_t_kind": "none",
                "strict_mdlm_audit_t_shape": [],
                "strict_mdlm_audit_one_scalar_t_per_sequence": False,
                "strict_mdlm_audit_t_broadcast_from_batch_scalar": False,
                "strict_mdlm_audit_per_row_t_count": 0,
                "strict_mdlm_audit_per_row_t_values": [],
                "strict_mdlm_audit_per_row_t_unique_counts": [],
                "strict_mdlm_audit_per_row_t_unique_value_count": 0,
                "strict_mdlm_audit_per_row_t_min": 0.0,
                "strict_mdlm_audit_per_row_t_max": 0.0,
                "strict_mdlm_audit_per_row_t_mean": 0.0,
                "strict_mdlm_audit_per_row_t_std": 0.0,
            },
            None,
            ["active corruption path returned no t value"],
        )

    if isinstance(t_val, torch.Tensor):
        t_shape = [int(dim) for dim in t_val.shape]
        t_tensor = t_val.detach().to(device=x_0.device, dtype=torch.float32)
        if t_tensor.ndim == 0:
            per_row_t = t_tensor.reshape(1).repeat(batch_size)
            t_kind = "scalar_tensor"
            unique_counts = torch.ones(batch_size, dtype=torch.long, device=x_0.device)
            broadcast_from_scalar = True
        elif t_tensor.ndim == 1 and t_tensor.shape[0] == batch_size:
            per_row_t = t_tensor
            t_kind = "per_row_tensor"
            unique_counts = torch.ones(batch_size, dtype=torch.long, device=x_0.device)
            broadcast_from_scalar = False
        elif t_tensor.shape == x_0.shape:
            per_row_values: list[float] = []
            unique_count_values: list[int] = []
            for row_idx in range(batch_size):
                row_t = t_tensor[row_idx][loss_mask[row_idx]]
                if row_t.numel() == 0:
                    per_row_values.append(0.0)
                    unique_count_values.append(0)
                    continue
                unique_count_values.append(int(torch.unique(row_t).numel()))
                per_row_values.append(float(row_t.flatten()[0].item()))
            per_row_t = torch.tensor(per_row_values, device=x_0.device, dtype=torch.float32)
            unique_counts = torch.tensor(unique_count_values, device=x_0.device, dtype=torch.long)
            t_kind = "per_token_tensor"
            broadcast_from_scalar = False
            failures.append("t is token-shaped; strict MDLM requires one scalar t per sequence")
        else:
            per_row_t = None
            unique_counts = torch.zeros(batch_size, dtype=torch.long, device=x_0.device)
            t_kind = "unsupported_tensor"
            broadcast_from_scalar = False
            failures.append(f"unsupported t tensor shape {tuple(t_tensor.shape)}")
    else:
        per_row_t = torch.full((batch_size,), float(t_val), device=x_0.device, dtype=torch.float32)
        t_shape = []
        t_kind = "python_scalar"
        unique_counts = torch.ones(batch_size, dtype=torch.long, device=x_0.device)
        broadcast_from_scalar = True

    if per_row_t is not None:
        if bool((per_row_t <= 0).any().item()):
            failures.append("t must be positive for schedule weighting")
        if bool((per_row_t > 1).any().item()):
            failures.append("t must be <= 1")
        one_scalar_per_sequence = bool((unique_counts == 1).all().item())
        if not one_scalar_per_sequence:
            failures.append("at least one sequence has multiple t values")
        if bool(broadcast_from_scalar):
            failures.append("t was broadcast from a batch-global scalar")
        per_row_values = _json_float_list(per_row_t)
        per_row_t_count = int(per_row_t.numel())
        per_row_t_unique_value_count = int(torch.unique(per_row_t).numel()) if per_row_t_count else 0
        per_row_t_min = float(per_row_t.min().item()) if per_row_t_count else 0.0
        per_row_t_max = float(per_row_t.max().item()) if per_row_t_count else 0.0
        per_row_t_mean = float(per_row_t.mean().item()) if per_row_t_count else 0.0
        per_row_t_std = float(per_row_t.std(unbiased=False).item()) if per_row_t_count > 1 else 0.0
    else:
        one_scalar_per_sequence = False
        per_row_values = []
        per_row_t_count = 0
        per_row_t_unique_value_count = 0
        per_row_t_min = 0.0
        per_row_t_max = 0.0
        per_row_t_mean = 0.0
        per_row_t_std = 0.0

    return (
        {
            "strict_mdlm_audit_t_kind": t_kind,
            "strict_mdlm_audit_t_shape": t_shape,
            "strict_mdlm_audit_one_scalar_t_per_sequence": one_scalar_per_sequence,
            "strict_mdlm_audit_t_broadcast_from_batch_scalar": bool(broadcast_from_scalar),
            "strict_mdlm_audit_per_row_t_count": per_row_t_count,
            "strict_mdlm_audit_per_row_t_values": per_row_values,
            "strict_mdlm_audit_per_row_t_unique_counts": _json_int_list(unique_counts),
            "strict_mdlm_audit_per_row_t_unique_value_count": per_row_t_unique_value_count,
            "strict_mdlm_audit_per_row_t_min": per_row_t_min,
            "strict_mdlm_audit_per_row_t_max": per_row_t_max,
            "strict_mdlm_audit_per_row_t_mean": per_row_t_mean,
            "strict_mdlm_audit_per_row_t_std": per_row_t_std,
        },
        per_row_t,
        failures,
    )


def _strict_mdlm_schedule_metrics(
    t_val: float | torch.Tensor,
    x_0: torch.Tensor,
    target_indices: torch.Tensor,
    active_token_count: int,
    loss_denominator_tokens: int,
) -> dict[str, bool | float | int | str]:
    if active_token_count <= 0:
        return {
            "strict_mdlm_audit_schedule_weight_status": "failed",
            "strict_mdlm_audit_schedule_weight_error": "no active target tokens",
            "strict_mdlm_audit_schedule_weight_target_count": 0,
            "strict_mdlm_audit_schedule_weight_mean": 0.0,
            "strict_mdlm_audit_expected_inv_t_mean": 0.0,
            "strict_mdlm_audit_schedule_weight_max_abs_error": 0.0,
            "strict_mdlm_audit_loss_normalization_scale": 0.0,
            "strict_mdlm_audit_expected_loss_normalization_scale": 0.0,
            "strict_mdlm_audit_loss_normalization_scale_abs_error": 0.0,
        }

    if isinstance(t_val, torch.Tensor):
        t_tensor = t_val.detach().to(device=x_0.device, dtype=torch.float32)
        try:
            target_t = _per_target_t_values(t_tensor, x_0, target_indices)
            if t_tensor.shape == x_0.shape:
                expected_target_t = t_tensor.reshape(-1).index_select(0, target_indices).float()
            elif t_tensor.ndim == 1 and t_tensor.shape[0] == x_0.shape[0]:
                target_rows = torch.div(target_indices, x_0.shape[1], rounding_mode="floor")
                expected_target_t = t_tensor.index_select(0, target_rows).float()
            else:
                raise ValueError(f"unsupported tensor t shape {tuple(t_tensor.shape)}")
        except ValueError as exc:
            return {
                "strict_mdlm_audit_schedule_weight_status": "failed",
                "strict_mdlm_audit_schedule_weight_error": str(exc),
                "strict_mdlm_audit_schedule_weight_target_count": int(active_token_count),
                "strict_mdlm_audit_schedule_weight_mean": 0.0,
                "strict_mdlm_audit_expected_inv_t_mean": 0.0,
                "strict_mdlm_audit_schedule_weight_max_abs_error": 0.0,
                "strict_mdlm_audit_loss_normalization_scale": 0.0,
                "strict_mdlm_audit_expected_loss_normalization_scale": 0.0,
                "strict_mdlm_audit_loss_normalization_scale_abs_error": 0.0,
            }
        token_weights = 1.0 / target_t.float().clamp_min(1e-8)
        expected_weights = 1.0 / expected_target_t.float().clamp_min(1e-8)
        loss_normalization_scale = float(token_weights.sum().item()) / max(1, int(loss_denominator_tokens))
    else:
        schedule_weight = 1.0 / max(float(t_val), 1e-8)
        token_weights = torch.full(
            (int(active_token_count),),
            float(schedule_weight),
            dtype=torch.float32,
            device=x_0.device,
        )
        expected_weights = torch.full_like(token_weights, float(schedule_weight), dtype=torch.float32)
        loss_normalization_scale = _fast_dllm_loss_normalization_scale(
            int(active_token_count),
            int(loss_denominator_tokens),
            float(t_val),
            schedule_weight=True,
        )

    expected_loss_normalization_scale = float(expected_weights.sum().item()) / max(1, int(loss_denominator_tokens))
    max_abs_error = (
        float((token_weights.float() - expected_weights.float()).abs().max().item()) if token_weights.numel() else 0.0
    )
    scale_error = abs(float(loss_normalization_scale) - expected_loss_normalization_scale)
    return {
        "strict_mdlm_audit_schedule_weight_status": "ok" if max_abs_error <= 1e-6 and scale_error <= 1e-6 else "failed",
        "strict_mdlm_audit_schedule_weight_error": "",
        "strict_mdlm_audit_schedule_weight_target_count": int(active_token_count),
        "strict_mdlm_audit_schedule_weight_mean": float(token_weights.mean().item()),
        "strict_mdlm_audit_expected_inv_t_mean": float(expected_weights.mean().item()),
        "strict_mdlm_audit_schedule_weight_max_abs_error": max_abs_error,
        "strict_mdlm_audit_loss_normalization_scale": float(loss_normalization_scale),
        "strict_mdlm_audit_expected_loss_normalization_scale": expected_loss_normalization_scale,
        "strict_mdlm_audit_loss_normalization_scale_abs_error": scale_error,
    }


STRICT_MDLM_ALL_TOKEN_BAYES_AUDIT_CORRUPTIONS = 16

STRICT_MDLM_ALL_TOKEN_BAYES_AUDIT_SAMPLE_LIMIT = 4

STRICT_MDLM_ALL_TOKEN_BAYES_ANALYTIC_SCOPE = "empirical_toy_prior_data_and_corruption_only"

STRICT_MDLM_ALL_TOKEN_BAYES_HISTORICAL_SCOPE = "optional_historical_context"

STRICT_MDLM_ALL_TOKEN_BAYES_HISTORICAL_COMPARABILITY = "not_assessed"


def _strict_mdlm_all_token_bayes_audit_enabled(config: ExperimentConfig) -> bool:
    return bool(getattr(config.train, "strict_mdlm_all_token_bayes_audit_only", False))


def _strict_mdlm_all_token_bayes_audit_static_failures(config: ExperimentConfig) -> list[str]:
    failures = _strict_mdlm_all_token_bayes_data_failures(config)
    if config.data.synthetic_eos_overfit_loss_mask_policy != "all_lossable_tokens":
        failures.append("all-token Bayes audit requires all_lossable_tokens loss mask policy")
    if not bool(config.train.strict_mdlm_all_token_diagnostic_telemetry):
        failures.append("all-token Bayes audit requires diagnostic telemetry enabled")
    expected_t_values = [0.55, 0.95]
    actual_t_values = _strict_mdlm_all_token_diagnostic_t_values(config)
    if len(actual_t_values) != len(expected_t_values) or any(
        abs(actual - expected) > 1e-8 for actual, expected in zip(actual_t_values, expected_t_values, strict=False)
    ):
        failures.append("all-token Bayes audit requires fixed t values [0.55, 0.95]")
    return failures


def _strict_mdlm_all_token_bayes_data_failures(config: ExperimentConfig) -> list[str]:
    failures = _strict_mdlm_static_invariant_failures(config)
    if config.data.synthetic_dataset != "eos_overfit_canary":
        failures.append("data.synthetic_dataset must be eos_overfit_canary")
    if config.data.synthetic_eos_overfit_loss_mask_policy not in ("eos_only", "all_lossable_tokens"):
        failures.append("data.synthetic_eos_overfit_loss_mask_policy must be eos_only or all_lossable_tokens")
    if config.data.tokenizer != "byte":
        failures.append("data.tokenizer must be byte")
    if not config.data.append_eos:
        failures.append("data.append_eos must be true")
    if not config.data.loss_on_eos:
        failures.append("data.loss_on_eos must be true")
    if float(config.data.random_length_ratio) != 0.0:
        failures.append("data.random_length_ratio must be 0")
    if float(config.data.continuation_ratio) != 0.0:
        failures.append("data.continuation_ratio must be 0")
    return failures


def _strict_mdlm_all_token_bayes_model_config(config: ExperimentConfig, tokenizer) -> DiffusionConfig:
    return DiffusionConfig(
        sequence_len=int(config.model.max_seq_len),
        vocab_size=int(tokenizer.get_vocab_size()),
        n_layer=int(config.model.depth),
        n_head=int(config.model.n_head),
        n_embd=int(config.model.n_embd),
        mask_token_id=int(tokenizer.get_mask_token_id()),
        mlp_hidden_dim=int(config.model.mlp_hidden_dim),
        attention_scale=float(config.model.attention_scale),
        local_attention_lower_layers=int(config.model.local_attention_lower_layers),
        local_attention_window=int(config.model.local_attention_window),
        auxiliary_head_layers=tuple(int(i) for i in config.model.auxiliary_head_layers),
        time_conditioning=bool(config.noise.time_conditioning),
        bd3lm_block_size=0,
        output_factorization=config.model.output_factorization,
        causal=bool(config.model.causal),
        residual_output_projection_init=config.model.residual_output_projection_init,
        embedding_head_init_std_multiplier=float(config.model.embedding_head_init_std_multiplier),
        length_head=bool(config.model.length_head),
        target_length_conditioning=bool(config.model.target_length_conditioning),
        mask_position_embedding=bool(config.model.mask_position_embedding),
    )


def _strict_mdlm_bayes_token_label(tokenizer, token_id: int) -> str:
    token_id = int(token_id)
    if token_id == int(tokenizer.get_eos_token_id()):
        return "<eos>"
    if token_id == int(tokenizer.get_mask_token_id()):
        return "<mask>"
    pad_id = tokenizer.get_pad_token_id()
    if pad_id is not None and token_id == int(pad_id):
        return "<pad>"
    bos_id = tokenizer.get_bos_token_id()
    if bos_id is not None and token_id == int(bos_id):
        return "<bos>"
    if 32 <= token_id <= 126:
        return chr(token_id)
    if 0 <= token_id <= 255:
        return f"\\x{token_id:02x}"
    return f"<{token_id}>"


def _strict_mdlm_bayes_decode_row(tokenizer, tokens: torch.Tensor, attention_mask: torch.Tensor) -> str:
    token_values = tokens.detach().to(device="cpu", dtype=torch.long).flatten().tolist()
    mask_values = attention_mask.detach().to(device="cpu", dtype=torch.bool).flatten().tolist()
    return " ".join(
        _strict_mdlm_bayes_token_label(tokenizer, token_id)
        for token_id, valid in zip(token_values, mask_values, strict=True)
        if bool(valid)
    )


def _strict_mdlm_all_token_diagnostic_t_values(config: ExperimentConfig) -> list[float]:
    configured_values = list(getattr(config.train, "strict_mdlm_all_token_diagnostic_t_values", []))
    if configured_values:
        return [float(value) for value in configured_values]
    low_t = float(config.noise.t_min)
    high_t = float(config.noise.t_max)
    if abs(low_t - high_t) <= 1e-8:
        return [low_t]
    return [low_t, high_t]


def _strict_mdlm_all_token_bayes_expected_loss_mask(
    batch: DLMDataBatch,
    tokenizer,
    data_config,
) -> torch.Tensor:
    eos_token_id = int(tokenizer.get_eos_token_id())
    expected = batch.attention_mask.clone()

    if not data_config.loss_on_bos:
        bos_id = tokenizer.get_bos_token_id()
        if bos_id is not None:
            expected &= batch.x_0 != int(bos_id)
    if not data_config.loss_on_eos:
        expected &= batch.x_0 != eos_token_id
    return expected


def _strict_mdlm_all_token_bayes_config_summary(
    config: ExperimentConfig,
    model_config: DiffusionConfig,
) -> dict[str, bool | float | int | str | dict]:
    return {
        "objective": "strict_mdlm_masked_ce",
        "loss_mask_policy": "all_lossable_tokens",
        "loss_denominator": "all_lossable_tokens",
        "per_row_t_enabled": bool(_uses_strict_mdlm_per_row_t(config)),
        "training_t_min": float(config.noise.t_min),
        "training_t_max": float(config.noise.t_max),
        "schedule_weight": bool(config.noise.schedule_weight),
        "complement_masking": bool(config.noise.complement_masking),
        "time_conditioning": bool(config.noise.time_conditioning),
        "suffix_masking": bool(config.noise.suffix_masking),
        "per_token_noise_levels": bool(_uses_per_token_noise_levels(config)),
        "auxiliary_head_layers": len(config.model.auxiliary_head_layers),
        "auxiliary_loss_weight": float(config.train.auxiliary_loss_weight),
        "sampler": str(config.eval.sampler),
        "sample_token_selection": str(config.eval.sample_token_selection),
        "model": asdict(model_config),
    }


def _safe_float(value, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else default
    return default


def _strict_mdlm_bayes_value_distribution(values: list[int]) -> tuple[dict[int, int], int, float, float]:
    counts: dict[int, int] = {}
    for value in values:
        token_id = int(value)
        counts[token_id] = counts.get(token_id, 0) + 1
    if not counts:
        return counts, -1, 0.0, 0.0
    total = float(sum(counts.values()))
    top_token, top_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
    probs = [count / total for count in counts.values()]
    entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
    return counts, int(top_token), top_count / total, entropy


def _strict_mdlm_bayes_decile_metrics(prefix: str, values: list[float]) -> dict[str, float | int | list[float] | bool]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    count = len(finite_values)
    if count <= 0:
        deciles = [0.0 for _ in range(11)]
        metrics: dict[str, float | int | list[float] | bool] = {
            f"{prefix}measured": False,
            f"{prefix}count": 0,
            f"{prefix}mean": 0.0,
            f"{prefix}min": 0.0,
            f"{prefix}max": 0.0,
            f"{prefix}deciles": deciles,
        }
    else:
        arr = np.asarray(finite_values, dtype=np.float64)
        deciles = [float(np.quantile(arr, pct / 10.0)) for pct in range(11)]
        metrics = {
            f"{prefix}measured": True,
            f"{prefix}count": count,
            f"{prefix}mean": float(arr.mean()),
            f"{prefix}min": float(arr.min()),
            f"{prefix}max": float(arr.max()),
            f"{prefix}deciles": deciles,
        }
    for idx, value in enumerate(deciles):
        metrics[f"{prefix}decile_{idx:02d}"] = float(value)
    return metrics


def _strict_mdlm_all_token_bayes_batch_provenance(
    config: ExperimentConfig,
    tokenizer,
    batch: DLMDataBatch,
) -> tuple[dict[str, bool | float | int | str | list], list[dict], list[str]]:
    failures: list[str] = []
    eos_token_id = int(tokenizer.get_eos_token_id())
    expected_docs = encode_document_batch(tokenizer, list(EOS_OVERFIT_CANARY_DOCUMENTS), config.data)
    if len(expected_docs) != len(EOS_OVERFIT_CANARY_DOCUMENTS):
        failures.append("encoded EOS overfit canary document count does not match source documents")

    expected_loss_mask = _strict_mdlm_all_token_bayes_expected_loss_mask(
        batch,
        tokenizer,
        config.data,
    )
    loss_mask_matches = bool(torch.equal(batch.loss_mask, expected_loss_mask))
    if not loss_mask_matches:
        failures.append("all-token toy loss mask does not match all lossable positions")

    row_records: list[dict] = []
    doc_cycle = max(1, len(expected_docs))
    row_count = int(batch.x_0.shape[0])
    seq_len = int(batch.x_0.shape[1])
    for row_idx in range(row_count):
        doc_idx = row_idx % doc_cycle
        expected = expected_docs[doc_idx] if expected_docs else []
        expected_len = len(expected)
        row_attention = batch.attention_mask[row_idx]
        row_loss_mask = batch.loss_mask[row_idx]
        row_tokens = batch.x_0[row_idx]
        actual_len = int(row_attention.sum().item())
        expected_attention = torch.zeros_like(row_attention)
        expected_attention[:expected_len] = True
        token_prefix = row_tokens[:expected_len].detach().to(device="cpu", dtype=torch.long).tolist()
        eos_positions = ((row_tokens == eos_token_id) & row_attention).nonzero(as_tuple=True)[0].detach().cpu().tolist()
        if actual_len != expected_len:
            failures.append(f"row {row_idx} attention length {actual_len} != expected {expected_len}")
        if token_prefix != [int(token_id) for token_id in expected]:
            failures.append(f"row {row_idx} clean tokens do not match expected synthetic document")
        if not bool(torch.equal(row_attention, expected_attention)):
            failures.append(f"row {row_idx} attention mask does not match expected synthetic document length")
        if not bool(torch.equal(row_loss_mask, row_attention)):
            failures.append(f"row {row_idx} loss mask is not all valid tokens")
        if len(eos_positions) != 1:
            failures.append(f"row {row_idx} expected exactly one EOS position, got {len(eos_positions)}")
        row_records.append(
            {
                "row": int(row_idx),
                "source_document_index": int(doc_idx),
                "source_document_text": str(EOS_OVERFIT_CANARY_DOCUMENTS[doc_idx])
                if doc_idx < len(EOS_OVERFIT_CANARY_DOCUMENTS)
                else "",
                "expected_length": int(expected_len),
                "attention_length": int(actual_len),
                "lossable_token_count": int(row_loss_mask.sum().item()),
                "eos_positions": [int(pos) for pos in eos_positions],
                "clean_tokens": [int(token_id) for token_id in token_prefix],
                "decoded": _strict_mdlm_bayes_decode_row(tokenizer, row_tokens, row_attention),
            }
        )

    lossable_tokens = int(batch.loss_mask.sum().item())
    eos_lossable = int(((batch.x_0 == eos_token_id) & batch.loss_mask).sum().item())
    non_eos_lossable = int(((batch.x_0 != eos_token_id) & batch.loss_mask).sum().item())
    metrics: dict[str, bool | float | int | str | list] = {
        "all_token_bayes_row_provenance_measured": True,
        "all_token_bayes_source_document_count": int(len(EOS_OVERFIT_CANARY_DOCUMENTS)),
        "all_token_bayes_encoded_document_count": int(len(expected_docs)),
        "all_token_bayes_batch_row_count": row_count,
        "all_token_bayes_batch_seq_len": seq_len,
        "all_token_bayes_lossable_position_provenance_measured": True,
        "all_token_bayes_loss_mask_matches_all_lossable_tokens": loss_mask_matches,
        "all_token_bayes_lossable_token_count": lossable_tokens,
        "all_token_bayes_eos_lossable_token_count": eos_lossable,
        "all_token_bayes_non_eos_lossable_token_count": non_eos_lossable,
        "all_token_bayes_attention_token_count": int(batch.attention_mask.sum().item()),
        "all_token_bayes_lossable_attention_mismatch_count": int(
            (batch.loss_mask & ~batch.attention_mask).sum().item()
        ),
        "all_token_bayes_row_attention_lengths": [
            int(value) for value in batch.attention_mask.sum(dim=1).detach().cpu().tolist()
        ],
        "all_token_bayes_row_lossable_counts": [
            int(value) for value in batch.loss_mask.sum(dim=1).detach().cpu().tolist()
        ],
        "all_token_bayes_row_provenance_status": "failed" if failures else "ok",
        "all_token_bayes_row_provenance_error": "; ".join(failures),
    }
    return metrics, row_records, failures


def _strict_mdlm_all_token_bayes_candidate_support(
    batch: DLMDataBatch,
    observed_row: int,
    active: torch.Tensor,
    z_t: torch.Tensor,
) -> list[int]:
    observed_attention = batch.attention_mask[observed_row]
    observed_loss = batch.loss_mask[observed_row]
    observed_active = active[observed_row]
    visible_positions = observed_attention & ~observed_active
    support: list[int] = []
    for candidate_idx in range(int(batch.x_0.shape[0])):
        if not bool(torch.equal(batch.attention_mask[candidate_idx], observed_attention)):
            continue
        if not bool(torch.equal(batch.loss_mask[candidate_idx], observed_loss)):
            continue
        if not bool(torch.equal(batch.x_0[candidate_idx][visible_positions], z_t[observed_row][visible_positions])):
            continue
        support.append(int(candidate_idx))
    return support


def _strict_mdlm_all_token_bayes_fixed_t_record(
    config: ExperimentConfig,
    tokenizer,
    noise,
    batch: DLMDataBatch,
    fixed_t: float,
    t_index: int,
    corruption_count: int,
) -> tuple[dict, list[str]]:
    x_0 = batch.x_0
    loss_mask = batch.loss_mask
    attention_mask = batch.attention_mask
    eos_token_id = int(tokenizer.get_eos_token_id())
    mask_token_id = int(tokenizer.get_mask_token_id())
    seed_base = int(config.runtime.training_seed if config.runtime.training_seed >= 0 else config.runtime.seed)
    failures: list[str] = []
    posterior_records: list[dict] = []
    decoded_samples: list[dict] = []
    support_counts: list[int] = []
    top_concentrations: list[float] = []
    all_entropies: list[float] = []
    eos_posteriors: list[float] = []
    eos_margins: list[float] = []
    eos_entropies: list[float] = []
    target_count = 0
    correct_count = 0
    eos_count = 0
    eos_correct_count = 0
    non_eos_count = 0
    non_eos_correct_count = 0
    posterior_support_empty_count = 0
    posterior_tie_count = 0
    active_token_total = 0
    active_eos_total = 0
    active_non_eos_total = 0
    mask_token_mismatch_count = 0
    t_summary_metrics: dict[str, bool | float | int | list[float] | list[int] | str] = {}
    schedule_metrics: dict[str, bool | float | int | str] = {}
    lag1_pair_total = 0
    lag1_both_total = 0
    lag1_expected_both_total = 0.0

    saved_state = _rng_state(config)
    try:
        for corruption_idx in range(int(corruption_count)):
            seed = seed_base + 4_000_003 + int(t_index) * 100_003 + int(corruption_idx) * 10_007
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32 - 1))
            random.seed(seed)
            if config.runtime.device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            t_tensor = torch.full(
                (x_0.shape[0],),
                float(fixed_t),
                device=x_0.device,
                dtype=torch.float32,
            )
            z_t, w = noise.corrupt(x_0, t_tensor, maskable=loss_mask)
            active = (w > 0) & loss_mask
            target_indices = active.reshape(-1).nonzero(as_tuple=True)[0]
            active_token_total += int(active.sum().item())
            active_eos_total += int(((x_0 == eos_token_id) & active).sum().item())
            active_non_eos_total += int(((x_0 != eos_token_id) & active).sum().item())
            mask_token_mismatch_count += int(((z_t == mask_token_id) != active).sum().item())

            pair_mask = loss_mask[:, :-1] & loss_mask[:, 1:]
            pair_count = int(pair_mask.sum().item())
            if pair_count > 0:
                both_count = int((active[:, :-1] & active[:, 1:] & pair_mask).sum().item())
                pair_counts_per_row = pair_mask.sum(dim=1).to(dtype=torch.float32)
                lag1_pair_total += pair_count
                lag1_both_total += both_count
                lag1_expected_both_total += float(((t_tensor.float() ** 2) * pair_counts_per_row).sum().item())

            if corruption_idx == 0:
                t_summary_metrics, _per_row_t, t_failures = _strict_mdlm_t_summary(
                    t_tensor,
                    x_0,
                    loss_mask,
                )
                if t_failures:
                    failures.extend(f"fixed t {fixed_t:g}: {failure}" for failure in t_failures)
                schedule_metrics = _strict_mdlm_schedule_metrics(
                    t_tensor,
                    x_0,
                    target_indices,
                    int(target_indices.numel()),
                    int(loss_mask.sum().item()),
                )
                if schedule_metrics["strict_mdlm_audit_schedule_weight_status"] != "ok":
                    failures.append(
                        f"fixed t {fixed_t:g}: {schedule_metrics['strict_mdlm_audit_schedule_weight_error']}"
                    )
            if target_indices.numel() <= 0:
                failures.append(f"fixed t {fixed_t:g} corruption {corruption_idx} produced no active targets")
                continue

            corruption_sample_rows: list[dict] = []
            for row_idx in range(int(x_0.shape[0])):
                support = _strict_mdlm_all_token_bayes_candidate_support(batch, row_idx, active, z_t)
                row_active_positions = active[row_idx].nonzero(as_tuple=True)[0].detach().cpu().tolist()
                if not support:
                    posterior_support_empty_count += int(len(row_active_positions))
                    failures.append(
                        f"fixed t {fixed_t:g} corruption {corruption_idx} row {row_idx} has empty posterior support"
                    )
                    continue
                support_counts.extend([len(support) for _ in row_active_positions])
                row_position_records: list[dict] = []
                for position in row_active_positions:
                    pos = int(position)
                    true_token = int(x_0[row_idx, pos].item())
                    support_values = [int(x_0[candidate_idx, pos].item()) for candidate_idx in support]
                    counts, pred_token, top_concentration, entropy = _strict_mdlm_bayes_value_distribution(
                        support_values
                    )
                    if not counts:
                        continue
                    max_count = max(counts.values())
                    tie_count = sum(1 for count in counts.values() if count == max_count)
                    posterior_tie_count += int(tie_count > 1)
                    eos_posterior = counts.get(eos_token_id, 0) / max(1, len(support_values))
                    competitor_count = max(
                        [count for token_id, count in counts.items() if token_id != eos_token_id],
                        default=0,
                    )
                    eos_margin = eos_posterior - competitor_count / max(1, len(support_values))
                    correct = pred_token == true_token
                    target_count += 1
                    correct_count += int(correct)
                    top_concentrations.append(float(top_concentration))
                    all_entropies.append(float(entropy))
                    is_eos_target = true_token == eos_token_id
                    if is_eos_target:
                        eos_count += 1
                        eos_correct_count += int(correct)
                        eos_posteriors.append(float(eos_posterior))
                        eos_margins.append(float(eos_margin))
                        eos_entropies.append(float(entropy))
                    else:
                        non_eos_count += 1
                        non_eos_correct_count += int(correct)
                    position_record = {
                        "position": pos,
                        "true_token_id": true_token,
                        "true_token": _strict_mdlm_bayes_token_label(tokenizer, true_token),
                        "bayes_pred_token_id": int(pred_token),
                        "bayes_pred_token": _strict_mdlm_bayes_token_label(tokenizer, int(pred_token)),
                        "posterior_support_count": int(len(support)),
                        "posterior_unique_token_count": int(len(counts)),
                        "posterior_token_counts": {
                            str(token_id): int(count) for token_id, count in sorted(counts.items())
                        },
                        "posterior_token_probabilities": {
                            str(token_id): count / max(1, len(support_values))
                            for token_id, count in sorted(counts.items())
                        },
                        "posterior_top_token_probability": float(top_concentration),
                        "posterior_entropy": float(entropy),
                        "posterior_eos_probability": float(eos_posterior),
                        "posterior_eos_margin": float(eos_margin),
                        "bayes_correct": bool(correct),
                    }
                    posterior_records.append(
                        {
                            "corruption_index": int(corruption_idx),
                            "row": int(row_idx),
                            "fixed_t": float(fixed_t),
                            **position_record,
                        }
                    )
                    row_position_records.append(position_record)
                if corruption_idx == 0 and row_idx < int(STRICT_MDLM_ALL_TOKEN_BAYES_AUDIT_SAMPLE_LIMIT):
                    corruption_sample_rows.append(
                        {
                            "row": int(row_idx),
                            "fixed_t": float(fixed_t),
                            "active_positions": [int(pos) for pos in row_active_positions],
                            "posterior_support_rows": [int(idx) for idx in support],
                            "posterior_support_count": int(len(support)),
                            "original": _strict_mdlm_bayes_decode_row(
                                tokenizer,
                                x_0[row_idx],
                                attention_mask[row_idx],
                            ),
                            "corrupted": _strict_mdlm_bayes_decode_row(
                                tokenizer,
                                z_t[row_idx],
                                attention_mask[row_idx],
                            ),
                            "positions": row_position_records,
                        }
                    )
            if corruption_idx == 0:
                decoded_samples = corruption_sample_rows
    finally:
        _restore_rng_state(saved_state, config)

    if eos_count <= 0:
        failures.append(f"fixed t {fixed_t:g} produced no active true-EOS positions")
    if non_eos_count <= 0:
        failures.append(f"fixed t {fixed_t:g} produced no active non-EOS positions")
    if mask_token_mismatch_count:
        failures.append(
            f"fixed t {fixed_t:g} corrupted-state mask token pattern mismatch count {mask_token_mismatch_count}"
        )

    lag1_observed = lag1_both_total / max(1, lag1_pair_total)
    lag1_expected = lag1_expected_both_total / max(1, lag1_pair_total)
    lag1_tolerance = max(
        0.20,
        6.0 * math.sqrt(max(lag1_expected * (1.0 - lag1_expected), 0.0) / max(1, lag1_pair_total)),
    )
    lag1_abs_error = abs(lag1_observed - lag1_expected)
    if lag1_pair_total <= 0:
        failures.append(f"fixed t {fixed_t:g} has no lag-1 lossable pairs for independence measurement")
    elif lag1_abs_error > lag1_tolerance:
        failures.append(f"fixed t {fixed_t:g} lag-1 mask independence outside tolerance")

    support_metrics = _strict_mdlm_bayes_decile_metrics("bayes_posterior_support_", [float(v) for v in support_counts])
    top_metrics = _strict_mdlm_bayes_decile_metrics("bayes_top_token_concentration_", top_concentrations)
    entropy_metrics = _strict_mdlm_bayes_decile_metrics("bayes_posterior_entropy_", all_entropies)
    eos_posterior_metrics = _strict_mdlm_bayes_decile_metrics("bayes_eos_posterior_at_true_eos_", eos_posteriors)
    eos_margin_metrics = _strict_mdlm_bayes_decile_metrics("bayes_eos_margin_at_true_eos_", eos_margins)
    eos_entropy_metrics = _strict_mdlm_bayes_decile_metrics("bayes_entropy_at_true_eos_", eos_entropies)
    record = {
        "event": "strict_mdlm_all_token_bayes_audit_fixed_t",
        "all_token_bayes_fixed_t": float(fixed_t),
        "all_token_bayes_fixed_t_index": int(t_index),
        "all_token_bayes_requested_corruption_count": int(corruption_count),
        "all_token_bayes_target_count": int(target_count),
        "all_token_bayes_correct_count": int(correct_count),
        "all_token_bayes_accuracy": correct_count / max(1, target_count),
        "all_token_bayes_eos_target_count": int(eos_count),
        "all_token_bayes_eos_correct_count": int(eos_correct_count),
        "all_token_bayes_eos_accuracy": eos_correct_count / max(1, eos_count),
        "all_token_bayes_non_eos_target_count": int(non_eos_count),
        "all_token_bayes_non_eos_correct_count": int(non_eos_correct_count),
        "all_token_bayes_non_eos_accuracy": non_eos_correct_count / max(1, non_eos_count),
        "all_token_bayes_posterior_support_empty_count": int(posterior_support_empty_count),
        "all_token_bayes_posterior_tie_count": int(posterior_tie_count),
        "all_token_bayes_active_token_total": int(active_token_total),
        "all_token_bayes_active_eos_total": int(active_eos_total),
        "all_token_bayes_active_non_eos_total": int(active_non_eos_total),
        "all_token_bayes_mask_token_pattern_mismatch_count": int(mask_token_mismatch_count),
        "all_token_bayes_lag1_pair_count": int(lag1_pair_total),
        "all_token_bayes_lag1_both_masked_fraction": float(lag1_observed),
        "all_token_bayes_lag1_expected_both_masked_fraction": float(lag1_expected),
        "all_token_bayes_lag1_abs_error": float(lag1_abs_error),
        "all_token_bayes_lag1_tolerance": float(lag1_tolerance),
        "all_token_bayes_lag1_independence_within_tolerance": bool(
            lag1_pair_total > 0 and lag1_abs_error <= lag1_tolerance
        ),
        "all_token_bayes_decoded_sample_count": int(len(decoded_samples)),
        "all_token_bayes_decoded_samples": decoded_samples,
        "all_token_bayes_position_record_count": int(len(posterior_records)),
        "all_token_bayes_position_records": posterior_records[:64],
        "all_token_bayes_status": "failed" if failures else "ok",
        "all_token_bayes_error": "; ".join(failures),
        **{f"all_token_bayes_fixed_{key}": value for key, value in t_summary_metrics.items()},
        **{f"all_token_bayes_fixed_{key}": value for key, value in schedule_metrics.items()},
        **support_metrics,
        **top_metrics,
        **entropy_metrics,
        **eos_posterior_metrics,
        **eos_margin_metrics,
        **eos_entropy_metrics,
    }
    return record, failures


def _strict_mdlm_all_token_bayes_comparison_table(config: ExperimentConfig) -> tuple[list[dict], dict, list[str]]:
    issues: list[str] = []
    table: list[dict] = []
    requested_run_ids = [str(run_id) for run_id in config.train.comparison_run_ids]
    required_metric_keys = [
        "status",
        "experiment_type",
        "eos_overfit_canary_fixed_t",
        "canary_eos_reconstruction_accuracy",
        "canary_eos_reconstruction_count",
        "canary_eos_mean_rank",
        "canary_non_eos_reconstruction_accuracy",
    ]
    optional_metric_keys = [
        "last_all_token_diag_fixed_t",
        "last_all_token_diag_eos_accuracy",
        "last_all_token_diag_eos_target_count",
        "last_all_token_diag_eos_rank_mean",
        "last_all_token_diag_eos_margin_mean",
        "last_all_token_diag_decoded_eos_position_accuracy",
    ]
    for run_id in requested_run_ids:
        entry: dict[str, object] = {
            "run_id": run_id,
            "comparison_status": "not_measured",
            "comparison_scope": STRICT_MDLM_ALL_TOKEN_BAYES_HISTORICAL_SCOPE,
            "comparability_status": STRICT_MDLM_ALL_TOKEN_BAYES_HISTORICAL_COMPARABILITY,
            "missing_required_metric_keys": [],
            "missing_optional_metric_keys": [],
        }
        try:
            manifest_path, artifact_root = _comparison_manifest_path(config.train.run_dir, run_id)
            manifest = _load_json_if_exists(manifest_path)
            metrics_path = _find_manifest_artifact_path(
                manifest,
                artifact_root,
                name="metrics.json",
                kind="metrics",
            )
            if metrics_path is None:
                raise FileNotFoundError(f"comparison run {run_id} manifest has no existing metrics.json artifact")
            metrics = _load_json_if_exists(metrics_path)
            if not metrics:
                raise FileNotFoundError(f"comparison run {run_id} metrics.json is empty or invalid")
            missing_required = [key for key in required_metric_keys if key not in metrics]
            missing_optional = [key for key in optional_metric_keys if key not in metrics]
            if missing_required:
                issues.append(f"comparison run {run_id} missing required metrics {missing_required}")
            entry.update(
                {
                    "comparison_status": "failed" if missing_required else "ok",
                    "source_artifacts_manifest": str(manifest_path),
                    "source_metrics_artifact": str(
                        _run_artifact_manifest_relative_path(config.train.run_dir, metrics_path)
                    ),
                    "missing_required_metric_keys": missing_required,
                    "missing_optional_metric_keys": missing_optional,
                }
            )
            for key in required_metric_keys + optional_metric_keys:
                if key in metrics:
                    entry[key] = metrics[key]
        except Exception as exc:
            issues.append(f"comparison run {run_id} not measured: {exc}")
            entry["comparison_status"] = "failed"
            entry["comparison_error"] = str(exc)
        table.append(entry)

    loaded_count = sum(1 for entry in table if entry.get("comparison_status") == "ok")
    requested_count = len(requested_run_ids)
    if requested_count == 0:
        historical_status = "not_requested"
    elif loaded_count == requested_count:
        historical_status = "complete"
    elif loaded_count > 0:
        historical_status = "partial"
    else:
        historical_status = "unavailable"
    metrics_out = {
        "all_token_bayes_historical_comparison_requested": bool(requested_run_ids),
        "all_token_bayes_historical_comparison_scope": STRICT_MDLM_ALL_TOKEN_BAYES_HISTORICAL_SCOPE,
        "all_token_bayes_historical_comparison_gates_analytic_audit": False,
        "all_token_bayes_historical_comparison_comparability_status": (
            STRICT_MDLM_ALL_TOKEN_BAYES_HISTORICAL_COMPARABILITY
        ),
        "all_token_bayes_historical_comparison_status": historical_status,
        "all_token_bayes_historical_comparison_issues": list(issues),
        "all_token_bayes_historical_comparison_error": "; ".join(issues),
        "all_token_bayes_historical_comparison_requested_run_ids": requested_run_ids,
        "all_token_bayes_historical_comparison_requested_count": int(requested_count),
        "all_token_bayes_historical_comparison_loaded_count": int(loaded_count),
        "all_token_bayes_comparison_table_measured": bool(requested_count > 0 and loaded_count == requested_count),
        "all_token_bayes_comparison_table_status": (
            "not_requested" if historical_status == "not_requested" else ("ok" if not issues else "failed")
        ),
        "all_token_bayes_comparison_table_error": "; ".join(issues),
        "all_token_bayes_comparison_run_count": int(requested_count),
        "all_token_bayes_comparison_loaded_count": int(loaded_count),
    }
    return table, metrics_out, issues


def _strict_mdlm_all_token_bayes_audit_summary(
    records: list[dict],
    expected_t_values: list[float],
) -> tuple[dict, list[str]]:
    fixed_records = [record for record in records if record.get("event") == "strict_mdlm_all_token_bayes_audit_fixed_t"]
    failures: list[str] = []
    fixed_t_values = sorted(float(record.get("all_token_bayes_fixed_t", 0.0)) for record in fixed_records)
    if len(fixed_t_values) != len(expected_t_values) or any(
        abs(a - b) > 1e-8 for a, b in zip(fixed_t_values, sorted(expected_t_values), strict=False)
    ):
        failures.append("Bayes audit fixed t records are incomplete")
    for record in fixed_records:
        if record.get("all_token_bayes_status") != "ok":
            failures.append(str(record.get("all_token_bayes_error", "fixed-t Bayes audit failed")))
        for required_key in (
            "bayes_eos_posterior_at_true_eos_measured",
            "bayes_eos_margin_at_true_eos_measured",
            "bayes_entropy_at_true_eos_measured",
            "bayes_top_token_concentration_measured",
            "bayes_posterior_support_measured",
        ):
            if not bool(record.get(required_key, False)):
                failures.append(f"{required_key} was not measured at fixed t {record.get('all_token_bayes_fixed_t')}")
        if not bool(record.get("all_token_bayes_lag1_independence_within_tolerance", False)):
            failures.append(
                f"mask independence was not measured as ok at fixed t {record.get('all_token_bayes_fixed_t')}"
            )

    by_t = {round(float(record.get("all_token_bayes_fixed_t", 0.0)), 8): record for record in fixed_records}
    low = by_t.get(0.55, {})
    high = by_t.get(0.95, {})
    low_eos_accuracy = _safe_float(low.get("all_token_bayes_eos_accuracy"))
    high_eos_accuracy = _safe_float(high.get("all_token_bayes_eos_accuracy"))
    low_eos_posterior = _safe_float(low.get("bayes_eos_posterior_at_true_eos_mean"))
    high_eos_posterior = _safe_float(high.get("bayes_eos_posterior_at_true_eos_mean"))
    if low and high:
        if high_eos_accuracy < low_eos_accuracy - 0.25 or high_eos_posterior < low_eos_posterior - 0.25:
            interpretation = "high_noise_ambiguity_supported_under_empirical_toy_prior"
        elif high_eos_accuracy >= 0.90 and high_eos_posterior >= 0.90:
            interpretation = "high_t_eos_identifiable_under_empirical_toy_prior"
        else:
            interpretation = "inconclusive"
    else:
        interpretation = "inconclusive"
    if interpretation == "inconclusive":
        failures.append("Bayes low/high t comparison is inconclusive")

    summary = {
        "all_token_bayes_audit_fixed_t_record_count": int(len(fixed_records)),
        "all_token_bayes_audit_fixed_t_values": fixed_t_values,
        "all_token_bayes_audit_expected_fixed_t_values": [float(value) for value in expected_t_values],
        "all_token_bayes_t055_eos_accuracy": low_eos_accuracy,
        "all_token_bayes_t095_eos_accuracy": high_eos_accuracy,
        "all_token_bayes_t055_non_eos_accuracy": _safe_float(low.get("all_token_bayes_non_eos_accuracy")),
        "all_token_bayes_t095_non_eos_accuracy": _safe_float(high.get("all_token_bayes_non_eos_accuracy")),
        "all_token_bayes_t055_eos_posterior_mean": low_eos_posterior,
        "all_token_bayes_t095_eos_posterior_mean": high_eos_posterior,
        "all_token_bayes_t055_eos_margin_mean": _safe_float(low.get("bayes_eos_margin_at_true_eos_mean")),
        "all_token_bayes_t095_eos_margin_mean": _safe_float(high.get("bayes_eos_margin_at_true_eos_mean")),
        "all_token_bayes_t055_entropy_at_true_eos_mean": _safe_float(low.get("bayes_entropy_at_true_eos_mean")),
        "all_token_bayes_t095_entropy_at_true_eos_mean": _safe_float(high.get("bayes_entropy_at_true_eos_mean")),
        "all_token_bayes_t055_top_token_concentration_mean": _safe_float(low.get("bayes_top_token_concentration_mean")),
        "all_token_bayes_t095_top_token_concentration_mean": _safe_float(
            high.get("bayes_top_token_concentration_mean")
        ),
        "all_token_bayes_eos_accuracy_delta_t095_minus_t055": high_eos_accuracy - low_eos_accuracy,
        "all_token_bayes_eos_posterior_delta_t095_minus_t055": high_eos_posterior - low_eos_posterior,
        "all_token_bayes_analytic_scope": STRICT_MDLM_ALL_TOKEN_BAYES_ANALYTIC_SCOPE,
        "all_token_bayes_empirical_prior": "uniform_over_batch_rows",
        "all_token_bayes_conditioning_fields": ["corrupted_tokens", "attention_mask", "loss_mask"],
        "all_token_bayes_model_forward_evaluated": False,
        "all_token_bayes_training_evaluated": False,
        "all_token_bayes_analytic_interpretation": interpretation,
        "all_token_bayes_analytic_interpretation_promoted": bool(interpretation != "inconclusive"),
        "all_token_bayes_interpretation": interpretation,
        "all_token_bayes_interpretation_promoted": bool(interpretation != "inconclusive"),
    }
    for label, record in (("t055", low), ("t095", high)):
        for source_key in (
            "bayes_eos_posterior_at_true_eos_deciles",
            "bayes_eos_margin_at_true_eos_deciles",
            "bayes_entropy_at_true_eos_deciles",
            "bayes_top_token_concentration_deciles",
            "bayes_posterior_support_deciles",
        ):
            summary[f"all_token_bayes_{label}_{source_key}"] = list(record.get(source_key, []))
    return summary, failures


def _strict_mdlm_all_token_bayes_catalog(
    trace_artifact: str,
    catalog_artifact: str,
    records: list[dict],
) -> dict:
    events = sorted({str(record.get("event", "")) for record in records if record.get("event")})
    return {
        "schema_version": 1,
        "telemetry_catalog": "strict_mdlm_all_token_bayes_audit",
        "registration_status": "ok",
        "streams": [
            {
                "name": "strict_mdlm_all_token_bayes_audit",
                "path": trace_artifact,
                "kind": "diagnostics",
                "format": "jsonl",
                "required": True,
                "events": events,
            }
        ],
        "catalog_artifact": catalog_artifact,
    }


def _run_strict_mdlm_all_token_bayes_audit_only(
    config: ExperimentConfig,
    tokenizer,
    noise,
    rank: int,
    world_size: int,
    device,
    is_ddp: bool,
    audit_start: float,
) -> dict:
    artifact_name = "strict_mdlm_all_token_bayes_audit.jsonl"
    artifact_path = os.path.join(config.train.run_dir, "telemetry", artifact_name)
    catalog_artifact = "telemetry/strict_mdlm_all_token_bayes_audit_catalog.json"
    catalog_path = os.path.join(config.train.run_dir, catalog_artifact)
    t_values = _strict_mdlm_all_token_diagnostic_t_values(config)
    static_failures = _strict_mdlm_all_token_bayes_audit_static_failures(config)
    batch = build_synthetic_eos_overfit_canary_batch(
        config.data,
        tokenizer,
        int(config.train.device_batch_size),
        int(config.model.max_seq_len),
        device,
    )
    model_config = _strict_mdlm_all_token_bayes_model_config(config, tokenizer)
    strict_mdlm_config_summary = _strict_mdlm_all_token_bayes_config_summary(
        config,
        model_config,
    )
    provenance_metrics, row_records, provenance_failures = _strict_mdlm_all_token_bayes_batch_provenance(
        config,
        tokenizer,
        batch,
    )
    records: list[dict] = [
        {
            "event": "strict_mdlm_all_token_bayes_audit_config",
            **run_config_summary(config, "train"),
            **_noise_telemetry(config),
            "model": asdict(model_config),
            "tokenizer": _tokenizer_summary(tokenizer),
            "strict_mdlm_config_summary": strict_mdlm_config_summary,
            "all_token_bayes_fixed_t_values": [float(value) for value in t_values],
            "all_token_bayes_corruption_count": int(STRICT_MDLM_ALL_TOKEN_BAYES_AUDIT_CORRUPTIONS),
            **provenance_metrics,
            "all_token_bayes_rows": row_records,
        }
    ]
    fixed_failures: list[str] = []
    for t_index, fixed_t in enumerate(t_values):
        fixed_record, failures = _strict_mdlm_all_token_bayes_fixed_t_record(
            config,
            tokenizer,
            noise,
            batch,
            float(fixed_t),
            int(t_index),
            int(STRICT_MDLM_ALL_TOKEN_BAYES_AUDIT_CORRUPTIONS),
        )
        records.append(fixed_record)
        fixed_failures.extend(failures)

    summary_metrics, summary_failures = _strict_mdlm_all_token_bayes_audit_summary(records, t_values)
    analytic_failures = [*static_failures, *provenance_failures, *fixed_failures, *summary_failures]
    analytic_failed = bool(analytic_failures)
    analytic_status = "failed" if analytic_failed else "ok"
    analytic_error = "; ".join(analytic_failures)
    records.append(
        {
            "event": "strict_mdlm_all_token_bayes_audit_summary",
            "strict_mdlm_all_token_bayes_analytic_status": analytic_status,
            "strict_mdlm_all_token_bayes_analytic_error": analytic_error,
            "strict_mdlm_all_token_bayes_analytic_scope": STRICT_MDLM_ALL_TOKEN_BAYES_ANALYTIC_SCOPE,
            "strict_mdlm_all_token_bayes_audit_status": analytic_status,
            "strict_mdlm_all_token_bayes_audit_error": analytic_error,
            **summary_metrics,
        }
    )

    comparison_table, comparison_metrics, _comparison_issues = _strict_mdlm_all_token_bayes_comparison_table(config)
    records.append(
        {
            "event": "strict_mdlm_all_token_bayes_audit_comparison_table",
            "comparison_run_ids": [str(run_id) for run_id in config.train.comparison_run_ids],
            "comparison_table": comparison_table,
            **comparison_metrics,
        }
    )

    source_metrics = source_provenance_metrics()
    metrics_status = "failed" if analytic_failed else "ok"
    promotion_blocked_reason = "strict_mdlm_all_token_bayes_audit_failed" if analytic_failed else ""
    audit_seconds = float(time.time() - audit_start)
    metrics = {
        **source_metrics,
        "scale": "small",
        "experiment_type": "strict-mdlm-all-token-bayes-audit",
        "causal_ar": False,
        **run_config_summary(config, "train"),
        **_noise_telemetry(config),
        "training_corruption": _training_corruption_label(config),
        "strict_mdlm_config_summary": strict_mdlm_config_summary,
        "strict_mdlm_all_token_bayes_audit_only": True,
        "strict_mdlm_all_token_bayes_audit_enabled": True,
        "strict_mdlm_all_token_bayes_analytic_status": analytic_status,
        "strict_mdlm_all_token_bayes_analytic_error": analytic_error,
        "strict_mdlm_all_token_bayes_analytic_scope": STRICT_MDLM_ALL_TOKEN_BAYES_ANALYTIC_SCOPE,
        "strict_mdlm_all_token_bayes_audit_status": analytic_status,
        "strict_mdlm_all_token_bayes_audit_error": analytic_error,
        "strict_mdlm_all_token_bayes_audit_artifact": f"telemetry/{artifact_name}",
        "strict_mdlm_all_token_bayes_audit_path": artifact_path,
        "strict_mdlm_all_token_bayes_audit_catalog_artifact": catalog_artifact,
        "strict_mdlm_all_token_bayes_audit_catalog_path": catalog_path,
        "strict_mdlm_all_token_bayes_audit_artifact_registration": "artifacts_json",
        "strict_mdlm_all_token_bayes_audit_artifact_registration_status": "ok",
        "strict_mdlm_all_token_diagnostic_telemetry_enabled": True,
        "strict_mdlm_all_token_diagnostic_fixed_t_values": [float(value) for value in t_values],
        "strict_mdlm_static_invariant_status": "failed" if static_failures else "ok",
        "strict_mdlm_static_invariant_error": "; ".join(static_failures),
        "status": metrics_status,
        "promotion_blocked_reason": promotion_blocked_reason,
        "run_completion_note": "analytic audit only; no optimizer step",
        "training_seconds": 0.0,
        "diagnostic_seconds": audit_seconds,
        "total_tokens": 0,
        "total_tokens_M": 0.0,
        "total_valid_tokens_M": 0.0,
        "total_lossable_tokens_M": 0.0,
        "total_content_tokens_M": 0.0,
        "total_content_bytes_M": 0.0,
        "session_tokens_M": 0.0,
        "session_valid_tokens_M": 0.0,
        "session_lossable_tokens_M": 0.0,
        "session_content_tokens_M": 0.0,
        "session_content_bytes_M": 0.0,
        "avg_tok_sec": 0,
        "avg_valid_tok_sec": 0,
        "avg_content_tok_sec": 0,
        "avg_content_byte_sec": 0,
        "world_size": int(world_size),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "mask_token_id": int(tokenizer.get_mask_token_id()),
        "bos_token_id": tokenizer.get_bos_token_id(),
        "eos_token_id": tokenizer.get_eos_token_id(),
        "pad_token_id": tokenizer.get_pad_token_id(),
        "val_nelbo_bpb": 0.0,
        "val_nelbo_bpb_is_audit_placeholder": True,
        **provenance_metrics,
        **comparison_metrics,
        "all_token_bayes_comparison_table": comparison_table,
        **summary_metrics,
    }

    if rank == 0:
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        for record in records:
            append_jsonl(artifact_path, record)
        write_json_atomic(
            catalog_path, _strict_mdlm_all_token_bayes_catalog(f"telemetry/{artifact_name}", catalog_artifact, records)
        )
        resolved_config_path = os.path.join(config.train.run_dir, "resolved_config.json")
        if not os.path.exists(resolved_config_path):
            write_config(resolved_config_path, config)
        _register_run_artifacts(
            config.train.run_dir,
            [
                {
                    "path": artifact_path,
                    "kind": "diagnostics",
                    "name": artifact_name,
                    "phase": "run",
                    "required": True,
                    "metadata": {"stream": "strict_mdlm_all_token_bayes_audit"},
                },
                {
                    "path": catalog_path,
                    "kind": "diagnostics",
                    "name": Path(catalog_artifact).name,
                    "phase": "run",
                    "required": True,
                    "metadata": {"catalog": "strict_mdlm_all_token_bayes_audit"},
                },
                {
                    "path": resolved_config_path,
                    "kind": "artifact",
                    "name": "resolved_config.json",
                    "phase": "run",
                    "required": True,
                    "metadata": {"role": "resolved_config"},
                },
            ],
        )
        metrics_path = os.path.join(config.train.run_dir, "metrics.json")
        write_json_atomic(metrics_path, metrics)
        _register_run_artifacts(
            config.train.run_dir,
            [
                {
                    "path": metrics_path,
                    "kind": "metrics",
                    "name": "metrics.json",
                    "phase": "run",
                    "required": True,
                    "metadata": {},
                }
            ],
        )
    if is_ddp:
        dist.barrier()
    if analytic_failed:
        raise RuntimeError(
            f"strict MDLM all-token Bayes audit failed: {metrics['strict_mdlm_all_token_bayes_audit_error']}"
        )
    return metrics
