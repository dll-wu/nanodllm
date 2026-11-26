"""Fixed-depth AR width mechanics; campaign selection and budgets live elsewhere.

The scope deliberately permits short/small-vocabulary mechanical fixtures.
Its campaign adapter must additionally admit the frozen data, grid and budget.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .model import DiffusionConfig

SCOPE = "ar_width_transfer_mup_v1"
RULE_SET = "nanodllm.ar_fixed_depth_audited_mup.v1"
ATTENTION_RULE = "sqrt(base_head_dim32)/target_head_dim_fixed_four_heads"
NORM_EPSILON = 1e-5
WIDTHS = (128, 256, 512)


def active(config: ExperimentConfig) -> bool:
    return config.parameterization.execution_scope == SCOPE


def runtime_active(config: object) -> bool:
    return getattr(config, "parameterization_rule_set", "") == RULE_SET


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _optional_roles(model: object) -> bool:
    return (
        any(
            getattr(model, name, False)
            for name in (
                "local_attention_lower_layers",
                "local_attention_window",
                "auxiliary_head_layers",
                "length_head",
                "target_length_conditioning",
                "mask_position_embedding",
                "bd3lm_block_size",
                "bd3lm_self_conditioning_dim",
                "block_causal_pilot_size",
                "token_edit_adapter_bottleneck_dim",
                "token_edit_adapter_top_layers",
                "token_edit_provenance_embedding",
                "token_edit_provenance_confidence_bins",
                "recurrent_core_start",
                "recurrent_core_length",
            )
        )
        or getattr(model, "mtp_mode", "none") != "none"
        or getattr(model, "recurrent_iterations", 1) != 1
    )


def validate_config(config: ExperimentConfig) -> None:
    model, p = config.model, config.parameterization
    if not (
        active(config)
        and p.mode == "mup"
        and p.contract_binding == "required"
        and (p.base_depth, p.base_width) == (4, 128)
        and p.hidden_init_std_multiplier == 1.0
        and model.backbone_profile == "nanollada_v1"
        and model.causal
        and model.depth == 4
        and model.n_embd in WIDTHS
        and model.n_head == 4
        and model.mlp_hidden_dim in (-1, 4 * model.n_embd)
        and model.attention_scale == -1
        and model.residual_output_projection_init == "zero"
        and model.embedding_head_init_std_multiplier == 1.0
        and model.output_factorization == "flat"
        and model.rope_base == 10000
        and config.optim.type == "adamw"
        and config.noise.type == "mdlm"
        and not config.noise.time_conditioning
        and not _optional_roles(model)
    ):
        raise ValueError("AR width muP requires its plain causal depth4/head4, widths128/256/512, base128 AdamW scope")
    if config.train.scale_lr_by_batch or config.optim.grad_clip != 0 or config.runtime.compile:
        raise ValueError("AR width muP disables batch LR scaling, gradient clipping and compilation")
    if (config.optim.adam_beta1, config.optim.adam_beta2) != (0.9, 0.999):
        raise ValueError("AR width muP requires Adam betas (0.9, 0.999)")
    if config.runtime.dtype not in {"bfloat16", "float32", "float64"} or config.runtime.allow_tf32:
        raise ValueError("AR width muP requires BF16/FP32 or FP64 reference arithmetic with TF32 disabled")
    if config.runtime.cudnn_allow_tf32 or config.runtime.float32_matmul_precision not in {"", "highest"}:
        raise ValueError("AR width muP requires full-precision FP32 matrix operations")
    if config.train.label_smoothing_epsilon or config.train.mdlm_auxiliary.mode != "none":
        raise ValueError("AR width muP requires hard-label causal loss without auxiliary objectives")


def mechanics(config: ExperimentConfig) -> dict:
    validate_config(config)
    width = config.model.n_embd
    m = width / 128
    material = {
        "schema": "nanodllm.parameterization_mechanics.v1",
        "mode": "mup",
        "rule_set": RULE_SET,
        "source_prescription": "frozen ar_width_transfer.contract.json; finite AdamW raw-coordinate audit",
        "base_coordinates": {"depth": 4, "width": 128, "head_dim": 32},
        "target_coordinates": {
            "depth": 4,
            "width": width,
            "head_dim": width // 4,
            "depth_multiplier": 1.0,
            "width_multiplier": m,
        },
        "initialization": {
            "resolved_hidden_std": width**-0.5,
            "resolved_embedding_head_std": 1.0,
            "resolved_output_head_std": 0.001,
            "hidden_distribution": "uniform",
            "zero_residual_outputs_preserved": True,
            "embedding_master_dtype": "float32",
        },
        "forward": {
            "attention_scale": math.sqrt(32) / (width / 4),
            "attention_scale_rule": ATTENTION_RULE,
            "attention_residual_multiplier": 1.0,
            "mlp_residual_multiplier": 1.0,
            "unembedding_multiplier": 1 / m,
            "norm_epsilon": NORM_EPSILON,
            "norm_arithmetic": "fp32_reduce_cast_back_or_entirely_fp64_reference",
            "attention_backend": "sdpa_math_no_reduced_precision",
        },
        "optimizer_multipliers": {
            "hidden_lr": 1 / m,
            "hidden_weight_decay": m,
            "residual_block_epsilon": 1 / m,
            "embedding_final_norm_unembedding_epsilon": 1 / m,
        },
    }
    return {**material, "mechanics_sha256": _digest(material)}


def validate_runtime(config: DiffusionConfig) -> None:
    if not (
        runtime_active(config)
        and config.parameterization_mode == "mup"
        and config.backbone_profile == "nanollada_v1"
        and config.causal
        and config.n_layer == 4
        and config.n_head == 4
        and config.n_embd in WIDTHS
        and (config.parameterization_base_depth, config.parameterization_base_width) == (4, 128)
        and config.mlp_hidden_dim in (-1, 4 * config.n_embd)
        and config.attention_scale == -1
        and config.parameterization_attention_scale_rule == ATTENTION_RULE
        and config.residual_output_projection_init == "zero"
        and config.embedding_head_init_std_multiplier == 1.0
        and config.output_factorization == "flat"
        and not config.time_conditioning
        and not _optional_roles(config)
    ):
        raise ValueError("AR width muP runtime shape or rule differs from the scoped mechanics")
    expected = {
        "parameterization_hidden_init_std": config.n_embd**-0.5,
        "parameterization_embedding_head_init_std": 1.0,
        "parameterization_residual_multiplier": 1.0,
        "parameterization_output_multiplier": 128 / config.n_embd,
        "parameterization_attention_scale": math.sqrt(32) / (config.n_embd / 4),
    }
    if any(not math.isclose(getattr(config, name), value, rel_tol=1e-12) for name, value in expected.items()):
        raise ValueError("AR width muP runtime multipliers differ")
    digest = config.parameterization_mechanics_sha256
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("AR width muP requires its mechanics digest")


def finish_contract(contract: dict) -> dict:
    material = {key: value for key, value in contract.items() if key != "contract_sha256"}
    material.update(
        execution_scope=SCOPE,
        adaptation_status="implemented_fixed_depth_ar_width_mechanics",
        scientific_training_authorized=False,
        compiled_bf16_qualification_status="separate_outcome_blind_qualification_required",
    )
    material["model_conventions"].update(
        backbone_profile="nanollada_v1",
        input_norm="parameter_free_rms_norm",
        final_norm="parameter_free_rms_norm",
        norm_epsilon=NORM_EPSILON,
        norm_arithmetic="fp32_reduce_cast_back_or_entirely_fp64_reference",
        mlp="squared_relu",
        objective="causal_ar",
        embedding_master_dtype="float32",
        optimizer_kernel="adamw_nonfused_nonforeach",
        attention_backend="sdpa_math_no_reduced_precision",
    )
    return {**material, "contract_sha256": _digest(material)}


def validate_model(model: object) -> None:
    """Reject a storage or norm substitution before constructing Adam state."""
    validate_runtime(model.config)
    expected_dtype = torch.float64 if model.compute_dtype == torch.float64 else torch.float32
    if any(parameter.dtype != expected_dtype for parameter in model.parameters()):
        raise ValueError("AR width muP requires FP32 master parameters, or FP64 for its reference fixture")
    if model.norm_epsilon != NORM_EPSILON or model.final_norm.epsilon != NORM_EPSILON:
        raise ValueError("AR width muP input/final normalization differs")
    if any(
        block.norm_epsilon != NORM_EPSILON or block.attn.norm_epsilon != NORM_EPSILON or not block.attn.ar_width_scope
        for block in model.blocks
    ):
        raise ValueError("AR width muP block/QK normalization or attention backend differs")


def configure_backend() -> None:
    """Pin process-level switches; scoped attention also selects MATH explicitly."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)


def assert_backend() -> None:
    if torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed():
        raise RuntimeError("AR width muP forbids reduced-precision math SDPA reductions")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("AR width muP forbids TF32 backend fallback")
