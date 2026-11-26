"""Pure paper-derived parameterization mechanics for NanoDLLM."""

from __future__ import annotations

import hashlib
import json
import math

from . import ar_width_mup, muon_residual_depth, nanochat_backbone_config, nanollada_completep
from .config import ExperimentConfig
from .d4_hidden_init_confirmation import (
    is_d4_hidden_init_confirmation,
    validate_d4_hidden_init_confirmation_config,
)
from .model import DiffusionConfig, embedding_head_init_std, residual_output_projection_init_std
from .model_mtp import MTP_ROLES, mtp_initialization
from .model_profiles import NANOLLADA_PROFILE, nanollada_initialization_spec
from .parameterization_calibration import is_d4_root_calibration, validate_d4_root_calibration_config
from .training_features import _uses_strict_mdlm_per_row_t

PARAMETERIZATION_RULE_SET = "nanodllm.ordinary_parameterization.v1"
PARAMETERIZATION_MECHANICS_SCHEMA = "nanodllm.parameterization_mechanics.v1"
PARAMETERIZATION_CALIBRATION_MECHANICS_SCHEMA = "nanodllm.parameterization_mechanics.v2"
MUP_RULE_SET = "nanodllm.mup_tpv_table8_completep_v4_extensions.ratio64.v1"
COMPLETEP_RULE_SET = "nanodllm.completep_alpha1_table1_v4.ratio64.v1"
MECHANICS_EXECUTABLE_PARAMETERIZATION_MODES = ("ordinary", "mup", "completep")


BLOCK_MATRIX_ROLES = frozenset(
    {
        "attention_query",
        "attention_key",
        "attention_value",
        "attention_output",
        "mlp_gate",
        "mlp_up",
        "mlp_output",
    }
)


def initialization_spec(role: str, model_config: DiffusionConfig) -> dict[str, float | str]:
    if role in MTP_ROLES:
        return mtp_initialization(role, model_config.n_embd)
    if model_config.backbone_profile == nanochat_backbone_config.PROFILE:
        return nanochat_backbone_config.initialization(role, model_config.n_embd)
    if model_config.backbone_profile == NANOLLADA_PROFILE:
        return nanollada_initialization_spec(role, model_config.n_embd)
    embedding_base = embedding_head_init_std(int(model_config.n_embd))
    embedding_head = embedding_head_init_std(
        int(model_config.n_embd),
        float(model_config.embedding_head_init_std_multiplier),
    )
    block_base_std = residual_output_projection_init_std(int(model_config.n_layer))
    block_std = (
        float(model_config.parameterization_hidden_init_std)
        if model_config.parameterization_mechanics_sha256 and float(model_config.parameterization_hidden_init_std) > 0
        else block_base_std
    )
    if role in {"token_embedding", "output_head"}:
        return {"rule": "normal", "resolved_scale": embedding_head, "scale_multiplier": 1.0}
    if role in {"attention_output", "mlp_output"} and model_config.residual_output_projection_init == "zero":
        return {"rule": "zeros", "resolved_scale": 0.0, "scale_multiplier": 1.0}
    if role == "block_self_conditioning_down":
        return {"rule": "normal", "resolved_scale": 0.02, "scale_multiplier": 1.0}
    if role in {"token_edit_adapter_up", "token_edit_provenance_embedding", "block_self_conditioning_up"}:
        return {"rule": "zeros", "resolved_scale": 0.0, "scale_multiplier": 1.0}
    if role == "token_edit_adapter_down":
        return {
            "rule": "normal",
            "resolved_scale": block_std,
            "scale_multiplier": block_std / block_base_std,
        }
    if role in BLOCK_MATRIX_ROLES:
        return {
            "rule": "normal",
            "resolved_scale": block_std,
            "scale_multiplier": block_std / block_base_std,
        }
    if role == "final_norm_scale":
        return {"rule": "ones", "resolved_scale": 1.0, "scale_multiplier": 1.0}
    if role in {"final_norm_bias", "length_output_head_bias", "noise_output_projection"}:
        return {"rule": "zeros", "resolved_scale": 0.0, "scale_multiplier": 1.0}
    return {"rule": "normal", "resolved_scale": embedding_base, "scale_multiplier": 1.0}


def nonordinary_initialization_spec(
    role: str,
    model_config: DiffusionConfig,
) -> dict[str, float | str]:
    if ar_width_mup.runtime_active(model_config):
        return nanollada_initialization_spec(role, model_config.n_embd)
    if model_config.parameterization_rule_set in nanollada_completep.RULE_SETS:
        return nanollada_completep.initialization_spec(role, model_config, nanollada_initialization_spec)
    if role in {"token_embedding", "output_head"}:
        return {
            "rule": "normal",
            "resolved_scale": float(model_config.parameterization_embedding_head_init_std),
            "scale_multiplier": 1.0,
        }
    if role in {"attention_output", "mlp_output"} and model_config.residual_output_projection_init == "zero":
        return {"rule": "zeros", "resolved_scale": 0.0, "scale_multiplier": 1.0}
    if role in BLOCK_MATRIX_ROLES:
        base_std = residual_output_projection_init_std(int(model_config.parameterization_base_depth))
        return {
            "rule": "normal",
            "resolved_scale": float(model_config.parameterization_hidden_init_std),
            "scale_multiplier": float(model_config.parameterization_hidden_init_std) / base_std,
        }
    if role == "final_norm_scale":
        return {"rule": "ones", "resolved_scale": 1.0, "scale_multiplier": 1.0}
    if role == "final_norm_bias":
        return {"rule": "zeros", "resolved_scale": 0.0, "scale_multiplier": 1.0}
    raise ValueError(f"nonordinary parameterization has no initialization adaptation for role {role!r}")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_nonordinary_config(config: ExperimentConfig) -> None:
    """Fail closed outside the reviewed NanoDLLM adaptation envelope."""

    mode = config.parameterization.mode
    if mode not in {"mup", "completep"}:
        raise ValueError(f"nonordinary mechanics require mode 'mup' or 'completep', got {mode!r}")
    if config.parameterization.contract_binding != "required":
        raise ValueError("mup and completep require parameterization.contract_binding='required'")
    if config.optim.type != "adamw":
        raise ValueError("mup and completep support only optim.type='adamw'")
    if (int(config.parameterization.base_depth), int(config.parameterization.base_width)) != (4, 256):
        raise ValueError("mup and completep require the frozen D4 base coordinates (4, 256)")
    if int(config.model.depth) * 64 != int(config.model.n_embd):
        raise ValueError("mup and completep require the frozen ratio-64 ladder")
    if int(config.model.n_head) <= 0 or int(config.model.n_embd) % int(config.model.n_head):
        raise ValueError("mup and completep require an integral attention head dimension")
    if int(config.model.n_embd) // int(config.model.n_head) != 128:
        raise ValueError("mup and completep require fixed head_dim=128")
    if int(config.model.mlp_hidden_dim) not in {-1, 4 * int(config.model.n_embd)}:
        raise ValueError("mup and completep require the 4 * n_embd SwiGLU hidden width")
    unsupported = {
        "explicit_attention_scale": float(config.model.attention_scale) != -1.0,
        "local_attention": bool(config.model.local_attention_lower_layers or config.model.local_attention_window),
        "auxiliary_heads": bool(config.model.auxiliary_head_layers),
        "output_factorization": config.model.output_factorization != "flat",
        "causal": bool(config.model.causal),
        "length_head": bool(config.model.length_head),
        "target_length_conditioning": bool(config.model.target_length_conditioning),
        "mask_position_embedding": bool(config.model.mask_position_embedding),
        "non_mdlm_noise": config.noise.type != "mdlm",
        "time_conditioning": bool(config.noise.time_conditioning),
    }
    enabled = sorted(name for name, value in unsupported.items() if value)
    if enabled:
        raise ValueError(f"mup and completep encountered unsupported NanoDLLM adaptations: {enabled}")
    if not _uses_strict_mdlm_per_row_t(config):
        raise ValueError("mup and completep require the frozen strict-MDLM objective and corruption recipe")
    hidden_root = float(config.parameterization.hidden_init_std_multiplier)
    if not math.isfinite(hidden_root) or hidden_root <= 0:
        raise ValueError("parameterization.hidden_init_std_multiplier must be finite and positive")


def resolve_parameterization_mechanics(config: ExperimentConfig) -> dict[str, object]:
    """Resolve model/optimizer multipliers without constructing a model."""

    if ar_width_mup.active(config):
        return ar_width_mup.mechanics(config)
    if muon_residual_depth.active(config):
        muon_residual_depth.validate_config(config, optimizer=False)
        return muon_residual_depth.mechanics()
    if nanollada_completep.is_completep(config):
        return nanollada_completep.mechanics(config)
    mode = config.parameterization.mode
    d4_root_init_mechanics = is_d4_root_calibration(config) or is_d4_hidden_init_confirmation(config)
    if is_d4_root_calibration(config):
        validate_d4_root_calibration_config(config)
    elif is_d4_hidden_init_confirmation(config):
        validate_d4_hidden_init_confirmation_config(config)
    if mode == "ordinary":
        if d4_root_init_mechanics:
            hidden_root_std = residual_output_projection_init_std(4) * float(
                config.parameterization.hidden_init_std_multiplier
            )
            material: dict[str, object] = {
                "schema": PARAMETERIZATION_CALIBRATION_MECHANICS_SCHEMA,
                "mode": "ordinary",
                "rule_set": PARAMETERIZATION_RULE_SET,
                "identity": float(config.parameterization.hidden_init_std_multiplier) == 1.0,
                "base_coordinates": {"depth": 4, "width": 256, "head_dim": 128},
                "target_coordinates": {
                    "depth": 4,
                    "width": 256,
                    "head_dim": 128,
                    "depth_multiplier": 1.0,
                    "width_multiplier": 1.0,
                },
                "initialization": {
                    "hidden_root_std_multiplier": float(config.parameterization.hidden_init_std_multiplier),
                    "hidden_root_std": hidden_root_std,
                    "hidden_std_multiplier": 1.0,
                    "resolved_hidden_std": hidden_root_std,
                    "embedding_head_root_std": embedding_head_init_std(256),
                    "embedding_head_std_multiplier": 1.0,
                    "resolved_embedding_head_std": embedding_head_init_std(256),
                    "zero_residual_outputs_preserved": True,
                    "depth_init_multiplier": 1.0,
                },
                "forward": {
                    "attention_residual_multiplier": 1.0,
                    "mlp_residual_multiplier": 1.0,
                    "unembedding_multiplier": 1.0,
                },
                "optimizer_multipliers": {
                    "hidden_lr": 1.0,
                    "embedding_lr": 1.0,
                    "final_norm_lr": 1.0,
                    "unembedding_lr": 1.0,
                    "hidden_weight_decay": 1.0,
                    "residual_block_epsilon": 1.0,
                    "embedding_final_norm_unembedding_epsilon": 1.0,
                },
            }
            return {**material, "mechanics_sha256": _canonical_sha256(material)}
        return {
            "schema": PARAMETERIZATION_MECHANICS_SCHEMA,
            "mode": "ordinary",
            "rule_set": PARAMETERIZATION_RULE_SET,
            "identity": True,
        }
    _validate_nonordinary_config(config)
    depth = int(config.model.depth)
    width = int(config.model.n_embd)
    base_depth = int(config.parameterization.base_depth)
    base_width = int(config.parameterization.base_width)
    head_dim = width // int(config.model.n_head)
    base_head_dim = 128
    width_multiplier = width / base_width
    depth_multiplier = depth / base_depth
    hidden_root_std = residual_output_projection_init_std(base_depth) * float(
        config.parameterization.hidden_init_std_multiplier
    )
    embedding_head_root_std = embedding_head_init_std(
        base_width,
        float(config.model.embedding_head_init_std_multiplier),
    )
    residual_multiplier = 1.0 if mode == "mup" else depth_multiplier**-1.0
    residual_eps_multiplier = width_multiplier**-1.0 if mode == "mup" else (width_multiplier * depth_multiplier) ** -1.0
    rule_set = MUP_RULE_SET if mode == "mup" else COMPLETEP_RULE_SET
    material: dict[str, object] = {
        "schema": PARAMETERIZATION_MECHANICS_SCHEMA,
        "mode": mode,
        "rule_set": rule_set,
        "source_prescriptions": {
            "mup": "Tensor Programs V arXiv:2203.03466v2 Table 8 easy implementation",
            "completep": "CompleteP arXiv:2505.01618v4 Table 1 alpha=1 with corrected epsilon rows",
            "official_code_repository": "EleutherAI/nanoGPT-mup",
            "official_code_commit": "88458c3f063b5c3c573f87435dc91f680174f0da",
            "official_code_source_path": "model.py",
            "official_code_resolution": "per-head 1/d_head attention and final-LN first/last-layer epsilon mapping",
        },
        "base_coordinates": {"depth": base_depth, "width": base_width, "head_dim": base_head_dim},
        "target_coordinates": {
            "depth": depth,
            "width": width,
            "head_dim": head_dim,
            "depth_multiplier": depth_multiplier,
            "width_multiplier": width_multiplier,
        },
        "initialization": {
            "hidden_root_std_multiplier": float(config.parameterization.hidden_init_std_multiplier),
            "hidden_root_std": hidden_root_std,
            "hidden_std_multiplier": width_multiplier**-0.5,
            "resolved_hidden_std": hidden_root_std * width_multiplier**-0.5,
            "embedding_head_root_std": embedding_head_root_std,
            "embedding_head_std_multiplier": 1.0,
            "resolved_embedding_head_std": embedding_head_root_std,
            "zero_residual_outputs_preserved": config.model.residual_output_projection_init == "zero",
            "depth_init_multiplier": 1.0,
        },
        "forward": {
            "attention_scale": math.sqrt(base_head_dim) / head_dim,
            "attention_scale_rule": "sqrt(base_head_dim)/target_head_dim_per_head_base_compatible",
            "attention_source_boundary": (
                "TPV Table 8 per-head mapping; CompleteP paper QK/N notation resolved to official-code per-head rule"
            ),
            "attention_residual_multiplier": residual_multiplier,
            "mlp_residual_multiplier": residual_multiplier,
            "unembedding_multiplier": width_multiplier**-1.0,
        },
        "optimizer_multipliers": {
            "hidden_lr": width_multiplier**-1.0,
            "embedding_lr": 1.0,
            "final_norm_lr": 1.0,
            "unembedding_lr": 1.0,
            "hidden_weight_decay": width_multiplier,
            "residual_block_epsilon": residual_eps_multiplier,
            "embedding_final_norm_unembedding_epsilon": width_multiplier**-1.0,
        },
        "nanodllm_adaptations": {
            "objective": "masked_diffusion_mdlm_not_autoregressive",
            "position_encoding": "rope_not_alibi",
            "activation": "swiglu_not_relu_squared",
            "block_norm": "parameter_free_rms_norm",
            "final_norm": "trainable_layer_norm",
            "embedding_head": "untied",
            "backbone_bias": "none",
            "head_dim": "fixed_128_ratio64_ladder",
            "residual_output_initialization": str(config.model.residual_output_projection_init),
        },
    }
    return {**material, "mechanics_sha256": _canonical_sha256(material)}


def parameterization_model_config_kwargs(config: ExperimentConfig) -> dict[str, object]:
    """Return explicit ``DiffusionConfig`` fields for nonordinary mechanics."""

    mechanics = resolve_parameterization_mechanics(config)
    if muon_residual_depth.active(config):
        return muon_residual_depth.model_kwargs()
    if config.parameterization.mode == "ordinary":
        if not (is_d4_root_calibration(config) or is_d4_hidden_init_confirmation(config)):
            return {}
        initialization = mechanics["initialization"]
        forward = mechanics["forward"]
        assert isinstance(initialization, dict) and isinstance(forward, dict)
        return {
            "parameterization_rule_set": str(mechanics["rule_set"]),
            "parameterization_base_depth": int(config.parameterization.base_depth),
            "parameterization_base_width": int(config.parameterization.base_width),
            "parameterization_mechanics_sha256": str(mechanics["mechanics_sha256"]),
            "parameterization_hidden_init_std": float(initialization["resolved_hidden_std"]),
            "parameterization_embedding_head_init_std": float(initialization["resolved_embedding_head_std"]),
            "parameterization_residual_multiplier": float(forward["attention_residual_multiplier"]),
            "parameterization_output_multiplier": float(forward["unembedding_multiplier"]),
        }
    initialization = mechanics["initialization"]
    forward = mechanics["forward"]
    assert isinstance(initialization, dict) and isinstance(forward, dict)
    return {
        "parameterization_mode": str(mechanics["mode"]),
        "parameterization_rule_set": str(mechanics["rule_set"]),
        "parameterization_base_depth": int(config.parameterization.base_depth),
        "parameterization_base_width": int(config.parameterization.base_width),
        "parameterization_mechanics_sha256": str(mechanics["mechanics_sha256"]),
        "parameterization_hidden_init_std": float(initialization["resolved_hidden_std"]),
        "parameterization_embedding_head_init_std": float(initialization["resolved_embedding_head_std"]),
        "parameterization_residual_multiplier": float(forward["attention_residual_multiplier"]),
        "parameterization_output_multiplier": float(forward["unembedding_multiplier"]),
        "parameterization_attention_scale": float(forward["attention_scale"]),
        "parameterization_attention_scale_rule": str(forward["attention_scale_rule"]),
    }
