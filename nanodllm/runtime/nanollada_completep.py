"""Base-compatible Complete(d)P alpha=1 for plain squared-ReLU AR and MDLM.

Only width/depth transfer is implemented here. Batch/duration choices belong
to the experiment, not to hidden optimizer transformations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .model import DiffusionConfig

AR_SCOPE = "ar_completep_transfer"
AR_RULE_SET = "nanodllm.ar_completedp_alpha1_width_depth.v1"
MDLM_SCOPE = "mdlm_completep_transfer"
MDLM_RULE_SET = "nanodllm.mdlm_completedp_alpha1_width_depth.v1"
RULE_SETS = frozenset({AR_RULE_SET, MDLM_RULE_SET})


def is_completep(config: ExperimentConfig) -> bool:
    return config.parameterization.execution_scope in {AR_SCOPE, MDLM_SCOPE}


def validate_config(config: ExperimentConfig) -> None:
    model, parameterization = config.model, config.parameterization
    causal = parameterization.execution_scope == AR_SCOPE
    if not (
        is_completep(config)
        and parameterization.mode == "completep"
        and parameterization.contract_binding == "required"
        and (parameterization.base_depth, parameterization.base_width) == (4, 256)
        and parameterization.hidden_init_std_multiplier == 1.0
        and model.backbone_profile == "nanollada_v1"
        and model.causal is causal
        and (model.depth in {4, 6, 8, 10, 12, 16, 20} if causal else model.depth >= 4)
        and model.n_embd == model.depth * 64
        and model.n_head == model.depth
        and model.mlp_hidden_dim in {-1, 4 * model.n_embd}
        and model.attention_scale == -1
        and config.optim.type == "adamw"
        and config.noise.type == "mdlm"
        and not config.noise.time_conditioning
    ):
        raise ValueError("CompleteP requires its matching AR/MDLM squared-ReLU, ratio64, head64, AdamW scope")
    if (
        any(
            (
                model.local_attention_lower_layers,
                model.local_attention_window,
                model.auxiliary_head_layers,
                model.length_head,
                model.target_length_conditioning,
                model.mask_position_embedding,
                model.bd3lm_block_size,
                model.block_causal_pilot_size,
            )
        )
        or model.output_factorization != "flat"
    ):
        raise ValueError("squared-ReLU CompleteP does not support optional model roles")
    if not causal:
        from .training_features import _uses_strict_mdlm_per_row_t

        if (
            not _uses_strict_mdlm_per_row_t(config)
            or not config.noise.schedule_weight
            or config.train.mdlm_auxiliary.mode != "none"
            or config.train.label_smoothing_epsilon
            or config.train.strict_mdlm_uniform_schedule_weight
            or config.train.strict_mdlm_eos_loss_mass_balance
            or config.train.strict_mdlm_t_curriculum_enabled
        ):
            raise ValueError(
                "MDLM CompleteP requires basic schedule-weighted per-row MDLM, without auxiliary objectives"
            )


def mechanics(config: ExperimentConfig) -> dict:
    validate_config(config)
    width, depth = config.model.n_embd, config.model.depth
    m, ell = width / 256, depth / 4
    material = {
        "schema": "nanodllm.parameterization_mechanics.v1",
        "mode": "completep",
        "rule_set": AR_RULE_SET if config.model.causal else MDLM_RULE_SET,
        "source_prescription": "arXiv:2512.22382v1 Table 1, alpha=1, width/depth columns only",
        "base_coordinates": {"depth": 4, "width": 256, "head_dim": 64},
        "target_coordinates": {
            "depth": depth,
            "width": width,
            "head_dim": 64,
            "depth_multiplier": ell,
            "width_multiplier": m,
        },
        "initialization": {
            "resolved_hidden_std": width**-0.5,
            "resolved_embedding_head_std": 1.0,
            "resolved_output_head_std": 0.001 / m,
            "hidden_distribution": "uniform",
            "zero_residual_outputs_preserved": True,
        },
        "forward": {
            "attention_scale": 64**-0.5,
            "attention_scale_rule": "fixed_head64_base_compatible",
            "attention_residual_multiplier": 1 / ell,
            "mlp_residual_multiplier": 1 / ell,
            "unembedding_multiplier": 1.0,
        },
        "optimizer_multipliers": {
            "hidden_lr": 1 / m,
            "unembedding_lr": 1 / m,
            "hidden_weight_decay": m,
            "unembedding_weight_decay": m,
            "residual_block_epsilon": 1 / (m * ell),
            "embedding_final_norm_unembedding_epsilon": 1 / m,
        },
        "adaptations": {
            "qk_norm": "parameter_free; no QK norm weights to rescale",
            "initialization": "preserve AR base: uniform hidden, embedding std1, output std0.001, zero residual outputs",
            "weight_decay": "all matrices including input embedding; output head scales with width",
            "duration_batch_optimizer_scaling": "not applied",
        },
    }
    if not config.model.causal:
        material["adaptations"].update(
            initialization="preserve ordinary squared-ReLU base: uniform hidden, embedding std1, output std0.001, zero residual outputs",
            duration_batch_optimizer_scaling="explicit root config only; no implicit scaling",
        )
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**material, "mechanics_sha256": digest}


def validate_runtime(config: DiffusionConfig) -> None:
    causal = config.parameterization_rule_set == AR_RULE_SET
    if not (
        config.parameterization_rule_set in RULE_SETS
        and config.parameterization_mode == "completep"
        and config.backbone_profile == "nanollada_v1"
        and config.causal is causal
        and (config.n_layer in {4, 6, 8, 10, 12, 16, 20} if causal else config.n_layer >= 4)
        and config.n_embd == 64 * config.n_layer
        and config.n_head == config.n_layer
        and (config.parameterization_base_depth, config.parameterization_base_width) == (4, 256)
        and config.parameterization_attention_scale_rule == "fixed_head64_base_compatible"
        and config.attention_scale == -1
        and config.residual_output_projection_init == "zero"
        and not config.time_conditioning
    ):
        raise ValueError("squared-ReLU CompleteP runtime shape/rule differs")
    if (
        config.mlp_hidden_dim not in {-1, 4 * config.n_embd}
        or config.output_factorization != "flat"
        or config.local_attention_lower_layers
        or config.local_attention_window
        or config.auxiliary_head_layers
        or config.bd3lm_block_size
        or config.bd3lm_self_conditioning_dim
        or config.length_head
        or config.target_length_conditioning
        or config.mask_position_embedding
        or config.token_edit_adapter_bottleneck_dim
        or config.token_edit_adapter_top_layers
        or config.token_edit_provenance_embedding
        or config.token_edit_provenance_confidence_bins
    ):
        raise ValueError("squared-ReLU CompleteP runtime has unsupported model roles")
    expected = {
        "parameterization_hidden_init_std": config.n_embd**-0.5,
        "parameterization_embedding_head_init_std": 1.0,
        "parameterization_residual_multiplier": 4 / config.n_layer,
        "parameterization_output_multiplier": 1.0,
        "parameterization_attention_scale": 64**-0.5,
    }
    if any(not math.isclose(getattr(config, key), value, rel_tol=1e-12) for key, value in expected.items()):
        raise ValueError("squared-ReLU CompleteP runtime multipliers differ")
    digest = config.parameterization_mechanics_sha256
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("squared-ReLU CompleteP requires its mechanics digest")


def head_std(config: DiffusionConfig) -> float:
    return 0.001 * 256 / config.n_embd if config.parameterization_rule_set in RULE_SETS else 0.001


def initialization_spec(
    role: str,
    config: DiffusionConfig,
    ordinary_spec: Callable[[str, int], dict[str, float | str]],
) -> dict[str, float | str]:
    value = ordinary_spec(role, config.n_embd)
    if role == "output_head":
        value.update(resolved_scale=head_std(config), scale_multiplier=256 / config.n_embd)
    return value


def finish_contract(contract: dict, *, scope: str) -> dict:
    if scope not in {AR_SCOPE, MDLM_SCOPE}:
        raise ValueError("unknown squared-ReLU CompleteP execution scope")
    causal = scope == AR_SCOPE
    material = {key: value for key, value in contract.items() if key != "contract_sha256"}
    material.update(
        adaptation_status="implemented_ar_width_depth_screen" if causal else "implemented_mdlm_width_depth_transfer",
        scientific_training_authorized=True,
        compiled_bf16_qualification_status=(
            "CPU integration tested; CUDA memory/throughput observed in first scheduled cell"
            if causal
            else "CPU integration tested; CUDA memory/throughput not yet tested"
        ),
        execution_scope=scope,
    )
    material["model_conventions"].update(
        backbone_profile="nanollada_v1",
        input_norm="parameter_free_rms_norm",
        final_norm="parameter_free_rms_norm",
        mlp="squared_relu",
        objective="causal_ar" if causal else "mdlm",
    )
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**material, "contract_sha256": digest}


def validate_model_initialization(checkpoint: Mapping, model) -> None:
    """Allow attention conversion, never reinterpret a parent's scaled weights.

    Only the new MDLM target is constrained here; historical initialization routes
    are unchanged. The normal loader separately reconstructs the source contract.
    """

    from .model import DiffusionConfig

    target = model.config
    if getattr(target, "parameterization_rule_set", None) != MDLM_RULE_SET:
        return
    validate_runtime(target)
    raw_source = checkpoint.get("config")
    if not isinstance(raw_source, Mapping):
        raise ValueError("MDLM CompleteP initialization requires a resolved source model config")
    source = DiffusionConfig(**raw_source)
    if source.parameterization_rule_set in RULE_SETS:
        validate_runtime(source)
    elif not (
        source.parameterization_mode == "ordinary"
        and source.backbone_profile == "nanollada_v1"
        and (source.n_layer, source.n_embd, source.n_head) == (4, 256, 4)
        and not source.parameterization_mechanics_sha256
    ):
        raise ValueError("MDLM CompleteP initialization requires matching CompleteP or ordinary D4 base weights")
    # Rule identity and topology differ deliberately. All actual forward/shape
    # conventions must agree; ordinary D4 has no resolved mechanics identity.
    identity_fields = {
        "causal",
        "parameterization_mode",
        "parameterization_rule_set",
        "parameterization_mechanics_sha256",
    }
    if source.parameterization_mode == "ordinary":
        identity_fields.update(field.name for field in fields(source) if field.name.startswith("parameterization_"))
        if source.parameterization_residual_multiplier != 1.0 or source.parameterization_output_multiplier != 1.0:
            raise ValueError("ordinary D4 initialization must have identity forward multipliers")
    mismatches = [
        field.name
        for field in fields(source)
        if field.name not in identity_fields and getattr(source, field.name) != getattr(target, field.name)
    ]
    train_config = checkpoint.get("train_config")
    source_model = train_config.get("model") if isinstance(train_config, Mapping) else None
    if not isinstance(source_model, Mapping) or source_model.get("rope_base") != model.rope_base:
        mismatches.append("rope_base")
    if mismatches:
        raise ValueError(f"MDLM CompleteP parent forward/shape conventions differ: {mismatches}")
