"""Model-boundary validation and provenance for resolved parameterization."""

from __future__ import annotations

import math
import re
from typing import Protocol

from . import ar_width_mup, muon_residual_depth, nanollada_completep


class ParameterizationRuntimeConfig(Protocol):
    n_layer: int
    n_head: int
    n_embd: int
    mlp_hidden_dim: int
    attention_scale: float
    residual_output_projection_init: str
    parameterization_mode: str
    parameterization_rule_set: str
    parameterization_base_depth: int
    parameterization_base_width: int
    parameterization_mechanics_sha256: str
    parameterization_hidden_init_std: float
    parameterization_embedding_head_init_std: float
    parameterization_residual_multiplier: float
    parameterization_output_multiplier: float
    parameterization_attention_scale: float
    parameterization_attention_scale_rule: str
    output_factorization: str
    causal: bool
    auxiliary_head_layers: tuple[int, ...]
    time_conditioning: bool
    length_head: bool
    target_length_conditioning: bool
    mask_position_embedding: bool
    local_attention_lower_layers: int
    local_attention_window: int


def validate_parameterization_runtime(config: ParameterizationRuntimeConfig) -> None:
    """Reject incomplete or unsupported resolved mechanics at model creation."""

    if ar_width_mup.runtime_active(config):
        ar_width_mup.validate_runtime(config)
        return
    if config.parameterization_rule_set == muon_residual_depth.RULE:
        muon_residual_depth.validate_runtime(config)
        return
    if config.parameterization_rule_set in nanollada_completep.RULE_SETS:
        nanollada_completep.validate_runtime(config)
        return
    if config.parameterization_mode not in {"ordinary", "mup", "completep"}:
        raise ValueError(f"unknown parameterization mode {config.parameterization_mode!r}")
    if config.parameterization_mode == "ordinary":
        if not config.parameterization_mechanics_sha256:
            return
        if config.parameterization_rule_set != "nanodllm.ordinary_parameterization.v1":
            raise ValueError("scoped ordinary parameterization rule set is missing or inconsistent")
        if (int(config.parameterization_base_depth), int(config.parameterization_base_width)) != (4, 256):
            raise ValueError("scoped ordinary parameterization requires the frozen D4 base coordinates")
        if (int(config.n_layer), int(config.n_embd), int(config.n_head), int(config.mlp_hidden_dim)) != (
            4,
            256,
            2,
            1024,
        ):
            raise ValueError("scoped ordinary parameterization requires the exact D4 calibration shape")
        if re.fullmatch(r"[0-9a-f]{64}", config.parameterization_mechanics_sha256) is None:
            raise ValueError("scoped ordinary parameterization requires a resolved mechanics SHA-256")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in (
                config.parameterization_hidden_init_std,
                config.parameterization_embedding_head_init_std,
            )
        ):
            raise ValueError("scoped ordinary parameterization requires finite positive initialization scales")
        base_hidden_std = 0.02 / math.sqrt(8)
        if (
            float(config.parameterization_hidden_init_std)
            not in {base_hidden_std * multiplier for multiplier in (0.5, 1.0, 2.0)}
            or float(config.parameterization_embedding_head_init_std) != 0.025
        ):
            raise ValueError("scoped ordinary parameterization initialization scale is outside the frozen grid")
        if not math.isclose(float(config.parameterization_residual_multiplier), 1.0) or not math.isclose(
            float(config.parameterization_output_multiplier), 1.0
        ):
            raise ValueError("scoped ordinary parameterization must preserve identity forward multipliers")
        if (
            float(config.attention_scale) != -1.0
            or float(config.parameterization_attention_scale) != -1.0
            or config.parameterization_attention_scale_rule
        ):
            raise ValueError("scoped ordinary parameterization must preserve ordinary attention-scale resolution")
        if (
            config.residual_output_projection_init != "zero"
            or config.output_factorization != "flat"
            or config.causal
            or config.auxiliary_head_layers
            or config.time_conditioning
            or config.length_head
            or config.target_length_conditioning
            or config.mask_position_embedding
            or config.local_attention_lower_layers
            or config.local_attention_window
        ):
            raise ValueError("scoped ordinary parameterization encountered an unsupported calibration model role")
        return
    resolved_scalars = (
        config.parameterization_hidden_init_std,
        config.parameterization_embedding_head_init_std,
        config.parameterization_residual_multiplier,
        config.parameterization_output_multiplier,
        config.parameterization_attention_scale,
    )
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in resolved_scalars):
        raise ValueError("nonordinary parameterization requires finite positive resolved model mechanics")
    if len(config.parameterization_mechanics_sha256) != 64:
        raise ValueError("nonordinary parameterization requires a resolved mechanics SHA-256")
    expected_rule_set = {
        "mup": "nanodllm.mup_tpv_table8_completep_v4_extensions.ratio64.v1",
        "completep": "nanodllm.completep_alpha1_table1_v4.ratio64.v1",
    }[config.parameterization_mode]
    if config.parameterization_rule_set != expected_rule_set:
        raise ValueError("nonordinary parameterization rule set is missing or inconsistent")
    if (int(config.parameterization_base_depth), int(config.parameterization_base_width)) != (4, 256):
        raise ValueError("nonordinary parameterization requires the frozen D4 base coordinates")
    if int(config.n_layer) <= 0 or int(config.n_embd) != 64 * int(config.n_layer):
        raise ValueError("nonordinary parameterization requires the ratio-64 depth/width ladder")
    if int(config.n_head) <= 0 or int(config.n_embd) % int(config.n_head):
        raise ValueError("nonordinary parameterization requires an integral attention head dimension")
    head_dim = int(config.n_embd) // int(config.n_head)
    if head_dim != 128:
        raise ValueError("nonordinary parameterization requires fixed head_dim=128")
    if int(config.mlp_hidden_dim) not in {-1, 4 * int(config.n_embd)}:
        raise ValueError("nonordinary parameterization requires the 4x SwiGLU hidden width")
    expected_attention_scale = math.sqrt(128) / head_dim
    if not math.isclose(float(config.parameterization_attention_scale), expected_attention_scale):
        raise ValueError("nonordinary parameterization attention scale is inconsistent with the per-head rule")
    if config.parameterization_attention_scale_rule != "sqrt(base_head_dim)/target_head_dim_per_head_base_compatible":
        raise ValueError("nonordinary parameterization attention-scale rule is missing or inconsistent")
    expected_residual_multiplier = 1.0 if config.parameterization_mode == "mup" else 4 / int(config.n_layer)
    if not math.isclose(float(config.parameterization_residual_multiplier), expected_residual_multiplier):
        raise ValueError("nonordinary parameterization residual multiplier is inconsistent with the depth rule")
    if not math.isclose(float(config.parameterization_output_multiplier), 256 / int(config.n_embd)):
        raise ValueError("nonordinary parameterization output multiplier is inconsistent with the width rule")
    if (
        config.output_factorization != "flat"
        or config.causal
        or config.auxiliary_head_layers
        or config.time_conditioning
        or config.length_head
        or config.target_length_conditioning
        or config.mask_position_embedding
        or config.local_attention_lower_layers
        or config.local_attention_window
    ):
        raise ValueError("nonordinary parameterization encountered an unsupported optional model role")


def parameterization_fingerprint(config: ParameterizationRuntimeConfig) -> dict[str, object] | None:
    """Return architecture-bound mechanics for nonordinary or scoped ordinary runs."""

    if config.parameterization_mode == "ordinary" and not config.parameterization_mechanics_sha256:
        return None
    return {
        "mode": str(config.parameterization_mode),
        "rule_set": str(config.parameterization_rule_set),
        "base_depth": int(config.parameterization_base_depth),
        "base_width": int(config.parameterization_base_width),
        "mechanics_sha256": str(config.parameterization_mechanics_sha256),
        "hidden_init_std": float(config.parameterization_hidden_init_std),
        "embedding_head_init_std": float(config.parameterization_embedding_head_init_std),
        "residual_multiplier": float(config.parameterization_residual_multiplier),
        "output_multiplier": float(config.parameterization_output_multiplier),
        "attention_scale": float(config.parameterization_attention_scale),
        "attention_scale_rule": str(config.parameterization_attention_scale_rule),
    }
