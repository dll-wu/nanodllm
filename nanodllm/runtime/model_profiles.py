"""Opt-in backbone bundles; the historical NanoDLLM path remains unchanged."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from . import ar_width_mup, muon_residual_depth, nanochat_backbone_config, nanollada_completep
from .model_attention import Linear, rms_norm

if TYPE_CHECKING:
    from .config import ExperimentConfig, ModelConfig
    from .model import DiffusionConfig

NANOLLADA_PROFILE = "nanollada_v1"


class ParameterFreeRMSNorm(nn.Module):
    def __init__(self, epsilon: float | None = None):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, epsilon=self.epsilon)


class SquaredReLUMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        # Reuse the existing semantic up/output roles for accounting and AdamW.
        self.up = Linear(width, 4 * width, bias=False)
        self.down = Linear(4 * width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)).square())


def validate_backbone_profile(config: ModelConfig | DiffusionConfig) -> None:
    if config.backbone_profile == "nanodllm":
        return
    if config.backbone_profile not in {NANOLLADA_PROFILE, nanochat_backbone_config.PROFILE}:
        raise ValueError("model.backbone_profile must be 'nanodllm', 'nanollada_v1' or 'nanochat_block_v1'")
    if config.mlp_hidden_dim not in (-1, 4 * config.n_embd):
        raise ValueError("nanollada_v1 requires a 4x squared-ReLU MLP")
    if config.residual_output_projection_init != "zero" or config.embedding_head_init_std_multiplier != 1.0:
        raise ValueError("nanollada_v1 owns its initialization; legacy initialization overrides are unsupported")
    if (
        config.output_factorization != "flat"
        or config.local_attention_lower_layers
        or config.local_attention_window
        or config.auxiliary_head_layers
        or config.bd3lm_self_conditioning_dim
        or config.length_head
        or config.target_length_conditioning
        or config.mask_position_embedding
        or config.token_edit_adapter_top_layers
        or config.token_edit_adapter_bottleneck_dim
        or config.token_edit_provenance_embedding
        or config.token_edit_provenance_confidence_bins
    ):
        raise ValueError("nanollada_v1 supports the plain flat-output backbone without auxiliary model extensions")


def validate_nanollada_experiment(config: ExperimentConfig) -> None:
    validate_backbone_profile(config.model)
    if config.model.backbone_profile == nanochat_backbone_config.PROFILE:
        nanochat_backbone_config.validate_experiment(config)
        return
    if config.model.backbone_profile != NANOLLADA_PROFILE:
        return
    if ar_width_mup.active(config):
        ar_width_mup.validate_config(config)
        return
    if muon_residual_depth.active(config):
        # The optimizer validator also uses an AdamW projection of this profile.
        muon_residual_depth.validate_config(config, optimizer=False)
        return
    if nanollada_completep.is_completep(config):
        nanollada_completep.validate_config(config)
        return
    if (
        config.parameterization.mode != "ordinary"
        or config.parameterization.execution_scope != "ordinary"
        or config.parameterization.hidden_init_std_multiplier != 1.0
        or config.optim.type != "adamw"
    ):
        raise ValueError("nanollada_v1 requires ordinary AdamW without scoped hidden-init mechanics")
    block_m2t = (
        config.noise.type == "bd3lm"
        and config.model.bd3lm_clean_stream_mode == "causal"
        and config.train.bd3lm_uniform_t2t_batch_fraction == 0
    )
    if bool(config.model.bd3lm_block_size) != block_m2t:
        raise ValueError("nanollada_v1 block sizes require the causal-clean block M2T objective")
    if (
        (config.noise.type != "mdlm" and not block_m2t)
        or config.noise.time_conditioning
        or config.model.block_causal_pilot_size
    ):
        raise ValueError(
            "nanollada_v1 supports unconditioned MDLM, causal-clean block M2T or causal next-token training"
        )


def validate_nanollada_runtime(config: DiffusionConfig) -> None:
    validate_backbone_profile(config)
    if ar_width_mup.runtime_active(config):
        ar_width_mup.validate_runtime(config)
        return
    if config.backbone_profile == nanochat_backbone_config.PROFILE:
        nanochat_backbone_config.validate_runtime(config)
        return
    if config.parameterization_rule_set == muon_residual_depth.RULE:
        muon_residual_depth.validate_runtime(config)
        return
    if (
        config.backbone_profile == NANOLLADA_PROFILE
        and config.parameterization_rule_set not in nanollada_completep.RULE_SETS
        and (
            config.parameterization_mode != "ordinary"
            or config.parameterization_mechanics_sha256
            or config.time_conditioning
            or config.parameterization_residual_multiplier != 1.0
            or config.parameterization_output_multiplier != 1.0
        )
    ):
        raise ValueError("nanollada_v1 requires ordinary unconditioned model mechanics")


@torch.no_grad()
def init_nanollada_weights(
    embedding: nn.Embedding,
    head: nn.Linear,
    blocks: nn.ModuleList,
    compute_dtype: torch.dtype,
    *,
    output_head_std: float = 0.001,
    keep_embedding_fp32: bool = False,
) -> None:
    # Upstream samples the embedding in FP32 before storing it in compute dtype.
    # Restore FP32 first so explicit reinitialization has the same semantics.
    embedding.to(dtype=torch.float32)
    nn.init.normal_(embedding.weight, std=1.0)
    nn.init.normal_(head.weight, std=output_head_std)
    bound = math.sqrt(3 / embedding.embedding_dim)
    for block in blocks:
        for projection in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
            nn.init.uniform_(projection.weight, -bound, bound)
        nn.init.zeros_(block.attn.c_proj.weight)
        nn.init.uniform_(block.mlp.up.weight, -bound, bound)
        nn.init.zeros_(block.mlp.down.weight)
    if compute_dtype != torch.float16 and not keep_embedding_fp32:
        embedding.to(dtype=compute_dtype)


def nanollada_initialization_spec(role: str, width: int) -> dict[str, float | str]:
    if role in {"token_embedding", "output_head"}:
        return {
            "rule": "normal",
            "resolved_scale": 1.0 if role == "token_embedding" else 0.001,
            "scale_multiplier": 1.0,
        }
    if role in {"attention_output", "mlp_output"}:
        return {"rule": "zeros", "resolved_scale": 0.0, "scale_multiplier": 1.0}
    if role in {"attention_query", "attention_key", "attention_value", "mlp_up"}:
        return {
            "rule": "uniform",
            "resolved_scale": width**-0.5,
            "scale_multiplier": 1.0,
            "uniform_bound": math.sqrt(3 / width),
        }
    raise ValueError(f"nanollada_v1 has no initialization rule for role {role!r}")


def nanollada_initialization_fingerprint(width: int) -> dict[str, object]:
    return {
        "profile": NANOLLADA_PROFILE,
        "embedding_std": 1.0,
        "output_head_std": 0.001,
        "nonzero_block_matrix_distribution": "uniform",
        "nonzero_block_matrix_std": width**-0.5,
        "nonzero_block_matrix_uniform_bound": math.sqrt(3 / width),
        "residual_output_projection_init": "zero",
        # Bind the profile policy, not a device-specific resolution. Evaluation
        # precision belongs to runtime provenance, not architecture identity.
        "embedding_parameter_dtype_rule": "compute_dtype_except_float16_keeps_float32",
    }
