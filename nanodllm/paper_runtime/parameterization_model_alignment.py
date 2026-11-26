"""Verify instantiated model mechanics against their experiment configuration."""

from torch import nn

from .config import ExperimentConfig
from .model import DiffusionConfig, resolved_attention_scale
from .parameterization_mechanics import parameterization_model_config_kwargs


def _validate_model_config_alignment(model_config: DiffusionConfig, config: ExperimentConfig) -> None:
    """Prove that resolved model semantics agree with their experiment source."""

    expected = {
        "sequence_len": int(config.model.max_seq_len),
        "n_layer": int(config.model.depth),
        "n_head": int(config.model.n_head),
        "n_embd": int(config.model.n_embd),
        "mlp_hidden_dim": int(config.model.mlp_hidden_dim),
        "attention_scale": float(config.model.attention_scale),
        "local_attention_lower_layers": int(config.model.local_attention_lower_layers),
        "local_attention_window": int(config.model.local_attention_window),
        "backbone_profile": config.model.backbone_profile,
        "mtp_mode": config.model.mtp_mode,
        "recurrent_core_start": config.model.recurrent_core_start,
        "recurrent_core_length": config.model.recurrent_core_length,
        "recurrent_iterations": config.model.recurrent_iterations,
        "auxiliary_head_layers": tuple(int(index) for index in config.model.auxiliary_head_layers),
        "time_conditioning": bool(config.noise.time_conditioning),
        "bd3lm_block_size": int(config.model.bd3lm_block_size) if config.noise.type == "bd3lm" else 0,
        "bd3lm_self_conditioning_dim": config.model.bd3lm_self_conditioning_dim,
        "output_factorization": str(config.model.output_factorization),
        "causal": bool(config.model.causal),
        "residual_output_projection_init": str(config.model.residual_output_projection_init),
        "embedding_head_init_std_multiplier": float(config.model.embedding_head_init_std_multiplier),
        "length_head": bool(config.model.length_head),
        "target_length_conditioning": bool(config.model.target_length_conditioning),
        "mask_position_embedding": bool(config.model.mask_position_embedding),
        "token_edit_adapter_bottleneck_dim": int(config.model.token_edit_adapter_bottleneck_dim),
        "token_edit_adapter_top_layers": int(config.model.token_edit_adapter_top_layers),
        "token_edit_provenance_embedding": bool(config.model.token_edit_provenance_embedding),
        "token_edit_provenance_confidence_bins": int(config.model.token_edit_provenance_confidence_bins),
    }
    expected.update(parameterization_model_config_kwargs(config))
    mismatches = {
        field: {"experiment_config": expected_value, "instantiated_model": getattr(model_config, field)}
        for field, expected_value in expected.items()
        if getattr(model_config, field) != expected_value
    }
    if mismatches:
        raise ValueError(f"experiment and instantiated model configs disagree for parameterization: {mismatches}")


def _resolved_model_config(model: nn.Module, config: ExperimentConfig) -> DiffusionConfig:
    """Return the instantiated model config after proving experiment alignment."""

    model_config = getattr(model, "config", None)
    if not isinstance(model_config, DiffusionConfig):
        raise TypeError("parameterization contracts require a model with a resolved DiffusionConfig")
    _validate_model_config_alignment(model_config, config)
    return model_config


def _validate_implemented_nonordinary_mechanics(model: nn.Module, model_config: DiffusionConfig) -> None:
    expected_attention_scale = resolved_attention_scale(model_config)
    for layer_index, block in enumerate(model.blocks):
        if float(block.attn.attention_scale) != expected_attention_scale:
            raise ValueError(
                "instantiated attention scale disagrees with resolved parameterization mechanics: "
                f"layer={layer_index}, instantiated={block.attn.attention_scale}, resolved={expected_attention_scale}"
            )
        if float(block.residual_multiplier) != float(model_config.parameterization_residual_multiplier):
            raise ValueError(
                "instantiated residual multiplier disagrees with resolved parameterization mechanics: "
                f"layer={layer_index}, instantiated={block.residual_multiplier}, "
                f"resolved={model_config.parameterization_residual_multiplier}"
            )
