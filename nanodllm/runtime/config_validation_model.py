"""Model-shape and architecture-extension validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_model_config(config: ExperimentConfig) -> None:
    if config.model.depth <= 0:
        raise ValueError("model.depth must be positive")
    if config.model.n_embd <= 0:
        raise ValueError("model.n_embd must be positive")
    if config.model.n_head <= 0:
        raise ValueError("model.n_head must be positive")
    if config.model.n_embd % config.model.n_head != 0:
        raise ValueError("model.n_embd must be divisible by model.n_head")
    if not isinstance(config.model.mlp_hidden_dim, int) or isinstance(config.model.mlp_hidden_dim, bool):
        raise ValueError("model.mlp_hidden_dim must be an integer")
    if config.model.mlp_hidden_dim != -1 and config.model.mlp_hidden_dim <= 0:
        raise ValueError("model.mlp_hidden_dim must be positive or -1 for the historical 4 * n_embd width")
    if isinstance(config.model.attention_scale, bool) or not isinstance(config.model.attention_scale, (int, float)):
        raise ValueError("model.attention_scale must be numeric")
    if config.model.attention_scale != -1.0 and config.model.attention_scale <= 0:
        raise ValueError(
            "model.attention_scale must be positive or -1 for PyTorch's historical 1 / sqrt(head_dim) default"
        )
    if config.model.local_attention_lower_layers < 0:
        raise ValueError("model.local_attention_lower_layers must be non-negative")
    if config.model.local_attention_lower_layers > 0 and config.model.local_attention_window <= 0:
        raise ValueError("model.local_attention_window must be positive when local attention is enabled")
    if not isinstance(config.model.causal, bool):
        raise ValueError("model.causal must be a boolean")
    if config.model.bd3lm_clean_stream_mode not in {"block_causal", "causal"}:
        raise ValueError("model.bd3lm_clean_stream_mode must be 'block_causal' or 'causal'")
    if config.model.output_factorization not in ("flat", "hierarchical_frequency_decile"):
        raise ValueError("model.output_factorization must be 'flat' or 'hierarchical_frequency_decile'")
    if config.model.residual_output_projection_init not in ("zero", "scaled_normal"):
        raise ValueError("model.residual_output_projection_init must be 'zero' or 'scaled_normal'")
    if config.model.embedding_head_init_std_multiplier <= 0:
        raise ValueError("model.embedding_head_init_std_multiplier must be positive")
    if not isinstance(config.model.length_head, bool):
        raise ValueError("model.length_head must be a boolean")
    if not isinstance(config.model.target_length_conditioning, bool):
        raise ValueError("model.target_length_conditioning must be a boolean")
    if not isinstance(config.model.mask_position_embedding, bool):
        raise ValueError("model.mask_position_embedding must be a boolean")
    if config.model.mask_position_embedding and config.model.causal:
        raise ValueError("model.mask_position_embedding is only supported for non-causal MDLM training")
    if (
        not isinstance(config.model.token_edit_adapter_bottleneck_dim, int)
        or isinstance(config.model.token_edit_adapter_bottleneck_dim, bool)
        or config.model.token_edit_adapter_bottleneck_dim < 0
    ):
        raise ValueError("model.token_edit_adapter_bottleneck_dim must be a non-negative integer")
    if (
        not isinstance(config.model.token_edit_adapter_top_layers, int)
        or isinstance(config.model.token_edit_adapter_top_layers, bool)
        or not 0 <= config.model.token_edit_adapter_top_layers <= config.model.depth
    ):
        raise ValueError("model.token_edit_adapter_top_layers must be an integer between zero and model.depth")
    if (config.model.token_edit_adapter_bottleneck_dim > 0) != (config.model.token_edit_adapter_top_layers > 0):
        raise ValueError("token-edit adapter bottleneck and top-layer count must be enabled together")
    if not isinstance(config.model.token_edit_provenance_embedding, bool):
        raise ValueError("model.token_edit_provenance_embedding must be a boolean")
    if (
        not isinstance(config.model.token_edit_provenance_confidence_bins, int)
        or isinstance(config.model.token_edit_provenance_confidence_bins, bool)
        or not 0 <= config.model.token_edit_provenance_confidence_bins <= 64
    ):
        raise ValueError("model.token_edit_provenance_confidence_bins must be an integer in [0, 64]")
    if config.model.token_edit_provenance_confidence_bins > 0 and not config.model.token_edit_provenance_embedding:
        raise ValueError("token-edit provenance confidence bins require token_edit_provenance_embedding=true")
    if (
        config.model.token_edit_adapter_bottleneck_dim > 0
        or config.model.token_edit_adapter_top_layers > 0
        or config.model.token_edit_provenance_embedding
    ) and config.parameterization.mode != "ordinary":
        raise ValueError("token-edit model extensions currently require parameterization.mode='ordinary'")
    if config.model.target_length_conditioning:
        if config.model.length_head:
            raise ValueError("model.target_length_conditioning cannot be coupled with model.length_head")
        if config.model.causal:
            raise ValueError("model.target_length_conditioning is only supported for non-causal MDLM training")
        if config.model.auxiliary_head_layers:
            raise ValueError("model.target_length_conditioning cannot be coupled with auxiliary heads")
        if config.model.output_factorization != "flat":
            raise ValueError("model.target_length_conditioning requires model.output_factorization='flat'")
        if config.data.boundary_aware_packing:
            raise ValueError("model.target_length_conditioning requires data.boundary_aware_packing=false")
        if config.data.content_budget_segmented_packing:
            raise ValueError("model.target_length_conditioning requires data.content_budget_segmented_packing=false")
        if config.data.short_document_curriculum:
            raise ValueError("model.target_length_conditioning requires data.short_document_curriculum=false")
        if not config.data.append_eos:
            raise ValueError("model.target_length_conditioning requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("model.target_length_conditioning requires data.loss_on_eos=true")
        if config.noise.type != "mdlm":
            raise ValueError("model.target_length_conditioning requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError(
                "model.target_length_conditioning preserves strict MDLM and requires complement_masking=false"
            )
        if config.noise.suffix_masking:
            raise ValueError(
                "model.target_length_conditioning preserves independent MDLM masking and requires suffix_masking=false"
            )
        if config.noise.time_conditioning:
            raise ValueError("model.target_length_conditioning preserves no time conditioning")
    if config.model.hierarchical_frequency_batches <= 0:
        raise ValueError("model.hierarchical_frequency_batches must be positive")
    for layer_idx in config.model.auxiliary_head_layers:
        if layer_idx < 0 or layer_idx >= config.model.depth:
            raise ValueError("model.auxiliary_head_layers entries must be valid zero-based layer indices")
    if len(set(config.model.auxiliary_head_layers)) != len(config.model.auxiliary_head_layers):
        raise ValueError("model.auxiliary_head_layers must not contain duplicates")
