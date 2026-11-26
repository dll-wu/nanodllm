"""Explicit native training-forward routing for plain and block objectives."""

import torch

from .config import ExperimentConfig
from .training_bd3lm import bd3lm_concat_inputs as _bd3lm_concat_inputs
from .training_features import _uses_bd3lm


def _forward_training_model(
    model,
    config: ExperimentConfig,
    z_t: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_indices: torch.Tensor,
    return_aux_logits: bool,
    noise_level,
    return_hierarchical_logits: bool = False,
    return_length_logits: bool = False,
    input_embeddings: torch.Tensor | None = None,
    input_provenance: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
    detach_final_hidden: bool = False,
    use_token_edit_adapter: bool = False,
    bd3lm_block_size: int | None = None,
    bd3lm_block_ids: torch.Tensor | None = None,
    recurrent_iterations: int | None = None,
):
    if input_embeddings is not None and _uses_bd3lm(config):
        raise ValueError("soft input embeddings are not supported for BD3LM concat training")
    if _uses_bd3lm(config):
        if detach_final_hidden or use_token_edit_adapter:
            raise ValueError("detached final hidden is not supported for BD3LM training")
        concat, bd3lm_mask, position_ids = _bd3lm_concat_inputs(
            z_t,
            x_0,
            attention_mask,
            position_ids=position_ids,
            block_size=config.model.bd3lm_block_size if bd3lm_block_size is None else bd3lm_block_size,
            clean_stream_mode=config.model.bd3lm_clean_stream_mode,
            block_ids=bd3lm_block_ids,
        )
        return model(
            concat,
            logits_indices=logits_indices,
            return_aux_logits=return_aux_logits,
            return_hierarchical_logits=return_hierarchical_logits,
            return_length_logits=return_length_logits,
            attention_mask=bd3lm_mask,
            position_ids=position_ids,
        )
    model_kwargs = {
        "logits_indices": logits_indices,
        "return_aux_logits": return_aux_logits,
        "return_hierarchical_logits": return_hierarchical_logits,
        "return_length_logits": return_length_logits,
        "attention_mask": attention_mask,
        "noise_level": noise_level if config.noise.time_conditioning else None,
        "input_embeddings": input_embeddings,
    }
    if input_provenance is not None:
        model_kwargs["input_provenance"] = input_provenance
    if detach_final_hidden:
        model_kwargs["detach_final_hidden"] = True
    if use_token_edit_adapter:
        model_kwargs["use_token_edit_adapter"] = True
    if position_ids is not None:
        model_kwargs["position_ids"] = position_ids
    if target_lengths is not None:
        model_kwargs["target_lengths"] = target_lengths
    if recurrent_iterations is not None:
        model_kwargs["recurrent_iterations"] = recurrent_iterations
    return model(z_t, **model_kwargs)
