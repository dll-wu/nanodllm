"""Model conditioning, packed attention masks and logit adapters for scoring."""

import torch


def _model_uses_noise_level(model) -> bool:
    config = getattr(model, "config", None)
    return bool(getattr(config, "time_conditioning", False))


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def model_uses_target_length_conditioning(model) -> bool:
    config = getattr(_unwrap_model(model), "config", None)
    return bool(getattr(config, "target_length_conditioning", False))


def _call_model_logits(
    model,
    z_t: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    model_logits_fn=None,
    noise_level=None,
    position_ids: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if model_logits_fn is not None:
        if target_lengths is not None:
            if position_ids is not None:
                if noise_level is None:
                    return model_logits_fn(
                        model,
                        z_t,
                        x_0,
                        attention_mask,
                        position_ids=position_ids,
                        target_lengths=target_lengths,
                    )
                return model_logits_fn(
                    model,
                    z_t,
                    x_0,
                    attention_mask,
                    noise_level=noise_level,
                    position_ids=position_ids,
                    target_lengths=target_lengths,
                )
            if noise_level is None:
                return model_logits_fn(model, z_t, x_0, attention_mask, target_lengths=target_lengths)
            return model_logits_fn(
                model,
                z_t,
                x_0,
                attention_mask,
                noise_level=noise_level,
                target_lengths=target_lengths,
            )
        if position_ids is not None:
            if noise_level is None:
                return model_logits_fn(model, z_t, x_0, attention_mask, position_ids=position_ids)
            return model_logits_fn(
                model,
                z_t,
                x_0,
                attention_mask,
                noise_level=noise_level,
                position_ids=position_ids,
            )
        if noise_level is None:
            return model_logits_fn(model, z_t, x_0, attention_mask)
        return model_logits_fn(model, z_t, x_0, attention_mask, noise_level=noise_level)
    model_kwargs = {"attention_mask": attention_mask}
    if noise_level is not None and _model_uses_noise_level(model):
        model_kwargs["noise_level"] = noise_level
    if position_ids is not None:
        model_kwargs["position_ids"] = position_ids
    if target_lengths is not None:
        model_kwargs["target_lengths"] = target_lengths
    return model(z_t, **model_kwargs)


def _batch_model_attention_mask(
    batch,
    attention_mask: torch.Tensor | None = None,
    query_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    base_mask = batch.attention_mask if attention_mask is None else attention_mask
    boundary_mask = getattr(batch, "document_boundary_mask", None)
    if boundary_mask is None:
        return base_mask
    if base_mask.dim() != 2:
        raise ValueError(f"base attention mask must be [batch, seq], got {tuple(base_mask.shape)}")
    b, t = base_mask.shape
    if boundary_mask.shape != (b, t, t):
        raise ValueError(f"document boundary mask must have shape {(b, t, t)}, got {tuple(boundary_mask.shape)}")
    query_mask = batch.attention_mask if query_attention_mask is None else query_attention_mask
    if query_mask.shape != base_mask.shape:
        raise ValueError(
            f"query attention mask must have shape {tuple(base_mask.shape)}, got {tuple(query_mask.shape)}"
        )
    return boundary_mask & query_mask[:, :, None] & base_mask[:, None, :]


def _block_generation_special_logits(logits: torch.Tensor, tokenizer) -> None:
    for token_id in tokenizer.get_generation_blocked_token_ids():
        if 0 <= int(token_id) < logits.size(-1):
            logits[..., int(token_id)] = -1e9
