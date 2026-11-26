"""Completed-prefix cache execution for nanochat-derived causal/block layers."""

import torch

from .model_attention import rms_norm
from .nanochat_backbone import windows


def forward(cache, tokens, *, causal):
    model, start, width = cache._model, cache.length, tokens.shape[1]
    extras = model.nanochat_extras
    positions = torch.arange(start, start + width, device=tokens.device)
    keys_positions = torch.arange(start + width, device=tokens.device)
    cos, sin = model.rope_cos[start : start + width], model.rope_sin[start : start + width]
    x = rms_norm(model.wte(tokens).to(model.compute_dtype))
    before = (
        rms_norm(model.wte(cache._tokens[:, start - 1 : start]).to(model.compute_dtype))
        if start
        else torch.zeros_like(x[:, :1])
    )
    initial = extras.smear(x, torch.cat((before, x[:, :-1]), 1))
    hidden, backout, pending = initial, None, []
    visible = positions[:, None] >= keys_positions[None, :] if causal else None
    distances = (positions[:, None] - keys_positions[None, :]).abs()
    for index, (block, window) in enumerate(zip(model.blocks, windows(model.config), strict=True)):
        hidden = extras.mix(hidden, initial, index)
        query, key, value = block.attn.project_with_values(
            rms_norm(hidden), cos, sin, extras.values(index, tokens, hidden.dtype)
        )
        keys = torch.cat((cache._keys[index][:, :, :start], key), 2) if start else key
        values = torch.cat((cache._values[index][:, :, :start], value), 2) if start else value
        mask = visible
        if window is not None:
            mask = distances.le(window) if visible is None else visible & distances.le(window)
        hidden = hidden + block.attn.attend_projected(query, keys, values, attention_mask=mask)
        hidden = hidden + block.mlp(rms_norm(hidden))
        if index == model.config.n_layer // 2:
            backout = hidden
        if causal:
            pending.append((key, value))
    return hidden - extras.backout_lambda.to(hidden.dtype) * backout, pending
