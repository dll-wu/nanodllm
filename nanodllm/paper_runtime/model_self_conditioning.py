"""Two-pass soft self-conditioning, identical at train and inference time.

The first pass is detached. Only MASK inputs receive the zero-initialized
adapter residual; caller-selected output positions never affect conditioning.
This is same-state refinement, not a cache of a previous denoising step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_attention import Linear, rms_norm


class BlockSelfConditioning(nn.Module):
    def __init__(self, width: int, bottleneck: int):
        super().__init__()
        self.down = Linear(width, bottleneck, bias=False)
        self.up = Linear(bottleneck, width, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, expected_embedding: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(rms_norm(expected_embedding))))


def conditioned_embeddings(model, tokens: torch.Tensor, *, attention_mask, position_ids) -> torch.Tensor:
    hard = model.wte(tokens).to(dtype=model.compute_dtype)
    indices = tokens.reshape(-1).eq(model.config.mask_token_id).nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return hard
    with torch.no_grad():
        logits = model(
            tokens,
            logits_indices=indices,
            input_embeddings=hard.detach(),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        # Chunk the vocabulary expectation to bound transient FP32 probabilities.
        expected = torch.cat(
            [
                torch.softmax(chunk.float(), dim=-1).to(model.compute_dtype)
                @ model.wte.weight[: logits.shape[-1]].detach().to(model.compute_dtype)
                for chunk in logits.split(256)
            ]
        )
        del logits
    residual = model.block_self_conditioning(expected.detach())
    return hard.flatten(0, 1).index_add(0, indices, residual.to(hard.dtype)).view_as(hard)
