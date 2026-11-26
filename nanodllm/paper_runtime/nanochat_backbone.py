"""Nanochat-derived layers with visibility-safe token smear and logical windows.

Adapted from karpathy/nanochat@92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
(MIT, Copyright (c)2025 Andrej Karpathy; docs/licenses/nanochat-MIT.txt).
Unlike the causal oracle, every input-dependent operation honors the supplied
document/clean/noisy visibility graph. No gold token crosses a block boundary
unless that token is in the query's completed prefix.
"""

import math

import torch
from torch import nn

from .model_attention import BidirectionalAttention, Linear, rms_norm
from .model_profiles import SquaredReLUMLP


def has_values(layer, depth):
    return layer % 2 == (depth - 1) % 2


def windows(config):
    short = -(-config.sequence_len // 4 // 128) * 128
    return [short if i % 4 != 3 and i != config.n_layer - 1 else None for i in range(config.n_layer)]


class Extras(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(torch.ones(1))
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(vocab, config.n_embd) for i in range(config.n_layer) if has_values(i, config.n_layer)}
        )

    def mix(self, x, initial, index):
        return self.resid_lambdas[index] * x + self.x0_lambdas[index] * initial

    def smear(self, x, previous):
        return x + self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[..., :24])) * previous

    def values(self, index, tokens, dtype):
        return self.value_embeds[str(index)](tokens).to(dtype) if str(index) in self.value_embeds else None


class Attention(BidirectionalAttention):
    def __init__(self, config, index):
        # Window masks are built from logical positions, never physical offsets
        # in the doubled [noisy; clean] sequence.
        super().__init__(config, causal=config.causal)
        self.ve_gate = Linear(12, config.n_head, bias=False) if has_values(index, config.n_layer) else None

    def sampled_attention_diagnostics(self, *args, **kwargs):
        raise ValueError("nanochat attention diagnostics require a profile-aware value/window probe")

    def project_with_values(self, x, cos, sin, values):
        q, k, v = self.project_qkv(x, cos, sin)
        if values is not None:
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :12]))
            v = v + (gate[..., None] * values.view(*x.shape[:2], self.n_head, self.head_dim)).transpose(1, 2)
        return q * 1.2, k * 1.2, v


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = Attention(config, layer_idx)
        self.mlp = SquaredReLUMLP(config.n_embd)
        self.residual_multiplier = 1.0

    def forward(self, x, cos, sin, *, attention_mask, values):
        q, k, v = self.attn.project_with_values(rms_norm(x), cos, sin, values)
        x = x + self.attn.attend_projected(q, k, v, attention_mask=attention_mask)
        return x + self.mlp(rms_norm(x))


@torch.no_grad()
def init_weights(model):
    e = model.nanochat_extras
    for embedding in [model.wte, *e.value_embeds.values()]:
        embedding.to(dtype=torch.float32)
    nn.init.normal_(model.wte.weight, std=0.8)
    nn.init.normal_(model.lm_head.weight, std=0.001)
    bound = math.sqrt(3 / model.config.n_embd)
    for block in model.blocks:
        for projection in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
            nn.init.uniform_(projection.weight, -bound, bound)
        nn.init.zeros_(block.attn.c_proj.weight)
        nn.init.uniform_(block.mlp.up.weight, -bound * 0.4, bound * 0.4)
        nn.init.zeros_(block.mlp.down.weight)
    for i in range(model.config.n_layer):
        e.resid_lambdas[i] = 1.15 - 0.10 * i / max(model.config.n_layer - 1, 1)
        e.x0_lambdas[i] = 0.20 - 0.15 * i / max(model.config.n_layer - 1, 1)
    nn.init.zeros_(e.smear_lambda)
    nn.init.constant_(e.backout_lambda, 0.2)
    nn.init.uniform_(e.smear_gate.weight, 0, 0.02)
    for embedding in e.value_embeds.values():
        nn.init.uniform_(embedding.weight, -bound, bound)
    for block in model.blocks:
        if block.attn.ve_gate is not None:
            nn.init.uniform_(block.attn.ve_gate.weight, 0, 0.02)
    if model.compute_dtype != torch.float16:
        for embedding in [model.wte, *e.value_embeds.values()]:
            embedding.to(dtype=model.compute_dtype)


def logical_positions(tokens, positions):
    if positions is None:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
    return positions.expand(tokens.shape)


def visible_predecessor(x, positions, visible):
    """Select an allowed predecessor, including clean-prefix→noisy-block edges.

    Multiplication after gather makes missing predecessors exactly zero. The
    graph chooses the stream; physical adjacency must not choose it for us.
    """
    candidates = (positions[:, :, None] - positions[:, None, :]) == 1
    if visible is not None:
        candidates = candidates & visible[:, 0]
    indices = torch.arange(x.shape[1], device=x.device)[None, None, :]
    previous = torch.where(candidates, indices, -1).amax(-1)
    gathered = x.gather(1, previous.clamp_min(0)[..., None].expand_as(x))
    return gathered * previous.ge(0)[..., None]


def hidden(model, tokens, x, cos, sin, attention_mask, position_ids):
    positions = logical_positions(tokens, position_ids)
    visible = model.blocks[0].attn._build_attention_mask(
        batch_size=tokens.shape[0], seq_len=tokens.shape[1], device=tokens.device, attention_mask=attention_mask
    )
    # Share one graph over heads: the supported packing/objective interfaces
    # never use different document visibility per head.
    if visible is not None and visible.shape[1] != 1:
        raise ValueError("nanochat smear requires shared head visibility")
    x = rms_norm(x)
    initial = model.nanochat_extras.smear(x, visible_predecessor(x, positions, visible))
    x, backout = initial, None
    distances = (positions[:, :, None] - positions[:, None, :]).abs()[:, None]
    for i, (block, window) in enumerate(zip(model.blocks, windows(model.config), strict=True)):
        mask = visible
        if window is not None:
            mask = distances.le(window) if visible is None else visible & distances.le(window)
        x = model.nanochat_extras.mix(x, initial, i)
        values = model.nanochat_extras.values(i, tokens, x.dtype)
        x = block(x, cos, sin, attention_mask=mask, values=values)
        if i == model.config.n_layer // 2:
            backout = x
    return x - model.nanochat_extras.backout_lambda.to(x.dtype) * backout


def softcap(logits):
    return 15 * torch.tanh(logits.float() / 15)


def fingerprint(model, payload):
    from .nanochat_backbone_config import PROFILE, initialization

    payload["architecture"].update(
        backbone_profile=PROFILE,
        mlp_type="squared_relu",
        input_norm="parameter_free_rms_norm",
        final_norm="parameter_free_rms_norm",
        learned_residual_input_mix=True,
        backout_layer=model.config.n_layer // 2,
        smear="visible_logical_predecessor",
        value_embedding_layers=list(model.nanochat_extras.value_embeds),
        logit_softcap=15,
    )
    payload["attention"].update(
        qk_norm="parameter_free_rms_norm_after_rope_times_1.2",
        window_pattern="SSSL_final_global",
        logical_windows=windows(model.config),
    )
    payload["initialization"] = {
        "profile": PROFILE,
        "policy": {
            role: initialization(role, model.config.n_embd)
            for role in (
                "token_embedding",
                "output_head",
                "attention_query",
                "attention_output",
                "mlp_up",
                "value_embedding",
                "value_gate",
                "residual_scale",
                "input_mix",
                "smear_gate",
                "smear_scale",
                "backout_scale",
            )
        },
        "embedding_parameter_dtype_rule": "compute_dtype_except_float16_keeps_float32",
    }
    payload["flops_estimate_method"] = (
        "6x_block_and_output_matrices_plus_dense_attention_upper_bound_excludes_embedding_lookups"
    )
    payload["nanochat_parameter_breakdown"] = {
        "token_embedding": model.wte.weight.numel(),
        "value_embeddings": sum(p.numel() for p in model.nanochat_extras.value_embeds.parameters()),
        "outer_scalars_and_smear_gate": sum(
            p.numel() for n, p in model.nanochat_extras.named_parameters() if not n.startswith("value_embeds.")
        ),
        "transformer_blocks_including_value_gates": sum(p.numel() for p in model.blocks.parameters()),
        "output_head": model.lm_head.weight.numel(),
    }


def estimated_flops(model, seq_len):
    # Dense SDPA masks do not promise sparse-kernel speedups; deliberately do
    # not discount local attention as if a window-aware kernel were in use.
    matrices = model.parameter_counts()["nanochat_scaling_parameters"]
    return 6 * matrices + 12 * model.config.n_embd * seq_len * model.config.n_layer
