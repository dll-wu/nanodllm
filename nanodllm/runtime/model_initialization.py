"""Profile-aware initialization with preserved legacy sampling order."""

import math

import torch
import torch.nn as nn

from . import ar_width_mup, nanochat_backbone, nanollada_completep
from .model_attention import precompute_rotary
from .model_profiles import NANOLLADA_PROFILE, init_nanollada_weights


def residual_output_projection_init_std(n_layer: int) -> float:
    return 0.02 / math.sqrt(2 * n_layer)


def embedding_head_init_std(n_embd: int, multiplier: float = 1.0) -> float:
    return 0.4 / math.sqrt(n_embd) * multiplier


@torch.no_grad()
def initialize_weights(self):
    if next(self.parameters()).device.type == "meta":
        if (
            self.config.backbone_profile == NANOLLADA_PROFILE
            and self.compute_dtype != torch.float16
            and not ar_width_mup.runtime_active(self.config)
        ):
            self.wte.to(dtype=self.compute_dtype)
        if self.nanochat_extras is not None and self.compute_dtype != torch.float16:
            for embedding in [self.wte, *self.nanochat_extras.value_embeds.values()]:
                embedding.to(dtype=self.compute_dtype)
        return
    head_dim = self.config.n_embd // self.config.n_head
    cos, sin = precompute_rotary(
        self.config.sequence_len,
        head_dim,
        self.rope_base,
        self.compute_dtype,
        device=self.wte.weight.device,
    )
    self.rope_cos.copy_(cos)
    self.rope_sin.copy_(sin)

    if self.nanochat_extras is not None:
        nanochat_backbone.init_weights(self)
        return
    if self.config.backbone_profile == NANOLLADA_PROFILE:
        init_nanollada_weights(
            self.wte,
            self.lm_head,
            self.blocks,
            self.compute_dtype,
            output_head_std=nanollada_completep.head_std(self.config),
            keep_embedding_fp32=ar_width_mup.runtime_active(self.config),
        )
        if getattr(self, "mtp", None) is not None:
            self.mtp.reset_parameters()
        return
    sigma_base = embedding_head_init_std(self.config.n_embd)
    if self.config.parameterization_mode == "ordinary":
        sigma_embedding_head = embedding_head_init_std(
            self.config.n_embd,
            float(self.config.embedding_head_init_std_multiplier),
        )
        sigma_aux = (
            float(self.config.parameterization_hidden_init_std)
            if self.config.parameterization_mechanics_sha256 and float(self.config.parameterization_hidden_init_std) > 0
            else residual_output_projection_init_std(self.config.n_layer)
        )
    else:
        sigma_embedding_head = float(self.config.parameterization_embedding_head_init_std)
        sigma_aux = float(self.config.parameterization_hidden_init_std)
    nn.init.normal_(self.wte.weight, std=sigma_embedding_head)
    if getattr(self, "block_self_conditioning", None) is not None:
        self.block_self_conditioning.reset_parameters()
    nn.init.normal_(self.lm_head.weight, std=sigma_embedding_head)
    if self.class_head is not None:
        nn.init.normal_(self.class_head.weight, std=sigma_base)
    if self.length_head is not None:
        nn.init.normal_(self.length_head.weight, std=sigma_base)
        nn.init.zeros_(self.length_head.bias)
    if self.target_length_embedding is not None:
        nn.init.normal_(self.target_length_embedding.weight, std=sigma_base)
    if self.target_distance_proj is not None:
        nn.init.normal_(self.target_distance_proj.weight, std=sigma_base)
    if self.mask_position_embedding is not None:
        nn.init.normal_(self.mask_position_embedding.weight, std=sigma_base)
    for adapter in self.token_edit_adapters:
        nn.init.normal_(adapter.down.weight, std=sigma_aux)
        nn.init.zeros_(adapter.up.weight)
    if self.token_edit_provenance_embedding is not None:
        nn.init.zeros_(self.token_edit_provenance_embedding.weight)
    for head in self.intermediate_heads:
        nn.init.normal_(head.weight, std=sigma_base)
    if self.noise_mlp is not None:
        nn.init.normal_(self.noise_mlp[0].weight, std=sigma_base)
        nn.init.zeros_(self.noise_mlp[2].weight)
    for block in self.blocks:
        for p in block.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=sigma_aux)
        if self.config.residual_output_projection_init == "zero":
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.zeros_(block.mlp.down.weight)
    nn.init.ones_(self.final_norm.weight)
    nn.init.zeros_(self.final_norm.bias)
