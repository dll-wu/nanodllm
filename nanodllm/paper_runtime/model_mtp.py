"""One optional future-token block with shared vocabulary projections."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from .model_attention import Linear, rms_norm

if TYPE_CHECKING:
    from .model import DiffusionConfig

MTP_MODES = ("none", "auxiliary_ntp", "independent", "conditional")
MTP_ROLES = (
    "mtp_attention_query",
    "mtp_attention_key",
    "mtp_attention_value",
    "mtp_attention_output",
    "mtp_mlp_up",
    "mtp_mlp_output",
    "mtp_concat_projection",
)


def strip_disabled_prediction_extensions(value: dict) -> None:
    """Omit only inert additions from an existing frozen config projection."""
    if value["train"].pop("recurrent_depth_schedule", "fixed") != "fixed":
        raise ValueError("historical projection requires train.recurrent_depth_schedule='fixed'")
    for section, field, default in (
        ("model", "mtp_mode", "none"),
        ("train", "mtp_loss_weight", 0.0),
        ("model", "recurrent_core_start", 0),
        ("model", "recurrent_core_length", 0),
        ("model", "recurrent_iterations", 1),
    ):
        if value[section].pop(field) != default:
            raise ValueError(f"historical projection requires {section}.{field}={default!r}")


def validate_mtp_model(config) -> None:
    if config.mtp_mode not in MTP_MODES:
        raise ValueError(f"model.mtp_mode must be one of {MTP_MODES}")
    if config.mtp_mode == "none":
        return
    if config.backbone_profile != "nanollada_v1" or not config.causal or config.bd3lm_block_size:
        raise ValueError("MTP requires plain causal nanollada_v1")
    if (config.recurrent_core_start, config.recurrent_core_length, config.recurrent_iterations) != (0, 0, 1):
        raise ValueError("MTP with recurrence is not supported")
    # The profile owner rejects all other optional model extensions.
    if getattr(config, "parameterization_mode", "ordinary") != "ordinary" or getattr(
        config, "parameterization_mechanics_sha256", ""
    ):
        raise ValueError("MTP requires ordinary unscoped parameterization")


class AuxiliaryMTP(nn.Module):
    """The parent owns embedding/head weights; do not register aliases here."""

    def __init__(self, config: DiffusionConfig, block: nn.Module):
        super().__init__()
        self.mode = config.mtp_mode
        self.block = block
        self.concat = Linear(2 * config.n_embd, config.n_embd, bias=False) if self.mode == "conditional" else None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        width = self.block.attn.c_q.in_features
        bound = math.sqrt(3 / width)
        for projection in (self.block.attn.c_q, self.block.attn.c_k, self.block.attn.c_v, self.block.mlp.up):
            nn.init.uniform_(projection.weight, -bound, bound)
        nn.init.zeros_(self.block.attn.c_proj.weight)
        nn.init.zeros_(self.block.mlp.down.weight)
        if self.concat is not None:
            nn.init.uniform_(self.concat.weight, -math.sqrt(3 / (2 * width)), math.sqrt(3 / (2 * width)))

    def forward(
        self,
        hidden,
        tokens,
        embedding,
        head,
        cos,
        sin,
        attention_mask,
        vocab_size,
        *,
        next_token=None,
        logits_indices=None,
    ):
        if next_token is not None and (
            self.concat is None
            or next_token.shape != (tokens.shape[0],)
            or next_token.dtype != torch.long
            or attention_mask is not None
        ):
            raise ValueError("explicit MTP next_token requires conditional unsegmented [batch] int64 input")
        x = rms_norm(hidden)
        if self.concat is not None:
            # Keep the full sequence shape. The final slot has no teacher token;
            # document-ending slots cannot import the following document either.
            gold = rms_norm(embedding(tokens[:, 1:]).to(dtype=x.dtype))
            adjacent = adjacent_support(tokens, attention_mask)
            gold = gold * adjacent.unsqueeze(-1).to(gold.dtype)
            tail = (
                gold.new_zeros((tokens.shape[0], 1, gold.shape[-1]))
                if next_token is None
                else rms_norm(embedding(next_token[:, None]).to(dtype=x.dtype))
            )
            gold = torch.cat((gold, tail), dim=1)
            x = self.concat(torch.cat((x, gold), dim=-1))
        x = self.block(x, cos, sin, attention_mask=attention_mask)
        if logits_indices is not None:
            x = x.flatten(0, 1).index_select(0, logits_indices)
        return head(rms_norm(x))[..., :vocab_size].float()


def adjacent_support(tokens: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Use the later query's visibility of its predecessor (also valid for causal masks)."""
    batch, length = tokens.shape
    if attention_mask is None:
        return torch.ones((batch, max(0, length - 1)), dtype=torch.bool, device=tokens.device)
    if attention_mask.dtype != torch.bool:
        raise ValueError("MTP requires boolean attention masks")
    if attention_mask.shape == tokens.shape:
        return attention_mask[:, 1:] & attention_mask[:, :-1]
    if attention_mask.shape == (batch, length, length):
        return attention_mask.diagonal(offset=-1, dim1=-2, dim2=-1)
    raise ValueError("MTP requires [batch, sequence] or [batch, sequence, sequence] attention masks")


def validate_mtp_forward(model, tokens, attention_mask, options) -> None:
    if model.mtp is None:
        raise ValueError("return_mtp_logits requires model.mtp_mode to be enabled")
    if tokens.shape[1] == 0:
        raise ValueError("MTP requires a nonempty input sequence")
    if any(value is not None and value is not False for value in options):
        raise ValueError("return_mtp_logits requires the plain full-sequence token forward")
    adjacent_support(tokens, attention_mask)


def validate_mtp_hidden(model, options) -> None:
    """A bounded inference interface, not a general hidden-state/feature API."""
    config = model.config
    if (
        model.training
        or config.backbone_profile != "nanollada_v1"
        or not config.causal
        or config.bd3lm_block_size
        or config.parameterization_mode != "ordinary"
        or config.parameterization_mechanics_sha256
        or config.recurrent_iterations != 1
        or any(value is not None and value is not False for value in options)
    ):
        raise ValueError("return_mtp_hidden requires eval-mode plain causal ordinary nanollada_v1")


def validate_mtp_request(model, tokens, mask, logits_requested, hidden_requested, indices, options):
    if hidden_requested:
        validate_mtp_hidden(model, (logits_requested, *options))
    else:
        validate_mtp_forward(model, tokens, mask, (indices, *options))


def mtp_output(model, tokens, hidden, logits, cos, sin, mask, hidden_requested):
    if hidden_requested:
        return logits, hidden
    return logits, model.mtp(hidden, tokens, model.wte, model.lm_head, cos, sin, mask, model.config.vocab_size)


def mtp_initialization(role: str, width: int) -> dict[str, float | str]:
    if role in {"mtp_attention_output", "mtp_mlp_output"}:
        return {"rule": "zeros", "resolved_scale": 0.0, "scale_multiplier": 1.0}
    fan_in = 2 * width if role == "mtp_concat_projection" else width
    return {
        "rule": "uniform",
        "resolved_scale": fan_in**-0.5,
        "scale_multiplier": 1.0,
        "uniform_bound": math.sqrt(3 / fan_in),
    }
