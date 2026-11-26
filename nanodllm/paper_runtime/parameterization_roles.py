"""Semantic parameter roles; historical taxonomy remains digest-compatible."""

import re

from . import nanochat_backbone_config
from .model_mtp import MTP_ROLES

PARAMETER_ROLES = (
    "token_embedding",
    "attention_query",
    "attention_key",
    "attention_value",
    "attention_output",
    "mlp_gate",
    "mlp_up",
    "mlp_output",
    "final_norm_scale",
    "final_norm_bias",
    "output_head",
    "class_output_head",
    "length_output_head_weight",
    "length_output_head_bias",
    "target_length_embedding",
    "target_distance_projection",
    "mask_position_embedding",
    "token_edit_adapter_down",
    "token_edit_adapter_up",
    "token_edit_provenance_embedding",
    "auxiliary_output_head",
    "noise_input_projection",
    "noise_output_projection",
    "block_self_conditioning_down",
    "block_self_conditioning_up",
)

_OPTIONAL_ADAPTATION_ROLES = frozenset(
    {
        "class_output_head",
        "length_output_head_weight",
        "length_output_head_bias",
        "target_length_embedding",
        "target_distance_projection",
        "mask_position_embedding",
        "token_edit_adapter_down",
        "token_edit_adapter_up",
        "token_edit_provenance_embedding",
        "auxiliary_output_head",
        "noise_input_projection",
        "noise_output_projection",
        "block_self_conditioning_down",
        "block_self_conditioning_up",
    }
)
_ADAMW_NO_DECAY_ROLES = frozenset(
    {
        "token_embedding",
        "final_norm_scale",
        "final_norm_bias",
        "length_output_head_bias",
        "token_edit_provenance_embedding",
    }
)
_MUON_ADAMW_NO_DECAY_ROLES = frozenset({"final_norm_scale", "final_norm_bias", "length_output_head_bias"})

_ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("token_embedding", re.compile(r"wte\.weight")),
    ("attention_query", re.compile(r"blocks\.\d+\.attn\.c_q\.weight")),
    ("attention_key", re.compile(r"blocks\.\d+\.attn\.c_k\.weight")),
    ("attention_value", re.compile(r"blocks\.\d+\.attn\.c_v\.weight")),
    ("attention_output", re.compile(r"blocks\.\d+\.attn\.c_proj\.weight")),
    ("mlp_gate", re.compile(r"blocks\.\d+\.mlp\.gate\.weight")),
    ("mlp_up", re.compile(r"blocks\.\d+\.mlp\.up\.weight")),
    ("mlp_output", re.compile(r"blocks\.\d+\.mlp\.down\.weight")),
    ("final_norm_scale", re.compile(r"final_norm\.weight")),
    ("final_norm_bias", re.compile(r"final_norm\.bias")),
    ("output_head", re.compile(r"lm_head\.weight")),
    ("class_output_head", re.compile(r"class_head\.weight")),
    ("length_output_head_weight", re.compile(r"length_head\.weight")),
    ("length_output_head_bias", re.compile(r"length_head\.bias")),
    ("target_length_embedding", re.compile(r"target_length_embedding\.weight")),
    ("target_distance_projection", re.compile(r"target_distance_proj\.weight")),
    ("mask_position_embedding", re.compile(r"mask_position_embedding\.weight")),
    ("token_edit_adapter_down", re.compile(r"token_edit_adapters\.\d+\.down\.weight")),
    ("token_edit_adapter_up", re.compile(r"token_edit_adapters\.\d+\.up\.weight")),
    ("token_edit_provenance_embedding", re.compile(r"token_edit_provenance_embedding\.weight")),
    ("auxiliary_output_head", re.compile(r"intermediate_heads\.\d+\.weight")),
    ("noise_input_projection", re.compile(r"noise_mlp\.0\.weight")),
    ("noise_output_projection", re.compile(r"noise_mlp\.2\.weight")),
    ("block_self_conditioning_down", re.compile(r"block_self_conditioning\.down\.weight")),
    ("block_self_conditioning_up", re.compile(r"block_self_conditioning\.up\.weight")),
)


backbone_roles = {
    "nanochat_extras.resid_lambdas": "residual_scale",
    "nanochat_extras.x0_lambdas": "input_mix",
    "nanochat_extras.smear_gate.weight": "smear_gate",
    "nanochat_extras.smear_lambda": "smear_scale",
    "nanochat_extras.backout_lambda": "backout_scale",
}


def ordinary_parameter_roles(config) -> tuple[str, ...]:
    """Include optional roles only for an enabled architecture, preserving old digests."""
    roles = PARAMETER_ROLES
    if config.backbone_profile == nanochat_backbone_config.PROFILE:
        roles += nanochat_backbone_config.ROLES
    if config.mtp_mode != "none":
        roles += MTP_ROLES
    return roles


def parameter_role(name: str) -> str:
    """Return the one semantic role assigned to a NanoDLLM parameter name."""

    if name == "mtp.concat.weight":
        return "mtp_concat_projection"
    if name.startswith("mtp.block."):
        return "mtp_" + parameter_role("blocks.0." + name.removeprefix("mtp.block."))
    extra = backbone_roles.get(name)
    if extra is not None:
        return extra
    if re.fullmatch(r"nanochat_extras\.value_embeds\.\d+\.weight", name):
        return "value_embedding"
    if re.fullmatch(r"blocks\.\d+\.attn\.ve_gate\.weight", name):
        return "value_gate"
    matches = [role for role, pattern in _ROLE_PATTERNS if pattern.fullmatch(name)]
    if len(matches) != 1:
        raise ValueError(f"parameter {name!r} must match exactly one semantic role, observed {matches}")
    return matches[0]
