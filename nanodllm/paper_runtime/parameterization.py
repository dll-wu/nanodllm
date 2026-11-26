"""Explicit parameterization taxonomy, contracts, and scale telemetry.

Ordinary parameterization retains its historical numerical behavior. The
explicit ``mup`` and ``completep`` modes implement the frozen NanoDLLM mapping
of Tensor Programs V Table 8 and CompleteP v4 Table 1, respectively.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from . import ar_width_mup, nanollada_completep
from . import nanochat_backbone_config as nanochat_backbone_config
from .completep_hidden_init_transfer import (
    completep_hidden_init_transfer_parameterization_contract,
    is_completep_hidden_init_transfer,
)
from .config import ExperimentConfig
from .d4_hidden_init_confirmation import (
    hidden_init_confirmation_parameterization_contract,
    is_d4_hidden_init_confirmation,
)
from .d8_batch_objective_config import SCOPE as D8_BATCH_OBJECTIVE_SCOPE
from .d8_batch_objective_config import parameterization_contract as d8_batch_objective_contract
from .d20_parameterization_transfer import (
    d20_completep_parameterization_contract,
    is_d20_completep_tpp20,
)
from .model import (
    DiffusionConfig,
    resolved_attention_scale,
    resolved_attention_scale_rule,
)
from .model_profiles import NANOLLADA_PROFILE
from .parameterization_calibration import calibration_parameterization_contract, is_d4_root_calibration
from .parameterization_mechanics import BLOCK_MATRIX_ROLES as _BLOCK_MATRIX_ROLES
from .parameterization_mechanics import (
    MECHANICS_EXECUTABLE_PARAMETERIZATION_MODES,
    PARAMETERIZATION_RULE_SET,
    resolve_parameterization_mechanics,
)
from .parameterization_mechanics import initialization_spec as _initialization_spec
from .parameterization_mechanics import nonordinary_initialization_spec as _nonordinary_initialization_spec
from .parameterization_mechanics import parameterization_model_config_kwargs as parameterization_model_config_kwargs
from .parameterization_model_alignment import (
    _resolved_model_config,
    _validate_implemented_nonordinary_mechanics,
    _validate_model_config_alignment,
)
from .parameterization_optimizer_groups import optimizer_group_parameters as optimizer_group_parameters
from .parameterization_optimizer_groups import validate_trainable_inventory as _validate_trainable_inventory
from .parameterization_roles import (
    _ADAMW_NO_DECAY_ROLES,
    _MUON_ADAMW_NO_DECAY_ROLES,
    _OPTIONAL_ADAPTATION_ROLES,
    ordinary_parameter_roles,
)
from .parameterization_roles import (
    PARAMETER_ROLES as PARAMETER_ROLES,
)
from .parameterization_roles import (
    parameter_role as parameter_role,
)
from .parameterization_transfer import (
    d12_transfer_parameterization_contract,
    is_d8_center_transfer,
    is_d8_response_neighborhood,
    is_d12_transfer,
    response_parameterization_contract,
    transfer_parameterization_contract,
)

if TYPE_CHECKING:
    from torch import Tensor


PARAMETERIZATION_CONTRACT_SCHEMA = "nanodllm.parameterization_contract.v1"
# The reserved list is retained in the v1 ordinary contract for digest
# compatibility. New v2 contracts carry ``mechanics_executable_modes`` instead.
PARAMETERIZATION_CONTRACT_SCHEMA_V2 = "nanodllm.parameterization_contract.v2"
RESERVED_PARAMETERIZATION_MODES = ("mup", "completep")


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optimizer_kind_and_group(role: str, config: ExperimentConfig) -> tuple[str, str]:
    if config.model.backbone_profile == NANOLLADA_PROFILE:
        if config.optim.type != "adamw":
            raise ValueError("nanollada_v1 requires AdamW")
        # All supported parameters are matrices, including the input embedding.
        return "adamw", "adamw.decay"
    if config.optim.type == "muon":
        if role in _BLOCK_MATRIX_ROLES:
            return "muon", "muon.block_matrices"
        decay = role not in _MUON_ADAMW_NO_DECAY_ROLES
        return "adamw", "adamw.decay" if decay else "adamw.no_decay"
    decay = role not in _ADAMW_NO_DECAY_ROLES
    return "adamw", "adamw.decay" if decay else "adamw.no_decay"


def _resolved_root_lr(config: ExperimentConfig, optimizer_kind: str, batch_lr_scale: float) -> float:
    if optimizer_kind == "muon":
        root = config.optim.lr if config.optim.muon_lr < 0 else config.optim.muon_lr
    elif optimizer_kind == "adamw":
        root = config.optim.lr if config.optim.adamw_lr < 0 else config.optim.adamw_lr
    else:  # pragma: no cover - the optimizer-kind producer is closed above.
        raise ValueError(f"unknown optimizer kind {optimizer_kind!r}")
    return float(root) * float(batch_lr_scale)


def _parameter_record(
    name: str,
    parameter: nn.Parameter,
    config: ExperimentConfig,
    model_config: DiffusionConfig,
) -> dict[str, object]:
    role = parameter_role(name)
    optimizer_kind, optimizer_group = _optimizer_kind_and_group(role, config)
    decay_multiplier = 0.0 if optimizer_group == "adamw.no_decay" else 1.0
    adam_eps = float(config.optim.adam_eps) if optimizer_kind == "adamw" else None
    return {
        "name": name,
        "role": role,
        "shape": list(parameter.shape),
        "num_parameters": int(parameter.numel()),
        "optimizer_kind": optimizer_kind,
        "optimizer_group": optimizer_group,
        "initialization": _initialization_spec(role, model_config),
        "lr_multiplier": 1.0,
        "weight_decay_multiplier": decay_multiplier,
        "adam_eps_multiplier": 1.0 if adam_eps is not None else None,
        "forward_multiplier": 1.0,
        "prospective_nonordinary_status": (
            "unsupported_until_explicit_adaptation" if role in _OPTIONAL_ADAPTATION_ROLES else "adaptation_not_frozen"
        ),
    }


def _resolve_ordinary_parameterization_contract(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    batch_lr_scale: float,
    _resolved_model_config_override: DiffusionConfig | None = None,
    _excluded_parameter_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Resolve the v1 ordinary contract without changing its established math."""

    if config.parameterization.mode != "ordinary":
        raise ValueError(f"ordinary resolver cannot handle parameterization.mode={config.parameterization.mode!r}")
    if not math.isfinite(float(batch_lr_scale)) or float(batch_lr_scale) <= 0:
        raise ValueError("batch_lr_scale must be finite and positive")
    if _resolved_model_config_override is None:
        model_config = _resolved_model_config(model, config)
    else:
        model_config = _resolved_model_config_override
        _validate_model_config_alignment(model_config, config)

    records = [
        _parameter_record(name, parameter, config, model_config)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(_excluded_parameter_prefixes)
    ]
    _validate_trainable_inventory(
        model,
        records,
        excluded_parameter_prefixes=_excluded_parameter_prefixes,
    )
    roles = ordinary_parameter_roles(model_config)
    by_role = {role: [] for role in roles}
    for record in records:
        by_role[str(record["role"])].append(record)

    parameter_groups: list[dict[str, object]] = []
    for role in by_role:
        role_records = by_role[role]
        if not role_records:
            continue
        first = role_records[0]
        optimizer_kind = str(first["optimizer_kind"])
        optimizer_group = str(first["optimizer_group"])
        if any(
            record["optimizer_kind"] != optimizer_kind or record["optimizer_group"] != optimizer_group
            for record in role_records
        ):
            raise RuntimeError(f"semantic role {role!r} resolves to inconsistent optimizer groups")
        root_lr = _resolved_root_lr(config, optimizer_kind, batch_lr_scale)
        weight_decay_multiplier = float(first["weight_decay_multiplier"])
        parameter_groups.append(
            {
                "role": role,
                "parameter_names": [str(record["name"]) for record in role_records],
                "parameter_count": len(role_records),
                "num_parameters": sum(int(record["num_parameters"]) for record in role_records),
                "initialization": first["initialization"],
                "optimizer_kind": optimizer_kind,
                "optimizer_group": optimizer_group,
                "lr_multiplier": 1.0,
                "resolved_lr": root_lr,
                "weight_decay_multiplier": weight_decay_multiplier,
                "resolved_weight_decay": float(config.optim.weight_decay) * weight_decay_multiplier,
                "adam_eps_multiplier": first["adam_eps_multiplier"],
                "resolved_adam_eps": float(config.optim.adam_eps) if optimizer_kind == "adamw" else None,
                "forward_multiplier": 1.0,
                "prospective_nonordinary_status": first["prospective_nonordinary_status"],
            }
        )

    optimizer_group_ids = (
        ("muon.block_matrices", "adamw.decay", "adamw.no_decay")
        if config.optim.type == "muon"
        else ("adamw.decay", "adamw.no_decay")
    )
    optimizer_groups = []
    for group_id in optimizer_group_ids:
        group_records = [record for record in records if record["optimizer_group"] == group_id]
        if not group_records:
            continue
        optimizer_kind = str(group_records[0]["optimizer_kind"])
        optimizer_groups.append(
            {
                "group_id": group_id,
                "optimizer_kind": optimizer_kind,
                "parameter_names": [str(record["name"]) for record in group_records],
                "parameter_count": len(group_records),
                "num_parameters": sum(int(record["num_parameters"]) for record in group_records),
                "resolved_lr": _resolved_root_lr(config, optimizer_kind, batch_lr_scale),
                "resolved_weight_decay": (0.0 if group_id == "adamw.no_decay" else float(config.optim.weight_decay)),
                "resolved_adam_eps": float(config.optim.adam_eps) if optimizer_kind == "adamw" else None,
            }
        )

    material: dict[str, object] = {
        "schema": PARAMETERIZATION_CONTRACT_SCHEMA,
        "rule_set": PARAMETERIZATION_RULE_SET,
        "mode": "ordinary",
        "adaptation_status": "implemented_identity",
        "launchable": True,
        "reserved_modes": list(RESERVED_PARAMETERIZATION_MODES),
        "base_coordinates": {
            "depth": int(config.parameterization.base_depth),
            "width": int(config.parameterization.base_width),
        },
        "target_coordinates": {
            "depth": int(model_config.n_layer),
            "width": int(model_config.n_embd),
            "depth_multiplier": int(model_config.n_layer) / int(config.parameterization.base_depth),
            "width_multiplier": int(model_config.n_embd) / int(config.parameterization.base_width),
        },
        "model_conventions": {
            "block_norm": "parameter_free_rms_norm_before_attention_and_mlp",
            "qk_norm": "parameter_free_rms_norm_after_rope",
            "final_norm": "trainable_layer_norm_weight_and_bias",
            "embedding_head_tied": bool(model.wte.weight is model.lm_head.weight),
            "attention_scale_configured": float(model_config.attention_scale),
            "attention_scale_resolved": resolved_attention_scale(model_config),
            "attention_scale_rule": resolved_attention_scale_rule(model_config),
            "head_width": int(model_config.n_embd) // int(model_config.n_head),
        },
        "optimizer": {
            "type": config.optim.type,
            "batch_lr_scale": float(batch_lr_scale),
            "root_lr": float(config.optim.lr),
            "root_muon_lr": float(config.optim.muon_lr),
            "root_adamw_lr": float(config.optim.adamw_lr),
            "root_weight_decay": float(config.optim.weight_decay),
            "root_adam_eps": float(config.optim.adam_eps),
        },
        "parameter_records": records,
        "parameter_groups": parameter_groups,
        "optimizer_groups": optimizer_groups,
    }
    if model_config.backbone_profile == NANOLLADA_PROFILE:
        material["model_conventions"].update(
            {
                "backbone_profile": NANOLLADA_PROFILE,
                "input_norm": "parameter_free_rms_norm",
                "final_norm": "parameter_free_rms_norm",
                "mlp_type": "squared_relu",
            }
        )
    return {**material, "contract_sha256": canonical_sha256(material)}


def _resolve_nonordinary_parameterization_contract(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    batch_lr_scale: float,
    _resolved_model_config_override: DiffusionConfig | None = None,
) -> dict[str, object]:
    if not math.isfinite(float(batch_lr_scale)) or float(batch_lr_scale) <= 0:
        raise ValueError("batch_lr_scale must be finite and positive")
    mechanics = resolve_parameterization_mechanics(config)
    if _resolved_model_config_override is None:
        model_config = _resolved_model_config(model, config)
    else:
        model_config = _resolved_model_config_override
        _validate_model_config_alignment(model_config, config)
    _validate_implemented_nonordinary_mechanics(model, model_config)
    if not hasattr(model, "wte") or not hasattr(model, "lm_head"):
        raise TypeError("mup and completep require explicit token embedding and language-model head modules")
    if model.wte.weight is model.lm_head.weight:
        raise ValueError("the frozen NanoDLLM mup and completep adaptation requires untied embedding and head")

    optimizer_multipliers = mechanics["optimizer_multipliers"]
    forward = mechanics["forward"]
    target = mechanics["target_coordinates"]
    assert isinstance(optimizer_multipliers, dict)
    assert isinstance(forward, dict)
    assert isinstance(target, dict)
    mode = str(mechanics["mode"])
    hidden_lr_multiplier = float(optimizer_multipliers["hidden_lr"])
    hidden_weight_decay_multiplier = float(optimizer_multipliers["hidden_weight_decay"])
    residual_eps_multiplier = float(optimizer_multipliers["residual_block_epsilon"])
    boundary_eps_multiplier = float(optimizer_multipliers["embedding_final_norm_unembedding_epsilon"])
    residual_multiplier = float(forward["attention_residual_multiplier"])
    output_multiplier = float(forward["unembedding_multiplier"])

    records: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = parameter_role(name)
        if role in _OPTIONAL_ADAPTATION_ROLES:
            raise ValueError(f"{mode} has no frozen adaptation for optional role {role!r} ({name})")
        hidden = role in _BLOCK_MATRIX_ROLES
        lr_multiplier = hidden_lr_multiplier if hidden else 1.0
        if role in _ADAMW_NO_DECAY_ROLES:
            weight_decay_multiplier = 0.0
        elif hidden:
            weight_decay_multiplier = hidden_weight_decay_multiplier
        else:
            weight_decay_multiplier = 1.0
        eps_multiplier = residual_eps_multiplier if hidden else boundary_eps_multiplier
        if ar_width_mup.active(config) and role == "token_embedding":
            weight_decay_multiplier = 1.0
        if nanollada_completep.is_completep(config):
            if role == "token_embedding":
                weight_decay_multiplier = 1.0
            elif role == "output_head":
                lr_multiplier = float(optimizer_multipliers["unembedding_lr"])
                weight_decay_multiplier = float(optimizer_multipliers["unembedding_weight_decay"])
        if role == "output_head":
            forward_multiplier = output_multiplier
        elif role in {"attention_output", "mlp_output"}:
            forward_multiplier = residual_multiplier
        else:
            forward_multiplier = 1.0
        records.append(
            {
                "name": name,
                "role": role,
                "shape": list(parameter.shape),
                "num_parameters": int(parameter.numel()),
                "optimizer_kind": "adamw",
                "optimizer_group": f"adamw.role.{role}",
                "initialization": _nonordinary_initialization_spec(role, model_config),
                "lr_multiplier": lr_multiplier,
                "weight_decay_multiplier": weight_decay_multiplier,
                "adam_eps_multiplier": eps_multiplier,
                "forward_multiplier": forward_multiplier,
                "prospective_nonordinary_status": "implemented_frozen_adaptation",
            }
        )
    _validate_trainable_inventory(model, records)

    root_lr = _resolved_root_lr(config, "adamw", batch_lr_scale)
    by_role = {role: [record for record in records if record["role"] == role] for role in PARAMETER_ROLES}
    parameter_groups: list[dict[str, object]] = []
    optimizer_groups: list[dict[str, object]] = []
    for role in PARAMETER_ROLES:
        role_records = by_role[role]
        if not role_records:
            continue
        first = role_records[0]
        names = [str(record["name"]) for record in role_records]
        lr_multiplier = float(first["lr_multiplier"])
        weight_decay_multiplier = float(first["weight_decay_multiplier"])
        eps_multiplier = float(first["adam_eps_multiplier"])
        group = {
            "role": role,
            "parameter_names": names,
            "parameter_count": len(role_records),
            "num_parameters": sum(int(record["num_parameters"]) for record in role_records),
            "initialization": first["initialization"],
            "optimizer_kind": "adamw",
            "optimizer_group": str(first["optimizer_group"]),
            "lr_multiplier": lr_multiplier,
            "resolved_lr": root_lr * lr_multiplier,
            "weight_decay_multiplier": weight_decay_multiplier,
            "resolved_weight_decay": float(config.optim.weight_decay) * weight_decay_multiplier,
            "adam_eps_multiplier": eps_multiplier,
            "resolved_adam_eps": float(config.optim.adam_eps) * eps_multiplier,
            "forward_multiplier": float(first["forward_multiplier"]),
            "prospective_nonordinary_status": "implemented_frozen_adaptation",
        }
        parameter_groups.append(group)
        optimizer_groups.append(
            {
                "group_id": str(first["optimizer_group"]),
                "role": role,
                "optimizer_kind": "adamw",
                "parameter_names": names,
                "parameter_count": len(role_records),
                "num_parameters": group["num_parameters"],
                "lr_multiplier": lr_multiplier,
                "resolved_lr": group["resolved_lr"],
                "weight_decay_multiplier": weight_decay_multiplier,
                "resolved_weight_decay": group["resolved_weight_decay"],
                "adam_eps_multiplier": eps_multiplier,
                "resolved_adam_eps": group["resolved_adam_eps"],
            }
        )

    material: dict[str, object] = {
        "schema": PARAMETERIZATION_CONTRACT_SCHEMA_V2,
        "rule_set": str(mechanics["rule_set"]),
        "mode": mode,
        "adaptation_status": "implemented_mechanics_scientific_root_calibration_pending",
        "mechanics_executable": True,
        "compiled_bf16_qualification_status": "pending",
        "scientific_training_authorized": False,
        "mechanics_executable_modes": list(MECHANICS_EXECUTABLE_PARAMETERIZATION_MODES),
        "mechanics": mechanics,
        "base_coordinates": {
            "depth": int(config.parameterization.base_depth),
            "width": int(config.parameterization.base_width),
        },
        "target_coordinates": {
            "depth": int(model_config.n_layer),
            "width": int(model_config.n_embd),
            "depth_multiplier": float(target["depth_multiplier"]),
            "width_multiplier": float(target["width_multiplier"]),
        },
        "model_conventions": {
            "block_norm": "parameter_free_rms_norm_before_attention_and_mlp",
            "qk_norm": "parameter_free_rms_norm_after_rope",
            "final_norm": "trainable_layer_norm_weight_and_bias",
            "embedding_head_tied": False,
            "attention_scale_configured": float(config.model.attention_scale),
            "attention_scale_resolved": resolved_attention_scale(model_config),
            "attention_scale_rule": resolved_attention_scale_rule(model_config),
            "head_width": int(model_config.n_embd) // int(model_config.n_head),
            "position_encoding": "rope",
            "mlp": "swiglu",
            "objective": "mdlm",
            "backbone_bias": False,
            "residual_output_projection_init": str(model_config.residual_output_projection_init),
        },
        "optimizer": {
            "type": "adamw",
            "batch_lr_scale": float(batch_lr_scale),
            "root_lr": float(config.optim.lr),
            "root_adamw_lr": float(config.optim.adamw_lr),
            "root_weight_decay": float(config.optim.weight_decay),
            "root_adam_eps": float(config.optim.adam_eps),
        },
        "parameter_records": records,
        "parameter_groups": parameter_groups,
        "optimizer_groups": optimizer_groups,
    }
    return {**material, "contract_sha256": canonical_sha256(material)}


def resolve_parameterization_contract(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    batch_lr_scale: float,
    _resolved_model_config_override: DiffusionConfig | None = None,
    _excluded_parameter_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Resolve the mode-specific named-parameter and optimizer contract."""

    if config.parameterization.mode == "ordinary":
        contract = _resolve_ordinary_parameterization_contract(
            model,
            replace(config, optim=replace(config.optim, type="adamw"))
            if config.optim.type == "nanochat_muon"
            else config,
            batch_lr_scale=batch_lr_scale,
            _resolved_model_config_override=_resolved_model_config_override,
            _excluded_parameter_prefixes=_excluded_parameter_prefixes,
        )
        if config.optim.type == "nanochat_muon":
            from . import muon_residual_depth
            from .nanochat_muon import finish_contract

            if muon_residual_depth.active(config):
                muon_residual_depth.validate_model(model)
            return finish_contract(contract, config, batch_lr_scale=batch_lr_scale)
    elif config.parameterization.mode in {"mup", "completep"}:
        contract = _resolve_nonordinary_parameterization_contract(
            model,
            config,
            batch_lr_scale=batch_lr_scale,
            _resolved_model_config_override=_resolved_model_config_override,
        )
    else:
        raise ValueError(f"unknown parameterization mode {config.parameterization.mode!r}")
    mechanics = resolve_parameterization_mechanics(config)
    if ar_width_mup.active(config):
        return ar_width_mup.finish_contract(contract)
    if nanollada_completep.is_completep(config):
        return nanollada_completep.finish_contract(contract, scope=config.parameterization.execution_scope)
    if is_d4_root_calibration(config):
        return calibration_parameterization_contract(contract, mechanics)
    if is_d4_hidden_init_confirmation(config):
        return hidden_init_confirmation_parameterization_contract(contract, mechanics)
    if is_d8_center_transfer(config):
        return transfer_parameterization_contract(contract, mechanics)
    if is_d8_response_neighborhood(config):
        return response_parameterization_contract(contract, mechanics)
    if is_d12_transfer(config):
        return d12_transfer_parameterization_contract(contract, mechanics)
    if is_d20_completep_tpp20(config):
        return d20_completep_parameterization_contract(contract, mechanics)
    if is_completep_hidden_init_transfer(config):
        return completep_hidden_init_transfer_parameterization_contract(contract, mechanics)
    if config.parameterization.execution_scope == D8_BATCH_OBJECTIVE_SCOPE:
        return d8_batch_objective_contract(contract, mechanics, probe=config.recipe.name.endswith("-probe"))
    return contract


def bind_parameterization_contract(config: ExperimentConfig, contract: Mapping[str, object]) -> None:
    """Bind a derived contract digest into the resolved configuration."""

    observed = contract.get("contract_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError("parameterization contract has no valid contract_sha256")
    if config.parameterization.contract_binding == "legacy_unbound":
        if config.parameterization.contract_sha256:
            raise ValueError("legacy_unbound parameterization cannot carry a contract digest")
        return
    if config.parameterization.contract_binding != "required":
        raise ValueError("unknown parameterization contract binding policy")
    configured = config.parameterization.contract_sha256
    if configured and configured != observed:
        raise ValueError(
            "parameterization.contract_sha256 does not match the resolved named-parameter contract: "
            f"configured={configured}, resolved={observed}"
        )
    config.parameterization.contract_sha256 = observed


def validate_checkpoint_parameterization_contract(
    checkpoint: Mapping[str, object],
    model: nn.Module,
    *,
    allow_token_edit_extensions: bool = False,
    allow_block_self_conditioning: bool = False,
) -> dict[str, object] | None:
    from .parameterization_checkpoint import validate_checkpoint_parameterization_contract as validate

    return validate(
        checkpoint,
        model,
        allow_token_edit_extensions=allow_token_edit_extensions,
        allow_block_self_conditioning=allow_block_self_conditioning,
    )


def validate_checkpoint_eval_architecture_fingerprint(
    checkpoint: Mapping[str, object],
    model: nn.Module,
) -> str | None:
    from .parameterization_checkpoint import validate_checkpoint_eval_architecture_fingerprint as validate

    return validate(checkpoint, model)


def _trainable_names_and_parameters(model: nn.Module) -> tuple[list[str], list[nn.Parameter]]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    return [name for name, _ in named], [parameter for _, parameter in named]


def parameterization_gradient_rms_metrics(
    model: nn.Module,
    contract: Mapping[str, object],
) -> dict[str, float]:
    """Return zero-filled pre-clipping gradient RMS and coverage by semantic role."""

    named = dict(model.named_parameters())
    metrics: dict[str, float] = {}
    for group in contract.get("parameter_groups", []):
        role = str(group["role"])
        parameters = [named[name] for name in group["parameter_names"]]
        grads = [parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None]
        role_numel = sum(int(parameter.numel()) for parameter in parameters)
        present_numel = sum(int(gradient.numel()) for gradient in grads)
        squared = sum(
            (gradient.square().sum() for gradient in grads), start=torch.tensor(0.0, device=parameters[0].device)
        )
        metrics[f"parameterization_{role}_gradient_rms"] = math.sqrt(float(squared.item()) / max(1, role_numel))
        metrics[f"parameterization_{role}_gradient_present_numel_fraction"] = present_numel / max(1, role_numel)
    return metrics


def parameterization_parameter_update_rms_metrics(
    model: nn.Module,
    contract: Mapping[str, object],
    pre_step_parameters: Sequence[Tensor],
) -> dict[str, float]:
    """Return post-step parameter/update RMS for every present semantic role."""

    names, parameters = _trainable_names_and_parameters(model)
    if len(pre_step_parameters) != len(parameters):
        raise ValueError(
            f"pre-step snapshot has {len(pre_step_parameters)} tensors for {len(parameters)} trainable parameters"
        )
    pre_by_name = dict(zip(names, pre_step_parameters, strict=True))
    named = dict(zip(names, parameters, strict=True))
    metrics: dict[str, float] = {}
    for group in contract.get("parameter_groups", []):
        role = str(group["role"])
        role_names = group["parameter_names"]
        numel = sum(int(named[name].numel()) for name in role_names)
        parameter_squared = sum(
            (named[name].detach().float().square().sum() for name in role_names),
            start=torch.tensor(0.0, device=parameters[0].device),
        )
        update_squared = sum(
            ((named[name].detach().float() - pre_by_name[name].float()).square().sum() for name in role_names),
            start=torch.tensor(0.0, device=parameters[0].device),
        )
        parameter_rms = math.sqrt(float(parameter_squared.item()) / max(1, numel))
        update_rms = math.sqrt(float(update_squared.item()) / max(1, numel))
        metrics[f"parameterization_{role}_parameter_rms"] = parameter_rms
        metrics[f"parameterization_{role}_update_rms"] = update_rms
        metrics[f"parameterization_{role}_update_to_parameter_rms"] = update_rms / max(parameter_rms, 1e-30)
    return metrics
