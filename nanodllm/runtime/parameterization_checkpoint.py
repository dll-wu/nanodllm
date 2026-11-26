"""Checkpoint reconstruction and architecture binding for parameterization."""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch.nn as nn

from .config import config_from_dict
from .model import DiffusionConfig


def validate_checkpoint_parameterization_contract(
    checkpoint: Mapping[str, object],
    model: nn.Module,
    *,
    allow_token_edit_extensions: bool = False,
    allow_block_self_conditioning: bool = False,
) -> dict[str, object] | None:
    """Reconstruct a bound contract while accepting canonical historical forms."""

    from .parameterization import bind_parameterization_contract, canonical_sha256, resolve_parameterization_contract

    diagnostic_status = checkpoint.get("parameterization_status")
    if diagnostic_status is not None:
        if diagnostic_status != "deliberately_unbound_diagnostic_only_nonloadable":
            raise ValueError("checkpoint has an unknown parameterization status")
        raise ValueError("diagnostic-only checkpoint is deliberately unbound and cannot be loaded")

    raw_train_config = checkpoint.get("train_config")
    resume_contract = checkpoint.get("resume_contract")
    trajectory = resume_contract.get("trajectory") if isinstance(resume_contract, dict) else None
    record = trajectory.get("record") if isinstance(trajectory, dict) else None
    resolved_trajectory_config = record.get("resolved_trajectory_config") if isinstance(record, dict) else None
    trajectory_parameterization = (
        resolved_trajectory_config.get("parameterization") if isinstance(resolved_trajectory_config, dict) else None
    )
    raw_parameterization = raw_train_config.get("parameterization") if isinstance(raw_train_config, dict) else None
    if raw_parameterization is None and trajectory_parameterization is None:
        return None
    if raw_parameterization is None or trajectory_parameterization is None:
        raise ValueError("checkpoint train config and trajectory must carry the same parameterization claim")
    if not isinstance(raw_train_config, dict):
        raise TypeError("checkpoint parameterization binding requires a train_config mapping")
    if not isinstance(raw_parameterization, dict):
        raise TypeError("checkpoint train_config.parameterization must be a mapping")
    if not isinstance(trajectory_parameterization, dict) or trajectory_parameterization != raw_parameterization:
        raise ValueError("checkpoint train config and trajectory disagree on the parameterization contract")
    binding = raw_parameterization.get("contract_binding")
    if binding == "legacy_unbound":
        historical_unbound = {
            "mode": "ordinary",
            "base_depth": 4,
            "base_width": 256,
            "contract_binding": "legacy_unbound",
            "contract_sha256": "",
        }
        current_unbound = {
            **historical_unbound,
            "hidden_init_std_multiplier": 1.0,
        }
        scoped_unbound = {**current_unbound, "execution_scope": "ordinary"}
        if raw_parameterization not in (historical_unbound, current_unbound, scoped_unbound):
            raise ValueError("checkpoint legacy_unbound parameterization claim is noncanonical")
        return None
    if binding != "required":
        raise ValueError("checkpoint parameterization contract binding policy is unknown")
    embedded_sha256 = raw_parameterization.get("contract_sha256", "")
    if not embedded_sha256:
        raise ValueError("checkpoint parameterization section has no bound contract_sha256")
    if (
        not isinstance(embedded_sha256, str)
        or len(embedded_sha256) != 64
        or any(character not in "0123456789abcdef" for character in embedded_sha256)
    ):
        raise ValueError("checkpoint parameterization.contract_sha256 is malformed")

    if not isinstance(record, dict) or trajectory.get("sha256") != canonical_sha256(record):
        raise ValueError("checkpoint parameterization binding requires a valid canonical trajectory record")
    derived_budget = record.get("derived_training_budget")
    total_batch_size = derived_budget.get("total_batch_size") if isinstance(derived_budget, dict) else None
    if not isinstance(total_batch_size, int) or isinstance(total_batch_size, bool) or total_batch_size <= 0:
        raise ValueError("checkpoint parameterization binding has no valid resolved total batch size")

    embedded_config = config_from_dict(raw_train_config)
    raw_resolved_model_config = checkpoint.get("config")
    if not isinstance(raw_resolved_model_config, dict):
        raise ValueError("checkpoint parameterization binding requires a resolved model config")
    try:
        checkpoint_model_config = DiffusionConfig(**raw_resolved_model_config)
    except TypeError as exc:
        raise ValueError("checkpoint parameterization resolved model config is malformed") from exc
    canonical_model_config = {
        field: getattr(checkpoint_model_config, field) for field in checkpoint_model_config.__dataclass_fields__
    }
    # Missing profile is the historical architecture, never the opt-in bundle.
    # Normalize only additive disabled defaults; keep other completeness checks.
    if "backbone_profile" not in raw_resolved_model_config:
        raw_resolved_model_config = {**raw_resolved_model_config, "backbone_profile": "nanodllm"}
    if "bd3lm_self_conditioning_dim" not in raw_resolved_model_config:
        raw_resolved_model_config = {**raw_resolved_model_config, "bd3lm_self_conditioning_dim": 0}
    for field, default in {
        "mtp_mode": "none",
        "recurrent_core_start": 0,
        "recurrent_core_length": 0,
        "recurrent_iterations": 1,
    }.items():
        if field not in raw_resolved_model_config:
            raw_resolved_model_config = {**raw_resolved_model_config, field: default}
    token_edit_extension_defaults = {
        "token_edit_adapter_bottleneck_dim": 0,
        "token_edit_adapter_top_layers": 0,
        "token_edit_provenance_embedding": False,
        "token_edit_provenance_confidence_bins": 0,
    }
    canonical_without_token_edit_extensions = {
        field: value for field, value in canonical_model_config.items() if field not in token_edit_extension_defaults
    }
    historical_runtime_fields = {
        "parameterization_mode",
        "parameterization_rule_set",
        "parameterization_base_depth",
        "parameterization_base_width",
        "parameterization_mechanics_sha256",
        "parameterization_hidden_init_std",
        "parameterization_embedding_head_init_std",
        "parameterization_residual_multiplier",
        "parameterization_output_multiplier",
        "parameterization_attention_scale",
        "parameterization_attention_scale_rule",
    }
    historical_ordinary_model_config = {
        field: value
        for field, value in canonical_model_config.items()
        if field not in historical_runtime_fields and field not in token_edit_extension_defaults
    }
    model_config_is_canonical = raw_resolved_model_config in (
        canonical_model_config,
        canonical_without_token_edit_extensions,
    )
    model_config_is_historical_ordinary = (
        raw_parameterization.get("mode") == "ordinary" and raw_resolved_model_config == historical_ordinary_model_config
    )
    if not model_config_is_canonical and not model_config_is_historical_ordinary:
        raise ValueError("checkpoint parameterization resolved model config is incomplete or noncanonical")
    checkpoint_has_token_edit_extensions = any(
        getattr(checkpoint_model_config, field) != default for field, default in token_edit_extension_defaults.items()
    )
    excluded_parameter_prefixes = (
        ("token_edit_adapters.", "token_edit_provenance_embedding.")
        if allow_token_edit_extensions and not checkpoint_has_token_edit_extensions
        else ()
    )
    if allow_block_self_conditioning and checkpoint_model_config.bd3lm_self_conditioning_dim == 0:
        excluded_parameter_prefixes += ("block_self_conditioning.",)
    batch_lr_scale = (
        (total_batch_size / int(embedded_config.train.base_total_batch_size)) ** 0.5
        if embedded_config.train.scale_lr_by_batch
        else 1.0
    )
    resolved = resolve_parameterization_contract(
        model,
        embedded_config,
        batch_lr_scale=batch_lr_scale,
        _resolved_model_config_override=checkpoint_model_config,
        _excluded_parameter_prefixes=excluded_parameter_prefixes,
    )
    if resolved["contract_sha256"] != embedded_sha256:
        raise ValueError(
            "checkpoint parameterization contract does not reconstruct: "
            f"checkpoint={embedded_sha256}, resolved={resolved['contract_sha256']}"
        )
    bind_parameterization_contract(embedded_config, resolved)
    return resolved


def validate_checkpoint_eval_architecture_fingerprint(
    checkpoint: Mapping[str, object],
    model: nn.Module,
) -> str | None:
    """Require any trajectory-bearing checkpoint's architecture to equal active eval."""

    from .parameterization import canonical_sha256

    resume_contract = checkpoint.get("resume_contract")
    if resume_contract is None:
        return None
    if not isinstance(resume_contract, dict):
        raise TypeError("checkpoint resume_contract must be a mapping or null")
    trajectory = resume_contract.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("checkpoint resume contract has no trajectory mapping")
    record = trajectory.get("record")
    if not isinstance(record, dict) or trajectory.get("sha256") != canonical_sha256(record):
        raise ValueError("checkpoint eval requires a valid canonical trajectory record")
    observed = record.get("architecture_fingerprint_sha256")
    active = model.architecture_fingerprint().get("fingerprint_sha256")
    if not isinstance(observed, str) or re.fullmatch(r"[0-9a-f]{64}", observed) is None:
        raise ValueError("checkpoint trajectory has no valid architecture fingerprint")
    if observed != active:
        raise ValueError(
            "checkpoint architecture fingerprint does not match the active eval model: "
            f"checkpoint={observed}, active={active}"
        )
    return observed
