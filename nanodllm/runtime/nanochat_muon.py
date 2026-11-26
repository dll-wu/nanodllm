"""Pinned optimizer with historical single-rank and opt-in replicated DDP scopes."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import torch

from . import muon_residual_depth
from . import nanochat_backbone_config as backbone
from .core_task_data import json_digest
from .nanochat_muon_schedule import contract_fields as schedule_contract_fields
from .nanochat_muon_schedule import validate_config as validate_schedules
from .nanochat_reference import SOURCE_HASHES, UPSTREAM_COMMIT, _module, verified_source

VARIANT = "nanochat-92d63d4-constant-momentum-decay-v1"
MATRIX_ROLES = {"attention_query", "attention_key", "attention_value", "attention_output", "mlp_up", "mlp_output"}
BLOCK_SCOPE = "single_rank_causal_block_d4"
D8_BLOCK_SCOPE = "single_rank_causal_block_d8"
D12_SCOPE = "ddp_replicated_plain_d12"
D12_ATTRIBUTION_SCOPE = "ddp_replicated_plain_d12_attribution"
D20_AR_SCOPE = "ddp_replicated_plain_d20_ar"
D20_SCOPE = "ddp_replicated_plain_d20"
DDP_BLOCK_SHAPES = {
    "ddp_replicated_causal_block_d12": (12, 768),
    "ddp_replicated_causal_block_d16": (16, 1024),
    "ddp_replicated_causal_block_d20": (20, 1280),
}
DDP_SCOPES = {"ddp_replicated_d4_d8", D12_SCOPE, D12_ATTRIBUTION_SCOPE, D20_AR_SCOPE, D20_SCOPE, *DDP_BLOCK_SHAPES}


def validate_config(config) -> None:
    validate_schedules(config)
    if config.optim.type != "nanochat_muon":
        return
    o, m, p = config.optim, config.model, config.parameterization
    if o.nanochat_execution_scope not in {"single_rank", *DDP_SCOPES, BLOCK_SCOPE, D8_BLOCK_SCOPE, backbone.SCOPE}:
        raise ValueError("unknown nanochat_execution_scope")
    new_backbone = m.backbone_profile == backbone.PROFILE
    if new_backbone != (o.nanochat_execution_scope == backbone.SCOPE):
        raise ValueError("nanochat backbone and optimizer scope must be admitted together")
    if new_backbone:
        backbone.validate_experiment(config)
        if (
            p.execution_scope != "ordinary"
            or o.nanochat_matrix_optimizer != "muon"
            or (config.runtime.device_type != "cpu" and config.train.total_batch_size != 65536)
        ):
            raise ValueError("nanochat backbone screen requires ordinary Muon with explicit EB65536")
        if config.runtime.device_type != "cpu" and (m.n_head != 2 or m.max_seq_len != 1024):
            raise ValueError("nanochat backbone GPU screen requires heads2/context1024")
    shapes = {(p.base_depth, p.base_width)}
    if o.nanochat_execution_scope == "ddp_replicated_d4_d8":
        shapes = {(4, 256), (8, 512)}
    if o.nanochat_execution_scope in {D12_SCOPE, D12_ATTRIBUTION_SCOPE}:
        shapes = {(12, 768)}
        if (
            p.execution_scope != "ordinary"
            or (o.nanochat_execution_scope == D12_SCOPE and o.nanochat_matrix_optimizer != "muon")
            or config.train.token_edit_corruption_batch_fraction
        ):
            raise ValueError("D12 Muon scope admits ordinary plain AR/MDLM only, not editing or depth scaling")
        if config.runtime.device_type != "cpu" and (
            m.n_head != 12 or m.mlp_hidden_dim != 3072 or m.max_seq_len != 1024
        ):
            raise ValueError("D12 Muon GPU scope requires heads12/MLP3072/context1024")
    if o.nanochat_execution_scope == D20_AR_SCOPE:
        shapes = {(20, 1280)}
        if not m.causal or p.execution_scope != "ordinary" or o.nanochat_matrix_optimizer != "muon":
            raise ValueError("D20 Muon scope admits ordinary plain AR only, not diffusion or depth scaling")
    if o.nanochat_execution_scope == D20_SCOPE:
        shapes = {(20, 1280)}
        if (
            p.execution_scope != "ordinary"
            or o.nanochat_matrix_optimizer != "muon"
            or config.train.token_edit_corruption_batch_fraction
        ):
            raise ValueError("D20 baseline Muon scope admits ordinary plain AR/MDLM only, not editing or depth scaling")
        if config.runtime.device_type != "cpu" and (
            m.n_head != 20 or m.mlp_hidden_dim != 5120 or m.max_seq_len != 1024
        ):
            raise ValueError("D20 baseline Muon GPU scope requires heads20/MLP5120/context1024")
    block = o.nanochat_execution_scope in {BLOCK_SCOPE, D8_BLOCK_SCOPE, *DDP_BLOCK_SHAPES} or (
        new_backbone and config.noise.type == "bd3lm"
    )
    if new_backbone:
        shapes = {(4, 256)}
    if block:
        shapes = {(8, 512)} if o.nanochat_execution_scope == D8_BLOCK_SCOPE else {(4, 256)}
        if o.nanochat_execution_scope in DDP_BLOCK_SHAPES:
            depth, width = DDP_BLOCK_SHAPES[o.nanochat_execution_scope]
            shapes = {(depth, width)}
            if (
                o.nanochat_matrix_optimizer != "muon"
                or m.causal
                or config.train.token_edit_corruption_batch_fraction
                or config.train.bd3lm_causal_ar_loss_weight
            ) or (
                config.runtime.device_type != "cpu"
                and (m.n_head, m.mlp_hidden_dim, m.max_seq_len) != (depth, 4 * width, 1024)
            ):
                raise ValueError("distributed block Muon requires its exact plain heads/MLP/context1024 shape")
        if (
            config.noise.type != "bd3lm"
            or m.bd3lm_clean_stream_mode != "causal"
            or m.bd3lm_block_size <= 0
            or p.execution_scope != "ordinary"
            or config.train.bd3lm_uniform_t2t_batch_fraction
            or config.train.bd3lm_recipe_mechanism != "none"
        ):
            raise ValueError("block Muon scope requires plain causal-clean masked diffusion without recipe extensions")
    if o.nanochat_matrix_optimizer not in {"muon", "adamw"}:
        raise ValueError("nanochat_matrix_optimizer must be muon or adamw")
    if o.nanochat_matrix_optimizer == "adamw" and (
        not all(math.isfinite(b) and 0 <= b < 1 for b in (o.adam_beta1, o.adam_beta2))
        or not math.isfinite(o.adam_eps)
        or o.adam_eps <= 0
    ):
        raise ValueError("nanochat matrix AdamW requires valid betas and epsilon")
    if (
        m.backbone_profile not in {"nanollada_v1", backbone.PROFILE}
        or p.mode != "ordinary"
        or p.execution_scope not in {"ordinary", muon_residual_depth.SCOPE}
        or (config.runtime.device_type != "cpu" and (m.depth, m.n_embd) not in shapes)
        or (not block and (config.noise.type != "mdlm" or m.bd3lm_block_size))
    ):
        raise ValueError("nanochat_muon requires a plain ordinary shape admitted by its execution scope, AR or MDLM")
    if muon_residual_depth.active(config):
        muon_residual_depth.validate_config(config)
    if o.nanochat_checkout and not Path(o.nanochat_checkout).is_absolute():
        raise ValueError("nanochat_muon requires an explicit absolute pinned source checkout")
    if config.runtime.dtype not in {"float32", "bfloat16"} or config.train.scale_lr_by_batch:
        raise ValueError("nanochat_muon requires float32/BF16 and explicit non-automatic batch LR roots")
    if o.muon_lr != -1 or o.adamw_lr != -1:
        raise ValueError("nanochat_muon uses optim.lr and explicit embedding/head LRs, not legacy overrides")
    for name in ("lr", "nanochat_embedding_lr", "nanochat_head_lr"):
        value = getattr(o, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"optim.{name} must be finite and positive")
    for name in ("weight_decay", "nanochat_embedding_weight_decay", "nanochat_head_weight_decay"):
        value = getattr(o, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"optim.{name} must be finite and nonnegative")
    if not 0 <= o.nanochat_muon_beta2 < 1 or not 0 <= o.muon_momentum < 1 or o.muon_ns_steps != 5:
        raise ValueError("nanochat_muon requires finite momentum/beta2 in [0,1) and five Polar Express steps")


def finish_contract(base: dict, config, *, batch_lr_scale: float) -> dict:
    validate_config(config)
    if batch_lr_scale != 1.0:
        raise ValueError("nanochat_muon first batch requires explicit LRs without automatic batch scaling")
    result = copy.deepcopy(base)
    result.pop("contract_sha256")
    o = config.optim
    groups = {}
    for record in result["parameter_records"]:
        role, shape = record["role"], record["shape"]
        if len(shape) != 2 and config.model.backbone_profile != backbone.PROFILE:
            raise ValueError("plain nanochat_muon scope requires matrix parameters only")
        if role in MATRIX_ROLES or role == "value_gate":
            kind = o.nanochat_matrix_optimizer
            group_id = kind + ".shape." + "x".join(map(str, shape))
            lr, wd, eps = o.lr, o.weight_decay, None if kind == "muon" else o.adam_eps
            settings = (
                {"momentum": o.muon_momentum, "ns_steps": 5, "beta2": o.nanochat_muon_beta2}
                if kind == "muon"
                else {"betas": [o.adam_beta1, o.adam_beta2], "eps": eps}
            )
        elif role in {"token_embedding", "output_head"}:
            embedding = role == "token_embedding"
            group_id, kind, eps = "adamw." + role, "adamw", 1e-10
            lr = o.nanochat_embedding_lr if embedding else o.nanochat_head_lr
            wd = o.nanochat_embedding_weight_decay if embedding else o.nanochat_head_weight_decay
            settings = {"betas": [0.8, 0.995 if embedding else 0.96], "eps": eps}
        elif config.model.backbone_profile == backbone.PROFILE and role in backbone.ROLES:
            group_id, kind, lr, wd, eps, settings = backbone.auxiliary_optimizer(role, o)
        else:
            raise ValueError(f"unqualified nanochat_muon role: {role}")
        fields = {
            "optimizer_kind": kind,
            "optimizer_group": group_id,
            "lr_multiplier": lr / o.lr,
            "resolved_lr": lr,
            "resolved_weight_decay": wd,
            "resolved_adam_eps": eps,
            "weight_decay_multiplier": 1.0,
            "adam_eps_multiplier": 1.0 if eps else None,
        }
        record.update(fields)
        group = groups.setdefault(
            group_id,
            {
                **fields,
                "group_id": group_id,
                "parameter_names": [],
                "parameter_count": 0,
                "num_parameters": 0,
                "settings": settings,
                "shape_lr_factor": math.sqrt(max(1, shape[0] / shape[1])) if kind == "muon" else 1.0,
            },
        )
        group["parameter_names"].append(record["name"])
        group["parameter_count"] += 1
        group["num_parameters"] += record["num_parameters"]
    records = {r["name"]: r for r in result["parameter_records"]}
    for role in result["parameter_groups"]:
        first = records[role["parameter_names"][0]]
        for key in fields:
            role[key] = first[key]
    # Keep the matrix root first: generic trainer LR telemetry names the first
    # group as primary. Auxiliary roots remain independently serialized.
    result["optimizer_groups"] = sorted(groups.values(), key=lambda g: ".shape." not in g["group_id"])
    result["optimizer"] = {
        "type": o.type,
        "variant": VARIANT,
        "upstream_commit": UPSTREAM_COMMIT,
        "source_sha256": SOURCE_HASHES["nanochat/optim.py"],
        "license_sha256": SOURCE_HASHES["LICENSE"],
        "root_lr": o.lr,
        "batch_lr_scale": 1.0,
        "momentum_schedule": "constant",
        "weight_decay_schedule": "constant",
        "lr_schedule": "shared NanoDLLM fractional schedule, independently scaled role roots",
        "shape_factor_applies_to": "Muon update and cautious decay, once inside upstream kernel",
        "distributed_scope": "single_rank_only",
        "width_depth_transfer": "not_implemented",
    }
    if o.nanochat_matrix_optimizer == "adamw":
        result["optimizer"].update(
            variant=VARIANT + "-adamw-matrices",
            matrix_optimizer="adamw",
            shape_factor_applies_to="none; ordinary AdamW update and decoupled decay",
        )
    if o.nanochat_execution_scope == "ddp_replicated_d4_d8":
        result["optimizer"].update(
            distributed_scope="ddp_replicated",
            distribution_adapter="local_native_kernels_v1",
            width_depth_transfer="empirical_d4_d8_screen_not_completep",
        )
    if o.nanochat_execution_scope == D12_SCOPE:
        result["optimizer"].update(
            distributed_scope="ddp_replicated",
            distribution_adapter="local_native_kernels_v1",
            execution_scope=D12_SCOPE,
            width_depth_transfer="d12_empirical_lr_neighborhood_not_completep",
        )
    if o.nanochat_execution_scope == D12_ATTRIBUTION_SCOPE:
        result["optimizer"].update(
            distributed_scope="ddp_replicated",
            distribution_adapter="local_native_kernels_v1",
            execution_scope=D12_ATTRIBUTION_SCOPE,
            width_depth_transfer="d12_matched_auxiliary_optimizer_attribution_not_completep",
        )
    if o.nanochat_execution_scope == D20_AR_SCOPE:
        result["optimizer"].update(
            distributed_scope="ddp_replicated",
            distribution_adapter="local_native_kernels_v1",
            execution_scope=D20_AR_SCOPE,
            width_depth_transfer="d20_ar_empirical_batch_width_transfer_not_completep",
        )
    if o.nanochat_execution_scope == D20_SCOPE:
        result["optimizer"].update(
            distributed_scope="ddp_replicated",
            distribution_adapter="local_native_kernels_v1",
            execution_scope=D20_SCOPE,
            width_depth_transfer="d12_to_d20_fixed_matrix_root_aux_inverse_sqrt_width_not_completep",
        )
    if muon_residual_depth.active(config):
        muon_residual_depth.finish_contract(result)
    if o.nanochat_execution_scope in {BLOCK_SCOPE, D8_BLOCK_SCOPE, *DDP_BLOCK_SHAPES}:
        result["optimizer"].update(
            execution_scope=o.nanochat_execution_scope,
            training_topology="causal_clean_prefix_bidirectional_masked_blocks",
            width_depth_transfer=(
                "fixed_matrix_root_aux_inverse_sqrt_width_not_completep"
                if o.nanochat_execution_scope in {D8_BLOCK_SCOPE, *DDP_BLOCK_SHAPES}
                else "not_implemented"
            ),
        )
        if o.nanochat_execution_scope in DDP_BLOCK_SHAPES:
            result["optimizer"].update(
                distributed_scope="ddp_replicated", distribution_adapter="local_native_kernels_v1"
            )
    if config.model.backbone_profile == backbone.PROFILE:
        result["optimizer"].update(
            execution_scope=backbone.SCOPE, auxiliary_policy="pinned_value_embedding_half_root_and_EB64K_scalars_v1"
        )
        result["model_conventions"].update(
            backbone_profile=backbone.PROFILE,
            input_norm="parameter_free_rms_norm",
            final_norm="parameter_free_rms_norm",
            mlp_type="squared_relu",
            qk_norm="parameter_free_rms_norm_after_rope_times_1.2",
            logit_softcap=15,
        )
    result["optimizer"].update(schedule_contract_fields(config))
    return {**result, "contract_sha256": json_digest(result)}


def make_optimizer(parameters: dict, contract: dict, config):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
        torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1
    )
    if distributed and config.optim.nanochat_execution_scope not in DDP_SCOPES:
        raise ValueError("nanochat_muon stage one is single-rank only; DDP integration is not qualified")
    checkout = (
        Path(config.optim.nanochat_checkout)
        if config.optim.nanochat_checkout
        else Path(__file__).parent / "vendor" / "nanochat"
    )
    verified_source(checkout, "LICENSE")
    compiled = config.runtime.compile and config.runtime.device_type == "cuda"
    module = _module(
        checkout,
        "nanochat/optim.py",
        {
            "COMPUTE_DTYPE": torch.bfloat16 if config.runtime.dtype == "bfloat16" else torch.float32,
        },
        eager_optimizer=not compiled,
        persistent=compiled,
    )
    groups = []
    for group in contract["optimizer_groups"]:
        groups.append(
            {
                "params": parameters[group["group_id"]],
                "kind": group["optimizer_kind"],
                "lr": group["resolved_lr"],
                "base_lr": group["resolved_lr"],
                "weight_decay": group["resolved_weight_decay"],
                **group["settings"],
                "parameterization_lr_multiplier": group["lr_multiplier"],
                "parameterization_group_id": group["group_id"],
            }
        )
    if config.optim.nanochat_execution_scope in DDP_SCOPES:
        from .nanochat_muon_replicated import replicated_optimizer

        return replicated_optimizer(module.MuonAdamW, groups)
    return module.MuonAdamW(groups)
