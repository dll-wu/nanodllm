"""Fail-closed D8/D12 CompleteP hidden-initialization transfer profiles."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config import ExperimentConfig


COMPLETEP_HIDDEN_INIT_TRANSFER_EXECUTION_SCOPE = "d8_d12_mdlm_completep_hidden_init_0p5_transfer"
COMPLETEP_HIDDEN_INIT_TRANSFER_NAMESPACE = "nanodllm-d8-d12-completep-hidden-init-0p5-transfer-v1"
COMPLETEP_HIDDEN_INIT_TRANSFER_RECIPES = {
    8: "d8-mdlm-completep-hidden-init-0p5-transfer-v1",
    12: "d12-mdlm-completep-hidden-init-0p5-transfer-v1",
}
COMPLETEP_HIDDEN_INIT_TRANSFER_MULTIPLIERS = (1.0, 0.5)
COMPLETEP_HIDDEN_INIT_TRANSFER_ROOT_LR = 4e-4
COMPLETEP_HIDDEN_INIT_TRANSFER_SPECS = {
    8: {
        "width": 512,
        "heads": 4,
        "mlp_hidden_dim": 2048,
        "nanochat_scaling_parameters": 50_331_648,
        "optimizer_updates": 30_720,
        "source_tokens": 1_006_632_960,
        "save_every": 7_680,
        "evaluation_tokens": 2_097_152,
        "evaluation_sequences": 1_024,
    },
    12: {
        "width": 768,
        "heads": 6,
        "mlp_hidden_dim": 3072,
        "nanochat_scaling_parameters": 138_412_032,
        "optimizer_updates": 84_480,
        "source_tokens": 2_768_240_640,
        "save_every": 21_120,
        "evaluation_tokens": 4_194_304,
        "evaluation_sequences": 2_048,
    },
}


class _TokenizerShape(Protocol):
    def get_vocab_size(self) -> int: ...

    def get_mask_token_id(self) -> int: ...


def derive_completep_hidden_init_transfer_seed(depth: int, block: str, role: str) -> int:
    """Derive one frozen unsigned 31-bit role seed for the scale transfer."""

    if depth not in COMPLETEP_HIDDEN_INIT_TRANSFER_SPECS:
        raise ValueError(f"unsupported hidden-init transfer depth: {depth}")
    material = f"{COMPLETEP_HIDDEN_INIT_TRANSFER_NAMESPACE}:d{depth}:{block}:{role}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def completep_hidden_init_transfer_seed_block(depth: int, block: str) -> tuple[int, int, int, int]:
    if block not in {"training", "evaluation"}:
        raise ValueError(f"unsupported hidden-init transfer seed block: {block}")
    return tuple(
        derive_completep_hidden_init_transfer_seed(depth, block, role)
        for role in ("runtime", "model", "training", "data")
    )


def is_completep_hidden_init_transfer(config: ExperimentConfig) -> bool:
    return config.parameterization.execution_scope == COMPLETEP_HIDDEN_INIT_TRANSFER_EXECUTION_SCOPE


def _equal_number(observed: object, expected: float | int) -> bool:
    return (
        isinstance(observed, int | float)
        and not isinstance(observed, bool)
        and math.isfinite(float(observed))
        and float(observed) == float(expected)
    )


def _require(config: ExperimentConfig, expected: Mapping[str, object], *, profile: str) -> None:
    for dotted_path, expected_value in expected.items():
        section, field = dotted_path.split(".", 1)
        observed = getattr(getattr(config, section), field)
        matches = (
            _equal_number(observed, expected_value)
            if isinstance(expected_value, int | float) and not isinstance(expected_value, bool)
            else observed == expected_value
        )
        if not matches:
            raise ValueError(
                f"CompleteP hidden-init transfer {profile} requires "
                f"{dotted_path}={expected_value!r}, observed {observed!r}"
            )


def validate_completep_hidden_init_transfer_config(config: ExperimentConfig) -> str:
    """Admit exactly the paired D8/D12 TPP20 CompleteP trajectories and evaluations."""

    depth = int(config.model.depth)
    spec = COMPLETEP_HIDDEN_INIT_TRANSFER_SPECS.get(depth)
    if spec is None:
        raise ValueError("CompleteP hidden-init transfer permits only D8 and D12")
    profile = "evaluation" if config.train.eval_only else "training"
    _require(
        config,
        {
            "recipe.name": COMPLETEP_HIDDEN_INIT_TRANSFER_RECIPES[depth],
            "recipe.status": "locked",
            "model.depth": depth,
            "model.n_embd": spec["width"],
            "model.n_head": spec["heads"],
            "model.max_seq_len": 2048,
            "model.mlp_hidden_dim": spec["mlp_hidden_dim"],
            "model.attention_scale": -1.0,
            "model.local_attention_lower_layers": 0,
            "model.local_attention_window": 0,
            "model.auxiliary_head_layers": [],
            "model.output_factorization": "flat",
            "model.causal": False,
            "model.residual_output_projection_init": "zero",
            "model.embedding_head_init_std_multiplier": 1.0,
            "model.length_head": False,
            "model.target_length_conditioning": False,
            "model.mask_position_embedding": False,
            "data.tokenizer": "hf",
            "data.prepend_bos": True,
            "data.append_eos": True,
            "data.loss_on_bos": False,
            "data.loss_on_eos": True,
            "data.boundary_aware_packing": False,
            "data.content_budget_segmented_packing": True,
            "noise.type": "mdlm",
            "noise.t_min": 0.05,
            "noise.t_max": 1.0,
            "noise.schedule_weight": True,
            "noise.complement_masking": False,
            "noise.per_token_noise_levels": False,
            "noise.time_conditioning": False,
            "noise.suffix_masking": False,
            "optim.type": "adamw",
            "optim.lr": COMPLETEP_HIDDEN_INIT_TRANSFER_ROOT_LR,
            "optim.muon_lr": -1.0,
            "optim.adamw_lr": -1.0,
            "optim.weight_decay": 0.1,
            "optim.adam_beta1": 0.9,
            "optim.adam_beta2": 0.95,
            "optim.adam_eps": 1e-8,
            "optim.warmup_steps": -1,
            "optim.warmup_ratio": 0.005,
            "optim.warmdown_ratio": 0.2,
            "optim.final_lr_frac": 0.1,
            "optim.grad_clip": 1.0,
            "parameterization.mode": "completep",
            "parameterization.execution_scope": COMPLETEP_HIDDEN_INIT_TRANSFER_EXECUTION_SCOPE,
            "parameterization.base_depth": 4,
            "parameterization.base_width": 256,
            "parameterization.contract_binding": "required",
            "runtime.device_type": "cuda",
            "runtime.dtype": "bfloat16",
            "runtime.cudnn_benchmark": False,
            "runtime.deterministic_algorithms": True,
            "train.target_param_count_basis": "nanochat_scaling_parameters",
            "train.target_param_data_ratio": 20.0,
            "train.total_batch_size": 32768,
            "train.device_batch_size": 2,
            "train.base_total_batch_size": 32768,
            "train.allow_batch_size_round_up": False,
            "train.scale_lr_by_batch": False,
            "train.tensorboard": False,
            "train.eval_every": -1,
            "train.auxiliary_loss_weight": 0.0,
            "eval.nelbo_tokens": spec["evaluation_tokens"],
            "eval.nelbo_t_min": 0.0,
            "eval.nelbo_t_max": 1.0,
            "eval.nelbo_time_sampling": "stratified_per_sequence",
            "eval.gen_eval": False,
            "eval.gen_eval_periodic": False,
            "eval.reconstruction_eval": False,
            "eval.conditional_rank_eval": False,
            "eval.conditional_rank_loo_eval": False,
            "eval.conditional_rank_stratified_eval": False,
            "eval.conditional_rank_rollout_eval": False,
            "eval.conditional_gen_eval": False,
            "diagnostics.parameter_state_hash_steps": [],
        },
        profile=profile,
    )
    if not any(
        _equal_number(config.parameterization.hidden_init_std_multiplier, value)
        for value in COMPLETEP_HIDDEN_INIT_TRANSFER_MULTIPLIERS
    ):
        raise ValueError("CompleteP hidden-init transfer permits only multipliers 1.0 and 0.5")

    observed_seeds = (
        int(config.runtime.seed),
        int(config.runtime.model_seed),
        int(config.runtime.training_seed),
        int(config.data.shuffle_seed),
    )
    expected_seeds = completep_hidden_init_transfer_seed_block(depth, profile)
    if observed_seeds != expected_seeds:
        raise ValueError(f"CompleteP hidden-init transfer D{depth} {profile} seed block differs")

    if profile == "training":
        if config.train.resume_from and not Path(config.train.resume_from).expanduser().is_absolute():
            raise ValueError("CompleteP hidden-init transfer resume checkpoint must be absolute")
        _require(
            config,
            {
                "train.num_steps": spec["optimizer_updates"],
                "train.target_tokens": spec["source_tokens"],
                "train.log_interval": 100,
                "train.save_every": spec["save_every"],
                "runtime.compile": True,
                "runtime.compile_dynamic": True,
                "eval.enabled": False,
            },
            profile=profile,
        )
    else:
        if not config.train.resume_from or not Path(config.train.resume_from).expanduser().is_absolute():
            raise ValueError("CompleteP hidden-init transfer evaluation requires an absolute checkpoint")
        _require(
            config,
            {
                "train.num_steps": -1,
                "train.target_tokens": -1,
                "train.save_every": -1,
                "runtime.compile": False,
                "runtime.compile_dynamic": False,
                "eval.enabled": True,
                "eval.nelbo_seed": derive_completep_hidden_init_transfer_seed(depth, "evaluation", "nelbo-plan"),
                "eval.source_checkpoint_identity_mode": "sha256",
            },
            profile=profile,
        )
    return profile


def completep_hidden_init_transfer_parameterization_contract(
    contract: Mapping[str, object], mechanics: Mapping[str, object]
) -> dict[str, object]:
    """Bind CompleteP mechanics to the exact paired D8/D12 transfer screen."""

    material = copy.deepcopy({key: value for key, value in contract.items() if key != "contract_sha256"})
    material.pop("launchable", None)
    material.pop("reserved_modes", None)
    material["mechanics_executable_modes"] = ["completep"]
    material["mechanics"] = copy.deepcopy(dict(mechanics))
    for collection_name in ("parameter_records", "parameter_groups"):
        collection = material.get(collection_name)
        if isinstance(collection, list):
            for record in collection:
                if isinstance(record, dict):
                    record.pop("prospective_nonordinary_status", None)
    material.update(
        {
            "schema": "nanodllm.parameterization_contract.v8",
            "adaptation_status": "implemented_mechanics_completep_hidden_init_transfer_authorized",
            "mechanics_executable": True,
            "compiled_bf16_qualification_status": (
                "reuses_completed_d4_hidden_init_and_d12_distributed_completep_properties"
            ),
            "scientific_training_authorized": True,
            "scientific_training_authorization_condition": (
                "exact_locked_paired_d8_d12_completep_hidden_init_plan_on_homogeneous_a100_or_a800_world_size_8"
            ),
            "scientific_execution_scope": COMPLETEP_HIDDEN_INIT_TRANSFER_EXECUTION_SCOPE,
            "campaign_plan_binding_required": True,
            "authorization_boundary": "exact_locked_d8_d12_completep_init1_vs_init0p5_tpp20_pairs_only",
            "qualification_evidence_owner": (
                "completed_d4_hidden_init_confirmation_and_d12_distributed_parameterization_qualification"
            ),
        }
    )
    payload = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {**material, "contract_sha256": hashlib.sha256(payload).hexdigest()}


def resolve_completep_hidden_init_transfer_parameterization_contract(
    config: ExperimentConfig,
    tokenizer: _TokenizerShape,
) -> tuple[dict[str, int], dict[str, object]]:
    """Resolve model counts and the scoped contract on the meta device."""

    import torch

    from .model import DiffusionTransformer
    from .parameterization import resolve_parameterization_contract
    from .training_plan import _resolve_diffusion_config

    validate_completep_hidden_init_transfer_config(config)
    model_config = _resolve_diffusion_config(config, tokenizer)
    with torch.device("meta"):
        model = DiffusionTransformer(model_config)
    counts = {key: int(value) for key, value in model.parameter_counts().items()}
    contract = resolve_parameterization_contract(model, config, batch_lr_scale=1.0)
    return counts, contract
