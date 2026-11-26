"""Fail-closed configuration admission for D8 parameterization experiments."""

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


D8_CENTER_TRANSFER_EXECUTION_SCOPE = "d8_mdlm_parameterization_center_transfer_screen"
D8_CENTER_TRANSFER_RECIPE = "d8-mdlm-parameterization-center-transfer-screen-v1"
D8_CENTER_TRANSFER_MODES = ("ordinary", "mup", "completep")
D8_CENTER_TRANSFER_TRAINING_SEEDS = (1272566586, 1879295888, 514259456, 1837729767)
D8_CENTER_TRANSFER_EVALUATION_SEEDS = (1499681107, 1244259746, 1327244235, 756497010)
D8_RESPONSE_NEIGHBORHOOD_EXECUTION_SCOPE = "d8_mdlm_parameterization_response_neighborhood"
D8_RESPONSE_NEIGHBORHOOD_RECIPE = "d8-mdlm-parameterization-response-neighborhood-v1"
D8_RESPONSE_NEIGHBORHOOD_ROOT_LRS = (2e-4, 4e-4, 8e-4)
D12_TRANSFER_EXECUTION_SCOPE = "d12_mdlm_parameterization_transfer_confirmation"
D12_TRANSFER_RECIPE = "d12-mdlm-parameterization-transfer-confirmation-v1"
D12_TRANSFER_CANARY_RECIPE = "d12-mdlm-parameterization-distributed-canary-v1"
D12_TRANSFER_MODES = ("ordinary", "mup", "completep")
D12_TRANSFER_CANDIDATE_MODES = ("mup", "completep")
D12_TRANSFER_ROOT_LRS = (2e-4, 4e-4, 8e-4)
D12_TRANSFER_TRAINING_SEEDS = (50_457_831, 276_992_044, 1_116_198_750, 1_615_077_665)
D12_TRANSFER_EVALUATION_SEEDS = (875_138_733, 1_174_791_555, 1_886_762_343, 1_137_542_844)
D12_TRANSFER_CANARY_SEEDS = (91_467_651, 729_063_322, 1_834_183_236, 1_129_946_911)


class _TokenizerShape(Protocol):
    def get_vocab_size(self) -> int: ...

    def get_mask_token_id(self) -> int: ...


class FixedTokenizerShape:
    """Minimal tokenizer boundary needed to resolve a model contract off-cluster."""

    def __init__(self, *, vocab_size: int, mask_token_id: int) -> None:
        self._vocab_size = vocab_size
        self._mask_token_id = mask_token_id

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_mask_token_id(self) -> int:
        return self._mask_token_id


def is_d8_center_transfer(config: ExperimentConfig) -> bool:
    return config.parameterization.execution_scope == D8_CENTER_TRANSFER_EXECUTION_SCOPE


def is_d8_response_neighborhood(config: ExperimentConfig) -> bool:
    return config.parameterization.execution_scope == D8_RESPONSE_NEIGHBORHOOD_EXECUTION_SCOPE


def is_d12_transfer(config: ExperimentConfig) -> bool:
    return config.parameterization.execution_scope == D12_TRANSFER_EXECUTION_SCOPE


def _equal_number(observed: object, expected: float | int) -> bool:
    return (
        isinstance(observed, int | float)
        and not isinstance(observed, bool)
        and math.isfinite(float(observed))
        and float(observed) == float(expected)
    )


def _require(config: ExperimentConfig, expected: dict[str, object], *, profile: str) -> None:
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
                f"D8 center-transfer {profile} profile requires {dotted_path}={expected_value!r}, observed {observed!r}"
            )


def _require_labeled(
    config: ExperimentConfig,
    expected: dict[str, object],
    *,
    label: str,
    profile: str,
) -> None:
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
                f"{label} {profile} profile requires {dotted_path}={expected_value!r}, observed {observed!r}"
            )


def _validate_d8_config(
    config: ExperimentConfig,
    *,
    recipe: str,
    allowed_root_lrs: tuple[float, ...],
    label: str,
) -> str:
    """Validate the shared full-horizon D8 training and evaluation profile."""

    _require(
        config,
        {
            "recipe.name": recipe,
            "recipe.status": "locked",
            "model.depth": 8,
            "model.n_embd": 512,
            "model.n_head": 4,
            "model.max_seq_len": 2048,
            "model.mlp_hidden_dim": 2048,
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
            "noise.type": "mdlm",
            "noise.t_min": 0.05,
            "noise.t_max": 1.0,
            "noise.schedule_weight": True,
            "noise.complement_masking": False,
            "noise.per_token_noise_levels": False,
            "noise.time_conditioning": False,
            "noise.suffix_masking": False,
            "optim.type": "adamw",
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
            "parameterization.base_depth": 4,
            "parameterization.base_width": 256,
            "parameterization.hidden_init_std_multiplier": 1.0,
            "parameterization.contract_binding": "required",
            "runtime.device_type": "cuda",
            "runtime.dtype": "bfloat16",
            "runtime.cudnn_benchmark": False,
            "runtime.deterministic_algorithms": True,
            "train.target_param_count_basis": "nanochat_scaling_parameters",
            "train.target_param_data_ratio": 20.0,
            "train.total_batch_size": 32768,
            "train.device_batch_size": 4,
            "train.base_total_batch_size": 32768,
            "train.allow_batch_size_round_up": False,
            "train.scale_lr_by_batch": False,
            "train.tensorboard": False,
            "train.eval_every": -1,
            "train.auxiliary_loss_weight": 0.0,
            "diagnostics.parameter_state_hash_steps": [],
        },
        profile="common",
    )
    if not any(_equal_number(config.optim.lr, learning_rate) for learning_rate in allowed_root_lrs):
        raise ValueError(f"{label} requires optim.lr in {allowed_root_lrs!r}, observed {config.optim.lr!r}")
    if config.parameterization.mode not in D8_CENTER_TRANSFER_MODES:
        raise ValueError(f"{label} permits only ordinary, mup, and completep")

    profile = "evaluation" if config.train.eval_only else "training"
    observed_seeds = (
        int(config.runtime.seed),
        int(config.runtime.model_seed),
        int(config.runtime.training_seed),
        int(config.data.shuffle_seed),
    )
    if profile == "training":
        if observed_seeds != D8_CENTER_TRANSFER_TRAINING_SEEDS:
            raise ValueError("D8 center-transfer training seed block differs")
        if config.train.resume_from and not Path(config.train.resume_from).expanduser().is_absolute():
            raise ValueError("D8 center-transfer resume checkpoint must be absolute")
        _require(
            config,
            {
                "train.num_steps": 30720,
                "train.target_tokens": 1006632960,
                "train.log_interval": 100,
                "train.save_every": 7680,
                "runtime.compile": True,
                "runtime.compile_dynamic": True,
                "eval.enabled": False,
            },
            profile=profile,
        )
    else:
        if observed_seeds != D8_CENTER_TRANSFER_EVALUATION_SEEDS:
            raise ValueError("D8 center-transfer evaluation seed block differs")
        if not config.train.resume_from or not Path(config.train.resume_from).expanduser().is_absolute():
            raise ValueError("D8 center-transfer evaluation requires an absolute checkpoint")
        _require(
            config,
            {
                "train.num_steps": -1,
                "train.target_tokens": -1,
                "train.save_every": -1,
                "runtime.compile": False,
                "runtime.compile_dynamic": False,
                "eval.enabled": True,
                "eval.nelbo_tokens": 2097152,
                "eval.nelbo_t_min": 0.0,
                "eval.nelbo_t_max": 1.0,
                "eval.nelbo_seed": 1804324978,
                "eval.nelbo_time_sampling": "stratified_per_sequence",
                "eval.source_checkpoint_identity_mode": "sha256",
                "eval.gen_eval": False,
                "eval.gen_eval_periodic": False,
                "eval.reconstruction_eval": False,
                "eval.conditional_rank_eval": False,
                "eval.conditional_rank_loo_eval": False,
                "eval.conditional_rank_stratified_eval": False,
                "eval.conditional_rank_rollout_eval": False,
                "eval.conditional_gen_eval": False,
            },
            profile=profile,
        )
    return profile


def validate_d8_center_transfer_config(config: ExperimentConfig) -> str:
    """Admit exactly the three historical full-horizon centers and their evaluation."""

    return _validate_d8_config(
        config,
        recipe=D8_CENTER_TRANSFER_RECIPE,
        allowed_root_lrs=(4e-4,),
        label="D8 center-transfer",
    )


def validate_d8_response_neighborhood_config(config: ExperimentConfig) -> str:
    """Admit the locked D8 root-LR response coordinates and their shared evaluation."""

    return _validate_d8_config(
        config,
        recipe=D8_RESPONSE_NEIGHBORHOOD_RECIPE,
        allowed_root_lrs=D8_RESPONSE_NEIGHBORHOOD_ROOT_LRS,
        label="D8 response-neighborhood",
    )


def validate_d12_transfer_config(config: ExperimentConfig) -> str:
    """Admit only the frozen seven-cell D12 confirmation and paired evaluation profile."""

    label = "D12 parameterization-transfer"
    _require_labeled(
        config,
        {
            "recipe.status": "locked",
            "model.depth": 12,
            "model.n_embd": 768,
            "model.n_head": 6,
            "model.max_seq_len": 2048,
            "model.mlp_hidden_dim": 3072,
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
            "noise.type": "mdlm",
            "noise.t_min": 0.05,
            "noise.t_max": 1.0,
            "noise.schedule_weight": True,
            "noise.complement_masking": False,
            "noise.per_token_noise_levels": False,
            "noise.time_conditioning": False,
            "noise.suffix_masking": False,
            "optim.type": "adamw",
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
            "parameterization.execution_scope": D12_TRANSFER_EXECUTION_SCOPE,
            "parameterization.base_depth": 4,
            "parameterization.base_width": 256,
            "parameterization.hidden_init_std_multiplier": 1.0,
            "parameterization.contract_binding": "required",
            "runtime.device_type": "cuda",
            "runtime.dtype": "bfloat16",
            "runtime.cudnn_benchmark": False,
            "runtime.deterministic_algorithms": True,
            "train.target_param_count_basis": "nanochat_scaling_parameters",
            "train.total_batch_size": 32768,
            "train.device_batch_size": 2,
            "train.base_total_batch_size": 32768,
            "train.allow_batch_size_round_up": False,
            "train.scale_lr_by_batch": False,
            "train.tensorboard": False,
            "train.eval_every": -1,
            "train.auxiliary_loss_weight": 0.0,
            "diagnostics.parameter_state_hash_steps": [],
        },
        label=label,
        profile="common",
    )
    if config.recipe.name not in {D12_TRANSFER_RECIPE, D12_TRANSFER_CANARY_RECIPE}:
        raise ValueError(f"{label} recipe identity differs")
    mode = config.parameterization.mode
    if mode not in D12_TRANSFER_MODES:
        raise ValueError(f"{label} permits only ordinary, mup, and completep")
    allowed_root_lrs = (
        (4e-4,) if mode == "ordinary" or config.recipe.name == D12_TRANSFER_CANARY_RECIPE else D12_TRANSFER_ROOT_LRS
    )
    if not any(_equal_number(config.optim.lr, learning_rate) for learning_rate in allowed_root_lrs):
        raise ValueError(f"{label} mode {mode!r} requires optim.lr in {allowed_root_lrs!r}")

    profile = (
        "evaluation"
        if config.train.eval_only
        else "canary"
        if config.recipe.name == D12_TRANSFER_CANARY_RECIPE
        else "training"
    )
    observed_seeds = (
        int(config.runtime.seed),
        int(config.runtime.model_seed),
        int(config.runtime.training_seed),
        int(config.data.shuffle_seed),
    )
    if profile == "canary":
        if observed_seeds != D12_TRANSFER_CANARY_SEEDS:
            raise ValueError(f"{label} canary seed block differs")
        if config.train.resume_from:
            raise ValueError(f"{label} canary requires fresh training")
        _require_labeled(
            config,
            {
                "optim.lr": 4e-4,
                "train.target_param_data_ratio": -1.0,
                "train.num_steps": 8,
                "train.target_tokens": 262144,
                "train.log_interval": 1,
                "train.save_every": 8,
                "runtime.compile": True,
                "runtime.compile_dynamic": True,
                "eval.enabled": False,
            },
            label=label,
            profile=profile,
        )
    elif profile == "training":
        if observed_seeds != D12_TRANSFER_TRAINING_SEEDS:
            raise ValueError(f"{label} training seed block differs")
        if config.train.resume_from:
            raise ValueError(f"{label} material arms require fresh training")
        _require_labeled(
            config,
            {
                "train.num_steps": 84480,
                "train.target_tokens": 2768240640,
                "train.target_param_data_ratio": 20.0,
                "train.log_interval": 100,
                "train.save_every": 21120,
                "runtime.compile": True,
                "runtime.compile_dynamic": True,
                "eval.enabled": False,
            },
            label=label,
            profile=profile,
        )
    else:
        if observed_seeds != D12_TRANSFER_EVALUATION_SEEDS:
            raise ValueError(f"{label} evaluation seed block differs")
        if not config.train.resume_from or not Path(config.train.resume_from).expanduser().is_absolute():
            raise ValueError(f"{label} evaluation requires an absolute checkpoint")
        _require_labeled(
            config,
            {
                "train.num_steps": -1,
                "train.target_tokens": -1,
                "train.target_param_data_ratio": 20.0,
                "train.save_every": -1,
                "runtime.compile": False,
                "runtime.compile_dynamic": False,
                "eval.enabled": True,
                "eval.nelbo_tokens": 4194304,
                "eval.nelbo_t_min": 0.0,
                "eval.nelbo_t_max": 1.0,
                "eval.nelbo_seed": 59097123,
                "eval.nelbo_time_sampling": "stratified_per_sequence",
                "eval.source_checkpoint_identity_mode": "sha256",
                "eval.gen_eval": False,
                "eval.gen_eval_periodic": False,
                "eval.reconstruction_eval": False,
                "eval.conditional_rank_eval": False,
                "eval.conditional_rank_loo_eval": False,
                "eval.conditional_rank_stratified_eval": False,
                "eval.conditional_rank_rollout_eval": False,
                "eval.conditional_gen_eval": False,
            },
            label=label,
            profile=profile,
        )
    return profile


def resolve_d8_parameterization_contract(
    config: ExperimentConfig,
    tokenizer: _TokenizerShape,
) -> tuple[dict[str, int], dict[str, object]]:
    """Resolve the exact D8 model counts and parameterization contract on meta."""

    import torch

    from .model import DiffusionTransformer
    from .parameterization import resolve_parameterization_contract
    from .training_plan import _resolve_diffusion_config

    validate_d8_center_transfer_config(config)
    model_config = _resolve_diffusion_config(config, tokenizer)
    with torch.device("meta"):
        model = DiffusionTransformer(model_config)
    counts = {key: int(value) for key, value in model.parameter_counts().items()}
    contract = resolve_parameterization_contract(model, config, batch_lr_scale=1.0)
    return counts, contract


def resolve_d8_response_parameterization_contract(
    config: ExperimentConfig,
    tokenizer: _TokenizerShape,
) -> tuple[dict[str, int], dict[str, object]]:
    """Resolve the exact D8 response-neighborhood parameterization contract on meta."""

    import torch

    from .model import DiffusionTransformer
    from .parameterization import resolve_parameterization_contract
    from .training_plan import _resolve_diffusion_config

    validate_d8_response_neighborhood_config(config)
    model_config = _resolve_diffusion_config(config, tokenizer)
    with torch.device("meta"):
        model = DiffusionTransformer(model_config)
    counts = {key: int(value) for key, value in model.parameter_counts().items()}
    contract = resolve_parameterization_contract(model, config, batch_lr_scale=1.0)
    return counts, contract


def resolve_d12_transfer_parameterization_contract(
    config: ExperimentConfig,
    tokenizer: _TokenizerShape,
) -> tuple[dict[str, int], dict[str, object]]:
    """Resolve the exact D12 model counts and scoped parameterization contract on meta."""

    import torch

    from .model import DiffusionTransformer
    from .parameterization import resolve_parameterization_contract
    from .training_plan import _resolve_diffusion_config

    validate_d12_transfer_config(config)
    model_config = _resolve_diffusion_config(config, tokenizer)
    with torch.device("meta"):
        model = DiffusionTransformer(model_config)
    counts = {key: int(value) for key, value in model.parameter_counts().items()}
    contract = resolve_parameterization_contract(model, config, batch_lr_scale=1.0)
    return counts, contract


def transfer_contract_material() -> dict[str, object]:
    """Return truthful scoped authorization for the active D8 screen."""

    return {
        "schema": "nanodllm.parameterization_contract.v4",
        "adaptation_status": "implemented_mechanics_d8_center_screen_authorized",
        "mechanics_executable": True,
        "compiled_bf16_qualification_status": "historical_a100_mechanics_qualification_plus_current_release_gates",
        "scientific_training_authorized": True,
        "scientific_training_authorization_condition": "exact_locked_d8_center_plan_on_accepted_a100",
        "scientific_execution_scope": D8_CENTER_TRANSFER_EXECUTION_SCOPE,
        "campaign_plan_binding_required": True,
        "authorization_boundary": "exact_locked_d8_center_screen_recipe_and_immutable_campaign_plan_only",
        "qualification_evidence_owner": "historical_a100_mechanics_run_and_current_release_tests",
    }


def transfer_parameterization_contract(
    contract: Mapping[str, object], mechanics: Mapping[str, object]
) -> dict[str, object]:
    """Upgrade an admitted model contract to the exact D8 campaign semantics."""

    material = copy.deepcopy({key: value for key, value in contract.items() if key != "contract_sha256"})
    material.pop("launchable", None)
    material.pop("reserved_modes", None)
    material["mechanics_executable_modes"] = list(D8_CENTER_TRANSFER_MODES)
    material["mechanics"] = copy.deepcopy(dict(mechanics))
    for collection_name in ("parameter_records", "parameter_groups"):
        collection = material.get(collection_name)
        if isinstance(collection, list):
            for record in collection:
                if isinstance(record, dict):
                    record.pop("prospective_nonordinary_status", None)
    material.update(transfer_contract_material())
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {**material, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def response_parameterization_contract(
    contract: Mapping[str, object], mechanics: Mapping[str, object]
) -> dict[str, object]:
    """Upgrade an admitted model contract to the exact D8 response-screen semantics."""

    material = copy.deepcopy({key: value for key, value in contract.items() if key != "contract_sha256"})
    material.pop("launchable", None)
    material.pop("reserved_modes", None)
    material["mechanics_executable_modes"] = list(D8_CENTER_TRANSFER_MODES)
    material["mechanics"] = copy.deepcopy(dict(mechanics))
    for collection_name in ("parameter_records", "parameter_groups"):
        collection = material.get(collection_name)
        if isinstance(collection, list):
            for record in collection:
                if isinstance(record, dict):
                    record.pop("prospective_nonordinary_status", None)
    material.update(
        {
            "schema": "nanodllm.parameterization_contract.v5",
            "adaptation_status": "implemented_mechanics_d8_response_neighborhood_authorized",
            "mechanics_executable": True,
            "compiled_bf16_qualification_status": (
                "historical_a100_mechanics_qualification_plus_current_release_gates"
            ),
            "scientific_training_authorized": True,
            "scientific_training_authorization_condition": "exact_locked_d8_response_plan_on_accepted_a100_or_a800",
            "scientific_execution_scope": D8_RESPONSE_NEIGHBORHOOD_EXECUTION_SCOPE,
            "campaign_plan_binding_required": True,
            "authorization_boundary": "exact_locked_d8_response_recipe_and_immutable_campaign_plan_only",
            "qualification_evidence_owner": "historical_d8_centers_and_current_release_tests",
        }
    )
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {**material, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def d12_transfer_parameterization_contract(
    contract: Mapping[str, object], mechanics: Mapping[str, object]
) -> dict[str, object]:
    """Upgrade an admitted model contract to the exact D12 confirmation semantics."""

    material = copy.deepcopy({key: value for key, value in contract.items() if key != "contract_sha256"})
    material.pop("launchable", None)
    material.pop("reserved_modes", None)
    material["mechanics_executable_modes"] = list(D12_TRANSFER_MODES)
    material["mechanics"] = copy.deepcopy(dict(mechanics))
    for collection_name in ("parameter_records", "parameter_groups"):
        collection = material.get(collection_name)
        if isinstance(collection, list):
            for record in collection:
                if isinstance(record, dict):
                    record.pop("prospective_nonordinary_status", None)
    material.update(
        {
            "schema": "nanodllm.parameterization_contract.v6",
            "adaptation_status": "implemented_mechanics_d12_transfer_confirmation_authorized",
            "mechanics_executable": True,
            "compiled_bf16_qualification_status": "same_release_d12_distributed_canary_required_before_material_arms",
            "scientific_training_authorized": True,
            "scientific_training_authorization_condition": (
                "exact_locked_d12_confirmation_plan_on_homogeneous_a100_or_a800_80gb_world_size_8"
            ),
            "scientific_execution_scope": D12_TRANSFER_EXECUTION_SCOPE,
            "campaign_plan_binding_required": True,
            "authorization_boundary": "exact_locked_d12_confirmation_recipe_and_immutable_arm_plan_only",
            "qualification_evidence_owner": "same_release_d12_distributed_canary_and_current_release_tests",
        }
    )
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {**material, "contract_sha256": hashlib.sha256(encoded).hexdigest()}
