"""Fail-closed admission for the bounded D20 CompleteP TPP20 campaign."""

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


D20_COMPLETEP_TPP20_NAMESPACE = "nanodllm-d20-mdlm-completep-tpp20-v1"
D20_COMPLETEP_TPP20_EXECUTION_SCOPE = "d20_mdlm_completep_tpp20"
D20_COMPLETEP_TPP20_RECIPE = "d20-mdlm-completep-tpp20-v1"
D20_COMPLETEP_TPP20_CANARY_RECIPE = "d20-mdlm-completep-tpp20-distributed-canary-v1"
D20_COMPLETEP_TPP20_PREFIX_RECIPE = "d20-mdlm-completep-tpp20-ordinary-prefix-v1"
D20_LEGACY_ORDINARY_EVALUATION_RECIPE = "d20-mdlm-ordinary-tpp20-historical-evaluation-v1"
D20_COMPLETEP_TPP20_MODES = ("ordinary", "completep")
D20_COMPLETEP_TPP20_MATERIAL_SEEDS = (1_273_347_104, 940_816_789, 909_970_692, 319_180_585)
D20_COMPLETEP_TPP20_QUALIFICATION_SEEDS = (1_529_168_483, 312_429_347, 735_826_918, 92_650_095)
D20_COMPLETEP_TPP20_EVALUATION_SEEDS = (1_129_497_981, 946_126_166, 2_012_589_833, 1_312_167_263)
D20_COMPLETEP_TPP20_NELBO_SEEDS = (
    1_527_547_018,
    2_086_911_719,
    243_780_285,
    1_378_509_031,
    1_837_093_018,
    1_120_997_644,
    341_860_605,
    2_129_275_378,
)
D20_COMPLETEP_TPP20_BOOTSTRAP_SEED = 1_149_113_500
D20_COMPLETEP_TPP20_ROOT_LR = 4e-4
D20_LEGACY_ORDINARY_ROOT_LR = 3e-4
D20_COMPLETEP_TPP20_UPDATES = 345_600
D20_COMPLETEP_TPP20_SOURCE_TOKENS = 11_324_620_800
D20_COMPLETEP_TPP20_WARMUP_STEPS = 1_728
D20_COMPLETEP_TPP20_PREFIX_UPDATES = 1_024
D20_COMPLETEP_TPP20_PREFIX_SOURCE_TOKENS = 33_554_432
D20_COMPLETEP_TPP20_CANARY_UPDATES = 8
D20_COMPLETEP_TPP20_CANARY_SOURCE_TOKENS = 262_144

_LOCKED_FIELDS = (
    "model.depth",
    "model.n_embd",
    "model.n_head",
    "model.max_seq_len",
    "model.mlp_hidden_dim",
    "model.attention_scale",
    "model.rope_base",
    "model.output_factorization",
    "model.causal",
    "model.residual_output_projection_init",
    "data.tokenizer",
    "data.tokenizer_path",
    "data.train_data",
    "data.train_exclude_data",
    "data.val_data",
    "data.prepend_bos",
    "data.append_eos",
    "data.shuffle_seed",
    "data.content_budget_segmented_packing",
    "data.identity_mode",
    "noise.type",
    "noise.t_min",
    "noise.t_max",
    "noise.schedule_weight",
    "optim.type",
    "optim.lr",
    "optim.weight_decay",
    "optim.adam_beta1",
    "optim.adam_beta2",
    "optim.adam_eps",
    "optim.warmup_steps",
    "optim.warmup_ratio",
    "optim.warmdown_ratio",
    "optim.final_lr_frac",
    "optim.grad_clip",
    "parameterization.mode",
    "parameterization.execution_scope",
    "parameterization.base_depth",
    "parameterization.base_width",
    "parameterization.hidden_init_std_multiplier",
    "parameterization.contract_binding",
    "runtime.seed",
    "runtime.model_seed",
    "runtime.training_seed",
    "runtime.dtype",
    "runtime.compile",
    "runtime.compile_dynamic",
    "runtime.cudnn_benchmark",
    "runtime.deterministic_algorithms",
    "train.num_steps",
    "train.target_tokens",
    "train.target_param_data_ratio",
    "train.target_param_count_basis",
    "train.total_batch_size",
    "train.device_batch_size",
    "train.base_total_batch_size",
    "train.allow_batch_size_round_up",
    "train.scale_lr_by_batch",
    "train.save_every",
    "eval.enabled",
    "eval.nelbo_tokens",
    "eval.nelbo_seed",
    "diagnostics.parameter_state_hash_steps",
)


class _TokenizerShape(Protocol):
    def get_vocab_size(self) -> int: ...

    def get_mask_token_id(self) -> int: ...


def is_d20_completep_tpp20(config: ExperimentConfig) -> bool:
    return config.parameterization.execution_scope == D20_COMPLETEP_TPP20_EXECUTION_SCOPE


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
                f"D20 CompleteP TPP20 {profile} profile requires {dotted_path}={expected_value!r}, "
                f"observed {observed!r}"
            )


def _common_expected(*, attention_scale: float, root_lr: float) -> dict[str, object]:
    return {
        "recipe.status": "locked",
        "recipe.locked_fields": list(_LOCKED_FIELDS),
        "recipe.provisional_fields": [],
        "model.depth": 20,
        "model.n_embd": 1280,
        "model.n_head": 10,
        "model.max_seq_len": 2048,
        "model.mlp_hidden_dim": 5120,
        "model.attention_scale": attention_scale,
        "model.rope_base": 100000.0,
        "model.pad_vocab_to": 64,
        "model.local_attention_lower_layers": 0,
        "model.local_attention_window": 0,
        "model.auxiliary_head_layers": [],
        "model.bd3lm_block_size": 0,
        "model.block_causal_pilot_size": 0,
        "model.output_factorization": "flat",
        "model.causal": False,
        "model.residual_output_projection_init": "zero",
        "model.embedding_head_init_std_multiplier": 1.0,
        "model.length_head": False,
        "model.target_length_conditioning": False,
        "model.mask_position_embedding": False,
        "model.token_edit_adapter_bottleneck_dim": 0,
        "model.token_edit_adapter_top_layers": 0,
        "model.token_edit_provenance_embedding": False,
        "model.token_edit_provenance_confidence_bins": 0,
        "data.tokenizer": "hf",
        "data.text_column": "text",
        "data.val_fraction": 0.0,
        "data.tokenizer_batch_size": 128,
        "data.prepend_bos": True,
        "data.append_eos": True,
        "data.loss_on_bos": False,
        "data.loss_on_eos": True,
        "data.random_length_ratio": 0.01,
        "data.random_length_min": 1,
        "data.continuation_ratio": 0.1,
        "data.continuation_min_prefix_frac": 0.125,
        "data.continuation_max_prefix_frac": 0.75,
        "data.shuffle_row_groups": True,
        "data.boundary_aware_packing": False,
        "data.content_budget_segmented_packing": True,
        "data.short_document_curriculum": False,
        "data.identity_mode": "metadata",
        "noise.type": "mdlm",
        "noise.t_min": 0.05,
        "noise.t_max": 1.0,
        "noise.schedule_weight": True,
        "noise.complement_masking": False,
        "noise.per_token_noise_levels": False,
        "noise.time_conditioning": False,
        "noise.suffix_masking": False,
        "noise.soft_input_curriculum": False,
        "optim.type": "adamw",
        "optim.lr": root_lr,
        "optim.muon_lr": -1.0,
        "optim.adamw_lr": -1.0,
        "optim.weight_decay": 0.1,
        "optim.adam_beta1": 0.9,
        "optim.adam_beta2": 0.95,
        "optim.adam_eps": 1e-8,
        "optim.final_lr_frac": 0.1,
        "optim.grad_clip": 1.0,
        "parameterization.base_depth": 4,
        "parameterization.base_width": 256,
        "parameterization.hidden_init_std_multiplier": 1.0,
        "runtime.device_type": "cuda",
        "runtime.dtype": "bfloat16",
        "runtime.min_free_gpu_mib": 5120,
        "runtime.init_single_rank_process_group": False,
        "runtime.float32_matmul_precision": "high",
        "runtime.allow_tf32": True,
        "runtime.cudnn_allow_tf32": True,
        "runtime.cudnn_benchmark": False,
        "runtime.deterministic_algorithms": True,
        "runtime.sdpa_backend": "auto",
        "train.target_param_count_basis": "nanochat_scaling_parameters",
        "train.total_batch_size": 32768,
        "train.device_batch_size": 2,
        "train.base_total_batch_size": 32768,
        "train.allow_batch_size_round_up": False,
        "train.scale_lr_by_batch": False,
        "train.tensorboard": False,
        "train.tensorboard_dir": "",
        "train.eval_every": -1,
        "train.auxiliary_loss_weight": 0.0,
        "train.divergence_check_interval": 10,
        "train.divergence_loss_threshold": 100.0,
        "eval.gen_eval": False,
        "eval.gen_eval_periodic": False,
        "eval.nelbo_tokens": 4194304,
        "eval.nelbo_t_min": 0.0,
        "eval.nelbo_t_max": 1.0,
        "eval.nelbo_time_sampling": "stratified_per_sequence",
        "eval.fixed_ce_batches": 1,
        "eval.fixed_ce_trace_enabled": False,
        "eval.reconstruction_eval": False,
        "eval.gen_samples": 1,
        "eval.conditional_rank_eval": False,
        "eval.conditional_rank_loo_eval": False,
        "eval.conditional_rank_stratified_eval": False,
        "eval.conditional_rank_rollout_eval": False,
        "eval.conditional_gen_eval": False,
    }


_PROFILE_BOUND_FIELDS = (
    "data.shuffle_seed",
    "optim.warmup_steps",
    "optim.warmup_ratio",
    "optim.warmdown_ratio",
    "parameterization.mode",
    "parameterization.execution_scope",
    "parameterization.contract_binding",
    "parameterization.contract_sha256",
    "runtime.seed",
    "runtime.model_seed",
    "runtime.training_seed",
    "runtime.compile",
    "runtime.compile_dynamic",
    "train.run_dir",
    "train.num_steps",
    "train.target_tokens",
    "train.target_param_data_ratio",
    "train.log_interval",
    "train.save_every",
    "train.resume_from",
    "train.eval_only",
    "eval.enabled",
    "eval.nelbo_seed",
    "eval.source_checkpoint_identity_mode",
    "eval.source_checkpoint_expected_sha256",
    "eval.source_checkpoint_expected_size_bytes",
    "diagnostics.parameter_state_hash_steps",
    "recipe.name",
)


def _require_no_unadmitted_overrides(
    config: ExperimentConfig,
    common_expected: Mapping[str, object],
) -> None:
    """Reject every non-profile field that differs from its inert dataclass default."""

    admitted: dict[str, set[str]] = {}
    for dotted_path in (*common_expected, *_PROFILE_BOUND_FIELDS):
        section, field = dotted_path.split(".", 1)
        admitted.setdefault(section, set()).add(field)
    for section_name in (
        "model",
        "data",
        "noise",
        "optim",
        "parameterization",
        "runtime",
        "train",
        "eval",
        "diagnostics",
        "recipe",
    ):
        observed_section = getattr(config, section_name)
        default_section = type(observed_section)()
        for field_name, expected in vars(default_section).items():
            if field_name in admitted.get(section_name, set()):
                continue
            if getattr(observed_section, field_name) != expected:
                raise ValueError(f"D20 CompleteP TPP20 forbids {section_name}.{field_name} outside the locked profile")


def _seed_tuple(config: ExperimentConfig) -> tuple[int, int, int, int]:
    return (
        int(config.runtime.seed),
        int(config.runtime.model_seed),
        int(config.runtime.training_seed),
        int(config.data.shuffle_seed),
    )


def _require_seed_tuple(config: ExperimentConfig, expected: tuple[int, int, int, int], *, profile: str) -> None:
    if _seed_tuple(config) != expected:
        raise ValueError(f"D20 CompleteP TPP20 {profile} seed block differs")


def _require_evaluation(config: ExperimentConfig, *, profile: str) -> None:
    _require_seed_tuple(config, D20_COMPLETEP_TPP20_EVALUATION_SEEDS, profile=profile)
    if not config.train.resume_from or not Path(config.train.resume_from).expanduser().is_absolute():
        raise ValueError(f"D20 CompleteP TPP20 {profile} requires an absolute checkpoint")
    if config.eval.nelbo_seed not in D20_COMPLETEP_TPP20_NELBO_SEEDS:
        raise ValueError(f"D20 CompleteP TPP20 {profile} NELBO seed is outside the frozen replicate block")
    _require(
        config,
        {
            "train.num_steps": -1,
            "train.target_tokens": -1,
            "train.target_param_data_ratio": 20.0,
            "train.log_interval": 128,
            "train.save_every": -1,
            "train.eval_only": True,
            "runtime.compile": False,
            "runtime.compile_dynamic": False,
            "eval.enabled": True,
            "eval.source_checkpoint_identity_mode": "sha256",
        },
        profile=profile,
    )


def validate_d20_completep_tpp20_config(config: ExperimentConfig) -> str:
    """Admit only the D20 two-mode canary, ordinary prefix, CompleteP full, or bound evaluation."""

    if config.recipe.name == D20_COMPLETEP_TPP20_CANARY_RECIPE:
        profile = "canary"
    elif config.recipe.name == D20_COMPLETEP_TPP20_PREFIX_RECIPE:
        profile = "ordinary-prefix"
    elif config.recipe.name == D20_COMPLETEP_TPP20_RECIPE and config.train.eval_only:
        profile = "completep-evaluation"
    elif config.recipe.name == D20_COMPLETEP_TPP20_RECIPE:
        profile = "completep-full"
    else:
        raise ValueError("D20 CompleteP TPP20 recipe identity differs")

    common_expected = _common_expected(attention_scale=-1.0, root_lr=D20_COMPLETEP_TPP20_ROOT_LR)
    _require(config, common_expected, profile=profile)
    _require_no_unadmitted_overrides(config, common_expected)
    _require(
        config,
        {
            "parameterization.execution_scope": D20_COMPLETEP_TPP20_EXECUTION_SCOPE,
            "parameterization.contract_binding": "required",
        },
        profile=profile,
    )

    if profile == "canary":
        if config.parameterization.mode not in D20_COMPLETEP_TPP20_MODES:
            raise ValueError("D20 CompleteP TPP20 canary permits only ordinary and completep")
        _require_seed_tuple(config, D20_COMPLETEP_TPP20_QUALIFICATION_SEEDS, profile=profile)
        if config.train.resume_from:
            raise ValueError("D20 CompleteP TPP20 canary requires fresh training")
        _require(
            config,
            {
                "optim.warmup_steps": -1,
                "optim.warmup_ratio": 0.005,
                "optim.warmdown_ratio": 0.2,
                "train.num_steps": D20_COMPLETEP_TPP20_CANARY_UPDATES,
                "train.target_tokens": D20_COMPLETEP_TPP20_CANARY_SOURCE_TOKENS,
                "train.target_param_data_ratio": -1.0,
                "train.log_interval": 1,
                "train.save_every": D20_COMPLETEP_TPP20_CANARY_UPDATES,
                "train.eval_only": False,
                "runtime.compile": True,
                "runtime.compile_dynamic": True,
                "eval.enabled": False,
                "eval.source_checkpoint_identity_mode": "off",
                "diagnostics.parameter_state_hash_steps": [1, 8],
            },
            profile=profile,
        )
    elif profile == "ordinary-prefix":
        _require_seed_tuple(config, D20_COMPLETEP_TPP20_MATERIAL_SEEDS, profile=profile)
        if config.train.resume_from:
            raise ValueError("D20 CompleteP TPP20 ordinary-prefix requires fresh training")
        _require(
            config,
            {
                "parameterization.mode": "ordinary",
                "optim.warmup_steps": D20_COMPLETEP_TPP20_WARMUP_STEPS,
                "optim.warmup_ratio": -1.0,
                "optim.warmdown_ratio": 0.0,
                "train.num_steps": D20_COMPLETEP_TPP20_PREFIX_UPDATES,
                "train.target_tokens": D20_COMPLETEP_TPP20_PREFIX_SOURCE_TOKENS,
                "train.target_param_data_ratio": -1.0,
                "train.log_interval": 128,
                "train.save_every": D20_COMPLETEP_TPP20_PREFIX_UPDATES,
                "train.eval_only": False,
                "runtime.compile": True,
                "runtime.compile_dynamic": True,
                "eval.enabled": False,
                "eval.source_checkpoint_identity_mode": "off",
                "diagnostics.parameter_state_hash_steps": [D20_COMPLETEP_TPP20_PREFIX_UPDATES],
            },
            profile=profile,
        )
    elif profile == "completep-full":
        _require_seed_tuple(config, D20_COMPLETEP_TPP20_MATERIAL_SEEDS, profile=profile)
        if config.train.resume_from and not Path(config.train.resume_from).expanduser().is_absolute():
            raise ValueError("D20 CompleteP TPP20 full-run resume checkpoint must be absolute")
        _require(
            config,
            {
                "parameterization.mode": "completep",
                "optim.warmup_steps": -1,
                "optim.warmup_ratio": 0.005,
                "optim.warmdown_ratio": 0.2,
                "train.num_steps": D20_COMPLETEP_TPP20_UPDATES,
                "train.target_tokens": D20_COMPLETEP_TPP20_SOURCE_TOKENS,
                "train.target_param_data_ratio": 20.0,
                "train.log_interval": 128,
                "train.save_every": 34560,
                "train.eval_only": False,
                "runtime.compile": True,
                "runtime.compile_dynamic": True,
                "eval.enabled": False,
                "eval.source_checkpoint_identity_mode": "off",
                "diagnostics.parameter_state_hash_steps": [D20_COMPLETEP_TPP20_PREFIX_UPDATES],
            },
            profile=profile,
        )
    else:
        _require(
            config,
            {
                "parameterization.mode": "completep",
                "optim.warmup_steps": -1,
                "optim.warmup_ratio": 0.005,
                "optim.warmdown_ratio": 0.2,
                "diagnostics.parameter_state_hash_steps": [],
            },
            profile=profile,
        )
        _require_evaluation(config, profile=profile)
    return profile


def validate_d20_legacy_ordinary_evaluation_config(config: ExperimentConfig) -> str:
    """Validate the historical ordinary checkpoint only under the separate ordinary scope."""

    profile = "legacy-ordinary-evaluation"
    common_expected = _common_expected(
        attention_scale=0.08838834764831843,
        root_lr=D20_LEGACY_ORDINARY_ROOT_LR,
    )
    _require(config, common_expected, profile=profile)
    _require_no_unadmitted_overrides(config, common_expected)
    _require(
        config,
        {
            "recipe.name": D20_LEGACY_ORDINARY_EVALUATION_RECIPE,
            "parameterization.mode": "ordinary",
            "parameterization.execution_scope": "ordinary",
            "parameterization.contract_binding": "legacy_unbound",
            "parameterization.contract_sha256": "",
            "optim.warmup_steps": -1,
            "optim.warmup_ratio": 0.005,
            "optim.warmdown_ratio": 0.2,
            "diagnostics.parameter_state_hash_steps": [],
        },
        profile=profile,
    )
    _require_evaluation(config, profile=profile)
    return profile


def resolve_d20_parameterization_contract(
    config: ExperimentConfig,
    tokenizer: _TokenizerShape,
) -> tuple[dict[str, int], dict[str, object]]:
    """Resolve exact D20 counts and the scoped parameterization contract on meta."""

    import torch

    from .model import DiffusionTransformer
    from .parameterization import resolve_parameterization_contract
    from .training_plan import _resolve_diffusion_config

    validate_d20_completep_tpp20_config(config)
    model_config = _resolve_diffusion_config(config, tokenizer)
    with torch.device("meta"):
        model = DiffusionTransformer(model_config)
    counts = {key: int(value) for key, value in model.parameter_counts().items()}
    contract = resolve_parameterization_contract(model, config, batch_lr_scale=1.0)
    return counts, contract


def d20_completep_parameterization_contract(
    contract: Mapping[str, object], mechanics: Mapping[str, object]
) -> dict[str, object]:
    """Upgrade an admitted contract to the exact D20 campaign authorization boundary."""

    material = copy.deepcopy({key: value for key, value in contract.items() if key != "contract_sha256"})
    material.pop("launchable", None)
    material.pop("reserved_modes", None)
    material["mechanics_executable_modes"] = list(D20_COMPLETEP_TPP20_MODES)
    material["mechanics"] = copy.deepcopy(dict(mechanics))
    for collection_name in ("parameter_records", "parameter_groups"):
        collection = material.get(collection_name)
        if isinstance(collection, list):
            for record in collection:
                if isinstance(record, dict):
                    record.pop("prospective_nonordinary_status", None)
    material.update(
        {
            "schema": "nanodllm.parameterization_contract.v7",
            "adaptation_status": "implemented_mechanics_d20_completep_tpp20_authorized",
            "mechanics_executable": True,
            "compiled_bf16_qualification_status": (
                "same_release_d20_two_mode_distributed_canary_required_before_completep_full"
            ),
            "scientific_training_authorized": True,
            "scientific_training_authorization_condition": (
                "exact_locked_d20_completep_tpp20_plan_on_homogeneous_a100_or_a800_80gb_world_size_8"
            ),
            "scientific_execution_scope": D20_COMPLETEP_TPP20_EXECUTION_SCOPE,
            "campaign_plan_binding_required": True,
            "authorization_boundary": (
                "exact_locked_d20_canary_ordinary_prefix_completep_full_and_candidate_evaluation_profiles_only"
            ),
            "qualification_evidence_owner": "same_release_d20_two_mode_distributed_canary_and_current_release_tests",
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
