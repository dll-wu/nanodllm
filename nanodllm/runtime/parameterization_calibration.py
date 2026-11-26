"""Fail-closed admission and contract metadata for D4 root calibration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig


D4_ROOT_CALIBRATION_EXECUTION_SCOPE = "d4_mdlm_parameterization_root_calibration"
D4_ROOT_CALIBRATION_RECIPE = "d4-mdlm-parameterization-root-calibration-v1"
D4_ROOT_CALIBRATION_HIDDEN_INIT_MULTIPLIERS = (0.5, 1.0, 2.0)
D4_ROOT_CALIBRATION_ROOT_LRS = (2e-4, 4e-4, 8e-4)
D4_ROOT_CALIBRATION_RESPONSE_POINTS = frozenset(
    {
        (2e-4, 1.0),
        (4e-4, 0.5),
        (4e-4, 1.0),
        (4e-4, 2.0),
        (8e-4, 1.0),
    }
)
D4_ROOT_CALIBRATION_CANARY_POINT = (8e-4, 2.0)
D4_ROOT_CALIBRATION_SEED_BLOCKS = frozenset(
    {
        (1727378116, 1658537070, 965807166, 563885120),
        (1568148591, 1303904326, 1196331066, 1124590496),
        (117630881, 627291353, 211480254, 1406210302),
    }
)
D4_ROOT_CALIBRATION_EVALUATION_SEED_BLOCK = (963720344, 1901168930, 698728796, 839404335)
D4_ROOT_CALIBRATION_NELBO_SEED = 218165712
_D4_ROOT_CALIBRATION_COMMON_PROJECTION_SHA256 = "b427599bdc1d2259b9efb71b15f8d222b7c26418c1fb97a13be39a62fc3db39c"
_COMMON_PROJECTION_VARIABLE_PATHS = (
    "data.shuffle_seed",
    "optim.lr",
    "parameterization.mode",
    "parameterization.hidden_init_std_multiplier",
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
    "train.total_batch_size",
    "train.device_batch_size",
    "train.base_total_batch_size",
    "train.log_interval",
    "train.save_every",
    "train.resume_from",
    "train.eval_only",
    "eval.enabled",
    "eval.source_checkpoint_identity_mode",
    "eval.source_checkpoint_expected_sha256",
    "eval.source_checkpoint_expected_size_bytes",
    "diagnostics.parameter_state_hash_steps",
)


def is_d4_root_calibration(config: ExperimentConfig) -> bool:
    return config.parameterization.execution_scope == D4_ROOT_CALIBRATION_EXECUTION_SCOPE


def _equals_number(observed: object, expected: float | int) -> bool:
    return (
        not isinstance(observed, bool)
        and isinstance(observed, (int, float))
        and math.isfinite(float(observed))
        and float(observed) == float(expected)
    )


def _require_fields(config: ExperimentConfig, expected: Mapping[str, object], *, profile: str) -> None:
    for dotted_path, expected_value in expected.items():
        section_name, field_name = dotted_path.split(".", 1)
        observed = getattr(getattr(config, section_name), field_name)
        matches = (
            _equals_number(observed, expected_value)
            if isinstance(expected_value, int | float) and not isinstance(expected_value, bool)
            else observed == expected_value
        )
        if not matches:
            raise ValueError(
                f"D4 root-calibration {profile} profile requires {dotted_path}={expected_value!r}, "
                f"observed {observed!r}"
            )


def _calibration_profile(config: ExperimentConfig) -> str:
    if config.train.eval_only:
        return "evaluation"
    if int(config.train.num_steps) == 8:
        return "canary"
    if int(config.train.num_steps) == 7680:
        return "training"
    raise ValueError("D4 root calibration admits only the exact canary, full-TPP20 training, and evaluation profiles")


def _common_projection_sha256(config: ExperimentConfig) -> str:
    from .data_snapshot import strip_disabled_snapshot
    from .mdlm_auxiliary_config import strip_disabled_auxiliary
    from .model_mtp import strip_disabled_prediction_extensions
    from .telemetry_config import strip_disabled_telemetry

    value = asdict(config)
    strip_disabled_telemetry(value)
    strip_disabled_snapshot(value)
    strip_disabled_prediction_extensions(value)
    strip_disabled_auxiliary(value)
    from .data_mask_tail_config import strip_disabled_mask_tail

    strip_disabled_mask_tail(value)
    from .data_bos_packing_config import strip_disabled_bos_packing

    strip_disabled_bos_packing(value)
    from .lr_schedule import strip_legacy_lr_schedule

    strip_legacy_lr_schedule(value)
    from .config_validation_optim import strip_disabled_nanochat_muon

    strip_disabled_nanochat_muon(value)
    if value["noise"].pop("mdlm_time_sampling") != "legacy_auto":
        raise ValueError("historical root calibration requires legacy MDLM time sampling")
    if value["noise"].pop("mdlm_unweighted_loss_scale") != 1.0:
        raise ValueError("historical root calibration requires default loss scale")
    if value["model"].pop("backbone_profile") != "nanodllm":
        raise ValueError("the root calibration requires the historical backbone")
    if value["train"].pop("token_edit_backbone_gradient_mode") != "shared":
        raise ValueError("D4 root calibration requires shared token-edit backbone gradients")
    if value["train"].pop("token_edit_batch_layout") != "whole_microbatch_cadence":
        raise ValueError("D4 root calibration requires the historical token-edit batch layout")
    if value["model"].pop("token_edit_adapter_bottleneck_dim") != 0:
        raise ValueError("D4 root calibration requires token-edit adapters to be disabled")
    if value["model"].pop("token_edit_adapter_top_layers") != 0:
        raise ValueError("D4 root calibration requires token-edit adapters to be disabled")
    if value["model"].pop("token_edit_provenance_embedding") is not False:
        raise ValueError("D4 root calibration requires token provenance to be disabled")
    if value["model"].pop("token_edit_provenance_confidence_bins") != 0:
        raise ValueError("D4 root calibration requires confidence provenance to be disabled")
    if value["train"].pop("model_init_allow_token_edit_extensions") is not False:
        raise ValueError("D4 root calibration forbids extended model initialization")
    if value["train"].pop("token_edit_canvas_mode") != "full_draft":
        raise ValueError("D4 root calibration requires the historical full-draft canvas")
    if value["train"].pop("token_edit_mixed_canvas_draft_fraction") != 1.0:
        raise ValueError("D4 root calibration requires the historical full-draft fraction")
    if value["train"].pop("token_edit_trajectory_completion_fraction") != 0.5:
        raise ValueError("D4 root calibration requires the default trajectory fraction")
    if value["train"].pop("token_edit_trajectory_steps") != 4:
        raise ValueError("D4 root calibration requires the default trajectory steps")
    if value["train"].pop("token_edit_correction_objective") != "cross_entropy":
        raise ValueError("D4 root calibration requires the historical correction objective")
    if value["train"].pop("token_edit_correction_aux_weight") != 0.0:
        raise ValueError("D4 root calibration requires no correction auxiliary")
    if value["train"].pop("token_edit_expected_accuracy_power") != 1.0:
        raise ValueError("D4 root calibration requires the historical expected-accuracy power")
    if value["train"].pop("token_edit_expected_accuracy_power_schedule") != "constant":
        raise ValueError("D4 root calibration requires constant expected-accuracy power")
    if value["train"].pop("token_edit_expected_accuracy_corruption_threshold") != 0.8:
        raise ValueError("D4 root calibration requires the default expected-accuracy threshold")
    if value["train"].pop("token_edit_cross_view_nonregression_weight") != 0.0:
        raise ValueError("D4 root calibration requires cross-view preservation to be disabled")
    if value["train"].pop("token_edit_draft_m2t_loss_weight") != 0.0:
        raise ValueError("D4 root calibration requires no joint draft objective")
    frozen_draft_defaults = {
        "token_edit_draft_model_source": "live_student",
        "token_edit_draft_checkpoint": "",
        "token_edit_draft_checkpoint_expected_sha256": "",
        "token_edit_draft_checkpoint_expected_size_bytes": -1,
    }
    for field_name, expected in frozen_draft_defaults.items():
        if value["train"].pop(field_name) != expected:
            raise ValueError(f"D4 root calibration requires train.{field_name}={expected!r}")
    credit_defaults = {
        "token_edit_draft_m2t_credit_mode": "uniform",
        "token_edit_draft_m2t_credit_strength": 1.0,
        "token_edit_draft_m2t_credit_clip_min": 0.5,
        "token_edit_draft_m2t_credit_clip_max": 1.5,
    }
    for field_name, expected in credit_defaults.items():
        if value["train"].pop(field_name) != expected:
            raise ValueError(f"D4 root calibration requires train.{field_name}={expected!r}")
    if value["train"].pop("token_edit_full_draft_provenance") is not False:
        raise ValueError("D4 root calibration requires full-draft provenance to be disabled")
    prefix_defaults = {
        "token_edit_prefix_enabled": False,
        "token_edit_prefix_canvas_mode": "fallible_draft",
        "token_edit_prefix_loss_scope": "all_active",
        "token_edit_prefix_min_fraction": 0.125,
        "token_edit_prefix_max_fraction": 0.75,
        "token_edit_prefix_boundary_seed": 271828,
    }
    for field_name, expected in prefix_defaults.items():
        if value["train"].pop(field_name) != expected:
            raise ValueError(f"D4 root calibration requires train.{field_name}={expected!r}")
    block_conversion_defaults = {
        ("model", "bd3lm_clean_stream_mode"): "block_causal",
        ("train", "bd3lm_uniform_t2t_batch_fraction"): 0.0,
        ("train", "bd3lm_causal_ar_loss_weight"): 0.0,
        ("train", "bd3lm_block_size_schedule"): "fixed",
        ("train", "bd3lm_block_size_schedule_start_step_indices"): [],
        ("train", "bd3lm_block_size_schedule_values"): [],
    }
    for (section, field_name), expected in block_conversion_defaults.items():
        if value[section].pop(field_name) != expected:
            raise ValueError(f"D4 root calibration requires {section}.{field_name}={expected!r}")
    from .config_validation_block_recipe import project_disabled_block_recipe_defaults

    project_disabled_block_recipe_defaults(value)
    for dotted_path in _COMMON_PROJECTION_VARIABLE_PATHS:
        section_name, field_name = dotted_path.split(".", 1)
        value[section_name][field_name] = "<campaign-bound>"
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_d4_root_calibration_config(config: ExperimentConfig) -> str:
    """Admit only the exact campaign envelope; return its inferred profile."""

    _require_fields(
        config,
        {
            "recipe.name": D4_ROOT_CALIBRATION_RECIPE,
            "recipe.status": "locked",
            "model.depth": 4,
            "model.n_embd": 256,
            "model.n_head": 2,
            "model.max_seq_len": 2048,
            "model.mlp_hidden_dim": 1024,
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
            "parameterization.contract_binding": "required",
            "runtime.device_type": "cuda",
            "runtime.dtype": "bfloat16",
            "runtime.cudnn_benchmark": False,
            "runtime.deterministic_algorithms": True,
            "train.target_param_count_basis": "nanochat_scaling_parameters",
            "train.allow_batch_size_round_up": False,
            "train.scale_lr_by_batch": False,
            "train.tensorboard": False,
            "train.eval_every": -1,
            "train.auxiliary_loss_weight": 0.0,
        },
        profile="common",
    )
    if config.parameterization.mode not in {"ordinary", "mup", "completep"}:
        raise ValueError("D4 root calibration permits only ordinary, mup, and completep modes")
    hidden_init = float(config.parameterization.hidden_init_std_multiplier)
    if hidden_init not in D4_ROOT_CALIBRATION_HIDDEN_INIT_MULTIPLIERS:
        raise ValueError("D4 root calibration permits hidden_init_std_multiplier only at 0.5, 1.0, or 2.0")
    root_lr = float(config.optim.lr)
    if root_lr not in D4_ROOT_CALIBRATION_ROOT_LRS:
        raise ValueError("D4 root calibration permits root learning rate only at 2e-4, 4e-4, or 8e-4")

    profile = _calibration_profile(config)
    seed_block = (
        int(config.runtime.seed),
        int(config.runtime.model_seed),
        int(config.runtime.training_seed),
        int(config.data.shuffle_seed),
    )
    if profile == "canary":
        if seed_block != (1727378116, 1658537070, 965807166, 563885120):
            raise ValueError("D4 root-calibration canary requires exact seed block b0")
        if (root_lr, hidden_init) != D4_ROOT_CALIBRATION_CANARY_POINT:
            raise ValueError("D4 root-calibration canary requires the aggressive lr=8e-4, init=2 point")
        _require_fields(
            config,
            {
                "train.num_steps": 8,
                "train.target_tokens": 131072,
                "train.target_param_data_ratio": -1.0,
                "train.total_batch_size": 16384,
                "train.device_batch_size": 8,
                "train.base_total_batch_size": 16384,
                "train.log_interval": 1,
                "train.save_every": -1,
                "train.resume_from": "",
                "train.eval_only": False,
                "train.divergence_check_interval": 10,
                "runtime.compile": True,
                "runtime.compile_dynamic": True,
                "eval.enabled": False,
                "diagnostics.parameter_state_hash_steps": [1, 8],
            },
            profile=profile,
        )
    elif profile == "training":
        if seed_block not in D4_ROOT_CALIBRATION_SEED_BLOCKS:
            raise ValueError("D4 root-calibration training requires one of its three exact role-separated seed blocks")
        if (root_lr, hidden_init) not in D4_ROOT_CALIBRATION_RESPONSE_POINTS:
            raise ValueError("D4 root-calibration training requires one of the five frozen response points")
        if config.train.resume_from and not Path(config.train.resume_from).expanduser().is_absolute():
            raise ValueError("D4 root-calibration training resume checkpoint must be an absolute path")
        _require_fields(
            config,
            {
                "train.num_steps": 7680,
                "train.target_tokens": 251658240,
                "train.target_param_data_ratio": 20.0,
                "train.total_batch_size": 32768,
                "train.device_batch_size": 16,
                "train.base_total_batch_size": 32768,
                "train.log_interval": 100,
                "train.save_every": 1920,
                "train.eval_only": False,
                "train.divergence_check_interval": 10,
                "runtime.compile": True,
                "runtime.compile_dynamic": True,
                "eval.enabled": False,
                "diagnostics.parameter_state_hash_steps": [],
            },
            profile=profile,
        )
    else:
        if seed_block != D4_ROOT_CALIBRATION_EVALUATION_SEED_BLOCK:
            raise ValueError("D4 root-calibration evaluation requires the dedicated selection-eval seed block")
        if (root_lr, hidden_init) not in D4_ROOT_CALIBRATION_RESPONSE_POINTS:
            raise ValueError("D4 root-calibration evaluation requires one of the five frozen response points")
        if not config.train.resume_from:
            raise ValueError("D4 root-calibration evaluation requires its bound training checkpoint")
        if not Path(config.train.resume_from).expanduser().is_absolute():
            raise ValueError("D4 root-calibration evaluation checkpoint must be an absolute path")
        _require_fields(
            config,
            {
                "train.num_steps": -1,
                "train.target_tokens": -1,
                "train.target_param_data_ratio": 20.0,
                "train.total_batch_size": 32768,
                "train.device_batch_size": 16,
                "train.base_total_batch_size": 32768,
                "train.log_interval": 100,
                "train.save_every": -1,
                "train.eval_only": True,
                "runtime.compile": False,
                "runtime.compile_dynamic": False,
                "eval.enabled": True,
                "eval.gen_eval": False,
                "eval.gen_eval_periodic": False,
                "eval.nelbo_tokens": 1048576,
                "eval.nelbo_t_min": 0.0,
                "eval.nelbo_t_max": 1.0,
                "eval.nelbo_seed": D4_ROOT_CALIBRATION_NELBO_SEED,
                "eval.nelbo_time_sampling": "stratified_per_sequence",
                "eval.source_checkpoint_identity_mode": "sha256",
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
    observed_projection = _common_projection_sha256(config)
    if observed_projection != _D4_ROOT_CALIBRATION_COMMON_PROJECTION_SHA256:
        raise ValueError(
            f"D4 root calibration differs from its exact frozen common field projection: observed={observed_projection}"
        )
    return profile


def calibration_contract_material() -> dict[str, object]:
    """Return truthful v3 authorization metadata for an admitted config."""

    return {
        "schema": "nanodllm.parameterization_contract.v3",
        "adaptation_status": "implemented_mechanics_d4_root_calibration_authorized",
        "mechanics_executable": True,
        "compiled_bf16_qualification_status": "same_release_campaign_gate_required_before_full_training",
        "scientific_training_authorized": True,
        "scientific_training_authorization_condition": "exact_same_release_three_mode_canary_gate_passed",
        "scientific_execution_scope": D4_ROOT_CALIBRATION_EXECUTION_SCOPE,
        "campaign_plan_binding_required": True,
        "authorization_boundary": "exact_locked_d4_root_calibration_recipe_and_immutable_campaign_plan_only",
        "qualification_evidence_owner": "immutable_campaign_plan_and_same_release_canary",
    }


def calibration_parameterization_contract(
    contract: Mapping[str, object], mechanics: Mapping[str, object]
) -> dict[str, object]:
    """Upgrade an admitted legacy mechanics contract to scoped v3 semantics."""

    material = copy.deepcopy({key: value for key, value in contract.items() if key != "contract_sha256"})
    material.pop("launchable", None)
    material.pop("reserved_modes", None)
    material["mechanics_executable_modes"] = ["ordinary", "mup", "completep"]
    material["mechanics"] = copy.deepcopy(dict(mechanics))
    for collection_name in ("parameter_records", "parameter_groups"):
        collection = material.get(collection_name)
        if isinstance(collection, list):
            for record in collection:
                if isinstance(record, dict):
                    record.pop("prospective_nonordinary_status", None)
    material.update(calibration_contract_material())
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {**material, "contract_sha256": hashlib.sha256(encoded).hexdigest()}
