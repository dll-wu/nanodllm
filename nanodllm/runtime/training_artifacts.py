from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import torch.nn as nn

from .config import ExperimentConfig
from .eval import NELBO_SEQUENCE_RECORD_SCHEMA_VERSION
from .model import DiffusionConfig
from .parameter_state_hash import (
    LEGACY_PRE_UPDATE_PARAMETER_STATE_HASH_METHOD,
    _legacy_pre_update_parameter_state_sha256,
)
from .parameter_state_hash import (
    parameter_state_sha256 as parameter_state_sha256,
)
from .telemetry import flush_tensorboard_writer, write_json_atomic
from .telemetry import write_tensorboard_scalars as _write_tensorboard_scalars
from .training_checkpoint import _load_json_if_exists, alias_checkpoint, save_checkpoint
from .training_data_identity import _prepare_data_identity as _prepare_data_identity
from .training_features import (
    _eos_supervision_audit_enabled,
    _token_edit_probability_bounds,
    _token_edit_support_policy,
    _uses_differentiable_self_generated_draft,
    _uses_fixed_row_token_edit_layout,
    _uses_per_token_noise_levels,
    _uses_self_generated_token_edit,
    _uses_token_edit_corruption,
    _uses_token_edit_prefix,
)
from .training_progress import (
    PER_TOKEN_NOISE_LEVEL_HIST_BINS,
    POSITION_DECILE_BALANCED_LOSS_DECILES,
    TOKEN_EDIT_CORRUPTION_DECILES,
)
from .training_runtime import _numeric_runtime_telemetry, print0
from .training_self_generated_edit import self_generated_objective_configuration_telemetry


def _pre_update_parameter_state_artifact(
    raw_model: nn.Module,
    *,
    architecture_fingerprint_sha256: str,
    parameterization_contract_sha256: str = "",
    model_seed: int,
    model_initialization_source: dict[str, object] | None = None,
) -> dict[str, object]:
    """Hash the exact initialized parameter state without materializing a full CPU copy."""
    hash_method = LEGACY_PRE_UPDATE_PARAMETER_STATE_HASH_METHOD
    if parameterization_contract_sha256 and (
        len(parameterization_contract_sha256) != 64
        or any(character not in "0123456789abcdef" for character in parameterization_contract_sha256)
    ):
        raise ValueError("parameterization_contract_sha256 must be empty or a lowercase 64-hex digest")
    if model_initialization_source is None:
        state_point = "after_seeded_model_construction_before_optimizer_or_checkpoint_load"
        initialization_contract = {
            "schema": (
                "nanodllm.initialization_contract.v3"
                if parameterization_contract_sha256
                else "nanodllm.initialization_contract.v1"
            ),
            "architecture_fingerprint_sha256": architecture_fingerprint_sha256,
            "model_seed": int(model_seed),
            "parameter_state_hash_method": hash_method,
            "state_point": state_point,
        }
        initialization_mode = "seeded_model_construction"
    else:
        state_point = "after_verified_model_state_only_checkpoint_load_before_optimizer_or_update"
        extended_token_edit_initialization = bool(
            model_initialization_source.get("retained_seeded_token_edit_extension_state", False)
        )
        initialization_contract = {
            "schema": (
                "nanodllm.initialization_contract.v4"
                if extended_token_edit_initialization
                else "nanodllm.initialization_contract.v3"
                if parameterization_contract_sha256
                else "nanodllm.initialization_contract.v2"
            ),
            "architecture_fingerprint_sha256": architecture_fingerprint_sha256,
            "initialization_mode": "model_state_only_checkpoint",
            "source_checkpoint_sha256": model_initialization_source["sha256"],
            "source_checkpoint_size_bytes": model_initialization_source["size_bytes"],
            "source_checkpoint_step": model_initialization_source["source_step"],
            "source_checkpoint_train_tokens": model_initialization_source["source_train_tokens"],
            "source_embedded_model_config_sha256": model_initialization_source["embedded_model_config_sha256"],
            "source_embedded_train_config_sha256": model_initialization_source["embedded_train_config_sha256"],
            "parameter_state_hash_method": hash_method,
            "state_point": state_point,
        }
        if extended_token_edit_initialization:
            initialization_contract.update(
                {
                    "load_mode": model_initialization_source["load_mode"],
                    "token_edit_extension_tensor_names": model_initialization_source[
                        "token_edit_extension_tensor_names"
                    ],
                    "token_edit_extension_state_sha256": model_initialization_source[
                        "token_edit_extension_state_sha256"
                    ],
                }
            )
        source_parameterization_sha256 = model_initialization_source.get("source_parameterization_contract_sha256")
        if source_parameterization_sha256:
            initialization_contract["source_parameterization_contract_sha256"] = source_parameterization_sha256
        initialization_mode = "model_state_only_checkpoint"
    if parameterization_contract_sha256:
        initialization_contract["parameterization_contract_sha256"] = parameterization_contract_sha256
    initialization_contract_sha256 = hashlib.sha256(
        json.dumps(
            initialization_contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    state_sha256, parameter_count, parameter_numel, parameter_bytes = _legacy_pre_update_parameter_state_sha256(
        raw_model
    )
    artifact: dict[str, object] = {
        "schema": "nanodllm.initial_state.v1",
        "hash_method": hash_method,
        "state_point": state_point,
        "initialization_mode": initialization_mode,
        "pre_update_parameter_state_sha256": state_sha256,
        "initial_state_sha256": state_sha256,
        "initialization_contract": initialization_contract,
        "initialization_contract_sha256": initialization_contract_sha256,
        "architecture_fingerprint_sha256": architecture_fingerprint_sha256,
        "model_seed": int(model_seed),
        "parameter_count": int(parameter_count),
        "parameter_numel": int(parameter_numel),
        "parameter_bytes": int(parameter_bytes),
    }
    if model_initialization_source is not None:
        artifact["model_initialization_source"] = {
            key: value for key, value in model_initialization_source.items() if key not in {"path", "identity_seconds"}
        }
    if parameterization_contract_sha256:
        artifact["parameterization_contract_sha256"] = parameterization_contract_sha256
    return artifact


def write_jsonl_atomic(path: str, records: list[dict[str, object]]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    digest = hashlib.sha256()
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    os.replace(tmp_path, path)
    return digest.hexdigest()


def validate_existing_initial_state_artifact(path: str, artifact: dict[str, object]) -> None:
    """Fail before other startup artifacts can overwrite a different lineage."""

    artifact_path = Path(path)
    if artifact_path.is_symlink():
        raise RuntimeError(f"initial-state artifact must not be a symlink: {artifact_path}")
    if artifact_path.exists():
        existing = _load_json_if_exists(artifact_path)
        if existing != artifact:
            raise RuntimeError("existing initial_state.v1.json differs from the current initialization contract")


def write_initial_state_artifact_once(path: str, artifact: dict[str, object]) -> None:
    """Preserve an existing exact initialization artifact across child retries."""

    validate_existing_initial_state_artifact(path, artifact)
    artifact_path = Path(path)
    if artifact_path.exists():
        return
    write_json_atomic(str(artifact_path), artifact)


def _source_checkpoint_identity(path: str) -> tuple[str, int]:
    """Return the SHA-256 and byte size of the exact checkpoint file read for replay."""

    digest = hashlib.sha256()
    size_bytes = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _validated_nelbo_sequence_records(
    value: object,
    *,
    expected_count: int,
    require_shared_time: bool,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("NELBO diagnostics must contain a sequence_records list")
    if len(value) != expected_count:
        raise ValueError(f"NELBO sequence record count {len(value)} does not match expected count {expected_count}")
    records: list[dict[str, object]] = []
    nonnegative_numeric_fields = (
        "nelbo_nats_numerator",
        "masked_nats_numerator",
        "byte_denominator",
        "lossable_content_target_count",
        "masked_target_count",
        "schedule_weight_sum",
        "schedule_weight_count",
    )
    for index, raw_record in enumerate(value):
        if not isinstance(raw_record, dict):
            raise TypeError(f"NELBO sequence record {index} must be a JSON object")
        record = dict(raw_record)
        if record.get("schema_version") != NELBO_SEQUENCE_RECORD_SCHEMA_VERSION:
            raise ValueError(f"NELBO sequence record {index} has unsupported schema_version")
        if record.get("sequence_index") != index:
            raise ValueError(f"NELBO sequence record {index} has non-contiguous sequence_index")
        for field in nonnegative_numeric_fields:
            field_value = record.get(field)
            if (
                not isinstance(field_value, int | float)
                or isinstance(field_value, bool)
                or not math.isfinite(float(field_value))
                or float(field_value) < 0
            ):
                raise ValueError(f"NELBO sequence record {index} has invalid {field}")
        for field in ("sampled_t", "sampled_t_min", "sampled_t_max"):
            field_value = record.get(field)
            if field_value is not None and (
                not isinstance(field_value, int | float)
                or isinstance(field_value, bool)
                or not math.isfinite(float(field_value))
            ):
                raise ValueError(f"NELBO sequence record {index} has invalid {field}")
        if require_shared_time:
            sampled_t = record.get("sampled_t")
            if (
                sampled_t is None
                or sampled_t != record.get("sampled_t_min")
                or sampled_t != record.get("sampled_t_max")
            ):
                raise ValueError(f"NELBO sequence record {index} does not contain one shared sequence time")
        required_digest_fields = ["input_identity_sha256", "corruption_mask_sha256"]
        if require_shared_time:
            required_digest_fields.extend(("corruption_uniform_sha256", "evaluation_sample_sha256"))
        for field in required_digest_fields:
            digest_value = record.get(field)
            if (
                not isinstance(digest_value, str)
                or len(digest_value) != 64
                or any(character not in "0123456789abcdef" for character in digest_value)
            ):
                raise ValueError(f"NELBO sequence record {index} has invalid {field}")
        records.append(record)
    if math.fsum(float(record["byte_denominator"]) for record in records) <= 0:
        raise ValueError("NELBO sequence records have no positive byte denominator")
    return records


def _ordered_nelbo_digest_root(domain: bytes, records: list[dict[str, object]], field: str) -> str:
    digest = hashlib.sha256(domain + b"\0")
    digest.update(len(records).to_bytes(8, "little"))
    for record in records:
        value = record[field]
        if not isinstance(value, str):
            raise TypeError(f"NELBO sequence record {field} must be a SHA-256 string")
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def source_promotion_metrics() -> dict:
    """Include only informational Git facts in shareable run reports."""
    from ..source_provenance import source_provenance_metrics

    return source_provenance_metrics()


def _numeric_runtime_config_summary(config: ExperimentConfig) -> dict[str, bool | str | None]:
    return {
        "runtime_deterministic_algorithms_requested": bool(config.runtime.deterministic_algorithms),
        "runtime_sdpa_backend_requested": str(config.runtime.sdpa_backend),
        "runtime_cudnn_allow_tf32_requested": config.runtime.cudnn_allow_tf32,
        "runtime_compile": bool(config.runtime.compile),
        "runtime_compile_dynamic": bool(config.runtime.compile_dynamic),
        "runtime_dtype": str(config.runtime.dtype),
        "runtime_device_type": str(config.runtime.device_type),
    }


def autoresearch_config_summary(
    config: ExperimentConfig, entrypoint_mode: str
) -> dict[str, bool | float | int | str | None]:
    token_edit_probability_min, token_edit_probability_max = _token_edit_probability_bounds(config)
    token_edit_enabled = _uses_token_edit_corruption(config)
    self_generated_token_edit = _uses_self_generated_token_edit(config)
    if _uses_fixed_row_token_edit_layout(config):
        token_edit_training_forward_contract = (
            "no_grad_argmax_draft_on_4_t2t_rows_then_combined_db16_gradient_correction"
        )
    elif self_generated_token_edit:
        token_edit_training_forward_contract = "no_grad_argmax_draft_then_gradient_correction"
    elif token_edit_enabled:
        token_edit_training_forward_contract = "single_gradient_correction"
    else:
        token_edit_training_forward_contract = "disabled"
    summary: dict[str, bool | float | int | str | None] = {
        "entrypoint_mode": entrypoint_mode,
        "run_mode": "eval-only" if config.train.eval_only else "train",
        **_numeric_runtime_config_summary(config),
        **_numeric_runtime_telemetry(config),
        "model_depth": int(config.model.depth),
        "model_n_embd": int(config.model.n_embd),
        "model_n_head": int(config.model.n_head),
        "model_output_factorization": config.model.output_factorization,
        "model_block_causal_pilot_size": int(config.model.block_causal_pilot_size),
        "model_residual_output_projection_init": config.model.residual_output_projection_init,
        "model_embedding_head_init_std_multiplier": float(config.model.embedding_head_init_std_multiplier),
        "model_length_head": bool(config.model.length_head),
        "model_target_length_conditioning": bool(config.model.target_length_conditioning),
        "model_mask_position_embedding": bool(config.model.mask_position_embedding),
        "tokenizer": config.data.tokenizer,
        "tokenizer_path": config.data.tokenizer_path if config.data.tokenizer == "hf" else "",
        "target_tokens": int(config.train.target_tokens),
        "total_batch_size": int(config.train.total_batch_size),
        "device_batch_size": int(config.train.device_batch_size),
        "strict_mdlm_corruption_audit_only": bool(config.train.strict_mdlm_corruption_audit_only),
        "strict_mdlm_target_ce_audit_only": bool(config.train.strict_mdlm_target_ce_audit_only),
        "strict_mdlm_right_context_canary_only": bool(config.train.strict_mdlm_right_context_canary_only),
        "strict_mdlm_eos_overfit_canary_only": bool(config.train.strict_mdlm_eos_overfit_canary_only),
        "strict_mdlm_sampler_eos_reachability_canary_only": bool(
            config.train.strict_mdlm_sampler_eos_reachability_canary_only
        ),
        "strict_mdlm_learned_eos_sampler_canary_only": bool(config.train.strict_mdlm_learned_eos_sampler_canary_only),
        "strict_mdlm_real_collator_micro_overfit_canary_only": bool(
            config.train.strict_mdlm_real_collator_micro_overfit_canary_only
        ),
        "strict_mdlm_real_collator_component_probe": bool(config.train.strict_mdlm_real_collator_component_probe),
        "strict_mdlm_all_token_bayes_audit_only": bool(config.train.strict_mdlm_all_token_bayes_audit_only),
        "strict_mdlm_all_token_diagnostic_telemetry": bool(config.train.strict_mdlm_all_token_diagnostic_telemetry),
        "strict_mdlm_teacher_forced_replay_diagnostic": bool(config.train.strict_mdlm_teacher_forced_replay_diagnostic),
        "strict_mdlm_eos_loss_mass_balance": bool(config.train.strict_mdlm_eos_loss_mass_balance),
        "real_packing_audit_only": bool(config.train.real_packing_audit_only),
        "label_smoothing_epsilon": float(config.train.label_smoothing_epsilon),
        "entropy_bonus_beta": float(config.train.entropy_bonus_beta),
        "confidence_penalty_weight": float(config.train.confidence_penalty_weight),
        "z_loss_weight": float(config.train.z_loss_weight),
        "eos_hazard_loss_weight": float(config.train.eos_hazard_loss_weight),
        "length_head_loss_weight": float(config.train.length_head_loss_weight),
        "position_decile_balanced_loss": bool(config.train.position_decile_balanced_loss),
        "uniform_loss_estimator": config.train.uniform_loss_estimator,
        "position_decile_balanced_loss_deciles": POSITION_DECILE_BALANCED_LOSS_DECILES,
        "token_edit_corruption_enabled": token_edit_enabled,
        "token_edit_corruption_batch_fraction": float(config.train.token_edit_corruption_batch_fraction),
        "token_edit_batch_layout": config.train.token_edit_batch_layout,
        "token_edit_corruption_schedule": config.train.token_edit_corruption_schedule,
        "token_edit_corruption_ratio": float(config.train.token_edit_corruption_ratio),
        "token_edit_corruption_probability_min": token_edit_probability_min,
        "token_edit_corruption_probability_max": token_edit_probability_max,
        "token_edit_corruption_support_policy": _token_edit_support_policy(config),
        "token_edit_corruption_deciles": TOKEN_EDIT_CORRUPTION_DECILES,
        "token_edit_input_source": config.train.token_edit_input_source,
        "token_edit_correct_draft_loss_weight": float(config.train.token_edit_correct_draft_loss_weight),
        "token_edit_wrong_draft_loss_weight": float(config.train.token_edit_wrong_draft_loss_weight),
        "token_edit_backbone_gradient_mode": config.train.token_edit_backbone_gradient_mode,
        "token_edit_draft_gradient_policy": (
            "differentiable_mdlm_loss_with_detached_argmax_canvas"
            if _uses_differentiable_self_generated_draft(config)
            else "stopped_gradient"
            if self_generated_token_edit
            else "not_applicable"
        ),
        "token_edit_draft_m2t_loss_weight": float(config.train.token_edit_draft_m2t_loss_weight),
        "token_edit_correction_loss_weight": 1.0 - float(config.train.token_edit_draft_m2t_loss_weight),
        "token_edit_correct_draft_loss_role": "preservation" if self_generated_token_edit else "not_applicable",
        "token_edit_wrong_draft_loss_role": "repair" if self_generated_token_edit else "not_applicable",
        "token_edit_training_forward_contract": (
            f"{token_edit_training_forward_contract}:correction_gradient="
            f"{config.train.token_edit_backbone_gradient_mode}"
            if self_generated_token_edit
            else token_edit_training_forward_contract
        ),
        "noise_per_token_noise_levels": _uses_per_token_noise_levels(config),
        "noise_per_token_level_hist_bins": PER_TOKEN_NOISE_LEVEL_HIST_BINS,
        "data_boundary_aware_packing": bool(config.data.boundary_aware_packing),
        "data_content_budget_segmented_packing": bool(config.data.content_budget_segmented_packing),
        "data_synthetic_dataset": str(config.data.synthetic_dataset),
        "data_synthetic_eos_overfit_loss_mask_policy": str(config.data.synthetic_eos_overfit_loss_mask_policy),
        "data_short_document_curriculum": bool(config.data.short_document_curriculum),
        "data_short_document_max_tokens": int(config.data.short_document_max_tokens),
        "eval_basin_diagnostic": bool(config.eval.basin_diagnostic),
        "eval_source_checkpoint_identity_mode": config.eval.source_checkpoint_identity_mode,
        "eval_source_checkpoint_expected_sha256": config.eval.source_checkpoint_expected_sha256,
        "eval_source_checkpoint_expected_size_bytes": int(config.eval.source_checkpoint_expected_size_bytes),
        "eval_basin_prefix_len": int(config.eval.basin_prefix_len),
        "eval_basin_nucleus_top_p": float(config.eval.basin_nucleus_top_p),
        "eval_only_temperature_sweep": bool(config.eval.eval_only_temperature_sweep),
        "eval_length_policy": config.eval.length_policy,
        "eval_sample_top_k": int(config.eval.sample_top_k),
        "eval_sample_top_p": float(config.eval.sample_top_p),
        "eval_target_length_distribution_batches": int(config.eval.target_length_distribution_batches),
        "eos_supervision_audit_enabled": False
        if (
            config.train.strict_mdlm_sampler_eos_reachability_canary_only
            or config.train.strict_mdlm_learned_eos_sampler_canary_only
        )
        else _eos_supervision_audit_enabled(config),
        "noise_soft_input_curriculum": bool(config.noise.soft_input_curriculum),
        "noise_soft_input_curriculum_start_mix": float(config.noise.soft_input_curriculum_start_mix),
        "noise_soft_input_curriculum_end_mix": float(config.noise.soft_input_curriculum_end_mix),
        "noise_soft_input_curriculum_anneal_fraction": float(config.noise.soft_input_curriculum_anneal_fraction),
        "optim_lr": float(config.optim.lr),
        "optim_muon_lr": float(config.optim.muon_lr),
        "optim_adamw_lr": float(config.optim.adamw_lr),
        "optim_warmup_steps": int(config.optim.warmup_steps),
        "optim_warmup_ratio": float(config.optim.warmup_ratio),
        "optim_warmdown_ratio": float(config.optim.warmdown_ratio),
        "optim_final_lr_frac": float(config.optim.final_lr_frac),
        "optim_grad_clip": float(config.optim.grad_clip),
        "optim_adam_beta1": float(config.optim.adam_beta1),
        "optim_adam_beta2": float(config.optim.adam_beta2),
        "optim_adam_eps": float(config.optim.adam_eps),
        "eval_fixed_ce_trace_enabled": bool(config.eval.fixed_ce_trace_enabled),
    }
    token_edit_modeling_candidate = (
        config.model.token_edit_adapter_bottleneck_dim > 0
        or config.model.token_edit_provenance_embedding
        or config.train.token_edit_canvas_mode != "full_draft"
    )
    if token_edit_modeling_candidate:
        summary.update(
            {
                "model_token_edit_adapter_bottleneck_dim": int(config.model.token_edit_adapter_bottleneck_dim),
                "model_token_edit_adapter_top_layers": int(config.model.token_edit_adapter_top_layers),
                "model_token_edit_provenance_embedding": bool(config.model.token_edit_provenance_embedding),
                "model_token_edit_provenance_confidence_bins": int(config.model.token_edit_provenance_confidence_bins),
                "token_edit_canvas_mode": config.train.token_edit_canvas_mode,
                "token_edit_mixed_canvas_draft_fraction": float(config.train.token_edit_mixed_canvas_draft_fraction),
                "token_edit_trajectory_completion_fraction": float(
                    config.train.token_edit_trajectory_completion_fraction
                ),
                "token_edit_trajectory_steps": int(config.train.token_edit_trajectory_steps),
                "token_edit_provenance_embedding": bool(config.model.token_edit_provenance_embedding),
                "token_edit_full_draft_provenance": bool(config.train.token_edit_full_draft_provenance),
                "token_edit_adapter_bottleneck_dim": int(config.model.token_edit_adapter_bottleneck_dim),
                "token_edit_adapter_top_layers": int(config.model.token_edit_adapter_top_layers),
                "model_init_allow_token_edit_extensions": bool(config.train.model_init_allow_token_edit_extensions),
            }
        )
    summary.update(self_generated_objective_configuration_telemetry(config))
    if _uses_differentiable_self_generated_draft(config):
        summary.update(
            {
                "token_edit_draft_m2t_credit_mode": config.train.token_edit_draft_m2t_credit_mode,
                "token_edit_draft_m2t_credit_strength": float(config.train.token_edit_draft_m2t_credit_strength),
                "token_edit_draft_m2t_credit_clip_min": float(config.train.token_edit_draft_m2t_credit_clip_min),
                "token_edit_draft_m2t_credit_clip_max": float(config.train.token_edit_draft_m2t_credit_clip_max),
            }
        )
    if _uses_token_edit_prefix(config):
        summary.update(
            {
                "token_edit_prefix_enabled": True,
                "token_edit_prefix_canvas_mode": config.train.token_edit_prefix_canvas_mode,
                "token_edit_prefix_loss_scope": config.train.token_edit_prefix_loss_scope,
                "token_edit_prefix_min_fraction": float(config.train.token_edit_prefix_min_fraction),
                "token_edit_prefix_max_fraction": float(config.train.token_edit_prefix_max_fraction),
                "token_edit_prefix_boundary_seed": int(config.train.token_edit_prefix_boundary_seed),
                "token_edit_prefix_boundary_sampling": "sha256_keyed_per_attention_segment_v1",
            }
        )
    if config.train.token_edit_canvas_mode in {"trajectory_gold_context", "trajectory_model_context"}:
        summary.update(
            {
                "token_edit_trajectory_context_source": (
                    "gold_tokens_at_model_selected_commit_positions"
                    if config.train.token_edit_canvas_mode == "trajectory_gold_context"
                    else "model_argmax_tokens"
                ),
                "token_edit_trajectory_selection": "generation_filtered_top2_logit_margin",
                "token_edit_trajectory_loss_scope": "remaining_initially_masked_targets",
            }
        )
    return summary


def build_blocked_autoresearch_metrics(
    config: ExperimentConfig,
    entrypoint_mode: str,
    scale: str,
) -> dict[str, bool | float | int | list[str] | str]:
    metrics: dict[str, bool | float | int | list[str] | str] = {
        **source_promotion_metrics(),
        **autoresearch_config_summary(config, entrypoint_mode),
        "scale": scale,
        "training_seconds": 0.0,
        "val_nelbo_bpb": 0.0,
    }
    if metrics["status"] != "blocked":
        raise ValueError("build_blocked_autoresearch_metrics called without a source blocker")
    return metrics


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _run_artifact_manifest_relative_path(run_dir: str, artifact_path: str | Path) -> str:
    run_path = Path(run_dir).expanduser()
    run_path = (Path.cwd() / run_path).resolve() if not run_path.is_absolute() else run_path.resolve()
    path = Path(artifact_path).expanduser()
    path = (run_path / path).resolve() if not path.is_absolute() else path.resolve()

    if run_path.parent.name == "runs":
        try:
            return path.relative_to(run_path.parent.parent).as_posix()
        except ValueError:
            pass
    try:
        return path.relative_to(run_path).as_posix()
    except ValueError:
        return str(path)


def _register_run_artifacts(
    run_dir: str,
    artifacts: list[dict[str, object]],
) -> Path:
    run_path = Path(run_dir).expanduser()
    run_path = (Path.cwd() / run_path).resolve() if not run_path.is_absolute() else run_path.resolve()
    manifest_path = run_path / "artifacts.json"
    now = _utc_now_iso()
    if manifest_path.exists():
        manifest = _load_json_if_exists(manifest_path)
        if not isinstance(manifest.get("artifacts"), list):
            raise ValueError(f"cannot update non-versioned artifact manifest at {manifest_path}")
        manifest.setdefault("schema_version", 1)
        manifest.setdefault("run_id", run_path.name)
        manifest.setdefault("created_at", now)
        manifest.setdefault(
            "telemetry_dir", _run_artifact_manifest_relative_path(str(run_path), run_path / "telemetry")
        )
        manifest.setdefault("metadata", {})
    else:
        manifest = {
            "schema_version": 1,
            "run_id": run_path.name,
            "created_at": now,
            "updated_at": now,
            "artifacts": [],
            "telemetry_dir": _run_artifact_manifest_relative_path(str(run_path), run_path / "telemetry"),
            "metadata": {},
        }

    manifest["updated_at"] = now
    existing = {
        (
            str(artifact.get("path", "")),
            str(artifact.get("kind", "")),
            str(artifact.get("name", "")),
        ): idx
        for idx, artifact in enumerate(manifest.get("artifacts", []))
        if isinstance(artifact, dict)
    }
    for spec in artifacts:
        raw_path = Path(str(spec["path"])).expanduser()
        if raw_path.is_absolute():
            path_obj = raw_path
        else:
            cwd_candidate = (Path.cwd() / raw_path).resolve()
            path_obj = cwd_candidate if cwd_candidate.exists() else run_path / raw_path
        rel_path = _run_artifact_manifest_relative_path(str(run_path), path_obj)
        entry = {
            "path": rel_path,
            "kind": str(spec.get("kind", "artifact")),
            "name": str(spec.get("name", path_obj.name)),
            "phase": str(spec.get("phase", "run")),
            "required": bool(spec.get("required", False)),
            "exists": bool(path_obj.exists()),
            "metadata": dict(spec.get("metadata", {})) if isinstance(spec.get("metadata", {}), dict) else {},
        }
        key = (entry["path"], entry["kind"], entry["name"])
        if key in existing:
            manifest["artifacts"][existing[key]] = entry
        else:
            existing[key] = len(manifest["artifacts"])
            manifest["artifacts"].append(entry)

    write_json_atomic(str(manifest_path), manifest)
    return manifest_path


def _tokenizer_summary(tokenizer) -> dict:
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "mask_token_id": tokenizer.get_mask_token_id(),
        "eos_token_id": tokenizer.get_eos_token_id(),
        "bos_token_id": tokenizer.get_bos_token_id(),
        "pad_token_id": tokenizer.get_pad_token_id(),
        "special_token_ids": sorted(tokenizer.get_special_token_ids()),
        "generation_blocked_token_ids": sorted(tokenizer.get_generation_blocked_token_ids()),
    }


def _publish_final_outputs(
    config: ExperimentConfig,
    *,
    rank: int,
    metrics: dict,
    tensorboard_writer,
    total_steps: int,
    raw_model: nn.Module,
    optimizers,
    scaler,
    model_config: DiffusionConfig,
    train_tokens: int,
    train_valid_tokens: int,
    resume_contract: dict[str, object],
) -> None:
    metrics_path = os.path.join(config.train.run_dir, "metrics.json")
    final_checkpoint_path = os.path.join(config.train.run_dir, "model_latest.pt")
    final_checkpoint_alias_path = os.path.join(config.train.run_dir, "model.pt")

    _write_tensorboard_scalars(tensorboard_writer, "final", metrics, total_steps)
    flush_tensorboard_writer(tensorboard_writer)
    save_checkpoint(
        final_checkpoint_path,
        raw_model,
        optimizers,
        scaler,
        config,
        model_config,
        total_steps,
        train_tokens,
        train_valid_tokens,
        metrics,
        resume_contract=resume_contract,
    )
    alias_checkpoint(final_checkpoint_path, final_checkpoint_alias_path)
    write_json_atomic(metrics_path, metrics)
    print0(rank, f"metrics saved to {metrics_path}")
    print0(rank, f"model saved to {final_checkpoint_alias_path}")
