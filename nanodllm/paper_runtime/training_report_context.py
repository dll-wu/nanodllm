from __future__ import annotations

from collections.abc import Mapping

from .training_artifacts import autoresearch_config_summary
from .training_checkpoint import RESUME_CONTRACT_VERSION
from .training_compute import (
    BD3LM_DENSE_CONCAT_FLOPS_ACCOUNTING_METHOD,
    compute_budget_fields,
    training_compute_estimate,
)
from .training_features import (
    _is_causal,
    _self_generated_draft_device_batch_size,
    _uses_bd3lm,
    _uses_bd3lm_causal_ar_auxiliary,
    _uses_differentiable_self_generated_draft,
    _uses_uniform_low_variance_estimator,
)
from .training_metrics import (
    _embedding_head_init_metrics,
    _residual_projection_init_metrics,
    _token_ratio_metrics,
)
from .training_mtp import mtp_training_enabled
from .training_objective import (
    _bd3lm_telemetry,
    _causal_ar_label_smoothing,
    _hierarchical_output_telemetry,
    _noise_telemetry,
    _token_edit_corruption_telemetry,
    _training_corruption_label,
)
from .training_plan import _lr_schedule_metric_fields, _selected_target_param_count
from .training_progress import EOS_HAZARD_RELATIVE_POSITION_DECILES
from .training_report_inputs import FinalReportInputs
from .training_report_schema import FieldGroup, ReportFragment, merge_field_groups
from .training_self_generated_edit import self_generated_objective_configuration_telemetry

_RECIPE_OWNED_NOISE_FIELDS = frozenset(
    {
        "noise_per_token_noise_levels",
        "noise_soft_input_curriculum",
        "noise_soft_input_curriculum_anneal_fraction",
        "noise_soft_input_curriculum_end_mix",
        "noise_soft_input_curriculum_start_mix",
    }
)
_RECIPE_OWNED_TOKEN_EDIT_FIELDS = frozenset(
    {
        "token_edit_corruption_batch_fraction",
        "token_edit_batch_layout",
        "token_edit_corruption_deciles",
        "token_edit_corruption_enabled",
        "token_edit_corruption_probability_max",
        "token_edit_corruption_probability_min",
        "token_edit_corruption_ratio",
        "token_edit_corruption_schedule",
        "token_edit_corruption_support_policy",
        "token_edit_backbone_gradient_mode",
        "token_edit_canvas_mode",
        "token_edit_mixed_canvas_draft_fraction",
        "token_edit_trajectory_completion_fraction",
        "token_edit_trajectory_steps",
        "token_edit_trajectory_context_source",
        "token_edit_trajectory_selection",
        "token_edit_trajectory_loss_scope",
        "token_edit_provenance_embedding",
        "token_edit_full_draft_provenance",
        "token_edit_adapter_bottleneck_dim",
        "token_edit_adapter_top_layers",
        "token_edit_correction_objective",
        "token_edit_correction_aux_weight",
        "token_edit_expected_accuracy_power",
        "token_edit_expected_accuracy_power_schedule",
        "token_edit_expected_accuracy_corruption_threshold",
        "token_edit_expected_accuracy_high_corruption_power",
        "token_edit_expected_accuracy_conditioning_signal",
        "token_edit_expected_accuracy_conditioning_rule",
        "token_edit_cross_view_nonregression_weight",
        "token_edit_cross_view_nonregression_scope",
        "token_edit_cross_view_nonregression_signal",
        "token_edit_draft_m2t_loss_weight",
        "token_edit_correction_loss_weight",
        "token_edit_draft_m2t_credit_mode",
        "token_edit_draft_m2t_credit_strength",
        "token_edit_draft_m2t_credit_clip_min",
        "token_edit_draft_m2t_credit_clip_max",
        "token_edit_prefix_enabled",
        "token_edit_prefix_canvas_mode",
        "token_edit_prefix_loss_scope",
        "token_edit_prefix_min_fraction",
        "token_edit_prefix_max_fraction",
        "token_edit_prefix_boundary_seed",
        "token_edit_prefix_boundary_sampling",
    }
)
_EOS_SHARED_FIELDS = frozenset(
    {
        "eos_supervision_audit_artifact",
        "eos_supervision_audit_enabled",
        "eos_supervision_audit_path",
        "eos_supervision_invariant_error",
        "eos_supervision_invariant_status",
    }
)

_RAW_PARAMETER_COUNT_FIELDS = (
    ("num_params", "total"),
    ("num_non_embedding_params", "non_embedding"),
    ("num_embedding_params", "wte"),
    ("num_wte_params", "wte"),
    ("num_lm_head_params", "lm_head"),
    ("num_class_head_params", "class_head"),
    ("num_length_head_params", "length_head"),
    ("num_target_length_conditioning_params", "target_length_conditioning"),
    ("num_mask_position_embedding_params", "mask_position_embedding"),
    ("num_auxiliary_head_params", "auxiliary_heads"),
    ("num_transformer_params", "transformer"),
)
_MILLION_PARAMETER_COUNT_FIELDS = (
    ("num_params_M", "total"),
    ("num_non_embedding_params_M", "non_embedding"),
    ("num_scaling_params_M", "scaling"),
    ("num_embedding_params_M", "wte"),
    ("num_wte_params_M", "wte"),
    ("num_lm_head_params_M", "lm_head"),
    ("num_class_head_params_M", "class_head"),
    ("num_length_head_params_M", "length_head"),
    ("num_target_length_conditioning_params_M", "target_length_conditioning"),
    ("num_mask_position_embedding_params_M", "mask_position_embedding"),
    ("num_auxiliary_head_params_M", "auxiliary_heads"),
    ("num_transformer_params_M", "transformer"),
)


def _without_fields(fields: Mapping[str, object], omitted: frozenset[str]) -> dict[str, object]:
    """Copy a helper mapping without fields owned by another report fragment."""

    missing = omitted - fields.keys()
    if missing:
        raise KeyError(f"report helper is missing expected fields: {sorted(missing)}")
    return {key: value for key, value in fields.items() if key not in omitted}


def _build_execution_context(inputs: FinalReportInputs) -> ReportFragment:
    return ReportFragment(
        "execution_context",
        {
            "scale": "small",
            "experiment_type": "causal-ar-diagnostic"
            if _is_causal(inputs.config)
            else "infrastructure-diagnostic"
            if inputs.config.noise.type == "mask_uniform"
            else "trained-model",
            "causal_ar": _is_causal(inputs.config),
            "causal_ar_auxiliary": _uses_bd3lm_causal_ar_auxiliary(inputs.config),
            "cpu_threading": inputs.cpu_threading,
            "distributed_cuda_memory": inputs.distributed_cuda_memory,
            "distributed_minimum_free_gpu_mib": inputs.distributed_cuda_memory["minimum_free_mib"],
            "distributed_minimum_free_gpu_fraction": inputs.distributed_cuda_memory["minimum_free_fraction"],
            "distributed_minimum_peak_reserved_headroom_fraction": inputs.distributed_cuda_memory[
                "minimum_peak_reserved_headroom_fraction"
            ],
            "distributed_maximum_peak_allocated_gpu_mib": inputs.distributed_cuda_memory["maximum_peak_allocated_mib"],
            "distributed_maximum_peak_reserved_gpu_mib": inputs.distributed_cuda_memory["maximum_peak_reserved_mib"],
        },
    )


def _build_recipe_context(inputs: FinalReportInputs) -> ReportFragment:
    """Resolve compatibility-header fields once and verify duplicated producers agree."""

    config = inputs.config
    fields = autoresearch_config_summary(config, inputs.entrypoint_mode)
    expected_config_values = {
        "model_depth": int(config.model.depth),
        "model_n_embd": int(config.model.n_embd),
        "model_n_head": int(config.model.n_head),
        "tokenizer": config.data.tokenizer,
        "tokenizer_path": config.data.tokenizer_path if config.data.tokenizer == "hf" else "",
        "target_tokens": int(config.train.target_tokens),
        "device_batch_size": int(config.train.device_batch_size),
        "label_smoothing_epsilon": float(config.train.label_smoothing_epsilon),
        "confidence_penalty_weight": float(config.train.confidence_penalty_weight),
        "z_loss_weight": float(config.train.z_loss_weight),
        "eos_hazard_loss_weight": float(config.train.eos_hazard_loss_weight),
        "length_head_loss_weight": float(config.train.length_head_loss_weight),
        "uniform_loss_estimator": config.train.uniform_loss_estimator,
        "token_edit_input_source": config.train.token_edit_input_source,
        "token_edit_correct_draft_loss_weight": float(config.train.token_edit_correct_draft_loss_weight),
        "token_edit_wrong_draft_loss_weight": float(config.train.token_edit_wrong_draft_loss_weight),
        "token_edit_backbone_gradient_mode": config.train.token_edit_backbone_gradient_mode,
        "token_edit_batch_layout": config.train.token_edit_batch_layout,
        "data_boundary_aware_packing": bool(config.data.boundary_aware_packing),
        "data_content_budget_segmented_packing": bool(config.data.content_budget_segmented_packing),
        "data_short_document_curriculum": bool(config.data.short_document_curriculum),
        "data_short_document_max_tokens": int(config.data.short_document_max_tokens),
    }
    if (
        config.model.token_edit_adapter_bottleneck_dim > 0
        or config.model.token_edit_provenance_embedding
        or config.train.token_edit_canvas_mode != "full_draft"
    ):
        expected_config_values.update(
            {
                "token_edit_canvas_mode": config.train.token_edit_canvas_mode,
                "token_edit_mixed_canvas_draft_fraction": float(config.train.token_edit_mixed_canvas_draft_fraction),
                "token_edit_trajectory_completion_fraction": float(
                    config.train.token_edit_trajectory_completion_fraction
                ),
                "token_edit_trajectory_steps": int(config.train.token_edit_trajectory_steps),
                "model_token_edit_adapter_bottleneck_dim": int(config.model.token_edit_adapter_bottleneck_dim),
                "model_token_edit_adapter_top_layers": int(config.model.token_edit_adapter_top_layers),
                "model_token_edit_provenance_embedding": bool(config.model.token_edit_provenance_embedding),
                "model_token_edit_provenance_confidence_bins": int(config.model.token_edit_provenance_confidence_bins),
                "token_edit_full_draft_provenance": bool(config.train.token_edit_full_draft_provenance),
                "model_init_allow_token_edit_extensions": bool(config.train.model_init_allow_token_edit_extensions),
            }
        )
    expected_config_values.update(self_generated_objective_configuration_telemetry(config))
    expected_config_values.update(
        {
            "token_edit_draft_m2t_loss_weight": float(config.train.token_edit_draft_m2t_loss_weight),
            "token_edit_correction_loss_weight": 1.0 - float(config.train.token_edit_draft_m2t_loss_weight),
        }
    )
    if _uses_differentiable_self_generated_draft(config):
        expected_config_values.update(
            {
                "token_edit_draft_m2t_credit_mode": config.train.token_edit_draft_m2t_credit_mode,
                "token_edit_draft_m2t_credit_strength": float(config.train.token_edit_draft_m2t_credit_strength),
                "token_edit_draft_m2t_credit_clip_min": float(config.train.token_edit_draft_m2t_credit_clip_min),
                "token_edit_draft_m2t_credit_clip_max": float(config.train.token_edit_draft_m2t_credit_clip_max),
            }
        )
    if config.train.token_edit_prefix_enabled:
        expected_config_values.update(
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
    for key, expected in expected_config_values.items():
        if fields.get(key) != expected:
            raise ValueError(f"recipe context field {key!r} disagrees with its configured value")

    noise_fields = _noise_telemetry(config)
    token_edit_fields = _token_edit_corruption_telemetry(config)
    for key in _RECIPE_OWNED_NOISE_FIELDS:
        if fields.get(key) != noise_fields[key]:
            raise ValueError(f"recipe context field {key!r} disagrees with noise telemetry")
    for key in _RECIPE_OWNED_TOKEN_EDIT_FIELDS:
        if key in fields or key in token_edit_fields:
            if fields.get(key) == token_edit_fields.get(key):
                continue
            raise ValueError(f"recipe context field {key!r} disagrees with token-edit telemetry")

    resolved_values = {
        "total_batch_size": inputs.total_batch_size,
        "eos_supervision_audit_enabled": bool(inputs.eos_supervision_train_metrics["eos_supervision_audit_enabled"]),
    }
    missing = resolved_values.keys() - fields.keys()
    if missing:
        raise KeyError(f"recipe summary is missing expected resolved fields: {sorted(missing)}")
    resolved_fields = {key: resolved_values.get(key, value) for key, value in fields.items()}
    return ReportFragment("recipe_context", resolved_fields)


def _build_eos_supervision(inputs: FinalReportInputs) -> ReportFragment:
    startup = inputs.eos_supervision_startup_metrics
    training = inputs.eos_supervision_train_metrics
    startup_missing = _EOS_SHARED_FIELDS - startup.keys()
    training_missing = _EOS_SHARED_FIELDS - training.keys()
    if startup_missing or training_missing:
        raise KeyError(
            "EOS supervision sources are missing shared fields: "
            f"startup={sorted(startup_missing)}, training={sorted(training_missing)}"
        )
    startup_and_final_status = {
        key: training[key] if key in _EOS_SHARED_FIELDS else value
        for key, value in startup.items()
        if key != "eos_supervision_audit_enabled"
    }
    training_counts = {key: value for key, value in training.items() if key not in _EOS_SHARED_FIELDS}
    return ReportFragment(
        "eos_supervision",
        merge_field_groups(
            FieldGroup("startup counts and final status", startup_and_final_status),
            FieldGroup("training counts", training_counts),
        ),
    )


def _build_model_resolution(inputs: FinalReportInputs) -> ReportFragment:
    return ReportFragment(
        "model_resolution",
        {
            "model_mlp_hidden_dim": int(inputs.model_config.mlp_hidden_dim),
            "model_resolved_mlp_hidden_dim": int(inputs.raw_model.blocks[0].mlp.up.out_features),
            "model_attention_scale": float(inputs.model_config.attention_scale),
            "model_resolved_attention_scale": float(
                inputs.raw_model.architecture_fingerprint()["attention"]["resolved_scale"]
            ),
            "architecture_fingerprint_sha256": inputs.architecture_fingerprint["fingerprint_sha256"],
            "pre_update_parameter_state_sha256": inputs.initial_state_artifact["pre_update_parameter_state_sha256"],
            "initial_state_sha256": inputs.initial_state_artifact["initial_state_sha256"],
            "initialization_contract_sha256": inputs.initial_state_artifact["initialization_contract_sha256"],
        },
    )


def _build_feature_configuration_fragments(inputs: FinalReportInputs) -> tuple[ReportFragment, ...]:
    token_edit_telemetry = _token_edit_corruption_telemetry(inputs.config)
    recipe_owned_token_edit_fields = frozenset(_RECIPE_OWNED_TOKEN_EDIT_FIELDS & token_edit_telemetry.keys())
    return (
        ReportFragment(
            "noise_configuration",
            _without_fields(_noise_telemetry(inputs.config), _RECIPE_OWNED_NOISE_FIELDS),
        ),
        ReportFragment(
            "token_edit_configuration",
            _without_fields(token_edit_telemetry, recipe_owned_token_edit_fields),
        ),
        ReportFragment("bd3lm_configuration", _bd3lm_telemetry(inputs.config)),
        ReportFragment(
            "hierarchical_output_configuration", _hierarchical_output_telemetry(inputs.token_class_metadata)
        ),
    )


def _build_initialization_diagnostics(inputs: FinalReportInputs) -> ReportFragment:
    return ReportFragment(
        "initialization_diagnostics",
        merge_field_groups(
            FieldGroup("residual projection initialization", _residual_projection_init_metrics(inputs.config)),
            FieldGroup("embedding and head initialization", _embedding_head_init_metrics(inputs.config)),
        ),
    )


def _build_objective_definition(inputs: FinalReportInputs) -> ReportFragment:
    return ReportFragment(
        "objective_definition",
        merge_field_groups(
            FieldGroup("training objective", {"training_corruption": _training_corruption_label(inputs.config)}),
            FieldGroup("clean label smoothing", inputs.clean_label_smoothing_metrics),
        ),
    )


def _build_data_identity(inputs: FinalReportInputs) -> ReportFragment:
    return ReportFragment(
        "data_identity",
        {
            "data_identity_enabled": bool(inputs.data_identity_summary["enabled"]),
            "data_identity_mode": str(inputs.data_identity_summary["mode"]),
            "data_identity_artifact": str(inputs.data_identity_summary["artifact"]),
            "data_identity_sha256": str(inputs.data_identity_summary["identity_sha256"]),
            "data_identity_sha256_strength": str(inputs.data_identity_summary["identity_sha256_strength"]),
        },
    )


def _build_objective_options(inputs: FinalReportInputs) -> ReportFragment:
    token_edit_metadata = inputs.token_edit_class_metadata or {}
    return ReportFragment(
        "objective_options",
        {
            "ar_label_smoothing_epsilon": _causal_ar_label_smoothing(inputs.config),
            "eos_hazard_relative_position_deciles": EOS_HAZARD_RELATIVE_POSITION_DECILES,
            "length_head_enabled": bool(inputs.config.model.length_head),
            "length_policy": inputs.config.eval.length_policy,
            "target_length_conditioning_enabled": bool(inputs.config.model.target_length_conditioning),
            "mask_position_embedding_enabled": bool(inputs.config.model.mask_position_embedding),
            "target_length_conditioning_empirical_prior_batches": int(
                inputs.config.eval.target_length_distribution_batches
            ),
            "target_length_conditioning_forced_eos_policy": inputs.config.eval.length_policy,
            "uniform_low_variance_estimator_enabled": _uses_uniform_low_variance_estimator(inputs.config),
            "uniform_estimator_configured_sample_count": 2
            if _uses_uniform_low_variance_estimator(inputs.config)
            else 1,
            "token_edit_frequency_batches": int(token_edit_metadata.get("frequency_batches", 0)),
            "token_edit_frequency_observed_tokens": int(token_edit_metadata.get("frequency_observed_tokens", 0)),
            "token_edit_frequency_nonzero_token_types": int(
                token_edit_metadata.get("frequency_nonzero_token_types", 0)
            ),
        },
    )


def _build_schedule_and_budget_context(inputs: FinalReportInputs) -> ReportFragment:
    target_param_count = _selected_target_param_count(inputs.config, inputs.counts)
    return ReportFragment(
        "schedule_and_budget",
        merge_field_groups(
            FieldGroup("learning-rate schedule", _lr_schedule_metric_fields(inputs.lr_schedule)),
            FieldGroup(
                "budget",
                {
                    "target_param_count_basis": inputs.config.train.target_param_count_basis,
                    "target_param_count": target_param_count,
                    "tokens_per_target_param": inputs.train_tokens / max(1, target_param_count),
                    "parameter_accounting": inputs.parameter_accounting,
                    "num_steps": inputs.total_steps,
                },
            ),
        ),
    )


def _build_data_and_tokenizer_context(inputs: FinalReportInputs) -> ReportFragment:
    return ReportFragment(
        "data_and_tokenizer",
        {
            "vocab_size": inputs.model_config.vocab_size,
            "mask_token_id": inputs.model_config.mask_token_id,
            "bos_token_id": inputs.tokenizer.get_bos_token_id(),
            "eos_token_id": inputs.tokenizer.get_eos_token_id(),
            "pad_token_id": inputs.tokenizer.get_pad_token_id(),
            "special_token_count": len(inputs.special_token_ids),
            "generation_blocked_special_token_count": len(inputs.tokenizer.get_generation_blocked_token_ids()),
            "data_prepend_bos": bool(inputs.config.data.prepend_bos),
            "data_append_eos": bool(inputs.config.data.append_eos),
            "data_loss_on_bos": bool(inputs.config.data.loss_on_bos),
            "data_loss_on_eos": bool(inputs.config.data.loss_on_eos),
        },
    )


def _parameter_count_fields(
    inputs: FinalReportInputs,
    ledger: tuple[tuple[str, str], ...],
    *,
    divisor: float = 1.0,
) -> dict[str, int | float]:
    fields: dict[str, int | float] = {}
    for report_key, count_key in ledger:
        if report_key in fields:
            raise ValueError(f"parameter-count ledger contains duplicate report field {report_key!r}")
        value = inputs.counts[count_key]
        fields[report_key] = value if divisor == 1.0 else value / divisor
    return fields


def _build_parameter_inventory(inputs: FinalReportInputs) -> ReportFragment:
    return ReportFragment(
        "parameter_inventory",
        merge_field_groups(
            FieldGroup("raw parameter counts", _parameter_count_fields(inputs, _RAW_PARAMETER_COUNT_FIELDS)),
            FieldGroup(
                "parameter counts in millions",
                _parameter_count_fields(inputs, _MILLION_PARAMETER_COUNT_FIELDS, divisor=1_000_000.0),
            ),
        ),
    )


def _build_compute_and_token_totals(inputs: FinalReportInputs) -> ReportFragment:
    compute = training_compute_estimate(
        training_flops_per_token=int(inputs.flops_per_token),
        training_tokens=int(inputs.train_tokens),
        self_generated_draft_forwards=int(
            inputs.progress.lineage.train_self_generated_edit_accumulator["draft_forwards"]
        ),
        device_batch_size=_self_generated_draft_device_batch_size(inputs.config),
        max_seq_len=int(inputs.config.model.max_seq_len),
        base_accounting_method=(
            BD3LM_DENSE_CONCAT_FLOPS_ACCOUNTING_METHOD
            if _uses_bd3lm(inputs.config)
            else "main_plus_mtp_dense_training_estimate"
            if mtp_training_enabled(inputs.config)
            else "base_training_estimate"
        ),
        token_edit_backbone_gradient_mode=inputs.config.train.token_edit_backbone_gradient_mode,
        differentiable_draft=_uses_differentiable_self_generated_draft(inputs.config),
    )
    return ReportFragment(
        "compute_and_token_totals",
        merge_field_groups(
            FieldGroup("compute budget", compute_budget_fields(inputs, compute)),
            FieldGroup("token-to-parameter ratios", _token_ratio_metrics(inputs.train_tokens, inputs.counts)),
            FieldGroup(
                "aggregate token counts",
                {
                    "total_tokens_M": inputs.train_tokens / 1_000_000.0,
                    "total_valid_tokens_M": inputs.train_valid_tokens / 1_000_000.0,
                    "total_content_tokens_M": inputs.train_content_tokens / 1_000_000.0,
                    "total_content_bytes_M": inputs.train_content_bytes / 1_000_000.0,
                    "session_tokens_M": inputs.session_train_tokens / 1_000_000.0,
                    "session_valid_tokens_M": inputs.session_train_valid_tokens / 1_000_000.0,
                    "session_content_tokens_M": inputs.session_train_content_tokens / 1_000_000.0,
                    "session_content_bytes_M": inputs.session_train_content_bytes / 1_000_000.0,
                },
            ),
        ),
    )


def _build_process_and_resume_context(inputs: FinalReportInputs) -> ReportFragment:
    return ReportFragment(
        "process_and_resume",
        {
            "avg_tok_sec": int(inputs.session_train_tokens / max(1e-08, inputs.training_seconds)),
            "avg_valid_tok_sec": int(inputs.session_train_valid_tokens / max(1e-08, inputs.training_seconds)),
            "avg_content_tok_sec": int(inputs.session_train_content_tokens / max(1e-08, inputs.training_seconds)),
            "avg_content_byte_sec": int(inputs.session_train_content_bytes / max(1e-08, inputs.training_seconds)),
            "train_content_tokens": int(inputs.train_content_tokens),
            "train_content_bytes": int(inputs.train_content_bytes),
            "session_content_tokens": int(inputs.session_train_content_tokens),
            "session_content_bytes": int(inputs.session_train_content_bytes),
            "train_content_token_byte_fertility": inputs.train_content_tokens / inputs.train_content_bytes
            if inputs.train_content_bytes
            else 0.0,
            "session_content_token_byte_fertility": inputs.session_train_content_tokens
            / inputs.session_train_content_bytes
            if inputs.session_train_content_bytes
            else 0.0,
            "training_seconds": inputs.training_seconds,
            "world_size": inputs.world_size,
            "effective_batch_size_tokens": inputs.total_batch_size,
            "grad_accum_steps": inputs.grad_accum_steps,
            "start_step": inputs.start_step,
            "resume_contract_version": int(inputs.resume_state["version"]) if inputs.resume_state is not None else None,
            "resume_contract_status": str(inputs.resume_state["status"])
            if inputs.resume_state is not None
            else "not_resumed",
            "resume_consumed_microbatches": int(inputs.resume_state["consumed_microbatches"])
            if inputs.resume_state is not None
            else 0,
            "resume_next_batch_digest_verified": bool(inputs.resume_next_batch_digest_verified),
            **inputs.source_checkpoint_load_receipt,
            "checkpoint_resume_contract_version": RESUME_CONTRACT_VERSION,
            "checkpoint_consumed_microbatches": inputs.total_steps * inputs.grad_accum_steps,
            "training_counter_reduction": "packed_int64_all_reduce",
        },
    )


def build_final_context_fragments_v1(inputs: FinalReportInputs) -> tuple[ReportFragment, ...]:
    """Build the ordered, single-owner run-context portion of final metrics v1."""

    return (
        _build_execution_context(inputs),
        _build_recipe_context(inputs),
        _build_eos_supervision(inputs),
        _build_model_resolution(inputs),
        *_build_feature_configuration_fragments(inputs),
        _build_initialization_diagnostics(inputs),
        _build_objective_definition(inputs),
        _build_data_identity(inputs),
        _build_objective_options(inputs),
        _build_schedule_and_budget_context(inputs),
        _build_data_and_tokenizer_context(inputs),
        _build_parameter_inventory(inputs),
        _build_compute_and_token_totals(inputs),
        _build_process_and_resume_context(inputs),
    )
