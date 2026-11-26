"""Strict MDLM diagnostic options and analytic Bayes audit controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_mdlm_audit_options(config: ExperimentConfig) -> None:
    if not isinstance(config.train.strict_mdlm_corruption_audit_only, bool):
        raise ValueError("train.strict_mdlm_corruption_audit_only must be a boolean")
    if not isinstance(config.train.strict_mdlm_target_ce_audit_only, bool):
        raise ValueError("train.strict_mdlm_target_ce_audit_only must be a boolean")
    if not isinstance(config.train.strict_mdlm_right_context_canary_only, bool):
        raise ValueError("train.strict_mdlm_right_context_canary_only must be a boolean")
    if not isinstance(config.train.strict_mdlm_eos_overfit_canary_only, bool):
        raise ValueError("train.strict_mdlm_eos_overfit_canary_only must be a boolean")
    if not isinstance(config.train.strict_mdlm_sampler_eos_reachability_canary_only, bool):
        raise ValueError("train.strict_mdlm_sampler_eos_reachability_canary_only must be a boolean")
    if not isinstance(config.train.strict_mdlm_learned_eos_sampler_canary_only, bool):
        raise ValueError("train.strict_mdlm_learned_eos_sampler_canary_only must be a boolean")
    if not isinstance(config.train.strict_mdlm_real_collator_micro_overfit_canary_only, bool):
        raise ValueError("train.strict_mdlm_real_collator_micro_overfit_canary_only must be a boolean")
    if not isinstance(config.train.strict_mdlm_real_collator_component_probe, bool):
        raise ValueError("train.strict_mdlm_real_collator_component_probe must be a boolean")
    if (
        config.train.strict_mdlm_real_collator_component_probe
        and not config.train.strict_mdlm_real_collator_micro_overfit_canary_only
    ):
        raise ValueError(
            "train.strict_mdlm_real_collator_component_probe requires "
            "strict_mdlm_real_collator_micro_overfit_canary_only=true"
        )
    if not isinstance(config.train.strict_mdlm_all_token_bayes_audit_only, bool):
        raise ValueError("train.strict_mdlm_all_token_bayes_audit_only must be a boolean")
    if not isinstance(config.train.strict_mdlm_all_token_diagnostic_telemetry, bool):
        raise ValueError("train.strict_mdlm_all_token_diagnostic_telemetry must be a boolean")
    if not isinstance(config.train.strict_mdlm_all_token_diagnostic_t_values, list):
        raise ValueError("train.strict_mdlm_all_token_diagnostic_t_values must be a list")
    if config.train.strict_mdlm_all_token_diagnostic_t_values:
        if not config.train.strict_mdlm_all_token_diagnostic_telemetry:
            raise ValueError(
                "train.strict_mdlm_all_token_diagnostic_t_values requires "
                "strict_mdlm_all_token_diagnostic_telemetry=true"
            )
        seen_diagnostic_t_values: set[float] = set()
        for value in config.train.strict_mdlm_all_token_diagnostic_t_values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("train.strict_mdlm_all_token_diagnostic_t_values entries must be numeric")
            value_f = float(value)
            if value_f <= 0.0 or value_f > 1.0:
                raise ValueError("train.strict_mdlm_all_token_diagnostic_t_values entries must be in (0, 1]")
            if value_f in seen_diagnostic_t_values:
                raise ValueError("train.strict_mdlm_all_token_diagnostic_t_values entries must be unique")
            seen_diagnostic_t_values.add(value_f)
    if not isinstance(config.train.strict_mdlm_eos_logit_bias_calibration, bool):
        raise ValueError("train.strict_mdlm_eos_logit_bias_calibration must be a boolean")
    if not isinstance(config.train.strict_mdlm_eos_logit_bias_candidates, list):
        raise ValueError("train.strict_mdlm_eos_logit_bias_candidates must be a list")
    if not isinstance(config.train.strict_mdlm_uniform_schedule_weight, bool):
        raise ValueError("train.strict_mdlm_uniform_schedule_weight must be a boolean")
    if config.train.strict_mdlm_uniform_schedule_weight and not config.train.strict_mdlm_eos_overfit_canary_only:
        raise ValueError("train.strict_mdlm_uniform_schedule_weight requires strict_mdlm_eos_overfit_canary_only=true")
    if config.train.strict_mdlm_eos_logit_bias_calibration:
        if not config.train.strict_mdlm_eos_overfit_canary_only:
            raise ValueError("EOS-logit-bias calibration requires strict_mdlm_eos_overfit_canary_only=true")
        if not config.train.strict_mdlm_all_token_diagnostic_telemetry:
            raise ValueError("EOS-logit-bias calibration requires strict_mdlm_all_token_diagnostic_telemetry=true")
        if config.train.strict_mdlm_teacher_forced_replay_diagnostic:
            raise ValueError("EOS-logit-bias calibration cannot be coupled with teacher-forced replay diagnostic")
        if len(config.train.strict_mdlm_eos_logit_bias_candidates) <= 0:
            raise ValueError(
                "train.strict_mdlm_eos_logit_bias_candidates must be non-empty when EOS logit-bias calibration is enabled"
            )
        if not config.train.strict_mdlm_all_token_diagnostic_t_values and (
            abs(float(config.noise.t_min) - 0.95) > 1e-8 or abs(float(config.noise.t_max) - 0.95) > 1e-8
        ):
            raise ValueError(
                "EOS-logit-bias calibration requires fixed high-t setup with noise.t_min=t_max=0.95 when no explicit "
                "strict_mdlm_all_token_diagnostic_t_values are configured"
            )
        if config.train.strict_mdlm_all_token_diagnostic_t_values and not any(
            abs(float(value) - 0.95) <= 1e-8 for value in config.train.strict_mdlm_all_token_diagnostic_t_values
        ):
            raise ValueError(
                "EOS-logit-bias calibration requires strict_mdlm_all_token_diagnostic_t_values to include 0.95"
            )
        seen_biases: set[float] = set()
        for value in config.train.strict_mdlm_eos_logit_bias_candidates:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("train.strict_mdlm_eos_logit_bias_candidates entries must be numeric")
            candidate = float(value)
            if candidate != candidate or candidate in (float("inf"), float("-inf")):
                raise ValueError("train.strict_mdlm_eos_logit_bias_candidates entries must be finite")
            if candidate in seen_biases:
                raise ValueError("train.strict_mdlm_eos_logit_bias_candidates entries must be unique")
            seen_biases.add(candidate)
    elif config.train.strict_mdlm_eos_logit_bias_candidates:
        raise ValueError(
            "train.strict_mdlm_eos_logit_bias_candidates requires strict_mdlm_eos_logit_bias_calibration=true"
        )
    if not isinstance(config.train.strict_mdlm_teacher_forced_replay_diagnostic, bool):
        raise ValueError("train.strict_mdlm_teacher_forced_replay_diagnostic must be a boolean")
    if not isinstance(config.train.strict_mdlm_t_curriculum_enabled, bool):
        raise ValueError("train.strict_mdlm_t_curriculum_enabled must be a boolean")
    if not isinstance(config.train.strict_mdlm_t_curriculum_warmup_steps, int):
        raise ValueError("train.strict_mdlm_t_curriculum_warmup_steps must be an integer")
    for name, value in (
        ("strict_mdlm_t_curriculum_warmup_t", config.train.strict_mdlm_t_curriculum_warmup_t),
        ("strict_mdlm_t_curriculum_final_t", config.train.strict_mdlm_t_curriculum_final_t),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"train.{name} must be numeric")
        if config.train.strict_mdlm_t_curriculum_enabled and (float(value) <= 0.0 or float(value) > 1.0):
            raise ValueError(f"train.{name} must be in (0, 1] when t curriculum is enabled")
    if config.train.strict_mdlm_t_curriculum_enabled:
        if config.train.strict_mdlm_t_curriculum_warmup_steps <= 0:
            raise ValueError(
                "train.strict_mdlm_t_curriculum_warmup_steps must be positive when t curriculum is enabled"
            )
        if config.train.num_steps > 0 and config.train.strict_mdlm_t_curriculum_warmup_steps >= config.train.num_steps:
            raise ValueError("train.strict_mdlm_t_curriculum_warmup_steps must be less than train.num_steps")
        if not config.train.strict_mdlm_eos_overfit_canary_only:
            raise ValueError("strict MDLM t curriculum requires strict_mdlm_eos_overfit_canary_only=true")
        if config.data.synthetic_eos_overfit_loss_mask_policy != "all_lossable_tokens":
            raise ValueError("strict MDLM t curriculum requires all_lossable_tokens loss mask policy")
        if not config.train.strict_mdlm_all_token_diagnostic_telemetry:
            raise ValueError("strict MDLM t curriculum requires all-token diagnostic telemetry")
        if config.train.strict_mdlm_eos_loss_mass_balance:
            raise ValueError("strict MDLM t curriculum cannot be coupled with EOS loss-mass balancing")
        expected_t_values = [
            float(config.train.strict_mdlm_t_curriculum_warmup_t),
            float(config.train.strict_mdlm_t_curriculum_final_t),
        ]
        actual_t_values = [float(value) for value in config.train.strict_mdlm_all_token_diagnostic_t_values]
        if len(actual_t_values) != 2 or any(
            abs(actual - expected) > 1e-8 for actual, expected in zip(actual_t_values, expected_t_values, strict=True)
        ):
            raise ValueError(
                "strict MDLM t curriculum requires "
                "strict_mdlm_all_token_diagnostic_t_values to match warmup/final t values"
            )
    if not isinstance(config.train.real_packing_audit_only, bool):
        raise ValueError("train.real_packing_audit_only must be a boolean")
    audit_flags = [
        bool(config.train.strict_mdlm_corruption_audit_only),
        bool(config.train.strict_mdlm_target_ce_audit_only),
        bool(config.train.strict_mdlm_right_context_canary_only),
        bool(config.train.strict_mdlm_eos_overfit_canary_only),
        bool(config.train.strict_mdlm_sampler_eos_reachability_canary_only),
        bool(config.train.strict_mdlm_learned_eos_sampler_canary_only),
        bool(config.train.strict_mdlm_real_collator_micro_overfit_canary_only),
        bool(config.train.strict_mdlm_all_token_bayes_audit_only),
        bool(config.train.real_packing_audit_only),
    ]
    if sum(1 for enabled in audit_flags if enabled) > 1:
        raise ValueError("audit-only modes must run one mechanism at a time")


def validate_all_token_bayes_audit(config: ExperimentConfig) -> None:
    if config.train.strict_mdlm_all_token_bayes_audit_only:
        if config.train.eval_only:
            raise ValueError("all-token Bayes audit is a train-mode smoke and requires train.eval_only=false")
        if config.data.synthetic_dataset != "eos_overfit_canary":
            raise ValueError("all-token Bayes audit requires data.synthetic_dataset='eos_overfit_canary'")
        if config.data.synthetic_eos_overfit_loss_mask_policy != "all_lossable_tokens":
            raise ValueError(
                "all-token Bayes audit requires data.synthetic_eos_overfit_loss_mask_policy='all_lossable_tokens'"
            )
        if config.data.train_data:
            raise ValueError("all-token Bayes audit uses only the synthetic in-memory dataset")
        if config.data.val_data:
            raise ValueError("all-token Bayes audit uses the synthetic in-memory validation batch")
        if config.data.tokenizer != "byte":
            raise ValueError("all-token Bayes audit requires data.tokenizer='byte'")
        if not config.data.append_eos:
            raise ValueError("all-token Bayes audit requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("all-token Bayes audit requires data.loss_on_eos=true")
        if config.data.random_length_ratio != 0.0:
            raise ValueError("all-token Bayes audit requires data.random_length_ratio=0")
        if config.data.continuation_ratio != 0.0:
            raise ValueError("all-token Bayes audit requires data.continuation_ratio=0")
        if config.data.boundary_aware_packing:
            raise ValueError("all-token Bayes audit requires data.boundary_aware_packing=false")
        if config.data.content_budget_segmented_packing:
            raise ValueError("all-token Bayes audit requires data.content_budget_segmented_packing=false")
        if config.data.short_document_curriculum:
            raise ValueError("all-token Bayes audit requires data.short_document_curriculum=false")
        if config.model.causal:
            raise ValueError("all-token Bayes audit requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("all-token Bayes audit requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("all-token Bayes audit cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("all-token Bayes audit cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("all-token Bayes audit cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("all-token Bayes audit cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("all-token Bayes audit cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("all-token Bayes audit requires noise.type='mdlm'")
        if not config.noise.schedule_weight:
            raise ValueError("all-token Bayes audit requires noise.schedule_weight=true")
        if config.noise.complement_masking:
            raise ValueError("all-token Bayes audit requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("all-token Bayes audit requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("all-token Bayes audit preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("all-token Bayes audit requires noise.per_token_noise_levels=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("all-token Bayes audit requires auxiliary_loss_weight=0")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("all-token Bayes audit requires entropy_bonus_beta=0")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("all-token Bayes audit requires confidence_penalty_weight=0")
        if config.train.z_loss_weight != 0:
            raise ValueError("all-token Bayes audit requires z_loss_weight=0")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("all-token Bayes audit requires eos_hazard_loss_weight=0")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("all-token Bayes audit requires length_head_loss_weight=0")
        if config.train.position_decile_balanced_loss:
            raise ValueError("all-token Bayes audit requires position_decile_balanced_loss=false")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("all-token Bayes audit requires uniform_loss_estimator='single_sample'")
        if config.train.token_edit_corruption_batch_fraction != 0:
            raise ValueError("all-token Bayes audit requires token_edit_corruption_batch_fraction=0")
        if config.train.token_edit_corruption_ratio != 0:
            raise ValueError("all-token Bayes audit requires token_edit_corruption_ratio=0")
        if config.train.strict_mdlm_eos_loss_mass_balance:
            raise ValueError("all-token Bayes audit cannot be coupled with EOS loss-mass balancing")
        if not config.train.strict_mdlm_all_token_diagnostic_telemetry:
            raise ValueError("all-token Bayes audit requires strict_mdlm_all_token_diagnostic_telemetry=true")
        expected_t_values = [0.55, 0.95]
        if len(config.train.strict_mdlm_all_token_diagnostic_t_values) != 2 or any(
            abs(float(actual) - expected) > 1e-8
            for actual, expected in zip(
                config.train.strict_mdlm_all_token_diagnostic_t_values,
                expected_t_values,
                strict=False,
            )
        ):
            raise ValueError("all-token Bayes audit requires strict_mdlm_all_token_diagnostic_t_values=[0.55, 0.95]")
