"""Pure validation rules for experiment configurations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from .config_validation_data import validate_data_config
from .config_validation_diagnostics import _validate_diagnostics_config
from .config_validation_evaluation import validate_runtime_evaluation_config
from .config_validation_model import validate_model_config
from .config_validation_noise import validate_noise_config
from .config_validation_optim import validate_model_optimizer_profile, validate_optimizer
from .config_validation_parameterization import validate_parameterization_config
from .config_validation_recipe import validate_recipe
from .data_mask_tail_config import validate_mask_tail
from .lr_schedule import validate_lr_schedule
from .mdlm_auxiliary_config import validate_auxiliary
from .telemetry_config import validate_telemetry_config
from .training_mtp import validate_mtp_config
from .training_variable_depth import validate_config as validate_depth

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_config(config: ExperimentConfig) -> None:
    validate_lr_schedule(config)
    validate_auxiliary(config)
    validate_mask_tail(config)
    validate_mtp_config(config)
    _validate_diagnostics_config(config)
    validate_model_optimizer_profile(config)
    validate_data_config(config)
    validate_model_config(config)
    validate_depth(config)
    validate_noise_config(config)
    validate_optimizer(config)
    validate_parameterization_config(config)
    validate_recipe(config)
    if config.train.device_batch_size <= 0:
        raise ValueError("train.device_batch_size must be positive")
    if config.train.target_tokens != -1 and config.train.target_tokens <= 0:
        raise ValueError("train.target_tokens must be positive or -1")
    if config.train.target_param_count_basis not in (
        "transformer_body_parameters",
        "nanochat_scaling_parameters",
        "non_input_embedding_parameters",
        "total_parameters",
    ):
        raise ValueError(
            "train.target_param_count_basis must name one of the supported instantiated-model accounting views"
        )
    if config.train.base_total_batch_size <= 0:
        raise ValueError("train.base_total_batch_size must be positive")
    validate_telemetry_config(config.train)
    if config.train.eval_only and not config.train.resume_from and not config.train.comparison_run_ids:
        raise ValueError("train.eval_only requires train.resume_from or train.comparison_run_ids")
    model_init_from = config.train.model_init_from.strip()
    if not isinstance(config.train.model_init_allow_token_edit_extensions, bool):
        raise ValueError("train.model_init_allow_token_edit_extensions must be a boolean")
    if model_init_from:
        if not Path(model_init_from).expanduser().is_absolute():
            raise ValueError("train.model_init_from must be an absolute checkpoint path")
        expected_sha256 = config.train.model_init_expected_sha256
        if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
            raise ValueError(
                "train.model_init_expected_sha256 must be a lowercase 64-hex digest when model_init_from is set"
            )
        if (
            not isinstance(config.train.model_init_expected_size_bytes, int)
            or isinstance(config.train.model_init_expected_size_bytes, bool)
            or config.train.model_init_expected_size_bytes <= 0
        ):
            raise ValueError(
                "train.model_init_expected_size_bytes must be a positive integer when model_init_from is set"
            )
        if config.train.eval_only:
            raise ValueError("train.model_init_from is a training-only model-state initialization interface")
        if config.train.comparison_run_ids:
            raise ValueError("train.model_init_from cannot be combined with train.comparison_run_ids")
        if (
            config.train.resume_from
            and Path(config.train.resume_from).expanduser() == Path(model_init_from).expanduser()
        ):
            raise ValueError("train.model_init_from and train.resume_from must not name the same checkpoint")
    elif config.train.model_init_expected_sha256 or config.train.model_init_expected_size_bytes != -1:
        raise ValueError("model-init expected identity fields require train.model_init_from")
    if config.train.model_init_allow_token_edit_extensions:
        if not model_init_from:
            raise ValueError("extended token-edit model initialization requires train.model_init_from")
        if (
            config.model.token_edit_adapter_bottleneck_dim <= 0
            and config.model.token_edit_adapter_top_layers <= 0
            and not config.model.token_edit_provenance_embedding
        ):
            raise ValueError("extended token-edit model initialization requires an enabled token-edit model extension")
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
    if config.train.strict_mdlm_learned_eos_sampler_canary_only:
        if config.train.eval_only:
            raise ValueError("learned EOS sampler canary requires train.eval_only=false")
        if config.data.synthetic_dataset != "eos_overfit_canary":
            raise ValueError("learned EOS sampler canary requires data.synthetic_dataset='eos_overfit_canary'")
        if config.data.synthetic_eos_overfit_loss_mask_policy != "eos_only":
            raise ValueError(
                "learned EOS sampler canary requires data.synthetic_eos_overfit_loss_mask_policy='eos_only'"
            )
        if config.data.train_data:
            raise ValueError("learned EOS sampler canary uses only the synthetic in-memory dataset")
        if config.data.val_data:
            raise ValueError("learned EOS sampler canary uses the synthetic in-memory validation batch")
        if config.data.tokenizer != "byte":
            raise ValueError("learned EOS sampler canary requires data.tokenizer='byte'")
        if not config.data.append_eos:
            raise ValueError("learned EOS sampler canary requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("learned EOS sampler canary requires data.loss_on_eos=true")
        if config.data.random_length_ratio != 0.0:
            raise ValueError("learned EOS sampler canary requires data.random_length_ratio=0")
        if config.data.continuation_ratio != 0.0:
            raise ValueError("learned EOS sampler canary requires data.continuation_ratio=0")
        if config.data.boundary_aware_packing:
            raise ValueError("learned EOS sampler canary requires data.boundary_aware_packing=false")
        if config.data.content_budget_segmented_packing:
            raise ValueError("learned EOS sampler canary requires data.content_budget_segmented_packing=false")
        if config.data.short_document_curriculum:
            raise ValueError("learned EOS sampler canary requires data.short_document_curriculum=false")
        if config.model.causal:
            raise ValueError("learned EOS sampler canary requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("learned EOS sampler canary requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("learned EOS sampler canary cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("learned EOS sampler canary cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("learned EOS sampler canary cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("learned EOS sampler canary cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("learned EOS sampler canary cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("learned EOS sampler canary requires noise.type='mdlm'")
        if not config.noise.schedule_weight:
            raise ValueError("learned EOS sampler canary requires noise.schedule_weight=true")
        if config.noise.complement_masking:
            raise ValueError("learned EOS sampler canary requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("learned EOS sampler canary requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("learned EOS sampler canary preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("learned EOS sampler canary requires noise.per_token_noise_levels=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("learned EOS sampler canary requires auxiliary_loss_weight=0")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("learned EOS sampler canary requires entropy_bonus_beta=0")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("learned EOS sampler canary requires confidence_penalty_weight=0")
        if config.train.z_loss_weight != 0:
            raise ValueError("learned EOS sampler canary requires z_loss_weight=0")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("learned EOS sampler canary requires eos_hazard_loss_weight=0")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("learned EOS sampler canary requires length_head_loss_weight=0")
        if config.train.position_decile_balanced_loss:
            raise ValueError("learned EOS sampler canary requires position_decile_balanced_loss=false")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("learned EOS sampler canary requires uniform_loss_estimator='single_sample'")
        if config.train.token_edit_corruption_batch_fraction != 0:
            raise ValueError("learned EOS sampler canary requires token_edit_corruption_batch_fraction=0")
        if config.train.token_edit_corruption_ratio != 0:
            raise ValueError("learned EOS sampler canary requires token_edit_corruption_ratio=0")
        if config.eval.sampler != "fhs":
            raise ValueError("learned EOS sampler canary requires eval.sampler='fhs'")
        if config.eval.sample_token_selection != "argmax":
            raise ValueError("learned EOS sampler canary requires eval.sample_token_selection='argmax'")
        if not config.eval.sample_trace:
            raise ValueError("learned EOS sampler canary requires eval.sample_trace=true")
        if config.eval.length_policy != "none":
            raise ValueError("learned EOS sampler canary requires eval.length_policy='none'")
        if config.eval.fhs_tau >= 1.0:
            raise ValueError("learned EOS sampler canary requires eval.fhs_tau < 1")
        if config.eval.gen_len < 4:
            raise ValueError("learned EOS sampler canary requires eval.gen_len >= 4")
    if config.train.strict_mdlm_sampler_eos_reachability_canary_only:
        if config.train.eval_only:
            raise ValueError("sampler EOS reachability canary requires train.eval_only=false")
        if config.data.tokenizer != "byte":
            raise ValueError("sampler EOS reachability canary requires data.tokenizer='byte'")
        if config.data.train_data:
            raise ValueError("sampler EOS reachability canary uses controlled logits and requires data.train_data=[]")
        if config.data.val_data:
            raise ValueError("sampler EOS reachability canary uses controlled logits and requires data.val_data=[]")
        if config.data.synthetic_dataset:
            raise ValueError("sampler EOS reachability canary uses controlled logits and requires no synthetic dataset")
        if not config.data.append_eos:
            raise ValueError("sampler EOS reachability canary requires data.append_eos=true")
        if config.data.boundary_aware_packing:
            raise ValueError("sampler EOS reachability canary requires data.boundary_aware_packing=false")
        if config.data.content_budget_segmented_packing:
            raise ValueError("sampler EOS reachability canary requires data.content_budget_segmented_packing=false")
        if config.data.short_document_curriculum:
            raise ValueError("sampler EOS reachability canary requires data.short_document_curriculum=false")
        if config.model.causal:
            raise ValueError("sampler EOS reachability canary requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("sampler EOS reachability canary requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("sampler EOS reachability canary cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("sampler EOS reachability canary cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("sampler EOS reachability canary cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("sampler EOS reachability canary cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("sampler EOS reachability canary cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("sampler EOS reachability canary requires noise.type='mdlm'")
        if not config.noise.schedule_weight:
            raise ValueError("sampler EOS reachability canary requires noise.schedule_weight=true")
        if config.noise.complement_masking:
            raise ValueError("sampler EOS reachability canary requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("sampler EOS reachability canary requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("sampler EOS reachability canary preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("sampler EOS reachability canary requires noise.per_token_noise_levels=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("sampler EOS reachability canary requires auxiliary_loss_weight=0")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("sampler EOS reachability canary requires entropy_bonus_beta=0")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("sampler EOS reachability canary requires confidence_penalty_weight=0")
        if config.train.z_loss_weight != 0:
            raise ValueError("sampler EOS reachability canary requires z_loss_weight=0")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("sampler EOS reachability canary requires eos_hazard_loss_weight=0")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("sampler EOS reachability canary requires length_head_loss_weight=0")
        if config.train.position_decile_balanced_loss:
            raise ValueError("sampler EOS reachability canary requires position_decile_balanced_loss=false")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("sampler EOS reachability canary requires uniform_loss_estimator='single_sample'")
        if config.train.token_edit_corruption_batch_fraction != 0:
            raise ValueError("sampler EOS reachability canary requires token_edit_corruption_batch_fraction=0")
        if config.train.token_edit_corruption_ratio != 0:
            raise ValueError("sampler EOS reachability canary requires token_edit_corruption_ratio=0")
        if config.eval.sampler != "fhs":
            raise ValueError("sampler EOS reachability canary requires eval.sampler='fhs'")
        if config.eval.sample_token_selection != "argmax":
            raise ValueError("sampler EOS reachability canary requires eval.sample_token_selection='argmax'")
        if not config.eval.sample_trace:
            raise ValueError("sampler EOS reachability canary requires eval.sample_trace=true")
        if config.eval.length_policy != "none":
            raise ValueError("sampler EOS reachability canary requires eval.length_policy='none'")
        if config.eval.fhs_tau >= 1.0:
            raise ValueError("sampler EOS reachability canary requires eval.fhs_tau < 1")
        if config.eval.gen_len < 4:
            raise ValueError("sampler EOS reachability canary requires eval.gen_len >= 4")
    if config.train.strict_mdlm_eos_overfit_canary_only:
        if config.train.eval_only:
            raise ValueError("EOS overfit canary requires train.eval_only=false")
        if config.data.synthetic_dataset != "eos_overfit_canary":
            raise ValueError("EOS overfit canary requires data.synthetic_dataset='eos_overfit_canary'")
        if config.data.train_data:
            raise ValueError("EOS overfit canary uses only the synthetic in-memory dataset")
        if config.data.val_data:
            raise ValueError("EOS overfit canary uses the synthetic in-memory validation batch")
        if config.data.tokenizer != "byte":
            raise ValueError("EOS overfit canary requires data.tokenizer='byte'")
        if not config.data.append_eos:
            raise ValueError("EOS overfit canary requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("EOS overfit canary requires data.loss_on_eos=true")
        if config.data.random_length_ratio != 0.0:
            raise ValueError("EOS overfit canary requires data.random_length_ratio=0")
        if config.data.continuation_ratio != 0.0:
            raise ValueError("EOS overfit canary requires data.continuation_ratio=0")
        if config.data.boundary_aware_packing:
            raise ValueError("EOS overfit canary requires data.boundary_aware_packing=false")
        if config.data.content_budget_segmented_packing:
            raise ValueError("EOS overfit canary requires data.content_budget_segmented_packing=false")
        if config.data.short_document_curriculum:
            raise ValueError("EOS overfit canary requires data.short_document_curriculum=false")
        if config.model.causal:
            raise ValueError("EOS overfit canary requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("EOS overfit canary requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("EOS overfit canary cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("EOS overfit canary cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("EOS overfit canary cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("EOS overfit canary cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("EOS overfit canary cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("EOS overfit canary requires noise.type='mdlm'")
        if not config.noise.schedule_weight and not config.train.strict_mdlm_uniform_schedule_weight:
            raise ValueError("EOS overfit canary requires noise.schedule_weight=true")
        if config.train.strict_mdlm_uniform_schedule_weight and config.noise.schedule_weight:
            raise ValueError("strict_mdlm_uniform_schedule_weight requires noise.schedule_weight=false")
        if config.noise.complement_masking:
            raise ValueError("EOS overfit canary requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("EOS overfit canary requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("EOS overfit canary preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("EOS overfit canary requires noise.per_token_noise_levels=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("EOS overfit canary requires auxiliary_loss_weight=0")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("EOS overfit canary requires entropy_bonus_beta=0")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("EOS overfit canary requires confidence_penalty_weight=0")
        if config.train.z_loss_weight != 0:
            raise ValueError("EOS overfit canary requires z_loss_weight=0")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("EOS overfit canary requires eos_hazard_loss_weight=0")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("EOS overfit canary requires length_head_loss_weight=0")
        if config.train.position_decile_balanced_loss:
            raise ValueError("EOS overfit canary requires position_decile_balanced_loss=false")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("EOS overfit canary requires uniform_loss_estimator='single_sample'")
        if config.train.token_edit_corruption_batch_fraction != 0:
            raise ValueError("EOS overfit canary requires token_edit_corruption_batch_fraction=0")
        if config.train.token_edit_corruption_ratio != 0:
            raise ValueError("EOS overfit canary requires token_edit_corruption_ratio=0")
        if (
            config.train.strict_mdlm_all_token_diagnostic_telemetry
            and config.data.synthetic_eos_overfit_loss_mask_policy != "all_lossable_tokens"
        ):
            raise ValueError(
                "all-token diagnostic telemetry requires "
                "data.synthetic_eos_overfit_loss_mask_policy='all_lossable_tokens'"
            )
        if config.train.strict_mdlm_teacher_forced_replay_diagnostic:
            if config.data.synthetic_eos_overfit_loss_mask_policy != "all_lossable_tokens":
                raise ValueError(
                    "teacher-forced replay diagnostic requires "
                    "data.synthetic_eos_overfit_loss_mask_policy='all_lossable_tokens'"
                )
            if not config.train.strict_mdlm_all_token_diagnostic_telemetry:
                raise ValueError(
                    "teacher-forced replay diagnostic requires strict_mdlm_all_token_diagnostic_telemetry=true"
                )
            if abs(float(config.noise.t_min) - 0.95) > 1e-8 or abs(float(config.noise.t_max) - 0.95) > 1e-8:
                raise ValueError("teacher-forced replay diagnostic requires fixed train noise t_min=t_max=0.95")
            replay_t_values = [float(value) for value in config.train.strict_mdlm_all_token_diagnostic_t_values]
            if replay_t_values and (len(replay_t_values) != 1 or abs(replay_t_values[0] - 0.95) > 1e-8):
                raise ValueError(
                    "teacher-forced replay diagnostic requires strict_mdlm_all_token_diagnostic_t_values=[0.95]"
                )
            if config.train.strict_mdlm_t_curriculum_enabled:
                raise ValueError("teacher-forced replay diagnostic cannot be coupled with strict MDLM t curriculum")
            if config.train.strict_mdlm_eos_loss_mass_balance:
                raise ValueError("teacher-forced replay diagnostic cannot be coupled with EOS loss-mass balancing")
            if config.train.comparison_run_ids != ["r-0014", "r-0015", "r-0017", "r-0020"]:
                raise ValueError(
                    "teacher-forced replay diagnostic requires "
                    "comparison_run_ids=['r-0014', 'r-0015', 'r-0017', 'r-0020']"
                )
            if config.eval.sampler != "default":
                raise ValueError("teacher-forced replay diagnostic requires eval.sampler='default'")
            if config.eval.sample_token_selection != "argmax":
                raise ValueError("teacher-forced replay diagnostic requires eval.sample_token_selection='argmax'")
            if config.eval.sample_trace:
                raise ValueError("teacher-forced replay diagnostic requires eval.sample_trace=false")
            if config.eval.length_policy != "none":
                raise ValueError("teacher-forced replay diagnostic requires eval.length_policy='none'")
    elif (
        config.train.strict_mdlm_all_token_diagnostic_telemetry
        and not config.train.strict_mdlm_all_token_bayes_audit_only
    ):
        raise ValueError(
            "all-token diagnostic telemetry requires strict_mdlm_eos_overfit_canary_only=true "
            "or strict_mdlm_all_token_bayes_audit_only=true"
        )
    elif config.train.strict_mdlm_teacher_forced_replay_diagnostic:
        raise ValueError("teacher-forced replay diagnostic requires strict_mdlm_eos_overfit_canary_only=true")
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
    if config.train.strict_mdlm_real_collator_micro_overfit_canary_only:
        if config.train.eval_only:
            raise ValueError("real-collator micro-overfit canary requires train.eval_only=false")
        if config.data.synthetic_dataset:
            raise ValueError("real-collator micro-overfit canary requires real train_data, not a synthetic dataset")
        if not config.data.train_data:
            raise ValueError("real-collator micro-overfit canary requires data.train_data")
        if not config.data.append_eos:
            raise ValueError("real-collator micro-overfit canary requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("real-collator micro-overfit canary requires data.loss_on_eos=true")
        if config.data.boundary_aware_packing:
            raise ValueError(
                "real-collator micro-overfit canary supports the default stream packer or "
                "content-budget segmented packer, not boundary-aware packing"
            )
        if config.data.short_document_curriculum:
            raise ValueError(
                "real-collator micro-overfit canary supports the default stream packer or "
                "content-budget segmented packer, not the short-document curriculum"
            )
        if config.train.strict_mdlm_real_collator_component_probe:
            if not config.data.content_budget_segmented_packing:
                raise ValueError("real-collator component probe requires data.content_budget_segmented_packing=true")
            if config.train.device_batch_size != 1:
                raise ValueError("real-collator component probe requires train.device_batch_size=1")
        if config.model.causal:
            raise ValueError("real-collator micro-overfit canary requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("real-collator micro-overfit canary requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("real-collator micro-overfit canary cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("real-collator micro-overfit canary cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("real-collator micro-overfit canary cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("real-collator micro-overfit canary cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("real-collator micro-overfit canary cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("real-collator micro-overfit canary requires noise.type='mdlm'")
        if not config.noise.schedule_weight:
            raise ValueError("real-collator micro-overfit canary requires noise.schedule_weight=true")
        if config.noise.complement_masking:
            raise ValueError("real-collator micro-overfit canary requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("real-collator micro-overfit canary requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("real-collator micro-overfit canary preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("real-collator micro-overfit canary requires noise.per_token_noise_levels=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("real-collator micro-overfit canary requires auxiliary_loss_weight=0")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("real-collator micro-overfit canary requires entropy_bonus_beta=0")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("real-collator micro-overfit canary requires confidence_penalty_weight=0")
        if config.train.z_loss_weight != 0:
            raise ValueError("real-collator micro-overfit canary requires z_loss_weight=0")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("real-collator micro-overfit canary requires eos_hazard_loss_weight=0")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("real-collator micro-overfit canary requires length_head_loss_weight=0")
        if config.train.position_decile_balanced_loss:
            raise ValueError("real-collator micro-overfit canary requires position_decile_balanced_loss=false")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("real-collator micro-overfit canary requires uniform_loss_estimator='single_sample'")
        if config.train.token_edit_corruption_batch_fraction != 0:
            raise ValueError("real-collator micro-overfit canary requires token_edit_corruption_batch_fraction=0")
        if config.train.token_edit_corruption_ratio != 0:
            raise ValueError("real-collator micro-overfit canary requires token_edit_corruption_ratio=0")
    if config.train.real_packing_audit_only:
        if config.train.eval_only:
            raise ValueError("real packing audit is a train-loader smoke and requires train.eval_only=false")
        if config.model.causal:
            raise ValueError("real packing audit requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("real packing audit requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("real packing audit cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("real packing audit cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("real packing audit cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("real packing audit cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("real packing audit cannot be coupled with target-length conditioning")
        if config.data.boundary_aware_packing:
            raise ValueError(
                "real packing audit targets the default stream packer and requires boundary_aware_packing=false"
            )
        if config.data.content_budget_segmented_packing:
            raise ValueError(
                "real packing audit targets the default stream packer and requires content_budget_segmented_packing=false"
            )
        if config.data.short_document_curriculum:
            raise ValueError(
                "real packing audit targets the default stream packer and requires short_document_curriculum=false"
            )
        if not config.data.append_eos:
            raise ValueError("real packing audit requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("real packing audit requires data.loss_on_eos=true")
        if config.noise.type != "mdlm":
            raise ValueError("real packing audit requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("real packing audit requires complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("real packing audit requires suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("real packing audit preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("real packing audit requires per_token_noise_levels=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("real packing audit cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("real packing audit cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("real packing audit cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("real packing audit cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("real packing audit cannot be coupled with EOS hazard loss")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("real packing audit cannot be coupled with length-head loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("real packing audit cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("real packing audit cannot be coupled with uniform loss estimators")
        if config.train.token_edit_corruption_batch_fraction != 0 or config.train.token_edit_corruption_ratio != 0:
            raise ValueError("real packing audit cannot be coupled with token-edit corruption")
    if config.train.strict_mdlm_corruption_audit_only:
        if config.train.eval_only:
            raise ValueError("strict MDLM corruption audit is a train-mode smoke and requires train.eval_only=false")
        if config.model.causal:
            raise ValueError("strict MDLM corruption audit requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("strict MDLM corruption audit requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("strict MDLM corruption audit cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("strict MDLM corruption audit cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("strict MDLM corruption audit cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("strict MDLM corruption audit requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("strict MDLM corruption audit requires complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("strict MDLM corruption audit requires suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("strict MDLM corruption audit preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("strict MDLM corruption audit requires per_token_noise_levels=false")
        if not config.noise.schedule_weight:
            raise ValueError("strict MDLM corruption audit requires schedule_weight=true")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with EOS hazard loss")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with length-head loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("strict MDLM corruption audit cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("strict MDLM corruption audit cannot be coupled with uniform loss estimators")
        if config.train.token_edit_corruption_batch_fraction != 0 or config.train.token_edit_corruption_ratio != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with token-edit corruption")
    if config.train.strict_mdlm_target_ce_audit_only:
        if config.train.eval_only:
            raise ValueError("strict MDLM target/CE audit is a train-mode smoke and requires train.eval_only=false")
        if config.model.causal:
            raise ValueError("strict MDLM target/CE audit requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("strict MDLM target/CE audit requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with target-length conditioning")
        if not config.data.append_eos:
            raise ValueError("strict MDLM target/CE audit requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("strict MDLM target/CE audit requires data.loss_on_eos=true")
        if config.noise.type != "mdlm":
            raise ValueError("strict MDLM target/CE audit requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("strict MDLM target/CE audit requires complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("strict MDLM target/CE audit requires suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("strict MDLM target/CE audit preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("strict MDLM target/CE audit requires per_token_noise_levels=false")
        if not config.noise.schedule_weight:
            raise ValueError("strict MDLM target/CE audit requires schedule_weight=true")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with EOS hazard loss")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with length-head loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("strict MDLM target/CE audit cannot be coupled with uniform loss estimators")
        if config.train.token_edit_corruption_batch_fraction != 0 or config.train.token_edit_corruption_ratio != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with token-edit corruption")
    if config.train.strict_mdlm_right_context_canary_only:
        if config.train.eval_only:
            raise ValueError(
                "strict MDLM right-context canary is a train-mode smoke and requires train.eval_only=false"
            )
        if config.model.causal:
            raise ValueError("strict MDLM right-context canary requires model.causal=false")
        if config.model.local_attention_lower_layers != 0 or config.model.local_attention_window != 0:
            raise ValueError("strict MDLM right-context canary requires global attention in all layers")
        if config.model.output_factorization != "flat":
            raise ValueError("strict MDLM right-context canary requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("strict MDLM right-context canary cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("strict MDLM right-context canary cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("strict MDLM right-context canary cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("strict MDLM right-context canary requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("strict MDLM right-context canary requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("strict MDLM right-context canary requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("strict MDLM right-context canary preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("strict MDLM right-context canary requires per_token_noise_levels=false")
        if not config.noise.schedule_weight:
            raise ValueError("strict MDLM right-context canary requires schedule_weight=true")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with EOS hazard loss")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with length-head loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("strict MDLM right-context canary cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("strict MDLM right-context canary cannot be coupled with uniform loss estimators")
        if config.train.token_edit_corruption_batch_fraction != 0 or config.train.token_edit_corruption_ratio != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with token-edit corruption")
    if (
        config.train.eval_only
        and config.train.comparison_run_ids
        and len(config.train.comparison_run_ids) != 1
        and not config.eval.basin_diagnostic
    ):
        raise ValueError("train.comparison_run_ids must contain exactly one run id for eval-only replay")
    for run_id in config.train.comparison_run_ids:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("train.comparison_run_ids entries must be non-empty strings")
    if config.train.auxiliary_loss_weight < 0:
        raise ValueError("train.auxiliary_loss_weight must be non-negative")
    if config.train.auxiliary_loss_weight > 0 and not config.model.auxiliary_head_layers:
        raise ValueError("train.auxiliary_loss_weight requires model.auxiliary_head_layers")
    if config.train.entropy_bonus_beta < 0:
        raise ValueError("train.entropy_bonus_beta must be non-negative")
    if config.train.confidence_penalty_weight < 0:
        raise ValueError("train.confidence_penalty_weight must be non-negative")
    if config.train.confidence_penalty_weight > 0:
        if config.model.causal:
            raise ValueError("train.confidence_penalty_weight is only supported for non-causal MDLM training")
        if config.noise.type != "mdlm":
            raise ValueError("train.confidence_penalty_weight requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("train.confidence_penalty_weight requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("train.confidence_penalty_weight requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("train.confidence_penalty_weight requires noise.time_conditioning=false")
        if config.model.output_factorization != "flat":
            raise ValueError("train.confidence_penalty_weight requires model.output_factorization='flat'")
        if config.train.entropy_bonus_beta > 0:
            raise ValueError("train.confidence_penalty_weight cannot be combined with train.entropy_bonus_beta")
    if config.train.z_loss_weight < 0:
        raise ValueError("train.z_loss_weight must be non-negative")
    if config.train.z_loss_weight > 0:
        if config.model.causal:
            raise ValueError("train.z_loss_weight is only supported for non-causal MDLM training")
        if config.noise.type != "mdlm":
            raise ValueError("train.z_loss_weight requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("train.z_loss_weight requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("train.z_loss_weight requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("train.z_loss_weight requires noise.time_conditioning=false")
        if config.model.output_factorization != "flat":
            raise ValueError("train.z_loss_weight requires model.output_factorization='flat'")
        if config.train.auxiliary_loss_weight > 0:
            raise ValueError("train.z_loss_weight cannot be combined with train.auxiliary_loss_weight")
        if config.train.entropy_bonus_beta > 0:
            raise ValueError("train.z_loss_weight cannot be combined with train.entropy_bonus_beta")
        if config.train.confidence_penalty_weight > 0:
            raise ValueError("train.z_loss_weight cannot be combined with train.confidence_penalty_weight")
        if config.train.eos_hazard_loss_weight > 0:
            raise ValueError("train.z_loss_weight cannot be combined with train.eos_hazard_loss_weight")
    if config.train.eos_hazard_loss_weight < 0:
        raise ValueError("train.eos_hazard_loss_weight must be non-negative")
    if config.train.eos_hazard_loss_weight > 0:
        if config.model.causal:
            raise ValueError("train.eos_hazard_loss_weight is only supported for non-causal MDLM training")
        if config.noise.type != "mdlm":
            raise ValueError("train.eos_hazard_loss_weight requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("train.eos_hazard_loss_weight requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("train.eos_hazard_loss_weight requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("train.eos_hazard_loss_weight requires noise.time_conditioning=false")
        if config.model.output_factorization != "flat":
            raise ValueError("train.eos_hazard_loss_weight requires model.output_factorization='flat'")
        if not config.data.loss_on_eos:
            raise ValueError("train.eos_hazard_loss_weight requires data.loss_on_eos=true")
        if config.train.auxiliary_loss_weight > 0:
            raise ValueError("train.eos_hazard_loss_weight cannot be combined with train.auxiliary_loss_weight")
        if config.train.entropy_bonus_beta > 0:
            raise ValueError("train.eos_hazard_loss_weight cannot be combined with train.entropy_bonus_beta")
        if config.train.confidence_penalty_weight > 0:
            raise ValueError("train.eos_hazard_loss_weight cannot be combined with train.confidence_penalty_weight")
    if config.train.length_head_loss_weight < 0:
        raise ValueError("train.length_head_loss_weight must be non-negative")
    if config.train.length_head_loss_weight > 0:
        if not config.model.length_head:
            raise ValueError("train.length_head_loss_weight requires model.length_head=true")
        if config.model.causal:
            raise ValueError("train.length_head_loss_weight is only supported for non-causal MDLM training")
        if config.noise.type != "mdlm":
            raise ValueError("train.length_head_loss_weight requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("train.length_head_loss_weight requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("train.length_head_loss_weight requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("train.length_head_loss_weight requires noise.time_conditioning=false")
        if config.model.output_factorization != "flat":
            raise ValueError("train.length_head_loss_weight requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("train.length_head_loss_weight cannot be coupled with auxiliary heads")
        if not config.data.loss_on_eos:
            raise ValueError("train.length_head_loss_weight requires data.loss_on_eos=true")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("train.length_head_loss_weight cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("train.length_head_loss_weight cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("train.length_head_loss_weight cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("train.length_head_loss_weight cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("train.length_head_loss_weight cannot be coupled with EOS hazard loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("train.length_head_loss_weight cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("train.length_head_loss_weight cannot be coupled with uniform loss estimators")
    if config.model.target_length_conditioning:
        if config.train.length_head_loss_weight != 0:
            raise ValueError("model.target_length_conditioning cannot be coupled with train.length_head_loss_weight")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("model.target_length_conditioning cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("model.target_length_conditioning cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("model.target_length_conditioning cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("model.target_length_conditioning cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("model.target_length_conditioning cannot be coupled with EOS hazard loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("model.target_length_conditioning cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("model.target_length_conditioning cannot be coupled with uniform loss estimators")
    if config.noise.per_token_noise_levels:
        if config.model.causal:
            raise ValueError("noise.per_token_noise_levels is only supported for non-causal MDLM training")
        if config.model.length_head:
            raise ValueError("noise.per_token_noise_levels cannot be coupled with model.length_head")
        if config.model.target_length_conditioning:
            raise ValueError("noise.per_token_noise_levels cannot be coupled with target length conditioning")
        if config.model.auxiliary_head_layers:
            raise ValueError("noise.per_token_noise_levels cannot be coupled with auxiliary heads")
        if config.model.output_factorization != "flat":
            raise ValueError("noise.per_token_noise_levels requires model.output_factorization='flat'")
        if config.data.boundary_aware_packing:
            raise ValueError("noise.per_token_noise_levels requires data.boundary_aware_packing=false")
        if config.data.content_budget_segmented_packing:
            raise ValueError("noise.per_token_noise_levels requires data.content_budget_segmented_packing=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("noise.per_token_noise_levels cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("noise.per_token_noise_levels cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("noise.per_token_noise_levels cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("noise.per_token_noise_levels cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("noise.per_token_noise_levels cannot be coupled with EOS hazard loss")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("noise.per_token_noise_levels cannot be coupled with length-head loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("noise.per_token_noise_levels cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("noise.per_token_noise_levels cannot be coupled with uniform loss estimators")
        if config.train.token_edit_corruption_batch_fraction != 0 or config.train.token_edit_corruption_ratio != 0:
            raise ValueError("noise.per_token_noise_levels cannot be coupled with token-edit corruption")
    if config.train.label_smoothing_epsilon < 0 or config.train.label_smoothing_epsilon >= 1:
        raise ValueError("train.label_smoothing_epsilon must be in [0, 1)")
    if config.train.label_smoothing_epsilon > 0 and not config.model.causal:
        raise ValueError("train.label_smoothing_epsilon is only supported for causal AR training")
    if config.train.divergence_check_interval == 0:
        raise ValueError("train.divergence_check_interval must be positive or -1")
    if config.train.divergence_loss_threshold <= 0:
        raise ValueError("train.divergence_loss_threshold must be positive")
    validate_runtime_evaluation_config(config)
    if not isinstance(config.train.position_decile_balanced_loss, bool):
        raise ValueError("train.position_decile_balanced_loss must be a boolean")
    if config.train.position_decile_balanced_loss:
        if config.model.causal:
            raise ValueError("train.position_decile_balanced_loss requires a non-causal MDLM model")
        if config.noise.type != "mdlm":
            raise ValueError("train.position_decile_balanced_loss requires noise.type='mdlm'")
        if config.noise.suffix_masking:
            raise ValueError("train.position_decile_balanced_loss requires suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("train.position_decile_balanced_loss requires time_conditioning=false")
        if not config.noise.schedule_weight:
            raise ValueError("train.position_decile_balanced_loss requires schedule_weight=true")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("train.position_decile_balanced_loss cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("train.position_decile_balanced_loss cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("train.position_decile_balanced_loss cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("train.position_decile_balanced_loss cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("train.position_decile_balanced_loss cannot be coupled with EOS hazard loss")
    if config.train.uniform_loss_estimator not in ("single_sample", "two_sample_mean"):
        raise ValueError("train.uniform_loss_estimator must be 'single_sample' or 'two_sample_mean'")
    if config.train.uniform_loss_estimator != "single_sample":
        if config.model.causal:
            raise ValueError("train.uniform_loss_estimator requires a non-causal uniform model")
        if config.noise.type != "uniform":
            raise ValueError("train.uniform_loss_estimator requires noise.type='uniform'")
        if not config.noise.time_conditioning:
            raise ValueError("train.uniform_loss_estimator requires noise.time_conditioning=true")
        if config.noise.soft_input_curriculum:
            raise ValueError("train.uniform_loss_estimator cannot be coupled with soft-input curriculum")
        if config.model.output_factorization != "flat":
            raise ValueError("train.uniform_loss_estimator requires model.output_factorization='flat'")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("train.uniform_loss_estimator cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("train.uniform_loss_estimator cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("train.uniform_loss_estimator cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("train.uniform_loss_estimator cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("train.uniform_loss_estimator cannot be coupled with EOS hazard loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("train.uniform_loss_estimator cannot be coupled with position decile balancing")
    for field_name in (
        "token_edit_corruption_batch_fraction",
        "token_edit_corruption_ratio",
        "token_edit_correct_draft_loss_weight",
        "token_edit_wrong_draft_loss_weight",
        "token_edit_mixed_canvas_draft_fraction",
        "token_edit_trajectory_completion_fraction",
        "token_edit_correction_aux_weight",
        "token_edit_expected_accuracy_power",
        "token_edit_expected_accuracy_corruption_threshold",
        "token_edit_cross_view_nonregression_weight",
        "token_edit_draft_m2t_loss_weight",
        "token_edit_draft_m2t_credit_strength",
        "token_edit_draft_m2t_credit_clip_min",
        "token_edit_draft_m2t_credit_clip_max",
        "token_edit_prefix_min_fraction",
        "token_edit_prefix_max_fraction",
    ):
        field_value = getattr(config.train, field_name)
        if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
            raise ValueError(f"train.{field_name} must be numeric")
        if not math.isfinite(float(field_value)):
            raise ValueError(f"train.{field_name} must be finite")
    if config.train.token_edit_corruption_batch_fraction < 0 or config.train.token_edit_corruption_batch_fraction > 1:
        raise ValueError("train.token_edit_corruption_batch_fraction must be in [0, 1]")
    if config.train.token_edit_batch_layout not in (
        "whole_microbatch_cadence",
        "fixed_12_m2t_4_t2t_rows",
    ):
        raise ValueError(
            "train.token_edit_batch_layout must be 'whole_microbatch_cadence' or 'fixed_12_m2t_4_t2t_rows'"
        )
    if config.train.token_edit_corruption_schedule not in ("legacy_fixed_ratio", "match_mdlm_per_row"):
        raise ValueError("train.token_edit_corruption_schedule must be 'legacy_fixed_ratio' or 'match_mdlm_per_row'")
    if config.train.token_edit_corruption_ratio < 0 or config.train.token_edit_corruption_ratio > 1:
        raise ValueError("train.token_edit_corruption_ratio must be in [0, 1]")
    if config.train.token_edit_input_source not in ("random_replacement", "self_generated_argmax"):
        raise ValueError("train.token_edit_input_source must be 'random_replacement' or 'self_generated_argmax'")
    if config.train.token_edit_draft_model_source not in ("live_student", "frozen_checkpoint"):
        raise ValueError("train.token_edit_draft_model_source must be 'live_student' or 'frozen_checkpoint'")
    draft_checkpoint = config.train.token_edit_draft_checkpoint
    draft_checkpoint_sha256 = config.train.token_edit_draft_checkpoint_expected_sha256
    draft_checkpoint_size = config.train.token_edit_draft_checkpoint_expected_size_bytes
    if not isinstance(draft_checkpoint, str) or not isinstance(draft_checkpoint_sha256, str):
        raise ValueError("train token-edit draft checkpoint path and digest must be strings")
    if isinstance(draft_checkpoint_size, bool) or not isinstance(draft_checkpoint_size, int):
        raise ValueError("train.token_edit_draft_checkpoint_expected_size_bytes must be an integer")
    if config.train.token_edit_draft_model_source == "live_student":
        if draft_checkpoint or draft_checkpoint_sha256 or draft_checkpoint_size != -1:
            raise ValueError("live-student token-edit drafts require empty checkpoint binding fields")
    else:
        if not Path(draft_checkpoint).expanduser().is_absolute():
            raise ValueError("frozen token-edit draft checkpoint must be an absolute path")
        if len(draft_checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in draft_checkpoint_sha256
        ):
            raise ValueError("frozen token-edit draft checkpoint SHA-256 must be lowercase 64-hex")
        if draft_checkpoint_size <= 0:
            raise ValueError("frozen token-edit draft checkpoint size must be positive")
    correction_objectives = {
        "cross_entropy",
        "expected_accuracy",
        "cross_entropy_plus_expected_accuracy",
        "cross_entropy_plus_draft_preference",
    }
    if config.train.token_edit_correction_objective not in correction_objectives:
        raise ValueError(
            "train.token_edit_correction_objective must be 'cross_entropy', 'expected_accuracy', "
            "'cross_entropy_plus_expected_accuracy', or 'cross_entropy_plus_draft_preference'"
        )
    correction_aux_weight = float(config.train.token_edit_correction_aux_weight)
    expected_accuracy_power = float(config.train.token_edit_expected_accuracy_power)
    expected_accuracy_power_schedule = config.train.token_edit_expected_accuracy_power_schedule
    expected_accuracy_corruption_threshold = float(config.train.token_edit_expected_accuracy_corruption_threshold)
    cross_view_nonregression_weight = float(config.train.token_edit_cross_view_nonregression_weight)
    if not 0.0 < expected_accuracy_power <= 1.0:
        raise ValueError("train.token_edit_expected_accuracy_power must be in (0, 1]")
    if expected_accuracy_power_schedule not in ("constant", "corruption_threshold"):
        raise ValueError(
            "train.token_edit_expected_accuracy_power_schedule must be 'constant' or 'corruption_threshold'"
        )
    if not 0.0 < expected_accuracy_corruption_threshold < 1.0:
        raise ValueError("train.token_edit_expected_accuracy_corruption_threshold must be in (0, 1)")
    if expected_accuracy_power_schedule == "constant":
        if expected_accuracy_corruption_threshold != 0.8:
            raise ValueError("constant expected-accuracy power requires the default corruption threshold 0.8")
    elif expected_accuracy_power >= 1.0:
        raise ValueError("corruption-threshold expected-accuracy power requires a configured power below 1")
    if cross_view_nonregression_weight < 0.0:
        raise ValueError("train.token_edit_cross_view_nonregression_weight must be non-negative")
    correction_objective_is_default = config.train.token_edit_correction_objective == "cross_entropy"
    correction_objective_is_pure_reward = config.train.token_edit_correction_objective == "expected_accuracy"
    if correction_objective_is_default or correction_objective_is_pure_reward:
        if correction_aux_weight != 0.0:
            raise ValueError(
                f"train.token_edit_correction_objective={config.train.token_edit_correction_objective!r} "
                "requires token_edit_correction_aux_weight=0"
            )
    elif correction_aux_weight <= 0.0:
        raise ValueError("combined token-edit correction objectives require token_edit_correction_aux_weight>0")
    if config.train.token_edit_backbone_gradient_mode not in ("shared", "output_head_only", "residual_adapter"):
        raise ValueError(
            "train.token_edit_backbone_gradient_mode must be 'shared', 'output_head_only', or 'residual_adapter'"
        )
    trajectory_canvas_modes = {"trajectory_gold_context", "trajectory_model_context"}
    if config.train.token_edit_canvas_mode not in {"full_draft", "mixed_mask_draft", *trajectory_canvas_modes}:
        raise ValueError(
            "train.token_edit_canvas_mode must be 'full_draft', 'mixed_mask_draft', "
            "'trajectory_gold_context', or 'trajectory_model_context'"
        )
    if not isinstance(config.train.token_edit_full_draft_provenance, bool):
        raise ValueError("train.token_edit_full_draft_provenance must be a boolean")
    if not isinstance(config.train.token_edit_prefix_enabled, bool):
        raise ValueError("train.token_edit_prefix_enabled must be a boolean")
    if config.train.token_edit_prefix_canvas_mode not in ("fallible_draft", "clean"):
        raise ValueError("train.token_edit_prefix_canvas_mode must be 'fallible_draft' or 'clean'")
    if config.train.token_edit_prefix_loss_scope not in ("all_active", "suffix_active"):
        raise ValueError("train.token_edit_prefix_loss_scope must be 'all_active' or 'suffix_active'")
    prefix_min_fraction = float(config.train.token_edit_prefix_min_fraction)
    prefix_max_fraction = float(config.train.token_edit_prefix_max_fraction)
    if not 0.0 < prefix_min_fraction <= prefix_max_fraction < 1.0:
        raise ValueError("train token-edit prefix fractions must satisfy 0 < min <= max < 1")
    if (
        isinstance(config.train.token_edit_prefix_boundary_seed, bool)
        or not isinstance(config.train.token_edit_prefix_boundary_seed, int)
        or config.train.token_edit_prefix_boundary_seed < 0
    ):
        raise ValueError("train.token_edit_prefix_boundary_seed must be a non-negative integer")
    mixed_canvas = config.train.token_edit_canvas_mode == "mixed_mask_draft"
    trajectory_canvas = config.train.token_edit_canvas_mode in trajectory_canvas_modes
    token_edit_enabled = (
        config.train.token_edit_corruption_batch_fraction > 0
        or config.train.token_edit_corruption_ratio > 0
        or config.train.token_edit_corruption_schedule != "legacy_fixed_ratio"
        or config.train.token_edit_input_source != "random_replacement"
        or config.train.token_edit_batch_layout != "whole_microbatch_cadence"
        or config.train.token_edit_prefix_enabled
        or config.train.token_edit_draft_model_source != "live_student"
    )
    if config.train.token_edit_draft_model_source == "frozen_checkpoint":
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("frozen token-edit draft checkpoints require self-generated argmax drafts")
        if config.train.token_edit_canvas_mode != "full_draft":
            raise ValueError("frozen token-edit draft checkpoints currently require the full-draft canvas")
        if config.train.token_edit_prefix_enabled:
            raise ValueError("frozen token-edit draft checkpoints do not support token-edit prefix training")
        if float(config.train.token_edit_draft_m2t_loss_weight) != 0.0:
            raise ValueError("frozen token-edit draft checkpoints require a stopped-gradient draft pass")
        if config.train.token_edit_backbone_gradient_mode != "shared":
            raise ValueError("frozen token-edit draft checkpoints currently require shared backbone gradients")
    if config.train.token_edit_prefix_enabled:
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("token-edit prefix training requires self-generated token editing")
        if config.train.token_edit_batch_layout != "whole_microbatch_cadence":
            raise ValueError("token-edit prefix training requires whole_microbatch_cadence")
        if config.train.token_edit_canvas_mode != "full_draft":
            raise ValueError("token-edit prefix training requires token_edit_canvas_mode='full_draft'")
        if config.train.token_edit_prefix_loss_scope != "suffix_active":
            raise ValueError("enabled token-edit prefix training requires suffix_active loss")
    elif (
        config.train.token_edit_prefix_canvas_mode != "fallible_draft"
        or config.train.token_edit_prefix_loss_scope != "all_active"
        or prefix_min_fraction != 0.125
        or prefix_max_fraction != 0.75
        or config.train.token_edit_prefix_boundary_seed != 271828
    ):
        raise ValueError("disabled token-edit prefix controls must retain their defaults")
    fixed_row_layout = config.train.token_edit_batch_layout == "fixed_12_m2t_4_t2t_rows"
    if fixed_row_layout:
        if config.train.device_batch_size != 16:
            raise ValueError("fixed_12_m2t_4_t2t_rows requires train.device_batch_size=16")
        if config.train.total_batch_size != 16 * config.model.max_seq_len:
            raise ValueError(
                "fixed_12_m2t_4_t2t_rows requires one DB16 microbatch per optimizer update "
                "(train.total_batch_size=16*model.max_seq_len)"
            )
        if config.train.allow_batch_size_round_up:
            raise ValueError("fixed_12_m2t_4_t2t_rows requires train.allow_batch_size_round_up=false")
        if float(config.train.token_edit_corruption_batch_fraction) != 0.25:
            raise ValueError("fixed_12_m2t_4_t2t_rows requires token_edit_corruption_batch_fraction=0.25")
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("fixed_12_m2t_4_t2t_rows requires self-generated token editing")
        if config.train.token_edit_corruption_schedule != "match_mdlm_per_row":
            raise ValueError("fixed_12_m2t_4_t2t_rows requires token_edit_corruption_schedule='match_mdlm_per_row'")
        if config.train.token_edit_backbone_gradient_mode != "shared":
            raise ValueError("fixed_12_m2t_4_t2t_rows requires shared token-edit backbone gradients")
        if config.train.token_edit_canvas_mode != "full_draft":
            raise ValueError("fixed_12_m2t_4_t2t_rows requires the full-draft correction canvas")
        if config.train.token_edit_correction_objective != "expected_accuracy":
            raise ValueError("fixed_12_m2t_4_t2t_rows requires the expected-accuracy correction objective")
    draft_fraction = float(config.train.token_edit_mixed_canvas_draft_fraction)
    if mixed_canvas:
        if not 0.0 < draft_fraction < 1.0:
            raise ValueError("mixed-mask/draft canvases require token_edit_mixed_canvas_draft_fraction in (0, 1)")
    elif draft_fraction != 1.0:
        raise ValueError("full-draft canvases require token_edit_mixed_canvas_draft_fraction=1")
    trajectory_fraction = float(config.train.token_edit_trajectory_completion_fraction)
    trajectory_steps = config.train.token_edit_trajectory_steps
    if isinstance(trajectory_steps, bool) or not isinstance(trajectory_steps, int):
        raise ValueError("train.token_edit_trajectory_steps must be an integer")
    if trajectory_canvas:
        if not 0.0 < trajectory_fraction < 1.0:
            raise ValueError("trajectory canvases require token_edit_trajectory_completion_fraction in (0, 1)")
        if trajectory_steps < 2:
            raise ValueError("trajectory canvases require token_edit_trajectory_steps>=2")
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("trajectory canvases require self-generated token editing")
        if config.train.token_edit_batch_layout != "whole_microbatch_cadence":
            raise ValueError("trajectory canvases require whole_microbatch_cadence")
        if config.train.token_edit_prefix_enabled:
            raise ValueError("trajectory canvases and token-edit prefix training must be screened independently")
        if config.train.token_edit_backbone_gradient_mode != "shared":
            raise ValueError("trajectory canvases require shared backbone gradients")
        if float(config.train.token_edit_draft_m2t_loss_weight) != 0.0:
            raise ValueError("trajectory canvases require stopped-gradient rollout passes")
        if config.model.token_edit_provenance_embedding or config.train.token_edit_full_draft_provenance:
            raise ValueError("trajectory canvases must be screened without provenance extensions")
    elif trajectory_fraction != 0.5 or trajectory_steps != 4:
        raise ValueError("non-trajectory canvases require default trajectory controls 0.5/4")
    if config.train.token_edit_input_source == "random_replacement":
        if config.train.token_edit_correct_draft_loss_weight != 0:
            raise ValueError("random-replacement token editing requires token_edit_correct_draft_loss_weight=0")
        if config.train.token_edit_wrong_draft_loss_weight != 1:
            raise ValueError("random-replacement token editing requires token_edit_wrong_draft_loss_weight=1")
    if not correction_objective_is_default:
        if not token_edit_enabled or config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("non-default correction objectives require enabled self-generated token editing")
        if (
            float(config.train.token_edit_correct_draft_loss_weight) != 1.0
            or float(config.train.token_edit_wrong_draft_loss_weight) != 1.0
        ):
            raise ValueError("non-default correction objectives require correct/wrong draft weights 1/1")
    corruption_conditioned_expected_accuracy = expected_accuracy_power_schedule == "corruption_threshold"
    generalized_expected_accuracy = expected_accuracy_power != 1.0
    cross_view_nonregression = cross_view_nonregression_weight > 0.0
    if generalized_expected_accuracy and cross_view_nonregression:
        raise ValueError("generalized expected accuracy and cross-view non-regression must be screened independently")
    if generalized_expected_accuracy or cross_view_nonregression:
        mechanism_name = (
            "corruption-conditioned expected accuracy"
            if corruption_conditioned_expected_accuracy
            else "generalized expected accuracy"
            if generalized_expected_accuracy
            else "cross-view non-regression"
        )
        if not token_edit_enabled or config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError(f"{mechanism_name} requires enabled self-generated token editing")
        if config.train.token_edit_correction_objective != "expected_accuracy":
            raise ValueError(f"{mechanism_name} requires the expected_accuracy correction objective")
        if config.train.token_edit_correction_aux_weight != 0.0:
            raise ValueError(f"{mechanism_name} requires token_edit_correction_aux_weight=0")
        if config.train.token_edit_batch_layout != "whole_microbatch_cadence":
            raise ValueError(f"{mechanism_name} requires whole_microbatch_cadence")
        if config.train.token_edit_canvas_mode != "full_draft":
            raise ValueError(f"{mechanism_name} requires token_edit_canvas_mode='full_draft'")
        if config.train.token_edit_prefix_enabled:
            raise ValueError(f"{mechanism_name} does not support token-edit prefix training")
        if config.train.token_edit_backbone_gradient_mode != "shared":
            raise ValueError(f"{mechanism_name} requires shared backbone gradients")
        if float(config.train.token_edit_draft_m2t_loss_weight) != 0.0:
            raise ValueError(f"{mechanism_name} requires a stopped-gradient draft pass")
        if config.model.token_edit_provenance_embedding or config.train.token_edit_full_draft_provenance:
            raise ValueError(f"{mechanism_name} must be screened without provenance extensions")
    isolated_gradient_mode = config.train.token_edit_backbone_gradient_mode in {
        "output_head_only",
        "residual_adapter",
    }
    if isolated_gradient_mode and (
        not token_edit_enabled or config.train.token_edit_input_source != "self_generated_argmax"
    ):
        raise ValueError("isolated token-edit gradients require enabled self-generated token editing")
    if isolated_gradient_mode and config.runtime.init_single_rank_process_group:
        raise ValueError("isolated token-edit gradients currently require single-process training")
    if config.train.token_edit_backbone_gradient_mode == "residual_adapter":
        if config.model.token_edit_adapter_bottleneck_dim <= 0:
            raise ValueError("residual-adapter gradients require model.token_edit_adapter_bottleneck_dim>0")
        if config.model.token_edit_adapter_top_layers <= 0:
            raise ValueError("residual-adapter gradients require model.token_edit_adapter_top_layers>0")
        if mixed_canvas:
            raise ValueError("residual-adapter and mixed-canvas mechanisms must be screened independently")
    elif config.model.token_edit_adapter_bottleneck_dim != 0 or config.model.token_edit_adapter_top_layers != 0:
        raise ValueError("token-edit adapter parameters require token_edit_backbone_gradient_mode='residual_adapter'")
    full_draft_provenance = config.train.token_edit_full_draft_provenance
    if mixed_canvas:
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("mixed-mask/draft canvases require self-generated token editing")
        if not config.model.token_edit_provenance_embedding:
            raise ValueError("mixed-mask/draft canvases require model.token_edit_provenance_embedding=true")
        if full_draft_provenance:
            raise ValueError("mixed-mask/draft canvases cannot enable full-draft provenance")
        if config.model.token_edit_provenance_confidence_bins > 0:
            raise ValueError("confidence-binned provenance is restricted to full-draft canvases")
    elif full_draft_provenance:
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("full-draft provenance requires self-generated token editing")
        if not config.model.token_edit_provenance_embedding:
            raise ValueError("full-draft provenance requires model.token_edit_provenance_embedding=true")
        if config.train.token_edit_prefix_enabled:
            raise ValueError("full-draft provenance does not support token-edit prefix training")
    elif config.model.token_edit_provenance_embedding:
        if token_edit_enabled:
            raise ValueError("token-edit provenance embeddings require a provenance-enabled token-edit canvas")
    draft_m2t_weight = float(config.train.token_edit_draft_m2t_loss_weight)
    if not 0.0 <= draft_m2t_weight <= 1.0:
        raise ValueError("train.token_edit_draft_m2t_loss_weight must be in [0, 1]")
    credit_modes = {
        "uniform",
        "aligned_expected_error_reward",
        "shuffled_expected_error_reward",
    }
    credit_mode = config.train.token_edit_draft_m2t_credit_mode
    if credit_mode not in credit_modes:
        raise ValueError(
            "train.token_edit_draft_m2t_credit_mode must be 'uniform', "
            "'aligned_expected_error_reward', or 'shuffled_expected_error_reward'"
        )
    credit_strength = float(config.train.token_edit_draft_m2t_credit_strength)
    credit_clip_min = float(config.train.token_edit_draft_m2t_credit_clip_min)
    credit_clip_max = float(config.train.token_edit_draft_m2t_credit_clip_max)
    if credit_strength < 0.0:
        raise ValueError("train.token_edit_draft_m2t_credit_strength must be non-negative")
    if not 0.0 < credit_clip_min <= 1.0 <= credit_clip_max:
        raise ValueError("train draft-M2T credit clipping must satisfy 0 < clip_min <= 1 <= clip_max")
    if credit_mode == "uniform":
        if (credit_strength, credit_clip_min, credit_clip_max) != (1.0, 0.5, 1.5):
            raise ValueError("uniform draft-M2T credit controls must retain their defaults")
    else:
        if credit_strength <= 0.0:
            raise ValueError("non-uniform draft-M2T credit requires token_edit_draft_m2t_credit_strength>0")
        if draft_m2t_weight <= 0.0:
            raise ValueError("non-uniform draft-M2T credit requires token_edit_draft_m2t_loss_weight>0")
    if draft_m2t_weight > 0.0:
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("draft-M2T joint loss requires self-generated argmax drafts")
        if config.train.token_edit_batch_layout != "whole_microbatch_cadence":
            raise ValueError("draft-M2T joint loss requires whole_microbatch_cadence")
        if config.train.token_edit_canvas_mode != "full_draft":
            raise ValueError("draft-M2T joint loss requires token_edit_canvas_mode='full_draft'")
        if config.train.token_edit_prefix_enabled:
            raise ValueError("draft-M2T joint loss does not support token-edit prefix training")
        if config.train.token_edit_backbone_gradient_mode != "shared":
            raise ValueError("draft-M2T joint loss requires shared backbone gradients")
        if config.runtime.init_single_rank_process_group:
            raise ValueError("draft-M2T joint loss currently requires non-DDP single-process training")
        if config.train.token_edit_correction_objective != "expected_accuracy":
            raise ValueError("draft-M2T joint loss requires the expected_accuracy correction objective")
        if config.train.token_edit_correction_aux_weight != 0:
            raise ValueError("draft-M2T joint loss requires token_edit_correction_aux_weight=0")
    if token_edit_enabled:
        if config.train.token_edit_corruption_batch_fraction <= 0:
            raise ValueError("token-edit corruption requires a positive batch fraction")
        if config.train.token_edit_input_source == "self_generated_argmax":
            if config.train.token_edit_corruption_schedule != "match_mdlm_per_row":
                raise ValueError(
                    "self-generated token editing requires token_edit_corruption_schedule='match_mdlm_per_row'"
                )
            if config.train.token_edit_corruption_ratio != 0:
                raise ValueError("self-generated token editing requires token_edit_corruption_ratio=0")
            if config.train.token_edit_correct_draft_loss_weight <= 0:
                raise ValueError("self-generated token editing requires token_edit_correct_draft_loss_weight>0")
            if config.train.token_edit_wrong_draft_loss_weight <= 0:
                raise ValueError("self-generated token editing requires token_edit_wrong_draft_loss_weight>0")
            if config.data.loss_on_bos:
                raise ValueError("self-generated token editing requires data.loss_on_bos=false")
        elif config.train.token_edit_corruption_schedule == "legacy_fixed_ratio":
            if config.train.token_edit_corruption_ratio <= 0:
                raise ValueError("legacy fixed-ratio token-edit corruption requires a positive token ratio")
        elif config.train.token_edit_corruption_ratio != 0:
            raise ValueError("matched-MDLM token-edit corruption requires token_edit_corruption_ratio=0")
        if config.model.causal:
            raise ValueError("token-edit corruption is only supported for non-causal MDLM training")
        if config.noise.type != "mdlm":
            raise ValueError("token-edit corruption requires noise.type='mdlm'")
        if not config.noise.schedule_weight:
            raise ValueError("token-edit corruption preserves strict MDLM and requires schedule_weight=true")
        if config.noise.complement_masking:
            raise ValueError("token-edit corruption preserves strict MDLM and requires complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError(
                "token-edit corruption preserves independent MDLM masking and requires suffix_masking=false"
            )
        if config.noise.time_conditioning:
            raise ValueError("token-edit corruption preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError(
                "token-edit corruption preserves per-row MDLM times and requires per_token_noise_levels=false"
            )
        if config.model.bd3lm_block_size != 0:
            raise ValueError("token-edit corruption cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("token-edit corruption cannot be coupled with block-causal pilot training")
        if config.model.output_factorization != "flat":
            raise ValueError("token-edit corruption requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("token-edit corruption requires model.auxiliary_head_layers to be empty")
        if config.model.length_head:
            raise ValueError("token-edit corruption cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("token-edit corruption cannot be coupled with target-length conditioning")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("token-edit corruption cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("token-edit corruption cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("token-edit corruption cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("token-edit corruption cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("token-edit corruption cannot be coupled with EOS hazard loss")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("token-edit corruption cannot be coupled with length-head loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("token-edit corruption cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("token-edit corruption cannot be coupled with uniform loss estimators")
