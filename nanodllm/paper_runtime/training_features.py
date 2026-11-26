"""Pure configuration feature predicates used by training services.

This module is deliberately dependency-light so training and diagnostic services
can share feature decisions without relying on the legacy injected namespace.
"""

from .config import ExperimentConfig


def _training_diagnostic_requested(config: ExperimentConfig) -> bool:
    return any(
        (
            config.train.strict_mdlm_sampler_eos_reachability_canary_only,
            config.train.strict_mdlm_learned_eos_sampler_canary_only,
            config.train.strict_mdlm_all_token_bayes_audit_only,
            config.train.strict_mdlm_eos_overfit_canary_only,
            config.train.strict_mdlm_real_collator_micro_overfit_canary_only,
            config.train.strict_mdlm_right_context_canary_only,
            config.train.real_packing_audit_only,
            config.train.strict_mdlm_target_ce_audit_only,
            config.train.strict_mdlm_corruption_audit_only,
        )
    )


def _is_causal(config: ExperimentConfig) -> bool:
    return bool(getattr(config.model, "causal", False))


def _uses_block_causal_pilot(config: ExperimentConfig) -> bool:
    return int(getattr(config.model, "block_causal_pilot_size", 0)) > 0


def _uses_content_budget_segmented_packing(config: ExperimentConfig) -> bool:
    return bool(getattr(config.data, "content_budget_segmented_packing", False))


def _uses_bd3lm(config: ExperimentConfig) -> bool:
    return config.noise.type == "bd3lm"


def _uses_bd3lm_causal_clean_stream(config: ExperimentConfig) -> bool:
    return _uses_bd3lm(config) and config.model.bd3lm_clean_stream_mode == "causal"


def _uses_bd3lm_uniform_t2t(config: ExperimentConfig) -> bool:
    return _uses_bd3lm_causal_clean_stream(config) and float(config.train.bd3lm_uniform_t2t_batch_fraction) > 0.0


def _uses_bd3lm_causal_ar_auxiliary(config: ExperimentConfig) -> bool:
    return _uses_bd3lm_causal_clean_stream(config) and float(config.train.bd3lm_causal_ar_loss_weight) > 0.0


def _effective_eval_sampler(config: ExperimentConfig) -> str:
    """Name the sampler implementation selected after model-family routing."""

    if _is_causal(config):
        return "ar"
    if _uses_bd3lm(config):
        return "bd3lm"
    if _uses_block_causal_pilot(config):
        return "block_causal_pilot"
    if config.eval.sampler == "default":
        return config.noise.type
    return config.eval.sampler


def _uses_hierarchical_output(config: ExperimentConfig) -> bool:
    return config.model.output_factorization == "hierarchical_frequency_decile"


def _uses_length_head(config: ExperimentConfig) -> bool:
    return bool(config.model.length_head)


def _uses_target_length_conditioning(config: ExperimentConfig) -> bool:
    return bool(getattr(config.model, "target_length_conditioning", False))


def _uses_uniform_low_variance_estimator(config: ExperimentConfig) -> bool:
    return config.train.uniform_loss_estimator == "two_sample_mean"


def _uses_token_edit_corruption(config: ExperimentConfig) -> bool:
    if float(config.train.token_edit_corruption_batch_fraction) <= 0.0:
        return False
    return _uses_matched_token_edit_schedule(config) or float(config.train.token_edit_corruption_ratio) > 0.0


def _uses_self_generated_token_edit(config: ExperimentConfig) -> bool:
    return config.train.token_edit_input_source == "self_generated_argmax"


def _uses_frozen_token_edit_draft_model(config: ExperimentConfig) -> bool:
    return config.train.token_edit_draft_model_source == "frozen_checkpoint"


def _uses_fixed_row_token_edit_layout(config: ExperimentConfig) -> bool:
    return config.train.token_edit_batch_layout == "fixed_12_m2t_4_t2t_rows"


def _self_generated_draft_device_batch_size(config: ExperimentConfig) -> int:
    if _uses_fixed_row_token_edit_layout(config):
        return 4
    return int(config.train.device_batch_size)


def _token_edit_detaches_backbone(config: ExperimentConfig) -> bool:
    return config.train.token_edit_backbone_gradient_mode == "output_head_only"


def _token_edit_uses_residual_adapter(config: ExperimentConfig) -> bool:
    return config.train.token_edit_backbone_gradient_mode == "residual_adapter"


def _uses_mixed_token_edit_canvas(config: ExperimentConfig) -> bool:
    return config.train.token_edit_canvas_mode == "mixed_mask_draft"


def _uses_trajectory_token_edit_canvas(config: ExperimentConfig) -> bool:
    return config.train.token_edit_canvas_mode in {"trajectory_gold_context", "trajectory_model_context"}


def _uses_full_draft_token_edit_provenance(config: ExperimentConfig) -> bool:
    return bool(config.train.token_edit_full_draft_provenance)


def _uses_differentiable_self_generated_draft(config: ExperimentConfig) -> bool:
    return float(config.train.token_edit_draft_m2t_loss_weight) > 0.0


def _uses_token_edit_prefix(config: ExperimentConfig) -> bool:
    return bool(config.train.token_edit_prefix_enabled)


def _uses_matched_token_edit_schedule(config: ExperimentConfig) -> bool:
    return config.train.token_edit_corruption_schedule == "match_mdlm_per_row"


def _uses_legacy_fixed_token_edit_schedule(config: ExperimentConfig) -> bool:
    return config.train.token_edit_corruption_schedule == "legacy_fixed_ratio"


def _token_edit_support_policy(config: ExperimentConfig) -> str:
    if _uses_self_generated_token_edit(config):
        return "mdlm_masked_lossable_positions_with_generation_blocked_argmax"
    if _uses_matched_token_edit_schedule(config):
        return "ordinary_positive_byte_tokens"
    return "legacy_all_lossable_non_mask_vocabulary"


def _token_edit_probability_bounds(config: ExperimentConfig) -> tuple[float, float]:
    if _uses_matched_token_edit_schedule(config):
        return float(config.noise.t_min), float(config.noise.t_max)
    ratio = float(config.train.token_edit_corruption_ratio)
    return ratio, ratio


def _uses_per_token_noise_levels(config: ExperimentConfig) -> bool:
    return bool(getattr(config.noise, "per_token_noise_levels", False))


def _uses_mdlm_m2t_per_row_t(config: ExperimentConfig) -> bool:
    """Whether an M2T branch uses the strict MDLM per-row time estimator.

    A mixed M2T/T2T recipe is not a strict-MDLM recipe as a whole, but its M2T
    microbatches still use the same per-row time sampling as the baseline.
    """

    return (
        not _is_causal(config)
        and config.noise.type == "mdlm"
        and (bool(config.noise.schedule_weight) or config.noise.mdlm_time_sampling == "per_row")
        and not bool(config.noise.complement_masking)
        and not bool(config.noise.suffix_masking)
        and not bool(config.noise.time_conditioning)
        and not _uses_per_token_noise_levels(config)
        and not _uses_block_causal_pilot(config)
        and not _uses_bd3lm(config)
        and (not _uses_token_edit_corruption(config) or _uses_matched_token_edit_schedule(config))
        and config.model.output_factorization == "flat"
        and not config.model.auxiliary_head_layers
        and not bool(config.model.length_head)
        and not bool(config.model.target_length_conditioning)
        and float(config.train.auxiliary_loss_weight) == 0.0
        and float(config.train.entropy_bonus_beta) == 0.0
        and float(config.train.confidence_penalty_weight) == 0.0
        and float(config.train.z_loss_weight) == 0.0
        and float(config.train.eos_hazard_loss_weight) == 0.0
        and float(config.train.length_head_loss_weight) == 0.0
        and not bool(config.train.position_decile_balanced_loss)
        and config.train.uniform_loss_estimator == "single_sample"
    )


def _uses_strict_mdlm_per_row_t(config: ExperimentConfig) -> bool:
    """Whether the complete recipe is strict MDLM with per-row time sampling."""

    return _uses_mdlm_m2t_per_row_t(config) and not _uses_token_edit_corruption(config)


def _uses_schedule_weighted_nelbo(config: ExperimentConfig) -> bool:
    """Whether the model supports the campaign's shared-time absorbing-MDLM evaluator.

    This is deliberately independent of the training estimator. In particular, an
    arm trained with per-token noise levels or without the schedule weight must be
    scored with the same shared-time reference objective as the baseline. The
    integration range determines whether that score is the full objective or a
    truncated reference integral.
    """
    return (
        not _is_causal(config)
        and config.noise.type == "mdlm"
        and not _uses_block_causal_pilot(config)
        and not _uses_bd3lm(config)
    )


def _eos_supervision_audit_enabled(config: ExperimentConfig) -> bool:
    return (
        not _is_causal(config)
        and config.noise.type == "mdlm"
        and not bool(config.noise.complement_masking)
        and not bool(config.noise.suffix_masking)
        and not bool(config.noise.time_conditioning)
        and not _uses_per_token_noise_levels(config)
        and not _uses_block_causal_pilot(config)
        and not _uses_token_edit_corruption(config)
        and config.model.output_factorization == "flat"
        and not config.model.auxiliary_head_layers
        and not bool(config.model.length_head)
        and not bool(config.model.target_length_conditioning)
        and bool(config.data.append_eos)
        and bool(config.data.loss_on_eos)
        and float(config.train.auxiliary_loss_weight) == 0.0
        and float(config.train.entropy_bonus_beta) == 0.0
        and float(config.train.confidence_penalty_weight) == 0.0
        and float(config.train.z_loss_weight) == 0.0
        and float(config.train.eos_hazard_loss_weight) == 0.0
        and float(config.train.length_head_loss_weight) == 0.0
        and not bool(config.train.position_decile_balanced_loss)
    )


def _ddp_broadcast_buffers(config: ExperimentConfig) -> bool:
    return not _uses_uniform_low_variance_estimator(config)
