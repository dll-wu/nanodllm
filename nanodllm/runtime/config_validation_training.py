"""Training budgets, checkpoint initialization, and objective compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .telemetry_config import validate_telemetry_config

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_training_budget_and_initialization(config: ExperimentConfig) -> None:
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


def validate_replay_ids(config: ExperimentConfig) -> None:
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


def validate_training_losses(config: ExperimentConfig) -> None:
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


def validate_loss_estimators(config: ExperimentConfig) -> None:
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
