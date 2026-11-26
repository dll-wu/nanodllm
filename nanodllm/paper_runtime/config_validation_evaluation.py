"""Runtime numerical-policy and evaluation/sampling configuration validation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .nelbo_time_sampling import validate_nelbo_time_sampling

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_runtime_evaluation_config(config: ExperimentConfig) -> None:
    if config.runtime.dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("runtime.dtype must be one of bfloat16, float16, float32")
    if config.runtime.min_free_gpu_mib < 0:
        raise ValueError("runtime.min_free_gpu_mib must be non-negative")
    if config.runtime.float32_matmul_precision not in ("", "highest", "high", "medium"):
        raise ValueError("runtime.float32_matmul_precision must be '', 'highest', 'high', or 'medium'")
    if not isinstance(config.runtime.allow_tf32, bool):
        raise ValueError("runtime.allow_tf32 must be a boolean")
    if config.runtime.cudnn_allow_tf32 is not None and not isinstance(config.runtime.cudnn_allow_tf32, bool):
        raise ValueError("runtime.cudnn_allow_tf32 must be a boolean or null")
    if (
        config.runtime.device_type == "cuda"
        and config.runtime.float32_matmul_precision in ("high", "medium")
        and not config.runtime.allow_tf32
    ):
        raise ValueError("runtime.float32_matmul_precision high/medium requires runtime.allow_tf32=true")
    if (
        config.runtime.device_type == "cuda"
        and config.runtime.float32_matmul_precision == "highest"
        and config.runtime.allow_tf32
    ):
        raise ValueError("runtime.float32_matmul_precision highest requires runtime.allow_tf32=false")
    if not isinstance(config.runtime.deterministic_algorithms, bool):
        raise ValueError("runtime.deterministic_algorithms must be a boolean")
    if config.runtime.sdpa_backend not in ("auto", "math"):
        raise ValueError("runtime.sdpa_backend must be 'auto' or 'math'")
    if (
        config.runtime.device_type == "cuda"
        and config.runtime.deterministic_algorithms
        and config.runtime.cudnn_benchmark
    ):
        raise ValueError("deterministic CUDA requires runtime.cudnn_benchmark=false")
    validate_nelbo_time_sampling(config)
    if config.eval.source_checkpoint_identity_mode not in ("off", "sha256"):
        raise ValueError("eval.source_checkpoint_identity_mode must be 'off' or 'sha256'")
    if config.eval.source_checkpoint_identity_mode == "sha256":
        expected_sha256 = config.eval.source_checkpoint_expected_sha256
        if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
            raise ValueError("eval.source_checkpoint_expected_sha256 must be a lowercase 64-hex digest in sha256 mode")
        if config.eval.source_checkpoint_expected_size_bytes <= 0:
            raise ValueError("eval.source_checkpoint_expected_size_bytes must be positive in sha256 mode")
        if not config.train.eval_only:
            raise ValueError("eval.source_checkpoint_identity_mode='sha256' requires train.eval_only=true")
    elif config.eval.source_checkpoint_expected_sha256 or config.eval.source_checkpoint_expected_size_bytes != -1:
        raise ValueError("source-checkpoint expected identity fields require identity_mode='sha256'")
    if config.eval.fixed_ce_batches <= 0:
        raise ValueError("eval.fixed_ce_batches must be positive")
    if config.eval.fixed_ce_trace_enabled and not config.eval.enabled:
        raise ValueError("eval.fixed_ce_trace_enabled requires eval.enabled=true")
    if config.eval.fixed_ce_trace_enabled and config.model.causal:
        raise ValueError("eval.fixed_ce_trace_enabled is only supported for non-causal diffusion training")
    if config.eval.reconstruction_batches <= 0:
        raise ValueError("eval.reconstruction_batches must be positive")
    if config.eval.reconstruction_log_samples < 0:
        raise ValueError("eval.reconstruction_log_samples must be non-negative")
    if not config.eval.reconstruction_t_values:
        raise ValueError("eval.reconstruction_t_values must contain at least one t value")
    for t_val in config.eval.reconstruction_t_values:
        if t_val < 0 or t_val > 1:
            raise ValueError("eval.reconstruction_t_values entries must be in [0, 1]")
    if not isinstance(config.eval.token_edit_transition_eval, bool):
        raise ValueError("eval.token_edit_transition_eval must be a boolean")
    if (
        isinstance(config.eval.token_edit_transition_batches, bool)
        or not isinstance(config.eval.token_edit_transition_batches, int)
        or config.eval.token_edit_transition_batches <= 0
    ):
        raise ValueError("eval.token_edit_transition_batches must be a positive integer")
    if (
        isinstance(config.eval.token_edit_transition_frequency_batches, bool)
        or not isinstance(config.eval.token_edit_transition_frequency_batches, int)
        or config.eval.token_edit_transition_frequency_batches <= 0
    ):
        raise ValueError("eval.token_edit_transition_frequency_batches must be a positive integer")
    if (
        isinstance(config.eval.token_edit_transition_seed, bool)
        or not isinstance(config.eval.token_edit_transition_seed, int)
        or config.eval.token_edit_transition_seed < 0
        or config.eval.token_edit_transition_seed >= 2**63
    ):
        raise ValueError("eval.token_edit_transition_seed must be a non-negative integer smaller than 2**63")
    if (
        isinstance(config.eval.token_edit_transition_log_samples, bool)
        or not isinstance(config.eval.token_edit_transition_log_samples, int)
        or config.eval.token_edit_transition_log_samples < 0
    ):
        raise ValueError("eval.token_edit_transition_log_samples must be a non-negative integer")
    fractions = config.eval.token_edit_transition_mask_fractions
    if not fractions:
        raise ValueError("eval.token_edit_transition_mask_fractions must contain at least one fraction")
    normalized_fractions: list[float] = []
    metric_labels: list[str] = []
    for fraction in fractions:
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(float(fraction)):
            raise ValueError("eval.token_edit_transition_mask_fractions entries must be finite numbers")
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("eval.token_edit_transition_mask_fractions entries must be in (0, 1]")
        normalized_fractions.append(float(fraction))
        metric_labels.append(f"{int(round(100.0 * float(fraction))):03d}")
    if len(set(normalized_fractions)) != len(normalized_fractions):
        raise ValueError("eval.token_edit_transition_mask_fractions entries must be unique")
    if len(set(metric_labels)) != len(metric_labels):
        raise ValueError("eval.token_edit_transition_mask_fractions entries must have unique percentage labels")
    if config.eval.token_edit_transition_eval:
        transition_failures: list[str] = []
        if not config.eval.enabled:
            transition_failures.append("eval.enabled must be true")
        if not config.train.eval_only:
            transition_failures.append("train.eval_only must be true")
        if config.eval.source_checkpoint_identity_mode != "sha256":
            transition_failures.append("source_checkpoint_identity_mode='sha256' is required")
        if config.model.causal:
            transition_failures.append("model.causal must be false")
        if config.noise.type != "mdlm":
            transition_failures.append("noise.type must be 'mdlm'")
        if config.noise.time_conditioning:
            transition_failures.append("noise.time_conditioning must be false")
        if config.model.output_factorization != "flat":
            transition_failures.append("model.output_factorization must be 'flat'")
        if config.model.auxiliary_head_layers:
            transition_failures.append("model.auxiliary_head_layers must be empty")
        if config.model.bd3lm_block_size != 0 or config.model.block_causal_pilot_size != 0:
            transition_failures.append("block-factorized model modes must be disabled")
        if config.model.length_head or config.model.target_length_conditioning:
            transition_failures.append("length prediction and conditioning must be disabled")
        if transition_failures:
            raise ValueError(
                "eval.token_edit_transition_eval requires the Stage-1 flat MDLM contract: "
                + "; ".join(transition_failures)
            )
    if config.eval.gen_score_mode not in ("none", "self", "judge"):
        raise ValueError("eval.gen_score_mode must be one of 'none', 'self', or 'judge'")
    if config.eval.gen_samples <= 0:
        raise ValueError("eval.gen_samples must be positive")
    if config.eval.gen_len <= 0:
        raise ValueError("eval.gen_len must be positive")
    if config.eval.gen_log_samples < 0:
        raise ValueError("eval.gen_log_samples must be non-negative")
    if config.eval.gen_self_score_t < 0 or config.eval.gen_self_score_t > 1:
        raise ValueError("eval.gen_self_score_t must be in [0, 1]")
    if config.eval.sample_steps <= 0:
        raise ValueError("eval.sample_steps must be positive")
    if config.eval.sample_scoring_mode not in ("random", "maxprob", "margin", "entropy", "margin_jitter"):
        raise ValueError("eval.sample_scoring_mode must be a supported sampler scoring mode")
    if config.eval.sample_token_selection not in ("argmax", "multinomial"):
        raise ValueError("eval.sample_token_selection must be either 'argmax' or 'multinomial'")
    if not math.isfinite(float(config.eval.sample_jitter_scale)) or config.eval.sample_jitter_scale < 0:
        raise ValueError("eval.sample_jitter_scale must be finite and non-negative")
    if config.eval.conditional_rank_batches <= 0:
        raise ValueError("eval.conditional_rank_batches must be positive")
    if config.eval.conditional_rank_loo_eval and not config.eval.conditional_rank_eval:
        raise ValueError("eval.conditional_rank_loo_eval requires eval.conditional_rank_eval=true")
    if config.eval.conditional_rank_loo_positions_per_sequence <= 0:
        raise ValueError("eval.conditional_rank_loo_positions_per_sequence must be positive")
    if config.eval.conditional_rank_stratified_eval and not config.eval.conditional_rank_eval:
        raise ValueError("eval.conditional_rank_stratified_eval requires eval.conditional_rank_eval=true")
    if config.eval.conditional_rank_stratified_target_per_decile <= 0:
        raise ValueError("eval.conditional_rank_stratified_target_per_decile must be positive")
    if (
        config.eval.conditional_rank_stratified_partial_mask_fraction < 0
        or config.eval.conditional_rank_stratified_partial_mask_fraction > 1
    ):
        raise ValueError("eval.conditional_rank_stratified_partial_mask_fraction must be in [0, 1]")
    if config.eval.conditional_rank_stratified_log_samples < 0:
        raise ValueError("eval.conditional_rank_stratified_log_samples must be non-negative")
    if config.eval.conditional_rank_rollout_eval and not config.eval.conditional_rank_eval:
        raise ValueError("eval.conditional_rank_rollout_eval requires eval.conditional_rank_eval=true")
    if config.eval.conditional_rank_rollout_eval and config.eval.sampler not in ("default", "fhs"):
        raise ValueError("eval.conditional_rank_rollout_eval supports eval.sampler='default' or 'fhs'")
    if config.eval.conditional_rank_rollout_batches <= 0:
        raise ValueError("eval.conditional_rank_rollout_batches must be positive")
    if not config.eval.conditional_rank_rollout_commit_fractions:
        raise ValueError("eval.conditional_rank_rollout_commit_fractions must contain at least one value")
    for fraction in config.eval.conditional_rank_rollout_commit_fractions:
        if fraction <= 0 or fraction >= 1:
            raise ValueError("eval.conditional_rank_rollout_commit_fractions entries must be in (0, 1)")
    if config.eval.conditional_rank_rollout_log_samples < 0:
        raise ValueError("eval.conditional_rank_rollout_log_samples must be non-negative")
    if not isinstance(config.eval.conditional_gen_eval, bool):
        raise ValueError("eval.conditional_gen_eval must be a boolean")
    if not config.eval.conditional_gen_prefix_fracs:
        raise ValueError("eval.conditional_gen_prefix_fracs must contain at least one value")
    for fraction in config.eval.conditional_gen_prefix_fracs:
        if fraction <= 0 or fraction >= 1:
            raise ValueError("eval.conditional_gen_prefix_fracs entries must be in (0, 1)")
    if config.eval.conditional_gen_samples <= 0:
        raise ValueError("eval.conditional_gen_samples must be positive")
    if config.eval.conditional_gen_len <= 0:
        raise ValueError("eval.conditional_gen_len must be positive")
    if config.eval.conditional_gen_log_samples < 0:
        raise ValueError("eval.conditional_gen_log_samples must be non-negative")
    if config.eval.conditional_gen_eval and config.noise.type not in ("mdlm", "bd3lm"):
        raise ValueError("eval.conditional_gen_eval supports noise.type='mdlm' or 'bd3lm'")
    if config.eval.conditional_gen_eval and config.eval.gen_score_mode == "judge":
        raise ValueError("eval.conditional_gen_eval supports eval.gen_score_mode 'none' or 'self'")
    if config.eval.conditional_gen_eval and config.noise.type == "bd3lm":
        block_size = int(config.model.bd3lm_block_size)
        continuation_len = int(config.eval.conditional_gen_len)
        if config.eval.gen_score_mode != "none":
            raise ValueError("BD3-LM conditional generation currently requires eval.gen_score_mode='none'")
        if continuation_len % block_size != 0:
            raise ValueError("BD3-LM conditional generation requires conditional_gen_len to be block-aligned")
        if continuation_len >= int(config.model.max_seq_len):
            raise ValueError("BD3-LM conditional generation requires conditional_gen_len < model.max_seq_len")
        for fraction in config.eval.conditional_gen_prefix_fracs:
            prefix_len = int(round(continuation_len * float(fraction)))
            if prefix_len <= 0 or prefix_len % block_size != 0:
                raise ValueError(
                    "BD3-LM conditional generation requires every requested prefix length to be block-aligned"
                )
            if prefix_len + continuation_len > int(config.model.max_seq_len):
                raise ValueError("BD3-LM conditional generation prefix plus continuation exceeds model.max_seq_len")
    if not math.isfinite(float(config.eval.gen_temperature)) or config.eval.gen_temperature <= 0:
        raise ValueError("eval.gen_temperature must be finite and positive")
    if not isinstance(config.eval.eval_only_temperature_sweep, bool):
        raise ValueError("eval.eval_only_temperature_sweep must be a boolean")
    if config.eval.length_policy not in ("none", "argmax_eos"):
        raise ValueError("eval.length_policy must be one of 'none' or 'argmax_eos'")
    if config.eval.length_policy != "none" and not config.model.length_head:
        raise ValueError("eval.length_policy requires model.length_head=true")
    if config.model.target_length_conditioning and config.eval.length_policy != "none":
        raise ValueError("model.target_length_conditioning forbids forced EOS eval.length_policy")
    if config.eval.target_length_distribution_batches <= 0:
        raise ValueError("eval.target_length_distribution_batches must be positive")
    if config.eval.gen_prior_correction_strength < 0:
        raise ValueError("eval.gen_prior_correction_strength must be non-negative")
    if config.eval.gen_prior_correction_batches < 0:
        raise ValueError("eval.gen_prior_correction_batches must be non-negative")
    if config.eval.gen_prior_correction_smoothing <= 0:
        raise ValueError("eval.gen_prior_correction_smoothing must be positive")
    if config.eval.gen_prior_correction_strength > 0 and not config.train.eval_only:
        raise ValueError("eval.gen_prior_correction_strength is only supported for train.eval_only runs")
    if config.eval.gen_prior_correction_strength > 0 and config.eval.gen_prior_correction_batches <= 0:
        raise ValueError("eval.gen_prior_correction_batches must be positive when prior correction is enabled")
    if config.eval.sampler not in (
        "default",
        "left_to_right",
        "remdm",
        "fhs",
        "bd3lm",
        "block_causal_pilot",
    ):
        raise ValueError(
            "eval.sampler must be one of 'default', 'left_to_right', 'remdm', 'fhs', 'bd3lm', 'block_causal_pilot'"
        )
    if config.noise.type == "bd3lm" and config.eval.sampler not in ("default", "bd3lm"):
        raise ValueError("noise.type='bd3lm' requires eval.sampler='default' or 'bd3lm'")
    if config.eval.sampler == "bd3lm" and config.noise.type != "bd3lm":
        raise ValueError("eval.sampler='bd3lm' requires noise.type='bd3lm'")
    if not isinstance(config.eval.sample_top_k, int) or isinstance(config.eval.sample_top_k, bool):
        raise ValueError("eval.sample_top_k must be an integer")
    if config.eval.sample_top_k < 0:
        raise ValueError("eval.sample_top_k must be non-negative")
    if isinstance(config.eval.sample_top_p, bool) or not isinstance(config.eval.sample_top_p, (int, float)):
        raise ValueError("eval.sample_top_p must be a number")
    if (
        not math.isfinite(float(config.eval.sample_top_p))
        or config.eval.sample_top_p <= 0
        or config.eval.sample_top_p > 1
    ):
        raise ValueError("eval.sample_top_p must be in (0, 1]")
    if config.eval.sampler != "left_to_right" and (config.eval.sample_top_k != 0 or config.eval.sample_top_p != 1.0):
        raise ValueError("eval.sample_top_k and eval.sample_top_p are currently supported only by left_to_right")
    if config.eval.sampler == "left_to_right":
        if config.noise.type != "mdlm":
            raise ValueError("eval.sampler='left_to_right' currently supports noise.type='mdlm'")
        if config.model.causal:
            raise ValueError("eval.sampler='left_to_right' requires model.causal=false")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("eval.sampler='left_to_right' cannot be combined with block-causal pilot mode")
        if config.eval.sample_jitter_scale != 0:
            raise ValueError("eval.sampler='left_to_right' requires eval.sample_jitter_scale=0")
        if config.eval.remasking_comparison:
            raise ValueError("eval.sampler='left_to_right' cannot be combined with eval.remasking_comparison")
        if config.eval.sample_token_selection != "multinomial" and (
            config.eval.sample_top_k != 0 or config.eval.sample_top_p != 1.0
        ):
            raise ValueError("left_to_right top-k/top-p filtering requires multinomial token selection")
        if config.eval.gen_eval and config.eval.sample_steps != config.eval.gen_len:
            raise ValueError("left_to_right unconditional generation requires eval.sample_steps == eval.gen_len")
        if config.eval.conditional_gen_eval and config.eval.sample_steps != config.eval.conditional_gen_len:
            raise ValueError(
                "left_to_right conditional generation requires eval.sample_steps == eval.conditional_gen_len"
            )
    if config.eval.fhs_tau < 0 or config.eval.fhs_tau > 1:
        raise ValueError("eval.fhs_tau must be in [0, 1]")
    if config.eval.remdm_tau_edit < 0 or config.eval.remdm_tau_edit > 1:
        raise ValueError("eval.remdm_tau_edit must be in [0, 1]")
    if config.eval.basin_prefix_len <= 0:
        raise ValueError("eval.basin_prefix_len must be positive")
    if not config.eval.basin_forced_argmax_depths:
        raise ValueError("eval.basin_forced_argmax_depths must contain at least one depth")
    for depth in config.eval.basin_forced_argmax_depths:
        if depth < 0:
            raise ValueError("eval.basin_forced_argmax_depths entries must be non-negative")
    if config.eval.basin_nucleus_top_p <= 0 or config.eval.basin_nucleus_top_p > 1:
        raise ValueError("eval.basin_nucleus_top_p must be in (0, 1]")
