import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    depth: int = 24
    n_embd: int = 1536
    n_head: int = 12
    max_seq_len: int = 2048
    # -1 preserves the historical 4 * n_embd SwiGLU width.
    mlp_hidden_dim: int = -1
    # -1 preserves PyTorch SDPA's historical 1 / sqrt(head_dim) default.
    attention_scale: float = -1.0
    rope_base: float = 10000.0
    pad_vocab_to: int = 64
    local_attention_lower_layers: int = 0
    local_attention_window: int = 0
    auxiliary_head_layers: list[int] = field(default_factory=list)
    bd3lm_block_size: int = 0
    block_causal_pilot_size: int = 0
    output_factorization: str = "flat"
    hierarchical_frequency_batches: int = 64
    causal: bool = False
    residual_output_projection_init: str = "zero"
    embedding_head_init_std_multiplier: float = 1.0
    length_head: bool = False
    target_length_conditioning: bool = False
    mask_position_embedding: bool = False


@dataclass
class DataConfig:
    tokenizer: str = "hf"
    tokenizer_path: str = "tokenizers/climbmix-32k"
    train_data: list[str] = field(default_factory=list)
    train_exclude_data: list[str] = field(default_factory=list)
    val_data: list[str] = field(default_factory=list)
    synthetic_dataset: str = ""
    synthetic_eos_overfit_loss_mask_policy: str = "eos_only"
    synthetic_eos_overfit_unique_non_eos_tokens: bool = False
    synthetic_eos_overfit_unique_non_eos_token_count: int = 0
    synthetic_eos_overfit_fixed_row_text: str = ""
    synthetic_eos_overfit_document_lengths: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 1, 2, 3, 4])
    text_column: str = "text"
    val_fraction: float = 0.10
    tokenizer_batch_size: int = 128
    prepend_bos: bool = False
    append_eos: bool = True
    loss_on_bos: bool = False
    loss_on_eos: bool = True
    random_length_ratio: float = 0.01
    random_length_min: int = 1
    continuation_ratio: float = 0.10
    continuation_min_prefix_frac: float = 0.125
    continuation_max_prefix_frac: float = 0.75
    shuffle_row_groups: bool = True
    shuffle_seed: int = 1337
    boundary_aware_packing: bool = False
    content_budget_segmented_packing: bool = False
    short_document_curriculum: bool = False
    short_document_max_tokens: int = 256
    identity_mode: str = "off"


@dataclass
class NoiseConfig:
    type: str = "mdlm"
    t_min: float = 0.05
    t_max: float = 0.95
    schedule_weight: bool = True
    complement_masking: bool = True
    per_token_noise_levels: bool = False
    mask_uniform_mix_prob: float = 0.25
    time_conditioning: bool = False
    suffix_masking: bool = False
    uniform_t_block_size: int = 16
    uniform_t_coarse_block_size: int = 64
    soft_input_curriculum: bool = False
    soft_input_curriculum_start_mix: float = 1.0
    soft_input_curriculum_end_mix: float = 0.0
    soft_input_curriculum_anneal_fraction: float = 1.0


@dataclass
class OptimConfig:
    type: str = "adamw"
    lr: float = 3e-4
    muon_lr: float = -1.0
    adamw_lr: float = -1.0
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_steps: int = 40
    warmup_ratio: float = -1.0
    warmdown_ratio: float = 0.2
    final_lr_frac: float = 0.1
    grad_clip: float = 1.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"


@dataclass
class RuntimeConfig:
    device_type: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 1337
    model_seed: int = -1
    training_seed: int = -1
    compile: bool = True
    compile_dynamic: bool = False
    min_free_gpu_mib: int = 0
    init_single_rank_process_group: bool = False
    float32_matmul_precision: str = ""
    allow_tf32: bool = False
    cudnn_allow_tf32: bool | None = None
    cudnn_benchmark: bool = False
    deterministic_algorithms: bool = False
    sdpa_backend: str = "auto"


@dataclass
class TrainConfig:
    run_dir: str = "runs/nanodllm"
    num_steps: int = -1
    # Token budget, in priority order: num_steps > target_tokens > target_param_data_ratio.
    target_tokens: int = -1
    # Ratio denominator is selected explicitly from an instantiated-model
    # parameter-accounting view. The default preserves historical configs.
    target_param_data_ratio: float = 12.0
    target_param_count_basis: str = "transformer_body_parameters"
    total_batch_size: int = -1
    device_batch_size: int = 16
    base_total_batch_size: int = 2**19
    allow_batch_size_round_up: bool = False
    scale_lr_by_batch: bool = True
    log_interval: int = 10
    tensorboard: bool = True
    tensorboard_dir: str = ""
    save_every: int = -1
    eval_every: int = -1
    resume_from: str = ""
    comparison_run_ids: list[str] = field(default_factory=list)
    eval_only: bool = False
    strict_mdlm_corruption_audit_only: bool = False
    strict_mdlm_target_ce_audit_only: bool = False
    strict_mdlm_right_context_canary_only: bool = False
    strict_mdlm_eos_overfit_canary_only: bool = False
    strict_mdlm_sampler_eos_reachability_canary_only: bool = False
    strict_mdlm_learned_eos_sampler_canary_only: bool = False
    strict_mdlm_real_collator_micro_overfit_canary_only: bool = False
    strict_mdlm_real_collator_component_probe: bool = False
    strict_mdlm_all_token_bayes_audit_only: bool = False
    strict_mdlm_all_token_diagnostic_telemetry: bool = False
    strict_mdlm_all_token_diagnostic_t_values: list[float] = field(default_factory=list)
    strict_mdlm_eos_logit_bias_calibration: bool = False
    strict_mdlm_eos_logit_bias_candidates: list[float] = field(default_factory=list)
    strict_mdlm_teacher_forced_replay_diagnostic: bool = False
    strict_mdlm_t_curriculum_enabled: bool = False
    strict_mdlm_t_curriculum_warmup_steps: int = 0
    strict_mdlm_t_curriculum_warmup_t: float = 0.0
    strict_mdlm_t_curriculum_final_t: float = 0.0
    strict_mdlm_uniform_schedule_weight: bool = False
    strict_mdlm_eos_loss_mass_balance: bool = False
    real_packing_audit_only: bool = False
    auxiliary_loss_weight: float = 0.0
    label_smoothing_epsilon: float = 0.0
    entropy_bonus_beta: float = 0.0
    confidence_penalty_weight: float = 0.0
    z_loss_weight: float = 0.0
    eos_hazard_loss_weight: float = 0.0
    length_head_loss_weight: float = 0.0
    position_decile_balanced_loss: bool = False
    uniform_loss_estimator: str = "single_sample"
    token_edit_corruption_batch_fraction: float = 0.0
    token_edit_corruption_ratio: float = 0.0
    divergence_check_interval: int = -1
    divergence_loss_threshold: float = 100.0


@dataclass
class EvalConfig:
    enabled: bool = True
    gen_eval: bool = True
    gen_eval_periodic: bool = False
    gen_score_mode: str = "self"
    nelbo_tokens: int = 40 * 524288
    nelbo_t_min: float = 1e-3
    nelbo_t_max: float = 1.0
    nelbo_seed: int = 314159
    nelbo_time_sampling: str = "stratified_per_sequence"
    source_checkpoint_identity_mode: str = "off"
    source_checkpoint_expected_sha256: str = ""
    source_checkpoint_expected_size_bytes: int = -1
    fixed_ce_batches: int = 5
    fixed_ce_trace_enabled: bool = False
    reconstruction_eval: bool = True
    reconstruction_batches: int = 2
    reconstruction_t_values: list[float] = field(default_factory=lambda: [0.7, 0.9])
    reconstruction_log_samples: int = 3
    gen_samples: int = 512
    gen_len: int = 512
    gen_log_samples: int = 5
    gen_self_score_t: float = 0.5
    sample_steps: int = 128
    sample_scoring_mode: str = "margin_jitter"
    sample_token_selection: str = "argmax"
    sample_jitter_scale: float = 0.02
    sample_trace: bool = True
    judge_model_id: str = "Qwen/Qwen3-0.6B-Base"
    conditional_rank_eval: bool = True
    conditional_rank_batches: int = 5
    conditional_rank_loo_eval: bool = False
    conditional_rank_loo_positions_per_sequence: int = 8
    conditional_rank_stratified_eval: bool = False
    conditional_rank_stratified_target_per_decile: int = 64
    conditional_rank_stratified_partial_mask_fraction: float = 0.25
    conditional_rank_stratified_log_samples: int = 8
    conditional_rank_rollout_eval: bool = False
    conditional_rank_rollout_batches: int = 5
    conditional_rank_rollout_commit_fractions: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    conditional_rank_rollout_log_samples: int = 8
    conditional_gen_eval: bool = False
    conditional_gen_prefix_fracs: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    conditional_gen_samples: int = 64
    conditional_gen_len: int = 256
    conditional_gen_log_samples: int = 8
    gen_temperature: float = 1.0
    length_policy: str = "none"
    gen_prior_correction_strength: float = 0.0
    gen_prior_correction_batches: int = 0
    gen_prior_correction_smoothing: float = 1.0
    sampler: str = "default"
    fhs_tau: float = 0.9
    remdm_tau_edit: float = 0.9
    remasking_comparison: bool = False
    basin_diagnostic: bool = False
    basin_forced_argmax_depths: list[int] = field(default_factory=lambda: [0, 1, 2, 4])
    basin_prefix_len: int = 16
    basin_nucleus_top_p: float = 0.95
    target_length_distribution_batches: int = 8


@dataclass
class RecipeMetadataConfig:
    """Human- and machine-readable readiness state for a checked-in recipe."""

    name: str = ""
    status: str = "unspecified"
    locked_fields: list[str] = field(default_factory=list)
    provisional_fields: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    recipe: RecipeMetadataConfig = field(default_factory=RecipeMetadataConfig)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _expand_environment_variables(value: Any) -> Any:
    """Expand environment variables in all string-valued config fields."""

    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    return value


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    data = _expand_environment_variables(data)
    defaults = asdict(ExperimentConfig())
    merged = _deep_update(defaults, data)
    if "target_flops" in merged["train"]:
        raise ValueError("train.target_flops has been replaced by train.target_tokens")
    cfg = ExperimentConfig(
        model=ModelConfig(**merged["model"]),
        data=DataConfig(**merged["data"]),
        noise=NoiseConfig(**merged["noise"]),
        optim=OptimConfig(**merged["optim"]),
        runtime=RuntimeConfig(**merged["runtime"]),
        train=TrainConfig(**merged["train"]),
        eval=EvalConfig(**merged["eval"]),
        recipe=RecipeMetadataConfig(**merged["recipe"]),
    )
    validate_config(cfg)
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        return config_from_dict(json.load(f))


def to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def write_config(path: str | Path, config: ExperimentConfig) -> None:
    with open(path, "w") as f:
        json.dump(to_dict(config), f, indent=2)


def validate_config(config: ExperimentConfig) -> None:
    sampler_eos_reachability_canary = bool(
        getattr(config.train, "strict_mdlm_sampler_eos_reachability_canary_only", False)
    )
    if config.data.tokenizer not in ("byte", "hf"):
        raise ValueError("data.tokenizer must be 'byte' or 'hf'")
    if config.data.tokenizer == "hf" and not config.data.tokenizer_path:
        raise ValueError("data.tokenizer_path is required when data.tokenizer is 'hf'")
    if config.data.synthetic_dataset not in ("", "eos_overfit_canary"):
        raise ValueError("data.synthetic_dataset must be empty or 'eos_overfit_canary'")
    if config.data.synthetic_eos_overfit_loss_mask_policy not in ("eos_only", "all_lossable_tokens"):
        raise ValueError("data.synthetic_eos_overfit_loss_mask_policy must be 'eos_only' or 'all_lossable_tokens'")
    if not isinstance(config.data.synthetic_eos_overfit_unique_non_eos_tokens, bool):
        raise ValueError("data.synthetic_eos_overfit_unique_non_eos_tokens must be a boolean")
    if not isinstance(config.data.synthetic_eos_overfit_unique_non_eos_token_count, int):
        raise ValueError("data.synthetic_eos_overfit_unique_non_eos_token_count must be an integer")
    if config.data.synthetic_eos_overfit_unique_non_eos_token_count < 0:
        raise ValueError("data.synthetic_eos_overfit_unique_non_eos_token_count must be >= 0")
    if config.data.synthetic_eos_overfit_unique_non_eos_token_count > 62:
        raise ValueError("data.synthetic_eos_overfit_unique_non_eos_token_count must be <= 62")
    if (
        config.data.synthetic_eos_overfit_unique_non_eos_tokens
        and config.data.synthetic_dataset != "eos_overfit_canary"
    ):
        raise ValueError("data.synthetic_eos_overfit_unique_non_eos_tokens is only configurable for eos_overfit_canary")
    if (
        config.data.synthetic_eos_overfit_unique_non_eos_token_count
        and config.data.synthetic_dataset != "eos_overfit_canary"
    ):
        raise ValueError(
            "data.synthetic_eos_overfit_unique_non_eos_token_count is only configurable for eos_overfit_canary"
        )
    if config.data.synthetic_eos_overfit_fixed_row_text and config.data.synthetic_dataset != "eos_overfit_canary":
        raise ValueError("data.synthetic_eos_overfit_fixed_row_text is only configurable for eos_overfit_canary")
    if config.data.synthetic_eos_overfit_fixed_row_text and (
        config.data.synthetic_eos_overfit_unique_non_eos_tokens
        or config.data.synthetic_eos_overfit_unique_non_eos_token_count
    ):
        raise ValueError(
            "data.synthetic_eos_overfit_fixed_row_text is mutually exclusive with "
            "synthetic_eos_overfit_unique_non_eos_tokens and "
            "synthetic_eos_overfit_unique_non_eos_token_count"
        )
    document_lengths = getattr(
        config.data,
        "synthetic_eos_overfit_document_lengths",
        [1, 2, 3, 4, 1, 2, 3, 4],
    )
    if not isinstance(document_lengths, list):
        raise ValueError("data.synthetic_eos_overfit_document_lengths must be a list")
    if not document_lengths:
        raise ValueError("data.synthetic_eos_overfit_document_lengths must be non-empty")
    for length in document_lengths:
        if not isinstance(length, int):
            raise ValueError("data.synthetic_eos_overfit_document_lengths entries must be integers")
        if length <= 0:
            raise ValueError("data.synthetic_eos_overfit_document_lengths entries must be positive")
        if length > config.model.max_seq_len:
            raise ValueError("data.synthetic_eos_overfit_document_lengths entries must not exceed model.max_seq_len")
    if config.data.synthetic_dataset != "eos_overfit_canary" and document_lengths != [1, 2, 3, 4, 1, 2, 3, 4]:
        raise ValueError("data.synthetic_eos_overfit_document_lengths is only configurable for eos_overfit_canary")
    if (
        config.data.synthetic_eos_overfit_unique_non_eos_tokens
        and config.data.synthetic_eos_overfit_unique_non_eos_token_count
    ):
        raise ValueError(
            "set either synthetic_eos_overfit_unique_non_eos_tokens "
            "or synthetic_eos_overfit_unique_non_eos_token_count, not both"
        )
    if (
        config.data.synthetic_dataset != "eos_overfit_canary"
        and config.data.synthetic_eos_overfit_loss_mask_policy != "eos_only"
    ):
        raise ValueError("data.synthetic_eos_overfit_loss_mask_policy is only configurable for eos_overfit_canary")
    if config.data.train_exclude_data and not config.data.train_data:
        raise ValueError("data.train_exclude_data requires data.train_data")
    if config.data.train_exclude_data and config.data.synthetic_dataset:
        raise ValueError("data.train_exclude_data cannot be combined with data.synthetic_dataset")
    if not config.data.train_data and not config.data.synthetic_dataset and not sampler_eos_reachability_canary:
        raise ValueError(
            "data.train_data must contain at least one parquet file, directory, glob, or synthetic dataset"
        )
    if config.data.val_fraction < 0 or config.data.val_fraction >= 1:
        raise ValueError("data.val_fraction must be in [0, 1)")
    if config.data.tokenizer_batch_size <= 0:
        raise ValueError("data.tokenizer_batch_size must be positive")
    if not config.data.append_eos:
        raise ValueError("data.append_eos must be true for dLLM pretraining")
    if config.data.random_length_ratio < 0 or config.data.random_length_ratio > 1:
        raise ValueError("data.random_length_ratio must be in [0, 1]")
    if config.data.continuation_ratio < 0 or config.data.continuation_ratio > 1:
        raise ValueError("data.continuation_ratio must be in [0, 1]")
    if config.data.random_length_ratio + config.data.continuation_ratio > 1:
        raise ValueError("random_length_ratio + continuation_ratio must be <= 1")
    if config.data.random_length_min <= 0:
        raise ValueError("data.random_length_min must be positive")
    if config.data.continuation_min_prefix_frac < 0:
        raise ValueError("data.continuation_min_prefix_frac must be non-negative")
    if config.data.continuation_max_prefix_frac > 1:
        raise ValueError("data.continuation_max_prefix_frac must be <= 1")
    if config.data.continuation_min_prefix_frac >= config.data.continuation_max_prefix_frac:
        raise ValueError("data.continuation_min_prefix_frac must be < data.continuation_max_prefix_frac")
    if not isinstance(config.data.boundary_aware_packing, bool):
        raise ValueError("data.boundary_aware_packing must be a boolean")
    if not isinstance(config.data.content_budget_segmented_packing, bool):
        raise ValueError("data.content_budget_segmented_packing must be a boolean")
    if config.data.content_budget_segmented_packing:
        if config.data.boundary_aware_packing:
            raise ValueError("data.content_budget_segmented_packing cannot be combined with boundary_aware_packing")
        if config.data.short_document_curriculum:
            raise ValueError("data.content_budget_segmented_packing cannot be combined with short_document_curriculum")
        if config.model.causal:
            raise ValueError("data.content_budget_segmented_packing requires model.causal=false")
        if config.noise.type != "mdlm":
            raise ValueError("data.content_budget_segmented_packing requires noise.type='mdlm'")
        if config.noise.time_conditioning:
            raise ValueError(
                "data.content_budget_segmented_packing preserves MDLM parity and requires time_conditioning=false"
            )
        if config.noise.suffix_masking:
            raise ValueError(
                "data.content_budget_segmented_packing preserves independent MDLM masking and requires suffix_masking=false"
            )
    if not isinstance(config.data.short_document_curriculum, bool):
        raise ValueError("data.short_document_curriculum must be a boolean")
    if config.data.identity_mode not in ("off", "metadata", "full_sha256"):
        raise ValueError("data.identity_mode must be one of 'off', 'metadata', or 'full_sha256'")
    if config.data.identity_mode != "off" and config.data.synthetic_dataset:
        raise ValueError("data.identity_mode requires parquet data rather than data.synthetic_dataset")
    if config.data.short_document_max_tokens <= 0:
        raise ValueError("data.short_document_max_tokens must be positive")
    if config.data.short_document_curriculum:
        if config.data.boundary_aware_packing:
            raise ValueError("data.short_document_curriculum cannot be combined with boundary_aware_packing")
        if config.data.short_document_max_tokens < 2:
            raise ValueError("data.short_document_max_tokens must be at least 2 when short curriculum is enabled")
        if config.data.short_document_max_tokens > config.model.max_seq_len:
            raise ValueError("data.short_document_max_tokens must not exceed model.max_seq_len")
    if config.model.depth <= 0:
        raise ValueError("model.depth must be positive")
    if config.model.n_embd <= 0:
        raise ValueError("model.n_embd must be positive")
    if config.model.n_head <= 0:
        raise ValueError("model.n_head must be positive")
    if config.model.n_embd % config.model.n_head != 0:
        raise ValueError("model.n_embd must be divisible by model.n_head")
    if not isinstance(config.model.mlp_hidden_dim, int) or isinstance(config.model.mlp_hidden_dim, bool):
        raise ValueError("model.mlp_hidden_dim must be an integer")
    if config.model.mlp_hidden_dim != -1 and config.model.mlp_hidden_dim <= 0:
        raise ValueError("model.mlp_hidden_dim must be positive or -1 for the historical 4 * n_embd width")
    if isinstance(config.model.attention_scale, bool) or not isinstance(config.model.attention_scale, (int, float)):
        raise ValueError("model.attention_scale must be numeric")
    if config.model.attention_scale != -1.0 and config.model.attention_scale <= 0:
        raise ValueError(
            "model.attention_scale must be positive or -1 for PyTorch's historical 1 / sqrt(head_dim) default"
        )
    if config.model.local_attention_lower_layers < 0:
        raise ValueError("model.local_attention_lower_layers must be non-negative")
    if config.model.local_attention_lower_layers > 0 and config.model.local_attention_window <= 0:
        raise ValueError("model.local_attention_window must be positive when local attention is enabled")
    if not isinstance(config.model.causal, bool):
        raise ValueError("model.causal must be a boolean")
    if config.model.output_factorization not in ("flat", "hierarchical_frequency_decile"):
        raise ValueError("model.output_factorization must be 'flat' or 'hierarchical_frequency_decile'")
    if config.model.residual_output_projection_init not in ("zero", "scaled_normal"):
        raise ValueError("model.residual_output_projection_init must be 'zero' or 'scaled_normal'")
    if config.model.embedding_head_init_std_multiplier <= 0:
        raise ValueError("model.embedding_head_init_std_multiplier must be positive")
    if not isinstance(config.model.length_head, bool):
        raise ValueError("model.length_head must be a boolean")
    if not isinstance(config.model.target_length_conditioning, bool):
        raise ValueError("model.target_length_conditioning must be a boolean")
    if not isinstance(config.model.mask_position_embedding, bool):
        raise ValueError("model.mask_position_embedding must be a boolean")
    if config.model.mask_position_embedding and config.model.causal:
        raise ValueError("model.mask_position_embedding is only supported for non-causal MDLM training")
    if config.model.target_length_conditioning:
        if config.model.length_head:
            raise ValueError("model.target_length_conditioning cannot be coupled with model.length_head")
        if config.model.causal:
            raise ValueError("model.target_length_conditioning is only supported for non-causal MDLM training")
        if config.model.auxiliary_head_layers:
            raise ValueError("model.target_length_conditioning cannot be coupled with auxiliary heads")
        if config.model.output_factorization != "flat":
            raise ValueError("model.target_length_conditioning requires model.output_factorization='flat'")
        if config.data.boundary_aware_packing:
            raise ValueError("model.target_length_conditioning requires data.boundary_aware_packing=false")
        if config.data.content_budget_segmented_packing:
            raise ValueError("model.target_length_conditioning requires data.content_budget_segmented_packing=false")
        if config.data.short_document_curriculum:
            raise ValueError("model.target_length_conditioning requires data.short_document_curriculum=false")
        if not config.data.append_eos:
            raise ValueError("model.target_length_conditioning requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("model.target_length_conditioning requires data.loss_on_eos=true")
        if config.noise.type != "mdlm":
            raise ValueError("model.target_length_conditioning requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError(
                "model.target_length_conditioning preserves strict MDLM and requires complement_masking=false"
            )
        if config.noise.suffix_masking:
            raise ValueError(
                "model.target_length_conditioning preserves independent MDLM masking and requires suffix_masking=false"
            )
        if config.noise.time_conditioning:
            raise ValueError("model.target_length_conditioning preserves no time conditioning")
    if config.model.hierarchical_frequency_batches <= 0:
        raise ValueError("model.hierarchical_frequency_batches must be positive")
    for layer_idx in config.model.auxiliary_head_layers:
        if layer_idx < 0 or layer_idx >= config.model.depth:
            raise ValueError("model.auxiliary_head_layers entries must be valid zero-based layer indices")
    if len(set(config.model.auxiliary_head_layers)) != len(config.model.auxiliary_head_layers):
        raise ValueError("model.auxiliary_head_layers must not contain duplicates")
    if config.noise.type not in ("mdlm", "uniform", "mask_uniform", "bd3lm"):
        raise ValueError("noise.type must be one of 'mdlm', 'uniform', 'mask_uniform', or 'bd3lm'")
    if config.noise.t_min < 0 or config.noise.t_max > 1 or config.noise.t_min > config.noise.t_max:
        raise ValueError("noise.t_min and noise.t_max must satisfy 0 <= t_min <= t_max <= 1")
    if config.noise.type == "mdlm" and config.noise.schedule_weight and config.noise.t_max == 0:
        raise ValueError("schedule-weighted MDLM requires noise.t_max > 0")
    if config.noise.mask_uniform_mix_prob < 0 or config.noise.mask_uniform_mix_prob > 1:
        raise ValueError("noise.mask_uniform_mix_prob must be in [0, 1]")
    if config.noise.type == "mdlm" and config.noise.time_conditioning:
        raise ValueError("noise.time_conditioning is only allowed for non-MDLM corruption kernels")
    if config.noise.suffix_masking and config.noise.type != "mdlm":
        raise ValueError("noise.suffix_masking is only supported for noise.type='mdlm'")
    if not isinstance(config.noise.per_token_noise_levels, bool):
        raise ValueError("noise.per_token_noise_levels must be a boolean")
    if config.noise.per_token_noise_levels:
        if config.noise.type != "mdlm":
            raise ValueError("noise.per_token_noise_levels requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("noise.per_token_noise_levels requires complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("noise.per_token_noise_levels requires suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("noise.per_token_noise_levels preserves no time conditioning")
        if not config.noise.schedule_weight:
            raise ValueError("noise.per_token_noise_levels requires schedule_weight=true")
    if config.noise.type == "bd3lm":
        if config.noise.time_conditioning:
            raise ValueError("noise.time_conditioning is not supported for BD3LM")
        if config.model.bd3lm_block_size <= 0:
            raise ValueError("model.bd3lm_block_size must be positive for noise.type='bd3lm'")
        if config.model.bd3lm_block_size > config.model.max_seq_len:
            raise ValueError("model.bd3lm_block_size must not exceed model.max_seq_len")
    elif config.model.bd3lm_block_size < 0:
        raise ValueError("model.bd3lm_block_size must be non-negative")
    if config.model.block_causal_pilot_size < 0:
        raise ValueError("model.block_causal_pilot_size must be non-negative")
    if config.model.block_causal_pilot_size > 0:
        if config.noise.type != "mdlm":
            raise ValueError("model.block_causal_pilot_size > 0 requires noise.type='mdlm'")
        if config.model.causal:
            raise ValueError("model.block_causal_pilot_size > 0 requires model.causal=false")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("model.block_causal_pilot_size cannot be combined with model.bd3lm_block_size")
        if config.noise.suffix_masking:
            raise ValueError("model.block_causal_pilot_size > 0 requires noise.suffix_masking=false")
        if config.noise.complement_masking:
            raise ValueError("model.block_causal_pilot_size > 0 requires noise.complement_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("model.block_causal_pilot_size > 0 requires noise.time_conditioning=false")
        if config.model.block_causal_pilot_size > config.model.max_seq_len:
            raise ValueError("model.block_causal_pilot_size must not exceed model.max_seq_len")
    if config.noise.uniform_t_block_size <= 0:
        raise ValueError("noise.uniform_t_block_size must be positive")
    if config.noise.uniform_t_coarse_block_size <= 0:
        raise ValueError("noise.uniform_t_coarse_block_size must be positive")
    if not isinstance(config.noise.soft_input_curriculum, bool):
        raise ValueError("noise.soft_input_curriculum must be a boolean")
    if (
        config.noise.soft_input_curriculum_start_mix < 0
        or config.noise.soft_input_curriculum_start_mix > 1
        or config.noise.soft_input_curriculum_end_mix < 0
        or config.noise.soft_input_curriculum_end_mix > 1
    ):
        raise ValueError("noise.soft_input_curriculum mix values must be in [0, 1]")
    if config.noise.soft_input_curriculum_start_mix < config.noise.soft_input_curriculum_end_mix:
        raise ValueError("noise.soft_input_curriculum_start_mix must be >= end_mix for soft-to-hard annealing")
    if (
        config.noise.soft_input_curriculum_anneal_fraction <= 0
        or config.noise.soft_input_curriculum_anneal_fraction > 1
    ):
        raise ValueError("noise.soft_input_curriculum_anneal_fraction must be in (0, 1]")
    if config.noise.soft_input_curriculum:
        if config.noise.type != "uniform":
            raise ValueError("noise.soft_input_curriculum requires noise.type='uniform'")
        if not config.noise.time_conditioning:
            raise ValueError("noise.soft_input_curriculum requires noise.time_conditioning=true")
        if config.model.causal:
            raise ValueError("noise.soft_input_curriculum requires model.causal=false")
    if config.optim.type not in ("adamw", "muon"):
        raise ValueError("optim.type must be one of 'adamw' or 'muon'")
    if config.optim.lr <= 0:
        raise ValueError("optim.lr must be positive")
    if config.optim.muon_lr != -1.0 and config.optim.muon_lr <= 0:
        raise ValueError("optim.muon_lr must be positive or -1 to inherit optim.lr")
    if config.optim.adamw_lr != -1.0 and config.optim.adamw_lr <= 0:
        raise ValueError("optim.adamw_lr must be positive or -1 to inherit optim.lr")
    if config.optim.adam_eps <= 0:
        raise ValueError("optim.adam_eps must be positive")
    if config.recipe.status not in ("unspecified", "historical", "provisional", "locked"):
        raise ValueError("recipe.status must be one of 'unspecified', 'historical', 'provisional', or 'locked'")
    if config.recipe.status != "unspecified" and not config.recipe.name:
        raise ValueError("recipe.name is required when recipe.status is not 'unspecified'")
    for field_name, paths in (
        ("locked_fields", config.recipe.locked_fields),
        ("provisional_fields", config.recipe.provisional_fields),
    ):
        if not isinstance(paths, list) or any(not isinstance(path, str) or not path for path in paths):
            raise ValueError(f"recipe.{field_name} must be a list of non-empty dotted field paths")
        if len(set(paths)) != len(paths):
            raise ValueError(f"recipe.{field_name} must not contain duplicates")
    overlap = set(config.recipe.locked_fields) & set(config.recipe.provisional_fields)
    if overlap:
        raise ValueError(f"recipe locked_fields and provisional_fields overlap: {sorted(overlap)}")
    if config.recipe.status == "locked" and config.recipe.provisional_fields:
        raise ValueError("recipe.status='locked' cannot declare provisional_fields")
    if config.recipe.status == "provisional" and not config.recipe.provisional_fields:
        raise ValueError("recipe.status='provisional' requires at least one provisional field")
    config_tree = asdict(config)
    for dotted_path in config.recipe.locked_fields + config.recipe.provisional_fields:
        node: Any = config_tree
        for part in dotted_path.split("."):
            if not isinstance(node, dict) or part not in node or part == "recipe":
                raise ValueError(f"recipe.provisional_fields contains unknown config path: {dotted_path}")
            node = node[part]
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
    if config.train.log_interval <= 0:
        raise ValueError("train.log_interval must be positive")
    if config.train.tensorboard_dir and not config.train.tensorboard:
        raise ValueError("train.tensorboard_dir requires train.tensorboard=true")
    if config.train.save_every == 0:
        raise ValueError("train.save_every must be positive or -1")
    if config.train.eval_every == 0:
        raise ValueError("train.eval_every must be positive or -1")
    if config.train.eval_only and not config.train.resume_from and not config.train.comparison_run_ids:
        raise ValueError("train.eval_only requires train.resume_from or train.comparison_run_ids")
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
            if len(config.train.comparison_run_ids) != 1:
                raise ValueError(
                    "teacher-forced replay diagnostic requires exactly one comparison run "
                    "providing a compatible concentration metric"
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
    if config.eval.nelbo_tokens <= 0:
        raise ValueError("eval.nelbo_tokens must be positive")
    if not 0.0 <= config.eval.nelbo_t_min < config.eval.nelbo_t_max <= 1.0:
        raise ValueError("eval NELBO times must satisfy 0 <= nelbo_t_min < nelbo_t_max <= 1")
    if config.eval.nelbo_seed < 0:
        raise ValueError("eval.nelbo_seed must be non-negative")
    if config.eval.nelbo_time_sampling != "stratified_per_sequence":
        raise ValueError("eval.nelbo_time_sampling must be 'stratified_per_sequence'")
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
    if config.eval.sample_jitter_scale < 0:
        raise ValueError("eval.sample_jitter_scale must be non-negative")
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
    if config.eval.conditional_gen_eval and config.noise.type != "mdlm":
        raise ValueError("eval.conditional_gen_eval currently supports noise.type='mdlm'")
    if config.eval.conditional_gen_eval and config.eval.gen_score_mode == "judge":
        raise ValueError("eval.conditional_gen_eval supports eval.gen_score_mode 'none' or 'self'")
    if config.eval.gen_temperature <= 0:
        raise ValueError("eval.gen_temperature must be positive")
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
    if config.eval.sampler not in ("default", "remdm", "fhs", "bd3lm", "block_causal_pilot"):
        raise ValueError("eval.sampler must be one of 'default', 'remdm', 'fhs', 'bd3lm', 'block_causal_pilot'")
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
    if config.train.token_edit_corruption_batch_fraction < 0 or config.train.token_edit_corruption_batch_fraction > 1:
        raise ValueError("train.token_edit_corruption_batch_fraction must be in [0, 1]")
    if config.train.token_edit_corruption_ratio < 0 or config.train.token_edit_corruption_ratio > 1:
        raise ValueError("train.token_edit_corruption_ratio must be in [0, 1]")
    token_edit_enabled = (
        config.train.token_edit_corruption_batch_fraction > 0 or config.train.token_edit_corruption_ratio > 0
    )
    if token_edit_enabled:
        if config.train.token_edit_corruption_batch_fraction <= 0:
            raise ValueError("token-edit corruption requires a positive batch fraction")
        if config.train.token_edit_corruption_ratio <= 0:
            raise ValueError("token-edit corruption requires a positive token ratio")
        if config.model.causal:
            raise ValueError("token-edit corruption is only supported for non-causal MDLM training")
        if config.noise.type != "mdlm":
            raise ValueError("token-edit corruption requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("token-edit corruption preserves strict MDLM and requires complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError(
                "token-edit corruption preserves independent MDLM masking and requires suffix_masking=false"
            )
        if config.noise.time_conditioning:
            raise ValueError("token-edit corruption preserves no time conditioning")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("token-edit corruption cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("token-edit corruption cannot be coupled with block-causal pilot training")
        if config.model.output_factorization != "flat":
            raise ValueError("token-edit corruption requires model.output_factorization='flat'")
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


def parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def set_by_dotted_key(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = data
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    data = to_dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override!r}")
        key, raw_value = override.split("=", 1)
        set_by_dotted_key(data, key, parse_value(raw_value))
    return config_from_dict(data)
