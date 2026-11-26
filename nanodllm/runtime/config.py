import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config_validation import validate_config
from .mdlm_auxiliary_config import MDLMAuxiliaryConfig


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
    # Historical BD3 uses block-causal clean states. AR-to-block conversion can
    # instead retain a strictly causal clean stream while keeping the noisy
    # active block bidirectional.
    bd3lm_clean_stream_mode: str = "block_causal"
    bd3lm_self_conditioning_dim: int = 0
    block_causal_pilot_size: int = 0
    output_factorization: str = "flat"
    hierarchical_frequency_batches: int = 64
    causal: bool = False
    residual_output_projection_init: str = "zero"
    embedding_head_init_std_multiplier: float = 1.0
    length_head: bool = False
    target_length_conditioning: bool = False
    mask_position_embedding: bool = False
    # Optional modeling paths for self-generated M2T/T2T experiments. Zero
    # adapter dimensions and a disabled provenance embedding preserve the
    # historical model architecture and checkpoint state exactly.
    token_edit_adapter_bottleneck_dim: int = 0
    token_edit_adapter_top_layers: int = 0
    token_edit_provenance_embedding: bool = False
    token_edit_provenance_confidence_bins: int = 0
    # A versioned opt-in bundle; omitted fields retain historical checkpoints.
    backbone_profile: str = "nanodllm"
    mtp_mode: str = "none"
    recurrent_core_start: int = 0
    recurrent_core_length: int = 0
    recurrent_iterations: int = 1


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
    bos_packing_mode: str = "none"
    bos_packing_buffer_size: int = 1000
    eos_mask_tail_mode: str = "none"
    eos_mask_tail_size: int = 16
    short_document_curriculum: bool = False
    short_document_max_tokens: int = 256
    identity_mode: str = "off"
    dataset_snapshot: str = ""
    dataset_snapshot_sha256: str = ""


@dataclass
class NoiseConfig:
    type: str = "mdlm"
    t_min: float = 0.05
    t_max: float = 0.95
    schedule_weight: bool = True
    mdlm_time_sampling: str = "legacy_auto"
    mdlm_unweighted_loss_scale: float = 1.0
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
    lr_decay_style: str = "legacy_linear"
    grad_clip: float = 1.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"
    # Pinned nanochat update kernel, distinct from torch.optim.Muon. Historical
    # momentum/decay defaults stay constant; time profiles are explicit opt-ins.
    nanochat_checkout: str = ""
    nanochat_embedding_lr: float = 0.18371173070873836
    nanochat_head_lr: float = 0.004898979485566356
    nanochat_embedding_weight_decay: float = 0.001
    nanochat_head_weight_decay: float = 0.01
    nanochat_muon_beta2: float = 0.9
    nanochat_matrix_optimizer: str = "muon"
    nanochat_execution_scope: str = "single_rank"
    nanochat_momentum_schedule: str = "constant"
    nanochat_weight_decay_schedule: str = "constant"


@dataclass
class ParameterizationConfig:
    """Base-shape coordinates and the derived parameterization identity."""

    mode: str = "ordinary"
    # Execution admission is separate from scientific authorization. Ordinary
    # preserves the compatibility path; nonordinary is canary-only for now.
    execution_scope: str = "ordinary"
    base_depth: int = 4
    base_width: int = 256
    # A tunable root scale for nonzero Transformer-backbone matrix
    # initialization. Parameterization rules derive target-width scales from
    # this value; the default preserves the D4 ordinary initialization.
    hidden_init_std_multiplier: float = 1.0
    # Compatibility campaigns retain empty binding and v1/v2 initialization;
    # full provenance and detailed role telemetry require an explicit opt-in.
    contract_binding: str = "legacy_unbound"
    # Derived from the instantiated named-parameter contract before artifacts,
    # optimizer construction, or exact-resume admission. Empty means unbound.
    contract_sha256: str = ""


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
    mtp_loss_weight: float = 0.0
    recurrent_depth_schedule: str = "fixed"
    mdlm_auxiliary: MDLMAuxiliaryConfig = field(default_factory=MDLMAuxiliaryConfig)
    run_dir: str = "runs/full_scale_masked"
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
    telemetry_interval_seconds: float = 0.0
    resume_from: str = ""
    model_init_from: str = ""
    model_init_expected_sha256: str = ""
    model_init_expected_size_bytes: int = -1
    model_init_allow_token_edit_extensions: bool = False
    model_init_allow_block_self_conditioning: bool = False
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
    # Historical runs select whole device microbatches on a deterministic
    # cadence. The fixed row layout is a bounded DB16/GA1 utilization probe.
    token_edit_batch_layout: str = "whole_microbatch_cadence"
    token_edit_corruption_schedule: str = "legacy_fixed_ratio"
    token_edit_corruption_ratio: float = 0.0
    token_edit_input_source: str = "random_replacement"
    # Self-generated drafts normally come from the live student. A frozen
    # checkpoint source is an explicit teacher-draft intervention and must bind
    # the exact checkpoint bytes below.
    token_edit_draft_model_source: str = "live_student"
    token_edit_draft_checkpoint: str = ""
    token_edit_draft_checkpoint_expected_sha256: str = ""
    token_edit_draft_checkpoint_expected_size_bytes: int = -1
    token_edit_correct_draft_loss_weight: float = 0.0
    token_edit_wrong_draft_loss_weight: float = 1.0
    # The default retains historical weighted cross-entropy. Reward and
    # preference modes are restricted to self-generated full-draft screens.
    token_edit_correction_objective: str = "cross_entropy"
    token_edit_correction_aux_weight: float = 0.0
    # Generalized expected-accuracy loss: (1 - p(target)^q) / q. The exact
    # q=1 branch retains the historical 1-p(target) objective.
    token_edit_expected_accuracy_power: float = 1.0
    # ``corruption_threshold`` applies the configured power only to T2T rows
    # whose sampled MDLM corruption probability is below the threshold; rows
    # at or above it retain the exact historical q=1 objective.
    token_edit_expected_accuracy_power_schedule: str = "constant"
    token_edit_expected_accuracy_corruption_threshold: float = 0.8
    # Preserve initially correct masked-view predictions when the visible-draft
    # correction view would otherwise reduce their detached target probability.
    token_edit_cross_view_nonregression_weight: float = 0.0
    # The correction weight is exactly one minus this draft-pass MDLM weight.
    token_edit_draft_m2t_loss_weight: float = 0.0
    # Optional row-local credit assignment for the differentiable draft-M2T
    # view. Expected-error rewards are detached from the correction graph;
    # the shuffled control preserves each row's exact credit multiset.
    token_edit_draft_m2t_credit_mode: str = "uniform"
    token_edit_draft_m2t_credit_strength: float = 1.0
    token_edit_draft_m2t_credit_clip_min: float = 0.5
    token_edit_draft_m2t_credit_clip_max: float = 1.5
    # ``output_head_only`` detaches the T2T correction-pass final hidden state;
    # ``residual_adapter`` restricts T2T updates to configured upper adapters.
    # M2T microbatches always retain the ordinary full-model route.
    token_edit_backbone_gradient_mode: str = "shared"
    # ``mixed_mask_draft`` leaves a random fixed fraction of initially masked
    # targets unresolved. The two trajectory modes instead expose the remaining
    # targets to margin-ranked partial rollouts with model or matched gold commits.
    token_edit_canvas_mode: str = "full_draft"
    token_edit_mixed_canvas_draft_fraction: float = 1.0
    token_edit_trajectory_completion_fraction: float = 0.5
    token_edit_trajectory_steps: int = 4
    token_edit_full_draft_provenance: bool = False
    # Optional packed-segment prefix intervention for self-generated T2T. The
    # dedicated boundary seed is stateless and never consumes the training RNG.
    token_edit_prefix_enabled: bool = False
    token_edit_prefix_canvas_mode: str = "fallible_draft"
    token_edit_prefix_loss_scope: str = "all_active"
    token_edit_prefix_min_fraction: float = 0.125
    token_edit_prefix_max_fraction: float = 0.75
    token_edit_prefix_boundary_seed: int = 271828
    # Opt-in AR-to-block conversion objective. The uniform fraction selects a
    # deterministic whole-microbatch T2T cadence; remaining microbatches use
    # absorbing-mask M2T. The AR term is computed from the causal clean half of
    # the same doubled forward and is separately token-normalized.
    bd3lm_uniform_t2t_batch_fraction: float = 0.0
    bd3lm_causal_ar_loss_weight: float = 0.0
    # Replace this fraction of M2T microbatches with all-lossable MASK targets.
    bd3lm_full_mask_batch_fraction: float = 0.0
    # Causal-clean conversion may use an explicit optimizer-step curriculum for
    # the active training block length. Starts are zero-based inclusive optimizer
    # step indices. Empty index/value lists preserve the historical fixed length.
    bd3lm_block_size_schedule: str = "fixed"
    bd3lm_block_size_schedule_start_step_indices: list[int] = field(default_factory=list)
    bd3lm_block_size_schedule_values: list[int] = field(default_factory=list)
    bd3lm_partition_mode: str = "fixed"
    bd3lm_partition_lengths: list[int] = field(default_factory=list)
    bd3lm_partition_seed: int = 0
    # One-factor block-recipe screens; disabled leaves the historical route unchanged.
    bd3lm_recipe_mechanism: str = "none"
    bd3lm_recipe_start_step: int = 0
    bd3lm_recipe_teacher_path: str = ""
    bd3lm_recipe_teacher_sha256: str = ""
    bd3lm_recipe_teacher_size_bytes: int = 0
    bd3lm_recipe_teacher_weight: float = 0.5
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
    token_edit_transition_eval: bool = False
    token_edit_transition_batches: int = 2
    token_edit_transition_frequency_batches: int = 64
    token_edit_transition_mask_fractions: list[float] = field(default_factory=lambda: [0.7, 0.9, 1.0])
    token_edit_transition_seed: int = 271828
    token_edit_transition_log_samples: int = 3
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
    eval_only_temperature_sweep: bool = True
    length_policy: str = "none"
    gen_prior_correction_strength: float = 0.0
    gen_prior_correction_batches: int = 0
    gen_prior_correction_smoothing: float = 1.0
    sampler: str = "default"
    sample_top_k: int = 0
    sample_top_p: float = 1.0
    fhs_tau: float = 0.9
    remdm_tau_edit: float = 0.9
    remasking_comparison: bool = False
    basin_diagnostic: bool = False
    basin_forced_argmax_depths: list[int] = field(default_factory=lambda: [0, 1, 2, 4])
    basin_prefix_len: int = 16
    basin_nucleus_top_p: float = 0.95
    target_length_distribution_batches: int = 8


@dataclass
class DiagnosticsConfig:
    """Resource budgets for optional training audits and canaries."""

    eos_supervision_prepack_document_batches: int = 8
    eos_supervision_packed_batches: int = 64
    eos_supervision_forced_mask_t: float = 1.0
    real_packing_audit_batches: int = 64
    real_packing_target_ce_fixed_mask_t: float = 1.0
    real_packing_target_ce_example_limit: int = 8
    # Optional optimizer-step state receipts for short mechanics canaries.
    parameter_state_hash_steps: list[int] = field(default_factory=list)


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
    parameterization: ParameterizationConfig = field(default_factory=ParameterizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    recipe: RecipeMetadataConfig = field(default_factory=RecipeMetadataConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


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
        parameterization=ParameterizationConfig(**merged["parameterization"]),
        runtime=RuntimeConfig(**merged["runtime"]),
        train=TrainConfig(
            **{**merged["train"], "mdlm_auxiliary": MDLMAuxiliaryConfig(**merged["train"]["mdlm_auxiliary"])}
        ),
        eval=EvalConfig(**merged["eval"]),
        diagnostics=DiagnosticsConfig(**merged["diagnostics"]),
        recipe=RecipeMetadataConfig(**merged["recipe"]),
    )
    validate_config(cfg)
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        return config_from_dict(json.load(f))


def to_dict(config: ExperimentConfig) -> dict[str, Any]:
    from .data_snapshot import strip_disabled_snapshot
    from .telemetry_config import strip_disabled_telemetry

    value = asdict(config)
    strip_disabled_snapshot(value)
    strip_disabled_telemetry(value)
    if config.data.bos_packing_mode == "none":
        from .data_bos_packing_config import strip_disabled_bos_packing

        strip_disabled_bos_packing(value)
    if config.train.recurrent_depth_schedule == "fixed":
        value["train"].pop("recurrent_depth_schedule")
    from .nanochat_muon_schedule import strip_constant_schedules

    strip_constant_schedules(value)
    if config.optim.type != "nanochat_muon":
        from .config_validation_optim import strip_disabled_nanochat_muon

        strip_disabled_nanochat_muon(value)
    elif config.optim.nanochat_matrix_optimizer == "muon":
        # Preserve the serialized identity of the already executed calibration.
        value["optim"].pop("nanochat_matrix_optimizer")
    if config.optim.nanochat_execution_scope == "single_rank":
        value["optim"].pop("nanochat_execution_scope", None)
    return value


def write_config(path: str | Path, config: ExperimentConfig) -> None:
    with open(path, "w") as f:
        json.dump(to_dict(config), f, indent=2)


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


def _parse_list_override(raw: str) -> list[Any] | None:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(raw, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, list) else None


def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    data = to_dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override!r}")
        key, raw_value = override.split("=", 1)
        current: Any = data
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        value = parse_value(raw_value)
        if isinstance(current, list) and raw_value.lstrip().startswith("["):
            decoded = _parse_list_override(raw_value)
            if decoded is not None:
                value = decoded
        set_by_dotted_key(data, key, value)
    return config_from_dict(data)
