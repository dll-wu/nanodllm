from __future__ import annotations

import gc
import json
import os
import random
import time
from contextlib import ExitStack, nullcontext
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist

from .config import ExperimentConfig, write_config
from .data import DLMDataBatch, build_frequency_token_classes, build_token_bytes, build_tokenizer, build_train_loader
from .lr_schedule import _lr_schedule_telemetry, _resolved_warmup_steps
from .model import DiffusionTransformer
from .noise import build_noise
from .parameter_state_hash import (
    PARAMETER_STATE_HASH_METHOD,
    PARAMETER_STATE_HASH_SCHEMA,
    resolve_parameter_state_hash_steps,
)
from .parameterization import bind_parameterization_contract, resolve_parameterization_contract
from .telemetry import append_jsonl, close_tensorboard_writer, write_json_atomic
from .telemetry import build_tensorboard_writer as _build_tensorboard_writer
from .telemetry import resolve_tensorboard_dir as _resolve_tensorboard_dir
from .telemetry import write_tensorboard_scalars as _write_tensorboard_scalars
from .training_artifacts import (
    _pre_update_parameter_state_artifact,
    _prepare_data_identity,
    _publish_final_outputs,
    _tokenizer_summary,
    source_promotion_metrics,
    validate_existing_initial_state_artifact,
    write_initial_state_artifact_once,
)
from .training_bd3lm import bd3lm_training_flops_per_source_token
from .training_checkpoint import (
    RESUME_BATCH_DIGEST_VERSION,
    RESUME_BATCH_TENSOR_FIELDS,
    RESUME_CONTRACT_VERSION,
    _collect_resume_contract,
    _load_checkpoint,
    _packed_all_reduce_int_counters,
    _resolve_resume_path,
    _resume_trajectory_contract,
    _synchronize_resume_phase_error,
    _verify_replayed_next_batch,
)
from .training_compute import resolve_training_model_accounting
from .training_evaluation import EvaluationRuntime, _run_eval_with_runtime
from .training_features import (
    _ddp_broadcast_buffers,
    _eos_supervision_audit_enabled,
    _token_edit_detaches_backbone,
    _token_edit_probability_bounds,
    _token_edit_support_policy,
    _token_edit_uses_residual_adapter,
    _uses_bd3lm,
    _uses_differentiable_self_generated_draft,
    _uses_fixed_row_token_edit_layout,
    _uses_frozen_token_edit_draft_model,
    _uses_hierarchical_output,
    _uses_matched_token_edit_schedule,
    _uses_per_token_noise_levels,
    _uses_self_generated_token_edit,
    _uses_token_edit_corruption,
    _uses_token_edit_prefix,
    _uses_trajectory_token_edit_canvas,
    _uses_uniform_low_variance_estimator,
)
from .training_frozen_draft import load_frozen_token_edit_draft_model
from .training_initialization import initialize_from_parent
from .training_metrics import (
    _embedding_head_init_diagnostics_enabled,
    _embedding_head_init_metrics,
    _evaluate_length_head_metrics,
    _gradient_component_norm_metrics,
    _lm_head_row_norm_metrics,
    _parameter_norm_metrics,
    _prefixed_metric_keys,
    _residual_projection_init_metrics,
)
from .training_monitor import initialize_monitor, memory_snapshot
from .training_objective import (
    _assert_clean_label_smoothing_ar_config,
    _bd3lm_telemetry,
    _eos_supervision_count_metrics,
    _eos_supervision_startup_audit,
    _eos_supervision_training_invariant_failures,
    _hierarchical_output_telemetry,
    _new_eos_supervision_counts,
    _noise_telemetry,
    _ordinary_token_candidate_ids,
    _reduce_eos_supervision_counts,
)
from .training_optim import _set_optimizer_base_lrs, make_optimizer
from .training_plan import (
    _fixed_ce_trace_steps,
    _parameter_accounting_summary,
    _resolve_diffusion_config,
    _training_budget,
)
from .training_progress import (
    EOS_HAZARD_RELATIVE_POSITION_DECILES,
    PER_TOKEN_NOISE_LEVEL_HIST_BINS,
    POSITION_DECILE_BALANCED_LOSS_DECILES,
    TOKEN_EDIT_CORRUPTION_DECILES,
    TrainerProgress,
)
from .training_report_inputs import FinalReportInputs
from .training_reporting import build_final_metrics_v1
from .training_rng import _restore_active_rank_rng_state, _restore_rng_state, _rng_state
from .training_runtime import (
    _check_gpu_preflight,
    _configure_numeric_runtime,
    _cpu_threading_telemetry,
    _destroy_process_group_if_initialized,
    _distributed_cuda_memory_telemetry,
    _restore_gc_state,
    dtype_from_name,
    init_distributed,
    print0,
    validate_training_entrypoint,
)
from .training_runtime import (
    run_requested_training_diagnostic as _run_requested_training_diagnostic,
)
from .training_state import TrainingSession, _current_trainer_state
from .training_step import _run_training_steps
from .training_token_edit_trajectory import trajectory_draft_forwards_per_microbatch


def _bootstrap_training_session(config: ExperimentConfig, resources: ExitStack) -> TrainingSession:
    validate_training_entrypoint(config)
    session = TrainingSession(config, resources)
    os.makedirs(config.train.run_dir, exist_ok=True)

    _configure_numeric_runtime(config)

    session.is_ddp, session.rank, session.local_rank, session.world_size, session.device = init_distributed(
        config.runtime.device_type,
        config.train.run_dir,
        init_single_rank_process_group=config.runtime.init_single_rank_process_group,
    )
    if session.is_ddp and (_token_edit_detaches_backbone(config) or _token_edit_uses_residual_adapter(config)):
        raise RuntimeError("isolated token-edit gradients currently require single-process training")

    _check_gpu_preflight(config, session.rank)

    session.compute_dtype = dtype_from_name(config.runtime.dtype)

    session.autocast_ctx = (
        torch.autocast(device_type=config.runtime.device_type, dtype=session.compute_dtype)
        if config.runtime.device_type == "cuda" and session.compute_dtype != torch.float32
        else nullcontext()
    )

    session.training_seed = config.runtime.seed if config.runtime.training_seed < 0 else config.runtime.training_seed

    session.model_seed = config.runtime.seed if config.runtime.model_seed < 0 else config.runtime.model_seed

    torch.manual_seed(session.training_seed + session.rank)

    np.random.seed(session.training_seed + session.rank)

    random.seed(session.training_seed + session.rank)

    if config.runtime.device_type == "cuda":
        torch.cuda.manual_seed(session.training_seed + session.rank)

    session.tokenizer = build_tokenizer(config)

    session.clean_label_smoothing_metrics = _assert_clean_label_smoothing_ar_config(config, session.tokenizer)

    return session


def _token_edit_prefix_run_summary_fragment(config: ExperimentConfig) -> dict[str, object]:
    """Return enabled-only prefix metadata without changing historical summaries."""

    if not _uses_token_edit_prefix(config):
        return {}
    return {
        "prefix": {
            "enabled": True,
            "canvas_mode": config.train.token_edit_prefix_canvas_mode,
            "loss_scope": config.train.token_edit_prefix_loss_scope,
            "min_fraction": float(config.train.token_edit_prefix_min_fraction),
            "max_fraction": float(config.train.token_edit_prefix_max_fraction),
            "boundary_seed": int(config.train.token_edit_prefix_boundary_seed),
            "boundary_sampling": "sha256_keyed_per_attention_segment_v1",
            "receipt_pattern": "telemetry/token_edit_prefix_plans.rank{rank}.jsonl",
        }
    }


def _initialize_standard_training_session(session: TrainingSession) -> TrainingSession:
    config = session.config

    session.data_identity_summary = _prepare_data_identity(
        config, rank=session.rank, world_size=session.world_size, is_ddp=session.is_ddp
    )

    session.training_token_bytes = build_token_bytes(session.tokenizer).to(session.device)

    session.generation_blocked_token_ids = {
        int(token_id) for token_id in session.tokenizer.get_generation_blocked_token_ids()
    }

    session.token_edit_candidate_ids = None

    if _uses_matched_token_edit_schedule(config) and not _uses_self_generated_token_edit(config):
        excluded_token_ids = {int(token_id) for token_id in session.tokenizer.get_special_token_ids()}
        excluded_token_ids.add(int(session.tokenizer.get_mask_token_id()))
        pad_token_id = session.tokenizer.get_pad_token_id()
        if pad_token_id is not None:
            excluded_token_ids.add(int(pad_token_id))
        session.token_edit_candidate_ids = _ordinary_token_candidate_ids(
            session.training_token_bytes,
            excluded_token_ids,
        )

    session.token_class_ids = None

    session.token_class_metadata: dict | None = None

    session.token_edit_class_ids = None

    session.token_edit_class_metadata: dict | None = None

    if _uses_hierarchical_output(config):
        session.saved_state = _rng_state(config)
        session.token_class_ids, session.token_class_metadata = build_frequency_token_classes(
            config.data,
            session.tokenizer,
            int(config.train.device_batch_size),
            int(config.model.max_seq_len),
            session.device,
            int(config.model.hierarchical_frequency_batches),
        )
        _restore_rng_state(session.saved_state, config)
        session.token_edit_class_ids = session.token_class_ids
        session.token_edit_class_metadata = session.token_class_metadata
    elif _uses_token_edit_corruption(config) and not _uses_self_generated_token_edit(config):
        session.saved_state = _rng_state(config)
        session.token_edit_class_ids, session.token_edit_class_metadata = build_frequency_token_classes(
            config.data,
            session.tokenizer,
            int(config.train.device_batch_size),
            int(config.model.max_seq_len),
            session.device,
            int(config.model.hierarchical_frequency_batches),
        )
        _restore_rng_state(session.saved_state, config)

    session.model_config = _resolve_diffusion_config(config, session.tokenizer)

    session.saved_state = _rng_state(config)

    torch.manual_seed(session.model_seed)

    if config.runtime.device_type == "cuda":
        torch.cuda.manual_seed(session.model_seed)

    session.raw_model = DiffusionTransformer(
        session.model_config,
        rope_base=config.model.rope_base,
        compute_dtype=session.compute_dtype,
        pad_vocab_to=config.model.pad_vocab_to,
        token_class_ids=session.token_class_ids,
    ).to(session.device)

    _restore_rng_state(session.saved_state, config)

    initialize_from_parent(session)

    from .training_block_recipe_views import setup_block_recipe

    setup_block_recipe(session)
    resolve_training_model_accounting(session, bd3lm_training_flops_per_source_token if _uses_bd3lm(config) else None)

    session.saved_state = _rng_state(config)
    try:
        (
            session.raw_frozen_draft_model,
            session.frozen_draft_model_source_artifact,
        ) = load_frozen_token_edit_draft_model(
            config,
            model_config=session.model_config,
            device=session.device,
            compute_dtype=session.compute_dtype,
            token_class_ids=session.token_class_ids,
            rank=session.rank,
            world_size=session.world_size,
            student_architecture_fingerprint=session.architecture_fingerprint,
        )
    finally:
        _restore_rng_state(session.saved_state, config)

    session.parameter_accounting = _parameter_accounting_summary(
        config, session.counts, session.architecture_fingerprint["parameter_inventory"]
    )

    session.total_batch_size, session.grad_accum_steps, session.total_steps, session.batch_lr_scale = _training_budget(
        config, session.counts, session.world_size, raw_model=session.raw_model
    )
    if _uses_fixed_row_token_edit_layout(config) and (
        session.total_batch_size != 16 * config.model.max_seq_len or session.grad_accum_steps != 1
    ):
        raise RuntimeError("fixed 12-M2T/4-T2T row layout requires DB16/GA1 after distributed batch planning")

    session.parameterization_contract = resolve_parameterization_contract(
        session.raw_model,
        config,
        batch_lr_scale=session.batch_lr_scale,
    )
    bind_parameterization_contract(config, session.parameterization_contract)
    session.parameter_state_hash_steps = resolve_parameter_state_hash_steps(
        config.diagnostics.parameter_state_hash_steps,
        total_steps=session.total_steps,
    )
    session.parameter_state_hash_path = os.path.join(
        config.train.run_dir,
        "telemetry",
        "parameter_state_hashes.jsonl",
    )

    if session.is_ddp:
        initial_state_receipt: list[object | None] = [None]
        if session.rank == 0:
            try:
                initial_state_receipt[0] = {
                    "status": "ok",
                    "artifact": _pre_update_parameter_state_artifact(
                        session.raw_model,
                        architecture_fingerprint_sha256=str(session.architecture_fingerprint["fingerprint_sha256"]),
                        parameterization_contract_sha256=config.parameterization.contract_sha256,
                        model_seed=session.model_seed,
                        model_initialization_source=session.model_initialization_source,
                    ),
                }
            except Exception as exc:  # noqa: BLE001 - peers must receive a terminal artifact result.
                initial_state_receipt[0] = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
        dist.broadcast_object_list(initial_state_receipt, src=0)
        receipt = initial_state_receipt[0]
        if not isinstance(receipt, dict) or receipt.get("status") != "ok":
            raise RuntimeError(f"rank-zero initial-state artifact construction failed: {receipt}")
        session.initial_state_artifact = receipt.get("artifact")
    else:
        session.initial_state_artifact = _pre_update_parameter_state_artifact(
            session.raw_model,
            architecture_fingerprint_sha256=str(session.architecture_fingerprint["fingerprint_sha256"]),
            parameterization_contract_sha256=config.parameterization.contract_sha256,
            model_seed=session.model_seed,
            model_initialization_source=session.model_initialization_source,
        )

    if not isinstance(session.initial_state_artifact, dict):
        raise RuntimeError("rank 0 did not provide a valid pre-update parameter-state artifact")

    session.total_budget_tokens = session.total_steps * session.total_batch_size

    session.warmup_steps = _resolved_warmup_steps(config, session.total_steps)

    session.lr_schedule = _lr_schedule_telemetry(config, session.total_steps, session.warmup_steps)

    session.trajectory_contract = _resume_trajectory_contract(
        config,
        device=session.device,
        model_config=asdict(session.model_config),
        architecture_fingerprint=session.architecture_fingerprint,
        data_identity_summary=session.data_identity_summary,
        total_batch_size=session.total_batch_size,
        grad_accum_steps=session.grad_accum_steps,
        total_steps=session.total_steps,
        warmup_steps=session.warmup_steps,
        lr_schedule=session.lr_schedule,
    )

    session.cpu_threading = _cpu_threading_telemetry()

    session.record_embedding_head_init_diagnostics = _embedding_head_init_diagnostics_enabled(config)

    print0(
        session.rank,
        json.dumps(
            {
                "model": asdict(session.model_config),
                "param_counts": session.counts,
                "parameter_accounting": session.parameter_accounting,
                "parameter_state_hash_telemetry": {
                    "enabled": bool(session.parameter_state_hash_steps),
                    "steps": sorted(session.parameter_state_hash_steps),
                    "artifact": "telemetry/parameter_state_hashes.jsonl" if session.parameter_state_hash_steps else "",
                    "hash_method": PARAMETER_STATE_HASH_METHOD if session.parameter_state_hash_steps else "",
                    "state_point": "after_successful_optimizer_update_before_periodic_checkpoint",
                },
                "parameterization_contract_sha256": session.parameterization_contract["contract_sha256"],
                "flops_per_token": session.flops_per_token,
                "total_batch_size": session.total_batch_size,
                "effective_batch_size_tokens": session.total_batch_size,
                "grad_accum_steps": session.grad_accum_steps,
                "total_steps": session.total_steps,
                "target_tokens": config.train.target_tokens,
                "total_budget_tokens": session.total_budget_tokens,
                "world_size": session.world_size,
                "compute_dtype": str(session.compute_dtype),
                "lr_schedule": session.lr_schedule,
                "cpu_threading": session.cpu_threading,
            },
            indent=2,
        ),
    )

    token_edit_enabled = _uses_token_edit_corruption(config)
    matched_token_edit_schedule = _uses_matched_token_edit_schedule(config)
    self_generated_token_edit = _uses_self_generated_token_edit(config)
    token_edit_probability_min, token_edit_probability_max = _token_edit_probability_bounds(config)
    if session.rank == 0:
        validate_existing_initial_state_artifact(
            os.path.join(config.train.run_dir, "initial_state.v1.json"),
            session.initial_state_artifact,
        )
        write_config(os.path.join(config.train.run_dir, "resolved_config.json"), config)
        write_json_atomic(
            os.path.join(config.train.run_dir, "architecture_fingerprint.json"), session.architecture_fingerprint
        )
    session.optimizers = make_optimizer(session.raw_model, config, session.batch_lr_scale)

    _set_optimizer_base_lrs(session.optimizers, config, session.batch_lr_scale)

    if session.rank == 0:
        write_json_atomic(
            os.path.join(config.train.run_dir, "run_summary.json"),
            {
                "model": asdict(session.model_config),
                "architecture_fingerprint": session.architecture_fingerprint,
                "initial_state": session.initial_state_artifact,
                "parameter_accounting": session.parameter_accounting,
                "parameter_state_hash_telemetry": {
                    "enabled": bool(session.parameter_state_hash_steps),
                    "steps": sorted(session.parameter_state_hash_steps),
                    "artifact": "telemetry/parameter_state_hashes.jsonl" if session.parameter_state_hash_steps else "",
                    "schema": PARAMETER_STATE_HASH_SCHEMA if session.parameter_state_hash_steps else "",
                    "hash_method": PARAMETER_STATE_HASH_METHOD if session.parameter_state_hash_steps else "",
                    "state_point": "after_successful_optimizer_update_before_periodic_checkpoint",
                },
                **(
                    {"parameterization": session.parameterization_contract}
                    if config.parameterization.contract_binding == "required"
                    else {}
                ),
                "tokenizer": _tokenizer_summary(session.tokenizer),
                "noise": _noise_telemetry(config),
                "objective": _bd3lm_telemetry(config),
                "per_token_noise_levels": {
                    "enabled": _uses_per_token_noise_levels(config),
                    "histogram_bins": PER_TOKEN_NOISE_LEVEL_HIST_BINS,
                    "schedule_weight_source": "target_token_t" if _uses_per_token_noise_levels(config) else "",
                    "loss_normalization": "sum_ce_over_t_div_lossable_tokens"
                    if _uses_per_token_noise_levels(config)
                    else "",
                    "fixed_ce_eval_grid": "standard_scalar_t",
                },
                "confidence_penalty": {
                    "weight": float(config.train.confidence_penalty_weight),
                    "enabled": bool(config.train.confidence_penalty_weight > 0),
                },
                "z_loss": {
                    "weight": float(config.train.z_loss_weight),
                    "enabled": bool(config.train.z_loss_weight > 0),
                },
                "eos_hazard_calibration": {
                    "weight": float(config.train.eos_hazard_loss_weight),
                    "enabled": bool(config.train.eos_hazard_loss_weight > 0),
                    "relative_position_deciles": EOS_HAZARD_RELATIVE_POSITION_DECILES,
                },
                "length_head": {
                    "enabled": bool(config.model.length_head),
                    "loss_weight": float(config.train.length_head_loss_weight),
                    "length_policy": config.eval.length_policy,
                },
                "target_length_conditioning": {
                    "enabled": bool(config.model.target_length_conditioning),
                    "empirical_prior_batches": int(config.eval.target_length_distribution_batches),
                    "forced_eos_policy": config.eval.length_policy,
                },
                "data_identity": session.data_identity_summary,
                "data_packing": {
                    "boundary_aware_packing": bool(config.data.boundary_aware_packing),
                    "content_budget_segmented_packing": bool(config.data.content_budget_segmented_packing),
                    "short_document_curriculum": bool(config.data.short_document_curriculum),
                },
                "eos_supervision_audit": {
                    "enabled": _eos_supervision_audit_enabled(config),
                    "artifact": "telemetry/eos_supervision_audit.jsonl",
                    "prepack_document_batches": int(config.diagnostics.eos_supervision_prepack_document_batches),
                    "packed_batches": int(config.diagnostics.eos_supervision_packed_batches),
                    "forced_mask_t": float(config.diagnostics.eos_supervision_forced_mask_t),
                },
                "position_decile_balancing": {
                    "enabled": bool(config.train.position_decile_balanced_loss),
                    "relative_position_deciles": POSITION_DECILE_BALANCED_LOSS_DECILES,
                },
                "uniform_loss_estimator": {
                    "name": config.train.uniform_loss_estimator,
                    "enabled": _uses_uniform_low_variance_estimator(config),
                    "sample_count": 2 if _uses_uniform_low_variance_estimator(config) else 1,
                },
                "token_edit_corruption": {
                    "enabled": token_edit_enabled,
                    "batch_layout": config.train.token_edit_batch_layout,
                    "input_source": config.train.token_edit_input_source,
                    "draft_model_source": config.train.token_edit_draft_model_source,
                    "frozen_draft_model_source": session.frozen_draft_model_source_artifact,
                    "canvas_mode": config.train.token_edit_canvas_mode,
                    "mixed_canvas_draft_fraction": float(config.train.token_edit_mixed_canvas_draft_fraction),
                    "trajectory_completion_fraction": float(config.train.token_edit_trajectory_completion_fraction),
                    "trajectory_steps": int(config.train.token_edit_trajectory_steps),
                    "adapter_bottleneck_dim": int(config.model.token_edit_adapter_bottleneck_dim),
                    "adapter_top_layers": int(config.model.token_edit_adapter_top_layers),
                    "provenance_embedding": bool(config.model.token_edit_provenance_embedding),
                    **_token_edit_prefix_run_summary_fragment(config),
                    "batch_fraction": float(config.train.token_edit_corruption_batch_fraction),
                    "schedule": config.train.token_edit_corruption_schedule,
                    "ratio": float(config.train.token_edit_corruption_ratio),
                    "probability_min": token_edit_probability_min,
                    "probability_max": token_edit_probability_max,
                    "support_policy": _token_edit_support_policy(config),
                    "candidate_source": (
                        (
                            "frozen_checkpoint_model_blocked_argmax_on_mdlm_masked_positions"
                            if _uses_frozen_token_edit_draft_model(config)
                            else "live_model_blocked_argmax_on_mdlm_masked_positions"
                        )
                        if self_generated_token_edit
                        else (
                            "tokenizer_token_bytes_gt_zero_excluding_special_mask_pad_and_vocab_padding"
                            if matched_token_edit_schedule
                            else "full_model_vocabulary_excluding_mask_and_original"
                        )
                    )
                    if token_edit_enabled
                    else "disabled",
                    "candidate_token_count": (
                        max(0, int(session.model_config.vocab_size) - len(session.generation_blocked_token_ids))
                        if self_generated_token_edit
                        else (
                            int(session.token_edit_candidate_ids.numel())
                            if matched_token_edit_schedule and session.token_edit_candidate_ids is not None
                            else max(0, int(session.model_config.vocab_size) - 2)
                        )
                    )
                    if token_edit_enabled
                    else 0,
                    "target_scope": (
                        (
                            "initially_masked_lossable_positions_mixed_between_unresolved_masks_and_inserted_drafts"
                            if config.train.token_edit_canvas_mode == "mixed_mask_draft"
                            else (
                                "remaining_initially_masked_positions_after_partial_margin_ranked_rollout"
                                if config.train.token_edit_canvas_mode
                                in {"trajectory_gold_context", "trajectory_model_context"}
                                else (
                                    "initially_masked_suffix_lossable_positions_with_all_active_correction_logits"
                                    if _uses_token_edit_prefix(config)
                                    else "initially_masked_lossable_positions"
                                )
                            )
                        )
                        if self_generated_token_edit
                        else ("ordinary_gold_positions" if matched_token_edit_schedule else "all_lossable_positions")
                    )
                    if token_edit_enabled
                    else "disabled",
                    "empty_draw_policy": (
                        "mdlm_corruption_policy"
                        if self_generated_token_edit
                        else (
                            "force_one_ordinary_gold_edit" if matched_token_edit_schedule else "force_one_lossable_edit"
                        )
                    )
                    if token_edit_enabled
                    else "disabled",
                    "no_eligible_target_policy": (
                        "skip_empty_rank_with_ddp_graph_parity"
                        if self_generated_token_edit
                        else ("fail_closed" if matched_token_edit_schedule else "legacy_empty_microbatch")
                    )
                    if token_edit_enabled
                    else "disabled",
                    "loss_normalization": "disabled"
                    if not token_edit_enabled
                    else (
                        "inverse_per_row_edit_probability_over_suffix_lossable_tokens"
                        if _uses_token_edit_prefix(config)
                        else "inverse_per_row_edit_probability_over_all_lossable_tokens"
                        if config.train.token_edit_corruption_schedule == "match_mdlm_per_row"
                        else "inverse_fixed_edit_probability_over_all_lossable_tokens"
                    ),
                    "correct_draft_loss_weight": float(config.train.token_edit_correct_draft_loss_weight),
                    "wrong_draft_loss_weight": float(config.train.token_edit_wrong_draft_loss_weight),
                    **(
                        {
                            "correction_objective": config.train.token_edit_correction_objective,
                            "correction_aux_weight": float(config.train.token_edit_correction_aux_weight),
                            "expected_accuracy_power": float(config.train.token_edit_expected_accuracy_power),
                            "expected_accuracy_power_schedule": (
                                config.train.token_edit_expected_accuracy_power_schedule
                            ),
                            "expected_accuracy_corruption_threshold": float(
                                config.train.token_edit_expected_accuracy_corruption_threshold
                            ),
                            "cross_view_nonregression_weight": float(
                                config.train.token_edit_cross_view_nonregression_weight
                            ),
                        }
                        if config.train.token_edit_correction_objective != "cross_entropy"
                        else {}
                    ),
                    "draft_gradient_policy": (
                        "differentiable_mdlm_loss_with_detached_argmax_canvas"
                        if _uses_differentiable_self_generated_draft(config)
                        else "stopped_gradient"
                        if self_generated_token_edit
                        else "not_applicable"
                    ),
                    **(
                        {
                            "draft_m2t_loss_weight": float(config.train.token_edit_draft_m2t_loss_weight),
                            "correction_loss_weight": 1.0 - float(config.train.token_edit_draft_m2t_loss_weight),
                            "draft_m2t_credit": {
                                "mode": config.train.token_edit_draft_m2t_credit_mode,
                                "strength": float(config.train.token_edit_draft_m2t_credit_strength),
                                "clip_min": float(config.train.token_edit_draft_m2t_credit_clip_min),
                                "clip_max": float(config.train.token_edit_draft_m2t_credit_clip_max),
                                "signal": "detached_correction_expected_error",
                                "advantage": "row_mean_expected_error_minus_token_expected_error",
                                "normalization": "mean_one_per_active_row",
                                "shuffle": (
                                    "stateless_training_seed_global_microbatch_row"
                                    if config.train.token_edit_draft_m2t_credit_mode == "shuffled_expected_error_reward"
                                    else "disabled"
                                ),
                            },
                        }
                        if _uses_differentiable_self_generated_draft(config)
                        else {}
                    ),
                    "backbone_gradient_mode": config.train.token_edit_backbone_gradient_mode,
                    "draft_decoding": "generation_blocked_argmax" if self_generated_token_edit else "not_applicable",
                    "forward_calls_per_t2t_microbatch": (
                        1 + trajectory_draft_forwards_per_microbatch(config)
                        if _uses_trajectory_token_edit_canvas(config)
                        else 2
                        if self_generated_token_edit
                        else 1
                    ),
                    "fixed_m2t_rows_per_correction_forward": (12 if _uses_fixed_row_token_edit_layout(config) else 0),
                    "fixed_t2t_rows_per_correction_forward": (4 if _uses_fixed_row_token_edit_layout(config) else 0),
                    "compute_matching_note": (
                        "token-matched arms are not compute-matched because T2T uses a draft and correction forward"
                        if self_generated_token_edit
                        else ""
                    ),
                    "frequency_reconstruction_telemetry_enabled": bool(
                        token_edit_enabled and not self_generated_token_edit
                    ),
                    "frequency_reconstruction_prediction_policy": (
                        "raw_unblocked_argmax"
                        if token_edit_enabled and not self_generated_token_edit
                        else "not_applicable"
                    ),
                    "self_generated_transition_prediction_policy": (
                        "generation_blocked_argmax" if self_generated_token_edit else "not_applicable"
                    ),
                    "frequency_deciles": TOKEN_EDIT_CORRUPTION_DECILES,
                    "frequency_batches": int(session.token_edit_class_metadata.get("frequency_batches", 0))
                    if session.token_edit_class_metadata
                    else 0,
                    "frequency_observed_tokens": int(
                        session.token_edit_class_metadata.get("frequency_observed_tokens", 0)
                    )
                    if session.token_edit_class_metadata
                    else 0,
                },
                "residual_output_projection_init": _residual_projection_init_metrics(config),
                "embedding_head_init": _embedding_head_init_metrics(config),
                "embedding_head_init_diagnostics": {
                    "enabled": bool(session.record_embedding_head_init_diagnostics),
                    "artifact": "telemetry/embedding_head_init_diagnostics.jsonl",
                    "row_norm_frequency_deciles": 7,
                    "early_snapshot_step": 1,
                },
                "output_factorization": _hierarchical_output_telemetry(session.token_class_metadata),
                "param_counts": session.counts,
                "num_params_M": session.counts["total"] / 1000000.0,
                "num_non_embedding_params_M": session.counts["non_embedding"] / 1000000.0,
                "num_scaling_params_M": session.counts["scaling"] / 1000000.0,
                "flops_per_token": session.flops_per_token,
                "total_batch_size": session.total_batch_size,
                "effective_batch_size_tokens": session.total_batch_size,
                "grad_accum_steps": session.grad_accum_steps,
                "total_steps": session.total_steps,
                "target_tokens": config.train.target_tokens,
                "total_budget_tokens": session.total_budget_tokens,
                "world_size": session.world_size,
                "checkpoint_resume_contract": {
                    "version": RESUME_CONTRACT_VERSION,
                    "batch_digest_version": RESUME_BATCH_DIGEST_VERSION,
                    "batch_tensor_fields": list(RESUME_BATCH_TENSOR_FIELDS),
                    "counter_reduction": "packed_int64_all_reduce",
                    "trajectory_sha256": session.trajectory_contract["sha256"],
                    "data_identity_bound": bool(session.data_identity_summary["enabled"]),
                },
                "batch_lr_scale": session.batch_lr_scale,
                "lr_config": {
                    "lr": config.optim.lr,
                    "muon_lr": config.optim.muon_lr,
                    "adamw_lr": config.optim.adamw_lr,
                },
                "lr_schedule": session.lr_schedule,
                "cpu_threading": session.cpu_threading,
                "fixed_ce_trace": {
                    "enabled": bool(config.eval.fixed_ce_trace_enabled),
                    "artifact": "telemetry/fixed_noise_ce_trace.jsonl",
                    "probe_steps": _fixed_ce_trace_steps(session.total_steps, session.warmup_steps)
                    if config.eval.fixed_ce_trace_enabled
                    else [],
                    "batches": int(config.eval.fixed_ce_batches),
                    "t_values": [0.1, 0.3, 0.5, 0.7, 0.9],
                },
                "optimizers": [
                    {
                        "class": optimizer.__class__.__name__,
                        "param_groups": len(optimizer.param_groups),
                        "base_lr": optimizer.param_groups[0].get("base_lr", optimizer.param_groups[0].get("lr", 0.0))
                        if optimizer.param_groups
                        else 0.0,
                        "params": sum(p.numel() for group in optimizer.param_groups for p in group.get("params", [])),
                    }
                    for optimizer in session.optimizers
                ],
            },
        )

    session.scaler = torch.amp.GradScaler(
        "cuda", enabled=config.runtime.device_type == "cuda" and session.compute_dtype == torch.float16
    )

    session.start_step = 0

    session.train_tokens = 0

    session.train_valid_tokens = 0

    session.train_content_tokens = 0

    session.train_content_bytes = 0

    session.resume_metrics: dict = {}

    session.resume_state: dict[str, object] | None = None

    session.resume_next_batch_digest_verified = False

    session.source_checkpoint_load_receipt: dict[str, object] = {}

    session.resume_path = _resolve_resume_path(config.train.run_dir, config.train.resume_from)

    if session.model_initialization_source is not None and session.resume_path is not None:
        same_source_receipt: list[object | None] = [None]
        if session.rank == 0:
            try:
                model_init_path = str(session.model_initialization_source["path"])
                resume_path = os.path.realpath(os.path.expanduser(session.resume_path))
                same_source_receipt[0] = {
                    "status": "ok",
                    "same_file": os.path.exists(resume_path) and os.path.samefile(model_init_path, resume_path),
                }
            except Exception as exc:  # noqa: BLE001 - DDP peers need one terminal path-admission result.
                same_source_receipt[0] = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
        if session.is_ddp:
            dist.broadcast_object_list(same_source_receipt, src=0)
        same_source = same_source_receipt[0]
        if not isinstance(same_source, dict) or same_source.get("status") != "ok":
            raise RuntimeError(f"rank-zero model-init/resume path admission failed: {same_source}")
        if same_source.get("same_file") is True:
            raise RuntimeError("train.model_init_from and train.resume_from resolve to the same checkpoint file")

    if session.resume_path is not None:
        (
            session.start_step,
            session.train_tokens,
            session.train_valid_tokens,
            session.resume_metrics,
            session.resume_state,
        ) = _load_checkpoint(
            session.resume_path,
            session.raw_model,
            session.optimizers,
            session.scaler,
            config,
            session.device,
            rank=session.rank,
            world_size=session.world_size,
            grad_accum_steps=session.grad_accum_steps,
            trajectory_contract=session.trajectory_contract,
            checkpoint_load_receipt=session.source_checkpoint_load_receipt,
        )
        _set_optimizer_base_lrs(session.optimizers, config, session.batch_lr_scale)
        print0(session.rank, f"resumed checkpoint {session.resume_path} at step {session.start_step}")
        session.train_content_tokens = int(session.resume_metrics.get("train_content_tokens", 0))
        session.train_content_bytes = int(session.resume_metrics.get("train_content_bytes", 0))
        if session.start_step > session.total_steps:
            raise RuntimeError(
                f"checkpoint step {session.start_step} exceeds the current training horizon {session.total_steps}"
            )

    if session.rank == 0:
        write_initial_state_artifact_once(
            os.path.join(config.train.run_dir, "initial_state.v1.json"),
            session.initial_state_artifact,
        )
        frozen_draft_source_path = os.path.join(config.train.run_dir, "frozen_draft_model_source.v1.json")
        if session.frozen_draft_model_source_artifact is not None:
            write_initial_state_artifact_once(
                frozen_draft_source_path,
                session.frozen_draft_model_source_artifact,
            )
        elif os.path.exists(frozen_draft_source_path):
            raise RuntimeError("live-student draft run directory contains a stale frozen teacher receipt")

    session.resume_train_tokens = session.train_tokens

    session.resume_train_valid_tokens = session.train_valid_tokens

    session.resume_train_content_tokens = session.train_content_tokens

    session.resume_train_content_bytes = session.train_content_bytes

    session.use_hierarchical_output = session.raw_model.uses_hierarchical_output()

    session.return_aux_logits = (
        config.train.auxiliary_loss_weight > 0 and len(session.model_config.auxiliary_head_layers) > 0
    )

    session.model = (
        torch.compile(session.raw_model, dynamic=config.runtime.compile_dynamic)
        if config.runtime.compile
        else session.raw_model
    )

    # Drafts use the live student unless an exact frozen teacher was admitted.
    # Compiling the teacher matches the live-student draft execution route; RNG
    # restoration keeps compilation from perturbing the training streams.
    if session.raw_frozen_draft_model is None:
        session.self_generated_draft_model = session.model
    else:
        session.saved_state = _rng_state(config)
        try:
            session.self_generated_draft_model = (
                torch.compile(session.raw_frozen_draft_model, dynamic=config.runtime.compile_dynamic)
                if config.runtime.compile
                else session.raw_frozen_draft_model
            )
        finally:
            _restore_rng_state(session.saved_state, config)

    if session.is_ddp:
        session.ddp_kwargs = {"broadcast_buffers": _ddp_broadcast_buffers(config)}
        if config.runtime.device_type == "cuda":
            session.ddp_kwargs["device_ids"] = [session.local_rank]
        session.model = torch.nn.parallel.DistributedDataParallel(
            session.model,
            find_unused_parameters=bool(session.model_config.auxiliary_head_layers and (not session.return_aux_logits)),
            **session.ddp_kwargs,
        )

    session.noise = build_noise(config.noise, session.model_config.mask_token_id, session.model_config.vocab_size)

    session.evaluation_runtime = EvaluationRuntime(
        raw_model=session.raw_model,
        config=config,
        tokenizer=session.tokenizer,
        noise=session.noise,
        autocast_ctx=session.autocast_ctx,
        device=session.device,
    )

    session.eos_supervision_audit_enabled = _eos_supervision_audit_enabled(config)

    session.eos_supervision_startup_metrics = _eos_supervision_startup_audit(
        config, session.tokenizer, session.noise, session.rank, session.world_size, session.device, session.is_ddp
    )

    session.train_loader = build_train_loader(
        config, session.tokenizer, session.rank, session.world_size, session.device
    )
    from .ar_width_transfer_pairing import wrap_training_loader

    session.train_loader = wrap_training_loader(
        config, session.train_loader, grad_accum_steps=session.grad_accum_steps, total_steps=session.total_steps
    )

    session.batch: DLMDataBatch | None = None

    session.prefetch_error: Exception | None = None

    try:
        session.batch = next(session.train_loader)
    except Exception as exc:
        session.prefetch_error = exc

    if session.resume_state is not None:
        _synchronize_resume_phase_error(
            "initial loader prefetch",
            session.prefetch_error,
            rank=session.rank,
            world_size=session.world_size,
            is_ddp=session.is_ddp,
        )
    elif session.prefetch_error is not None:
        raise session.prefetch_error

    assert session.batch is not None

    session.x_0 = session.batch.x_0

    session.loss_mask = session.batch.loss_mask

    session.attention_mask = session.batch.attention_mask

    session.epoch = session.batch.epoch

    if session.resume_state is not None:
        session.fast_forward = int(session.resume_state["consumed_microbatches"])
        print0(session.rank, f"fast-forwarding train loader by {session.fast_forward} micro-batches")
        session.fast_forward_error: Exception | None = None
        try:
            for _ in range(session.fast_forward):
                session.batch = next(session.train_loader)
        except Exception as exc:
            session.fast_forward_error = exc
        _synchronize_resume_phase_error(
            "loader fast-forward",
            session.fast_forward_error,
            rank=session.rank,
            world_size=session.world_size,
            is_ddp=session.is_ddp,
        )
        session.x_0 = session.batch.x_0
        session.loss_mask = session.batch.loss_mask
        session.attention_mask = session.batch.attention_mask
        session.epoch = session.batch.epoch
        session.replay_error: Exception | None = None
        try:
            session.resume_next_batch_digest_verified = _verify_replayed_next_batch(
                session.resume_state, session.batch, rank=session.rank
            )
        except Exception as exc:
            session.replay_error = exc
        _synchronize_resume_phase_error(
            "next-batch replay verification",
            session.replay_error,
            rank=session.rank,
            world_size=session.world_size,
            is_ddp=session.is_ddp,
        )
        session.restore_error: Exception | None = None
        try:
            if int(session.resume_state["version"]) == RESUME_CONTRACT_VERSION:
                _restore_active_rank_rng_state(session.resume_state["local_rng_state"], config, session.device)
            else:
                _restore_rng_state(session.resume_state["local_rng_state"], config)
        except Exception as exc:
            session.restore_error = exc
        _synchronize_resume_phase_error(
            "RNG restoration",
            session.restore_error,
            rank=session.rank,
            world_size=session.world_size,
            is_ddp=session.is_ddp,
        )

    session.bos_token_id = session.tokenizer.get_bos_token_id()

    session.eos_token_id = session.tokenizer.get_eos_token_id()

    session.special_token_ids = {int(i) for i in session.tokenizer.get_special_token_ids()}

    initial_parameter_norm_metrics = {
        f"initial_{key}": value for key, value in _parameter_norm_metrics(session.raw_model).items()
    }
    session.progress = TrainerProgress.from_legacy_metrics(
        max_seq_len=int(config.model.max_seq_len),
        resume_metrics=session.resume_metrics,
        initial_parameter_norm_metrics=initial_parameter_norm_metrics,
        gradient_component_metric_keys=_gradient_component_norm_metrics(session.raw_model),
    )

    session.grad_clip_threshold = float(config.optim.grad_clip)
    session.grad_clip_enabled = session.grad_clip_threshold > 0

    session.t0 = time.time()

    session.last_log_t = session.t0

    session.last_log_step = session.start_step

    session.gc_was_enabled = gc.isenabled()

    session.gc_was_already_frozen = gc.get_freeze_count() > 0

    session.resources.callback(_restore_gc_state, session.gc_was_enabled, not session.gc_was_already_frozen)

    gc.collect()

    if not session.gc_was_already_frozen:
        gc.freeze()

    gc.disable()

    session.train_log_path = os.path.join(config.train.run_dir, "train_log.jsonl")

    session.eval_log_path = os.path.join(config.train.run_dir, "eval_log.jsonl")

    session.residual_diagnostics_path = os.path.join(config.train.run_dir, "telemetry", "residual_diagnostics.jsonl")

    session.fixed_ce_trace_path = os.path.join(config.train.run_dir, "telemetry", "fixed_noise_ce_trace.jsonl")

    session.embedding_head_init_diagnostics_path = os.path.join(
        config.train.run_dir, "telemetry", "embedding_head_init_diagnostics.jsonl"
    )

    session.token_edit_prefix_plan_path = os.path.join(
        config.train.run_dir, "telemetry", f"token_edit_prefix_plans.rank{session.rank}.jsonl"
    )
    if _uses_token_edit_prefix(config):
        os.makedirs(os.path.dirname(session.token_edit_prefix_plan_path), exist_ok=True)

    session.fixed_ce_trace_steps = (
        _fixed_ce_trace_steps(session.total_steps, session.warmup_steps)
        if config.eval.enabled and config.eval.fixed_ce_trace_enabled
        else []
    )

    session.tensorboard_writer = _build_tensorboard_writer(config, session.rank)

    session.resources.callback(close_tensorboard_writer, session.tensorboard_writer)

    if session.tensorboard_writer is not None:
        print0(session.rank, f"tensorboard logs to {_resolve_tensorboard_dir(config)}")

    if session.resume_state is not None and session.resume_state.get("local_trainer_state") is not None:
        session.progress = TrainerProgress.from_checkpoint(
            session.resume_state["local_trainer_state"],
            max_seq_len=int(config.model.max_seq_len),
        )

    # Logical interval counters survive exact resume. Throughput wall time does
    # not, so draft-forward FLOP/s counts only work performed by this process.
    session.runtime_interval_start_self_generated_draft_forwards = int(
        session.progress.interval.self_generated_edit_accumulator["draft_forwards"]
    )

    session.all_logits_indices = torch.arange(
        int(config.train.device_batch_size) * config.model.max_seq_len, device=session.device, dtype=torch.long
    )

    if session.record_embedding_head_init_diagnostics and session.rank == 0 and (session.start_step == 0):
        session.saved_state = _rng_state(config)
        try:
            session.progress.diagnostics.initial_lm_head_row_norm_metrics = _prefixed_metric_keys(
                "initial_",
                _lm_head_row_norm_metrics(
                    session.raw_model,
                    session.tokenizer,
                    config.data,
                    session.device,
                    int(config.train.device_batch_size),
                    int(config.model.max_seq_len),
                    batches=10,
                ),
            )
            os.makedirs(os.path.dirname(session.embedding_head_init_diagnostics_path), exist_ok=True)
            append_jsonl(
                session.embedding_head_init_diagnostics_path,
                {
                    "event": "initial_lm_head_row_norms",
                    "step": int(session.start_step),
                    **_embedding_head_init_metrics(config),
                    **session.progress.diagnostics.initial_lm_head_row_norm_metrics,
                },
            )
        finally:
            _restore_rng_state(session.saved_state, config)

    return session


def _collect_final_lm_head_row_norm_metrics(session: TrainingSession) -> dict[str, object]:
    """Run the rank-zero probe without letting a failed DDP rank continue."""

    row_norm_metrics: dict[str, object] = {}
    local_error: Exception | None = None
    if session.rank == 0:
        try:
            saved_state = _rng_state(session.config)
            try:
                row_norm_metrics = _lm_head_row_norm_metrics(
                    session.raw_model,
                    session.tokenizer,
                    session.config.data,
                    session.device,
                    int(session.config.train.device_batch_size),
                    int(session.config.model.max_seq_len),
                    batches=10,
                )
            finally:
                _restore_rng_state(saved_state, session.config)
        except Exception as exc:
            # Delay the raise until every DDP rank knows that finalization failed.
            local_error = exc

    if session.is_ddp:
        local_result = {
            "rank": int(session.rank),
            "error": f"{type(local_error).__name__}: {local_error}" if local_error is not None else "",
        }
        gathered: list[object] = [None for _ in range(session.world_size)]
        dist.all_gather_object(gathered, local_result)
        failures = [item for item in gathered if isinstance(item, dict) and item.get("error")]
        if failures:
            details = "; ".join(f"rank {item['rank']}: {item['error']}" for item in failures)
            raise RuntimeError(f"final lm-head row-norm probe failed across ranks: {details}") from local_error
    if local_error is not None:
        raise local_error
    return row_norm_metrics


def _finalize_training_session(session: TrainingSession) -> dict:

    if session.config.runtime.device_type == "cuda":
        torch.cuda.synchronize()

    session.final_resume_contract = _collect_resume_contract(
        session.config,
        session.batch,
        device=session.device,
        rank=session.rank,
        world_size=session.world_size,
        is_ddp=session.is_ddp,
        grad_accum_steps=session.grad_accum_steps,
        consumed_microbatches=session.total_steps * session.grad_accum_steps,
        trajectory_contract=session.trajectory_contract,
        trainer_state=_current_trainer_state(session),
    )

    session.training_seconds = time.time() - session.t0

    session.progress.lineage.ar_boundary_counts = _packed_all_reduce_int_counters(
        {key: int(value) for key, value in session.progress.lineage.ar_boundary_counts.items()},
        device=session.device,
        is_ddp=session.is_ddp,
    )

    session.progress.lineage.boundary_telemetry = _packed_all_reduce_int_counters(
        {key: int(value) for key, value in session.progress.lineage.boundary_telemetry.items()},
        device=session.device,
        is_ddp=session.is_ddp,
    )

    session.row_sum_keys = tuple(
        key for key in session.progress.lineage.row_telemetry if not key.endswith("_min") and (not key.endswith("_max"))
    )

    session.row_extrema_min_keys = tuple(key for key in session.progress.lineage.row_telemetry if key.endswith("_min"))

    session.row_extrema_max_keys = tuple(key for key in session.progress.lineage.row_telemetry if key.endswith("_max"))

    session.row_sums = _packed_all_reduce_int_counters(
        {key: int(session.progress.lineage.row_telemetry[key]) for key in session.row_sum_keys},
        device=session.device,
        is_ddp=session.is_ddp,
    )

    session.row_mins = _packed_all_reduce_int_counters(
        {key: int(session.progress.lineage.row_telemetry[key]) for key in session.row_extrema_min_keys},
        device=session.device,
        is_ddp=session.is_ddp,
        op=dist.ReduceOp.MIN,
    )

    session.row_maxes = _packed_all_reduce_int_counters(
        {key: int(session.progress.lineage.row_telemetry[key]) for key in session.row_extrema_max_keys},
        device=session.device,
        is_ddp=session.is_ddp,
        op=dist.ReduceOp.MAX,
    )

    session.progress.lineage.row_telemetry = {**session.row_sums, **session.row_mins, **session.row_maxes}

    session.session_train_tokens = max(0, session.train_tokens - session.resume_train_tokens)

    session.session_train_valid_tokens = max(0, session.train_valid_tokens - session.resume_train_valid_tokens)

    session.session_train_content_tokens = max(0, session.train_content_tokens - session.resume_train_content_tokens)

    session.session_train_content_bytes = max(0, session.train_content_bytes - session.resume_train_content_bytes)

    session.eos_supervision_train_counts_global = (
        _reduce_eos_supervision_counts(
            session.progress.lineage.eos_supervision_train_counts, session.device, session.is_ddp
        )
        if session.eos_supervision_audit_enabled
        else _new_eos_supervision_counts()
    )

    session.eos_supervision_train_failures = (
        _eos_supervision_training_invariant_failures(
            session.eos_supervision_train_counts_global, session.eos_supervision_startup_metrics
        )
        if session.eos_supervision_audit_enabled
        else []
    )

    session.eos_supervision_train_metrics = {
        "eos_supervision_audit_enabled": bool(session.eos_supervision_audit_enabled),
        "eos_supervision_audit_artifact": "telemetry/eos_supervision_audit.jsonl",
        "eos_supervision_audit_path": os.path.join(
            session.config.train.run_dir, "telemetry", "eos_supervision_audit.jsonl"
        ),
        "eos_supervision_invariant_status": "ok"
        if session.eos_supervision_audit_enabled and (not session.eos_supervision_train_failures)
        else "disabled"
        if not session.eos_supervision_audit_enabled
        else "failed",
        "eos_supervision_invariant_error": "; ".join(session.eos_supervision_train_failures),
    }

    session.eos_supervision_train_metrics.update(
        _eos_supervision_count_metrics("train_eos_supervision_", session.eos_supervision_train_counts_global)
    )

    if session.eos_supervision_audit_enabled and session.rank == 0:
        append_jsonl(
            os.path.join(session.config.train.run_dir, "telemetry", "eos_supervision_audit.jsonl"),
            {"event": "final_train_eos_supervision_audit", **session.eos_supervision_train_metrics},
        )

    if (
        session.eos_supervision_audit_enabled
        and session.eos_supervision_train_metrics["eos_supervision_invariant_status"] != "ok"
    ):
        raise RuntimeError(
            f"EOS supervision invariant failed: {session.eos_supervision_train_metrics['eos_supervision_invariant_error']}"
        )

    session.distributed_cuda_memory, peak_vram_mb = memory_snapshot(session, _distributed_cuda_memory_telemetry)
    source_metrics = source_promotion_metrics()

    train_length_prior_counts = (
        session.progress.lineage.train_length_head_accumulator["target_counts"]
        if session.config.train.length_head_loss_weight > 0
        and isinstance(session.progress.lineage.train_length_head_accumulator.get("target_counts"), list)
        else None
    )

    final_eval_metrics: dict[str, object] = {}
    if session.rank == 0 and session.config.eval.enabled:
        eval_start = time.time()
        final_eval_metrics = _run_eval_with_runtime(
            session.evaluation_runtime,
            session.total_steps,
            include_gen=session.config.eval.gen_eval,
        )
        final_eval_metrics["evaluation_seconds"] = time.time() - eval_start
        if session.config.train.length_head_loss_weight > 0:
            final_eval_metrics.update(
                _evaluate_length_head_metrics(
                    session.raw_model,
                    session.config,
                    session.tokenizer,
                    session.device,
                    session.autocast_ctx,
                    train_prior_counts=train_length_prior_counts,
                )
            )
        append_jsonl(session.eval_log_path, final_eval_metrics)
        _write_tensorboard_scalars(session.tensorboard_writer, "eval", final_eval_metrics, session.total_steps)

    row_norm_metrics = _collect_final_lm_head_row_norm_metrics(session)

    final_metrics = build_final_metrics_v1(
        FinalReportInputs.from_session(
            session,
            source_metrics=source_metrics,
            peak_vram_mb=peak_vram_mb,
            final_eval_metrics=final_eval_metrics,
            row_norm_metrics=row_norm_metrics,
            entrypoint_mode=os.environ.get("NANODLLM_ENTRYPOINT_MODE", "train"),
        )
    )

    if session.rank == 0:
        _publish_final_outputs(
            session.config,
            rank=session.rank,
            metrics=final_metrics,
            tensorboard_writer=session.tensorboard_writer,
            total_steps=session.total_steps,
            raw_model=session.raw_model,
            optimizers=session.optimizers,
            scaler=session.scaler,
            model_config=session.model_config,
            train_tokens=session.train_tokens,
            train_valid_tokens=session.train_valid_tokens,
            resume_contract=session.final_resume_contract,
        )

    if session.is_ddp:
        dist.barrier()

    return final_metrics


def _train_impl(config: ExperimentConfig, resources: ExitStack) -> dict:
    session = _bootstrap_training_session(config, resources)
    diagnostic_metrics = _run_requested_training_diagnostic(session)
    if diagnostic_metrics is not None:
        return diagnostic_metrics
    session = _initialize_standard_training_session(session)
    initialize_monitor(session)
    from .ar_width_transfer_diagnostics import diagnostic_scope

    with diagnostic_scope(session):
        _run_training_steps(session)
    from .ar_width_transfer_pairing import finish_training_loader

    finish_training_loader(session.train_loader)
    if session.monitor is not None:
        session.monitor.switch("finalize")
    return _finalize_training_session(session)


def train(config: ExperimentConfig) -> dict:
    process_group_was_initialized = dist.is_available() and dist.is_initialized()
    with ExitStack() as resources:
        if not process_group_was_initialized:
            resources.callback(_destroy_process_group_if_initialized)
        return _train_impl(config, resources)
