from __future__ import annotations

import hashlib
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import (
    ParquetTextSource,
    build_train_loader,
    encode_document_batch,
    make_dataloader,
)
from .eval import target_eos_lengths
from .model import DiffusionTransformer
from .nelbo_time_sampling import build_nelbo_time_plan
from .telemetry import append_jsonl
from .training_bd3lm import (
    bd3lm_concat_inputs as _bd3lm_concat_inputs,
)
from .training_bd3lm import (
    sample_bd3lm_block_times,
)
from .training_block_conversion import (
    block_conversion_objective_label,
    block_conversion_objective_telemetry,
)
from .training_features import (
    _eos_supervision_audit_enabled,
    _is_causal,
    _token_edit_probability_bounds,
    _token_edit_support_policy,
    _uses_bd3lm,
    _uses_bd3lm_causal_clean_stream,
    _uses_block_causal_pilot,
    _uses_content_budget_segmented_packing,
    _uses_differentiable_self_generated_draft,
    _uses_legacy_fixed_token_edit_schedule,
    _uses_matched_token_edit_schedule,
    _uses_mdlm_m2t_per_row_t,
    _uses_mixed_token_edit_canvas,
    _uses_per_token_noise_levels,
    _uses_target_length_conditioning,
    _uses_token_edit_corruption,
    _uses_token_edit_prefix,
    _uses_trajectory_token_edit_canvas,
)
from .training_forward import _forward_training_model as _forward_training_model
from .training_noise_time import sample_training_scalar_noise_time as _sample_training_scalar_noise_time
from .training_noise_time import sample_training_tensor_noise_time as _sample_training_tensor_noise_time
from .training_progress import TOKEN_EDIT_CORRUPTION_DECILES
from .training_rng import _restore_rng_state, _rng_state
from .training_self_generated_edit import (
    self_generated_objective_configuration_telemetry,
)

BD3LM_OBJECTIVE_TIME_SAMPLING_CONTRACT_VERSION = 1

EOS_SUPERVISION_COUNT_KEYS = (
    "microbatches",
    "rows",
    "valid_tokens",
    "loss_denominator_tokens",
    "attention_lossable_tokens",
    "lossable_attention_mismatch_tokens",
    "special_tokens",
    "special_loss_tokens",
    "eos_tokens",
    "eos_loss_tokens",
    "eos_rows",
    "eos_loss_rows",
    "active_target_tokens",
    "active_eos_target_tokens",
    "active_special_target_tokens",
    "row_eos_loss_tokens_0",
    "row_eos_loss_tokens_1",
    "row_eos_loss_tokens_2",
    "row_eos_loss_tokens_3",
    "row_eos_loss_tokens_4plus",
)


def _noise_telemetry(config: ExperimentConfig) -> dict[str, bool | float | str]:
    telemetry = {
        "noise_type": config.noise.type,
        "noise_t_min": float(config.noise.t_min),
        "noise_t_max": float(config.noise.t_max),
        "noise_schedule_weight": bool(config.noise.schedule_weight),
        "noise_complement_masking": bool(config.noise.complement_masking),
        "noise_per_token_noise_levels": _uses_per_token_noise_levels(config),
        "noise_time_conditioning": bool(config.noise.time_conditioning),
        "noise_suffix_masking": bool(config.noise.suffix_masking),
        "mask_uniform_mix_prob": float(config.noise.mask_uniform_mix_prob),
        "noise_soft_input_curriculum": bool(config.noise.soft_input_curriculum),
        "noise_soft_input_curriculum_start_mix": float(config.noise.soft_input_curriculum_start_mix),
        "noise_soft_input_curriculum_end_mix": float(config.noise.soft_input_curriculum_end_mix),
        "noise_soft_input_curriculum_anneal_fraction": float(config.noise.soft_input_curriculum_anneal_fraction),
    }
    if config.noise.mdlm_time_sampling != "legacy_auto":
        telemetry.update(
            noise_mdlm_time_sampling=config.noise.mdlm_time_sampling,
            noise_mdlm_unweighted_loss_scale=float(config.noise.mdlm_unweighted_loss_scale),
        )
    if config.noise.mdlm_time_sampling == "stratified_per_row":
        telemetry["noise_mdlm_stratification_scope"] = "local_microbatch_rows"
    return telemetry


def _token_edit_corruption_telemetry(
    config: ExperimentConfig,
) -> dict[str, bool | float | int | str]:
    enabled = _uses_token_edit_corruption(config)
    probability_min, probability_max = _token_edit_probability_bounds(config)
    telemetry: dict[str, bool | float | int | str] = {
        "token_edit_corruption_enabled": enabled,
        "token_edit_corruption_batch_fraction": float(config.train.token_edit_corruption_batch_fraction),
        "token_edit_batch_layout": config.train.token_edit_batch_layout,
        "token_edit_corruption_schedule": config.train.token_edit_corruption_schedule,
        "token_edit_corruption_ratio": float(config.train.token_edit_corruption_ratio),
        "token_edit_corruption_probability_min": probability_min,
        "token_edit_corruption_probability_max": probability_max,
        "token_edit_corruption_support_policy": _token_edit_support_policy(config),
        "token_edit_corruption_deciles": TOKEN_EDIT_CORRUPTION_DECILES,
        "token_edit_backbone_gradient_mode": config.train.token_edit_backbone_gradient_mode,
        "token_edit_draft_m2t_loss_weight": float(config.train.token_edit_draft_m2t_loss_weight),
        "token_edit_correction_loss_weight": 1.0 - float(config.train.token_edit_draft_m2t_loss_weight),
        "training_objective": (
            "mdlm_m2t_plus_partial_trajectory_t2t_edit"
            if enabled and _uses_trajectory_token_edit_canvas(config)
            else "mdlm_m2t_plus_mixed_mask_draft_denoising"
            if enabled and _uses_mixed_token_edit_canvas(config)
            else "mdlm_m2t_plus_t2t_edit"
            if enabled
            else "mdlm_m2t"
        ),
    }
    if (
        config.model.token_edit_adapter_bottleneck_dim > 0
        or config.model.token_edit_provenance_embedding
        or _uses_mixed_token_edit_canvas(config)
        or _uses_trajectory_token_edit_canvas(config)
    ):
        telemetry.update(
            {
                "token_edit_canvas_mode": config.train.token_edit_canvas_mode,
                "token_edit_mixed_canvas_draft_fraction": float(config.train.token_edit_mixed_canvas_draft_fraction),
                "token_edit_trajectory_completion_fraction": float(
                    config.train.token_edit_trajectory_completion_fraction
                ),
                "token_edit_trajectory_steps": int(config.train.token_edit_trajectory_steps),
                "token_edit_provenance_embedding": bool(config.model.token_edit_provenance_embedding),
                "token_edit_provenance_confidence_bins": int(config.model.token_edit_provenance_confidence_bins),
                "token_edit_full_draft_provenance": bool(config.train.token_edit_full_draft_provenance),
                "token_edit_adapter_bottleneck_dim": int(config.model.token_edit_adapter_bottleneck_dim),
                "token_edit_adapter_top_layers": int(config.model.token_edit_adapter_top_layers),
            }
        )
    telemetry.update(self_generated_objective_configuration_telemetry(config))
    if float(config.train.token_edit_cross_view_nonregression_weight) > 0.0:
        telemetry["training_objective"] = "mdlm_m2t_plus_t2t_expected_accuracy_cross_view_nonregression"
    elif config.train.token_edit_expected_accuracy_power_schedule == "corruption_threshold":
        telemetry["training_objective"] = "mdlm_m2t_plus_t2t_corruption_conditioned_expected_accuracy"
    elif float(config.train.token_edit_expected_accuracy_power) != 1.0:
        telemetry["training_objective"] = "mdlm_m2t_plus_t2t_generalized_expected_accuracy"
    if _uses_differentiable_self_generated_draft(config):
        telemetry.update(
            {
                "training_objective": "mdlm_m2t_plus_joint_draft_m2t_and_t2t_expected_accuracy",
                "token_edit_draft_m2t_credit_mode": config.train.token_edit_draft_m2t_credit_mode,
                "token_edit_draft_m2t_credit_strength": float(config.train.token_edit_draft_m2t_credit_strength),
                "token_edit_draft_m2t_credit_clip_min": float(config.train.token_edit_draft_m2t_credit_clip_min),
                "token_edit_draft_m2t_credit_clip_max": float(config.train.token_edit_draft_m2t_credit_clip_max),
                "token_edit_draft_m2t_credit_signal": "detached_correction_expected_error",
                "token_edit_draft_m2t_credit_advantage": "row_mean_expected_error_minus_token_expected_error",
                "token_edit_draft_m2t_credit_normalization": "mean_one_per_active_row",
                "token_edit_draft_m2t_credit_shuffle": (
                    "stateless_training_seed_global_microbatch_row"
                    if config.train.token_edit_draft_m2t_credit_mode == "shuffled_expected_error_reward"
                    else "disabled"
                ),
            }
        )
    if _uses_token_edit_prefix(config):
        telemetry.update(
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
    if _uses_trajectory_token_edit_canvas(config):
        telemetry.update(
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
    return telemetry


def _training_corruption_label(config: ExperimentConfig) -> str:
    if _is_causal(config):
        return "causal-ar"
    if _uses_bd3lm(config):
        if _uses_bd3lm_causal_clean_stream(config):
            return (
                "content-budget-segmented-" + block_conversion_objective_label(config)
                if _uses_content_budget_segmented_packing(config)
                else block_conversion_objective_label(config)
            )
        return (
            "content-budget-segmented-bd3lm-independent-block-time"
            if _uses_content_budget_segmented_packing(config)
            else "bd3lm-independent-block-time"
        )
    if _uses_content_budget_segmented_packing(config):
        return "content-budget-segmented-mdlm"
    if _uses_block_causal_pilot(config):
        return "block-causal-pilot-mdlm"
    if config.noise.type == "mdlm" and config.noise.suffix_masking:
        return "suffix-mask-mdlm"
    if config.noise.type == "mdlm" and _uses_per_token_noise_levels(config):
        return "per-token-noise-mdlm"
    return config.noise.type


def _causal_ar_label_smoothing(config: ExperimentConfig) -> float:
    if not _is_causal(config):
        return 0.0
    return float(config.train.label_smoothing_epsilon)


def _stratified_per_sequence_times(
    count: int,
    t_min: float,
    t_max: float,
    seed: int,
) -> torch.Tensor:
    """Return a deterministic, permuted midpoint stratification of ``[t_min, t_max]``."""

    return build_nelbo_time_plan(count, t_min, t_max, seed).times


NELBO_CORRUPTION_RNG_SCHEME = "torch_cpu_sequence_major_uniform_v1"


def _sequence_major_nelbo_corruption_plan(
    count: int,
    seq_len: int,
    base_seed: int,
) -> tuple[torch.Tensor, int]:
    """Build one CPU random plan whose shape is independent of device batching."""

    if count <= 0 or seq_len <= 0:
        raise ValueError("NELBO corruption plan dimensions must be positive")
    seed_material = f"nanodllm-fixed-mdlm-corruption-v1:{int(base_seed)}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed)
    return torch.rand((count, seq_len), generator=generator, dtype=torch.float32), derived_seed


def _new_eos_supervision_counts() -> dict[str, int]:
    return {key: 0 for key in EOS_SUPERVISION_COUNT_KEYS}


def _update_eos_row_loss_distribution(counts: dict[str, int], row_counts: torch.Tensor) -> None:
    if row_counts.numel() == 0:
        return
    row_counts_cpu = row_counts.detach().to(device="cpu", dtype=torch.long)
    counts["row_eos_loss_tokens_0"] += int((row_counts_cpu == 0).sum().item())
    counts["row_eos_loss_tokens_1"] += int((row_counts_cpu == 1).sum().item())
    counts["row_eos_loss_tokens_2"] += int((row_counts_cpu == 2).sum().item())
    counts["row_eos_loss_tokens_3"] += int((row_counts_cpu == 3).sum().item())
    counts["row_eos_loss_tokens_4plus"] += int((row_counts_cpu >= 4).sum().item())


def _update_eos_supervision_batch_counts(
    counts: dict[str, int],
    x_batch: torch.Tensor,
    loss_mask_batch: torch.Tensor,
    attention_mask_batch: torch.Tensor,
    special_token_ids: set[int],
    eos_token_id: int | None,
    active_mask_batch: torch.Tensor | None = None,
) -> None:
    if eos_token_id is None:
        return
    if x_batch.shape != loss_mask_batch.shape or x_batch.shape != attention_mask_batch.shape:
        raise ValueError(
            "EOS audit batch tensors must share shape: "
            f"x={tuple(x_batch.shape)} loss={tuple(loss_mask_batch.shape)} "
            f"attention={tuple(attention_mask_batch.shape)}"
        )
    valid_mask = attention_mask_batch.to(dtype=torch.bool)
    loss_mask_bool = loss_mask_batch.to(dtype=torch.bool)
    attention_lossable = loss_mask_bool & valid_mask
    special_mask = torch.zeros_like(valid_mask)
    for token_id in special_token_ids:
        special_mask |= x_batch == int(token_id)
    special_mask &= valid_mask
    eos_mask = (x_batch == int(eos_token_id)) & valid_mask
    eos_loss_mask = eos_mask & loss_mask_bool

    counts["microbatches"] += 1
    counts["rows"] += int(x_batch.shape[0])
    counts["valid_tokens"] += int(valid_mask.sum().item())
    counts["loss_denominator_tokens"] += int(loss_mask_bool.sum().item())
    counts["attention_lossable_tokens"] += int(attention_lossable.sum().item())
    counts["lossable_attention_mismatch_tokens"] += int((loss_mask_bool & ~valid_mask).sum().item())
    counts["special_tokens"] += int(special_mask.sum().item())
    counts["special_loss_tokens"] += int((special_mask & loss_mask_bool).sum().item())
    counts["eos_tokens"] += int(eos_mask.sum().item())
    counts["eos_loss_tokens"] += int(eos_loss_mask.sum().item())
    counts["eos_rows"] += int(eos_mask.any(dim=1).sum().item())
    counts["eos_loss_rows"] += int(eos_loss_mask.any(dim=1).sum().item())
    _update_eos_row_loss_distribution(counts, eos_loss_mask.sum(dim=1))

    if active_mask_batch is None:
        return
    if active_mask_batch.shape != x_batch.shape:
        raise ValueError(
            f"EOS audit active mask shape {tuple(active_mask_batch.shape)} != x shape {tuple(x_batch.shape)}"
        )
    active_mask = active_mask_batch.to(dtype=torch.bool) & loss_mask_bool
    counts["active_target_tokens"] += int(active_mask.sum().item())
    counts["active_eos_target_tokens"] += int((active_mask & eos_mask).sum().item())
    counts["active_special_target_tokens"] += int((active_mask & special_mask).sum().item())


def _reduce_eos_supervision_counts(
    counts: dict[str, int],
    device,
    is_ddp: bool,
) -> dict[str, int]:
    if not is_ddp:
        return {key: int(counts.get(key, 0)) for key in EOS_SUPERVISION_COUNT_KEYS}
    values = torch.tensor(
        [int(counts.get(key, 0)) for key in EOS_SUPERVISION_COUNT_KEYS],
        dtype=torch.long,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {key: int(value) for key, value in zip(EOS_SUPERVISION_COUNT_KEYS, values.cpu().tolist(), strict=True)}


def _eos_supervision_count_metrics(
    prefix: str,
    counts: dict[str, int],
) -> dict[str, bool | float | int | str]:
    valid_tokens = int(counts.get("valid_tokens", 0))
    loss_denominator_tokens = int(counts.get("loss_denominator_tokens", 0))
    eos_tokens = int(counts.get("eos_tokens", 0))
    eos_loss_tokens = int(counts.get("eos_loss_tokens", 0))
    active_target_tokens = int(counts.get("active_target_tokens", 0))
    active_eos_target_tokens = int(counts.get("active_eos_target_tokens", 0))
    special_tokens = int(counts.get("special_tokens", 0))
    special_loss_tokens = int(counts.get("special_loss_tokens", 0))
    rows = int(counts.get("rows", 0))
    metrics: dict[str, bool | float | int | str] = {
        f"{prefix}eos_supervision_microbatch_count": int(counts.get("microbatches", 0)),
        f"{prefix}eos_supervision_row_count": rows,
        f"{prefix}eos_supervision_valid_token_count": valid_tokens,
        f"{prefix}eos_supervision_loss_denominator_token_count": loss_denominator_tokens,
        f"{prefix}eos_supervision_attention_lossable_token_count": int(counts.get("attention_lossable_tokens", 0)),
        f"{prefix}eos_supervision_lossable_attention_mismatch_token_count": int(
            counts.get("lossable_attention_mismatch_tokens", 0)
        ),
        f"{prefix}eos_token_count": eos_tokens,
        f"{prefix}eos_token_frequency": eos_tokens / max(1, valid_tokens),
        f"{prefix}eos_loss_token_count": eos_loss_tokens,
        f"{prefix}eos_loss_token_fraction_of_loss_denominator": eos_loss_tokens / max(1, loss_denominator_tokens),
        f"{prefix}eos_loss_token_fraction_of_eos_tokens": eos_loss_tokens / max(1, eos_tokens),
        f"{prefix}eos_loss_denominator_includes_eos": bool(eos_loss_tokens > 0),
        f"{prefix}eos_bearing_row_count": int(counts.get("eos_rows", 0)),
        f"{prefix}eos_bearing_row_fraction": int(counts.get("eos_rows", 0)) / max(1, rows),
        f"{prefix}eos_loss_bearing_row_count": int(counts.get("eos_loss_rows", 0)),
        f"{prefix}eos_loss_bearing_row_fraction": int(counts.get("eos_loss_rows", 0)) / max(1, rows),
        f"{prefix}row_eos_loss_tokens_0_count": int(counts.get("row_eos_loss_tokens_0", 0)),
        f"{prefix}row_eos_loss_tokens_1_count": int(counts.get("row_eos_loss_tokens_1", 0)),
        f"{prefix}row_eos_loss_tokens_2_count": int(counts.get("row_eos_loss_tokens_2", 0)),
        f"{prefix}row_eos_loss_tokens_3_count": int(counts.get("row_eos_loss_tokens_3", 0)),
        f"{prefix}row_eos_loss_tokens_4plus_count": int(counts.get("row_eos_loss_tokens_4plus", 0)),
        f"{prefix}special_token_count": special_tokens,
        f"{prefix}special_token_frequency": special_tokens / max(1, valid_tokens),
        f"{prefix}special_loss_token_count": special_loss_tokens,
        f"{prefix}special_loss_token_fraction_of_loss_denominator": special_loss_tokens
        / max(1, loss_denominator_tokens),
        f"{prefix}special_loss_token_fraction_of_special_tokens": special_loss_tokens / max(1, special_tokens),
        f"{prefix}masked_target_count": active_target_tokens,
        f"{prefix}masked_eos_target_count": active_eos_target_tokens,
        f"{prefix}masked_eos_target_fraction": active_eos_target_tokens / max(1, active_target_tokens),
        f"{prefix}masked_special_target_count": int(counts.get("active_special_target_tokens", 0)),
    }
    return metrics


def _collect_eos_prepack_counts(
    config: ExperimentConfig,
    tokenizer,
    rank: int,
    world_size: int,
) -> dict[str, int]:
    counts = {
        "document_batches": 0,
        "documents": 0,
        "eos_documents": 0,
        "tokens": 0,
        "loss_tokens": 0,
        "special_tokens": 0,
        "special_loss_tokens": 0,
        "eos_tokens": 0,
        "eos_loss_tokens": 0,
    }
    eos_token_id = tokenizer.get_eos_token_id()
    special_token_ids = {int(i) for i in tokenizer.get_special_token_ids()}
    if config.data.synthetic_dataset:
        training_seed = int(config.runtime.training_seed if config.runtime.training_seed >= 0 else config.runtime.seed)
        batches = make_dataloader(
            config.data,
            tokenizer,
            int(config.train.device_batch_size),
            config.model.max_seq_len,
            "train",
            device="cpu",
            rank=rank,
            world_size=world_size,
            augmentation_seed=training_seed + int(rank),
        )
        for _ in range(config.diagnostics.eos_supervision_prepack_document_batches):
            batch = next(batches)
            counts["document_batches"] += 1
            for row, row_attention, row_loss in zip(batch.x_0, batch.attention_mask, batch.loss_mask, strict=True):
                visible = row_attention.to(dtype=torch.bool)
                if not bool(visible.any()):
                    continue
                tokens = row[visible]
                lossable = row_loss[visible].to(dtype=torch.bool)
                special = torch.zeros_like(tokens, dtype=torch.bool)
                for token_id in special_token_ids:
                    special |= tokens == token_id
                eos = tokens == int(eos_token_id) if eos_token_id is not None else torch.zeros_like(special)
                counts["documents"] += 1
                counts["tokens"] += int(tokens.numel())
                counts["loss_tokens"] += int(lossable.sum())
                counts["special_tokens"] += int(special.sum())
                counts["special_loss_tokens"] += int((special & lossable).sum())
                counts["eos_tokens"] += int(eos.sum())
                counts["eos_loss_tokens"] += int((eos & lossable).sum())
                counts["eos_documents"] += int(bool(eos.any()))
        return counts
    source = ParquetTextSource(config.data)
    batches = source.document_batches("train", rank=rank, world_size=world_size)
    for _ in range(config.diagnostics.eos_supervision_prepack_document_batches):
        texts, _epoch = next(batches)
        counts["document_batches"] += 1
        for doc_ids in encode_document_batch(tokenizer, texts, config.data):
            if not doc_ids:
                continue
            counts["documents"] += 1
            counts["tokens"] += len(doc_ids)
            doc_loss_tokens = 0
            doc_special_tokens = 0
            doc_special_loss_tokens = 0
            doc_eos_tokens = 0
            doc_eos_loss_tokens = 0
            for token_id in doc_ids:
                token_id = int(token_id)
                token_lossable = True
                if token_id in special_token_ids:
                    doc_special_tokens += 1
                if tokenizer.get_bos_token_id() is not None and token_id == int(tokenizer.get_bos_token_id()):
                    token_lossable = bool(config.data.loss_on_bos)
                if eos_token_id is not None and token_id == int(eos_token_id):
                    doc_eos_tokens += 1
                    token_lossable = bool(config.data.loss_on_eos)
                if token_lossable:
                    doc_loss_tokens += 1
                    if token_id in special_token_ids:
                        doc_special_loss_tokens += 1
                    if eos_token_id is not None and token_id == int(eos_token_id):
                        doc_eos_loss_tokens += 1
            counts["loss_tokens"] += doc_loss_tokens
            counts["special_tokens"] += doc_special_tokens
            counts["special_loss_tokens"] += doc_special_loss_tokens
            counts["eos_tokens"] += doc_eos_tokens
            counts["eos_loss_tokens"] += doc_eos_loss_tokens
            counts["eos_documents"] += int(doc_eos_tokens > 0)
    return counts


def _reduce_prepack_counts(
    counts: dict[str, int],
    device,
    is_ddp: bool,
) -> dict[str, int]:
    keys = tuple(counts.keys())
    if not is_ddp:
        return {key: int(counts[key]) for key in keys}
    values = torch.tensor([int(counts[key]) for key in keys], dtype=torch.long, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {key: int(value) for key, value in zip(keys, values.cpu().tolist(), strict=True)}


def _eos_prepack_metrics(prefix: str, counts: dict[str, int]) -> dict[str, bool | float | int | str]:
    tokens = int(counts.get("tokens", 0))
    documents = int(counts.get("documents", 0))
    eos_tokens = int(counts.get("eos_tokens", 0))
    eos_loss_tokens = int(counts.get("eos_loss_tokens", 0))
    special_tokens = int(counts.get("special_tokens", 0))
    special_loss_tokens = int(counts.get("special_loss_tokens", 0))
    loss_tokens = int(counts.get("loss_tokens", 0))
    return {
        f"{prefix}eos_prepack_document_batch_count": int(counts.get("document_batches", 0)),
        f"{prefix}eos_prepack_document_count": documents,
        f"{prefix}eos_prepack_eos_document_count": int(counts.get("eos_documents", 0)),
        f"{prefix}eos_prepack_eos_document_fraction": int(counts.get("eos_documents", 0)) / max(1, documents),
        f"{prefix}eos_prepack_token_count": tokens,
        f"{prefix}eos_prepack_loss_token_count": loss_tokens,
        f"{prefix}eos_prepack_eos_token_count": eos_tokens,
        f"{prefix}eos_prepack_eos_token_frequency": eos_tokens / max(1, tokens),
        f"{prefix}eos_prepack_eos_loss_token_count": eos_loss_tokens,
        f"{prefix}eos_prepack_eos_loss_token_fraction_of_loss_tokens": eos_loss_tokens / max(1, loss_tokens),
        f"{prefix}eos_prepack_special_token_count": special_tokens,
        f"{prefix}eos_prepack_special_token_frequency": special_tokens / max(1, tokens),
        f"{prefix}eos_prepack_special_loss_token_count": special_loss_tokens,
        f"{prefix}eos_prepack_special_loss_token_fraction_of_loss_tokens": special_loss_tokens / max(1, loss_tokens),
    }


def _eos_supervision_training_invariant_failures(
    train_counts: dict[str, int],
    startup_metrics: dict[str, bool | float | int | str],
) -> list[str]:
    failures: list[str] = []
    if int(train_counts.get("eos_rows", 0)) <= 0:
        failures.append("no EOS-bearing rows observed in training batches")
    if int(train_counts.get("eos_loss_tokens", 0)) <= 0:
        failures.append("no EOS loss tokens observed in training batches")
    if int(startup_metrics.get("startup_postpack_masked_eos_target_count", 0)) <= 0:
        failures.append("no EOS tokens became masked targets in the forced startup corruption canary")
    if int(startup_metrics.get("startup_postpack_eos_loss_token_count", 0)) <= 0:
        failures.append("no EOS tokens included in the CE loss denominator after startup packing")
    if int(train_counts.get("lossable_attention_mismatch_tokens", 0)) > 0:
        failures.append("loss_mask contains tokens outside attention_mask in training batches")
    return failures


def _eos_supervision_startup_audit(
    config: ExperimentConfig,
    tokenizer,
    noise,
    rank: int,
    world_size: int,
    device,
    is_ddp: bool,
) -> dict[str, bool | float | int | str]:
    artifact_path = os.path.join(config.train.run_dir, "telemetry", "eos_supervision_audit.jsonl")
    enabled = _eos_supervision_audit_enabled(config)
    base_metrics: dict[str, bool | float | int | str] = {
        "eos_supervision_audit_enabled": bool(enabled),
        "eos_supervision_audit_artifact": "telemetry/eos_supervision_audit.jsonl",
        "eos_supervision_audit_path": artifact_path,
        "eos_supervision_audit_prepack_document_batches": int(
            config.diagnostics.eos_supervision_prepack_document_batches
        ),
        "eos_supervision_audit_packed_batches": int(config.diagnostics.eos_supervision_packed_batches),
        "eos_supervision_audit_forced_mask_t": float(config.diagnostics.eos_supervision_forced_mask_t),
        "eos_supervision_invariant_status": "disabled",
        "eos_supervision_invariant_error": "",
    }
    if not enabled:
        return base_metrics
    eos_token_id = tokenizer.get_eos_token_id()
    if eos_token_id is None:
        raise RuntimeError("EOS supervision invariant failed: tokenizer has no eos_token_id")

    saved_state = _rng_state(config)
    try:
        prepack_counts = _reduce_prepack_counts(
            _collect_eos_prepack_counts(config, tokenizer, rank, world_size),
            device,
            is_ddp,
        )
        packed_counts = _new_eos_supervision_counts()
        special_token_ids = {int(i) for i in tokenizer.get_special_token_ids()}
        audit_loader = build_train_loader(config, tokenizer, rank, world_size, device)
        for _ in range(config.diagnostics.eos_supervision_packed_batches):
            batch = next(audit_loader)
            z_audit, w_audit = noise.corrupt(
                batch.x_0,
                float(config.diagnostics.eos_supervision_forced_mask_t),
                maskable=batch.loss_mask,
            )
            del z_audit
            active = (w_audit > 0) & batch.loss_mask
            _update_eos_supervision_batch_counts(
                packed_counts,
                batch.x_0,
                batch.loss_mask,
                batch.attention_mask,
                special_token_ids,
                eos_token_id,
                active,
            )
        packed_counts = _reduce_eos_supervision_counts(packed_counts, device, is_ddp)
    finally:
        _restore_rng_state(saved_state, config)

    metrics: dict[str, bool | float | int | str] = {
        **base_metrics,
        **_eos_prepack_metrics("startup_", prepack_counts),
        **_eos_supervision_count_metrics("startup_postpack_", packed_counts),
    }
    failures: list[str] = []
    if int(prepack_counts.get("documents", 0)) <= 0:
        failures.append("no train documents observed before packing")
    if int(prepack_counts.get("eos_tokens", 0)) <= 0:
        failures.append("no EOS tokens observed before packing")
    if int(prepack_counts.get("eos_loss_tokens", 0)) <= 0:
        failures.append("no EOS loss tokens observed before packing")
    if int(packed_counts.get("eos_tokens", 0)) <= 0:
        failures.append("no EOS tokens observed after active packing")
    if int(packed_counts.get("eos_loss_tokens", 0)) <= 0:
        failures.append("no EOS tokens included in the CE loss denominator after active packing")
    if int(packed_counts.get("active_eos_target_tokens", 0)) <= 0:
        failures.append("no EOS tokens became masked targets in the forced startup corruption canary")
    if int(packed_counts.get("lossable_attention_mismatch_tokens", 0)) > 0:
        failures.append("loss_mask contains tokens outside attention_mask")

    metrics["eos_supervision_invariant_status"] = "failed" if failures else "ok"
    metrics["eos_supervision_invariant_error"] = "; ".join(failures)
    if rank == 0:
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        append_jsonl(artifact_path, {"event": "startup_eos_supervision_audit", **metrics})
    if failures:
        raise RuntimeError(f"EOS supervision invariant failed: {metrics['eos_supervision_invariant_error']}")
    return metrics


def _token_edit_branch_enabled_for_microbatch(config: ExperimentConfig, microbatch_index: int) -> bool:
    fraction = float(config.train.token_edit_corruption_batch_fraction)
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    index = int(microbatch_index)
    return math.floor((index + 1) * fraction) > math.floor(index * fraction)


def _population_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _assert_clean_label_smoothing_ar_config(config: ExperimentConfig, tokenizer) -> dict[str, bool | float | int | str]:
    if _causal_ar_label_smoothing(config) <= 0:
        return {}
    return {"ar_label_smoothing_vocab_size": int(tokenizer.get_vocab_size())}


def _hierarchical_output_telemetry(
    metadata: dict | None,
) -> dict[str, float | int | str]:
    if not metadata:
        return {"output_factorization": "flat", "output_class_count": 0}
    out: dict[str, float | int | str] = {
        "output_factorization": "hierarchical_frequency_decile",
        "output_class_count": int(metadata.get("output_class_count", 0)),
        "output_frequency_batches": int(metadata.get("frequency_batches", 0)),
        "output_frequency_observed_tokens": int(metadata.get("frequency_observed_tokens", 0)),
        "output_frequency_nonzero_token_types": int(metadata.get("frequency_nonzero_token_types", 0)),
        "output_special_token_count": int(metadata.get("special_token_count", 0)),
    }
    class_token_counts = metadata.get("class_token_counts", {})
    if isinstance(class_token_counts, dict):
        for key, value in class_token_counts.items():
            out[f"output_class_{key}_token_count"] = int(value)
    return out


def _bd3lm_telemetry(config: ExperimentConfig) -> dict[str, object]:
    if config.noise.type in {"mdlm", "bd3lm"} or config.noise.type == "uniform" and config.noise.time_conditioning:
        loss_normalization = "fast_dllm_token"
    else:
        loss_normalization = "active_token_mean"
    if _uses_block_causal_pilot(config):
        block_size = int(config.model.block_causal_pilot_size)
        num_blocks = (int(config.model.max_seq_len) + block_size - 1) // block_size
        return {
            "objective": "block_causal_pilot",
            "objective_label": "simple-block-causal-pilot",
            "bd3lm_enabled": False,
            "block_causal_pilot_enabled": True,
            "block_causal_pilot_not_true_bd3lm": True,
            "loss_normalization": loss_normalization,
            "block_causal_pilot_size": block_size,
            "block_causal_pilot_num_blocks": num_blocks,
        }
    if not _uses_bd3lm(config):
        return {
            "objective": config.noise.type,
            "bd3lm_enabled": False,
            "block_causal_pilot_enabled": False,
            "loss_normalization": loss_normalization,
        }
    clean_seq_len = int(config.model.max_seq_len)
    if _uses_bd3lm_causal_clean_stream(config):
        return block_conversion_objective_telemetry(config)
    return {
        "objective": "bd3lm",
        "objective_label": "bd3lm-independent-block-time",
        "bd3lm_enabled": True,
        "bd3lm_objective_time_sampling_contract_version": BD3LM_OBJECTIVE_TIME_SAMPLING_CONTRACT_VERSION,
        "loss_normalization": loss_normalization,
        "bd3lm_block_size": int(config.model.bd3lm_block_size),
        "bd3lm_clean_seq_len": clean_seq_len,
        "bd3lm_concat_seq_len": clean_seq_len * 2,
        "bd3lm_clean_stream_mode": config.model.bd3lm_clean_stream_mode,
        "bd3lm_corruption_time_sampling": "independent-per-segment-block",
        "bd3lm_loss_weighting": "inverse-block-time-over-clean-lossable-tokens",
        "bd3lm_document_mask_policy": "clean-visibility-intersection-in-every-concat-quadrant",
        "bd3lm_position_policy": "duplicate-clean-segment-reset-positions",
    }


def _bd3lm_noised_half_indices(active: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = active.shape
    row_idx, col_idx = active.reshape(batch_size, seq_len).nonzero(as_tuple=True)
    return row_idx.to(torch.long) * (2 * seq_len) + col_idx.to(torch.long)


def _fast_dllm_loss_normalization_scale(
    active_token_count: int,
    lossable_token_count: int,
    t_val: float | None,
    schedule_weight: bool,
) -> float:
    scale = float(active_token_count) / max(1, int(lossable_token_count))
    if schedule_weight and t_val is not None:
        scale /= max(float(t_val), 1e-8)
    return scale


def _training_targets_from_indices(x_0: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    return x_0.reshape(-1).index_select(0, target_indices)


def _soft_input_curriculum_mix(config: ExperimentConfig, step: int, total_steps: int) -> float:
    if not config.noise.soft_input_curriculum:
        return 0.0
    anneal_steps = max(
        1,
        round(max(1, int(total_steps)) * float(config.noise.soft_input_curriculum_anneal_fraction)),
    )
    start_mix = float(config.noise.soft_input_curriculum_start_mix)
    end_mix = float(config.noise.soft_input_curriculum_end_mix)
    if anneal_steps <= 1:
        return end_mix
    progress = min(1.0, max(0.0, float(step) / float(anneal_steps - 1)))
    return start_mix + (end_mix - start_mix) * progress


def _soft_input_curriculum_phase(config: ExperimentConfig, mix: float) -> str:
    if not config.noise.soft_input_curriculum:
        return "disabled"
    if mix <= 1e-6:
        return "hard"
    if mix >= 1.0 - 1e-6:
        return "soft"
    return "annealing"


def _uniform_support_mean_embedding(raw_model: DiffusionTransformer) -> torch.Tensor:
    vocab_size = int(raw_model.config.vocab_size)
    mask_token_id = int(raw_model.config.mask_token_id)
    embedding_weight = raw_model.wte.weight[:vocab_size]
    if 0 <= mask_token_id < vocab_size:
        if vocab_size <= 1:
            raise ValueError("uniform soft input curriculum requires at least two vocabulary tokens")
        return (embedding_weight.sum(dim=0) - embedding_weight[mask_token_id]) / float(vocab_size - 1)
    return embedding_weight.mean(dim=0)


def _build_uniform_soft_input_embeddings(
    raw_model: DiffusionTransformer,
    config: ExperimentConfig,
    x_0: torch.Tensor,
    z_t: torch.Tensor,
    loss_mask: torch.Tensor,
    t_val,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor | None, dict[str, float | str]]:
    mix = _soft_input_curriculum_mix(config, step, total_steps)
    phase = _soft_input_curriculum_phase(config, mix)
    if t_val is None:
        t_mean = 0.0
    elif torch.is_tensor(t_val):
        t_mean = float(t_val.detach().float().mean().item())
    else:
        t_mean = float(t_val)
    lossable_count = loss_mask.float().sum().detach().clamp_min(1.0)
    hard_changed = ((z_t != x_0) & loss_mask).float().sum().detach() / lossable_count
    lossable_fraction = loss_mask.float().mean().detach()
    stats: dict[str, float | str] = {
        "soft_input_curriculum_phase": phase,
        "soft_input_mix_prob": float(mix),
        "soft_input_hard_embedding_mix": float(1.0 - mix),
        "soft_input_expected_uniform_mass": float(mix * t_mean),
        "soft_input_sampled_t": float(t_mean),
        "soft_input_hard_changed_token_frac": float(hard_changed.item()),
        "soft_input_lossable_token_frac": float(lossable_fraction.item()),
    }
    if not config.noise.soft_input_curriculum or mix <= 0.0:
        return None, stats
    if config.noise.type != "uniform" or not config.noise.time_conditioning:
        raise ValueError("soft input curriculum is only implemented for time-conditioned uniform corruption")
    if t_val is None:
        raise ValueError("soft input curriculum requires a sampled uniform t value")

    if torch.is_tensor(t_val):
        t_tensor = t_val.to(device=x_0.device, dtype=raw_model.wte.weight.dtype).reshape(-1, 1, 1)
        if t_tensor.shape[0] != x_0.shape[0]:
            raise ValueError(f"tensor t must have one value per sequence, got {tuple(t_tensor.shape)}")
    else:
        t_tensor = torch.full(
            (x_0.shape[0], 1, 1),
            float(t_val),
            device=x_0.device,
            dtype=raw_model.wte.weight.dtype,
        )

    hard_embeddings = raw_model.wte(z_t)
    clean_embeddings = raw_model.wte(x_0)
    uniform_mean = _uniform_support_mean_embedding(raw_model).view(1, 1, -1)
    expected_embeddings = clean_embeddings * (1.0 - t_tensor) + uniform_mean * t_tensor
    expected_embeddings = torch.where(loss_mask[:, :, None], expected_embeddings, clean_embeddings)
    input_embeddings = hard_embeddings * (1.0 - float(mix)) + expected_embeddings * float(mix)
    return input_embeddings, stats


def _uniform_two_sample_mean_loss(
    model,
    config: ExperimentConfig,
    noise,
    x_0: torch.Tensor,
    z_t: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    target_indices: torch.Tensor,
    t_val: float,
    noise_level,
    lossable_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, dict[str, float | int | str]]:
    if config.noise.type != "uniform" or not config.noise.time_conditioning:
        raise ValueError("two-sample uniform estimator requires time-conditioned uniform corruption")
    if config.train.uniform_loss_estimator != "two_sample_mean":
        raise ValueError("two-sample uniform estimator called when train.uniform_loss_estimator is disabled")
    if target_indices.numel() == 0:
        raise ValueError("two-sample uniform estimator requires at least one target token")

    targets = _training_targets_from_indices(x_0, target_indices)
    loss_normalization_scale = _fast_dllm_loss_normalization_scale(
        int(target_indices.numel()),
        int(lossable_tokens),
        float(t_val),
        bool(config.noise.schedule_weight),
    )
    z_second, _ = noise.corrupt(x_0, float(t_val), maskable=loss_mask)
    raw_ces: list[torch.Tensor] = []
    scaled_losses: list[torch.Tensor] = []
    changed_fracs: list[float] = []
    lossable_count = loss_mask.float().sum().detach().clamp_min(1.0)
    for z_sample in (z_t, z_second):
        logits = _forward_training_model(
            model,
            config,
            z_sample,
            x_0,
            attention_mask,
            logits_indices=target_indices,
            return_aux_logits=False,
            noise_level=noise_level,
            return_hierarchical_logits=False,
            input_embeddings=None,
        )
        ce_values = F.cross_entropy(logits, targets, reduction="none")
        raw_ce = ce_values.mean()
        raw_ces.append(raw_ce)
        scaled_losses.append(raw_ce * float(loss_normalization_scale))
        changed_frac = ((z_sample != x_0) & loss_mask).float().sum().detach() / lossable_count
        changed_fracs.append(float(changed_frac.item()))

    raw_ce_stack = torch.stack(raw_ces)
    scaled_loss_stack = torch.stack(scaled_losses)
    raw_ce_mean = raw_ce_stack.mean()
    scaled_loss_mean = scaled_loss_stack.mean()
    raw_ce_var = raw_ce_stack.float().var(unbiased=False) if raw_ce_stack.numel() > 1 else raw_ce_stack.new_tensor(0.0)
    scaled_loss_var = (
        scaled_loss_stack.float().var(unbiased=False)
        if scaled_loss_stack.numel() > 1
        else scaled_loss_stack.new_tensor(0.0)
    )
    stats: dict[str, float | int | str] = {
        "uniform_loss_estimator": config.train.uniform_loss_estimator,
        "uniform_estimator_sample_count": int(raw_ce_stack.numel()),
        "uniform_estimator_loss_normalization_scale": float(loss_normalization_scale),
        "uniform_estimator_raw_ce_mean": float(raw_ce_mean.detach().item()),
        "uniform_estimator_scaled_loss_mean": float(scaled_loss_mean.detach().item()),
        "uniform_estimator_raw_ce_sample_variance": float(raw_ce_var.detach().item()),
        "uniform_estimator_scaled_loss_sample_variance": float(scaled_loss_var.detach().item()),
        "uniform_estimator_hard_changed_token_frac_mean": sum(changed_fracs) / len(changed_fracs),
    }
    for idx, (raw_ce, scaled_loss, changed_frac) in enumerate(zip(raw_ces, scaled_losses, changed_fracs, strict=True)):
        stats[f"uniform_estimator_raw_ce_sample{idx}"] = float(raw_ce.detach().item())
        stats[f"uniform_estimator_scaled_loss_sample{idx}"] = float(scaled_loss.detach().item())
        stats[f"uniform_estimator_hard_changed_token_frac_sample{idx}"] = float(changed_frac)
    return (
        scaled_loss_mean,
        raw_ce_mean,
        scaled_loss_mean,
        float(loss_normalization_scale),
        stats,
    )


def _forward_eval_logits(
    model,
    config: ExperimentConfig,
    z_t: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    noise_level=None,
    position_ids: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if _uses_bd3lm(config):
        concat, bd3lm_mask, position_ids = _bd3lm_concat_inputs(
            z_t,
            x_0,
            attention_mask,
            block_size=config.model.bd3lm_block_size,
            clean_stream_mode=config.model.bd3lm_clean_stream_mode,
            position_ids=position_ids,
        )
        logits = model(concat, attention_mask=bd3lm_mask, position_ids=position_ids)
        return logits[:, : z_t.shape[1]].contiguous()
    model_kwargs = {
        "attention_mask": attention_mask,
        "noise_level": noise_level if config.noise.time_conditioning else None,
    }
    if position_ids is not None:
        model_kwargs["position_ids"] = position_ids
    if target_lengths is not None:
        model_kwargs["target_lengths"] = target_lengths
    return model(z_t, **model_kwargs)


def _forward_eval_hierarchical_components(
    model,
    config: ExperimentConfig,
    z_t: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    noise_level=None,
    position_ids: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _uses_bd3lm(config):
        concat, bd3lm_mask, position_ids = _bd3lm_concat_inputs(
            z_t,
            x_0,
            attention_mask,
            block_size=config.model.bd3lm_block_size,
            clean_stream_mode=config.model.bd3lm_clean_stream_mode,
            position_ids=position_ids,
        )
        class_logits, token_logits = model(
            concat,
            return_hierarchical_logits=True,
            attention_mask=bd3lm_mask,
            position_ids=position_ids,
        )
        seq_len = z_t.shape[1]
        return class_logits[:, :seq_len].contiguous(), token_logits[:, :seq_len].contiguous()
    model_kwargs = {
        "return_hierarchical_logits": True,
        "attention_mask": attention_mask,
        "noise_level": noise_level if config.noise.time_conditioning else None,
    }
    if position_ids is not None:
        model_kwargs["position_ids"] = position_ids
    if target_lengths is not None:
        model_kwargs["target_lengths"] = target_lengths
    return model(z_t, **model_kwargs)


def _target_lengths_for_conditioning(
    config: ExperimentConfig,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None,
) -> torch.Tensor | None:
    if not _uses_target_length_conditioning(config):
        return None
    if attention_mask.dim() != 2:
        raise ValueError("target-length conditioning requires a 2D row attention mask")
    return target_eos_lengths(x_0, attention_mask, eos_token_id)


def _target_length_prior_metrics(
    prefix: str, counts: torch.Tensor, batches: int, observed_rows: int
) -> dict[str, float | int | bool]:
    counts_cpu = counts.detach().cpu().to(dtype=torch.float64)
    total = float(counts_cpu.sum().item())
    nonzero = (counts_cpu > 0).nonzero(as_tuple=True)[0].tolist()
    metrics: dict[str, float | int | bool] = {
        f"{prefix}target_length_conditioning_enabled": True,
        f"{prefix}target_length_prior_batches": int(batches),
        f"{prefix}target_length_prior_rows": int(observed_rows),
        f"{prefix}target_length_prior_total_count": int(total),
        f"{prefix}target_length_prior_hist_nonzero_bins": int(len(nonzero)),
    }
    if total > 0:
        lengths = torch.arange(1, counts_cpu.numel() + 1, dtype=torch.float64)
        metrics[f"{prefix}target_length_prior_mean"] = float((lengths * counts_cpu).sum().item() / total)
        metrics[f"{prefix}target_length_prior_min"] = int(nonzero[0] + 1) if nonzero else 0
        metrics[f"{prefix}target_length_prior_max"] = int(nonzero[-1] + 1) if nonzero else 0
    else:
        metrics[f"{prefix}target_length_prior_mean"] = 0.0
        metrics[f"{prefix}target_length_prior_min"] = 0
        metrics[f"{prefix}target_length_prior_max"] = 0
    for idx in nonzero:
        count = int(counts_cpu[idx].item())
        metrics[f"{prefix}target_length_prior_length_{idx + 1:04d}_count"] = count
        metrics[f"{prefix}target_length_prior_length_{idx + 1:04d}_rate"] = count / total if total else 0.0
    return metrics


def _estimate_target_length_prior(
    config: ExperimentConfig,
    tokenizer,
    batch_size: int,
    seq_len: int,
    device,
    batches: int,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    counts = torch.zeros(int(seq_len), dtype=torch.long, device=device)
    observed_rows = 0
    if not _uses_target_length_conditioning(config):
        return counts, {"target_length_conditioning_enabled": False}
    train_loader = make_dataloader(config.data, tokenizer, int(batch_size), int(seq_len), "train", device=device)
    eos_token_id = tokenizer.get_eos_token_id()
    for _ in range(int(batches)):
        batch = next(train_loader)
        lengths = target_eos_lengths(batch.x_0, batch.attention_mask, eos_token_id).clamp(1, int(seq_len))
        counts.scatter_add_(0, lengths - 1, torch.ones_like(lengths, dtype=torch.long))
        observed_rows += int(lengths.numel())
    metrics = _target_length_prior_metrics("", counts, int(batches), int(observed_rows))
    return counts, metrics


def _sample_target_lengths_from_prior(
    counts: torch.Tensor | None,
    n: int,
    length: int,
    device,
) -> torch.Tensor | None:
    if counts is None:
        return None
    counts = counts.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if counts.numel() == 0 or float(counts.sum().item()) <= 0:
        return torch.full((int(n),), int(length), device=device, dtype=torch.long)
    visible = counts[: int(length)]
    if visible.numel() > 0 and float(visible.sum().item()) > 0:
        sampled = torch.multinomial(visible, int(n), replacement=True) + 1
        return sampled.to(dtype=torch.long)
    sampled = torch.multinomial(counts, int(n), replacement=True) + 1
    return sampled.clamp(1, int(length)).to(dtype=torch.long)


def _corrupt_training_batch(
    config: ExperimentConfig,
    noise,
    x_0: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
):
    if config.noise.type == "mask_uniform":
        t_val = _sample_training_scalar_noise_time(config)
        z_t, w = noise.corrupt(x_0, t_val, maskable=loss_mask)
        return z_t, w, None, t_val

    if config.noise.type == "uniform":
        if config.noise.time_conditioning:
            t_val = _sample_training_scalar_noise_time(config)
            z_t, w = noise.corrupt(x_0, t_val, maskable=loss_mask)
            return z_t, w, t_val, t_val
        z_t, w = noise.corrupt_with_hierarchical_block_t(
            x_0,
            config.noise.t_min,
            config.noise.t_max,
            coarse_block_size=int(config.noise.uniform_t_coarse_block_size),
            fine_block_size=int(config.noise.uniform_t_block_size),
            maskable=loss_mask,
        )
        noise_level = 0.5 * (float(config.noise.t_min) + float(config.noise.t_max))
        return z_t, w, None, noise_level

    if _uses_bd3lm(config):
        t_val = sample_bd3lm_block_times(
            x_0.shape,
            block_size=int(config.model.bd3lm_block_size),
            t_min=float(config.noise.t_min),
            t_max=float(config.noise.t_max),
            device=x_0.device,
            position_ids=position_ids,
        )
        z_t, w = noise.corrupt(x_0, t_val, maskable=loss_mask)
        return z_t, w, t_val, None

    if config.noise.suffix_masking:
        t_val = _sample_training_tensor_noise_time(config, (x_0.shape[0],), x_0.device)
        z_t, w = noise.corrupt_suffix(x_0, t_val, maskable=loss_mask)
    elif _uses_per_token_noise_levels(config):
        z_t, w, t_val = noise.corrupt_per_token_t(
            x_0,
            float(config.noise.t_min),
            float(config.noise.t_max),
            maskable=loss_mask,
        )
    elif config.noise.complement_masking:
        t_val = _sample_training_scalar_noise_time(config)
        z_t, w = noise.corrupt_with_complement(x_0, t_val, maskable=loss_mask)
    elif _uses_mdlm_m2t_per_row_t(config):
        t_val = _sample_training_tensor_noise_time(config, (x_0.shape[0],), x_0.device)
        z_t, w = noise.corrupt(x_0, t_val, maskable=loss_mask)
    else:
        t_val = _sample_training_scalar_noise_time(config)
        z_t, w = noise.corrupt(x_0, t_val, maskable=loss_mask)
    return z_t, w, t_val, None


def _ordinary_token_candidate_ids(
    token_bytes: torch.Tensor,
    excluded_token_ids: set[int] | frozenset[int] | tuple[int, ...] = (),
) -> torch.Tensor:
    """Return sorted tokenizer IDs that decode to bytes and are not special IDs."""

    if token_bytes.ndim != 1:
        raise ValueError("token byte lengths must be a one-dimensional tensor")
    ordinary = token_bytes > 0
    for token_id in excluded_token_ids:
        token_id = int(token_id)
        if 0 <= token_id < ordinary.numel():
            ordinary[token_id] = False
    candidate_ids = ordinary.nonzero(as_tuple=True)[0].to(dtype=torch.long)
    if candidate_ids.numel() < 2:
        raise ValueError("token-edit corruption requires at least two ordinary tokenizer tokens")
    return candidate_ids


def _tokens_in_sorted_support(token_ids: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    if support.ndim != 1 or support.numel() == 0:
        raise ValueError("token-edit candidate IDs must be a nonempty one-dimensional tensor")
    positions = torch.searchsorted(support, token_ids)
    in_bounds = positions < support.numel()
    safe_positions = positions.clamp_max(support.numel() - 1)
    return in_bounds & (support[safe_positions] == token_ids)


def _token_edit_ordinary_eligible_mask(
    x_0: torch.Tensor,
    loss_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Return lossable targets whose clean IDs belong to the ordinary-token support."""

    if x_0.shape != loss_mask.shape:
        raise ValueError("token-edit clean tokens and loss mask must have the same shape")
    return loss_mask & _tokens_in_sorted_support(x_0, candidate_ids)


def _sample_non_original_tokens(
    originals: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    if originals.dtype != torch.long:
        raise ValueError("token-edit originals must be int64 token ids")
    if candidate_ids.device != originals.device:
        raise ValueError("token-edit candidate IDs and originals must be on the same device")
    if candidate_ids.dtype != torch.long or candidate_ids.ndim != 1 or candidate_ids.numel() < 2:
        raise ValueError("token-edit candidate IDs must contain at least two int64 IDs")
    if candidate_ids.numel() > 1 and not bool(torch.all(candidate_ids[1:] > candidate_ids[:-1]).item()):
        raise ValueError("token-edit candidate IDs must be sorted and unique")
    original_ranks = torch.searchsorted(candidate_ids, originals)
    if bool((original_ranks >= candidate_ids.numel()).any().item()) or not bool(
        torch.equal(candidate_ids[original_ranks], originals)
    ):
        raise ValueError("token-edit originals must belong to the ordinary-token candidate set")
    raw_ranks = torch.randint(
        0,
        candidate_ids.numel() - 1,
        originals.shape,
        device=originals.device,
        dtype=torch.long,
    )
    sampled_ranks = raw_ranks + (raw_ranks >= original_ranks).long()
    sampled = candidate_ids[sampled_ranks]
    if bool((sampled == originals).any().item()):
        raise RuntimeError("token-edit corruption sampled an unchanged token")
    return sampled


def _sample_legacy_non_mask_non_original_tokens(
    originals: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
) -> torch.Tensor:
    """Reproduce the archived fixed-ratio replacement support exactly."""

    if originals.dtype != torch.long:
        raise ValueError("token-edit originals must be int64 token ids")
    vocab_size = int(vocab_size)
    mask_token_id = int(mask_token_id)
    if vocab_size < 3:
        raise ValueError("token-edit corruption requires at least three vocabulary tokens")
    if 0 <= mask_token_id < vocab_size:
        if bool((originals == mask_token_id).any().item()):
            raise ValueError("token-edit corruption cannot corrupt existing mask-token targets")
        support_size = vocab_size - 2
        raw = torch.randint(0, support_size, originals.shape, device=originals.device, dtype=torch.long)
        low = torch.minimum(originals, torch.full_like(originals, mask_token_id))
        high = torch.maximum(originals, torch.full_like(originals, mask_token_id))
        sampled = raw + (raw >= low).long()
        sampled = sampled + (sampled >= high).long()
    else:
        raw = torch.randint(
            0,
            vocab_size - 1,
            originals.shape,
            device=originals.device,
            dtype=torch.long,
        )
        sampled = raw + (raw >= originals).long()
    if bool((sampled == originals).any().item()):
        raise RuntimeError("token-edit corruption sampled an unchanged token")
    if 0 <= mask_token_id < vocab_size and bool((sampled == mask_token_id).any().item()):
        raise RuntimeError("token-edit corruption sampled the mask token")
    return sampled


def _token_edit_loss_normalization_scale(
    active_token_count: int,
    lossable_token_count: int,
    edit_probability: float,
) -> float:
    """Return the inverse-propensity T2T scale without treating it as diffusion time."""

    probability = float(edit_probability)
    if not 0.0 < probability <= 1.0:
        raise ValueError("token-edit probability must be in (0, 1]")
    return float(active_token_count) / max(1, int(lossable_token_count)) / probability


def _corrupt_token_edit_training_batch(
    config: ExperimentConfig,
    x_0: torch.Tensor,
    loss_mask: torch.Tensor,
    candidate_ids: torch.Tensor | None,
    *,
    vocab_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | float | None, None]:
    matched_schedule = _uses_matched_token_edit_schedule(config)
    if matched_schedule:
        if candidate_ids is None:
            raise RuntimeError("matched token-edit corruption requires ordinary-token candidates")
        eligible_mask = _token_edit_ordinary_eligible_mask(x_0, loss_mask, candidate_ids)
    else:
        eligible_mask = loss_mask.bool()
    if not bool(eligible_mask.any().item()):
        if matched_schedule:
            raise RuntimeError("token-edit microbatch has no lossable ordinary-token targets")
        ratio = float(config.train.token_edit_corruption_ratio)
        return (
            x_0.clone(),
            torch.zeros_like(loss_mask, dtype=torch.float32),
            ratio,
            None,
        )
    if matched_schedule:
        edit_probability = _sample_training_tensor_noise_time(config, (x_0.shape[0],), x_0.device)
        edit_mask = (torch.rand(x_0.shape, device=x_0.device) < edit_probability[:, None]) & eligible_mask
        t_val: torch.Tensor | float | None = edit_probability
    else:
        if not _uses_legacy_fixed_token_edit_schedule(config):
            raise RuntimeError("unsupported token-edit corruption schedule reached training")
        ratio = float(config.train.token_edit_corruption_ratio)
        edit_mask = (torch.rand(x_0.shape, device=x_0.device) < ratio) & eligible_mask
        t_val = ratio
    if not edit_mask.any():
        eligible_indices = eligible_mask.reshape(-1).nonzero(as_tuple=True)[0]
        selected = eligible_indices[torch.randint(0, eligible_indices.numel(), (1,), device=x_0.device)]
        edit_mask = edit_mask.reshape(-1)
        edit_mask[selected] = True
        edit_mask = edit_mask.view_as(x_0)
    z_t = x_0.clone()
    if edit_mask.any():
        originals = x_0[edit_mask]
        if matched_schedule:
            z_t[edit_mask] = _sample_non_original_tokens(originals, candidate_ids)
        else:
            z_t[edit_mask] = _sample_legacy_non_mask_non_original_tokens(
                originals,
                vocab_size,
                mask_token_id,
            )
    return z_t, edit_mask.float(), t_val, None


def _corrupt_eval_batch(
    config: ExperimentConfig,
    noise,
    x_eval: torch.Tensor,
    loss_mask: torch.Tensor,
    t_val: float | None = None,
):
    if config.noise.type == "mask_uniform":
        sampled_t = _sample_training_scalar_noise_time(config) if t_val is None else t_val
        z_eval, w_eval = noise.corrupt(x_eval, sampled_t, maskable=loss_mask)
        return z_eval, w_eval, None, sampled_t

    if config.noise.type == "uniform":
        if config.noise.time_conditioning:
            sampled_t = _sample_training_scalar_noise_time(config) if t_val is None else t_val
            z_eval, w_eval = noise.corrupt(x_eval, sampled_t, maskable=loss_mask)
            return z_eval, w_eval, None, sampled_t
        if t_val is None:
            z_eval, w_eval = noise.corrupt_with_hierarchical_block_t(
                x_eval,
                config.noise.t_min,
                config.noise.t_max,
                coarse_block_size=int(config.noise.uniform_t_coarse_block_size),
                fine_block_size=int(config.noise.uniform_t_block_size),
                maskable=loss_mask,
            )
            noise_level = 0.5 * (float(config.noise.t_min) + float(config.noise.t_max))
            return z_eval, w_eval, None, noise_level
        z_eval, w_eval = noise.corrupt_with_hierarchical_block_t(
            x_eval,
            t_val,
            t_val,
            coarse_block_size=int(config.noise.uniform_t_coarse_block_size),
            fine_block_size=int(config.noise.uniform_t_block_size),
            maskable=loss_mask,
        )
        return z_eval, w_eval, None, t_val

    if config.noise.suffix_masking:
        sampled_t = (
            _sample_training_tensor_noise_time(config, (x_eval.shape[0],), x_eval.device) if t_val is None else t_val
        )
        z_eval, w_eval = noise.corrupt_suffix(x_eval, sampled_t, maskable=loss_mask)
    else:
        sampled_t = _sample_training_scalar_noise_time(config) if t_val is None else t_val
        z_eval, w_eval = noise.corrupt(x_eval, sampled_t, maskable=loss_mask)
    return z_eval, w_eval, sampled_t, None
