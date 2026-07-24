import copy
import datetime
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
import uuid
from contextlib import ExitStack, nullcontext, suppress
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig, config_from_dict, to_dict, write_config
from .data import (
    DLMDataBatch,
    ParquetTextSource,
    build_frequency_token_classes,
    build_token_bytes,
    build_tokenizer,
    build_train_loader,
    build_val_loader,
    encode_document_batch,
    make_dataloader,
)
from .data_identity import build_data_identity
from .eval import (
    NELBO_SEQUENCE_RECORD_SCHEMA_VERSION,
    evaluate_ar_next_token_conditional_rank_suffix,
    evaluate_ar_validation_prefix_continuations,
    evaluate_argmax_basin_diagnostic,
    evaluate_bos_only_first_token,
    evaluate_conditional_generation_metrics,
    evaluate_conditional_rank_rollout,
    evaluate_conditional_rank_suffix,
    evaluate_generation_metrics,
    evaluate_nelbo,
    evaluate_reconstruction_metrics,
    target_eos_lengths,
)
from .model import (
    DiffusionConfig,
    DiffusionTransformer,
    bd3lm_attention_mask,
    duplicated_position_ids,
    embedding_head_init_std,
    estimate_non_embedding_params,
    residual_output_projection_init_std,
    rms_norm,
)
from .noise import build_noise, sample_positive_uniform_scalar, sample_positive_uniform_tensor
from .sampling import (
    sample_ar,
    sample_bd3lm,
    sample_block_causal_pilot,
    sample_fhs,
    sample_mask_uniform,
    sample_mdlm,
    sample_mdlm_from_prefixes,
    sample_remdm,
    sample_uniform,
)
from .telemetry import (
    append_jsonl,
    close_tensorboard_writer,
    flush_tensorboard_writer,
    write_json_atomic,
)
from .telemetry import (
    build_tensorboard_writer as _build_tensorboard_writer,
)
from .telemetry import (
    resolve_tensorboard_dir as _resolve_tensorboard_dir,
)
from .telemetry import (
    write_tensorboard_scalars as _write_tensorboard_scalars,
)

TOP1_PROB_HIST_BINS = 10
LOGIT_SCALE_HIST_BINS = 16
EOS_HAZARD_RELATIVE_POSITION_DECILES = 10
POSITION_DECILE_BALANCED_LOSS_DECILES = 10
TOKEN_EDIT_CORRUPTION_DECILES = 7
PER_TOKEN_NOISE_LEVEL_HIST_BINS = 10
CPU_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ARROW_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
)
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})
NUMERIC_RUNTIME_TELEMETRY_ATTR = "_nanodllm_numeric_runtime_telemetry"
RESUME_CONTRACT_VERSION = 1
RESUME_BATCH_DIGEST_VERSION = 1
RESUME_RNG_STATE_VERSION = 1
RESUME_TRAINER_STATE_VERSION = 1
RESUME_TRAJECTORY_VERSION = 1
RESUME_TRAINER_GROUP_FIELDS = {
    "lineage": (
        "ar_boundary_counts",
        "boundary_telemetry",
        "row_telemetry",
        "eos_supervision_train_counts",
        "block_causal_visible_prefix_frac_sum",
        "block_causal_visible_prefix_count",
        "train_m2t_ce_sum",
        "train_m2t_loss_sum",
        "train_m2t_loss_count",
        "train_m2t_microbatches",
        "train_m2t_active_tokens",
        "train_t2t_ce_sum",
        "train_t2t_loss_sum",
        "train_t2t_loss_count",
        "train_t2t_microbatches",
        "train_t2t_active_tokens",
        "train_length_head_accumulator",
        "train_token_edit_reconstruction_accumulator",
        "train_per_token_noise_level_hist_counts",
        "train_per_token_effective_masked_fraction_sum",
        "train_per_token_effective_masked_fraction_count",
        "train_per_token_effective_masked_fraction_min",
        "train_per_token_effective_masked_fraction_max",
        "grad_clip_event_count",
        "grad_clip_observed_steps",
    ),
    "interval": (
        "smooth_loss",
        "smooth_count",
        "m2t_ce_sum",
        "m2t_loss_sum",
        "m2t_loss_count",
        "m2t_microbatches",
        "m2t_active_tokens",
        "t2t_ce_sum",
        "t2t_loss_sum",
        "t2t_loss_count",
        "t2t_microbatches",
        "t2t_active_tokens",
    ),
    "last": (
        "loss_value",
        "main_ce",
        "main_loss",
        "confidence_penalty",
        "confidence_penalty_loss",
        "z_loss",
        "z_loss_loss",
        "eos_hazard_calibration_loss",
        "eos_hazard_loss",
        "eos_hazard_accumulator",
        "length_head_accumulator",
        "position_decile_ce_accumulator",
        "token_edit_reconstruction_accumulator",
        "m2t_ce",
        "m2t_loss",
        "m2t_microbatches",
        "m2t_active_tokens",
        "t2t_ce",
        "t2t_loss",
        "t2t_microbatches",
        "t2t_active_tokens",
        "grad_norm",
        "grad_clipped",
        "grad_clip_ratio",
        "active_tokens",
        "lossable_tokens",
        "valid_tokens",
        "content_tokens",
        "content_bytes",
        "class_ce",
        "within_class_ce",
        "ar_unsmoothed_ce",
        "second_moment_norm",
        "update_norm",
        "update_grad_ratio",
        "entropy_mean",
        "top1_prob_mean",
        "argmax_mode_frac",
        "top1_prob_hist_counts",
        "entropy_hist_counts",
        "logsumexp_hist_counts",
        "logit_norm_hist_counts",
        "logit_norm_mean",
        "logit_norm_min",
        "logit_norm_max",
        "logit_rms_mean",
        "logsumexp_mean",
        "logsumexp_min",
        "logsumexp_max",
        "soft_input_stats",
        "uniform_estimator_stats",
        "per_token_noise_level_hist_counts",
        "per_token_effective_masked_fraction_sum",
        "per_token_effective_masked_fraction_count",
        "per_token_effective_masked_fraction_min",
        "per_token_effective_masked_fraction_max",
        "gradient_component_norms",
        "update_component_norms",
    ),
    "diagnostics": (
        "initial_parameter_norm_metrics",
        "initial_gradient_metrics",
        "initial_update_metrics",
        "initial_lm_head_row_norm_metrics",
        "early_lm_head_row_norm_metrics",
        "early_training_diagnostics",
        "early_residual_diagnostics",
        "fixed_ce_trace_records",
        "last_fixed_ce_trace_record",
    ),
}
RESUME_BATCH_TENSOR_FIELDS = (
    "x_0",
    "loss_mask",
    "attention_mask",
    "position_ids",
    "document_boundary_mask",
    "document_start_mask",
)
CGROUP_ROOT = Path("/sys/fs/cgroup")
EOS_SUPERVISION_PREPACK_DOCUMENT_BATCHES = 8
EOS_SUPERVISION_PACKED_BATCHES = 64
EOS_SUPERVISION_FORCED_MASK_T = 1.0
REAL_PACKING_AUDIT_BATCHES = 64
REAL_PACKING_AUDIT_DOCS_PER_ROW_MAX_BUCKET = 8
REAL_PACKING_TARGET_CE_AUDIT_FIXED_MASK_T = 1.0
REAL_PACKING_TARGET_CE_AUDIT_EXAMPLE_LIMIT = 8
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
REAL_PACKING_AUDIT_SUM_KEYS = (
    "microbatches",
    "rows",
    "row_slots",
    "packed_tokens",
    "effective_tokens",
    "lossable_tokens",
    "attention_drop_tail_tokens",
    "lossable_attention_mismatch_tokens",
    "packed_eos_tokens",
    "active_eos_tokens",
    "lossable_eos_tokens",
    "pre_cap_eos_tokens",
    "pre_cap_active_eos_tokens",
    "pre_cap_lossable_eos_tokens",
    "final_slot_eos_tokens",
    "final_slot_active_eos_tokens",
    "final_slot_lossable_eos_tokens",
    "eos_rows",
    "active_eos_rows",
    "lossable_eos_rows",
    "pre_cap_eos_rows",
    "pre_cap_active_eos_rows",
    "pre_cap_lossable_eos_rows",
    "final_slot_eos_rows",
    "final_slot_active_eos_rows",
    "final_slot_lossable_eos_rows",
    "row_cap_split_rows",
    "attention_drop_tail_rows",
    "documents_per_row_sum",
    "document_boundary_count",
    "active_document_boundary_count",
    "lossable_document_boundary_count",
    "pre_cap_document_boundary_count",
    "final_slot_document_boundary_count",
    "packer_truncation_count",
    "packer_drop_tail_document_count",
    "packer_drop_tail_token_count",
    "target_ce_lossable_label_count",
    "target_ce_input_label_match_count",
    "target_ce_input_label_mismatch_count",
    "target_ce_target_count",
    "target_ce_framework_target_mismatch_count",
    "target_ce_mask_target_count",
    "target_ce_pad_target_count",
    "target_ce_eos_loss_target_count",
    "target_ce_active_mask_token_count",
    "target_ce_noised_active_non_mask_count",
    "target_ce_noised_inactive_changed_count",
    "target_ce_pad_token_count",
    "target_ce_pad_excluded_count",
    "target_ce_pad_loss_token_count",
    "target_ce_pad_attention_token_count",
    "target_ce_loss_mask_attention_mismatch_count",
    "target_ce_eos_boundary_position_count",
    "target_ce_eos_boundary_lossable_position_count",
    "target_ce_eos_boundary_target_count",
    "target_ce_eos_boundary_attention_not_lossable_count",
    "target_ce_batch_with_target_count",
    "target_ce_batch_with_eos_target_count",
)
REAL_PACKING_AUDIT_MIN_KEYS = (
    "effective_sequence_length_min",
    "lossable_sequence_length_min",
    "documents_per_row_min",
)
REAL_PACKING_AUDIT_MAX_KEYS = (
    "effective_sequence_length_max",
    "lossable_sequence_length_max",
    "documents_per_row_max",
)


def _read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _positive_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _cpu_cgroup_telemetry(root: Path = CGROUP_ROOT) -> dict[str, object]:
    """Observe the effective cgroup CPU quota and cpuset without changing them."""
    version: int | None = None
    quota_status = "unavailable"
    quota_us: int | None = None
    period_us: int | None = None

    cpu_max = _read_optional_text(root / "cpu.max")
    if cpu_max is not None:
        version = 2
        parts = cpu_max.split()
        if len(parts) == 2:
            period_us = _positive_int(parts[1])
            if parts[0] == "max" and period_us is not None:
                quota_status = "unlimited"
            else:
                quota_us = _positive_int(parts[0])
                if quota_us is not None and period_us is not None:
                    quota_status = "limited"
                else:
                    quota_us = None
                    period_us = None
                    quota_status = "invalid"
        else:
            quota_status = "invalid"
    else:
        for cpu_root in (root / "cpu", root / "cpu,cpuacct", root / "cpuacct,cpu"):
            raw_quota = _read_optional_text(cpu_root / "cpu.cfs_quota_us")
            raw_period = _read_optional_text(cpu_root / "cpu.cfs_period_us")
            if raw_quota is None and raw_period is None:
                continue
            version = 1
            try:
                parsed_quota = int(raw_quota) if raw_quota is not None else None
            except ValueError:
                parsed_quota = None
            period_us = _positive_int(raw_period) if raw_period is not None else None
            if parsed_quota == -1 and period_us is not None:
                quota_status = "unlimited"
            elif parsed_quota is not None and parsed_quota > 0 and period_us is not None:
                quota_status = "limited"
                quota_us = parsed_quota
            else:
                quota_status = "invalid"
                period_us = None
            break

    cpuset_cpus = None
    cpuset_source = None
    cpuset_candidates = (
        (root / "cpuset.cpus.effective", "cgroup_v2_cpuset.cpus.effective"),
        (root / "cpuset.cpus", "cgroup_v2_cpuset.cpus"),
        (root / "cpuset" / "cpuset.effective_cpus", "cgroup_v1_cpuset.effective_cpus"),
        (root / "cpuset" / "cpuset.cpus", "cgroup_v1_cpuset.cpus"),
    )
    for path, source in cpuset_candidates:
        value = _read_optional_text(path)
        if value:
            cpuset_cpus = value
            cpuset_source = source
            break

    quota_cpu_count = None
    if quota_status == "limited" and quota_us is not None and period_us is not None:
        quota_cpu_count = quota_us / period_us
    return {
        "version": version,
        "cpu_quota_status": quota_status,
        "cpu_quota_us": quota_us,
        "cpu_period_us": period_us,
        "cpu_quota_cpu_count": quota_cpu_count,
        "cpuset_cpus": cpuset_cpus,
        "cpuset_source": cpuset_source,
    }


def _scheduler_affinity_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _cpu_threading_telemetry() -> dict[str, object]:
    """Observe CPU threading and allocation state without changing runtime behavior."""
    return {
        "os_cpu_count": os.cpu_count(),
        "scheduler_affinity_cpu_count": _scheduler_affinity_cpu_count(),
        "cgroup": _cpu_cgroup_telemetry(),
        "torch_intra_op_threads": int(torch.get_num_threads()),
        "torch_inter_op_threads": int(torch.get_num_interop_threads()),
        "environment": {key: os.environ.get(key) for key in CPU_THREAD_ENV_KEYS},
    }


def _distributed_cuda_memory_telemetry(
    *,
    device_type: str,
    rank: int,
    local_rank: int,
    world_size: int,
    is_ddp: bool,
) -> dict[str, object]:
    """Collect comparable post-training CUDA memory observations from every rank."""
    if device_type != "cuda" or not torch.cuda.is_available():
        return {
            "enabled": False,
            "scope": "not_cuda",
            "rank_records": [],
            "minimum_free_mib": None,
            "minimum_free_fraction": None,
            "minimum_peak_reserved_headroom_fraction": None,
            "maximum_peak_allocated_mib": None,
            "maximum_peak_reserved_mib": None,
        }

    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info(local_rank)
    mib = 1024 * 1024
    local_record: dict[str, object] = {
        "rank": int(rank),
        "local_rank": int(local_rank),
        "device_index": int(torch.cuda.current_device()),
        "device_name": str(torch.cuda.get_device_name(local_rank)),
        "free_mib": float(free_bytes / mib),
        "total_mib": float(total_bytes / mib),
        "free_fraction": float(free_bytes / total_bytes) if total_bytes > 0 else 0.0,
        "current_allocated_mib": float(torch.cuda.memory_allocated(local_rank) / mib),
        "current_reserved_mib": float(torch.cuda.memory_reserved(local_rank) / mib),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(local_rank) / mib),
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved(local_rank) / mib),
    }
    local_record["peak_reserved_headroom_fraction"] = (
        max(0.0, float(local_record["total_mib"]) - float(local_record["peak_reserved_mib"]))
        / float(local_record["total_mib"])
        if float(local_record["total_mib"]) > 0
        else 0.0
    )
    if is_ddp:
        gathered: list[object | None] = [None] * int(world_size)
        dist.all_gather_object(gathered, local_record)
    else:
        gathered = [local_record]

    if len(gathered) != int(world_size):
        raise RuntimeError(f"gathered {len(gathered)} CUDA memory records for world size {world_size}")
    rank_records: list[dict[str, object]] = []
    seen_ranks: set[int] = set()
    for index, record in enumerate(gathered):
        if not isinstance(record, dict):
            raise RuntimeError(f"CUDA memory rank record {index} is not an object")
        record_rank = record.get("rank")
        if not isinstance(record_rank, int) or isinstance(record_rank, bool):
            raise RuntimeError(f"CUDA memory rank record {index} has invalid rank={record_rank!r}")
        if record_rank in seen_ranks:
            raise RuntimeError(f"CUDA memory telemetry contains duplicate rank {record_rank}")
        seen_ranks.add(record_rank)
        rank_records.append(record)
    if seen_ranks != set(range(int(world_size))):
        raise RuntimeError(
            f"CUDA memory telemetry ranks {sorted(seen_ranks)} differ from expected {list(range(int(world_size)))}"
        )
    rank_records.sort(key=lambda item: int(item["rank"]))
    return {
        "enabled": True,
        "scope": "all_ranks_post_training_before_final_evaluation_and_checkpoint",
        "rank_records": rank_records,
        "minimum_free_mib": min(float(record["free_mib"]) for record in rank_records),
        "minimum_free_fraction": min(float(record["free_fraction"]) for record in rank_records),
        "minimum_peak_reserved_headroom_fraction": min(
            float(record["peak_reserved_headroom_fraction"]) for record in rank_records
        ),
        "maximum_peak_allocated_mib": max(float(record["peak_allocated_mib"]) for record in rank_records),
        "maximum_peak_reserved_mib": max(float(record["peak_reserved_mib"]) for record in rank_records),
    }


def _training_diagnostics_module():
    from . import training_diagnostics

    return training_diagnostics


def _training_diagnostic_requested(config: ExperimentConfig) -> bool:
    return bool(config.train.strict_mdlm_all_token_bayes_audit_only)


def _is_causal(config: ExperimentConfig) -> bool:
    return bool(getattr(config.model, "causal", False))


def _uses_block_causal_pilot(config: ExperimentConfig) -> bool:
    return int(getattr(config.model, "block_causal_pilot_size", 0)) > 0


def _uses_content_budget_segmented_packing(config: ExperimentConfig) -> bool:
    return bool(getattr(config.data, "content_budget_segmented_packing", False))


def _batch_model_attention_mask(batch, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    base_mask = batch.attention_mask if attention_mask is None else attention_mask
    boundary_mask = getattr(batch, "document_boundary_mask", None)
    if boundary_mask is None:
        return base_mask
    if base_mask.dim() != 2:
        raise ValueError(f"base attention mask must be [batch, seq], got {tuple(base_mask.shape)}")
    b, t = base_mask.shape
    if boundary_mask.shape != (b, t, t):
        raise ValueError(f"document boundary mask must have shape {(b, t, t)}, got {tuple(boundary_mask.shape)}")
    valid_q = base_mask[:, :, None]
    valid_k = base_mask[:, None, :]
    return boundary_mask & valid_q & valid_k


def _block_causal_pilot_attention_mask(
    attention_mask_2d: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("block_causal_pilot block_size must be positive")
    _batch_size, seq_len = attention_mask_2d.shape
    positions = torch.arange(seq_len, device=attention_mask_2d.device)
    block_ids = positions // block_size
    causal = block_ids[None, :] <= block_ids[:, None]
    base = attention_mask_2d[:, :, None] & attention_mask_2d[:, None, :]
    return base & causal[None, :, :]


def _apply_block_causal_pilot_training(
    config: ExperimentConfig,
    x_0: torch.Tensor,
    z_t: torch.Tensor,
    w: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_size <= 0:
        raise ValueError("block_causal_pilot block_size must be positive")
    batch_size, seq_len = x_0.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    positions = torch.arange(seq_len, device=x_0.device).view(1, seq_len)
    position_block_ids = (positions // block_size).expand(batch_size, -1)
    valid_lossable = loss_mask & attention_mask
    block_lossable_counts = torch.zeros((batch_size, num_blocks), device=x_0.device, dtype=torch.long)
    block_lossable_counts.scatter_add_(1, position_block_ids, valid_lossable.to(torch.long))
    valid_blocks = block_lossable_counts > 0
    block_scores = torch.rand((batch_size, num_blocks), device=x_0.device).masked_fill(~valid_blocks, -1.0)
    current_blocks = block_scores.argmax(dim=1)

    block_starts = current_blocks.view(batch_size, 1) * block_size
    block_ends = torch.min(block_starts + block_size, torch.tensor(seq_len, device=x_0.device))
    in_current_block = (positions >= block_starts) & (positions < block_ends)

    z_t = torch.where(in_current_block, z_t, x_0)
    w = w * in_current_block.float()

    model_attention_mask = _block_causal_pilot_attention_mask(attention_mask, block_size)

    visible_prefix_tokens = (attention_mask & (positions < block_starts)).sum(dim=1)
    visible_prefix_frac = visible_prefix_tokens.float() / attention_mask.sum(dim=1).clamp_min(1).float()

    return z_t, w, model_attention_mask, in_current_block, visible_prefix_frac


def init_distributed(device_type: str, run_dir: str, init_single_rank_process_group: bool = False):
    is_ddp = "RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl" if device_type == "cuda" else "gloo")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(device_type, local_rank) if device_type == "cuda" else torch.device(device_type)
        if device_type == "cuda":
            torch.cuda.set_device(device)
    elif init_single_rank_process_group:
        rank = local_rank = 0
        world_size = 1
        device = torch.device(device_type, local_rank) if device_type == "cuda" else torch.device(device_type)
        rendezvous_dir = os.path.join(os.path.abspath(run_dir), ".single_rank_rendezvous")
        os.makedirs(rendezvous_dir, exist_ok=True)
        rendezvous_path = os.path.join(rendezvous_dir, f"pg_{os.getpid()}_{uuid.uuid4().hex}")
        dist.init_process_group(
            backend="nccl" if device_type == "cuda" else "gloo",
            init_method=f"file://{rendezvous_path}",
            rank=0,
            world_size=1,
            timeout=datetime.timedelta(seconds=60),
        )
        if device_type == "cuda":
            torch.cuda.set_device(device)
        is_ddp = True
    else:
        rank = local_rank = 0
        world_size = 1
        device = torch.device(device_type)
    return is_ddp, rank, local_rank, world_size, device


def print0(rank: int, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def get_lr_multiplier(
    step: int, total_steps: int, warmup_steps: int, warmdown_ratio: float, final_lr_frac: float
) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    warmdown_steps = round(warmdown_ratio * total_steps)
    warmdown_start = total_steps - warmdown_steps
    if step < warmdown_start:
        return 1.0
    progress = (total_steps - step) / max(1, warmdown_steps)
    return max(final_lr_frac, progress)


def _resolved_warmup_steps(config: ExperimentConfig, total_steps: int) -> int:
    if config.optim.warmup_ratio >= 0:
        return round(config.optim.warmup_ratio * total_steps)
    return config.optim.warmup_steps


def _lr_schedule_telemetry(config: ExperimentConfig, total_steps: int, warmup_steps: int) -> dict:
    warmdown_steps = round(config.optim.warmdown_ratio * total_steps)
    warmdown_start_step = max(0, total_steps - warmdown_steps)
    probe_steps = sorted(
        {
            0,
            max(0, warmup_steps - 1),
            min(max(0, total_steps - 1), warmup_steps),
            max(0, warmdown_start_step - 1),
            min(max(0, total_steps - 1), warmdown_start_step),
            max(0, total_steps - 1),
        }
    )
    trace = [
        {
            "step": step + 1,
            "lr_multiplier": get_lr_multiplier(
                step,
                total_steps,
                warmup_steps,
                config.optim.warmdown_ratio,
                config.optim.final_lr_frac,
            ),
        }
        for step in probe_steps
        if 0 <= step < total_steps
    ]
    return {
        "warmup_steps_config": int(config.optim.warmup_steps),
        "warmup_ratio_config": float(config.optim.warmup_ratio),
        "warmup_steps": int(warmup_steps),
        "warmup_fraction": warmup_steps / max(1, total_steps),
        "warmdown_ratio": float(config.optim.warmdown_ratio),
        "warmdown_steps": int(warmdown_steps),
        "warmdown_start_step": int(warmdown_start_step + 1),
        "final_lr_frac": float(config.optim.final_lr_frac),
        "trace_artifact": "train_log.jsonl",
        "trace": trace,
    }


def _lr_schedule_metric_fields(lr_schedule: dict) -> dict[str, bool | float | int | str]:
    return {
        f"lr_schedule_{key}": value for key, value in lr_schedule.items() if isinstance(value, (bool, float, int, str))
    }


def _fixed_ce_trace_steps(total_steps: int, warmup_steps: int) -> list[int]:
    candidates = {
        1,
        max(1, min(total_steps, warmup_steps)),
        max(1, min(total_steps, round(total_steps * 0.5))),
        max(1, total_steps),
    }
    return sorted(candidates)


def _resolved_optimizer_lr(config: ExperimentConfig, optimizer_kind: str, batch_lr_scale: float) -> float:
    if optimizer_kind == "muon":
        lr = config.optim.lr if config.optim.muon_lr < 0 else config.optim.muon_lr
    elif optimizer_kind == "adamw":
        lr = config.optim.lr if config.optim.adamw_lr < 0 else config.optim.adamw_lr
    else:
        raise ValueError(f"unknown optimizer kind {optimizer_kind!r}")
    return lr * batch_lr_scale


def _optimizer_base_lrs(config: ExperimentConfig, batch_lr_scale: float) -> tuple[float, ...]:
    if config.optim.type == "muon":
        return (
            _resolved_optimizer_lr(config, "muon", batch_lr_scale),
            _resolved_optimizer_lr(config, "adamw", batch_lr_scale),
        )
    return (_resolved_optimizer_lr(config, "adamw", batch_lr_scale),)


def _set_optimizer_base_lrs(optimizers, config: ExperimentConfig, batch_lr_scale: float) -> None:
    base_lrs = _optimizer_base_lrs(config, batch_lr_scale)
    optimizers = tuple(optimizers)
    if len(base_lrs) != len(optimizers):
        raise ValueError(f"expected {len(base_lrs)} optimizers for {config.optim.type}, got {len(optimizers)}")
    for optimizer, base_lr in zip(optimizers, base_lrs, strict=True):
        for group in optimizer.param_groups:
            group["base_lr"] = base_lr
            group["lr"] = base_lr


def _set_optimizer_lrs(optimizers, lr_multiplier: float) -> None:
    for group in _iter_optimizer_param_groups(optimizers):
        group["lr"] = float(group["base_lr"]) * lr_multiplier


def make_optimizer(model: nn.Module, config: ExperimentConfig, batch_lr_scale: float):
    optim = config.optim
    base_lrs = _optimizer_base_lrs(config, batch_lr_scale)
    if optim.type == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("optim.type='muon' requires a PyTorch build with torch.optim.Muon")
        muon_lr, adamw_lr = base_lrs
        muon_params, adamw_decay, adamw_no_decay = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() == 2 and name.startswith("blocks."):
                muon_params.append(p)
            elif p.dim() >= 2:
                adamw_decay.append(p)
            else:
                adamw_no_decay.append(p)
        muon_kwargs = {
            "lr": muon_lr,
            "weight_decay": optim.weight_decay,
            "momentum": optim.muon_momentum,
            "ns_steps": optim.muon_ns_steps,
        }
        if optim.muon_adjust_lr_fn:
            muon_kwargs["adjust_lr_fn"] = optim.muon_adjust_lr_fn
        return (
            torch.optim.Muon(muon_params, **muon_kwargs),
            torch.optim.AdamW(
                [
                    {"params": adamw_decay, "weight_decay": optim.weight_decay},
                    {"params": adamw_no_decay, "weight_decay": 0.0},
                ],
                lr=adamw_lr,
                betas=(optim.adam_beta1, optim.adam_beta2),
                eps=optim.adam_eps,
                fused=config.runtime.device_type == "cuda",
            ),
        )

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and not name.endswith("wte.weight"):
            decay.append(p)
        else:
            no_decay.append(p)
    return (
        torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": optim.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=base_lrs[0],
            betas=(optim.adam_beta1, optim.adam_beta2),
            eps=optim.adam_eps,
            fused=config.runtime.device_type == "cuda",
        ),
    )


def _optimizer_state_dict(optimizers) -> dict:
    optimizers = tuple(optimizers)
    if len(optimizers) == 1:
        return optimizers[0].state_dict()
    return {
        "format": "optimizer_list",
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
    }


def _load_optimizer_state_dict(optimizers, state: dict) -> None:
    optimizers = tuple(optimizers)
    if "optimizers" in state:
        if len(state["optimizers"]) != len(optimizers):
            raise ValueError(
                f"checkpoint has {len(state['optimizers'])} optimizer states, "
                f"but this run built {len(optimizers)} optimizers"
            )
        for optimizer, optimizer_state in zip(optimizers, state["optimizers"], strict=True):
            optimizer.load_state_dict(optimizer_state)
        return
    if len(optimizers) != 1:
        raise ValueError("single-optimizer checkpoint cannot be loaded into a multi-optimizer run")
    optimizers[0].load_state_dict(state)


def _iter_optimizer_param_groups(optimizers):
    for optimizer in optimizers:
        yield from optimizer.param_groups


def _optimizer_lr_metrics(optimizers) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for optimizer_idx, optimizer in enumerate(optimizers):
        group_lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
        if not group_lrs:
            continue
        metrics[f"optimizer_{optimizer_idx}_lr"] = group_lrs[0]
        if len(set(group_lrs)) > 1:
            metrics[f"optimizer_{optimizer_idx}_lr_min"] = min(group_lrs)
            metrics[f"optimizer_{optimizer_idx}_lr_max"] = max(group_lrs)
    return metrics


def _primary_optimizer_lr(optimizers) -> float:
    for group in _iter_optimizer_param_groups(optimizers):
        return float(group.get("lr", 0.0))
    return 0.0


def _zero_grad(optimizers) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def _unscale_optimizers(scaler, optimizers) -> None:
    for optimizer in optimizers:
        scaler.unscale_(optimizer)


def _step_optimizers(scaler, optimizers) -> None:
    for optimizer in optimizers:
        scaler.step(optimizer)


def _check_gpu_preflight(config: ExperimentConfig, rank: int) -> None:
    if config.runtime.device_type != "cuda" or config.runtime.min_free_gpu_mib <= 0 or rank != 0:
        return
    import subprocess

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    free_mib = int(result.stdout.strip().split("\n")[0].strip())
    if free_mib <= config.runtime.min_free_gpu_mib:
        raise RuntimeError(
            f"GPU pre-flight failed: only {free_mib} MiB free, need >{config.runtime.min_free_gpu_mib} MiB"
        )


def _pre_update_parameter_state_artifact(
    raw_model: nn.Module,
    *,
    architecture_fingerprint_sha256: str,
    model_seed: int,
) -> dict[str, object]:
    """Hash the exact initialized parameter state without materializing a full CPU copy."""
    hash_method = "sha256_domain_separated_sorted_named_parameter_metadata_and_raw_bytes_v1"
    state_point = "after_seeded_model_construction_before_optimizer_or_checkpoint_load"
    initialization_contract = {
        "schema": "nanodllm.initialization_contract.v1",
        "architecture_fingerprint_sha256": architecture_fingerprint_sha256,
        "model_seed": int(model_seed),
        "parameter_state_hash_method": hash_method,
        "state_point": state_point,
    }
    initialization_contract_sha256 = hashlib.sha256(
        json.dumps(
            initialization_contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    digest = hashlib.sha256()
    digest.update(b"nanodllm.pre_update_parameter_state.v1\0")
    parameter_count = 0
    parameter_numel = 0
    parameter_bytes = 0
    for name, parameter in sorted(raw_model.named_parameters(), key=lambda item: item[0]):
        value = parameter.detach().contiguous()
        metadata = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "nbytes": int(value.numel() * value.element_size()),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        byte_view = value.reshape(-1).view(torch.uint8).cpu().numpy()
        digest.update(memoryview(byte_view))
        digest.update(b"\0")
        parameter_count += 1
        parameter_numel += int(value.numel())
        parameter_bytes += int(value.numel() * value.element_size())

    state_sha256 = digest.hexdigest()
    return {
        "schema": "nanodllm.initial_state.v1",
        "hash_method": hash_method,
        "state_point": state_point,
        "pre_update_parameter_state_sha256": state_sha256,
        "initial_state_sha256": state_sha256,
        "initialization_contract": initialization_contract,
        "initialization_contract_sha256": initialization_contract_sha256,
        "architecture_fingerprint_sha256": architecture_fingerprint_sha256,
        "model_seed": int(model_seed),
        "parameter_count": int(parameter_count),
        "parameter_numel": int(parameter_numel),
        "parameter_bytes": int(parameter_bytes),
    }


def write_jsonl_atomic(path: str, records: list[dict[str, object]]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    digest = hashlib.sha256()
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    os.replace(tmp_path, path)
    return digest.hexdigest()


def _source_checkpoint_identity(path: str) -> tuple[str, int]:
    """Return the SHA-256 and byte size of the exact checkpoint file read for replay."""

    digest = hashlib.sha256()
    size_bytes = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _validated_nelbo_sequence_records(
    value: object,
    *,
    expected_count: int,
    require_shared_time: bool,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("NELBO diagnostics must contain a sequence_records list")
    if len(value) != expected_count:
        raise ValueError(f"NELBO sequence record count {len(value)} does not match expected count {expected_count}")
    records: list[dict[str, object]] = []
    nonnegative_numeric_fields = (
        "nelbo_nats_numerator",
        "masked_nats_numerator",
        "byte_denominator",
        "lossable_content_target_count",
        "masked_target_count",
        "schedule_weight_sum",
        "schedule_weight_count",
    )
    for index, raw_record in enumerate(value):
        if not isinstance(raw_record, dict):
            raise TypeError(f"NELBO sequence record {index} must be a JSON object")
        record = dict(raw_record)
        if record.get("schema_version") != NELBO_SEQUENCE_RECORD_SCHEMA_VERSION:
            raise ValueError(f"NELBO sequence record {index} has unsupported schema_version")
        if record.get("sequence_index") != index:
            raise ValueError(f"NELBO sequence record {index} has non-contiguous sequence_index")
        for field in nonnegative_numeric_fields:
            field_value = record.get(field)
            if (
                not isinstance(field_value, int | float)
                or isinstance(field_value, bool)
                or not math.isfinite(float(field_value))
                or float(field_value) < 0
            ):
                raise ValueError(f"NELBO sequence record {index} has invalid {field}")
        for field in ("sampled_t", "sampled_t_min", "sampled_t_max"):
            field_value = record.get(field)
            if field_value is not None and (
                not isinstance(field_value, int | float)
                or isinstance(field_value, bool)
                or not math.isfinite(float(field_value))
            ):
                raise ValueError(f"NELBO sequence record {index} has invalid {field}")
        if require_shared_time:
            sampled_t = record.get("sampled_t")
            if (
                sampled_t is None
                or sampled_t != record.get("sampled_t_min")
                or sampled_t != record.get("sampled_t_max")
            ):
                raise ValueError(f"NELBO sequence record {index} does not contain one shared sequence time")
        required_digest_fields = ["input_identity_sha256", "corruption_mask_sha256"]
        if require_shared_time:
            required_digest_fields.extend(("corruption_uniform_sha256", "evaluation_sample_sha256"))
        for field in required_digest_fields:
            digest_value = record.get(field)
            if (
                not isinstance(digest_value, str)
                or len(digest_value) != 64
                or any(character not in "0123456789abcdef" for character in digest_value)
            ):
                raise ValueError(f"NELBO sequence record {index} has invalid {field}")
        records.append(record)
    if math.fsum(float(record["byte_denominator"]) for record in records) <= 0:
        raise ValueError("NELBO sequence records have no positive byte denominator")
    return records


def _ordered_nelbo_digest_root(domain: bytes, records: list[dict[str, object]], field: str) -> str:
    digest = hashlib.sha256(domain + b"\0")
    digest.update(len(records).to_bytes(8, "little"))
    for record in records:
        value = record[field]
        if not isinstance(value, str):
            raise TypeError(f"NELBO sequence record {field} must be a SHA-256 string")
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _prepare_data_identity(
    config: ExperimentConfig,
    *,
    rank: int,
    world_size: int,
    is_ddp: bool,
) -> dict[str, bool | str]:
    """Freeze and cross-rank-check the data contract before constructing a loader."""

    mode = str(config.data.identity_mode)
    if mode == "off":
        return {
            "enabled": False,
            "mode": mode,
            "artifact": "",
            "identity_sha256": "",
            "identity_sha256_strength": "",
        }

    record: dict | None = None
    local_error = ""
    try:
        record = build_data_identity(
            config,
            full_sha256=mode == "full_sha256",
            world_size=int(world_size),
            assignment_epoch=1,
        )
    except Exception as exc:  # Gather failures so one rank cannot silently disagree.
        local_error = f"{type(exc).__name__}: {exc}"

    local_result = {
        "rank": int(rank),
        "error": local_error,
        "identity_sha256": str(record["hashes"]["identity_sha256"]) if record is not None else "",
    }
    gathered = [local_result]
    if is_ddp:
        gathered = [None for _ in range(int(world_size))]
        dist.all_gather_object(gathered, local_result)
    errors = [f"rank {item['rank']}: {item['error']}" for item in gathered if item and item["error"]]
    if errors:
        raise RuntimeError("data identity failed: " + "; ".join(errors))
    digests = {str(item["identity_sha256"]) for item in gathered if item}
    if len(digests) != 1:
        raise RuntimeError(f"data identity differs across ranks: {sorted(digests)}")
    assert record is not None

    artifact_name = "data_identity.v1.json"
    artifact_path = Path(config.train.run_dir) / artifact_name
    digest = str(record["hashes"]["identity_sha256"])
    if rank == 0:
        if artifact_path.exists():
            try:
                existing = json.loads(artifact_path.read_text(encoding="utf-8"))
                existing_digest = str(existing["hashes"]["identity_sha256"])
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"cannot validate existing data identity {artifact_path}: {exc}") from exc
            if existing_digest != digest:
                raise RuntimeError(
                    f"refusing to resume with a changed data identity: existing={existing_digest} current={digest}"
                )
        write_json_atomic(str(artifact_path), record)
    if is_ddp:
        dist.barrier()
    return {
        "enabled": True,
        "mode": mode,
        "artifact": artifact_name,
        "identity_sha256": digest,
        "identity_sha256_strength": str(record["hashes"]["identity_sha256_strength"]),
    }


def _live_source_provenance() -> dict[str, bool | int | None | str]:
    """Inspect Git metadata when this module comes directly from its checkout."""

    unavailable: dict[str, bool | int | None | str] = {
        "source_repository_detected": False,
        "source_git_status_available": False,
        "source_git_commit": "",
        "source_tracked_worktree_clean": None,
        "source_untracked_path_count": 0,
    }
    try:
        import subprocess

        source_path = Path(__file__).resolve()
        repo_hint = source_path.parents[1]
        repo_root = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo_hint}",
                "-C",
                str(repo_hint),
                "rev-parse",
                "--show-toplevel",
            ],
            capture_output=True,
            text=True,
        )
        if repo_root.returncode != 0:
            return unavailable

        repo_root_path = Path(repo_root.stdout.strip()).resolve()
        if (repo_root_path / "nanodllm" / "training.py").resolve() != source_path:
            return unavailable

        commit = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root_path}", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root_path,
        )
        status = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root_path}", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_root_path,
        )
        if status.returncode != 0:
            return {
                **unavailable,
                "source_repository_detected": True,
                "source_git_commit": commit.stdout.strip() if commit.returncode == 0 else "",
            }

        lines = [line for line in status.stdout.splitlines() if line.strip()]
        tracked_dirty = any(not line.startswith("?? ") for line in lines)
        untracked_path_count = sum(1 for line in lines if line.startswith("?? "))
        return {
            "source_repository_detected": True,
            "source_git_status_available": True,
            "source_git_commit": commit.stdout.strip() if commit.returncode == 0 else "",
            "source_tracked_worktree_clean": not tracked_dirty,
            "source_untracked_path_count": untracked_path_count,
        }
    except Exception:
        return unavailable


def source_provenance_metrics() -> dict[str, bool | int | None | list[str] | str]:
    """Return informational source facts that never gate an otherwise successful run."""

    provenance = _live_source_provenance()
    repository_detected = bool(provenance["source_repository_detected"])
    status_available = bool(provenance["source_git_status_available"])
    tracked_clean = provenance["source_tracked_worktree_clean"]
    untracked_path_count = int(provenance["source_untracked_path_count"])
    warnings: list[str] = []
    if not repository_detected:
        warnings.append("Git checkout metadata is unavailable; this may be an installed distribution")
    elif not status_available:
        warnings.append("Git checkout detected, but worktree status could not be inspected")
    else:
        if tracked_clean is False:
            warnings.append("Git checkout contains tracked modifications")
        if untracked_path_count:
            warnings.append("Git checkout contains untracked paths")
    worktree_dirty = None if not status_available else tracked_clean is False or untracked_path_count > 0
    return {
        "status": "ok",
        **provenance,
        "source_worktree_dirty": worktree_dirty,
        "source_provenance_is_informational": True,
        "source_provenance_warnings": warnings,
    }


def _numeric_runtime_config_summary(config: ExperimentConfig) -> dict[str, bool | str | None]:
    return {
        "runtime_deterministic_algorithms_requested": bool(config.runtime.deterministic_algorithms),
        "runtime_sdpa_backend_requested": str(config.runtime.sdpa_backend),
        "runtime_cudnn_allow_tf32_requested": config.runtime.cudnn_allow_tf32,
        "runtime_compile": bool(config.runtime.compile),
        "runtime_compile_dynamic": bool(config.runtime.compile_dynamic),
        "runtime_dtype": str(config.runtime.dtype),
        "runtime_device_type": str(config.runtime.device_type),
    }


def _query_numeric_runtime_telemetry(config: ExperimentConfig) -> dict[str, bool | str]:
    """Query process-wide numeric state after the config has been applied."""
    cudnn_allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    return {
        "runtime_deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "runtime_deterministic_algorithms_warn_only_enabled": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "runtime_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "runtime_cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "runtime_cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "runtime_sdpa_flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "runtime_sdpa_mem_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "runtime_sdpa_math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "runtime_sdpa_cudnn_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        "runtime_cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "runtime_cudnn_allow_tf32_resolved": cudnn_allow_tf32,
        "runtime_cudnn_allow_tf32": cudnn_allow_tf32,
        "runtime_float32_matmul_precision": str(torch.get_float32_matmul_precision()),
    }


def _numeric_runtime_telemetry(config: ExperimentConfig) -> dict[str, bool | str]:
    """Return cached effective telemetry only after runtime configuration."""
    return dict(getattr(config, NUMERIC_RUNTIME_TELEMETRY_ATTR, {}))


def _configure_numeric_runtime(config: ExperimentConfig) -> dict[str, bool | str]:
    """Apply config-owned process-wide numeric controls and report their effective state."""
    with suppress(AttributeError):
        delattr(config, NUMERIC_RUNTIME_TELEMETRY_ATTR)
    deterministic = bool(config.runtime.deterministic_algorithms)
    if config.runtime.device_type == "cuda" and deterministic:
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        if workspace_config not in DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS:
            allowed = ", ".join(sorted(DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS))
            raise RuntimeError(
                f"deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG to be one of {allowed}; got {workspace_config!r}"
            )
        if config.runtime.cudnn_benchmark:
            raise RuntimeError("deterministic CUDA requires runtime.cudnn_benchmark=false")

    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = deterministic
    if config.runtime.float32_matmul_precision:
        torch.set_float32_matmul_precision(config.runtime.float32_matmul_precision)
    if config.runtime.device_type == "cuda":
        if not config.runtime.float32_matmul_precision:
            torch.backends.cuda.matmul.allow_tf32 = bool(config.runtime.allow_tf32)
        if config.runtime.cudnn_allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = bool(config.runtime.cudnn_allow_tf32)
        torch.backends.cudnn.benchmark = bool(config.runtime.cudnn_benchmark)
    if config.runtime.sdpa_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
    else:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(True)
    telemetry = _query_numeric_runtime_telemetry(config)
    setattr(config, NUMERIC_RUNTIME_TELEMETRY_ATTR, telemetry)
    return dict(telemetry)


def run_config_summary(config: ExperimentConfig, run_mode: str) -> dict[str, bool | float | int | str | None]:
    return {
        "run_mode": run_mode,
        **_numeric_runtime_config_summary(config),
        **_numeric_runtime_telemetry(config),
        "model_depth": int(config.model.depth),
        "model_n_embd": int(config.model.n_embd),
        "model_n_head": int(config.model.n_head),
        "model_output_factorization": config.model.output_factorization,
        "model_block_causal_pilot_size": int(config.model.block_causal_pilot_size),
        "model_residual_output_projection_init": config.model.residual_output_projection_init,
        "model_embedding_head_init_std_multiplier": float(config.model.embedding_head_init_std_multiplier),
        "model_length_head": bool(config.model.length_head),
        "model_target_length_conditioning": bool(config.model.target_length_conditioning),
        "model_mask_position_embedding": bool(config.model.mask_position_embedding),
        "tokenizer": config.data.tokenizer,
        "tokenizer_path": config.data.tokenizer_path if config.data.tokenizer == "hf" else "",
        "target_tokens": int(config.train.target_tokens),
        "total_batch_size": int(config.train.total_batch_size),
        "device_batch_size": int(config.train.device_batch_size),
        "strict_mdlm_all_token_bayes_audit_only": bool(config.train.strict_mdlm_all_token_bayes_audit_only),
        "strict_mdlm_all_token_diagnostic_telemetry": bool(config.train.strict_mdlm_all_token_diagnostic_telemetry),
        "label_smoothing_epsilon": float(config.train.label_smoothing_epsilon),
        "entropy_bonus_beta": float(config.train.entropy_bonus_beta),
        "confidence_penalty_weight": float(config.train.confidence_penalty_weight),
        "z_loss_weight": float(config.train.z_loss_weight),
        "eos_hazard_loss_weight": float(config.train.eos_hazard_loss_weight),
        "length_head_loss_weight": float(config.train.length_head_loss_weight),
        "position_decile_balanced_loss": bool(config.train.position_decile_balanced_loss),
        "uniform_loss_estimator": config.train.uniform_loss_estimator,
        "position_decile_balanced_loss_deciles": POSITION_DECILE_BALANCED_LOSS_DECILES,
        "token_edit_corruption_enabled": _uses_token_edit_corruption(config),
        "token_edit_corruption_batch_fraction": float(config.train.token_edit_corruption_batch_fraction),
        "token_edit_corruption_ratio": float(config.train.token_edit_corruption_ratio),
        "token_edit_corruption_deciles": TOKEN_EDIT_CORRUPTION_DECILES,
        "noise_per_token_noise_levels": _uses_per_token_noise_levels(config),
        "noise_per_token_level_hist_bins": PER_TOKEN_NOISE_LEVEL_HIST_BINS,
        "data_boundary_aware_packing": bool(config.data.boundary_aware_packing),
        "data_content_budget_segmented_packing": bool(config.data.content_budget_segmented_packing),
        "data_synthetic_dataset": str(config.data.synthetic_dataset),
        "data_synthetic_eos_overfit_loss_mask_policy": str(config.data.synthetic_eos_overfit_loss_mask_policy),
        "data_short_document_curriculum": bool(config.data.short_document_curriculum),
        "data_short_document_max_tokens": int(config.data.short_document_max_tokens),
        "eval_basin_diagnostic": bool(config.eval.basin_diagnostic),
        "eval_source_checkpoint_identity_mode": config.eval.source_checkpoint_identity_mode,
        "eval_source_checkpoint_expected_sha256": config.eval.source_checkpoint_expected_sha256,
        "eval_source_checkpoint_expected_size_bytes": int(config.eval.source_checkpoint_expected_size_bytes),
        "eval_basin_prefix_len": int(config.eval.basin_prefix_len),
        "eval_basin_nucleus_top_p": float(config.eval.basin_nucleus_top_p),
        "eval_length_policy": config.eval.length_policy,
        "eval_target_length_distribution_batches": int(config.eval.target_length_distribution_batches),
        "eos_supervision_audit_enabled": _eos_supervision_audit_enabled(config),
        "noise_soft_input_curriculum": bool(config.noise.soft_input_curriculum),
        "noise_soft_input_curriculum_start_mix": float(config.noise.soft_input_curriculum_start_mix),
        "noise_soft_input_curriculum_end_mix": float(config.noise.soft_input_curriculum_end_mix),
        "noise_soft_input_curriculum_anneal_fraction": float(config.noise.soft_input_curriculum_anneal_fraction),
        "optim_lr": float(config.optim.lr),
        "optim_muon_lr": float(config.optim.muon_lr),
        "optim_adamw_lr": float(config.optim.adamw_lr),
        "optim_warmup_steps": int(config.optim.warmup_steps),
        "optim_warmup_ratio": float(config.optim.warmup_ratio),
        "optim_warmdown_ratio": float(config.optim.warmdown_ratio),
        "optim_final_lr_frac": float(config.optim.final_lr_frac),
        "optim_grad_clip": float(config.optim.grad_clip),
        "optim_adam_beta1": float(config.optim.adam_beta1),
        "optim_adam_beta2": float(config.optim.adam_beta2),
        "optim_adam_eps": float(config.optim.adam_eps),
        "eval_fixed_ce_trace_enabled": bool(config.eval.fixed_ce_trace_enabled),
    }


def _noise_telemetry(config: ExperimentConfig) -> dict[str, bool | float | str]:
    return {
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


def _token_edit_corruption_telemetry(config: ExperimentConfig) -> dict[str, bool | float | int | str]:
    enabled = _uses_token_edit_corruption(config)
    return {
        "token_edit_corruption_enabled": enabled,
        "token_edit_corruption_batch_fraction": float(config.train.token_edit_corruption_batch_fraction),
        "token_edit_corruption_ratio": float(config.train.token_edit_corruption_ratio),
        "token_edit_corruption_deciles": TOKEN_EDIT_CORRUPTION_DECILES,
        "training_objective": "mdlm_m2t_plus_t2t_edit" if enabled else "mdlm_m2t",
    }


def _training_corruption_label(config: ExperimentConfig) -> str:
    if _is_causal(config):
        return "causal-ar"
    if _uses_content_budget_segmented_packing(config):
        return "content-budget-segmented-mdlm"
    if _uses_block_causal_pilot(config):
        return "block-causal-pilot-mdlm"
    if config.noise.type == "mdlm" and config.noise.suffix_masking:
        return "suffix-mask-mdlm"
    if config.noise.type == "mdlm" and _uses_per_token_noise_levels(config):
        return "per-token-noise-mdlm"
    return config.noise.type


def _uses_bd3lm(config: ExperimentConfig) -> bool:
    return config.noise.type == "bd3lm"


def _uses_hierarchical_output(config: ExperimentConfig) -> bool:
    return config.model.output_factorization == "hierarchical_frequency_decile"


def _uses_length_head(config: ExperimentConfig) -> bool:
    return bool(config.model.length_head)


def _uses_target_length_conditioning(config: ExperimentConfig) -> bool:
    return bool(getattr(config.model, "target_length_conditioning", False))


def _causal_ar_label_smoothing(config: ExperimentConfig) -> float:
    if not _is_causal(config):
        return 0.0
    return float(config.train.label_smoothing_epsilon)


def _uses_uniform_low_variance_estimator(config: ExperimentConfig) -> bool:
    return config.train.uniform_loss_estimator == "two_sample_mean"


def _uses_token_edit_corruption(config: ExperimentConfig) -> bool:
    return (
        float(config.train.token_edit_corruption_batch_fraction) > 0.0
        and float(config.train.token_edit_corruption_ratio) > 0.0
    )


def _uses_per_token_noise_levels(config: ExperimentConfig) -> bool:
    return bool(getattr(config.noise, "per_token_noise_levels", False))


def _uses_strict_mdlm_per_row_t(config: ExperimentConfig) -> bool:
    return (
        not _is_causal(config)
        and config.noise.type == "mdlm"
        and bool(config.noise.schedule_weight)
        and not bool(config.noise.complement_masking)
        and not bool(config.noise.suffix_masking)
        and not bool(config.noise.time_conditioning)
        and not _uses_per_token_noise_levels(config)
        and not _uses_block_causal_pilot(config)
        and not _uses_bd3lm(config)
        and not _uses_token_edit_corruption(config)
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


def _uses_schedule_weighted_nelbo(config: ExperimentConfig) -> bool:
    """Whether the model supports the shared-time absorbing-MDLM evaluator.

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


def _stratified_per_sequence_times(
    count: int,
    t_min: float,
    t_max: float,
    seed: int,
) -> torch.Tensor:
    """Return a deterministic, permuted midpoint stratification of ``[t_min, t_max]``."""

    if count <= 0:
        raise ValueError("stratified NELBO evaluation requires at least one sequence")
    if not 0.0 <= float(t_min) < float(t_max) <= 1.0:
        raise ValueError("stratified NELBO times require 0 <= t_min < t_max <= 1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    unit_midpoints = (torch.arange(count, dtype=torch.float64) + 0.5) / float(count)
    times = float(t_min) + (float(t_max) - float(t_min)) * unit_midpoints
    return times[torch.randperm(count, generator=generator)].to(torch.float32)


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
    source = ParquetTextSource(config.data)
    batches = source.document_batches("train", rank=rank, world_size=world_size)
    for _ in range(EOS_SUPERVISION_PREPACK_DOCUMENT_BATCHES):
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
        "eos_supervision_audit_prepack_document_batches": EOS_SUPERVISION_PREPACK_DOCUMENT_BATCHES,
        "eos_supervision_audit_packed_batches": EOS_SUPERVISION_PACKED_BATCHES,
        "eos_supervision_audit_forced_mask_t": EOS_SUPERVISION_FORCED_MASK_T,
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
        for _ in range(EOS_SUPERVISION_PACKED_BATCHES):
            batch = next(audit_loader)
            z_audit, w_audit = noise.corrupt(
                batch.x_0,
                EOS_SUPERVISION_FORCED_MASK_T,
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


def _ddp_broadcast_buffers(config: ExperimentConfig) -> bool:
    return not _uses_uniform_low_variance_estimator(config)


def _population_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _label_smoothing_ar_metrics(config: ExperimentConfig, tokenizer) -> dict[str, bool | float | int | str]:
    epsilon = _causal_ar_label_smoothing(config)
    if epsilon <= 0:
        return {
            "ar_label_smoothing_enabled": False,
            "ar_label_smoothing_epsilon": 0.0,
        }
    vocab_size = int(tokenizer.get_vocab_size())
    return {
        "ar_label_smoothing_enabled": True,
        "ar_label_smoothing_epsilon": float(epsilon),
        "ar_label_smoothing_vocab_size": vocab_size,
    }


def _hierarchical_output_telemetry(metadata: dict | None) -> dict[str, float | int | str]:
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


def _bd3lm_telemetry(config: ExperimentConfig) -> dict[str, bool | int | str]:
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
    return {
        "objective": "bd3lm",
        "bd3lm_enabled": True,
        "loss_normalization": loss_normalization,
        "bd3lm_block_size": int(config.model.bd3lm_block_size),
        "bd3lm_clean_seq_len": clean_seq_len,
        "bd3lm_concat_seq_len": clean_seq_len * 2,
    }


def _bd3lm_concat_inputs(
    config: ExperimentConfig,
    z_t: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if z_t.shape != x_0.shape:
        raise ValueError(f"BD3LM requires z_t and x_0 to have the same shape, got {z_t.shape} and {x_0.shape}")
    concat = torch.cat((z_t, x_0), dim=1)
    mask = bd3lm_attention_mask(attention_mask, int(config.model.bd3lm_block_size))
    position_ids = duplicated_position_ids(z_t.shape[1], device=z_t.device)
    return concat, mask, position_ids


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


def _forward_training_model(
    model,
    config: ExperimentConfig,
    z_t: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_indices: torch.Tensor,
    return_aux_logits: bool,
    noise_level,
    return_hierarchical_logits: bool = False,
    return_length_logits: bool = False,
    input_embeddings: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
):
    if input_embeddings is not None and _uses_bd3lm(config):
        raise ValueError("soft input embeddings are not supported for BD3LM concat training")
    if _uses_bd3lm(config):
        concat, bd3lm_mask, position_ids = _bd3lm_concat_inputs(config, z_t, x_0, attention_mask)
        return model(
            concat,
            logits_indices=logits_indices,
            return_aux_logits=return_aux_logits,
            return_hierarchical_logits=return_hierarchical_logits,
            return_length_logits=return_length_logits,
            attention_mask=bd3lm_mask,
            position_ids=position_ids,
        )
    model_kwargs = {
        "logits_indices": logits_indices,
        "return_aux_logits": return_aux_logits,
        "return_hierarchical_logits": return_hierarchical_logits,
        "return_length_logits": return_length_logits,
        "attention_mask": attention_mask,
        "noise_level": noise_level if config.noise.time_conditioning else None,
        "input_embeddings": input_embeddings,
    }
    if position_ids is not None:
        model_kwargs["position_ids"] = position_ids
    if target_lengths is not None:
        model_kwargs["target_lengths"] = target_lengths
    return model(z_t, **model_kwargs)


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
        concat, bd3lm_mask, position_ids = _bd3lm_concat_inputs(config, z_t, x_0, attention_mask)
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
        concat, bd3lm_mask, position_ids = _bd3lm_concat_inputs(config, z_t, x_0, attention_mask)
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


@torch.no_grad()
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


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _run_artifact_manifest_relative_path(run_dir: str, artifact_path: str | Path) -> str:
    run_path = Path(run_dir).expanduser()
    run_path = (Path.cwd() / run_path).resolve() if not run_path.is_absolute() else run_path.resolve()
    path = Path(artifact_path).expanduser()
    path = (run_path / path).resolve() if not path.is_absolute() else path.resolve()

    if run_path.parent.name == "runs":
        try:
            return path.relative_to(run_path.parent.parent).as_posix()
        except ValueError:
            pass
    try:
        return path.relative_to(run_path).as_posix()
    except ValueError:
        return str(path)


def _register_run_artifacts(
    run_dir: str,
    artifacts: list[dict[str, object]],
) -> Path:
    run_path = Path(run_dir).expanduser()
    run_path = (Path.cwd() / run_path).resolve() if not run_path.is_absolute() else run_path.resolve()
    manifest_path = run_path / "artifacts.json"
    now = _utc_now_iso()
    if manifest_path.exists():
        manifest = _load_json_if_exists(manifest_path)
        if not isinstance(manifest.get("artifacts"), list):
            raise ValueError(f"cannot update non-versioned artifact manifest at {manifest_path}")
        manifest.setdefault("schema_version", 1)
        manifest.setdefault("run_id", run_path.name)
        manifest.setdefault("created_at", now)
        manifest.setdefault(
            "telemetry_dir", _run_artifact_manifest_relative_path(str(run_path), run_path / "telemetry")
        )
        manifest.setdefault("metadata", {})
    else:
        manifest = {
            "schema_version": 1,
            "run_id": run_path.name,
            "created_at": now,
            "updated_at": now,
            "artifacts": [],
            "telemetry_dir": _run_artifact_manifest_relative_path(str(run_path), run_path / "telemetry"),
            "metadata": {},
        }

    manifest["updated_at"] = now
    existing = {
        (
            str(artifact.get("path", "")),
            str(artifact.get("kind", "")),
            str(artifact.get("name", "")),
        ): idx
        for idx, artifact in enumerate(manifest.get("artifacts", []))
        if isinstance(artifact, dict)
    }
    for spec in artifacts:
        raw_path = Path(str(spec["path"])).expanduser()
        if raw_path.is_absolute():
            path_obj = raw_path
        else:
            cwd_candidate = (Path.cwd() / raw_path).resolve()
            path_obj = cwd_candidate if cwd_candidate.exists() else run_path / raw_path
        rel_path = _run_artifact_manifest_relative_path(str(run_path), path_obj)
        entry = {
            "path": rel_path,
            "kind": str(spec.get("kind", "artifact")),
            "name": str(spec.get("name", path_obj.name)),
            "phase": str(spec.get("phase", "run")),
            "required": bool(spec.get("required", False)),
            "exists": bool(path_obj.exists()),
            "metadata": dict(spec.get("metadata", {})) if isinstance(spec.get("metadata", {}), dict) else {},
        }
        key = (entry["path"], entry["kind"], entry["name"])
        if key in existing:
            manifest["artifacts"][existing[key]] = entry
        else:
            existing[key] = len(manifest["artifacts"])
            manifest["artifacts"].append(entry)

    write_json_atomic(str(manifest_path), manifest)
    return manifest_path


def _tokenizer_summary(tokenizer) -> dict:
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "mask_token_id": tokenizer.get_mask_token_id(),
        "eos_token_id": tokenizer.get_eos_token_id(),
        "bos_token_id": tokenizer.get_bos_token_id(),
        "pad_token_id": tokenizer.get_pad_token_id(),
        "special_token_ids": sorted(tokenizer.get_special_token_ids()),
        "generation_blocked_token_ids": sorted(tokenizer.get_generation_blocked_token_ids()),
    }


def _rng_state(config: ExperimentConfig) -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if config.runtime.device_type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _active_rank_rng_state(config: ExperimentConfig, device: torch.device) -> dict[str, object]:
    """Capture only this process' RNG streams and active CUDA device."""
    state: dict[str, object] = {
        "version": RESUME_RNG_STATE_VERSION,
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "device_type": str(device.type),
        "distributed_backend": str(dist.get_backend()) if dist.is_available() and dist.is_initialized() else "none",
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        "group_rank": int(os.environ.get("GROUP_RANK", "0")),
        "cuda": None,
    }
    if config.runtime.device_type == "cuda" and torch.cuda.is_available():
        device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
        state["cuda"] = {
            "active_device_index": device_index,
            "visible_device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(device_index),
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
            "state": torch.cuda.get_rng_state(device_index),
        }
    return state


def _legacy_rng_state_from_active(state: dict[str, object]) -> dict[str, object]:
    """Keep the old top-level key readable without querying peer CUDA devices."""
    legacy: dict[str, object] = {
        "torch": state["torch_cpu"],
        "numpy": state["numpy"],
        "python": state["python"],
    }
    cuda_state = state.get("cuda")
    if isinstance(cuda_state, dict) and "state" in cuda_state:
        legacy["cuda"] = [cuda_state["state"]]
    return legacy


def _validate_active_rank_rng_topology(
    state: dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> None:
    if state.get("version") != RESUME_RNG_STATE_VERSION:
        raise RuntimeError(f"unsupported rank RNG state version {state.get('version')!r}")
    expected_topology = {
        "device_type": str(device.type),
        "distributed_backend": str(dist.get_backend()) if dist.is_available() and dist.is_initialized() else "none",
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        "group_rank": int(os.environ.get("GROUP_RANK", "0")),
    }
    mismatches = {
        key: (state.get(key), expected) for key, expected in expected_topology.items() if state.get(key) != expected
    }
    cuda_state = state.get("cuda")
    if config.runtime.device_type == "cuda" and torch.cuda.is_available():
        device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
        if not isinstance(cuda_state, dict):
            raise RuntimeError("rank RNG state has no active CUDA-device state")
        cuda_expected = {
            "active_device_index": device_index,
            "visible_device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(device_index),
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        }
        mismatches.update(
            {
                f"cuda.{key}": (cuda_state.get(key), expected)
                for key, expected in cuda_expected.items()
                if cuda_state.get(key) != expected
            }
        )
    elif cuda_state is not None:
        mismatches["cuda"] = ("present", "absent")
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={saved!r}, current={current!r}" for key, (saved, current) in mismatches.items()
        )
        raise RuntimeError(f"resume RNG topology mismatch: {details}")


def _restore_active_rank_rng_state(
    state: dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> None:
    _validate_active_rank_rng_topology(state, config, device)
    cuda_state = state.get("cuda")
    torch.set_rng_state(_rng_byte_tensor(state["torch_cpu"], field="torch_cpu"))
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if isinstance(cuda_state, dict):
        torch.cuda.set_rng_state(
            _rng_byte_tensor(cuda_state["state"], field="cuda.state"),
            device=int(cuda_state["active_device_index"]),
        )


def _source_implementation_identity() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_paths = ("nanodllm", "pretrain.py", "pyproject.toml", "uv.lock")
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all", "--", *runtime_paths],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if len(head) == 40 and not status:
            return {"mode": "clean_git_head", "git_head": head}
    except (OSError, subprocess.SubprocessError):
        head = ""

    files = [repo_root / name for name in ("pretrain.py", "pyproject.toml", "uv.lock")]
    files.extend(sorted((repo_root / "nanodllm").rglob("*.py")))
    digest = hashlib.sha256(b"nanodllm-source-tree-v1\0")
    included = 0
    for path in sorted({path.resolve() for path in files if path.is_file()}):
        relative = path.relative_to(repo_root.resolve()).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
        included += 1
    if not included:
        raise RuntimeError(f"could not identify any runtime source files below {repo_root}")
    return {
        "mode": "source_tree_sha256_fallback",
        "git_head": head,
        "source_tree_sha256": digest.hexdigest(),
        "source_file_count": included,
    }


def _runtime_compatibility_identity(config: ExperimentConfig, device: torch.device) -> dict[str, object]:
    identity: dict[str, object] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", "")),
        "device_type": str(device.type),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }
    if config.runtime.device_type == "cuda" and torch.cuda.is_available():
        device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
        nccl_version = None
        if hasattr(torch.cuda, "nccl") and hasattr(torch.cuda.nccl, "version"):
            try:
                nccl_version = torch.cuda.nccl.version()
            except (RuntimeError, AttributeError):
                nccl_version = None
        identity.update(
            {
                "cuda_runtime_version": str(torch.version.cuda),
                "cudnn_version": torch.backends.cudnn.version(),
                "nccl_version": list(nccl_version) if isinstance(nccl_version, tuple) else nccl_version,
                "active_gpu_name": torch.cuda.get_device_name(device_index),
                "active_gpu_compute_capability": list(torch.cuda.get_device_capability(device_index)),
            }
        )
    else:
        identity.update(
            {
                "cpu_processor": platform.processor(),
                "cpu_count": os.cpu_count(),
            }
        )
    return identity


def _trajectory_record_sha256(record: dict[str, object]) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _resume_trajectory_contract(
    config: ExperimentConfig,
    *,
    device: torch.device,
    model_config: dict[str, object],
    architecture_fingerprint: dict[str, object],
    data_identity_summary: dict[str, object],
    total_batch_size: int,
    grad_accum_steps: int,
    total_steps: int,
    warmup_steps: int,
    lr_schedule: dict[str, object],
) -> dict[str, object]:
    """Canonical trajectory identity with a narrow operational-field allowlist."""
    resolved = copy.deepcopy(to_dict(config))
    resolved.pop("eval", None)
    resolved.pop("recipe", None)
    train = resolved.get("train", {})
    if isinstance(train, dict):
        for key in (
            "run_dir",
            "resume_from",
            "comparison_run_ids",
            "tensorboard",
            "tensorboard_dir",
            "save_every",
            "eval_every",
        ):
            train.pop(key, None)
    runtime = resolved.get("runtime", {})
    if isinstance(runtime, dict):
        runtime.pop("min_free_gpu_mib", None)
    data_identity = {
        key: data_identity_summary.get(key)
        for key in ("enabled", "mode", "identity_sha256", "identity_sha256_strength")
    }
    record = {
        "version": RESUME_TRAJECTORY_VERSION,
        "implementation_identity": _source_implementation_identity(),
        "runtime_compatibility": _runtime_compatibility_identity(config, device),
        "resolved_trajectory_config": resolved,
        "resolved_model_config": model_config,
        "architecture_fingerprint_sha256": architecture_fingerprint.get("fingerprint_sha256"),
        "data_identity": data_identity,
        "derived_training_budget": {
            "total_batch_size": int(total_batch_size),
            "grad_accum_steps": int(grad_accum_steps),
            "total_steps": int(total_steps),
            "warmup_steps": int(warmup_steps),
            "lr_schedule": lr_schedule,
        },
        "allowed_operational_differences": {
            "config_sections": ["eval", "recipe"],
            "train_fields": [
                "run_dir",
                "resume_from",
                "comparison_run_ids",
                "tensorboard",
                "tensorboard_dir",
                "save_every",
                "eval_every",
            ],
            "runtime_fields": ["min_free_gpu_mib"],
        },
    }
    return {"version": RESUME_TRAJECTORY_VERSION, "sha256": _trajectory_record_sha256(record), "record": record}


def _tensor_content_digest(value: torch.Tensor | None) -> dict[str, object]:
    if value is None:
        return {"present": False}
    tensor = value.detach().to(device="cpu").contiguous()
    raw_bytes = tensor.view(torch.uint8).numpy().tobytes()
    return {
        "present": True,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def _batch_six_tensor_digests(batch: DLMDataBatch) -> dict[str, object]:
    """Hash every tensor that defines the next training batch."""
    tensors = {field: _tensor_content_digest(getattr(batch, field, None)) for field in RESUME_BATCH_TENSOR_FIELDS}
    canonical = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "version": RESUME_BATCH_DIGEST_VERSION,
        "epoch": int(batch.epoch),
        "tensor_fields": list(RESUME_BATCH_TENSOR_FIELDS),
        "tensors": tensors,
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _packed_all_reduce_int_counters(
    counters: dict[str, int],
    *,
    device: torch.device,
    is_ddp: bool,
    op=dist.ReduceOp.SUM,
) -> dict[str, int]:
    """Reduce an ordered group of integer counters with one collective."""
    keys = tuple(counters)
    values = torch.tensor([int(counters[key]) for key in keys], dtype=torch.int64, device=device)
    if is_ddp:
        dist.all_reduce(values, op=op)
    return {key: int(value) for key, value in zip(keys, values.cpu().tolist(), strict=True)}


def _packed_all_reduce_float_counters(
    counters: dict[str, float],
    *,
    device: torch.device,
    is_ddp: bool,
    op=dist.ReduceOp.SUM,
) -> dict[str, float]:
    keys = tuple(counters)
    values = torch.tensor([float(counters[key]) for key in keys], dtype=torch.float64, device=device)
    if is_ddp:
        dist.all_reduce(values, op=op)
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist(), strict=True)}


def _all_reduce_numeric_accumulator(
    accumulator: dict[str, object],
    *,
    device: torch.device,
    is_ddp: bool,
) -> dict[str, object]:
    """SUM-reduce a JSON-like numeric accumulator while preserving integer fields."""
    out = copy.deepcopy(accumulator)
    int_paths: list[tuple[object, ...]] = []
    int_values: list[int] = []
    float_paths: list[tuple[object, ...]] = []
    float_values: list[float] = []

    def visit(value: object, path: tuple[object, ...]) -> None:
        if isinstance(value, bool):
            raise TypeError(f"boolean accumulator field at {path!r} is unsupported")
        if isinstance(value, int):
            int_paths.append(path)
            int_values.append(value)
        elif isinstance(value, float):
            float_paths.append(path)
            float_values.append(value)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, index))
        else:
            raise TypeError(f"non-numeric accumulator field at {path!r}: {type(value).__name__}")

    def assign(path: tuple[object, ...], value: int | float) -> None:
        cursor: object = out
        for key in path[:-1]:
            cursor = cursor[key]  # type: ignore[index]
        cursor[path[-1]] = value  # type: ignore[index]

    visit(out, ())
    if int_values:
        reduced_ints = torch.tensor(int_values, dtype=torch.int64, device=device)
        if is_ddp:
            dist.all_reduce(reduced_ints, op=dist.ReduceOp.SUM)
        for path, value in zip(int_paths, reduced_ints.cpu().tolist(), strict=True):
            assign(path, int(value))
    if float_values:
        reduced_floats = torch.tensor(float_values, dtype=torch.float64, device=device)
        if is_ddp:
            dist.all_reduce(reduced_floats, op=dist.ReduceOp.SUM)
        for path, value in zip(float_paths, reduced_floats.cpu().tolist(), strict=True):
            assign(path, float(value))
    return out


def _merge_numeric_accumulator(dest: dict[str, object], src: dict[str, object]) -> None:
    for key, value in src.items():
        if isinstance(value, dict):
            child = dest.get(key)
            if not isinstance(child, dict):
                raise ValueError(f"accumulator field {key!r} is not a mapping")
            _merge_numeric_accumulator(child, value)
        elif isinstance(value, list):
            child = dest.get(key)
            if not isinstance(child, list) or len(child) != len(value):
                raise ValueError(f"accumulator list field {key!r} has a different shape")
            for index, item in enumerate(value):
                child[index] += item
        elif isinstance(value, int | float) and not isinstance(value, bool):
            dest[key] = dest[key] + value  # type: ignore[operator]
        else:
            raise TypeError(f"unsupported accumulator field {key!r}: {type(value).__name__}")


def _require_exact_mapping_keys(value: object, expected: set[str], *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a mapping")
    observed = set(value)
    if observed != expected:
        raise RuntimeError(
            f"{field} keys mismatch: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return value


def _validate_json_state(value: object, *, field: str) -> None:
    if value is None or isinstance(value, bool | int | float | str):
        return
    if isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _validate_json_state(child, field=f"{field}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{field} mapping keys must be strings")
        for key, child in value.items():
            _validate_json_state(child, field=f"{field}.{key}")
        return
    raise TypeError(f"{field} has unsupported type {type(value).__name__}")


def _validate_numeric_tree_shape(value: object, template: object, *, field: str) -> None:
    if isinstance(template, bool):
        if not isinstance(value, bool):
            raise TypeError(f"{field} must be a boolean")
    elif isinstance(template, int):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{field} must be an integer")
    elif isinstance(template, float):
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(f"{field} must be numeric")
    elif isinstance(template, list):
        if not isinstance(value, list) or len(value) != len(template):
            raise RuntimeError(f"{field} must be a list of length {len(template)}")
        for index, (child, expected) in enumerate(zip(value, template, strict=True)):
            _validate_numeric_tree_shape(child, expected, field=f"{field}[{index}]")
    elif isinstance(template, dict):
        mapping = _require_exact_mapping_keys(value, set(template), field=field)
        for key, expected in template.items():
            _validate_numeric_tree_shape(mapping[key], expected, field=f"{field}.{key}")
    else:
        raise TypeError(f"unsupported trainer schema template at {field}")


def _validate_trainer_state_schema(state: dict[str, object], config: ExperimentConfig) -> None:
    root = _require_exact_mapping_keys(
        state,
        {"version", *RESUME_TRAINER_GROUP_FIELDS},
        field="trainer_state",
    )
    if root["version"] != RESUME_TRAINER_STATE_VERSION:
        raise RuntimeError(f"unsupported trainer state version {root['version']!r}")
    groups = {
        name: _require_exact_mapping_keys(root[name], set(fields), field=f"trainer_state.{name}")
        for name, fields in RESUME_TRAINER_GROUP_FIELDS.items()
    }
    lineage = groups["lineage"]
    last = groups["last"]
    diagnostics = groups["diagnostics"]
    accumulator_templates = {
        "trainer_state.lineage.train_length_head_accumulator": (
            lineage["train_length_head_accumulator"],
            _new_length_head_accumulator(int(config.model.max_seq_len)),
        ),
        "trainer_state.lineage.train_token_edit_reconstruction_accumulator": (
            lineage["train_token_edit_reconstruction_accumulator"],
            _new_token_edit_reconstruction_accumulator(),
        ),
        "trainer_state.last.eos_hazard_accumulator": (
            last["eos_hazard_accumulator"],
            _new_eos_hazard_accumulator(),
        ),
        "trainer_state.last.length_head_accumulator": (
            last["length_head_accumulator"],
            _new_length_head_accumulator(int(config.model.max_seq_len)),
        ),
        "trainer_state.last.position_decile_ce_accumulator": (
            last["position_decile_ce_accumulator"],
            _new_position_decile_ce_accumulator(),
        ),
        "trainer_state.last.token_edit_reconstruction_accumulator": (
            last["token_edit_reconstruction_accumulator"],
            _new_token_edit_reconstruction_accumulator(),
        ),
    }
    for field, (value, template) in accumulator_templates.items():
        _validate_numeric_tree_shape(value, template, field=field)
    fixed_shapes = {
        "trainer_state.lineage.train_per_token_noise_level_hist_counts": (
            lineage["train_per_token_noise_level_hist_counts"],
            [0 for _ in range(PER_TOKEN_NOISE_LEVEL_HIST_BINS)],
        ),
        "trainer_state.last.top1_prob_hist_counts": (
            last["top1_prob_hist_counts"],
            [0 for _ in range(TOP1_PROB_HIST_BINS)],
        ),
        "trainer_state.last.entropy_hist_counts": (
            last["entropy_hist_counts"],
            [0 for _ in range(LOGIT_SCALE_HIST_BINS)],
        ),
        "trainer_state.last.logsumexp_hist_counts": (
            last["logsumexp_hist_counts"],
            [0 for _ in range(LOGIT_SCALE_HIST_BINS)],
        ),
        "trainer_state.last.logit_norm_hist_counts": (
            last["logit_norm_hist_counts"],
            [0 for _ in range(LOGIT_SCALE_HIST_BINS)],
        ),
        "trainer_state.last.per_token_noise_level_hist_counts": (
            last["per_token_noise_level_hist_counts"],
            [0 for _ in range(PER_TOKEN_NOISE_LEVEL_HIST_BINS)],
        ),
    }
    for field, (value, template) in fixed_shapes.items():
        _validate_numeric_tree_shape(value, template, field=field)
    counter_fields = {
        "ar_boundary_counts": {
            "microbatches",
            "microbatches_with_bos",
            "valid_tokens",
            "special_tokens",
            "special_loss_tokens",
            "bos_tokens",
            "bos_loss_tokens",
            "eos_tokens",
            "eos_loss_tokens",
        },
        "boundary_telemetry": {
            "documents_per_row_sum",
            "documents_per_row_count",
            "pad_tokens",
            "total_row_tokens",
            "blocked_pairs_sum",
            "causal_pairs_sum",
            "valid_pairs_sum",
            "allowed_pairs_sum",
            "segment_blocked_pairs_sum",
        },
        "row_telemetry": {
            "rows",
            "valid_tokens_sum",
            "valid_tokens_min",
            "valid_tokens_max",
            "lossable_tokens_sum",
            "lossable_tokens_min",
            "lossable_tokens_max",
            "pad_tokens",
            "total_row_tokens",
            "eos_rows",
            "eos_tokens",
            "eos_loss_tokens",
        },
        "eos_supervision_train_counts": set(_new_eos_supervision_counts()),
    }
    for field, expected_keys in counter_fields.items():
        value = _require_exact_mapping_keys(
            lineage[field],
            expected_keys,
            field=f"trainer_state.lineage.{field}",
        )
        template = {
            key: 0.0 if field == "boundary_telemetry" and key == "documents_per_row_sum" else 0 for key in expected_keys
        }
        _validate_numeric_tree_shape(value, template, field=f"trainer_state.lineage.{field}")
    mapping_fields = {
        "trainer_state.last.soft_input_stats": last["soft_input_stats"],
        "trainer_state.last.uniform_estimator_stats": last["uniform_estimator_stats"],
        "trainer_state.last.gradient_component_norms": last["gradient_component_norms"],
        "trainer_state.last.update_component_norms": last["update_component_norms"],
        **{
            f"trainer_state.diagnostics.{key}": value
            for key, value in diagnostics.items()
            if key != "fixed_ce_trace_records"
        },
    }
    for field, value in mapping_fields.items():
        if not isinstance(value, dict):
            raise TypeError(f"{field} must be a mapping")
        _validate_json_state(value, field=field)
    if not isinstance(diagnostics["fixed_ce_trace_records"], int) or isinstance(
        diagnostics["fixed_ce_trace_records"], bool
    ):
        raise TypeError("trainer_state.diagnostics.fixed_ce_trace_records must be an integer")
    container_fields = {
        "lineage": {
            "ar_boundary_counts",
            "boundary_telemetry",
            "row_telemetry",
            "eos_supervision_train_counts",
            "train_length_head_accumulator",
            "train_token_edit_reconstruction_accumulator",
            "train_per_token_noise_level_hist_counts",
        },
        "last": {
            "eos_hazard_accumulator",
            "length_head_accumulator",
            "position_decile_ce_accumulator",
            "token_edit_reconstruction_accumulator",
            "top1_prob_hist_counts",
            "entropy_hist_counts",
            "logsumexp_hist_counts",
            "logit_norm_hist_counts",
            "soft_input_stats",
            "uniform_estimator_stats",
            "per_token_noise_level_hist_counts",
            "gradient_component_norms",
            "update_component_norms",
            "grad_clipped",
        },
        "diagnostics": set(diagnostics),
    }
    for group_name in ("lineage", "interval", "last"):
        for key, value in groups[group_name].items():
            if key in container_fields.get(group_name, set()):
                continue
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"trainer_state.{group_name}.{key} must be numeric")
    if not isinstance(last["grad_clipped"], bool):
        raise TypeError("trainer_state.last.grad_clipped must be a boolean")


def _validate_rank_rng_payload(state: dict[str, object]) -> None:
    rng = _require_exact_mapping_keys(
        state,
        {
            "version",
            "torch_cpu",
            "numpy",
            "python",
            "device_type",
            "distributed_backend",
            "local_rank",
            "local_world_size",
            "group_rank",
            "cuda",
        },
        field="rank RNG state",
    )
    if rng["version"] != RESUME_RNG_STATE_VERSION:
        raise RuntimeError(f"unsupported rank RNG state version {rng['version']!r}")
    torch_state = _rng_byte_tensor(rng["torch_cpu"], field="torch_cpu")
    if torch_state.ndim != 1 or not torch_state.numel():
        raise RuntimeError("torch_cpu RNG state must be a nonempty byte vector")
    try:
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(rng["numpy"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid NumPy RNG state: {exc}") from exc
    try:
        python_probe = random.Random()
        python_probe.setstate(rng["python"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid Python RNG state: {exc}") from exc
    for key in ("device_type", "distributed_backend"):
        if not isinstance(rng[key], str) or not rng[key]:
            raise TypeError(f"rank RNG state {key} must be a nonempty string")
    for key in ("local_rank", "group_rank"):
        if not isinstance(rng[key], int) or isinstance(rng[key], bool) or int(rng[key]) < 0:
            raise TypeError(f"rank RNG state {key} must be a nonnegative integer")
    if (
        not isinstance(rng["local_world_size"], int)
        or isinstance(rng["local_world_size"], bool)
        or int(rng["local_world_size"]) <= 0
    ):
        raise TypeError("rank RNG state local_world_size must be a positive integer")
    cuda_state = rng["cuda"]
    if cuda_state is not None:
        cuda = _require_exact_mapping_keys(
            cuda_state,
            {"active_device_index", "visible_device_count", "device_name", "compute_capability", "state"},
            field="rank CUDA RNG state",
        )
        for key in ("active_device_index", "visible_device_count"):
            if not isinstance(cuda[key], int) or isinstance(cuda[key], bool) or int(cuda[key]) < 0:
                raise TypeError(f"rank CUDA RNG state {key} must be a nonnegative integer")
        if not isinstance(cuda["device_name"], str) or not cuda["device_name"]:
            raise TypeError("rank CUDA RNG device_name must be a nonempty string")
        capability = cuda["compute_capability"]
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in capability)
        ):
            raise TypeError("rank CUDA RNG compute_capability must contain two nonnegative integers")
        cuda_bytes = _rng_byte_tensor(cuda["state"], field="cuda.state")
        if cuda_bytes.ndim != 1 or not cuda_bytes.numel():
            raise RuntimeError("CUDA RNG state must be a nonempty byte vector")


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_batch_digest_schema(digest: dict[str, object]) -> None:
    root = _require_exact_mapping_keys(
        digest,
        {"version", "epoch", "tensor_fields", "tensors", "combined_sha256"},
        field="next-batch digest",
    )
    if root["version"] != RESUME_BATCH_DIGEST_VERSION:
        raise RuntimeError(f"unsupported next-batch digest version {root['version']!r}")
    if not isinstance(root["epoch"], int) or isinstance(root["epoch"], bool) or int(root["epoch"]) < 0:
        raise TypeError("next-batch digest epoch must be a nonnegative integer")
    if root["tensor_fields"] != list(RESUME_BATCH_TENSOR_FIELDS):
        raise RuntimeError("next-batch digest tensor field ordering is incompatible")
    tensors = _require_exact_mapping_keys(root["tensors"], set(RESUME_BATCH_TENSOR_FIELDS), field="batch tensors")
    for field in RESUME_BATCH_TENSOR_FIELDS:
        value = tensors[field]
        if not isinstance(value, dict) or not isinstance(value.get("present"), bool):
            raise TypeError(f"batch tensor digest {field} must declare boolean presence")
        if not value["present"]:
            _require_exact_mapping_keys(value, {"present"}, field=f"batch tensor digest {field}")
            continue
        item = _require_exact_mapping_keys(
            value,
            {"present", "dtype", "shape", "sha256"},
            field=f"batch tensor digest {field}",
        )
        if not isinstance(item["dtype"], str) or not item["dtype"]:
            raise TypeError(f"batch tensor digest {field} dtype must be a nonempty string")
        if not isinstance(item["shape"], list) or not all(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in item["shape"]
        ):
            raise TypeError(f"batch tensor digest {field} shape must contain nonnegative integers")
        if not _is_sha256_hex(item["sha256"]):
            raise TypeError(f"batch tensor digest {field} sha256 is malformed")
    if not _is_sha256_hex(root["combined_sha256"]):
        raise TypeError("next-batch combined sha256 is malformed")
    canonical = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != root["combined_sha256"]:
        raise RuntimeError("next-batch combined sha256 does not match its tensor records")


def _collect_resume_contract(
    config: ExperimentConfig,
    batch: DLMDataBatch,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    is_ddp: bool,
    grad_accum_steps: int,
    consumed_microbatches: int,
    trajectory_contract: dict[str, object],
    trainer_state: dict[str, object],
) -> dict[str, object]:
    if consumed_microbatches < 0:
        raise ValueError("consumed_microbatches must be nonnegative")
    if trainer_state.get("version") != RESUME_TRAINER_STATE_VERSION:
        raise ValueError("trainer_state has an unsupported or missing version")
    local_state = {
        "rank": int(rank),
        "rng_state": _active_rank_rng_state(config, device),
        "next_batch_six_tensor_digests": _batch_six_tensor_digests(batch),
        "trainer_state": copy.deepcopy(trainer_state),
    }
    if is_ddp:
        rank_states: list[object] = [None for _ in range(world_size)]
        dist.all_gather_object(rank_states, local_state)
    else:
        rank_states = [local_state]
    observed_ranks = [state.get("rank") if isinstance(state, dict) else None for state in rank_states]
    if observed_ranks != list(range(world_size)):
        raise RuntimeError(
            "resume-state gather did not return one ordered payload per rank: "
            f"observed={observed_ranks}, expected={list(range(world_size))}"
        )
    return {
        "version": RESUME_CONTRACT_VERSION,
        "world_size": int(world_size),
        "grad_accum_steps": int(grad_accum_steps),
        "consumed_microbatches": int(consumed_microbatches),
        "consumed_microbatches_scope": "processed_microbatches_per_rank",
        "replay_advance_count_per_rank": int(consumed_microbatches),
        "prefetched_next_batch_count_per_rank": 1,
        "loader_yields_per_rank": int(consumed_microbatches) + 1,
        "global_processed_microbatches": int(consumed_microbatches) * int(world_size),
        "checkpoint_boundary": "optimizer_step",
        "trajectory": copy.deepcopy(trajectory_contract),
        "rank_states": rank_states,
    }


def _load_resume_contract_local(
    checkpoint: dict,
    *,
    rank: int,
    world_size: int,
    grad_accum_steps: int,
    checkpoint_step: int,
    trajectory_contract: dict[str, object] | None,
    config: ExperimentConfig | None,
    device: torch.device | None,
) -> dict[str, object]:
    if checkpoint_step < 0:
        raise RuntimeError(f"checkpoint step must be nonnegative, got {checkpoint_step}")
    contract = checkpoint.get("resume_contract")
    if contract is None:
        if world_size != 1:
            raise RuntimeError(
                "checkpoint predates the versioned resume contract and cannot be resumed with world_size > 1; "
                "restart from a contract-bearing checkpoint"
            )
        legacy_rng_state = checkpoint.get("rng_state")
        if not isinstance(legacy_rng_state, dict) or not legacy_rng_state:
            raise RuntimeError("legacy single-rank checkpoint has no restorable RNG state")
        return {
            "version": 0,
            "status": "legacy_single_rank_compatibility",
            "world_size": 1,
            "grad_accum_steps": int(grad_accum_steps),
            "consumed_microbatches": int(checkpoint_step) * int(grad_accum_steps),
            "local_rng_state": legacy_rng_state,
            "local_trainer_state": None,
            "expected_next_batch_six_tensor_digests": None,
        }
    if not isinstance(contract, dict):
        raise TypeError("checkpoint resume_contract must be a mapping")
    version = contract.get("version")
    if version != RESUME_CONTRACT_VERSION:
        raise RuntimeError(
            f"unsupported checkpoint resume contract version {version!r}; expected {RESUME_CONTRACT_VERSION}"
        )
    saved_world_size = int(contract.get("world_size", -1))
    saved_grad_accum = int(contract.get("grad_accum_steps", -1))
    consumed_microbatches = int(contract.get("consumed_microbatches", -1))
    if consumed_microbatches < 0:
        raise RuntimeError(f"checkpoint consumed_microbatches must be nonnegative, got {consumed_microbatches}")
    if saved_world_size != world_size:
        raise RuntimeError(
            f"resume world_size mismatch: checkpoint={saved_world_size}, current={world_size}; exact resume "
            "requires the original world size"
        )
    if saved_grad_accum != grad_accum_steps:
        raise RuntimeError(
            f"resume grad_accum_steps mismatch: checkpoint={saved_grad_accum}, current={grad_accum_steps}"
        )
    if contract.get("checkpoint_boundary") != "optimizer_step":
        raise RuntimeError("checkpoint has an unsupported or missing checkpoint boundary")
    if contract.get("consumed_microbatches_scope") != "processed_microbatches_per_rank":
        raise RuntimeError("checkpoint has an unsupported consumed_microbatches scope")
    expected_consumed = int(checkpoint_step) * saved_grad_accum
    if consumed_microbatches != expected_consumed:
        raise RuntimeError(
            "checkpoint step and consumed microbatch count disagree: "
            f"step={checkpoint_step}, grad_accum_steps={saved_grad_accum}, "
            f"consumed_microbatches={consumed_microbatches}, expected={expected_consumed}"
        )
    expected_loader_yields = consumed_microbatches + 1
    if int(contract.get("replay_advance_count_per_rank", -1)) != consumed_microbatches:
        raise RuntimeError("checkpoint replay advance count disagrees with consumed microbatches")
    if int(contract.get("prefetched_next_batch_count_per_rank", -1)) != 1:
        raise RuntimeError("checkpoint resume contract must contain exactly one prefetched next batch")
    if int(contract.get("loader_yields_per_rank", -1)) != expected_loader_yields:
        raise RuntimeError("checkpoint loader-yield count disagrees with replay and prefetch counts")
    if int(contract.get("global_processed_microbatches", -1)) != consumed_microbatches * world_size:
        raise RuntimeError("checkpoint global processed-microbatch count is inconsistent")
    saved_trajectory = contract.get("trajectory")
    if not isinstance(saved_trajectory, dict) or saved_trajectory.get("version") != RESUME_TRAJECTORY_VERSION:
        raise RuntimeError("checkpoint has no supported trajectory contract")
    saved_trajectory_record = saved_trajectory.get("record")
    if not isinstance(saved_trajectory_record, dict):
        raise RuntimeError("checkpoint trajectory contract has no canonical record")
    recomputed_trajectory_sha256 = _trajectory_record_sha256(saved_trajectory_record)
    if saved_trajectory.get("sha256") != recomputed_trajectory_sha256:
        raise RuntimeError(
            "checkpoint trajectory record hash mismatch: "
            f"stored={saved_trajectory.get('sha256')}, recomputed={recomputed_trajectory_sha256}"
        )
    if trajectory_contract is None:
        raise RuntimeError("current trajectory contract is required for exact resume")
    if saved_trajectory.get("sha256") != trajectory_contract.get("sha256"):
        raise RuntimeError(
            "resume trajectory mismatch: "
            f"checkpoint={saved_trajectory.get('sha256')}, current={trajectory_contract.get('sha256')}"
        )
    rank_states = contract.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise RuntimeError("checkpoint resume contract does not contain one state for every rank")
    observed_ranks = [state.get("rank") if isinstance(state, dict) else None for state in rank_states]
    if observed_ranks != list(range(world_size)):
        raise RuntimeError(f"checkpoint rank states are not ordered by rank: {observed_ranks}")
    local_state = rank_states[rank]
    local_rng_state = local_state.get("rng_state")
    local_trainer_state = local_state.get("trainer_state")
    expected_batch_digests = local_state.get("next_batch_six_tensor_digests")
    if not isinstance(local_rng_state, dict) or not local_rng_state:
        raise RuntimeError(f"checkpoint has no restorable RNG state for rank {rank}")
    _validate_rank_rng_payload(local_rng_state)
    if config is None or device is None:
        raise RuntimeError("current config and device are required to validate rank RNG topology")
    _validate_active_rank_rng_topology(local_rng_state, config, device)
    if not isinstance(local_trainer_state, dict):
        raise RuntimeError(f"checkpoint has no trainer state for rank {rank}")
    _validate_trainer_state_schema(local_trainer_state, config)
    if not isinstance(expected_batch_digests, dict):
        raise RuntimeError(f"checkpoint has no next-batch digest for rank {rank}")
    _validate_batch_digest_schema(expected_batch_digests)
    return {
        "version": RESUME_CONTRACT_VERSION,
        "status": "exact_contract",
        "world_size": saved_world_size,
        "grad_accum_steps": saved_grad_accum,
        "consumed_microbatches": consumed_microbatches,
        "local_rng_state": local_rng_state,
        "local_trainer_state": local_trainer_state,
        "expected_next_batch_six_tensor_digests": expected_batch_digests,
    }


def _load_resume_contract(
    checkpoint: dict,
    *,
    rank: int,
    world_size: int,
    grad_accum_steps: int,
    checkpoint_step: int,
    trajectory_contract: dict[str, object] | None = None,
    config: ExperimentConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, object]:
    state: dict[str, object] | None = None
    error: Exception | None = None
    try:
        state = _load_resume_contract_local(
            checkpoint,
            rank=rank,
            world_size=world_size,
            grad_accum_steps=grad_accum_steps,
            checkpoint_step=checkpoint_step,
            trajectory_contract=trajectory_contract,
            config=config,
            device=device,
        )
    except Exception as exc:
        error = exc
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        local_result = {
            "rank": int(rank),
            "error": f"{type(error).__name__}: {error}" if error is not None else "",
        }
        gathered: list[object] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_result)
        failures = [item for item in gathered if isinstance(item, dict) and item.get("error")]
        if failures:
            details = "; ".join(f"rank {item['rank']}: {item['error']}" for item in failures)
            raise RuntimeError(f"resume contract validation failed across ranks: {details}")
    if error is not None:
        raise error
    assert state is not None
    return state


def _verify_replayed_next_batch(resume_state: dict[str, object], batch: DLMDataBatch, *, rank: int) -> bool:
    expected = resume_state.get("expected_next_batch_six_tensor_digests")
    if expected is None:
        return False
    observed = _batch_six_tensor_digests(batch)
    if observed != expected:
        expected_hash = expected.get("combined_sha256") if isinstance(expected, dict) else None
        raise RuntimeError(
            f"deterministic loader replay mismatch on rank {rank}: "
            f"expected next-batch digest {expected_hash}, observed {observed['combined_sha256']}"
        )
    return True


def _synchronize_resume_phase_error(
    phase: str,
    error: Exception | None,
    *,
    rank: int,
    world_size: int,
    is_ddp: bool,
) -> None:
    """Make every rank fail the resume phase before any rank can continue."""
    if is_ddp:
        local_result = {
            "rank": int(rank),
            "error": f"{type(error).__name__}: {error}" if error is not None else "",
        }
        gathered: list[object] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_result)
        failures = [item for item in gathered if isinstance(item, dict) and item.get("error")]
        if failures:
            details = "; ".join(f"rank {item['rank']}: {item['error']}" for item in failures)
            raise RuntimeError(f"resume {phase} failed across ranks: {details}")
    if error is not None:
        raise error


def _rng_byte_tensor(value, *, field: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if isinstance(value, (list, tuple, np.ndarray)):
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()
    raise TypeError(f"{field} RNG state must be a tensor or byte sequence, got {type(value).__name__}")


def _restore_rng_state(state: dict, config: ExperimentConfig) -> None:
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(_rng_byte_tensor(state["torch"], field="torch"))
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])
    if config.runtime.device_type == "cuda" and "cuda" in state and torch.cuda.is_available():
        cuda_states = state["cuda"]
        if not isinstance(cuda_states, (list, tuple)):
            raise TypeError(f"cuda RNG state must be a sequence, got {type(cuda_states).__name__}")
        normalized_cuda_states = [
            _rng_byte_tensor(cuda_state, field=f"cuda[{index}]") for index, cuda_state in enumerate(cuda_states)
        ]
        torch.cuda.set_rng_state_all(normalized_cuda_states)


def save_checkpoint(
    path: str,
    raw_model: nn.Module,
    optimizers,
    scaler,
    config: ExperimentConfig,
    model_config,
    step: int,
    train_tokens: int,
    train_valid_tokens: int,
    metrics: dict,
    *,
    resume_contract: dict[str, object] | None = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if resume_contract is not None:
        rank_states = resume_contract.get("rank_states")
        if not isinstance(rank_states, list) or not rank_states or not isinstance(rank_states[0], dict):
            raise RuntimeError("resume contract has no rank-0 RNG state")
        rank_zero_rng = rank_states[0].get("rng_state")
        if not isinstance(rank_zero_rng, dict):
            raise RuntimeError("resume contract has malformed rank-0 RNG state")
        legacy_rng_state = _legacy_rng_state_from_active(rank_zero_rng)
    else:
        legacy_rng_state = _rng_state(config)
    payload = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": _optimizer_state_dict(optimizers),
        "scaler_state_dict": scaler.state_dict(),
        # Retain rank-0 state under the legacy key for model-only replay and old readers.
        "rng_state": legacy_rng_state,
        "resume_contract": resume_contract,
        "config": asdict(model_config),
        "train_config": to_dict(config),
        "step": step,
        "train_tokens": train_tokens,
        "train_valid_tokens": train_valid_tokens,
        "metrics": metrics,
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def alias_checkpoint(source_path: str, alias_path: str) -> None:
    """Make alias_path load the same checkpoint without another full save."""
    if os.path.exists(alias_path):
        try:
            if os.path.samefile(source_path, alias_path):
                return
        except OSError:
            pass
    os.makedirs(os.path.dirname(alias_path), exist_ok=True)
    tmp_path = alias_path + ".tmp"
    with suppress(FileNotFoundError):
        os.unlink(tmp_path)
    try:
        os.link(source_path, tmp_path)
    except OSError:
        try:
            os.symlink(os.path.basename(source_path), tmp_path)
        except OSError:
            shutil.copy2(source_path, tmp_path)
    os.replace(tmp_path, alias_path)


def _resolve_resume_path(run_dir: str, resume_from: str) -> str | None:
    if not resume_from:
        return None
    if resume_from == "latest":
        candidates = [
            os.path.join(run_dir, "model_latest.pt"),
            os.path.join(run_dir, "model.pt"),
        ]
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        if os.path.isdir(checkpoints_dir):
            candidates.extend(
                str(p)
                for p in sorted(
                    Path(checkpoints_dir).glob("model_step*.pt"), key=lambda p: p.stat().st_mtime, reverse=True
                )
            )
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"no checkpoint found for train.resume_from='latest' in {run_dir}")
    return resume_from


def _dedupe_path_objects(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        out.append(path)
        seen.add(key)
    return out


def _artifact_root_candidates(run_dir: str) -> list[Path]:
    run_path = Path(run_dir).expanduser()
    if not run_path.is_absolute():
        run_path = Path.cwd() / run_path
    run_path = run_path.resolve()
    runs_anchor = next((path for path in (run_path, *run_path.parents) if path.name == "runs"), None)
    candidates = [runs_anchor.parent] if runs_anchor is not None else [run_path.parent]
    candidates.append(Path.cwd().resolve())
    return _dedupe_path_objects(candidates)


def _resolve_artifact_path(artifact_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return artifact_root / path


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _comparison_manifest_path(run_dir: str, run_id: str) -> tuple[Path, Path]:
    for artifact_root in _artifact_root_candidates(run_dir):
        manifest_path = artifact_root / "runs" / run_id / "artifacts.json"
        if manifest_path.exists():
            return manifest_path, artifact_root
    candidates = [str(root / "runs" / run_id / "artifacts.json") for root in _artifact_root_candidates(run_dir)]
    raise FileNotFoundError(f"could not find artifacts.json for comparison run {run_id!r}; checked {candidates}")


def _manifest_artifacts(manifest: dict) -> list[dict]:
    artifacts = manifest.get("artifacts", [])
    return [artifact for artifact in artifacts if isinstance(artifact, dict)] if isinstance(artifacts, list) else []


def _find_manifest_artifact_path(
    manifest: dict, artifact_root: Path, *, name: str, kind: str | None = None
) -> Path | None:
    for artifact in _manifest_artifacts(manifest):
        if artifact.get("name") != name:
            continue
        if kind is not None and artifact.get("kind") != kind:
            continue
        path = _resolve_artifact_path(artifact_root, str(artifact.get("path", "")))
        if path.exists():
            return path
    return None


def _source_run_dir_from_manifest(manifest: dict, artifact_root: Path, manifest_path: Path) -> Path:
    metrics_path = _find_manifest_artifact_path(manifest, artifact_root, name="metrics.json", kind="metrics")
    if metrics_path is not None:
        return metrics_path.parent
    return manifest_path.parent


def _find_checkpoint_from_manifest(manifest: dict, artifact_root: Path, source_run_dir: Path) -> Path:
    checkpoint_names = {"model_latest.pt", "model.pt"}
    for artifact in _manifest_artifacts(manifest):
        name = str(artifact.get("name", ""))
        path_text = str(artifact.get("path", ""))
        if artifact.get("kind") != "checkpoint" and Path(name).name not in checkpoint_names:
            continue
        path = _resolve_artifact_path(artifact_root, path_text)
        if path.exists() and path.suffix == ".pt":
            return path

    candidates = [source_run_dir / "model_latest.pt", source_run_dir / "model.pt"]
    checkpoints_dir = source_run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        candidates.extend(sorted(checkpoints_dir.glob("model_step*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no checkpoint artifact found for comparison run at {source_run_dir}")


def _resolve_comparison_sources(config: ExperimentConfig) -> list[dict[str, str | list[str] | dict]]:
    run_ids = [run_id.strip() for run_id in config.train.comparison_run_ids if run_id.strip()]
    if not run_ids:
        return []

    sources: list[dict[str, str | list[str] | dict]] = []
    for run_id in run_ids:
        manifest_path, artifact_root = _comparison_manifest_path(config.train.run_dir, run_id)
        manifest = _load_json_if_exists(manifest_path)
        manifest_run_id = str(manifest.get("run_id", ""))
        if manifest_run_id and manifest_run_id != run_id:
            raise ValueError(f"comparison manifest {manifest_path} has run_id={manifest_run_id!r}, expected {run_id!r}")
        source_run_dir = _source_run_dir_from_manifest(manifest, artifact_root, manifest_path)
        source_metrics_path = _find_manifest_artifact_path(manifest, artifact_root, name="metrics.json", kind="metrics")
        source_metrics = _load_json_if_exists(source_metrics_path) if source_metrics_path is not None else {}
        if source_metrics and source_metrics.get("status") != "ok":
            raise RuntimeError(
                f"comparison run {run_id} is not a clean source run: status={source_metrics.get('status')!r}"
            )
        checkpoint_path = _find_checkpoint_from_manifest(manifest, artifact_root, source_run_dir)
        sources.append(
            {
                "source_run_ids": [run_id],
                "source_run_id": run_id,
                "source_checkpoint": str(checkpoint_path),
                "source_checkpoint_path": str(checkpoint_path),
                "source_artifacts_manifest": str(manifest_path),
                "source_metrics_path": str(source_metrics_path) if source_metrics_path is not None else "",
                "source_run_dir": str(source_run_dir),
                "source_metrics": source_metrics,
            }
        )
    return sources


def _resolve_comparison_source(config: ExperimentConfig) -> dict[str, str | list[str] | dict]:
    sources = _resolve_comparison_sources(config)
    if not sources:
        return {"source_run_ids": [], "source_metrics": {}}
    if len(sources) != 1:
        raise ValueError("eval-only replay supports exactly one comparison run checkpoint")
    source = dict(sources[0])
    source["source_run_ids"] = [str(source.get("source_run_id", ""))]
    return source


def _resolve_eval_only_resume(config: ExperimentConfig) -> tuple[str | None, dict[str, str | list[str] | dict]]:
    source_info = _resolve_comparison_source(config)
    resume_from = config.train.resume_from.strip()
    if source_info.get("source_checkpoint") and resume_from in {"", "comparison", "comparison_latest"}:
        return str(source_info["source_checkpoint"]), source_info
    return _resolve_resume_path(config.train.run_dir, resume_from), source_info


def _load_checkpoint(
    path: str,
    raw_model: nn.Module,
    optimizers,
    scaler,
    config: ExperimentConfig,
    device,
    *,
    rank: int,
    world_size: int,
    grad_accum_steps: int,
    trajectory_contract: dict[str, object],
) -> tuple[int, int, int, dict, dict[str, object]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    checkpoint_step = int(ckpt.get("step", 0))
    resume_state = _load_resume_contract(
        ckpt,
        rank=rank,
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
        checkpoint_step=checkpoint_step,
        trajectory_contract=trajectory_contract,
        config=config,
        device=device,
    )
    raw_model.load_state_dict(ckpt["model_state_dict"])
    _load_optimizer_state_dict(optimizers, ckpt["optimizer_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    metrics = dict(ckpt.get("metrics", {}))
    return (
        checkpoint_step,
        int(ckpt.get("train_tokens", 0)),
        int(ckpt.get("train_valid_tokens", metrics.get("train_valid_tokens", 0))),
        metrics,
        resume_state,
    )


def _grad_norm(parameters) -> float:
    norms = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), 2).item())


def _parameter_norm(parameters) -> float:
    norms = [p.detach().float().norm(2) for p in parameters]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), 2).item())


def _parameter_norm_metrics(raw_model: nn.Module) -> dict[str, float]:
    return {
        "parameter_norm_transformer": _parameter_norm(raw_model.blocks.parameters()),
        "parameter_norm_embedding": _parameter_norm(raw_model.wte.parameters()),
        "parameter_norm_lm_head": _parameter_norm(raw_model.lm_head.parameters()),
        "parameter_norm_length_head": _parameter_norm(
            raw_model.length_head.parameters() if getattr(raw_model, "length_head", None) is not None else []
        ),
        "parameter_norm_target_length_conditioning": _parameter_norm(
            list(raw_model.target_length_embedding.parameters()) + list(raw_model.target_distance_proj.parameters())
            if getattr(raw_model, "target_length_embedding", None) is not None
            and getattr(raw_model, "target_distance_proj", None) is not None
            else []
        ),
        "parameter_norm_mask_position_embedding": _parameter_norm(
            raw_model.mask_position_embedding.parameters()
            if getattr(raw_model, "mask_position_embedding", None) is not None
            else []
        ),
        "parameter_norm_final_norm": _parameter_norm(raw_model.final_norm.parameters()),
        "parameter_norm_class_head": _parameter_norm(
            raw_model.class_head.parameters() if raw_model.class_head is not None else []
        ),
        "parameter_norm_auxiliary_heads": _parameter_norm(raw_model.intermediate_heads.parameters()),
    }


def _gradient_component_norm_metrics(raw_model: nn.Module) -> dict[str, float]:
    return {
        "grad_norm_transformer": _grad_norm(raw_model.blocks.parameters()),
        "grad_norm_embedding": _grad_norm(raw_model.wte.parameters()),
        "grad_norm_lm_head": _grad_norm(raw_model.lm_head.parameters()),
        "grad_norm_length_head": _grad_norm(
            raw_model.length_head.parameters() if getattr(raw_model, "length_head", None) is not None else []
        ),
        "grad_norm_target_length_conditioning": _grad_norm(
            list(raw_model.target_length_embedding.parameters()) + list(raw_model.target_distance_proj.parameters())
            if getattr(raw_model, "target_length_embedding", None) is not None
            and getattr(raw_model, "target_distance_proj", None) is not None
            else []
        ),
        "grad_norm_mask_position_embedding": _grad_norm(
            raw_model.mask_position_embedding.parameters()
            if getattr(raw_model, "mask_position_embedding", None) is not None
            else []
        ),
        "grad_norm_final_norm": _grad_norm(raw_model.final_norm.parameters()),
        "grad_norm_class_head": _grad_norm(
            raw_model.class_head.parameters() if raw_model.class_head is not None else []
        ),
        "grad_norm_auxiliary_heads": _grad_norm(raw_model.intermediate_heads.parameters()),
    }


@torch.no_grad()
def _second_moment_norm(optimizers) -> float:
    total_sq = 0.0
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for p in group.get("params", []):
                state = optimizer.state.get(p)
                if state is not None and "exp_avg_sq" in state:
                    total_sq += float(state["exp_avg_sq"].float().norm(2).item() ** 2)
    return math.sqrt(total_sq) if total_sq > 0 else 0.0


@torch.no_grad()
def _update_norm(pre_step_params, post_step_params) -> float:
    total_sq = 0.0
    for pre, post in zip(pre_step_params, post_step_params, strict=True):
        total_sq += float((pre.float() - post.detach().float()).norm(2).item() ** 2)
    return math.sqrt(total_sq) if total_sq > 0 else 0.0


@torch.no_grad()
def _component_update_norm_metrics(
    pre_step_components: dict[str, list[torch.Tensor]] | None,
    raw_model: nn.Module,
) -> dict[str, float]:
    metrics = {
        "update_norm_embedding": 0.0,
        "update_norm_lm_head": 0.0,
    }
    if pre_step_components is None:
        return metrics
    component_params = {
        "embedding": list(raw_model.wte.parameters()),
        "lm_head": list(raw_model.lm_head.parameters()),
    }
    for name, post_params in component_params.items():
        pre_params = pre_step_components.get(name)
        if pre_params is None:
            continue
        metrics[f"update_norm_{name}"] = _update_norm(pre_params, post_params)
    return metrics


def _component_update_grad_ratio_metrics(
    update_metrics: dict[str, float],
    grad_metrics: dict[str, float],
) -> dict[str, float]:
    return {
        "update_grad_ratio_embedding": update_metrics.get("update_norm_embedding", 0.0)
        / max(grad_metrics.get("grad_norm_embedding", 0.0), 1e-8),
        "update_grad_ratio_lm_head": update_metrics.get("update_norm_lm_head", 0.0)
        / max(grad_metrics.get("grad_norm_lm_head", 0.0), 1e-8),
    }


def _residual_projection_init_metrics(config: ExperimentConfig) -> dict[str, float | str]:
    return {
        "residual_output_projection_init": config.model.residual_output_projection_init,
        "residual_output_projection_init_std": residual_output_projection_init_std(int(config.model.depth)),
    }


def _embedding_head_init_metrics(config: ExperimentConfig) -> dict[str, float]:
    base_std = embedding_head_init_std(int(config.model.n_embd))
    multiplier = float(config.model.embedding_head_init_std_multiplier)
    return {
        "embedding_head_init_std_base": base_std,
        "embedding_head_init_std_multiplier": multiplier,
        "embedding_init_std": embedding_head_init_std(int(config.model.n_embd), multiplier),
        "lm_head_init_std": embedding_head_init_std(int(config.model.n_embd), multiplier),
    }


def _embedding_head_init_diagnostics_enabled(config: ExperimentConfig) -> bool:
    return not math.isclose(float(config.model.embedding_head_init_std_multiplier), 1.0)


def _prefixed_metric_keys(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _tensor_rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.sqrt(value.detach().float().pow(2).mean()).item())


def _logit_confidence_statistics(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=-1).float()
    probs = log_probs.exp()
    entropy_values = -(probs * log_probs).sum(dim=-1)
    max_probs, argmax_ids = probs.max(dim=-1)
    raw_entropy = entropy_values.mean()
    raw_max_prob = max_probs.mean()
    if argmax_ids.numel() > 0:
        mode_val = torch.mode(argmax_ids).values
        raw_argmax_mode_frac = (argmax_ids == mode_val).float().mean()
    else:
        raw_argmax_mode_frac = torch.tensor(0.0, device=logits.device)
    return entropy_values, max_probs, raw_entropy, raw_max_prob, raw_argmax_mode_frac


def _logit_scale_statistics(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_f = logits.float()
    logsumexp_values = torch.logsumexp(logits_f, dim=-1)
    z_loss_values = logsumexp_values.square()
    logit_norms = logits_f.norm(dim=-1)
    logit_rms = torch.sqrt(logits_f.square().mean(dim=-1))
    return z_loss_values, logsumexp_values, logit_norms, logit_rms


def _accumulate_probability_histogram(counts: list[int], values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    bins = len(counts)
    bucket_ids = torch.clamp((values.detach().float() * bins).to(torch.long), min=0, max=bins - 1)
    hist = torch.bincount(bucket_ids.cpu(), minlength=bins)
    for idx, count in enumerate(hist.tolist()[:bins]):
        counts[idx] += int(count)


def _probability_histogram_metrics(prefix: str, counts: list[int]) -> dict[str, float | int]:
    total = int(sum(counts))
    metrics: dict[str, float | int] = {f"{prefix}_count": total}
    bins = len(counts)
    for idx, count in enumerate(counts):
        low = int(100 * idx / bins)
        high = int(100 * (idx + 1) / bins)
        label = f"{low:02d}_{high:03d}" if high == 100 else f"{low:02d}_{high:02d}"
        metrics[f"{prefix}_bin_{label}_count"] = int(count)
        metrics[f"{prefix}_bin_{label}_frac"] = float(count) / max(1, total)
    return metrics


def _accumulate_value_histogram(
    counts: list[int],
    values: torch.Tensor,
    min_value: float,
    max_value: float,
) -> None:
    if values.numel() == 0:
        return
    bins = len(counts)
    if bins <= 0 or max_value <= min_value:
        raise ValueError("histogram requires positive bins and max_value > min_value")
    values_f = torch.nan_to_num(
        values.detach().float(),
        nan=float(min_value),
        posinf=float(max_value),
        neginf=float(min_value),
    )
    scaled = (values_f - float(min_value)) / (float(max_value) - float(min_value))
    bucket_ids = torch.clamp((scaled * bins).to(torch.long), min=0, max=bins - 1)
    hist = torch.bincount(bucket_ids.cpu(), minlength=bins)
    for idx, count in enumerate(hist.tolist()[:bins]):
        counts[idx] += int(count)


def _histogram_edge_label(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value))).replace("-", "m")
    return f"{value:.2f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _value_histogram_metrics(
    prefix: str,
    counts: list[int],
    min_value: float,
    max_value: float,
) -> dict[str, float | int]:
    total = int(sum(counts))
    bins = len(counts)
    metrics: dict[str, float | int] = {
        f"{prefix}_count": total,
        f"{prefix}_range_min": float(min_value),
        f"{prefix}_range_max": float(max_value),
    }
    if bins <= 0:
        return metrics
    width = (float(max_value) - float(min_value)) / bins
    for idx, count in enumerate(counts):
        low = float(min_value) + idx * width
        high = float(min_value) + (idx + 1) * width
        label = f"{_histogram_edge_label(low)}_{_histogram_edge_label(high)}"
        metrics[f"{prefix}_bin_{label}_count"] = int(count)
        metrics[f"{prefix}_bin_{label}_frac"] = float(count) / max(1, total)
    return metrics


def _per_token_effective_masked_fraction_metrics(
    prefix: str,
    fraction_sum: float,
    fraction_count: int,
    fraction_min: float,
    fraction_max: float,
) -> dict[str, float | int]:
    return {
        f"{prefix}per_token_noise_effective_masked_fraction_count": int(fraction_count),
        f"{prefix}per_token_noise_effective_masked_fraction_mean": (
            float(fraction_sum) / int(fraction_count) if fraction_count else 0.0
        ),
        f"{prefix}per_token_noise_effective_masked_fraction_min": (
            float(fraction_min) if fraction_count and math.isfinite(fraction_min) else 0.0
        ),
        f"{prefix}per_token_noise_effective_masked_fraction_max": (
            float(fraction_max) if fraction_count and math.isfinite(fraction_max) else 0.0
        ),
    }


def _per_target_t_values(t_val: torch.Tensor, x_0: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    if t_val.shape == x_0.shape:
        return t_val.reshape(-1).index_select(0, target_indices).float()
    if t_val.ndim == 1 and t_val.shape[0] == x_0.shape[0]:
        target_rows = torch.div(target_indices, x_0.shape[1], rounding_mode="floor")
        return t_val.index_select(0, target_rows).float()
    raise ValueError(
        "tensor t must be either per-token with shape "
        f"{tuple(x_0.shape)} or per-row with shape ({x_0.shape[0]},), got {tuple(t_val.shape)}"
    )


def _new_token_edit_reconstruction_accumulator(
    num_deciles: int = TOKEN_EDIT_CORRUPTION_DECILES,
) -> dict[str, list[int] | int]:
    return {
        "counts": [0 for _ in range(int(num_deciles) + 1)],
        "correct": [0 for _ in range(int(num_deciles) + 1)],
        "microbatches": 0,
    }


def _update_token_edit_reconstruction_accumulator(
    accumulator: dict[str, list[int] | int],
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_class_ids: torch.Tensor,
    world_size: int = 1,
) -> None:
    if logits.numel() == 0 or targets.numel() == 0:
        return
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(f"token-edit logits/targets mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    class_ids = token_class_ids.to(device=targets.device).index_select(0, targets.detach().to(torch.long))
    class_ids = class_ids.clamp(0, TOKEN_EDIT_CORRUPTION_DECILES)
    pred = logits.detach().argmax(dim=-1)
    correct = (pred == targets).to(torch.long)
    counts = accumulator["counts"]
    correct_counts = accumulator["correct"]
    if not isinstance(counts, list) or not isinstance(correct_counts, list):
        raise ValueError("token-edit reconstruction accumulator is malformed")
    for decile in range(TOKEN_EDIT_CORRUPTION_DECILES + 1):
        mask = class_ids == decile
        count = int(mask.sum().item()) * int(world_size)
        if count == 0:
            continue
        counts[decile] += count
        correct_counts[decile] += int(correct[mask].sum().item()) * int(world_size)


def _merge_token_edit_reconstruction_accumulator(
    dest: dict[str, list[int] | int],
    src: dict[str, list[int] | int],
) -> None:
    dest_counts = dest["counts"]
    dest_correct = dest["correct"]
    src_counts = src["counts"]
    src_correct = src["correct"]
    if not all(isinstance(values, list) for values in (dest_counts, dest_correct, src_counts, src_correct)):
        raise ValueError("token-edit reconstruction accumulator is malformed")
    for idx in range(min(len(dest_counts), len(src_counts))):
        dest_counts[idx] += int(src_counts[idx])
        dest_correct[idx] += int(src_correct[idx])
    dest["microbatches"] = int(dest["microbatches"]) + int(src["microbatches"])


def _token_edit_reconstruction_metrics(
    prefix: str,
    accumulator: dict[str, list[int] | int],
    enabled: bool,
) -> dict[str, float | int | bool]:
    counts = [int(v) for v in accumulator["counts"]] if isinstance(accumulator["counts"], list) else []
    correct = [int(v) for v in accumulator["correct"]] if isinstance(accumulator["correct"], list) else []
    total = sum(counts)
    total_correct = sum(correct)
    metrics: dict[str, float | int | bool] = {
        f"{prefix}t2t_corrupted_token_reconstruction_enabled": bool(enabled),
        f"{prefix}t2t_corrupted_token_reconstruction_deciles": TOKEN_EDIT_CORRUPTION_DECILES,
        f"{prefix}t2t_corrupted_token_reconstruction_microbatches": int(accumulator["microbatches"]),
        f"{prefix}t2t_corrupted_token_reconstruction_count": int(total),
        f"{prefix}t2t_corrupted_token_reconstruction_correct": int(total_correct),
        f"{prefix}t2t_corrupted_token_reconstruction_accuracy": total_correct / max(1, total),
    }
    if counts:
        special_count = counts[0]
        special_correct = correct[0]
        metrics[f"{prefix}t2t_corrupted_token_reconstruction_special_count"] = special_count
        metrics[f"{prefix}t2t_corrupted_token_reconstruction_special_accuracy"] = special_correct / max(
            1, special_count
        )
    for decile in range(1, TOKEN_EDIT_CORRUPTION_DECILES + 1):
        count = counts[decile] if decile < len(counts) else 0
        correct_count = correct[decile] if decile < len(correct) else 0
        metrics[f"{prefix}t2t_corrupted_token_reconstruction_decile_{decile}_count"] = int(count)
        metrics[f"{prefix}t2t_corrupted_token_reconstruction_decile_{decile}_correct"] = int(correct_count)
        metrics[f"{prefix}t2t_corrupted_token_reconstruction_decile_{decile}_accuracy"] = correct_count / max(1, count)
    return metrics


def _relative_position_decile_ids(
    flat_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    num_deciles: int,
) -> torch.Tensor:
    if flat_indices.numel() == 0:
        return torch.empty_like(flat_indices, dtype=torch.long)
    seq_len = int(attention_mask.shape[1])
    rows = torch.div(flat_indices.to(torch.long), seq_len, rounding_mode="floor")
    cols = flat_indices.to(torch.long) - rows * seq_len
    valid_lengths = attention_mask.to(torch.long).sum(dim=1).clamp_min(1)
    denom = valid_lengths.index_select(0, rows).sub(1).clamp_min(1)
    relative = cols.float() / denom.float()
    return torch.clamp((relative * int(num_deciles)).to(torch.long), min=0, max=int(num_deciles) - 1)


def _position_decile_balanced_ce_loss(
    ce_values: torch.Tensor,
    target_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    token_weights: torch.Tensor,
    num_deciles: int = POSITION_DECILE_BALANCED_LOSS_DECILES,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if ce_values.shape != target_indices.shape:
        raise ValueError(
            f"CE values and target indices must have the same shape, got {ce_values.shape} and {target_indices.shape}"
        )
    if token_weights.shape != ce_values.shape:
        raise ValueError(f"token_weights must match CE values, got {token_weights.shape} and {ce_values.shape}")
    if loss_mask.shape != attention_mask.shape:
        raise ValueError(f"loss_mask must match attention_mask, got {loss_mask.shape} and {attention_mask.shape}")
    device = ce_values.device
    num_deciles = int(num_deciles)
    active_counts = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    lossable_counts = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    raw_ce_sums = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    weighted_ce_sums = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    balanced_contributions = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    lossable_indices = (loss_mask & attention_mask).reshape(-1).nonzero(as_tuple=True)[0]
    if lossable_indices.numel() > 0:
        lossable_decile_ids = _relative_position_decile_ids(lossable_indices, attention_mask, num_deciles)
        lossable_counts.scatter_add_(
            0,
            lossable_decile_ids,
            torch.ones_like(lossable_decile_ids, dtype=torch.float32),
        )
    if ce_values.numel() == 0:
        return ce_values.float().sum() * 0.0, {
            "active_counts": active_counts,
            "lossable_counts": lossable_counts,
            "raw_ce_sums": raw_ce_sums,
            "weighted_ce_sums": weighted_ce_sums,
            "balanced_contributions": balanced_contributions,
            "loss": torch.tensor(0.0, device=device),
        }

    decile_ids = _relative_position_decile_ids(target_indices, attention_mask, num_deciles)
    raw_values = ce_values.float()
    weighted_values = raw_values * token_weights.float()
    ones = torch.ones_like(raw_values, dtype=torch.float32)
    active_counts.scatter_add_(0, decile_ids, ones)
    raw_ce_sums.scatter_add_(0, decile_ids, raw_values)
    weighted_ce_sums.scatter_add_(0, decile_ids, weighted_values)
    lossable_deciles = lossable_counts > 0
    decile_losses = weighted_ce_sums / lossable_counts.clamp_min(1.0)
    nonempty_count = lossable_deciles.float().sum().clamp_min(1.0)
    balanced_contributions = torch.where(
        lossable_deciles,
        decile_losses / nonempty_count,
        balanced_contributions,
    )
    loss = balanced_contributions.sum()
    return loss, {
        "active_counts": active_counts.detach(),
        "lossable_counts": lossable_counts.detach(),
        "raw_ce_sums": raw_ce_sums.detach(),
        "weighted_ce_sums": weighted_ce_sums.detach(),
        "balanced_contributions": balanced_contributions.detach(),
        "loss": loss.detach(),
    }


def _new_position_decile_ce_accumulator(
    num_deciles: int = POSITION_DECILE_BALANCED_LOSS_DECILES,
) -> dict[str, list[float] | float | int]:
    return {
        "active_counts": [0.0 for _ in range(int(num_deciles))],
        "lossable_counts": [0.0 for _ in range(int(num_deciles))],
        "raw_ce_sums": [0.0 for _ in range(int(num_deciles))],
        "weighted_ce_sums": [0.0 for _ in range(int(num_deciles))],
        "balanced_contributions": [0.0 for _ in range(int(num_deciles))],
        "loss_sum": 0.0,
        "microbatches": 0,
    }


def _update_position_decile_ce_accumulator(
    accumulator: dict[str, list[float] | float | int],
    stats: dict[str, torch.Tensor],
    world_size: int = 1,
) -> None:
    for key in ("active_counts", "lossable_counts", "raw_ce_sums", "weighted_ce_sums", "balanced_contributions"):
        values = stats[key].detach().float().tolist()
        accumulator_values = accumulator[key]
        if not isinstance(accumulator_values, list):
            raise ValueError(f"position decile accumulator field {key!r} is not a list")
        for idx, value in enumerate(values):
            accumulator_values[idx] += float(value) * int(world_size)
    accumulator["loss_sum"] = float(accumulator["loss_sum"]) + float(stats["loss"].detach().item()) * int(world_size)
    accumulator["microbatches"] = int(accumulator["microbatches"]) + int(world_size)


def _position_decile_ce_accumulator_metrics(
    prefix: str,
    accumulator: dict[str, list[float] | float | int],
    enabled: bool,
) -> dict[str, float | int | bool]:
    active_counts = (
        [float(v) for v in accumulator["active_counts"]] if isinstance(accumulator["active_counts"], list) else []
    )
    lossable_counts = (
        [float(v) for v in accumulator["lossable_counts"]] if isinstance(accumulator["lossable_counts"], list) else []
    )
    raw_ce_sums = [float(v) for v in accumulator["raw_ce_sums"]] if isinstance(accumulator["raw_ce_sums"], list) else []
    weighted_ce_sums = (
        [float(v) for v in accumulator["weighted_ce_sums"]] if isinstance(accumulator["weighted_ce_sums"], list) else []
    )
    balanced_contributions = (
        [float(v) for v in accumulator["balanced_contributions"]]
        if isinstance(accumulator["balanced_contributions"], list)
        else []
    )
    total_active_count = sum(active_counts)
    total_lossable_count = sum(lossable_counts)
    microbatches = int(accumulator["microbatches"])
    metrics: dict[str, float | int | bool] = {
        f"{prefix}position_decile_balanced_loss_enabled": bool(enabled),
        f"{prefix}position_decile_balanced_loss_deciles": len(active_counts),
        f"{prefix}position_decile_balanced_microbatches": microbatches,
        f"{prefix}position_decile_balanced_loss": float(accumulator["loss_sum"]) / max(1, microbatches),
        f"{prefix}position_decile_active_tokens": int(total_active_count),
        f"{prefix}position_decile_lossable_tokens": int(total_lossable_count),
    }
    for decile_idx, active_count in enumerate(active_counts):
        label = f"{decile_idx:02d}"
        lossable_count = lossable_counts[decile_idx]
        raw_mean = raw_ce_sums[decile_idx] / active_count if active_count else 0.0
        weighted_mean = weighted_ce_sums[decile_idx] / active_count if active_count else 0.0
        weighted_per_lossable = weighted_ce_sums[decile_idx] / lossable_count if lossable_count else 0.0
        contribution = balanced_contributions[decile_idx] / max(1, microbatches)
        metrics[f"{prefix}position_decile_{label}_active_tokens"] = int(active_count)
        metrics[f"{prefix}position_decile_{label}_lossable_tokens"] = int(lossable_count)
        metrics[f"{prefix}position_decile_{label}_active_frac"] = active_count / max(1.0, total_active_count)
        metrics[f"{prefix}position_decile_{label}_lossable_frac"] = lossable_count / max(1.0, total_lossable_count)
        metrics[f"{prefix}position_decile_{label}_ce_before_weighting"] = raw_mean
        metrics[f"{prefix}position_decile_{label}_ce_after_schedule_weighting"] = weighted_mean
        metrics[f"{prefix}position_decile_{label}_schedule_weighted_ce_per_lossable_token"] = weighted_per_lossable
        metrics[f"{prefix}position_decile_{label}_balanced_loss_contribution"] = contribution
    return metrics


def _eos_hazard_calibration_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None,
    num_deciles: int = EOS_HAZARD_RELATIVE_POSITION_DECILES,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = logits.device
    empty = {
        "pred_sums": torch.zeros(num_deciles, device=device, dtype=torch.float32),
        "target_sums": torch.zeros(num_deciles, device=device, dtype=torch.float32),
        "counts": torch.zeros(num_deciles, device=device, dtype=torch.float32),
    }
    if eos_token_id is None or logits.numel() == 0 or target_indices.numel() == 0:
        return torch.tensor(0.0, device=device), empty
    eos_id = int(eos_token_id)
    if eos_id < 0 or eos_id >= logits.shape[-1]:
        return torch.tensor(0.0, device=device), empty

    eos_probs = F.softmax(logits.float(), dim=-1)[:, eos_id]
    target_eos = (targets.to(torch.long) == eos_id).to(torch.float32)
    decile_ids = _relative_position_decile_ids(target_indices, attention_mask, int(num_deciles))
    counts = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    pred_sums = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    target_sums = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    ones = torch.ones_like(eos_probs, dtype=torch.float32)
    counts.scatter_add_(0, decile_ids, ones)
    pred_sums.scatter_add_(0, decile_ids, eos_probs)
    target_sums.scatter_add_(0, decile_ids, target_eos)
    nonempty = counts > 0
    pred_means = pred_sums / counts.clamp_min(1.0)
    target_means = target_sums / counts.clamp_min(1.0)
    loss = (pred_means[nonempty] - target_means[nonempty]).pow(2).mean() if nonempty.any() else eos_probs.sum() * 0.0
    return loss, {"pred_sums": pred_sums, "target_sums": target_sums, "counts": counts}


def _new_eos_hazard_accumulator(num_deciles: int = EOS_HAZARD_RELATIVE_POSITION_DECILES) -> dict[str, list[float]]:
    return {
        "pred_sums": [0.0 for _ in range(int(num_deciles))],
        "target_sums": [0.0 for _ in range(int(num_deciles))],
        "counts": [0.0 for _ in range(int(num_deciles))],
    }


def _update_eos_hazard_accumulator(
    accumulator: dict[str, list[float]],
    stats: dict[str, torch.Tensor],
    world_size: int = 1,
) -> None:
    for key in ("pred_sums", "target_sums", "counts"):
        values = stats.get(key)
        if values is None:
            continue
        detached = values.detach().float().cpu().tolist()
        for idx, value in enumerate(detached):
            accumulator[key][idx] += float(value) * int(world_size)


def _eos_hazard_accumulator_metrics(
    prefix: str,
    accumulator: dict[str, list[float]],
    calibration_loss: float,
    weighted_loss: float,
    weight: float,
) -> dict[str, float | int]:
    pred_sums = accumulator["pred_sums"]
    target_sums = accumulator["target_sums"]
    counts = accumulator["counts"]
    total_count = float(sum(counts))
    total_pred = float(sum(pred_sums))
    total_target = float(sum(target_sums))
    pred_mean = total_pred / total_count if total_count else 0.0
    target_mean = total_target / total_count if total_count else 0.0
    metrics: dict[str, float | int] = {
        f"{prefix}eos_hazard_loss_weight": float(weight),
        f"{prefix}eos_hazard_calibration_loss": float(calibration_loss),
        f"{prefix}eos_hazard_loss": float(weighted_loss),
        f"{prefix}eos_hazard_token_count": int(total_count),
        f"{prefix}eos_hazard_pred_prob": pred_mean,
        f"{prefix}eos_hazard_empirical_freq": target_mean,
        f"{prefix}eos_hazard_abs_error": abs(pred_mean - target_mean),
    }
    for decile_idx, count in enumerate(counts):
        pred = pred_sums[decile_idx] / count if count else 0.0
        target = target_sums[decile_idx] / count if count else 0.0
        label = f"{decile_idx:02d}"
        metrics[f"{prefix}eos_hazard_decile_{label}_count"] = int(count)
        metrics[f"{prefix}eos_hazard_decile_{label}_pred_prob"] = pred
        metrics[f"{prefix}eos_hazard_decile_{label}_empirical_freq"] = target
        metrics[f"{prefix}eos_hazard_decile_{label}_abs_error"] = abs(pred - target)
    return metrics


def _length_head_targets(
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x_0.shape != attention_mask.shape:
        raise ValueError(f"x_0 and attention_mask shapes must match, got {x_0.shape} and {attention_mask.shape}")
    batch_size, seq_len = x_0.shape
    valid_rows = attention_mask.any(dim=1)
    valid_lengths = attention_mask.to(torch.long).sum(dim=1).clamp_min(1)
    if eos_token_id is None:
        return (valid_lengths - 1).to(torch.long), valid_rows

    positions = torch.arange(seq_len, device=x_0.device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
    eos_hits = (x_0 == int(eos_token_id)) & attention_mask
    first_eos = torch.where(eos_hits, positions, torch.full_like(positions, seq_len)).min(dim=1).values
    targets = torch.where(first_eos < seq_len, first_eos, valid_lengths - 1)
    return targets.to(torch.long), valid_rows


def _length_head_loss_and_stats(
    length_logits: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets, valid_rows = _length_head_targets(x_0, attention_mask, eos_token_id)
    device = length_logits.device
    seq_len = int(x_0.shape[1])
    if length_logits.shape != (x_0.shape[0], seq_len):
        raise ValueError(f"length_logits must have shape {(x_0.shape[0], seq_len)}, got {tuple(length_logits.shape)}")
    empty_stats = {
        "count": torch.tensor(0.0, device=device),
        "ce": torch.tensor(0.0, device=device),
        "accuracy": torch.tensor(0.0, device=device),
        "mae": torch.tensor(0.0, device=device),
        "confidence": torch.tensor(0.0, device=device),
        "calibration_abs_error": torch.tensor(0.0, device=device),
        "target_counts": torch.zeros(seq_len, device=device, dtype=torch.float32),
        "pred_counts": torch.zeros(seq_len, device=device, dtype=torch.float32),
    }
    if not bool(valid_rows.any().item()):
        return length_logits.float().sum() * 0.0, empty_stats

    logits = length_logits[valid_rows].float()
    target = targets[valid_rows]
    ce = F.cross_entropy(logits, target, reduction="mean")
    pred = logits.argmax(dim=-1)
    correct = (pred == target).float()
    mae = (pred - target).abs().float().mean()
    probs = F.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values.mean()
    accuracy = correct.mean()
    target_counts = torch.bincount(target, minlength=seq_len).to(device=device, dtype=torch.float32)
    pred_counts = torch.bincount(pred, minlength=seq_len).to(device=device, dtype=torch.float32)
    return ce, {
        "count": torch.tensor(float(target.numel()), device=device),
        "ce": ce.detach(),
        "accuracy": accuracy.detach(),
        "mae": mae.detach(),
        "confidence": confidence.detach(),
        "calibration_abs_error": (confidence - accuracy).abs().detach(),
        "target_counts": target_counts.detach(),
        "pred_counts": pred_counts.detach(),
    }


def _new_length_head_accumulator(seq_len: int) -> dict[str, object]:
    return {
        "microbatches": 0,
        "count": 0.0,
        "ce_sum": 0.0,
        "accuracy_sum": 0.0,
        "mae_sum": 0.0,
        "confidence_sum": 0.0,
        "calibration_abs_error_sum": 0.0,
        "target_counts": [0.0 for _ in range(int(seq_len))],
        "pred_counts": [0.0 for _ in range(int(seq_len))],
    }


def _update_length_head_accumulator(
    accumulator: dict[str, object],
    stats: dict[str, torch.Tensor],
    world_size: int = 1,
) -> None:
    count = float(stats["count"].detach().item()) * int(world_size)
    if count <= 0:
        return
    accumulator["microbatches"] = int(accumulator["microbatches"]) + int(world_size)
    accumulator["count"] = float(accumulator["count"]) + count
    for key, stat_key in (
        ("ce_sum", "ce"),
        ("accuracy_sum", "accuracy"),
        ("mae_sum", "mae"),
        ("confidence_sum", "confidence"),
        ("calibration_abs_error_sum", "calibration_abs_error"),
    ):
        accumulator[key] = float(accumulator[key]) + float(stats[stat_key].detach().item()) * int(world_size)
    for key in ("target_counts", "pred_counts"):
        values = accumulator[key]
        if not isinstance(values, list):
            raise ValueError(f"length-head accumulator field {key!r} is not a list")
        for idx, value in enumerate(stats[key].detach().float().cpu().tolist()):
            values[idx] += float(value) * int(world_size)


def _constant_prior_from_counts(counts: list[float], mode: str) -> int:
    if not counts:
        return 0
    if mode == "mode":
        return max(range(len(counts)), key=lambda idx: (counts[idx], -idx))
    if mode == "median":
        total = float(sum(counts))
        if total <= 0:
            return 0
        running = 0.0
        midpoint = 0.5 * total
        for idx, count in enumerate(counts):
            running += float(count)
            if running >= midpoint:
                return idx
        return len(counts) - 1
    raise ValueError(f"unknown prior mode: {mode!r}")


def _length_head_accumulator_metrics(
    prefix: str,
    accumulator: dict[str, object],
    weight: float,
    prior_counts: list[float] | None = None,
) -> dict[str, float | int | bool]:
    microbatches = int(accumulator["microbatches"])
    count = float(accumulator["count"])
    target_counts = (
        [float(v) for v in accumulator["target_counts"]] if isinstance(accumulator["target_counts"], list) else []
    )
    pred_counts = [float(v) for v in accumulator["pred_counts"]] if isinstance(accumulator["pred_counts"], list) else []
    total_targets = float(sum(target_counts))
    metrics: dict[str, float | int | bool] = {
        f"{prefix}length_head_enabled": bool(weight > 0),
        f"{prefix}length_head_loss_weight": float(weight),
        f"{prefix}length_head_microbatches": microbatches,
        f"{prefix}length_head_count": int(count),
        f"{prefix}length_head_ce": float(accumulator["ce_sum"]) / max(1, microbatches),
        f"{prefix}length_head_loss": float(weight) * float(accumulator["ce_sum"]) / max(1, microbatches),
        f"{prefix}length_head_accuracy": float(accumulator["accuracy_sum"]) / max(1, microbatches),
        f"{prefix}length_head_mae": float(accumulator["mae_sum"]) / max(1, microbatches),
        f"{prefix}length_head_confidence": float(accumulator["confidence_sum"]) / max(1, microbatches),
        f"{prefix}length_head_calibration_abs_error": float(accumulator["calibration_abs_error_sum"])
        / max(1, microbatches),
    }
    if target_counts:
        target_mean = sum(idx * count_value for idx, count_value in enumerate(target_counts)) / max(1.0, total_targets)
        pred_total = float(sum(pred_counts))
        pred_mean = sum(idx * count_value for idx, count_value in enumerate(pred_counts)) / max(1.0, pred_total)
        metrics[f"{prefix}length_head_target_position_mean"] = target_mean
        metrics[f"{prefix}length_head_predicted_position_mean"] = pred_mean
        metrics[f"{prefix}length_head_target_length_mean"] = target_mean + 1.0
        metrics[f"{prefix}length_head_predicted_length_mean"] = pred_mean + 1.0
        metrics[f"{prefix}length_head_target_hist_nonzero_bins"] = sum(1 for value in target_counts if value > 0)
        metrics[f"{prefix}length_head_pred_hist_nonzero_bins"] = sum(1 for value in pred_counts if value > 0)
        for idx, (target_count, pred_count) in enumerate(zip(target_counts, pred_counts, strict=True)):
            if target_count or pred_count:
                metrics[f"{prefix}length_head_target_position_{idx:04d}_count"] = int(target_count)
                metrics[f"{prefix}length_head_predicted_position_{idx:04d}_count"] = int(pred_count)
    if prior_counts is not None:
        prior_counts_f = [float(v) for v in prior_counts]
        prior_mode = _constant_prior_from_counts(prior_counts_f, "mode")
        prior_median = _constant_prior_from_counts(prior_counts_f, "median")
        total = max(1.0, total_targets)
        target_mode_hits = target_counts[prior_mode] if prior_mode < len(target_counts) else 0.0
        prior_mae = sum(abs(idx - prior_median) * count_value for idx, count_value in enumerate(target_counts)) / total
        metrics[f"{prefix}length_head_prior_mode_position"] = int(prior_mode)
        metrics[f"{prefix}length_head_prior_median_position"] = int(prior_median)
        metrics[f"{prefix}length_head_prior_accuracy"] = target_mode_hits / total
        metrics[f"{prefix}length_head_prior_mae"] = prior_mae
        metrics[f"{prefix}length_head_beats_prior_accuracy"] = (
            metrics[f"{prefix}length_head_accuracy"] > metrics[f"{prefix}length_head_prior_accuracy"]
        )
        metrics[f"{prefix}length_head_beats_prior_mae"] = (
            metrics[f"{prefix}length_head_mae"] < metrics[f"{prefix}length_head_prior_mae"]
        )
    return metrics


@torch.no_grad()
def _evaluate_length_head_metrics(
    raw_model: nn.Module,
    config: ExperimentConfig,
    tokenizer,
    device,
    autocast_ctx,
    train_prior_counts: list[float] | None = None,
) -> dict[str, float | int | bool]:
    if not _uses_length_head(config):
        return {}
    val_loader = build_val_loader(config, tokenizer, device)
    accumulator = _new_length_head_accumulator(int(config.model.max_seq_len))
    eos_token_id = tokenizer.get_eos_token_id()
    batches = int(config.eval.fixed_ce_batches)
    for _ in range(batches):
        batch = next(val_loader)
        z_eval, _w_eval, _t_val, noise_level = _corrupt_eval_batch(
            config,
            build_noise(config.noise, int(tokenizer.get_mask_token_id()), int(tokenizer.get_vocab_size())),
            batch.x_0,
            batch.loss_mask,
            t_val=0.9,
        )
        with autocast_ctx:
            _logits, length_logits = raw_model(
                z_eval,
                attention_mask=batch.attention_mask,
                noise_level=noise_level if config.noise.time_conditioning else None,
                return_length_logits=True,
            )
        _loss, stats = _length_head_loss_and_stats(length_logits, batch.x_0, batch.attention_mask, eos_token_id)
        _update_length_head_accumulator(accumulator, stats)
    return _length_head_accumulator_metrics(
        "val_",
        accumulator,
        float(config.train.length_head_loss_weight),
        prior_counts=train_prior_counts,
    )


def _summarize_metric_values(metrics: dict[str, float], prefix: str, values: list[float]) -> None:
    if not values:
        return
    metrics[f"{prefix}mean"] = float(sum(values) / len(values))
    metrics[f"{prefix}min"] = float(min(values))
    metrics[f"{prefix}max"] = float(max(values))


@torch.no_grad()
def _residual_branch_rms_metrics(
    raw_model: DiffusionTransformer,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None = None,
) -> dict[str, float]:
    b, t = tokens.shape
    if position_ids is None:
        cos = raw_model.rope_cos[:t]
        sin = raw_model.rope_sin[:t]
    else:
        position_ids = position_ids.to(device=tokens.device, dtype=torch.long)
        if position_ids.dim() == 1:
            flat_positions = position_ids
            shape = (t, 1, -1)
        elif position_ids.dim() == 2:
            flat_positions = position_ids.reshape(-1)
            shape = (b, t, 1, -1)
        else:
            raise ValueError("position_ids must be a 1D or 2D tensor")
        cos = raw_model.rope_cos.index_select(0, flat_positions).view(shape)
        sin = raw_model.rope_sin.index_select(0, flat_positions).view(shape)

    x = raw_model.wte(tokens).to(dtype=raw_model.compute_dtype)
    if raw_model.noise_mlp is not None:
        noise_level = (tokens == int(raw_model.config.mask_token_id)).float().mean(dim=1)
        noise_level = noise_level.to(device=tokens.device, dtype=raw_model.compute_dtype)
        x = x + raw_model.noise_mlp(noise_level[:, None])[:, None, :]

    metrics: dict[str, float] = {}
    attn_values: list[float] = []
    mlp_values: list[float] = []
    activation_values: list[float] = []
    metrics["activation_embedding_rms"] = _tensor_rms(x)
    for layer_idx, block in enumerate(raw_model.blocks):
        attn_out = block.attn(rms_norm(x), cos, sin, attention_mask=attention_mask)
        attn_rms = _tensor_rms(attn_out)
        x = x + attn_out
        mlp_out = block.mlp(rms_norm(x))
        mlp_rms = _tensor_rms(mlp_out)
        x = x + mlp_out
        activation_rms = _tensor_rms(x)
        metrics[f"residual_branch_layer_{layer_idx}_attn_rms"] = attn_rms
        metrics[f"residual_branch_layer_{layer_idx}_mlp_rms"] = mlp_rms
        metrics[f"activation_layer_{layer_idx}_rms"] = activation_rms
        attn_values.append(attn_rms)
        mlp_values.append(mlp_rms)
        activation_values.append(activation_rms)
    _summarize_metric_values(metrics, "residual_branch_attn_rms_", attn_values)
    _summarize_metric_values(metrics, "residual_branch_mlp_rms_", mlp_values)
    _summarize_metric_values(metrics, "activation_rms_", activation_values)
    return metrics


def _layerwise_gradient_norm_metrics(raw_model: DiffusionTransformer) -> dict[str, float]:
    metrics: dict[str, float] = {}
    layer_values: list[float] = []
    attn_proj_values: list[float] = []
    mlp_down_values: list[float] = []
    for layer_idx, block in enumerate(raw_model.blocks):
        layer_norm = _grad_norm(block.parameters())
        attn_proj_norm = _grad_norm(block.attn.c_proj.parameters())
        mlp_down_norm = _grad_norm(block.mlp.down.parameters())
        metrics[f"layerwise_grad_layer_{layer_idx}_norm"] = layer_norm
        metrics[f"layerwise_grad_layer_{layer_idx}_attn_c_proj_norm"] = attn_proj_norm
        metrics[f"layerwise_grad_layer_{layer_idx}_mlp_down_norm"] = mlp_down_norm
        layer_values.append(layer_norm)
        attn_proj_values.append(attn_proj_norm)
        mlp_down_values.append(mlp_down_norm)
    _summarize_metric_values(metrics, "layerwise_grad_norm_", layer_values)
    _summarize_metric_values(metrics, "layerwise_grad_attn_c_proj_norm_", attn_proj_values)
    _summarize_metric_values(metrics, "layerwise_grad_mlp_down_norm_", mlp_down_values)
    return metrics


def _lm_head_row_norm_metrics(
    raw_model: nn.Module,
    tokenizer,
    data_config,
    device,
    batch_size: int,
    seq_len: int,
    batches: int = 10,
) -> dict[str, float]:
    vocab_size = tokenizer.get_vocab_size()
    token_bytes = build_token_bytes(tokenizer).to(device)
    special_ids = {int(i) for i in tokenizer.get_special_token_ids()}
    counts = torch.zeros(vocab_size, dtype=torch.long, device=device)
    loader = make_dataloader(data_config, tokenizer, int(batch_size), int(seq_len), "train", device=device)
    for _ in range(batches):
        batch = next(loader)
        content = batch.attention_mask & batch.loss_mask & (token_bytes[batch.x_0] > 0)
        for sid in special_ids:
            content &= batch.x_0 != sid
        if content.any():
            ids = batch.x_0[content]
            counts += torch.bincount(ids, minlength=vocab_size)
    lm_head_weight = raw_model.lm_head.weight[:vocab_size].float()
    row_norms = lm_head_weight.norm(dim=1)
    metrics: dict[str, float] = {}
    eos_id = tokenizer.get_eos_token_id()
    if eos_id is not None and 0 <= int(eos_id) < vocab_size:
        metrics["lm_head_row_norm_eos"] = float(row_norms[int(eos_id)].item())
    non_special = [tid for tid in range(vocab_size) if tid not in special_ids]
    non_special.sort(key=lambda tid: (int(counts[tid].item()), tid))
    num_deciles = 7
    decile_size = len(non_special) // num_deciles
    remainder = len(non_special) % num_deciles
    offset = 0
    for decile in range(1, num_deciles + 1):
        size = decile_size + (1 if decile <= remainder else 0)
        tokens = non_special[offset : offset + size]
        if tokens:
            norms = row_norms[tokens]
            metrics[f"lm_head_row_norm_decile_{decile}_mean"] = float(norms.mean().item())
            metrics[f"lm_head_row_norm_decile_{decile}_std"] = float(norms.std().item())
            metrics[f"lm_head_row_norm_decile_{decile}_min"] = float(norms.min().item())
            metrics[f"lm_head_row_norm_decile_{decile}_max"] = float(norms.max().item())
        offset += size
    return metrics


def _training_budget(
    config: ExperimentConfig,
    counts: dict,
    world_size: int,
    *,
    raw_model: DiffusionTransformer | None = None,
):
    train = config.train
    assert train.device_batch_size is not None
    target_param_count = _selected_target_param_count(config, counts)
    param_data_tokens = int(train.target_param_data_ratio * target_param_count)
    if train.total_batch_size < 0:
        reference_count = _reference_target_param_count(config, raw_model)
        ref_tokens = train.target_param_data_ratio * reference_count
        predicted_batch = train.base_total_batch_size * (param_data_tokens / ref_tokens) ** 0.383
        total_batch_size = 2 ** round(math.log2(predicted_batch))
    else:
        total_batch_size = train.total_batch_size

    tokens_per_micro = train.device_batch_size * config.model.max_seq_len
    world_tokens_per_micro = tokens_per_micro * world_size
    if total_batch_size % world_tokens_per_micro != 0:
        if not train.allow_batch_size_round_up:
            raise ValueError(
                f"total_batch_size={total_batch_size} must be divisible by "
                f"world_size * device_batch_size * max_seq_len = {world_tokens_per_micro}"
            )
        grad_accum_steps = max(1, total_batch_size // world_tokens_per_micro)
        total_batch_size = world_tokens_per_micro * grad_accum_steps
    else:
        grad_accum_steps = total_batch_size // world_tokens_per_micro
    if train.num_steps > 0:
        total_steps = train.num_steps
    elif train.target_tokens > 0:
        total_steps = max(1, round(train.target_tokens / total_batch_size))
    else:
        total_steps = max(1, param_data_tokens // total_batch_size)
    batch_lr_scale = (total_batch_size / train.base_total_batch_size) ** 0.5 if train.scale_lr_by_batch else 1.0
    return total_batch_size, grad_accum_steps, total_steps, batch_lr_scale


def _reference_target_param_count(
    config: ExperimentConfig,
    raw_model: DiffusionTransformer | None,
) -> int:
    """Read the batch-scaling reference count from a meta-device model."""
    if config.train.target_param_count_basis == "transformer_body_parameters":
        return estimate_non_embedding_params(n_layer=12, n_embd=768)
    if raw_model is None:
        raise ValueError(
            "automatic total_batch_size with a non-legacy target_param_count_basis requires "
            "the instantiated training model"
        )

    reference_config = replace(
        raw_model.config,
        n_layer=12,
        n_head=12,
        n_embd=768,
        mlp_hidden_dim=-1,
        attention_scale=-1.0,
    )
    token_class_ids = None
    if reference_config.output_factorization == "hierarchical_frequency_decile":
        output_class_count = int(raw_model.output_class_count)
        token_class_ids = torch.arange(reference_config.vocab_size, dtype=torch.long) % output_class_count
    with torch.device("meta"):
        reference_model = DiffusionTransformer(
            reference_config,
            rope_base=float(config.model.rope_base),
            compute_dtype=torch.float32,
            pad_vocab_to=int(config.model.pad_vocab_to),
            token_class_ids=token_class_ids,
        )
    return _selected_target_param_count(config, reference_model.parameter_counts())


def _selected_target_param_count(config: ExperimentConfig, counts: dict[str, int]) -> int:
    basis = config.train.target_param_count_basis
    if basis in counts:
        return int(counts[basis])
    compatibility_aliases = {
        "transformer_body_parameters": "non_embedding",
        "nanochat_scaling_parameters": "scaling",
        "total_parameters": "total",
    }
    alias = compatibility_aliases.get(basis)
    if alias is not None and alias in counts:
        return int(counts[alias])
    raise KeyError(f"parameter count view {basis!r} is missing from the instantiated-model inventory")


def _parameter_accounting_summary(
    config: ExperimentConfig,
    counts: dict[str, int],
    inventory: dict[str, object],
) -> dict[str, object]:
    return {
        "method": inventory["method"],
        "target_param_data_ratio": float(config.train.target_param_data_ratio),
        "target_param_count_basis": config.train.target_param_count_basis,
        "target_param_count": _selected_target_param_count(config, counts),
        "components": inventory["components"],
        "views": inventory["views"],
    }


def _token_ratio_metrics(train_tokens: int, counts: dict) -> dict[str, float]:
    """Both token/param ratio conventions, named explicitly to prevent confusion.

    - `tokens_per_nonembedding_param`: tokens per historical transformer-body
      parameter (the legacy repo/batch3 convention).
    - `tokens_per_scaling_param`: tokens per scaling param (transformer + lm_head),
      the nanochat/Karpathy convention used for GPT-2-scale budget comparisons.

    The two differ by the lm_head fraction (large for the 128k vocab at small models)
    and are NOT interchangeable: e.g. at m_deep a non-embedding ratio of 256 is a
    scaling ratio of ~67.6. Both remain logged for historical comparison; the
    active budget convention is recorded separately as `target_param_count_basis`.
    """
    tokens = max(0, int(train_tokens))
    non_emb = max(1, int(counts.get("non_embedding", 0)))
    scaling = max(1, int(counts.get("scaling", 0)))
    return {
        "tokens_per_nonembedding_param": tokens / non_emb,
        "tokens_per_scaling_param": tokens / scaling,
    }


def _sample_training_scalar_noise_time(config: ExperimentConfig) -> float:
    if config.noise.type == "mdlm" and bool(config.noise.schedule_weight):
        return sample_positive_uniform_scalar(config.noise.t_min, config.noise.t_max)
    return random.uniform(config.noise.t_min, config.noise.t_max)


def _sample_training_tensor_noise_time(config: ExperimentConfig, shape: tuple[int, ...], device) -> torch.Tensor:
    if config.noise.type == "mdlm" and bool(config.noise.schedule_weight):
        return sample_positive_uniform_tensor(
            shape,
            device=device,
            t_min=float(config.noise.t_min),
            t_max=float(config.noise.t_max),
        )
    return torch.empty(shape, device=device, dtype=torch.float32).uniform_(
        float(config.noise.t_min),
        float(config.noise.t_max),
    )


def _corrupt_training_batch(config: ExperimentConfig, noise, x_0: torch.Tensor, loss_mask: torch.Tensor):
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
    elif _uses_strict_mdlm_per_row_t(config):
        t_val = _sample_training_tensor_noise_time(config, (x_0.shape[0],), x_0.device)
        z_t, w = noise.corrupt(x_0, t_val, maskable=loss_mask)
    else:
        t_val = _sample_training_scalar_noise_time(config)
        z_t, w = noise.corrupt(x_0, t_val, maskable=loss_mask)
    return z_t, w, t_val, None


def _sample_non_mask_non_original_tokens(
    originals: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
) -> torch.Tensor:
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
        raw = torch.randint(0, vocab_size - 1, originals.shape, device=originals.device, dtype=torch.long)
        sampled = raw + (raw >= originals).long()
    if bool((sampled == originals).any().item()):
        raise RuntimeError("token-edit corruption sampled an unchanged token")
    if 0 <= mask_token_id < vocab_size and bool((sampled == mask_token_id).any().item()):
        raise RuntimeError("token-edit corruption sampled the mask token")
    return sampled


def _corrupt_token_edit_training_batch(
    config: ExperimentConfig,
    x_0: torch.Tensor,
    loss_mask: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, float, None]:
    ratio = float(config.train.token_edit_corruption_ratio)
    edit_mask = (torch.rand(x_0.shape, device=x_0.device) < ratio) & loss_mask
    if not edit_mask.any() and loss_mask.any():
        lossable_indices = loss_mask.reshape(-1).nonzero(as_tuple=True)[0]
        selected = lossable_indices[torch.randint(0, lossable_indices.numel(), (1,), device=x_0.device)]
        edit_mask = edit_mask.reshape(-1)
        edit_mask[selected] = True
        edit_mask = edit_mask.view_as(x_0)
    z_t = x_0.clone()
    if edit_mask.any():
        originals = x_0[edit_mask]
        z_t[edit_mask] = _sample_non_mask_non_original_tokens(originals, vocab_size, mask_token_id)
    return z_t, edit_mask.float(), ratio, None


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


def _run_eval_causal_ar(
    raw_model: nn.Module,
    config: ExperimentConfig,
    tokenizer,
    autocast_ctx,
    device,
    step: int,
    include_gen: bool,
) -> dict:
    token_bytes = build_token_bytes(tokenizer).to(device)
    val_loader = build_val_loader(config, tokenizer, device)
    bos_id = tokenizer.get_bos_token_id()
    eos_id = tokenizer.get_eos_token_id()
    total_ce = 0.0
    total_content_ce = 0.0
    total_tokens = 0
    total_content_tokens = 0
    total_content_bytes = 0
    eos_entropy_sum = 0.0
    eos_top1_hits = 0
    eos_entropy_count = 0
    bucket_sums = {
        "first_generated_token": 0.0,
        "post_bos": 0.0,
        "interior": 0.0,
        "pre_eos": 0.0,
        "post_eos": 0.0,
    }
    bucket_counts = {key: 0 for key in bucket_sums}

    def add_bucket(name: str, ce_values: torch.Tensor, mask: torch.Tensor) -> None:
        if not mask.any():
            return
        bucket_sums[name] += float(ce_values[mask].sum().item())
        bucket_counts[name] += int(mask.sum().item())

    steps = max(1, config.eval.nelbo_tokens // (int(config.train.device_batch_size) * config.model.max_seq_len))
    with torch.no_grad():
        for _ in range(steps):
            batch = next(val_loader)
            x_eval = batch.x_0
            with autocast_ctx:
                model_kwargs = {}
                if getattr(batch, "position_ids", None) is not None:
                    model_kwargs["position_ids"] = batch.position_ids
                model_kwargs["attention_mask"] = _batch_model_attention_mask(batch)
                logits = raw_model(x_eval, **model_kwargs)
            logits_pred = logits[:, :-1, :].contiguous()
            targets = x_eval[:, 1:].contiguous()
            target_mask = batch.loss_mask[:, 1:] & batch.attention_mask[:, 1:]
            ce = F.cross_entropy(
                logits_pred.view(-1, logits_pred.size(-1)),
                targets.view(-1),
                reduction="none",
            ).view_as(targets)
            if target_mask.any():
                total_ce += float(ce[target_mask].sum().item())
                total_tokens += int(target_mask.sum().item())
                content_mask = target_mask & (token_bytes[targets] > 0)
                if content_mask.any():
                    total_content_ce += float(ce[content_mask].sum().item())
                    total_content_tokens += int(content_mask.sum().item())
                    total_content_bytes += int(token_bytes[targets][content_mask].sum().item())
                target_positions = torch.arange(1, x_eval.shape[1], device=x_eval.device).view(1, -1)
                prev_tokens = x_eval[:, :-1].contiguous()
                prev_bos = (
                    prev_tokens == int(bos_id)
                    if bos_id is not None
                    else torch.zeros_like(target_mask, dtype=torch.bool)
                )
                prev_eos = (
                    prev_tokens == int(eos_id)
                    if eos_id is not None
                    else torch.zeros_like(target_mask, dtype=torch.bool)
                )
                target_eos = (
                    targets == int(eos_id) if eos_id is not None else torch.zeros_like(target_mask, dtype=torch.bool)
                )
                eos_eval_mask = target_mask & target_eos
                if eos_eval_mask.any():
                    eos_logits = logits_pred[eos_eval_mask].float()
                    eos_probs = F.softmax(eos_logits, dim=-1)
                    safe_eos_probs = eos_probs.clamp_min(1e-10)
                    eos_entropy_sum += float((-(safe_eos_probs * safe_eos_probs.log()).sum(dim=-1)).sum().item())
                    eos_top1_hits += int((eos_probs.argmax(dim=-1) == int(eos_id)).sum().item())
                    eos_entropy_count += int(eos_eval_mask.sum().item())
                add_bucket("first_generated_token", ce, target_mask & (target_positions == 1))
                add_bucket("post_bos", ce, target_mask & prev_bos)
                add_bucket("interior", ce, content_mask & ~prev_bos & ~prev_eos & ~target_eos)
                add_bucket("pre_eos", ce, target_mask & target_eos)
                add_bucket("post_eos", ce, target_mask & prev_eos)
    metrics: dict[str, float | int | str] = {
        "step": step,
        "eval_ar_batches": steps,
        "eval_ar_tokens": total_tokens,
        "eval_ar_content_tokens": total_content_tokens,
        "eval_ar_content_bytes": total_content_bytes,
        "eval_ar_content_token_byte_fertility": total_content_tokens / total_content_bytes
        if total_content_bytes
        else 0.0,
        "val_ce": total_ce / total_tokens if total_tokens else 0.0,
        "val_content_ce": total_content_ce / total_content_tokens if total_content_tokens else 0.0,
        "ar_gold_context_eos_token_entropy_mean": eos_entropy_sum / eos_entropy_count if eos_entropy_count else 0.0,
        "ar_gold_context_eos_token_top1_hit": eos_top1_hits / eos_entropy_count if eos_entropy_count else 0.0,
        "ar_gold_context_eos_token_count": eos_entropy_count,
    }
    for name, ce_sum in bucket_sums.items():
        count = bucket_counts[name]
        metrics[f"val_ar_ce_{name}"] = ce_sum / count if count else 0.0
        metrics[f"val_ar_{name}_tokens"] = count
    ar_bpb = total_content_ce / (math.log(2.0) * total_content_bytes) if total_content_bytes else 0.0
    metrics.update(
        {
            "val_bpb": ar_bpb,
            "val_ar_bpb": ar_bpb,
            "val_content_bpb": ar_bpb,
            "val_bpb_metric": "causal_ar_content_ce_bits_per_byte",
        }
    )
    with autocast_ctx:
        metrics.update(evaluate_bos_only_first_token(raw_model, tokenizer, device))
    if config.eval.conditional_rank_eval:
        with autocast_ctx:
            metrics.update(
                evaluate_ar_next_token_conditional_rank_suffix(
                    raw_model,
                    tokenizer,
                    config.data,
                    device,
                    batches=config.eval.conditional_rank_batches,
                    batch_size=int(config.train.device_batch_size),
                    seq_len=config.model.max_seq_len,
                )
            )
        metrics["ar_gold_context_suffix_top1_hit"] = metrics.get(
            "ar_next_token_conditional_rank_suffix_gold_top1_hit",
            0.0,
        )
        metrics["ar_gold_context_suffix_entropy_mean"] = metrics.get(
            "ar_next_token_conditional_rank_suffix_entropy_mean",
            0.0,
        )
        metrics["ar_gold_context_rare_token_top1_hit"] = metrics.get(
            "ar_next_token_conditional_rank_decile_1_gold_top1_hit",
            0.0,
        )
        metrics["ar_gold_context_rare_token_entropy_mean"] = metrics.get(
            "ar_next_token_conditional_rank_decile_1_entropy_mean",
            0.0,
        )
        metrics["ar_gold_context_common_token_top1_hit"] = metrics.get(
            "ar_next_token_conditional_rank_decile_7_gold_top1_hit",
            0.0,
        )
        metrics["ar_gold_context_common_token_entropy_mean"] = metrics.get(
            "ar_next_token_conditional_rank_decile_7_entropy_mean",
            0.0,
        )
    if include_gen:

        def copy_prefixed(prefix: str, source: dict) -> None:
            for key, value in source.items():
                suffix = key[4:] if key.startswith("gen_") else key
                metrics[f"{prefix}{suffix}"] = value

        def build_sample_fn(token_selection: str):
            def sample_fn(m, n, length):
                return sample_ar(
                    m,
                    tokenizer,
                    n,
                    length,
                    temperature=config.eval.gen_temperature,
                    token_selection=token_selection,
                )

            return sample_fn

        primary_token_selection = config.eval.sample_token_selection
        secondary_token_selection = "multinomial" if primary_token_selection == "argmax" else "argmax"
        metrics["gen_primary_token_selection"] = primary_token_selection
        metrics["gen_secondary_token_selection"] = secondary_token_selection
        for token_selection in (primary_token_selection, secondary_token_selection):
            with autocast_ctx:
                gen_metrics = evaluate_generation_metrics(
                    raw_model,
                    tokenizer,
                    None,
                    build_sample_fn(token_selection),
                    int(config.train.device_batch_size),
                    config.train.run_dir,
                    device=device,
                    judge_model_id=config.eval.judge_model_id,
                    gen_samples=config.eval.gen_samples,
                    gen_len=config.eval.gen_len,
                    log_samples=config.eval.gen_log_samples,
                    score_mode=config.eval.gen_score_mode,
                    self_score_t=config.eval.gen_self_score_t,
                    step=step,
                    sample_label=f"free_run_{token_selection}",
                    model_logits_fn=None,
                )
            if token_selection == primary_token_selection:
                metrics.update(gen_metrics)
            copy_prefixed(f"ar_free_run_{token_selection}_", gen_metrics)

        prefix_len = max(1, min(16, int(config.eval.gen_len) - 1, int(config.model.max_seq_len) - 1))
        for token_selection in (primary_token_selection, secondary_token_selection):
            with autocast_ctx:
                prefix_metrics = evaluate_ar_validation_prefix_continuations(
                    raw_model,
                    tokenizer,
                    config.data,
                    int(config.train.device_batch_size),
                    device=device,
                    seq_len=config.model.max_seq_len,
                    gen_samples=config.eval.gen_samples,
                    gen_len=config.eval.gen_len,
                    prefix_len=prefix_len,
                    temperature=config.eval.gen_temperature,
                    token_selection=token_selection,
                    run_dir=config.train.run_dir,
                    log_samples=config.eval.gen_log_samples,
                    step=step,
                )
            for key, value in prefix_metrics.items():
                metrics[f"ar_val_prefix_{token_selection}_{key}"] = value
    return metrics


def _run_eval_unisolated(
    raw_model: nn.Module,
    config: ExperimentConfig,
    tokenizer,
    noise,
    autocast_ctx,
    device,
    step: int,
    include_gen: bool,
) -> dict:
    was_training = raw_model.training
    raw_model.eval()
    if _is_causal(config):
        metrics = _run_eval_causal_ar(
            raw_model,
            config,
            tokenizer,
            autocast_ctx,
            device,
            step,
            include_gen,
        )
        if was_training:
            raw_model.train()
        return metrics
    corruption_label = _training_corruption_label(config)
    use_hierarchical_output = hasattr(raw_model, "uses_hierarchical_output") and raw_model.uses_hierarchical_output()
    schedule_weighted_nelbo = _uses_schedule_weighted_nelbo(config)
    nelbo_eval_noise = (
        "shared_sequence_time_stratified_schedule_weighted_mdlm"
        if schedule_weighted_nelbo
        else ("standard_scalar_t_mdlm" if _uses_per_token_noise_levels(config) else corruption_label)
    )
    nelbo_t_min = float(config.eval.nelbo_t_min)
    nelbo_t_max = float(config.eval.nelbo_t_max)
    nelbo_integration_width = nelbo_t_max - nelbo_t_min
    nelbo_full_interval = bool(nelbo_t_min == 0.0 and nelbo_t_max == 1.0)
    eval_batch_size = int(config.train.device_batch_size)
    nelbo_steps = max(1, int(config.eval.nelbo_tokens) // (eval_batch_size * int(config.model.max_seq_len)))
    nelbo_sequence_count = nelbo_steps * eval_batch_size
    nelbo_times = (
        _stratified_per_sequence_times(
            nelbo_sequence_count,
            nelbo_t_min,
            nelbo_t_max,
            int(config.eval.nelbo_seed),
        )
        if schedule_weighted_nelbo
        else None
    )
    nelbo_corruption_plan: torch.Tensor | None = None
    nelbo_corruption_derived_seed: int | None = None
    if schedule_weighted_nelbo:
        nelbo_corruption_plan, nelbo_corruption_derived_seed = _sequence_major_nelbo_corruption_plan(
            nelbo_sequence_count,
            int(config.model.max_seq_len),
            int(config.eval.nelbo_seed),
        )
    nelbo_time_offset = 0
    metrics: dict[str, float | int | str | bool | None] = {
        "step": step,
        "eval_nelbo_tokens": config.eval.nelbo_tokens,
        "eval_fixed_ce_batches": config.eval.fixed_ce_batches,
        "eval_corruption": corruption_label,
        "val_nelbo_corruption": nelbo_eval_noise,
        "val_fixed_ce_corruption": corruption_label,
        "val_nelbo_eval_noise": nelbo_eval_noise,
        "val_nelbo_time_sampling": config.eval.nelbo_time_sampling
        if schedule_weighted_nelbo
        else "training_corruption_sampler",
        "val_nelbo_seed": int(config.eval.nelbo_seed) if schedule_weighted_nelbo else None,
        "val_nelbo_t_min": nelbo_t_min if schedule_weighted_nelbo else None,
        "val_nelbo_t_max": nelbo_t_max if schedule_weighted_nelbo else None,
        "val_nelbo_integration_width": nelbo_integration_width if schedule_weighted_nelbo else None,
        "val_nelbo_full_interval": bool(schedule_weighted_nelbo and nelbo_full_interval),
        "val_nelbo_sequence_count": nelbo_sequence_count if schedule_weighted_nelbo else 0,
        "val_nelbo_corruption_rng_scheme": NELBO_CORRUPTION_RNG_SCHEME if schedule_weighted_nelbo else None,
        "val_nelbo_corruption_rng_base_seed": int(config.eval.nelbo_seed) if schedule_weighted_nelbo else None,
        "val_nelbo_corruption_rng_derived_seed": nelbo_corruption_derived_seed,
        "val_nelbo_corruption_plan_shape": [nelbo_sequence_count, int(config.model.max_seq_len)]
        if schedule_weighted_nelbo
        else None,
        "val_fixed_ce_eval_noise": "standard_scalar_t_mdlm"
        if _uses_per_token_noise_levels(config)
        else corruption_label,
    }
    target_length_prior_counts: torch.Tensor | None = None
    if _uses_target_length_conditioning(config):
        saved_state = _rng_state(config)
        try:
            target_length_prior_counts, target_length_prior_metrics = _estimate_target_length_prior(
                config,
                tokenizer,
                int(config.train.device_batch_size),
                int(config.model.max_seq_len),
                device,
                int(config.eval.target_length_distribution_batches),
            )
        finally:
            _restore_rng_state(saved_state, config)
        metrics.update(target_length_prior_metrics)
        total_prior = float(target_length_prior_counts.float().sum().item())
        visible_prior = float(target_length_prior_counts[: int(config.eval.gen_len)].float().sum().item())
        metrics["target_length_prior_visible_count_at_gen_len"] = int(visible_prior)
        metrics["target_length_prior_visible_fraction_at_gen_len"] = visible_prior / total_prior if total_prior else 0.0

    def noise_fn(x_eval, loss_mask):
        nonlocal nelbo_time_offset
        if schedule_weighted_nelbo:
            assert nelbo_times is not None
            assert nelbo_corruption_plan is not None
            start_offset = nelbo_time_offset
            next_offset = nelbo_time_offset + int(x_eval.shape[0])
            if next_offset > int(nelbo_times.numel()):
                raise RuntimeError("NELBO evaluator consumed more sequences than its fixed time design")
            t_rows = nelbo_times[start_offset:next_offset].to(device=x_eval.device)
            uniform_rows_cpu = nelbo_corruption_plan[start_offset:next_offset]
            nelbo_time_offset = next_offset
            z_eval, w_eval = noise.corrupt(
                x_eval,
                t_rows,
                maskable=loss_mask,
                uniforms=uniform_rows_cpu.to(device=x_eval.device),
            )
            noise_level = t_rows if config.noise.time_conditioning else None
            return z_eval, t_rows, w_eval, noise_level, {"corruption_uniforms": uniform_rows_cpu}
        z_eval, w_eval, t_val, noise_level = _corrupt_eval_batch(config, noise, x_eval, loss_mask)
        return z_eval, t_val, w_eval, noise_level

    def model_logits_fn(
        m,
        z_eval,
        x_eval,
        attention_mask,
        noise_level=None,
        position_ids=None,
        target_lengths=None,
    ):
        if target_lengths is None:
            target_lengths = _target_lengths_for_conditioning(
                config,
                x_eval,
                attention_mask,
                tokenizer.get_eos_token_id(),
            )
        return _forward_eval_logits(
            m,
            config,
            z_eval,
            x_eval,
            attention_mask,
            noise_level=noise_level,
            position_ids=position_ids,
            target_lengths=target_lengths,
        )

    with autocast_ctx:
        nelbo_result = evaluate_nelbo(
            raw_model,
            config.data,
            tokenizer,
            noise_fn,
            int(config.train.device_batch_size),
            device=device,
            seq_len=config.model.max_seq_len,
            eval_tokens=config.eval.nelbo_tokens,
            model_logits_fn=model_logits_fn,
            schedule_weight=schedule_weighted_nelbo,
            schedule_weight_scale=nelbo_integration_width,
            return_diagnostics=True,
        )
        has_nelbo_diagnostics = len(nelbo_result) == 5
        if has_nelbo_diagnostics:
            if schedule_weighted_nelbo and nelbo_time_offset != nelbo_sequence_count:
                raise RuntimeError(
                    "NELBO evaluator did not consume its complete fixed corruption plan: "
                    f"{nelbo_time_offset} != {nelbo_sequence_count}"
                )
            nelbo, masked_ce_bpb, weight_mean, schedule_weighted, nelbo_diagnostics = nelbo_result
        else:  # Backward-compatible with small monkeypatched evaluator tests.
            nelbo, masked_ce_bpb, weight_mean, schedule_weighted = nelbo_result
            nelbo_diagnostics = {}
        if has_nelbo_diagnostics:
            sequence_records = _validated_nelbo_sequence_records(
                nelbo_diagnostics.pop("sequence_records", None),
                expected_count=nelbo_sequence_count,
                require_shared_time=schedule_weighted_nelbo,
            )
            diagnostic_sequence_count = nelbo_diagnostics.get("sequence_count")
            if diagnostic_sequence_count != len(sequence_records):
                raise ValueError(
                    "NELBO diagnostic sequence_count does not match the persisted sequence record count: "
                    f"{diagnostic_sequence_count!r} != {len(sequence_records)}"
                )
            total_record_bytes = math.fsum(float(record["byte_denominator"]) for record in sequence_records)
            reconstructed_nelbo = math.fsum(float(record["nelbo_nats_numerator"]) for record in sequence_records) / (
                math.log(2) * total_record_bytes
            )
            reconstructed_masked_ce = math.fsum(
                float(record["masked_nats_numerator"]) for record in sequence_records
            ) / (math.log(2) * total_record_bytes)
            if not math.isclose(reconstructed_nelbo, float(nelbo), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("NELBO sequence records do not reconstruct val_nelbo_bpb")
            if not math.isclose(reconstructed_masked_ce, float(masked_ce_bpb), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("NELBO sequence records do not reconstruct val_masked_ce_bpb")
            expected_totals = {
                "total_bytes": total_record_bytes,
                "masked_target_count": sum(int(record["masked_target_count"]) for record in sequence_records),
                "schedule_weight_count": sum(int(record["schedule_weight_count"]) for record in sequence_records),
            }
            for diagnostic_key, expected_value in expected_totals.items():
                if nelbo_diagnostics.get(diagnostic_key) != expected_value:
                    raise ValueError(
                        f"NELBO diagnostic {diagnostic_key} does not match sequence records: "
                        f"{nelbo_diagnostics.get(diagnostic_key)!r} != {expected_value!r}"
                    )
            expected_identity_roots = {
                "input_identity_root_sha256": _ordered_nelbo_digest_root(
                    b"nanodllm-nelbo-input-identity-root-v2", sequence_records, "input_identity_sha256"
                ),
                "corruption_mask_root_sha256": _ordered_nelbo_digest_root(
                    b"nanodllm-nelbo-corruption-mask-root-v2", sequence_records, "corruption_mask_sha256"
                ),
            }
            if schedule_weighted_nelbo:
                expected_identity_roots.update(
                    {
                        "corruption_uniform_root_sha256": _ordered_nelbo_digest_root(
                            b"nanodllm-nelbo-corruption-uniform-root-v1",
                            sequence_records,
                            "corruption_uniform_sha256",
                        ),
                        "evaluation_sample_root_sha256": _ordered_nelbo_digest_root(
                            b"nanodllm-nelbo-evaluation-sample-root-v2",
                            sequence_records,
                            "evaluation_sample_sha256",
                        ),
                    }
                )
            for diagnostic_key, expected_root in expected_identity_roots.items():
                if nelbo_diagnostics.get(diagnostic_key) != expected_root:
                    raise ValueError(
                        f"NELBO diagnostic {diagnostic_key} does not match sequence records: "
                        f"{nelbo_diagnostics.get(diagnostic_key)!r} != {expected_root!r}"
                    )
                metrics[f"val_nelbo_{diagnostic_key}"] = expected_root
            sequence_artifact_relpath = (
                f"telemetry/nelbo_sequences_step{step}.v{NELBO_SEQUENCE_RECORD_SCHEMA_VERSION}.jsonl"
            )
            sequence_artifact_path = os.path.join(config.train.run_dir, sequence_artifact_relpath)
            enriched_sequence_records = [
                {
                    **record,
                    "evaluation_step": int(step),
                    "nelbo_seed": metrics["val_nelbo_seed"],
                    "time_sampling": metrics["val_nelbo_time_sampling"],
                    "corruption_rng_scheme": metrics["val_nelbo_corruption_rng_scheme"],
                }
                for record in sequence_records
            ]
            metrics["val_nelbo_sequence_records_artifact"] = sequence_artifact_relpath
            metrics["val_nelbo_sequence_records_schema_version"] = NELBO_SEQUENCE_RECORD_SCHEMA_VERSION
            metrics["val_nelbo_sequence_records_count"] = len(enriched_sequence_records)
            metrics["val_nelbo_sequence_records_sha256"] = write_jsonl_atomic(
                sequence_artifact_path,
                enriched_sequence_records,
            )
        metrics["val_nelbo_bpb"] = nelbo
        metrics["val_masked_ce_bpb"] = masked_ce_bpb
        metrics["val_nelbo_is_schedule_weighted_bound"] = bool(schedule_weighted and nelbo_full_interval)
        metrics["val_nelbo_is_truncated_reference_integral"] = bool(schedule_weighted and not nelbo_full_interval)
        if use_hierarchical_output:
            metrics["hierarchical_val_nelbo_bpb"] = nelbo
        metrics["val_nelbo_schedule_weighted"] = schedule_weighted
        metrics["val_nelbo_weight_mean"] = weight_mean
        metrics["val_nelbo_effective_importance_weight_mean_masked_content_targets"] = weight_mean
        metrics["val_nelbo_weight_mean_semantics"] = (
            "integration_width_over_t_mean_on_masked_content_targets" if schedule_weighted else "not_applicable"
        )
        for source_key, metric_key in (
            ("nelbo_bpb_se", "val_nelbo_bpb_se"),
            ("masked_ce_bpb_se", "val_masked_ce_bpb_se"),
            ("se_method", "val_nelbo_se_method"),
            ("se_unit", "val_nelbo_jackknife_unit"),
            ("block_count", "val_nelbo_block_count"),
            ("sequence_count", "val_nelbo_jackknife_sequence_count"),
            ("total_bytes", "val_nelbo_total_bytes"),
            ("masked_target_count", "val_nelbo_masked_target_count"),
            ("schedule_weight_count", "val_nelbo_schedule_weight_count"),
        ):
            if source_key in nelbo_diagnostics:
                metrics[metric_key] = nelbo_diagnostics[source_key]

    token_bytes = build_token_bytes(tokenizer).to(device)
    eos_token_id = tokenizer.get_eos_token_id()
    val_loader = build_val_loader(config, tokenizer, device)
    with torch.no_grad():
        for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
            all_ce, content_ce = [], []
            class_ce, within_class_ce = [], []
            content_class_ce, content_within_class_ce = [], []
            class_entropy, class_max_prob = [], []
            class_prob_eos_special, class_prob_low_freq, class_prob_high_freq = [], [], []
            eos_hazard_losses = []
            eos_hazard_accumulator = _new_eos_hazard_accumulator()
            for _ in range(config.eval.fixed_ce_batches):
                batch = next(val_loader)
                eval_attention_mask = _batch_model_attention_mask(batch)
                x_eval = batch.x_0
                z_eval, w_eval, _, noise_level = _corrupt_eval_batch(
                    config,
                    noise,
                    x_eval,
                    batch.loss_mask,
                    t_val=t_val,
                )
                with autocast_ctx:
                    if use_hierarchical_output:
                        class_logits, token_logits = _forward_eval_hierarchical_components(
                            raw_model,
                            config,
                            z_eval,
                            x_eval,
                            eval_attention_mask,
                            noise_level=noise_level,
                            position_ids=getattr(batch, "position_ids", None),
                        )
                        ce, class_ce_values, within_class_ce_values = raw_model.hierarchical_ce(
                            class_logits,
                            token_logits,
                            x_eval,
                        )
                    else:
                        logits = _forward_eval_logits(
                            raw_model,
                            config,
                            z_eval,
                            x_eval,
                            eval_attention_mask,
                            noise_level=noise_level,
                            position_ids=getattr(batch, "position_ids", None),
                            target_lengths=_target_lengths_for_conditioning(
                                config,
                                x_eval,
                                batch.attention_mask,
                                eos_token_id,
                            ),
                        )
                        ce = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), x_eval.view(-1), reduction="none"
                        ).view_as(x_eval)
                        class_ce_values = None
                        within_class_ce_values = None
                active = (w_eval > 0) & batch.loss_mask
                content = active & (token_bytes[x_eval] > 0)
                all_ce.append(float(ce[active].mean().item()))
                content_ce.append(float(ce[content].mean().item()))
                if (
                    config.train.eos_hazard_loss_weight > 0
                    and not use_hierarchical_output
                    and eos_token_id is not None
                    and active.any()
                ):
                    active_indices = active.reshape(-1).nonzero(as_tuple=True)[0]
                    active_targets = x_eval.reshape(-1).index_select(0, active_indices)
                    active_logits = logits.reshape(-1, logits.size(-1)).index_select(0, active_indices)
                    eos_hazard_loss, eos_hazard_stats = _eos_hazard_calibration_loss(
                        active_logits,
                        active_targets,
                        active_indices,
                        batch.attention_mask,
                        eos_token_id,
                    )
                    eos_hazard_losses.append(float(eos_hazard_loss.detach().item()))
                    _update_eos_hazard_accumulator(eos_hazard_accumulator, eos_hazard_stats)
                if use_hierarchical_output and class_ce_values is not None and within_class_ce_values is not None:
                    class_ce.append(float(class_ce_values[active].mean().item()))
                    within_class_ce.append(float(within_class_ce_values[active].mean().item()))
                    content_class_ce.append(float(class_ce_values[content].mean().item()))
                    content_within_class_ce.append(float(within_class_ce_values[content].mean().item()))
                    probs = F.softmax(class_logits.float(), dim=-1)
                    active_probs = probs[active]
                    safe_probs = active_probs.clamp_min(1e-10)
                    class_entropy.append(float((-(safe_probs * safe_probs.log()).sum(dim=-1)).mean().item()))
                    class_max_prob.append(float(active_probs.max(dim=-1).values.mean().item()))
                    class_prob_eos_special.append(float(active_probs[:, 0].mean().item()))
                    if active_probs.shape[-1] > 1:
                        class_prob_low_freq.append(float(active_probs[:, 1].mean().item()))
                    if active_probs.shape[-1] > 7:
                        class_prob_high_freq.append(float(active_probs[:, 7].mean().item()))
            key = f"t{int(t_val * 10):02d}"
            metrics[f"val_ce_{key}"] = sum(all_ce) / len(all_ce)
            metrics[f"val_ce_{key}_batch_variance"] = _population_variance(all_ce)
            metrics[f"val_content_ce_{key}"] = sum(content_ce) / len(content_ce)
            metrics[f"val_content_ce_{key}_batch_variance"] = _population_variance(content_ce)
            if config.train.eos_hazard_loss_weight > 0:
                eos_hazard_loss = sum(eos_hazard_losses) / len(eos_hazard_losses) if eos_hazard_losses else 0.0
                metrics.update(
                    _eos_hazard_accumulator_metrics(
                        f"val_{key}_",
                        eos_hazard_accumulator,
                        eos_hazard_loss,
                        eos_hazard_loss * float(config.train.eos_hazard_loss_weight),
                        float(config.train.eos_hazard_loss_weight),
                    )
                )
            if use_hierarchical_output:
                metrics[f"val_class_ce_{key}"] = sum(class_ce) / len(class_ce) if class_ce else 0.0
                metrics[f"val_within_class_ce_{key}"] = (
                    sum(within_class_ce) / len(within_class_ce) if within_class_ce else 0.0
                )
                metrics[f"val_content_class_ce_{key}"] = (
                    sum(content_class_ce) / len(content_class_ce) if content_class_ce else 0.0
                )
                metrics[f"val_content_within_class_ce_{key}"] = (
                    sum(content_within_class_ce) / len(content_within_class_ce) if content_within_class_ce else 0.0
                )
                metrics[f"val_class_entropy_{key}"] = sum(class_entropy) / len(class_entropy) if class_entropy else 0.0
                metrics[f"val_class_max_prob_{key}"] = (
                    sum(class_max_prob) / len(class_max_prob) if class_max_prob else 0.0
                )
                metrics[f"val_class_prob_eos_special_{key}"] = (
                    sum(class_prob_eos_special) / len(class_prob_eos_special) if class_prob_eos_special else 0.0
                )
                metrics[f"val_class_prob_low_freq_{key}"] = (
                    sum(class_prob_low_freq) / len(class_prob_low_freq) if class_prob_low_freq else 0.0
                )
                metrics[f"val_class_prob_high_freq_{key}"] = (
                    sum(class_prob_high_freq) / len(class_prob_high_freq) if class_prob_high_freq else 0.0
                )

    if config.noise.type == "mdlm" and config.noise.suffix_masking:
        metrics["val_indep_fixed_ce_corruption"] = "independent-mdlm-cross-objective"
        with torch.no_grad():
            for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
                all_ce, content_ce = [], []
                class_ce, within_class_ce = [], []
                content_class_ce, content_within_class_ce = [], []
                for _ in range(config.eval.fixed_ce_batches):
                    batch = next(val_loader)
                    eval_attention_mask = _batch_model_attention_mask(batch)
                    x_eval = batch.x_0
                    z_eval, w_eval = noise.corrupt(x_eval, t_val, maskable=batch.loss_mask)
                    with autocast_ctx:
                        if use_hierarchical_output:
                            class_logits, token_logits = _forward_eval_hierarchical_components(
                                raw_model,
                                config,
                                z_eval,
                                x_eval,
                                eval_attention_mask,
                                noise_level=None,
                                position_ids=getattr(batch, "position_ids", None),
                            )
                            ce, class_ce_values, within_class_ce_values = raw_model.hierarchical_ce(
                                class_logits,
                                token_logits,
                                x_eval,
                            )
                        else:
                            logits = _forward_eval_logits(
                                raw_model,
                                config,
                                z_eval,
                                x_eval,
                                eval_attention_mask,
                                noise_level=None,
                                position_ids=getattr(batch, "position_ids", None),
                                target_lengths=_target_lengths_for_conditioning(
                                    config,
                                    x_eval,
                                    batch.attention_mask,
                                    eos_token_id,
                                ),
                            )
                            ce = F.cross_entropy(
                                logits.view(-1, logits.size(-1)), x_eval.view(-1), reduction="none"
                            ).view_as(x_eval)
                            class_ce_values = None
                            within_class_ce_values = None
                    active = (w_eval > 0) & batch.loss_mask
                    content = active & (token_bytes[x_eval] > 0)
                    all_ce.append(float(ce[active].mean().item()))
                    content_ce.append(float(ce[content].mean().item()))
                    if use_hierarchical_output and class_ce_values is not None and within_class_ce_values is not None:
                        class_ce.append(float(class_ce_values[active].mean().item()))
                        within_class_ce.append(float(within_class_ce_values[active].mean().item()))
                        content_class_ce.append(float(class_ce_values[content].mean().item()))
                        content_within_class_ce.append(float(within_class_ce_values[content].mean().item()))
                key = f"t{int(t_val * 10):02d}"
                metrics[f"val_indep_ce_{key}"] = sum(all_ce) / len(all_ce)
                metrics[f"val_indep_content_ce_{key}"] = sum(content_ce) / len(content_ce)
                if use_hierarchical_output:
                    metrics[f"val_indep_class_ce_{key}"] = sum(class_ce) / len(class_ce) if class_ce else 0.0
                    metrics[f"val_indep_within_class_ce_{key}"] = (
                        sum(within_class_ce) / len(within_class_ce) if within_class_ce else 0.0
                    )
                    metrics[f"val_indep_content_class_ce_{key}"] = (
                        sum(content_class_ce) / len(content_class_ce) if content_class_ce else 0.0
                    )
                    metrics[f"val_indep_content_within_class_ce_{key}"] = (
                        sum(content_within_class_ce) / len(content_within_class_ce) if content_within_class_ce else 0.0
                    )

    if config.eval.reconstruction_eval:
        reconstruction_corrupt_fn = None
        reconstruction_corruption_label = "independent-mdlm"
        if config.noise.type == "mdlm" and config.noise.suffix_masking:
            reconstruction_corruption_label = corruption_label

            def reconstruction_corrupt_fn(x_eval, loss_mask, t_val):
                z_eval, w_eval, _t, noise_level = _corrupt_eval_batch(
                    config,
                    noise,
                    x_eval,
                    loss_mask,
                    t_val=t_val,
                )
                return z_eval, w_eval, noise_level

        with autocast_ctx:
            metrics.update(
                evaluate_reconstruction_metrics(
                    raw_model,
                    config.data,
                    tokenizer,
                    noise,
                    int(config.train.device_batch_size),
                    device=device,
                    seq_len=config.model.max_seq_len,
                    t_values=config.eval.reconstruction_t_values,
                    batches=config.eval.reconstruction_batches,
                    run_dir=config.train.run_dir,
                    log_samples=config.eval.reconstruction_log_samples,
                    step=step,
                    model_logits_fn=model_logits_fn,
                    corrupt_fn=reconstruction_corrupt_fn,
                    corruption_label=reconstruction_corruption_label,
                )
            )

    if config.eval.conditional_rank_eval:
        with autocast_ctx:
            metrics.update(
                evaluate_conditional_rank_suffix(
                    raw_model,
                    tokenizer,
                    config.data,
                    device,
                    batches=config.eval.conditional_rank_batches,
                    batch_size=int(config.train.device_batch_size),
                    seq_len=config.model.max_seq_len,
                    model_logits_fn=model_logits_fn,
                    include_leave_one_out=config.eval.conditional_rank_loo_eval,
                    loo_positions_per_sequence=config.eval.conditional_rank_loo_positions_per_sequence,
                    include_stratified_matched=config.eval.conditional_rank_stratified_eval,
                    stratified_target_per_decile=config.eval.conditional_rank_stratified_target_per_decile,
                    stratified_partial_mask_fraction=config.eval.conditional_rank_stratified_partial_mask_fraction,
                    run_dir=config.train.run_dir,
                    stratified_log_samples=config.eval.conditional_rank_stratified_log_samples,
                    step=step,
                )
            )

    if config.eval.conditional_rank_rollout_eval:
        with autocast_ctx:
            metrics.update(
                evaluate_conditional_rank_rollout(
                    raw_model,
                    tokenizer,
                    config.data,
                    device,
                    batches=config.eval.conditional_rank_rollout_batches,
                    batch_size=int(config.train.device_batch_size),
                    seq_len=config.model.max_seq_len,
                    model_logits_fn=model_logits_fn,
                    commit_fractions=config.eval.conditional_rank_rollout_commit_fractions,
                    sample_steps=config.eval.sample_steps,
                    sampler=config.eval.sampler,
                    scoring_mode=config.eval.sample_scoring_mode,
                    token_selection=config.eval.sample_token_selection,
                    temperature=config.eval.gen_temperature,
                    jitter_scale=config.eval.sample_jitter_scale,
                    fhs_tau=config.eval.fhs_tau,
                    run_dir=config.train.run_dir,
                    log_samples=config.eval.conditional_rank_rollout_log_samples,
                    step=step,
                )
            )

    if config.eval.conditional_gen_eval:
        if config.noise.type != "mdlm":
            raise ValueError("eval.conditional_gen_eval currently supports noise.type='mdlm'")

        def conditional_sample_fn(m, prefixes, continuation_len):
            target_lengths = None
            if _uses_target_length_conditioning(config):
                total_len = int(prefixes.shape[1]) + int(continuation_len)
                target_lengths = torch.full(
                    (int(prefixes.shape[0]),),
                    total_len,
                    dtype=torch.long,
                    device=prefixes.device,
                )
            return sample_mdlm_from_prefixes(
                m,
                noise,
                tokenizer,
                prefixes,
                int(continuation_len),
                config.eval.sample_steps,
                scoring_mode=config.eval.sample_scoring_mode,
                token_selection=config.eval.sample_token_selection,
                temperature=config.eval.gen_temperature,
                jitter_scale=config.eval.sample_jitter_scale,
                return_trace=config.eval.sample_trace,
                target_lengths=target_lengths,
            )

        with autocast_ctx:
            metrics.update(
                evaluate_conditional_generation_metrics(
                    raw_model,
                    tokenizer,
                    config.data,
                    noise,
                    conditional_sample_fn,
                    int(config.train.device_batch_size),
                    config.train.run_dir,
                    device=device,
                    seq_len=config.model.max_seq_len,
                    prefix_fracs=config.eval.conditional_gen_prefix_fracs,
                    gen_samples=config.eval.conditional_gen_samples,
                    continuation_len=config.eval.conditional_gen_len,
                    log_samples=config.eval.conditional_gen_log_samples,
                    score_mode=config.eval.gen_score_mode,
                    self_score_t=config.eval.gen_self_score_t,
                    step=step,
                    sample_label=config.eval.sample_token_selection,
                    model_logits_fn=model_logits_fn,
                )
            )

    if include_gen:

        def build_sample_fn(token_selection: str):
            def sample_fn(m, n, length):
                target_lengths = _sample_target_lengths_from_prior(
                    target_length_prior_counts,
                    n,
                    length,
                    device,
                )
                if _uses_bd3lm(config):
                    return sample_bd3lm(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        block_size=int(config.model.bd3lm_block_size),
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=token_selection,
                        temperature=config.eval.gen_temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        return_trace=config.eval.sample_trace,
                    )
                if _uses_block_causal_pilot(config):
                    return sample_block_causal_pilot(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        block_size=int(config.model.block_causal_pilot_size),
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=token_selection,
                        temperature=config.eval.gen_temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        tau=config.eval.fhs_tau,
                        length_policy=config.eval.length_policy,
                        return_trace=config.eval.sample_trace,
                    )
                if config.eval.sampler == "remdm":
                    return sample_remdm(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=token_selection,
                        temperature=config.eval.gen_temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        tau_edit=config.eval.remdm_tau_edit,
                        return_trace=config.eval.sample_trace,
                        target_lengths=target_lengths,
                    )
                if config.eval.sampler == "fhs":
                    return sample_fhs(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=token_selection,
                        temperature=config.eval.gen_temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        tau=config.eval.fhs_tau,
                        length_policy=config.eval.length_policy,
                        return_trace=config.eval.sample_trace,
                        target_lengths=target_lengths,
                    )
                if config.noise.type == "mdlm":
                    return sample_mdlm(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=token_selection,
                        temperature=config.eval.gen_temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        return_trace=config.eval.sample_trace,
                        target_lengths=target_lengths,
                    )
                sample_impl = {
                    "uniform": sample_uniform,
                    "mask_uniform": sample_mask_uniform,
                }[config.noise.type]
                return sample_impl(
                    m,
                    noise,
                    tokenizer,
                    n,
                    length,
                    config.eval.sample_steps,
                    scoring_mode=config.eval.sample_scoring_mode,
                    token_selection=token_selection,
                    temperature=config.eval.gen_temperature,
                    jitter_scale=config.eval.sample_jitter_scale,
                    return_trace=config.eval.sample_trace,
                )

            return sample_fn

        primary_token_selection = config.eval.sample_token_selection
        metrics["gen_primary_token_selection"] = primary_token_selection
        with autocast_ctx:
            metrics.update(
                evaluate_generation_metrics(
                    raw_model,
                    tokenizer,
                    noise,
                    build_sample_fn(primary_token_selection),
                    int(config.train.device_batch_size),
                    config.train.run_dir,
                    device=device,
                    judge_model_id=config.eval.judge_model_id,
                    gen_samples=config.eval.gen_samples,
                    gen_len=config.eval.gen_len,
                    log_samples=config.eval.gen_log_samples,
                    score_mode=config.eval.gen_score_mode,
                    self_score_t=config.eval.gen_self_score_t,
                    step=step,
                    sample_label=primary_token_selection,
                    model_logits_fn=model_logits_fn,
                    block_size=int(config.model.block_causal_pilot_size) if _uses_block_causal_pilot(config) else 0,
                )
            )
        if primary_token_selection == "argmax":
            metrics["gen_argmax_eos_rate"] = float(metrics.get("gen_eos_rate", 0.0))
        elif primary_token_selection == "multinomial":
            metrics["gen_stochastic_eos_rate"] = float(metrics.get("gen_eos_rate", 0.0))

        if config.noise.type == "mdlm" and (config.noise.suffix_masking or config.train.eos_hazard_loss_weight > 0):
            secondary_token_selection = "multinomial" if primary_token_selection == "argmax" else "argmax"
            secondary_label = "stochastic" if secondary_token_selection == "multinomial" else "argmax"
            metrics["gen_secondary_token_selection"] = secondary_token_selection
            with autocast_ctx:
                secondary_metrics = evaluate_generation_metrics(
                    raw_model,
                    tokenizer,
                    noise,
                    build_sample_fn(secondary_token_selection),
                    int(config.train.device_batch_size),
                    config.train.run_dir,
                    device=device,
                    judge_model_id=config.eval.judge_model_id,
                    gen_samples=config.eval.gen_samples,
                    gen_len=config.eval.gen_len,
                    log_samples=config.eval.gen_log_samples,
                    score_mode=config.eval.gen_score_mode,
                    self_score_t=config.eval.gen_self_score_t,
                    step=step,
                    sample_label=secondary_label,
                    model_logits_fn=model_logits_fn,
                )
            for key, value in secondary_metrics.items():
                suffix = key[4:] if key.startswith("gen_") else key
                metrics[f"gen_{secondary_label}_{suffix}"] = value
            if secondary_token_selection == "argmax":
                metrics["gen_argmax_eos_rate"] = float(secondary_metrics.get("gen_eos_rate", 0.0))
            elif secondary_token_selection == "multinomial":
                metrics["gen_stochastic_eos_rate"] = float(secondary_metrics.get("gen_eos_rate", 0.0))

    if was_training:
        raw_model.train()
    return metrics


def _run_eval(
    raw_model: nn.Module,
    config: ExperimentConfig,
    tokenizer,
    noise,
    autocast_ctx,
    device,
    step: int,
    include_gen: bool,
) -> dict:
    """Run the fixed evaluation bundle without advancing the training RNG streams."""

    saved_state = _rng_state(config)
    was_training = raw_model.training
    eval_seed = int(config.eval.nelbo_seed)
    try:
        random.seed(eval_seed)
        np.random.seed(eval_seed % (2**32))
        torch.manual_seed(eval_seed)
        if config.runtime.device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        return _run_eval_unisolated(
            raw_model,
            config,
            tokenizer,
            noise,
            autocast_ctx,
            device,
            step,
            include_gen,
        )
    finally:
        _restore_rng_state(saved_state, config)
        if was_training:
            raw_model.train()


@torch.no_grad()
def _evaluate_fixed_noise_ce_trace(
    raw_model: nn.Module,
    config: ExperimentConfig,
    tokenizer,
    noise,
    autocast_ctx,
    device,
    step: int,
    total_steps: int,
    train_tokens: int,
    train_valid_tokens: int,
) -> dict[str, bool | float | int | list[float] | str]:
    """Evaluate fixed validation examples and corruption masks for CE traces."""
    saved_state = _rng_state(config)
    was_training = raw_model.training
    raw_model.eval()
    token_bytes = build_token_bytes(tokenizer).to(device)
    eos_token_id = tokenizer.get_eos_token_id()
    use_hierarchical_output = hasattr(raw_model, "uses_hierarchical_output") and raw_model.uses_hierarchical_output()
    t_values = (0.1, 0.3, 0.5, 0.7, 0.9)
    record: dict[str, bool | float | int | list[float] | str] = {
        "step": int(step),
        "checkpoint_step": int(step),
        "total_steps": int(total_steps),
        "train_tokens": int(train_tokens),
        "train_valid_tokens": int(train_valid_tokens),
        "fixed_ce_trace_checkpoint_step": int(step),
        "fixed_ce_trace_batches": int(config.eval.fixed_ce_batches),
        "fixed_ce_trace_corruption": _training_corruption_label(config),
        "fixed_ce_trace_eval_noise": "standard_scalar_t_mdlm"
        if _uses_per_token_noise_levels(config)
        else _training_corruption_label(config),
        "fixed_ce_trace_t_values": list(t_values),
    }
    seed_base = config.runtime.training_seed if config.runtime.training_seed >= 0 else config.runtime.seed
    try:
        for t_idx, t_val in enumerate(t_values):
            fixed_seed = int(seed_base) + 1_048_583 + t_idx * 8_191
            torch.manual_seed(fixed_seed)
            np.random.seed(fixed_seed % (2**32 - 1))
            random.seed(fixed_seed)
            if config.runtime.device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed(fixed_seed)
            val_loader = build_val_loader(config, tokenizer, device)
            ce_sum = 0.0
            content_ce_sum = 0.0
            active_count = 0
            content_count = 0
            lossable_count = 0
            for _ in range(int(config.eval.fixed_ce_batches)):
                batch = next(val_loader)
                x_eval = batch.x_0
                eval_attention_mask = _batch_model_attention_mask(batch)
                z_eval, w_eval, _, noise_level = _corrupt_eval_batch(
                    config,
                    noise,
                    x_eval,
                    batch.loss_mask,
                    t_val=t_val,
                )
                with autocast_ctx:
                    if use_hierarchical_output:
                        class_logits, token_logits = _forward_eval_hierarchical_components(
                            raw_model,
                            config,
                            z_eval,
                            x_eval,
                            eval_attention_mask,
                            noise_level=noise_level,
                            position_ids=getattr(batch, "position_ids", None),
                        )
                        ce, _class_ce_values, _within_class_ce_values = raw_model.hierarchical_ce(
                            class_logits,
                            token_logits,
                            x_eval,
                        )
                    else:
                        logits = _forward_eval_logits(
                            raw_model,
                            config,
                            z_eval,
                            x_eval,
                            eval_attention_mask,
                            noise_level=noise_level,
                            position_ids=getattr(batch, "position_ids", None),
                            target_lengths=_target_lengths_for_conditioning(
                                config,
                                x_eval,
                                batch.attention_mask,
                                eos_token_id,
                            ),
                        )
                        ce = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            x_eval.view(-1),
                            reduction="none",
                        ).view_as(x_eval)
                active = (w_eval > 0) & batch.loss_mask
                content = active & (token_bytes[x_eval] > 0)
                active_values = ce[active].detach().float()
                content_values = ce[content].detach().float()
                ce_sum += float(active_values.sum().item()) if active_values.numel() else 0.0
                content_ce_sum += float(content_values.sum().item()) if content_values.numel() else 0.0
                active_count += int(active.sum().item())
                content_count += int(content.sum().item())
                lossable_count += int(batch.loss_mask.sum().item())
            key = f"t{int(round(t_val * 10)):02d}"
            record[f"fixed_ce_trace_ce_{key}"] = ce_sum / max(1, active_count)
            record[f"fixed_ce_trace_content_ce_{key}"] = content_ce_sum / max(1, content_count)
            record[f"fixed_ce_trace_active_tokens_{key}"] = int(active_count)
            record[f"fixed_ce_trace_content_tokens_{key}"] = int(content_count)
            record[f"fixed_ce_trace_active_fraction_{key}"] = active_count / max(1, lossable_count)
            record[f"fixed_ce_trace_seed_{key}"] = int(fixed_seed)
    finally:
        _restore_rng_state(saved_state, config)
        if was_training:
            raw_model.train()
    return record


def _publish_final_outputs(
    config: ExperimentConfig,
    *,
    rank: int,
    metrics: dict,
    tensorboard_writer,
    total_steps: int,
    raw_model: nn.Module,
    optimizers,
    scaler,
    model_config: DiffusionConfig,
    train_tokens: int,
    train_valid_tokens: int,
    resume_contract: dict[str, object],
) -> None:
    metrics_path = os.path.join(config.train.run_dir, "metrics.json")
    final_checkpoint_path = os.path.join(config.train.run_dir, "model_latest.pt")
    final_checkpoint_alias_path = os.path.join(config.train.run_dir, "model.pt")

    _write_tensorboard_scalars(tensorboard_writer, "final", metrics, total_steps)
    flush_tensorboard_writer(tensorboard_writer)
    save_checkpoint(
        final_checkpoint_path,
        raw_model,
        optimizers,
        scaler,
        config,
        model_config,
        total_steps,
        train_tokens,
        train_valid_tokens,
        metrics,
        resume_contract=resume_contract,
    )
    alias_checkpoint(final_checkpoint_path, final_checkpoint_alias_path)
    write_json_atomic(metrics_path, metrics)
    print0(rank, f"metrics saved to {metrics_path}")
    print0(rank, f"model saved to {final_checkpoint_alias_path}")


def _destroy_process_group_if_initialized() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _restore_gc_state(was_enabled: bool, unfreeze: bool) -> None:
    if unfreeze and gc.get_freeze_count() > 0:
        gc.unfreeze()
    if was_enabled:
        gc.enable()
    else:
        gc.disable()


def train(config: ExperimentConfig) -> dict:
    process_group_was_initialized = dist.is_available() and dist.is_initialized()
    with ExitStack() as resources:
        if not process_group_was_initialized:
            resources.callback(_destroy_process_group_if_initialized)
        return _train_impl(config, resources)


def _train_impl(config: ExperimentConfig, resources: ExitStack) -> dict:
    os.makedirs(config.train.run_dir, exist_ok=True)
    _configure_numeric_runtime(config)
    is_ddp, rank, local_rank, world_size, device = init_distributed(
        config.runtime.device_type,
        config.train.run_dir,
        init_single_rank_process_group=config.runtime.init_single_rank_process_group,
    )
    _check_gpu_preflight(config, rank)
    compute_dtype = dtype_from_name(config.runtime.dtype)
    autocast_ctx = (
        torch.autocast(device_type=config.runtime.device_type, dtype=compute_dtype)
        if config.runtime.device_type == "cuda" and compute_dtype != torch.float32
        else nullcontext()
    )

    training_seed = config.runtime.seed if config.runtime.training_seed < 0 else config.runtime.training_seed
    model_seed = config.runtime.seed if config.runtime.model_seed < 0 else config.runtime.model_seed
    torch.manual_seed(training_seed + rank)
    np.random.seed(training_seed + rank)
    random.seed(training_seed + rank)
    if config.runtime.device_type == "cuda":
        torch.cuda.manual_seed(training_seed + rank)

    tokenizer = build_tokenizer(config)
    label_smoothing_metrics = _label_smoothing_ar_metrics(config, tokenizer)
    if _training_diagnostic_requested(config):
        diagnostic_metrics = _training_diagnostics_module().run_configured_diagnostic(
            config,
            tokenizer,
            rank=rank,
            world_size=world_size,
            device=device,
            is_ddp=is_ddp,
        )
        if diagnostic_metrics is None:
            raise RuntimeError("configured training diagnostic did not select a diagnostic runner")
        return diagnostic_metrics
    data_identity_summary = _prepare_data_identity(
        config,
        rank=rank,
        world_size=world_size,
        is_ddp=is_ddp,
    )
    training_token_bytes = build_token_bytes(tokenizer).to(device)
    token_class_ids = None
    token_class_metadata: dict | None = None
    token_edit_class_ids = None
    token_edit_class_metadata: dict | None = None
    if _uses_hierarchical_output(config):
        saved_state = _rng_state(config)
        token_class_ids, token_class_metadata = build_frequency_token_classes(
            config.data,
            tokenizer,
            int(config.train.device_batch_size),
            int(config.model.max_seq_len),
            device,
            int(config.model.hierarchical_frequency_batches),
        )
        _restore_rng_state(saved_state, config)
        token_edit_class_ids = token_class_ids
        token_edit_class_metadata = token_class_metadata
    elif _uses_token_edit_corruption(config):
        saved_state = _rng_state(config)
        token_edit_class_ids, token_edit_class_metadata = build_frequency_token_classes(
            config.data,
            tokenizer,
            int(config.train.device_batch_size),
            int(config.model.max_seq_len),
            device,
            int(config.model.hierarchical_frequency_batches),
        )
        _restore_rng_state(saved_state, config)
    model_config = DiffusionConfig(
        sequence_len=config.model.max_seq_len,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=int(config.model.depth),
        n_head=int(config.model.n_head),
        n_embd=int(config.model.n_embd),
        mask_token_id=tokenizer.get_mask_token_id(),
        mlp_hidden_dim=int(config.model.mlp_hidden_dim),
        attention_scale=float(config.model.attention_scale),
        local_attention_lower_layers=int(config.model.local_attention_lower_layers),
        local_attention_window=int(config.model.local_attention_window),
        auxiliary_head_layers=tuple(int(i) for i in config.model.auxiliary_head_layers),
        time_conditioning=bool(config.noise.time_conditioning),
        bd3lm_block_size=int(config.model.bd3lm_block_size) if _uses_bd3lm(config) else 0,
        output_factorization=config.model.output_factorization,
        causal=bool(config.model.causal),
        residual_output_projection_init=config.model.residual_output_projection_init,
        embedding_head_init_std_multiplier=float(config.model.embedding_head_init_std_multiplier),
        length_head=bool(config.model.length_head),
        target_length_conditioning=bool(config.model.target_length_conditioning),
        mask_position_embedding=bool(config.model.mask_position_embedding),
    )

    saved_state = _rng_state(config)
    torch.manual_seed(model_seed)
    if config.runtime.device_type == "cuda":
        torch.cuda.manual_seed(model_seed)
    raw_model = DiffusionTransformer(
        model_config,
        rope_base=config.model.rope_base,
        compute_dtype=compute_dtype,
        pad_vocab_to=config.model.pad_vocab_to,
        token_class_ids=token_class_ids,
    ).to(device)
    _restore_rng_state(saved_state, config)

    counts = raw_model.parameter_counts()
    flops_per_token = raw_model.estimate_flops()
    architecture_fingerprint = raw_model.architecture_fingerprint()
    initial_state_artifact = (
        _pre_update_parameter_state_artifact(
            raw_model,
            architecture_fingerprint_sha256=str(architecture_fingerprint["fingerprint_sha256"]),
            model_seed=model_seed,
        )
        if rank == 0
        else None
    )
    if is_ddp:
        initial_state_objects: list[object | None] = [initial_state_artifact]
        dist.broadcast_object_list(initial_state_objects, src=0)
        initial_state_artifact = initial_state_objects[0]
    if not isinstance(initial_state_artifact, dict):
        raise RuntimeError("rank 0 did not provide a valid pre-update parameter-state artifact")
    parameter_accounting = _parameter_accounting_summary(
        config, counts, architecture_fingerprint["parameter_inventory"]
    )
    total_batch_size, grad_accum_steps, total_steps, batch_lr_scale = _training_budget(
        config, counts, world_size, raw_model=raw_model
    )
    total_budget_tokens = total_steps * total_batch_size
    warmup_steps = _resolved_warmup_steps(config, total_steps)
    lr_schedule = _lr_schedule_telemetry(config, total_steps, warmup_steps)
    trajectory_contract = _resume_trajectory_contract(
        config,
        device=device,
        model_config=asdict(model_config),
        architecture_fingerprint=architecture_fingerprint,
        data_identity_summary=data_identity_summary,
        total_batch_size=total_batch_size,
        grad_accum_steps=grad_accum_steps,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        lr_schedule=lr_schedule,
    )
    cpu_threading = _cpu_threading_telemetry()
    record_embedding_head_init_diagnostics = _embedding_head_init_diagnostics_enabled(config)

    print0(
        rank,
        json.dumps(
            {
                "model": asdict(model_config),
                "param_counts": counts,
                "parameter_accounting": parameter_accounting,
                "flops_per_token": flops_per_token,
                "total_batch_size": total_batch_size,
                "effective_batch_size_tokens": total_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "total_steps": total_steps,
                "target_tokens": config.train.target_tokens,
                "total_budget_tokens": total_budget_tokens,
                "world_size": world_size,
                "compute_dtype": str(compute_dtype),
                "lr_schedule": lr_schedule,
                "cpu_threading": cpu_threading,
            },
            indent=2,
        ),
    )
    if rank == 0:
        write_config(os.path.join(config.train.run_dir, "resolved_config.json"), config)
        write_json_atomic(os.path.join(config.train.run_dir, "architecture_fingerprint.json"), architecture_fingerprint)
        write_json_atomic(os.path.join(config.train.run_dir, "initial_state.v1.json"), initial_state_artifact)

    optimizers = make_optimizer(raw_model, config, batch_lr_scale)
    _set_optimizer_base_lrs(optimizers, config, batch_lr_scale)
    if rank == 0:
        write_json_atomic(
            os.path.join(config.train.run_dir, "run_summary.json"),
            {
                "model": asdict(model_config),
                "architecture_fingerprint": architecture_fingerprint,
                "initial_state": initial_state_artifact,
                "parameter_accounting": parameter_accounting,
                "tokenizer": _tokenizer_summary(tokenizer),
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
                "data_identity": data_identity_summary,
                "data_packing": {
                    "boundary_aware_packing": bool(config.data.boundary_aware_packing),
                    "content_budget_segmented_packing": bool(config.data.content_budget_segmented_packing),
                    "short_document_curriculum": bool(config.data.short_document_curriculum),
                },
                "eos_supervision_audit": {
                    "enabled": _eos_supervision_audit_enabled(config),
                    "artifact": "telemetry/eos_supervision_audit.jsonl",
                    "prepack_document_batches": EOS_SUPERVISION_PREPACK_DOCUMENT_BATCHES,
                    "packed_batches": EOS_SUPERVISION_PACKED_BATCHES,
                    "forced_mask_t": EOS_SUPERVISION_FORCED_MASK_T,
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
                    "enabled": _uses_token_edit_corruption(config),
                    "batch_fraction": float(config.train.token_edit_corruption_batch_fraction),
                    "ratio": float(config.train.token_edit_corruption_ratio),
                    "frequency_deciles": TOKEN_EDIT_CORRUPTION_DECILES,
                    "frequency_batches": int(token_edit_class_metadata.get("frequency_batches", 0))
                    if token_edit_class_metadata
                    else 0,
                    "frequency_observed_tokens": int(token_edit_class_metadata.get("frequency_observed_tokens", 0))
                    if token_edit_class_metadata
                    else 0,
                },
                "residual_output_projection_init": _residual_projection_init_metrics(config),
                "embedding_head_init": _embedding_head_init_metrics(config),
                "embedding_head_init_diagnostics": {
                    "enabled": bool(record_embedding_head_init_diagnostics),
                    "artifact": "telemetry/embedding_head_init_diagnostics.jsonl",
                    "row_norm_frequency_deciles": 7,
                    "early_snapshot_step": 1,
                },
                "output_factorization": _hierarchical_output_telemetry(token_class_metadata),
                "param_counts": counts,
                "num_params_M": counts["total"] / 1e6,
                "num_non_embedding_params_M": counts["non_embedding"] / 1e6,
                "num_scaling_params_M": counts["scaling"] / 1e6,
                "flops_per_token": flops_per_token,
                "total_batch_size": total_batch_size,
                "effective_batch_size_tokens": total_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "total_steps": total_steps,
                "target_tokens": config.train.target_tokens,
                "total_budget_tokens": total_budget_tokens,
                "world_size": world_size,
                "checkpoint_resume_contract": {
                    "version": RESUME_CONTRACT_VERSION,
                    "batch_digest_version": RESUME_BATCH_DIGEST_VERSION,
                    "batch_tensor_fields": list(RESUME_BATCH_TENSOR_FIELDS),
                    "counter_reduction": "packed_int64_all_reduce",
                    "trajectory_sha256": trajectory_contract["sha256"],
                    "data_identity_bound": bool(data_identity_summary["enabled"]),
                },
                "batch_lr_scale": batch_lr_scale,
                "lr_config": {
                    "lr": config.optim.lr,
                    "muon_lr": config.optim.muon_lr,
                    "adamw_lr": config.optim.adamw_lr,
                },
                "lr_schedule": lr_schedule,
                "cpu_threading": cpu_threading,
                "fixed_ce_trace": {
                    "enabled": bool(config.eval.fixed_ce_trace_enabled),
                    "artifact": "telemetry/fixed_noise_ce_trace.jsonl",
                    "probe_steps": _fixed_ce_trace_steps(total_steps, warmup_steps)
                    if config.eval.fixed_ce_trace_enabled
                    else [],
                    "batches": int(config.eval.fixed_ce_batches),
                    "t_values": [0.1, 0.3, 0.5, 0.7, 0.9],
                },
                "optimizers": [
                    {
                        "class": optimizer.__class__.__name__,
                        "param_groups": len(optimizer.param_groups),
                        "base_lr": (
                            optimizer.param_groups[0].get(
                                "base_lr",
                                optimizer.param_groups[0].get("lr", 0.0),
                            )
                            if optimizer.param_groups
                            else 0.0
                        ),
                        "params": sum(p.numel() for group in optimizer.param_groups for p in group.get("params", [])),
                    }
                    for optimizer in optimizers
                ],
            },
        )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.runtime.device_type == "cuda" and compute_dtype == torch.float16,
    )

    start_step = 0
    train_tokens = 0
    train_valid_tokens = 0
    train_content_tokens = 0
    train_content_bytes = 0
    block_causal_visible_prefix_frac_sum = 0.0
    block_causal_visible_prefix_count = 0
    resume_metrics: dict = {}
    resume_state: dict[str, object] | None = None
    resume_next_batch_digest_verified = False
    resume_path = _resolve_resume_path(config.train.run_dir, config.train.resume_from)
    if resume_path is not None:
        start_step, train_tokens, train_valid_tokens, resume_metrics, resume_state = _load_checkpoint(
            resume_path,
            raw_model,
            optimizers,
            scaler,
            config,
            device,
            rank=rank,
            world_size=world_size,
            grad_accum_steps=grad_accum_steps,
            trajectory_contract=trajectory_contract,
        )
        _set_optimizer_base_lrs(optimizers, config, batch_lr_scale)
        print0(rank, f"resumed checkpoint {resume_path} at step {start_step}")
        train_content_tokens = int(resume_metrics.get("train_content_tokens", 0))
        train_content_bytes = int(resume_metrics.get("train_content_bytes", 0))
        if start_step > total_steps:
            raise RuntimeError(f"checkpoint step {start_step} exceeds the current training horizon {total_steps}")
    resume_train_tokens = train_tokens
    resume_train_valid_tokens = train_valid_tokens
    resume_train_content_tokens = train_content_tokens
    resume_train_content_bytes = train_content_bytes

    use_hierarchical_output = raw_model.uses_hierarchical_output()
    return_aux_logits = config.train.auxiliary_loss_weight > 0 and len(model_config.auxiliary_head_layers) > 0
    model = torch.compile(raw_model, dynamic=config.runtime.compile_dynamic) if config.runtime.compile else raw_model
    if is_ddp:
        ddp_kwargs = {"broadcast_buffers": _ddp_broadcast_buffers(config)}
        if config.runtime.device_type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=bool(model_config.auxiliary_head_layers and not return_aux_logits),
            **ddp_kwargs,
        )
    noise = build_noise(config.noise, model_config.mask_token_id, model_config.vocab_size)
    eos_supervision_audit_enabled = _eos_supervision_audit_enabled(config)
    eos_supervision_startup_metrics = _eos_supervision_startup_audit(
        config,
        tokenizer,
        noise,
        rank,
        world_size,
        device,
        is_ddp,
    )
    eos_supervision_train_counts = _new_eos_supervision_counts()

    train_loader = build_train_loader(config, tokenizer, rank, world_size, device)
    batch: DLMDataBatch | None = None
    prefetch_error: Exception | None = None
    try:
        batch = next(train_loader)
    except Exception as exc:
        prefetch_error = exc
    if resume_state is not None:
        _synchronize_resume_phase_error(
            "initial loader prefetch",
            prefetch_error,
            rank=rank,
            world_size=world_size,
            is_ddp=is_ddp,
        )
    elif prefetch_error is not None:
        raise prefetch_error
    assert batch is not None
    x_0 = batch.x_0
    loss_mask = batch.loss_mask
    attention_mask = batch.attention_mask
    epoch = batch.epoch
    if resume_state is not None:
        fast_forward = int(resume_state["consumed_microbatches"])
        print0(rank, f"fast-forwarding train loader by {fast_forward} micro-batches")
        fast_forward_error: Exception | None = None
        try:
            for _ in range(fast_forward):
                batch = next(train_loader)
        except Exception as exc:
            fast_forward_error = exc
        _synchronize_resume_phase_error(
            "loader fast-forward",
            fast_forward_error,
            rank=rank,
            world_size=world_size,
            is_ddp=is_ddp,
        )
        x_0 = batch.x_0
        loss_mask = batch.loss_mask
        attention_mask = batch.attention_mask
        epoch = batch.epoch
        replay_error: Exception | None = None
        try:
            resume_next_batch_digest_verified = _verify_replayed_next_batch(resume_state, batch, rank=rank)
        except Exception as exc:
            replay_error = exc
        _synchronize_resume_phase_error(
            "next-batch replay verification",
            replay_error,
            rank=rank,
            world_size=world_size,
            is_ddp=is_ddp,
        )
        restore_error: Exception | None = None
        try:
            if int(resume_state["version"]) == RESUME_CONTRACT_VERSION:
                _restore_active_rank_rng_state(resume_state["local_rng_state"], config, device)
            else:
                _restore_rng_state(resume_state["local_rng_state"], config)
        except Exception as exc:
            restore_error = exc
        _synchronize_resume_phase_error(
            "RNG restoration",
            restore_error,
            rank=rank,
            world_size=world_size,
            is_ddp=is_ddp,
        )

    bos_token_id = tokenizer.get_bos_token_id()
    eos_token_id = tokenizer.get_eos_token_id()
    special_token_ids = {int(i) for i in tokenizer.get_special_token_ids()}
    ar_boundary_counts = {
        "microbatches": 0,
        "microbatches_with_bos": 0,
        "valid_tokens": 0,
        "special_tokens": 0,
        "special_loss_tokens": 0,
        "bos_tokens": 0,
        "bos_loss_tokens": 0,
        "eos_tokens": 0,
        "eos_loss_tokens": 0,
    }
    boundary_telemetry = {
        "documents_per_row_sum": 0.0,
        "documents_per_row_count": 0,
        "pad_tokens": 0,
        "total_row_tokens": 0,
        "blocked_pairs_sum": 0,
        "causal_pairs_sum": 0,
        "valid_pairs_sum": 0,
        "allowed_pairs_sum": 0,
        "segment_blocked_pairs_sum": 0,
    }
    row_telemetry = {
        "rows": 0,
        "valid_tokens_sum": 0,
        "valid_tokens_min": 0,
        "valid_tokens_max": 0,
        "lossable_tokens_sum": 0,
        "lossable_tokens_min": 0,
        "lossable_tokens_max": 0,
        "pad_tokens": 0,
        "total_row_tokens": 0,
        "eos_rows": 0,
        "eos_tokens": 0,
        "eos_loss_tokens": 0,
    }

    def content_byte_counts(
        x_batch: torch.Tensor,
        attention_mask_batch: torch.Tensor,
    ) -> tuple[int, int]:
        byte_values = training_token_bytes[x_batch]
        content_mask = attention_mask_batch & (byte_values > 0)
        content_tokens = int(content_mask.sum().item())
        content_bytes = int(byte_values[attention_mask_batch].sum().item())
        return content_tokens, content_bytes

    def record_ar_boundary_batch(
        x_batch: torch.Tensor,
        loss_mask_batch: torch.Tensor,
        attention_mask_batch: torch.Tensor,
        position_ids_batch: torch.Tensor | None = None,
        document_boundary_mask_batch: torch.Tensor | None = None,
        document_start_mask_batch: torch.Tensor | None = None,
    ) -> None:
        if not (_is_causal(config) or _uses_content_budget_segmented_packing(config) or eos_supervision_audit_enabled):
            return
        ar_boundary_counts["microbatches"] += 1
        ar_boundary_counts["valid_tokens"] += int(attention_mask_batch.sum().item())
        b, t = x_batch.shape
        row_valid = attention_mask_batch.sum(dim=1)
        row_lossable = (loss_mask_batch & attention_mask_batch).sum(dim=1)
        row_count = int(row_valid.numel())
        row_telemetry["rows"] += row_count
        row_telemetry["valid_tokens_sum"] += int(row_valid.sum().item())
        row_telemetry["lossable_tokens_sum"] += int(row_lossable.sum().item())
        row_telemetry["pad_tokens"] += int((~attention_mask_batch).sum().item())
        row_telemetry["total_row_tokens"] += b * t
        if row_valid.numel():
            row_valid_min = int(row_valid.min().item())
            row_valid_max = int(row_valid.max().item())
            row_lossable_min = int(row_lossable.min().item())
            row_lossable_max = int(row_lossable.max().item())
            if row_telemetry["rows"] == row_count:
                row_telemetry["valid_tokens_min"] = row_valid_min
                row_telemetry["lossable_tokens_min"] = row_lossable_min
            else:
                row_telemetry["valid_tokens_min"] = min(int(row_telemetry["valid_tokens_min"]), row_valid_min)
                row_telemetry["lossable_tokens_min"] = min(int(row_telemetry["lossable_tokens_min"]), row_lossable_min)
            row_telemetry["valid_tokens_max"] = max(int(row_telemetry["valid_tokens_max"]), row_valid_max)
            row_telemetry["lossable_tokens_max"] = max(int(row_telemetry["lossable_tokens_max"]), row_lossable_max)
        if special_token_ids:
            special_mask = torch.zeros_like(attention_mask_batch, dtype=torch.bool)
            for token_id in special_token_ids:
                special_mask |= x_batch == token_id
            special_mask &= attention_mask_batch
            ar_boundary_counts["special_tokens"] += int(special_mask.sum().item())
            ar_boundary_counts["special_loss_tokens"] += int((special_mask & loss_mask_batch).sum().item())
        if eos_token_id is not None:
            eos_mask = (x_batch == int(eos_token_id)) & attention_mask_batch
            ar_boundary_counts["eos_tokens"] += int(eos_mask.sum().item())
            ar_boundary_counts["eos_loss_tokens"] += int((eos_mask & loss_mask_batch).sum().item())
            row_telemetry["eos_rows"] += int(eos_mask.any(dim=1).sum().item())
            row_telemetry["eos_tokens"] += int(eos_mask.sum().item())
            row_telemetry["eos_loss_tokens"] += int((eos_mask & loss_mask_batch).sum().item())
        if bos_token_id is not None:
            bos_mask = (x_batch == int(bos_token_id)) & attention_mask_batch
            bos_count = int(bos_mask.sum().item())
            ar_boundary_counts["bos_tokens"] += bos_count
            ar_boundary_counts["bos_loss_tokens"] += int((bos_mask & loss_mask_batch).sum().item())
            if bos_count:
                ar_boundary_counts["microbatches_with_bos"] += 1
        if position_ids_batch is not None and document_boundary_mask_batch is not None:
            if document_start_mask_batch is not None:
                docs_per_row = (document_start_mask_batch & attention_mask_batch).sum(dim=1).float()
            else:
                docs_per_row = ((position_ids_batch == 0) & attention_mask_batch).sum(dim=1).float()
            boundary_telemetry["documents_per_row_sum"] += int(docs_per_row.sum().item())
            boundary_telemetry["documents_per_row_count"] += b
            causal_mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x_batch.device))
            causal_mask = causal_mask.unsqueeze(0).expand(b, -1, -1)
            valid_q = attention_mask_batch[:, :, None]
            valid_k = attention_mask_batch[:, None, :]
            valid_pairs = valid_q & valid_k
            allowed_pairs = valid_pairs & document_boundary_mask_batch
            total_pairs = int(valid_pairs.sum().item())
            allowed_pair_count = int(allowed_pairs.sum().item())
            boundary_telemetry["valid_pairs_sum"] += total_pairs
            boundary_telemetry["allowed_pairs_sum"] += allowed_pair_count
            boundary_telemetry["segment_blocked_pairs_sum"] += total_pairs - allowed_pair_count
            valid_causal = causal_mask & valid_q & valid_k
            total_causal = int(valid_causal.sum().item())
            allowed_causal = int((valid_causal & document_boundary_mask_batch).sum().item())
            boundary_telemetry["causal_pairs_sum"] += total_causal
            boundary_telemetry["blocked_pairs_sum"] += total_causal - allowed_causal

    smooth_loss = 0.0
    smooth_count = 0
    last_loss_value = float(resume_metrics.get("train_loss", 0.0)) if resume_metrics else 0.0
    last_main_ce = (
        float(resume_metrics.get("last_train_main_ce", resume_metrics.get("raw_main_ce", 0.0)))
        if resume_metrics
        else 0.0
    )
    last_main_loss = (
        float(resume_metrics.get("last_train_main_loss", resume_metrics.get("raw_main_ce", 0.0)))
        if resume_metrics
        else 0.0
    )
    last_confidence_penalty = float(resume_metrics.get("last_train_confidence_penalty", 0.0)) if resume_metrics else 0.0
    last_confidence_penalty_loss = (
        float(resume_metrics.get("last_train_confidence_penalty_loss", 0.0)) if resume_metrics else 0.0
    )
    last_z_loss = float(resume_metrics.get("last_train_z_loss", 0.0)) if resume_metrics else 0.0
    last_z_loss_loss = float(resume_metrics.get("last_train_z_loss_loss", 0.0)) if resume_metrics else 0.0
    last_eos_hazard_calibration_loss = (
        float(resume_metrics.get("last_train_eos_hazard_calibration_loss", 0.0)) if resume_metrics else 0.0
    )
    last_eos_hazard_loss = float(resume_metrics.get("last_train_eos_hazard_loss", 0.0)) if resume_metrics else 0.0
    last_eos_hazard_accumulator = _new_eos_hazard_accumulator()
    last_length_head_accumulator = _new_length_head_accumulator(int(config.model.max_seq_len))
    train_length_head_accumulator = _new_length_head_accumulator(int(config.model.max_seq_len))
    last_position_decile_ce_accumulator = _new_position_decile_ce_accumulator()
    last_token_edit_reconstruction_accumulator = _new_token_edit_reconstruction_accumulator()
    train_token_edit_reconstruction_accumulator = _new_token_edit_reconstruction_accumulator()
    last_m2t_ce = float(resume_metrics.get("last_train_m2t_ce", 0.0)) if resume_metrics else 0.0
    last_m2t_loss = float(resume_metrics.get("last_train_m2t_loss", 0.0)) if resume_metrics else 0.0
    last_m2t_microbatches = int(resume_metrics.get("last_train_m2t_microbatches", 0)) if resume_metrics else 0
    last_m2t_active_tokens = int(resume_metrics.get("last_train_m2t_active_tokens", 0)) if resume_metrics else 0
    last_t2t_ce = float(resume_metrics.get("last_train_t2t_ce", 0.0)) if resume_metrics else 0.0
    last_t2t_loss = float(resume_metrics.get("last_train_t2t_loss", 0.0)) if resume_metrics else 0.0
    last_t2t_microbatches = int(resume_metrics.get("last_train_t2t_microbatches", 0)) if resume_metrics else 0
    last_t2t_active_tokens = int(resume_metrics.get("last_train_t2t_active_tokens", 0)) if resume_metrics else 0
    train_m2t_ce_sum = 0.0
    train_m2t_loss_sum = 0.0
    train_m2t_loss_count = 0
    train_m2t_microbatches = 0
    train_m2t_active_tokens = 0
    train_t2t_ce_sum = 0.0
    train_t2t_loss_sum = 0.0
    train_t2t_loss_count = 0
    train_t2t_microbatches = 0
    train_t2t_active_tokens = 0
    interval_m2t_ce_sum = 0.0
    interval_m2t_loss_sum = 0.0
    interval_m2t_loss_count = 0
    interval_m2t_microbatches = 0
    interval_m2t_active_tokens = 0
    interval_t2t_ce_sum = 0.0
    interval_t2t_loss_sum = 0.0
    interval_t2t_loss_count = 0
    interval_t2t_microbatches = 0
    interval_t2t_active_tokens = 0
    last_grad_norm = float(resume_metrics.get("grad_norm", 0.0)) if resume_metrics else 0.0
    last_grad_clipped = bool(resume_metrics.get("last_grad_clipped", False)) if resume_metrics else False
    last_grad_clip_ratio = float(resume_metrics.get("last_grad_clip_ratio", 1.0)) if resume_metrics else 1.0
    grad_clip_threshold = float(config.optim.grad_clip)
    grad_clip_enabled = grad_clip_threshold > 0
    grad_clip_event_count = int(resume_metrics.get("grad_clip_event_count", 0)) if resume_metrics else 0
    grad_clip_observed_steps = int(resume_metrics.get("grad_clip_observed_steps", 0)) if resume_metrics else 0
    last_active_tokens = int(resume_metrics.get("active_tokens", 0)) if resume_metrics else 0
    last_lossable_tokens = int(resume_metrics.get("lossable_tokens", 0)) if resume_metrics else 0
    last_valid_tokens = int(resume_metrics.get("valid_tokens", 0)) if resume_metrics else 0
    last_content_tokens = int(resume_metrics.get("content_tokens", 0)) if resume_metrics else 0
    last_content_bytes = int(resume_metrics.get("content_bytes", 0)) if resume_metrics else 0
    last_class_ce = float(resume_metrics.get("raw_class_ce", 0.0)) if resume_metrics else 0.0
    last_within_class_ce = float(resume_metrics.get("raw_within_class_ce", 0.0)) if resume_metrics else 0.0
    last_ar_unsmoothed_ce = float(resume_metrics.get("ar_unsmoothed_ce", last_loss_value)) if resume_metrics else 0.0
    last_second_moment_norm = float(resume_metrics.get("second_moment_norm", 0.0)) if resume_metrics else 0.0
    last_update_norm = float(resume_metrics.get("update_norm", 0.0)) if resume_metrics else 0.0
    last_update_grad_ratio = float(resume_metrics.get("update_grad_ratio", 0.0)) if resume_metrics else 0.0
    last_entropy_mean = float(resume_metrics.get("train_entropy_mean", 0.0)) if resume_metrics else 0.0
    last_top1_prob_mean = float(resume_metrics.get("train_top1_prob_mean", 0.0)) if resume_metrics else 0.0
    last_argmax_mode_frac = float(resume_metrics.get("train_argmax_mode_frac", 0.0)) if resume_metrics else 0.0
    last_top1_prob_hist_counts = [0 for _ in range(TOP1_PROB_HIST_BINS)]
    last_entropy_hist_counts = [0 for _ in range(LOGIT_SCALE_HIST_BINS)]
    last_logsumexp_hist_counts = [0 for _ in range(LOGIT_SCALE_HIST_BINS)]
    last_logit_norm_hist_counts = [0 for _ in range(LOGIT_SCALE_HIST_BINS)]
    last_logit_norm_mean = (
        float(resume_metrics.get("last_train_masked_logit_norm_mean", 0.0)) if resume_metrics else 0.0
    )
    last_logit_norm_min = float(resume_metrics.get("last_train_masked_logit_norm_min", 0.0)) if resume_metrics else 0.0
    last_logit_norm_max = float(resume_metrics.get("last_train_masked_logit_norm_max", 0.0)) if resume_metrics else 0.0
    last_logit_rms_mean = float(resume_metrics.get("last_train_masked_logit_rms_mean", 0.0)) if resume_metrics else 0.0
    last_logsumexp_mean = float(resume_metrics.get("last_train_masked_logsumexp_mean", 0.0)) if resume_metrics else 0.0
    last_logsumexp_min = float(resume_metrics.get("last_train_masked_logsumexp_min", 0.0)) if resume_metrics else 0.0
    last_logsumexp_max = float(resume_metrics.get("last_train_masked_logsumexp_max", 0.0)) if resume_metrics else 0.0
    last_soft_input_stats: dict[str, float | str] = {}
    last_uniform_estimator_stats: dict[str, float | int | str] = {}
    last_per_token_noise_level_hist_counts = [0 for _ in range(PER_TOKEN_NOISE_LEVEL_HIST_BINS)]
    train_per_token_noise_level_hist_counts = [0 for _ in range(PER_TOKEN_NOISE_LEVEL_HIST_BINS)]
    last_per_token_effective_masked_fraction_sum = 0.0
    last_per_token_effective_masked_fraction_count = 0
    last_per_token_effective_masked_fraction_min = float("inf")
    last_per_token_effective_masked_fraction_max = float("-inf")
    train_per_token_effective_masked_fraction_sum = 0.0
    train_per_token_effective_masked_fraction_count = 0
    train_per_token_effective_masked_fraction_min = float("inf")
    train_per_token_effective_masked_fraction_max = float("-inf")
    initial_parameter_norm_metrics = {
        f"initial_{key}": value for key, value in _parameter_norm_metrics(raw_model).items()
    }
    last_gradient_component_norms = {
        key: float(resume_metrics.get(key, 0.0)) for key in _gradient_component_norm_metrics(raw_model)
    }
    initial_gradient_metrics: dict[str, float | int] = {}
    last_update_component_norms: dict[str, float] = {
        "update_norm_embedding": 0.0,
        "update_norm_lm_head": 0.0,
        "update_grad_ratio_embedding": 0.0,
        "update_grad_ratio_lm_head": 0.0,
    }
    initial_update_metrics: dict[str, float | int] = {}
    initial_lm_head_row_norm_metrics: dict[str, float | int] = {}
    early_lm_head_row_norm_metrics: dict[str, float | int] = {}
    early_training_diagnostics: dict[str, float | int] = {}
    t0 = time.time()
    last_log_t = t0
    last_log_step = start_step
    gc_was_enabled = gc.isenabled()
    gc_was_already_frozen = gc.get_freeze_count() > 0
    resources.callback(_restore_gc_state, gc_was_enabled, not gc_was_already_frozen)
    gc.collect()
    if not gc_was_already_frozen:
        gc.freeze()
    gc.disable()

    train_log_path = os.path.join(config.train.run_dir, "train_log.jsonl")
    eval_log_path = os.path.join(config.train.run_dir, "eval_log.jsonl")
    residual_diagnostics_path = os.path.join(config.train.run_dir, "telemetry", "residual_diagnostics.jsonl")
    fixed_ce_trace_path = os.path.join(config.train.run_dir, "telemetry", "fixed_noise_ce_trace.jsonl")
    embedding_head_init_diagnostics_path = os.path.join(
        config.train.run_dir,
        "telemetry",
        "embedding_head_init_diagnostics.jsonl",
    )
    fixed_ce_trace_steps = (
        _fixed_ce_trace_steps(total_steps, warmup_steps)
        if config.eval.enabled and config.eval.fixed_ce_trace_enabled
        else []
    )
    fixed_ce_trace_records = 0
    last_fixed_ce_trace_record: dict[str, bool | float | int | list[float] | str] = {}
    early_residual_diagnostics: dict[str, float | int | str] = {}
    tensorboard_writer = _build_tensorboard_writer(config, rank)
    resources.callback(close_tensorboard_writer, tensorboard_writer)
    if tensorboard_writer is not None:
        print0(rank, f"tensorboard logs to {_resolve_tensorboard_dir(config)}")
    checkpoint_metrics: dict = {}

    def current_trainer_state() -> dict[str, object]:
        return {
            "version": RESUME_TRAINER_STATE_VERSION,
            "lineage": {
                "ar_boundary_counts": copy.deepcopy(ar_boundary_counts),
                "boundary_telemetry": copy.deepcopy(boundary_telemetry),
                "row_telemetry": copy.deepcopy(row_telemetry),
                "eos_supervision_train_counts": copy.deepcopy(eos_supervision_train_counts),
                "block_causal_visible_prefix_frac_sum": block_causal_visible_prefix_frac_sum,
                "block_causal_visible_prefix_count": block_causal_visible_prefix_count,
                "train_m2t_ce_sum": train_m2t_ce_sum,
                "train_m2t_loss_sum": train_m2t_loss_sum,
                "train_m2t_loss_count": train_m2t_loss_count,
                "train_m2t_microbatches": train_m2t_microbatches,
                "train_m2t_active_tokens": train_m2t_active_tokens,
                "train_t2t_ce_sum": train_t2t_ce_sum,
                "train_t2t_loss_sum": train_t2t_loss_sum,
                "train_t2t_loss_count": train_t2t_loss_count,
                "train_t2t_microbatches": train_t2t_microbatches,
                "train_t2t_active_tokens": train_t2t_active_tokens,
                "train_length_head_accumulator": copy.deepcopy(train_length_head_accumulator),
                "train_token_edit_reconstruction_accumulator": copy.deepcopy(
                    train_token_edit_reconstruction_accumulator
                ),
                "train_per_token_noise_level_hist_counts": list(train_per_token_noise_level_hist_counts),
                "train_per_token_effective_masked_fraction_sum": train_per_token_effective_masked_fraction_sum,
                "train_per_token_effective_masked_fraction_count": train_per_token_effective_masked_fraction_count,
                "train_per_token_effective_masked_fraction_min": train_per_token_effective_masked_fraction_min,
                "train_per_token_effective_masked_fraction_max": train_per_token_effective_masked_fraction_max,
                "grad_clip_event_count": grad_clip_event_count,
                "grad_clip_observed_steps": grad_clip_observed_steps,
            },
            "interval": {
                "smooth_loss": smooth_loss,
                "smooth_count": smooth_count,
                "m2t_ce_sum": interval_m2t_ce_sum,
                "m2t_loss_sum": interval_m2t_loss_sum,
                "m2t_loss_count": interval_m2t_loss_count,
                "m2t_microbatches": interval_m2t_microbatches,
                "m2t_active_tokens": interval_m2t_active_tokens,
                "t2t_ce_sum": interval_t2t_ce_sum,
                "t2t_loss_sum": interval_t2t_loss_sum,
                "t2t_loss_count": interval_t2t_loss_count,
                "t2t_microbatches": interval_t2t_microbatches,
                "t2t_active_tokens": interval_t2t_active_tokens,
            },
            "last": {
                "loss_value": last_loss_value,
                "main_ce": last_main_ce,
                "main_loss": last_main_loss,
                "confidence_penalty": last_confidence_penalty,
                "confidence_penalty_loss": last_confidence_penalty_loss,
                "z_loss": last_z_loss,
                "z_loss_loss": last_z_loss_loss,
                "eos_hazard_calibration_loss": last_eos_hazard_calibration_loss,
                "eos_hazard_loss": last_eos_hazard_loss,
                "eos_hazard_accumulator": copy.deepcopy(last_eos_hazard_accumulator),
                "length_head_accumulator": copy.deepcopy(last_length_head_accumulator),
                "position_decile_ce_accumulator": copy.deepcopy(last_position_decile_ce_accumulator),
                "token_edit_reconstruction_accumulator": copy.deepcopy(last_token_edit_reconstruction_accumulator),
                "m2t_ce": last_m2t_ce,
                "m2t_loss": last_m2t_loss,
                "m2t_microbatches": last_m2t_microbatches,
                "m2t_active_tokens": last_m2t_active_tokens,
                "t2t_ce": last_t2t_ce,
                "t2t_loss": last_t2t_loss,
                "t2t_microbatches": last_t2t_microbatches,
                "t2t_active_tokens": last_t2t_active_tokens,
                "grad_norm": last_grad_norm,
                "grad_clipped": last_grad_clipped,
                "grad_clip_ratio": last_grad_clip_ratio,
                "active_tokens": last_active_tokens,
                "lossable_tokens": last_lossable_tokens,
                "valid_tokens": last_valid_tokens,
                "content_tokens": last_content_tokens,
                "content_bytes": last_content_bytes,
                "class_ce": last_class_ce,
                "within_class_ce": last_within_class_ce,
                "ar_unsmoothed_ce": last_ar_unsmoothed_ce,
                "second_moment_norm": last_second_moment_norm,
                "update_norm": last_update_norm,
                "update_grad_ratio": last_update_grad_ratio,
                "entropy_mean": last_entropy_mean,
                "top1_prob_mean": last_top1_prob_mean,
                "argmax_mode_frac": last_argmax_mode_frac,
                "top1_prob_hist_counts": list(last_top1_prob_hist_counts),
                "entropy_hist_counts": list(last_entropy_hist_counts),
                "logsumexp_hist_counts": list(last_logsumexp_hist_counts),
                "logit_norm_hist_counts": list(last_logit_norm_hist_counts),
                "logit_norm_mean": last_logit_norm_mean,
                "logit_norm_min": last_logit_norm_min,
                "logit_norm_max": last_logit_norm_max,
                "logit_rms_mean": last_logit_rms_mean,
                "logsumexp_mean": last_logsumexp_mean,
                "logsumexp_min": last_logsumexp_min,
                "logsumexp_max": last_logsumexp_max,
                "soft_input_stats": copy.deepcopy(last_soft_input_stats),
                "uniform_estimator_stats": copy.deepcopy(last_uniform_estimator_stats),
                "per_token_noise_level_hist_counts": list(last_per_token_noise_level_hist_counts),
                "per_token_effective_masked_fraction_sum": last_per_token_effective_masked_fraction_sum,
                "per_token_effective_masked_fraction_count": last_per_token_effective_masked_fraction_count,
                "per_token_effective_masked_fraction_min": last_per_token_effective_masked_fraction_min,
                "per_token_effective_masked_fraction_max": last_per_token_effective_masked_fraction_max,
                "gradient_component_norms": copy.deepcopy(last_gradient_component_norms),
                "update_component_norms": copy.deepcopy(last_update_component_norms),
            },
            "diagnostics": {
                "initial_parameter_norm_metrics": copy.deepcopy(initial_parameter_norm_metrics),
                "initial_gradient_metrics": copy.deepcopy(initial_gradient_metrics),
                "initial_update_metrics": copy.deepcopy(initial_update_metrics),
                "initial_lm_head_row_norm_metrics": copy.deepcopy(initial_lm_head_row_norm_metrics),
                "early_lm_head_row_norm_metrics": copy.deepcopy(early_lm_head_row_norm_metrics),
                "early_training_diagnostics": copy.deepcopy(early_training_diagnostics),
                "early_residual_diagnostics": copy.deepcopy(early_residual_diagnostics),
                "fixed_ce_trace_records": fixed_ce_trace_records,
                "last_fixed_ce_trace_record": copy.deepcopy(last_fixed_ce_trace_record),
            },
        }

    if resume_state is not None and resume_state.get("local_trainer_state") is not None:
        progress = resume_state["local_trainer_state"]
        if not isinstance(progress, dict) or progress.get("version") != RESUME_TRAINER_STATE_VERSION:
            raise RuntimeError("resume contract contains malformed trainer progress")
        lineage = progress.get("lineage")
        interval = progress.get("interval")
        last = progress.get("last")
        diagnostics = progress.get("diagnostics")
        if not all(isinstance(group, dict) for group in (lineage, interval, last, diagnostics)):
            raise RuntimeError("resume trainer progress is missing a required state group")
        ar_boundary_counts = copy.deepcopy(lineage["ar_boundary_counts"])
        boundary_telemetry = copy.deepcopy(lineage["boundary_telemetry"])
        row_telemetry = copy.deepcopy(lineage["row_telemetry"])
        eos_supervision_train_counts = copy.deepcopy(lineage["eos_supervision_train_counts"])
        block_causal_visible_prefix_frac_sum = float(lineage["block_causal_visible_prefix_frac_sum"])
        block_causal_visible_prefix_count = int(lineage["block_causal_visible_prefix_count"])
        train_m2t_ce_sum = float(lineage["train_m2t_ce_sum"])
        train_m2t_loss_sum = float(lineage["train_m2t_loss_sum"])
        train_m2t_loss_count = int(lineage["train_m2t_loss_count"])
        train_m2t_microbatches = int(lineage["train_m2t_microbatches"])
        train_m2t_active_tokens = int(lineage["train_m2t_active_tokens"])
        train_t2t_ce_sum = float(lineage["train_t2t_ce_sum"])
        train_t2t_loss_sum = float(lineage["train_t2t_loss_sum"])
        train_t2t_loss_count = int(lineage["train_t2t_loss_count"])
        train_t2t_microbatches = int(lineage["train_t2t_microbatches"])
        train_t2t_active_tokens = int(lineage["train_t2t_active_tokens"])
        train_length_head_accumulator = copy.deepcopy(lineage["train_length_head_accumulator"])
        train_token_edit_reconstruction_accumulator = copy.deepcopy(
            lineage["train_token_edit_reconstruction_accumulator"]
        )
        train_per_token_noise_level_hist_counts = list(lineage["train_per_token_noise_level_hist_counts"])
        train_per_token_effective_masked_fraction_sum = float(lineage["train_per_token_effective_masked_fraction_sum"])
        train_per_token_effective_masked_fraction_count = int(
            lineage["train_per_token_effective_masked_fraction_count"]
        )
        train_per_token_effective_masked_fraction_min = float(lineage["train_per_token_effective_masked_fraction_min"])
        train_per_token_effective_masked_fraction_max = float(lineage["train_per_token_effective_masked_fraction_max"])
        grad_clip_event_count = int(lineage["grad_clip_event_count"])
        grad_clip_observed_steps = int(lineage["grad_clip_observed_steps"])
        smooth_loss = float(interval["smooth_loss"])
        smooth_count = int(interval["smooth_count"])
        interval_m2t_ce_sum = float(interval["m2t_ce_sum"])
        interval_m2t_loss_sum = float(interval["m2t_loss_sum"])
        interval_m2t_loss_count = int(interval["m2t_loss_count"])
        interval_m2t_microbatches = int(interval["m2t_microbatches"])
        interval_m2t_active_tokens = int(interval["m2t_active_tokens"])
        interval_t2t_ce_sum = float(interval["t2t_ce_sum"])
        interval_t2t_loss_sum = float(interval["t2t_loss_sum"])
        interval_t2t_loss_count = int(interval["t2t_loss_count"])
        interval_t2t_microbatches = int(interval["t2t_microbatches"])
        interval_t2t_active_tokens = int(interval["t2t_active_tokens"])
        last_loss_value = float(last["loss_value"])
        last_main_ce = float(last["main_ce"])
        last_main_loss = float(last["main_loss"])
        last_confidence_penalty = float(last["confidence_penalty"])
        last_confidence_penalty_loss = float(last["confidence_penalty_loss"])
        last_z_loss = float(last["z_loss"])
        last_z_loss_loss = float(last["z_loss_loss"])
        last_eos_hazard_calibration_loss = float(last["eos_hazard_calibration_loss"])
        last_eos_hazard_loss = float(last["eos_hazard_loss"])
        last_eos_hazard_accumulator = copy.deepcopy(last["eos_hazard_accumulator"])
        last_length_head_accumulator = copy.deepcopy(last["length_head_accumulator"])
        last_position_decile_ce_accumulator = copy.deepcopy(last["position_decile_ce_accumulator"])
        last_token_edit_reconstruction_accumulator = copy.deepcopy(last["token_edit_reconstruction_accumulator"])
        last_m2t_ce = float(last["m2t_ce"])
        last_m2t_loss = float(last["m2t_loss"])
        last_m2t_microbatches = int(last["m2t_microbatches"])
        last_m2t_active_tokens = int(last["m2t_active_tokens"])
        last_t2t_ce = float(last["t2t_ce"])
        last_t2t_loss = float(last["t2t_loss"])
        last_t2t_microbatches = int(last["t2t_microbatches"])
        last_t2t_active_tokens = int(last["t2t_active_tokens"])
        last_grad_norm = float(last["grad_norm"])
        last_grad_clipped = bool(last["grad_clipped"])
        last_grad_clip_ratio = float(last["grad_clip_ratio"])
        last_active_tokens = int(last["active_tokens"])
        last_lossable_tokens = int(last["lossable_tokens"])
        last_valid_tokens = int(last["valid_tokens"])
        last_content_tokens = int(last["content_tokens"])
        last_content_bytes = int(last["content_bytes"])
        last_class_ce = float(last["class_ce"])
        last_within_class_ce = float(last["within_class_ce"])
        last_ar_unsmoothed_ce = float(last["ar_unsmoothed_ce"])
        last_second_moment_norm = float(last["second_moment_norm"])
        last_update_norm = float(last["update_norm"])
        last_update_grad_ratio = float(last["update_grad_ratio"])
        last_entropy_mean = float(last["entropy_mean"])
        last_top1_prob_mean = float(last["top1_prob_mean"])
        last_argmax_mode_frac = float(last["argmax_mode_frac"])
        last_top1_prob_hist_counts = list(last["top1_prob_hist_counts"])
        last_entropy_hist_counts = list(last["entropy_hist_counts"])
        last_logsumexp_hist_counts = list(last["logsumexp_hist_counts"])
        last_logit_norm_hist_counts = list(last["logit_norm_hist_counts"])
        last_logit_norm_mean = float(last["logit_norm_mean"])
        last_logit_norm_min = float(last["logit_norm_min"])
        last_logit_norm_max = float(last["logit_norm_max"])
        last_logit_rms_mean = float(last["logit_rms_mean"])
        last_logsumexp_mean = float(last["logsumexp_mean"])
        last_logsumexp_min = float(last["logsumexp_min"])
        last_logsumexp_max = float(last["logsumexp_max"])
        last_soft_input_stats = copy.deepcopy(last["soft_input_stats"])
        last_uniform_estimator_stats = copy.deepcopy(last["uniform_estimator_stats"])
        last_per_token_noise_level_hist_counts = list(last["per_token_noise_level_hist_counts"])
        last_per_token_effective_masked_fraction_sum = float(last["per_token_effective_masked_fraction_sum"])
        last_per_token_effective_masked_fraction_count = int(last["per_token_effective_masked_fraction_count"])
        last_per_token_effective_masked_fraction_min = float(last["per_token_effective_masked_fraction_min"])
        last_per_token_effective_masked_fraction_max = float(last["per_token_effective_masked_fraction_max"])
        last_gradient_component_norms = copy.deepcopy(last["gradient_component_norms"])
        last_update_component_norms = copy.deepcopy(last["update_component_norms"])
        initial_parameter_norm_metrics = copy.deepcopy(diagnostics["initial_parameter_norm_metrics"])
        initial_gradient_metrics = copy.deepcopy(diagnostics["initial_gradient_metrics"])
        initial_update_metrics = copy.deepcopy(diagnostics["initial_update_metrics"])
        initial_lm_head_row_norm_metrics = copy.deepcopy(diagnostics["initial_lm_head_row_norm_metrics"])
        early_lm_head_row_norm_metrics = copy.deepcopy(diagnostics["early_lm_head_row_norm_metrics"])
        early_training_diagnostics = copy.deepcopy(diagnostics["early_training_diagnostics"])
        early_residual_diagnostics = copy.deepcopy(diagnostics["early_residual_diagnostics"])
        fixed_ce_trace_records = int(diagnostics["fixed_ce_trace_records"])
        last_fixed_ce_trace_record = copy.deepcopy(diagnostics["last_fixed_ce_trace_record"])
    all_logits_indices = torch.arange(
        int(config.train.device_batch_size) * config.model.max_seq_len,
        device=device,
        dtype=torch.long,
    )

    if record_embedding_head_init_diagnostics and rank == 0 and start_step == 0:
        saved_state = _rng_state(config)
        try:
            initial_lm_head_row_norm_metrics = _prefixed_metric_keys(
                "initial_",
                _lm_head_row_norm_metrics(
                    raw_model,
                    tokenizer,
                    config.data,
                    device,
                    int(config.train.device_batch_size),
                    int(config.model.max_seq_len),
                    batches=10,
                ),
            )
            os.makedirs(os.path.dirname(embedding_head_init_diagnostics_path), exist_ok=True)
            append_jsonl(
                embedding_head_init_diagnostics_path,
                {
                    "event": "initial_lm_head_row_norms",
                    "step": int(start_step),
                    **_embedding_head_init_metrics(config),
                    **initial_lm_head_row_norm_metrics,
                },
            )
        finally:
            _restore_rng_state(saved_state, config)

    for step in range(start_step, total_steps):
        lrm = get_lr_multiplier(
            step,
            total_steps,
            warmup_steps,
            config.optim.warmdown_ratio,
            config.optim.final_lr_frac,
        )
        _set_optimizer_lrs(optimizers, lrm)
        _zero_grad(optimizers)
        record_early_residual_diagnostics = (
            step == 0 and config.model.residual_output_projection_init == "scaled_normal"
        )
        residual_diagnostic_batch: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None

        step_loss_sum = 0.0
        step_main_ce_sum = 0.0
        step_main_loss_sum = 0.0
        step_confidence_penalty_sum = 0.0
        step_confidence_penalty_loss_sum = 0.0
        step_z_loss_sum = 0.0
        step_z_loss_loss_sum = 0.0
        step_eos_hazard_calibration_loss_sum = 0.0
        step_eos_hazard_loss_sum = 0.0
        step_eos_hazard_accumulator = _new_eos_hazard_accumulator()
        step_length_head_accumulator = _new_length_head_accumulator(int(config.model.max_seq_len))
        step_position_decile_ce_accumulator = _new_position_decile_ce_accumulator()
        step_token_edit_reconstruction_accumulator = _new_token_edit_reconstruction_accumulator()
        step_class_ce_sum = 0.0
        step_within_class_ce_sum = 0.0
        step_ar_unsmoothed_ce_sum = 0.0
        step_aux_ce_sums = [0.0 for _ in model_config.auxiliary_head_layers] if return_aux_logits else []
        step_loss_count = 0
        step_m2t_ce_sum = 0.0
        step_m2t_loss_sum = 0.0
        step_m2t_loss_count = 0
        step_m2t_microbatches = 0
        step_m2t_active_tokens = 0
        step_t2t_ce_sum = 0.0
        step_t2t_loss_sum = 0.0
        step_t2t_loss_count = 0
        step_t2t_microbatches = 0
        step_t2t_active_tokens = 0
        step_processed_tokens = 0
        step_active_tokens = 0
        step_lossable_tokens = 0
        step_valid_tokens = 0
        step_content_tokens = 0
        step_content_bytes = 0
        step_t_sum = 0.0
        step_t_min = float("inf")
        step_t_max = float("-inf")
        step_t_count = 0
        step_schedule_multiplier_sum = 0.0
        step_loss_normalization_scale_sum = 0.0
        step_entropy_sum = 0.0
        step_top1_prob_sum = 0.0
        step_argmax_mode_sum = 0.0
        step_top1_prob_hist_counts = [0 for _ in range(TOP1_PROB_HIST_BINS)]
        step_entropy_hist_counts = [0 for _ in range(LOGIT_SCALE_HIST_BINS)]
        step_logsumexp_hist_counts = [0 for _ in range(LOGIT_SCALE_HIST_BINS)]
        step_logit_norm_hist_counts = [0 for _ in range(LOGIT_SCALE_HIST_BINS)]
        step_logit_norm_sum = 0.0
        step_logit_norm_min = float("inf")
        step_logit_norm_max = float("-inf")
        step_logit_rms_sum = 0.0
        step_logsumexp_sum = 0.0
        step_logsumexp_min = float("inf")
        step_logsumexp_max = float("-inf")
        step_visible_prefix_frac_sum = 0.0
        step_visible_prefix_count = 0
        step_soft_input_stat_sums: dict[str, float] = {}
        step_soft_input_stat_count = 0
        step_uniform_estimator_stat_sums: dict[str, float] = {}
        step_uniform_estimator_stat_count = 0
        step_uniform_estimator_label = config.train.uniform_loss_estimator
        step_per_token_noise_level_hist_counts = [0 for _ in range(PER_TOKEN_NOISE_LEVEL_HIST_BINS)]
        step_per_token_effective_masked_fraction_sum = 0.0
        step_per_token_effective_masked_fraction_count = 0
        step_per_token_effective_masked_fraction_min = float("inf")
        step_per_token_effective_masked_fraction_max = float("-inf")
        step_soft_input_phase = _soft_input_curriculum_phase(
            config,
            _soft_input_curriculum_mix(config, step, total_steps),
        )
        for micro in range(grad_accum_steps):
            step_processed_tokens += int(x_0.numel())
            if _is_causal(config):
                if record_early_residual_diagnostics and residual_diagnostic_batch is None:
                    diagnostic_attention_mask = _batch_model_attention_mask(batch, attention_mask)
                    diagnostic_position_ids = (
                        batch.position_ids.detach() if getattr(batch, "position_ids", None) is not None else None
                    )
                    residual_diagnostic_batch = (
                        x_0.detach(),
                        diagnostic_attention_mask.detach() if diagnostic_attention_mask is not None else None,
                        diagnostic_position_ids,
                    )
                record_ar_boundary_batch(
                    x_0,
                    loss_mask,
                    attention_mask,
                    getattr(batch, "position_ids", None),
                    getattr(batch, "document_boundary_mask", None),
                    getattr(batch, "document_start_mask", None),
                )
                valid_tokens = int(attention_mask.sum().item())
                lossable_tokens = int(loss_mask.sum().item())
                content_tokens, content_bytes = content_byte_counts(x_0, attention_mask)
                step_valid_tokens += valid_tokens
                step_lossable_tokens += lossable_tokens
                step_content_tokens += content_tokens
                step_content_bytes += content_bytes
                active_mask = loss_mask & attention_mask
                step_active_tokens += int(active_mask.sum().item())
                sync_context = model.no_sync if (is_ddp and micro < grad_accum_steps - 1) else nullcontext
                with sync_context():
                    with autocast_ctx:
                        model_kwargs = {}
                        if getattr(batch, "position_ids", None) is not None:
                            model_kwargs["position_ids"] = batch.position_ids
                        model_kwargs["attention_mask"] = _batch_model_attention_mask(batch, attention_mask)
                        logits = model(x_0, **model_kwargs)
                        logits_pred = logits[:, :-1, :].contiguous()
                        targets = x_0[:, 1:].contiguous()
                        target_mask = loss_mask[:, 1:] & attention_mask[:, 1:]
                        label_smoothing = _causal_ar_label_smoothing(config)
                        ce = F.cross_entropy(
                            logits_pred.view(-1, logits_pred.size(-1)),
                            targets.view(-1),
                            reduction="none",
                            label_smoothing=label_smoothing,
                        ).view_as(targets)
                        if label_smoothing > 0:
                            unsmoothed_ce = F.cross_entropy(
                                logits_pred.view(-1, logits_pred.size(-1)),
                                targets.view(-1),
                                reduction="none",
                            ).view_as(targets)
                        else:
                            unsmoothed_ce = ce
                        if target_mask.any():
                            raw_main_ce = ce[target_mask].mean()
                            raw_unsmoothed_ce = unsmoothed_ce[target_mask].mean()
                        else:
                            raw_main_ce = torch.tensor(0.0, device=device)
                            raw_unsmoothed_ce = torch.tensor(0.0, device=device)
                        if config.train.entropy_bonus_beta > 0:
                            pred_probs = F.softmax(logits_pred, dim=-1).float()
                            pred_log_probs = F.log_softmax(logits_pred, dim=-1).float()
                            entropy_values = -(pred_probs * pred_log_probs).sum(dim=-1)
                            if target_mask.any():
                                raw_entropy = entropy_values[target_mask].mean()
                                raw_max_prob = pred_probs.max(dim=-1).values[target_mask].mean()
                                argmax_ids = pred_probs.argmax(dim=-1)[target_mask]
                                if argmax_ids.numel() > 0:
                                    mode_val = torch.mode(argmax_ids).values
                                    raw_argmax_mode_frac = (argmax_ids == mode_val).float().mean()
                                else:
                                    raw_argmax_mode_frac = torch.tensor(0.0, device=device)
                            else:
                                raw_entropy = torch.tensor(0.0, device=device)
                                raw_max_prob = torch.tensor(0.0, device=device)
                                raw_argmax_mode_frac = torch.tensor(0.0, device=device)
                            entropy_bonus = -config.train.entropy_bonus_beta * raw_entropy
                        else:
                            entropy_bonus = torch.tensor(0.0, device=device)
                            raw_entropy = torch.tensor(0.0, device=device)
                            raw_max_prob = torch.tensor(0.0, device=device)
                            raw_argmax_mode_frac = torch.tensor(0.0, device=device)
                        loss = raw_main_ce + entropy_bonus
                        loss_normalization_scale = 1.0
                        main_loss_for_log = raw_main_ce
                    loss_value = float(loss.detach().item())
                    if not math.isfinite(loss_value):
                        raise RuntimeError(f"non-finite loss at step {step}, micro {micro}: {loss_value}")
                    step_loss_sum += loss_value
                    step_main_ce_sum += float(raw_main_ce.detach().item())
                    step_main_loss_sum += float(main_loss_for_log.detach().item())
                    step_ar_unsmoothed_ce_sum += float(raw_unsmoothed_ce.detach().item())
                    step_entropy_sum += float(raw_entropy.detach().item())
                    step_top1_prob_sum += float(raw_max_prob.detach().item())
                    step_argmax_mode_sum += float(raw_argmax_mode_frac.detach().item())
                    step_loss_normalization_scale_sum += 1.0
                    step_loss_count += 1
                    scaler.scale(loss / grad_accum_steps).backward()
                batch = next(train_loader)
                x_0 = batch.x_0
                loss_mask = batch.loss_mask
                attention_mask = batch.attention_mask
                epoch = batch.epoch
                continue
            global_microbatch_idx = step * grad_accum_steps + micro
            token_edit_branch = _token_edit_branch_enabled_for_microbatch(config, global_microbatch_idx)
            if token_edit_branch:
                if token_edit_class_ids is None:
                    raise RuntimeError("token-edit corruption requires frequency decile token classes")
                z_t, w, t_val, noise_level = _corrupt_token_edit_training_batch(
                    config,
                    x_0,
                    loss_mask,
                    int(model_config.vocab_size),
                    int(model_config.mask_token_id),
                )
            else:
                z_t, w, t_val, noise_level = _corrupt_training_batch(config, noise, x_0, loss_mask)
            soft_input_embeddings = None
            if config.noise.soft_input_curriculum:
                soft_input_embeddings, soft_input_stats = _build_uniform_soft_input_embeddings(
                    raw_model,
                    config,
                    x_0,
                    z_t,
                    loss_mask,
                    t_val,
                    step,
                    total_steps,
                )
                step_soft_input_phase = str(soft_input_stats["soft_input_curriculum_phase"])
                for key, value in soft_input_stats.items():
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        step_soft_input_stat_sums[key] = step_soft_input_stat_sums.get(key, 0.0) + float(value)
                step_soft_input_stat_count += 1
            if record_early_residual_diagnostics and residual_diagnostic_batch is None:
                residual_diagnostic_batch = (
                    z_t.detach(),
                    _batch_model_attention_mask(batch, attention_mask).detach(),
                    batch.position_ids.detach() if getattr(batch, "position_ids", None) is not None else None,
                )
            forward_attention_mask = _batch_model_attention_mask(batch, attention_mask)
            if _uses_block_causal_pilot(config):
                block_size = int(config.model.block_causal_pilot_size)
                z_t, w, forward_attention_mask, _in_current_block, visible_prefix_frac = (
                    _apply_block_causal_pilot_training(config, x_0, z_t, w, attention_mask, loss_mask, block_size)
                )
                step_visible_prefix_frac_sum += float(visible_prefix_frac.sum().item())
                step_visible_prefix_count += x_0.shape[0]
            record_ar_boundary_batch(
                x_0,
                loss_mask,
                attention_mask,
                getattr(batch, "position_ids", None),
                getattr(batch, "document_boundary_mask", None),
                getattr(batch, "document_start_mask", None),
            )
            active = (w > 0) & loss_mask
            valid_tokens = int(attention_mask.sum().item())
            lossable_tokens = int(loss_mask.sum().item())
            content_tokens, content_bytes = content_byte_counts(x_0, attention_mask)
            step_valid_tokens += valid_tokens
            step_lossable_tokens += lossable_tokens
            step_content_tokens += content_tokens
            step_content_bytes += content_bytes
            if eos_supervision_audit_enabled:
                _update_eos_supervision_batch_counts(
                    eos_supervision_train_counts,
                    x_0,
                    loss_mask,
                    attention_mask,
                    special_token_ids,
                    eos_token_id,
                    active,
                )
            if _uses_per_token_noise_levels(config):
                if token_edit_branch:
                    raise ValueError("per-token noise-level telemetry is only defined for MDLM M2T corruption")
                if not isinstance(t_val, torch.Tensor) or t_val.shape != x_0.shape:
                    raise ValueError(
                        "noise.per_token_noise_levels requires token-shaped t values from training corruption"
                    )
                lossable_t_values = t_val[loss_mask].detach().float()
                _accumulate_value_histogram(
                    step_per_token_noise_level_hist_counts,
                    lossable_t_values,
                    0.0,
                    1.0,
                )
                effective_masked_fraction = int(active.sum().item()) / max(1, lossable_tokens)
                step_per_token_effective_masked_fraction_sum += effective_masked_fraction
                step_per_token_effective_masked_fraction_count += 1
                step_per_token_effective_masked_fraction_min = min(
                    step_per_token_effective_masked_fraction_min,
                    effective_masked_fraction,
                )
                step_per_token_effective_masked_fraction_max = max(
                    step_per_token_effective_masked_fraction_max,
                    effective_masked_fraction,
                )
            if bool(active.all().item()):
                target_indices = all_logits_indices
            else:
                target_indices = active.reshape(-1).nonzero(as_tuple=True)[0]
            if target_indices.numel() == 0:
                batch = next(train_loader)
                x_0 = batch.x_0
                loss_mask = batch.loss_mask
                attention_mask = batch.attention_mask
                epoch = batch.epoch
                continue
            logits_indices = _bd3lm_noised_half_indices(active) if _uses_bd3lm(config) else target_indices
            targets = _training_targets_from_indices(x_0, target_indices)
            step_active_tokens += int(target_indices.numel())
            sync_context = model.no_sync if (is_ddp and micro < grad_accum_steps - 1) else nullcontext
            if _uses_uniform_low_variance_estimator(config):
                if isinstance(t_val, torch.Tensor) or t_val is None:
                    raise ValueError("two-sample uniform estimator requires scalar uniform t")
                if use_hierarchical_output or return_aux_logits:
                    raise ValueError("two-sample uniform estimator requires flat logits without auxiliary heads")
                with sync_context():
                    with autocast_ctx:
                        (
                            loss,
                            raw_main_ce,
                            main_loss_for_log,
                            loss_normalization_scale,
                            uniform_estimator_stats,
                        ) = _uniform_two_sample_mean_loss(
                            model,
                            config,
                            noise,
                            x_0,
                            z_t,
                            loss_mask,
                            forward_attention_mask,
                            target_indices,
                            float(t_val),
                            noise_level,
                            lossable_tokens,
                        )
                    loss_value = float(loss.detach().item())
                    if not math.isfinite(loss_value):
                        raise RuntimeError(f"non-finite loss at step {step}, micro {micro}: {loss_value}")
                    step_loss_sum += loss_value
                    step_main_ce_sum += float(raw_main_ce.detach().item())
                    step_main_loss_sum += float(main_loss_for_log.detach().item())
                    step_loss_normalization_scale_sum += float(loss_normalization_scale)
                    for key, value in uniform_estimator_stats.items():
                        if isinstance(value, int | float) and not isinstance(value, bool):
                            step_uniform_estimator_stat_sums[key] = step_uniform_estimator_stat_sums.get(
                                key, 0.0
                            ) + float(value)
                        elif key == "uniform_loss_estimator":
                            step_uniform_estimator_label = str(value)
                    step_uniform_estimator_stat_count += 1
                    if t_val is not None:
                        step_t_sum += float(t_val)
                        step_t_min = min(step_t_min, float(t_val))
                        step_t_max = max(step_t_max, float(t_val))
                        step_t_count += 1
                        if config.noise.schedule_weight:
                            step_schedule_multiplier_sum += 1.0 / max(float(t_val), 1e-8)
                    step_loss_count += 1
                    scaler.scale(loss / grad_accum_steps).backward()
                batch = next(train_loader)
                x_0 = batch.x_0
                loss_mask = batch.loss_mask
                attention_mask = batch.attention_mask
                epoch = batch.epoch
                continue
            with sync_context():
                with autocast_ctx:
                    target_lengths = _target_lengths_for_conditioning(
                        config,
                        x_0,
                        attention_mask,
                        eos_token_id,
                    )
                    model_out = _forward_training_model(
                        model,
                        config,
                        z_t,
                        x_0,
                        forward_attention_mask,
                        logits_indices=logits_indices,
                        return_aux_logits=return_aux_logits,
                        noise_level=noise_level,
                        return_hierarchical_logits=use_hierarchical_output,
                        return_length_logits=_uses_length_head(config),
                        input_embeddings=soft_input_embeddings,
                        position_ids=getattr(batch, "position_ids", None),
                        target_lengths=target_lengths,
                    )
                    entropy_values = None
                    raw_entropy = torch.tensor(0.0, device=device)
                    raw_max_prob = torch.tensor(0.0, device=device)
                    raw_argmax_mode_frac = torch.tensor(0.0, device=device)
                    z_loss_values = None
                    logsumexp_values = None
                    logit_norms = None
                    logit_rms = None
                    length_logits = None
                    if use_hierarchical_output:
                        if return_aux_logits and _uses_length_head(config):
                            class_logits, token_logits, aux_logits, length_logits = model_out
                        elif return_aux_logits:
                            class_logits, token_logits, aux_logits = model_out
                        elif _uses_length_head(config):
                            class_logits, token_logits, length_logits = model_out
                            aux_logits = []
                        else:
                            class_logits, token_logits = model_out
                            aux_logits = []
                        main_ce_values, class_ce_values, within_class_ce_values = raw_model.hierarchical_ce(
                            class_logits,
                            token_logits,
                            targets,
                        )
                    else:
                        if return_aux_logits and _uses_length_head(config):
                            logits, aux_logits, length_logits = model_out
                        elif return_aux_logits:
                            logits, aux_logits = model_out
                        elif _uses_length_head(config):
                            logits, length_logits = model_out
                            aux_logits = []
                        else:
                            logits, aux_logits = model_out, []
                        main_ce_values = F.cross_entropy(logits, targets, reduction="none")
                        class_ce_values = None
                        within_class_ce_values = None
                        (
                            entropy_values,
                            max_probs,
                            raw_entropy,
                            raw_max_prob,
                            raw_argmax_mode_frac,
                        ) = _logit_confidence_statistics(logits)
                        _accumulate_probability_histogram(step_top1_prob_hist_counts, max_probs)
                        _accumulate_value_histogram(step_entropy_hist_counts, entropy_values, 0.0, 16.0)
                        if config.train.z_loss_weight > 0 or record_embedding_head_init_diagnostics:
                            z_loss_values, logsumexp_values, logit_norms, logit_rms = _logit_scale_statistics(logits)
                            _accumulate_value_histogram(step_logsumexp_hist_counts, logsumexp_values, 0.0, 64.0)
                            _accumulate_value_histogram(step_logit_norm_hist_counts, logit_norms, 0.0, 512.0)
                        else:
                            z_loss_values = None
                            logsumexp_values = None
                            logit_norms = None
                            logit_rms = None
                    raw_confidence_penalty = (
                        -raw_entropy if config.train.confidence_penalty_weight > 0 else torch.tensor(0.0, device=device)
                    )
                    confidence_penalty_loss_for_log = torch.tensor(0.0, device=device)
                    raw_z_loss = torch.tensor(0.0, device=device)
                    z_loss_for_log = torch.tensor(0.0, device=device)
                    if config.train.z_loss_weight > 0:
                        if z_loss_values is None:
                            raise RuntimeError("z-loss requires flat logits")
                        raw_z_loss = z_loss_values.mean()
                    raw_eos_hazard_calibration_loss = torch.tensor(0.0, device=device)
                    eos_hazard_loss_for_log = torch.tensor(0.0, device=device)
                    eos_hazard_stats = None
                    if config.train.eos_hazard_loss_weight > 0:
                        if use_hierarchical_output:
                            raise RuntimeError("EOS hazard calibration requires flat logits")
                        raw_eos_hazard_calibration_loss, eos_hazard_stats = _eos_hazard_calibration_loss(
                            logits,
                            targets,
                            target_indices,
                            attention_mask,
                            eos_token_id,
                        )
                    raw_length_head_ce = torch.tensor(0.0, device=device)
                    length_head_loss_for_log = torch.tensor(0.0, device=device)
                    length_head_stats = None
                    if config.train.length_head_loss_weight > 0:
                        if length_logits is None:
                            raise RuntimeError("length-head training requires length logits")
                        raw_length_head_ce, length_head_stats = _length_head_loss_and_stats(
                            length_logits,
                            x_0,
                            attention_mask,
                            eos_token_id,
                        )
                        length_head_loss_for_log = float(config.train.length_head_loss_weight) * raw_length_head_ce
                    aux_ce_values = [F.cross_entropy(aux, targets, reduction="none") for aux in aux_logits]
                    raw_main_ce = main_ce_values.mean()
                    raw_class_ce = class_ce_values.mean() if class_ce_values is not None else None
                    raw_within_class_ce = within_class_ce_values.mean() if within_class_ce_values is not None else None
                    raw_aux_ces = [aux_ce.mean() for aux_ce in aux_ce_values]
                    position_decile_stats = None
                    if isinstance(t_val, torch.Tensor):
                        if config.noise.type not in {"mdlm", "bd3lm"}:
                            raise ValueError("tensor t loss path is only supported for MDLM-style objectives")
                        if config.noise.schedule_weight:
                            token_weights = 1.0 / _per_target_t_values(t_val, x_0, target_indices).clamp_min(1e-8)
                        else:
                            token_weights = torch.ones_like(main_ce_values, dtype=torch.float32)
                        loss_denominator = max(1, lossable_tokens)
                        if config.train.position_decile_balanced_loss:
                            main_loss, position_decile_stats = _position_decile_balanced_ce_loss(
                                main_ce_values,
                                target_indices,
                                attention_mask,
                                loss_mask,
                                token_weights,
                            )
                            aux_losses = [
                                _position_decile_balanced_ce_loss(
                                    aux_ce,
                                    target_indices,
                                    attention_mask,
                                    loss_mask,
                                    token_weights,
                                )[0]
                                for aux_ce in aux_ce_values
                            ]
                            loss_normalization_scale = float(token_weights.detach().mean().item())
                        else:
                            main_loss = (main_ce_values.float() * token_weights).sum() / loss_denominator
                            aux_losses = [
                                (aux_ce.float() * token_weights).sum() / loss_denominator for aux_ce in aux_ce_values
                            ]
                            loss_normalization_scale = float(token_weights.detach().sum().item()) / loss_denominator
                        loss = main_loss
                        main_loss_for_log = main_loss
                        for aux_loss in aux_losses:
                            loss = loss + config.train.auxiliary_loss_weight * aux_loss
                        if config.train.entropy_bonus_beta > 0 and entropy_values is not None:
                            entropy_loss = (
                                -config.train.entropy_bonus_beta * entropy_values.float() * token_weights
                            ).sum() / loss_denominator
                            loss = loss + entropy_loss
                        if config.train.confidence_penalty_weight > 0:
                            if entropy_values is None:
                                raise RuntimeError("confidence penalty requires flat logits entropy values")
                            confidence_penalty_loss = (
                                config.train.confidence_penalty_weight * -entropy_values.float() * token_weights
                            ).sum() / loss_denominator
                            confidence_penalty_loss_for_log = confidence_penalty_loss
                            loss = loss + confidence_penalty_loss
                        if config.train.z_loss_weight > 0:
                            if z_loss_values is None:
                                raise RuntimeError("z-loss requires flat logits")
                            z_loss = (
                                config.train.z_loss_weight * z_loss_values.float() * token_weights
                            ).sum() / loss_denominator
                            z_loss_for_log = z_loss
                            loss = loss + z_loss
                        if config.train.eos_hazard_loss_weight > 0:
                            eos_hazard_loss = config.train.eos_hazard_loss_weight * raw_eos_hazard_calibration_loss
                            eos_hazard_loss_for_log = eos_hazard_loss
                            loss = loss + eos_hazard_loss
                    else:
                        if config.train.position_decile_balanced_loss:
                            schedule_multiplier = (
                                1.0 / max(float(t_val), 1e-8)
                                if config.noise.schedule_weight and t_val is not None
                                else 1.0
                            )
                            token_weights = torch.full_like(
                                main_ce_values,
                                float(schedule_multiplier),
                                dtype=torch.float32,
                            )
                            main_loss, position_decile_stats = _position_decile_balanced_ce_loss(
                                main_ce_values,
                                target_indices,
                                attention_mask,
                                loss_mask,
                                token_weights,
                            )
                            aux_losses = [
                                _position_decile_balanced_ce_loss(
                                    aux_ce,
                                    target_indices,
                                    attention_mask,
                                    loss_mask,
                                    token_weights,
                                )[0]
                                for aux_ce in aux_ce_values
                            ]
                            loss = main_loss
                            for aux_loss in aux_losses:
                                loss = loss + config.train.auxiliary_loss_weight * aux_loss
                            loss_normalization_scale = float(schedule_multiplier)
                            main_loss_for_log = main_loss
                        else:
                            main_loss = raw_main_ce
                            aux_losses = raw_aux_ces
                            loss = main_loss
                            for aux_loss in aux_losses:
                                loss = loss + config.train.auxiliary_loss_weight * aux_loss
                            if config.train.entropy_bonus_beta > 0 and entropy_values is not None:
                                entropy_loss = -config.train.entropy_bonus_beta * raw_entropy
                                loss = loss + entropy_loss
                            if config.train.confidence_penalty_weight > 0:
                                confidence_penalty_loss = (
                                    config.train.confidence_penalty_weight * raw_confidence_penalty
                                )
                                loss = loss + confidence_penalty_loss
                            if config.train.z_loss_weight > 0:
                                z_loss = config.train.z_loss_weight * raw_z_loss
                                loss = loss + z_loss
                            if config.train.eos_hazard_loss_weight > 0:
                                eos_hazard_loss = config.train.eos_hazard_loss_weight * raw_eos_hazard_calibration_loss
                                loss = loss + eos_hazard_loss
                            uses_fast_dllm_scale = config.noise.type in {"mdlm", "bd3lm"} or (
                                config.noise.type == "uniform" and config.noise.time_conditioning
                            )
                            loss_normalization_scale = (
                                _fast_dllm_loss_normalization_scale(
                                    int(target_indices.numel()),
                                    lossable_tokens,
                                    t_val,
                                    bool(config.noise.schedule_weight),
                                )
                                if uses_fast_dllm_scale
                                else 1.0
                            )
                            loss = loss * loss_normalization_scale
                            main_loss_for_log = main_loss * loss_normalization_scale
                            if config.train.confidence_penalty_weight > 0:
                                confidence_penalty_loss_for_log = confidence_penalty_loss * loss_normalization_scale
                            if config.train.z_loss_weight > 0:
                                z_loss_for_log = z_loss * loss_normalization_scale
                            if config.train.eos_hazard_loss_weight > 0:
                                eos_hazard_loss_for_log = eos_hazard_loss * loss_normalization_scale
                    if config.train.length_head_loss_weight > 0:
                        loss = loss + length_head_loss_for_log
                loss_value = float(loss.detach().item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"non-finite loss at step {step}, micro {micro}: {loss_value}")
                step_loss_sum += loss_value
                step_main_ce_sum += float(raw_main_ce.detach().item())
                step_main_loss_sum += float(main_loss_for_log.detach().item())
                step_confidence_penalty_sum += float(raw_confidence_penalty.detach().item())
                step_confidence_penalty_loss_sum += float(confidence_penalty_loss_for_log.detach().item())
                step_z_loss_sum += float(raw_z_loss.detach().item())
                step_z_loss_loss_sum += float(z_loss_for_log.detach().item())
                step_eos_hazard_calibration_loss_sum += float(raw_eos_hazard_calibration_loss.detach().item())
                step_eos_hazard_loss_sum += float(eos_hazard_loss_for_log.detach().item())
                if eos_hazard_stats is not None:
                    _update_eos_hazard_accumulator(step_eos_hazard_accumulator, eos_hazard_stats, 1)
                if length_head_stats is not None:
                    _update_length_head_accumulator(step_length_head_accumulator, length_head_stats, 1)
                if position_decile_stats is not None:
                    _update_position_decile_ce_accumulator(
                        step_position_decile_ce_accumulator,
                        position_decile_stats,
                        1,
                    )
                if token_edit_branch:
                    if token_edit_class_ids is None:
                        raise RuntimeError("token-edit corruption requires frequency decile token classes")
                    _update_token_edit_reconstruction_accumulator(
                        step_token_edit_reconstruction_accumulator,
                        logits,
                        targets,
                        token_edit_class_ids,
                        1,
                    )
                    step_token_edit_reconstruction_accumulator["microbatches"] = (
                        int(step_token_edit_reconstruction_accumulator["microbatches"]) + 1
                    )
                if raw_class_ce is not None and raw_within_class_ce is not None:
                    step_class_ce_sum += float(raw_class_ce.detach().item())
                    step_within_class_ce_sum += float(raw_within_class_ce.detach().item())
                step_entropy_sum += float(raw_entropy.detach().item())
                step_top1_prob_sum += float(raw_max_prob.detach().item())
                step_argmax_mode_sum += float(raw_argmax_mode_frac.detach().item())
                if logsumexp_values is not None and logit_norms is not None and logit_rms is not None:
                    logit_norms_f = logit_norms.detach().float()
                    logsumexp_values_f = logsumexp_values.detach().float()
                    logit_rms_f = logit_rms.detach().float()
                    step_logit_norm_sum += float(logit_norms_f.mean().item())
                    step_logit_norm_min = min(step_logit_norm_min, float(logit_norms_f.min().item()))
                    step_logit_norm_max = max(step_logit_norm_max, float(logit_norms_f.max().item()))
                    step_logit_rms_sum += float(logit_rms_f.mean().item())
                    step_logsumexp_sum += float(logsumexp_values_f.mean().item())
                    step_logsumexp_min = min(step_logsumexp_min, float(logsumexp_values_f.min().item()))
                    step_logsumexp_max = max(step_logsumexp_max, float(logsumexp_values_f.max().item()))
                step_loss_normalization_scale_sum += float(loss_normalization_scale)
                if token_edit_branch:
                    step_t2t_ce_sum += float(raw_main_ce.detach().item())
                    step_t2t_loss_sum += float(main_loss_for_log.detach().item())
                    step_t2t_loss_count += 1
                    step_t2t_microbatches += 1
                    step_t2t_active_tokens += int(target_indices.numel())
                else:
                    step_m2t_ce_sum += float(raw_main_ce.detach().item())
                    step_m2t_loss_sum += float(main_loss_for_log.detach().item())
                    step_m2t_loss_count += 1
                    step_m2t_microbatches += 1
                    step_m2t_active_tokens += int(target_indices.numel())
                for aux_idx, aux_ce in enumerate(raw_aux_ces):
                    step_aux_ce_sums[aux_idx] += float(aux_ce.detach().item())
                if t_val is not None:
                    if isinstance(t_val, torch.Tensor):
                        if t_val.shape == x_0.shape:
                            t_values = t_val[loss_mask].detach().float().flatten()
                        else:
                            t_values = t_val.detach().float().flatten()
                        if t_values.numel() > 0:
                            step_t_sum += float(t_values.sum().item())
                            step_t_min = min(step_t_min, float(t_values.min().item()))
                            step_t_max = max(step_t_max, float(t_values.max().item()))
                            step_t_count += int(t_values.numel())
                            if config.noise.schedule_weight:
                                step_schedule_multiplier_sum += float((1.0 / t_values.clamp_min(1e-8)).sum().item())
                    else:
                        step_t_sum += float(t_val)
                        step_t_min = min(step_t_min, float(t_val))
                        step_t_max = max(step_t_max, float(t_val))
                        step_t_count += 1
                        if config.noise.schedule_weight:
                            step_schedule_multiplier_sum += 1.0 / max(float(t_val), 1e-8)
                step_loss_count += 1
                scaler.scale(loss / grad_accum_steps).backward()
            batch = next(train_loader)
            x_0 = batch.x_0
            loss_mask = batch.loss_mask
            attention_mask = batch.attention_mask
            epoch = batch.epoch

        step_counters = _packed_all_reduce_int_counters(
            {
                "processed_tokens": step_processed_tokens,
                "valid_tokens": step_valid_tokens,
                "lossable_tokens": step_lossable_tokens,
                "active_tokens": step_active_tokens,
                "content_tokens": step_content_tokens,
                "content_bytes": step_content_bytes,
                "m2t_microbatches": step_m2t_microbatches,
                "m2t_active_tokens": step_m2t_active_tokens,
                "t2t_microbatches": step_t2t_microbatches,
                "t2t_active_tokens": step_t2t_active_tokens,
                "visible_prefix_count": step_visible_prefix_count,
                "loss_count": step_loss_count,
                "m2t_loss_count": step_m2t_loss_count,
                "t2t_loss_count": step_t2t_loss_count,
                "t_count": step_t_count,
                "soft_input_stat_count": step_soft_input_stat_count,
                "uniform_estimator_stat_count": step_uniform_estimator_stat_count,
                "per_token_effective_masked_fraction_count": step_per_token_effective_masked_fraction_count,
            },
            device=device,
            is_ddp=is_ddp,
        )
        step_processed_tokens = step_counters["processed_tokens"]
        step_valid_tokens = step_counters["valid_tokens"]
        step_lossable_tokens = step_counters["lossable_tokens"]
        step_active_tokens = step_counters["active_tokens"]
        step_content_tokens = step_counters["content_tokens"]
        step_content_bytes = step_counters["content_bytes"]
        step_m2t_microbatches = step_counters["m2t_microbatches"]
        step_m2t_active_tokens = step_counters["m2t_active_tokens"]
        step_t2t_microbatches = step_counters["t2t_microbatches"]
        step_t2t_active_tokens = step_counters["t2t_active_tokens"]
        step_visible_prefix_count = step_counters["visible_prefix_count"]
        step_loss_count = step_counters["loss_count"]
        step_m2t_loss_count = step_counters["m2t_loss_count"]
        step_t2t_loss_count = step_counters["t2t_loss_count"]
        step_t_count = step_counters["t_count"]
        step_soft_input_stat_count = step_counters["soft_input_stat_count"]
        step_uniform_estimator_stat_count = step_counters["uniform_estimator_stat_count"]
        step_per_token_effective_masked_fraction_count = step_counters["per_token_effective_masked_fraction_count"]
        step_float_sums = _packed_all_reduce_float_counters(
            {
                "loss_sum": step_loss_sum,
                "main_ce_sum": step_main_ce_sum,
                "main_loss_sum": step_main_loss_sum,
                "confidence_penalty_sum": step_confidence_penalty_sum,
                "confidence_penalty_loss_sum": step_confidence_penalty_loss_sum,
                "z_loss_sum": step_z_loss_sum,
                "z_loss_loss_sum": step_z_loss_loss_sum,
                "eos_hazard_calibration_loss_sum": step_eos_hazard_calibration_loss_sum,
                "eos_hazard_loss_sum": step_eos_hazard_loss_sum,
                "class_ce_sum": step_class_ce_sum,
                "within_class_ce_sum": step_within_class_ce_sum,
                "ar_unsmoothed_ce_sum": step_ar_unsmoothed_ce_sum,
                "m2t_ce_sum": step_m2t_ce_sum,
                "m2t_loss_sum": step_m2t_loss_sum,
                "t2t_ce_sum": step_t2t_ce_sum,
                "t2t_loss_sum": step_t2t_loss_sum,
                "t_sum": step_t_sum,
                "schedule_multiplier_sum": step_schedule_multiplier_sum,
                "loss_normalization_scale_sum": step_loss_normalization_scale_sum,
                "entropy_sum": step_entropy_sum,
                "top1_prob_sum": step_top1_prob_sum,
                "argmax_mode_sum": step_argmax_mode_sum,
                "logit_norm_sum": step_logit_norm_sum,
                "logit_rms_sum": step_logit_rms_sum,
                "logsumexp_sum": step_logsumexp_sum,
                "visible_prefix_frac_sum": step_visible_prefix_frac_sum,
                "per_token_effective_masked_fraction_sum": step_per_token_effective_masked_fraction_sum,
            },
            device=device,
            is_ddp=is_ddp,
        )
        step_loss_sum = step_float_sums["loss_sum"]
        step_main_ce_sum = step_float_sums["main_ce_sum"]
        step_main_loss_sum = step_float_sums["main_loss_sum"]
        step_confidence_penalty_sum = step_float_sums["confidence_penalty_sum"]
        step_confidence_penalty_loss_sum = step_float_sums["confidence_penalty_loss_sum"]
        step_z_loss_sum = step_float_sums["z_loss_sum"]
        step_z_loss_loss_sum = step_float_sums["z_loss_loss_sum"]
        step_eos_hazard_calibration_loss_sum = step_float_sums["eos_hazard_calibration_loss_sum"]
        step_eos_hazard_loss_sum = step_float_sums["eos_hazard_loss_sum"]
        step_class_ce_sum = step_float_sums["class_ce_sum"]
        step_within_class_ce_sum = step_float_sums["within_class_ce_sum"]
        step_ar_unsmoothed_ce_sum = step_float_sums["ar_unsmoothed_ce_sum"]
        step_m2t_ce_sum = step_float_sums["m2t_ce_sum"]
        step_m2t_loss_sum = step_float_sums["m2t_loss_sum"]
        step_t2t_ce_sum = step_float_sums["t2t_ce_sum"]
        step_t2t_loss_sum = step_float_sums["t2t_loss_sum"]
        step_t_sum = step_float_sums["t_sum"]
        step_schedule_multiplier_sum = step_float_sums["schedule_multiplier_sum"]
        step_loss_normalization_scale_sum = step_float_sums["loss_normalization_scale_sum"]
        step_entropy_sum = step_float_sums["entropy_sum"]
        step_top1_prob_sum = step_float_sums["top1_prob_sum"]
        step_argmax_mode_sum = step_float_sums["argmax_mode_sum"]
        step_logit_norm_sum = step_float_sums["logit_norm_sum"]
        step_logit_rms_sum = step_float_sums["logit_rms_sum"]
        step_logsumexp_sum = step_float_sums["logsumexp_sum"]
        step_visible_prefix_frac_sum = step_float_sums["visible_prefix_frac_sum"]
        step_per_token_effective_masked_fraction_sum = step_float_sums["per_token_effective_masked_fraction_sum"]
        step_aux_ce_sums = list(
            _all_reduce_numeric_accumulator(
                {"values": step_aux_ce_sums},
                device=device,
                is_ddp=is_ddp,
            )["values"]
        )
        step_soft_input_stat_sums = dict(
            _all_reduce_numeric_accumulator(
                step_soft_input_stat_sums,
                device=device,
                is_ddp=is_ddp,
            )
        )
        step_uniform_estimator_stat_sums = dict(
            _all_reduce_numeric_accumulator(
                step_uniform_estimator_stat_sums,
                device=device,
                is_ddp=is_ddp,
            )
        )
        reduced_histograms = _all_reduce_numeric_accumulator(
            {
                "top1": step_top1_prob_hist_counts,
                "entropy": step_entropy_hist_counts,
                "logsumexp": step_logsumexp_hist_counts,
                "logit_norm": step_logit_norm_hist_counts,
                "per_token_noise": step_per_token_noise_level_hist_counts,
            },
            device=device,
            is_ddp=is_ddp,
        )
        step_top1_prob_hist_counts = list(reduced_histograms["top1"])
        step_entropy_hist_counts = list(reduced_histograms["entropy"])
        step_logsumexp_hist_counts = list(reduced_histograms["logsumexp"])
        step_logit_norm_hist_counts = list(reduced_histograms["logit_norm"])
        step_per_token_noise_level_hist_counts = list(reduced_histograms["per_token_noise"])
        step_eos_hazard_accumulator = _all_reduce_numeric_accumulator(
            step_eos_hazard_accumulator,
            device=device,
            is_ddp=is_ddp,
        )
        step_length_head_accumulator = _all_reduce_numeric_accumulator(
            step_length_head_accumulator,
            device=device,
            is_ddp=is_ddp,
        )
        step_position_decile_ce_accumulator = _all_reduce_numeric_accumulator(
            step_position_decile_ce_accumulator,
            device=device,
            is_ddp=is_ddp,
        )
        step_token_edit_reconstruction_accumulator = _all_reduce_numeric_accumulator(
            step_token_edit_reconstruction_accumulator,
            device=device,
            is_ddp=is_ddp,
        )
        step_mins = _packed_all_reduce_float_counters(
            {
                "t": step_t_min,
                "logit_norm": step_logit_norm_min,
                "logsumexp": step_logsumexp_min,
                "per_token_effective_masked_fraction": step_per_token_effective_masked_fraction_min,
            },
            device=device,
            is_ddp=is_ddp,
            op=dist.ReduceOp.MIN,
        )
        step_maxes = _packed_all_reduce_float_counters(
            {
                "t": step_t_max,
                "logit_norm": step_logit_norm_max,
                "logsumexp": step_logsumexp_max,
                "per_token_effective_masked_fraction": step_per_token_effective_masked_fraction_max,
            },
            device=device,
            is_ddp=is_ddp,
            op=dist.ReduceOp.MAX,
        )
        step_t_min = step_mins["t"]
        step_logit_norm_min = step_mins["logit_norm"]
        step_logsumexp_min = step_mins["logsumexp"]
        step_per_token_effective_masked_fraction_min = step_mins["per_token_effective_masked_fraction"]
        step_t_max = step_maxes["t"]
        step_logit_norm_max = step_maxes["logit_norm"]
        step_logsumexp_max = step_maxes["logsumexp"]
        step_per_token_effective_masked_fraction_max = step_maxes["per_token_effective_masked_fraction"]
        if step_processed_tokens != total_batch_size:
            raise RuntimeError(
                "globally reduced processed-token count does not match the configured effective batch: "
                f"observed={step_processed_tokens}, expected={total_batch_size}"
            )
        train_m2t_ce_sum += step_m2t_ce_sum
        train_m2t_loss_sum += step_m2t_loss_sum
        train_m2t_loss_count += step_m2t_loss_count
        train_m2t_microbatches += step_m2t_microbatches
        train_m2t_active_tokens += step_m2t_active_tokens
        interval_m2t_ce_sum += step_m2t_ce_sum
        interval_m2t_loss_sum += step_m2t_loss_sum
        interval_m2t_loss_count += step_m2t_loss_count
        interval_m2t_microbatches += step_m2t_microbatches
        interval_m2t_active_tokens += step_m2t_active_tokens
        train_t2t_ce_sum += step_t2t_ce_sum
        train_t2t_loss_sum += step_t2t_loss_sum
        train_t2t_loss_count += step_t2t_loss_count
        train_t2t_microbatches += step_t2t_microbatches
        train_t2t_active_tokens += step_t2t_active_tokens
        interval_t2t_ce_sum += step_t2t_ce_sum
        interval_t2t_loss_sum += step_t2t_loss_sum
        interval_t2t_loss_count += step_t2t_loss_count
        interval_t2t_microbatches += step_t2t_microbatches
        interval_t2t_active_tokens += step_t2t_active_tokens
        _merge_numeric_accumulator(train_length_head_accumulator, step_length_head_accumulator)
        for index, value in enumerate(step_per_token_noise_level_hist_counts):
            train_per_token_noise_level_hist_counts[index] += int(value)
        train_per_token_effective_masked_fraction_sum += step_per_token_effective_masked_fraction_sum
        train_per_token_effective_masked_fraction_count += step_per_token_effective_masked_fraction_count
        train_per_token_effective_masked_fraction_min = min(
            train_per_token_effective_masked_fraction_min,
            step_per_token_effective_masked_fraction_min,
        )
        train_per_token_effective_masked_fraction_max = max(
            train_per_token_effective_masked_fraction_max,
            step_per_token_effective_masked_fraction_max,
        )

        _unscale_optimizers(scaler, optimizers)
        step_gradient_component_norms = _gradient_component_norm_metrics(raw_model)
        if not initial_gradient_metrics:
            initial_gradient_metrics = {"initial_gradient_step": step + 1}
            initial_gradient_metrics.update(
                {f"initial_{key}": value for key, value in step_gradient_component_norms.items()}
            )
        if record_early_residual_diagnostics and rank == 0 and residual_diagnostic_batch is not None:
            diagnostic_tokens, diagnostic_attention_mask, diagnostic_position_ids = residual_diagnostic_batch
            residual_record: dict[str, float | int | str] = {
                "step": step + 1,
                **_residual_projection_init_metrics(config),
                **_residual_branch_rms_metrics(
                    raw_model,
                    diagnostic_tokens,
                    diagnostic_attention_mask,
                    diagnostic_position_ids,
                ),
                **_layerwise_gradient_norm_metrics(raw_model),
            }
            os.makedirs(os.path.dirname(residual_diagnostics_path), exist_ok=True)
            append_jsonl(residual_diagnostics_path, residual_record)
            early_residual_diagnostics = {
                "early_residual_diagnostics_step": step + 1,
                "early_residual_diagnostics_path": residual_diagnostics_path,
            }
            early_residual_diagnostics.update(
                {
                    f"early_{key}": value
                    for key, value in residual_record.items()
                    if key != "step" and isinstance(value, int | float)
                }
            )
        record_total_update_norm = (step + 1) % config.train.log_interval == 0 or step + 1 == total_steps
        record_component_update_norm = record_total_update_norm or (
            record_embedding_head_init_diagnostics and step == start_step
        )
        pre_step_params = None
        pre_step_components = None
        if record_total_update_norm:
            pre_step_params = [p.detach().clone() for p in raw_model.parameters() if p.requires_grad]
        if record_component_update_norm:
            pre_step_components = {
                "embedding": [p.detach().clone() for p in raw_model.wte.parameters()],
                "lm_head": [p.detach().clone() for p in raw_model.lm_head.parameters()],
            }
        if config.optim.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.optim.grad_clip))
            last_grad_clipped = grad_norm > grad_clip_threshold
            last_grad_clip_ratio = min(1.0, grad_clip_threshold / max(grad_norm, 1e-12))
            grad_clip_event_count += int(last_grad_clipped)
            grad_clip_observed_steps += 1
        else:
            grad_norm = _grad_norm(raw_model.parameters())
            last_grad_clipped = False
            last_grad_clip_ratio = 1.0
        _step_optimizers(scaler, optimizers)
        scaler.update()
        second_moment_norm = 0.0
        update_norm = 0.0
        if pre_step_params is not None:
            second_moment_norm = _second_moment_norm(optimizers)
            post_step_params = [p for p in raw_model.parameters() if p.requires_grad]
            update_norm = _update_norm(pre_step_params, post_step_params)
        update_component_norms = _component_update_norm_metrics(pre_step_components, raw_model)
        update_component_norms.update(
            _component_update_grad_ratio_metrics(update_component_norms, step_gradient_component_norms)
        )
        if pre_step_components is not None:
            last_update_component_norms = update_component_norms
            if not initial_update_metrics:
                initial_update_metrics = {"initial_update_step": step + 1}
                initial_update_metrics.update(
                    {f"initial_{key}": value for key, value in update_component_norms.items()}
                )
        train_tokens += step_processed_tokens
        train_valid_tokens += step_valid_tokens
        train_content_tokens += step_content_tokens
        train_content_bytes += step_content_bytes
        if _uses_block_causal_pilot(config):
            block_causal_visible_prefix_frac_sum += step_visible_prefix_frac_sum
            block_causal_visible_prefix_count += step_visible_prefix_count
        last_loss_value = step_loss_sum / max(1, step_loss_count)
        last_main_ce = step_main_ce_sum / max(1, step_loss_count)
        last_main_loss = step_main_loss_sum / max(1, step_loss_count)
        last_confidence_penalty = step_confidence_penalty_sum / max(1, step_loss_count)
        last_confidence_penalty_loss = step_confidence_penalty_loss_sum / max(1, step_loss_count)
        last_z_loss = step_z_loss_sum / max(1, step_loss_count)
        last_z_loss_loss = step_z_loss_loss_sum / max(1, step_loss_count)
        last_eos_hazard_calibration_loss = step_eos_hazard_calibration_loss_sum / max(1, step_loss_count)
        last_eos_hazard_loss = step_eos_hazard_loss_sum / max(1, step_loss_count)
        last_eos_hazard_accumulator = step_eos_hazard_accumulator
        last_length_head_accumulator = step_length_head_accumulator
        last_position_decile_ce_accumulator = step_position_decile_ce_accumulator
        last_token_edit_reconstruction_accumulator = step_token_edit_reconstruction_accumulator
        _merge_token_edit_reconstruction_accumulator(
            train_token_edit_reconstruction_accumulator,
            step_token_edit_reconstruction_accumulator,
        )
        last_m2t_ce = step_m2t_ce_sum / max(1, step_m2t_loss_count)
        last_m2t_loss = step_m2t_loss_sum / max(1, step_m2t_loss_count)
        last_m2t_microbatches = step_m2t_microbatches
        last_m2t_active_tokens = step_m2t_active_tokens
        last_t2t_ce = step_t2t_ce_sum / max(1, step_t2t_loss_count)
        last_t2t_loss = step_t2t_loss_sum / max(1, step_t2t_loss_count)
        last_t2t_microbatches = step_t2t_microbatches
        last_t2t_active_tokens = step_t2t_active_tokens
        last_grad_norm = grad_norm
        last_gradient_component_norms = step_gradient_component_norms
        last_second_moment_norm = second_moment_norm
        last_update_norm = update_norm
        last_update_grad_ratio = update_norm / max(grad_norm, 1e-8)
        last_active_tokens = step_active_tokens
        last_lossable_tokens = step_lossable_tokens
        last_valid_tokens = step_valid_tokens
        last_content_tokens = step_content_tokens
        last_content_bytes = step_content_bytes
        if use_hierarchical_output:
            last_class_ce = step_class_ce_sum / max(1, step_loss_count)
            last_within_class_ce = step_within_class_ce_sum / max(1, step_loss_count)
        if _is_causal(config):
            last_ar_unsmoothed_ce = step_ar_unsmoothed_ce_sum / max(1, step_loss_count)
        if step_loss_count:
            last_entropy_mean = step_entropy_sum / step_loss_count
            last_top1_prob_mean = step_top1_prob_sum / step_loss_count
            last_argmax_mode_frac = step_argmax_mode_sum / step_loss_count
            last_top1_prob_hist_counts = list(step_top1_prob_hist_counts)
            last_entropy_hist_counts = list(step_entropy_hist_counts)
            last_logsumexp_hist_counts = list(step_logsumexp_hist_counts)
            last_logit_norm_hist_counts = list(step_logit_norm_hist_counts)
            last_logit_norm_mean = step_logit_norm_sum / step_loss_count
            last_logit_norm_min = step_logit_norm_min if math.isfinite(step_logit_norm_min) else 0.0
            last_logit_norm_max = step_logit_norm_max if math.isfinite(step_logit_norm_max) else 0.0
            last_logit_rms_mean = step_logit_rms_sum / step_loss_count
            last_logsumexp_mean = step_logsumexp_sum / step_loss_count
            last_logsumexp_min = step_logsumexp_min if math.isfinite(step_logsumexp_min) else 0.0
            last_logsumexp_max = step_logsumexp_max if math.isfinite(step_logsumexp_max) else 0.0
        if step_soft_input_stat_count:
            last_soft_input_stats = {
                key: value / step_soft_input_stat_count for key, value in step_soft_input_stat_sums.items()
            }
            last_soft_input_stats["soft_input_curriculum_phase"] = step_soft_input_phase
        if step_uniform_estimator_stat_count:
            last_uniform_estimator_stats = {
                key: value / step_uniform_estimator_stat_count
                for key, value in step_uniform_estimator_stat_sums.items()
            }
            last_uniform_estimator_stats["uniform_loss_estimator"] = step_uniform_estimator_label
        last_per_token_noise_level_hist_counts = list(step_per_token_noise_level_hist_counts)
        last_per_token_effective_masked_fraction_sum = step_per_token_effective_masked_fraction_sum
        last_per_token_effective_masked_fraction_count = step_per_token_effective_masked_fraction_count
        last_per_token_effective_masked_fraction_min = step_per_token_effective_masked_fraction_min
        last_per_token_effective_masked_fraction_max = step_per_token_effective_masked_fraction_max

        if record_embedding_head_init_diagnostics and not early_training_diagnostics:
            early_training_diagnostics = {
                "early_train_step": step + 1,
                "early_train_entropy_mean": last_entropy_mean,
                "early_train_top1_prob_mean": last_top1_prob_mean,
                "early_train_argmax_mode_frac": last_argmax_mode_frac,
                "early_train_masked_logit_norm_mean": last_logit_norm_mean,
                "early_train_masked_logit_norm_min": last_logit_norm_min,
                "early_train_masked_logit_norm_max": last_logit_norm_max,
                "early_train_masked_logit_rms_mean": last_logit_rms_mean,
                "early_train_masked_logsumexp_mean": last_logsumexp_mean,
                "early_train_masked_logsumexp_min": last_logsumexp_min,
                "early_train_masked_logsumexp_max": last_logsumexp_max,
            }
            early_training_diagnostics.update(
                _probability_histogram_metrics("early_train_masked_top1_prob_hist", last_top1_prob_hist_counts)
            )
            early_training_diagnostics.update(
                _value_histogram_metrics("early_train_masked_entropy_hist", last_entropy_hist_counts, 0.0, 16.0)
            )
            early_training_diagnostics.update(
                _value_histogram_metrics(
                    "early_train_masked_logsumexp_hist",
                    last_logsumexp_hist_counts,
                    0.0,
                    64.0,
                )
            )
            early_training_diagnostics.update(
                _value_histogram_metrics(
                    "early_train_masked_logit_norm_hist",
                    last_logit_norm_hist_counts,
                    0.0,
                    512.0,
                )
            )
            if rank == 0:
                os.makedirs(os.path.dirname(embedding_head_init_diagnostics_path), exist_ok=True)
                append_jsonl(
                    embedding_head_init_diagnostics_path,
                    {
                        "event": "early_training_confidence",
                        **early_training_diagnostics,
                        **{f"early_{key}": value for key, value in last_update_component_norms.items()},
                        **{f"early_{key}": value for key, value in step_gradient_component_norms.items()},
                    },
                )
        if record_embedding_head_init_diagnostics and rank == 0 and not early_lm_head_row_norm_metrics:
            saved_state = _rng_state(config)
            try:
                early_lm_head_row_norm_metrics = {
                    "early_lm_head_row_norm_step": step + 1,
                    **_prefixed_metric_keys(
                        "early_",
                        _lm_head_row_norm_metrics(
                            raw_model,
                            tokenizer,
                            config.data,
                            device,
                            int(config.train.device_batch_size),
                            int(config.model.max_seq_len),
                            batches=10,
                        ),
                    ),
                }
                os.makedirs(os.path.dirname(embedding_head_init_diagnostics_path), exist_ok=True)
                append_jsonl(
                    embedding_head_init_diagnostics_path,
                    {
                        "event": "early_lm_head_row_norms",
                        **early_lm_head_row_norm_metrics,
                    },
                )
            finally:
                _restore_rng_state(saved_state, config)

        if config.train.divergence_check_interval > 0 and (
            (step + 1) % config.train.divergence_check_interval == 0 or step + 1 == total_steps
        ):
            failed = (not math.isfinite(last_loss_value)) or last_loss_value > config.train.divergence_loss_threshold
            if is_ddp:
                fail_tensor = torch.tensor(int(failed), device=device)
                dist.all_reduce(fail_tensor, op=dist.ReduceOp.MAX)
                failed = bool(fail_tensor.item())
            if failed:
                raise RuntimeError(f"training diverged at step {step + 1}: loss={last_loss_value}")

        if (step + 1) % config.train.log_interval == 0 or step + 1 == total_steps:
            if config.runtime.device_type == "cuda":
                torch.cuda.synchronize()
            smooth_count += 1
            smooth_loss = 0.9 * smooth_loss + 0.1 * last_loss_value
            debiased = smooth_loss / (1 - 0.9**smooth_count)
            now = time.time()
            interval_steps = step + 1 - last_log_step
            tok_sec = int(total_batch_size * interval_steps / max(1e-8, now - last_log_t))
            mfu = flops_per_token * total_batch_size * interval_steps / max(1e-8, now - last_log_t)
            log_record = {
                "step": step + 1,
                "total_steps": total_steps,
                "train_loss": debiased,
                "raw_train_loss": last_loss_value,
                "raw_main_ce": step_main_ce_sum / max(1, step_loss_count),
                "raw_total_loss": last_loss_value,
                "train_main_ce": last_main_ce,
                "train_main_loss": last_main_loss,
                "token_edit_corruption_enabled": _uses_token_edit_corruption(config),
                "token_edit_corruption_batch_fraction": float(config.train.token_edit_corruption_batch_fraction),
                "token_edit_corruption_ratio": float(config.train.token_edit_corruption_ratio),
                "train_m2t_ce": interval_m2t_ce_sum / max(1, interval_m2t_loss_count),
                "train_m2t_loss": interval_m2t_loss_sum / max(1, interval_m2t_loss_count),
                "train_m2t_microbatches": interval_m2t_microbatches,
                "train_m2t_active_tokens": interval_m2t_active_tokens,
                "train_m2t_active_token_frac": interval_m2t_active_tokens / max(1, last_lossable_tokens),
                "train_t2t_ce": interval_t2t_ce_sum / max(1, interval_t2t_loss_count),
                "train_t2t_loss": interval_t2t_loss_sum / max(1, interval_t2t_loss_count),
                "train_t2t_microbatches": interval_t2t_microbatches,
                "train_t2t_active_tokens": interval_t2t_active_tokens,
                "train_t2t_active_token_frac": interval_t2t_active_tokens / max(1, last_lossable_tokens),
                "last_step_m2t_ce": last_m2t_ce,
                "last_step_m2t_loss": last_m2t_loss,
                "last_step_m2t_microbatches": last_m2t_microbatches,
                "last_step_t2t_ce": last_t2t_ce,
                "last_step_t2t_loss": last_t2t_loss,
                "last_step_t2t_microbatches": last_t2t_microbatches,
                "train_confidence_penalty": last_confidence_penalty,
                "train_confidence_penalty_loss": last_confidence_penalty_loss,
                "confidence_penalty_weight": float(config.train.confidence_penalty_weight),
                "train_z_loss": last_z_loss,
                "train_z_loss_loss": last_z_loss_loss,
                "z_loss_weight": float(config.train.z_loss_weight),
                "train_eos_hazard_calibration_loss": last_eos_hazard_calibration_loss,
                "train_eos_hazard_loss": last_eos_hazard_loss,
                "eos_hazard_loss_weight": float(config.train.eos_hazard_loss_weight),
                "lr_multiplier": lrm,
                "lr": _primary_optimizer_lr(optimizers),
                "lr_schedule_warmup_steps": warmup_steps,
                "lr_schedule_warmup_fraction": lr_schedule["warmup_fraction"],
                "lr_schedule_warmup_ratio_config": lr_schedule["warmup_ratio_config"],
                "tok_sec": tok_sec,
                "flops_sec": mfu,
                "grad_norm": last_grad_norm,
                "grad_clip_enabled": grad_clip_enabled,
                "grad_clip_threshold": grad_clip_threshold,
                "grad_clipped": last_grad_clipped,
                "grad_clip_ratio": last_grad_clip_ratio,
                "grad_clip_event_count": grad_clip_event_count,
                "grad_clip_fraction": grad_clip_event_count / max(1, grad_clip_observed_steps),
                "second_moment_norm": last_second_moment_norm,
                "update_norm": last_update_norm,
                "update_grad_ratio": last_update_grad_ratio,
                "valid_tokens": last_valid_tokens,
                "lossable_tokens": last_lossable_tokens,
                "active_tokens": last_active_tokens,
                "active_token_frac": last_active_tokens / max(1, last_lossable_tokens),
                "content_tokens": last_content_tokens,
                "content_bytes": last_content_bytes,
                "content_token_byte_fertility": last_content_tokens / last_content_bytes if last_content_bytes else 0.0,
                "epoch": epoch,
                "train_tokens": train_tokens,
                "train_valid_tokens": train_valid_tokens,
                "train_content_tokens": train_content_tokens,
                "train_content_bytes": train_content_bytes,
                "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
                if config.runtime.device_type == "cuda"
                else 0.0,
                "train_entropy_mean": last_entropy_mean,
                "train_top1_prob_mean": last_top1_prob_mean,
                "train_argmax_mode_frac": last_argmax_mode_frac,
                "train_masked_logit_norm_mean": last_logit_norm_mean,
                "train_masked_logit_norm_min": last_logit_norm_min,
                "train_masked_logit_norm_max": last_logit_norm_max,
                "train_masked_logit_rms_mean": last_logit_rms_mean,
                "train_masked_logsumexp_mean": last_logsumexp_mean,
                "train_masked_logsumexp_min": last_logsumexp_min,
                "train_masked_logsumexp_max": last_logsumexp_max,
            }
            log_record.update(_probability_histogram_metrics("train_masked_top1_prob_hist", step_top1_prob_hist_counts))
            log_record.update(
                _value_histogram_metrics("train_masked_entropy_hist", step_entropy_hist_counts, 0.0, 16.0)
            )
            log_record.update(
                _value_histogram_metrics("train_masked_logsumexp_hist", step_logsumexp_hist_counts, 0.0, 64.0)
            )
            log_record.update(
                _value_histogram_metrics("train_masked_logit_norm_hist", step_logit_norm_hist_counts, 0.0, 512.0)
            )
            log_record.update(
                _eos_hazard_accumulator_metrics(
                    "train_",
                    step_eos_hazard_accumulator,
                    last_eos_hazard_calibration_loss,
                    last_eos_hazard_loss,
                    float(config.train.eos_hazard_loss_weight),
                )
            )
            if config.train.length_head_loss_weight > 0:
                log_record.update(
                    _length_head_accumulator_metrics(
                        "train_",
                        step_length_head_accumulator,
                        float(config.train.length_head_loss_weight),
                    )
                )
            if config.train.position_decile_balanced_loss:
                log_record.update(
                    _position_decile_ce_accumulator_metrics(
                        "train_",
                        step_position_decile_ce_accumulator,
                        True,
                    )
                )
            if _uses_token_edit_corruption(config):
                log_record.update(
                    _token_edit_reconstruction_metrics(
                        "train_",
                        step_token_edit_reconstruction_accumulator,
                        True,
                    )
                )
            if _is_causal(config):
                log_record["ar_train_ce"] = debiased
                log_record["ar_train_label_smoothed_ce"] = step_main_ce_sum / max(1, step_loss_count)
                log_record["ar_train_unsmoothed_ce"] = step_ar_unsmoothed_ce_sum / max(1, step_loss_count)
                log_record["ar_label_smoothing_epsilon"] = _causal_ar_label_smoothing(config)
            if use_hierarchical_output:
                log_record["raw_class_ce"] = step_class_ce_sum / max(1, step_loss_count)
                log_record["raw_within_class_ce"] = step_within_class_ce_sum / max(1, step_loss_count)
            if _uses_bd3lm(config):
                log_record.update(
                    {
                        "bd3lm_block_size": int(config.model.bd3lm_block_size),
                        "bd3lm_concat_seq_len": int(config.model.max_seq_len) * 2,
                        "bd3lm_loss_target_tokens": last_active_tokens,
                        "bd3lm_lossable_noised_half_tokens": last_lossable_tokens,
                    }
                )
            if _uses_block_causal_pilot(config):
                log_record["block_causal_pilot_size"] = int(config.model.block_causal_pilot_size)
                log_record["block_causal_pilot_num_blocks"] = (
                    int(config.model.max_seq_len) + int(config.model.block_causal_pilot_size) - 1
                ) // int(config.model.block_causal_pilot_size)
                if step_visible_prefix_count:
                    log_record["block_causal_pilot_visible_prefix_frac_mean"] = (
                        step_visible_prefix_frac_sum / step_visible_prefix_count
                    )
            if step_soft_input_stat_count:
                log_record.update(last_soft_input_stats)
            if step_uniform_estimator_stat_count:
                log_record.update(last_uniform_estimator_stats)
            if _uses_per_token_noise_levels(config):
                log_record.update(
                    _value_histogram_metrics(
                        "per_token_noise_level_hist",
                        step_per_token_noise_level_hist_counts,
                        0.0,
                        1.0,
                    )
                )
                log_record.update(
                    _per_token_effective_masked_fraction_metrics(
                        "",
                        step_per_token_effective_masked_fraction_sum,
                        step_per_token_effective_masked_fraction_count,
                        step_per_token_effective_masked_fraction_min,
                        step_per_token_effective_masked_fraction_max,
                    )
                )
            for aux_idx, aux_ce_sum in enumerate(step_aux_ce_sums):
                log_record[f"raw_aux_ce_{aux_idx}"] = aux_ce_sum / max(1, step_loss_count)
            if step_t_count:
                log_record["sampled_t_mean"] = step_t_sum / step_t_count
                log_record["sampled_t_min"] = step_t_min
                log_record["sampled_t_max"] = step_t_max
                if config.noise.schedule_weight:
                    log_record["schedule_multiplier_mean"] = step_schedule_multiplier_sum / step_t_count
                log_record["loss_normalization_scale_mean"] = step_loss_normalization_scale_sum / max(
                    1,
                    step_loss_count,
                )
            log_record.update(_optimizer_lr_metrics(optimizers))
            log_record.update(last_gradient_component_norms)
            log_record.update(last_update_component_norms)
            log_record.update(_parameter_norm_metrics(raw_model))
            print0(
                rank,
                f"step {step + 1:05d}/{total_steps:05d} "
                f"loss {debiased:.4f} lrm {lrm:.3f} grad {last_grad_norm:.2f} "
                f"{tok_sec:,} tok/s epoch {epoch}",
                flush=True,
            )
            if rank == 0:
                append_jsonl(train_log_path, log_record)
                _write_tensorboard_scalars(tensorboard_writer, "train", log_record, step + 1)
                flush_tensorboard_writer(tensorboard_writer)
            interval_m2t_ce_sum = 0.0
            interval_m2t_loss_sum = 0.0
            interval_m2t_loss_count = 0
            interval_m2t_microbatches = 0
            interval_m2t_active_tokens = 0
            interval_t2t_ce_sum = 0.0
            interval_t2t_loss_sum = 0.0
            interval_t2t_loss_count = 0
            interval_t2t_microbatches = 0
            interval_t2t_active_tokens = 0
            last_log_t = now
            last_log_step = step + 1

        if rank == 0 and step + 1 in fixed_ce_trace_steps:
            trace_record = _evaluate_fixed_noise_ce_trace(
                raw_model,
                config,
                tokenizer,
                noise,
                autocast_ctx,
                device,
                step + 1,
                total_steps,
                train_tokens,
                train_valid_tokens,
            )
            trace_record["fixed_ce_trace_checkpoint_index"] = fixed_ce_trace_records + 1
            os.makedirs(os.path.dirname(fixed_ce_trace_path), exist_ok=True)
            append_jsonl(fixed_ce_trace_path, trace_record)
            _write_tensorboard_scalars(tensorboard_writer, "fixed_ce_trace", trace_record, step + 1)
            flush_tensorboard_writer(tensorboard_writer)
            fixed_ce_trace_records += 1
            last_fixed_ce_trace_record = trace_record

        periodic_resume_contract = None
        if config.train.save_every > 0 and (step + 1) % config.train.save_every == 0:
            periodic_resume_contract = _collect_resume_contract(
                config,
                batch,
                device=device,
                rank=rank,
                world_size=world_size,
                is_ddp=is_ddp,
                grad_accum_steps=grad_accum_steps,
                consumed_microbatches=(step + 1) * grad_accum_steps,
                trajectory_contract=trajectory_contract,
                trainer_state=current_trainer_state(),
            )
        if periodic_resume_contract is not None and rank == 0:
            checkpoint_metrics = {
                "train_loss": last_loss_value,
                "raw_main_ce": last_main_ce,
                "train_main_loss": last_main_loss,
                "raw_total_loss": last_loss_value,
                "token_edit_corruption_enabled": _uses_token_edit_corruption(config),
                "last_train_m2t_ce": last_m2t_ce,
                "last_train_m2t_loss": last_m2t_loss,
                "last_train_m2t_microbatches": last_m2t_microbatches,
                "last_train_m2t_active_tokens": last_m2t_active_tokens,
                "last_train_t2t_ce": last_t2t_ce,
                "last_train_t2t_loss": last_t2t_loss,
                "last_train_t2t_microbatches": last_t2t_microbatches,
                "last_train_t2t_active_tokens": last_t2t_active_tokens,
                "train_confidence_penalty": last_confidence_penalty,
                "train_confidence_penalty_loss": last_confidence_penalty_loss,
                "confidence_penalty_weight": float(config.train.confidence_penalty_weight),
                "last_train_z_loss": last_z_loss,
                "last_train_z_loss_loss": last_z_loss_loss,
                "z_loss_weight": float(config.train.z_loss_weight),
                "last_train_eos_hazard_calibration_loss": last_eos_hazard_calibration_loss,
                "last_train_eos_hazard_loss": last_eos_hazard_loss,
                "eos_hazard_loss_weight": float(config.train.eos_hazard_loss_weight),
                "ar_unsmoothed_ce": last_ar_unsmoothed_ce,
                "grad_norm": last_grad_norm,
                "last_grad_clipped": last_grad_clipped,
                "last_grad_clip_ratio": last_grad_clip_ratio,
                "grad_clip_event_count": grad_clip_event_count,
                "grad_clip_observed_steps": grad_clip_observed_steps,
                **last_gradient_component_norms,
                **last_update_component_norms,
                "valid_tokens": last_valid_tokens,
                "lossable_tokens": last_lossable_tokens,
                "active_tokens": last_active_tokens,
                "train_valid_tokens": train_valid_tokens,
                "train_content_tokens": train_content_tokens,
                "train_content_bytes": train_content_bytes,
                "train_entropy_mean": last_entropy_mean,
                "train_top1_prob_mean": last_top1_prob_mean,
                "train_argmax_mode_frac": last_argmax_mode_frac,
                "last_train_masked_logit_norm_mean": last_logit_norm_mean,
                "last_train_masked_logit_rms_mean": last_logit_rms_mean,
                "last_train_masked_logsumexp_mean": last_logsumexp_mean,
                **{f"last_{key}": value for key, value in last_soft_input_stats.items()},
                **{f"last_{key}": value for key, value in last_uniform_estimator_stats.items()},
                **_probability_histogram_metrics("train_masked_top1_prob_hist", last_top1_prob_hist_counts),
                **_value_histogram_metrics("train_masked_entropy_hist", last_entropy_hist_counts, 0.0, 16.0),
                **_value_histogram_metrics("train_masked_logsumexp_hist", last_logsumexp_hist_counts, 0.0, 64.0),
                **_value_histogram_metrics("train_masked_logit_norm_hist", last_logit_norm_hist_counts, 0.0, 512.0),
            }
            if _uses_per_token_noise_levels(config):
                checkpoint_metrics.update(
                    _value_histogram_metrics(
                        "last_train_per_token_noise_level_hist",
                        last_per_token_noise_level_hist_counts,
                        0.0,
                        1.0,
                    )
                )
                checkpoint_metrics.update(
                    _per_token_effective_masked_fraction_metrics(
                        "last_train_",
                        last_per_token_effective_masked_fraction_sum,
                        last_per_token_effective_masked_fraction_count,
                        last_per_token_effective_masked_fraction_min,
                        last_per_token_effective_masked_fraction_max,
                    )
                )
            if _uses_token_edit_corruption(config):
                checkpoint_metrics.update(
                    _token_edit_reconstruction_metrics(
                        "last_train_",
                        last_token_edit_reconstruction_accumulator,
                        True,
                    )
                )
            if config.train.length_head_loss_weight > 0:
                checkpoint_metrics.update(
                    _length_head_accumulator_metrics(
                        "last_train_",
                        last_length_head_accumulator,
                        float(config.train.length_head_loss_weight),
                    )
                )
            if config.train.position_decile_balanced_loss:
                checkpoint_metrics.update(
                    _position_decile_ce_accumulator_metrics(
                        "last_train_",
                        last_position_decile_ce_accumulator,
                        True,
                    )
                )
            if use_hierarchical_output:
                checkpoint_metrics["raw_class_ce"] = last_class_ce
                checkpoint_metrics["raw_within_class_ce"] = last_within_class_ce
            periodic_checkpoint_path = os.path.join(config.train.run_dir, f"model_step{step + 1}.pt")
            save_checkpoint(
                periodic_checkpoint_path,
                raw_model,
                optimizers,
                scaler,
                config,
                model_config,
                step + 1,
                train_tokens,
                train_valid_tokens,
                checkpoint_metrics,
                resume_contract=periodic_resume_contract,
            )
            alias_checkpoint(
                periodic_checkpoint_path,
                os.path.join(config.train.run_dir, "model_latest.pt"),
            )

        if (
            config.eval.enabled
            and config.train.eval_every > 0
            and rank == 0
            and (step + 1) % config.train.eval_every == 0
            and step + 1 < total_steps
        ):
            eval_start = time.time()
            eval_metrics = _run_eval(
                raw_model,
                config,
                tokenizer,
                noise,
                autocast_ctx,
                device,
                step + 1,
                include_gen=config.eval.gen_eval and config.eval.gen_eval_periodic,
            )
            eval_metrics["evaluation_seconds"] = time.time() - eval_start
            append_jsonl(eval_log_path, eval_metrics)
            _write_tensorboard_scalars(tensorboard_writer, "eval", eval_metrics, step + 1)
            flush_tensorboard_writer(tensorboard_writer)
            if _is_causal(config):
                print0(
                    rank,
                    f"eval step {step + 1:05d}: val_ce {eval_metrics.get('val_ce', 0.0):.4f} "
                    f"val_bpb {eval_metrics.get('val_bpb', 0.0):.4f}",
                    flush=True,
                )
            else:
                print0(
                    rank,
                    f"eval step {step + 1:05d}: val_nelbo_bpb {eval_metrics['val_nelbo_bpb']:.4f}",
                    flush=True,
                )

    if config.runtime.device_type == "cuda":
        torch.cuda.synchronize()
    final_resume_contract = _collect_resume_contract(
        config,
        batch,
        device=device,
        rank=rank,
        world_size=world_size,
        is_ddp=is_ddp,
        grad_accum_steps=grad_accum_steps,
        consumed_microbatches=total_steps * grad_accum_steps,
        trajectory_contract=trajectory_contract,
        trainer_state=current_trainer_state(),
    )
    training_seconds = time.time() - t0
    ar_boundary_counts = _packed_all_reduce_int_counters(
        {key: int(value) for key, value in ar_boundary_counts.items()},
        device=device,
        is_ddp=is_ddp,
    )
    boundary_telemetry = _packed_all_reduce_int_counters(
        {key: int(value) for key, value in boundary_telemetry.items()},
        device=device,
        is_ddp=is_ddp,
    )
    row_sum_keys = tuple(key for key in row_telemetry if not key.endswith("_min") and not key.endswith("_max"))
    row_extrema_min_keys = tuple(key for key in row_telemetry if key.endswith("_min"))
    row_extrema_max_keys = tuple(key for key in row_telemetry if key.endswith("_max"))
    row_sums = _packed_all_reduce_int_counters(
        {key: int(row_telemetry[key]) for key in row_sum_keys},
        device=device,
        is_ddp=is_ddp,
    )
    row_mins = _packed_all_reduce_int_counters(
        {key: int(row_telemetry[key]) for key in row_extrema_min_keys},
        device=device,
        is_ddp=is_ddp,
        op=dist.ReduceOp.MIN,
    )
    row_maxes = _packed_all_reduce_int_counters(
        {key: int(row_telemetry[key]) for key in row_extrema_max_keys},
        device=device,
        is_ddp=is_ddp,
        op=dist.ReduceOp.MAX,
    )
    row_telemetry = {**row_sums, **row_mins, **row_maxes}
    session_train_tokens = max(0, train_tokens - resume_train_tokens)
    session_train_valid_tokens = max(0, train_valid_tokens - resume_train_valid_tokens)
    session_train_content_tokens = max(0, train_content_tokens - resume_train_content_tokens)
    session_train_content_bytes = max(0, train_content_bytes - resume_train_content_bytes)
    eos_supervision_train_counts_global = (
        _reduce_eos_supervision_counts(eos_supervision_train_counts, device, is_ddp)
        if eos_supervision_audit_enabled
        else _new_eos_supervision_counts()
    )
    eos_supervision_train_failures = (
        _eos_supervision_training_invariant_failures(
            eos_supervision_train_counts_global,
            eos_supervision_startup_metrics,
        )
        if eos_supervision_audit_enabled
        else []
    )
    eos_supervision_train_metrics = {
        "eos_supervision_audit_enabled": bool(eos_supervision_audit_enabled),
        "eos_supervision_audit_artifact": "telemetry/eos_supervision_audit.jsonl",
        "eos_supervision_audit_path": os.path.join(config.train.run_dir, "telemetry", "eos_supervision_audit.jsonl"),
        "eos_supervision_invariant_status": (
            "ok"
            if (eos_supervision_audit_enabled and not eos_supervision_train_failures)
            else ("disabled" if not eos_supervision_audit_enabled else "failed")
        ),
        "eos_supervision_invariant_error": "; ".join(eos_supervision_train_failures),
    }
    eos_supervision_train_metrics.update(
        _eos_supervision_count_metrics("train_eos_supervision_", eos_supervision_train_counts_global)
    )
    if eos_supervision_audit_enabled and rank == 0:
        append_jsonl(
            os.path.join(config.train.run_dir, "telemetry", "eos_supervision_audit.jsonl"),
            {"event": "final_train_eos_supervision_audit", **eos_supervision_train_metrics},
        )
    if eos_supervision_audit_enabled and eos_supervision_train_metrics["eos_supervision_invariant_status"] != "ok":
        raise RuntimeError(
            f"EOS supervision invariant failed: {eos_supervision_train_metrics['eos_supervision_invariant_error']}"
        )

    distributed_cuda_memory = _distributed_cuda_memory_telemetry(
        device_type=config.runtime.device_type,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_ddp=is_ddp,
    )

    metrics = {
        **source_provenance_metrics(),
        "scale": "small",
        "experiment_type": (
            "causal-ar-diagnostic"
            if _is_causal(config)
            else ("infrastructure-diagnostic" if config.noise.type == "mask_uniform" else "trained-model")
        ),
        "causal_ar": _is_causal(config),
        "cpu_threading": cpu_threading,
        "distributed_cuda_memory": distributed_cuda_memory,
        "distributed_minimum_free_gpu_mib": distributed_cuda_memory["minimum_free_mib"],
        "distributed_minimum_free_gpu_fraction": distributed_cuda_memory["minimum_free_fraction"],
        "distributed_minimum_peak_reserved_headroom_fraction": distributed_cuda_memory[
            "minimum_peak_reserved_headroom_fraction"
        ],
        "distributed_maximum_peak_allocated_gpu_mib": distributed_cuda_memory["maximum_peak_allocated_mib"],
        "distributed_maximum_peak_reserved_gpu_mib": distributed_cuda_memory["maximum_peak_reserved_mib"],
        **run_config_summary(config, "train"),
        **eos_supervision_startup_metrics,
        **eos_supervision_train_metrics,
        "model_depth": config.model.depth,
        "model_n_embd": config.model.n_embd,
        "model_n_head": config.model.n_head,
        "model_mlp_hidden_dim": int(model_config.mlp_hidden_dim),
        "model_resolved_mlp_hidden_dim": int(raw_model.blocks[0].mlp.gate.out_features),
        "model_attention_scale": float(model_config.attention_scale),
        "model_resolved_attention_scale": float(raw_model.architecture_fingerprint()["attention"]["resolved_scale"]),
        "architecture_fingerprint_sha256": architecture_fingerprint["fingerprint_sha256"],
        "pre_update_parameter_state_sha256": initial_state_artifact["pre_update_parameter_state_sha256"],
        "initial_state_sha256": initial_state_artifact["initial_state_sha256"],
        "initialization_contract_sha256": initial_state_artifact["initialization_contract_sha256"],
        **_noise_telemetry(config),
        **_token_edit_corruption_telemetry(config),
        **_bd3lm_telemetry(config),
        **_hierarchical_output_telemetry(token_class_metadata),
        **_residual_projection_init_metrics(config),
        **_embedding_head_init_metrics(config),
        "training_corruption": _training_corruption_label(config),
        **label_smoothing_metrics,
        "tokenizer": config.data.tokenizer,
        "tokenizer_path": config.data.tokenizer_path if config.data.tokenizer == "hf" else "",
        "data_identity_enabled": bool(data_identity_summary["enabled"]),
        "data_identity_mode": str(data_identity_summary["mode"]),
        "data_identity_artifact": str(data_identity_summary["artifact"]),
        "data_identity_sha256": str(data_identity_summary["identity_sha256"]),
        "data_identity_sha256_strength": str(data_identity_summary["identity_sha256_strength"]),
        "label_smoothing_epsilon": float(config.train.label_smoothing_epsilon),
        "ar_label_smoothing_epsilon": _causal_ar_label_smoothing(config),
        "confidence_penalty_weight": float(config.train.confidence_penalty_weight),
        "z_loss_weight": float(config.train.z_loss_weight),
        "eos_hazard_loss_weight": float(config.train.eos_hazard_loss_weight),
        "eos_hazard_relative_position_deciles": EOS_HAZARD_RELATIVE_POSITION_DECILES,
        "length_head_enabled": bool(config.model.length_head),
        "length_head_loss_weight": float(config.train.length_head_loss_weight),
        "length_policy": config.eval.length_policy,
        "target_length_conditioning_enabled": bool(config.model.target_length_conditioning),
        "mask_position_embedding_enabled": bool(config.model.mask_position_embedding),
        "target_length_conditioning_empirical_prior_batches": int(config.eval.target_length_distribution_batches),
        "target_length_conditioning_forced_eos_policy": config.eval.length_policy,
        "uniform_loss_estimator": config.train.uniform_loss_estimator,
        "uniform_low_variance_estimator_enabled": _uses_uniform_low_variance_estimator(config),
        "uniform_estimator_configured_sample_count": 2 if _uses_uniform_low_variance_estimator(config) else 1,
        "token_edit_frequency_batches": int(token_edit_class_metadata.get("frequency_batches", 0))
        if token_edit_class_metadata
        else 0,
        "token_edit_frequency_observed_tokens": int(token_edit_class_metadata.get("frequency_observed_tokens", 0))
        if token_edit_class_metadata
        else 0,
        "token_edit_frequency_nonzero_token_types": int(
            token_edit_class_metadata.get("frequency_nonzero_token_types", 0)
        )
        if token_edit_class_metadata
        else 0,
        **_lr_schedule_metric_fields(lr_schedule),
        "target_tokens": config.train.target_tokens,
        "target_param_count_basis": config.train.target_param_count_basis,
        "target_param_count": _selected_target_param_count(config, counts),
        "tokens_per_target_param": train_tokens / max(1, _selected_target_param_count(config, counts)),
        "parameter_accounting": parameter_accounting,
        "num_steps": total_steps,
        "vocab_size": model_config.vocab_size,
        "mask_token_id": model_config.mask_token_id,
        "bos_token_id": tokenizer.get_bos_token_id(),
        "eos_token_id": tokenizer.get_eos_token_id(),
        "pad_token_id": tokenizer.get_pad_token_id(),
        "special_token_count": len(special_token_ids),
        "generation_blocked_special_token_count": len(tokenizer.get_generation_blocked_token_ids()),
        "data_prepend_bos": bool(config.data.prepend_bos),
        "data_append_eos": bool(config.data.append_eos),
        "data_loss_on_bos": bool(config.data.loss_on_bos),
        "data_loss_on_eos": bool(config.data.loss_on_eos),
        "data_short_document_curriculum": bool(config.data.short_document_curriculum),
        "data_short_document_max_tokens": int(config.data.short_document_max_tokens),
        "num_params": counts["total"],
        "num_non_embedding_params": counts["non_embedding"],
        "num_embedding_params": counts["wte"],
        "num_wte_params": counts["wte"],
        "num_lm_head_params": counts["lm_head"],
        "num_class_head_params": counts["class_head"],
        "num_length_head_params": counts["length_head"],
        "num_target_length_conditioning_params": counts["target_length_conditioning"],
        "num_mask_position_embedding_params": counts["mask_position_embedding"],
        "num_auxiliary_head_params": counts["auxiliary_heads"],
        "num_transformer_params": counts["transformer"],
        "num_params_M": counts["total"] / 1e6,
        "num_non_embedding_params_M": counts["non_embedding"] / 1e6,
        "num_scaling_params_M": counts["scaling"] / 1e6,
        "num_embedding_params_M": counts["wte"] / 1e6,
        "num_wte_params_M": counts["wte"] / 1e6,
        "num_lm_head_params_M": counts["lm_head"] / 1e6,
        "num_class_head_params_M": counts["class_head"] / 1e6,
        "num_length_head_params_M": counts["length_head"] / 1e6,
        "num_target_length_conditioning_params_M": counts["target_length_conditioning"] / 1e6,
        "num_mask_position_embedding_params_M": counts["mask_position_embedding"] / 1e6,
        "num_auxiliary_head_params_M": counts["auxiliary_heads"] / 1e6,
        "num_transformer_params_M": counts["transformer"] / 1e6,
        "flops_per_token": flops_per_token,
        "flops_used": flops_per_token * train_tokens,
        "total_budget_tokens": total_budget_tokens,
        "total_tokens": train_tokens,
        **_token_ratio_metrics(train_tokens, counts),
        "total_tokens_M": train_tokens / 1e6,
        "total_valid_tokens_M": train_valid_tokens / 1e6,
        "total_content_tokens_M": train_content_tokens / 1e6,
        "total_content_bytes_M": train_content_bytes / 1e6,
        "session_tokens_M": session_train_tokens / 1e6,
        "session_valid_tokens_M": session_train_valid_tokens / 1e6,
        "session_content_tokens_M": session_train_content_tokens / 1e6,
        "session_content_bytes_M": session_train_content_bytes / 1e6,
        "avg_tok_sec": int(session_train_tokens / max(1e-8, training_seconds)),
        "avg_valid_tok_sec": int(session_train_valid_tokens / max(1e-8, training_seconds)),
        "avg_content_tok_sec": int(session_train_content_tokens / max(1e-8, training_seconds)),
        "avg_content_byte_sec": int(session_train_content_bytes / max(1e-8, training_seconds)),
        "train_content_tokens": int(train_content_tokens),
        "train_content_bytes": int(train_content_bytes),
        "session_content_tokens": int(session_train_content_tokens),
        "session_content_bytes": int(session_train_content_bytes),
        "train_content_token_byte_fertility": train_content_tokens / train_content_bytes
        if train_content_bytes
        else 0.0,
        "session_content_token_byte_fertility": session_train_content_tokens / session_train_content_bytes
        if session_train_content_bytes
        else 0.0,
        "training_seconds": training_seconds,
        "world_size": world_size,
        "total_batch_size": total_batch_size,
        "effective_batch_size_tokens": total_batch_size,
        "device_batch_size": config.train.device_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "start_step": start_step,
        "resume_contract_version": int(resume_state["version"]) if resume_state is not None else None,
        "resume_contract_status": str(resume_state["status"]) if resume_state is not None else "not_resumed",
        "resume_consumed_microbatches": int(resume_state["consumed_microbatches"]) if resume_state is not None else 0,
        "resume_next_batch_digest_verified": bool(resume_next_batch_digest_verified),
        "checkpoint_resume_contract_version": RESUME_CONTRACT_VERSION,
        "checkpoint_consumed_microbatches": total_steps * grad_accum_steps,
        "training_counter_reduction": "packed_int64_all_reduce",
        "last_train_loss": last_loss_value,
        "last_train_total_loss": last_loss_value,
        "last_train_main_ce": last_main_ce,
        "last_train_main_loss": last_main_loss,
        "last_train_m2t_ce": last_m2t_ce,
        "last_train_m2t_loss": last_m2t_loss,
        "last_train_m2t_microbatches": last_m2t_microbatches,
        "last_train_m2t_active_tokens": last_m2t_active_tokens,
        "last_train_m2t_active_token_frac": last_m2t_active_tokens / max(1, last_lossable_tokens),
        "last_train_t2t_ce": last_t2t_ce,
        "last_train_t2t_loss": last_t2t_loss,
        "last_train_t2t_microbatches": last_t2t_microbatches,
        "last_train_t2t_active_tokens": last_t2t_active_tokens,
        "last_train_t2t_active_token_frac": last_t2t_active_tokens / max(1, last_lossable_tokens),
        "train_m2t_ce_mean": train_m2t_ce_sum / max(1, train_m2t_loss_count),
        "train_m2t_loss_mean": train_m2t_loss_sum / max(1, train_m2t_loss_count),
        "train_m2t_microbatches": int(train_m2t_microbatches),
        "train_m2t_active_tokens": int(train_m2t_active_tokens),
        "train_t2t_ce_mean": train_t2t_ce_sum / max(1, train_t2t_loss_count),
        "train_t2t_loss_mean": train_t2t_loss_sum / max(1, train_t2t_loss_count),
        "train_t2t_microbatches": int(train_t2t_microbatches),
        "train_t2t_active_tokens": int(train_t2t_active_tokens),
        "last_train_confidence_penalty": last_confidence_penalty,
        "last_train_confidence_penalty_loss": last_confidence_penalty_loss,
        "last_train_z_loss": last_z_loss,
        "last_train_z_loss_loss": last_z_loss_loss,
        "last_train_eos_hazard_calibration_loss": last_eos_hazard_calibration_loss,
        "last_train_eos_hazard_loss": last_eos_hazard_loss,
        "ar_last_train_ce": last_loss_value if _is_causal(config) else 0.0,
        "ar_last_train_label_smoothed_ce": last_loss_value if _is_causal(config) else 0.0,
        "ar_last_train_unsmoothed_ce": last_ar_unsmoothed_ce if _is_causal(config) else 0.0,
        "last_train_entropy_mean": last_entropy_mean,
        "last_train_top1_prob_mean": last_top1_prob_mean,
        "last_train_argmax_mode_frac": last_argmax_mode_frac,
        "last_train_masked_logit_norm_mean": last_logit_norm_mean,
        "last_train_masked_logit_norm_min": last_logit_norm_min,
        "last_train_masked_logit_norm_max": last_logit_norm_max,
        "last_train_masked_logit_rms_mean": last_logit_rms_mean,
        "last_train_masked_logsumexp_mean": last_logsumexp_mean,
        "last_train_masked_logsumexp_min": last_logsumexp_min,
        "last_train_masked_logsumexp_max": last_logsumexp_max,
        **{f"last_{key}": value for key, value in last_soft_input_stats.items()},
        **{f"last_{key}": value for key, value in last_uniform_estimator_stats.items()},
        **_probability_histogram_metrics("last_train_masked_top1_prob_hist", last_top1_prob_hist_counts),
        **_value_histogram_metrics("last_train_masked_entropy_hist", last_entropy_hist_counts, 0.0, 16.0),
        **_value_histogram_metrics("last_train_masked_logsumexp_hist", last_logsumexp_hist_counts, 0.0, 64.0),
        **_value_histogram_metrics("last_train_masked_logit_norm_hist", last_logit_norm_hist_counts, 0.0, 512.0),
        **_eos_hazard_accumulator_metrics(
            "last_train_",
            last_eos_hazard_accumulator,
            last_eos_hazard_calibration_loss,
            last_eos_hazard_loss,
            float(config.train.eos_hazard_loss_weight),
        ),
        "last_class_ce": last_class_ce,
        "last_within_class_ce": last_within_class_ce,
        "last_grad_norm": last_grad_norm,
        "grad_clip_enabled": grad_clip_enabled,
        "grad_clip_threshold": grad_clip_threshold,
        "last_grad_clipped": last_grad_clipped,
        "last_grad_clip_ratio": last_grad_clip_ratio,
        "grad_clip_event_count": grad_clip_event_count,
        "grad_clip_observed_steps": grad_clip_observed_steps,
        "grad_clip_fraction": grad_clip_event_count / max(1, grad_clip_observed_steps),
        **last_gradient_component_norms,
        **initial_gradient_metrics,
        **initial_parameter_norm_metrics,
        **initial_update_metrics,
        **initial_lm_head_row_norm_metrics,
        **early_lm_head_row_norm_metrics,
        **early_training_diagnostics,
        "last_second_moment_norm": last_second_moment_norm,
        "last_update_norm": last_update_norm,
        "last_update_grad_ratio": last_update_grad_ratio,
        **{f"last_{key}": value for key, value in last_update_component_norms.items()},
        "embedding_head_init_diagnostics_enabled": bool(record_embedding_head_init_diagnostics),
        "embedding_head_init_diagnostics_artifact": "telemetry/embedding_head_init_diagnostics.jsonl",
        "embedding_head_init_diagnostics_path": embedding_head_init_diagnostics_path,
        "last_valid_tokens": last_valid_tokens,
        "last_lossable_tokens": last_lossable_tokens,
        "last_active_tokens": last_active_tokens,
        "last_active_token_frac": last_active_tokens / max(1, last_lossable_tokens),
        "last_content_tokens": last_content_tokens,
        "last_content_bytes": last_content_bytes,
        "last_content_token_byte_fertility": last_content_tokens / last_content_bytes if last_content_bytes else 0.0,
        "train_special_token_count": int(ar_boundary_counts["special_tokens"]),
        "train_special_loss_token_count": int(ar_boundary_counts["special_loss_tokens"]),
        "train_special_token_fraction": ar_boundary_counts["special_tokens"]
        / max(1, int(ar_boundary_counts["valid_tokens"])),
        "train_special_loss_token_fraction": ar_boundary_counts["special_loss_tokens"]
        / max(1, int(ar_boundary_counts["special_tokens"])),
        "train_bos_token_observed": bool(ar_boundary_counts["bos_tokens"] > 0),
        "train_bos_token_count": int(ar_boundary_counts["bos_tokens"]),
        "train_bos_loss_token_count": int(ar_boundary_counts["bos_loss_tokens"]),
        "train_bos_token_fraction": ar_boundary_counts["bos_tokens"] / max(1, int(ar_boundary_counts["valid_tokens"])),
        "train_bos_loss_token_fraction": ar_boundary_counts["bos_loss_tokens"]
        / max(1, int(ar_boundary_counts["bos_tokens"])),
        "train_bos_batch_rate": ar_boundary_counts["microbatches_with_bos"]
        / max(1, int(ar_boundary_counts["microbatches"])),
        "train_eos_token_count": int(ar_boundary_counts["eos_tokens"]),
        "train_eos_loss_token_count": int(ar_boundary_counts["eos_loss_tokens"]),
        "train_eos_token_fraction": ar_boundary_counts["eos_tokens"] / max(1, int(ar_boundary_counts["valid_tokens"])),
        "train_eos_loss_token_fraction": ar_boundary_counts["eos_loss_tokens"]
        / max(1, int(ar_boundary_counts["eos_tokens"])),
        "data_boundary_aware_packing": bool(config.data.boundary_aware_packing),
        "data_content_budget_segmented_packing": bool(config.data.content_budget_segmented_packing),
        "content_budget_segmented_packing_enabled": _uses_content_budget_segmented_packing(config),
        "content_budget_segmented_target_tokens": int(config.train.target_tokens),
        "content_budget_segmented_train_valid_tokens": int(train_valid_tokens)
        if _uses_content_budget_segmented_packing(config)
        else 0,
        "content_budget_segmented_train_content_tokens": int(train_content_tokens)
        if _uses_content_budget_segmented_packing(config)
        else 0,
        "train_row_count": int(row_telemetry["rows"]),
        "train_row_valid_tokens_mean": row_telemetry["valid_tokens_sum"] / max(1, int(row_telemetry["rows"])),
        "train_row_valid_tokens_min": int(row_telemetry["valid_tokens_min"]),
        "train_row_valid_tokens_max": int(row_telemetry["valid_tokens_max"]),
        "train_row_lossable_tokens_mean": row_telemetry["lossable_tokens_sum"] / max(1, int(row_telemetry["rows"])),
        "train_row_lossable_tokens_min": int(row_telemetry["lossable_tokens_min"]),
        "train_row_lossable_tokens_max": int(row_telemetry["lossable_tokens_max"]),
        "train_pad_token_count": int(row_telemetry["pad_tokens"]),
        "train_pad_fraction": row_telemetry["pad_tokens"] / max(1, int(row_telemetry["total_row_tokens"])),
        "train_row_eos_fraction": row_telemetry["eos_rows"] / max(1, int(row_telemetry["rows"])),
        "train_row_eos_tokens_mean": row_telemetry["eos_tokens"] / max(1, int(row_telemetry["rows"])),
        "train_row_eos_loss_tokens_mean": row_telemetry["eos_loss_tokens"] / max(1, int(row_telemetry["rows"])),
        "short_document_train_row_count": int(row_telemetry["rows"]) if config.data.short_document_curriculum else 0,
        "short_document_train_valid_tokens_mean": row_telemetry["valid_tokens_sum"] / max(1, int(row_telemetry["rows"]))
        if config.data.short_document_curriculum
        else 0.0,
        "short_document_train_valid_tokens_min": int(row_telemetry["valid_tokens_min"])
        if config.data.short_document_curriculum
        else 0,
        "short_document_train_valid_tokens_max": int(row_telemetry["valid_tokens_max"])
        if config.data.short_document_curriculum
        else 0,
        "short_document_train_lossable_tokens_mean": row_telemetry["lossable_tokens_sum"]
        / max(1, int(row_telemetry["rows"]))
        if config.data.short_document_curriculum
        else 0.0,
        "short_document_train_pad_fraction": row_telemetry["pad_tokens"]
        / max(1, int(row_telemetry["total_row_tokens"]))
        if config.data.short_document_curriculum
        else 0.0,
        "short_document_train_eos_row_fraction": row_telemetry["eos_rows"] / max(1, int(row_telemetry["rows"]))
        if config.data.short_document_curriculum
        else 0.0,
        "short_document_train_eos_tokens_mean": row_telemetry["eos_tokens"] / max(1, int(row_telemetry["rows"]))
        if config.data.short_document_curriculum
        else 0.0,
        "short_document_train_eos_loss_tokens_mean": row_telemetry["eos_loss_tokens"]
        / max(1, int(row_telemetry["rows"]))
        if config.data.short_document_curriculum
        else 0.0,
        "boundary_documents_per_row_mean": boundary_telemetry["documents_per_row_sum"]
        / max(1, boundary_telemetry["documents_per_row_count"]),
        "boundary_pad_fraction": row_telemetry["pad_tokens"] / max(1, int(row_telemetry["total_row_tokens"])),
        "boundary_cross_document_blocked_pairs": boundary_telemetry["blocked_pairs_sum"],
        "boundary_causal_pairs_total": boundary_telemetry["causal_pairs_sum"],
        "boundary_cross_document_blocked_fraction": boundary_telemetry["blocked_pairs_sum"]
        / max(1, boundary_telemetry["causal_pairs_sum"]),
        "segment_documents_per_row_mean": boundary_telemetry["documents_per_row_sum"]
        / max(1, boundary_telemetry["documents_per_row_count"])
        if _uses_content_budget_segmented_packing(config)
        else 0.0,
        "segment_pad_fraction": row_telemetry["pad_tokens"] / max(1, int(row_telemetry["total_row_tokens"]))
        if _uses_content_budget_segmented_packing(config)
        else 0.0,
        "segment_mask_valid_pairs_total": int(boundary_telemetry["valid_pairs_sum"])
        if _uses_content_budget_segmented_packing(config)
        else 0,
        "segment_mask_allowed_pairs_total": int(boundary_telemetry["allowed_pairs_sum"])
        if _uses_content_budget_segmented_packing(config)
        else 0,
        "segment_mask_blocked_pairs_total": int(boundary_telemetry["segment_blocked_pairs_sum"])
        if _uses_content_budget_segmented_packing(config)
        else 0,
        "segment_mask_allowed_fraction": boundary_telemetry["allowed_pairs_sum"]
        / max(1, boundary_telemetry["valid_pairs_sum"])
        if _uses_content_budget_segmented_packing(config)
        else 0.0,
        "segment_mask_blocked_fraction": boundary_telemetry["segment_blocked_pairs_sum"]
        / max(1, boundary_telemetry["valid_pairs_sum"])
        if _uses_content_budget_segmented_packing(config)
        else 0.0,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
        if config.runtime.device_type == "cuda"
        else 0.0,
        **_parameter_norm_metrics(raw_model),
        **early_residual_diagnostics,
    }
    if _uses_per_token_noise_levels(config):
        metrics.update(
            {
                "per_token_noise_levels_enabled": True,
                "per_token_noise_level_hist_bins": PER_TOKEN_NOISE_LEVEL_HIST_BINS,
                "per_token_noise_schedule_weight_source": "target_token_t",
                "per_token_noise_loss_normalization": "sum_ce_over_t_div_lossable_tokens",
                "per_token_noise_fixed_ce_eval_grid": "standard_scalar_t",
            }
        )
        metrics.update(
            _value_histogram_metrics(
                "last_train_per_token_noise_level_hist",
                last_per_token_noise_level_hist_counts,
                0.0,
                1.0,
            )
        )
        metrics.update(
            _value_histogram_metrics(
                "train_per_token_noise_level_hist",
                train_per_token_noise_level_hist_counts,
                0.0,
                1.0,
            )
        )
        metrics.update(
            _per_token_effective_masked_fraction_metrics(
                "last_train_",
                last_per_token_effective_masked_fraction_sum,
                last_per_token_effective_masked_fraction_count,
                last_per_token_effective_masked_fraction_min,
                last_per_token_effective_masked_fraction_max,
            )
        )
        metrics.update(
            _per_token_effective_masked_fraction_metrics(
                "train_",
                train_per_token_effective_masked_fraction_sum,
                train_per_token_effective_masked_fraction_count,
                train_per_token_effective_masked_fraction_min,
                train_per_token_effective_masked_fraction_max,
            )
        )
    if _uses_bd3lm(config):
        metrics["bd3lm_loss_target_tokens_last"] = last_active_tokens
        metrics["bd3lm_lossable_noised_half_tokens_last"] = last_lossable_tokens
    if _uses_block_causal_pilot(config):
        metrics["block_causal_pilot_size"] = int(config.model.block_causal_pilot_size)
        metrics["block_causal_pilot_num_blocks"] = (
            int(config.model.max_seq_len) + int(config.model.block_causal_pilot_size) - 1
        ) // int(config.model.block_causal_pilot_size)
        metrics["block_causal_pilot_enabled"] = True
        metrics["block_causal_pilot_not_true_bd3lm"] = True
        metrics["block_causal_pilot_visible_prefix_frac_mean"] = (
            block_causal_visible_prefix_frac_sum / block_causal_visible_prefix_count
            if block_causal_visible_prefix_count
            else 0.0
        )
    metrics.update(
        {
            "fixed_ce_trace_enabled": bool(config.eval.fixed_ce_trace_enabled),
            "fixed_ce_trace_artifact": "telemetry/fixed_noise_ce_trace.jsonl",
            "fixed_ce_trace_path": fixed_ce_trace_path,
            "fixed_ce_trace_steps": fixed_ce_trace_steps,
            "fixed_ce_trace_record_count": int(fixed_ce_trace_records),
            "fixed_ce_trace_batches": int(config.eval.fixed_ce_batches),
        }
    )
    for key, value in last_fixed_ce_trace_record.items():
        if isinstance(value, bool) or not isinstance(value, (float, int, str)):
            continue
        metric_key = f"last_{key}" if key.startswith("fixed_ce_trace_") else f"fixed_ce_trace_last_{key}"
        metrics[metric_key] = value
    if config.train.position_decile_balanced_loss:
        metrics.update(
            _position_decile_ce_accumulator_metrics(
                "last_train_",
                last_position_decile_ce_accumulator,
                True,
            )
        )
    if _uses_token_edit_corruption(config):
        metrics.update(
            _token_edit_reconstruction_metrics(
                "last_train_",
                last_token_edit_reconstruction_accumulator,
                True,
            )
        )
        metrics.update(
            _token_edit_reconstruction_metrics(
                "train_",
                train_token_edit_reconstruction_accumulator,
                True,
            )
        )
    if config.train.length_head_loss_weight > 0:
        train_length_prior_counts = (
            train_length_head_accumulator["target_counts"]
            if isinstance(train_length_head_accumulator.get("target_counts"), list)
            else None
        )
        metrics.update(
            _length_head_accumulator_metrics(
                "last_train_",
                last_length_head_accumulator,
                float(config.train.length_head_loss_weight),
            )
        )
        metrics.update(
            _length_head_accumulator_metrics(
                "train_",
                train_length_head_accumulator,
                float(config.train.length_head_loss_weight),
            )
        )
    else:
        train_length_prior_counts = None

    if rank == 0 and config.eval.enabled:
        eval_start = time.time()
        final_eval_metrics = _run_eval(
            raw_model,
            config,
            tokenizer,
            noise,
            autocast_ctx,
            device,
            total_steps,
            include_gen=config.eval.gen_eval,
        )
        final_eval_metrics["evaluation_seconds"] = time.time() - eval_start
        if config.train.length_head_loss_weight > 0:
            final_eval_metrics.update(
                _evaluate_length_head_metrics(
                    raw_model,
                    config,
                    tokenizer,
                    device,
                    autocast_ctx,
                    train_prior_counts=train_length_prior_counts,
                )
            )
        metrics.update(final_eval_metrics)
        append_jsonl(eval_log_path, final_eval_metrics)
        _write_tensorboard_scalars(tensorboard_writer, "eval", final_eval_metrics, total_steps)

    if rank == 0:
        saved_state = _rng_state(config)
        try:
            row_norm_metrics = _lm_head_row_norm_metrics(
                raw_model,
                tokenizer,
                config.data,
                device,
                int(config.train.device_batch_size),
                int(config.model.max_seq_len),
                batches=10,
            )
            metrics.update(row_norm_metrics)
        except Exception:
            pass
        _restore_rng_state(saved_state, config)

    if rank == 0:
        _publish_final_outputs(
            config,
            rank=rank,
            metrics=metrics,
            tensorboard_writer=tensorboard_writer,
            total_steps=total_steps,
            raw_model=raw_model,
            optimizers=optimizers,
            scaler=scaler,
            model_config=model_config,
            train_tokens=train_tokens,
            train_valid_tokens=train_valid_tokens,
            resume_contract=final_resume_contract,
        )

    if is_ddp:
        dist.barrier()

    return metrics


def _metric_label(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_")


def _load_source_train_config_dict(source_info: dict[str, str | list[str] | dict]) -> dict:
    source_run_dir = Path(str(source_info.get("source_run_dir", "")))
    resolved_config_path = source_run_dir / "resolved_config.json"
    raw_config = _load_json_if_exists(resolved_config_path)
    if raw_config:
        return raw_config

    checkpoint_path = str(source_info.get("source_checkpoint_path", source_info.get("source_checkpoint", "")))
    if not checkpoint_path:
        raise FileNotFoundError("comparison source has no checkpoint path")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_config = checkpoint.get("train_config", {})
    if not isinstance(raw_config, dict) or not raw_config:
        raise ValueError(f"checkpoint {checkpoint_path} does not contain a train_config for replay")
    return raw_config


def _source_replay_config(
    active_config: ExperimentConfig,
    source_info: dict[str, str | list[str] | dict],
) -> tuple[ExperimentConfig, list[str]]:
    raw_config = copy.deepcopy(_load_source_train_config_dict(source_info))
    normalizations: list[str] = []
    model_section = raw_config.setdefault("model", {})
    if isinstance(model_section, dict) and model_section.get("output_factorization") == "eos_hazard":
        model_section["output_factorization"] = "flat"
        normalizations.append("legacy_eos_hazard_output_factorization_to_flat")

    source_config = config_from_dict(raw_config)
    source_run_id = str(source_info.get("source_run_id", ""))
    source_config.runtime = copy.deepcopy(active_config.runtime)
    source_config.eval = copy.deepcopy(active_config.eval)
    source_config.train.run_dir = active_config.train.run_dir
    source_config.train.eval_only = True
    source_config.train.resume_from = "comparison"
    source_config.train.comparison_run_ids = [source_run_id] if source_run_id else []
    source_config.train.device_batch_size = int(active_config.train.device_batch_size)
    source_config.train.tensorboard = False
    source_config.train.tensorboard_dir = ""
    return source_config, normalizations


def _build_eval_model_config(config: ExperimentConfig, tokenizer) -> DiffusionConfig:
    return DiffusionConfig(
        sequence_len=config.model.max_seq_len,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=int(config.model.depth),
        n_head=int(config.model.n_head),
        n_embd=int(config.model.n_embd),
        mask_token_id=tokenizer.get_mask_token_id(),
        mlp_hidden_dim=int(config.model.mlp_hidden_dim),
        attention_scale=float(config.model.attention_scale),
        local_attention_lower_layers=int(config.model.local_attention_lower_layers),
        local_attention_window=int(config.model.local_attention_window),
        auxiliary_head_layers=tuple(int(i) for i in config.model.auxiliary_head_layers),
        time_conditioning=bool(config.noise.time_conditioning),
        bd3lm_block_size=int(config.model.bd3lm_block_size) if _uses_bd3lm(config) else 0,
        output_factorization=config.model.output_factorization,
        causal=bool(config.model.causal),
        residual_output_projection_init=config.model.residual_output_projection_init,
        embedding_head_init_std_multiplier=float(config.model.embedding_head_init_std_multiplier),
        length_head=bool(config.model.length_head),
        target_length_conditioning=bool(config.model.target_length_conditioning),
        mask_position_embedding=bool(config.model.mask_position_embedding),
    )


def _load_model_checkpoint_for_eval(
    path: str,
    raw_model: nn.Module,
    device,
    *,
    restore_rng_config: ExperimentConfig | None = None,
) -> tuple[int, int, int, dict, list[str]]:
    # Training checkpoints use PyTorch's zip serialization. Memory mapping lets
    # eval-only workers fault in the model tensors they copy to the model while
    # leaving optimizer tensor pages untouched.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state_dict = checkpoint["model_state_dict"]
    ignored_legacy_keys: list[str] = []
    model_keys = set(raw_model.state_dict().keys())
    legacy_eval_only_keys = ("stop_head.weight", "stop_head.bias")
    removable_keys = [key for key in legacy_eval_only_keys if key in state_dict and key not in model_keys]
    if removable_keys:
        state_dict = dict(state_dict)
        for key in removable_keys:
            state_dict.pop(key)
        ignored_legacy_keys = removable_keys
    raw_model.load_state_dict(state_dict)
    if restore_rng_config is not None:
        _restore_rng_state(checkpoint.get("rng_state", {}), restore_rng_config)
    checkpoint_metrics = dict(checkpoint.get("metrics", {})) if isinstance(checkpoint.get("metrics", {}), dict) else {}
    checkpoint_step = int(checkpoint.get("step", 0))
    checkpoint_train_tokens = int(checkpoint.get("train_tokens", 0))
    checkpoint_train_valid_tokens = int(
        checkpoint.get("train_valid_tokens", checkpoint_metrics.get("train_valid_tokens", 0))
    )
    del state_dict
    del checkpoint
    return (
        checkpoint_step,
        checkpoint_train_tokens,
        checkpoint_train_valid_tokens,
        checkpoint_metrics,
        ignored_legacy_keys,
    )


def _selected_basin_aggregate_key(key: str) -> bool:
    if not key.startswith("basin_depth_"):
        return False
    suffixes = (
        "_samples",
        "_eos_rate",
        "_exact_duplicate_rate",
        "_repeat_4gram_rate",
        "_distinct_1",
        "_distinct_2",
        "_distinct_3",
        "_max_token_frac_mean",
        "_max_token_fraction",
        "_token_entropy_mean",
        "_argmax_minus_nucleus_eos_rate",
        "_argmax_minus_nucleus_exact_duplicate_rate",
        "_argmax_minus_nucleus_repeat_4gram_rate",
        "_argmax_minus_nucleus_distinct_1",
        "_argmax_minus_nucleus_distinct_2",
        "_argmax_minus_nucleus_distinct_3",
        "_argmax_minus_nucleus_max_token_frac_mean",
        "_argmax_minus_nucleus_token_entropy_mean",
    )
    return any(key.endswith(suffix) for suffix in suffixes)


def eval_only_basin_replay(config: ExperimentConfig) -> dict:
    """Eval-only source replay for prefix-conditioned argmax-attractor diagnostics."""
    os.makedirs(config.train.run_dir, exist_ok=True)
    _configure_numeric_runtime(config)
    is_ddp, rank, _local_rank, world_size, device = init_distributed(
        config.runtime.device_type,
        config.train.run_dir,
        init_single_rank_process_group=config.runtime.init_single_rank_process_group,
    )
    _check_gpu_preflight(config, rank)
    compute_dtype = dtype_from_name(config.runtime.dtype)
    autocast_ctx = (
        torch.autocast(device_type=config.runtime.device_type, dtype=compute_dtype)
        if config.runtime.device_type == "cuda" and compute_dtype != torch.float32
        else nullcontext()
    )

    training_seed = config.runtime.seed if config.runtime.training_seed < 0 else config.runtime.training_seed
    torch.manual_seed(training_seed + rank)
    np.random.seed(training_seed + rank)
    random.seed(training_seed + rank)
    if config.runtime.device_type == "cuda":
        torch.cuda.manual_seed(training_seed + rank)

    if rank == 0:
        write_config(os.path.join(config.train.run_dir, "resolved_config.json"), config)

    sources = _resolve_comparison_sources(config)
    if not sources:
        raise ValueError("basin diagnostic requires train.comparison_run_ids")

    metrics: dict[str, object] = {
        **source_provenance_metrics(),
        "scale": "eval-only",
        "experiment_type": "eval-only-basin-diagnostic",
        "eval_only": True,
        "train_eval_only": bool(config.train.eval_only),
        "replay_metric_scope": "checkpoint-replay-argmax-basin-diagnostic",
        **run_config_summary(config, "eval-only"),
        "source_run_ids": [str(source.get("source_run_id", "")) for source in sources],
        "comparison_source_count": int(len(sources)),
        "basin_diagnostic": True,
        "basin_forced_argmax_depths": [int(d) for d in config.eval.basin_forced_argmax_depths],
        "basin_nucleus_top_p": float(config.eval.basin_nucleus_top_p),
        "basin_stochastic_policy": "nucleus_multinomial",
        "basin_prefix_len": int(config.eval.basin_prefix_len),
        "basin_gen_samples": int(config.eval.gen_samples),
        "basin_gen_len": int(config.eval.gen_len),
        "training_seconds": 0.0,
        "world_size": world_size,
        "device_batch_size": int(config.train.device_batch_size),
    }
    if metrics.get("status") != "ok":
        if rank == 0:
            write_json_atomic(os.path.join(config.train.run_dir, "metrics.json"), metrics)
        if is_ddp:
            dist.barrier()
            dist.destroy_process_group()
        return metrics

    aggregate_values: dict[str, list[float]] = {}
    source_val_nelbo_values: list[float] = []
    source_val_ce_values: list[float] = []

    for source_idx, source_info in enumerate(sources):
        source_run_id = str(source_info.get("source_run_id", ""))
        source_label = f"source_{source_idx:02d}_{_metric_label(source_run_id)}"
        source_config, normalizations = _source_replay_config(config, source_info)
        tokenizer = build_tokenizer(source_config)
        token_class_ids = None
        token_class_metadata: dict | None = None
        if _uses_hierarchical_output(source_config):
            saved_state = _rng_state(source_config)
            token_class_ids, token_class_metadata = build_frequency_token_classes(
                source_config.data,
                tokenizer,
                int(source_config.train.device_batch_size),
                int(source_config.model.max_seq_len),
                device,
                int(source_config.model.hierarchical_frequency_batches),
            )
            _restore_rng_state(saved_state, source_config)

        model_config = _build_eval_model_config(source_config, tokenizer)
        raw_model = DiffusionTransformer(
            model_config,
            rope_base=source_config.model.rope_base,
            compute_dtype=compute_dtype,
            pad_vocab_to=source_config.model.pad_vocab_to,
            token_class_ids=token_class_ids,
        ).to(device)
        checkpoint_path = str(source_info.get("source_checkpoint_path", source_info.get("source_checkpoint", "")))
        (
            checkpoint_step,
            checkpoint_train_tokens,
            checkpoint_train_valid_tokens,
            checkpoint_metrics,
            ignored_checkpoint_keys,
        ) = _load_model_checkpoint_for_eval(checkpoint_path, raw_model, device)
        raw_model.eval()
        noise = build_noise(source_config.noise, model_config.mask_token_id, model_config.vocab_size)
        source_eos_token_id = int(tokenizer.get_eos_token_id())

        def model_logits_fn(
            m,
            z_eval,
            x_eval,
            attention_mask,
            noise_level=None,
            position_ids=None,
            target_lengths=None,
            source_config=source_config,
            eos_token_id=source_eos_token_id,
        ):
            if target_lengths is None:
                target_lengths = _target_lengths_for_conditioning(
                    source_config,
                    x_eval,
                    attention_mask,
                    eos_token_id,
                )
            return _forward_eval_logits(
                m,
                source_config,
                z_eval,
                x_eval,
                attention_mask,
                noise_level=noise_level,
                position_ids=position_ids,
                target_lengths=target_lengths,
            )

        source_metrics = source_info.get("source_metrics", {})
        if not isinstance(source_metrics, dict):
            source_metrics = {}
        source_val_nelbo = source_metrics.get("val_nelbo_bpb", checkpoint_metrics.get("val_nelbo_bpb"))
        source_val_ce = source_metrics.get("val_ce", checkpoint_metrics.get("val_ce"))
        if isinstance(source_val_nelbo, (int, float)):
            source_val_nelbo_values.append(float(source_val_nelbo))
        if isinstance(source_val_ce, (int, float)):
            source_val_ce_values.append(float(source_val_ce))

        with autocast_ctx:
            diag_start = time.time()
            basin_metrics = evaluate_argmax_basin_diagnostic(
                raw_model,
                tokenizer,
                source_config.data,
                int(source_config.train.device_batch_size),
                device=device,
                seq_len=int(source_config.model.max_seq_len),
                gen_samples=int(config.eval.gen_samples),
                gen_len=int(config.eval.gen_len),
                prefix_len=int(config.eval.basin_prefix_len),
                forced_argmax_depths=config.eval.basin_forced_argmax_depths,
                temperature=float(config.eval.gen_temperature),
                nucleus_top_p=float(config.eval.basin_nucleus_top_p),
                is_causal=_is_causal(source_config),
                run_dir=config.train.run_dir if rank == 0 else None,
                log_samples=int(config.eval.gen_log_samples),
                source_run_id=source_run_id,
                noise=noise,
                model_logits_fn=model_logits_fn,
            )
            basin_seconds = time.time() - diag_start

        source_prefix = f"{source_label}_"
        source_metadata: dict[str, object] = {
            f"{source_prefix}run_id": source_run_id,
            f"{source_prefix}checkpoint": checkpoint_path,
            f"{source_prefix}artifacts_manifest": str(source_info.get("source_artifacts_manifest", "")),
            f"{source_prefix}metrics_path": str(source_info.get("source_metrics_path", "")),
            f"{source_prefix}checkpoint_step": int(checkpoint_step),
            f"{source_prefix}checkpoint_train_tokens": int(checkpoint_train_tokens),
            f"{source_prefix}checkpoint_train_valid_tokens": int(checkpoint_train_valid_tokens),
            f"{source_prefix}source_val_nelbo_bpb": float(source_val_nelbo)
            if isinstance(source_val_nelbo, (int, float))
            else 0.0,
            f"{source_prefix}source_val_ce": float(source_val_ce) if isinstance(source_val_ce, (int, float)) else 0.0,
            f"{source_prefix}source_training_seconds": float(source_metrics.get("training_seconds", 0.0))
            if isinstance(source_metrics.get("training_seconds", 0.0), (int, float))
            else 0.0,
            f"{source_prefix}training_seconds": 0.0,
            f"{source_prefix}diagnostic_seconds": float(basin_seconds),
            f"{source_prefix}model_depth": int(source_config.model.depth),
            f"{source_prefix}model_n_embd": int(source_config.model.n_embd),
            f"{source_prefix}model_n_head": int(source_config.model.n_head),
            f"{source_prefix}model_causal": bool(source_config.model.causal),
            f"{source_prefix}training_corruption": _training_corruption_label(source_config),
            f"{source_prefix}noise_type": source_config.noise.type,
            f"{source_prefix}noise_time_conditioning": bool(source_config.noise.time_conditioning),
            f"{source_prefix}noise_complement_masking": bool(source_config.noise.complement_masking),
            f"{source_prefix}tokenizer_path": source_config.data.tokenizer_path
            if source_config.data.tokenizer == "hf"
            else "",
            f"{source_prefix}config_normalizations": normalizations,
            f"{source_prefix}ignored_checkpoint_keys": ignored_checkpoint_keys,
            f"{source_prefix}output_factorization": source_config.model.output_factorization,
            f"{source_prefix}output_class_count": int(
                token_class_metadata.get("output_class_count", 0) if token_class_metadata else 0
            ),
        }
        metrics.update(source_metadata)
        for key, value in basin_metrics.items():
            metrics[f"{source_prefix}{key}"] = value
            if isinstance(value, (int, float)) and _selected_basin_aggregate_key(key):
                aggregate_values.setdefault(key, []).append(float(value))

        del raw_model, noise, tokenizer
        gc.collect()
        if config.runtime.device_type == "cuda":
            torch.cuda.empty_cache()

    for key, values in aggregate_values.items():
        metrics[f"source_mean_{key}"] = sum(values) / len(values) if values else 0.0
    if source_val_nelbo_values:
        metrics["source_val_nelbo_bpb_mean"] = sum(source_val_nelbo_values) / len(source_val_nelbo_values)
        metrics["val_nelbo_bpb"] = metrics["source_val_nelbo_bpb_mean"]
        metrics["val_nelbo_bpb_is_source_copy"] = True
    else:
        metrics["val_nelbo_bpb"] = 0.0
        metrics["val_nelbo_bpb_is_source_copy"] = False
    if source_val_ce_values:
        metrics["source_val_ce_mean"] = sum(source_val_ce_values) / len(source_val_ce_values)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if config.runtime.device_type == "cuda" else 0.0
    metrics["peak_vram_mb"] = peak_vram_mb

    if rank == 0:
        write_json_atomic(os.path.join(config.train.run_dir, "metrics.json"), metrics)
        print0(rank, f"metrics saved to {os.path.join(config.train.run_dir, 'metrics.json')}")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()

    return metrics


def _merge_eval_only_replay_nelbo_metrics(
    metrics: dict,
    replay_metrics: dict,
    source_val_nelbo_bpb: float,
) -> None:
    """Keep source-checkpoint metrics separate from newly replayed NELBO semantics."""

    metrics.update(replay_metrics)
    replay_metric_keys = [
        key
        for key in replay_metrics
        if key.startswith("val_nelbo") or key in {"val_masked_ce_bpb", "hierarchical_val_nelbo_bpb"}
    ]
    for key in replay_metric_keys:
        metrics.pop(key, None)
        metrics[f"eval_only_replay_{key}"] = replay_metrics[key]
    metrics["source_val_nelbo_bpb"] = float(source_val_nelbo_bpb)
    metrics["val_nelbo_bpb"] = float(source_val_nelbo_bpb)
    metrics["val_nelbo_metric_origin"] = "source_checkpoint_metrics"


def eval_only_replay(config: ExperimentConfig) -> dict:
    """Eval-only checkpoint replay: load a checkpoint and run the configured eval bundle."""
    if config.eval.basin_diagnostic:
        return eval_only_basin_replay(config)

    os.makedirs(config.train.run_dir, exist_ok=True)
    _configure_numeric_runtime(config)
    is_ddp, rank, local_rank, world_size, device = init_distributed(
        config.runtime.device_type,
        config.train.run_dir,
        init_single_rank_process_group=config.runtime.init_single_rank_process_group,
    )
    _check_gpu_preflight(config, rank)
    compute_dtype = dtype_from_name(config.runtime.dtype)
    autocast_ctx = (
        torch.autocast(device_type=config.runtime.device_type, dtype=compute_dtype)
        if config.runtime.device_type == "cuda" and compute_dtype != torch.float32
        else nullcontext()
    )

    training_seed = config.runtime.seed if config.runtime.training_seed < 0 else config.runtime.training_seed
    torch.manual_seed(training_seed + rank)
    np.random.seed(training_seed + rank)
    random.seed(training_seed + rank)
    if config.runtime.device_type == "cuda":
        torch.cuda.manual_seed(training_seed + rank)

    tokenizer = build_tokenizer(config)
    data_identity_summary = _prepare_data_identity(
        config,
        rank=rank,
        world_size=world_size,
        is_ddp=is_ddp,
    )
    token_class_ids = None
    token_class_metadata: dict | None = None
    if _uses_hierarchical_output(config):
        saved_state = _rng_state(config)
        token_class_ids, token_class_metadata = build_frequency_token_classes(
            config.data,
            tokenizer,
            int(config.train.device_batch_size),
            int(config.model.max_seq_len),
            device,
            int(config.model.hierarchical_frequency_batches),
        )
        _restore_rng_state(saved_state, config)
    model_config = DiffusionConfig(
        sequence_len=config.model.max_seq_len,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=int(config.model.depth),
        n_head=int(config.model.n_head),
        n_embd=int(config.model.n_embd),
        mask_token_id=tokenizer.get_mask_token_id(),
        mlp_hidden_dim=int(config.model.mlp_hidden_dim),
        attention_scale=float(config.model.attention_scale),
        local_attention_lower_layers=int(config.model.local_attention_lower_layers),
        local_attention_window=int(config.model.local_attention_window),
        auxiliary_head_layers=tuple(int(i) for i in config.model.auxiliary_head_layers),
        time_conditioning=bool(config.noise.time_conditioning),
        bd3lm_block_size=int(config.model.bd3lm_block_size) if _uses_bd3lm(config) else 0,
        output_factorization=config.model.output_factorization,
        causal=bool(config.model.causal),
        residual_output_projection_init=config.model.residual_output_projection_init,
        embedding_head_init_std_multiplier=float(config.model.embedding_head_init_std_multiplier),
        length_head=bool(config.model.length_head),
        target_length_conditioning=bool(config.model.target_length_conditioning),
        mask_position_embedding=bool(config.model.mask_position_embedding),
    )

    saved_state = _rng_state(config)
    model_seed = config.runtime.seed if config.runtime.model_seed < 0 else config.runtime.model_seed
    torch.manual_seed(model_seed)
    if config.runtime.device_type == "cuda":
        torch.cuda.manual_seed(model_seed)
    raw_model = DiffusionTransformer(
        model_config,
        rope_base=config.model.rope_base,
        compute_dtype=compute_dtype,
        pad_vocab_to=config.model.pad_vocab_to,
        token_class_ids=token_class_ids,
    ).to(device)
    _restore_rng_state(saved_state, config)

    counts = raw_model.parameter_counts()
    flops_per_token = raw_model.estimate_flops()
    parameter_accounting = _parameter_accounting_summary(config, counts, raw_model.parameter_inventory())
    total_batch_size, grad_accum_steps, total_steps, _batch_lr_scale = _training_budget(
        config, counts, world_size, raw_model=raw_model
    )

    print0(
        rank,
        json.dumps(
            {
                "model": asdict(model_config),
                "param_counts": counts,
                "parameter_accounting": parameter_accounting,
                "flops_per_token": flops_per_token,
                "total_batch_size": total_batch_size,
                "effective_batch_size_tokens": total_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "total_steps": total_steps,
                "target_tokens": config.train.target_tokens,
                "world_size": world_size,
                "compute_dtype": str(compute_dtype),
                "eval_only": True,
            },
            indent=2,
        ),
    )
    if rank == 0:
        write_config(os.path.join(config.train.run_dir, "resolved_config.json"), config)

    resume_metrics: dict = {}
    resume_path, source_info = _resolve_eval_only_resume(config)
    if resume_path is None:
        raise ValueError(
            "eval_only_replay requires train.resume_from or train.comparison_run_ids to point to a checkpoint"
        )
    checkpoint_identity_sha256 = ""
    checkpoint_identity_size_bytes = -1
    checkpoint_identity_seconds = 0.0
    if config.eval.source_checkpoint_identity_mode == "sha256":
        checkpoint_identity_start = time.time()
        checkpoint_identity_payload: list[object] = ["", -1]
        if rank == 0:
            checkpoint_identity_payload[:] = _source_checkpoint_identity(resume_path)
        if is_ddp:
            dist.broadcast_object_list(checkpoint_identity_payload, src=0)
        checkpoint_identity_sha256 = str(checkpoint_identity_payload[0])
        checkpoint_identity_size_bytes = int(checkpoint_identity_payload[1])
        checkpoint_identity_seconds = time.time() - checkpoint_identity_start
        if checkpoint_identity_sha256 != config.eval.source_checkpoint_expected_sha256:
            raise ValueError(
                "source checkpoint SHA-256 mismatch: "
                f"observed {checkpoint_identity_sha256}, expected {config.eval.source_checkpoint_expected_sha256}"
            )
        if checkpoint_identity_size_bytes != config.eval.source_checkpoint_expected_size_bytes:
            raise ValueError(
                "source checkpoint size mismatch: "
                f"observed {checkpoint_identity_size_bytes}, "
                f"expected {config.eval.source_checkpoint_expected_size_bytes}"
            )
    checkpoint_load_start = time.time()
    (
        checkpoint_step,
        checkpoint_train_tokens,
        checkpoint_train_valid_tokens,
        resume_metrics,
        ignored_checkpoint_keys,
    ) = _load_model_checkpoint_for_eval(
        resume_path,
        raw_model,
        device,
        restore_rng_config=config,
    )
    checkpoint_load_seconds = time.time() - checkpoint_load_start
    checkpoint_load_peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if config.runtime.device_type == "cuda" else 0.0
    )
    if config.runtime.device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print0(rank, f"eval_only_replay loaded checkpoint {resume_path}")

    noise = build_noise(config.noise, model_config.mask_token_id, model_config.vocab_size)

    # Run standard non-generation eval bundle once
    base_metrics: dict[str, float | int | str] = {}
    if config.eval.enabled:
        eval_start = time.time()
        base_metrics = _run_eval(
            raw_model,
            config,
            tokenizer,
            noise,
            autocast_ctx,
            device,
            step=0,
            include_gen=False,
        )
        base_metrics["evaluation_seconds"] = time.time() - eval_start

    source_metrics = source_info.get("source_metrics", {})
    if not isinstance(source_metrics, dict):
        source_metrics = {}
    source_val_nelbo_bpb = float(
        source_metrics.get("val_nelbo_bpb", resume_metrics.get("val_nelbo_bpb", base_metrics.get("val_nelbo_bpb", 0.0)))
    )
    source_val_ce = source_metrics.get("val_ce", resume_metrics.get("val_ce"))
    replay_val_ce = base_metrics.get("val_ce")
    source_train_tokens = int(
        source_metrics.get("train_tokens", source_metrics.get("total_tokens", checkpoint_train_tokens))
    )

    def model_logits_fn(
        m,
        z_eval,
        x_eval,
        attention_mask,
        noise_level=None,
        position_ids=None,
        target_lengths=None,
    ):
        if target_lengths is None:
            target_lengths = _target_lengths_for_conditioning(
                config,
                x_eval,
                attention_mask,
                tokenizer.get_eos_token_id(),
            )
        return _forward_eval_logits(
            m,
            config,
            z_eval,
            x_eval,
            attention_mask,
            noise_level=noise_level,
            position_ids=position_ids,
            target_lengths=target_lengths,
        )

    if config.eval.conditional_gen_eval and not any(k.startswith("conditional_gen_p") for k in base_metrics):
        if config.noise.type != "mdlm":
            raise ValueError("eval.conditional_gen_eval currently supports noise.type='mdlm'")

        def conditional_sample_fn(m, prefixes, continuation_len):
            target_lengths = None
            if _uses_target_length_conditioning(config):
                total_len = int(prefixes.shape[1]) + int(continuation_len)
                target_lengths = torch.full(
                    (int(prefixes.shape[0]),),
                    total_len,
                    dtype=torch.long,
                    device=prefixes.device,
                )
            return sample_mdlm_from_prefixes(
                m,
                noise,
                tokenizer,
                prefixes,
                int(continuation_len),
                config.eval.sample_steps,
                scoring_mode=config.eval.sample_scoring_mode,
                token_selection=config.eval.sample_token_selection,
                temperature=config.eval.gen_temperature,
                jitter_scale=config.eval.sample_jitter_scale,
                return_trace=config.eval.sample_trace,
                target_lengths=target_lengths,
            )

        with autocast_ctx:
            conditional_start = time.time()
            conditional_metrics = evaluate_conditional_generation_metrics(
                raw_model,
                tokenizer,
                config.data,
                noise,
                conditional_sample_fn,
                int(config.train.device_batch_size),
                config.train.run_dir,
                device=device,
                seq_len=config.model.max_seq_len,
                prefix_fracs=config.eval.conditional_gen_prefix_fracs,
                gen_samples=config.eval.conditional_gen_samples,
                continuation_len=config.eval.conditional_gen_len,
                log_samples=config.eval.conditional_gen_log_samples,
                score_mode=config.eval.gen_score_mode,
                self_score_t=config.eval.gen_self_score_t,
                step=0,
                sample_label=config.eval.sample_token_selection,
                model_logits_fn=model_logits_fn,
            )
        conditional_metrics["conditional_gen_seconds"] = time.time() - conditional_start
        base_metrics.update(conditional_metrics)

    if config.eval.gen_eval:
        temperatures = [0.5, 0.8, 1.0]

        def _make_sample_fn(sampler_name: str, fixed_temp: float, tau_edit: float | None = None):
            def sample_fn(m, n, length, temperature=fixed_temp):
                if _is_causal(config):
                    return sample_ar(
                        m,
                        tokenizer,
                        n,
                        length,
                        temperature=temperature,
                        token_selection=config.eval.sample_token_selection,
                    )
                if _uses_bd3lm(config):
                    return sample_bd3lm(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        block_size=int(config.model.bd3lm_block_size),
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=config.eval.sample_token_selection,
                        temperature=temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        return_trace=config.eval.sample_trace,
                    )
                if sampler_name == "remdm":
                    return sample_remdm(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=config.eval.sample_token_selection,
                        temperature=temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        tau_edit=tau_edit if tau_edit is not None else config.eval.remdm_tau_edit,
                        return_trace=config.eval.sample_trace,
                    )
                if sampler_name == "fhs":
                    return sample_fhs(
                        m,
                        noise,
                        tokenizer,
                        n,
                        length,
                        config.eval.sample_steps,
                        scoring_mode=config.eval.sample_scoring_mode,
                        token_selection=config.eval.sample_token_selection,
                        temperature=temperature,
                        jitter_scale=config.eval.sample_jitter_scale,
                        tau=config.eval.fhs_tau,
                        length_policy=config.eval.length_policy,
                        return_trace=config.eval.sample_trace,
                    )
                sample_impl = {
                    "mdlm": sample_mdlm,
                    "uniform": sample_uniform,
                    "mask_uniform": sample_mask_uniform,
                }[config.noise.type]
                return sample_impl(
                    m,
                    noise,
                    tokenizer,
                    n,
                    length,
                    config.eval.sample_steps,
                    scoring_mode=config.eval.sample_scoring_mode,
                    token_selection=config.eval.sample_token_selection,
                    temperature=temperature,
                    jitter_scale=config.eval.sample_jitter_scale,
                    return_trace=config.eval.sample_trace,
                )

            return sample_fn

        def _run_generation_pass(sampler_name: str, label_prefix: str, tau_edit: float | None = None):
            for temp in temperatures:
                sample_fn = _make_sample_fn(sampler_name, fixed_temp=temp, tau_edit=tau_edit)
                with autocast_ctx:
                    gen_start = time.time()
                    gen_metrics = evaluate_generation_metrics(
                        raw_model,
                        tokenizer,
                        noise,
                        sample_fn,
                        int(config.train.device_batch_size),
                        config.train.run_dir,
                        device=device,
                        judge_model_id=config.eval.judge_model_id,
                        gen_samples=config.eval.gen_samples,
                        gen_len=config.eval.gen_len,
                        log_samples=config.eval.gen_log_samples,
                        score_mode=config.eval.gen_score_mode,
                        self_score_t=config.eval.gen_self_score_t,
                        step=0,
                        sample_label=f"{label_prefix}temp{int(temp * 10):02d}",
                        model_logits_fn=model_logits_fn,
                    )
                    gen_seconds = time.time() - gen_start

                temp_key = f"{label_prefix}temp{int(temp * 10):02d}"
                if temp == 1.0 and label_prefix == "":
                    base_metrics.update(gen_metrics)
                elif temp == 1.0:
                    for k, v in gen_metrics.items():
                        if isinstance(v, (int, float)):
                            base_metrics[f"gen_{label_prefix}{k[4:] if k.startswith('gen_') else k}"] = v
                for k, v in gen_metrics.items():
                    if isinstance(v, (int, float)):
                        base_metrics[f"gen_{temp_key}_{k[4:] if k.startswith('gen_') else k}"] = v
                base_metrics[f"gen_{temp_key}_seconds"] = gen_seconds

        _run_generation_pass(config.eval.sampler, "")
        if config.eval.remasking_comparison:
            _run_generation_pass("remdm", "remask_", tau_edit=config.eval.remdm_tau_edit)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if config.runtime.device_type == "cuda" else 0.0

    metrics = {
        **source_provenance_metrics(),
        "scale": "eval-only",
        "experiment_type": "eval-only",
        "eval_only": True,
        "train_eval_only": bool(config.train.eval_only),
        "replay_metric_scope": "checkpoint-replay-diagnostic",
        **run_config_summary(config, "eval-only"),
        "source_run_ids": source_info.get("source_run_ids", config.train.comparison_run_ids),
        "source_run_id": str((source_info.get("source_run_ids", config.train.comparison_run_ids) or [""])[0]),
        "source_checkpoint": source_info.get("source_checkpoint", resume_path),
        "source_checkpoint_path": source_info.get(
            "source_checkpoint_path", source_info.get("source_checkpoint", resume_path)
        ),
        "source_checkpoint_identity_mode": config.eval.source_checkpoint_identity_mode,
        "source_checkpoint_sha256": checkpoint_identity_sha256 or None,
        "source_checkpoint_size_bytes": checkpoint_identity_size_bytes if checkpoint_identity_size_bytes >= 0 else None,
        "source_checkpoint_identity_seconds": float(checkpoint_identity_seconds),
        "eval_only_checkpoint_load_mode": "model_state_only_cpu_mmap",
        "eval_only_checkpoint_load_seconds": float(checkpoint_load_seconds),
        "eval_only_checkpoint_load_peak_vram_mb": float(checkpoint_load_peak_vram_mb),
        "eval_only_checkpoint_rng_restored": True,
        "eval_only_ignored_checkpoint_keys": ignored_checkpoint_keys,
        "source_artifacts_manifest": source_info.get("source_artifacts_manifest", ""),
        "source_metrics_path": source_info.get("source_metrics_path", ""),
        "source_val_nelbo_bpb": source_val_nelbo_bpb,
        "source_val_ce": float(source_val_ce) if isinstance(source_val_ce, (int, float)) else 0.0,
        "val_nelbo_bpb": source_val_nelbo_bpb,
        "val_nelbo_metric_origin": "source_checkpoint_metrics",
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_train_tokens": int(checkpoint_train_tokens),
        "checkpoint_train_valid_tokens": int(checkpoint_train_valid_tokens),
        "source_train_tokens": source_train_tokens,
        "replay_train_tokens_start": int(checkpoint_train_tokens),
        "replay_train_tokens_end": int(checkpoint_train_tokens),
        "replay_train_tokens_delta": 0,
        "sampler": config.eval.sampler,
        "sample_steps": config.eval.sample_steps,
        "sample_scoring_mode": config.eval.sample_scoring_mode,
        "sample_token_selection": config.eval.sample_token_selection,
        "sample_jitter_scale": config.eval.sample_jitter_scale,
        "remdm_tau_edit": config.eval.remdm_tau_edit,
        "remasking_comparison": config.eval.remasking_comparison,
        "model_depth": config.model.depth,
        "model_n_embd": config.model.n_embd,
        "model_n_head": config.model.n_head,
        **_noise_telemetry(config),
        **_bd3lm_telemetry(config),
        **_hierarchical_output_telemetry(token_class_metadata),
        **_residual_projection_init_metrics(config),
        **_embedding_head_init_metrics(config),
        "training_corruption": _training_corruption_label(config),
        "tokenizer": config.data.tokenizer,
        "data_identity_enabled": bool(data_identity_summary["enabled"]),
        "data_identity_mode": str(data_identity_summary["mode"]),
        "data_identity_artifact": str(data_identity_summary["artifact"]),
        "data_identity_sha256": str(data_identity_summary["identity_sha256"]),
        "data_identity_sha256_strength": str(data_identity_summary["identity_sha256_strength"]),
        "target_tokens": config.train.target_tokens,
        "target_param_count_basis": config.train.target_param_count_basis,
        "target_param_count": _selected_target_param_count(config, counts),
        "tokens_per_target_param": checkpoint_train_tokens / max(1, _selected_target_param_count(config, counts)),
        "parameter_accounting": parameter_accounting,
        "vocab_size": model_config.vocab_size,
        "mask_token_id": model_config.mask_token_id,
        "eos_token_id": tokenizer.get_eos_token_id(),
        "num_params_M": counts["total"] / 1e6,
        "num_non_embedding_params_M": counts["non_embedding"] / 1e6,
        "num_scaling_params_M": counts["scaling"] / 1e6,
        "num_wte_params_M": counts["wte"] / 1e6,
        "num_lm_head_params_M": counts["lm_head"] / 1e6,
        "num_class_head_params_M": counts["class_head"] / 1e6,
        "num_length_head_params_M": counts["length_head"] / 1e6,
        "num_auxiliary_head_params_M": counts["auxiliary_heads"] / 1e6,
        "num_transformer_params_M": counts["transformer"] / 1e6,
        **_token_ratio_metrics(checkpoint_train_tokens, counts),
        "flops_per_token": flops_per_token,
        "training_seconds": 0.0,
        "world_size": world_size,
        "total_batch_size": total_batch_size,
        "device_batch_size": config.train.device_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "peak_vram_mb": peak_vram_mb,
    }
    _merge_eval_only_replay_nelbo_metrics(metrics, base_metrics, source_val_nelbo_bpb)
    if isinstance(replay_val_ce, int | float):
        metrics["eval_only_replay_val_ce"] = float(replay_val_ce)

    if rank == 0:
        write_json_atomic(os.path.join(config.train.run_dir, "metrics.json"), metrics)
        print0(rank, f"metrics saved to {os.path.join(config.train.run_dir, 'metrics.json')}")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()

    return metrics
