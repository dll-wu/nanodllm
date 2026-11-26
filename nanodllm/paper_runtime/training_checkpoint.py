from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .checkpoint_load_binding import load_checkpoint_with_optional_binding
from .config import ExperimentConfig, to_dict
from .data import DLMDataBatch
from .parameter_state_hash import (
    parameter_state_hash_record,
    validate_checkpoint_parameter_state_hash,
    validate_resume_parameter_state_hashes,
)
from .parameterization import validate_checkpoint_parameterization_contract
from .source_identity import source_implementation_identity as _source_implementation_identity
from .telemetry_config import TELEMETRY_OPERATIONAL_FIELDS
from .training_checkpoint_paths import (
    _checkpoint_step as _checkpoint_step,
)
from .training_checkpoint_paths import (
    _latest_checkpoint_in_run_dir as _latest_checkpoint_in_run_dir,
)
from .training_checkpoint_paths import (
    _resolve_resume_path as _resolve_resume_path,
)
from .training_checkpoint_paths import (
    _step_checkpoint_candidates as _step_checkpoint_candidates,
)
from .training_checkpoint_paths import (
    periodic_path as periodic_path,
)
from .training_optim import _load_optimizer_state_dict, _optimizer_state_dict
from .training_progress import RESUME_TRAINER_STATE_VERSION, TrainerProgress
from .training_rng import (
    _active_rank_rng_state,
    _legacy_rng_state_from_active,
    _rng_state,
    _validate_active_rank_rng_topology,
    _validate_rank_rng_payload,
)

RESUME_CONTRACT_VERSION = 1
RESUME_BATCH_DIGEST_VERSION = 1
RESUME_TRAJECTORY_VERSION = 1
RESUME_BATCH_TENSOR_FIELDS = (
    "x_0",
    "loss_mask",
    "attention_mask",
    "position_ids",
    "document_boundary_mask",
    "document_start_mask",
)


def runtime_compatibility_identity(config: ExperimentConfig, device: torch.device) -> dict[str, object]:
    """Return the canonical runtime identity bound into exact-resume checkpoints."""

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


def _validated_trajectory_record(trajectory: object) -> dict[str, object]:
    if not isinstance(trajectory, dict) or trajectory.get("version") != RESUME_TRAJECTORY_VERSION:
        raise RuntimeError("checkpoint has no supported trajectory contract")
    record = trajectory.get("record")
    if not isinstance(record, dict):
        raise RuntimeError("checkpoint trajectory contract has no canonical record")
    recomputed_sha256 = _trajectory_record_sha256(record)
    if trajectory.get("sha256") != recomputed_sha256:
        raise RuntimeError(
            "checkpoint trajectory record hash mismatch: "
            f"stored={trajectory.get('sha256')}, recomputed={recomputed_sha256}"
        )
    return record


def validate_checkpoint_runtime_compatibility(
    checkpoint: Mapping[str, object],
    expected: Mapping[str, object],
    expected_implementation_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate and return a checkpoint's canonical exact-resume runtime identity."""

    resume_contract = checkpoint.get("resume_contract")
    if not isinstance(resume_contract, dict):
        raise RuntimeError("checkpoint has no versioned resume contract")
    record = _validated_trajectory_record(resume_contract.get("trajectory"))
    if expected_implementation_identity is not None and record.get("implementation_identity") != dict(
        expected_implementation_identity
    ):
        raise RuntimeError(
            "checkpoint implementation identity mismatch: "
            f"checkpoint={record.get('implementation_identity')}, expected={dict(expected_implementation_identity)}"
        )
    observed = record.get("runtime_compatibility")
    if not isinstance(observed, dict):
        raise RuntimeError("checkpoint trajectory record has no runtime compatibility identity")
    if observed != dict(expected):
        raise RuntimeError(
            f"checkpoint runtime compatibility mismatch: checkpoint={observed}, expected={dict(expected)}"
        )
    return copy.deepcopy(observed)


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
            "model_init_from",
            "token_edit_draft_checkpoint",
            "comparison_run_ids",
            *TELEMETRY_OPERATIONAL_FIELDS,
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
        "runtime_compatibility": runtime_compatibility_identity(config, device),
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
                "model_init_from",
                "comparison_run_ids",
                *TELEMETRY_OPERATIONAL_FIELDS,
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


def _validate_trainer_state_schema(state: dict[str, object], config: ExperimentConfig) -> None:
    TrainerProgress.from_checkpoint(state, max_seq_len=int(config.model.max_seq_len))


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
        if config is not None and config.train.model_init_from:
            raise RuntimeError(
                "a parent-initialized child run cannot resume a legacy checkpoint without a versioned trajectory contract"
            )
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
    _validated_trajectory_record(saved_trajectory)
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
    diagnostic_only_unbound: bool = False,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if diagnostic_only_unbound:
        if resume_contract is not None or not config.train.strict_mdlm_learned_eos_sampler_canary_only:
            raise ValueError("unbound diagnostic checkpoints are restricted to the learned-EOS sampler canary")
        train_config = to_dict(config)
        parameterization = train_config.pop("parameterization", None)
        if not isinstance(parameterization, dict):
            raise ValueError("unbound diagnostic checkpoint requires an explicit ordinary parameterization")
        binding = parameterization.get("contract_binding")
        digest = parameterization.get("contract_sha256")
        if not ((binding == "legacy_unbound" and digest == "") or (binding == "required" and digest)):
            raise ValueError("unbound diagnostic checkpoint has an invalid parameterization binding")
    else:
        if config.parameterization.contract_sha256 and resume_contract is None:
            raise ValueError("a bound parameterization checkpoint requires a trajectory-bearing resume contract")
        train_config = to_dict(config)
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
        "train_config": train_config,
        "step": step,
        "train_tokens": train_tokens,
        "train_valid_tokens": train_valid_tokens,
        "metrics": metrics,
    }
    if config.diagnostics.parameter_state_hash_steps:
        architecture_fingerprint = raw_model.architecture_fingerprint()
        payload["parameter_state_hash"] = parameter_state_hash_record(
            raw_model,
            step=step,
            train_tokens=train_tokens,
            architecture_fingerprint_sha256=str(architecture_fingerprint["fingerprint_sha256"]),
            parameterization_contract_sha256=config.parameterization.contract_sha256,
        )
    if diagnostic_only_unbound:
        payload["parameterization_status"] = "deliberately_unbound_diagnostic_only_nonloadable"
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
        if path.is_file() and path.suffix == ".pt":
            return path

    candidate = _latest_checkpoint_in_run_dir(source_run_dir)
    if candidate is not None:
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
    checkpoint_load_receipt: dict[str, object] | None = None,
) -> tuple[int, int, int, dict, dict[str, object]]:
    ckpt = load_checkpoint_with_optional_binding(
        path,
        map_location=device,
        weights_only=False,
        receipt=checkpoint_load_receipt,
    )
    checkpoint_parameterization = validate_checkpoint_parameterization_contract(ckpt, raw_model)
    checkpoint_parameterization_sha256 = (
        str(checkpoint_parameterization.get("contract_sha256")) if checkpoint_parameterization is not None else ""
    )
    if checkpoint_parameterization_sha256 != config.parameterization.contract_sha256:
        raise RuntimeError("resume checkpoint parameterization contract differs from the active training contract")
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
    if config.diagnostics.parameter_state_hash_steps:
        checkpoint_run_dir = Path(path).resolve().parent
        active_run_dir = Path(config.train.run_dir).resolve()
        if checkpoint_run_dir != active_run_dir:
            raise RuntimeError(
                "parameter-state receipt resume requires the checkpoint and active run directory to match"
            )
        architecture_fingerprint = raw_model.architecture_fingerprint()
        architecture_fingerprint_sha256 = str(architecture_fingerprint["fingerprint_sha256"])
        validate_checkpoint_parameter_state_hash(
            ckpt,
            raw_model,
            architecture_fingerprint_sha256=architecture_fingerprint_sha256,
            parameterization_contract_sha256=config.parameterization.contract_sha256,
        )
        validate_resume_parameter_state_hashes(
            checkpoint_run_dir / "telemetry" / "parameter_state_hashes.jsonl",
            raw_model,
            configured_steps=config.diagnostics.parameter_state_hash_steps,
            checkpoint_step=checkpoint_step,
            checkpoint_train_tokens=int(ckpt.get("train_tokens", 0)),
            architecture_fingerprint_sha256=architecture_fingerprint_sha256,
            parameterization_contract_sha256=config.parameterization.contract_sha256,
        )
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
