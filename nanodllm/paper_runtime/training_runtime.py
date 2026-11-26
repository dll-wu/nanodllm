import datetime
import gc
import os
import uuid
from contextlib import suppress
from pathlib import Path

import torch
import torch.distributed as dist

from . import ar_width_mup
from .config import ExperimentConfig
from .training_mtp import validate_mtp_config, validate_mtp_process
from .training_variable_depth import validate_process as validate_variable_depth_process

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
CGROUP_ROOT = Path("/sys/fs/cgroup")


def run_requested_training_diagnostic(session) -> None:
    """Paper training supports ordinary updates. The preview CLI owns standalone audits."""
    from .training_features import _training_diagnostic_requested

    if _training_diagnostic_requested(session.config):
        raise ValueError("Standalone diagnostics are outside the paper training recipes.")


def validate_training_entrypoint(config: ExperimentConfig) -> None:
    """Reject unsupported process layouts before allocating the training session."""
    validate_mtp_config(config)
    validate_mtp_process(config)
    validate_variable_depth_process(config)


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
    lifetime_peaks: tuple[float, float] | None = None,
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
    if lifetime_peaks is not None:
        local_record["peak_allocated_mib"], local_record["peak_reserved_mib"] = lifetime_peaks
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
    if ar_width_mup.active(config):
        ar_width_mup.configure_backend()
    telemetry = _query_numeric_runtime_telemetry(config)
    setattr(config, NUMERIC_RUNTIME_TELEMETRY_ATTR, telemetry)
    return dict(telemetry)


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
