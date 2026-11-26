"""Training RNG snapshots, restoration, and versioned rank-state validation."""

import os
import random

import numpy as np
import torch
import torch.distributed as dist

from .config import ExperimentConfig

RESUME_RNG_STATE_VERSION = 1

RANK_RNG_STATE_FIELDS = frozenset(
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
    }
)
RANK_CUDA_RNG_STATE_FIELDS = frozenset(
    {"active_device_index", "visible_device_count", "device_name", "compute_capability", "state"}
)


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


def _require_exact_mapping_keys(value: object, expected: frozenset[str], *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a mapping")
    observed = set(value)
    if observed != expected:
        raise RuntimeError(
            f"{field} keys mismatch: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return value


def _validate_rank_rng_payload(state: dict[str, object]) -> None:
    rng = _require_exact_mapping_keys(state, RANK_RNG_STATE_FIELDS, field="rank RNG state")
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
        cuda = _require_exact_mapping_keys(cuda_state, RANK_CUDA_RNG_STATE_FIELDS, field="rank CUDA RNG state")
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
