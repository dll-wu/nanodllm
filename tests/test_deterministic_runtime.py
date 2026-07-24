import os

import pytest
import torch
from nanodllm.config import DataConfig, ExperimentConfig, RuntimeConfig, validate_config
from nanodllm.training import (
    _configure_numeric_runtime,
    _numeric_runtime_telemetry,
    run_config_summary,
)


def _config(**runtime_overrides) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(tokenizer="byte", train_data=["dummy.parquet"]),
        runtime=RuntimeConfig(**runtime_overrides),
    )


@pytest.fixture(autouse=True)
def _restore_process_numeric_runtime():
    workspace_was_set = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    state = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(state["deterministic"], warn_only=state["warn_only"])
        torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
        torch.backends.cuda.enable_flash_sdp(state["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(state["mem_efficient"])
        torch.backends.cuda.enable_math_sdp(state["math"])
        torch.backends.cuda.enable_cudnn_sdp(state["cudnn_sdp"])
        torch.set_float32_matmul_precision(state["matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = state["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = state["cudnn_allow_tf32"]
        if workspace_was_set:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace_config
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def test_deterministic_runtime_defaults_are_opt_in():
    runtime = RuntimeConfig()

    assert runtime.deterministic_algorithms is False
    assert runtime.sdpa_backend == "auto"
    assert runtime.cudnn_allow_tf32 is None


@pytest.mark.parametrize("backend", ["flash", "efficient", "MATH"])
def test_runtime_rejects_unknown_sdpa_backend(backend):
    config = _config(device_type="cpu", sdpa_backend=backend)

    with pytest.raises(ValueError, match="runtime.sdpa_backend must be 'auto' or 'math'"):
        validate_config(config)


def test_runtime_rejects_non_boolean_deterministic_algorithms():
    config = _config(device_type="cpu")
    config.runtime.deterministic_algorithms = "true"

    with pytest.raises(ValueError, match="runtime.deterministic_algorithms must be a boolean"):
        validate_config(config)


@pytest.mark.parametrize("value", [True, False, None])
def test_runtime_accepts_nullable_cudnn_tf32_override(value):
    config = _config(device_type="cpu", cudnn_allow_tf32=value)

    validate_config(config)


def test_runtime_rejects_non_boolean_cudnn_tf32_override():
    config = _config(device_type="cpu")
    config.runtime.cudnn_allow_tf32 = "true"

    with pytest.raises(ValueError, match="runtime.cudnn_allow_tf32 must be a boolean or null"):
        validate_config(config)


@pytest.mark.parametrize(
    ("precision", "allow_tf32", "error"),
    [
        ("high", False, "high/medium requires runtime.allow_tf32=true"),
        ("medium", False, "high/medium requires runtime.allow_tf32=true"),
        ("highest", True, "highest requires runtime.allow_tf32=false"),
    ],
)
def test_runtime_rejects_contradictory_cuda_matmul_controls(precision, allow_tf32, error):
    config = _config(
        device_type="cuda",
        float32_matmul_precision=precision,
        allow_tf32=allow_tf32,
    )

    with pytest.raises(ValueError, match=error):
        validate_config(config)


@pytest.mark.parametrize(
    ("device_type", "precision", "allow_tf32"),
    [
        ("cuda", "", False),
        ("cuda", "", True),
        ("cuda", "high", True),
        ("cuda", "medium", True),
        ("cuda", "highest", False),
        ("cpu", "high", False),
        ("cpu", "highest", True),
    ],
)
def test_runtime_accepts_consistent_or_non_cuda_matmul_controls(device_type, precision, allow_tf32):
    config = _config(
        device_type=device_type,
        float32_matmul_precision=precision,
        allow_tf32=allow_tf32,
    )

    validate_config(config)


def test_runtime_validation_rejects_deterministic_cuda_with_cudnn_benchmark():
    config = _config(device_type="cuda", deterministic_algorithms=True, cudnn_benchmark=True)

    with pytest.raises(ValueError, match="deterministic CUDA requires runtime.cudnn_benchmark=false"):
        validate_config(config)


@pytest.mark.parametrize("workspace_config", [None, "", ":8:8", "4096:8"])
def test_deterministic_cuda_fails_closed_without_supported_cublas_workspace(monkeypatch, workspace_config):
    config = _config(device_type="cuda", deterministic_algorithms=True, cudnn_benchmark=False)
    if workspace_config is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace_config)

    before = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError, match="deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG"):
        _configure_numeric_runtime(config)
    assert torch.are_deterministic_algorithms_enabled() is before
    assert _numeric_runtime_telemetry(config) == {}


def test_preconfiguration_summary_omits_effective_runtime_telemetry():
    config = _config(device_type="cuda", cudnn_allow_tf32=True)

    summary = run_config_summary(config, "train")

    assert summary["runtime_cudnn_allow_tf32_requested"] is True
    assert summary["runtime_compile"] is True
    assert summary["runtime_dtype"] == "bfloat16"
    for key in (
        "runtime_deterministic_algorithms_enabled",
        "runtime_cudnn_allow_tf32_resolved",
        "runtime_cudnn_allow_tf32",
        "runtime_sdpa_math_enabled",
        "runtime_float32_matmul_precision",
    ):
        assert key not in summary
    assert _numeric_runtime_telemetry(config) == {}


def test_failed_reconfiguration_clears_stale_effective_telemetry(monkeypatch):
    config = _config(device_type="cpu")
    _configure_numeric_runtime(config)
    assert _numeric_runtime_telemetry(config)
    config.runtime.device_type = "cuda"
    config.runtime.deterministic_algorithms = True
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with pytest.raises(RuntimeError, match="deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG"):
        _configure_numeric_runtime(config)

    assert _numeric_runtime_telemetry(config) == {}
    summary = run_config_summary(config, "train")
    assert "runtime_deterministic_algorithms_enabled" not in summary


def test_medium_matmul_precision_remains_medium():
    config = _config(
        device_type="cuda",
        float32_matmul_precision="medium",
        allow_tf32=True,
    )

    telemetry = _configure_numeric_runtime(config)

    assert telemetry["runtime_float32_matmul_precision"] == "medium"
    assert telemetry["runtime_cuda_matmul_allow_tf32"] is True


@pytest.mark.parametrize("workspace_config", [":4096:8", ":16:8"])
def test_deterministic_math_runtime_is_applied_and_reported(monkeypatch, workspace_config):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace_config)
    torch.backends.cudnn.allow_tf32 = False
    config = _config(
        device_type="cuda",
        dtype="bfloat16",
        compile=True,
        compile_dynamic=True,
        float32_matmul_precision="high",
        allow_tf32=True,
        cudnn_allow_tf32=True,
        cudnn_benchmark=False,
        deterministic_algorithms=True,
        sdpa_backend="math",
    )

    telemetry = _configure_numeric_runtime(config)

    assert telemetry == _numeric_runtime_telemetry(config)
    assert telemetry["runtime_deterministic_algorithms_enabled"] is True
    assert telemetry["runtime_deterministic_algorithms_warn_only_enabled"] is False
    assert telemetry["runtime_cudnn_benchmark"] is False
    assert telemetry["runtime_cudnn_deterministic"] is True
    assert telemetry["runtime_cublas_workspace_config"] == workspace_config
    assert telemetry["runtime_sdpa_flash_enabled"] is False
    assert telemetry["runtime_sdpa_mem_efficient_enabled"] is False
    assert telemetry["runtime_sdpa_math_enabled"] is True
    assert telemetry["runtime_sdpa_cudnn_enabled"] is False
    assert telemetry["runtime_cuda_matmul_allow_tf32"] is True
    assert telemetry["runtime_cudnn_allow_tf32_resolved"] is True
    assert telemetry["runtime_cudnn_allow_tf32"] is True
    assert telemetry["runtime_float32_matmul_precision"] == "high"

    summary = run_config_summary(config, "train")
    assert summary["runtime_deterministic_algorithms_requested"] is True
    assert summary["runtime_sdpa_backend_requested"] == "math"
    assert summary["runtime_cudnn_allow_tf32_requested"] is True
    assert summary["runtime_compile"] is True
    assert summary["runtime_compile_dynamic"] is True
    assert summary["runtime_dtype"] == "bfloat16"
    assert summary["runtime_device_type"] == "cuda"
    for key, value in telemetry.items():
        assert summary[key] == value


def test_default_runtime_preserves_cudnn_tf32_and_restores_other_runtime_defaults():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    config = _config(
        device_type="cuda",
        allow_tf32=False,
        deterministic_algorithms=False,
        sdpa_backend="auto",
    )

    telemetry = _configure_numeric_runtime(config)

    assert telemetry["runtime_deterministic_algorithms_enabled"] is False
    assert telemetry["runtime_deterministic_algorithms_warn_only_enabled"] is False
    assert telemetry["runtime_cudnn_deterministic"] is False
    assert telemetry["runtime_cuda_matmul_allow_tf32"] is False
    assert telemetry["runtime_cudnn_allow_tf32_resolved"] is True
    assert telemetry["runtime_cudnn_allow_tf32"] is True
    assert telemetry["runtime_sdpa_flash_enabled"] is True
    assert telemetry["runtime_sdpa_mem_efficient_enabled"] is True
    assert telemetry["runtime_sdpa_math_enabled"] is True
    assert telemetry["runtime_sdpa_cudnn_enabled"] is True

    summary = run_config_summary(config, "train")
    assert summary["runtime_cudnn_allow_tf32_requested"] is None


def test_explicit_cudnn_tf32_override_is_authoritative():
    torch.backends.cudnn.allow_tf32 = True
    config = _config(
        device_type="cuda",
        cudnn_allow_tf32=False,
    )

    telemetry = _configure_numeric_runtime(config)

    assert telemetry["runtime_cudnn_allow_tf32_resolved"] is False
    assert telemetry["runtime_cudnn_allow_tf32"] is False
    summary = run_config_summary(config, "train")
    assert summary["runtime_cudnn_allow_tf32_requested"] is False
