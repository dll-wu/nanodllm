"""CPU, sequence-major time plans for the shared-time MDLM evaluator."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .config import ExperimentConfig

MIDPOINT_TIME_SAMPLING = "stratified_per_sequence"
NELBO_TIME_SAMPLING_METHODS = (
    MIDPOINT_TIME_SAMPLING,
    "independent_uniform",
    "randomized_stratified_per_sequence",
)


def validate_nelbo_time_sampling(config: ExperimentConfig) -> None:
    if config.eval.nelbo_tokens <= 0:
        raise ValueError("eval.nelbo_tokens must be positive")
    if not 0.0 <= config.eval.nelbo_t_min < config.eval.nelbo_t_max <= 1.0:
        raise ValueError("eval NELBO times must satisfy 0 <= nelbo_t_min < nelbo_t_max <= 1")
    if config.eval.nelbo_seed < 0:
        raise ValueError("eval.nelbo_seed must be non-negative")
    if config.eval.nelbo_time_sampling not in NELBO_TIME_SAMPLING_METHODS:
        raise ValueError(f"eval.nelbo_time_sampling must be one of {NELBO_TIME_SAMPLING_METHODS}")
    if config.eval.nelbo_time_sampling != MIDPOINT_TIME_SAMPLING:
        if not config.train.eval_only:
            raise ValueError("non-default eval.nelbo_time_sampling requires train.eval_only=true")
        if config.model.causal or config.noise.type != "mdlm" or config.model.block_causal_pilot_size > 0:
            raise ValueError("non-default eval.nelbo_time_sampling requires the native shared-time MDLM evaluator")


@dataclass(frozen=True)
class NelboTimePlan:
    times: torch.Tensor
    method: str
    endpoint_redraw_count: int

    def metadata(self) -> dict[str, str | int]:
        return {
            "val_nelbo_time_plan_sha256": hashlib.sha256(
                self.times.numpy().astype("<f4", copy=False).tobytes()
            ).hexdigest(),
            "val_nelbo_time_rng_scheme": "torch_cpu_sequence_major_time_v1",
            "val_nelbo_time_endpoint_policy": (
                "legacy_float64_midpoints_cast_float32"
                if self.method == MIDPOINT_TIME_SAMPLING
                else "float64_uniform_cast_float32_reject_closed_stratum_endpoints"
            ),
            "val_nelbo_time_endpoint_redraw_count": self.endpoint_redraw_count,
        }


def _open_float32_uniforms(
    lower: torch.Tensor,
    upper: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Reject only float32 rounding to/outside a stratum, never clip importance weights.

    The recorded proposal is a finite-precision approximation to continuous
    uniform sampling. In particular, it assigns no mass to the singular t=0.
    Refuse intervals with no representable interior rather than invent epsilon.
    """

    first = lower.float()
    first = torch.where(
        first.double() <= lower,
        torch.nextafter(first, torch.full_like(first, float("inf"))),
        first,
    )
    if bool((first.double() >= upper).any()):
        raise ValueError("NELBO time stratum has no float32 interior; reduce count or widen interval")
    result = torch.empty(lower.shape, dtype=torch.float32)
    pending = torch.arange(lower.numel())
    redraw_count = 0
    for _ in range(128):
        draws = torch.rand(pending.numel(), dtype=torch.float64, generator=generator)
        values = (lower[pending] + (upper[pending] - lower[pending]) * draws).float()
        valid = (values.double() > lower[pending]) & (values.double() < upper[pending])
        result[pending[valid]] = values[valid]
        pending = pending[~valid]
        redraw_count += pending.numel()
        if pending.numel() == 0:
            return result, redraw_count
    raise RuntimeError("NELBO time endpoint rejection exceeded 128 rounds")


def build_nelbo_time_plan(
    count: int,
    t_min: float,
    t_max: float,
    seed: int,
    method: str = MIDPOINT_TIME_SAMPLING,
) -> NelboTimePlan:
    """Build a plan independent of device batching and global/training RNG state.

    Midpoint values and permutation reproduce the historical default exactly.
    The two stratified designs share bin assignment at the same count/seed;
    token-mask uniforms remain a separate, unchanged RNG stream.
    """

    if count <= 0:
        raise ValueError("stratified NELBO evaluation requires at least one sequence")
    if not 0.0 <= float(t_min) < float(t_max) <= 1.0:
        raise ValueError("stratified NELBO times require 0 <= t_min < t_max <= 1")
    if method not in NELBO_TIME_SAMPLING_METHODS:
        raise ValueError(f"unknown NELBO time sampling method: {method!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(count, generator=generator)
    width = float(t_max) - float(t_min)
    if method == MIDPOINT_TIME_SAMPLING:
        unit_midpoints = (torch.arange(count, dtype=torch.float64) + 0.5) / float(count)
        times = float(t_min) + width * unit_midpoints
        return NelboTimePlan(times[permutation].float(), method, 0)
    if method == "independent_uniform":
        lower = torch.full((count,), float(t_min), dtype=torch.float64)
        upper = torch.full((count,), float(t_max), dtype=torch.float64)
    else:
        boundaries = float(t_min) + width * torch.arange(count + 1, dtype=torch.float64) / float(count)
        lower, upper = boundaries[:-1], boundaries[1:]
    times, redraw_count = _open_float32_uniforms(lower, upper, generator)
    return NelboTimePlan(times[permutation], method, redraw_count)
