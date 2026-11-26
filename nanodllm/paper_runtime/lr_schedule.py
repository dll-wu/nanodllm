"""Opt-in cooldown shapes; historical clamped-linear scheduling stays unchanged."""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig

STYLES = ("legacy_linear", "linear", "cosine", "one_minus_sqrt")


def decay_multiplier(remaining: float, floor: float, style: str) -> float:
    if style == "legacy_linear":
        return max(floor, remaining)
    if style not in STYLES:
        raise ValueError("unknown optim.lr_decay_style")
    remaining = min(1.0, max(0.0, remaining))
    if style == "cosine":
        shape = (1 - math.cos(math.pi * remaining)) / 2
    elif style == "one_minus_sqrt":
        shape = 1 - math.sqrt(1 - remaining)
    else:
        shape = remaining
    return floor + (1 - floor) * shape


def validate_lr_schedule(config: "ExperimentConfig") -> None:
    optim = config.optim
    if optim.lr_decay_style not in STYLES:
        raise ValueError(f"optim.lr_decay_style must be one of {STYLES}")
    if optim.lr_decay_style == "legacy_linear":
        return
    if config.parameterization.execution_scope != "ordinary":
        raise ValueError("new LR schedules require ordinary execution scope; historical scaling contracts stay frozen")
    if not (0 <= optim.final_lr_frac <= 1 and 0 < optim.warmdown_ratio <= 1):
        raise ValueError("new LR schedules require finite floor in [0,1] and cooldown in (0,1]")
    if optim.warmup_ratio != -1 and not 0 <= optim.warmup_ratio < 1:
        raise ValueError("new LR schedules require warmup_ratio=-1 or a fraction in [0,1)")
    if optim.warmup_ratio == -1 and (type(optim.warmup_steps) is not int or optim.warmup_steps < 0):
        raise ValueError("new LR schedules require nonnegative integer warmup_steps")
    if config.train.num_steps > 0:
        validate_resolved_schedule(config, config.train.num_steps)


def validate_resolved_schedule(config: "ExperimentConfig", total_steps: int) -> None:
    if config.optim.lr_decay_style == "legacy_linear":
        return
    optim = config.optim
    warmup = round(optim.warmup_ratio * total_steps) if optim.warmup_ratio >= 0 else optim.warmup_steps
    cooldown = round(optim.warmdown_ratio * total_steps)
    if cooldown < 1 or warmup + cooldown > total_steps:
        raise ValueError("new LR schedules require nonempty cooldown and no warmup/cooldown overlap")


def strip_legacy_lr_schedule(value: dict) -> None:
    if value["optim"].pop("lr_decay_style", "legacy_linear") != "legacy_linear":
        raise ValueError("historical configuration requires legacy_linear LR schedule")


def get_lr_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    warmdown_ratio: float,
    final_lr_frac: float,
    lr_decay_style: str = "legacy_linear",
) -> float:
    # Delayed import avoids the config/validation cycle. The legacy owner also
    # belongs to a pinned evaluator projection, so its bytes remain untouched.
    from .training_plan import get_lr_multiplier as legacy_multiplier

    if lr_decay_style not in STYLES:
        raise ValueError("unknown optim.lr_decay_style")
    legacy = legacy_multiplier(step, total_steps, warmup_steps, warmdown_ratio, final_lr_frac)
    cooldown = round(warmdown_ratio * total_steps)
    if lr_decay_style == "legacy_linear" or step < warmup_steps or step < total_steps - cooldown:
        return legacy
    return decay_multiplier((total_steps - step) / max(1, cooldown), final_lr_frac, lr_decay_style)


def _resolved_warmup_steps(config: "ExperimentConfig", total_steps: int) -> int:
    from .training_plan import _resolved_warmup_steps as legacy_warmup

    validate_resolved_schedule(config, total_steps)
    return legacy_warmup(config, total_steps)


def _lr_schedule_telemetry(config: "ExperimentConfig", total_steps: int, warmup_steps: int) -> dict:
    from .training_plan import _lr_schedule_telemetry as legacy_telemetry

    result = legacy_telemetry(config, total_steps, warmup_steps)
    optim = config.optim
    if optim.lr_decay_style == "legacy_linear":
        return result
    result["decay_style"] = optim.lr_decay_style
    for entry in result["trace"]:
        entry["lr_multiplier"] = get_lr_multiplier(
            entry["step"] - 1,
            total_steps,
            warmup_steps,
            optim.warmdown_ratio,
            optim.final_lr_frac,
            optim.lr_decay_style,
        )
    return result
