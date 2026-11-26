"""Observation cadence and fields excluded from numerical resume identity."""

import math

TELEMETRY_OPERATIONAL_FIELDS = (
    "tensorboard",
    "tensorboard_dir",
    "save_every",
    "eval_every",
    "telemetry_interval_seconds",
)


def validate_telemetry_config(train):
    if not math.isfinite(train.telemetry_interval_seconds) or train.telemetry_interval_seconds < 0:
        raise ValueError("train.telemetry_interval_seconds must be finite and nonnegative")
    if train.log_interval <= 0:
        raise ValueError("train.log_interval must be positive")
    if train.tensorboard_dir and not train.tensorboard:
        raise ValueError("train.tensorboard_dir requires train.tensorboard=true")
    if train.save_every == 0:
        raise ValueError("train.save_every must be positive or -1")
    if train.eval_every == 0:
        raise ValueError("train.eval_every must be positive or -1")


def strip_disabled_telemetry(value):
    """Preserve identities of configurations predating this opt-in observation field."""
    if value.get("train", {}).get("telemetry_interval_seconds", 0) == 0:
        value["train"].pop("telemetry_interval_seconds", None)
