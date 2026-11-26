"""Pure validation of training-diagnostic configuration fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig


def _validate_diagnostics_config(config: ExperimentConfig) -> None:
    for name in (
        "eos_supervision_prepack_document_batches",
        "eos_supervision_packed_batches",
        "real_packing_audit_batches",
        "real_packing_target_ce_example_limit",
    ):
        value = getattr(config.diagnostics, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"diagnostics.{name} must be an integer")
        if value <= 0:
            raise ValueError(f"diagnostics.{name} must be positive")

    for name in (
        "eos_supervision_forced_mask_t",
        "real_packing_target_ce_fixed_mask_t",
    ):
        value = getattr(config.diagnostics, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"diagnostics.{name} must be numeric")
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"diagnostics.{name} must be in (0, 1]")

    hash_steps = config.diagnostics.parameter_state_hash_steps
    if not isinstance(hash_steps, list) or any(
        not isinstance(step, int) or isinstance(step, bool) or step <= 0 for step in hash_steps
    ):
        raise ValueError("diagnostics.parameter_state_hash_steps must be a list of positive integers")
    if hash_steps != sorted(set(hash_steps)):
        raise ValueError("diagnostics.parameter_state_hash_steps must be unique and ascending")
