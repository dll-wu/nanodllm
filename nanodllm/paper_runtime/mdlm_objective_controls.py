"""Opt-in time-sampling and constant-weight controls for flat MDLM training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .config import ExperimentConfig


def target_weights(times: torch.Tensor, *, schedule_weight: bool, constant_scale: float) -> torch.Tensor:
    """Weights on already selected targets; the denominator remains all lossable tokens."""
    if schedule_weight:
        return 1.0 / times.clamp_min(1e-8)
    return torch.full_like(times, constant_scale, dtype=torch.float32)


def validate_mdlm_objective_controls(config: ExperimentConfig) -> None:
    if not isinstance(config.noise.per_token_noise_levels, bool):
        raise ValueError("noise.per_token_noise_levels must be a boolean")
    if config.noise.per_token_noise_levels:
        if config.noise.type != "mdlm":
            raise ValueError("noise.per_token_noise_levels requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("noise.per_token_noise_levels requires complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("noise.per_token_noise_levels requires suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("noise.per_token_noise_levels preserves no time conditioning")
        if not config.noise.schedule_weight:
            raise ValueError("noise.per_token_noise_levels requires schedule_weight=true")
    # Preserve legacy auto-routing, including historical unweighted scalar-time runs.
    if config.noise.mdlm_time_sampling not in {"legacy_auto", "per_row", "stratified_per_row"}:
        raise ValueError("noise.mdlm_time_sampling must be 'legacy_auto', 'per_row' or 'stratified_per_row'")
    scale = config.noise.mdlm_unweighted_loss_scale
    if isinstance(scale, bool) or not isinstance(scale, int | float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("noise.mdlm_unweighted_loss_scale must be finite and positive")
    if config.noise.mdlm_time_sampling in {"per_row", "stratified_per_row"}:
        from .training_features import _uses_strict_mdlm_per_row_t

        if not _uses_strict_mdlm_per_row_t(config):
            raise ValueError("explicit per-row time sampling requires the flat pure-M2T MDLM route")
        if config.noise.t_min <= 0:
            raise ValueError("explicit per_row training requires noise.t_min > 0")
    if config.noise.mdlm_time_sampling == "stratified_per_row":
        if not config.noise.schedule_weight:
            raise ValueError("stratified_per_row requires schedule_weight=true")
        if not 0 < config.noise.t_min < config.noise.t_max <= 1:
            raise ValueError("stratified_per_row requires 0 < t_min < t_max <= 1")
    if scale != 1.0 and (config.noise.mdlm_time_sampling != "per_row" or config.noise.schedule_weight):
        raise ValueError("a non-unit MDLM loss scale requires explicit per_row and schedule_weight=false")
