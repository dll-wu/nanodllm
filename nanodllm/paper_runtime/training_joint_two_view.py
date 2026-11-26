"""Differentiable draft-view objective integration for training steps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .training_features import _uses_differentiable_self_generated_draft
from .training_metrics import _per_target_t_values, _weighted_token_mean
from .training_self_generated_edit import joint_two_view_loss

_DRAFT_M2T_CREDIT_MODES = frozenset(
    {
        "uniform",
        "aligned_expected_error_reward",
        "shuffled_expected_error_reward",
    }
)

_JOINT_TWO_VIEW_INT_COUNTER_ATTRIBUTES = (
    ("joint_two_view_count", "step_joint_two_view_count"),
    ("joint_draft_m2t_credit_target_count", "step_joint_draft_m2t_credit_target_count"),
    ("joint_draft_m2t_credit_row_count", "step_joint_draft_m2t_credit_row_count"),
    ("joint_draft_m2t_credit_clipped_low_count", "step_joint_draft_m2t_credit_clipped_low_count"),
    ("joint_draft_m2t_credit_clipped_high_count", "step_joint_draft_m2t_credit_clipped_high_count"),
)
_JOINT_TWO_VIEW_FLOAT_SUM_ATTRIBUTES = (
    ("joint_draft_m2t_loss_sum", "step_joint_draft_m2t_loss_sum"),
    ("joint_correction_loss_sum", "step_joint_correction_loss_sum"),
    ("joint_draft_m2t_credit_expected_error_sum", "step_joint_draft_m2t_credit_expected_error_sum"),
    ("joint_draft_m2t_credit_multiplier_sum", "step_joint_draft_m2t_credit_multiplier_sum"),
    (
        "joint_draft_m2t_credit_expected_error_multiplier_sum",
        "step_joint_draft_m2t_credit_expected_error_multiplier_sum",
    ),
    ("joint_draft_m2t_credit_base_schedule_mass_sum", "step_joint_draft_m2t_credit_base_schedule_mass_sum"),
    ("joint_draft_m2t_credit_schedule_mass_sum", "step_joint_draft_m2t_credit_schedule_mass_sum"),
)
_JOINT_TWO_VIEW_MINIMUM_ATTRIBUTES = (
    ("joint_draft_m2t_credit_multiplier", "step_joint_draft_m2t_credit_multiplier_min"),
)
_JOINT_TWO_VIEW_MAXIMUM_ATTRIBUTES = (
    ("joint_draft_m2t_credit_multiplier", "step_joint_draft_m2t_credit_multiplier_max"),
)


@dataclass(frozen=True, slots=True)
class DraftM2TCredit:
    """Detached row-local multipliers and their audit inputs."""

    multipliers: torch.Tensor
    expected_errors: torch.Tensor
    clipped_low: torch.Tensor
    clipped_high: torch.Tensor
    row_count: int


def _stateless_shuffle_keys(
    target_positions: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    sequence_length: int,
    training_seed: int,
    global_microbatch_idx: int,
) -> torch.Tensor:
    """Build unique pseudo-random integer keys without touching an RNG stream."""

    # All intermediates remain below signed-int64 overflow before each modulo.
    # Appending the position makes keys unique even if the quadratic mix
    # collides, so argsort never depends on device-specific tie handling.
    modulus = 2_147_483_647
    seed = int(training_seed) % modulus
    microbatch = int(global_microbatch_idx) % modulus
    mixed = torch.remainder(
        target_positions + seed * 104_729 + microbatch * 13_007 + row_ids * 433_494_437,
        modulus,
    )
    mixed = torch.remainder(mixed * mixed + mixed * 48_271 + 12_820_163, modulus)
    mixed = torch.remainder(mixed * mixed + mixed * 40_692 + 37_811, modulus)
    return mixed * int(sequence_length) + target_positions


def _shuffle_credit_within_rows(
    multipliers: torch.Tensor,
    target_indices: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    sequence_length: int,
    training_seed: int,
    global_microbatch_idx: int,
) -> torch.Tensor:
    """Apply a stateless within-row permutation while preserving each multiset."""

    target_positions = torch.remainder(target_indices, int(sequence_length))
    keys = _stateless_shuffle_keys(
        target_positions,
        row_ids,
        sequence_length=sequence_length,
        training_seed=training_seed,
        global_microbatch_idx=global_microbatch_idx,
    )
    shuffled = torch.empty_like(multipliers)
    for row_id in torch.unique(row_ids, sorted=True).unbind():
        row_targets = (row_ids == row_id).nonzero(as_tuple=True)[0]
        permutation = torch.argsort(keys.index_select(0, row_targets))
        source_targets = row_targets.index_select(0, permutation)
        shuffled.index_copy_(0, row_targets, multipliers.index_select(0, source_targets))
    return shuffled


def build_row_local_draft_m2t_credit(
    expected_errors: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    sequence_length: int,
    mode: str,
    strength: float,
    clip_min: float,
    clip_max: float,
    training_seed: int,
    global_microbatch_idx: int,
) -> DraftM2TCredit:
    """Turn detached correction difficulty into mean-one row-local credit.

    Lower correction expected error is treated as higher downstream reward:
    ``advantage = row_mean(error) - error``. Clipping precedes row-wise
    normalization, so multiplying the ordinary ``1/t`` draft weights preserves
    their exact mass in every row. The shuffled mode permutes only the final
    normalized multipliers.
    """

    if mode not in _DRAFT_M2T_CREDIT_MODES:
        raise ValueError(f"unsupported draft-M2T credit mode {mode!r}")
    if expected_errors.dim() != 1 or not expected_errors.is_floating_point():
        raise ValueError("draft-M2T expected errors must be a one-dimensional floating tensor")
    if target_indices.dim() != 1 or target_indices.dtype != torch.long:
        raise ValueError("draft-M2T target indices must be a one-dimensional int64 tensor")
    if target_indices.device != expected_errors.device or target_indices.numel() != expected_errors.numel():
        raise ValueError("draft-M2T expected errors and target indices must align on one device")
    if int(sequence_length) <= 0:
        raise ValueError("draft-M2T sequence length must be positive")
    if not math.isfinite(float(strength)) or float(strength) < 0.0:
        raise ValueError("draft-M2T credit strength must be finite and non-negative")
    if mode != "uniform" and float(strength) <= 0.0:
        raise ValueError("non-uniform draft-M2T credit strength must be positive")
    if not 0.0 < float(clip_min) <= 1.0 <= float(clip_max):
        raise ValueError("draft-M2T credit clipping must satisfy 0 < clip_min <= 1 <= clip_max")
    difficulty = expected_errors.detach().float()
    if difficulty.numel() == 0:
        empty_bool = torch.empty_like(difficulty, dtype=torch.bool)
        return DraftM2TCredit(difficulty.clone(), difficulty, empty_bool, empty_bool.clone(), 0)
    if bool((target_indices < 0).any().item()):
        raise ValueError("draft-M2T target indices must be non-negative")
    if not bool(torch.isfinite(difficulty).all().item()):
        raise ValueError("draft-M2T expected errors must be finite")
    if bool(((difficulty < 0.0) | (difficulty > 1.0)).any().item()):
        raise ValueError("draft-M2T expected errors must be in [0, 1]")

    row_ids = torch.div(target_indices, int(sequence_length), rounding_mode="floor")
    unique_rows, inverse, row_counts = torch.unique(
        row_ids,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    if mode == "uniform":
        unclipped = torch.ones_like(difficulty)
    else:
        row_sums = torch.zeros(unique_rows.numel(), dtype=difficulty.dtype, device=difficulty.device)
        row_sums.scatter_add_(0, inverse, difficulty)
        row_means = row_sums / row_counts.to(dtype=difficulty.dtype)
        advantages = row_means.index_select(0, inverse) - difficulty
        unclipped = 1.0 + float(strength) * advantages
    clipped_low = unclipped < float(clip_min)
    clipped_high = unclipped > float(clip_max)
    clipped = unclipped.clamp(min=float(clip_min), max=float(clip_max))
    row_credit_sums = torch.zeros(unique_rows.numel(), dtype=clipped.dtype, device=clipped.device)
    row_credit_sums.scatter_add_(0, inverse, clipped)
    row_credit_means = row_credit_sums / row_counts.to(dtype=clipped.dtype)
    multipliers = clipped / row_credit_means.index_select(0, inverse)
    if mode == "shuffled_expected_error_reward":
        multipliers = _shuffle_credit_within_rows(
            multipliers,
            target_indices,
            row_ids,
            sequence_length=sequence_length,
            training_seed=training_seed,
            global_microbatch_idx=global_microbatch_idx,
        )
    return DraftM2TCredit(
        multipliers=multipliers.detach(),
        expected_errors=difficulty,
        clipped_low=clipped_low.detach(),
        clipped_high=clipped_high.detach(),
        row_count=int(unique_rows.numel()),
    )


def reset_joint_two_view_step_metrics(session) -> None:
    session.step_joint_draft_m2t_loss_sum = 0.0
    session.step_joint_correction_loss_sum = 0.0
    session.step_joint_two_view_count = 0
    session.step_joint_draft_m2t_credit_target_count = 0
    session.step_joint_draft_m2t_credit_row_count = 0
    session.step_joint_draft_m2t_credit_clipped_low_count = 0
    session.step_joint_draft_m2t_credit_clipped_high_count = 0
    session.step_joint_draft_m2t_credit_expected_error_sum = 0.0
    session.step_joint_draft_m2t_credit_multiplier_sum = 0.0
    session.step_joint_draft_m2t_credit_expected_error_multiplier_sum = 0.0
    session.step_joint_draft_m2t_credit_base_schedule_mass_sum = 0.0
    session.step_joint_draft_m2t_credit_schedule_mass_sum = 0.0
    session.step_joint_draft_m2t_credit_multiplier_min = float("inf")
    session.step_joint_draft_m2t_credit_multiplier_max = float("-inf")


def _step_metric_values(session, attributes: tuple[tuple[str, str], ...]) -> dict[str, int | float]:
    return {key: getattr(session, attribute) for key, attribute in attributes}


def _apply_reduced_step_metrics(
    session,
    reduced: Mapping[str, int | float],
    attributes: tuple[tuple[str, str], ...],
) -> None:
    for key, attribute in attributes:
        setattr(session, attribute, reduced[key])


def joint_two_view_step_int_counters(session) -> dict[str, int | float]:
    """Return joint-two-view integer counters for the packed step reduction."""

    return _step_metric_values(session, _JOINT_TWO_VIEW_INT_COUNTER_ATTRIBUTES)


def apply_reduced_joint_two_view_step_int_counters(session, reduced: Mapping[str, int | float]) -> None:
    _apply_reduced_step_metrics(session, reduced, _JOINT_TWO_VIEW_INT_COUNTER_ATTRIBUTES)


def joint_two_view_step_float_sums(session) -> dict[str, int | float]:
    """Return joint-two-view sums for the packed step reduction."""

    return _step_metric_values(session, _JOINT_TWO_VIEW_FLOAT_SUM_ATTRIBUTES)


def apply_reduced_joint_two_view_step_float_sums(session, reduced: Mapping[str, int | float]) -> None:
    _apply_reduced_step_metrics(session, reduced, _JOINT_TWO_VIEW_FLOAT_SUM_ATTRIBUTES)


def joint_two_view_step_minima(session) -> dict[str, int | float]:
    return _step_metric_values(session, _JOINT_TWO_VIEW_MINIMUM_ATTRIBUTES)


def joint_two_view_step_maxima(session) -> dict[str, int | float]:
    return _step_metric_values(session, _JOINT_TWO_VIEW_MAXIMUM_ATTRIBUTES)


def apply_reduced_joint_two_view_step_extrema(
    session,
    minima: Mapping[str, int | float],
    maxima: Mapping[str, int | float],
) -> None:
    _apply_reduced_step_metrics(session, minima, _JOINT_TWO_VIEW_MINIMUM_ATTRIBUTES)
    _apply_reduced_step_metrics(session, maxima, _JOINT_TWO_VIEW_MAXIMUM_ATTRIBUTES)


def apply_joint_two_view_loss(session) -> None:
    """Mix the correction loss with same-view draft MDLM when configured."""

    if not session.self_generated_edit_branch or not _uses_differentiable_self_generated_draft(session.config):
        return
    if session.self_generated_draft_m2t_logits is None:
        raise RuntimeError("joint two-view training is missing differentiable draft logits")
    draft_ce_values = F.cross_entropy(
        session.self_generated_draft_m2t_logits,
        session.targets,
        reduction="none",
    )
    base_draft_weights = 1.0 / _per_target_t_values(
        session.t_val,
        session.x_0,
        session.target_indices,
    ).clamp_min(1e-08)
    if session.self_generated_correction_objective is None:
        raise RuntimeError("joint two-view training is missing the correction expected-error signal")
    credit = build_row_local_draft_m2t_credit(
        session.self_generated_correction_objective.expected_error_values,
        session.target_indices,
        sequence_length=int(session.x_0.shape[1]),
        mode=session.config.train.token_edit_draft_m2t_credit_mode,
        strength=float(session.config.train.token_edit_draft_m2t_credit_strength),
        clip_min=float(session.config.train.token_edit_draft_m2t_credit_clip_min),
        clip_max=float(session.config.train.token_edit_draft_m2t_credit_clip_max),
        training_seed=int(session.training_seed),
        global_microbatch_idx=int(session.global_microbatch_idx),
    )
    draft_weights = base_draft_weights * credit.multipliers
    draft_m2t_loss = _weighted_token_mean(draft_ce_values, draft_weights, session.lossable_tokens)
    correction_loss = session.loss
    session.loss = joint_two_view_loss(
        correction_loss,
        draft_m2t_loss,
        draft_m2t_weight=float(session.config.train.token_edit_draft_m2t_loss_weight),
    )
    session.main_loss_for_log = session.loss
    session.step_joint_draft_m2t_loss_sum += float(draft_m2t_loss.detach().item())
    session.step_joint_correction_loss_sum += float(correction_loss.detach().item())
    session.step_joint_two_view_count += 1
    session.step_joint_draft_m2t_credit_target_count += int(credit.multipliers.numel())
    session.step_joint_draft_m2t_credit_row_count += credit.row_count
    session.step_joint_draft_m2t_credit_clipped_low_count += int(credit.clipped_low.sum().item())
    session.step_joint_draft_m2t_credit_clipped_high_count += int(credit.clipped_high.sum().item())
    session.step_joint_draft_m2t_credit_expected_error_sum += float(credit.expected_errors.sum().item())
    session.step_joint_draft_m2t_credit_multiplier_sum += float(credit.multipliers.sum().item())
    session.step_joint_draft_m2t_credit_expected_error_multiplier_sum += float(
        (credit.expected_errors * credit.multipliers).sum().item()
    )
    session.step_joint_draft_m2t_credit_base_schedule_mass_sum += float(base_draft_weights.sum().item())
    session.step_joint_draft_m2t_credit_schedule_mass_sum += float(draft_weights.sum().item())
    if credit.multipliers.numel():
        session.step_joint_draft_m2t_credit_multiplier_min = min(
            session.step_joint_draft_m2t_credit_multiplier_min,
            float(credit.multipliers.min().item()),
        )
        session.step_joint_draft_m2t_credit_multiplier_max = max(
            session.step_joint_draft_m2t_credit_multiplier_max,
            float(credit.multipliers.max().item()),
        )


def _joint_two_view_common_metrics(session) -> dict[str, float | str]:
    if not _uses_differentiable_self_generated_draft(session.config):
        return {}
    weight = float(session.config.train.token_edit_draft_m2t_loss_weight)
    return {
        "token_edit_draft_m2t_loss_weight": weight,
        "token_edit_correction_loss_weight": 1.0 - weight,
        "token_edit_draft_m2t_credit_mode": session.config.train.token_edit_draft_m2t_credit_mode,
        "token_edit_draft_m2t_credit_strength": float(session.config.train.token_edit_draft_m2t_credit_strength),
        "token_edit_draft_m2t_credit_clip_min": float(session.config.train.token_edit_draft_m2t_credit_clip_min),
        "token_edit_draft_m2t_credit_clip_max": float(session.config.train.token_edit_draft_m2t_credit_clip_max),
    }


def _joint_two_view_credit_metrics(session, prefix: str) -> dict[str, float | int]:
    count = session.step_joint_draft_m2t_credit_target_count
    expected_error_mean = session.step_joint_draft_m2t_credit_expected_error_sum / max(1, count)
    multiplier_mean = session.step_joint_draft_m2t_credit_multiplier_sum / max(1, count)
    covariance = (
        session.step_joint_draft_m2t_credit_expected_error_multiplier_sum / max(1, count)
        - expected_error_mean * multiplier_mean
    )
    base_schedule_mass = session.step_joint_draft_m2t_credit_base_schedule_mass_sum
    schedule_mass = session.step_joint_draft_m2t_credit_schedule_mass_sum
    return {
        f"{prefix}joint_draft_m2t_credit_target_tokens": count,
        f"{prefix}joint_draft_m2t_credit_rows": session.step_joint_draft_m2t_credit_row_count,
        f"{prefix}joint_draft_m2t_credit_expected_error_mean": expected_error_mean,
        f"{prefix}joint_draft_m2t_credit_multiplier_mean": multiplier_mean,
        f"{prefix}joint_draft_m2t_credit_multiplier_min": (
            session.step_joint_draft_m2t_credit_multiplier_min if count else 0.0
        ),
        f"{prefix}joint_draft_m2t_credit_multiplier_max": (
            session.step_joint_draft_m2t_credit_multiplier_max if count else 0.0
        ),
        f"{prefix}joint_draft_m2t_credit_expected_error_multiplier_covariance": covariance,
        f"{prefix}joint_draft_m2t_credit_base_schedule_coefficient_mass": base_schedule_mass,
        f"{prefix}joint_draft_m2t_credit_schedule_coefficient_mass": schedule_mass,
        f"{prefix}joint_draft_m2t_credit_schedule_coefficient_mass_ratio": (
            schedule_mass / base_schedule_mass if base_schedule_mass > 0.0 else 0.0
        ),
        f"{prefix}joint_draft_m2t_credit_clipped_low_tokens": (session.step_joint_draft_m2t_credit_clipped_low_count),
        f"{prefix}joint_draft_m2t_credit_clipped_high_tokens": (session.step_joint_draft_m2t_credit_clipped_high_count),
    }


def joint_two_view_log_metrics(session) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = _joint_two_view_common_metrics(session)
    if not metrics:
        return metrics
    metrics.update(
        {
            "last_step_joint_draft_m2t_loss": session.step_joint_draft_m2t_loss_sum
            / max(1, session.step_joint_two_view_count),
            "last_step_joint_correction_loss": session.step_joint_correction_loss_sum
            / max(1, session.step_joint_two_view_count),
            "last_step_joint_two_view_microbatches": session.step_joint_two_view_count,
            **_joint_two_view_credit_metrics(session, "last_step_"),
        }
    )
    return metrics


def joint_two_view_checkpoint_metrics(session) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = _joint_two_view_common_metrics(session)
    if not metrics:
        return metrics
    metrics.update(
        {
            "last_train_joint_draft_m2t_loss": session.step_joint_draft_m2t_loss_sum
            / max(1, session.step_joint_two_view_count),
            "last_train_joint_correction_loss": session.step_joint_correction_loss_sum
            / max(1, session.step_joint_two_view_count),
            "last_train_joint_two_view_microbatches": session.step_joint_two_view_count,
            **_joint_two_view_credit_metrics(session, "last_train_"),
        }
    )
    return metrics
