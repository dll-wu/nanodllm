"""Resumable clipped-schedule selection for the D4 BD3 reconstruction.

BD3 (arXiv:2503.09573v3, sections 5.2–5.3 and appendix C.4) selects
mask-rate intervals by loss variance. The released implementation at
1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7 averages within-batch variances of
block-mean losses. It retains 1/t weighting with clipped t, without a width
factor, and calls the clipped estimator biased. We reproduce that criterion,
not a learned importance distribution or an unbiased full-interval estimator.
The small grid and update cadence below are explicit D4 budget adaptations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import variance

import torch


def positive_unit_samples(count: int, *, device, generator: torch.Generator) -> torch.Tensor:
    """Draw the nonzero float32 uniform grid; reject the singular zero endpoint."""
    resolution = 1 << 24
    return torch.randint(1, resolution, (count,), device=device, generator=generator).float() / resolution


def mask_rates(
    count: int,
    *,
    low: float,
    high: float,
    stratified: bool,
    device,
    generator: torch.Generator,
) -> torch.Tensor:
    if count < 1 or not 0 <= low < high <= 1:
        raise ValueError("mask rates require a positive count and 0 <= low < high <= 1")
    unit = positive_unit_samples(count, device=device, generator=generator)
    if stratified:
        unit = (torch.arange(count, device=device, dtype=torch.float32) + unit) / count
    return low + (high - low) * unit


@dataclass
class ClippedScheduleState:
    """JSON-only sufficient state; caller separately checkpoints the tuning RNG."""

    intervals: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 0.5), (0.3, 0.8), (0.5, 1.0))
    interval_steps: int = 352
    selected_index: int = 0
    last_tuned_step: int = 0
    tuning_count: int = 0
    calibration_forward_count: int = 0
    last_scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if any(
            type(value) is not int
            for value in (
                self.interval_steps,
                self.selected_index,
                self.last_tuned_step,
                self.tuning_count,
                self.calibration_forward_count,
            )
        ):
            raise ValueError("clipped-schedule progress and cadence must be integers")
        self.intervals = tuple(tuple(float(value) for value in pair) for pair in self.intervals)
        if not self.intervals or self.intervals[0] != (0.0, 1.0):
            raise ValueError("the first candidate must be the fixed full-range control (0, 1)")
        if any(len(pair) != 2 or not 0 <= pair[0] < pair[1] <= 1 or pair[0] > 0.5 for pair in self.intervals):
            raise ValueError("require 0 <= low <= 0.5 and low < high <= 1; upstream low>0.5 changes weighting")
        if len(set(self.intervals)) != len(self.intervals):
            raise ValueError("clipped-schedule candidates must be unique")
        if self.interval_steps <= 0 or not 0 <= self.selected_index < len(self.intervals):
            raise ValueError("invalid clipped-schedule cadence or selected index")
        if min(self.last_tuned_step, self.tuning_count, self.calibration_forward_count) < 0:
            raise ValueError("clipped-schedule progress cannot be negative")
        if self.last_scores and (
            len(self.last_scores) != len(self.intervals)
            or any(not math.isfinite(score) or score < 0 for score in self.last_scores)
        ):
            raise ValueError("invalid clipped-schedule variance scores")

    @property
    def bounds(self) -> tuple[float, float]:
        return self.intervals[self.selected_index]

    def should_tune(self, step: int) -> bool:
        if step < 0 or step < self.last_tuned_step:
            raise ValueError("optimizer step predates clipped-schedule state")
        return step // self.interval_steps > self.last_tuned_step // self.interval_steps

    def select(self, scores: list[float], *, step: int, forward_count: int) -> None:
        if not self.should_tune(step):
            raise ValueError("clipped-schedule selection is not due")
        if len(scores) != len(self.intervals) or any(not math.isfinite(x) or x < 0 for x in scores):
            raise ValueError("one finite nonnegative variance is required per candidate")
        if forward_count < len(self.intervals) * 2:
            raise ValueError("selection requires at least two batches per candidate")
        self.selected_index = min(range(len(scores)), key=scores.__getitem__)
        self.last_scores = list(scores)
        self.last_tuned_step = step
        self.tuning_count += 1
        self.calibration_forward_count += forward_count

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "intervals": [list(pair) for pair in self.intervals],
            "interval_steps": self.interval_steps,
            "selected_index": self.selected_index,
            "last_tuned_step": self.last_tuned_step,
            "tuning_count": self.tuning_count,
            "calibration_forward_count": self.calibration_forward_count,
            "last_scores": list(self.last_scores),
        }

    def load_state_dict(self, payload: dict) -> None:
        expected = set(self.state_dict())
        if set(payload) != expected or payload["version"] != 1:
            raise ValueError("invalid clipped-schedule state schema")
        restored = type(self)(**{key: value for key, value in payload.items() if key != "version"})
        if restored.intervals != self.intervals or restored.interval_steps != self.interval_steps:
            raise ValueError("clipped-schedule configuration changed across resume")
        self.selected_index = restored.selected_index
        self.last_tuned_step = restored.last_tuned_step
        self.tuning_count = restored.tuning_count
        self.calibration_forward_count = restored.calibration_forward_count
        self.last_scores = restored.last_scores


def loss_variance_score(block_losses: list[torch.Tensor]) -> float:
    """Mean sample variance across blocks/examples within each calibration batch.

    This follows upstream `_clipped_schedule_search`, rather than silently
    substituting variance of batch means or variance across gradient entries.
    """
    if len(block_losses) < 2:
        raise ValueError("variance calibration needs at least two independent batches")
    scores = []
    for values in block_losses:
        flat = values.detach().float().reshape(-1)
        if flat.numel() < 2 or not bool(torch.isfinite(flat).all()):
            raise ValueError("each calibration batch requires two finite block losses")
        scores.append(float(flat.var(unbiased=True)))
    return sum(scores) / len(scores)


def batch_mean_variance(block_losses: list[torch.Tensor]) -> float:
    """Secondary paper-motivated diagnostic; never used instead of the score."""
    return variance(float(values.detach().float().mean()) for values in block_losses)
