"""Training-only scalar and per-row noise draws; evaluation owns separate plans."""

import random

import torch

from .config import ExperimentConfig
from .noise import sample_positive_uniform_scalar, sample_positive_uniform_tensor, sample_stratified_row_times


def sample_training_scalar_noise_time(config: ExperimentConfig) -> float:
    if config.noise.type == "mdlm" and bool(config.noise.schedule_weight):
        return sample_positive_uniform_scalar(config.noise.t_min, config.noise.t_max)
    return random.uniform(config.noise.t_min, config.noise.t_max)


def sample_training_tensor_noise_time(config: ExperimentConfig, shape: tuple[int, ...], device) -> torch.Tensor:
    if config.noise.mdlm_time_sampling == "stratified_per_row":
        if len(shape) != 1:
            raise ValueError("stratified_per_row requires one noise time per row")
        return sample_stratified_row_times(shape[0], device=device, t_min=config.noise.t_min, t_max=config.noise.t_max)
    if config.noise.type == "mdlm" and bool(config.noise.schedule_weight):
        return sample_positive_uniform_tensor(
            shape,
            device=device,
            t_min=float(config.noise.t_min),
            t_max=float(config.noise.t_max),
        )
    return torch.empty(shape, device=device, dtype=torch.float32).uniform_(
        float(config.noise.t_min),
        float(config.noise.t_max),
    )
