"""Checkpoint destination policy and legacy-compatible filesystem discovery."""

from __future__ import annotations

import os

from ..training_checkpoint_paths import (
    _checkpoint_step as _checkpoint_step,
)
from ..training_checkpoint_paths import (
    _latest_checkpoint_in_run_dir as _latest_checkpoint_in_run_dir,
)
from ..training_checkpoint_paths import (
    _resolve_resume_path as _resolve_resume_path,
)
from ..training_checkpoint_paths import (
    _step_checkpoint_candidates as _step_checkpoint_candidates,
)
from .ar_width_mup import active as ar_width_active
from .config import ExperimentConfig


def periodic_path(config: ExperimentConfig, step: int) -> str:
    """The scoped experiment retains one recovery checkpoint; ordinary runs keep steps."""
    name = "model_latest.pt" if ar_width_active(config) else f"model_step{step}.pt"
    return os.path.join(config.train.run_dir, name)
