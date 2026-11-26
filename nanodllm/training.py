"""Public training API for AR, full-sequence MDLM, and block diffusion."""

from .runtime.lr_schedule import get_lr_multiplier
from .runtime.training import eval_only_replay, train
from .runtime.training_artifacts import autoresearch_config_summary, write_jsonl_atomic
from .runtime.training_checkpoint import alias_checkpoint, save_checkpoint
from .runtime.training_optim import make_optimizer
from .runtime.training_replay import eval_only_basin_replay
from .runtime.training_runtime import dtype_from_name, init_distributed, print0
from .source_provenance import source_provenance_metrics


def run_config_summary(config, run_mode):
    """Include the original preview's run mode in the native config summary."""
    return {**autoresearch_config_summary(config, run_mode), "run_mode": run_mode}


__all__ = [
    "alias_checkpoint",
    "dtype_from_name",
    "eval_only_basin_replay",
    "eval_only_replay",
    "get_lr_multiplier",
    "init_distributed",
    "make_optimizer",
    "print0",
    "run_config_summary",
    "save_checkpoint",
    "source_provenance_metrics",
    "train",
    "write_jsonl_atomic",
]
