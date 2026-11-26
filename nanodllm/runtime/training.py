"""Stable training API façade.

Implementation lives in the flat ``training_*`` services. External callers use
the small public surface below rather than importing private service helpers.
"""

from .telemetry import write_json_atomic
from .training_artifacts import (
    autoresearch_config_summary,
    build_blocked_autoresearch_metrics,
    source_promotion_metrics,
)
from .training_loop import train
from .training_replay import eval_only_replay

__all__ = [
    "train",
    "eval_only_replay",
    "autoresearch_config_summary",
    "build_blocked_autoresearch_metrics",
    "source_promotion_metrics",
    "write_json_atomic",
]
