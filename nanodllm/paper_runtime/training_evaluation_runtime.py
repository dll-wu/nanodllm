"""RNG-isolated evaluation dependencies and periodic observation lifecycle."""

import random
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import ExperimentConfig
from .telemetry import append_jsonl, flush_tensorboard_writer
from .telemetry import write_tensorboard_scalars as _write_tensorboard_scalars
from .training_features import _is_causal
from .training_monitor import TrainingMonitor
from .training_rng import _restore_rng_state, _rng_state
from .training_runtime import print0


@dataclass(frozen=True, slots=True)
class EvaluationRuntime:
    """Dependencies shared by each evaluation call for one loaded model."""

    raw_model: nn.Module
    config: ExperimentConfig
    tokenizer: object
    noise: object
    autocast_ctx: object
    device: torch.device
    monitor: TrainingMonitor | None = None


def run_isolated_evaluation(runtime: EvaluationRuntime, step: int, include_gen: bool, evaluate) -> dict:
    """Run the fixed evaluation bundle without advancing the training RNG streams."""

    raw_model = runtime.raw_model
    config = runtime.config
    saved_state = _rng_state(config)
    was_training = raw_model.training
    eval_seed = int(config.eval.nelbo_seed)
    try:
        random.seed(eval_seed)
        np.random.seed(eval_seed % (2**32))
        torch.manual_seed(eval_seed)
        if config.runtime.device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        with runtime.monitor.phase("evaluation") if runtime.monitor is not None else nullcontext():
            return evaluate(
                raw_model,
                config,
                runtime.tokenizer,
                runtime.noise,
                runtime.autocast_ctx,
                runtime.device,
                step,
                include_gen,
            )
    finally:
        _restore_rng_state(saved_state, config)
        if was_training:
            raw_model.train()


def run_periodic_evaluation(session, step, evaluate):
    session.eval_start = time.time()
    session.eval_metrics = evaluate(
        session.evaluation_runtime,
        step + 1,
        include_gen=session.config.eval.gen_eval and session.config.eval.gen_eval_periodic,
    )
    session.eval_metrics["evaluation_seconds"] = time.time() - session.eval_start
    append_jsonl(session.eval_log_path, session.eval_metrics)
    _write_tensorboard_scalars(session.tensorboard_writer, "eval", session.eval_metrics, step + 1)
    flush_tensorboard_writer(session.tensorboard_writer)
    if _is_causal(session.config):
        print0(
            session.rank,
            f"eval step {step + 1:05d}: val_ce {session.eval_metrics.get('val_ce', 0.0):.4f} val_bpb {session.eval_metrics.get('val_bpb', 0.0):.4f}",
            flush=True,
        )
    else:
        print0(
            session.rank,
            f"eval step {step + 1:05d}: val_nelbo_bpb {session.eval_metrics['val_nelbo_bpb']:.4f}",
            flush=True,
        )
