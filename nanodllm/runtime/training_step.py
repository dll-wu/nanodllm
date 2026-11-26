"""Run optimizer updates over the prefetched training stream."""

from __future__ import annotations

import os

from .telemetry import append_jsonl, flush_tensorboard_writer
from .telemetry import write_tensorboard_scalars as _write_tensorboard_scalars
from .training_block_conversion_step import run_block_conversion_microbatch
from .training_causal_microbatch import run_causal_microbatch
from .training_diffusion_microbatch import run_diffusion_microbatch
from .training_empty_target import backward_empty_ddp_microbatch
from .training_evaluation import _evaluate_fixed_noise_ce_trace, _run_eval_with_runtime
from .training_evaluation_runtime import run_periodic_evaluation
from .training_features import _is_causal, _uses_bd3lm_causal_clean_stream
from .training_mdlm_auxiliary import run_mdlm_auxiliary_microbatch
from .training_periodic_checkpoint import checkpoint_if_due
from .training_state import TrainingSession
from .training_step_counters import _prepare_training_step, _reduce_step_accumulators
from .training_step_log import log_training_interval
from .training_step_statistics import accumulate_objective_totals, check_training_diagnostics, record_step_outcome
from .training_step_update import apply_optimizer_step
from .training_variable_depth import microbatch_kwargs as variable_depth_kwargs


def _accumulate_microbatch(session: TrainingSession, step: int, micro: int) -> None:
    depth_kwargs = variable_depth_kwargs(session.config, step, micro, session.grad_accum_steps)
    session.step_processed_tokens += int(session.x_0.numel())
    if _is_causal(session.config):
        run_causal_microbatch(session, step, micro, depth_kwargs)
        return
    if _uses_bd3lm_causal_clean_stream(session.config):
        run_block_conversion_microbatch(
            session, step=step, micro=micro, backward_empty_ddp_microbatch=backward_empty_ddp_microbatch
        )
        return
    if run_mdlm_auxiliary_microbatch(session, step=step, micro=micro):
        return
    run_diffusion_microbatch(session, step, micro, depth_kwargs)


def _advance_training_batch(session: TrainingSession) -> None:
    """Prefetch the next batch before the completed step's resume state is captured."""
    session.batch = next(session.train_loader)
    session.x_0 = session.batch.x_0
    session.loss_mask = session.batch.loss_mask
    session.attention_mask = session.batch.attention_mask
    session.epoch = session.batch.epoch


def _record_fixed_ce_trace(session: TrainingSession, step: int) -> None:
    session.trace_record = _evaluate_fixed_noise_ce_trace(
        session.raw_model,
        session.config,
        session.tokenizer,
        session.noise,
        session.autocast_ctx,
        session.device,
        step + 1,
        session.total_steps,
        session.train_tokens,
        session.train_valid_tokens,
    )
    session.trace_record["fixed_ce_trace_checkpoint_index"] = session.progress.diagnostics.fixed_ce_trace_records + 1
    os.makedirs(os.path.dirname(session.fixed_ce_trace_path), exist_ok=True)
    append_jsonl(session.fixed_ce_trace_path, session.trace_record)
    _write_tensorboard_scalars(session.tensorboard_writer, "fixed_ce_trace", session.trace_record, step + 1)
    flush_tensorboard_writer(session.tensorboard_writer)
    session.progress.diagnostics.fixed_ce_trace_records += 1
    session.progress.diagnostics.last_fixed_ce_trace_record = session.trace_record


def _run_training_steps(session: TrainingSession) -> None:
    for step in range(session.start_step, session.total_steps):
        _prepare_training_step(session, step)
        for micro in range(session.grad_accum_steps):
            _accumulate_microbatch(session, step, micro)
            _advance_training_batch(session)
        _reduce_step_accumulators(session)
        accumulate_objective_totals(session)
        apply_optimizer_step(session, step)
        record_step_outcome(session, step)
        check_training_diagnostics(session, step)
        if (step + 1) % session.config.train.log_interval == 0 or step + 1 == session.total_steps:
            log_training_interval(session, step)
        if session.rank == 0 and step + 1 in session.fixed_ce_trace_steps:
            _record_fixed_ce_trace(session, step)
        checkpoint_if_due(session, step)
        if (
            session.config.eval.enabled
            and session.config.train.eval_every > 0
            and (session.rank == 0)
            and ((step + 1) % session.config.train.eval_every == 0)
            and (step + 1 < session.total_steps)
        ):
            run_periodic_evaluation(session, step, _run_eval_with_runtime)
