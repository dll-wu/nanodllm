"""Unconditional generation scoring, sampling traces and sample artifacts."""

import json
import math
from collections import Counter

import torch

from . import data
from .eval_generation_artifacts import GenerationArtifacts
from .eval_generation_metrics import (
    _eos_block_metrics,
    _target_length_generation_metrics,
)
from .eval_generation_prior import _generation_prior_correction
from .eval_generation_surface import GenerationStats
from .eval_judge import _get_judge
from .eval_model import model_uses_target_length_conditioning
from .eval_reconstruction import _self_reconstruction_ce
from .eval_sample_trace import merge_sample_trace as _merge_sample_trace
from .eval_sample_trace import new_sample_trace_totals as _new_sample_trace_totals
from .eval_sample_trace import sample_trace_metrics as _sample_trace_metrics
from .sampling import sampling_logit_bias


@torch.no_grad()
def evaluate_generation_metrics(
    model,
    tokenizer,
    noise,
    sample_fn,
    batch_size: int,
    run_dir: str | None = None,
    device=None,
    gen_samples: int = 512,
    gen_len: int = 512,
    log_samples: int = 5,
    score_mode: str = "self",
    self_score_t: float = 0.5,
    judge_model_id: str = "Qwen/Qwen3-0.6B-Base",
    step: int | None = None,
    sample_label: str | None = None,
    model_logits_fn=None,
    block_size: int = 0,
) -> dict[str, float | int | str]:
    if score_mode not in ("none", "self", "judge"):
        raise ValueError(f"unknown generation score mode: {score_mode!r}")

    device = next(model.parameters()).device if device is None else device
    n_batches = max(1, math.ceil(gen_samples / batch_size))
    stats = GenerationStats(tokenizer, data.build_token_bytes(tokenizer).cpu())
    prior_logit_bias, prior_metrics = _generation_prior_correction(
        run_dir,
        tokenizer,
        batch_size,
        device,
        gen_len,
    )

    eos_block_counts: Counter[int] = Counter()
    self_ce_total = 0.0
    self_ce_count = 0
    judge_ce_total = 0.0
    target_trace_lengths: list[int] = []
    target_trace_pre_eos_positions: list[int] = []
    target_trace_post_eos_positions: list[int] = []
    target_trace_forced_eos_count = 0
    trace_totals = _new_sample_trace_totals()

    judge = _get_judge(device, judge_model_id) if score_mode == "judge" else None

    with GenerationArtifacts(run_dir, tokenizer, gen_len, step, sample_label) as artifacts:
        log = artifacts.log
        log(
            f"[gen] step={step if step is not None else '-'} "
            f"score_mode={score_mode} samples={gen_samples} len={gen_len}"
            f"{f' label={sample_label}' if sample_label else ''}"
            f"{' prior_correction=true' if prior_logit_bias is not None else ''}"
        )
        logged = 0
        remaining = int(gen_samples)
        for batch_idx in range(n_batches):
            current_batch = min(int(batch_size), remaining)
            remaining -= current_batch
            with sampling_logit_bias(prior_logit_bias):
                sample_out = sample_fn(model, current_batch, gen_len)
            if isinstance(sample_out, tuple):
                tokens, sample_trace = sample_out
                _merge_sample_trace(trace_totals, sample_trace)
                trace_lengths = sample_trace.get("target_length_values")
                trace_pre = sample_trace.get("target_length_pre_policy_eos_positions")
                trace_post = sample_trace.get("target_length_post_policy_eos_positions")
                if isinstance(trace_lengths, list) and isinstance(trace_pre, list) and isinstance(trace_post, list):
                    target_trace_lengths.extend(int(v) for v in trace_lengths)
                    target_trace_pre_eos_positions.extend(int(v) for v in trace_pre)
                    target_trace_post_eos_positions.extend(int(v) for v in trace_post)
                    target_trace_forced_eos_count += int(sample_trace.get("target_length_forced_eos_count", 0))
                log(
                    f"[gen-trace] step={step if step is not None else '-'} batch={batch_idx} "
                    f"{f'label={sample_label} ' if sample_label else ''}"
                    f"{json.dumps(sample_trace, sort_keys=True)}"
                )
            else:
                tokens = sample_out
            batch_texts = []
            for row_idx in range(tokens.shape[0]):
                row = [int(i) for i in tokens[row_idx].detach().cpu().tolist()]
                scored = stats.add(row)
                batch_texts.append(scored.text)
                if scored.has_eos and block_size > 0:
                    eos_block_counts[(len(scored.truncated) - 1) // block_size] += 1
                artifacts.record(
                    batch_idx, row_idx, row, scored.truncated, scored.has_eos, len(scored.content), scored.content_bytes
                )

                if logged < log_samples:
                    log(
                        f"[gen] step={step if step is not None else '-'} batch={batch_idx} item={row_idx} "
                        f"tokens={len(scored.truncated)} chars={len(scored.text)} eos={scored.has_eos} empty={not scored.text.strip()}"
                        f"{f' label={sample_label}' if sample_label else ''}"
                    )
                    log(f"[gen] text={scored.text[:300]!r}")
                    logged += 1

            artifacts.flush_samples()

            empty = sum(1 for txt in batch_texts if len(txt.strip()) == 0)
            if empty:
                log(
                    f"[gen] WARNING: step={step if step is not None else '-'} "
                    f"batch={batch_idx} empty={empty}/{current_batch}"
                )

            if score_mode == "self":
                ce_sum, ce_count = _self_reconstruction_ce(
                    model,
                    tokenizer,
                    noise,
                    tokens.to(device),
                    device,
                    float(self_score_t),
                    model_logits_fn=model_logits_fn,
                )
                self_ce_total += ce_sum
                self_ce_count += ce_count
                if ce_count:
                    log(
                        f"[gen] step={step if step is not None else '-'} batch={batch_idx} "
                        f"self_ce_t={float(self_score_t):.3f} self_ce={ce_sum / ce_count:.4f}"
                    )
            elif score_mode == "judge" and judge is not None:
                batch_ce = judge.score(batch_texts)
                log(
                    f"[gen] step={step if step is not None else '-'} batch={batch_idx} "
                    f"judge_ce={batch_ce:.4f} judge_ppl={math.exp(batch_ce):.2f}"
                )
                judge_ce_total += batch_ce

    sample_count = len(stats.texts)
    metrics = {f"gen_{key}": value for key, value in stats.metrics(int(gen_len)).items()}
    metrics["gen_score_mode"] = score_mode
    metrics["gen_token_entropy"] = metrics["gen_token_entropy_mean"]
    metrics["gen_eos_position_mean"] = metrics["gen_avg_decoded_tokens"]
    metrics.update(artifacts.sample_metrics(sample_count))
    for frac in (0.25, 0.5, 0.75):
        threshold = int(gen_len * frac)
        count = sum(1 for length in stats.decoded_lengths if length < threshold)
        metrics[f"gen_eos_before_{int(frac * 100)}pct"] = count / sample_count if sample_count else 0.0
    if block_size > 0:
        metrics.update(
            _eos_block_metrics("gen_", eos_block_counts, sample_count, (gen_len + block_size - 1) // block_size)
        )
    metrics.update(prior_metrics)
    if score_mode == "self":
        metrics["gen_self_score_t"] = float(self_score_t)
        metrics["gen_self_ce"] = self_ce_total / self_ce_count if self_ce_count else 0.0
        metrics["gen_self_ce_tokens"] = self_ce_count
    elif score_mode == "judge":
        metrics["gen_ppl"] = math.exp(judge_ce_total / n_batches)
    if int(trace_totals["batches"] or 0) > 0:
        trace_metrics = _sample_trace_metrics(trace_totals)
        metrics.update(trace_metrics)
        if "gen_trace_commit_tokens" in trace_metrics:
            metrics["sampler_commit_count"] = trace_metrics["gen_trace_commit_tokens"]
        if "gen_trace_candidate_entropy_mean" in trace_metrics:
            metrics["sampler_candidate_entropy"] = trace_metrics["gen_trace_candidate_entropy_mean"]
        if "gen_trace_pred_max_token_frac_mean" in trace_metrics:
            metrics["sampler_pred_max_token_frac"] = trace_metrics["gen_trace_pred_max_token_frac_mean"]
    if target_trace_lengths or model_uses_target_length_conditioning(model):
        metrics.update(
            _target_length_generation_metrics(
                target_trace_lengths,
                target_trace_pre_eos_positions,
                target_trace_post_eos_positions,
                target_trace_forced_eos_count,
                int(gen_len),
            )
        )

    return metrics
