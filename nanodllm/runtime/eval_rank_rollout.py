"""Conditional rank under paired model-committed and gold-committed contexts."""

import json
import os
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F

from . import data
from .eval_generation_artifacts import _decode_attention_row, _decode_masked_attention_row
from .eval_model import _batch_model_attention_mask, _call_model_logits
from .eval_rank_metrics import (
    _conditional_rank_frequency_deciles,
    _content_position_mask,
    _fill_conditional_rank_metrics,
    _new_conditional_rank_decile_stats,
    _new_conditional_rank_stats,
    _record_conditional_rank_position,
    _score_conditional_rank_position,
    _single_token_text,
)
from .eval_rollout_commits import _fill_rollout_trace_metrics, _new_rollout_trace, _rollout_suffix_commit_order


def _commit_fraction_label(fraction: float) -> str:
    return f"commit{int(round(float(fraction) * 100)):02d}"


def _rollout_target_counts(candidate: torch.Tensor, fraction: float) -> torch.Tensor:
    totals = candidate.sum(dim=-1)
    raw = torch.ceil(totals.float() * float(fraction)).to(dtype=torch.long)
    bounded = torch.minimum(raw.clamp(min=1), (totals - 1).clamp(min=0))
    return torch.where(totals > 1, bounded, torch.zeros_like(raw))


def _score_rollout_context(
    model,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    context: torch.Tensor,
    score_mask: torch.Tensor,
    tokenizer,
    token_to_decile: dict[int, int],
    model_logits_fn,
    vocab_size: int,
    topk_k: int,
) -> tuple[dict[int, dict], dict]:
    decile_stats = _new_conditional_rank_decile_stats()
    suffix_stats = _new_conditional_rank_stats()
    if not score_mask.any():
        return decile_stats, suffix_stats
    logits = _call_model_logits(
        model,
        context,
        x_0,
        attention_mask,
        model_logits_fn=model_logits_fn,
        position_ids=position_ids,
    )
    probs = F.softmax(logits.float(), dim=-1)
    rows, positions = score_mask.nonzero(as_tuple=True)
    for row_idx, pos in zip(rows.detach().cpu().tolist(), positions.detach().cpu().tolist(), strict=True):
        token_id = int(x_0[row_idx, pos].item())
        scored = _score_conditional_rank_position(probs[row_idx, pos], token_id, vocab_size, topk_k)
        _record_conditional_rank_position(decile_stats, suffix_stats, token_to_decile, token_id, scored)
    return decile_stats, suffix_stats


def _update_rollout_commit_metrics(
    metrics: dict,
    label: str,
    committed: torch.Tensor,
    model_tokens: torch.Tensor,
    x_0: torch.Tensor,
    tokenizer,
    eos_id: int | None,
    token_counter: Counter[int],
) -> None:
    prefix = f"conditional_rank_rollout_{label}_"
    committed_count = int(committed.sum().item())
    if committed_count:
        committed_model = model_tokens[committed]
        committed_gold = x_0[committed]
        exact = int((committed_model == committed_gold).sum().item())
        ids = [int(i) for i in committed_model.detach().cpu().tolist()]
        token_counter.update(ids)
        max_token_id, max_token_count = Counter(ids).most_common(1)[0]
        eos_count = sum(1 for token_id in ids if eos_id is not None and token_id == int(eos_id))
    else:
        exact = 0
        max_token_id, max_token_count = -1, 0
        eos_count = 0
    total_committed = int(metrics.get(f"{prefix}committed_content_count", 0)) + committed_count
    total_exact = int(metrics.get(f"{prefix}commit_exact_match_count", 0)) + exact
    total_eos = int(metrics.get(f"{prefix}eos_commit_count", 0)) + eos_count
    if token_counter:
        max_token_id, max_token_count = token_counter.most_common(1)[0]
    metrics[f"{prefix}committed_content_count"] = total_committed
    metrics[f"{prefix}commit_exact_match_count"] = total_exact
    metrics[f"{prefix}commit_exact_match_rate"] = total_exact / total_committed if total_committed else 0.0
    metrics[f"{prefix}model_committed_max_token_id"] = int(max_token_id)
    metrics[f"{prefix}model_committed_max_token_text"] = _single_token_text(tokenizer, int(max_token_id))
    metrics[f"{prefix}model_committed_max_token_count"] = int(max_token_count)
    metrics[f"{prefix}model_committed_max_token_fraction"] = (
        max_token_count / total_committed if total_committed else 0.0
    )
    metrics[f"{prefix}eos_commit_count"] = int(total_eos)
    metrics[f"{prefix}eos_commit_rate"] = total_eos / total_committed if total_committed else 0.0


def _write_rollout_token_counts(
    run_dir: str | None,
    token_counts: dict[str, Counter[int]],
    tokenizer,
    top_k: int = 50,
) -> tuple[str, int]:
    if run_dir is None:
        return "", 0
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "conditional_rank_rollout_committed_tokens.jsonl")
    records = 0
    with open(path, "w", encoding="utf-8") as f:
        for label in sorted(token_counts):
            total = sum(token_counts[label].values())
            for token_id, count in token_counts[label].most_common(int(top_k)):
                f.write(
                    json.dumps(
                        {
                            "commit_label": label,
                            "token_id": int(token_id),
                            "token_text": _single_token_text(tokenizer, int(token_id)),
                            "count": int(count),
                            "fraction": count / total if total else 0.0,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                records += 1
    return path, records


def _log_rollout_examples(
    log_file,
    tokenizer,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    model_context: torch.Tensor,
    gold_context: torch.Tensor,
    committed: torch.Tensor,
    label: str,
    logged: int,
    log_samples: int,
    step: int | None,
) -> int:
    if log_file is None or logged >= log_samples:
        return logged
    mask_id = int(tokenizer.get_mask_token_id())
    eos_id = tokenizer.get_eos_token_id()
    for row_idx in range(x_0.shape[0]):
        if logged >= log_samples:
            break
        if not bool(committed[row_idx].any().item()):
            continue
        commit_count = int(committed[row_idx].sum().item())
        exact_count = int((model_context[row_idx, committed[row_idx]] == x_0[row_idx, committed[row_idx]]).sum().item())
        log_file.write(
            f"[conditional-rank-rollout] step={step if step is not None else '-'} "
            f"item={logged} {label} row={row_idx} commit_exact={exact_count}/{commit_count}\n"
        )
        log_file.write(
            "[conditional-rank-rollout] target="
            f"{_decode_attention_row(tokenizer, x_0[row_idx], attention_mask[row_idx])!r}\n"
        )
        model_text = _decode_masked_attention_row(
            tokenizer,
            model_context[row_idx],
            attention_mask[row_idx],
            mask_id,
            eos_id,
        )
        log_file.write(f"[conditional-rank-rollout] model_context={model_text!r}\n")
        gold_text = _decode_masked_attention_row(
            tokenizer,
            gold_context[row_idx],
            attention_mask[row_idx],
            mask_id,
            eos_id,
        )
        log_file.write(f"[conditional-rank-rollout] gold_context={gold_text!r}\n")
        logged += 1
    return logged


@torch.no_grad()
def evaluate_conditional_rank_rollout(
    model,
    tokenizer,
    data_config,
    device,
    batches=5,
    batch_size: int = 8,
    seq_len: int = data.MAX_SEQ_LEN,
    model_logits_fn=None,
    commit_fractions: list[float] | tuple[float, ...] = (0.25, 0.5, 0.75),
    sample_steps: int = 64,
    sampler: str = "default",
    scoring_mode: str = "margin_jitter",
    token_selection: str = "argmax",
    temperature: float = 1.0,
    jitter_scale: float = 0.02,
    fhs_tau: float = 0.9,
    run_dir: str | None = None,
    log_samples: int = 8,
    step: int | None = None,
) -> dict[str, float | int | str]:
    """Score paired model-committed versus gold-committed suffix contexts."""
    if batches <= 0:
        return {}

    start_time = time.time()
    special_ids = tokenizer.get_special_token_ids()
    token_bytes = data.build_token_bytes(tokenizer).to(device)
    token_to_decile, freq_batches = _conditional_rank_frequency_deciles(
        data_config,
        tokenizer,
        batch_size,
        seq_len,
        device,
        batches,
        token_bytes,
        special_ids,
    )
    val_loader = data.make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)
    mask_id = int(tokenizer.get_mask_token_id())
    eos_id = tokenizer.get_eos_token_id()
    prefix_len = seq_len // 2
    vocab_size = tokenizer.get_vocab_size()
    topk_k = min(20, vocab_size)
    fractions = [float(fraction) for fraction in commit_fractions]
    labels = [_commit_fraction_label(fraction) for fraction in fractions]

    decile_stats: dict[tuple[str, str], dict[int, dict]] = {}
    suffix_stats: dict[tuple[str, str], dict] = {}
    for label in labels:
        for source in ("model", "gold"):
            decile_stats[(source, label)] = _new_conditional_rank_decile_stats()
            suffix_stats[(source, label)] = _new_conditional_rank_stats()

    trace = _new_rollout_trace()
    token_counts: dict[str, Counter[int]] = {label: Counter() for label in labels}
    metrics: dict[str, float | int | str] = {
        "conditional_rank_rollout_batches": int(batches),
        "conditional_rank_rollout_freq_batches": int(freq_batches),
        "conditional_rank_rollout_sample_steps": int(sample_steps),
        "conditional_rank_rollout_sampler": sampler,
        "conditional_rank_rollout_scoring_mode": scoring_mode,
        "conditional_rank_rollout_token_selection": token_selection,
        "conditional_rank_rollout_temperature": float(temperature),
        "conditional_rank_rollout_jitter_scale": float(jitter_scale),
        "conditional_rank_rollout_fhs_tau": float(fhs_tau),
        "conditional_rank_rollout_commit_fraction_count": len(fractions),
    }
    for label, fraction in zip(labels, fractions, strict=True):
        metrics[f"conditional_rank_rollout_{label}_target_fraction"] = float(fraction)

    logged = 0
    samples_path = ""
    log_file = None
    log_stack = ExitStack()
    if run_dir is not None and log_samples > 0:
        os.makedirs(run_dir, exist_ok=True)
        samples_path = os.path.join(run_dir, "conditional_rank_rollout_samples.log")
        log_file = log_stack.enter_context(Path(samples_path).open("a", encoding="utf-8"))  # noqa: SIM115
        log_file.write(
            f"[conditional-rank-rollout] step={step if step is not None else '-'} "
            f"batches={int(batches)} fractions={','.join(labels)} sampler={sampler} "
            f"scoring={scoring_mode} token_selection={token_selection}\n"
        )

    try:
        for _ in range(int(batches)):
            batch = next(val_loader)
            x_0 = batch.x_0
            attention_mask = batch.attention_mask
            model_attention_mask = _batch_model_attention_mask(batch)
            position_ids = getattr(batch, "position_ids", None)
            loss_mask = batch.loss_mask
            content = _content_position_mask(x_0, attention_mask, loss_mask, token_bytes, special_ids)
            candidate = content.clone()
            candidate[:, :prefix_len] = False
            suffix_mask = attention_mask.clone()
            suffix_mask[:, :prefix_len] = False
            commit_rank, model_tokens = _rollout_suffix_commit_order(
                model,
                x_0,
                model_attention_mask,
                position_ids,
                candidate,
                suffix_mask,
                tokenizer,
                model_logits_fn,
                int(sample_steps),
                sampler,
                scoring_mode,
                token_selection,
                float(temperature),
                float(jitter_scale),
                float(fhs_tau),
                trace,
            )

            for label, fraction in zip(labels, fractions, strict=True):
                target_counts = _rollout_target_counts(candidate, fraction)
                committed = torch.zeros_like(candidate)
                for row_idx in range(candidate.shape[0]):
                    target_count = int(target_counts[row_idx].item())
                    if target_count <= 0:
                        continue
                    committed[row_idx] = (
                        candidate[row_idx] & (commit_rank[row_idx] > 0) & (commit_rank[row_idx] <= target_count)
                    )
                remaining = candidate & ~committed
                model_context = x_0.clone()
                gold_context = x_0.clone()
                model_context[suffix_mask] = mask_id
                gold_context[suffix_mask] = mask_id
                model_context[committed] = model_tokens[committed]
                gold_context[committed] = x_0[committed]

                candidate_count = int(candidate.sum().item())
                remaining_count = int(remaining.sum().item())
                metrics[f"conditional_rank_rollout_{label}_candidate_content_count"] = (
                    int(metrics.get(f"conditional_rank_rollout_{label}_candidate_content_count", 0)) + candidate_count
                )
                metrics[f"conditional_rank_rollout_{label}_remaining_masked_content_count"] = (
                    int(metrics.get(f"conditional_rank_rollout_{label}_remaining_masked_content_count", 0))
                    + remaining_count
                )
                metrics[f"conditional_rank_rollout_{label}_remaining_mask_count"] = int(
                    metrics.get(f"conditional_rank_rollout_{label}_remaining_mask_count", 0)
                ) + int(((model_context == mask_id) & attention_mask).sum().item())
                _update_rollout_commit_metrics(
                    metrics,
                    label,
                    committed,
                    model_tokens,
                    x_0,
                    tokenizer,
                    eos_id,
                    token_counts[label],
                )
                if candidate_count:
                    running_committed = int(metrics[f"conditional_rank_rollout_{label}_committed_content_count"])
                    running_candidate = int(metrics[f"conditional_rank_rollout_{label}_candidate_content_count"])
                    metrics[f"conditional_rank_rollout_{label}_actual_fraction"] = (
                        running_committed / running_candidate if running_candidate else 0.0
                    )

                model_deciles, model_suffix = _score_rollout_context(
                    model,
                    x_0,
                    model_attention_mask,
                    position_ids,
                    model_context,
                    remaining,
                    tokenizer,
                    token_to_decile,
                    model_logits_fn,
                    vocab_size,
                    topk_k,
                )
                gold_deciles, gold_suffix = _score_rollout_context(
                    model,
                    x_0,
                    model_attention_mask,
                    position_ids,
                    gold_context,
                    remaining,
                    tokenizer,
                    token_to_decile,
                    model_logits_fn,
                    vocab_size,
                    topk_k,
                )
                for decile in range(1, 8):
                    for key in model_deciles[decile]:
                        if isinstance(model_deciles[decile][key], list):
                            decile_stats[("model", label)][decile][key].extend(model_deciles[decile][key])
                        else:
                            decile_stats[("model", label)][decile][key] += model_deciles[decile][key]
                    for key in gold_deciles[decile]:
                        if isinstance(gold_deciles[decile][key], list):
                            decile_stats[("gold", label)][decile][key].extend(gold_deciles[decile][key])
                        else:
                            decile_stats[("gold", label)][decile][key] += gold_deciles[decile][key]
                for key in model_suffix:
                    if isinstance(model_suffix[key], list):
                        suffix_stats[("model", label)][key].extend(model_suffix[key])
                    else:
                        suffix_stats[("model", label)][key] += model_suffix[key]
                for key in gold_suffix:
                    if isinstance(gold_suffix[key], list):
                        suffix_stats[("gold", label)][key].extend(gold_suffix[key])
                    else:
                        suffix_stats[("gold", label)][key] += gold_suffix[key]

                logged = _log_rollout_examples(
                    log_file,
                    tokenizer,
                    x_0,
                    attention_mask,
                    model_context,
                    gold_context,
                    committed,
                    label,
                    logged,
                    int(log_samples),
                    step,
                )
    finally:
        log_stack.close()

    for label in labels:
        for source in ("model", "gold"):
            for decile in range(1, 8):
                _fill_conditional_rank_metrics(
                    decile_stats[(source, label)][decile],
                    f"conditional_rank_rollout_{source}_{label}_decile_{decile}_",
                    metrics,
                    tokenizer,
                )
            _fill_conditional_rank_metrics(
                suffix_stats[(source, label)],
                f"conditional_rank_rollout_{source}_{label}_suffix_",
                metrics,
                tokenizer,
            )
    for label in labels:
        candidate_count = int(metrics.get(f"conditional_rank_rollout_{label}_candidate_content_count", 0))
        committed_count = int(metrics.get(f"conditional_rank_rollout_{label}_committed_content_count", 0))
        metrics[f"conditional_rank_rollout_{label}_actual_fraction"] = (
            committed_count / candidate_count if candidate_count else 0.0
        )

    token_counts_path, token_count_records = _write_rollout_token_counts(run_dir, token_counts, tokenizer)
    _fill_rollout_trace_metrics(trace, metrics)
    metrics["conditional_rank_rollout_logged_samples"] = int(logged)
    metrics["conditional_rank_rollout_samples_path"] = samples_path
    metrics["conditional_rank_rollout_committed_token_counts_path"] = token_counts_path
    metrics["conditional_rank_rollout_committed_token_count_records"] = int(token_count_records)
    metrics["conditional_rank_rollout_seconds"] = time.time() - start_time
    metrics["conditional_rank_rollout_peak_vram_mb"] = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    )
    return metrics
