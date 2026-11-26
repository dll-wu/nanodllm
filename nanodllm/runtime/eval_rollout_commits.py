"""Commit ordering and sampling traces for conditional rank rollouts."""

import torch
import torch.nn.functional as F

from .eval_model import _block_generation_special_logits, _call_model_logits


def _new_rollout_trace() -> dict[str, float | int | None]:
    return {
        "batches": 0,
        "steps": 0,
        "candidate_count": 0,
        "commit_count": 0,
        "commit_count_min": None,
        "commit_count_max": None,
        "candidate_entropy_sum": 0.0,
        "candidate_top1_prob_sum": 0.0,
        "pred_max_token_frac_sum": 0.0,
        "pred_max_token_frac_steps": 0,
        "final_mask_count": 0,
    }


def _update_rollout_trace(
    trace: dict[str, float | int | None],
    candidate_mask: torch.Tensor,
    probs: torch.Tensor,
    x_pred: torch.Tensor,
    commit_mask: torch.Tensor,
) -> None:
    candidate_count = int(candidate_mask.sum().item())
    commit_count = int(commit_mask.sum().item())
    trace["steps"] = int(trace["steps"] or 0) + 1
    trace["candidate_count"] = int(trace["candidate_count"] or 0) + candidate_count
    trace["commit_count"] = int(trace["commit_count"] or 0) + commit_count
    trace["commit_count_min"] = (
        commit_count if trace["commit_count_min"] is None else min(int(trace["commit_count_min"]), commit_count)
    )
    trace["commit_count_max"] = (
        commit_count if trace["commit_count_max"] is None else max(int(trace["commit_count_max"]), commit_count)
    )
    if candidate_count == 0:
        return

    top1_prob = probs.max(dim=-1).values[candidate_mask]
    safe_probs = probs.clamp_min(1e-10)
    entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)[candidate_mask]
    trace["candidate_top1_prob_sum"] = float(trace["candidate_top1_prob_sum"] or 0.0) + float(top1_prob.sum().item())
    trace["candidate_entropy_sum"] = float(trace["candidate_entropy_sum"] or 0.0) + float(entropy.sum().item())

    pred_ids = x_pred[candidate_mask]
    if pred_ids.numel() > 0:
        counts = torch.unique(pred_ids, return_counts=True)[1]
        trace["pred_max_token_frac_sum"] = float(trace["pred_max_token_frac_sum"] or 0.0) + float(
            counts.max().item() / max(1, pred_ids.numel())
        )
        trace["pred_max_token_frac_steps"] = int(trace["pred_max_token_frac_steps"] or 0) + 1


def _fill_rollout_trace_metrics(trace: dict[str, float | int | None], metrics: dict) -> None:
    batches = int(trace["batches"] or 0)
    steps = int(trace["steps"] or 0)
    candidate_count = int(trace["candidate_count"] or 0)
    pred_frac_steps = int(trace["pred_max_token_frac_steps"] or 0)
    prefix = "conditional_rank_rollout_sampler_"
    metrics[f"{prefix}batches"] = batches
    metrics[f"{prefix}steps_mean"] = steps / batches if batches else 0.0
    metrics[f"{prefix}candidate_tokens"] = candidate_count
    metrics[f"{prefix}commit_tokens"] = int(trace["commit_count"] or 0)
    metrics[f"{prefix}commit_tokens_mean"] = int(trace["commit_count"] or 0) / steps if steps else 0.0
    metrics[f"{prefix}commit_tokens_min"] = int(trace["commit_count_min"] or 0)
    metrics[f"{prefix}commit_tokens_max"] = int(trace["commit_count_max"] or 0)
    metrics[f"{prefix}candidate_entropy_mean"] = (
        float(trace["candidate_entropy_sum"] or 0.0) / candidate_count if candidate_count else 0.0
    )
    metrics[f"{prefix}candidate_top1_prob_mean"] = (
        float(trace["candidate_top1_prob_sum"] or 0.0) / candidate_count if candidate_count else 0.0
    )
    metrics[f"{prefix}pred_max_token_frac_mean"] = (
        float(trace["pred_max_token_frac_sum"] or 0.0) / pred_frac_steps if pred_frac_steps else 0.0
    )
    metrics[f"{prefix}final_mask_count"] = int(trace["final_mask_count"] or 0)
    metrics[f"{prefix}final_mask_count_mean"] = int(trace["final_mask_count"] or 0) / batches if batches else 0.0


def _rollout_confidence(
    probs: torch.Tensor,
    scoring_mode: str,
    jitter_scale: float,
) -> torch.Tensor:
    if scoring_mode == "maxprob":
        return probs.max(dim=-1).values
    if scoring_mode in ("margin", "margin_jitter"):
        top2 = probs.topk(2, dim=-1).values
        conf = top2[..., 0] - top2[..., 1]
        if scoring_mode == "margin_jitter" and jitter_scale > 0:
            conf = conf + (-torch.empty_like(conf).exponential_().log() - 0.5772156649) * float(jitter_scale)
        return conf
    if scoring_mode == "entropy":
        safe_probs = probs.clamp_min(1e-10)
        return (safe_probs * safe_probs.log()).sum(dim=-1)
    if scoring_mode == "random":
        return torch.rand(probs.shape[:-1], device=probs.device)
    raise ValueError(f"unknown scoring mode: {scoring_mode}")


def _rollout_commit_mask(
    still_masked: torch.Tensor,
    probs: torch.Tensor,
    x_pred: torch.Tensor,
    step: int,
    sampler: str,
    scoring_mode: str,
    jitter_scale: float,
    fhs_tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampler == "fhs":
        selected_prob = probs.gather(-1, x_pred.unsqueeze(-1)).squeeze(-1)
        commit = still_masked & (selected_prob >= float(fhs_tau))
        priority = selected_prob
        if commit.any():
            return commit, priority

        p_unmask = 1.0 / max(1, int(step))
        k_per_seq = (p_unmask * still_masked.sum(dim=-1)).long().clamp(min=1)
        scores = selected_prob.masked_fill(~still_masked, -1e9)
        commit = torch.zeros_like(still_masked)
        for row_idx in range(still_masked.shape[0]):
            remaining = int(still_masked[row_idx].sum().item())
            if remaining == 0:
                continue
            k = min(remaining, int(k_per_seq[row_idx].item()))
            _, idx = torch.topk(scores[row_idx], k)
            commit[row_idx, idx] = True
        return commit, priority

    priority = _rollout_confidence(probs, scoring_mode, jitter_scale)
    p_unmask = 1.0 / max(1, int(step))
    if scoring_mode == "random":
        commit = still_masked & (priority < p_unmask)
        return commit, priority

    k_per_seq = (p_unmask * still_masked.sum(dim=-1)).long().clamp(min=1)
    scores = priority.masked_fill(~still_masked, -1e9)
    commit = torch.zeros_like(still_masked)
    for row_idx in range(still_masked.shape[0]):
        remaining = int(still_masked[row_idx].sum().item())
        if remaining == 0:
            continue
        k = min(remaining, int(k_per_seq[row_idx].item()))
        _, idx = torch.topk(scores[row_idx], k)
        commit[row_idx, idx] = True
    return commit, priority


def _assign_rollout_commit_ranks(
    commit_rank: torch.Tensor,
    model_tokens: torch.Tensor,
    row_commit_counts: torch.Tensor,
    commit: torch.Tensor,
    x_pred: torch.Tensor,
    priority: torch.Tensor,
) -> None:
    for row_idx in range(commit.shape[0]):
        positions = torch.nonzero(commit[row_idx], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        order = torch.argsort(priority[row_idx, positions].float(), descending=True)
        positions = positions[order]
        start = int(row_commit_counts[row_idx].item())
        ranks = torch.arange(
            start + 1,
            start + 1 + int(positions.numel()),
            device=commit.device,
            dtype=commit_rank.dtype,
        )
        commit_rank[row_idx, positions] = ranks
        model_tokens[row_idx, positions] = x_pred[row_idx, positions]
        row_commit_counts[row_idx] += int(positions.numel())


def _rollout_suffix_commit_order(
    model,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    candidate: torch.Tensor,
    suffix_mask: torch.Tensor,
    tokenizer,
    model_logits_fn,
    sample_steps: int,
    sampler: str,
    scoring_mode: str,
    token_selection: str,
    temperature: float,
    jitter_scale: float,
    fhs_tau: float,
    trace: dict[str, float | int | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_id = int(tokenizer.get_mask_token_id())
    z_model = x_0.clone()
    z_model[suffix_mask] = mask_id
    commit_rank = torch.full_like(x_0, -1, dtype=torch.long)
    model_tokens = torch.full_like(x_0, mask_id)
    row_commit_counts = torch.zeros(x_0.shape[0], device=x_0.device, dtype=torch.long)
    trace["batches"] = int(trace["batches"] or 0) + 1

    for step in range(max(1, int(sample_steps)), 0, -1):
        still_masked = candidate & (commit_rank < 0)
        if not still_masked.any():
            break
        logits = _call_model_logits(
            model,
            z_model,
            x_0,
            attention_mask,
            model_logits_fn=model_logits_fn,
            position_ids=position_ids,
        ).float()
        _block_generation_special_logits(logits, tokenizer)
        probs = F.softmax(logits / float(temperature), dim=-1)
        if token_selection == "argmax":
            x_pred = logits.argmax(dim=-1)
        elif token_selection == "multinomial":
            x_pred = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view_as(x_0)
        else:
            raise ValueError(f"unknown token selection: {token_selection}")

        commit, priority = _rollout_commit_mask(
            still_masked,
            probs,
            x_pred,
            step,
            sampler,
            scoring_mode,
            jitter_scale,
            fhs_tau,
        )
        commit &= still_masked
        _update_rollout_trace(trace, still_masked, probs, x_pred, commit)
        if commit.any():
            _assign_rollout_commit_ranks(commit_rank, model_tokens, row_commit_counts, commit, x_pred, priority)
            z_model[commit] = x_pred[commit]

    remaining = candidate & (commit_rank < 0)
    trace["final_mask_count"] = int(trace["final_mask_count"] or 0) + int(remaining.sum().item())
    if remaining.any():
        logits = _call_model_logits(
            model,
            z_model,
            x_0,
            attention_mask,
            model_logits_fn=model_logits_fn,
            position_ids=position_ids,
        ).float()
        _block_generation_special_logits(logits, tokenizer)
        probs = F.softmax(logits / float(temperature), dim=-1)
        x_pred = logits.argmax(dim=-1)
        priority = probs.gather(-1, x_pred.unsqueeze(-1)).squeeze(-1)
        _assign_rollout_commit_ranks(
            commit_rank,
            model_tokens,
            row_commit_counts,
            remaining,
            x_pred,
            priority,
        )
    return commit_rank, model_tokens
