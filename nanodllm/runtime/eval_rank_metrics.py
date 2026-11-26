"""Conditional-rank scores and aggregation, independent of rollout execution."""

from collections import Counter

import torch

from . import data


def _single_token_text(tokenizer, token_id: int) -> str:
    if token_id < 0:
        return ""
    return tokenizer.decode([int(token_id)])


def _new_conditional_rank_stats() -> dict:
    return {
        "content_token_count": 0,
        "gold_ranks": [],
        "gold_probs": [],
        "top1_hits": 0,
        "top5_hits": 0,
        "top10_hits": 0,
        "top20_hits": 0,
        "top1_probs": [],
        "predicted_max_ids": [],
        "entropies": [],
        "margins": [],
    }


def _new_conditional_rank_decile_stats() -> dict[int, dict]:
    return {d: _new_conditional_rank_stats() for d in range(1, 8)}


def _content_position_mask(
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    token_bytes: torch.Tensor,
    special_ids: set[int],
) -> torch.Tensor:
    content = attention_mask & loss_mask & (token_bytes[x_0] > 0)
    for sid in special_ids:
        content &= x_0 != int(sid)
    return content


def _score_conditional_rank_position(
    pos_probs: torch.Tensor,
    token_id: int,
    vocab_size: int,
    topk_k: int,
) -> dict[str, float | int | bool]:
    gold_prob = float(pos_probs[token_id].item())
    rank = int((pos_probs > pos_probs[token_id]).sum().item()) + 1
    topk_vals, topk_ids = torch.topk(pos_probs, topk_k)
    top1_prob = float(topk_vals[0].item())
    predicted_max_id = int(topk_ids[0].item())
    top1_hit = bool((topk_ids[0] == token_id).item())
    top5_hit = bool((topk_ids[:5] == token_id).any().item()) if topk_k >= 5 else top1_hit
    top10_hit = bool((topk_ids[:10] == token_id).any().item()) if topk_k >= 10 else top5_hit
    top20_hit = bool((topk_ids[:20] == token_id).any().item()) if topk_k >= 20 else top10_hit
    entropy = float((-pos_probs * torch.log(pos_probs + 1e-10)).sum().item())
    margin = float((topk_vals[0] - topk_vals[1]).item()) if vocab_size > 1 else 0.0
    return {
        "rank": rank,
        "gold_prob": gold_prob,
        "top1_hit": top1_hit,
        "top5_hit": top5_hit,
        "top10_hit": top10_hit,
        "top20_hit": top20_hit,
        "top1_prob": top1_prob,
        "predicted_max_id": predicted_max_id,
        "entropy": entropy,
        "margin": margin,
    }


def _update_conditional_rank_stats(stats: dict, scored: dict[str, float | int | bool]) -> None:
    stats["content_token_count"] += 1
    stats["gold_ranks"].append(int(scored["rank"]))
    stats["gold_probs"].append(float(scored["gold_prob"]))
    stats["top1_hits"] += int(bool(scored["top1_hit"]))
    stats["top5_hits"] += int(bool(scored["top5_hit"]))
    stats["top10_hits"] += int(bool(scored["top10_hit"]))
    stats["top20_hits"] += int(bool(scored["top20_hit"]))
    stats["top1_probs"].append(float(scored["top1_prob"]))
    stats["predicted_max_ids"].append(int(scored["predicted_max_id"]))
    stats["entropies"].append(float(scored["entropy"]))
    stats["margins"].append(float(scored["margin"]))


def _record_conditional_rank_position(
    decile_stats: dict[int, dict],
    suffix_stats: dict,
    token_to_decile: dict[int, int],
    token_id: int,
    scored: dict[str, float | int | bool],
) -> None:
    decile = token_to_decile.get(token_id, 1)
    _update_conditional_rank_stats(decile_stats[decile], scored)
    _update_conditional_rank_stats(suffix_stats, scored)


def _fill_conditional_rank_metrics(stats: dict, prefix: str, metrics_out: dict, tokenizer) -> None:
    count = stats["content_token_count"]
    metrics_out[f"{prefix}content_token_count"] = count
    if count == 0:
        metrics_out[f"{prefix}gold_rank_mean"] = 0.0
        metrics_out[f"{prefix}gold_prob_mean"] = 0.0
        metrics_out[f"{prefix}gold_top1_hit"] = 0.0
        metrics_out[f"{prefix}gold_top5_hit"] = 0.0
        metrics_out[f"{prefix}gold_top10_hit"] = 0.0
        metrics_out[f"{prefix}gold_top20_hit"] = 0.0
        metrics_out[f"{prefix}top1_prob_mean"] = 0.0
        metrics_out[f"{prefix}predicted_max_token_id"] = -1
        metrics_out[f"{prefix}predicted_max_token_text"] = ""
        metrics_out[f"{prefix}predicted_max_token_count"] = 0
        metrics_out[f"{prefix}predicted_max_token_fraction"] = 0.0
        metrics_out[f"{prefix}entropy_mean"] = 0.0
        metrics_out[f"{prefix}margin_mean"] = 0.0
        return
    metrics_out[f"{prefix}gold_rank_mean"] = sum(stats["gold_ranks"]) / count
    metrics_out[f"{prefix}gold_prob_mean"] = sum(stats["gold_probs"]) / count
    metrics_out[f"{prefix}gold_top1_hit"] = stats["top1_hits"] / count
    metrics_out[f"{prefix}gold_top5_hit"] = stats["top5_hits"] / count
    metrics_out[f"{prefix}gold_top10_hit"] = stats["top10_hits"] / count
    metrics_out[f"{prefix}gold_top20_hit"] = stats["top20_hits"] / count
    metrics_out[f"{prefix}top1_prob_mean"] = sum(stats["top1_probs"]) / count
    max_counter = Counter(stats["predicted_max_ids"])
    most_common_id, most_common_count = max_counter.most_common(1)[0]
    metrics_out[f"{prefix}predicted_max_token_id"] = most_common_id
    metrics_out[f"{prefix}predicted_max_token_text"] = _single_token_text(tokenizer, most_common_id)
    metrics_out[f"{prefix}predicted_max_token_count"] = most_common_count
    metrics_out[f"{prefix}predicted_max_token_fraction"] = most_common_count / count
    metrics_out[f"{prefix}entropy_mean"] = sum(stats["entropies"]) / count
    metrics_out[f"{prefix}margin_mean"] = sum(stats["margins"]) / count


def _conditional_rank_frequency_deciles(
    data_config,
    tokenizer,
    batch_size: int,
    seq_len: int,
    device,
    batches: int,
    token_bytes: torch.Tensor,
    special_ids: set[int],
) -> tuple[dict[int, int], int]:
    train_loader = data.make_dataloader(data_config, tokenizer, batch_size, seq_len, "train", device=device)
    freq_counts = Counter()
    freq_batches = max(batches * 2, 10)
    for _ in range(freq_batches):
        batch = next(train_loader)
        content = _content_position_mask(batch.x_0, batch.attention_mask, batch.loss_mask, token_bytes, special_ids)
        if content.any():
            ids = batch.x_0[content].detach().cpu().tolist()
            freq_counts.update(ids)

    vocab_size = tokenizer.get_vocab_size()
    all_freqs = [(freq_counts.get(tid, 0), tid) for tid in range(vocab_size)]
    all_freqs.sort(key=lambda x: x[0])
    decile_size = vocab_size // 7
    remainder = vocab_size % 7
    token_to_decile: dict[int, int] = {}
    idx = 0
    for decile in range(1, 8):
        size = decile_size + (1 if decile <= remainder else 0)
        for _ in range(size):
            token_to_decile[all_freqs[idx][1]] = decile
            idx += 1
    return token_to_decile, freq_batches


def _selected_suffix_positions(content_positions: torch.Tensor, limit: int) -> list[int]:
    if content_positions.numel() <= limit:
        return [int(pos) for pos in content_positions.detach().cpu().tolist()]
    sample_offsets = torch.linspace(
        0,
        content_positions.numel() - 1,
        steps=limit,
        device=content_positions.device,
    ).round()
    return [int(pos) for pos in content_positions[sample_offsets.to(torch.long)].detach().cpu().tolist()]
