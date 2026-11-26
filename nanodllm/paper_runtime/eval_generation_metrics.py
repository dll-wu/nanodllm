"""Deterministic token, EOS and target-length metrics; no model or data I/O."""

import math
from collections import Counter

import torch
import torch.nn.functional as F


def target_eos_lengths(
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None,
) -> torch.Tensor:
    if x_0.shape != attention_mask.shape:
        raise ValueError(f"x_0 and attention_mask shapes must match, got {x_0.shape} and {attention_mask.shape}")
    batch_size, seq_len = x_0.shape
    valid_lengths = attention_mask.to(torch.long).sum(dim=1).clamp_min(1)
    if eos_token_id is None:
        return valid_lengths
    positions = torch.arange(seq_len, device=x_0.device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
    eos_hits = (x_0 == int(eos_token_id)) & attention_mask
    first_eos = torch.where(eos_hits, positions, torch.full_like(positions, seq_len)).min(dim=1).values
    return torch.where(first_eos < seq_len, first_eos + 1, valid_lengths).to(torch.long)


def _truncate_at_eos(ids: list[int], eos_id: int | None) -> tuple[list[int], bool]:
    if eos_id is None or eos_id not in ids:
        return ids, False
    eos_pos = ids.index(eos_id)
    return ids[: eos_pos + 1], True


def _content_ids(ids: list[int], special_ids: set[int], eos_id: int | None) -> list[int]:
    truncated, _ = _truncate_at_eos(ids, eos_id)
    return [int(i) for i in truncated if int(i) not in special_ids]


def _token_byte_total(ids: list[int], token_bytes: torch.Tensor) -> int:
    vocab_size = int(token_bytes.numel())
    total = 0
    for token_id in ids:
        token_id = int(token_id)
        if 0 <= token_id < vocab_size:
            total += int(token_bytes[token_id].item())
    return total


def _distinct_ngram_ratio(sequences: list[list[int]], n: int) -> float:
    total = 0
    unique: set[tuple[int, ...]] = set()
    for ids in sequences:
        if len(ids) < n:
            continue
        for idx in range(len(ids) - n + 1):
            unique.add(tuple(ids[idx : idx + n]))
            total += 1
    return len(unique) / total if total else 0.0


def _repeat_ngram_rate(sequences: list[list[int]], n: int) -> float:
    rates = []
    for ids in sequences:
        if len(ids) < n:
            rates.append(0.0)
            continue
        seen: set[tuple[int, ...]] = set()
        repeats = 0
        total = 0
        for idx in range(len(ids) - n + 1):
            ngram = tuple(ids[idx : idx + n])
            repeats += int(ngram in seen)
            seen.add(ngram)
            total += 1
        rates.append(repeats / total if total else 0.0)
    return sum(rates) / len(rates) if rates else 0.0


def _max_token_fraction(ids: list[int]) -> float:
    if not ids:
        return 0.0
    return max(Counter(ids).values()) / len(ids)


def _token_entropy(ids: list[int]) -> float:
    if not ids:
        return 0.0
    counts = Counter(ids)
    denom = len(ids)
    return -sum((count / denom) * math.log(count / denom) for count in counts.values())


def _entropy_std(entropies: list[float]) -> float:
    if not entropies:
        return 0.0
    mean = sum(entropies) / len(entropies)
    return math.sqrt(sum((value - mean) ** 2 for value in entropies) / len(entropies))


def _eos_position_metrics(
    prefix: str,
    eos_position_counts: Counter[int],
    sample_count: int,
    max_len: int,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        f"{prefix}eos_position_observed": int(sum(eos_position_counts.values())),
        f"{prefix}eos_position_max_len": int(max_len),
    }
    for pos in range(max(0, int(max_len))):
        count = int(eos_position_counts.get(pos, 0))
        metrics[f"{prefix}eos_position_{pos:04d}_count"] = count
        metrics[f"{prefix}eos_position_{pos:04d}_rate"] = count / sample_count if sample_count else 0.0
    return metrics


def _eos_block_metrics(
    prefix: str,
    eos_block_counts: Counter[int],
    sample_count: int,
    num_blocks: int,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        f"{prefix}eos_block_observed": int(sum(eos_block_counts.values())),
        f"{prefix}eos_block_count": int(num_blocks),
    }
    for block_idx in range(max(0, int(num_blocks))):
        count = int(eos_block_counts.get(block_idx, 0))
        metrics[f"{prefix}eos_block_{block_idx}_count"] = count
        metrics[f"{prefix}eos_block_{block_idx}_rate"] = count / sample_count if sample_count else 0.0
    return metrics


def _new_eos_suffix_decile_stats(num_deciles: int = 7) -> dict[str, list[float]]:
    return {
        "counts": [0.0 for _ in range(int(num_deciles))],
        "correct": [0.0 for _ in range(int(num_deciles))],
        "rank_sums": [0.0 for _ in range(int(num_deciles))],
        "prob_sums": [0.0 for _ in range(int(num_deciles))],
    }


def _update_eos_suffix_decile_stats(
    stats: dict[str, list[float]],
    logits: torch.Tensor,
    pred: torch.Tensor,
    x_0: torch.Tensor,
    active: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_id: int | None,
) -> None:
    if eos_id is None:
        return
    eos_token_id = int(eos_id)
    if eos_token_id < 0 or eos_token_id >= logits.shape[-1]:
        return
    eos_active = active & (x_0 == eos_token_id)
    if not eos_active.any():
        return
    rows, cols = eos_active.nonzero(as_tuple=True)
    valid_lengths = attention_mask.to(torch.long).sum(dim=1).clamp_min(1)
    denom = valid_lengths.index_select(0, rows).sub(1).clamp_min(1)
    num_deciles = len(stats["counts"])
    deciles = torch.clamp((cols.float() / denom.float() * num_deciles).to(torch.long), max=num_deciles - 1)
    selected_logits = logits[rows, cols].float()
    eos_logits = selected_logits[:, eos_token_id]
    ranks = (selected_logits > eos_logits[:, None]).sum(dim=-1).to(torch.float32) + 1.0
    eos_probs = F.softmax(selected_logits, dim=-1)[:, eos_token_id]
    correct = (pred[rows, cols] == eos_token_id).to(torch.float32)
    for decile_idx in range(num_deciles):
        decile_mask = deciles == decile_idx
        if not decile_mask.any():
            continue
        count = float(decile_mask.sum().item())
        stats["counts"][decile_idx] += count
        stats["correct"][decile_idx] += float(correct[decile_mask].sum().item())
        stats["rank_sums"][decile_idx] += float(ranks[decile_mask].sum().item())
        stats["prob_sums"][decile_idx] += float(eos_probs[decile_mask].sum().item())


def _eos_suffix_decile_metrics(prefix: str, stats: dict[str, list[float]]) -> dict[str, float | int]:
    counts = stats["counts"]
    correct = stats["correct"]
    rank_sums = stats["rank_sums"]
    prob_sums = stats["prob_sums"]
    total_count = float(sum(counts))
    metrics: dict[str, float | int] = {
        f"{prefix}eos_active_count": int(total_count),
        f"{prefix}eos_token_acc": sum(correct) / total_count if total_count else 0.0,
        f"{prefix}eos_rank_mean": sum(rank_sums) / total_count if total_count else 0.0,
        f"{prefix}eos_prob_mean": sum(prob_sums) / total_count if total_count else 0.0,
    }
    for decile_idx, count in enumerate(counts, start=1):
        metrics[f"{prefix}eos_suffix_decile_{decile_idx}_count"] = int(count)
        metrics[f"{prefix}eos_suffix_decile_{decile_idx}_token_acc"] = correct[decile_idx - 1] / count if count else 0.0
        metrics[f"{prefix}eos_suffix_decile_{decile_idx}_rank_mean"] = (
            rank_sums[decile_idx - 1] / count if count else 0.0
        )
        metrics[f"{prefix}eos_suffix_decile_{decile_idx}_prob_mean"] = (
            prob_sums[decile_idx - 1] / count if count else 0.0
        )
    return metrics


def _ngram_entropy(sequences: list[list[int]], n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    counts: Counter[tuple[int, ...]] = Counter()
    total = 0
    for ids in sequences:
        if len(ids) < n:
            continue
        for idx in range(len(ids) - n + 1):
            counts[tuple(ids[idx : idx + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _target_length_generation_metrics(
    target_lengths: list[int],
    pre_eos_positions: list[int],
    post_eos_positions: list[int],
    forced_eos_count: int,
    gen_len: int,
) -> dict[str, float | int]:
    if not target_lengths:
        return {
            "gen_target_length_conditioning_applied": 0,
            "gen_target_length_sample_count": 0,
            "gen_target_length_pre_policy_eos_count": 0,
            "gen_target_length_post_policy_eos_count": 0,
            "gen_target_length_forced_eos_count": 0,
            "gen_target_length_no_forced_eos_success_count": 0,
        }
    sample_count = len(target_lengths)
    pre_errors = [
        abs(int(pos) - (int(target_length) - 1))
        for target_length, pos in zip(target_lengths, pre_eos_positions, strict=True)
        if int(pos) >= 0
    ]
    post_errors = [
        abs(int(pos) - (int(target_length) - 1))
        for target_length, pos in zip(target_lengths, post_eos_positions, strict=True)
        if int(pos) >= 0
    ]

    def median(values: list[int]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[mid])
        return 0.5 * float(ordered[mid - 1] + ordered[mid])

    pre_eos_count = sum(1 for pos in pre_eos_positions if int(pos) >= 0)
    post_eos_count = sum(1 for pos in post_eos_positions if int(pos) >= 0)
    target_counter = Counter(int(v) for v in target_lengths)
    metrics: dict[str, float | int] = {
        "gen_target_length_conditioning_applied": 1,
        "gen_target_length_sample_count": int(sample_count),
        "gen_target_length_mean": sum(target_lengths) / sample_count,
        "gen_target_length_min": int(min(target_lengths)),
        "gen_target_length_max": int(max(target_lengths)),
        "gen_target_length_hist_nonzero_bins": int(len(target_counter)),
        "gen_target_length_pre_policy_eos_count": int(pre_eos_count),
        "gen_target_length_pre_policy_eos_rate": pre_eos_count / sample_count,
        "gen_target_length_post_policy_eos_count": int(post_eos_count),
        "gen_target_length_post_policy_eos_rate": post_eos_count / sample_count,
        "gen_target_length_forced_eos_count": int(forced_eos_count),
        "gen_target_length_forced_eos_rate": forced_eos_count / sample_count,
        "gen_target_length_no_forced_eos_success_count": int(pre_eos_count),
        "gen_target_length_pre_policy_eos_abs_error_mean": sum(pre_errors) / len(pre_errors) if pre_errors else 0.0,
        "gen_target_length_pre_policy_eos_abs_error_median": median(pre_errors),
        "gen_target_length_post_policy_eos_abs_error_mean": sum(post_errors) / len(post_errors) if post_errors else 0.0,
        "gen_target_length_post_policy_eos_abs_error_median": median(post_errors),
        "gen_target_length_position_max_len": int(gen_len),
    }
    for target_length, count in sorted(target_counter.items()):
        metrics[f"gen_target_length_length_{target_length:04d}_count"] = int(count)
        metrics[f"gen_target_length_length_{target_length:04d}_rate"] = count / sample_count
    return metrics


def _new_target_length_recon_stats(num_deciles: int = 10) -> dict[str, list[float]]:
    return {
        "active_counts": [0.0 for _ in range(int(num_deciles))],
        "active_ce_sums": [0.0 for _ in range(int(num_deciles))],
        "active_correct": [0.0 for _ in range(int(num_deciles))],
        "content_counts": [0.0 for _ in range(int(num_deciles))],
        "content_ce_sums": [0.0 for _ in range(int(num_deciles))],
        "content_correct": [0.0 for _ in range(int(num_deciles))],
    }


def _target_length_decile_ids(target_lengths: torch.Tensor, seq_len: int, num_deciles: int = 10) -> torch.Tensor:
    scaled = (target_lengths.to(torch.long).clamp_min(1) - 1) * int(num_deciles)
    return torch.clamp(torch.div(scaled, max(1, int(seq_len)), rounding_mode="floor"), min=0, max=int(num_deciles) - 1)


def _update_target_length_recon_stats(
    stats: dict[str, list[float]],
    ce: torch.Tensor,
    correct: torch.Tensor,
    active: torch.Tensor,
    content: torch.Tensor,
    target_lengths: torch.Tensor,
) -> None:
    if not active.any():
        return
    num_deciles = len(stats["active_counts"])
    row_deciles = _target_length_decile_ids(target_lengths, ce.shape[1], num_deciles)
    for label, mask, count_key, ce_key, correct_key in (
        ("active", active, "active_counts", "active_ce_sums", "active_correct"),
        ("content", content, "content_counts", "content_ce_sums", "content_correct"),
    ):
        del label
        if not mask.any():
            continue
        rows, _cols = mask.nonzero(as_tuple=True)
        deciles = row_deciles.index_select(0, rows)
        values = ce[mask].detach().float()
        correct_values = correct[mask].detach().float()
        for decile_idx in range(num_deciles):
            decile_mask = deciles == decile_idx
            if not decile_mask.any():
                continue
            count = int(decile_mask.sum().item())
            stats[count_key][decile_idx] += float(count)
            stats[ce_key][decile_idx] += float(values[decile_mask].sum().item())
            stats[correct_key][decile_idx] += float(correct_values[decile_mask].sum().item())


def _target_length_recon_metrics(prefix: str, stats: dict[str, list[float]]) -> dict[str, float | int]:
    num_deciles = len(stats["active_counts"])
    metrics: dict[str, float | int] = {f"{prefix}target_length_decile_count": int(num_deciles)}
    for decile_idx in range(num_deciles):
        label = f"{decile_idx:02d}"
        active_count = float(stats["active_counts"][decile_idx])
        content_count = float(stats["content_counts"][decile_idx])
        metrics[f"{prefix}target_length_decile_{label}_active_tokens"] = int(active_count)
        metrics[f"{prefix}target_length_decile_{label}_ce"] = (
            stats["active_ce_sums"][decile_idx] / active_count if active_count else 0.0
        )
        metrics[f"{prefix}target_length_decile_{label}_token_acc"] = (
            stats["active_correct"][decile_idx] / active_count if active_count else 0.0
        )
        metrics[f"{prefix}target_length_decile_{label}_content_tokens"] = int(content_count)
        metrics[f"{prefix}target_length_decile_{label}_content_ce"] = (
            stats["content_ce_sums"][decile_idx] / content_count if content_count else 0.0
        )
        metrics[f"{prefix}target_length_decile_{label}_content_token_acc"] = (
            stats["content_correct"][decile_idx] / content_count if content_count else 0.0
        )
    return metrics
