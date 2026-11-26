"""Conditional rank of autoregressive, masked-suffix and leave-one-out predictions."""

import torch
import torch.nn.functional as F

from . import data
from .eval_generation_metrics import target_eos_lengths
from .eval_model import _batch_model_attention_mask, _call_model_logits, model_uses_target_length_conditioning
from .eval_rank_metrics import (
    _conditional_rank_frequency_deciles,
    _content_position_mask,
    _fill_conditional_rank_metrics,
    _new_conditional_rank_decile_stats,
    _new_conditional_rank_stats,
    _record_conditional_rank_position,
    _score_conditional_rank_position,
    _selected_suffix_positions,
)
from .eval_rank_stratified import _evaluate_stratified_conditional_rank


@torch.no_grad()
def evaluate_ar_next_token_conditional_rank_suffix(
    model,
    tokenizer,
    data_config,
    device,
    batches=5,
    batch_size: int = 8,
    seq_len: int = data.MAX_SEQ_LEN,
) -> dict[str, float | int | str]:
    if batches <= 0:
        return {}

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
    short_document = bool(getattr(data_config, "short_document_curriculum", False))
    prefix_len = seq_len // 2
    if short_document:
        prefix_len = max(1, min(prefix_len, int(getattr(data_config, "short_document_max_tokens", seq_len)) // 2))
    vocab_size = tokenizer.get_vocab_size()
    topk_k = min(20, vocab_size)
    decile_stats = _new_conditional_rank_decile_stats()
    suffix_stats = _new_conditional_rank_stats()
    prefix_len_sum = 0
    prefix_len_min = 0
    prefix_len_max = 0
    prefix_len_count = 0

    for _ in range(int(batches)):
        batch = next(val_loader)
        x_0 = batch.x_0
        content = _content_position_mask(x_0, batch.attention_mask, batch.loss_mask, token_bytes, special_ids)
        if x_0.shape[1] < 2:
            continue
        target_content = content[:, 1:].clone()
        if short_document:
            valid_lens = batch.attention_mask.sum(dim=1)
            row_prefix_lens = torch.clamp(valid_lens // 2, min=1)
            row_prefix_lens = torch.clamp(row_prefix_lens, max=max(1, int(prefix_len)))
            target_positions = torch.arange(1, x_0.shape[1], device=x_0.device).view(1, -1)
            target_content &= target_positions >= row_prefix_lens.view(-1, 1)
            row_prefix_cpu = row_prefix_lens.detach().cpu()
            prefix_len_sum += int(row_prefix_cpu.sum().item())
            prefix_len_count += int(row_prefix_cpu.numel())
            row_min = int(row_prefix_cpu.min().item()) if row_prefix_cpu.numel() else 0
            row_max = int(row_prefix_cpu.max().item()) if row_prefix_cpu.numel() else 0
            prefix_len_min = (
                row_min if prefix_len_count == int(row_prefix_cpu.numel()) else min(prefix_len_min, row_min)
            )
            prefix_len_max = max(prefix_len_max, row_max)
        else:
            target_content[:, : max(0, prefix_len - 1)] = False
        if not target_content.any():
            continue

        model_kwargs = {}
        if getattr(batch, "position_ids", None) is not None:
            model_kwargs["position_ids"] = batch.position_ids
        if getattr(batch, "document_boundary_mask", None) is not None:
            model_kwargs["attention_mask"] = batch.document_boundary_mask & batch.attention_mask[:, None, :]
        else:
            model_kwargs["attention_mask"] = batch.attention_mask
        logits = model(x_0, **model_kwargs)[:, :-1, :].float()
        probs = F.softmax(logits, dim=-1)
        rows, target_offsets = target_content.nonzero(as_tuple=True)
        for row_idx, target_offset in zip(
            rows.detach().cpu().tolist(),
            target_offsets.detach().cpu().tolist(),
            strict=True,
        ):
            target_pos = int(target_offset) + 1
            token_id = int(x_0[int(row_idx), target_pos].item())
            scored = _score_conditional_rank_position(
                probs[int(row_idx), int(target_offset)],
                token_id,
                vocab_size,
                topk_k,
            )
            _record_conditional_rank_position(decile_stats, suffix_stats, token_to_decile, token_id, scored)

    metrics: dict[str, float | int | str] = {
        "ar_next_token_conditional_rank_batches": int(batches),
        "ar_next_token_conditional_rank_freq_batches": int(freq_batches),
        "ar_next_token_conditional_rank_prefix_len": int(prefix_len),
        "ar_next_token_conditional_rank_short_document": int(short_document),
    }
    if short_document:
        metrics["ar_next_token_conditional_rank_prefix_len_mean"] = (
            prefix_len_sum / prefix_len_count if prefix_len_count else 0.0
        )
        metrics["ar_next_token_conditional_rank_prefix_len_min"] = int(prefix_len_min)
        metrics["ar_next_token_conditional_rank_prefix_len_max"] = int(prefix_len_max)
    for decile in range(1, 8):
        _fill_conditional_rank_metrics(
            decile_stats[decile],
            f"ar_next_token_conditional_rank_decile_{decile}_",
            metrics,
            tokenizer,
        )
    _fill_conditional_rank_metrics(
        suffix_stats,
        "ar_next_token_conditional_rank_suffix_",
        metrics,
        tokenizer,
    )
    return metrics


def _evaluate_loo_conditional_rank_batch(
    model,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    model_attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    loss_mask: torch.Tensor,
    tokenizer,
    model_logits_fn,
    token_bytes: torch.Tensor,
    special_ids: set[int],
    token_to_decile: dict[int, int],
    decile_stats: dict[int, dict],
    suffix_stats: dict,
    prefix_len: int,
    positions_per_sequence: int,
    expanded_batch_size: int,
    vocab_size: int,
    topk_k: int,
) -> int:
    content = _content_position_mask(x_0, attention_mask, loss_mask, token_bytes, special_ids)
    content[:, :prefix_len] = False
    selected: list[tuple[int, int]] = []
    for row_idx in range(x_0.shape[0]):
        row_positions = torch.nonzero(content[row_idx], as_tuple=False).flatten()
        for pos in _selected_suffix_positions(row_positions, positions_per_sequence):
            selected.append((row_idx, pos))

    mask_id = int(tokenizer.get_mask_token_id())
    selected_count = len(selected)
    for chunk_start in range(0, selected_count, expanded_batch_size):
        chunk = selected[chunk_start : chunk_start + expanded_batch_size]
        source_rows = torch.tensor([row_idx for row_idx, _pos in chunk], device=x_0.device, dtype=torch.long)
        chunk_positions = torch.tensor([pos for _row_idx, pos in chunk], device=x_0.device, dtype=torch.long)
        z_chunk = x_0.index_select(0, source_rows).clone()
        x_chunk = x_0.index_select(0, source_rows)
        attention_source = attention_mask if model_attention_mask is None else model_attention_mask
        attention_chunk = attention_source.index_select(0, source_rows)
        position_chunk = position_ids.index_select(0, source_rows) if position_ids is not None else None
        z_chunk[torch.arange(len(chunk), device=x_0.device), chunk_positions] = mask_id

        logits = _call_model_logits(
            model,
            z_chunk,
            x_chunk,
            attention_chunk,
            model_logits_fn=model_logits_fn,
            position_ids=position_chunk,
        )
        selected_logits = logits[torch.arange(len(chunk), device=x_0.device), chunk_positions].float()
        selected_probs = F.softmax(selected_logits, dim=-1)
        for local_idx, (source_row, pos) in enumerate(chunk):
            token_id = int(x_0[source_row, pos].item())
            scored = _score_conditional_rank_position(selected_probs[local_idx], token_id, vocab_size, topk_k)
            _record_conditional_rank_position(decile_stats, suffix_stats, token_to_decile, token_id, scored)
    return selected_count


@torch.no_grad()
def evaluate_conditional_rank_suffix(
    model,
    tokenizer,
    data_config,
    device,
    batches=5,
    batch_size: int = 8,
    seq_len: int = data.MAX_SEQ_LEN,
    model_logits_fn=None,
    include_leave_one_out: bool = False,
    loo_positions_per_sequence: int = 8,
    include_stratified_matched: bool = False,
    stratified_target_per_decile: int = 64,
    stratified_partial_mask_fraction: float = 0.25,
    run_dir: str | None = None,
    stratified_log_samples: int = 8,
    step: int | None = None,
) -> dict[str, float | int | str]:
    """Evaluate suffix conditional rank, optionally adding matched suffix contexts.

    The standard metric masks the full suffix and scores every content suffix
    token from prefix-only context. The leave-one-out variant keeps the suffix
    clean, masks only a bounded sampled set of content suffix positions, and
    scores those positions in expanded minibatches. The stratified matched
    variant chooses the same bounded frequency-decile-balanced suffix positions
    and scores them under prefix-only, partial-suffix, and leave-one-out context.
    """
    if batches <= 0:
        return {}

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
    prefix_len = seq_len // 2
    vocab_size = tokenizer.get_vocab_size()
    topk_k = min(20, vocab_size)
    expanded_batch_size = max(1, int(batch_size))

    decile_stats = _new_conditional_rank_decile_stats()
    suffix_stats = _new_conditional_rank_stats()
    loo_decile_stats = _new_conditional_rank_decile_stats()
    loo_suffix_stats = _new_conditional_rank_stats()
    loo_selected_positions = 0
    stratified_batch_records: list[dict[str, torch.Tensor]] = []

    for _ in range(batches):
        batch = next(val_loader)
        x_0 = batch.x_0
        attention_mask = batch.attention_mask
        model_attention_mask = _batch_model_attention_mask(batch)
        loss_mask = batch.loss_mask
        target_lengths = (
            target_eos_lengths(x_0, attention_mask, tokenizer.get_eos_token_id())
            if model_uses_target_length_conditioning(model)
            else None
        )
        content = _content_position_mask(x_0, attention_mask, loss_mask, token_bytes, special_ids)
        if include_stratified_matched:
            suffix_content = content.clone()
            suffix_content[:, :prefix_len] = False
            stratified_batch_records.append(
                {
                    "x_0": x_0,
                    "attention_mask": attention_mask,
                    "model_attention_mask": model_attention_mask,
                    "position_ids": getattr(batch, "position_ids", None),
                    "document_boundary_mask": getattr(batch, "document_boundary_mask", None),
                    "suffix_content": suffix_content,
                }
            )

        z_eval = x_0.clone()
        z_eval[:, prefix_len:] = mask_id
        eval_attention_mask = attention_mask.clone()
        eval_attention_mask[:, prefix_len:] = False
        eval_model_attention_mask = _batch_model_attention_mask(
            batch,
            eval_attention_mask,
            query_attention_mask=attention_mask,
        )

        logits = _call_model_logits(
            model,
            z_eval,
            x_0,
            eval_model_attention_mask,
            model_logits_fn=model_logits_fn,
            position_ids=getattr(batch, "position_ids", None),
            target_lengths=target_lengths,
        )
        probs = F.softmax(logits.float(), dim=-1)

        for row_idx in range(x_0.shape[0]):
            for pos in range(prefix_len, seq_len):
                if not bool(content[row_idx, pos].item()):
                    continue
                token_id = int(x_0[row_idx, pos].item())
                scored = _score_conditional_rank_position(probs[row_idx, pos], token_id, vocab_size, topk_k)
                _record_conditional_rank_position(decile_stats, suffix_stats, token_to_decile, token_id, scored)

        if include_leave_one_out:
            loo_selected_positions += _evaluate_loo_conditional_rank_batch(
                model,
                x_0,
                attention_mask,
                model_attention_mask,
                getattr(batch, "position_ids", None),
                loss_mask,
                tokenizer,
                model_logits_fn,
                token_bytes,
                special_ids,
                token_to_decile,
                loo_decile_stats,
                loo_suffix_stats,
                prefix_len,
                loo_positions_per_sequence,
                expanded_batch_size,
                vocab_size,
                topk_k,
            )

    metrics: dict[str, float | int | str] = {
        "conditional_rank_batches": batches,
        "conditional_rank_freq_batches": freq_batches,
    }
    for decile in range(1, 8):
        _fill_conditional_rank_metrics(decile_stats[decile], f"conditional_rank_decile_{decile}_", metrics, tokenizer)
    _fill_conditional_rank_metrics(suffix_stats, "conditional_rank_suffix_", metrics, tokenizer)

    if include_leave_one_out:
        metrics.update(
            {
                "conditional_rank_loo_batches": batches,
                "conditional_rank_loo_freq_batches": freq_batches,
                "conditional_rank_loo_positions_per_sequence": int(loo_positions_per_sequence),
                "conditional_rank_loo_selected_positions": int(loo_selected_positions),
            }
        )
        for decile in range(1, 8):
            _fill_conditional_rank_metrics(
                loo_decile_stats[decile],
                f"conditional_rank_loo_decile_{decile}_",
                metrics,
                tokenizer,
            )
        _fill_conditional_rank_metrics(loo_suffix_stats, "conditional_rank_loo_suffix_", metrics, tokenizer)
    if include_stratified_matched:
        metrics.update(
            _evaluate_stratified_conditional_rank(
                model,
                tokenizer,
                stratified_batch_records,
                token_to_decile,
                model_logits_fn,
                prefix_len,
                int(stratified_target_per_decile),
                float(stratified_partial_mask_fraction),
                expanded_batch_size,
                vocab_size,
                topk_k,
                run_dir,
                int(stratified_log_samples),
                step,
            )
        )
    return metrics
