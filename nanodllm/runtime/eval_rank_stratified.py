"""Matched prefix-only, partial-mask and leave-one-out rank by frequency stratum."""

import os

import torch
import torch.nn.functional as F

from .eval_generation_artifacts import _decode_attention_row
from .eval_model import _call_model_logits
from .eval_rank_metrics import (
    _fill_conditional_rank_metrics,
    _new_conditional_rank_decile_stats,
    _new_conditional_rank_stats,
    _score_conditional_rank_position,
    _selected_suffix_positions,
    _single_token_text,
    _update_conditional_rank_stats,
)


def _conditional_rank_decile(decile: int) -> int:
    return min(7, max(1, int(decile)))


def _collect_stratified_conditional_rank_positions(
    batch_records: list[dict[str, torch.Tensor]],
    token_to_decile: dict[int, int],
    target_per_decile: int,
) -> tuple[list[dict[str, int]], dict[int, int]]:
    selected_by_decile: dict[int, list[dict[str, int]]] = {decile: [] for decile in range(1, 8)}
    for batch_idx, record in enumerate(batch_records):
        x_0 = record["x_0"]
        suffix_content = record["suffix_content"]
        for row_idx in range(x_0.shape[0]):
            row_positions = torch.nonzero(suffix_content[row_idx], as_tuple=False).flatten()
            for pos_tensor in row_positions:
                pos = int(pos_tensor.item())
                token_id = int(x_0[row_idx, pos].item())
                decile = _conditional_rank_decile(token_to_decile.get(token_id, 1))
                if len(selected_by_decile[decile]) >= target_per_decile:
                    continue
                selected_by_decile[decile].append(
                    {
                        "batch_idx": int(batch_idx),
                        "row_idx": int(row_idx),
                        "pos": int(pos),
                        "token_id": int(token_id),
                        "decile": int(decile),
                    }
                )
    selected: list[dict[str, int]] = []
    selected_counts: dict[int, int] = {}
    for decile in range(1, 8):
        decile_selected = selected_by_decile[decile]
        selected.extend(decile_selected)
        selected_counts[decile] = len(decile_selected)
    return selected, selected_counts


def _group_conditional_rank_records_by_batch(
    selected: list[dict[str, int]],
) -> dict[int, list[dict[str, int]]]:
    grouped: dict[int, list[dict[str, int]]] = {}
    for record in selected:
        grouped.setdefault(int(record["batch_idx"]), []).append(record)
    return grouped


def _record_stratified_conditional_rank_position(
    decile_stats: dict[int, dict],
    suffix_stats: dict,
    record: dict[str, int],
    scored: dict[str, float | int | bool],
) -> None:
    decile = _conditional_rank_decile(record["decile"])
    _update_conditional_rank_stats(decile_stats[decile], scored)
    _update_conditional_rank_stats(suffix_stats, scored)


def _score_stratified_prefix_context(
    model,
    batch_records: list[dict[str, torch.Tensor]],
    selected: list[dict[str, int]],
    model_logits_fn,
    prefix_len: int,
    mask_id: int,
    vocab_size: int,
    topk_k: int,
) -> tuple[dict[int, dict], dict]:
    decile_stats = _new_conditional_rank_decile_stats()
    suffix_stats = _new_conditional_rank_stats()
    grouped = _group_conditional_rank_records_by_batch(selected)
    for batch_idx, records in grouped.items():
        batch_record = batch_records[batch_idx]
        x_0 = batch_record["x_0"]
        attention_mask = batch_record["attention_mask"]
        position_ids = batch_record.get("position_ids")
        z_eval = x_0.clone()
        z_eval[:, prefix_len:] = mask_id
        eval_attention_mask = attention_mask.clone()
        eval_attention_mask[:, prefix_len:] = False
        boundary_mask = batch_record.get("document_boundary_mask")
        if boundary_mask is not None:
            eval_model_attention_mask = boundary_mask & attention_mask[:, :, None] & eval_attention_mask[:, None, :]
        else:
            eval_model_attention_mask = eval_attention_mask
        logits = _call_model_logits(
            model,
            z_eval,
            x_0,
            eval_model_attention_mask,
            model_logits_fn=model_logits_fn,
            position_ids=position_ids,
        )
        probs = F.softmax(logits.float(), dim=-1)
        for record in records:
            row_idx = int(record["row_idx"])
            pos = int(record["pos"])
            token_id = int(record["token_id"])
            scored = _score_conditional_rank_position(probs[row_idx, pos], token_id, vocab_size, topk_k)
            _record_stratified_conditional_rank_position(decile_stats, suffix_stats, record, scored)
    return decile_stats, suffix_stats


def _partial_suffix_mask_positions(
    suffix_content: torch.Tensor,
    row_idx: int,
    selected_pos: int,
    partial_mask_fraction: float,
) -> list[int]:
    if partial_mask_fraction <= 0:
        return []
    row_positions = torch.nonzero(suffix_content[row_idx], as_tuple=False).flatten()
    other_positions = row_positions[row_positions != int(selected_pos)]
    if other_positions.numel() == 0:
        return []
    other_count = int(round(float(partial_mask_fraction) * int(other_positions.numel())))
    other_count = min(int(other_positions.numel()), max(1, other_count))
    return _selected_suffix_positions(other_positions, other_count)


def _score_stratified_masked_context(
    model,
    batch_records: list[dict[str, torch.Tensor]],
    selected: list[dict[str, int]],
    tokenizer,
    model_logits_fn,
    expanded_batch_size: int,
    vocab_size: int,
    topk_k: int,
    partial_mask_fraction: float | None,
) -> tuple[dict[int, dict], dict]:
    decile_stats = _new_conditional_rank_decile_stats()
    suffix_stats = _new_conditional_rank_stats()
    grouped = _group_conditional_rank_records_by_batch(selected)
    mask_id = int(tokenizer.get_mask_token_id())
    for batch_idx, records in grouped.items():
        batch_record = batch_records[batch_idx]
        x_0 = batch_record["x_0"]
        attention_mask = batch_record["attention_mask"]
        model_attention_mask = batch_record.get("model_attention_mask", attention_mask)
        position_source = batch_record.get("position_ids")
        suffix_content = batch_record["suffix_content"]
        for chunk_start in range(0, len(records), expanded_batch_size):
            chunk = records[chunk_start : chunk_start + expanded_batch_size]
            source_rows = torch.tensor(
                [int(record["row_idx"]) for record in chunk],
                device=x_0.device,
                dtype=torch.long,
            )
            chunk_positions = torch.tensor(
                [int(record["pos"]) for record in chunk],
                device=x_0.device,
                dtype=torch.long,
            )
            z_chunk = x_0.index_select(0, source_rows).clone()
            x_chunk = x_0.index_select(0, source_rows)
            attention_chunk = model_attention_mask.index_select(0, source_rows)
            position_chunk = position_source.index_select(0, source_rows) if position_source is not None else None
            for local_idx, record in enumerate(chunk):
                selected_pos = int(record["pos"])
                z_chunk[local_idx, selected_pos] = mask_id
                if partial_mask_fraction is None:
                    continue
                for other_pos in _partial_suffix_mask_positions(
                    suffix_content,
                    int(record["row_idx"]),
                    selected_pos,
                    partial_mask_fraction,
                ):
                    z_chunk[local_idx, int(other_pos)] = mask_id

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
            for local_idx, record in enumerate(chunk):
                token_id = int(record["token_id"])
                scored = _score_conditional_rank_position(selected_probs[local_idx], token_id, vocab_size, topk_k)
                _record_stratified_conditional_rank_position(decile_stats, suffix_stats, record, scored)
    return decile_stats, suffix_stats


def _fill_stratified_conditional_rank_metrics(
    decile_stats: dict[int, dict],
    suffix_stats: dict,
    base_prefix: str,
    metrics_out: dict,
    tokenizer,
    selected_counts: dict[int, int],
    target_per_decile: int,
) -> None:
    total_target = int(target_per_decile) * 7
    total_selected = sum(int(selected_counts.get(decile, 0)) for decile in range(1, 8))
    for decile in range(1, 8):
        prefix = f"{base_prefix}decile_{decile}_"
        selected = int(selected_counts.get(decile, 0))
        _fill_conditional_rank_metrics(decile_stats[decile], prefix, metrics_out, tokenizer)
        metrics_out[f"{prefix}target_count"] = int(target_per_decile)
        metrics_out[f"{prefix}selected_count"] = selected
        metrics_out[f"{prefix}shortfall"] = max(0, int(target_per_decile) - selected)
    suffix_prefix = f"{base_prefix}suffix_"
    _fill_conditional_rank_metrics(suffix_stats, suffix_prefix, metrics_out, tokenizer)
    metrics_out[f"{suffix_prefix}target_count"] = total_target
    metrics_out[f"{suffix_prefix}selected_count"] = total_selected
    metrics_out[f"{suffix_prefix}shortfall"] = max(0, total_target - total_selected)


def _log_stratified_conditional_rank_examples(
    batch_records: list[dict[str, torch.Tensor]],
    selected: list[dict[str, int]],
    tokenizer,
    run_dir: str | None,
    log_samples: int,
    step: int | None,
    target_per_decile: int,
    partial_mask_fraction: float,
    prefix_len: int,
) -> tuple[str, int]:
    if run_dir is None or log_samples <= 0 or not selected:
        return "", 0
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "conditional_rank_stratified_samples.log")
    logged = 0
    with open(path, "a") as log_file:
        log_file.write(
            f"[conditional-rank-stratified] step={step if step is not None else '-'} "
            f"selected_positions={len(selected)} target_per_decile={int(target_per_decile)} "
            f"partial_mask_fraction={float(partial_mask_fraction):.3f}\n"
        )
        for record in selected:
            if logged >= log_samples:
                break
            batch_record = batch_records[int(record["batch_idx"])]
            x_0 = batch_record["x_0"]
            attention_mask = batch_record["attention_mask"]
            row_idx = int(record["row_idx"])
            token_id = int(record["token_id"])
            prefix_attention = attention_mask[row_idx].clone()
            prefix_attention[prefix_len:] = False
            suffix_attention = attention_mask[row_idx].clone()
            suffix_attention[:prefix_len] = False
            log_file.write(
                f"[conditional-rank-stratified] item={logged} decile={int(record['decile'])} "
                f"batch={int(record['batch_idx'])} row={row_idx} pos={int(record['pos'])} "
                f"token_id={token_id} token_text={_single_token_text(tokenizer, token_id)!r}\n"
            )
            log_file.write(
                "[conditional-rank-stratified] prefix="
                f"{_decode_attention_row(tokenizer, x_0[row_idx], prefix_attention)!r}\n"
            )
            log_file.write(
                "[conditional-rank-stratified] suffix="
                f"{_decode_attention_row(tokenizer, x_0[row_idx], suffix_attention)!r}\n"
            )
            logged += 1
    return path, logged


def _evaluate_stratified_conditional_rank(
    model,
    tokenizer,
    batch_records: list[dict[str, torch.Tensor]],
    token_to_decile: dict[int, int],
    model_logits_fn,
    prefix_len: int,
    target_per_decile: int,
    partial_mask_fraction: float,
    expanded_batch_size: int,
    vocab_size: int,
    topk_k: int,
    run_dir: str | None,
    log_samples: int,
    step: int | None,
) -> dict[str, float | int | str]:
    selected, selected_counts = _collect_stratified_conditional_rank_positions(
        batch_records,
        token_to_decile,
        int(target_per_decile),
    )
    prefix_decile_stats, prefix_suffix_stats = _score_stratified_prefix_context(
        model,
        batch_records,
        selected,
        model_logits_fn,
        prefix_len,
        int(tokenizer.get_mask_token_id()),
        vocab_size,
        topk_k,
    )
    partial_decile_stats, partial_suffix_stats = _score_stratified_masked_context(
        model,
        batch_records,
        selected,
        tokenizer,
        model_logits_fn,
        expanded_batch_size,
        vocab_size,
        topk_k,
        partial_mask_fraction=float(partial_mask_fraction),
    )
    loo_decile_stats, loo_suffix_stats = _score_stratified_masked_context(
        model,
        batch_records,
        selected,
        tokenizer,
        model_logits_fn,
        expanded_batch_size,
        vocab_size,
        topk_k,
        partial_mask_fraction=None,
    )
    samples_path, logged_samples = _log_stratified_conditional_rank_examples(
        batch_records,
        selected,
        tokenizer,
        run_dir,
        int(log_samples),
        step,
        int(target_per_decile),
        float(partial_mask_fraction),
        prefix_len,
    )

    selected_total = sum(selected_counts.values())
    target_total = int(target_per_decile) * 7
    metrics: dict[str, float | int | str] = {
        "conditional_rank_stratified_batches": len(batch_records),
        "conditional_rank_stratified_target_per_decile": int(target_per_decile),
        "conditional_rank_stratified_partial_mask_fraction": float(partial_mask_fraction),
        "conditional_rank_stratified_selected_positions": int(selected_total),
        "conditional_rank_stratified_target_positions": int(target_total),
        "conditional_rank_stratified_shortfall": max(0, int(target_total) - int(selected_total)),
        "conditional_rank_stratified_logged_samples": int(logged_samples),
        "conditional_rank_stratified_samples_path": samples_path,
    }
    for decile in range(1, 8):
        selected_count = int(selected_counts.get(decile, 0))
        metrics[f"conditional_rank_stratified_decile_{decile}_selected_count"] = selected_count
        metrics[f"conditional_rank_stratified_decile_{decile}_target_count"] = int(target_per_decile)
        metrics[f"conditional_rank_stratified_decile_{decile}_shortfall"] = max(
            0,
            int(target_per_decile) - selected_count,
        )

    _fill_stratified_conditional_rank_metrics(
        prefix_decile_stats,
        prefix_suffix_stats,
        "conditional_rank_stratified_prefix_",
        metrics,
        tokenizer,
        selected_counts,
        int(target_per_decile),
    )
    _fill_stratified_conditional_rank_metrics(
        partial_decile_stats,
        partial_suffix_stats,
        "conditional_rank_stratified_partial25_",
        metrics,
        tokenizer,
        selected_counts,
        int(target_per_decile),
    )
    _fill_stratified_conditional_rank_metrics(
        loo_decile_stats,
        loo_suffix_stats,
        "conditional_rank_stratified_loo_",
        metrics,
        tokenizer,
        selected_counts,
        int(target_per_decile),
    )
    return metrics
