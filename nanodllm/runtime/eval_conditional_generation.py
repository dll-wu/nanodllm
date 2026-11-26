"""Generation from validation prefixes and continuation scoring."""

import json
import os

import torch

from . import data
from .config import DataConfig
from .eval_generation_metrics import (
    _content_ids,
)
from .eval_generation_surface import GenerationStats
from .eval_reconstruction import _self_reconstruction_ce_with_score_mask
from .eval_sample_trace import merge_sample_trace as _merge_sample_trace
from .eval_sample_trace import new_sample_trace_totals as _new_sample_trace_totals
from .eval_sample_trace import sample_trace_metrics as _sample_trace_metrics


@torch.no_grad()
def evaluate_conditional_generation_metrics(
    model,
    tokenizer,
    data_config: DataConfig,
    noise,
    sample_fn,
    batch_size: int,
    run_dir: str | None = None,
    device=None,
    seq_len: int = data.MAX_SEQ_LEN,
    prefix_fracs: list[float] | tuple[float, ...] = (0.25, 0.5, 0.75),
    gen_samples: int = 64,
    continuation_len: int = 256,
    log_samples: int = 8,
    score_mode: str = "self",
    self_score_t: float = 0.5,
    step: int | None = None,
    sample_label: str | None = None,
    model_logits_fn=None,
) -> dict[str, float | int | str]:
    if score_mode not in ("none", "self"):
        raise ValueError(f"conditional generation score mode must be 'none' or 'self', got {score_mode!r}")
    device = next(model.parameters()).device if device is None else device
    batch_size = max(1, int(batch_size))
    target_samples = max(1, int(gen_samples))
    continuation_len = int(continuation_len)
    seq_len = int(seq_len)
    if continuation_len <= 0:
        raise ValueError("continuation_len must be positive")
    if continuation_len >= seq_len:
        raise ValueError("continuation_len must be smaller than seq_len")

    eos_id = tokenizer.get_eos_token_id()
    special_ids = tokenizer.get_special_token_ids()
    token_bytes_cpu = data.build_token_bytes(tokenizer).cpu()
    metrics: dict[str, float | int | str] = {
        "conditional_gen_samples_requested": int(target_samples),
        "conditional_gen_continuation_len": int(continuation_len),
        "conditional_gen_score_mode": score_mode,
    }

    prefix_records_path = None
    samples_path = None
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        prefix_records_path = os.path.join(run_dir, "prefix_records.jsonl")
        samples_path = os.path.join(run_dir, "conditional_generation_samples.jsonl")
        metrics["conditional_gen_prefix_records_path"] = prefix_records_path
        metrics["conditional_gen_samples_path"] = samples_path

    for raw_frac in prefix_fracs:
        prefix_frac = float(raw_frac)
        label = f"p{int(round(prefix_frac * 100)):02d}"
        prefix = f"conditional_gen_{label}_"
        prefix_len = max(1, min(int(round(continuation_len * prefix_frac)), seq_len - continuation_len))
        total_len = prefix_len + continuation_len
        val_loader = data.make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)

        prefixes: list[torch.Tensor] = []
        prefix_records: list[dict[str, object]] = []
        attempts = 0
        max_attempts = max(8, target_samples * 8)
        while len(prefixes) < target_samples and attempts < max_attempts:
            batch = next(val_loader)
            batch_idx = attempts
            attempts += 1
            for row_idx in range(batch.x_0.shape[0]):
                if len(prefixes) >= target_samples:
                    break
                valid_len = int(batch.attention_mask[row_idx].sum().item())
                if valid_len < total_len:
                    continue
                prefix_tokens = batch.x_0[row_idx, :prefix_len].detach().clone()
                if eos_id is not None and bool((prefix_tokens == int(eos_id)).any().item()):
                    continue
                prefixes.append(prefix_tokens)
                prefix_ids = [int(i) for i in prefix_tokens.detach().cpu().tolist()]
                gold_ids = [int(i) for i in batch.x_0[row_idx, prefix_len:total_len].detach().cpu().tolist()]
                prefix_records.append(
                    {
                        "step": step,
                        "sample_label": sample_label or "",
                        "source_split": "val",
                        "source_batch_index": int(batch_idx),
                        "source_row_index": int(row_idx),
                        "source_valid_len": int(valid_len),
                        "prefix_fraction": float(prefix_frac),
                        "prefix_len": int(prefix_len),
                        "continuation_len": int(continuation_len),
                        "prefix_token_ids": prefix_ids,
                        "prefix_text": tokenizer.decode(prefix_ids),
                        "gold_continuation_token_ids": gold_ids,
                        "gold_continuation_text": tokenizer.decode(gold_ids),
                    }
                )

        if prefix_records_path is not None:
            with open(prefix_records_path, "a", encoding="utf-8") as f:
                for record in prefix_records:
                    f.write(json.dumps(record, sort_keys=True) + "\n")

        if not prefixes:
            metrics.update(
                {
                    f"{prefix}samples": 0,
                    f"{prefix}prefix_fraction": float(prefix_frac),
                    f"{prefix}prefix_len": int(prefix_len),
                    f"{prefix}continuation_len": int(continuation_len),
                    f"{prefix}collection_attempts": int(attempts),
                }
            )
            continue

        stats = GenerationStats(tokenizer, token_bytes_cpu)
        prefix_copy_hits = 0
        prefix_copy_total = 0
        prefix_preserved = 0
        self_ce_total = 0.0
        self_ce_count = 0
        trace_totals = _new_sample_trace_totals()
        logged = 0

        for start in range(0, len(prefixes), batch_size):
            end = min(len(prefixes), start + batch_size)
            prefix_tensor = torch.stack(prefixes[start:end], dim=0).to(device=device)
            sample_out = sample_fn(model, prefix_tensor, continuation_len)
            if isinstance(sample_out, tuple):
                tokens, sample_trace = sample_out
                _merge_sample_trace(trace_totals, sample_trace)
            else:
                tokens = sample_out
            if tokens.shape[1] != total_len:
                raise RuntimeError(f"conditional sampler returned length {tokens.shape[1]}, expected {total_len}")
            prefix_preserved += int((tokens[:, :prefix_len] == prefix_tensor).all(dim=1).sum().item())

            score_mask = torch.zeros_like(tokens, dtype=torch.bool, device=device)
            score_mask[:, prefix_len:] = True
            if score_mode == "self":
                ce_sum, ce_count = _self_reconstruction_ce_with_score_mask(
                    model,
                    tokenizer,
                    noise,
                    tokens,
                    score_mask,
                    device,
                    float(self_score_t),
                    model_logits_fn=model_logits_fn,
                )
                self_ce_total += ce_sum
                self_ce_count += ce_count

            for local_idx in range(tokens.shape[0]):
                global_idx = start + local_idx
                row = [int(i) for i in tokens[local_idx].detach().cpu().tolist()]
                prefix_ids = row[:prefix_len]
                continuation = row[prefix_len:]
                scored = stats.add(continuation)
                prefix_content = set(_content_ids(prefix_ids, special_ids, eos_id))
                prefix_copy_hits += sum(1 for token_id in scored.content if int(token_id) in prefix_content)
                prefix_copy_total += len(scored.content)

                if samples_path is not None and logged < int(log_samples):
                    record = dict(prefix_records[global_idx])
                    record.update(
                        {
                            "sample_index": int(global_idx),
                            "prefix_preserved": bool(prefix_ids == record["prefix_token_ids"]),
                            "continuation_token_ids": continuation,
                            "continuation_text": scored.text,
                            "full_text": tokenizer.decode(row),
                            "has_eos": scored.has_eos,
                        }
                    )
                    with open(samples_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, sort_keys=True) + "\n")
                    logged += 1

        sample_count = len(stats.texts)
        frac_metrics = {f"{prefix}{key}": value for key, value in stats.metrics(continuation_len).items()}
        frac_metrics[f"{prefix}mean_continuation_len"] = frac_metrics.pop(f"{prefix}avg_decoded_tokens")
        frac_metrics.update(
            {
                f"{prefix}prefix_fraction": float(prefix_frac),
                f"{prefix}prefix_len": int(prefix_len),
                f"{prefix}continuation_len": int(continuation_len),
                f"{prefix}total_len": int(total_len),
                f"{prefix}collection_attempts": int(attempts),
                f"{prefix}logged_samples": int(logged),
                f"{prefix}prefix_preserved_rate": prefix_preserved / sample_count if sample_count else 0.0,
                f"{prefix}prefix_copy_rate": prefix_copy_hits / prefix_copy_total if prefix_copy_total else 0.0,
            }
        )
        if score_mode == "self":
            frac_metrics[f"{prefix}self_score_t"] = float(self_score_t)
            frac_metrics[f"{prefix}self_ce"] = self_ce_total / self_ce_count if self_ce_count else 0.0
            frac_metrics[f"{prefix}self_ce_tokens"] = int(self_ce_count)
        if int(trace_totals["batches"] or 0) > 0:
            for key, value in _sample_trace_metrics(trace_totals).items():
                suffix = key[len("gen_trace_") :] if key.startswith("gen_trace_") else key
                frac_metrics[f"{prefix}trace_{suffix}"] = value
            frac_metrics[f"{prefix}trace_pred_max_token_frac_mean"] = frac_metrics.get(
                f"{prefix}trace_pred_max_token_frac_mean",
                0.0,
            )
        metrics.update(frac_metrics)

    return metrics
