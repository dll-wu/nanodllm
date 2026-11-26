"""Continuation diagnostics after a forced argmax prefix."""

import json
import os

import torch
import torch.nn.functional as F

from . import data
from .eval_ar_generation import _collect_validation_prefixes
from .eval_generation_surface import GenerationStats
from .eval_model import _block_generation_special_logits, _call_model_logits, _model_uses_noise_level
from .eval_rank_metrics import _single_token_text


def _nucleus_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    probs = F.softmax(logits / float(temperature), dim=-1)
    if float(top_p) >= 1.0:
        return torch.multinomial(probs, 1).squeeze(-1)

    sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = sorted_probs.masked_fill(remove, 0.0)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sampled = torch.multinomial(filtered, 1)
    return sorted_ids.gather(-1, sampled).squeeze(-1)


def _basin_next_logits(
    model,
    tokenizer,
    tokens: torch.Tensor,
    pos: int,
    is_causal: bool,
    noise_level=None,
    model_logits_fn=None,
) -> torch.Tensor:
    if is_causal:
        return model(tokens[:, : int(pos)])[:, -1, :]

    attention_mask = torch.ones_like(tokens, dtype=torch.bool, device=tokens.device)
    if noise_level is None and _model_uses_noise_level(model):
        mask_id = int(tokenizer.get_mask_token_id())
        noise_level = (tokens == mask_id).float().mean(dim=1)
    logits = _call_model_logits(
        model,
        tokens,
        tokens,
        attention_mask,
        model_logits_fn=model_logits_fn,
        noise_level=noise_level,
    )
    return logits[:, int(pos), :]


def _basin_initial_tokens(
    tokenizer,
    prefixes: torch.Tensor,
    total_len: int,
    is_causal: bool,
    noise=None,
) -> torch.Tensor:
    device = prefixes.device
    pad_id = tokenizer.get_pad_token_id()
    fill_id = pad_id if pad_id is not None else 0
    if is_causal:
        tokens = torch.full(
            (prefixes.shape[0], int(total_len)),
            int(fill_id),
            dtype=torch.long,
            device=device,
        )
    elif noise is not None and hasattr(noise, "sample_terminal_tokens"):
        tokens = noise.sample_terminal_tokens((prefixes.shape[0], int(total_len)), device=device)
    elif noise is not None and hasattr(noise, "sample_uniform_tokens"):
        tokens = noise.sample_uniform_tokens((prefixes.shape[0], int(total_len)), device=device)
    else:
        tokens = torch.full(
            (prefixes.shape[0], int(total_len)),
            int(tokenizer.get_mask_token_id()),
            dtype=torch.long,
            device=device,
        )
    tokens[:, : prefixes.shape[1]] = prefixes
    return tokens


@torch.no_grad()
def _sample_basin_from_prefixes(
    model,
    tokenizer,
    prefixes: torch.Tensor,
    total_len: int,
    forced_argmax_depth: int,
    continuation_policy: str,
    temperature: float,
    nucleus_top_p: float,
    is_causal: bool,
    noise=None,
    model_logits_fn=None,
) -> tuple[torch.Tensor, list[dict[str, float | int | str]]]:
    if continuation_policy not in ("argmax", "nucleus"):
        raise ValueError(f"unknown continuation_policy: {continuation_policy!r}")
    if prefixes.dim() != 2:
        raise ValueError(f"prefixes must be 2D, got shape {tuple(prefixes.shape)}")
    if prefixes.shape[1] >= total_len:
        raise ValueError("prefix length must be smaller than total generation length")

    device = next(model.parameters()).device
    prefix_len = int(prefixes.shape[1])
    prefixes = prefixes.to(device=device, dtype=torch.long)
    tokens = _basin_initial_tokens(
        tokenizer,
        prefixes,
        int(total_len),
        is_causal=bool(is_causal),
        noise=noise,
    )

    traces: list[dict[str, float | int | str]] = []
    max_forced = min(max(0, int(forced_argmax_depth)), int(total_len) - prefix_len)
    for step_idx in range(max_forced):
        pos = prefix_len + step_idx
        step_noise_level = None if is_causal else (int(total_len) - int(pos)) / max(1, int(total_len))
        logits = _basin_next_logits(
            model,
            tokenizer,
            tokens,
            pos,
            is_causal=is_causal,
            noise_level=step_noise_level,
            model_logits_fn=model_logits_fn,
        ).float()
        _block_generation_special_logits(logits, tokenizer)
        probs = F.softmax(logits / float(temperature), dim=-1)
        safe_probs = probs.clamp_min(1e-10)
        entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)
        top1_prob, top1_id = probs.max(dim=-1)
        next_token = logits.argmax(dim=-1)
        tokens[:, pos] = next_token
        counts = torch.bincount(next_token.detach().cpu(), minlength=tokenizer.get_vocab_size())
        max_count = int(counts.max().item()) if counts.numel() else 0
        most_common_id = int(counts.argmax().item()) if counts.numel() else -1
        traces.append(
            {
                "forced_step": int(step_idx + 1),
                "position": int(pos),
                "entropy_mean": float(entropy.mean().item()),
                "top1_prob_mean": float(top1_prob.mean().item()),
                "top1_prob_min": float(top1_prob.min().item()),
                "top1_prob_max": float(top1_prob.max().item()),
                "top1_token_id": most_common_id,
                "top1_token_text": _single_token_text(tokenizer, most_common_id),
                "top1_token_count": max_count,
                "top1_token_fraction": max_count / max(1, int(next_token.numel())),
            }
        )

    for pos in range(prefix_len + max_forced, int(total_len)):
        step_noise_level = None if is_causal else (int(total_len) - int(pos)) / max(1, int(total_len))
        logits = _basin_next_logits(
            model,
            tokenizer,
            tokens,
            pos,
            is_causal=is_causal,
            noise_level=step_noise_level,
            model_logits_fn=model_logits_fn,
        ).float()
        _block_generation_special_logits(logits, tokenizer)
        if continuation_policy == "argmax":
            next_token = logits.argmax(dim=-1)
        else:
            next_token = _nucleus_sample(logits, temperature=float(temperature), top_p=float(nucleus_top_p))
        tokens[:, pos] = next_token
    return tokens, traces


def _basin_continuation_metrics(
    tokenizer,
    tokens: torch.Tensor,
    prefix_len: int,
    total_len: int,
) -> dict[str, float | int | str]:
    stats = GenerationStats(tokenizer, data.build_token_bytes(tokenizer).cpu())
    for row in tokens:
        stats.add([int(i) for i in row[int(prefix_len) :].detach().cpu().tolist()])
    generated_len = int(total_len) - int(prefix_len)
    return {"generated_len": generated_len, **stats.metrics(generated_len)}


@torch.no_grad()
def evaluate_argmax_basin_diagnostic(
    model,
    tokenizer,
    data_config,
    batch_size: int,
    device,
    seq_len: int = data.MAX_SEQ_LEN,
    gen_samples: int = 16,
    gen_len: int = 256,
    prefix_len: int = 16,
    forced_argmax_depths: list[int] | tuple[int, ...] = (0, 1, 2, 4),
    temperature: float = 1.0,
    nucleus_top_p: float = 0.95,
    is_causal: bool = False,
    run_dir: str | None = None,
    log_samples: int = 5,
    source_run_id: str = "",
    noise=None,
    model_logits_fn=None,
) -> dict[str, float | int | str]:
    total_len = max(2, int(gen_len))
    prefix_len = max(1, min(int(prefix_len), total_len - 1, int(seq_len) - 1))
    target_samples = max(1, int(gen_samples))
    batch_size = max(1, int(batch_size))
    eos_id = tokenizer.get_eos_token_id()
    bos_id = tokenizer.get_bos_token_id()
    val_loader = data.make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)

    prefixes, gold_rows, attempts = _collect_validation_prefixes(
        val_loader, target_samples, prefix_len, total_len, bos_id, eos_id
    )

    metrics: dict[str, float | int | str] = {
        "basin_samples": int(len(prefixes)),
        "basin_collection_attempts": int(attempts),
        "basin_prefix_len": int(prefix_len),
        "basin_gen_len": int(total_len),
        "basin_generated_len": int(total_len - prefix_len),
        "basin_nucleus_top_p": float(nucleus_top_p),
        "basin_temperature": float(temperature),
        "basin_model_kind": "causal_ar" if is_causal else "masked_prefix",
        "basin_stochastic_policy": "nucleus_multinomial",
    }
    if not prefixes:
        return metrics

    prefix_tensor = torch.stack(prefixes, dim=0).to(device=device)
    examples_path = os.path.join(run_dir, "basin_diagnostic_examples.jsonl") if run_dir is not None else None
    traces_path = os.path.join(run_dir, "basin_diagnostic_traces.jsonl") if run_dir is not None else None
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)

    per_policy: dict[tuple[int, str], dict[str, float | int | str]] = {}
    for raw_depth in forced_argmax_depths:
        depth = max(0, int(raw_depth))
        trace_written = False
        for policy in ("argmax", "nucleus"):
            tokens, traces = _sample_basin_from_prefixes(
                model,
                tokenizer,
                prefix_tensor,
                total_len,
                forced_argmax_depth=depth,
                continuation_policy=policy,
                temperature=float(temperature),
                nucleus_top_p=float(nucleus_top_p),
                is_causal=bool(is_causal),
                noise=noise,
                model_logits_fn=model_logits_fn,
            )
            depth_key = f"basin_depth_{depth:02d}_{policy}_"
            policy_metrics = _basin_continuation_metrics(tokenizer, tokens, prefix_len, total_len)
            per_policy[(depth, policy)] = policy_metrics
            for key, value in policy_metrics.items():
                metrics[f"{depth_key}{key}"] = value
            metrics[f"{depth_key}forced_argmax_depth"] = int(depth)
            metrics[f"{depth_key}continuation_policy"] = "nucleus_multinomial" if policy == "nucleus" else policy

            if not trace_written:
                for trace in traces:
                    trace_key = f"basin_depth_{depth:02d}_forced_step_{int(trace['forced_step']):02d}_"
                    for key, value in trace.items():
                        if key == "forced_step":
                            continue
                        metrics[f"{trace_key}{key}"] = value
                    if traces_path is not None:
                        with open(traces_path, "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "source_run_id": source_run_id,
                                        "forced_argmax_depth": int(depth),
                                        **trace,
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                trace_written = True

            if examples_path is not None and log_samples > 0:
                with open(examples_path, "a", encoding="utf-8") as f:
                    for row_idx in range(min(int(log_samples), tokens.shape[0])):
                        row = [int(i) for i in tokens[row_idx].detach().cpu().tolist()]
                        forced_end = min(prefix_len + depth, total_len)
                        f.write(
                            json.dumps(
                                {
                                    "source_run_id": source_run_id,
                                    "forced_argmax_depth": int(depth),
                                    "continuation_policy": policy,
                                    "sample_index": int(row_idx),
                                    "prefix_text": tokenizer.decode(row[:prefix_len]),
                                    "forced_text": tokenizer.decode(row[prefix_len:forced_end]),
                                    "continuation_text": tokenizer.decode(row[forced_end:]),
                                    "generated_text": tokenizer.decode(row[prefix_len:]),
                                    "gold_text": tokenizer.decode(gold_rows[row_idx]),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )

        pair_keys = (
            "eos_rate",
            "exact_duplicate_rate",
            "repeat_4gram_rate",
            "distinct_1",
            "distinct_2",
            "distinct_3",
            "max_token_frac_mean",
            "max_token_fraction",
            "token_entropy_mean",
        )
        argmax_metrics = per_policy.get((depth, "argmax"), {})
        nucleus_metrics = per_policy.get((depth, "nucleus"), {})
        for key in pair_keys:
            argmax_value = argmax_metrics.get(key)
            nucleus_value = nucleus_metrics.get(key)
            if isinstance(argmax_value, (int, float)) and isinstance(nucleus_value, (int, float)):
                metrics[f"basin_depth_{depth:02d}_argmax_minus_nucleus_{key}"] = float(argmax_value) - float(
                    nucleus_value
                )

    return metrics
