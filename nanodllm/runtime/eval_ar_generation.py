"""Autoregressive first-token and validation-prefix diagnostics."""

import os
from collections.abc import Iterator

import torch
import torch.nn.functional as F

from . import data
from .eval_generation_surface import GenerationStats
from .eval_model import _block_generation_special_logits
from .eval_rank_metrics import _single_token_text


def _collect_validation_prefixes(
    val_loader: Iterator[data.DLMDataBatch],
    target_samples: int,
    prefix_len: int,
    total_len: int,
    bos_id: int | None,
    eos_id: int | None,
) -> tuple[list[torch.Tensor], list[list[int]], int]:
    prefixes: list[torch.Tensor] = []
    gold_rows: list[list[int]] = []
    attempts = 0
    max_attempts = max(8, target_samples * 8)
    while len(prefixes) < target_samples and attempts < max_attempts:
        attempts += 1
        batch = next(val_loader)
        for row_idx in range(batch.x_0.shape[0]):
            if len(prefixes) >= target_samples:
                break
            start = _ar_prefix_start(batch.x_0[row_idx], batch.attention_mask[row_idx], bos_id, prefix_len)
            if start < 0:
                continue
            prefix = batch.x_0[row_idx, start : start + prefix_len].detach().clone()
            if eos_id is not None and bool((prefix == int(eos_id)).any().item()):
                continue
            valid_len = int(batch.attention_mask[row_idx].sum().item())
            gold_end = min(valid_len, start + total_len)
            prefixes.append(prefix)
            gold_rows.append([int(i) for i in batch.x_0[row_idx, start:gold_end].detach().cpu().tolist()])
    return prefixes, gold_rows, attempts


@torch.no_grad()
def evaluate_bos_only_first_token(model, tokenizer, device) -> dict[str, float | int]:
    """Evaluate the first-token distribution when the model is fed only BOS."""
    bos_id = tokenizer.get_bos_token_id()
    if bos_id is None:
        return {}
    tokens = torch.full((1, 1), int(bos_id), dtype=torch.long, device=device)
    logits = model(tokens)[:, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = float((-probs * log_probs).sum().item())
    top_prob, top_id = probs.max(dim=-1)
    return {
        "bos_only_first_token_entropy": entropy,
        "bos_only_first_token_top_id": int(top_id.item()),
        "bos_only_first_token_top_prob": float(top_prob.item()),
        "bos_only_first_token_top_text": _single_token_text(tokenizer, int(top_id.item())),
    }


@torch.no_grad()
def _sample_ar_from_prefixes(
    model,
    tokenizer,
    prefixes: torch.Tensor,
    total_len: int,
    temperature: float,
    token_selection: str,
) -> torch.Tensor:
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection!r}")
    if prefixes.dim() != 2:
        raise ValueError(f"prefixes must be 2D, got shape {tuple(prefixes.shape)}")
    if prefixes.shape[1] >= total_len:
        raise ValueError("prefix length must be smaller than total generation length")

    device = next(model.parameters()).device
    pad_id = tokenizer.get_pad_token_id()
    tokens = torch.full(
        (prefixes.shape[0], int(total_len)),
        int(pad_id) if pad_id is not None else 0,
        dtype=torch.long,
        device=device,
    )
    prefixes = prefixes.to(device=device, dtype=torch.long)
    prefix_len = prefixes.shape[1]
    tokens[:, :prefix_len] = prefixes
    for pos in range(prefix_len, int(total_len)):
        logits = model(tokens[:, :pos])[:, -1, :]
        _block_generation_special_logits(logits, tokenizer)
        if token_selection == "argmax":
            next_token = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / float(temperature), dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
        tokens[:, pos] = next_token
    return tokens


def _ar_prefix_start(row: torch.Tensor, attention_mask: torch.Tensor, bos_id: int | None, prefix_len: int) -> int:
    valid_len = int(attention_mask.sum().item())
    if valid_len <= prefix_len:
        return -1
    if bos_id is not None:
        candidates = ((row == int(bos_id)) & attention_mask).nonzero(as_tuple=True)[0]
        for candidate in candidates.detach().cpu().tolist():
            if int(candidate) + prefix_len < valid_len:
                return int(candidate)
    return 0


@torch.no_grad()
def evaluate_ar_validation_prefix_continuations(
    model,
    tokenizer,
    data_config,
    batch_size: int,
    device,
    seq_len: int = data.MAX_SEQ_LEN,
    gen_samples: int = 16,
    gen_len: int = 256,
    prefix_len: int = 16,
    temperature: float = 1.0,
    token_selection: str = "argmax",
    run_dir: str | None = None,
    log_samples: int = 5,
    step: int | None = None,
) -> dict[str, float | int | str]:
    total_len = max(2, int(gen_len))
    prefix_len = max(1, min(int(prefix_len), total_len - 1, int(seq_len) - 1))
    target_samples = max(1, int(gen_samples))
    batch_size = max(1, int(batch_size))
    eos_id = tokenizer.get_eos_token_id()
    bos_id = tokenizer.get_bos_token_id()
    stats = GenerationStats(tokenizer, data.build_token_bytes(tokenizer).cpu())
    val_loader = data.make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)

    prefixes, gold_rows, attempts = _collect_validation_prefixes(
        val_loader, target_samples, prefix_len, total_len, bos_id, eos_id
    )

    if not prefixes:
        return {
            "samples": 0,
            "prefix_len": int(prefix_len),
            "gen_len": int(total_len),
            "token_selection": token_selection,
            "collection_attempts": int(attempts),
        }

    prefix_tensor = torch.stack(prefixes, dim=0).to(device=device)
    tokens = _sample_ar_from_prefixes(
        model,
        tokenizer,
        prefix_tensor,
        total_len,
        temperature=float(temperature),
        token_selection=token_selection,
    )

    for row in tokens:
        stats.add([int(i) for i in row[prefix_len:].detach().cpu().tolist()])

    if run_dir is not None and log_samples > 0:
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "ar_validation_prefix_samples.log")
        with open(log_path, "a") as log_file:
            log_file.write(
                f"[ar-val-prefix] step={step if step is not None else '-'} "
                f"token_selection={token_selection} samples={len(prefixes)} "
                f"prefix_len={prefix_len} gen_len={total_len}\n"
            )
            for row_idx in range(min(int(log_samples), tokens.shape[0])):
                row = [int(i) for i in tokens[row_idx].detach().cpu().tolist()]
                log_file.write(
                    f"[ar-val-prefix] item={row_idx} "
                    f"generated_tokens={len(row) - prefix_len} "
                    f"eos={eos_id in row[prefix_len:] if eos_id is not None else False}\n"
                )
                log_file.write(f"[ar-val-prefix] prefix={tokenizer.decode(row[:prefix_len])!r}\n")
                log_file.write(f"[ar-val-prefix] gold={tokenizer.decode(gold_rows[row_idx])!r}\n")
                log_file.write(f"[ar-val-prefix] generated={tokenizer.decode(row)!r}\n")
                log_file.write(f"[ar-val-prefix] generated_suffix={stats.texts[row_idx]!r}\n")

    return {
        **stats.metrics(int(total_len - prefix_len)),
        "prefix_len": int(prefix_len),
        "gen_len": int(total_len),
        "generated_len": int(total_len - prefix_len),
        "token_selection": token_selection,
        "temperature": float(temperature),
        "collection_attempts": int(attempts),
    }
