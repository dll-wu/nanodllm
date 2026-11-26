"""Training-corpus unigram prior correction for generation."""

import json
import os

import torch

from . import data
from .config import DataConfig
from .eval_rank_metrics import _single_token_text

_PRIOR_CORRECTION_CACHE: dict[tuple, tuple[torch.Tensor, dict[str, float | int | str]]] = {}


def _load_generation_prior_settings(run_dir: str | None) -> dict | None:
    if run_dir is None:
        return None
    config_path = os.path.join(run_dir, "resolved_config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    eval_cfg = payload.get("eval", {})
    train_cfg = payload.get("train", {})
    data_cfg = payload.get("data", {})
    if not isinstance(eval_cfg, dict) or not isinstance(train_cfg, dict) or not isinstance(data_cfg, dict):
        return None
    strength = float(eval_cfg.get("gen_prior_correction_strength", 0.0))
    batches = int(eval_cfg.get("gen_prior_correction_batches", 0))
    smoothing = float(eval_cfg.get("gen_prior_correction_smoothing", 1.0))
    if strength <= 0 or not bool(train_cfg.get("eval_only", False)):
        return None
    if batches <= 0:
        raise ValueError("eval.gen_prior_correction_batches must be positive when prior correction is enabled")
    return {
        "config_path": config_path,
        "data_config": DataConfig(**data_cfg),
        "strength": strength,
        "batches": batches,
        "smoothing": smoothing,
    }


@torch.no_grad()
def _estimate_unigram_prior_logit_bias(
    data_config,
    tokenizer,
    batch_size: int,
    device,
    seq_len: int,
    batches: int,
    strength: float,
    smoothing: float,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    vocab_size = tokenizer.get_vocab_size()
    counts = torch.full((vocab_size,), float(smoothing), dtype=torch.float64)
    token_bytes = data.build_token_bytes(tokenizer).to(device)
    loader = data.make_dataloader(data_config, tokenizer, batch_size, seq_len, "train", device=device)
    special_ids = tokenizer.get_special_token_ids()
    observed_tokens = 0

    for _ in range(int(batches)):
        batch = next(loader)
        content = batch.attention_mask & batch.loss_mask & (token_bytes[batch.x_0] > 0)
        for token_id in special_ids:
            if 0 <= int(token_id) < vocab_size:
                content &= batch.x_0 != int(token_id)
        if not content.any():
            continue
        ids = batch.x_0[content].detach().cpu()
        counts += torch.bincount(ids, minlength=vocab_size).to(dtype=torch.float64)
        observed_tokens += int(ids.numel())

    log_prior = (counts / counts.sum()).log().to(device=device, dtype=torch.float32)
    logit_bias = -float(strength) * log_prior
    zero_bias_count = 0
    zero_byte_ids = (token_bytes.detach().cpu() <= 0).nonzero(as_tuple=True)[0].tolist()
    for token_id in zero_byte_ids:
        if 0 <= int(token_id) < vocab_size:
            logit_bias[int(token_id)] = 0.0
            zero_bias_count += 1

    observed_counts = counts - float(smoothing)
    top_count, top_id = torch.max(observed_counts, dim=0)
    top_token_id = int(top_id.item()) if observed_tokens else -1
    top_token_count = int(top_count.item()) if observed_tokens else 0
    metrics: dict[str, float | int | str] = {
        "gen_prior_correction_strength": float(strength),
        "gen_prior_correction_batches": int(batches),
        "gen_prior_correction_smoothing": float(smoothing),
        "gen_prior_correction_token_count": int(observed_tokens),
        "gen_prior_correction_nonzero_token_types": int((observed_counts > 0).sum().item()),
        "gen_prior_correction_zero_bias_token_types": int(zero_bias_count),
        "gen_prior_correction_top_token_id": top_token_id,
        "gen_prior_correction_top_token_text": _single_token_text(tokenizer, top_token_id),
        "gen_prior_correction_top_token_count": top_token_count,
        "gen_prior_correction_top_token_fraction": top_token_count / observed_tokens if observed_tokens else 0.0,
    }
    return logit_bias, metrics


def _generation_prior_correction(
    run_dir: str | None,
    tokenizer,
    batch_size: int,
    device,
    seq_len: int,
) -> tuple[torch.Tensor | None, dict[str, float | int | str]]:
    settings = _load_generation_prior_settings(run_dir)
    if settings is None:
        return None, {}

    config_path = str(settings["config_path"])
    strength = float(settings["strength"])
    batches = int(settings["batches"])
    smoothing = float(settings["smoothing"])
    key = (
        config_path,
        os.path.getmtime(config_path),
        str(torch.device(device)),
        tokenizer.get_vocab_size(),
        int(batch_size),
        int(seq_len),
        strength,
        batches,
        smoothing,
    )
    cached = _PRIOR_CORRECTION_CACHE.get(key)
    if cached is None:
        cached = _estimate_unigram_prior_logit_bias(
            settings["data_config"],
            tokenizer,
            batch_size,
            device,
            seq_len,
            batches,
            strength,
            smoothing,
        )
        _PRIOR_CORRECTION_CACHE[key] = cached
    logit_bias, metrics = cached
    return logit_bias, dict(metrics)
