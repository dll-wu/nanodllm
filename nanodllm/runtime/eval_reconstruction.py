"""Fixed-noise reconstruction and generated-token self-reconstruction scores."""

import os
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from . import data
from .eval_generation_artifacts import _decode_attention_row, _decode_masked_attention_row
from .eval_generation_metrics import (
    _eos_suffix_decile_metrics,
    _new_eos_suffix_decile_stats,
    _new_target_length_recon_stats,
    _target_length_recon_metrics,
    _update_eos_suffix_decile_stats,
    _update_target_length_recon_stats,
    target_eos_lengths,
)
from .eval_model import _batch_model_attention_mask, _call_model_logits, model_uses_target_length_conditioning


def _t_key(t_val: float) -> str:
    return f"t{int(round(float(t_val) * 10)):02d}"


@torch.no_grad()
def _self_reconstruction_ce(
    model,
    tokenizer,
    noise,
    tokens: torch.Tensor,
    device,
    t_val: float,
    model_logits_fn=None,
) -> tuple[float, int]:
    score_mask = torch.ones_like(tokens, dtype=torch.bool, device=device)
    return _self_reconstruction_ce_with_score_mask(
        model, tokenizer, noise, tokens, score_mask, device, t_val, model_logits_fn=model_logits_fn
    )


@torch.no_grad()
def _self_reconstruction_ce_with_score_mask(
    model,
    tokenizer,
    noise,
    tokens: torch.Tensor,
    score_mask: torch.Tensor,
    device,
    t_val: float,
    model_logits_fn=None,
) -> tuple[float, int]:
    eos_id = tokenizer.get_eos_token_id()
    special_ids = tokenizer.get_special_token_ids()
    token_bytes = data.build_token_bytes(tokenizer).to(device)
    tokens = tokens.to(device=device)
    score_mask = score_mask.to(device=device, dtype=torch.bool)
    attention_mask = torch.ones_like(tokens, dtype=torch.bool, device=device)
    if eos_id is not None:
        eos_hits = tokens == int(eos_id)
        for row_idx in range(tokens.shape[0]):
            positions = eos_hits[row_idx].nonzero(as_tuple=True)[0]
            if positions.numel() > 0:
                attention_mask[row_idx, int(positions[0].item()) + 1 :] = False

    score_mask &= attention_mask & (token_bytes[tokens] > 0)
    for token_id in special_ids:
        if 0 <= int(token_id) < token_bytes.numel():
            score_mask &= tokens != int(token_id)
    if not score_mask.any() or not hasattr(noise, "corrupt"):
        return 0.0, 0

    z_t, loss_weight = noise.corrupt(tokens, float(t_val), maskable=score_mask)
    active = (loss_weight > 0) & score_mask
    if not active.any():
        return 0.0, 0

    logits = _call_model_logits(
        model,
        z_t,
        tokens,
        attention_mask,
        model_logits_fn=model_logits_fn,
        noise_level=float(t_val),
    )
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), tokens.view(-1), reduction="none").view_as(tokens)
    total = float(ce[active].sum().item())
    count = int(active.sum().item())
    return total, count


@torch.no_grad()
def evaluate_reconstruction_metrics(
    model,
    data_config,
    tokenizer,
    noise,
    batch_size: int,
    device,
    seq_len: int = data.MAX_SEQ_LEN,
    t_values: list[float] | tuple[float, ...] = (0.7, 0.9),
    batches: int = 2,
    run_dir: str | None = None,
    log_samples: int = 3,
    step: int | None = None,
    model_logits_fn=None,
    corrupt_fn=None,
    corruption_label: str = "independent-mdlm",
) -> dict[str, float | int | str]:
    if batches <= 0 or not t_values:
        return {}

    val_loader = data.make_dataloader(data_config, tokenizer, batch_size, seq_len, "val", device=device)
    token_bytes = data.build_token_bytes(tokenizer).to(device)
    mask_id = int(tokenizer.get_mask_token_id())
    eos_id = tokenizer.get_eos_token_id()
    metrics: dict[str, float | int | str] = {
        "recon_batches": int(batches),
        "recon_log_samples": int(log_samples),
        "recon_corruption": corruption_label,
    }

    with ExitStack() as stack:
        log_file = None
        if run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            log_file = stack.enter_context(open(os.path.join(run_dir, "reconstruction_samples.log"), "a"))

        def log(msg: str) -> None:
            if log_file is not None:
                log_file.write(msg + "\n")
                log_file.flush()
            else:
                print(msg)

        log(
            f"[recon] step={step if step is not None else '-'} "
            f"batches={int(batches)} t_values={','.join(f'{float(t):.2f}' for t in t_values)} "
            f"corruption={corruption_label}"
        )

        for t_val in t_values:
            ce_sum = 0.0
            content_ce_sum = 0.0
            active_count = 0
            content_count = 0
            correct_count = 0
            content_correct_count = 0
            lossable_count = 0
            rows_with_content = 0
            exact_content_rows = 0
            logged = 0
            eos_suffix_stats = _new_eos_suffix_decile_stats()
            target_length_stats = _new_target_length_recon_stats()
            use_target_length_conditioning = model_uses_target_length_conditioning(model)

            for batch_idx in range(int(batches)):
                batch = next(val_loader)
                x_eval = batch.x_0
                target_lengths = (
                    target_eos_lengths(x_eval, batch.attention_mask, eos_id) if use_target_length_conditioning else None
                )
                noise_level = float(t_val)
                if corrupt_fn is None:
                    z_eval, w_eval = noise.corrupt(x_eval, float(t_val), maskable=batch.loss_mask)
                else:
                    corrupt_out = corrupt_fn(x_eval, batch.loss_mask, float(t_val))
                    if len(corrupt_out) == 2:
                        z_eval, w_eval = corrupt_out
                        noise_level = None
                    elif len(corrupt_out) == 3:
                        z_eval, w_eval, noise_level = corrupt_out
                    else:
                        raise ValueError(f"corrupt_fn must return 2 or 3 values, got {len(corrupt_out)}")
                logits = _call_model_logits(
                    model,
                    z_eval,
                    x_eval,
                    _batch_model_attention_mask(batch),
                    model_logits_fn=model_logits_fn,
                    noise_level=noise_level,
                    position_ids=getattr(batch, "position_ids", None),
                    target_lengths=target_lengths,
                )
                pred = logits.argmax(dim=-1)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    x_eval.reshape(-1),
                    reduction="none",
                ).view_as(x_eval)

                active = (w_eval > 0) & batch.loss_mask & batch.attention_mask
                content = active & (token_bytes[x_eval] > 0)
                correct = pred == x_eval
                if target_lengths is not None:
                    _update_target_length_recon_stats(
                        target_length_stats,
                        ce,
                        correct,
                        active,
                        content,
                        target_lengths,
                    )
                _update_eos_suffix_decile_stats(
                    eos_suffix_stats,
                    logits,
                    pred,
                    x_eval,
                    active,
                    batch.attention_mask,
                    eos_id,
                )
                lossable_count += int((batch.loss_mask & batch.attention_mask).sum().item())
                if active.any():
                    ce_sum += float(ce[active].sum().item())
                    active_count += int(active.sum().item())
                    correct_count += int((correct & active).sum().item())
                if content.any():
                    content_ce_sum += float(ce[content].sum().item())
                    content_count += int(content.sum().item())
                    content_correct_count += int((correct & content).sum().item())
                    row_content_counts = content.sum(dim=-1)
                    row_content_correct = (correct & content).sum(dim=-1)
                    has_content = row_content_counts > 0
                    rows_with_content += int(has_content.sum().item())
                    exact_content_rows += int(((row_content_correct == row_content_counts) & has_content).sum().item())

                if logged < log_samples:
                    reconstructed = x_eval.clone()
                    reconstructed[active] = pred[active]
                    for row_idx in range(x_eval.shape[0]):
                        if logged >= log_samples:
                            break
                        if not bool(content[row_idx].any().item()):
                            continue
                        row_content = int(content[row_idx].sum().item())
                        row_correct = int((correct[row_idx] & content[row_idx]).sum().item())
                        log(
                            f"[recon] step={step if step is not None else '-'} t={float(t_val):.2f} "
                            f"batch={batch_idx} item={row_idx} content_correct={row_correct}/{row_content}"
                        )
                        log(
                            "[recon] target="
                            f"{_decode_attention_row(tokenizer, x_eval[row_idx], batch.attention_mask[row_idx])!r}"
                        )
                        log(
                            "[recon] masked="
                            f"{_decode_masked_attention_row(tokenizer, z_eval[row_idx], batch.attention_mask[row_idx], mask_id, eos_id)!r}"
                        )
                        log(
                            "[recon] pred="
                            f"{_decode_attention_row(tokenizer, reconstructed[row_idx], batch.attention_mask[row_idx])!r}"
                        )
                        logged += 1

            key = _t_key(float(t_val))
            metrics[f"recon_{key}_ce"] = ce_sum / active_count if active_count else 0.0
            metrics[f"recon_{key}_content_ce"] = content_ce_sum / content_count if content_count else 0.0
            metrics[f"recon_{key}_token_acc"] = correct_count / active_count if active_count else 0.0
            metrics[f"recon_{key}_content_token_acc"] = content_correct_count / content_count if content_count else 0.0
            metrics[f"recon_{key}_content_exact_row_rate"] = (
                exact_content_rows / rows_with_content if rows_with_content else 0.0
            )
            metrics[f"recon_{key}_active_frac"] = active_count / lossable_count if lossable_count else 0.0
            metrics[f"recon_{key}_content_tokens"] = content_count
            metrics.update(_eos_suffix_decile_metrics(f"recon_{key}_", eos_suffix_stats))
            if use_target_length_conditioning:
                metrics.update(_target_length_recon_metrics(f"recon_{key}_", target_length_stats))

    return metrics
