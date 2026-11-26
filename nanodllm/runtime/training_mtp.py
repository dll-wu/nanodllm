"""Admission, aligned targets and telemetry for single-future-token training."""

from __future__ import annotations

import math
import os

import torch
from torch.nn import functional as F

from .model_mtp import adjacent_support, validate_mtp_model


def validate_mtp_config(config) -> None:
    validate_mtp_model(config.model)
    weight = config.train.mtp_loss_weight
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("train.mtp_loss_weight must be finite and nonnegative")
    if config.model.mtp_mode == "none":
        if weight:
            raise ValueError("train.mtp_loss_weight requires model.mtp_mode")
        return
    if (
        config.optim.type != "adamw"
        or config.parameterization.mode != "ordinary"
        or config.parameterization.execution_scope != "ordinary"
        or config.parameterization.hidden_init_std_multiplier != 1.0
    ):
        raise ValueError("MTP requires ordinary AdamW without scoped parameterization")
    if config.noise.type != "mdlm" or config.noise.time_conditioning or config.model.block_causal_pilot_size:
        raise ValueError("MTP supports plain causal next-token training only")
    if not (config.data.content_budget_segmented_packing or config.data.boundary_aware_packing):
        raise ValueError("MTP training requires document-aware packing")
    if config.runtime.init_single_rank_process_group:
        raise ValueError("MTP requires single-process training without DDP")
    if config.train.mdlm_auxiliary.mode != "none" or any(
        (
            config.train.auxiliary_loss_weight,
            config.train.label_smoothing_epsilon,
            config.train.entropy_bonus_beta,
            config.train.confidence_penalty_weight,
            config.train.z_loss_weight,
            config.train.eos_hazard_loss_weight,
            config.train.length_head_loss_weight,
            config.train.position_decile_balanced_loss,
            config.train.token_edit_corruption_batch_fraction,
            config.train.bd3lm_causal_ar_loss_weight,
        )
    ):
        raise ValueError("MTP does not support coupled training objectives")


def validate_mtp_process(config) -> None:
    """Reject torchrun before distributed initialization or model allocation."""
    if config.model.mtp_mode != "none" and (
        "RANK" in os.environ
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
        or config.runtime.init_single_rank_process_group
        or (torch.distributed.is_initialized())
    ):
        raise ValueError("MTP requires single-process training without DDP")


def mtp_training_enabled(config) -> bool:
    return config.model.mtp_mode != "none" and config.train.mtp_loss_weight > 0


def mtp_compute_metrics(config, model, tokens: int) -> dict:
    if config.model.mtp_mode == "none":
        return {}
    extra = model.mtp_flops() if mtp_training_enabled(config) else 0
    return {
        "mtp_parameters": sum(p.numel() for p in model.mtp.parameters()),
        "mtp_main_training_flops_per_token": model.estimate_flops(include_mtp=False),
        "mtp_auxiliary_training_flops_per_token": extra,
        "mtp_auxiliary_training_flops_used": extra * tokens,
        "mtp_compute_estimate_method": "dense_training_including_shared_head_reuse",
    }


def future_targets(batch, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Require every intervening edge to stay within one valid document segment."""
    if horizon not in (1, 2):
        raise ValueError("MTP target horizon must be one or two")
    tokens, valid = batch.x_0, batch.attention_mask
    length = max(0, tokens.shape[1] - horizon)
    targets = tokens[:, horizon:]
    support = batch.loss_mask[:, horizon:] & valid[:, horizon:]
    adjacent = valid[:, 1:] & valid[:, :-1]
    boundary = getattr(batch, "document_boundary_mask", None)
    if boundary is not None:
        adjacent = adjacent & adjacent_support(tokens, boundary)
    positions = getattr(batch, "position_ids", None)
    if positions is not None:
        adjacent = adjacent & (positions[:, 1:] == positions[:, :-1] + 1)
    starts = getattr(batch, "document_start_mask", None)
    if starts is not None:
        adjacent = adjacent & ~starts[:, 1:]
    for offset in range(horizon):
        support = support & adjacent[:, offset : offset + length]
    return targets, support


def future_token_loss(logits: torch.Tensor, batch, horizon: int) -> torch.Tensor:
    targets, support = future_targets(batch, horizon)
    predictions = logits[:, : targets.shape[1]]
    # Index before CE so ignored targets cannot create NaNs. An empty sum stays
    # connected to both the branch and backbone without inventing a target.
    selected = predictions[support].float()
    if selected.shape[0] == 0:
        return selected.sum()
    return F.cross_entropy(selected, targets[support])


def mtp_metrics(config, *, main_loss: float, total_loss: float) -> dict:
    if config.model.mtp_mode == "none":
        return {}
    weight = config.train.mtp_loss_weight
    weighted = total_loss - main_loss if weight > 0 else 0.0
    return {
        "mtp_mode": config.model.mtp_mode,
        "mtp_loss_weight": weight,
        "mtp_main_loss": main_loss,
        "mtp_auxiliary_loss": weighted / weight if weight > 0 else 0.0,
        "mtp_weighted_auxiliary_loss": weighted,
        "mtp_branch_executed": weight > 0,
    }


def prediction_quality_metrics(config, last, *, causal: bool, has_ar_measurement: bool) -> dict:
    """Keep autoregressive quality main-head-only when total loss also includes MTP."""
    ar_ce = last.loss_value if causal and config.model.mtp_mode == "none" else last.ar_unsmoothed_ce
    return {
        "ar_last_train_ce": ar_ce if has_ar_measurement else 0.0,
        "ar_last_train_label_smoothed_ce": ar_ce if has_ar_measurement else 0.0,
        "ar_last_train_unsmoothed_ce": last.ar_unsmoothed_ce if has_ar_measurement else 0.0,
        "last_train_entropy_mean": last.entropy_mean,
        "last_train_top1_prob_mean": last.top1_prob_mean,
        "last_train_argmax_mode_frac": last.argmax_mode_frac,
        "last_train_masked_logit_norm_mean": last.logit_norm_mean,
        "last_train_masked_logit_norm_min": last.logit_norm_min,
        "last_train_masked_logit_norm_max": last.logit_norm_max,
        "last_train_masked_logit_rms_mean": last.logit_rms_mean,
        "last_train_masked_logsumexp_mean": last.logsumexp_mean,
        "last_train_masked_logsumexp_min": last.logsumexp_min,
        "last_train_masked_logsumexp_max": last.logsumexp_max,
    }


def causal_training_loss(session, *, label_smoothing) -> None:
    """Populate the existing causal loss state under the caller's autocast scope."""
    if mtp_training_enabled(session.config):
        session.logits, mtp_logits = session.model(session.x_0, return_mtp_logits=True, **session.model_kwargs)
    else:
        session.logits = session.model(session.x_0, **session.model_kwargs)
    session.logits_pred = session.logits[:, :-1, :].contiguous()
    session.targets = session.x_0[:, 1:].contiguous()
    session.target_mask = session.loss_mask[:, 1:] & session.attention_mask[:, 1:]
    if session.config.model.mtp_mode != "none":
        _, session.target_mask = future_targets(session.batch, 1)
    session.label_smoothing = label_smoothing(session.config)
    session.ce = F.cross_entropy(
        session.logits_pred.view(-1, session.logits_pred.size(-1)),
        session.targets.view(-1),
        reduction="none",
        label_smoothing=session.label_smoothing,
    ).view_as(session.targets)
    if session.label_smoothing > 0:
        session.unsmoothed_ce = F.cross_entropy(
            session.logits_pred.view(-1, session.logits_pred.size(-1)),
            session.targets.view(-1),
            reduction="none",
        ).view_as(session.targets)
    else:
        session.unsmoothed_ce = session.ce
    if session.target_mask.any():
        session.raw_main_ce = session.ce[session.target_mask].mean()
        session.raw_unsmoothed_ce = session.unsmoothed_ce[session.target_mask].mean()
    else:
        session.raw_main_ce = torch.tensor(0.0, device=session.device)
        session.raw_unsmoothed_ce = torch.tensor(0.0, device=session.device)
        if session.config.model.mtp_mode != "none":
            session.raw_main_ce = session.logits_pred.float().sum() * 0.0
            session.raw_unsmoothed_ce = session.raw_main_ce
    if session.config.train.entropy_bonus_beta > 0:
        session.pred_probs = F.softmax(session.logits_pred, dim=-1).float()
        session.pred_log_probs = F.log_softmax(session.logits_pred, dim=-1).float()
        session.entropy_values = -(session.pred_probs * session.pred_log_probs).sum(dim=-1)
        if session.target_mask.any():
            session.raw_entropy = session.entropy_values[session.target_mask].mean()
            session.raw_max_prob = session.pred_probs.max(dim=-1).values[session.target_mask].mean()
            session.argmax_ids = session.pred_probs.argmax(dim=-1)[session.target_mask]
            if session.argmax_ids.numel() > 0:
                session.mode_val = torch.mode(session.argmax_ids).values
                session.raw_argmax_mode_frac = (session.argmax_ids == session.mode_val).float().mean()
            else:
                session.raw_argmax_mode_frac = torch.tensor(0.0, device=session.device)
        else:
            session.raw_entropy = torch.tensor(0.0, device=session.device)
            session.raw_max_prob = torch.tensor(0.0, device=session.device)
            session.raw_argmax_mode_frac = torch.tensor(0.0, device=session.device)
        session.entropy_bonus = -session.config.train.entropy_bonus_beta * session.raw_entropy
    else:
        session.entropy_bonus = torch.tensor(0.0, device=session.device)
        session.raw_entropy = torch.tensor(0.0, device=session.device)
        session.raw_max_prob = torch.tensor(0.0, device=session.device)
        session.raw_argmax_mode_frac = torch.tensor(0.0, device=session.device)
    session.loss = session.raw_main_ce + session.entropy_bonus
    if mtp_training_enabled(session.config):
        horizon = 1 if session.config.model.mtp_mode == "auxiliary_ntp" else 2
        session.loss = session.loss + session.config.train.mtp_loss_weight * future_token_loss(
            mtp_logits, session.batch, horizon
        )
    session.loss_normalization_scale = 1.0
    session.main_loss_for_log = session.raw_main_ce
