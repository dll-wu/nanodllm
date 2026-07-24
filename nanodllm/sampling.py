from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .model import bd3lm_attention_mask, duplicated_position_ids
from .noise import MDLMNoise

_LOGIT_BIAS_STACK: list[torch.Tensor | None] = []


@contextmanager
def sampling_logit_bias(logit_bias: torch.Tensor | None):
    _LOGIT_BIAS_STACK.append(logit_bias)
    try:
        yield
    finally:
        _LOGIT_BIAS_STACK.pop()


def _apply_logit_bias(logits: torch.Tensor) -> torch.Tensor:
    if not _LOGIT_BIAS_STACK:
        return logits
    logit_bias = _LOGIT_BIAS_STACK[-1]
    if logit_bias is None:
        return logits
    return logits + logit_bias.to(device=logits.device, dtype=logits.dtype).view(1, 1, -1)


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _hierarchical_class_probs(model, probs: torch.Tensor) -> torch.Tensor | None:
    raw_model = _unwrap_model(model)
    token_class_ids = getattr(raw_model, "token_class_ids", None)
    class_count = int(getattr(raw_model, "output_class_count", 0) or 0)
    if token_class_ids is None or class_count <= 0 or token_class_ids.numel() != probs.shape[-1]:
        return None
    class_probs = torch.zeros(*probs.shape[:-1], class_count, dtype=probs.dtype, device=probs.device)
    class_ids = token_class_ids.to(device=probs.device, dtype=torch.long)
    index = class_ids.view(*([1] * (probs.dim() - 1)), -1).expand_as(probs)
    return class_probs.scatter_add(-1, index, probs)


def block_special_logits(logits: torch.Tensor, tokenizer) -> None:
    token_ids = tokenizer.get_generation_blocked_token_ids()
    for token_id in token_ids:
        if 0 <= token_id < logits.size(-1):
            logits[..., token_id] = -1e9


def _uses_noise_level(model) -> bool:
    config = getattr(model, "config", None)
    return bool(getattr(config, "time_conditioning", False))


def _model_logits(
    model,
    z: torch.Tensor,
    noise_level: float | None = None,
    target_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    model_kwargs = {}
    if noise_level is not None and _uses_noise_level(model):
        level = torch.full((z.shape[0],), float(noise_level), device=z.device)
        model_kwargs["noise_level"] = level
    if target_lengths is not None:
        model_kwargs["target_lengths"] = target_lengths
    return _apply_logit_bias(model(z, **model_kwargs))


def _model_length_logits(model, z: torch.Tensor) -> torch.Tensor:
    out = model(z, return_length_logits=True)
    if not isinstance(out, tuple):
        raise RuntimeError("model did not return length logits")
    length_logits = out[-1]
    if not torch.is_tensor(length_logits) or length_logits.dim() != 2 or length_logits.shape[0] != z.shape[0]:
        raise RuntimeError(
            f"length logits must have shape [batch, positions], got "
            f"{tuple(length_logits.shape) if torch.is_tensor(length_logits) else type(length_logits)}"
        )
    return length_logits.float()


def _first_eos_positions(tokens: torch.Tensor, eos_id: int | None) -> torch.Tensor:
    batch_size, length = tokens.shape
    out = torch.full((batch_size,), -1, dtype=torch.long, device=tokens.device)
    if eos_id is None:
        return out
    hits = tokens == int(eos_id)
    if not hits.any():
        return out
    positions = torch.arange(length, dtype=torch.long, device=tokens.device).view(1, length)
    first = torch.where(hits, positions, torch.full_like(positions, length)).min(dim=1).values
    has_eos = first < length
    out[has_eos] = first[has_eos]
    return out


def _record_position_counts(
    trace: dict[str, float | int] | None,
    prefix: str,
    positions: torch.Tensor,
    length: int,
) -> None:
    if trace is None:
        return
    observed = 0
    for pos in positions.detach().cpu().tolist():
        pos = int(pos)
        if pos < 0 or pos >= int(length):
            continue
        observed += 1
        key = f"length_policy_{prefix}_position_{pos:04d}_count"
        trace[key] = int(trace.get(key, 0)) + 1
    trace[f"length_policy_{prefix}_position_observed"] = (
        int(trace.get(f"length_policy_{prefix}_position_observed", 0)) + observed
    )
    trace[f"length_policy_{prefix}_position_max_len"] = int(length)


def _record_sampler_position_counts(
    trace: dict | None,
    prefix: str,
    positions: torch.Tensor,
    length: int,
) -> None:
    if trace is None:
        return
    observed = 0
    for pos in positions.detach().cpu().tolist():
        pos = int(pos)
        if pos < 0 or pos >= int(length):
            continue
        observed += 1
        key = f"{prefix}_position_{pos:04d}_count"
        trace[key] = int(trace.get(key, 0)) + 1
    trace[f"{prefix}_position_observed"] = int(trace.get(f"{prefix}_position_observed", 0)) + observed
    trace[f"{prefix}_position_max_len"] = int(length)


def _record_sampler_eos_trace(
    trace: dict | None,
    candidate_mask: torch.Tensor,
    x_pred: torch.Tensor,
    commit_mask: torch.Tensor,
    eos_id: int | None,
    length: int,
) -> torch.Tensor | None:
    if trace is None or eos_id is None:
        return None
    eos_id = int(eos_id)
    selected = candidate_mask & (x_pred == eos_id)
    committed = commit_mask & selected
    selected_count = int(selected.sum().item())
    committed_count = int(committed.sum().item())
    trace["sampler_eos_token_id"] = eos_id
    trace["sampler_eos_selected_count"] = int(trace.get("sampler_eos_selected_count", 0)) + selected_count
    trace["sampler_eos_commit_count"] = int(trace.get("sampler_eos_commit_count", 0)) + committed_count
    trace["sampler_eos_candidate_count"] = int(trace.get("sampler_eos_candidate_count", 0)) + int(
        candidate_mask.sum().item()
    )
    step_index = int(trace.get("steps", 0))
    if selected_count and "sampler_eos_first_selected_step" not in trace:
        first = selected.nonzero(as_tuple=False)[0]
        trace["sampler_eos_first_selected_step"] = step_index
        trace["sampler_eos_first_selected_row"] = int(first[0].item())
        trace["sampler_eos_first_selected_position"] = int(first[1].item())
    if committed_count and "sampler_eos_first_commit_step" not in trace:
        first = committed.nonzero(as_tuple=False)[0]
        trace["sampler_eos_first_commit_step"] = step_index
        trace["sampler_eos_first_commit_row"] = int(first[0].item())
        trace["sampler_eos_first_commit_position"] = int(first[1].item())
    if selected_count:
        _record_sampler_position_counts(trace, "sampler_eos_selected", selected.nonzero(as_tuple=False)[:, 1], length)
    if committed_count:
        _record_sampler_position_counts(trace, "sampler_eos_commit", committed.nonzero(as_tuple=False)[:, 1], length)
    return committed


def _record_sampler_final_eos_trace(
    trace: dict | None,
    tokens: torch.Tensor,
    eos_id: int | None,
    committed_eos_mask: torch.Tensor | None = None,
) -> None:
    if trace is None or eos_id is None:
        return
    eos_id = int(eos_id)
    final_eos = tokens == eos_id
    first_eos = _first_eos_positions(tokens, eos_id)
    committed_count = int(committed_eos_mask.sum().item()) if committed_eos_mask is not None else 0
    preserved_count = int((committed_eos_mask & final_eos).sum().item()) if committed_eos_mask is not None else 0
    trace["sampler_final_eos_count"] = int(final_eos.sum().item())
    trace["sampler_final_eos_positions"] = [int(v) for v in first_eos.detach().cpu().tolist()]
    trace["sampler_final_pre_cap_eos_count"] = int(((first_eos >= 0) & (first_eos < tokens.shape[1] - 1)).sum().item())
    trace["sampler_eos_committed_final_count"] = committed_count
    trace["sampler_eos_preserved_count"] = preserved_count
    trace["sampler_eos_preserved_all"] = int(preserved_count == committed_count)
    if final_eos.any():
        _record_sampler_position_counts(
            trace, "sampler_final_eos", final_eos.nonzero(as_tuple=False)[:, 1], tokens.shape[1]
        )


def _normalize_target_lengths(
    target_lengths: torch.Tensor | None,
    n: int,
    length: int,
    device,
) -> torch.Tensor | None:
    if target_lengths is None:
        return None
    out = target_lengths.to(device=device, dtype=torch.long).reshape(-1)
    if out.numel() == 1:
        out = out.expand(int(n))
    if out.numel() != int(n):
        raise ValueError(f"target_lengths must have one value or {int(n)} values, got {out.numel()}")
    if out.numel() and (int(out.min().item()) < 1 or int(out.max().item()) > int(length)):
        raise ValueError("generation target_lengths entries must be in [1, length]")
    return out


def _record_target_length_trace(
    trace: dict | None,
    target_lengths: torch.Tensor | None,
    pre_policy_tokens: torch.Tensor,
    post_policy_tokens: torch.Tensor,
    eos_id: int | None,
) -> None:
    if trace is None or target_lengths is None:
        return
    target_lengths = target_lengths.detach().to(device=post_policy_tokens.device, dtype=torch.long).reshape(-1)
    target_positions = target_lengths - 1
    pre_eos = _first_eos_positions(pre_policy_tokens, eos_id)
    post_eos = _first_eos_positions(post_policy_tokens, eos_id)
    pre_has_eos = pre_eos >= 0
    post_has_eos = post_eos >= 0
    forced_eos = post_has_eos & ~pre_has_eos
    trace["target_length_conditioning_applied"] = 1
    trace["target_length_sample_count"] = int(target_lengths.numel())
    trace["target_length_values"] = [int(v) for v in target_lengths.detach().cpu().tolist()]
    trace["target_length_pre_policy_eos_positions"] = [int(v) for v in pre_eos.detach().cpu().tolist()]
    trace["target_length_post_policy_eos_positions"] = [int(v) for v in post_eos.detach().cpu().tolist()]
    trace["target_length_pre_policy_eos_count"] = int(pre_has_eos.sum().item())
    trace["target_length_post_policy_eos_count"] = int(post_has_eos.sum().item())
    trace["target_length_forced_eos_count"] = int(forced_eos.sum().item())
    trace["target_length_no_forced_eos_success_count"] = int(pre_has_eos.sum().item())
    if pre_has_eos.any():
        errors = (pre_eos[pre_has_eos] - target_positions[pre_has_eos]).abs()
        trace["target_length_pre_policy_eos_abs_error_values"] = [int(v) for v in errors.detach().cpu().tolist()]
        trace["target_length_pre_policy_eos_abs_error_sum"] = int(errors.sum().item())
    else:
        trace["target_length_pre_policy_eos_abs_error_values"] = []
        trace["target_length_pre_policy_eos_abs_error_sum"] = 0
    if post_has_eos.any():
        errors = (post_eos[post_has_eos] - target_positions[post_has_eos]).abs()
        trace["target_length_post_policy_eos_abs_error_values"] = [int(v) for v in errors.detach().cpu().tolist()]
        trace["target_length_post_policy_eos_abs_error_sum"] = int(errors.sum().item())
    else:
        trace["target_length_post_policy_eos_abs_error_values"] = []
        trace["target_length_post_policy_eos_abs_error_sum"] = 0


@torch.no_grad()
def _apply_argmax_length_policy(
    model,
    tokens: torch.Tensor,
    tokenizer,
    token_selection: str,
    length_policy: str,
    trace: dict[str, float | int] | None,
) -> torch.Tensor:
    if length_policy == "none":
        return tokens
    if length_policy != "argmax_eos":
        raise ValueError(f"unknown length policy: {length_policy!r}")
    if token_selection != "argmax":
        if trace is not None:
            trace["length_policy_applied"] = 0
        return tokens
    eos_id = tokenizer.get_eos_token_id()
    if eos_id is None:
        if trace is not None:
            trace["length_policy_applied"] = 0
        return tokens

    pre_eos = _first_eos_positions(tokens, int(eos_id))
    length_logits = _model_length_logits(model, tokens)
    if length_logits.shape[1] < tokens.shape[1]:
        raise RuntimeError(
            f"length logits expose {length_logits.shape[1]} positions for generation length {tokens.shape[1]}"
        )
    predicted_positions = length_logits[:, : tokens.shape[1]].argmax(dim=-1).clamp(0, tokens.shape[1] - 1)
    out = tokens.clone()
    out[torch.arange(tokens.shape[0], device=tokens.device), predicted_positions] = int(eos_id)
    post_eos = _first_eos_positions(out, int(eos_id))

    if trace is not None:
        trace["length_policy_applied"] = 1
        trace["length_policy_predicted_position_sum"] = int(predicted_positions.sum().item())
        trace["length_policy_predicted_length_sum"] = int((predicted_positions + 1).sum().item())
        trace["length_policy_predicted_position_count"] = int(predicted_positions.numel())
        trace["length_policy_pre_eos_count"] = int((pre_eos >= 0).sum().item())
        trace["length_policy_post_eos_count"] = int((post_eos >= 0).sum().item())
        _record_position_counts(trace, "pre_eos", pre_eos, tokens.shape[1])
        _record_position_counts(trace, "post_eos", post_eos, tokens.shape[1])
        _record_position_counts(trace, "predicted", predicted_positions, tokens.shape[1])
    return out


def _bd3lm_model_logits(
    model,
    z_t: torch.Tensor,
    x0_context: torch.Tensor,
    attention_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    concat = torch.cat((z_t, x0_context), dim=1)
    bd3lm_mask = bd3lm_attention_mask(attention_mask, block_size)
    position_ids = duplicated_position_ids(z_t.shape[1], device=z_t.device)
    logits = model(concat, attention_mask=bd3lm_mask, position_ids=position_ids)
    return _apply_logit_bias(logits[:, : z_t.shape[1]])


def _new_trace() -> dict[str, float | int]:
    return {
        "steps": 0,
        "candidate_count": 0,
        "commit_count": 0,
        "commit_count_min": 0,
        "commit_count_max": 0,
        "candidate_entropy_sum": 0.0,
        "candidate_top1_prob_sum": 0.0,
        "pred_max_token_frac_sum": 0.0,
        "pred_max_token_frac_steps": 0,
        "class_entropy_sum": 0.0,
        "class_max_prob_sum": 0.0,
        "class_prob_eos_special_sum": 0.0,
        "class_prob_low_freq_sum": 0.0,
        "class_prob_high_freq_sum": 0.0,
        "final_mask_count": 0,
        "remask_count": 0,
    }


def _update_trace(
    trace: dict[str, float | int] | None,
    candidate_mask: torch.Tensor,
    probs: torch.Tensor,
    x_pred: torch.Tensor,
    commit_mask: torch.Tensor,
    remask_mask: torch.Tensor | None = None,
    class_probs: torch.Tensor | None = None,
) -> None:
    if trace is None:
        return

    candidate_count = int(candidate_mask.sum().item())
    commit_count = int(commit_mask.sum().item())
    trace["steps"] = int(trace["steps"]) + 1
    trace["candidate_count"] = int(trace["candidate_count"]) + candidate_count
    trace["commit_count"] = int(trace["commit_count"]) + commit_count
    if remask_mask is not None:
        trace["remask_count"] = int(trace["remask_count"]) + int(remask_mask.sum().item())
    if int(trace["steps"]) == 1:
        trace["commit_count_min"] = commit_count
        trace["commit_count_max"] = commit_count
    else:
        trace["commit_count_min"] = min(int(trace["commit_count_min"]), commit_count)
        trace["commit_count_max"] = max(int(trace["commit_count_max"]), commit_count)

    if candidate_count == 0:
        return
    top1_prob = probs.max(dim=-1).values[candidate_mask]
    safe_probs = probs.clamp_min(1e-10)
    entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)[candidate_mask]
    trace["candidate_top1_prob_sum"] = float(trace["candidate_top1_prob_sum"]) + float(top1_prob.sum().item())
    trace["candidate_entropy_sum"] = float(trace["candidate_entropy_sum"]) + float(entropy.sum().item())

    pred_ids = x_pred[candidate_mask]
    if pred_ids.numel() > 0:
        counts = torch.unique(pred_ids, return_counts=True)[1]
        trace["pred_max_token_frac_sum"] = float(trace["pred_max_token_frac_sum"]) + float(
            counts.max().item() / max(1, pred_ids.numel())
        )
        trace["pred_max_token_frac_steps"] = int(trace["pred_max_token_frac_steps"]) + 1
    if class_probs is not None:
        candidate_class_probs = class_probs[candidate_mask]
        safe_class_probs = candidate_class_probs.clamp_min(1e-10)
        trace["class_entropy_sum"] = float(trace["class_entropy_sum"]) + float(
            (-(safe_class_probs * safe_class_probs.log()).sum(dim=-1)).sum().item()
        )
        trace["class_max_prob_sum"] = float(trace["class_max_prob_sum"]) + float(
            candidate_class_probs.max(dim=-1).values.sum().item()
        )
        trace["class_prob_eos_special_sum"] = float(trace["class_prob_eos_special_sum"]) + float(
            candidate_class_probs[:, 0].sum().item()
        )
        if candidate_class_probs.shape[-1] > 1:
            trace["class_prob_low_freq_sum"] = float(trace["class_prob_low_freq_sum"]) + float(
                candidate_class_probs[:, 1].sum().item()
            )
        if candidate_class_probs.shape[-1] > 7:
            trace["class_prob_high_freq_sum"] = float(trace["class_prob_high_freq_sum"]) + float(
                candidate_class_probs[:, 7].sum().item()
            )


@torch.no_grad()
def sample_mdlm(
    model,
    noise: MDLMNoise,
    tokenizer,
    n: int,
    length: int,
    n_steps: int,
    scoring_mode: str = "margin",
    token_selection: str = "argmax",
    temperature: float = 1.0,
    jitter_scale: float = 0.0,
    return_trace: bool = False,
    target_lengths: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    if scoring_mode not in ("random", "maxprob", "margin", "entropy", "margin_jitter"):
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection}")

    device = next(model.parameters()).device
    mask_id = noise.mask_token_id
    z = torch.full((n, length), mask_id, dtype=torch.long, device=device)
    target_lengths = _normalize_target_lengths(target_lengths, n, length, device)
    trace = _new_trace() if return_trace else None
    for step in range(n_steps, 0, -1):
        still_masked = z == mask_id
        if not still_masked.any():
            break
        logits = _model_logits(model, z, target_lengths=target_lengths)
        block_special_logits(logits, tokenizer)
        probs = F.softmax(logits / temperature, dim=-1)
        if token_selection == "argmax":
            x_pred = logits.argmax(dim=-1)
        else:
            x_pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(n, length)

        p_unmask = 1.0 / step
        if scoring_mode == "random":
            unmask = still_masked & (torch.rand_like(still_masked.float()) < p_unmask)
        else:
            k_per_seq = (p_unmask * still_masked.sum(dim=-1)).long().clamp(min=1)
            if scoring_mode == "maxprob":
                conf = probs.max(dim=-1).values
            elif scoring_mode in ("margin", "margin_jitter"):
                top2 = probs.topk(2, dim=-1).values
                conf = top2[..., 0] - top2[..., 1]
                if scoring_mode == "margin_jitter" and jitter_scale > 0:
                    conf = conf + (-torch.empty_like(conf).exponential_().log() - 0.5772156649) * jitter_scale
            else:
                p = probs.clamp_min(1e-10)
                conf = (p * p.log()).sum(dim=-1)
            scores = conf.masked_fill(~still_masked, -1e9)
            unmask = torch.zeros_like(still_masked)
            for i in range(n):
                _, idx = torch.topk(scores[i], int(k_per_seq[i].item()))
                unmask[i, idx] = True
        _update_trace(trace, still_masked, probs, x_pred, unmask, class_probs=_hierarchical_class_probs(model, probs))
        z[unmask] = x_pred[unmask]

    if (z == mask_id).any():
        logits = _model_logits(model, z, target_lengths=target_lengths)
        block_special_logits(logits, tokenizer)
        z[z == mask_id] = logits.argmax(dim=-1)[z == mask_id]
    if trace is not None:
        _record_target_length_trace(trace, target_lengths, z, z, tokenizer.get_eos_token_id())
        trace["final_mask_count"] = int((z == mask_id).sum().item())
        return z, trace
    return z


@torch.no_grad()
def sample_mdlm_from_prefixes(
    model,
    noise: MDLMNoise,
    tokenizer,
    prefixes: torch.Tensor,
    continuation_len: int,
    n_steps: int,
    scoring_mode: str = "margin",
    token_selection: str = "argmax",
    temperature: float = 1.0,
    jitter_scale: float = 0.0,
    return_trace: bool = False,
    target_lengths: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    if scoring_mode not in ("random", "maxprob", "margin", "entropy", "margin_jitter"):
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection}")
    if prefixes.dim() != 2:
        raise ValueError(f"prefixes must be 2D, got shape {tuple(prefixes.shape)}")
    if int(continuation_len) <= 0:
        raise ValueError("continuation_len must be positive")

    device = next(model.parameters()).device
    mask_id = noise.mask_token_id
    prefixes = prefixes.to(device=device, dtype=torch.long)
    n, prefix_len = prefixes.shape
    length = int(prefix_len) + int(continuation_len)
    z = torch.full((n, length), int(mask_id), dtype=torch.long, device=device)
    z[:, :prefix_len] = prefixes
    continuation_mask = torch.zeros_like(z, dtype=torch.bool)
    continuation_mask[:, prefix_len:] = True
    target_lengths = _normalize_target_lengths(target_lengths, n, length, device)
    trace = _new_trace() if return_trace else None
    for step in range(n_steps, 0, -1):
        still_masked = (z == mask_id) & continuation_mask
        if not still_masked.any():
            break
        logits = _model_logits(model, z, target_lengths=target_lengths)
        block_special_logits(logits, tokenizer)
        probs = F.softmax(logits / temperature, dim=-1)
        if token_selection == "argmax":
            x_pred = logits.argmax(dim=-1)
        else:
            x_pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(n, length)

        p_unmask = 1.0 / step
        if scoring_mode == "random":
            unmask = still_masked & (torch.rand_like(still_masked.float()) < p_unmask)
        else:
            k_per_seq = (p_unmask * still_masked.sum(dim=-1)).long().clamp(min=1)
            if scoring_mode == "maxprob":
                conf = probs.max(dim=-1).values
            elif scoring_mode in ("margin", "margin_jitter"):
                top2 = probs.topk(2, dim=-1).values
                conf = top2[..., 0] - top2[..., 1]
                if scoring_mode == "margin_jitter" and jitter_scale > 0:
                    conf = conf + (-torch.empty_like(conf).exponential_().log() - 0.5772156649) * jitter_scale
            else:
                p = probs.clamp_min(1e-10)
                conf = (p * p.log()).sum(dim=-1)
            scores = conf.masked_fill(~still_masked, -1e9)
            unmask = torch.zeros_like(still_masked)
            for i in range(n):
                _, idx = torch.topk(scores[i], int(k_per_seq[i].item()))
                unmask[i, idx] = True
        _update_trace(trace, still_masked, probs, x_pred, unmask, class_probs=_hierarchical_class_probs(model, probs))
        z[unmask] = x_pred[unmask]

    final_mask = (z == mask_id) & continuation_mask
    if final_mask.any():
        logits = _model_logits(model, z, target_lengths=target_lengths)
        block_special_logits(logits, tokenizer)
        z[final_mask] = logits.argmax(dim=-1)[final_mask]
    if trace is not None:
        _record_target_length_trace(trace, target_lengths, z, z, tokenizer.get_eos_token_id())
        trace["final_mask_count"] = int(((z == mask_id) & continuation_mask).sum().item())
        return z, trace
    return z


@torch.no_grad()
def sample_remdm(
    model,
    noise: MDLMNoise,
    tokenizer,
    n: int,
    length: int,
    n_steps: int,
    scoring_mode: str = "margin",
    token_selection: str = "argmax",
    temperature: float = 1.0,
    jitter_scale: float = 0.0,
    tau_edit: float = 0.9,
    return_trace: bool = False,
    target_lengths: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    if scoring_mode not in ("random", "maxprob", "margin", "entropy", "margin_jitter"):
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection}")

    device = next(model.parameters()).device
    mask_id = noise.mask_token_id
    z = torch.full((n, length), mask_id, dtype=torch.long, device=device)
    target_lengths = _normalize_target_lengths(target_lengths, n, length, device)
    trace = _new_trace() if return_trace else None
    for step in range(n_steps, 0, -1):
        still_masked = z == mask_id
        if not still_masked.any():
            break
        logits = _model_logits(model, z, target_lengths=target_lengths)
        block_special_logits(logits, tokenizer)
        probs = F.softmax(logits / temperature, dim=-1)
        if token_selection == "argmax":
            x_pred = logits.argmax(dim=-1)
        else:
            x_pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(n, length)

        p_unmask = 1.0 / step
        if scoring_mode == "random":
            unmask = still_masked & (torch.rand_like(still_masked.float()) < p_unmask)
        else:
            k_per_seq = (p_unmask * still_masked.sum(dim=-1)).long().clamp(min=1)
            if scoring_mode == "maxprob":
                conf = probs.max(dim=-1).values
            elif scoring_mode in ("margin", "margin_jitter"):
                top2 = probs.topk(2, dim=-1).values
                conf = top2[..., 0] - top2[..., 1]
                if scoring_mode == "margin_jitter" and jitter_scale > 0:
                    conf = conf + (-torch.empty_like(conf).exponential_().log() - 0.5772156649) * jitter_scale
            else:
                p = probs.clamp_min(1e-10)
                conf = (p * p.log()).sum(dim=-1)
            scores = conf.masked_fill(~still_masked, -1e9)
            unmask = torch.zeros_like(still_masked)
            for i in range(n):
                _, idx = torch.topk(scores[i], int(k_per_seq[i].item()))
                unmask[i, idx] = True

        committed = ~still_masked
        remask = torch.zeros_like(still_masked)
        if committed.any():
            top1_prob = probs.max(dim=-1).values
            remask = committed & (top1_prob > tau_edit) & (x_pred != z)

        _update_trace(
            trace,
            still_masked,
            probs,
            x_pred,
            unmask,
            remask,
            class_probs=_hierarchical_class_probs(model, probs),
        )
        z[remask] = mask_id
        z[unmask] = x_pred[unmask]

    if (z == mask_id).any():
        logits = _model_logits(model, z, target_lengths=target_lengths)
        block_special_logits(logits, tokenizer)
        z[z == mask_id] = logits.argmax(dim=-1)[z == mask_id]
    if trace is not None:
        _record_target_length_trace(trace, target_lengths, z, z, tokenizer.get_eos_token_id())
        trace["final_mask_count"] = int((z == mask_id).sum().item())
        return z, trace
    return z


@torch.no_grad()
def sample_uniform(
    model,
    noise,
    tokenizer,
    n: int,
    length: int,
    n_steps: int,
    scoring_mode: str = "margin",
    token_selection: str = "argmax",
    temperature: float = 1.0,
    jitter_scale: float = 0.0,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    if scoring_mode not in ("random", "maxprob", "margin", "entropy", "margin_jitter"):
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection}")

    device = next(model.parameters()).device
    z = noise.sample_uniform_tokens((n, length), device=device)
    committed = torch.zeros((n, length), dtype=torch.bool, device=device)
    trace = _new_trace() if return_trace else None
    for step in range(max(1, n_steps), 0, -1):
        uncommitted = ~committed
        if not uncommitted.any():
            break
        logits = _model_logits(model, z, noise_level=step / max(1, n_steps))
        block_special_logits(logits, tokenizer)
        probs = F.softmax(logits / temperature, dim=-1)
        if token_selection == "argmax":
            x_pred = logits.argmax(dim=-1)
        else:
            x_pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(n, length)

        p_commit = 1.0 / step
        if scoring_mode == "random":
            commit = uncommitted & (torch.rand_like(uncommitted.float()) < p_commit)
        else:
            if scoring_mode == "maxprob":
                conf = probs.max(dim=-1).values
            elif scoring_mode in ("margin", "margin_jitter"):
                top2 = probs.topk(2, dim=-1).values
                conf = top2[..., 0] - top2[..., 1]
                if scoring_mode == "margin_jitter" and jitter_scale > 0:
                    conf = conf + (-torch.empty_like(conf).exponential_().log() - 0.5772156649) * jitter_scale
            else:
                p = probs.clamp_min(1e-10)
                conf = (p * p.log()).sum(dim=-1)
            scores = conf.masked_fill(~uncommitted, -1e9)
            commit = torch.zeros_like(uncommitted)
            for i in range(n):
                remaining = int(uncommitted[i].sum().item())
                if remaining == 0:
                    continue
                k = max(1, min(remaining, int(p_commit * remaining)))
                _, idx = torch.topk(scores[i], k)
                commit[i, idx] = True
        _update_trace(trace, uncommitted, probs, x_pred, commit, class_probs=_hierarchical_class_probs(model, probs))
        z[commit] = x_pred[commit]
        committed[commit] = True

    if (~committed).any():
        remaining = ~committed
        logits = _model_logits(model, z, noise_level=0.0)
        block_special_logits(logits, tokenizer)
        z[remaining] = logits.argmax(dim=-1)[remaining]
        committed[remaining] = True
    if trace is not None:
        trace["final_mask_count"] = int((~committed).sum().item())
        return z, trace
    return z


@torch.no_grad()
def sample_mask_uniform(
    model,
    noise,
    tokenizer,
    n: int,
    length: int,
    n_steps: int,
    scoring_mode: str = "margin",
    token_selection: str = "argmax",
    temperature: float = 1.0,
    jitter_scale: float = 0.0,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    if scoring_mode not in ("random", "maxprob", "margin", "entropy", "margin_jitter"):
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection}")

    device = next(model.parameters()).device
    mask_id = noise.mask_token_id
    if hasattr(noise, "sample_terminal_tokens"):
        z = noise.sample_terminal_tokens((n, length), device=device)
    else:
        z = torch.full((n, length), mask_id, dtype=torch.long, device=device)
    committed = torch.zeros((n, length), dtype=torch.bool, device=device)
    trace = _new_trace() if return_trace else None
    for step in range(max(1, n_steps), 0, -1):
        editable = ~committed
        if not editable.any():
            break
        logits = _model_logits(model, z, noise_level=step / max(1, n_steps))
        block_special_logits(logits, tokenizer)
        probs = F.softmax(logits / temperature, dim=-1)
        if token_selection == "argmax":
            x_pred = logits.argmax(dim=-1)
        else:
            x_pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(n, length)

        p_commit = 1.0 / step
        if scoring_mode == "random":
            commit = editable & (torch.rand_like(editable.float()) < p_commit)
        else:
            if scoring_mode == "maxprob":
                conf = probs.max(dim=-1).values
            elif scoring_mode in ("margin", "margin_jitter"):
                top2 = probs.topk(2, dim=-1).values
                conf = top2[..., 0] - top2[..., 1]
                if scoring_mode == "margin_jitter" and jitter_scale > 0:
                    conf = conf + (-torch.empty_like(conf).exponential_().log() - 0.5772156649) * jitter_scale
            else:
                p = probs.clamp_min(1e-10)
                conf = (p * p.log()).sum(dim=-1)
            scores = conf.masked_fill(~editable, -1e9)
            commit = torch.zeros_like(editable)
            for i in range(n):
                remaining = int(editable[i].sum().item())
                if remaining == 0:
                    continue
                k = max(1, min(remaining, int(p_commit * remaining)))
                _, idx = torch.topk(scores[i], k)
                commit[i, idx] = True
        _update_trace(trace, editable, probs, x_pred, commit, class_probs=_hierarchical_class_probs(model, probs))
        z[editable] = x_pred[editable]
        committed[commit] = True

    if (~committed).any() or (z == mask_id).any():
        remaining = (~committed) | (z == mask_id)
        logits = _model_logits(model, z, noise_level=0.0)
        block_special_logits(logits, tokenizer)
        z[remaining] = logits.argmax(dim=-1)[remaining]
        committed[remaining] = True
    if trace is not None:
        trace["final_mask_count"] = int((z == mask_id).sum().item())
        return z, trace
    return z


@torch.no_grad()
def sample_bd3lm(
    model,
    noise: MDLMNoise,
    tokenizer,
    n: int,
    length: int,
    n_steps: int,
    block_size: int,
    scoring_mode: str = "margin",
    token_selection: str = "argmax",
    temperature: float = 1.0,
    jitter_scale: float = 0.0,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    if scoring_mode not in ("random", "maxprob", "margin", "entropy", "margin_jitter"):
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    device = next(model.parameters()).device
    mask_id = noise.mask_token_id
    z = torch.full((n, length), mask_id, dtype=torch.long, device=device)
    x0_context = torch.full_like(z, mask_id)
    attention_mask = torch.ones((n, length), dtype=torch.bool, device=device)
    trace = _new_trace() if return_trace else None

    for block_start in range(0, length, int(block_size)):
        block_end = min(length, block_start + int(block_size))
        block_slice = slice(block_start, block_end)
        for step in range(max(1, n_steps), 0, -1):
            still_masked = z[:, block_slice] == mask_id
            if not still_masked.any():
                break
            block_logits = _bd3lm_model_logits(model, z, x0_context, attention_mask, int(block_size))[:, block_slice]
            block_special_logits(block_logits, tokenizer)
            probs = F.softmax(block_logits / temperature, dim=-1)
            if token_selection == "argmax":
                x_pred = block_logits.argmax(dim=-1)
            else:
                x_pred = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view(n, block_end - block_start)

            p_unmask = 1.0 / step
            if scoring_mode == "random":
                commit = still_masked & (torch.rand_like(still_masked.float()) < p_unmask)
            else:
                k_per_seq = (p_unmask * still_masked.sum(dim=-1)).long().clamp(min=1)
                if scoring_mode == "maxprob":
                    conf = probs.max(dim=-1).values
                elif scoring_mode in ("margin", "margin_jitter"):
                    top2 = probs.topk(2, dim=-1).values
                    conf = top2[..., 0] - top2[..., 1]
                    if scoring_mode == "margin_jitter" and jitter_scale > 0:
                        conf = conf + (-torch.empty_like(conf).exponential_().log() - 0.5772156649) * jitter_scale
                else:
                    p = probs.clamp_min(1e-10)
                    conf = (p * p.log()).sum(dim=-1)
                scores = conf.masked_fill(~still_masked, -1e9)
                commit = torch.zeros_like(still_masked)
                for i in range(n):
                    remaining = int(still_masked[i].sum().item())
                    if remaining == 0:
                        continue
                    k = min(remaining, int(k_per_seq[i].item()))
                    _, idx = torch.topk(scores[i], k)
                    commit[i, idx] = True

            _update_trace(
                trace,
                still_masked,
                probs,
                x_pred,
                commit,
                class_probs=_hierarchical_class_probs(model, probs),
            )
            block_tokens = z[:, block_slice]
            block_tokens[commit] = x_pred[commit]

        if (z[:, block_slice] == mask_id).any():
            remaining = z[:, block_slice] == mask_id
            block_logits = _bd3lm_model_logits(model, z, x0_context, attention_mask, int(block_size))[:, block_slice]
            block_special_logits(block_logits, tokenizer)
            block_tokens = z[:, block_slice]
            block_tokens[remaining] = block_logits.argmax(dim=-1)[remaining]

        x0_context[:, block_slice] = z[:, block_slice]

    if trace is not None:
        trace["final_mask_count"] = int((z == mask_id).sum().item())
        return z, trace
    return z


@torch.no_grad()
def sample_block_causal_pilot(
    model,
    noise: MDLMNoise,
    tokenizer,
    n: int,
    length: int,
    n_steps: int,
    block_size: int,
    scoring_mode: str = "margin",
    token_selection: str = "argmax",
    temperature: float = 1.0,
    jitter_scale: float = 0.0,
    tau: float = 0.9,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    if scoring_mode not in ("random", "maxprob", "margin", "entropy", "margin_jitter"):
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    device = next(model.parameters()).device
    mask_id = noise.mask_token_id
    z = torch.full((n, length), mask_id, dtype=torch.long, device=device)
    trace = _new_trace() if return_trace else None
    num_blocks = (length + block_size - 1) // block_size

    positions = torch.arange(length, device=device)
    block_ids = positions // block_size
    causal_mask = block_ids[None, :] <= block_ids[:, None]

    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(length, block_start + block_size)
        block_slice = slice(block_start, block_end)

        for step in range(max(1, n_steps), 0, -1):
            still_masked = z[:, block_slice] == mask_id
            if not still_masked.any():
                break

            attn_mask = causal_mask[None, :, :].expand(n, -1, -1)
            logits = _apply_logit_bias(model(z, attention_mask=attn_mask))[:, block_slice]
            block_special_logits(logits, tokenizer)
            probs = F.softmax(logits / temperature, dim=-1)
            if token_selection == "argmax":
                x_pred = logits.argmax(dim=-1)
            else:
                x_pred = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view(n, block_end - block_start)

            selected_prob = probs.gather(-1, x_pred.unsqueeze(-1)).squeeze(-1)
            commit = still_masked & (selected_prob >= tau)
            if not commit.any():
                p_unmask = 1.0 / step
                k_per_seq = (p_unmask * still_masked.sum(dim=-1)).long().clamp(min=1)
                scores = selected_prob.masked_fill(~still_masked, -1e9)
                commit = torch.zeros_like(still_masked)
                for i in range(n):
                    _, idx = torch.topk(scores[i], int(k_per_seq[i].item()))
                    commit[i, idx] = True

            _update_trace(
                trace,
                still_masked,
                probs,
                x_pred,
                commit,
                class_probs=_hierarchical_class_probs(model, probs),
            )
            block_tokens = z[:, block_slice]
            block_tokens[commit] = x_pred[commit]

        if (z[:, block_slice] == mask_id).any():
            remaining = z[:, block_slice] == mask_id
            attn_mask = causal_mask[None, :, :].expand(n, -1, -1)
            logits = _apply_logit_bias(model(z, attention_mask=attn_mask))[:, block_slice]
            block_special_logits(logits, tokenizer)
            block_tokens = z[:, block_slice]
            block_tokens[remaining] = logits.argmax(dim=-1)[remaining]

    if trace is not None:
        trace["final_mask_count"] = int((z == mask_id).sum().item())
        trace["num_blocks"] = num_blocks
        trace["nfe_per_block"] = n_steps
        return z, trace
    return z


@torch.no_grad()
@torch.no_grad()
def sample_ar(
    model,
    tokenizer,
    n: int,
    length: int,
    temperature: float = 1.0,
    token_selection: str = "multinomial",
) -> torch.Tensor:
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection!r}")
    device = next(model.parameters()).device
    bos_id = tokenizer.get_bos_token_id()
    pad_id = tokenizer.get_pad_token_id()
    blocked_ids = tokenizer.get_generation_blocked_token_ids()
    tokens = torch.full((n, length), pad_id if pad_id is not None else 0, dtype=torch.long, device=device)
    start_pos = 0
    if bos_id is not None:
        tokens[:, 0] = bos_id
        start_pos = 1
    else:
        # Seed first token from a single PAD input so the model has something to attend to.
        seed = torch.full((n, 1), pad_id if pad_id is not None else 0, dtype=torch.long, device=device)
        logits = model(seed)[:, -1, :]
        for token_id in blocked_ids:
            if 0 <= token_id < logits.size(-1):
                logits[..., token_id] = -1e9
        logits = _apply_logit_bias(logits)
        if token_selection == "argmax":
            tokens[:, 0] = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            tokens[:, 0] = torch.multinomial(probs, 1).squeeze(-1)
        start_pos = 1
    for pos in range(start_pos, length):
        logits = model(tokens[:, :pos])[:, -1, :]
        for token_id in blocked_ids:
            if 0 <= token_id < logits.size(-1):
                logits[..., token_id] = -1e9
        logits = _apply_logit_bias(logits)
        if token_selection == "argmax":
            next_token = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
        tokens[:, pos] = next_token
    return tokens


def sample_fhs(
    model,
    noise: MDLMNoise,
    tokenizer,
    n: int,
    length: int,
    n_steps: int,
    scoring_mode: str = "margin",
    token_selection: str = "argmax",
    temperature: float = 1.0,
    jitter_scale: float = 0.0,
    tau: float = 0.9,
    length_policy: str = "none",
    return_trace: bool = False,
    target_lengths: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    if token_selection not in ("argmax", "multinomial"):
        raise ValueError(f"unknown token selection: {token_selection}")

    device = next(model.parameters()).device
    mask_id = noise.mask_token_id
    z = torch.full((n, length), mask_id, dtype=torch.long, device=device)
    target_lengths = _normalize_target_lengths(target_lengths, n, length, device)
    trace = _new_trace() if return_trace else None
    eos_id = tokenizer.get_eos_token_id()
    committed_eos_mask = torch.zeros_like(z, dtype=torch.bool) if trace is not None and eos_id is not None else None
    for step in range(n_steps, 0, -1):
        still_masked = z == mask_id
        if not still_masked.any():
            break
        logits = _model_logits(model, z, target_lengths=target_lengths)
        block_special_logits(logits, tokenizer)
        probs = F.softmax(logits / temperature, dim=-1)
        if token_selection == "argmax":
            x_pred = logits.argmax(dim=-1)
        else:
            x_pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(n, length)

        selected_prob = probs.gather(-1, x_pred.unsqueeze(-1)).squeeze(-1)
        commit = still_masked & (selected_prob >= tau)

        # Fallback: if threshold is too high and nothing commits, force top-k
        # by confidence following the MDLM unmasking schedule.
        if not commit.any():
            p_unmask = 1.0 / step
            k_per_seq = (p_unmask * still_masked.sum(dim=-1)).long().clamp(min=1)
            scores = selected_prob.masked_fill(~still_masked, -1e9)
            commit = torch.zeros_like(still_masked)
            for i in range(n):
                _, idx = torch.topk(scores[i], int(k_per_seq[i].item()))
                commit[i, idx] = True

        _update_trace(trace, still_masked, probs, x_pred, commit, class_probs=_hierarchical_class_probs(model, probs))
        eos_commit = _record_sampler_eos_trace(trace, still_masked, x_pred, commit, eos_id, length)
        if committed_eos_mask is not None and eos_commit is not None:
            committed_eos_mask |= eos_commit
        z[commit] = x_pred[commit]

    if (z == mask_id).any():
        logits = _model_logits(model, z, target_lengths=target_lengths)
        block_special_logits(logits, tokenizer)
        z[z == mask_id] = logits.argmax(dim=-1)[z == mask_id]
    pre_policy_z = z.clone() if target_lengths is not None else z
    z = _apply_argmax_length_policy(model, z, tokenizer, token_selection, length_policy, trace)
    if trace is not None:
        _record_target_length_trace(trace, target_lengths, pre_policy_z, z, tokenizer.get_eos_token_id())
        _record_sampler_final_eos_trace(trace, z, eos_id, committed_eos_mask)
        trace["final_mask_count"] = int((z == mask_id).sum().item())
        return z, trace
    return z
