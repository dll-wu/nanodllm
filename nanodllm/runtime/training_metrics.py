import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import training_progress as _training_progress
from .config import ExperimentConfig
from .data import build_token_bytes, build_val_loader, make_dataloader
from .model import (
    DiffusionTransformer,
    embedding_head_init_std,
    residual_output_projection_init_std,
    rms_norm,
)
from .noise import build_noise
from .training_features import _uses_length_head
from .training_objective import _corrupt_eval_batch

# Preserve the metrics-module compatibility exports while keeping the progress
# schema constants owned by ``training_progress``.
EOS_HAZARD_RELATIVE_POSITION_DECILES = _training_progress.EOS_HAZARD_RELATIVE_POSITION_DECILES
LOGIT_SCALE_HIST_BINS = _training_progress.LOGIT_SCALE_HIST_BINS
PER_TOKEN_NOISE_LEVEL_HIST_BINS = _training_progress.PER_TOKEN_NOISE_LEVEL_HIST_BINS
POSITION_DECILE_BALANCED_LOSS_DECILES = _training_progress.POSITION_DECILE_BALANCED_LOSS_DECILES
TOKEN_EDIT_CORRUPTION_DECILES = _training_progress.TOKEN_EDIT_CORRUPTION_DECILES
TOP1_PROB_HIST_BINS = _training_progress.TOP1_PROB_HIST_BINS


def _grad_norm(parameters) -> float:
    norms = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), 2).item())


def _parameter_norm(parameters) -> float:
    norms = [p.detach().float().norm(2) for p in parameters]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), 2).item())


def _parameter_norm_metrics(raw_model: nn.Module) -> dict[str, float]:
    metrics = {
        "parameter_norm_transformer": _parameter_norm(raw_model.blocks.parameters()),
        "parameter_norm_embedding": _parameter_norm(raw_model.wte.parameters()),
        "parameter_norm_lm_head": _parameter_norm(raw_model.lm_head.parameters()),
        "parameter_norm_length_head": _parameter_norm(
            raw_model.length_head.parameters() if getattr(raw_model, "length_head", None) is not None else []
        ),
        "parameter_norm_target_length_conditioning": _parameter_norm(
            list(raw_model.target_length_embedding.parameters()) + list(raw_model.target_distance_proj.parameters())
            if getattr(raw_model, "target_length_embedding", None) is not None
            and getattr(raw_model, "target_distance_proj", None) is not None
            else []
        ),
        "parameter_norm_mask_position_embedding": _parameter_norm(
            raw_model.mask_position_embedding.parameters()
            if getattr(raw_model, "mask_position_embedding", None) is not None
            else []
        ),
        "parameter_norm_final_norm": _parameter_norm(raw_model.final_norm.parameters()),
        "parameter_norm_class_head": _parameter_norm(
            raw_model.class_head.parameters() if raw_model.class_head is not None else []
        ),
        "parameter_norm_auxiliary_heads": _parameter_norm(raw_model.intermediate_heads.parameters()),
    }
    token_edit_adapters = getattr(raw_model, "token_edit_adapters", None)
    if token_edit_adapters is not None and len(token_edit_adapters) > 0:
        metrics["parameter_norm_token_edit_adapter"] = _parameter_norm(token_edit_adapters.parameters())
    token_edit_provenance_embedding = getattr(raw_model, "token_edit_provenance_embedding", None)
    if token_edit_provenance_embedding is not None:
        metrics["parameter_norm_token_edit_provenance_embedding"] = _parameter_norm(
            token_edit_provenance_embedding.parameters()
        )
    return metrics


def _gradient_component_norm_metrics(raw_model: nn.Module) -> dict[str, float]:
    metrics = {
        "grad_norm_transformer": _grad_norm(raw_model.blocks.parameters()),
        "grad_norm_embedding": _grad_norm(raw_model.wte.parameters()),
        "grad_norm_lm_head": _grad_norm(raw_model.lm_head.parameters()),
        "grad_norm_length_head": _grad_norm(
            raw_model.length_head.parameters() if getattr(raw_model, "length_head", None) is not None else []
        ),
        "grad_norm_target_length_conditioning": _grad_norm(
            list(raw_model.target_length_embedding.parameters()) + list(raw_model.target_distance_proj.parameters())
            if getattr(raw_model, "target_length_embedding", None) is not None
            and getattr(raw_model, "target_distance_proj", None) is not None
            else []
        ),
        "grad_norm_mask_position_embedding": _grad_norm(
            raw_model.mask_position_embedding.parameters()
            if getattr(raw_model, "mask_position_embedding", None) is not None
            else []
        ),
        "grad_norm_final_norm": _grad_norm(raw_model.final_norm.parameters()),
        "grad_norm_class_head": _grad_norm(
            raw_model.class_head.parameters() if raw_model.class_head is not None else []
        ),
        "grad_norm_auxiliary_heads": _grad_norm(raw_model.intermediate_heads.parameters()),
    }
    token_edit_adapters = getattr(raw_model, "token_edit_adapters", None)
    if token_edit_adapters is not None and len(token_edit_adapters) > 0:
        metrics["grad_norm_token_edit_adapter"] = _grad_norm(token_edit_adapters.parameters())
    token_edit_provenance_embedding = getattr(raw_model, "token_edit_provenance_embedding", None)
    if token_edit_provenance_embedding is not None:
        metrics["grad_norm_token_edit_provenance_embedding"] = _grad_norm(token_edit_provenance_embedding.parameters())
    return metrics


def _second_moment_norm(optimizers) -> float:
    total_sq = 0.0
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for p in group.get("params", []):
                state = optimizer.state.get(p)
                if state is not None and "exp_avg_sq" in state:
                    total_sq += float(state["exp_avg_sq"].float().norm(2).item() ** 2)
    return math.sqrt(total_sq) if total_sq > 0 else 0.0


def _update_norm(pre_step_params, post_step_params) -> float:
    total_sq = 0.0
    for pre, post in zip(pre_step_params, post_step_params, strict=True):
        total_sq += float((pre.float() - post.detach().float()).norm(2).item() ** 2)
    return math.sqrt(total_sq) if total_sq > 0 else 0.0


def _component_update_norm_metrics(
    pre_step_components: dict[str, list[torch.Tensor]] | None,
    raw_model: nn.Module,
) -> dict[str, float]:
    metrics = {
        "update_norm_embedding": 0.0,
        "update_norm_lm_head": 0.0,
    }
    if pre_step_components is None:
        return metrics
    component_params = {
        "embedding": list(raw_model.wte.parameters()),
        "lm_head": list(raw_model.lm_head.parameters()),
    }
    token_edit_adapters = getattr(raw_model, "token_edit_adapters", None)
    if token_edit_adapters is not None and len(token_edit_adapters) > 0:
        component_params["token_edit_adapter"] = list(token_edit_adapters.parameters())
    token_edit_provenance_embedding = getattr(raw_model, "token_edit_provenance_embedding", None)
    if token_edit_provenance_embedding is not None:
        component_params["token_edit_provenance_embedding"] = list(token_edit_provenance_embedding.parameters())
    for name, post_params in component_params.items():
        pre_params = pre_step_components.get(name)
        if pre_params is None:
            continue
        metrics[f"update_norm_{name}"] = _update_norm(pre_params, post_params)
    return metrics


def _component_update_grad_ratio_metrics(
    update_metrics: dict[str, float],
    grad_metrics: dict[str, float],
) -> dict[str, float]:
    return {
        "update_grad_ratio_embedding": update_metrics.get("update_norm_embedding", 0.0)
        / max(grad_metrics.get("grad_norm_embedding", 0.0), 1e-8),
        "update_grad_ratio_lm_head": update_metrics.get("update_norm_lm_head", 0.0)
        / max(grad_metrics.get("grad_norm_lm_head", 0.0), 1e-8),
    }


def _residual_projection_init_metrics(config: ExperimentConfig) -> dict[str, float | str]:
    return {
        "residual_output_projection_init": config.model.residual_output_projection_init,
        "residual_output_projection_init_std": (
            0.0
            if config.model.backbone_profile == "nanollada_v1"
            else residual_output_projection_init_std(int(config.model.depth))
        ),
    }


def _embedding_head_init_metrics(config: ExperimentConfig) -> dict[str, float | None]:
    if config.model.backbone_profile == "nanollada_v1":
        return {
            "embedding_head_init_std_base": None,
            "embedding_head_init_std_multiplier": None,
            "embedding_init_std": 1.0,
            "lm_head_init_std": 0.001,
        }
    base_std = embedding_head_init_std(int(config.model.n_embd))
    multiplier = float(config.model.embedding_head_init_std_multiplier)
    return {
        "embedding_head_init_std_base": base_std,
        "embedding_head_init_std_multiplier": multiplier,
        "embedding_init_std": embedding_head_init_std(int(config.model.n_embd), multiplier),
        "lm_head_init_std": embedding_head_init_std(int(config.model.n_embd), multiplier),
    }


def _embedding_head_init_diagnostics_enabled(config: ExperimentConfig) -> bool:
    return not math.isclose(float(config.model.embedding_head_init_std_multiplier), 1.0)


def _prefixed_metric_keys(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _tensor_rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.sqrt(value.detach().float().pow(2).mean()).item())


def _logit_confidence_statistics(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=-1).float()
    probs = log_probs.exp()
    entropy_values = -(probs * log_probs).sum(dim=-1)
    max_probs, argmax_ids = probs.max(dim=-1)
    raw_entropy = entropy_values.mean()
    raw_max_prob = max_probs.mean()
    if argmax_ids.numel() > 0:
        mode_val = torch.mode(argmax_ids).values
        raw_argmax_mode_frac = (argmax_ids == mode_val).float().mean()
    else:
        raw_argmax_mode_frac = torch.tensor(0.0, device=logits.device)
    return entropy_values, max_probs, raw_entropy, raw_max_prob, raw_argmax_mode_frac


def _logit_scale_statistics(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_f = logits.float()
    logsumexp_values = torch.logsumexp(logits_f, dim=-1)
    z_loss_values = logsumexp_values.square()
    logit_norms = logits_f.norm(dim=-1)
    logit_rms = torch.sqrt(logits_f.square().mean(dim=-1))
    return z_loss_values, logsumexp_values, logit_norms, logit_rms


def _accumulate_probability_histogram(counts: list[int], values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    bins = len(counts)
    bucket_ids = torch.clamp((values.detach().float() * bins).to(torch.long), min=0, max=bins - 1)
    hist = torch.bincount(bucket_ids.cpu(), minlength=bins)
    for idx, count in enumerate(hist.tolist()[:bins]):
        counts[idx] += int(count)


def _probability_histogram_metrics(prefix: str, counts: list[int]) -> dict[str, float | int]:
    total = int(sum(counts))
    metrics: dict[str, float | int] = {f"{prefix}_count": total}
    bins = len(counts)
    for idx, count in enumerate(counts):
        low = int(100 * idx / bins)
        high = int(100 * (idx + 1) / bins)
        label = f"{low:02d}_{high:03d}" if high == 100 else f"{low:02d}_{high:02d}"
        metrics[f"{prefix}_bin_{label}_count"] = int(count)
        metrics[f"{prefix}_bin_{label}_frac"] = float(count) / max(1, total)
    return metrics


def _accumulate_value_histogram(
    counts: list[int],
    values: torch.Tensor,
    min_value: float,
    max_value: float,
) -> None:
    if values.numel() == 0:
        return
    bins = len(counts)
    if bins <= 0 or max_value <= min_value:
        raise ValueError("histogram requires positive bins and max_value > min_value")
    values_f = torch.nan_to_num(
        values.detach().float(),
        nan=float(min_value),
        posinf=float(max_value),
        neginf=float(min_value),
    )
    scaled = (values_f - float(min_value)) / (float(max_value) - float(min_value))
    bucket_ids = torch.clamp((scaled * bins).to(torch.long), min=0, max=bins - 1)
    hist = torch.bincount(bucket_ids.cpu(), minlength=bins)
    for idx, count in enumerate(hist.tolist()[:bins]):
        counts[idx] += int(count)


def _histogram_edge_label(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value))).replace("-", "m")
    return f"{value:.2f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _value_histogram_metrics(
    prefix: str,
    counts: list[int],
    min_value: float,
    max_value: float,
) -> dict[str, float | int]:
    total = int(sum(counts))
    bins = len(counts)
    metrics: dict[str, float | int] = {
        f"{prefix}_count": total,
        f"{prefix}_range_min": float(min_value),
        f"{prefix}_range_max": float(max_value),
    }
    if bins <= 0:
        return metrics
    width = (float(max_value) - float(min_value)) / bins
    for idx, count in enumerate(counts):
        low = float(min_value) + idx * width
        high = float(min_value) + (idx + 1) * width
        label = f"{_histogram_edge_label(low)}_{_histogram_edge_label(high)}"
        metrics[f"{prefix}_bin_{label}_count"] = int(count)
        metrics[f"{prefix}_bin_{label}_frac"] = float(count) / max(1, total)
    return metrics


def _per_token_effective_masked_fraction_metrics(
    prefix: str,
    fraction_sum: float,
    fraction_count: int,
    fraction_min: float,
    fraction_max: float,
) -> dict[str, float | int]:
    return {
        f"{prefix}per_token_noise_effective_masked_fraction_count": int(fraction_count),
        f"{prefix}per_token_noise_effective_masked_fraction_mean": (
            float(fraction_sum) / int(fraction_count) if fraction_count else 0.0
        ),
        f"{prefix}per_token_noise_effective_masked_fraction_min": (
            float(fraction_min) if fraction_count and math.isfinite(fraction_min) else 0.0
        ),
        f"{prefix}per_token_noise_effective_masked_fraction_max": (
            float(fraction_max) if fraction_count and math.isfinite(fraction_max) else 0.0
        ),
    }


def _per_target_t_values(t_val: torch.Tensor, x_0: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    if t_val.shape == x_0.shape:
        return t_val.reshape(-1).index_select(0, target_indices).float()
    if t_val.ndim == 1 and t_val.shape[0] == x_0.shape[0]:
        target_rows = torch.div(target_indices, x_0.shape[1], rounding_mode="floor")
        return t_val.index_select(0, target_rows).float()
    raise ValueError(
        "tensor t must be either per-token with shape "
        f"{tuple(x_0.shape)} or per-row with shape ({x_0.shape[0]},), got {tuple(t_val.shape)}"
    )


def _weighted_token_mean(
    values: torch.Tensor,
    token_weights: torch.Tensor,
    denominator: int,
) -> torch.Tensor:
    """Return a weighted token sum normalized by one explicit population denominator."""

    if values.shape != token_weights.shape:
        raise ValueError(f"token weights must match values, got {token_weights.shape} and {values.shape}")
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("weighted token mean denominator must be a positive integer")
    return (values.float() * token_weights.float()).sum() / denominator


def _new_token_edit_reconstruction_accumulator(
    num_deciles: int = TOKEN_EDIT_CORRUPTION_DECILES,
) -> dict[str, list[int] | int]:
    return {
        "counts": [0 for _ in range(int(num_deciles) + 1)],
        "correct": [0 for _ in range(int(num_deciles) + 1)],
        "microbatches": 0,
    }


def _update_token_edit_reconstruction_accumulator(
    accumulator: dict[str, list[int] | int],
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_class_ids: torch.Tensor,
    world_size: int = 1,
) -> None:
    if logits.numel() == 0 or targets.numel() == 0:
        return
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(f"token-edit logits/targets mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    class_ids = token_class_ids.to(device=targets.device).index_select(0, targets.detach().to(torch.long))
    class_ids = class_ids.clamp(0, TOKEN_EDIT_CORRUPTION_DECILES)
    pred = logits.detach().argmax(dim=-1)
    correct = (pred == targets).to(torch.long)
    counts = accumulator["counts"]
    correct_counts = accumulator["correct"]
    if not isinstance(counts, list) or not isinstance(correct_counts, list):
        raise ValueError("token-edit reconstruction accumulator is malformed")
    for decile in range(TOKEN_EDIT_CORRUPTION_DECILES + 1):
        mask = class_ids == decile
        count = int(mask.sum().item()) * int(world_size)
        if count == 0:
            continue
        counts[decile] += count
        correct_counts[decile] += int(correct[mask].sum().item()) * int(world_size)


def _merge_token_edit_reconstruction_accumulator(
    dest: dict[str, list[int] | int],
    src: dict[str, list[int] | int],
) -> None:
    dest_counts = dest["counts"]
    dest_correct = dest["correct"]
    src_counts = src["counts"]
    src_correct = src["correct"]
    if not all(isinstance(values, list) for values in (dest_counts, dest_correct, src_counts, src_correct)):
        raise ValueError("token-edit reconstruction accumulator is malformed")
    for idx in range(min(len(dest_counts), len(src_counts))):
        dest_counts[idx] += int(src_counts[idx])
        dest_correct[idx] += int(src_correct[idx])
    dest["microbatches"] = int(dest["microbatches"]) + int(src["microbatches"])


def _token_edit_reconstruction_metrics(
    prefix: str,
    accumulator: dict[str, list[int] | int],
    enabled: bool,
) -> dict[str, float | int | bool]:
    counts = [int(v) for v in accumulator["counts"]] if isinstance(accumulator["counts"], list) else []
    correct = [int(v) for v in accumulator["correct"]] if isinstance(accumulator["correct"], list) else []
    total = sum(counts)
    total_correct = sum(correct)
    metrics: dict[str, float | int | bool] = {
        f"{prefix}t2t_corrupted_token_reconstruction_enabled": bool(enabled),
        f"{prefix}t2t_corrupted_token_reconstruction_deciles": TOKEN_EDIT_CORRUPTION_DECILES,
        f"{prefix}t2t_corrupted_token_reconstruction_microbatches": int(accumulator["microbatches"]),
        f"{prefix}t2t_corrupted_token_reconstruction_count": int(total),
        f"{prefix}t2t_corrupted_token_reconstruction_correct": int(total_correct),
        f"{prefix}t2t_corrupted_token_reconstruction_accuracy": total_correct / max(1, total),
    }
    if counts:
        special_count = counts[0]
        special_correct = correct[0]
        metrics[f"{prefix}t2t_corrupted_token_reconstruction_special_count"] = special_count
        metrics[f"{prefix}t2t_corrupted_token_reconstruction_special_accuracy"] = special_correct / max(
            1, special_count
        )
    for decile in range(1, TOKEN_EDIT_CORRUPTION_DECILES + 1):
        count = counts[decile] if decile < len(counts) else 0
        correct_count = correct[decile] if decile < len(correct) else 0
        metrics[f"{prefix}t2t_corrupted_token_reconstruction_decile_{decile}_count"] = int(count)
        metrics[f"{prefix}t2t_corrupted_token_reconstruction_decile_{decile}_correct"] = int(correct_count)
        metrics[f"{prefix}t2t_corrupted_token_reconstruction_decile_{decile}_accuracy"] = correct_count / max(1, count)
    return metrics


def _relative_position_decile_ids(
    flat_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    num_deciles: int,
) -> torch.Tensor:
    if flat_indices.numel() == 0:
        return torch.empty_like(flat_indices, dtype=torch.long)
    seq_len = int(attention_mask.shape[1])
    rows = torch.div(flat_indices.to(torch.long), seq_len, rounding_mode="floor")
    cols = flat_indices.to(torch.long) - rows * seq_len
    valid_lengths = attention_mask.to(torch.long).sum(dim=1).clamp_min(1)
    denom = valid_lengths.index_select(0, rows).sub(1).clamp_min(1)
    relative = cols.float() / denom.float()
    return torch.clamp((relative * int(num_deciles)).to(torch.long), min=0, max=int(num_deciles) - 1)


def _position_decile_balanced_ce_loss(
    ce_values: torch.Tensor,
    target_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    token_weights: torch.Tensor,
    num_deciles: int = POSITION_DECILE_BALANCED_LOSS_DECILES,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if ce_values.shape != target_indices.shape:
        raise ValueError(
            f"CE values and target indices must have the same shape, got {ce_values.shape} and {target_indices.shape}"
        )
    if token_weights.shape != ce_values.shape:
        raise ValueError(f"token_weights must match CE values, got {token_weights.shape} and {ce_values.shape}")
    if loss_mask.shape != attention_mask.shape:
        raise ValueError(f"loss_mask must match attention_mask, got {loss_mask.shape} and {attention_mask.shape}")
    device = ce_values.device
    num_deciles = int(num_deciles)
    active_counts = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    lossable_counts = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    raw_ce_sums = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    weighted_ce_sums = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    balanced_contributions = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    lossable_indices = (loss_mask & attention_mask).reshape(-1).nonzero(as_tuple=True)[0]
    if lossable_indices.numel() > 0:
        lossable_decile_ids = _relative_position_decile_ids(lossable_indices, attention_mask, num_deciles)
        lossable_counts.scatter_add_(
            0,
            lossable_decile_ids,
            torch.ones_like(lossable_decile_ids, dtype=torch.float32),
        )
    if ce_values.numel() == 0:
        return ce_values.float().sum() * 0.0, {
            "active_counts": active_counts,
            "lossable_counts": lossable_counts,
            "raw_ce_sums": raw_ce_sums,
            "weighted_ce_sums": weighted_ce_sums,
            "balanced_contributions": balanced_contributions,
            "loss": torch.tensor(0.0, device=device),
        }

    decile_ids = _relative_position_decile_ids(target_indices, attention_mask, num_deciles)
    raw_values = ce_values.float()
    weighted_values = raw_values * token_weights.float()
    ones = torch.ones_like(raw_values, dtype=torch.float32)
    active_counts.scatter_add_(0, decile_ids, ones)
    raw_ce_sums.scatter_add_(0, decile_ids, raw_values)
    weighted_ce_sums.scatter_add_(0, decile_ids, weighted_values)
    lossable_deciles = lossable_counts > 0
    decile_losses = weighted_ce_sums / lossable_counts.clamp_min(1.0)
    nonempty_count = lossable_deciles.float().sum().clamp_min(1.0)
    balanced_contributions = torch.where(
        lossable_deciles,
        decile_losses / nonempty_count,
        balanced_contributions,
    )
    loss = balanced_contributions.sum()
    return loss, {
        "active_counts": active_counts.detach(),
        "lossable_counts": lossable_counts.detach(),
        "raw_ce_sums": raw_ce_sums.detach(),
        "weighted_ce_sums": weighted_ce_sums.detach(),
        "balanced_contributions": balanced_contributions.detach(),
        "loss": loss.detach(),
    }


def _new_position_decile_ce_accumulator(
    num_deciles: int = POSITION_DECILE_BALANCED_LOSS_DECILES,
) -> dict[str, list[float] | float | int]:
    return {
        "active_counts": [0.0 for _ in range(int(num_deciles))],
        "lossable_counts": [0.0 for _ in range(int(num_deciles))],
        "raw_ce_sums": [0.0 for _ in range(int(num_deciles))],
        "weighted_ce_sums": [0.0 for _ in range(int(num_deciles))],
        "balanced_contributions": [0.0 for _ in range(int(num_deciles))],
        "loss_sum": 0.0,
        "microbatches": 0,
    }


def _update_position_decile_ce_accumulator(
    accumulator: dict[str, list[float] | float | int],
    stats: dict[str, torch.Tensor],
    world_size: int = 1,
) -> None:
    for key in ("active_counts", "lossable_counts", "raw_ce_sums", "weighted_ce_sums", "balanced_contributions"):
        values = stats[key].detach().float().tolist()
        accumulator_values = accumulator[key]
        if not isinstance(accumulator_values, list):
            raise ValueError(f"position decile accumulator field {key!r} is not a list")
        for idx, value in enumerate(values):
            accumulator_values[idx] += float(value) * int(world_size)
    accumulator["loss_sum"] = float(accumulator["loss_sum"]) + float(stats["loss"].detach().item()) * int(world_size)
    accumulator["microbatches"] = int(accumulator["microbatches"]) + int(world_size)


def _position_decile_ce_accumulator_metrics(
    prefix: str,
    accumulator: dict[str, list[float] | float | int],
    enabled: bool,
) -> dict[str, float | int | bool]:
    active_counts = (
        [float(v) for v in accumulator["active_counts"]] if isinstance(accumulator["active_counts"], list) else []
    )
    lossable_counts = (
        [float(v) for v in accumulator["lossable_counts"]] if isinstance(accumulator["lossable_counts"], list) else []
    )
    raw_ce_sums = [float(v) for v in accumulator["raw_ce_sums"]] if isinstance(accumulator["raw_ce_sums"], list) else []
    weighted_ce_sums = (
        [float(v) for v in accumulator["weighted_ce_sums"]] if isinstance(accumulator["weighted_ce_sums"], list) else []
    )
    balanced_contributions = (
        [float(v) for v in accumulator["balanced_contributions"]]
        if isinstance(accumulator["balanced_contributions"], list)
        else []
    )
    total_active_count = sum(active_counts)
    total_lossable_count = sum(lossable_counts)
    microbatches = int(accumulator["microbatches"])
    metrics: dict[str, float | int | bool] = {
        f"{prefix}position_decile_balanced_loss_enabled": bool(enabled),
        f"{prefix}position_decile_balanced_loss_deciles": len(active_counts),
        f"{prefix}position_decile_balanced_microbatches": microbatches,
        f"{prefix}position_decile_balanced_loss": float(accumulator["loss_sum"]) / max(1, microbatches),
        f"{prefix}position_decile_active_tokens": int(total_active_count),
        f"{prefix}position_decile_lossable_tokens": int(total_lossable_count),
    }
    for decile_idx, active_count in enumerate(active_counts):
        label = f"{decile_idx:02d}"
        lossable_count = lossable_counts[decile_idx]
        raw_mean = raw_ce_sums[decile_idx] / active_count if active_count else 0.0
        weighted_mean = weighted_ce_sums[decile_idx] / active_count if active_count else 0.0
        weighted_per_lossable = weighted_ce_sums[decile_idx] / lossable_count if lossable_count else 0.0
        contribution = balanced_contributions[decile_idx] / max(1, microbatches)
        metrics[f"{prefix}position_decile_{label}_active_tokens"] = int(active_count)
        metrics[f"{prefix}position_decile_{label}_lossable_tokens"] = int(lossable_count)
        metrics[f"{prefix}position_decile_{label}_active_frac"] = active_count / max(1.0, total_active_count)
        metrics[f"{prefix}position_decile_{label}_lossable_frac"] = lossable_count / max(1.0, total_lossable_count)
        metrics[f"{prefix}position_decile_{label}_ce_before_weighting"] = raw_mean
        metrics[f"{prefix}position_decile_{label}_ce_after_schedule_weighting"] = weighted_mean
        metrics[f"{prefix}position_decile_{label}_schedule_weighted_ce_per_lossable_token"] = weighted_per_lossable
        metrics[f"{prefix}position_decile_{label}_balanced_loss_contribution"] = contribution
    return metrics


def _eos_hazard_calibration_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None,
    num_deciles: int = EOS_HAZARD_RELATIVE_POSITION_DECILES,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = logits.device
    empty = {
        "pred_sums": torch.zeros(num_deciles, device=device, dtype=torch.float32),
        "target_sums": torch.zeros(num_deciles, device=device, dtype=torch.float32),
        "counts": torch.zeros(num_deciles, device=device, dtype=torch.float32),
    }
    if eos_token_id is None or logits.numel() == 0 or target_indices.numel() == 0:
        return torch.tensor(0.0, device=device), empty
    eos_id = int(eos_token_id)
    if eos_id < 0 or eos_id >= logits.shape[-1]:
        return torch.tensor(0.0, device=device), empty

    eos_probs = F.softmax(logits.float(), dim=-1)[:, eos_id]
    target_eos = (targets.to(torch.long) == eos_id).to(torch.float32)
    decile_ids = _relative_position_decile_ids(target_indices, attention_mask, int(num_deciles))
    counts = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    pred_sums = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    target_sums = torch.zeros(num_deciles, device=device, dtype=torch.float32)
    ones = torch.ones_like(eos_probs, dtype=torch.float32)
    counts.scatter_add_(0, decile_ids, ones)
    pred_sums.scatter_add_(0, decile_ids, eos_probs)
    target_sums.scatter_add_(0, decile_ids, target_eos)
    nonempty = counts > 0
    pred_means = pred_sums / counts.clamp_min(1.0)
    target_means = target_sums / counts.clamp_min(1.0)
    loss = (pred_means[nonempty] - target_means[nonempty]).pow(2).mean() if nonempty.any() else eos_probs.sum() * 0.0
    return loss, {"pred_sums": pred_sums, "target_sums": target_sums, "counts": counts}


def _new_eos_hazard_accumulator(num_deciles: int = EOS_HAZARD_RELATIVE_POSITION_DECILES) -> dict[str, list[float]]:
    return {
        "pred_sums": [0.0 for _ in range(int(num_deciles))],
        "target_sums": [0.0 for _ in range(int(num_deciles))],
        "counts": [0.0 for _ in range(int(num_deciles))],
    }


def _update_eos_hazard_accumulator(
    accumulator: dict[str, list[float]],
    stats: dict[str, torch.Tensor],
    world_size: int = 1,
) -> None:
    for key in ("pred_sums", "target_sums", "counts"):
        values = stats.get(key)
        if values is None:
            continue
        detached = values.detach().float().cpu().tolist()
        for idx, value in enumerate(detached):
            accumulator[key][idx] += float(value) * int(world_size)


def _eos_hazard_accumulator_metrics(
    prefix: str,
    accumulator: dict[str, list[float]],
    calibration_loss: float,
    weighted_loss: float,
    weight: float,
) -> dict[str, float | int]:
    pred_sums = accumulator["pred_sums"]
    target_sums = accumulator["target_sums"]
    counts = accumulator["counts"]
    total_count = float(sum(counts))
    total_pred = float(sum(pred_sums))
    total_target = float(sum(target_sums))
    pred_mean = total_pred / total_count if total_count else 0.0
    target_mean = total_target / total_count if total_count else 0.0
    metrics: dict[str, float | int] = {
        f"{prefix}eos_hazard_loss_weight": float(weight),
        f"{prefix}eos_hazard_calibration_loss": float(calibration_loss),
        f"{prefix}eos_hazard_loss": float(weighted_loss),
        f"{prefix}eos_hazard_token_count": int(total_count),
        f"{prefix}eos_hazard_pred_prob": pred_mean,
        f"{prefix}eos_hazard_empirical_freq": target_mean,
        f"{prefix}eos_hazard_abs_error": abs(pred_mean - target_mean),
    }
    for decile_idx, count in enumerate(counts):
        pred = pred_sums[decile_idx] / count if count else 0.0
        target = target_sums[decile_idx] / count if count else 0.0
        label = f"{decile_idx:02d}"
        metrics[f"{prefix}eos_hazard_decile_{label}_count"] = int(count)
        metrics[f"{prefix}eos_hazard_decile_{label}_pred_prob"] = pred
        metrics[f"{prefix}eos_hazard_decile_{label}_empirical_freq"] = target
        metrics[f"{prefix}eos_hazard_decile_{label}_abs_error"] = abs(pred - target)
    return metrics


def _length_head_targets(
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x_0.shape != attention_mask.shape:
        raise ValueError(f"x_0 and attention_mask shapes must match, got {x_0.shape} and {attention_mask.shape}")
    batch_size, seq_len = x_0.shape
    valid_rows = attention_mask.any(dim=1)
    valid_lengths = attention_mask.to(torch.long).sum(dim=1).clamp_min(1)
    if eos_token_id is None:
        return (valid_lengths - 1).to(torch.long), valid_rows

    positions = torch.arange(seq_len, device=x_0.device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
    eos_hits = (x_0 == int(eos_token_id)) & attention_mask
    first_eos = torch.where(eos_hits, positions, torch.full_like(positions, seq_len)).min(dim=1).values
    targets = torch.where(first_eos < seq_len, first_eos, valid_lengths - 1)
    return targets.to(torch.long), valid_rows


def _length_head_loss_and_stats(
    length_logits: torch.Tensor,
    x_0: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets, valid_rows = _length_head_targets(x_0, attention_mask, eos_token_id)
    device = length_logits.device
    seq_len = int(x_0.shape[1])
    if length_logits.shape != (x_0.shape[0], seq_len):
        raise ValueError(f"length_logits must have shape {(x_0.shape[0], seq_len)}, got {tuple(length_logits.shape)}")
    empty_stats = {
        "count": torch.tensor(0.0, device=device),
        "ce": torch.tensor(0.0, device=device),
        "accuracy": torch.tensor(0.0, device=device),
        "mae": torch.tensor(0.0, device=device),
        "confidence": torch.tensor(0.0, device=device),
        "calibration_abs_error": torch.tensor(0.0, device=device),
        "target_counts": torch.zeros(seq_len, device=device, dtype=torch.float32),
        "pred_counts": torch.zeros(seq_len, device=device, dtype=torch.float32),
    }
    if not bool(valid_rows.any().item()):
        return length_logits.float().sum() * 0.0, empty_stats

    logits = length_logits[valid_rows].float()
    target = targets[valid_rows]
    ce = F.cross_entropy(logits, target, reduction="mean")
    pred = logits.argmax(dim=-1)
    correct = (pred == target).float()
    mae = (pred - target).abs().float().mean()
    probs = F.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values.mean()
    accuracy = correct.mean()
    target_counts = torch.bincount(target, minlength=seq_len).to(device=device, dtype=torch.float32)
    pred_counts = torch.bincount(pred, minlength=seq_len).to(device=device, dtype=torch.float32)
    return ce, {
        "count": torch.tensor(float(target.numel()), device=device),
        "ce": ce.detach(),
        "accuracy": accuracy.detach(),
        "mae": mae.detach(),
        "confidence": confidence.detach(),
        "calibration_abs_error": (confidence - accuracy).abs().detach(),
        "target_counts": target_counts.detach(),
        "pred_counts": pred_counts.detach(),
    }


def _new_length_head_accumulator(seq_len: int) -> dict[str, object]:
    return {
        "microbatches": 0,
        "count": 0.0,
        "ce_sum": 0.0,
        "accuracy_sum": 0.0,
        "mae_sum": 0.0,
        "confidence_sum": 0.0,
        "calibration_abs_error_sum": 0.0,
        "target_counts": [0.0 for _ in range(int(seq_len))],
        "pred_counts": [0.0 for _ in range(int(seq_len))],
    }


def _update_length_head_accumulator(
    accumulator: dict[str, object],
    stats: dict[str, torch.Tensor],
    world_size: int = 1,
) -> None:
    count = float(stats["count"].detach().item()) * int(world_size)
    if count <= 0:
        return
    accumulator["microbatches"] = int(accumulator["microbatches"]) + int(world_size)
    accumulator["count"] = float(accumulator["count"]) + count
    for key, stat_key in (
        ("ce_sum", "ce"),
        ("accuracy_sum", "accuracy"),
        ("mae_sum", "mae"),
        ("confidence_sum", "confidence"),
        ("calibration_abs_error_sum", "calibration_abs_error"),
    ):
        accumulator[key] = float(accumulator[key]) + float(stats[stat_key].detach().item()) * int(world_size)
    for key in ("target_counts", "pred_counts"):
        values = accumulator[key]
        if not isinstance(values, list):
            raise ValueError(f"length-head accumulator field {key!r} is not a list")
        for idx, value in enumerate(stats[key].detach().float().cpu().tolist()):
            values[idx] += float(value) * int(world_size)


def _constant_prior_from_counts(counts: list[float], mode: str) -> int:
    if not counts:
        return 0
    if mode == "mode":
        return max(range(len(counts)), key=lambda idx: (counts[idx], -idx))
    if mode == "median":
        total = float(sum(counts))
        if total <= 0:
            return 0
        running = 0.0
        midpoint = 0.5 * total
        for idx, count in enumerate(counts):
            running += float(count)
            if running >= midpoint:
                return idx
        return len(counts) - 1
    raise ValueError(f"unknown prior mode: {mode!r}")


def _length_head_accumulator_metrics(
    prefix: str,
    accumulator: dict[str, object],
    weight: float,
    prior_counts: list[float] | None = None,
) -> dict[str, float | int | bool]:
    microbatches = int(accumulator["microbatches"])
    count = float(accumulator["count"])
    target_counts = (
        [float(v) for v in accumulator["target_counts"]] if isinstance(accumulator["target_counts"], list) else []
    )
    pred_counts = [float(v) for v in accumulator["pred_counts"]] if isinstance(accumulator["pred_counts"], list) else []
    total_targets = float(sum(target_counts))
    metrics: dict[str, float | int | bool] = {
        f"{prefix}length_head_enabled": bool(weight > 0),
        f"{prefix}length_head_loss_weight": float(weight),
        f"{prefix}length_head_microbatches": microbatches,
        f"{prefix}length_head_count": int(count),
        f"{prefix}length_head_ce": float(accumulator["ce_sum"]) / max(1, microbatches),
        f"{prefix}length_head_loss": float(weight) * float(accumulator["ce_sum"]) / max(1, microbatches),
        f"{prefix}length_head_accuracy": float(accumulator["accuracy_sum"]) / max(1, microbatches),
        f"{prefix}length_head_mae": float(accumulator["mae_sum"]) / max(1, microbatches),
        f"{prefix}length_head_confidence": float(accumulator["confidence_sum"]) / max(1, microbatches),
        f"{prefix}length_head_calibration_abs_error": float(accumulator["calibration_abs_error_sum"])
        / max(1, microbatches),
    }
    if target_counts:
        target_mean = sum(idx * count_value for idx, count_value in enumerate(target_counts)) / max(1.0, total_targets)
        pred_total = float(sum(pred_counts))
        pred_mean = sum(idx * count_value for idx, count_value in enumerate(pred_counts)) / max(1.0, pred_total)
        metrics[f"{prefix}length_head_target_position_mean"] = target_mean
        metrics[f"{prefix}length_head_predicted_position_mean"] = pred_mean
        metrics[f"{prefix}length_head_target_length_mean"] = target_mean + 1.0
        metrics[f"{prefix}length_head_predicted_length_mean"] = pred_mean + 1.0
        metrics[f"{prefix}length_head_target_hist_nonzero_bins"] = sum(1 for value in target_counts if value > 0)
        metrics[f"{prefix}length_head_pred_hist_nonzero_bins"] = sum(1 for value in pred_counts if value > 0)
        for idx, (target_count, pred_count) in enumerate(zip(target_counts, pred_counts, strict=True)):
            if target_count or pred_count:
                metrics[f"{prefix}length_head_target_position_{idx:04d}_count"] = int(target_count)
                metrics[f"{prefix}length_head_predicted_position_{idx:04d}_count"] = int(pred_count)
    if prior_counts is not None:
        prior_counts_f = [float(v) for v in prior_counts]
        prior_mode = _constant_prior_from_counts(prior_counts_f, "mode")
        prior_median = _constant_prior_from_counts(prior_counts_f, "median")
        total = max(1.0, total_targets)
        target_mode_hits = target_counts[prior_mode] if prior_mode < len(target_counts) else 0.0
        prior_mae = sum(abs(idx - prior_median) * count_value for idx, count_value in enumerate(target_counts)) / total
        metrics[f"{prefix}length_head_prior_mode_position"] = int(prior_mode)
        metrics[f"{prefix}length_head_prior_median_position"] = int(prior_median)
        metrics[f"{prefix}length_head_prior_accuracy"] = target_mode_hits / total
        metrics[f"{prefix}length_head_prior_mae"] = prior_mae
        metrics[f"{prefix}length_head_beats_prior_accuracy"] = (
            metrics[f"{prefix}length_head_accuracy"] > metrics[f"{prefix}length_head_prior_accuracy"]
        )
        metrics[f"{prefix}length_head_beats_prior_mae"] = (
            metrics[f"{prefix}length_head_mae"] < metrics[f"{prefix}length_head_prior_mae"]
        )
    return metrics


def _evaluate_length_head_metrics(
    raw_model: nn.Module,
    config: ExperimentConfig,
    tokenizer,
    device,
    autocast_ctx,
    train_prior_counts: list[float] | None = None,
) -> dict[str, float | int | bool]:
    if not _uses_length_head(config):
        return {}
    val_loader = build_val_loader(config, tokenizer, device)
    accumulator = _new_length_head_accumulator(int(config.model.max_seq_len))
    eos_token_id = tokenizer.get_eos_token_id()
    batches = int(config.eval.fixed_ce_batches)
    for _ in range(batches):
        batch = next(val_loader)
        z_eval, _w_eval, _t_val, noise_level = _corrupt_eval_batch(
            config,
            build_noise(config.noise, int(tokenizer.get_mask_token_id()), int(tokenizer.get_vocab_size())),
            batch.x_0,
            batch.loss_mask,
            t_val=0.9,
        )
        with autocast_ctx:
            _logits, length_logits = raw_model(
                z_eval,
                attention_mask=batch.attention_mask,
                noise_level=noise_level if config.noise.time_conditioning else None,
                return_length_logits=True,
            )
        _loss, stats = _length_head_loss_and_stats(length_logits, batch.x_0, batch.attention_mask, eos_token_id)
        _update_length_head_accumulator(accumulator, stats)
    return _length_head_accumulator_metrics(
        "val_",
        accumulator,
        float(config.train.length_head_loss_weight),
        prior_counts=train_prior_counts,
    )


def _summarize_metric_values(metrics: dict[str, float], prefix: str, values: list[float]) -> None:
    if not values:
        return
    metrics[f"{prefix}mean"] = float(sum(values) / len(values))
    metrics[f"{prefix}min"] = float(min(values))
    metrics[f"{prefix}max"] = float(max(values))


def _residual_branch_rms_metrics(
    raw_model: DiffusionTransformer,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None = None,
) -> dict[str, float]:
    b, t = tokens.shape
    if position_ids is None:
        cos = raw_model.rope_cos[:t]
        sin = raw_model.rope_sin[:t]
    else:
        position_ids = position_ids.to(device=tokens.device, dtype=torch.long)
        if position_ids.dim() == 1:
            flat_positions = position_ids
            shape = (t, 1, -1)
        elif position_ids.dim() == 2:
            flat_positions = position_ids.reshape(-1)
            shape = (b, t, 1, -1)
        else:
            raise ValueError("position_ids must be a 1D or 2D tensor")
        cos = raw_model.rope_cos.index_select(0, flat_positions).view(shape)
        sin = raw_model.rope_sin.index_select(0, flat_positions).view(shape)

    x = raw_model.wte(tokens).to(dtype=raw_model.compute_dtype)
    if raw_model.noise_mlp is not None:
        noise_level = (tokens == int(raw_model.config.mask_token_id)).float().mean(dim=1)
        noise_level = noise_level.to(device=tokens.device, dtype=raw_model.compute_dtype)
        x = x + raw_model.noise_mlp(noise_level[:, None])[:, None, :]

    metrics: dict[str, float] = {}
    attn_values: list[float] = []
    mlp_values: list[float] = []
    attn_contribution_values: list[float] = []
    mlp_contribution_values: list[float] = []
    activation_values: list[float] = []
    metrics["activation_embedding_rms"] = _tensor_rms(x)
    for layer_idx, block in enumerate(raw_model.blocks):
        attn_out = block.attn(rms_norm(x), cos, sin, attention_mask=attention_mask)
        attn_rms = _tensor_rms(attn_out)
        residual_multiplier = float(block.residual_multiplier)
        attn_contribution = attn_out if residual_multiplier == 1.0 else residual_multiplier * attn_out
        attn_contribution_rms = _tensor_rms(attn_contribution)
        x = x + attn_contribution
        mlp_out = block.mlp(rms_norm(x))
        mlp_rms = _tensor_rms(mlp_out)
        mlp_contribution = mlp_out if residual_multiplier == 1.0 else residual_multiplier * mlp_out
        mlp_contribution_rms = _tensor_rms(mlp_contribution)
        x = x + mlp_contribution
        activation_rms = _tensor_rms(x)
        metrics[f"residual_branch_layer_{layer_idx}_attn_rms"] = attn_rms
        metrics[f"residual_branch_layer_{layer_idx}_mlp_rms"] = mlp_rms
        metrics[f"residual_contribution_layer_{layer_idx}_attn_rms"] = attn_contribution_rms
        metrics[f"residual_contribution_layer_{layer_idx}_mlp_rms"] = mlp_contribution_rms
        metrics[f"activation_layer_{layer_idx}_rms"] = activation_rms
        attn_values.append(attn_rms)
        mlp_values.append(mlp_rms)
        attn_contribution_values.append(attn_contribution_rms)
        mlp_contribution_values.append(mlp_contribution_rms)
        activation_values.append(activation_rms)
    if raw_model.blocks:
        metrics["parameterization_residual_multiplier"] = float(raw_model.blocks[0].residual_multiplier)
    _summarize_metric_values(metrics, "residual_branch_attn_rms_", attn_values)
    _summarize_metric_values(metrics, "residual_branch_mlp_rms_", mlp_values)
    _summarize_metric_values(metrics, "residual_contribution_attn_rms_", attn_contribution_values)
    _summarize_metric_values(metrics, "residual_contribution_mlp_rms_", mlp_contribution_values)
    _summarize_metric_values(metrics, "activation_rms_", activation_values)
    return metrics


def _layerwise_gradient_norm_metrics(raw_model: DiffusionTransformer) -> dict[str, float]:
    metrics: dict[str, float] = {}
    layer_values: list[float] = []
    attn_proj_values: list[float] = []
    mlp_down_values: list[float] = []
    for layer_idx, block in enumerate(raw_model.blocks):
        layer_norm = _grad_norm(block.parameters())
        attn_proj_norm = _grad_norm(block.attn.c_proj.parameters())
        mlp_down_norm = _grad_norm(block.mlp.down.parameters())
        metrics[f"layerwise_grad_layer_{layer_idx}_norm"] = layer_norm
        metrics[f"layerwise_grad_layer_{layer_idx}_attn_c_proj_norm"] = attn_proj_norm
        metrics[f"layerwise_grad_layer_{layer_idx}_mlp_down_norm"] = mlp_down_norm
        layer_values.append(layer_norm)
        attn_proj_values.append(attn_proj_norm)
        mlp_down_values.append(mlp_down_norm)
    _summarize_metric_values(metrics, "layerwise_grad_norm_", layer_values)
    _summarize_metric_values(metrics, "layerwise_grad_attn_c_proj_norm_", attn_proj_values)
    _summarize_metric_values(metrics, "layerwise_grad_mlp_down_norm_", mlp_down_values)
    return metrics


def _lm_head_row_norm_metrics(
    raw_model: nn.Module,
    tokenizer,
    data_config,
    device,
    batch_size: int,
    seq_len: int,
    batches: int = 10,
) -> dict[str, float]:
    vocab_size = tokenizer.get_vocab_size()
    token_bytes = build_token_bytes(tokenizer).to(device)
    special_ids = {int(i) for i in tokenizer.get_special_token_ids()}
    counts = torch.zeros(vocab_size, dtype=torch.long, device=device)
    loader = make_dataloader(data_config, tokenizer, int(batch_size), int(seq_len), "train", device=device)
    for _ in range(batches):
        batch = next(loader)
        content = batch.attention_mask & batch.loss_mask & (token_bytes[batch.x_0] > 0)
        for sid in special_ids:
            content &= batch.x_0 != sid
        if content.any():
            ids = batch.x_0[content]
            counts += torch.bincount(ids, minlength=vocab_size)
    lm_head_weight = raw_model.lm_head.weight[:vocab_size].float()
    row_norms = lm_head_weight.norm(dim=1)
    metrics: dict[str, float] = {}
    eos_id = tokenizer.get_eos_token_id()
    if eos_id is not None and 0 <= int(eos_id) < vocab_size:
        metrics["lm_head_row_norm_eos"] = float(row_norms[int(eos_id)].item())
    non_special = [tid for tid in range(vocab_size) if tid not in special_ids]
    non_special.sort(key=lambda tid: (int(counts[tid].item()), tid))
    num_deciles = 7
    decile_size = len(non_special) // num_deciles
    remainder = len(non_special) % num_deciles
    offset = 0
    for decile in range(1, num_deciles + 1):
        size = decile_size + (1 if decile <= remainder else 0)
        tokens = non_special[offset : offset + size]
        if tokens:
            norms = row_norms[tokens]
            metrics[f"lm_head_row_norm_decile_{decile}_mean"] = float(norms.mean().item())
            metrics[f"lm_head_row_norm_decile_{decile}_std"] = float(norms.std().item())
            metrics[f"lm_head_row_norm_decile_{decile}_min"] = float(norms.min().item())
            metrics[f"lm_head_row_norm_decile_{decile}_max"] = float(norms.max().item())
        offset += size
    return metrics


def _token_ratio_metrics(train_tokens: int, counts: dict) -> dict[str, float]:
    """Both token/param ratio conventions, named explicitly to prevent confusion.

    - `tokens_per_nonembedding_param`: tokens per historical transformer-body
      parameter (the legacy repo/batch3 convention).
    - `tokens_per_scaling_param`: tokens per scaling param (transformer + lm_head),
      the nanochat/Karpathy convention used for GPT-2-scale budget comparisons.

    The two differ by the lm_head fraction (large for the 128k vocab at small models)
    and are NOT interchangeable: e.g. at m_deep a non-embedding ratio of 256 is a
    scaling ratio of ~67.6. Both remain logged for historical comparison; the
    active budget convention is recorded separately as `target_param_count_basis`.
    """
    tokens = max(0, int(train_tokens))
    non_emb = max(1, int(counts.get("non_embedding", 0)))
    scaling = max(1, int(counts.get("scaling", 0)))
    return {
        "tokens_per_nonembedding_param": tokens / non_emb,
        "tokens_per_scaling_param": tokens / scaling,
    }
