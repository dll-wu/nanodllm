"""Stopped-gradient self-generated edit objective and sufficient statistics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .training_features import _uses_trajectory_token_edit_canvas


@dataclass(frozen=True)
class SelfGeneratedDraftResult:
    """Draft outputs plus optional differentiable MDLM logits and confidence."""

    tokens: torch.Tensor
    predictions: torch.Tensor
    correct: torch.Tensor
    drafted: torch.Tensor
    confidence: torch.Tensor | None = None
    m2t_logits: torch.Tensor | None = None
    target_probabilities: torch.Tensor | None = None


@dataclass(frozen=True)
class SelfGeneratedCorrectionObjective:
    """Per-target correction losses and their interpretable components."""

    loss_values: torch.Tensor
    expected_error_values: torch.Tensor
    draft_preference_values: torch.Tensor
    generalized_expected_accuracy_values: torch.Tensor
    cross_view_nonregression_values: torch.Tensor


@dataclass(frozen=True)
class FixedRowSelfGeneratedDraftResult:
    """Draft outputs aligned to the T2T targets inside one mixed DB16 batch."""

    tokens: torch.Tensor
    row_mask: torch.Tensor
    target_mask: torch.Tensor
    predictions: torch.Tensor
    correct: torch.Tensor
    drafted: torch.Tensor


def self_generated_objective_configuration_telemetry(config: ExperimentConfig) -> dict[str, float | str]:
    """Return the shared configured telemetry for non-CE correction objectives."""

    if config.train.token_edit_correction_objective == "cross_entropy":
        return {}
    telemetry: dict[str, float | str] = {
        "token_edit_correction_objective": config.train.token_edit_correction_objective,
        "token_edit_correction_aux_weight": float(config.train.token_edit_correction_aux_weight),
        "token_edit_expected_accuracy_power": float(config.train.token_edit_expected_accuracy_power),
        "token_edit_expected_accuracy_power_schedule": config.train.token_edit_expected_accuracy_power_schedule,
        "token_edit_cross_view_nonregression_weight": float(config.train.token_edit_cross_view_nonregression_weight),
    }
    if config.train.token_edit_expected_accuracy_power_schedule == "corruption_threshold":
        telemetry.update(
            {
                "token_edit_expected_accuracy_corruption_threshold": float(
                    config.train.token_edit_expected_accuracy_corruption_threshold
                ),
                "token_edit_expected_accuracy_high_corruption_power": 1.0,
                "token_edit_expected_accuracy_conditioning_signal": "sampled_row_corruption_probability",
                "token_edit_expected_accuracy_conditioning_rule": "configured_power_below_threshold_q1_at_or_above",
            }
        )
    if float(config.train.token_edit_cross_view_nonregression_weight) > 0.0:
        telemetry.update(
            {
                "token_edit_cross_view_nonregression_scope": "initially_correct_drafts",
                "token_edit_cross_view_nonregression_signal": "detached_masked_view_target_probability_hinge",
            }
        )
    return telemetry


def fixed_12_m2t_4_t2t_row_mask(batch_size: int, device) -> torch.Tensor:
    """Return the prospectively fixed M,M,M,T row cadence for one DB16 batch."""

    if batch_size != 16:
        raise ValueError("fixed 12-M2T/4-T2T row layout requires batch_size=16")
    row_indices = torch.arange(batch_size, device=device)
    return row_indices.remainder(4).eq(3)


def mixed_canvas_input_provenance(
    tokens: torch.Tensor,
    target_indices: torch.Tensor,
    drafted: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    confidence_bins: int = 0,
) -> torch.Tensor:
    """Label trusted, unresolved-mask, and inserted-draft input positions."""

    if tokens.dim() != 2 or tokens.dtype != torch.long:
        raise ValueError("mixed-canvas tokens must be a two-dimensional int64 tensor")
    if target_indices.dim() != 1 or target_indices.dtype != torch.long:
        raise ValueError("mixed-canvas target_indices must be a one-dimensional int64 tensor")
    if drafted.dim() != 1 or drafted.dtype != torch.bool or drafted.shape != target_indices.shape:
        raise ValueError("mixed-canvas drafted labels must be boolean and match target_indices")
    if target_indices.device != tokens.device or drafted.device != tokens.device:
        raise ValueError("mixed-canvas tensors must share one device")
    if isinstance(confidence_bins, bool) or not isinstance(confidence_bins, int) or confidence_bins < 0:
        raise ValueError("provenance confidence_bins must be a non-negative integer")
    if confidence_bins:
        if confidence is None or confidence.shape != target_indices.shape or not confidence.is_floating_point():
            raise ValueError("confidence-binned provenance requires one floating confidence per target")
        if confidence.device != tokens.device:
            raise ValueError("provenance confidence must share the token device")
        if not bool(torch.isfinite(confidence).all().item()) or bool(
            ((confidence < 0) | (confidence > 1)).any().item()
        ):
            raise ValueError("provenance confidence must be finite and in [0, 1]")
    elif confidence is not None:
        raise ValueError("provenance confidence requires confidence_bins>0")
    provenance = torch.zeros_like(tokens, dtype=torch.long)
    flat = provenance.reshape(-1)
    flat.index_fill_(0, target_indices, 1)
    if bool(drafted.any().item()):
        if confidence_bins:
            assert confidence is not None
            bins = torch.floor(confidence.detach().float() * confidence_bins).long().clamp_max(confidence_bins - 1)
            flat.index_copy_(0, target_indices[drafted], 3 + bins[drafted])
        else:
            flat.index_fill_(0, target_indices[drafted], 2)
    return provenance


def generation_allowed_confidence(
    logits: torch.Tensor,
    predictions: torch.Tensor,
    generation_blocked_token_ids: set[int],
) -> torch.Tensor:
    """Return detached allowed-vocabulary top-1 probabilities."""

    filtered = logits.detach().clone()
    if generation_blocked_token_ids:
        blocked = torch.tensor(sorted(generation_blocked_token_ids), dtype=torch.long, device=logits.device)
        filtered.index_fill_(-1, blocked, -torch.inf)
    selected = filtered.gather(-1, predictions[:, None]).squeeze(-1).float()
    confidence = torch.exp(selected - torch.logsumexp(filtered.float(), dim=-1)).clamp(0.0, 1.0)
    if not bool(torch.isfinite(confidence).all().item()):
        raise RuntimeError("self-generated draft confidence is non-finite")
    return confidence.detach()


def _mixed_canvas_drafted_targets(target_count: int, draft_fraction: float, device) -> torch.Tensor:
    if not 0.0 < float(draft_fraction) <= 1.0:
        raise ValueError("self-generated draft fraction must be in (0, 1]")
    if target_count < 0:
        raise ValueError("self-generated target count must be non-negative")
    if target_count == 0:
        return torch.empty(0, dtype=torch.bool, device=device)
    if float(draft_fraction) == 1.0 or target_count == 1:
        return torch.ones(target_count, dtype=torch.bool, device=device)
    drafted_count = min(target_count - 1, max(1, round(target_count * float(draft_fraction))))
    drafted = torch.zeros(target_count, dtype=torch.bool, device=device)
    drafted[torch.randperm(target_count, device=device)[:drafted_count]] = True
    return drafted


def _coerce_generation_blocked_token_ids(
    generation_blocked_token_ids: set[int] | frozenset[int] | list[int] | tuple[int, ...],
    *,
    field_name: str,
) -> set[int]:
    blocked_ids: set[int] = set()
    for token_id in generation_blocked_token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"{field_name} blocked token IDs must be integers")
        blocked_ids.add(int(token_id))
    return blocked_ids


def generation_blocked_argmax(
    logits: torch.Tensor,
    generation_blocked_token_ids: set[int] | frozenset[int] | list[int] | tuple[int, ...],
    *,
    field_name: str,
) -> torch.Tensor:
    """Return detached argmax predictions after validating and blocking forbidden IDs."""

    if not torch.is_tensor(logits) or logits.dim() != 2 or not logits.is_floating_point():
        shape = tuple(logits.shape) if torch.is_tensor(logits) else type(logits)
        raise ValueError(f"{field_name} logits must be a floating [targets, vocab] tensor, got {shape}")
    if logits.shape[1] <= 0:
        raise ValueError(f"{field_name} logits must have a non-empty vocabulary dimension")

    blocked_ids = _coerce_generation_blocked_token_ids(
        generation_blocked_token_ids,
        field_name=field_name,
    )
    vocab_size = int(logits.shape[-1])
    if any(token_id < 0 or token_id >= vocab_size for token_id in blocked_ids):
        raise ValueError(f"{field_name} blocked token IDs must be in [0, {vocab_size})")
    if len(blocked_ids) >= vocab_size:
        raise ValueError(f"{field_name} blocks every model output token")

    # Reduce the allowed vocabulary segments independently instead of cloning the
    # full targets-by-vocabulary tensor merely to overwrite a handful of columns.
    # Strict ``>`` updates preserve torch.argmax's lowest-token-ID tie breaking.
    detached_logits = logits.detach()
    best_values: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None
    saw_nan = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
    segment_start = 0
    for segment_end in (*sorted(blocked_ids), vocab_size):
        if segment_start < segment_end:
            segment_values, segment_indices = detached_logits[:, segment_start:segment_end].max(dim=-1)
            segment_indices = segment_indices + segment_start
            saw_nan |= torch.isnan(segment_values)
            if best_values is None:
                best_values = segment_values
                best_indices = segment_indices
            else:
                replace = segment_values > best_values
                best_values = torch.where(replace, segment_values, best_values)
                best_indices = torch.where(replace, segment_indices, best_indices)
        segment_start = segment_end + 1

    if best_values is None or best_indices is None:  # Guarded above, but keep the failure local.
        raise RuntimeError(f"{field_name} has no allowed output tokens")
    if bool((saw_nan | ~torch.isfinite(best_values)).any().item()):
        raise RuntimeError(f"{field_name} logits do not contain a finite allowed token for every target")
    return best_indices.detach()


def build_self_generated_argmax_draft(
    model,
    config: ExperimentConfig,
    x_0: torch.Tensor,
    z_t: torch.Tensor,
    attention_mask: torch.Tensor,
    target_indices: torch.Tensor,
    generation_blocked_token_ids: set[int] | frozenset[int] | list[int] | tuple[int, ...],
    noise_level,
    position_ids: torch.Tensor | None = None,
    draft_fraction: float = 1.0,
) -> SelfGeneratedDraftResult:
    """Fill selected masked positions with detached, generation-safe argmax predictions."""

    if x_0.shape != z_t.shape:
        raise ValueError(
            f"self-generated draft token inputs must share shape: x_0={tuple(x_0.shape)} z_t={tuple(z_t.shape)}"
        )
    if x_0.dim() != 2:
        raise ValueError(f"self-generated draft inputs must be 2D, got {tuple(x_0.shape)}")
    if not torch.is_tensor(target_indices) or target_indices.dim() != 1 or target_indices.dtype != torch.long:
        raise ValueError("self-generated draft target_indices must be a 1D torch.long tensor")
    if target_indices.device != z_t.device:
        raise ValueError("self-generated draft target_indices must be on the same device as z_t")

    differentiable = float(config.train.token_edit_draft_m2t_loss_weight) > 0.0
    preserve_cross_view = float(config.train.token_edit_cross_view_nonregression_weight) > 0.0
    confidence_bins = int(config.model.token_edit_provenance_confidence_bins)
    draft = z_t.detach().clone()
    if target_indices.numel() == 0:
        return SelfGeneratedDraftResult(
            tokens=draft,
            predictions=torch.empty(0, dtype=torch.long, device=z_t.device),
            correct=torch.empty(0, dtype=torch.bool, device=z_t.device),
            drafted=torch.empty(0, dtype=torch.bool, device=z_t.device),
            confidence=(torch.empty(0, dtype=torch.float32, device=z_t.device) if confidence_bins else None),
            m2t_logits=None,
            target_probabilities=(
                torch.empty(0, dtype=torch.float32, device=z_t.device) if preserve_cross_view else None
            ),
        )

    blocked_ids = _coerce_generation_blocked_token_ids(
        generation_blocked_token_ids,
        field_name="self-generated draft",
    )
    # ``index_select`` performs bounds validation without separate min/max host
    # synchronizations. Production indices come from ``active.nonzero()`` and
    # are therefore unique by construction.
    clean_targets = x_0.reshape(-1).index_select(0, target_indices)
    if blocked_ids:
        blocked = torch.tensor(sorted(blocked_ids), dtype=clean_targets.dtype, device=clean_targets.device)
        if bool(torch.isin(clean_targets, blocked).any().item()):
            raise ValueError("self-generated draft clean targets contain generation-blocked token IDs")

    model_kwargs = {
        "logits_indices": target_indices,
        "return_aux_logits": False,
        "return_hierarchical_logits": False,
        "return_length_logits": False,
        "attention_mask": attention_mask,
        "noise_level": noise_level if config.noise.time_conditioning else None,
        "input_embeddings": None,
    }
    if position_ids is not None:
        model_kwargs["position_ids"] = position_ids
    with torch.set_grad_enabled(differentiable):
        logits = model(z_t, **model_kwargs)
    if not torch.is_tensor(logits) or logits.dim() != 2 or not logits.is_floating_point():
        shape = tuple(logits.shape) if torch.is_tensor(logits) else type(logits)
        raise ValueError(f"self-generated draft logits must be a floating [targets, vocab] tensor, got {shape}")
    if logits.shape[0] != target_indices.numel() or logits.shape[1] <= 0:
        raise ValueError(
            "self-generated draft logits must contain one non-empty vocabulary row per target: "
            f"logits={tuple(logits.shape)} targets={target_indices.numel()}"
        )

    predictions = generation_blocked_argmax(
        logits,
        blocked_ids,
        field_name="self-generated draft",
    )
    confidence = generation_allowed_confidence(logits, predictions, blocked_ids) if confidence_bins else None
    target_probabilities = None
    if preserve_cross_view:
        row_indices = torch.arange(target_indices.numel(), device=logits.device)
        target_probabilities = F.softmax(logits.detach().float(), dim=-1)[row_indices, clean_targets].detach()
    draft_correct = predictions.eq(clean_targets).detach()
    drafted = _mixed_canvas_drafted_targets(target_indices.numel(), draft_fraction, z_t.device)
    if bool(drafted.any().item()):
        draft.reshape(-1).index_copy_(
            0,
            target_indices[drafted],
            predictions[drafted].to(dtype=draft.dtype),
        )
    return SelfGeneratedDraftResult(
        tokens=draft.detach(),
        predictions=predictions,
        correct=draft_correct,
        drafted=drafted.detach(),
        confidence=confidence,
        m2t_logits=logits if differentiable else None,
        target_probabilities=target_probabilities,
    )


def joint_two_view_loss(
    correction_loss: torch.Tensor,
    draft_m2t_loss: torch.Tensor,
    *,
    draft_m2t_weight: float,
) -> torch.Tensor:
    """Convexly mix correction and same-view draft-MDLM objectives."""

    if correction_loss.ndim != 0 or draft_m2t_loss.ndim != 0:
        raise ValueError("joint two-view losses must be scalar tensors")
    if not 0.0 <= float(draft_m2t_weight) <= 1.0:
        raise ValueError("joint two-view draft-M2T weight must be in [0, 1]")
    return (1.0 - float(draft_m2t_weight)) * correction_loss + float(draft_m2t_weight) * draft_m2t_loss


@torch.no_grad()
def build_fixed_row_self_generated_argmax_draft(
    model,
    config: ExperimentConfig,
    x_0: torch.Tensor,
    z_t: torch.Tensor,
    active: torch.Tensor,
    attention_mask: torch.Tensor,
    target_indices: torch.Tensor,
    generation_blocked_token_ids: set[int] | frozenset[int] | list[int] | tuple[int, ...],
    noise_level,
    position_ids: torch.Tensor | None = None,
) -> FixedRowSelfGeneratedDraftResult:
    """Draft only four T2T rows and return one combined DB16 correction canvas."""

    if x_0.shape != z_t.shape or active.shape != x_0.shape:
        raise ValueError("fixed-row clean, corrupted, and active tensors must share one shape")
    if x_0.dim() != 2 or active.dtype != torch.bool:
        raise ValueError("fixed-row inputs must be 2D tokens with a boolean active mask")
    if target_indices.dim() != 1 or target_indices.dtype != torch.long:
        raise ValueError("fixed-row target_indices must be a one-dimensional int64 tensor")
    batch_size, seq_len = x_0.shape
    row_mask = fixed_12_m2t_4_t2t_row_mask(batch_size, x_0.device)
    row_indices = row_mask.nonzero(as_tuple=True)[0]
    target_rows = torch.div(target_indices, seq_len, rounding_mode="floor")
    target_mask = row_mask.index_select(0, target_rows)
    local_active = active.index_select(0, row_indices)
    local_target_indices = local_active.reshape(-1).nonzero(as_tuple=True)[0]
    if int(local_target_indices.numel()) != int(target_mask.sum().item()):
        raise RuntimeError("fixed-row local and combined T2T target counts disagree")

    local_attention_mask = attention_mask.index_select(0, row_indices)
    local_position_ids = position_ids.index_select(0, row_indices) if position_ids is not None else None
    draft_result = build_self_generated_argmax_draft(
        model,
        config,
        x_0.index_select(0, row_indices),
        z_t.index_select(0, row_indices),
        local_attention_mask,
        local_target_indices,
        generation_blocked_token_ids,
        noise_level,
        position_ids=local_position_ids,
        draft_fraction=1.0,
    )
    combined_tokens = z_t.detach().clone()
    combined_tokens.index_copy_(0, row_indices, draft_result.tokens)
    return FixedRowSelfGeneratedDraftResult(
        tokens=combined_tokens.detach(),
        row_mask=row_mask,
        target_mask=target_mask,
        predictions=draft_result.predictions,
        correct=draft_result.correct,
        drafted=draft_result.drafted,
    )


def self_generated_t2t_target_weights(
    target_t: torch.Tensor,
    draft_correct: torch.Tensor,
    *,
    correct_weight: float,
    wrong_weight: float,
    drafted: torch.Tensor | None = None,
    unresolved_mask_weight: float = 1.0,
) -> torch.Tensor:
    """Return detached ``1 / t`` weights adjusted by draft correctness.

    The caller remains responsible for applying the all-lossable-token denominator.
    """

    if not torch.is_tensor(target_t) or target_t.dim() != 1 or not target_t.is_floating_point():
        raise ValueError("self-generated T2T target_t must be a 1D floating tensor")
    if not torch.is_tensor(draft_correct) or draft_correct.dim() != 1 or draft_correct.dtype != torch.bool:
        raise ValueError("self-generated T2T draft_correct must be a 1D boolean tensor")
    if target_t.shape != draft_correct.shape or target_t.device != draft_correct.device:
        raise ValueError("self-generated T2T target_t and draft_correct must share shape and device")
    if drafted is None:
        drafted = torch.ones_like(draft_correct)
    if drafted.dtype != torch.bool or drafted.shape != draft_correct.shape or drafted.device != draft_correct.device:
        raise ValueError("self-generated T2T drafted labels must be boolean and match draft correctness")
    for name, value in (
        ("correct_weight", correct_weight),
        ("wrong_weight", wrong_weight),
        ("unresolved_mask_weight", unresolved_mask_weight),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"self-generated T2T {name} must be a finite non-negative number")
        if float(value) < 0.0:
            raise ValueError(f"self-generated T2T {name} must be a finite non-negative number")

    detached_t = target_t.detach().float()
    if not torch.isfinite(detached_t).all() or (detached_t <= 0).any():
        raise ValueError("self-generated T2T target_t values must be finite and positive")
    class_weights = torch.where(
        draft_correct,
        torch.full_like(detached_t, float(correct_weight)),
        torch.full_like(detached_t, float(wrong_weight)),
    )
    class_weights = torch.where(
        drafted,
        class_weights,
        torch.full_like(detached_t, float(unresolved_mask_weight)),
    )
    return (class_weights / detached_t).detach()


def self_generated_t2t_correction_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ce_values: torch.Tensor,
    draft_predictions: torch.Tensor,
    draft_correct: torch.Tensor,
    *,
    objective: str,
    auxiliary_weight: float,
    expected_accuracy_power: float = 1.0,
    expected_accuracy_powers: torch.Tensor | None = None,
    draft_target_probabilities: torch.Tensor | None = None,
    cross_view_nonregression_weight: float = 0.0,
) -> SelfGeneratedCorrectionObjective:
    """Build a zero-variance reward or draft-preference correction objective.

    ``(1 - p(target)^q) / q`` generalizes the exact negative expected terminal-
    token accuracy recovered at ``q=1``. An optional detached per-target power
    tensor supports corruption-conditioned rows while preserving the scalar as
    the configured low-corruption power. The optional cross-view hinge prevents
    the correction view from lowering target probability on initially correct
    drafts. Draft-view probabilities, labels, and selectors are detached control
    signals and never carry gradients from the draft pass.
    """

    if not torch.is_tensor(logits) or logits.dim() != 2 or not logits.is_floating_point():
        raise ValueError("self-generated correction logits must be a floating [targets, vocab] tensor")
    target_count, vocab_size = logits.shape
    if vocab_size <= 0:
        raise ValueError("self-generated correction logits must have a non-empty vocabulary")
    for name, value, dtype in (
        ("targets", targets, torch.long),
        ("draft_predictions", draft_predictions, torch.long),
        ("draft_correct", draft_correct, torch.bool),
    ):
        if not torch.is_tensor(value) or value.dim() != 1 or value.shape[0] != target_count or value.dtype != dtype:
            raise ValueError(f"self-generated correction {name} must match the target rows and dtype")
        if value.device != logits.device:
            raise ValueError(f"self-generated correction {name} must share the logits device")
    if (
        not torch.is_tensor(ce_values)
        or ce_values.dim() != 1
        or ce_values.shape[0] != target_count
        or not ce_values.is_floating_point()
        or ce_values.device != logits.device
    ):
        raise ValueError("self-generated correction CE values must match the logits rows")
    if target_count and (
        int(targets.min().item()) < 0
        or int(targets.max().item()) >= vocab_size
        or int(draft_predictions.min().item()) < 0
        or int(draft_predictions.max().item()) >= vocab_size
    ):
        raise ValueError("self-generated correction token IDs are outside the logits vocabulary")
    if not torch.equal(draft_correct, draft_predictions.eq(targets)):
        raise ValueError("self-generated correction draft labels disagree with predictions and targets")
    if isinstance(auxiliary_weight, bool) or not isinstance(auxiliary_weight, (int, float)):
        raise ValueError("self-generated correction auxiliary weight must be numeric")
    if not math.isfinite(float(auxiliary_weight)) or float(auxiliary_weight) < 0.0:
        raise ValueError("self-generated correction auxiliary weight must be finite and non-negative")
    for name, value in (
        ("expected-accuracy power", expected_accuracy_power),
        ("cross-view non-regression weight", cross_view_nonregression_weight),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"self-generated correction {name} must be finite and numeric")
    if not 0.0 < float(expected_accuracy_power) <= 1.0:
        raise ValueError("self-generated correction expected-accuracy power must be in (0, 1]")
    if expected_accuracy_powers is not None:
        if (
            not torch.is_tensor(expected_accuracy_powers)
            or expected_accuracy_powers.dim() != 1
            or expected_accuracy_powers.shape[0] != target_count
            or not expected_accuracy_powers.is_floating_point()
            or expected_accuracy_powers.device != logits.device
        ):
            raise ValueError("per-target expected-accuracy powers must be floating and match the logits rows")
        if expected_accuracy_powers.requires_grad:
            raise ValueError("per-target expected-accuracy powers must be detached")
        if not bool(torch.isfinite(expected_accuracy_powers).all().item()) or bool(
            ((expected_accuracy_powers <= 0) | (expected_accuracy_powers > 1)).any().item()
        ):
            raise ValueError("per-target expected-accuracy powers must be finite and in (0, 1]")
    if float(cross_view_nonregression_weight) < 0.0:
        raise ValueError("self-generated correction cross-view non-regression weight must be non-negative")
    if float(cross_view_nonregression_weight) > 0.0:
        if (
            draft_target_probabilities is None
            or not torch.is_tensor(draft_target_probabilities)
            or draft_target_probabilities.dim() != 1
            or draft_target_probabilities.shape[0] != target_count
            or not draft_target_probabilities.is_floating_point()
            or draft_target_probabilities.device != logits.device
        ):
            raise ValueError("cross-view non-regression requires one draft target probability per logits row")
        if draft_target_probabilities.requires_grad:
            raise ValueError("cross-view draft target probabilities must be detached")
        if not bool(torch.isfinite(draft_target_probabilities).all().item()) or bool(
            ((draft_target_probabilities < 0) | (draft_target_probabilities > 1)).any().item()
        ):
            raise ValueError("cross-view draft target probabilities must be finite and in [0, 1]")
    elif draft_target_probabilities is not None:
        raise ValueError("draft target probabilities require a positive cross-view non-regression weight")

    supported = {
        "cross_entropy",
        "expected_accuracy",
        "cross_entropy_plus_expected_accuracy",
        "cross_entropy_plus_draft_preference",
    }
    if objective not in supported:
        raise ValueError(f"unsupported self-generated correction objective: {objective!r}")
    generalized_expected_accuracy = float(expected_accuracy_power) != 1.0 or expected_accuracy_powers is not None
    if (
        generalized_expected_accuracy or float(cross_view_nonregression_weight) > 0.0
    ) and objective != "expected_accuracy":
        raise ValueError("generalized expected-accuracy controls require the pure expected_accuracy objective")
    if generalized_expected_accuracy and float(cross_view_nonregression_weight) > 0.0:
        raise ValueError("generalized expected accuracy and cross-view non-regression must be screened independently")
    combined = objective.startswith("cross_entropy_plus_")
    if combined != (float(auxiliary_weight) > 0.0):
        raise ValueError("combined self-generated correction objectives require one positive auxiliary weight")

    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    row_indices = torch.arange(target_count, device=logits.device)
    target_log_probabilities = log_probabilities[row_indices, targets]
    target_probabilities = target_log_probabilities.exp()
    expected_error_values = 1.0 - target_probabilities
    if expected_accuracy_powers is not None:
        power_values = expected_accuracy_powers.detach().float()
        powered_values = (1.0 - target_probabilities.pow(power_values)) / power_values
        generalized_expected_accuracy_values = torch.where(power_values.eq(1.0), expected_error_values, powered_values)
    elif float(expected_accuracy_power) == 1.0:
        generalized_expected_accuracy_values = expected_error_values
    else:
        generalized_expected_accuracy_values = (1.0 - target_probabilities.pow(float(expected_accuracy_power))) / float(
            expected_accuracy_power
        )
    cross_view_nonregression_values = torch.zeros_like(expected_error_values)
    if draft_target_probabilities is not None:
        cross_view_nonregression_values = torch.where(
            draft_correct,
            F.relu(draft_target_probabilities.float() - target_probabilities),
            torch.zeros_like(expected_error_values),
        )
    draft_log_probabilities = log_probabilities[row_indices, draft_predictions]
    draft_preference_values = torch.where(
        draft_correct,
        torch.zeros_like(target_log_probabilities),
        F.softplus(draft_log_probabilities - target_log_probabilities),
    )
    if objective == "cross_entropy":
        loss_values = ce_values
    elif objective == "expected_accuracy":
        loss_values = generalized_expected_accuracy_values + (
            float(cross_view_nonregression_weight) * cross_view_nonregression_values
        )
    elif objective == "cross_entropy_plus_expected_accuracy":
        loss_values = ce_values + float(auxiliary_weight) * expected_error_values
    else:
        loss_values = ce_values + float(auxiliary_weight) * draft_preference_values
    return SelfGeneratedCorrectionObjective(
        loss_values=loss_values,
        expected_error_values=expected_error_values,
        draft_preference_values=draft_preference_values,
        generalized_expected_accuracy_values=generalized_expected_accuracy_values,
        cross_view_nonregression_values=cross_view_nonregression_values,
    )


def corruption_conditioned_expected_accuracy_powers(
    target_t: torch.Tensor,
    *,
    configured_power: float,
    corruption_threshold: float,
) -> torch.Tensor:
    """Select q per target from the sampled row corruption probability."""

    if not torch.is_tensor(target_t) or target_t.dim() != 1 or not target_t.is_floating_point():
        raise ValueError("corruption-conditioned expected accuracy requires one floating target_t per target")
    if target_t.requires_grad:
        raise ValueError("corruption-conditioned target_t must be detached")
    if not bool(torch.isfinite(target_t).all().item()) or bool(((target_t <= 0) | (target_t > 1)).any().item()):
        raise ValueError("corruption-conditioned target_t must be finite and in (0, 1]")
    if isinstance(configured_power, bool) or not isinstance(configured_power, (int, float)):
        raise ValueError("corruption-conditioned expected-accuracy power must be numeric")
    if not math.isfinite(float(configured_power)) or not 0.0 < float(configured_power) < 1.0:
        raise ValueError("corruption-conditioned expected-accuracy power must be finite and in (0, 1)")
    if isinstance(corruption_threshold, bool) or not isinstance(corruption_threshold, (int, float)):
        raise ValueError("corruption-conditioned threshold must be numeric")
    if not math.isfinite(float(corruption_threshold)) or not 0.0 < float(corruption_threshold) < 1.0:
        raise ValueError("corruption-conditioned threshold must be finite and in (0, 1)")
    return torch.where(
        target_t.detach().float() < float(corruption_threshold),
        torch.full_like(target_t, float(configured_power), dtype=torch.float32),
        torch.ones_like(target_t, dtype=torch.float32),
    )


def build_self_generated_correction_objective_for_session(session) -> SelfGeneratedCorrectionObjective:
    """Bind one training session's tensors and configured correction controls."""

    train = session.config.train
    expected_accuracy_powers = None
    if train.token_edit_expected_accuracy_power_schedule == "corruption_threshold":
        if not isinstance(session.t_val, torch.Tensor):
            raise RuntimeError("corruption-conditioned expected accuracy requires per-target corruption probabilities")
        # Local import avoids a module cycle: training_metrics imports the
        # objective façade, which owns this module's configuration telemetry.
        from .training_metrics import _per_target_t_values

        expected_accuracy_powers = corruption_conditioned_expected_accuracy_powers(
            _per_target_t_values(session.t_val, session.x_0, session.target_indices),
            configured_power=float(train.token_edit_expected_accuracy_power),
            corruption_threshold=float(train.token_edit_expected_accuracy_corruption_threshold),
        )
    return self_generated_t2t_correction_objective(
        session.logits,
        session.targets,
        session.main_ce_values,
        session.self_generated_draft_predictions,
        session.self_generated_draft_correct,
        objective=train.token_edit_correction_objective,
        auxiliary_weight=float(train.token_edit_correction_aux_weight),
        expected_accuracy_power=float(train.token_edit_expected_accuracy_power),
        expected_accuracy_powers=expected_accuracy_powers,
        draft_target_probabilities=session.self_generated_draft_target_probabilities,
        cross_view_nonregression_weight=float(train.token_edit_cross_view_nonregression_weight),
    )


SELF_GENERATED_EDIT_ACCUMULATOR_FIELDS = (
    "microbatches",
    "draft_forwards",
    "correction_forwards",
    "draft_tokens",
    "draft_correct",
    "draft_wrong",
    "c_to_c",
    "c_to_w",
    "w_to_c",
    "w_to_w",
    "correct_ce_sum",
    "wrong_ce_sum",
)

_COUNT_FIELDS = SELF_GENERATED_EDIT_ACCUMULATOR_FIELDS[:-2]
_SUM_FIELDS = SELF_GENERATED_EDIT_ACCUMULATOR_FIELDS[-2:]


def new_self_generated_edit_accumulator() -> dict[str, int | float]:
    return {
        **{field: 0 for field in _COUNT_FIELDS},
        **{field: 0.0 for field in _SUM_FIELDS},
    }


def coerce_self_generated_edit_accumulator(
    value: object,
    *,
    field_name: str,
) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    keys = tuple(value.keys())
    if set(keys) != set(SELF_GENERATED_EDIT_ACCUMULATOR_FIELDS) or len(keys) != len(
        SELF_GENERATED_EDIT_ACCUMULATOR_FIELDS
    ):
        missing = sorted(set(SELF_GENERATED_EDIT_ACCUMULATOR_FIELDS) - set(keys))
        extra = sorted(set(keys) - set(SELF_GENERATED_EDIT_ACCUMULATOR_FIELDS))
        raise TypeError(f"{field_name} has unexpected fields: missing={missing}, extra={extra}")

    result = new_self_generated_edit_accumulator()
    for key in _COUNT_FIELDS:
        item = value[key]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise TypeError(f"{field_name}.{key} must be a non-negative integer")
        result[key] = item
    for key in _SUM_FIELDS:
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TypeError(f"{field_name}.{key} must be numeric")
        number = float(item)
        if not math.isfinite(number) or number < 0.0:
            raise TypeError(f"{field_name}.{key} must be finite and non-negative")
        result[key] = number

    _validate_self_generated_edit_conservation(result, field_name=field_name)
    return result


def _validate_self_generated_edit_conservation(
    accumulator: Mapping[str, int | float],
    *,
    field_name: str,
) -> None:
    if int(accumulator["draft_tokens"]) != int(accumulator["draft_correct"]) + int(accumulator["draft_wrong"]):
        raise ValueError(f"{field_name} draft-token counts do not conserve")
    if int(accumulator["draft_correct"]) != int(accumulator["c_to_c"]) + int(accumulator["c_to_w"]):
        raise ValueError(f"{field_name} correct-draft transitions do not conserve")
    if int(accumulator["draft_wrong"]) != int(accumulator["w_to_c"]) + int(accumulator["w_to_w"]):
        raise ValueError(f"{field_name} wrong-draft transitions do not conserve")
    microbatches = int(accumulator["microbatches"])
    if microbatches != int(accumulator["correction_forwards"]):
        raise ValueError(f"{field_name} correction-forward count must equal completed microbatches")
    if int(accumulator["draft_forwards"]) < microbatches:
        raise ValueError(f"{field_name} draft-forward count must cover every completed microbatch")


def update_self_generated_edit_accumulator(
    accumulator: dict[str, int | float],
    *,
    draft_correct: torch.Tensor,
    final_predictions: torch.Tensor,
    targets: torch.Tensor,
    ce_values: torch.Tensor,
    drafted: torch.Tensor | None = None,
    allow_empty: bool = False,
    draft_forward_count: int = 1,
) -> None:
    coerce_self_generated_edit_accumulator(accumulator, field_name="self_generated_edit_accumulator")
    if draft_correct.dtype != torch.bool or draft_correct.ndim != 1:
        raise ValueError("draft_correct must be a one-dimensional boolean tensor")
    if final_predictions.dtype != torch.long or targets.dtype != torch.long:
        raise ValueError("final predictions and targets must be int64 token IDs")
    if final_predictions.ndim != 1 or targets.ndim != 1 or ce_values.ndim != 1:
        raise ValueError("self-generated edit observations must be one-dimensional")
    if not (draft_correct.shape == final_predictions.shape == targets.shape == ce_values.shape):
        raise ValueError("self-generated edit observations must have identical shapes")
    if drafted is None:
        drafted = torch.ones_like(draft_correct)
    if drafted.dtype != torch.bool or drafted.shape != draft_correct.shape or drafted.device != draft_correct.device:
        raise ValueError("self-generated edit drafted labels must be boolean and match draft correctness")
    if isinstance(draft_forward_count, bool) or not isinstance(draft_forward_count, int) or draft_forward_count < 1:
        raise ValueError("self-generated edit draft_forward_count must be a positive integer")
    if not ce_values.is_floating_point():
        raise ValueError("self-generated edit CE values must be floating point")
    if ce_values.numel() == 0:
        if not allow_empty:
            raise ValueError("self-generated edit accumulator requires at least one drafted token")
        accumulator["microbatches"] = int(accumulator["microbatches"]) + 1
        accumulator["draft_forwards"] = int(accumulator["draft_forwards"]) + draft_forward_count
        accumulator["correction_forwards"] = int(accumulator["correction_forwards"]) + 1
        _validate_self_generated_edit_conservation(accumulator, field_name="self_generated_edit_accumulator")
        return
    if not bool(drafted.any().item()):
        if not allow_empty:
            raise ValueError("self-generated edit accumulator requires at least one drafted token")
        accumulator["microbatches"] = int(accumulator["microbatches"]) + 1
        accumulator["draft_forwards"] = int(accumulator["draft_forwards"]) + draft_forward_count
        accumulator["correction_forwards"] = int(accumulator["correction_forwards"]) + 1
        _validate_self_generated_edit_conservation(accumulator, field_name="self_generated_edit_accumulator")
        return
    if not bool(torch.isfinite(ce_values).all().item()) or bool((ce_values < 0).any().item()):
        raise ValueError("self-generated edit CE values must be finite and non-negative")

    draft_correct = draft_correct[drafted]
    final_correct = final_predictions.eq(targets)[drafted]
    ce_values = ce_values[drafted]
    draft_wrong = ~draft_correct
    accumulator["microbatches"] = int(accumulator["microbatches"]) + 1
    accumulator["draft_forwards"] = int(accumulator["draft_forwards"]) + draft_forward_count
    accumulator["correction_forwards"] = int(accumulator["correction_forwards"]) + 1
    accumulator["draft_tokens"] = int(accumulator["draft_tokens"]) + int(draft_correct.numel())
    accumulator["draft_correct"] = int(accumulator["draft_correct"]) + int(draft_correct.sum().item())
    accumulator["draft_wrong"] = int(accumulator["draft_wrong"]) + int(draft_wrong.sum().item())
    accumulator["c_to_c"] = int(accumulator["c_to_c"]) + int((draft_correct & final_correct).sum().item())
    accumulator["c_to_w"] = int(accumulator["c_to_w"]) + int((draft_correct & ~final_correct).sum().item())
    accumulator["w_to_c"] = int(accumulator["w_to_c"]) + int((draft_wrong & final_correct).sum().item())
    accumulator["w_to_w"] = int(accumulator["w_to_w"]) + int((draft_wrong & ~final_correct).sum().item())
    accumulator["correct_ce_sum"] = float(accumulator["correct_ce_sum"]) + float(
        ce_values[draft_correct].detach().float().sum().item()
    )
    accumulator["wrong_ce_sum"] = float(accumulator["wrong_ce_sum"]) + float(
        ce_values[draft_wrong].detach().float().sum().item()
    )
    _validate_self_generated_edit_conservation(accumulator, field_name="self_generated_edit_accumulator")


def accumulate_self_generated_correction_step(
    session,
    logits: torch.Tensor,
    targets: torch.Tensor,
    ce_values: torch.Tensor,
    *,
    target_mask: torch.Tensor | None,
) -> None:
    """Record correction transitions and objective components under an optional target scope."""

    draft_correct = session.self_generated_draft_correct
    drafted = session.self_generated_drafted
    if draft_correct is None:
        raise RuntimeError("self-generated edit branch is missing draft correctness labels")
    if drafted is None:
        raise RuntimeError("self-generated edit branch is missing drafted-position labels")
    if target_mask is not None:
        logits = logits[target_mask]
        targets = targets[target_mask]
        ce_values = ce_values[target_mask]
        draft_correct = draft_correct[target_mask]
        drafted = drafted[target_mask]
        if _uses_trajectory_token_edit_canvas(session.config):
            # These are the final rollout proposals for targets left unresolved
            # before the correction forward, so their transition is observable
            # even though the proposals were not inserted into the canvas.
            drafted = torch.ones_like(drafted)
    session.self_generated_final_predictions = generation_blocked_argmax(
        logits,
        session.generation_blocked_token_ids,
        field_name="self-generated correction",
    )
    update_self_generated_edit_accumulator(
        session.step_self_generated_edit_accumulator,
        draft_correct=draft_correct,
        final_predictions=session.self_generated_final_predictions,
        targets=targets,
        ce_values=ce_values,
        drafted=drafted,
        allow_empty=target_mask is not None,
        draft_forward_count=max(1, int(getattr(session, "token_edit_trajectory_model_forwards", 0))),
    )
    objective = session.self_generated_correction_objective
    if objective is None:
        raise RuntimeError("self-generated edit branch is missing correction-objective telemetry")
    selected = target_mask if target_mask is not None else slice(None)
    session.step_self_generated_correction_target_count += int(targets.numel())
    session.step_self_generated_expected_error_sum += float(
        objective.expected_error_values[selected].detach().sum().item()
    )
    session.step_self_generated_draft_preference_sum += float(
        objective.draft_preference_values[selected].detach().sum().item()
    )


def merge_self_generated_edit_accumulator(
    destination: dict[str, int | float],
    source: Mapping[str, int | float],
) -> None:
    coerce_self_generated_edit_accumulator(destination, field_name="self_generated_edit_destination")
    coerce_self_generated_edit_accumulator(source, field_name="self_generated_edit_source")
    for key in _COUNT_FIELDS:
        destination[key] = int(destination[key]) + int(source[key])
    for key in _SUM_FIELDS:
        destination[key] = float(destination[key]) + float(source[key])
    _validate_self_generated_edit_conservation(destination, field_name="self_generated_edit_accumulator")


def self_generated_edit_metrics(
    prefix: str,
    accumulator: Mapping[str, int | float],
) -> dict[str, int | float]:
    validated = coerce_self_generated_edit_accumulator(accumulator, field_name="self_generated_edit_accumulator")
    draft_tokens = int(validated["draft_tokens"])
    draft_correct = int(validated["draft_correct"])
    draft_wrong = int(validated["draft_wrong"])
    c_to_c = int(validated["c_to_c"])
    c_to_w = int(validated["c_to_w"])
    w_to_c = int(validated["w_to_c"])
    w_to_w = int(validated["w_to_w"])
    final_correct = c_to_c + w_to_c
    return {
        f"{prefix}microbatches": int(validated["microbatches"]),
        f"{prefix}draft_forwards": int(validated["draft_forwards"]),
        f"{prefix}correction_forwards": int(validated["correction_forwards"]),
        f"{prefix}draft_tokens": draft_tokens,
        f"{prefix}draft_correct_tokens": draft_correct,
        f"{prefix}draft_wrong_tokens": draft_wrong,
        f"{prefix}c_to_c": c_to_c,
        f"{prefix}c_to_w": c_to_w,
        f"{prefix}w_to_c": w_to_c,
        f"{prefix}w_to_w": w_to_w,
        f"{prefix}draft_accuracy": draft_correct / max(1, draft_tokens),
        f"{prefix}final_accuracy": final_correct / max(1, draft_tokens),
        f"{prefix}damage_rate": c_to_w / max(1, draft_correct),
        f"{prefix}repair_rate": w_to_c / max(1, draft_wrong),
        f"{prefix}net_repair_rate": (w_to_c - c_to_w) / max(1, draft_tokens),
        f"{prefix}correct_draft_ce": float(validated["correct_ce_sum"]) / max(1, draft_correct),
        f"{prefix}wrong_draft_ce": float(validated["wrong_ce_sum"]) / max(1, draft_wrong),
    }
