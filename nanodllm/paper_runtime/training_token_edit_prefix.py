"""Stateless, packed-segment-local prefix plans for self-generated T2T."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .telemetry import append_jsonl
from .training_features import _uses_full_draft_token_edit_provenance, _uses_mixed_token_edit_canvas
from .training_self_generated_edit import (
    SelfGeneratedDraftResult,
    build_self_generated_argmax_draft,
    mixed_canvas_input_provenance,
)
from .training_token_edit_trajectory import build_trajectory_argmax_draft, uses_trajectory_canvas

if TYPE_CHECKING:
    from .training_state import TrainingSession

TOKEN_EDIT_PREFIX_PLAN_SCHEMA_VERSION = 1
_BOUNDARY_HASH_DOMAIN = "nanodllm-token-edit-prefix-boundary-v1"


@dataclass(frozen=True)
class TokenEditPrefixSegment:
    row: int
    ordinal: int
    start: int
    end: int
    prefix_end: int
    sampled_fraction_u53: int
    lossable_tokens: int
    prefix_lossable_tokens: int
    suffix_lossable_tokens: int
    splittable: bool

    def receipt(self) -> dict[str, int | bool]:
        return {
            "row": self.row,
            "ordinal": self.ordinal,
            "start": self.start,
            "end": self.end,
            "prefix_end": self.prefix_end,
            "sampled_fraction_u53": self.sampled_fraction_u53,
            "lossable_tokens": self.lossable_tokens,
            "prefix_lossable_tokens": self.prefix_lossable_tokens,
            "suffix_lossable_tokens": self.suffix_lossable_tokens,
            "splittable": self.splittable,
        }


@dataclass(frozen=True)
class TokenEditPrefixPlan:
    prefix_mask: torch.Tensor
    suffix_lossable_mask: torch.Tensor
    segments: tuple[TokenEditPrefixSegment, ...]
    min_fraction: float
    max_fraction: float
    boundary_seed: int
    rank: int
    plan_index: int
    plan_sha256: str

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def splittable_segment_count(self) -> int:
        return sum(segment.splittable for segment in self.segments)

    @property
    def prefix_lossable_tokens(self) -> int:
        return sum(segment.prefix_lossable_tokens for segment in self.segments)

    @property
    def lossable_tokens(self) -> int:
        return sum(segment.lossable_tokens for segment in self.segments)

    @property
    def suffix_lossable_tokens(self) -> int:
        return sum(segment.suffix_lossable_tokens for segment in self.segments)


@dataclass(frozen=True)
class TokenEditPrefixInputs:
    draft_input_tokens: torch.Tensor
    draft_target_indices: torch.Tensor
    correction_target_indices: torch.Tensor
    correction_loss_target_mask: torch.Tensor
    correction_lossable_tokens: int
    restored_prefix_active_tokens: int


@dataclass(frozen=True)
class TokenEditPrefixDraft:
    tokens: torch.Tensor
    predictions: torch.Tensor
    correct: torch.Tensor
    drafted: torch.Tensor


def apply_token_edit_prefix_loss_scope(
    token_weights: torch.Tensor,
    inputs: TokenEditPrefixInputs,
) -> tuple[torch.Tensor, int]:
    """Zero prefix objective mass and return the exact suffix denominator."""

    if token_weights.ndim != 1 or not token_weights.is_floating_point():
        raise ValueError("token-edit prefix weights must be a one-dimensional floating tensor")
    target_mask = inputs.correction_loss_target_mask
    if target_mask.shape != token_weights.shape or target_mask.device != token_weights.device:
        raise ValueError("token-edit prefix loss-target mask must match objective weights")
    if inputs.correction_lossable_tokens < 0:
        raise ValueError("token-edit prefix lossable-token denominator must be non-negative")
    return token_weights * target_mask.to(dtype=token_weights.dtype), max(1, inputs.correction_lossable_tokens)


def _validate_plan_tensors(
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
) -> None:
    if loss_mask.ndim != 2 or loss_mask.dtype != torch.bool:
        raise ValueError("token-edit prefix loss_mask must be a two-dimensional boolean tensor")
    if attention_mask.shape != loss_mask.shape or attention_mask.dtype != torch.bool:
        raise ValueError("token-edit prefix attention_mask must be boolean and match loss_mask")
    if loss_mask.device != attention_mask.device:
        raise ValueError("token-edit prefix masks must share one device")
    if bool((loss_mask & ~attention_mask).any().item()):
        raise ValueError("token-edit prefix loss_mask must be contained in attention_mask")
    if position_ids is not None:
        if position_ids.shape != loss_mask.shape or position_ids.dtype != torch.long:
            raise ValueError("token-edit prefix position_ids must be int64 and match loss_mask")
        if position_ids.device != loss_mask.device:
            raise ValueError("token-edit prefix position_ids must share the mask device")


def _segment_uniform_u53(boundary_seed: int, rank: int, plan_index: int, row: int, ordinal: int) -> int:
    material = f"{_BOUNDARY_HASH_DOMAIN}:{boundary_seed}:{rank}:{plan_index}:{row}:{ordinal}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") >> 11


def _segment_spans(attention: list[bool], positions: list[int] | None) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    previous_position: int | None = None
    for index, valid in enumerate(attention):
        if not valid:
            if start is not None:
                spans.append((start, index))
                start = None
            previous_position = None
            continue
        position = positions[index] if positions is not None else None
        if start is None and positions is not None and position != 0:
            raise ValueError("token-edit prefix position_ids must start each valid span at zero")
        reset = start is None or (positions is not None and position == 0)
        if reset:
            if start is not None:
                spans.append((start, index))
            start = index
        elif positions is not None and position != int(previous_position) + 1:
            raise ValueError("token-edit prefix position_ids must advance by one or reset to zero")
        if positions is not None and position < 0:
            raise ValueError("token-edit prefix position_ids must be non-negative on valid tokens")
        previous_position = position
    if start is not None:
        spans.append((start, len(attention)))
    return spans


def sample_token_edit_prefix_plan(
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    *,
    min_fraction: float,
    max_fraction: float,
    boundary_seed: int,
    rank: int,
    plan_index: int,
) -> TokenEditPrefixPlan:
    """Sample one replayable boundary per actual packed attention segment.

    Position-ID resets define the same segment partition used by the segmented
    packer. Without position IDs, each contiguous valid span is one segment.
    The keyed SHA-256 draws consume neither the global CPU nor device RNG stream.
    """

    _validate_plan_tensors(loss_mask, attention_mask, position_ids)
    if not 0.0 < float(min_fraction) <= float(max_fraction) < 1.0:
        raise ValueError("token-edit prefix fractions must satisfy 0 < min <= max < 1")
    for name, value in (("boundary_seed", boundary_seed), ("rank", rank), ("plan_index", plan_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"token-edit prefix {name} must be a non-negative integer")

    device = loss_mask.device
    loss_rows = loss_mask.detach().cpu().tolist()
    attention_rows = attention_mask.detach().cpu().tolist()
    position_rows = position_ids.detach().cpu().tolist() if position_ids is not None else None
    prefix_cpu = torch.zeros(loss_mask.shape, dtype=torch.bool)
    suffix_cpu = torch.zeros(loss_mask.shape, dtype=torch.bool)
    segments: list[TokenEditPrefixSegment] = []
    resolution = 1 << 53

    for row, attention in enumerate(attention_rows):
        positions = position_rows[row] if position_rows is not None else None
        for ordinal, (start, end) in enumerate(_segment_spans(attention, positions)):
            lossable_positions = [index for index in range(start, end) if loss_rows[row][index]]
            u53 = _segment_uniform_u53(boundary_seed, rank, plan_index, row, ordinal)
            fraction = float(min_fraction) + (float(max_fraction) - float(min_fraction)) * (u53 / resolution)
            proposed_prefix_tokens = int(math.floor(fraction * (end - start) + 0.5))
            splittable = len(lossable_positions) >= 2
            if splittable:
                minimum_prefix_end = lossable_positions[0] + 1
                maximum_prefix_end = lossable_positions[-1]
                prefix_end = min(max(start + proposed_prefix_tokens, minimum_prefix_end), maximum_prefix_end)
                prefix_cpu[row, start:prefix_end] = True
                suffix_cpu[row, prefix_end:end] = torch.tensor(loss_rows[row][prefix_end:end], dtype=torch.bool)
            else:
                # No valid two-sided objective exists. C treats the whole tiny
                # segment as immutable context; B/C assign it no suffix loss.
                prefix_end = end
                prefix_cpu[row, start:end] = True
            prefix_lossable = sum(loss_rows[row][start:prefix_end])
            suffix_lossable = sum(loss_rows[row][prefix_end:end])
            if splittable and (prefix_lossable < 1 or suffix_lossable < 1):
                raise RuntimeError("token-edit prefix boundary did not preserve one lossable token on each side")
            segments.append(
                TokenEditPrefixSegment(
                    row=row,
                    ordinal=ordinal,
                    start=start,
                    end=end,
                    prefix_end=prefix_end,
                    sampled_fraction_u53=u53,
                    lossable_tokens=len(lossable_positions),
                    prefix_lossable_tokens=prefix_lossable,
                    suffix_lossable_tokens=suffix_lossable,
                    splittable=splittable,
                )
            )

    payload = {
        "schema_version": TOKEN_EDIT_PREFIX_PLAN_SCHEMA_VERSION,
        "boundary_seed": boundary_seed,
        "rank": rank,
        "plan_index": plan_index,
        "shape": list(loss_mask.shape),
        "min_fraction": float(min_fraction),
        "max_fraction": float(max_fraction),
        "segments": [segment.receipt() for segment in segments],
    }
    plan_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return TokenEditPrefixPlan(
        prefix_mask=prefix_cpu.to(device=device),
        suffix_lossable_mask=suffix_cpu.to(device=device),
        segments=tuple(segments),
        min_fraction=float(min_fraction),
        max_fraction=float(max_fraction),
        boundary_seed=boundary_seed,
        rank=rank,
        plan_index=plan_index,
        plan_sha256=plan_sha256,
    )


def prepare_token_edit_prefix_inputs(
    x_0: torch.Tensor,
    z_t: torch.Tensor,
    active: torch.Tensor,
    plan: TokenEditPrefixPlan,
    *,
    canvas_mode: str,
    loss_scope: str,
) -> TokenEditPrefixInputs:
    """Prepare A/B/C draft inputs while retaining all-active correction logits."""

    if x_0.shape != z_t.shape or active.shape != x_0.shape or active.dtype != torch.bool:
        raise ValueError("token-edit prefix clean, corrupted, and active tensors must share shape")
    if plan.prefix_mask.shape != x_0.shape or plan.suffix_lossable_mask.shape != x_0.shape:
        raise ValueError("token-edit prefix plan must match the token tensors")
    correction_target_indices = active.reshape(-1).nonzero(as_tuple=True)[0]
    if loss_scope == "all_active":
        if canvas_mode != "fallible_draft":
            raise ValueError("all-active prefix loss requires a fallible draft canvas")
        correction_loss_mask = torch.ones_like(correction_target_indices, dtype=torch.bool)
        correction_lossable_tokens = plan.lossable_tokens
        draft_input_tokens = z_t
        draft_target_indices = correction_target_indices
        restored_prefix_active_tokens = 0
    elif loss_scope == "suffix_active":
        correction_loss_mask = plan.suffix_lossable_mask.reshape(-1).index_select(0, correction_target_indices)
        correction_lossable_tokens = int(plan.suffix_lossable_mask.sum().item())
        if canvas_mode == "fallible_draft":
            draft_input_tokens = z_t
            draft_target_indices = correction_target_indices
            restored_prefix_active_tokens = 0
        elif canvas_mode == "clean":
            draft_input_tokens = torch.where(plan.prefix_mask, x_0, z_t)
            draft_target_indices = correction_target_indices[correction_loss_mask]
            restored_prefix_active_tokens = int((active & plan.prefix_mask).sum().item())
        else:
            raise ValueError(f"unsupported token-edit prefix canvas mode: {canvas_mode!r}")
    else:
        raise ValueError(f"unsupported token-edit prefix loss scope: {loss_scope!r}")
    return TokenEditPrefixInputs(
        draft_input_tokens=draft_input_tokens,
        draft_target_indices=draft_target_indices,
        correction_target_indices=correction_target_indices,
        correction_loss_target_mask=correction_loss_mask,
        correction_lossable_tokens=correction_lossable_tokens,
        restored_prefix_active_tokens=restored_prefix_active_tokens,
    )


def align_token_edit_prefix_draft(
    x_0: torch.Tensor,
    inputs: TokenEditPrefixInputs,
    *,
    draft_tokens: torch.Tensor,
    draft_predictions: torch.Tensor,
    draft_correct: torch.Tensor,
    drafted: torch.Tensor,
) -> TokenEditPrefixDraft:
    """Align a suffix-only C draft with the all-active correction output rows."""

    full_indices = inputs.correction_target_indices
    draft_indices = inputs.draft_target_indices
    if draft_predictions.shape != draft_indices.shape or draft_correct.shape != draft_indices.shape:
        raise ValueError("token-edit prefix draft observations must align with draft targets")
    if drafted.shape != draft_indices.shape:
        raise ValueError("token-edit prefix drafted labels must align with draft targets")
    if torch.equal(full_indices, draft_indices):
        return TokenEditPrefixDraft(draft_tokens, draft_predictions, draft_correct, drafted)
    full_targets = x_0.reshape(-1).index_select(0, full_indices)
    predictions = full_targets.clone()
    correct = torch.ones_like(full_targets, dtype=torch.bool)
    full_drafted = torch.zeros_like(full_targets, dtype=torch.bool)
    offsets = torch.searchsorted(full_indices, draft_indices)
    if draft_indices.numel() and not torch.equal(full_indices.index_select(0, offsets), draft_indices):
        raise ValueError("token-edit prefix draft targets must be a subset of correction targets")
    predictions.index_copy_(0, offsets, draft_predictions)
    correct.index_copy_(0, offsets, draft_correct)
    full_drafted.index_copy_(0, offsets, drafted)
    return TokenEditPrefixDraft(draft_tokens, predictions, correct, full_drafted)


def token_edit_prefix_plan_receipt(
    plan: TokenEditPrefixPlan,
    *,
    step: int,
    microbatch: int,
    canvas_mode: str,
    loss_scope: str,
    all_active_targets: int,
    draft_targets: int,
    correction_loss_targets: int,
    correction_lossable_tokens: int,
    restored_prefix_active_tokens: int,
) -> dict[str, object]:
    return {
        "schema_version": TOKEN_EDIT_PREFIX_PLAN_SCHEMA_VERSION,
        "step": step,
        "microbatch": microbatch,
        "global_microbatch_index": plan.plan_index,
        "rank": plan.rank,
        "boundary_seed": plan.boundary_seed,
        "min_fraction": plan.min_fraction,
        "max_fraction": plan.max_fraction,
        "plan_sha256": plan.plan_sha256,
        "canvas_mode": canvas_mode,
        "loss_scope": loss_scope,
        "segment_count": plan.segment_count,
        "splittable_segment_count": plan.splittable_segment_count,
        "skipped_segment_count": plan.segment_count - plan.splittable_segment_count,
        "prefix_lossable_tokens": plan.prefix_lossable_tokens,
        "suffix_lossable_tokens": plan.suffix_lossable_tokens,
        "all_active_targets": all_active_targets,
        "draft_targets": draft_targets,
        "correction_output_targets": all_active_targets,
        "correction_loss_targets": correction_loss_targets,
        "correction_lossable_tokens": correction_lossable_tokens,
        "restored_prefix_active_tokens": restored_prefix_active_tokens,
        "segments": [segment.receipt() for segment in plan.segments],
    }


def prepare_token_edit_prefix_microbatch(session: TrainingSession) -> None:
    """Attach one stateless packed-segment plan and its A/B/C tensor views."""

    plan = sample_token_edit_prefix_plan(
        session.loss_mask,
        session.attention_mask,
        getattr(session.batch, "position_ids", None),
        min_fraction=float(session.config.train.token_edit_prefix_min_fraction),
        max_fraction=float(session.config.train.token_edit_prefix_max_fraction),
        boundary_seed=int(session.config.train.token_edit_prefix_boundary_seed),
        rank=session.rank,
        plan_index=session.global_microbatch_idx,
    )
    inputs = prepare_token_edit_prefix_inputs(
        session.x_0,
        session.z_t,
        session.active,
        plan,
        canvas_mode=session.config.train.token_edit_prefix_canvas_mode,
        loss_scope=session.config.train.token_edit_prefix_loss_scope,
    )
    expected_targets = session.active.reshape(-1).nonzero(as_tuple=True)[0]
    if not torch.equal(inputs.correction_target_indices, expected_targets):
        raise RuntimeError("token-edit prefix intervention changed correction output targets")
    session.token_edit_prefix_plan = plan
    session.token_edit_prefix_inputs = inputs


def apply_token_edit_prefix_draft_result(
    session: TrainingSession,
    draft_result: SelfGeneratedDraftResult,
    *,
    step: int,
    microbatch: int,
) -> None:
    """Align one prefix draft, attach suffix scope, and persist its replay receipt."""

    plan = session.token_edit_prefix_plan
    inputs = session.token_edit_prefix_inputs
    if not isinstance(plan, TokenEditPrefixPlan) or not isinstance(inputs, TokenEditPrefixInputs):
        raise RuntimeError("token-edit prefix draft requires a prepared boundary plan")
    aligned = align_token_edit_prefix_draft(
        session.x_0,
        inputs,
        draft_tokens=draft_result.tokens,
        draft_predictions=draft_result.predictions,
        draft_correct=draft_result.correct,
        drafted=draft_result.drafted,
    )
    session.z_t = aligned.tokens
    session.self_generated_draft_predictions = aligned.predictions
    session.self_generated_draft_correct = aligned.correct
    session.self_generated_drafted = aligned.drafted
    target_mask = inputs.correction_loss_target_mask
    session.token_edit_correction_loss_target_mask = target_mask
    correction_loss_targets = int(target_mask.sum().item())
    session.step_token_edit_prefix_correction_loss_targets += correction_loss_targets
    if (
        session.config.train.token_edit_prefix_loss_scope == "all_active"
        and inputs.correction_lossable_tokens != session.lossable_tokens
    ):
        raise RuntimeError("all-active token-edit prefix denominator changed")
    append_jsonl(
        session.token_edit_prefix_plan_path,
        token_edit_prefix_plan_receipt(
            plan,
            step=step,
            microbatch=microbatch,
            canvas_mode=session.config.train.token_edit_prefix_canvas_mode,
            loss_scope=session.config.train.token_edit_prefix_loss_scope,
            all_active_targets=int(session.target_indices.numel()),
            draft_targets=int(inputs.draft_target_indices.numel()),
            correction_loss_targets=correction_loss_targets,
            correction_lossable_tokens=inputs.correction_lossable_tokens,
            restored_prefix_active_tokens=inputs.restored_prefix_active_tokens,
        ),
    )


def prepare_self_generated_edit_draft_with_prefix(
    session: TrainingSession,
    *,
    step: int,
    microbatch: int,
) -> None:
    """Run one detached draft while preserving default and prefix-specific semantics."""

    if not isinstance(session.t_val, torch.Tensor):
        raise RuntimeError("self-generated token editing requires per-row MDLM noise times")
    inputs = session.token_edit_prefix_inputs
    if inputs is None and uses_trajectory_canvas(session.config):
        with session.autocast_ctx:
            result = build_trajectory_argmax_draft(
                session.self_generated_draft_model,
                session.config,
                session.x_0,
                session.z_t,
                session.forward_attention_mask,
                session.target_indices,
                session.generation_blocked_token_ids,
                session.noise_level,
                position_ids=getattr(session.batch, "position_ids", None),
            )
        session.z_t = result.tokens
        session.self_generated_draft_predictions = result.predictions
        session.self_generated_draft_correct = result.correct
        session.self_generated_drafted = result.drafted
        session.token_edit_correction_loss_target_mask = result.correction_loss_target_mask
        session.token_edit_trajectory_model_forwards = result.model_forwards
        session.token_edit_trajectory_committed_targets = result.committed_targets
        session.token_edit_trajectory_remaining_targets = result.remaining_targets
        return
    draft_input_tokens = session.z_t if inputs is None else inputs.draft_input_tokens
    draft_target_indices = session.target_indices if inputs is None else inputs.draft_target_indices
    with session.autocast_ctx:
        result = build_self_generated_argmax_draft(
            session.self_generated_draft_model,
            session.config,
            session.x_0,
            draft_input_tokens,
            session.forward_attention_mask,
            draft_target_indices,
            session.generation_blocked_token_ids,
            session.noise_level,
            position_ids=getattr(session.batch, "position_ids", None),
            draft_fraction=float(session.config.train.token_edit_mixed_canvas_draft_fraction),
        )
        if inputs is not None:
            apply_token_edit_prefix_draft_result(session, result, step=step, microbatch=microbatch)
        else:
            session.z_t = result.tokens
            session.self_generated_draft_predictions = result.predictions
            session.self_generated_draft_correct = result.correct
            session.self_generated_drafted = result.drafted
            session.self_generated_draft_m2t_logits = result.m2t_logits
            session.self_generated_draft_target_probabilities = result.target_probabilities
        if _uses_mixed_token_edit_canvas(session.config) or _uses_full_draft_token_edit_provenance(session.config):
            session.token_edit_input_provenance = mixed_canvas_input_provenance(
                session.z_t,
                session.target_indices,
                result.drafted,
                confidence=result.confidence,
                confidence_bins=int(session.config.model.token_edit_provenance_confidence_bins),
            )


def token_edit_prefix_target_mean(values: torch.Tensor, target_mask: torch.Tensor | None) -> torch.Tensor:
    """Return the ordinary mean or the exact possibly-empty suffix-target mean."""

    if target_mask is None:
        return values.mean()
    if target_mask.dtype != torch.bool or target_mask.ndim != 1 or target_mask.shape != values.shape:
        raise ValueError("token-edit prefix target mask must be boolean and match objective values")
    return values[target_mask].mean() if bool(target_mask.any().item()) else values[:0].sum()


def token_edit_prefix_target_means(
    values: list[torch.Tensor],
    target_mask: torch.Tensor | None,
) -> list[torch.Tensor]:
    """Apply the same optional suffix target scope to auxiliary objectives."""

    return [token_edit_prefix_target_mean(value, target_mask) for value in values]


def apply_token_edit_prefix_loss_weights(session: TrainingSession) -> None:
    """Apply suffix objective weights and its exact lossable-token denominator."""

    if session.token_edit_prefix_inputs is None or session.token_edit_correction_loss_target_mask is None:
        return
    session.token_weights, session.prefix_loss_denominator = apply_token_edit_prefix_loss_scope(
        session.token_weights,
        session.token_edit_prefix_inputs,
    )
    session.token_edit_loss_denominator = session.prefix_loss_denominator
