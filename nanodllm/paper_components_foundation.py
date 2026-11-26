"""Per-microbatch F1–F4 operators; no optimizer, loader, or default mutations.

F1: MDLM SUBS (2406.07524v2 sections 3.2–3.5, D.3), its supported
time-agnostic variant, stratified row masking rates and zero MASK probability.
F2: LLaDA eq. (3), section 2.2 (2502.09992v3), IID row rates and 1% random
lengths. Released GUIDELINES.md at 9182493720ed723ef8031210d85959364e51cbe0
uses a whole-microbatch crop, no inserted EOS, and rates in [1e-3, 1]. The
context cap is adapted to 1024; source-position accounting still includes crops.
F3/F4: BD3 eq. (8), sections 3.2, 5 and appendix C.4 (2503.09573v3): independent
block corruption, SUBS and block-causal clean states, not causal clean states.

All arms preserve the supplied protected-prefix/document masks. Common plain
D4/Muon, segmented BOS handling and source-position TPP are recipe adaptations.
These routines do not purport to reproduce the original data/optimizer/scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from .paper_components_noise import ClippedScheduleState, batch_mean_variance, loss_variance_score, mask_rates
from .paper_runtime.data import DLMDataBatch
from .paper_runtime.noise import MDLMNoise
from .paper_runtime.training_bd3lm import bd3lm_block_times_from_candidates, bd3lm_concat_inputs


@dataclass(frozen=True)
class FoundationLoss:
    loss: torch.Tensor
    token_losses: torch.Tensor
    block_losses: torch.Tensor | None
    telemetry: dict[str, float | int]


def _state(policy: dict) -> ClippedScheduleState:
    state = ClippedScheduleState(
        intervals=tuple(tuple(pair) for pair in policy.get("noise_intervals", ClippedScheduleState().intervals)),
        interval_steps=policy.get("noise_update_interval", 352),
    )
    if policy.get("method_state"):
        state.load_state_dict(policy["method_state"])
    return state


def initialize_policy(policy: dict) -> None:
    """Install or validate JSON-only F4 state before checkpointing/training."""
    policy["method_state"] = _state(policy).state_dict()


def _check_model(model) -> int:
    config = model.config
    if config.causal or config.time_conditioning or config.output_factorization != "flat":
        raise ValueError("foundations require a time-agnostic, noncausal flat-output backbone")
    return int(config.mask_token_id)


def _crop_llada(batch: DLMDataBatch, *, probability: float, generator: torch.Generator) -> tuple[DLMDataBatch, int]:
    if not 0 <= probability <= 1:
        raise ValueError("variable-length probability must be in [0, 1]")
    rows, length = batch.x_0.shape
    device = batch.x_0.device
    if float(torch.rand((), device=device, generator=generator)) >= probability:
        return batch, 0
    crop_length = int(torch.randint(1, length + 1, (), device=device, generator=generator))
    boundary = batch.document_boundary_mask
    if boundary is not None:
        boundary = boundary[:, :crop_length, :crop_length]
    return replace(
        batch,
        **{
            name: value[:, :crop_length]
            for name in ("x_0", "loss_mask", "attention_mask", "position_ids", "document_start_mask")
            if (value := getattr(batch, name)) is not None
        },
        document_boundary_mask=boundary,
    ), rows


def _visibility(batch: DLMDataBatch) -> torch.Tensor:
    valid = batch.attention_mask
    visibility = valid[:, :, None] & valid[:, None, :]
    if batch.document_boundary_mask is not None:
        visibility &= batch.document_boundary_mask
    return visibility


def _block_structure(batch: DLMDataBatch, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size <= 0:
        raise ValueError("block size must be positive")
    rows, length = batch.x_0.shape
    positions = batch.position_ids
    if positions is None:
        positions = torch.arange(length, device=batch.x_0.device)[None, :].expand(rows, -1)
    starts = positions.remainder(block_size) == 0
    starts[:, 0] = True
    # Flattened owners are globally unique even when document positions reset.
    owners = starts.reshape(-1).cumsum(0).reshape(rows, length) - 1
    return starts, owners


def subs_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, *, mask_token_id: int, subs: bool) -> torch.Tensor:
    """SUBS is masked-target CE with MASK excluded; visible-token loss is zero."""
    if targets.numel() and bool((targets == mask_token_id).any()):
        raise ValueError("clean MASK targets cannot use the absorbing foundation objective")
    if not 0 <= mask_token_id < logits.shape[-1]:
        raise ValueError("MASK id is outside the model output vocabulary")
    logits = logits.float()
    if subs:
        forbidden = torch.arange(logits.shape[-1], device=logits.device) == mask_token_id
        logits = logits.masked_fill(forbidden, -torch.inf)
    return F.cross_entropy(logits, targets, reduction="none")


def foundation_loss(
    model,
    batch: DLMDataBatch,
    *,
    arm: str,
    generator: torch.Generator,
    policy: dict,
    bounds: tuple[float, float] | None = None,
) -> FoundationLoss:
    """Differentiable objective; caller owns autocast. Bounds override is for tuning."""
    if arm not in {"F1", "F2", "F3", "F4"}:
        raise ValueError(f"unknown scratch foundation arm: {arm}")
    mask_id = _check_model(model)
    source_positions = batch.x_0.numel()
    cropped_rows = 0
    if arm == "F2":
        batch, cropped_rows = _crop_llada(
            batch, probability=float(policy.get("variable_length_probability", 0.01)), generator=generator
        )
    x_0 = batch.x_0
    eligible = batch.loss_mask & batch.attention_mask
    if bool(((x_0 == mask_id) & eligible).any()):
        raise ValueError("foundation clean targets include MASK")
    rows, length = x_0.shape
    block_size = int(policy.get("block_size", 32))
    blocked = arm in {"F3", "F4"}
    reference_bounds = (float(policy.get("llada_time_epsilon", 0.001)) if arm == "F2" else 0.0, 1.0)
    low, high = bounds if bounds is not None else (_state(policy).bounds if arm == "F4" else reference_bounds)
    if not blocked and (low, high) != reference_bounds:
        raise ValueError("F1/F2 require their declared reference masking interval")
    if blocked:
        starts, owners = _block_structure(batch, block_size)
        draws = mask_rates(
            int(starts.sum()), low=low, high=high, stratified=True, device=x_0.device, generator=generator
        )
        candidates = torch.zeros_like(x_0, dtype=torch.float32)
        candidates[starts] = draws
        times = bd3lm_block_times_from_candidates(candidates, block_size=block_size, position_ids=batch.position_ids)
    else:
        times = mask_rates(rows, low=low, high=high, stratified=arm == "F1", device=x_0.device, generator=generator)[
            :, None
        ].expand_as(x_0)
    z_t, selected = MDLMNoise(mask_id).corrupt(
        x_0, times, eligible, uniforms=torch.rand(x_0.shape, device=x_0.device, generator=generator)
    )
    active = selected.bool()
    flat_targets = active.reshape(-1).nonzero(as_tuple=True)[0]
    flat_rows = torch.div(flat_targets, length, rounding_mode="floor")
    positions = batch.position_ids
    model_tokens, attention = z_t, _visibility(batch)
    logits_indices = flat_targets
    if blocked:
        model_tokens, attention, positions = bd3lm_concat_inputs(
            z_t, x_0, attention, block_size=block_size, clean_stream_mode="block_causal", position_ids=positions
        )
        logits_indices = flat_targets + flat_rows * length
    logits = model(model_tokens, logits_indices=logits_indices, attention_mask=attention, position_ids=positions)
    targets = x_0.reshape(-1).index_select(0, flat_targets)
    ce = subs_cross_entropy(logits, targets, mask_token_id=mask_id, subs=arm != "F2")
    target_times = times.reshape(-1).index_select(0, flat_targets)
    weighted = ce / target_times
    denominator = eligible.sum().clamp_min(1)
    loss = weighted.sum() / denominator
    token_losses = weighted.new_zeros(x_0.numel()).scatter(0, flat_targets, weighted).view_as(x_0)
    block_losses = None
    if blocked:
        # Native score is block-sum / BL, including zero loss on visible/protected
        # positions. Document-end partial blocks are zero-padded for this score.
        block_losses = weighted.new_zeros(draws.numel()).scatter_add(0, owners.reshape(-1), token_losses.reshape(-1))
        counts = torch.zeros_like(block_losses).scatter_add(0, owners.reshape(-1), eligible.reshape(-1).float())
        block_losses = (block_losses / block_size)[counts > 0]
    telemetry = {
        "loss": float(loss.detach()),
        "raw_masked_ce": float(ce.detach().mean()) if ce.numel() else 0.0,
        "source_positions": source_positions,
        "processed_positions": model_tokens.numel(),
        "lossable_tokens": int(eligible.sum()),
        "active_tokens": int(active.sum()),
        "valid_tokens": int(batch.attention_mask.sum()),
        "variable_length_rows": cropped_rows,
        "mask_rate_mean": float(times.mean()),
        "noise_low": low,
        "noise_high": high,
        "clipped_objective": int((low, high) != (0.0, 1.0)),
        "forwards": 1,
    }
    return FoundationLoss(loss, token_losses, block_losses, telemetry)


def backward_microbatch(
    model,
    batch: DLMDataBatch,
    *,
    arm: str,
    step: int,
    num_steps: int,
    generator: torch.Generator,
    accumulation: int = 1,
    policy: dict,
) -> dict[str, float | int]:
    """Backward once, normalized by accumulation; never step or clear gradients."""
    if accumulation < 1 or not 0 <= step < num_steps:
        raise ValueError("invalid accumulation or zero-based optimizer step")
    if arm == "F4":
        if not policy.get("method_state"):
            raise ValueError("F4 requires initialized and checkpointed clipped-schedule state")
        if _state(policy).should_tune(step):
            raise ValueError("F4 noise selection is due; call prepare_update before training")
    result = foundation_loss(model, batch, arm=arm, generator=generator, policy=policy)
    (result.loss / accumulation).backward()
    return result.telemetry


def _to_device(batch: DLMDataBatch, device: torch.device) -> DLMDataBatch:
    return replace(
        batch,
        **{
            name: value.to(device)
            for name in (
                "x_0",
                "loss_mask",
                "attention_mask",
                "position_ids",
                "document_boundary_mask",
                "document_start_mask",
            )
            if (value := getattr(batch, name)) is not None
        },
    )


@torch.no_grad()
def prepare_update(
    model,
    *,
    step: int,
    num_steps: int,
    generator: torch.Generator,
    policy: dict,
    calibration_batches: Sequence[DLMDataBatch],
) -> dict[str, float | int]:
    """F4-only update-boundary hook; no-op off cadence, JSON state committed last.

    Caller supplies frozen calibration TRAIN batches and a separately checkpointed
    tuning RNG. Shared examples across intervals are deliberate; no held-out test
    loss or generation score selects the noise bounds. Caller owns autocast.
    """
    if not 0 <= step < num_steps:
        raise ValueError("invalid optimizer step")
    initialize_policy(policy)
    state = _state(policy)
    if not state.should_tune(step):
        return {"noise_tuned": 0, "calibration_forwards": 0, "calibration_processed_positions": 0}
    if len(calibration_batches) < 2:
        raise ValueError("F4 needs at least two frozen calibration TRAIN batches")
    was_training = model.training
    model.eval()
    scores, batch_variances = [], []
    processed = 0
    device = next(model.parameters()).device
    try:
        for bounds in state.intervals:
            observations = []
            for batch in calibration_batches:
                result = foundation_loss(
                    model, _to_device(batch, device), arm="F4", generator=generator, policy=policy, bounds=bounds
                )
                observations.append(result.block_losses.cpu())
                processed += result.telemetry["processed_positions"]
            scores.append(loss_variance_score(observations))
            batch_variances.append(batch_mean_variance(observations))
    finally:
        model.train(was_training)
    forwards = len(state.intervals) * len(calibration_batches)
    state.select(scores, step=step, forward_count=forwards)
    policy["method_state"] = state.state_dict()
    return {
        "noise_tuned": 1,
        "noise_low": state.bounds[0],
        "noise_high": state.bounds[1],
        "calibration_forwards": forwards,
        "calibration_processed_positions": processed,
        **{f"noise_candidate_{i}_block_variance": score for i, score in enumerate(scores)},
        **{f"noise_candidate_{i}_batch_mean_variance": score for i, score in enumerate(batch_variances)},
    }
