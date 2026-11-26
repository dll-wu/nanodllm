"""Marginal transition strata for the paired M2T/T2T evaluation panel."""

from __future__ import annotations

import torch

DRAFT_ERROR_BAND_LABELS = (
    "exact",
    "gt_000_le_025",
    "gt_025_le_050",
    "gt_050_le_075",
    "gt_075_le_100",
)
POSITION_DECILE_LABELS = tuple(f"{index:02d}" for index in range(1, 11))
CONFIDENCE_DECILE_LABELS = tuple(f"{index:02d}" for index in range(1, 11))

_COUNT_FIELDS = (
    "eligible_count",
    "c_to_c_count",
    "c_to_w_count",
    "w_to_c_count",
    "w_to_w_count",
)


def _frequency_band_labels(band_count: int) -> tuple[str, ...]:
    if band_count <= 0:
        return ()
    return tuple("00_special" if index == 0 else f"{index:02d}" for index in range(band_count))


def _new_accumulator() -> dict[str, int | float]:
    return {
        "eligible_count": 0,
        "c_to_c_count": 0,
        "c_to_w_count": 0,
        "w_to_c_count": 0,
        "w_to_w_count": 0,
        "draft_confidence_sum": 0.0,
        "correction_confidence_sum": 0.0,
    }


def new_transition_strata(frequency_band_count: int) -> dict[str, dict[str, dict[str, int | float]]]:
    axis_labels = {
        "draft_error_band": DRAFT_ERROR_BAND_LABELS,
        "position_decile": POSITION_DECILE_LABELS,
        "draft_confidence_decile": CONFIDENCE_DECILE_LABELS,
        "correction_confidence_decile": CONFIDENCE_DECILE_LABELS,
    }
    frequency_labels = _frequency_band_labels(int(frequency_band_count))
    if frequency_labels:
        axis_labels["token_frequency_band"] = frequency_labels
    return {axis: {label: _new_accumulator() for label in labels} for axis, labels in axis_labels.items()}


def relative_position_decile_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.dtype != torch.bool or attention_mask.dim() != 2:
        raise ValueError("attention_mask must be a boolean [batch, sequence] tensor")
    sequence_length = int(attention_mask.shape[1])
    columns = torch.arange(sequence_length, device=attention_mask.device).view(1, -1)
    denominators = attention_mask.to(torch.long).sum(dim=1).sub(1).clamp_min(1).view(-1, 1)
    relative = columns.float() / denominators.float()
    return torch.clamp((relative * 10).to(torch.long), min=0, max=9)


def confidence_decile_ids(confidence: torch.Tensor) -> torch.Tensor:
    if not confidence.is_floating_point():
        raise ValueError("confidence must be floating point")
    if bool((~torch.isfinite(confidence)).any().item()) or bool(((confidence < 0) | (confidence > 1)).any().item()):
        raise ValueError("confidence values must be finite and lie in [0, 1]")
    return torch.clamp((confidence * 10).to(torch.long), min=0, max=9)


def draft_error_band_id(draft_correct: torch.Tensor, eligible: torch.Tensor) -> int | None:
    eligible_count = int(eligible.sum().item())
    if eligible_count == 0:
        return None
    error_count = int((eligible & ~draft_correct).sum().item())
    if error_count == 0:
        return 0
    error_rate = error_count / eligible_count
    if error_rate <= 0.25:
        return 1
    if error_rate <= 0.50:
        return 2
    if error_rate <= 0.75:
        return 3
    return 4


def _observation(
    draft_correct: torch.Tensor,
    final_correct: torch.Tensor,
    eligible: torch.Tensor,
    draft_confidence: torch.Tensor,
    correction_confidence: torch.Tensor,
) -> dict[str, int | float]:
    return {
        "eligible_count": int(eligible.sum().item()),
        "c_to_c_count": int((eligible & draft_correct & final_correct).sum().item()),
        "c_to_w_count": int((eligible & draft_correct & ~final_correct).sum().item()),
        "w_to_c_count": int((eligible & ~draft_correct & final_correct).sum().item()),
        "w_to_w_count": int((eligible & ~draft_correct & ~final_correct).sum().item()),
        "draft_confidence_sum": float(draft_confidence[eligible].double().sum().item()),
        "correction_confidence_sum": float(correction_confidence[eligible].double().sum().item()),
    }


def _add_observation(accumulator: dict[str, int | float], observation: dict[str, int | float]) -> None:
    for field in _COUNT_FIELDS:
        accumulator[field] = int(accumulator[field]) + int(observation[field])
    for field in ("draft_confidence_sum", "correction_confidence_sum"):
        accumulator[field] = float(accumulator[field]) + float(observation[field])


def _axis_assignments(
    strata: dict[str, dict[str, dict[str, int | float]]],
    *,
    draft_correct: torch.Tensor,
    eligible: torch.Tensor,
    position_deciles: torch.Tensor,
    draft_confidence: torch.Tensor,
    correction_confidence: torch.Tensor,
    frequency_band_ids: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    draft_error = draft_error_band_id(draft_correct, eligible)
    assignments = {
        "draft_error_band": torch.full_like(position_deciles, -1 if draft_error is None else draft_error),
        "position_decile": position_deciles,
        "draft_confidence_decile": confidence_decile_ids(draft_confidence),
        "correction_confidence_decile": confidence_decile_ids(correction_confidence),
    }
    if "token_frequency_band" in strata:
        if frequency_band_ids is None:
            raise ValueError("token-frequency strata require frequency-band IDs")
        assignments["token_frequency_band"] = frequency_band_ids
    return assignments


def update_transition_strata(
    strata: dict[str, dict[str, dict[str, int | float]]],
    *,
    draft_correct: torch.Tensor,
    final_correct: torch.Tensor,
    eligible: torch.Tensor,
    position_deciles: torch.Tensor,
    draft_confidence: torch.Tensor,
    correction_confidence: torch.Tensor,
    frequency_band_ids: torch.Tensor | None,
) -> dict[str, dict[str, dict[str, int | float]]]:
    """Update marginal axes for one sequence and return its nonempty observations."""

    reference_shape = eligible.shape
    for name, tensor in (
        ("draft_correct", draft_correct),
        ("final_correct", final_correct),
        ("position_deciles", position_deciles),
        ("draft_confidence", draft_confidence),
        ("correction_confidence", correction_confidence),
    ):
        if tensor.shape != reference_shape:
            raise ValueError(f"{name} must match eligible shape")
    if frequency_band_ids is not None and frequency_band_ids.shape != reference_shape:
        raise ValueError("frequency_band_ids must match eligible shape")

    assignments = _axis_assignments(
        strata,
        draft_correct=draft_correct,
        eligible=eligible,
        position_deciles=position_deciles,
        draft_confidence=draft_confidence,
        correction_confidence=correction_confidence,
        frequency_band_ids=frequency_band_ids,
    )
    row_records: dict[str, dict[str, dict[str, int | float]]] = {}
    for axis, bands in strata.items():
        assignment = assignments[axis]
        axis_record: dict[str, dict[str, int | float]] = {}
        for band_index, (label, accumulator) in enumerate(bands.items()):
            band_mask = eligible & (assignment == band_index)
            if not bool(band_mask.any().item()):
                continue
            observation = _observation(
                draft_correct,
                final_correct,
                band_mask,
                draft_confidence,
                correction_confidence,
            )
            _add_observation(accumulator, observation)
            axis_record[label] = observation
        row_records[axis] = axis_record
    return row_records


def merge_transition_strata(
    destination: dict[str, dict[str, dict[str, int | float]]],
    source: dict[str, dict[str, dict[str, int | float]]],
) -> None:
    if destination.keys() != source.keys():
        raise ValueError("transition-strata axes do not match")
    for axis, destination_bands in destination.items():
        source_bands = source[axis]
        if destination_bands.keys() != source_bands.keys():
            raise ValueError(f"transition-strata bands do not match for {axis}")
        for label, destination_accumulator in destination_bands.items():
            _add_observation(destination_accumulator, source_bands[label])


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def transition_stratum_metrics(accumulator: dict[str, int | float]) -> dict[str, int | float | bool]:
    eligible = int(accumulator["eligible_count"])
    c_to_c = int(accumulator["c_to_c_count"])
    c_to_w = int(accumulator["c_to_w_count"])
    w_to_c = int(accumulator["w_to_c_count"])
    w_to_w = int(accumulator["w_to_w_count"])
    initial_correct = c_to_c + c_to_w
    initial_wrong = w_to_c + w_to_w
    final_correct = c_to_c + w_to_c
    draft_confidence_mean = _ratio(accumulator["draft_confidence_sum"], eligible)
    correction_confidence_mean = _ratio(accumulator["correction_confidence_sum"], eligible)
    draft_accuracy = _ratio(initial_correct, eligible)
    final_accuracy = _ratio(final_correct, eligible)
    return {
        **accumulator,
        "initial_correct_count": initial_correct,
        "initial_wrong_count": initial_wrong,
        "final_correct_count": final_correct,
        "draft_error_count": initial_wrong,
        "final_error_count": c_to_w + w_to_w,
        "net_repair_count": w_to_c - c_to_w,
        "correct_token_retention_rate": _ratio(c_to_c, initial_correct),
        "damage_rate": _ratio(c_to_w, initial_correct),
        "repair_rate": _ratio(w_to_c, initial_wrong),
        "draft_error_rate": _ratio(initial_wrong, eligible),
        "final_error_rate": _ratio(c_to_w + w_to_w, eligible),
        "net_repair_gain": _ratio(w_to_c - c_to_w, eligible),
        "draft_confidence_mean": draft_confidence_mean,
        "correction_confidence_mean": correction_confidence_mean,
        "draft_accuracy": draft_accuracy,
        "final_accuracy": final_accuracy,
        "draft_calibration_gap": draft_confidence_mean - draft_accuracy,
        "correction_calibration_gap": correction_confidence_mean - final_accuracy,
    }


def flatten_transition_strata_metrics(
    prefix: str,
    strata: dict[str, dict[str, dict[str, int | float]]],
) -> dict[str, int | float | bool | str]:
    metrics: dict[str, int | float | bool | str] = {}
    for axis, bands in strata.items():
        metrics[f"{prefix}{axis}_stratum_count"] = len(bands)
        eligible_total = sum(int(accumulator["eligible_count"]) for accumulator in bands.values())
        draft_ece = 0.0
        correction_ece = 0.0
        for label, accumulator in bands.items():
            values = transition_stratum_metrics(accumulator)
            metrics.update({f"{prefix}{axis}_{label}_{key}": value for key, value in values.items()})
            if eligible_total:
                weight = int(accumulator["eligible_count"]) / eligible_total
                draft_ece += weight * abs(float(values["draft_calibration_gap"]))
                correction_ece += weight * abs(float(values["correction_calibration_gap"]))
        metrics[f"{prefix}{axis}_draft_weighted_absolute_group_calibration_gap"] = draft_ece
        metrics[f"{prefix}{axis}_correction_weighted_absolute_group_calibration_gap"] = correction_ece
        if axis == "draft_confidence_decile":
            metrics[f"{prefix}{axis}_draft_ece"] = draft_ece
        if axis == "correction_confidence_decile":
            metrics[f"{prefix}{axis}_correction_ece"] = correction_ece
    return metrics


def validate_transition_strata_conservation(
    strata: dict[str, dict[str, dict[str, int | float]]],
    expected_counts: dict[str, int],
) -> None:
    for axis, bands in strata.items():
        for field in _COUNT_FIELDS:
            observed = sum(int(accumulator[field]) for accumulator in bands.values())
            expected = int(expected_counts[field])
            if observed != expected:
                raise RuntimeError(
                    f"transition stratum {axis} does not conserve {field}: observed {observed}, expected {expected}"
                )


def transition_strata_definitions() -> dict[str, str]:
    return {
        "draft_error_band": "per_sequence_exact_then_(0,.25],(.25,.5],(.5,.75],(.75,1]",
        "token_frequency_band": "0=special;1..7=equal_token_type_frequency_rank_rarest_to_most_frequent",
        "position_decile": "floor(10*column/max(attended_length-1,1))_clamped_0_to_9",
        "draft_confidence_decile": "blocked_vocabulary_top1_probability_fixed_[0,.1),...,[.9,1]",
        "correction_confidence_decile": "blocked_vocabulary_top1_probability_fixed_[0,.1),...,[.9,1]",
        "axes": "marginal_not_cartesian",
        "confidence_calibration": "weighted_absolute_gap_between_mean_top1_confidence_and_accuracy",
    }
