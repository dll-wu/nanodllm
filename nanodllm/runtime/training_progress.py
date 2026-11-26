"""Typed ownership and versioned codecs for checkpointed trainer progress."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field

from .training_self_generated_edit import (
    coerce_self_generated_edit_accumulator,
    new_self_generated_edit_accumulator,
)

LEGACY_TRAINER_STATE_VERSION = 1
INTERVAL_DENOMINATOR_TRAINER_STATE_VERSION = 2
BRANCH_DENOMINATOR_TRAINER_STATE_VERSION = 3
SELF_GENERATED_EDIT_TRAINER_STATE_VERSION = 4
RESUME_TRAINER_STATE_VERSION = 5
TRAINER_PROGRESS_VERSION = RESUME_TRAINER_STATE_VERSION

TOP1_PROB_HIST_BINS = 10
LOGIT_SCALE_HIST_BINS = 16
EOS_HAZARD_RELATIVE_POSITION_DECILES = 10
POSITION_DECILE_BALANCED_LOSS_DECILES = 10
TOKEN_EDIT_CORRUPTION_DECILES = 7
PER_TOKEN_NOISE_LEVEL_HIST_BINS = 10

AR_BOUNDARY_COUNT_FIELDS = (
    "microbatches",
    "microbatches_with_bos",
    "valid_tokens",
    "special_tokens",
    "special_loss_tokens",
    "bos_tokens",
    "bos_loss_tokens",
    "eos_tokens",
    "eos_loss_tokens",
)
BOUNDARY_TELEMETRY_FIELDS = (
    "documents_per_row_sum",
    "documents_per_row_count",
    "pad_tokens",
    "total_row_tokens",
    "blocked_pairs_sum",
    "causal_pairs_sum",
    "valid_pairs_sum",
    "allowed_pairs_sum",
    "segment_blocked_pairs_sum",
)
ROW_TELEMETRY_FIELDS = (
    "rows",
    "valid_tokens_sum",
    "valid_tokens_min",
    "valid_tokens_max",
    "lossable_tokens_sum",
    "lossable_tokens_min",
    "lossable_tokens_max",
    "pad_tokens",
    "total_row_tokens",
    "eos_rows",
    "eos_tokens",
    "eos_loss_tokens",
)
EOS_SUPERVISION_COUNT_FIELDS = (
    "microbatches",
    "rows",
    "valid_tokens",
    "loss_denominator_tokens",
    "attention_lossable_tokens",
    "lossable_attention_mismatch_tokens",
    "special_tokens",
    "special_loss_tokens",
    "eos_tokens",
    "eos_loss_tokens",
    "eos_rows",
    "eos_loss_rows",
    "active_target_tokens",
    "active_eos_target_tokens",
    "active_special_target_tokens",
    "row_eos_loss_tokens_0",
    "row_eos_loss_tokens_1",
    "row_eos_loss_tokens_2",
    "row_eos_loss_tokens_3",
    "row_eos_loss_tokens_4plus",
)

LINEAGE_V1_FIELDS = (
    "ar_boundary_counts",
    "boundary_telemetry",
    "row_telemetry",
    "eos_supervision_train_counts",
    "block_causal_visible_prefix_frac_sum",
    "block_causal_visible_prefix_count",
    "train_m2t_ce_sum",
    "train_m2t_loss_sum",
    "train_m2t_loss_count",
    "train_m2t_microbatches",
    "train_m2t_active_tokens",
    "train_t2t_ce_sum",
    "train_t2t_loss_sum",
    "train_t2t_loss_count",
    "train_t2t_microbatches",
    "train_t2t_active_tokens",
    "train_length_head_accumulator",
    "train_token_edit_reconstruction_accumulator",
    "train_per_token_noise_level_hist_counts",
    "train_per_token_effective_masked_fraction_sum",
    "train_per_token_effective_masked_fraction_count",
    "train_per_token_effective_masked_fraction_min",
    "train_per_token_effective_masked_fraction_max",
    "grad_clip_event_count",
    "grad_clip_observed_steps",
)
BRANCH_DENOMINATOR_FIELDS = (
    "m2t_lossable_tokens",
    "t2t_lossable_tokens",
    "t2t_ordinary_eligible_tokens",
)
LINEAGE_V3_FIELDS = (
    *LINEAGE_V1_FIELDS,
    "train_m2t_lossable_tokens",
    "train_t2t_lossable_tokens",
    "train_t2t_ordinary_eligible_tokens",
)
LINEAGE_V4_FIELDS = (*LINEAGE_V3_FIELDS, "train_self_generated_edit_accumulator")
LINEAGE_V5_FIELDS = (
    *LINEAGE_V3_FIELDS,
    "train_t2t_eligible_tokens",
    "train_self_generated_edit_accumulator",
)
INTERVAL_V1_FIELDS = (
    "smooth_loss",
    "smooth_count",
    "m2t_ce_sum",
    "m2t_loss_sum",
    "m2t_loss_count",
    "m2t_microbatches",
    "m2t_active_tokens",
    "t2t_ce_sum",
    "t2t_loss_sum",
    "t2t_loss_count",
    "t2t_microbatches",
    "t2t_active_tokens",
)
INTERVAL_V2_FIELDS = (*INTERVAL_V1_FIELDS, "lossable_tokens")
INTERVAL_V3_FIELDS = (*INTERVAL_V2_FIELDS, *BRANCH_DENOMINATOR_FIELDS)
INTERVAL_V5_FIELDS = (*INTERVAL_V3_FIELDS, "t2t_eligible_tokens", "self_generated_edit_accumulator")
LAST_STEP_V1_FIELDS = (
    "loss_value",
    "main_ce",
    "main_loss",
    "confidence_penalty",
    "confidence_penalty_loss",
    "z_loss",
    "z_loss_loss",
    "eos_hazard_calibration_loss",
    "eos_hazard_loss",
    "eos_hazard_accumulator",
    "length_head_accumulator",
    "position_decile_ce_accumulator",
    "token_edit_reconstruction_accumulator",
    "m2t_ce",
    "m2t_loss",
    "m2t_microbatches",
    "m2t_active_tokens",
    "t2t_ce",
    "t2t_loss",
    "t2t_microbatches",
    "t2t_active_tokens",
    "grad_norm",
    "grad_clipped",
    "grad_clip_ratio",
    "active_tokens",
    "lossable_tokens",
    "valid_tokens",
    "content_tokens",
    "content_bytes",
    "class_ce",
    "within_class_ce",
    "ar_unsmoothed_ce",
    "second_moment_norm",
    "update_norm",
    "update_grad_ratio",
    "entropy_mean",
    "top1_prob_mean",
    "argmax_mode_frac",
    "top1_prob_hist_counts",
    "entropy_hist_counts",
    "logsumexp_hist_counts",
    "logit_norm_hist_counts",
    "logit_norm_mean",
    "logit_norm_min",
    "logit_norm_max",
    "logit_rms_mean",
    "logsumexp_mean",
    "logsumexp_min",
    "logsumexp_max",
    "soft_input_stats",
    "uniform_estimator_stats",
    "per_token_noise_level_hist_counts",
    "per_token_effective_masked_fraction_sum",
    "per_token_effective_masked_fraction_count",
    "per_token_effective_masked_fraction_min",
    "per_token_effective_masked_fraction_max",
    "gradient_component_norms",
    "update_component_norms",
)
LAST_STEP_V3_FIELDS = (*LAST_STEP_V1_FIELDS, *BRANCH_DENOMINATOR_FIELDS)
LAST_STEP_V4_FIELDS = (*LAST_STEP_V3_FIELDS, "self_generated_edit_accumulator")
LAST_STEP_V5_FIELDS = (*LAST_STEP_V3_FIELDS, "t2t_eligible_tokens", "self_generated_edit_accumulator")
DIAGNOSTICS_V1_FIELDS = (
    "initial_parameter_norm_metrics",
    "initial_gradient_metrics",
    "initial_update_metrics",
    "initial_lm_head_row_norm_metrics",
    "early_lm_head_row_norm_metrics",
    "early_training_diagnostics",
    "early_residual_diagnostics",
    "fixed_ce_trace_records",
    "last_fixed_ce_trace_record",
)
RESUME_TRAINER_V1_GROUP_FIELDS = {
    "lineage": LINEAGE_V1_FIELDS,
    "interval": INTERVAL_V1_FIELDS,
    "last": LAST_STEP_V1_FIELDS,
    "diagnostics": DIAGNOSTICS_V1_FIELDS,
}
RESUME_TRAINER_V2_GROUP_FIELDS = {
    **RESUME_TRAINER_V1_GROUP_FIELDS,
    "interval": INTERVAL_V2_FIELDS,
}
RESUME_TRAINER_V3_GROUP_FIELDS = {
    "lineage": LINEAGE_V3_FIELDS,
    "interval": INTERVAL_V3_FIELDS,
    "last": LAST_STEP_V3_FIELDS,
    "diagnostics": DIAGNOSTICS_V1_FIELDS,
}
RESUME_TRAINER_V4_GROUP_FIELDS = {
    "lineage": LINEAGE_V4_FIELDS,
    "interval": INTERVAL_V3_FIELDS,
    "last": LAST_STEP_V4_FIELDS,
    "diagnostics": DIAGNOSTICS_V1_FIELDS,
}
RESUME_TRAINER_GROUP_FIELDS = {
    "lineage": LINEAGE_V5_FIELDS,
    "interval": INTERVAL_V5_FIELDS,
    "last": LAST_STEP_V5_FIELDS,
    "diagnostics": DIAGNOSTICS_V1_FIELDS,
}


def _require_positive_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{field_name} must be a positive integer")
    return value


def _require_exact_dict(value: object, expected: tuple[str, ...], *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} keys must be strings")
    observed = set(value)
    expected_set = set(expected)
    if observed != expected_set:
        raise RuntimeError(
            f"{field_name} keys mismatch: "
            f"missing={sorted(expected_set - observed)}, extra={sorted(observed - expected_set)}"
        )
    return value


def _as_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric")
    return float(value)


def _as_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric")
    return int(value)


def _as_strict_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _int_list(value: object, *, length: int, field_name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise RuntimeError(f"{field_name} must be a list of length {length}")
    return [_as_strict_int(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)]


def _float_list(value: object, *, length: int, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise RuntimeError(f"{field_name} must be a list of length {length}")
    return [_as_float(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)]


def _int_counter(value: object, fields: tuple[str, ...], *, field_name: str) -> dict[str, int]:
    mapping = _require_exact_dict(value, fields, field_name=field_name)
    return {key: _as_strict_int(mapping[key], field_name=f"{field_name}.{key}") for key in fields}


def _boundary_telemetry(value: object, *, field_name: str) -> dict[str, int | float]:
    mapping = _require_exact_dict(value, BOUNDARY_TELEMETRY_FIELDS, field_name=field_name)
    return {
        "documents_per_row_sum": _as_float(
            mapping["documents_per_row_sum"], field_name=f"{field_name}.documents_per_row_sum"
        ),
        "documents_per_row_count": _as_strict_int(
            mapping["documents_per_row_count"], field_name=f"{field_name}.documents_per_row_count"
        ),
        "pad_tokens": _as_strict_int(mapping["pad_tokens"], field_name=f"{field_name}.pad_tokens"),
        "total_row_tokens": _as_strict_int(mapping["total_row_tokens"], field_name=f"{field_name}.total_row_tokens"),
        "blocked_pairs_sum": _as_strict_int(mapping["blocked_pairs_sum"], field_name=f"{field_name}.blocked_pairs_sum"),
        "causal_pairs_sum": _as_strict_int(mapping["causal_pairs_sum"], field_name=f"{field_name}.causal_pairs_sum"),
        "valid_pairs_sum": _as_strict_int(mapping["valid_pairs_sum"], field_name=f"{field_name}.valid_pairs_sum"),
        "allowed_pairs_sum": _as_strict_int(mapping["allowed_pairs_sum"], field_name=f"{field_name}.allowed_pairs_sum"),
        "segment_blocked_pairs_sum": _as_strict_int(
            mapping["segment_blocked_pairs_sum"], field_name=f"{field_name}.segment_blocked_pairs_sum"
        ),
    }


def _length_head_accumulator(value: object, *, max_seq_len: int, field_name: str) -> dict[str, object]:
    fields = (
        "microbatches",
        "count",
        "ce_sum",
        "accuracy_sum",
        "mae_sum",
        "confidence_sum",
        "calibration_abs_error_sum",
        "target_counts",
        "pred_counts",
    )
    mapping = _require_exact_dict(value, fields, field_name=field_name)
    return {
        "microbatches": _as_strict_int(mapping["microbatches"], field_name=f"{field_name}.microbatches"),
        "count": _as_float(mapping["count"], field_name=f"{field_name}.count"),
        "ce_sum": _as_float(mapping["ce_sum"], field_name=f"{field_name}.ce_sum"),
        "accuracy_sum": _as_float(mapping["accuracy_sum"], field_name=f"{field_name}.accuracy_sum"),
        "mae_sum": _as_float(mapping["mae_sum"], field_name=f"{field_name}.mae_sum"),
        "confidence_sum": _as_float(mapping["confidence_sum"], field_name=f"{field_name}.confidence_sum"),
        "calibration_abs_error_sum": _as_float(
            mapping["calibration_abs_error_sum"], field_name=f"{field_name}.calibration_abs_error_sum"
        ),
        "target_counts": _float_list(
            mapping["target_counts"], length=max_seq_len, field_name=f"{field_name}.target_counts"
        ),
        "pred_counts": _float_list(mapping["pred_counts"], length=max_seq_len, field_name=f"{field_name}.pred_counts"),
    }


def _token_edit_accumulator(value: object, *, field_name: str) -> dict[str, object]:
    mapping = _require_exact_dict(value, ("counts", "correct", "microbatches"), field_name=field_name)
    length = TOKEN_EDIT_CORRUPTION_DECILES + 1
    return {
        "counts": _int_list(mapping["counts"], length=length, field_name=f"{field_name}.counts"),
        "correct": _int_list(mapping["correct"], length=length, field_name=f"{field_name}.correct"),
        "microbatches": _as_strict_int(mapping["microbatches"], field_name=f"{field_name}.microbatches"),
    }


def _position_decile_accumulator(value: object, *, field_name: str) -> dict[str, object]:
    list_fields = (
        "active_counts",
        "lossable_counts",
        "raw_ce_sums",
        "weighted_ce_sums",
        "balanced_contributions",
    )
    mapping = _require_exact_dict(value, (*list_fields, "loss_sum", "microbatches"), field_name=field_name)
    return {
        **{
            key: _float_list(
                mapping[key], length=POSITION_DECILE_BALANCED_LOSS_DECILES, field_name=f"{field_name}.{key}"
            )
            for key in list_fields
        },
        "loss_sum": _as_float(mapping["loss_sum"], field_name=f"{field_name}.loss_sum"),
        "microbatches": _as_strict_int(mapping["microbatches"], field_name=f"{field_name}.microbatches"),
    }


def _eos_hazard_accumulator(value: object, *, field_name: str) -> dict[str, list[float]]:
    fields = ("pred_sums", "target_sums", "counts")
    mapping = _require_exact_dict(value, fields, field_name=field_name)
    return {
        key: _float_list(mapping[key], length=EOS_HAZARD_RELATIVE_POSITION_DECILES, field_name=f"{field_name}.{key}")
        for key in fields
    }


def _validate_json_state(value: object, *, field_name: str) -> None:
    if value is None or isinstance(value, bool | int | float | str):
        return
    if isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _validate_json_state(child, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} mapping keys must be strings")
        for key, child in value.items():
            _validate_json_state(child, field_name=f"{field_name}.{key}")
        return
    raise TypeError(f"{field_name} has unsupported type {type(value).__name__}")


def _json_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    _validate_json_state(value, field_name=field_name)
    return deepcopy(value)


def _new_length_head_accumulator(max_seq_len: int) -> dict[str, object]:
    return {
        "microbatches": 0,
        "count": 0.0,
        "ce_sum": 0.0,
        "accuracy_sum": 0.0,
        "mae_sum": 0.0,
        "confidence_sum": 0.0,
        "calibration_abs_error_sum": 0.0,
        "target_counts": [0.0 for _ in range(max_seq_len)],
        "pred_counts": [0.0 for _ in range(max_seq_len)],
    }


def _new_token_edit_accumulator() -> dict[str, object]:
    length = TOKEN_EDIT_CORRUPTION_DECILES + 1
    return {"counts": [0 for _ in range(length)], "correct": [0 for _ in range(length)], "microbatches": 0}


def _new_position_decile_accumulator() -> dict[str, object]:
    return {
        "active_counts": [0.0 for _ in range(POSITION_DECILE_BALANCED_LOSS_DECILES)],
        "lossable_counts": [0.0 for _ in range(POSITION_DECILE_BALANCED_LOSS_DECILES)],
        "raw_ce_sums": [0.0 for _ in range(POSITION_DECILE_BALANCED_LOSS_DECILES)],
        "weighted_ce_sums": [0.0 for _ in range(POSITION_DECILE_BALANCED_LOSS_DECILES)],
        "balanced_contributions": [0.0 for _ in range(POSITION_DECILE_BALANCED_LOSS_DECILES)],
        "loss_sum": 0.0,
        "microbatches": 0,
    }


def _new_eos_hazard_accumulator() -> dict[str, list[float]]:
    return {
        "pred_sums": [0.0 for _ in range(EOS_HAZARD_RELATIVE_POSITION_DECILES)],
        "target_sums": [0.0 for _ in range(EOS_HAZARD_RELATIVE_POSITION_DECILES)],
        "counts": [0.0 for _ in range(EOS_HAZARD_RELATIVE_POSITION_DECILES)],
    }


@dataclass(slots=True)
class LineageState:
    ar_boundary_counts: dict[str, int] = field(default_factory=dict)
    boundary_telemetry: dict[str, int | float] = field(default_factory=dict)
    row_telemetry: dict[str, int] = field(default_factory=dict)
    eos_supervision_train_counts: dict[str, int] = field(default_factory=dict)
    block_causal_visible_prefix_frac_sum: float = 0.0
    block_causal_visible_prefix_count: int = 0
    train_m2t_ce_sum: float = 0.0
    train_m2t_loss_sum: float = 0.0
    train_m2t_loss_count: int = 0
    train_m2t_microbatches: int = 0
    train_m2t_active_tokens: int = 0
    train_t2t_ce_sum: float = 0.0
    train_t2t_loss_sum: float = 0.0
    train_t2t_loss_count: int = 0
    train_t2t_microbatches: int = 0
    train_t2t_active_tokens: int = 0
    train_length_head_accumulator: dict[str, object] = field(default_factory=dict)
    train_token_edit_reconstruction_accumulator: dict[str, object] = field(default_factory=dict)
    train_per_token_noise_level_hist_counts: list[int] = field(default_factory=list)
    train_per_token_effective_masked_fraction_sum: float = 0.0
    train_per_token_effective_masked_fraction_count: int = 0
    train_per_token_effective_masked_fraction_min: float = float("inf")
    train_per_token_effective_masked_fraction_max: float = float("-inf")
    grad_clip_event_count: int = 0
    grad_clip_observed_steps: int = 0
    train_m2t_lossable_tokens: int = 0
    train_t2t_lossable_tokens: int = 0
    train_t2t_ordinary_eligible_tokens: int = 0
    train_t2t_eligible_tokens: int = 0
    train_self_generated_edit_accumulator: dict[str, int | float] = field(default_factory=dict)

    @classmethod
    def fresh(cls, max_seq_len: int) -> LineageState:
        max_seq_len = _require_positive_int(max_seq_len, field_name="max_seq_len")
        return cls(
            ar_boundary_counts={key: 0 for key in AR_BOUNDARY_COUNT_FIELDS},
            boundary_telemetry={key: 0.0 if key == "documents_per_row_sum" else 0 for key in BOUNDARY_TELEMETRY_FIELDS},
            row_telemetry={key: 0 for key in ROW_TELEMETRY_FIELDS},
            eos_supervision_train_counts={key: 0 for key in EOS_SUPERVISION_COUNT_FIELDS},
            train_length_head_accumulator=_new_length_head_accumulator(max_seq_len),
            train_token_edit_reconstruction_accumulator=_new_token_edit_accumulator(),
            train_self_generated_edit_accumulator=new_self_generated_edit_accumulator(),
            train_per_token_noise_level_hist_counts=[0 for _ in range(PER_TOKEN_NOISE_LEVEL_HIST_BINS)],
        )

    @classmethod
    def from_v1(cls, value: object, *, max_seq_len: int) -> LineageState:
        max_seq_len = _require_positive_int(max_seq_len, field_name="max_seq_len")
        state = _require_exact_dict(value, LINEAGE_V1_FIELDS, field_name="trainer_state.lineage")
        field_name = "trainer_state.lineage"
        return cls(
            ar_boundary_counts=_int_counter(
                state["ar_boundary_counts"], AR_BOUNDARY_COUNT_FIELDS, field_name=f"{field_name}.ar_boundary_counts"
            ),
            boundary_telemetry=_boundary_telemetry(
                state["boundary_telemetry"], field_name=f"{field_name}.boundary_telemetry"
            ),
            row_telemetry=_int_counter(
                state["row_telemetry"], ROW_TELEMETRY_FIELDS, field_name=f"{field_name}.row_telemetry"
            ),
            eos_supervision_train_counts=_int_counter(
                state["eos_supervision_train_counts"],
                EOS_SUPERVISION_COUNT_FIELDS,
                field_name=f"{field_name}.eos_supervision_train_counts",
            ),
            block_causal_visible_prefix_frac_sum=_as_float(
                state["block_causal_visible_prefix_frac_sum"],
                field_name=f"{field_name}.block_causal_visible_prefix_frac_sum",
            ),
            block_causal_visible_prefix_count=_as_int(
                state["block_causal_visible_prefix_count"],
                field_name=f"{field_name}.block_causal_visible_prefix_count",
            ),
            train_m2t_ce_sum=_as_float(state["train_m2t_ce_sum"], field_name=f"{field_name}.train_m2t_ce_sum"),
            train_m2t_loss_sum=_as_float(state["train_m2t_loss_sum"], field_name=f"{field_name}.train_m2t_loss_sum"),
            train_m2t_loss_count=_as_int(
                state["train_m2t_loss_count"], field_name=f"{field_name}.train_m2t_loss_count"
            ),
            train_m2t_microbatches=_as_int(
                state["train_m2t_microbatches"], field_name=f"{field_name}.train_m2t_microbatches"
            ),
            train_m2t_active_tokens=_as_int(
                state["train_m2t_active_tokens"], field_name=f"{field_name}.train_m2t_active_tokens"
            ),
            train_t2t_ce_sum=_as_float(state["train_t2t_ce_sum"], field_name=f"{field_name}.train_t2t_ce_sum"),
            train_t2t_loss_sum=_as_float(state["train_t2t_loss_sum"], field_name=f"{field_name}.train_t2t_loss_sum"),
            train_t2t_loss_count=_as_int(
                state["train_t2t_loss_count"], field_name=f"{field_name}.train_t2t_loss_count"
            ),
            train_t2t_microbatches=_as_int(
                state["train_t2t_microbatches"], field_name=f"{field_name}.train_t2t_microbatches"
            ),
            train_t2t_active_tokens=_as_int(
                state["train_t2t_active_tokens"], field_name=f"{field_name}.train_t2t_active_tokens"
            ),
            train_length_head_accumulator=_length_head_accumulator(
                state["train_length_head_accumulator"],
                max_seq_len=max_seq_len,
                field_name=f"{field_name}.train_length_head_accumulator",
            ),
            train_token_edit_reconstruction_accumulator=_token_edit_accumulator(
                state["train_token_edit_reconstruction_accumulator"],
                field_name=f"{field_name}.train_token_edit_reconstruction_accumulator",
            ),
            train_self_generated_edit_accumulator=new_self_generated_edit_accumulator(),
            train_per_token_noise_level_hist_counts=_int_list(
                state["train_per_token_noise_level_hist_counts"],
                length=PER_TOKEN_NOISE_LEVEL_HIST_BINS,
                field_name=f"{field_name}.train_per_token_noise_level_hist_counts",
            ),
            train_per_token_effective_masked_fraction_sum=_as_float(
                state["train_per_token_effective_masked_fraction_sum"],
                field_name=f"{field_name}.train_per_token_effective_masked_fraction_sum",
            ),
            train_per_token_effective_masked_fraction_count=_as_int(
                state["train_per_token_effective_masked_fraction_count"],
                field_name=f"{field_name}.train_per_token_effective_masked_fraction_count",
            ),
            train_per_token_effective_masked_fraction_min=_as_float(
                state["train_per_token_effective_masked_fraction_min"],
                field_name=f"{field_name}.train_per_token_effective_masked_fraction_min",
            ),
            train_per_token_effective_masked_fraction_max=_as_float(
                state["train_per_token_effective_masked_fraction_max"],
                field_name=f"{field_name}.train_per_token_effective_masked_fraction_max",
            ),
            grad_clip_event_count=_as_int(
                state["grad_clip_event_count"], field_name=f"{field_name}.grad_clip_event_count"
            ),
            grad_clip_observed_steps=_as_int(
                state["grad_clip_observed_steps"], field_name=f"{field_name}.grad_clip_observed_steps"
            ),
        )

    def to_v1(self) -> dict[str, object]:
        return {
            "ar_boundary_counts": deepcopy(self.ar_boundary_counts),
            "boundary_telemetry": deepcopy(self.boundary_telemetry),
            "row_telemetry": deepcopy(self.row_telemetry),
            "eos_supervision_train_counts": deepcopy(self.eos_supervision_train_counts),
            "block_causal_visible_prefix_frac_sum": self.block_causal_visible_prefix_frac_sum,
            "block_causal_visible_prefix_count": self.block_causal_visible_prefix_count,
            "train_m2t_ce_sum": self.train_m2t_ce_sum,
            "train_m2t_loss_sum": self.train_m2t_loss_sum,
            "train_m2t_loss_count": self.train_m2t_loss_count,
            "train_m2t_microbatches": self.train_m2t_microbatches,
            "train_m2t_active_tokens": self.train_m2t_active_tokens,
            "train_t2t_ce_sum": self.train_t2t_ce_sum,
            "train_t2t_loss_sum": self.train_t2t_loss_sum,
            "train_t2t_loss_count": self.train_t2t_loss_count,
            "train_t2t_microbatches": self.train_t2t_microbatches,
            "train_t2t_active_tokens": self.train_t2t_active_tokens,
            "train_length_head_accumulator": deepcopy(self.train_length_head_accumulator),
            "train_token_edit_reconstruction_accumulator": deepcopy(self.train_token_edit_reconstruction_accumulator),
            "train_per_token_noise_level_hist_counts": list(self.train_per_token_noise_level_hist_counts),
            "train_per_token_effective_masked_fraction_sum": self.train_per_token_effective_masked_fraction_sum,
            "train_per_token_effective_masked_fraction_count": self.train_per_token_effective_masked_fraction_count,
            "train_per_token_effective_masked_fraction_min": self.train_per_token_effective_masked_fraction_min,
            "train_per_token_effective_masked_fraction_max": self.train_per_token_effective_masked_fraction_max,
            "grad_clip_event_count": self.grad_clip_event_count,
            "grad_clip_observed_steps": self.grad_clip_observed_steps,
        }

    @classmethod
    def from_v3(cls, value: object, *, max_seq_len: int) -> LineageState:
        state = _require_exact_dict(value, LINEAGE_V3_FIELDS, field_name="trainer_state.lineage")
        result = cls.from_v1({key: state[key] for key in LINEAGE_V1_FIELDS}, max_seq_len=max_seq_len)
        result.train_self_generated_edit_accumulator = new_self_generated_edit_accumulator()
        result.train_m2t_lossable_tokens = _as_int(
            state["train_m2t_lossable_tokens"],
            field_name="trainer_state.lineage.train_m2t_lossable_tokens",
        )
        result.train_t2t_lossable_tokens = _as_int(
            state["train_t2t_lossable_tokens"],
            field_name="trainer_state.lineage.train_t2t_lossable_tokens",
        )
        result.train_t2t_ordinary_eligible_tokens = _as_int(
            state["train_t2t_ordinary_eligible_tokens"],
            field_name="trainer_state.lineage.train_t2t_ordinary_eligible_tokens",
        )
        result.train_t2t_eligible_tokens = result.train_t2t_ordinary_eligible_tokens
        return result

    def to_v3(self) -> dict[str, object]:
        return {
            **self.to_v1(),
            "train_m2t_lossable_tokens": self.train_m2t_lossable_tokens,
            "train_t2t_lossable_tokens": self.train_t2t_lossable_tokens,
            "train_t2t_ordinary_eligible_tokens": self.train_t2t_ordinary_eligible_tokens,
        }

    @classmethod
    def from_v4(cls, value: object, *, max_seq_len: int) -> LineageState:
        state = _require_exact_dict(value, LINEAGE_V4_FIELDS, field_name="trainer_state.lineage")
        result = cls.from_v3({key: state[key] for key in LINEAGE_V3_FIELDS}, max_seq_len=max_seq_len)
        result.train_self_generated_edit_accumulator = coerce_self_generated_edit_accumulator(
            state["train_self_generated_edit_accumulator"],
            field_name="trainer_state.lineage.train_self_generated_edit_accumulator",
        )
        result.train_t2t_eligible_tokens = result.train_t2t_ordinary_eligible_tokens
        return result

    def to_v4(self) -> dict[str, object]:
        return {
            **self.to_v3(),
            "train_self_generated_edit_accumulator": deepcopy(self.train_self_generated_edit_accumulator),
        }

    @classmethod
    def from_v5(cls, value: object, *, max_seq_len: int) -> LineageState:
        state = _require_exact_dict(value, LINEAGE_V5_FIELDS, field_name="trainer_state.lineage")
        result = cls.from_v3({key: state[key] for key in LINEAGE_V3_FIELDS}, max_seq_len=max_seq_len)
        result.train_t2t_eligible_tokens = _as_int(
            state["train_t2t_eligible_tokens"],
            field_name="trainer_state.lineage.train_t2t_eligible_tokens",
        )
        result.train_self_generated_edit_accumulator = coerce_self_generated_edit_accumulator(
            state["train_self_generated_edit_accumulator"],
            field_name="trainer_state.lineage.train_self_generated_edit_accumulator",
        )
        return result

    def to_v5(self) -> dict[str, object]:
        return {
            **self.to_v3(),
            "train_t2t_eligible_tokens": self.train_t2t_eligible_tokens,
            "train_self_generated_edit_accumulator": deepcopy(self.train_self_generated_edit_accumulator),
        }


@dataclass(slots=True)
class IntervalState:
    smooth_loss: float = 0.0
    smooth_count: int = 0
    m2t_ce_sum: float = 0.0
    m2t_loss_sum: float = 0.0
    m2t_loss_count: int = 0
    m2t_microbatches: int = 0
    m2t_active_tokens: int = 0
    t2t_ce_sum: float = 0.0
    t2t_loss_sum: float = 0.0
    t2t_loss_count: int = 0
    t2t_microbatches: int = 0
    t2t_active_tokens: int = 0
    lossable_tokens: int = 0
    m2t_lossable_tokens: int = 0
    t2t_lossable_tokens: int = 0
    t2t_ordinary_eligible_tokens: int = 0
    t2t_eligible_tokens: int = 0
    self_generated_edit_accumulator: dict[str, int | float] = field(default_factory=new_self_generated_edit_accumulator)

    @classmethod
    def fresh(cls) -> IntervalState:
        return cls()

    @classmethod
    def from_v1(cls, value: object) -> IntervalState:
        state = _require_exact_dict(value, INTERVAL_V1_FIELDS, field_name="trainer_state.interval")
        field_name = "trainer_state.interval"
        for key in ("m2t_ce_sum", "m2t_loss_sum", "t2t_ce_sum", "t2t_loss_sum"):
            _as_float(state[key], field_name=f"{field_name}.{key}")
        for key in (
            "m2t_loss_count",
            "m2t_microbatches",
            "m2t_active_tokens",
            "t2t_loss_count",
            "t2t_microbatches",
            "t2t_active_tokens",
        ):
            _as_int(state[key], field_name=f"{field_name}.{key}")
        # Version 1 did not retain the interval lossable-token denominator. Its
        # branch counters therefore cannot produce a truthful active-token
        # fraction after resume. Preserve the smoothing state, but start a new
        # branch-metric interval rather than publishing a fabricated ratio.
        return cls(
            smooth_loss=_as_float(state["smooth_loss"], field_name=f"{field_name}.smooth_loss"),
            smooth_count=_as_int(state["smooth_count"], field_name=f"{field_name}.smooth_count"),
        )

    @classmethod
    def from_v2(cls, value: object) -> IntervalState:
        state = _require_exact_dict(value, INTERVAL_V2_FIELDS, field_name="trainer_state.interval")
        field_name = "trainer_state.interval"
        return cls(
            smooth_loss=_as_float(state["smooth_loss"], field_name=f"{field_name}.smooth_loss"),
            smooth_count=_as_int(state["smooth_count"], field_name=f"{field_name}.smooth_count"),
            m2t_ce_sum=_as_float(state["m2t_ce_sum"], field_name=f"{field_name}.m2t_ce_sum"),
            m2t_loss_sum=_as_float(state["m2t_loss_sum"], field_name=f"{field_name}.m2t_loss_sum"),
            m2t_loss_count=_as_int(state["m2t_loss_count"], field_name=f"{field_name}.m2t_loss_count"),
            m2t_microbatches=_as_int(state["m2t_microbatches"], field_name=f"{field_name}.m2t_microbatches"),
            m2t_active_tokens=_as_int(state["m2t_active_tokens"], field_name=f"{field_name}.m2t_active_tokens"),
            t2t_ce_sum=_as_float(state["t2t_ce_sum"], field_name=f"{field_name}.t2t_ce_sum"),
            t2t_loss_sum=_as_float(state["t2t_loss_sum"], field_name=f"{field_name}.t2t_loss_sum"),
            t2t_loss_count=_as_int(state["t2t_loss_count"], field_name=f"{field_name}.t2t_loss_count"),
            t2t_microbatches=_as_int(state["t2t_microbatches"], field_name=f"{field_name}.t2t_microbatches"),
            t2t_active_tokens=_as_int(state["t2t_active_tokens"], field_name=f"{field_name}.t2t_active_tokens"),
            lossable_tokens=_as_int(state["lossable_tokens"], field_name=f"{field_name}.lossable_tokens"),
        )

    def to_v1(self) -> dict[str, object]:
        return {
            "smooth_loss": self.smooth_loss,
            "smooth_count": self.smooth_count,
            "m2t_ce_sum": self.m2t_ce_sum,
            "m2t_loss_sum": self.m2t_loss_sum,
            "m2t_loss_count": self.m2t_loss_count,
            "m2t_microbatches": self.m2t_microbatches,
            "m2t_active_tokens": self.m2t_active_tokens,
            "t2t_ce_sum": self.t2t_ce_sum,
            "t2t_loss_sum": self.t2t_loss_sum,
            "t2t_loss_count": self.t2t_loss_count,
            "t2t_microbatches": self.t2t_microbatches,
            "t2t_active_tokens": self.t2t_active_tokens,
        }

    def to_v2(self) -> dict[str, object]:
        return {**self.to_v1(), "lossable_tokens": self.lossable_tokens}

    @classmethod
    def from_v3(cls, value: object) -> IntervalState:
        state = _require_exact_dict(value, INTERVAL_V3_FIELDS, field_name="trainer_state.interval")
        result = cls.from_v2({key: state[key] for key in INTERVAL_V2_FIELDS})
        result.m2t_lossable_tokens = _as_int(
            state["m2t_lossable_tokens"], field_name="trainer_state.interval.m2t_lossable_tokens"
        )
        result.t2t_lossable_tokens = _as_int(
            state["t2t_lossable_tokens"], field_name="trainer_state.interval.t2t_lossable_tokens"
        )
        result.t2t_ordinary_eligible_tokens = _as_int(
            state["t2t_ordinary_eligible_tokens"],
            field_name="trainer_state.interval.t2t_ordinary_eligible_tokens",
        )
        result.t2t_eligible_tokens = result.t2t_ordinary_eligible_tokens
        result.self_generated_edit_accumulator = new_self_generated_edit_accumulator()
        return result

    def to_v3(self) -> dict[str, object]:
        return {
            **self.to_v2(),
            "m2t_lossable_tokens": self.m2t_lossable_tokens,
            "t2t_lossable_tokens": self.t2t_lossable_tokens,
            "t2t_ordinary_eligible_tokens": self.t2t_ordinary_eligible_tokens,
        }

    @classmethod
    def from_v5(cls, value: object) -> IntervalState:
        state = _require_exact_dict(value, INTERVAL_V5_FIELDS, field_name="trainer_state.interval")
        result = cls.from_v3({key: state[key] for key in INTERVAL_V3_FIELDS})
        result.t2t_eligible_tokens = _as_int(
            state["t2t_eligible_tokens"],
            field_name="trainer_state.interval.t2t_eligible_tokens",
        )
        result.self_generated_edit_accumulator = coerce_self_generated_edit_accumulator(
            state["self_generated_edit_accumulator"],
            field_name="trainer_state.interval.self_generated_edit_accumulator",
        )
        return result

    def to_v5(self) -> dict[str, object]:
        return {
            **self.to_v3(),
            "t2t_eligible_tokens": self.t2t_eligible_tokens,
            "self_generated_edit_accumulator": deepcopy(self.self_generated_edit_accumulator),
        }


@dataclass(slots=True)
class LastStepState:
    loss_value: float = 0.0
    main_ce: float = 0.0
    main_loss: float = 0.0
    confidence_penalty: float = 0.0
    confidence_penalty_loss: float = 0.0
    z_loss: float = 0.0
    z_loss_loss: float = 0.0
    eos_hazard_calibration_loss: float = 0.0
    eos_hazard_loss: float = 0.0
    eos_hazard_accumulator: dict[str, list[float]] = field(default_factory=dict)
    length_head_accumulator: dict[str, object] = field(default_factory=dict)
    position_decile_ce_accumulator: dict[str, object] = field(default_factory=dict)
    token_edit_reconstruction_accumulator: dict[str, object] = field(default_factory=dict)
    m2t_ce: float = 0.0
    m2t_loss: float = 0.0
    m2t_microbatches: int = 0
    m2t_active_tokens: int = 0
    t2t_ce: float = 0.0
    t2t_loss: float = 0.0
    t2t_microbatches: int = 0
    t2t_active_tokens: int = 0
    grad_norm: float = 0.0
    grad_clipped: bool = False
    grad_clip_ratio: float = 1.0
    active_tokens: int = 0
    lossable_tokens: int = 0
    valid_tokens: int = 0
    content_tokens: int = 0
    content_bytes: int = 0
    class_ce: float = 0.0
    within_class_ce: float = 0.0
    ar_unsmoothed_ce: float = 0.0
    second_moment_norm: float = 0.0
    update_norm: float = 0.0
    update_grad_ratio: float = 0.0
    entropy_mean: float = 0.0
    top1_prob_mean: float = 0.0
    argmax_mode_frac: float = 0.0
    top1_prob_hist_counts: list[int] = field(default_factory=list)
    entropy_hist_counts: list[int] = field(default_factory=list)
    logsumexp_hist_counts: list[int] = field(default_factory=list)
    logit_norm_hist_counts: list[int] = field(default_factory=list)
    logit_norm_mean: float = 0.0
    logit_norm_min: float = 0.0
    logit_norm_max: float = 0.0
    logit_rms_mean: float = 0.0
    logsumexp_mean: float = 0.0
    logsumexp_min: float = 0.0
    logsumexp_max: float = 0.0
    soft_input_stats: dict[str, object] = field(default_factory=dict)
    uniform_estimator_stats: dict[str, object] = field(default_factory=dict)
    per_token_noise_level_hist_counts: list[int] = field(default_factory=list)
    per_token_effective_masked_fraction_sum: float = 0.0
    per_token_effective_masked_fraction_count: int = 0
    per_token_effective_masked_fraction_min: float = float("inf")
    per_token_effective_masked_fraction_max: float = float("-inf")
    gradient_component_norms: dict[str, object] = field(default_factory=dict)
    update_component_norms: dict[str, object] = field(default_factory=dict)
    m2t_lossable_tokens: int = 0
    t2t_lossable_tokens: int = 0
    t2t_ordinary_eligible_tokens: int = 0
    t2t_eligible_tokens: int = 0
    self_generated_edit_accumulator: dict[str, int | float] = field(default_factory=dict)

    @classmethod
    def fresh(cls, max_seq_len: int) -> LastStepState:
        max_seq_len = _require_positive_int(max_seq_len, field_name="max_seq_len")
        return cls(
            eos_hazard_accumulator=_new_eos_hazard_accumulator(),
            length_head_accumulator=_new_length_head_accumulator(max_seq_len),
            position_decile_ce_accumulator=_new_position_decile_accumulator(),
            token_edit_reconstruction_accumulator=_new_token_edit_accumulator(),
            self_generated_edit_accumulator=new_self_generated_edit_accumulator(),
            top1_prob_hist_counts=[0 for _ in range(TOP1_PROB_HIST_BINS)],
            entropy_hist_counts=[0 for _ in range(LOGIT_SCALE_HIST_BINS)],
            logsumexp_hist_counts=[0 for _ in range(LOGIT_SCALE_HIST_BINS)],
            logit_norm_hist_counts=[0 for _ in range(LOGIT_SCALE_HIST_BINS)],
            per_token_noise_level_hist_counts=[0 for _ in range(PER_TOKEN_NOISE_LEVEL_HIST_BINS)],
            update_component_norms={
                "update_norm_embedding": 0.0,
                "update_norm_lm_head": 0.0,
                "update_grad_ratio_embedding": 0.0,
                "update_grad_ratio_lm_head": 0.0,
            },
        )

    @classmethod
    def from_v1(cls, value: object, *, max_seq_len: int) -> LastStepState:
        max_seq_len = _require_positive_int(max_seq_len, field_name="max_seq_len")
        state = _require_exact_dict(value, LAST_STEP_V1_FIELDS, field_name="trainer_state.last")
        field_name = "trainer_state.last"
        floats = {
            key: _as_float(state[key], field_name=f"{field_name}.{key}")
            for key in (
                "loss_value",
                "main_ce",
                "main_loss",
                "confidence_penalty",
                "confidence_penalty_loss",
                "z_loss",
                "z_loss_loss",
                "eos_hazard_calibration_loss",
                "eos_hazard_loss",
                "m2t_ce",
                "m2t_loss",
                "t2t_ce",
                "t2t_loss",
                "grad_norm",
                "grad_clip_ratio",
                "class_ce",
                "within_class_ce",
                "ar_unsmoothed_ce",
                "second_moment_norm",
                "update_norm",
                "update_grad_ratio",
                "entropy_mean",
                "top1_prob_mean",
                "argmax_mode_frac",
                "logit_norm_mean",
                "logit_norm_min",
                "logit_norm_max",
                "logit_rms_mean",
                "logsumexp_mean",
                "logsumexp_min",
                "logsumexp_max",
                "per_token_effective_masked_fraction_sum",
                "per_token_effective_masked_fraction_min",
                "per_token_effective_masked_fraction_max",
            )
        }
        integers = {
            key: _as_int(state[key], field_name=f"{field_name}.{key}")
            for key in (
                "m2t_microbatches",
                "m2t_active_tokens",
                "t2t_microbatches",
                "t2t_active_tokens",
                "active_tokens",
                "lossable_tokens",
                "valid_tokens",
                "content_tokens",
                "content_bytes",
                "per_token_effective_masked_fraction_count",
            )
        }
        return cls(
            loss_value=floats["loss_value"],
            main_ce=floats["main_ce"],
            main_loss=floats["main_loss"],
            confidence_penalty=floats["confidence_penalty"],
            confidence_penalty_loss=floats["confidence_penalty_loss"],
            z_loss=floats["z_loss"],
            z_loss_loss=floats["z_loss_loss"],
            eos_hazard_calibration_loss=floats["eos_hazard_calibration_loss"],
            eos_hazard_loss=floats["eos_hazard_loss"],
            eos_hazard_accumulator=_eos_hazard_accumulator(
                state["eos_hazard_accumulator"], field_name=f"{field_name}.eos_hazard_accumulator"
            ),
            length_head_accumulator=_length_head_accumulator(
                state["length_head_accumulator"],
                max_seq_len=max_seq_len,
                field_name=f"{field_name}.length_head_accumulator",
            ),
            position_decile_ce_accumulator=_position_decile_accumulator(
                state["position_decile_ce_accumulator"],
                field_name=f"{field_name}.position_decile_ce_accumulator",
            ),
            token_edit_reconstruction_accumulator=_token_edit_accumulator(
                state["token_edit_reconstruction_accumulator"],
                field_name=f"{field_name}.token_edit_reconstruction_accumulator",
            ),
            self_generated_edit_accumulator=new_self_generated_edit_accumulator(),
            m2t_ce=floats["m2t_ce"],
            m2t_loss=floats["m2t_loss"],
            m2t_microbatches=integers["m2t_microbatches"],
            m2t_active_tokens=integers["m2t_active_tokens"],
            t2t_ce=floats["t2t_ce"],
            t2t_loss=floats["t2t_loss"],
            t2t_microbatches=integers["t2t_microbatches"],
            t2t_active_tokens=integers["t2t_active_tokens"],
            grad_norm=floats["grad_norm"],
            grad_clipped=_as_bool(state["grad_clipped"], field_name=f"{field_name}.grad_clipped"),
            grad_clip_ratio=floats["grad_clip_ratio"],
            active_tokens=integers["active_tokens"],
            lossable_tokens=integers["lossable_tokens"],
            valid_tokens=integers["valid_tokens"],
            content_tokens=integers["content_tokens"],
            content_bytes=integers["content_bytes"],
            class_ce=floats["class_ce"],
            within_class_ce=floats["within_class_ce"],
            ar_unsmoothed_ce=floats["ar_unsmoothed_ce"],
            second_moment_norm=floats["second_moment_norm"],
            update_norm=floats["update_norm"],
            update_grad_ratio=floats["update_grad_ratio"],
            entropy_mean=floats["entropy_mean"],
            top1_prob_mean=floats["top1_prob_mean"],
            argmax_mode_frac=floats["argmax_mode_frac"],
            top1_prob_hist_counts=_int_list(
                state["top1_prob_hist_counts"],
                length=TOP1_PROB_HIST_BINS,
                field_name=f"{field_name}.top1_prob_hist_counts",
            ),
            entropy_hist_counts=_int_list(
                state["entropy_hist_counts"],
                length=LOGIT_SCALE_HIST_BINS,
                field_name=f"{field_name}.entropy_hist_counts",
            ),
            logsumexp_hist_counts=_int_list(
                state["logsumexp_hist_counts"],
                length=LOGIT_SCALE_HIST_BINS,
                field_name=f"{field_name}.logsumexp_hist_counts",
            ),
            logit_norm_hist_counts=_int_list(
                state["logit_norm_hist_counts"],
                length=LOGIT_SCALE_HIST_BINS,
                field_name=f"{field_name}.logit_norm_hist_counts",
            ),
            logit_norm_mean=floats["logit_norm_mean"],
            logit_norm_min=floats["logit_norm_min"],
            logit_norm_max=floats["logit_norm_max"],
            logit_rms_mean=floats["logit_rms_mean"],
            logsumexp_mean=floats["logsumexp_mean"],
            logsumexp_min=floats["logsumexp_min"],
            logsumexp_max=floats["logsumexp_max"],
            soft_input_stats=_json_mapping(state["soft_input_stats"], field_name=f"{field_name}.soft_input_stats"),
            uniform_estimator_stats=_json_mapping(
                state["uniform_estimator_stats"], field_name=f"{field_name}.uniform_estimator_stats"
            ),
            per_token_noise_level_hist_counts=_int_list(
                state["per_token_noise_level_hist_counts"],
                length=PER_TOKEN_NOISE_LEVEL_HIST_BINS,
                field_name=f"{field_name}.per_token_noise_level_hist_counts",
            ),
            per_token_effective_masked_fraction_sum=floats["per_token_effective_masked_fraction_sum"],
            per_token_effective_masked_fraction_count=integers["per_token_effective_masked_fraction_count"],
            per_token_effective_masked_fraction_min=floats["per_token_effective_masked_fraction_min"],
            per_token_effective_masked_fraction_max=floats["per_token_effective_masked_fraction_max"],
            gradient_component_norms=_json_mapping(
                state["gradient_component_norms"], field_name=f"{field_name}.gradient_component_norms"
            ),
            update_component_norms=_json_mapping(
                state["update_component_norms"], field_name=f"{field_name}.update_component_norms"
            ),
        )

    def to_v1(self) -> dict[str, object]:
        return {
            "loss_value": self.loss_value,
            "main_ce": self.main_ce,
            "main_loss": self.main_loss,
            "confidence_penalty": self.confidence_penalty,
            "confidence_penalty_loss": self.confidence_penalty_loss,
            "z_loss": self.z_loss,
            "z_loss_loss": self.z_loss_loss,
            "eos_hazard_calibration_loss": self.eos_hazard_calibration_loss,
            "eos_hazard_loss": self.eos_hazard_loss,
            "eos_hazard_accumulator": deepcopy(self.eos_hazard_accumulator),
            "length_head_accumulator": deepcopy(self.length_head_accumulator),
            "position_decile_ce_accumulator": deepcopy(self.position_decile_ce_accumulator),
            "token_edit_reconstruction_accumulator": deepcopy(self.token_edit_reconstruction_accumulator),
            "m2t_ce": self.m2t_ce,
            "m2t_loss": self.m2t_loss,
            "m2t_microbatches": self.m2t_microbatches,
            "m2t_active_tokens": self.m2t_active_tokens,
            "t2t_ce": self.t2t_ce,
            "t2t_loss": self.t2t_loss,
            "t2t_microbatches": self.t2t_microbatches,
            "t2t_active_tokens": self.t2t_active_tokens,
            "grad_norm": self.grad_norm,
            "grad_clipped": self.grad_clipped,
            "grad_clip_ratio": self.grad_clip_ratio,
            "active_tokens": self.active_tokens,
            "lossable_tokens": self.lossable_tokens,
            "valid_tokens": self.valid_tokens,
            "content_tokens": self.content_tokens,
            "content_bytes": self.content_bytes,
            "class_ce": self.class_ce,
            "within_class_ce": self.within_class_ce,
            "ar_unsmoothed_ce": self.ar_unsmoothed_ce,
            "second_moment_norm": self.second_moment_norm,
            "update_norm": self.update_norm,
            "update_grad_ratio": self.update_grad_ratio,
            "entropy_mean": self.entropy_mean,
            "top1_prob_mean": self.top1_prob_mean,
            "argmax_mode_frac": self.argmax_mode_frac,
            "top1_prob_hist_counts": list(self.top1_prob_hist_counts),
            "entropy_hist_counts": list(self.entropy_hist_counts),
            "logsumexp_hist_counts": list(self.logsumexp_hist_counts),
            "logit_norm_hist_counts": list(self.logit_norm_hist_counts),
            "logit_norm_mean": self.logit_norm_mean,
            "logit_norm_min": self.logit_norm_min,
            "logit_norm_max": self.logit_norm_max,
            "logit_rms_mean": self.logit_rms_mean,
            "logsumexp_mean": self.logsumexp_mean,
            "logsumexp_min": self.logsumexp_min,
            "logsumexp_max": self.logsumexp_max,
            "soft_input_stats": deepcopy(self.soft_input_stats),
            "uniform_estimator_stats": deepcopy(self.uniform_estimator_stats),
            "per_token_noise_level_hist_counts": list(self.per_token_noise_level_hist_counts),
            "per_token_effective_masked_fraction_sum": self.per_token_effective_masked_fraction_sum,
            "per_token_effective_masked_fraction_count": self.per_token_effective_masked_fraction_count,
            "per_token_effective_masked_fraction_min": self.per_token_effective_masked_fraction_min,
            "per_token_effective_masked_fraction_max": self.per_token_effective_masked_fraction_max,
            "gradient_component_norms": deepcopy(self.gradient_component_norms),
            "update_component_norms": deepcopy(self.update_component_norms),
        }

    @classmethod
    def from_v3(cls, value: object, *, max_seq_len: int) -> LastStepState:
        state = _require_exact_dict(value, LAST_STEP_V3_FIELDS, field_name="trainer_state.last")
        result = cls.from_v1({key: state[key] for key in LAST_STEP_V1_FIELDS}, max_seq_len=max_seq_len)
        result.m2t_lossable_tokens = _as_int(
            state["m2t_lossable_tokens"], field_name="trainer_state.last.m2t_lossable_tokens"
        )
        result.t2t_lossable_tokens = _as_int(
            state["t2t_lossable_tokens"], field_name="trainer_state.last.t2t_lossable_tokens"
        )
        result.t2t_ordinary_eligible_tokens = _as_int(
            state["t2t_ordinary_eligible_tokens"],
            field_name="trainer_state.last.t2t_ordinary_eligible_tokens",
        )
        result.t2t_eligible_tokens = result.t2t_ordinary_eligible_tokens
        return result

    def to_v3(self) -> dict[str, object]:
        return {
            **self.to_v1(),
            "m2t_lossable_tokens": self.m2t_lossable_tokens,
            "t2t_lossable_tokens": self.t2t_lossable_tokens,
            "t2t_ordinary_eligible_tokens": self.t2t_ordinary_eligible_tokens,
        }

    @classmethod
    def from_v4(cls, value: object, *, max_seq_len: int) -> LastStepState:
        state = _require_exact_dict(value, LAST_STEP_V4_FIELDS, field_name="trainer_state.last")
        result = cls.from_v3({key: state[key] for key in LAST_STEP_V3_FIELDS}, max_seq_len=max_seq_len)
        result.self_generated_edit_accumulator = coerce_self_generated_edit_accumulator(
            state["self_generated_edit_accumulator"],
            field_name="trainer_state.last.self_generated_edit_accumulator",
        )
        result.t2t_eligible_tokens = result.t2t_ordinary_eligible_tokens
        return result

    def to_v4(self) -> dict[str, object]:
        return {
            **self.to_v3(),
            "self_generated_edit_accumulator": deepcopy(self.self_generated_edit_accumulator),
        }

    @classmethod
    def from_v5(cls, value: object, *, max_seq_len: int) -> LastStepState:
        state = _require_exact_dict(value, LAST_STEP_V5_FIELDS, field_name="trainer_state.last")
        result = cls.from_v3({key: state[key] for key in LAST_STEP_V3_FIELDS}, max_seq_len=max_seq_len)
        result.t2t_eligible_tokens = _as_int(
            state["t2t_eligible_tokens"],
            field_name="trainer_state.last.t2t_eligible_tokens",
        )
        result.self_generated_edit_accumulator = coerce_self_generated_edit_accumulator(
            state["self_generated_edit_accumulator"],
            field_name="trainer_state.last.self_generated_edit_accumulator",
        )
        return result

    def to_v5(self) -> dict[str, object]:
        return {
            **self.to_v3(),
            "t2t_eligible_tokens": self.t2t_eligible_tokens,
            "self_generated_edit_accumulator": deepcopy(self.self_generated_edit_accumulator),
        }


@dataclass(slots=True)
class DiagnosticsState:
    initial_parameter_norm_metrics: dict[str, object] = field(default_factory=dict)
    initial_gradient_metrics: dict[str, object] = field(default_factory=dict)
    initial_update_metrics: dict[str, object] = field(default_factory=dict)
    initial_lm_head_row_norm_metrics: dict[str, object] = field(default_factory=dict)
    early_lm_head_row_norm_metrics: dict[str, object] = field(default_factory=dict)
    early_training_diagnostics: dict[str, object] = field(default_factory=dict)
    early_residual_diagnostics: dict[str, object] = field(default_factory=dict)
    fixed_ce_trace_records: int = 0
    last_fixed_ce_trace_record: dict[str, object] = field(default_factory=dict)

    @classmethod
    def fresh(cls) -> DiagnosticsState:
        return cls()

    @classmethod
    def from_v1(cls, value: object) -> DiagnosticsState:
        state = _require_exact_dict(value, DIAGNOSTICS_V1_FIELDS, field_name="trainer_state.diagnostics")
        field_name = "trainer_state.diagnostics"
        return cls(
            initial_parameter_norm_metrics=_json_mapping(
                state["initial_parameter_norm_metrics"],
                field_name=f"{field_name}.initial_parameter_norm_metrics",
            ),
            initial_gradient_metrics=_json_mapping(
                state["initial_gradient_metrics"], field_name=f"{field_name}.initial_gradient_metrics"
            ),
            initial_update_metrics=_json_mapping(
                state["initial_update_metrics"], field_name=f"{field_name}.initial_update_metrics"
            ),
            initial_lm_head_row_norm_metrics=_json_mapping(
                state["initial_lm_head_row_norm_metrics"],
                field_name=f"{field_name}.initial_lm_head_row_norm_metrics",
            ),
            early_lm_head_row_norm_metrics=_json_mapping(
                state["early_lm_head_row_norm_metrics"],
                field_name=f"{field_name}.early_lm_head_row_norm_metrics",
            ),
            early_training_diagnostics=_json_mapping(
                state["early_training_diagnostics"], field_name=f"{field_name}.early_training_diagnostics"
            ),
            early_residual_diagnostics=_json_mapping(
                state["early_residual_diagnostics"], field_name=f"{field_name}.early_residual_diagnostics"
            ),
            fixed_ce_trace_records=_as_strict_int(
                state["fixed_ce_trace_records"], field_name=f"{field_name}.fixed_ce_trace_records"
            ),
            last_fixed_ce_trace_record=_json_mapping(
                state["last_fixed_ce_trace_record"], field_name=f"{field_name}.last_fixed_ce_trace_record"
            ),
        )

    def to_v1(self) -> dict[str, object]:
        return {
            "initial_parameter_norm_metrics": deepcopy(self.initial_parameter_norm_metrics),
            "initial_gradient_metrics": deepcopy(self.initial_gradient_metrics),
            "initial_update_metrics": deepcopy(self.initial_update_metrics),
            "initial_lm_head_row_norm_metrics": deepcopy(self.initial_lm_head_row_norm_metrics),
            "early_lm_head_row_norm_metrics": deepcopy(self.early_lm_head_row_norm_metrics),
            "early_training_diagnostics": deepcopy(self.early_training_diagnostics),
            "early_residual_diagnostics": deepcopy(self.early_residual_diagnostics),
            "fixed_ce_trace_records": self.fixed_ce_trace_records,
            "last_fixed_ce_trace_record": deepcopy(self.last_fixed_ce_trace_record),
        }


def _migrate_pre_v3_branch_telemetry(
    progress: TrainerProgress,
    *,
    interval_has_combined_denominator: bool,
) -> TrainerProgress:
    """Recover pure-M2T denominators or restart inseparable legacy branch ledgers."""

    lineage = progress.lineage
    if lineage.train_t2t_microbatches == 0:
        if lineage.train_m2t_microbatches > 0:
            historical_boundary_microbatches = int(lineage.ar_boundary_counts["microbatches"])
            historical_lossable_tokens = int(lineage.row_telemetry["lossable_tokens_sum"])
            denominator_is_exact = (
                historical_boundary_microbatches == lineage.train_m2t_microbatches
                and historical_lossable_tokens > 0
                and historical_lossable_tokens >= lineage.train_m2t_active_tokens
            )
            if not denominator_is_exact:
                return _restart_pre_v3_branch_telemetry(progress)
            lineage.train_m2t_lossable_tokens = historical_lossable_tokens
            progress.last.m2t_lossable_tokens = int(progress.last.lossable_tokens)
            if interval_has_combined_denominator:
                progress.interval.m2t_lossable_tokens = int(progress.interval.lossable_tokens)
            else:
                progress.interval = IntervalState(
                    smooth_loss=progress.interval.smooth_loss,
                    smooth_count=progress.interval.smooth_count,
                )
        return progress

    return _restart_pre_v3_branch_telemetry(progress)


def _restart_pre_v3_branch_telemetry(progress: TrainerProgress) -> TrainerProgress:
    """Drop legacy branch-only counters when their denominators cannot be reconstructed exactly."""

    lineage = progress.lineage
    lineage.train_m2t_ce_sum = 0.0
    lineage.train_m2t_loss_sum = 0.0
    lineage.train_m2t_loss_count = 0
    lineage.train_m2t_microbatches = 0
    lineage.train_m2t_active_tokens = 0
    lineage.train_t2t_ce_sum = 0.0
    lineage.train_t2t_loss_sum = 0.0
    lineage.train_t2t_loss_count = 0
    lineage.train_t2t_microbatches = 0
    lineage.train_t2t_active_tokens = 0
    lineage.train_token_edit_reconstruction_accumulator = _new_token_edit_accumulator()
    progress.interval = IntervalState(
        smooth_loss=progress.interval.smooth_loss,
        smooth_count=progress.interval.smooth_count,
    )
    progress.last.m2t_ce = 0.0
    progress.last.m2t_loss = 0.0
    progress.last.m2t_microbatches = 0
    progress.last.m2t_active_tokens = 0
    progress.last.t2t_ce = 0.0
    progress.last.t2t_loss = 0.0
    progress.last.t2t_microbatches = 0
    progress.last.t2t_active_tokens = 0
    progress.last.token_edit_reconstruction_accumulator = _new_token_edit_accumulator()
    return progress


def _migrate_v4_t2t_eligibility(progress: TrainerProgress) -> TrainerProgress:
    """Recover the source-generic denominator from a validated v4 payload."""

    self_generated_run = any(progress.lineage.train_self_generated_edit_accumulator.values())
    if self_generated_run:
        progress.lineage.train_t2t_eligible_tokens = progress.lineage.train_t2t_lossable_tokens
        progress.interval.t2t_eligible_tokens = progress.interval.t2t_lossable_tokens
        progress.last.t2t_eligible_tokens = progress.last.t2t_lossable_tokens
    else:
        progress.lineage.train_t2t_eligible_tokens = progress.lineage.train_t2t_ordinary_eligible_tokens
        progress.interval.t2t_eligible_tokens = progress.interval.t2t_ordinary_eligible_tokens
        progress.last.t2t_eligible_tokens = progress.last.t2t_ordinary_eligible_tokens
    return progress


@dataclass(slots=True)
class TrainerProgress:
    lineage: LineageState
    interval: IntervalState
    last: LastStepState
    diagnostics: DiagnosticsState

    @classmethod
    def fresh(cls, max_seq_len: int) -> TrainerProgress:
        max_seq_len = _require_positive_int(max_seq_len, field_name="max_seq_len")
        return cls(
            lineage=LineageState.fresh(max_seq_len),
            interval=IntervalState.fresh(),
            last=LastStepState.fresh(max_seq_len),
            diagnostics=DiagnosticsState.fresh(),
        )

    @classmethod
    def from_legacy_metrics(
        cls,
        *,
        max_seq_len: int,
        resume_metrics: dict[str, object] | None,
        initial_parameter_norm_metrics: dict[str, object],
        gradient_component_metric_keys: Iterable[str],
    ) -> TrainerProgress:
        """Reproduce the pre-v1-checkpoint trainer initialization path."""
        if resume_metrics is not None and not isinstance(resume_metrics, dict):
            raise TypeError("resume_metrics must be a mapping or None")
        progress = cls.fresh(max_seq_len)
        metrics = resume_metrics if resume_metrics else {}

        progress.lineage.grad_clip_event_count = int(metrics.get("grad_clip_event_count", 0))
        progress.lineage.grad_clip_observed_steps = int(metrics.get("grad_clip_observed_steps", 0))

        progress.last.loss_value = float(metrics.get("train_loss", 0.0))
        raw_main_ce = metrics.get("raw_main_ce", 0.0)
        progress.last.main_ce = float(metrics.get("last_train_main_ce", raw_main_ce))
        progress.last.main_loss = float(metrics.get("last_train_main_loss", raw_main_ce))
        progress.last.confidence_penalty = float(metrics.get("last_train_confidence_penalty", 0.0))
        progress.last.confidence_penalty_loss = float(metrics.get("last_train_confidence_penalty_loss", 0.0))
        progress.last.z_loss = float(metrics.get("last_train_z_loss", 0.0))
        progress.last.z_loss_loss = float(metrics.get("last_train_z_loss_loss", 0.0))
        progress.last.eos_hazard_calibration_loss = float(metrics.get("last_train_eos_hazard_calibration_loss", 0.0))
        progress.last.eos_hazard_loss = float(metrics.get("last_train_eos_hazard_loss", 0.0))
        progress.last.m2t_ce = float(metrics.get("last_train_m2t_ce", 0.0))
        progress.last.m2t_loss = float(metrics.get("last_train_m2t_loss", 0.0))
        progress.last.m2t_microbatches = int(metrics.get("last_train_m2t_microbatches", 0))
        progress.last.m2t_active_tokens = int(metrics.get("last_train_m2t_active_tokens", 0))
        progress.last.m2t_lossable_tokens = int(metrics.get("last_train_m2t_lossable_tokens", 0))
        progress.last.t2t_ce = float(metrics.get("last_train_t2t_ce", 0.0))
        progress.last.t2t_loss = float(metrics.get("last_train_t2t_loss", 0.0))
        progress.last.t2t_microbatches = int(metrics.get("last_train_t2t_microbatches", 0))
        progress.last.t2t_active_tokens = int(metrics.get("last_train_t2t_active_tokens", 0))
        progress.last.t2t_lossable_tokens = int(metrics.get("last_train_t2t_lossable_tokens", 0))
        progress.last.t2t_ordinary_eligible_tokens = int(metrics.get("last_train_t2t_ordinary_eligible_tokens", 0))
        progress.last.t2t_eligible_tokens = int(
            metrics.get("last_train_t2t_eligible_tokens", progress.last.t2t_ordinary_eligible_tokens)
        )
        progress.last.grad_norm = float(metrics.get("grad_norm", 0.0))
        progress.last.grad_clipped = bool(metrics.get("last_grad_clipped", False))
        progress.last.grad_clip_ratio = float(metrics.get("last_grad_clip_ratio", 1.0))
        progress.last.active_tokens = int(metrics.get("active_tokens", 0))
        progress.last.lossable_tokens = int(metrics.get("lossable_tokens", 0))
        progress.last.valid_tokens = int(metrics.get("valid_tokens", 0))
        progress.last.content_tokens = int(metrics.get("content_tokens", 0))
        progress.last.content_bytes = int(metrics.get("content_bytes", 0))
        progress.last.class_ce = float(metrics.get("raw_class_ce", 0.0))
        progress.last.within_class_ce = float(metrics.get("raw_within_class_ce", 0.0))
        progress.last.ar_unsmoothed_ce = float(metrics.get("ar_unsmoothed_ce", progress.last.loss_value))
        progress.last.second_moment_norm = float(metrics.get("second_moment_norm", 0.0))
        progress.last.update_norm = float(metrics.get("update_norm", 0.0))
        progress.last.update_grad_ratio = float(metrics.get("update_grad_ratio", 0.0))
        progress.last.entropy_mean = float(metrics.get("train_entropy_mean", 0.0))
        progress.last.top1_prob_mean = float(metrics.get("train_top1_prob_mean", 0.0))
        progress.last.argmax_mode_frac = float(metrics.get("train_argmax_mode_frac", 0.0))
        progress.last.logit_norm_mean = float(metrics.get("last_train_masked_logit_norm_mean", 0.0))
        progress.last.logit_norm_min = float(metrics.get("last_train_masked_logit_norm_min", 0.0))
        progress.last.logit_norm_max = float(metrics.get("last_train_masked_logit_norm_max", 0.0))
        progress.last.logit_rms_mean = float(metrics.get("last_train_masked_logit_rms_mean", 0.0))
        progress.last.logsumexp_mean = float(metrics.get("last_train_masked_logsumexp_mean", 0.0))
        progress.last.logsumexp_min = float(metrics.get("last_train_masked_logsumexp_min", 0.0))
        progress.last.logsumexp_max = float(metrics.get("last_train_masked_logsumexp_max", 0.0))

        gradient_keys = tuple(gradient_component_metric_keys)
        if not all(isinstance(key, str) for key in gradient_keys):
            raise TypeError("gradient_component_metric_keys must contain only strings")
        progress.last.gradient_component_norms = {key: float(metrics.get(key, 0.0)) for key in gradient_keys}
        progress.diagnostics.initial_parameter_norm_metrics = _json_mapping(
            initial_parameter_norm_metrics,
            field_name="initial_parameter_norm_metrics",
        )
        return progress

    @classmethod
    def from_v1(cls, value: object, *, max_seq_len: int) -> TrainerProgress:
        max_seq_len = _require_positive_int(max_seq_len, field_name="max_seq_len")
        state = _require_exact_dict(
            value,
            ("version", "lineage", "interval", "last", "diagnostics"),
            field_name="trainer_state",
        )
        version = state["version"]
        if not isinstance(version, int) or isinstance(version, bool) or version != LEGACY_TRAINER_STATE_VERSION:
            raise RuntimeError(f"unsupported trainer state version {version!r}")
        progress = cls(
            lineage=LineageState.from_v1(state["lineage"], max_seq_len=max_seq_len),
            interval=IntervalState.from_v1(state["interval"]),
            last=LastStepState.from_v1(state["last"], max_seq_len=max_seq_len),
            diagnostics=DiagnosticsState.from_v1(state["diagnostics"]),
        )
        return _migrate_pre_v3_branch_telemetry(progress, interval_has_combined_denominator=False)

    def to_v1(self) -> dict[str, object]:
        return {
            "version": LEGACY_TRAINER_STATE_VERSION,
            "lineage": self.lineage.to_v1(),
            "interval": self.interval.to_v1(),
            "last": self.last.to_v1(),
            "diagnostics": self.diagnostics.to_v1(),
        }

    @classmethod
    def from_v2(cls, value: object, *, max_seq_len: int) -> TrainerProgress:
        max_seq_len = _require_positive_int(max_seq_len, field_name="max_seq_len")
        state = _require_exact_dict(
            value,
            ("version", "lineage", "interval", "last", "diagnostics"),
            field_name="trainer_state",
        )
        version = state["version"]
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version != INTERVAL_DENOMINATOR_TRAINER_STATE_VERSION
        ):
            raise RuntimeError(f"unsupported trainer state version {version!r}")
        progress = cls(
            lineage=LineageState.from_v1(state["lineage"], max_seq_len=max_seq_len),
            interval=IntervalState.from_v2(state["interval"]),
            last=LastStepState.from_v1(state["last"], max_seq_len=max_seq_len),
            diagnostics=DiagnosticsState.from_v1(state["diagnostics"]),
        )
        return _migrate_pre_v3_branch_telemetry(progress, interval_has_combined_denominator=True)

    @classmethod
    def from_checkpoint(cls, value: object, *, max_seq_len: int) -> TrainerProgress:
        if not isinstance(value, dict):
            raise TypeError("trainer_state must be a mapping")
        version = value.get("version")
        if version == LEGACY_TRAINER_STATE_VERSION:
            return cls.from_v1(value, max_seq_len=max_seq_len)
        if version == INTERVAL_DENOMINATOR_TRAINER_STATE_VERSION:
            return cls.from_v2(value, max_seq_len=max_seq_len)
        if version == BRANCH_DENOMINATOR_TRAINER_STATE_VERSION:
            return cls.from_v3(value, max_seq_len=max_seq_len)
        if version == SELF_GENERATED_EDIT_TRAINER_STATE_VERSION:
            return cls.from_v4(value, max_seq_len=max_seq_len)
        if version == RESUME_TRAINER_STATE_VERSION:
            return cls.from_v5(value, max_seq_len=max_seq_len)
        raise RuntimeError(f"unsupported trainer state version {version!r}")

    def to_v2(self) -> dict[str, object]:
        return {
            "version": INTERVAL_DENOMINATOR_TRAINER_STATE_VERSION,
            "lineage": self.lineage.to_v1(),
            "interval": self.interval.to_v2(),
            "last": self.last.to_v1(),
            "diagnostics": self.diagnostics.to_v1(),
        }

    @classmethod
    def from_v3(cls, value: object, *, max_seq_len: int) -> TrainerProgress:
        max_seq_len = _require_positive_int(max_seq_len, field_name="max_seq_len")
        state = _require_exact_dict(
            value,
            ("version", "lineage", "interval", "last", "diagnostics"),
            field_name="trainer_state",
        )
        version = state["version"]
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version != BRANCH_DENOMINATOR_TRAINER_STATE_VERSION
        ):
            raise RuntimeError(f"unsupported trainer state version {version!r}")
        return cls(
            lineage=LineageState.from_v3(state["lineage"], max_seq_len=max_seq_len),
            interval=IntervalState.from_v3(state["interval"]),
            last=LastStepState.from_v3(state["last"], max_seq_len=max_seq_len),
            diagnostics=DiagnosticsState.from_v1(state["diagnostics"]),
        )

    def to_v3(self) -> dict[str, object]:
        return {
            "version": BRANCH_DENOMINATOR_TRAINER_STATE_VERSION,
            "lineage": self.lineage.to_v3(),
            "interval": self.interval.to_v3(),
            "last": self.last.to_v3(),
            "diagnostics": self.diagnostics.to_v1(),
        }

    @classmethod
    def from_v4(cls, value: object, *, max_seq_len: int) -> TrainerProgress:
        max_seq_len = _require_positive_int(max_seq_len, field_name="max_seq_len")
        state = _require_exact_dict(
            value,
            ("version", "lineage", "interval", "last", "diagnostics"),
            field_name="trainer_state",
        )
        version = state["version"]
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version != SELF_GENERATED_EDIT_TRAINER_STATE_VERSION
        ):
            raise RuntimeError(f"unsupported trainer state version {version!r}")
        progress = cls(
            lineage=LineageState.from_v4(state["lineage"], max_seq_len=max_seq_len),
            interval=IntervalState.from_v3(state["interval"]),
            last=LastStepState.from_v4(state["last"], max_seq_len=max_seq_len),
            diagnostics=DiagnosticsState.from_v1(state["diagnostics"]),
        )
        return _migrate_v4_t2t_eligibility(progress)

    def to_v4(self) -> dict[str, object]:
        return {
            "version": SELF_GENERATED_EDIT_TRAINER_STATE_VERSION,
            "lineage": self.lineage.to_v4(),
            "interval": self.interval.to_v3(),
            "last": self.last.to_v4(),
            "diagnostics": self.diagnostics.to_v1(),
        }

    @classmethod
    def from_v5(cls, value: object, *, max_seq_len: int) -> TrainerProgress:
        max_seq_len = _require_positive_int(max_seq_len, field_name="max_seq_len")
        state = _require_exact_dict(
            value,
            ("version", "lineage", "interval", "last", "diagnostics"),
            field_name="trainer_state",
        )
        version = state["version"]
        if not isinstance(version, int) or isinstance(version, bool) or version != RESUME_TRAINER_STATE_VERSION:
            raise RuntimeError(f"unsupported trainer state version {version!r}")
        return cls(
            lineage=LineageState.from_v5(state["lineage"], max_seq_len=max_seq_len),
            interval=IntervalState.from_v5(state["interval"]),
            last=LastStepState.from_v5(state["last"], max_seq_len=max_seq_len),
            diagnostics=DiagnosticsState.from_v1(state["diagnostics"]),
        )

    def to_v5(self) -> dict[str, object]:
        return {
            "version": RESUME_TRAINER_STATE_VERSION,
            "lineage": self.lineage.to_v5(),
            "interval": self.interval.to_v5(),
            "last": self.last.to_v5(),
            "diagnostics": self.diagnostics.to_v1(),
        }

    def to_checkpoint(self) -> dict[str, object]:
        return self.to_v5()
