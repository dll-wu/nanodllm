"""Count-preserving repair summaries and diagnostic exact-edit scores."""

from __future__ import annotations

from collections.abc import Mapping

import torch

EXACT_EDIT_DIAGNOSTIC_SCHEMA = "nanodllm.token_edit_exact_edit_diagnostics.v1"

_TRANSITION_FIELDS = ("c_to_c_count", "c_to_w_count", "w_to_c_count", "w_to_w_count")
_COUNT_FIELDS = ("eligible_count", *_TRANSITION_FIELDS, "changed_count", "changed_wrong_to_wrong_count")


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def new_group_accumulator() -> dict[str, int]:
    return {
        **dict.fromkeys(_COUNT_FIELDS, 0),
        "rows_with_eligible": 0,
        "draft_exact_rows": 0,
        "final_exact_rows": 0,
    }


def row_transition_counts(
    draft_correct: torch.Tensor,
    final_correct: torch.Tensor,
    eligible: torch.Tensor,
    changed: torch.Tensor,
) -> dict[str, int]:
    wrong_to_wrong = eligible & ~draft_correct & ~final_correct
    return {
        "eligible_count": int(eligible.sum().item()),
        "c_to_c_count": int((eligible & draft_correct & final_correct).sum().item()),
        "c_to_w_count": int((eligible & draft_correct & ~final_correct).sum().item()),
        "w_to_c_count": int((eligible & ~draft_correct & final_correct).sum().item()),
        "w_to_w_count": int(wrong_to_wrong.sum().item()),
        "changed_count": int((eligible & changed).sum().item()),
        "changed_wrong_to_wrong_count": int((wrong_to_wrong & changed).sum().item()),
    }


def _count_metrics(counts: Mapping[str, int]) -> dict[str, float | int | bool]:
    eligible = counts["eligible_count"]
    c_to_c, c_to_w, w_to_c, w_to_w = (counts[field] for field in _TRANSITION_FIELDS)
    initial_correct = c_to_c + c_to_w
    initial_wrong = w_to_c + w_to_w
    final_error = c_to_w + w_to_w
    net_repair = w_to_c - c_to_w
    error_reduction = initial_wrong - final_error
    changed = counts["changed_count"]
    changed_wrong = counts["changed_wrong_to_wrong_count"]
    # Wrong-to-different-wrong is both an unsuccessful edit and an unrepaired error.
    # Aggregate these counts before computing ratios; do not average row-level F1.
    return {
        **counts,
        "initial_correct_count": initial_correct,
        "initial_wrong_count": initial_wrong,
        "draft_error_count": initial_wrong,
        "final_error_count": final_error,
        "net_repair_count": net_repair,
        "c_to_c_conditional_rate": ratio(c_to_c, initial_correct),
        "c_to_w_conditional_rate": ratio(c_to_w, initial_correct),
        "w_to_c_conditional_rate": ratio(w_to_c, initial_wrong),
        "w_to_w_conditional_rate": ratio(w_to_w, initial_wrong),
        "correct_token_retention_rate": ratio(c_to_c, initial_correct),
        "damage_rate": ratio(c_to_w, initial_correct),
        "repair_rate": ratio(w_to_c, initial_wrong),
        "draft_error_rate": ratio(initial_wrong, eligible),
        "final_error_rate": ratio(final_error, eligible),
        "net_repair_gain": ratio(net_repair, eligible),
        "error_rate_reduction": ratio(error_reduction, eligible),
        "net_repair_identity_count": net_repair - error_reduction,
        "net_repair_identity_holds": net_repair == error_reduction,
        "unchanged_wrong_count": w_to_w - changed_wrong,
        "exact_edit_true_positive_count": w_to_c,
        "exact_edit_false_positive_count": changed - w_to_c,
        "exact_edit_false_negative_count": initial_wrong - w_to_c,
        "exact_edit_precision": ratio(w_to_c, changed),
        "exact_edit_recall": ratio(w_to_c, initial_wrong),
        "exact_edit_f1": ratio(2 * w_to_c, changed + initial_wrong),
        "exact_edit_precision_defined": changed > 0,
        "exact_edit_recall_defined": initial_wrong > 0,
        "exact_edit_f1_defined": changed + initial_wrong > 0,
    }


def row_group_record(counts: dict[str, int]) -> dict[str, object]:
    eligible = counts["eligible_count"]
    return {
        **_count_metrics(counts),
        "draft_exact": None if eligible == 0 else counts["w_to_c_count"] + counts["w_to_w_count"] == 0,
        "final_exact": None if eligible == 0 else counts["c_to_w_count"] + counts["w_to_w_count"] == 0,
    }


def update_group_accumulator(accumulator: dict[str, int], row_counts: dict[str, int]) -> None:
    for key in _COUNT_FIELDS:
        accumulator[key] += row_counts[key]
    if row_counts["eligible_count"] == 0:
        return
    accumulator["rows_with_eligible"] += 1
    if row_counts["w_to_c_count"] + row_counts["w_to_w_count"] == 0:
        accumulator["draft_exact_rows"] += 1
    if row_counts["c_to_w_count"] + row_counts["w_to_w_count"] == 0:
        accumulator["final_exact_rows"] += 1


def merge_group_accumulator(destination: dict[str, int], source: dict[str, int]) -> None:
    if destination.keys() != source.keys():
        raise ValueError("transition group accumulators do not match")
    for key, value in source.items():
        destination[key] += int(value)


def transition_count_view(accumulator: dict[str, int]) -> dict[str, int]:
    """Keep the existing correctness-only marginal strata contract unchanged."""

    return {field: int(accumulator[field]) for field in ("eligible_count", *_TRANSITION_FIELDS)}


def group_metrics(accumulator: dict[str, int]) -> dict[str, float | int | bool]:
    rows = accumulator["rows_with_eligible"]
    return {
        **_count_metrics(accumulator),
        "draft_exact_row_rate": ratio(accumulator["draft_exact_rows"], rows),
        "final_exact_row_rate": ratio(accumulator["final_exact_rows"], rows),
    }
