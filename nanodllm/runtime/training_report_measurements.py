from __future__ import annotations

from .training_features import (
    _is_causal,
    _uses_bd3lm,
    _uses_bd3lm_causal_ar_auxiliary,
    _uses_block_causal_pilot,
    _uses_content_budget_segmented_packing,
    _uses_per_token_noise_levels,
    _uses_self_generated_token_edit,
    _uses_token_edit_corruption,
)
from .training_metrics import (
    _eos_hazard_accumulator_metrics,
    _length_head_accumulator_metrics,
    _parameter_norm_metrics,
    _per_token_effective_masked_fraction_metrics,
    _position_decile_ce_accumulator_metrics,
    _probability_histogram_metrics,
    _token_edit_reconstruction_metrics,
    _value_histogram_metrics,
)
from .training_mtp import mtp_metrics, prediction_quality_metrics
from .training_progress import PER_TOKEN_NOISE_LEVEL_HIST_BINS
from .training_report_inputs import FinalReportInputs
from .training_report_schema import FieldGroup, ReportFragment, merge_field_groups
from .training_self_generated_edit import self_generated_edit_metrics

_EOS_LOSS_FIELDS = frozenset({"last_train_eos_hazard_calibration_loss", "last_train_eos_hazard_loss"})


def _build_loss_measurements(inputs: FinalReportInputs) -> ReportFragment:
    last = inputs.progress.last
    lineage = inputs.progress.lineage
    return ReportFragment(
        "loss_measurements",
        {
            "last_train_loss": last.loss_value,
            "last_train_total_loss": last.loss_value,
            **mtp_metrics(inputs.config, main_loss=last.main_loss, total_loss=last.loss_value),
            "last_train_main_ce": last.main_ce,
            "last_train_main_loss": last.main_loss,
            "last_train_m2t_ce": last.m2t_ce,
            "last_train_m2t_loss": last.m2t_loss,
            "last_train_m2t_microbatches": last.m2t_microbatches,
            "last_train_m2t_active_tokens": last.m2t_active_tokens,
            "last_train_m2t_lossable_tokens": last.m2t_lossable_tokens,
            "last_train_m2t_active_token_frac": last.m2t_active_tokens / max(1, last.lossable_tokens),
            "last_train_m2t_active_token_combined_lossable_share": last.m2t_active_tokens
            / max(1, last.lossable_tokens),
            "last_train_m2t_active_token_frac_within_branch": last.m2t_active_tokens / max(1, last.m2t_lossable_tokens),
            "last_train_t2t_ce": last.t2t_ce,
            "last_train_t2t_loss": last.t2t_loss,
            "last_train_t2t_microbatches": last.t2t_microbatches,
            "last_train_t2t_active_tokens": last.t2t_active_tokens,
            "last_train_t2t_lossable_tokens": last.t2t_lossable_tokens,
            "last_train_t2t_eligible_tokens": last.t2t_eligible_tokens,
            "last_train_t2t_ordinary_eligible_tokens": last.t2t_ordinary_eligible_tokens,
            "last_train_t2t_active_token_frac": last.t2t_active_tokens / max(1, last.lossable_tokens),
            "last_train_t2t_active_token_combined_lossable_share": last.t2t_active_tokens
            / max(1, last.lossable_tokens),
            "last_train_t2t_active_token_frac_within_branch": last.t2t_active_tokens / max(1, last.t2t_lossable_tokens),
            "last_train_t2t_active_token_frac_within_eligible": last.t2t_active_tokens
            / max(1, last.t2t_eligible_tokens),
            "last_train_t2t_active_token_frac_within_ordinary_eligible": last.t2t_active_tokens
            / last.t2t_ordinary_eligible_tokens
            if last.t2t_ordinary_eligible_tokens
            else 0.0,
            "train_m2t_ce_mean": lineage.train_m2t_ce_sum / max(1, lineage.train_m2t_loss_count),
            "train_m2t_loss_mean": lineage.train_m2t_loss_sum / max(1, lineage.train_m2t_loss_count),
            "train_m2t_microbatches": int(lineage.train_m2t_microbatches),
            "train_m2t_active_tokens": int(lineage.train_m2t_active_tokens),
            "train_m2t_lossable_tokens": int(lineage.train_m2t_lossable_tokens),
            "train_m2t_active_token_combined_lossable_share": lineage.train_m2t_active_tokens
            / max(1, lineage.train_m2t_lossable_tokens + lineage.train_t2t_lossable_tokens),
            "train_m2t_active_token_frac_within_branch": lineage.train_m2t_active_tokens
            / max(1, lineage.train_m2t_lossable_tokens),
            "train_t2t_ce_mean": lineage.train_t2t_ce_sum / max(1, lineage.train_t2t_loss_count),
            "train_t2t_loss_mean": lineage.train_t2t_loss_sum / max(1, lineage.train_t2t_loss_count),
            "train_t2t_microbatches": int(lineage.train_t2t_microbatches),
            "train_t2t_active_tokens": int(lineage.train_t2t_active_tokens),
            "train_t2t_lossable_tokens": int(lineage.train_t2t_lossable_tokens),
            "train_t2t_eligible_tokens": int(lineage.train_t2t_eligible_tokens),
            "train_t2t_ordinary_eligible_tokens": int(lineage.train_t2t_ordinary_eligible_tokens),
            "train_t2t_active_token_combined_lossable_share": lineage.train_t2t_active_tokens
            / max(1, lineage.train_m2t_lossable_tokens + lineage.train_t2t_lossable_tokens),
            "train_t2t_active_token_frac_within_branch": lineage.train_t2t_active_tokens
            / max(1, lineage.train_t2t_lossable_tokens),
            "train_t2t_active_token_frac_within_eligible": lineage.train_t2t_active_tokens
            / max(1, lineage.train_t2t_eligible_tokens),
            "train_t2t_active_token_frac_within_ordinary_eligible": lineage.train_t2t_active_tokens
            / lineage.train_t2t_ordinary_eligible_tokens
            if lineage.train_t2t_ordinary_eligible_tokens
            else 0.0,
            "last_train_confidence_penalty": last.confidence_penalty,
            "last_train_confidence_penalty_loss": last.confidence_penalty_loss,
            "last_train_z_loss": last.z_loss,
            "last_train_z_loss_loss": last.z_loss_loss,
            "last_train_eos_hazard_calibration_loss": last.eos_hazard_calibration_loss,
            "last_train_eos_hazard_loss": last.eos_hazard_loss,
        },
    )


def _build_prediction_quality_measurements(inputs: FinalReportInputs) -> ReportFragment:
    last = inputs.progress.last
    has_ar_measurement = _is_causal(inputs.config) or _uses_bd3lm_causal_ar_auxiliary(inputs.config)
    return ReportFragment(
        "prediction_quality_measurements",
        merge_field_groups(
            FieldGroup(
                "autoregressive and logit summaries",
                prediction_quality_metrics(
                    inputs.config, last, causal=_is_causal(inputs.config), has_ar_measurement=has_ar_measurement
                ),
            ),
            FieldGroup("soft-input statistics", {f"last_{key}": value for key, value in last.soft_input_stats.items()}),
            FieldGroup(
                "uniform-estimator statistics",
                {f"last_{key}": value for key, value in last.uniform_estimator_stats.items()},
            ),
            FieldGroup(
                "top-one probability histogram",
                _probability_histogram_metrics("last_train_masked_top1_prob_hist", last.top1_prob_hist_counts),
            ),
            FieldGroup(
                "entropy histogram",
                _value_histogram_metrics("last_train_masked_entropy_hist", last.entropy_hist_counts, 0.0, 16.0),
            ),
            FieldGroup(
                "log-sum-exp histogram",
                _value_histogram_metrics("last_train_masked_logsumexp_hist", last.logsumexp_hist_counts, 0.0, 64.0),
            ),
            FieldGroup(
                "logit-norm histogram",
                _value_histogram_metrics("last_train_masked_logit_norm_hist", last.logit_norm_hist_counts, 0.0, 512.0),
            ),
        ),
    )


def _build_eos_hazard_measurements(inputs: FinalReportInputs) -> ReportFragment:
    last = inputs.progress.last
    fields = _eos_hazard_accumulator_metrics(
        "last_train_",
        last.eos_hazard_accumulator,
        last.eos_hazard_calibration_loss,
        last.eos_hazard_loss,
        float(inputs.config.train.eos_hazard_loss_weight),
    )
    missing = _EOS_LOSS_FIELDS - fields.keys()
    if missing:
        raise KeyError(f"EOS hazard helper is missing expected loss fields: {sorted(missing)}")
    return ReportFragment(
        "eos_hazard_measurements",
        {key: value for key, value in fields.items() if key not in _EOS_LOSS_FIELDS},
    )


def _build_optimization_measurements(inputs: FinalReportInputs) -> ReportFragment:
    last = inputs.progress.last
    lineage = inputs.progress.lineage
    diagnostics = inputs.progress.diagnostics
    return ReportFragment(
        "optimization_measurements",
        merge_field_groups(
            FieldGroup(
                "gradient clipping",
                {
                    "last_class_ce": last.class_ce,
                    "last_within_class_ce": last.within_class_ce,
                    "last_grad_norm": last.grad_norm,
                    "grad_clip_enabled": inputs.grad_clip_enabled,
                    "grad_clip_threshold": inputs.grad_clip_threshold,
                    "last_grad_clipped": last.grad_clipped,
                    "last_grad_clip_ratio": last.grad_clip_ratio,
                    "grad_clip_event_count": lineage.grad_clip_event_count,
                    "grad_clip_observed_steps": lineage.grad_clip_observed_steps,
                    "grad_clip_fraction": lineage.grad_clip_event_count / max(1, lineage.grad_clip_observed_steps),
                },
            ),
            FieldGroup("last gradient component norms", last.gradient_component_norms),
            FieldGroup("initial gradient metrics", diagnostics.initial_gradient_metrics),
            FieldGroup("initial parameter norm metrics", diagnostics.initial_parameter_norm_metrics),
            FieldGroup("initial update metrics", diagnostics.initial_update_metrics),
            FieldGroup("initial lm-head row norms", diagnostics.initial_lm_head_row_norm_metrics),
            FieldGroup("early lm-head row norms", diagnostics.early_lm_head_row_norm_metrics),
            FieldGroup("early training diagnostics", diagnostics.early_training_diagnostics),
            FieldGroup(
                "last update summary",
                {
                    "last_second_moment_norm": last.second_moment_norm,
                    "last_update_norm": last.update_norm,
                    "last_update_grad_ratio": last.update_grad_ratio,
                },
            ),
            FieldGroup(
                "last update component norms",
                {f"last_{key}": value for key, value in last.update_component_norms.items()},
            ),
            FieldGroup(
                "embedding-head diagnostic artifact",
                {
                    "embedding_head_init_diagnostics_enabled": bool(inputs.record_embedding_head_init_diagnostics),
                    "embedding_head_init_diagnostics_artifact": "telemetry/embedding_head_init_diagnostics.jsonl",
                    "embedding_head_init_diagnostics_path": inputs.embedding_head_init_diagnostics_path,
                },
            ),
        ),
    )


def _build_token_measurements(inputs: FinalReportInputs) -> ReportFragment:
    last = inputs.progress.last
    boundary = inputs.progress.lineage.ar_boundary_counts
    return ReportFragment(
        "token_measurements",
        {
            "last_valid_tokens": last.valid_tokens,
            "last_lossable_tokens": last.lossable_tokens,
            "last_active_tokens": last.active_tokens,
            "last_active_token_frac": last.active_tokens / max(1, last.lossable_tokens),
            "last_content_tokens": last.content_tokens,
            "last_content_bytes": last.content_bytes,
            "last_content_token_byte_fertility": last.content_tokens / last.content_bytes
            if last.content_bytes
            else 0.0,
            "train_special_token_count": int(boundary["special_tokens"]),
            "train_special_loss_token_count": int(boundary["special_loss_tokens"]),
            "train_special_token_fraction": boundary["special_tokens"] / max(1, int(boundary["valid_tokens"])),
            "train_special_loss_token_fraction": boundary["special_loss_tokens"]
            / max(1, int(boundary["special_tokens"])),
            "train_bos_token_observed": bool(boundary["bos_tokens"] > 0),
            "train_bos_token_count": int(boundary["bos_tokens"]),
            "train_bos_loss_token_count": int(boundary["bos_loss_tokens"]),
            "train_bos_token_fraction": boundary["bos_tokens"] / max(1, int(boundary["valid_tokens"])),
            "train_bos_loss_token_fraction": boundary["bos_loss_tokens"] / max(1, int(boundary["bos_tokens"])),
            "train_bos_batch_rate": boundary["microbatches_with_bos"] / max(1, int(boundary["microbatches"])),
            "train_eos_token_count": int(boundary["eos_tokens"]),
            "train_eos_loss_token_count": int(boundary["eos_loss_tokens"]),
            "train_eos_token_fraction": boundary["eos_tokens"] / max(1, int(boundary["valid_tokens"])),
            "train_eos_loss_token_fraction": boundary["eos_loss_tokens"] / max(1, int(boundary["eos_tokens"])),
        },
    )


def _build_packing_measurements(inputs: FinalReportInputs) -> ReportFragment:
    lineage = inputs.progress.lineage
    rows = lineage.row_telemetry
    boundary = lineage.boundary_telemetry
    segmented = _uses_content_budget_segmented_packing(inputs.config)
    short_documents = inputs.config.data.short_document_curriculum
    return ReportFragment(
        "packing_measurements",
        {
            "content_budget_segmented_packing_enabled": segmented,
            "content_budget_segmented_target_tokens": int(inputs.config.train.target_tokens),
            "content_budget_segmented_train_valid_tokens": int(inputs.train_valid_tokens) if segmented else 0,
            "content_budget_segmented_train_content_tokens": int(inputs.train_content_tokens) if segmented else 0,
            "train_row_count": int(rows["rows"]),
            "train_row_valid_tokens_mean": rows["valid_tokens_sum"] / max(1, int(rows["rows"])),
            "train_row_valid_tokens_min": int(rows["valid_tokens_min"]),
            "train_row_valid_tokens_max": int(rows["valid_tokens_max"]),
            "train_row_lossable_tokens_mean": rows["lossable_tokens_sum"] / max(1, int(rows["rows"])),
            "train_row_lossable_tokens_min": int(rows["lossable_tokens_min"]),
            "train_row_lossable_tokens_max": int(rows["lossable_tokens_max"]),
            "train_pad_token_count": int(rows["pad_tokens"]),
            "train_pad_fraction": rows["pad_tokens"] / max(1, int(rows["total_row_tokens"])),
            "train_row_eos_fraction": rows["eos_rows"] / max(1, int(rows["rows"])),
            "train_row_eos_tokens_mean": rows["eos_tokens"] / max(1, int(rows["rows"])),
            "train_row_eos_loss_tokens_mean": rows["eos_loss_tokens"] / max(1, int(rows["rows"])),
            "short_document_train_row_count": int(rows["rows"]) if short_documents else 0,
            "short_document_train_valid_tokens_mean": rows["valid_tokens_sum"] / max(1, int(rows["rows"]))
            if short_documents
            else 0.0,
            "short_document_train_valid_tokens_min": int(rows["valid_tokens_min"]) if short_documents else 0,
            "short_document_train_valid_tokens_max": int(rows["valid_tokens_max"]) if short_documents else 0,
            "short_document_train_lossable_tokens_mean": rows["lossable_tokens_sum"] / max(1, int(rows["rows"]))
            if short_documents
            else 0.0,
            "short_document_train_pad_fraction": rows["pad_tokens"] / max(1, int(rows["total_row_tokens"]))
            if short_documents
            else 0.0,
            "short_document_train_eos_row_fraction": rows["eos_rows"] / max(1, int(rows["rows"]))
            if short_documents
            else 0.0,
            "short_document_train_eos_tokens_mean": rows["eos_tokens"] / max(1, int(rows["rows"]))
            if short_documents
            else 0.0,
            "short_document_train_eos_loss_tokens_mean": rows["eos_loss_tokens"] / max(1, int(rows["rows"]))
            if short_documents
            else 0.0,
            "boundary_documents_per_row_mean": boundary["documents_per_row_sum"]
            / max(1, boundary["documents_per_row_count"]),
            "boundary_pad_fraction": rows["pad_tokens"] / max(1, int(rows["total_row_tokens"])),
            "boundary_cross_document_blocked_pairs": boundary["blocked_pairs_sum"],
            "boundary_causal_pairs_total": boundary["causal_pairs_sum"],
            "boundary_cross_document_blocked_fraction": boundary["blocked_pairs_sum"]
            / max(1, boundary["causal_pairs_sum"]),
            "segment_documents_per_row_mean": boundary["documents_per_row_sum"]
            / max(1, boundary["documents_per_row_count"])
            if segmented
            else 0.0,
            "segment_pad_fraction": rows["pad_tokens"] / max(1, int(rows["total_row_tokens"])) if segmented else 0.0,
            "segment_mask_valid_pairs_total": int(boundary["valid_pairs_sum"]) if segmented else 0,
            "segment_mask_allowed_pairs_total": int(boundary["allowed_pairs_sum"]) if segmented else 0,
            "segment_mask_blocked_pairs_total": int(boundary["segment_blocked_pairs_sum"]) if segmented else 0,
            "segment_mask_allowed_fraction": boundary["allowed_pairs_sum"] / max(1, boundary["valid_pairs_sum"])
            if segmented
            else 0.0,
            "segment_mask_blocked_fraction": boundary["segment_blocked_pairs_sum"] / max(1, boundary["valid_pairs_sum"])
            if segmented
            else 0.0,
        },
    )


def _build_resource_and_parameter_measurements(inputs: FinalReportInputs) -> ReportFragment:
    return ReportFragment(
        "resource_and_parameter_measurements",
        merge_field_groups(
            FieldGroup("peak device memory", {"peak_vram_mb": inputs.peak_vram_mb}),
            FieldGroup("final parameter norms", _parameter_norm_metrics(inputs.raw_model)),
            FieldGroup("early residual diagnostics", inputs.progress.diagnostics.early_residual_diagnostics),
        ),
    )


def _build_per_token_noise_measurements(inputs: FinalReportInputs) -> ReportFragment | None:
    if not _uses_per_token_noise_levels(inputs.config):
        return None
    last = inputs.progress.last
    lineage = inputs.progress.lineage
    return ReportFragment(
        "per_token_noise_measurements",
        merge_field_groups(
            FieldGroup(
                "per-token noise contract",
                {
                    "per_token_noise_levels_enabled": True,
                    "per_token_noise_level_hist_bins": PER_TOKEN_NOISE_LEVEL_HIST_BINS,
                    "per_token_noise_schedule_weight_source": "target_token_t",
                    "per_token_noise_loss_normalization": "sum_ce_over_t_div_lossable_tokens",
                    "per_token_noise_fixed_ce_eval_grid": "standard_scalar_t",
                },
            ),
            FieldGroup(
                "last per-token noise histogram",
                _value_histogram_metrics(
                    "last_train_per_token_noise_level_hist",
                    last.per_token_noise_level_hist_counts,
                    0.0,
                    1.0,
                ),
            ),
            FieldGroup(
                "aggregate per-token noise histogram",
                _value_histogram_metrics(
                    "train_per_token_noise_level_hist",
                    lineage.train_per_token_noise_level_hist_counts,
                    0.0,
                    1.0,
                ),
            ),
            FieldGroup(
                "last effective masked fraction",
                _per_token_effective_masked_fraction_metrics(
                    "last_train_",
                    last.per_token_effective_masked_fraction_sum,
                    last.per_token_effective_masked_fraction_count,
                    last.per_token_effective_masked_fraction_min,
                    last.per_token_effective_masked_fraction_max,
                ),
            ),
            FieldGroup(
                "aggregate effective masked fraction",
                _per_token_effective_masked_fraction_metrics(
                    "train_",
                    lineage.train_per_token_effective_masked_fraction_sum,
                    lineage.train_per_token_effective_masked_fraction_count,
                    lineage.train_per_token_effective_masked_fraction_min,
                    lineage.train_per_token_effective_masked_fraction_max,
                ),
            ),
        ),
    )


def _build_fixed_ce_trace_measurements(inputs: FinalReportInputs) -> ReportFragment:
    last_record: dict[str, object] = {}
    for key, value in inputs.progress.diagnostics.last_fixed_ce_trace_record.items():
        if isinstance(value, bool) or not isinstance(value, (float, int, str)):
            continue
        metric_key = f"last_{key}" if key.startswith("fixed_ce_trace_") else f"fixed_ce_trace_last_{key}"
        last_record[metric_key] = value
    return ReportFragment(
        "fixed_ce_trace_measurements",
        merge_field_groups(
            FieldGroup(
                "fixed CE trace contract",
                {
                    "fixed_ce_trace_enabled": bool(inputs.config.eval.fixed_ce_trace_enabled),
                    "fixed_ce_trace_artifact": "telemetry/fixed_noise_ce_trace.jsonl",
                    "fixed_ce_trace_path": inputs.fixed_ce_trace_path,
                    "fixed_ce_trace_steps": list(inputs.fixed_ce_trace_steps),
                    "fixed_ce_trace_record_count": int(inputs.progress.diagnostics.fixed_ce_trace_records),
                    "fixed_ce_trace_batches": int(inputs.config.eval.fixed_ce_batches),
                },
            ),
            FieldGroup("last fixed CE trace record", last_record),
        ),
    )


def _build_optional_measurements(inputs: FinalReportInputs) -> tuple[ReportFragment, ...]:
    fragments: list[ReportFragment] = []
    per_token_noise = _build_per_token_noise_measurements(inputs)
    if per_token_noise is not None:
        fragments.append(per_token_noise)
    if _uses_bd3lm(inputs.config):
        fragments.append(
            ReportFragment(
                "bd3lm_measurements",
                {
                    "bd3lm_loss_target_tokens_last": inputs.progress.last.active_tokens,
                    "bd3lm_lossable_noised_half_tokens_last": inputs.progress.last.lossable_tokens,
                },
            )
        )
    if _uses_block_causal_pilot(inputs.config):
        count = inputs.progress.lineage.block_causal_visible_prefix_count
        fragments.append(
            ReportFragment(
                "block_causal_pilot_measurements",
                {
                    "block_causal_pilot_visible_prefix_frac_mean": inputs.progress.lineage.block_causal_visible_prefix_frac_sum
                    / count
                    if count
                    else 0.0
                },
            )
        )
    fragments.append(_build_fixed_ce_trace_measurements(inputs))
    if inputs.config.train.position_decile_balanced_loss:
        fragments.append(
            ReportFragment(
                "position_decile_measurements",
                _position_decile_ce_accumulator_metrics(
                    "last_train_", inputs.progress.last.position_decile_ce_accumulator, True
                ),
            )
        )
    if _uses_token_edit_corruption(inputs.config) and not _uses_self_generated_token_edit(inputs.config):
        fragments.append(
            ReportFragment(
                "token_edit_reconstruction_measurements",
                merge_field_groups(
                    FieldGroup(
                        "last token-edit reconstruction",
                        _token_edit_reconstruction_metrics(
                            "last_train_", inputs.progress.last.token_edit_reconstruction_accumulator, True
                        ),
                    ),
                    FieldGroup(
                        "aggregate token-edit reconstruction",
                        _token_edit_reconstruction_metrics(
                            "train_",
                            inputs.progress.lineage.train_token_edit_reconstruction_accumulator,
                            True,
                        ),
                    ),
                ),
            )
        )
    if _uses_self_generated_token_edit(inputs.config):
        fragments.append(
            ReportFragment(
                "self_generated_token_edit_measurements",
                merge_field_groups(
                    FieldGroup(
                        "last self-generated token edit",
                        self_generated_edit_metrics(
                            "last_train_self_generated_edit_",
                            inputs.progress.last.self_generated_edit_accumulator,
                        ),
                    ),
                    FieldGroup(
                        "aggregate self-generated token edit",
                        self_generated_edit_metrics(
                            "train_self_generated_edit_",
                            inputs.progress.lineage.train_self_generated_edit_accumulator,
                        ),
                    ),
                ),
            )
        )
    if inputs.config.train.length_head_loss_weight > 0:
        loss_weight = float(inputs.config.train.length_head_loss_weight)
        fragments.append(
            ReportFragment(
                "length_head_measurements",
                merge_field_groups(
                    FieldGroup(
                        "last length-head metrics",
                        _length_head_accumulator_metrics(
                            "last_train_", inputs.progress.last.length_head_accumulator, loss_weight
                        ),
                    ),
                    FieldGroup(
                        "aggregate length-head metrics",
                        _length_head_accumulator_metrics(
                            "train_", inputs.progress.lineage.train_length_head_accumulator, loss_weight
                        ),
                    ),
                ),
            )
        )
    return tuple(fragments)


def build_final_measurement_fragments_v1(inputs: FinalReportInputs) -> tuple[ReportFragment, ...]:
    """Build the ordered, single-owner measured-outcome portion of final metrics v1."""

    return (
        _build_loss_measurements(inputs),
        _build_prediction_quality_measurements(inputs),
        _build_eos_hazard_measurements(inputs),
        _build_optimization_measurements(inputs),
        _build_token_measurements(inputs),
        _build_packing_measurements(inputs),
        _build_resource_and_parameter_measurements(inputs),
        *_build_optional_measurements(inputs),
    )
