"""Aggregation and reporting for generation sampler traces."""

from __future__ import annotations


def merge_sample_trace(totals: dict[str, float | int | None], trace: dict[str, float | int]) -> None:
    """Accumulate one sampler trace into a run-level sufficient-statistics ledger."""

    totals["batches"] = int(totals["batches"] or 0) + 1
    totals["steps"] = int(totals["steps"] or 0) + int(trace.get("steps", 0))
    totals["candidate_count"] = int(totals["candidate_count"] or 0) + int(trace.get("candidate_count", 0))
    totals["commit_count"] = int(totals["commit_count"] or 0) + int(trace.get("commit_count", 0))
    totals["candidate_entropy_sum"] = float(totals["candidate_entropy_sum"] or 0.0) + float(
        trace.get("candidate_entropy_sum", 0.0)
    )
    totals["candidate_top1_prob_sum"] = float(totals["candidate_top1_prob_sum"] or 0.0) + float(
        trace.get("candidate_top1_prob_sum", 0.0)
    )
    totals["pred_max_token_frac_sum"] = float(totals["pred_max_token_frac_sum"] or 0.0) + float(
        trace.get("pred_max_token_frac_sum", 0.0)
    )
    totals["pred_max_token_frac_steps"] = int(totals["pred_max_token_frac_steps"] or 0) + int(
        trace.get("pred_max_token_frac_steps", 0)
    )
    for key in (
        "class_entropy_sum",
        "class_max_prob_sum",
        "class_prob_eos_special_sum",
        "class_prob_low_freq_sum",
        "class_prob_high_freq_sum",
    ):
        totals[key] = float(totals.get(key) or 0.0) + float(trace.get(key, 0.0))
    totals["final_mask_count"] = int(totals["final_mask_count"] or 0) + int(trace.get("final_mask_count", 0))
    totals["remask_count"] = int(totals.get("remask_count") or 0) + int(trace.get("remask_count", 0))
    totals["num_blocks"] = int(totals.get("num_blocks") or 0) + int(trace.get("num_blocks", 0))
    totals["nfe_per_block"] = float(totals.get("nfe_per_block") or 0.0) + float(trace.get("nfe_per_block", 0.0))
    for key in (
        "bd3lm_actual_nfe",
        "bd3lm_model_forward_calls",
        "bd3lm_denoise_model_calls",
        "bd3lm_prefill_model_calls",
        "bd3lm_finalized_append_model_calls",
        "bd3lm_processed_query_tokens",
        "bd3lm_denoise_processed_query_tokens",
        "bd3lm_prefill_processed_query_tokens",
        "bd3lm_finalized_append_processed_query_tokens",
        "bd3lm_denoise_cached_kv_token_positions",
        "bd3lm_denoise_active_kv_token_positions",
        "bd3lm_generated_blocks",
        "bd3lm_prefix_sample_count",
        "bd3lm_prefix_preserved_rows",
        "bd3lm_prefix_token_count",
        "bd3lm_prefix_preserved_tokens",
        "bd3lm_full_recompute_no_cache",
        "bd3lm_topology_equivalent_completed_prefix_cache",
        "bd3lm_completed_prefix_dependency_topology_equivalent",
        "bd3lm_legacy_full_shape_trajectory_equivalence_guaranteed",
        "bd3lm_model_phase_timing_measured",
    ):
        totals[key] = int(totals.get(key) or 0) + int(trace.get(key, 0))
    for source_key, total_key in (
        ("bd3lm_cache_capacity_tokens_per_sequence", "bd3lm_cache_capacity_tokens_per_sequence_max"),
        ("bd3lm_cache_high_water_tokens_per_sequence", "bd3lm_cache_high_water_tokens_per_sequence_max"),
        ("bd3lm_cache_allocated_kv_bytes", "bd3lm_cache_allocated_kv_bytes_max"),
        ("bd3lm_cache_high_water_live_kv_bytes", "bd3lm_cache_high_water_live_kv_bytes_max"),
    ):
        totals[total_key] = max(int(totals.get(total_key) or 0), int(trace.get(source_key, 0)))
    for key in (
        "bd3lm_model_forward_seconds",
        "bd3lm_denoise_model_seconds",
        "bd3lm_prefill_model_seconds",
        "bd3lm_finalized_append_model_seconds",
    ):
        totals[key] = float(totals.get(key) or 0.0) + float(trace.get(key, 0.0))
    for key, value in trace.items():
        if not key.startswith("length_policy_") or not isinstance(value, (int, float)):
            continue
        if key.endswith("_position_max_len"):
            totals[key] = max(int(totals.get(key) or 0), int(value))
        else:
            totals[key] = float(totals.get(key) or 0.0) + float(value)

    commit_min = int(trace.get("commit_count_min", 0))
    commit_max = int(trace.get("commit_count_max", 0))
    totals["commit_count_min"] = (
        commit_min if totals["commit_count_min"] is None else min(int(totals["commit_count_min"]), commit_min)
    )
    totals["commit_count_max"] = (
        commit_max if totals["commit_count_max"] is None else max(int(totals["commit_count_max"]), commit_max)
    )


def new_sample_trace_totals() -> dict[str, float | int | None]:
    """Create an empty generation-trace sufficient-statistics ledger."""

    return {
        "batches": 0,
        "steps": 0,
        "candidate_count": 0,
        "commit_count": 0,
        "commit_count_min": None,
        "commit_count_max": None,
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
        "num_blocks": 0,
        "nfe_per_block": 0.0,
        "bd3lm_actual_nfe": 0,
        "bd3lm_model_forward_calls": 0,
        "bd3lm_denoise_model_calls": 0,
        "bd3lm_prefill_model_calls": 0,
        "bd3lm_finalized_append_model_calls": 0,
        "bd3lm_processed_query_tokens": 0,
        "bd3lm_denoise_processed_query_tokens": 0,
        "bd3lm_prefill_processed_query_tokens": 0,
        "bd3lm_finalized_append_processed_query_tokens": 0,
        "bd3lm_denoise_cached_kv_token_positions": 0,
        "bd3lm_denoise_active_kv_token_positions": 0,
        "bd3lm_cache_allocated_kv_bytes_max": 0,
        "bd3lm_cache_high_water_live_kv_bytes_max": 0,
        "bd3lm_cache_capacity_tokens_per_sequence_max": 0,
        "bd3lm_cache_high_water_tokens_per_sequence_max": 0,
        "bd3lm_model_forward_seconds": 0.0,
        "bd3lm_denoise_model_seconds": 0.0,
        "bd3lm_prefill_model_seconds": 0.0,
        "bd3lm_finalized_append_model_seconds": 0.0,
        "bd3lm_generated_blocks": 0,
        "bd3lm_prefix_sample_count": 0,
        "bd3lm_prefix_preserved_rows": 0,
        "bd3lm_prefix_token_count": 0,
        "bd3lm_prefix_preserved_tokens": 0,
        "bd3lm_full_recompute_no_cache": 0,
        "bd3lm_topology_equivalent_completed_prefix_cache": 0,
        "bd3lm_completed_prefix_dependency_topology_equivalent": 0,
        "bd3lm_legacy_full_shape_trajectory_equivalence_guaranteed": 0,
        "bd3lm_model_phase_timing_measured": 0,
    }


def sample_trace_metrics(totals: dict[str, float | int | None]) -> dict[str, float | int]:
    """Materialize public generation metrics from trace sufficient statistics."""

    batches = int(totals["batches"] or 0)
    steps = int(totals["steps"] or 0)
    candidate_count = int(totals["candidate_count"] or 0)
    pred_frac_steps = int(totals["pred_max_token_frac_steps"] or 0)
    num_blocks = int(totals.get("num_blocks") or 0)
    nfe_per_block = float(totals.get("nfe_per_block") or 0.0)
    bd3lm_actual_nfe = int(totals.get("bd3lm_actual_nfe") or 0)
    bd3lm_model_forward_calls = int(totals.get("bd3lm_model_forward_calls") or 0)
    bd3lm_denoise_model_calls = int(totals.get("bd3lm_denoise_model_calls") or 0)
    bd3lm_prefill_model_calls = int(totals.get("bd3lm_prefill_model_calls") or 0)
    bd3lm_append_model_calls = int(totals.get("bd3lm_finalized_append_model_calls") or 0)
    bd3lm_generated_blocks = int(totals.get("bd3lm_generated_blocks") or 0)
    bd3lm_prefix_sample_count = int(totals.get("bd3lm_prefix_sample_count") or 0)
    bd3lm_prefix_preserved_rows = int(totals.get("bd3lm_prefix_preserved_rows") or 0)
    bd3lm_prefix_token_count = int(totals.get("bd3lm_prefix_token_count") or 0)
    bd3lm_prefix_preserved_tokens = int(totals.get("bd3lm_prefix_preserved_tokens") or 0)
    bd3lm_full_recompute_batches = int(totals.get("bd3lm_full_recompute_no_cache") or 0)
    bd3lm_cached_batches = int(totals.get("bd3lm_topology_equivalent_completed_prefix_cache") or 0)
    bd3lm_dependency_topology_equivalent_batches = int(
        totals.get("bd3lm_completed_prefix_dependency_topology_equivalent") or 0
    )
    bd3lm_trajectory_guaranteed_batches = int(
        totals.get("bd3lm_legacy_full_shape_trajectory_equivalence_guaranteed") or 0
    )
    bd3lm_timed_batches = int(totals.get("bd3lm_model_phase_timing_measured") or 0)
    metrics: dict[str, float | int] = {
        "gen_trace_batches": batches,
        "gen_trace_steps_mean": steps / batches if batches else 0.0,
        "gen_trace_candidate_tokens": candidate_count,
        "gen_trace_commit_tokens": int(totals["commit_count"] or 0),
        "gen_trace_commit_tokens_mean": int(totals["commit_count"] or 0) / steps if steps else 0.0,
        "gen_trace_commit_tokens_min": int(totals["commit_count_min"] or 0),
        "gen_trace_commit_tokens_max": int(totals["commit_count_max"] or 0),
        "gen_trace_candidate_entropy_mean": float(totals["candidate_entropy_sum"] or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_candidate_top1_prob_mean": float(totals["candidate_top1_prob_sum"] or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_pred_max_token_frac_mean": float(totals["pred_max_token_frac_sum"] or 0.0) / pred_frac_steps
        if pred_frac_steps
        else 0.0,
        "gen_trace_class_entropy_mean": float(totals.get("class_entropy_sum") or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_class_max_prob_mean": float(totals.get("class_max_prob_sum") or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_class_prob_eos_special_mean": float(totals.get("class_prob_eos_special_sum") or 0.0)
        / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_class_prob_low_freq_mean": float(totals.get("class_prob_low_freq_sum") or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_class_prob_high_freq_mean": float(totals.get("class_prob_high_freq_sum") or 0.0) / candidate_count
        if candidate_count
        else 0.0,
        "gen_trace_final_mask_count": int(totals["final_mask_count"] or 0),
        "gen_trace_final_mask_count_mean": int(totals["final_mask_count"] or 0) / batches if batches else 0.0,
        "gen_trace_remask_count": int(totals.get("remask_count") or 0),
        "gen_trace_remask_tokens_mean": int(totals.get("remask_count") or 0) / steps if steps else 0.0,
        "gen_trace_num_blocks": num_blocks,
        "gen_trace_num_blocks_mean": num_blocks / batches if batches else 0.0,
        "gen_trace_nfe_per_block": nfe_per_block / batches if batches else 0.0,
    }
    if bd3lm_actual_nfe or bd3lm_generated_blocks or bd3lm_prefix_sample_count:
        metrics.update(
            {
                "gen_trace_bd3lm_actual_nfe": bd3lm_actual_nfe,
                "gen_trace_bd3lm_actual_nfe_mean": bd3lm_actual_nfe / batches if batches else 0.0,
                "gen_trace_bd3lm_model_forward_calls": bd3lm_model_forward_calls,
                "gen_trace_bd3lm_denoise_model_calls": bd3lm_denoise_model_calls,
                "gen_trace_bd3lm_prefill_model_calls": bd3lm_prefill_model_calls,
                "gen_trace_bd3lm_finalized_append_model_calls": bd3lm_append_model_calls,
                "gen_trace_bd3lm_processed_query_tokens": int(totals.get("bd3lm_processed_query_tokens") or 0),
                "gen_trace_bd3lm_denoise_processed_query_tokens": int(
                    totals.get("bd3lm_denoise_processed_query_tokens") or 0
                ),
                "gen_trace_bd3lm_prefill_processed_query_tokens": int(
                    totals.get("bd3lm_prefill_processed_query_tokens") or 0
                ),
                "gen_trace_bd3lm_finalized_append_processed_query_tokens": int(
                    totals.get("bd3lm_finalized_append_processed_query_tokens") or 0
                ),
                "gen_trace_bd3lm_denoise_cached_kv_token_positions": int(
                    totals.get("bd3lm_denoise_cached_kv_token_positions") or 0
                ),
                "gen_trace_bd3lm_denoise_active_kv_token_positions": int(
                    totals.get("bd3lm_denoise_active_kv_token_positions") or 0
                ),
                "gen_trace_bd3lm_cache_allocated_kv_bytes_max": int(
                    totals.get("bd3lm_cache_allocated_kv_bytes_max") or 0
                ),
                "gen_trace_bd3lm_cache_high_water_live_kv_bytes_max": int(
                    totals.get("bd3lm_cache_high_water_live_kv_bytes_max") or 0
                ),
                "gen_trace_bd3lm_cache_capacity_tokens_per_sequence_max": int(
                    totals.get("bd3lm_cache_capacity_tokens_per_sequence_max") or 0
                ),
                "gen_trace_bd3lm_cache_high_water_tokens_per_sequence_max": int(
                    totals.get("bd3lm_cache_high_water_tokens_per_sequence_max") or 0
                ),
                "gen_trace_bd3lm_generated_blocks": bd3lm_generated_blocks,
                "gen_trace_bd3lm_prefix_sample_count": bd3lm_prefix_sample_count,
                "gen_trace_bd3lm_prefix_preserved_rows": bd3lm_prefix_preserved_rows,
                "gen_trace_bd3lm_prefix_preserved_rate": bd3lm_prefix_preserved_rows / bd3lm_prefix_sample_count
                if bd3lm_prefix_sample_count
                else 0.0,
                "gen_trace_bd3lm_prefix_token_count": bd3lm_prefix_token_count,
                "gen_trace_bd3lm_prefix_preserved_tokens": bd3lm_prefix_preserved_tokens,
                "gen_trace_bd3lm_prefix_token_preserved_rate": (
                    bd3lm_prefix_preserved_tokens / bd3lm_prefix_token_count if bd3lm_prefix_token_count else 1.0
                ),
                "gen_trace_bd3lm_full_recompute_batches": bd3lm_full_recompute_batches,
                "gen_trace_bd3lm_full_recompute_rate": bd3lm_full_recompute_batches / batches if batches else 0.0,
                "gen_trace_bd3lm_topology_equivalent_completed_prefix_cache_batches": bd3lm_cached_batches,
                "gen_trace_bd3lm_topology_equivalent_completed_prefix_cache_rate": bd3lm_cached_batches / batches
                if batches
                else 0.0,
                "gen_trace_bd3lm_completed_prefix_dependency_topology_equivalent_batches": (
                    bd3lm_dependency_topology_equivalent_batches
                ),
                "gen_trace_bd3lm_completed_prefix_dependency_topology_equivalent_rate": (
                    bd3lm_dependency_topology_equivalent_batches / batches
                )
                if batches
                else 0.0,
                "gen_trace_bd3lm_legacy_full_shape_trajectory_equivalence_guaranteed_batches": (
                    bd3lm_trajectory_guaranteed_batches
                ),
                "gen_trace_bd3lm_legacy_full_shape_trajectory_equivalence_guaranteed_rate": (
                    bd3lm_trajectory_guaranteed_batches / batches if batches else 0.0
                ),
                "gen_trace_bd3lm_model_phase_timing_measured_batches": bd3lm_timed_batches,
                "gen_trace_bd3lm_model_phase_timing_coverage": bd3lm_timed_batches / batches if batches else 0.0,
            }
        )
        if batches and bd3lm_timed_batches == batches:
            metrics.update(
                {
                    "gen_trace_bd3lm_model_forward_seconds": float(totals.get("bd3lm_model_forward_seconds") or 0.0),
                    "gen_trace_bd3lm_denoise_model_seconds": float(totals.get("bd3lm_denoise_model_seconds") or 0.0),
                    "gen_trace_bd3lm_prefill_model_seconds": float(totals.get("bd3lm_prefill_model_seconds") or 0.0),
                    "gen_trace_bd3lm_finalized_append_model_seconds": float(
                        totals.get("bd3lm_finalized_append_model_seconds") or 0.0
                    ),
                }
            )
    for key, value in totals.items():
        if not key.startswith("length_policy_") or value is None:
            continue
        metric_key = f"gen_{key}"
        if isinstance(value, int) or float(value).is_integer():
            metrics[metric_key] = int(value)
        else:
            metrics[metric_key] = float(value)
    count = int(metrics.get("gen_length_policy_predicted_position_count", 0))
    if count:
        metrics["gen_length_policy_predicted_position_mean"] = (
            float(metrics.get("gen_length_policy_predicted_position_sum", 0)) / count
        )
        metrics["gen_length_policy_predicted_length_mean"] = (
            float(metrics.get("gen_length_policy_predicted_length_sum", 0)) / count
        )
    return metrics
