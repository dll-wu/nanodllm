"""Public eval API, implemented in :mod:`nanodllm.runtime.eval`."""

from .runtime.eval import (
    NELBO_SEQUENCE_RECORD_SCHEMA_VERSION,
    JudgeModel,
    evaluate_ar_next_token_conditional_rank_suffix,
    evaluate_ar_validation_prefix_continuations,
    evaluate_argmax_basin_diagnostic,
    evaluate_bos_only_first_token,
    evaluate_conditional_generation_metrics,
    evaluate_conditional_rank_rollout,
    evaluate_conditional_rank_suffix,
    evaluate_gen_ppl,
    evaluate_generation_metrics,
    evaluate_nelbo,
    evaluate_reconstruction_metrics,
    model_uses_target_length_conditioning,
    target_eos_lengths,
)

__all__ = [
    "JudgeModel",
    "NELBO_SEQUENCE_RECORD_SCHEMA_VERSION",
    "evaluate_ar_next_token_conditional_rank_suffix",
    "evaluate_ar_validation_prefix_continuations",
    "evaluate_argmax_basin_diagnostic",
    "evaluate_bos_only_first_token",
    "evaluate_conditional_generation_metrics",
    "evaluate_conditional_rank_rollout",
    "evaluate_conditional_rank_suffix",
    "evaluate_gen_ppl",
    "evaluate_generation_metrics",
    "evaluate_nelbo",
    "evaluate_reconstruction_metrics",
    "model_uses_target_length_conditioning",
    "target_eos_lengths",
]
