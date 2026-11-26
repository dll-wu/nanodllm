"""Training-compute estimates shared by interval and final reporting."""

from __future__ import annotations

from .training_mtp import mtp_compute_metrics, mtp_training_enabled
from .training_variable_depth import metrics as variable_depth_metrics

BASE_TRAINING_FLOPS_ACCOUNTING_METHOD = "base_training_estimate"
BD3LM_DENSE_CONCAT_FLOPS_ACCOUNTING_METHOD = "bd3lm_dense_x_t_x_0_concat_training_estimate"


def resolve_training_model_accounting(session, bd3lm_estimate=None) -> None:
    """Resolve architecture identity and count only branches executed in training."""
    model, config = session.raw_model, session.config
    session.counts = model.parameter_counts()
    session.flops_per_token = bd3lm_estimate(model) if bd3lm_estimate is not None else model.estimate_flops()
    if config.model.mtp_mode != "none" and not mtp_training_enabled(config):
        session.flops_per_token = model.estimate_flops(include_mtp=False)
    session.architecture_fingerprint = model.architecture_fingerprint()


def compute_budget_fields(inputs, compute) -> dict:
    """Map the compute estimate to the stable final-report ledger."""
    return {
        "flops_per_token": inputs.flops_per_token,
        **mtp_compute_metrics(inputs.config, inputs.raw_model, inputs.train_tokens),
        **variable_depth_metrics(inputs.config, inputs.train_tokens // inputs.config.train.total_batch_size),
        "flops_accounting_method": compute["accounting_method"],
        "base_training_flops_used": compute["base_training_flops"],
        "self_generated_draft_forward_flops_per_token": compute["draft_forward_flops_per_token"],
        "self_generated_draft_forward_input_tokens": compute["draft_forward_input_tokens"],
        "self_generated_draft_forward_flops_used": compute["draft_forward_flops"],
        "self_generated_draft_backward_flops_used": compute["draft_backward_flops"],
        "flops_used": compute["total_flops"],
        "total_budget_tokens": inputs.total_budget_tokens,
        "total_tokens": inputs.train_tokens,
    }


def training_compute_estimate(
    *,
    training_flops_per_token: int,
    training_tokens: int,
    self_generated_draft_forwards: int,
    device_batch_size: int,
    max_seq_len: int,
    base_accounting_method: str = BASE_TRAINING_FLOPS_ACCOUNTING_METHOD,
    token_edit_backbone_gradient_mode: str = "shared",
    differentiable_draft: bool = False,
) -> dict[str, int | str]:
    """Estimate base training plus draft forward and optional backward FLOPs.

    The model's existing training estimate is ``6P`` plus the corresponding
    quadratic-attention term. A forward-only pass uses ``2P`` plus one third
    of that attention term, so its per-token estimate is exactly one third of
    the existing training estimate. Every counted draft forward consumes one
    full fixed-shape device microbatch.
    """

    values = {
        "training_flops_per_token": training_flops_per_token,
        "training_tokens": training_tokens,
        "self_generated_draft_forwards": self_generated_draft_forwards,
        "device_batch_size": device_batch_size,
        "max_seq_len": max_seq_len,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if training_flops_per_token % 3 != 0:
        raise ValueError("training_flops_per_token must be divisible by three for the forward-only estimate")
    if not isinstance(base_accounting_method, str) or not base_accounting_method:
        raise ValueError("base_accounting_method must be a non-empty string")
    if token_edit_backbone_gradient_mode not in {"shared", "output_head_only", "residual_adapter"}:
        raise ValueError("token_edit_backbone_gradient_mode is unsupported")
    if not isinstance(differentiable_draft, bool):
        raise ValueError("differentiable_draft must be boolean")
    if self_generated_draft_forwards > 0 and (device_batch_size <= 0 or max_seq_len <= 0):
        raise ValueError("positive draft-forward counts require positive device_batch_size and max_seq_len")

    base_training_flops = training_flops_per_token * training_tokens
    draft_forward_flops_per_token = training_flops_per_token // 3
    draft_forward_input_tokens = self_generated_draft_forwards * device_batch_size * max_seq_len
    draft_forward_flops = draft_forward_flops_per_token * draft_forward_input_tokens
    draft_backward_flops = 2 * draft_forward_flops if differentiable_draft else 0
    if self_generated_draft_forwards > 0:
        accounting_method = (
            "base_training_plus_forward_only_draft_estimate"
            if base_accounting_method == BASE_TRAINING_FLOPS_ACCOUNTING_METHOD
            else f"{base_accounting_method}_plus_forward_only_draft_estimate"
        )
        if token_edit_backbone_gradient_mode == "output_head_only":
            accounting_method += "_dense_backward_upper_bound_for_output_head_only_t2t"
        elif token_edit_backbone_gradient_mode == "residual_adapter":
            accounting_method += "_dense_backward_upper_bound_for_residual_adapter_t2t"
        if differentiable_draft:
            accounting_method += "_plus_differentiable_draft_backward_estimate"
    else:
        accounting_method = base_accounting_method
    return {
        "accounting_method": accounting_method,
        "base_training_flops": base_training_flops,
        "draft_forward_flops_per_token": draft_forward_flops_per_token,
        "draft_forward_input_tokens": draft_forward_input_tokens,
        "draft_forward_flops": draft_forward_flops,
        "draft_backward_flops": draft_backward_flops,
        "total_flops": base_training_flops + draft_forward_flops + draft_backward_flops,
    }
