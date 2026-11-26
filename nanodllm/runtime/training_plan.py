import math
from dataclasses import replace
from typing import Protocol

import torch

from .config import ExperimentConfig
from .model import DiffusionConfig, DiffusionTransformer, estimate_non_embedding_params
from .parameterization import parameterization_model_config_kwargs


class _TokenizerMetadata(Protocol):
    def get_vocab_size(self) -> int: ...

    def get_mask_token_id(self) -> int: ...


def _resolve_diffusion_config(config: ExperimentConfig, tokenizer: _TokenizerMetadata) -> DiffusionConfig:
    """Resolve the model's runtime configuration from the experiment contract."""

    return DiffusionConfig(
        sequence_len=config.model.max_seq_len,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=int(config.model.depth),
        n_head=int(config.model.n_head),
        n_embd=int(config.model.n_embd),
        mask_token_id=tokenizer.get_mask_token_id(),
        backbone_profile=config.model.backbone_profile,
        mtp_mode=config.model.mtp_mode,
        recurrent_core_start=config.model.recurrent_core_start,
        recurrent_core_length=config.model.recurrent_core_length,
        recurrent_iterations=config.model.recurrent_iterations,
        mlp_hidden_dim=int(config.model.mlp_hidden_dim),
        attention_scale=float(config.model.attention_scale),
        local_attention_lower_layers=int(config.model.local_attention_lower_layers),
        local_attention_window=int(config.model.local_attention_window),
        auxiliary_head_layers=tuple(int(i) for i in config.model.auxiliary_head_layers),
        time_conditioning=bool(config.noise.time_conditioning),
        bd3lm_block_size=int(config.model.bd3lm_block_size) if config.noise.type == "bd3lm" else 0,
        bd3lm_self_conditioning_dim=config.model.bd3lm_self_conditioning_dim,
        output_factorization=config.model.output_factorization,
        causal=bool(config.model.causal),
        residual_output_projection_init=config.model.residual_output_projection_init,
        embedding_head_init_std_multiplier=float(config.model.embedding_head_init_std_multiplier),
        length_head=bool(config.model.length_head),
        target_length_conditioning=bool(config.model.target_length_conditioning),
        mask_position_embedding=bool(config.model.mask_position_embedding),
        token_edit_adapter_bottleneck_dim=int(config.model.token_edit_adapter_bottleneck_dim),
        token_edit_adapter_top_layers=int(config.model.token_edit_adapter_top_layers),
        token_edit_provenance_embedding=bool(config.model.token_edit_provenance_embedding),
        token_edit_provenance_confidence_bins=int(config.model.token_edit_provenance_confidence_bins),
        **parameterization_model_config_kwargs(config),
    )


def get_lr_multiplier(
    step: int, total_steps: int, warmup_steps: int, warmdown_ratio: float, final_lr_frac: float
) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    warmdown_steps = round(warmdown_ratio * total_steps)
    warmdown_start = total_steps - warmdown_steps
    if step < warmdown_start:
        return 1.0
    progress = (total_steps - step) / max(1, warmdown_steps)
    return max(final_lr_frac, progress)


def _resolved_warmup_steps(config: ExperimentConfig, total_steps: int) -> int:
    if config.optim.warmup_ratio >= 0:
        return round(config.optim.warmup_ratio * total_steps)
    return config.optim.warmup_steps


def _lr_schedule_telemetry(config: ExperimentConfig, total_steps: int, warmup_steps: int) -> dict:
    warmdown_steps = round(config.optim.warmdown_ratio * total_steps)
    warmdown_start_step = max(0, total_steps - warmdown_steps)
    probe_steps = sorted(
        {
            0,
            max(0, warmup_steps - 1),
            min(max(0, total_steps - 1), warmup_steps),
            max(0, warmdown_start_step - 1),
            min(max(0, total_steps - 1), warmdown_start_step),
            max(0, total_steps - 1),
        }
    )
    trace = [
        {
            "step": step + 1,
            "lr_multiplier": get_lr_multiplier(
                step,
                total_steps,
                warmup_steps,
                config.optim.warmdown_ratio,
                config.optim.final_lr_frac,
            ),
        }
        for step in probe_steps
        if 0 <= step < total_steps
    ]
    return {
        "warmup_steps_config": int(config.optim.warmup_steps),
        "warmup_ratio_config": float(config.optim.warmup_ratio),
        "warmup_steps": int(warmup_steps),
        "warmup_fraction": warmup_steps / max(1, total_steps),
        "warmdown_ratio": float(config.optim.warmdown_ratio),
        "warmdown_steps": int(warmdown_steps),
        "warmdown_start_step": int(warmdown_start_step + 1),
        "final_lr_frac": float(config.optim.final_lr_frac),
        "trace_artifact": "train_log.jsonl",
        "trace": trace,
    }


def _lr_schedule_metric_fields(lr_schedule: dict) -> dict[str, bool | float | int | str]:
    return {
        f"lr_schedule_{key}": value for key, value in lr_schedule.items() if isinstance(value, (bool, float, int, str))
    }


def _fixed_ce_trace_steps(total_steps: int, warmup_steps: int) -> list[int]:
    candidates = {
        1,
        max(1, min(total_steps, warmup_steps)),
        max(1, min(total_steps, round(total_steps * 0.5))),
        max(1, total_steps),
    }
    return sorted(candidates)


def _resolved_optimizer_lr(config: ExperimentConfig, optimizer_kind: str, batch_lr_scale: float) -> float:
    if optimizer_kind == "muon":
        lr = config.optim.lr if config.optim.muon_lr < 0 else config.optim.muon_lr
    elif optimizer_kind == "adamw":
        lr = config.optim.lr if config.optim.adamw_lr < 0 else config.optim.adamw_lr
    else:
        raise ValueError(f"unknown optimizer kind {optimizer_kind!r}")
    return lr * batch_lr_scale


def _training_budget(
    config: ExperimentConfig,
    counts: dict,
    world_size: int,
    *,
    raw_model: DiffusionTransformer | None = None,
):
    train = config.train
    assert train.device_batch_size is not None
    target_param_count = _selected_target_param_count(config, counts)
    param_data_tokens = int(train.target_param_data_ratio * target_param_count)
    if train.total_batch_size < 0:
        reference_count = _reference_target_param_count(config, raw_model)
        ref_tokens = train.target_param_data_ratio * reference_count
        predicted_batch = train.base_total_batch_size * (param_data_tokens / ref_tokens) ** 0.383
        total_batch_size = 2 ** round(math.log2(predicted_batch))
    else:
        total_batch_size = train.total_batch_size

    tokens_per_micro = train.device_batch_size * config.model.max_seq_len
    world_tokens_per_micro = tokens_per_micro * world_size
    if total_batch_size % world_tokens_per_micro != 0:
        if not train.allow_batch_size_round_up:
            raise ValueError(
                f"total_batch_size={total_batch_size} must be divisible by "
                f"world_size * device_batch_size * max_seq_len = {world_tokens_per_micro}"
            )
        grad_accum_steps = max(1, total_batch_size // world_tokens_per_micro)
        total_batch_size = world_tokens_per_micro * grad_accum_steps
    else:
        grad_accum_steps = total_batch_size // world_tokens_per_micro
    if train.num_steps > 0:
        total_steps = train.num_steps
    elif train.target_tokens > 0:
        total_steps = max(1, round(train.target_tokens / total_batch_size))
    else:
        total_steps = max(1, param_data_tokens // total_batch_size)
    batch_lr_scale = (total_batch_size / train.base_total_batch_size) ** 0.5 if train.scale_lr_by_batch else 1.0
    return total_batch_size, grad_accum_steps, total_steps, batch_lr_scale


def _reference_target_param_count(
    config: ExperimentConfig,
    raw_model: DiffusionTransformer | None,
) -> int:
    """Read the batch-scaling reference count from a meta-device model."""
    if config.train.target_param_count_basis == "transformer_body_parameters":
        return estimate_non_embedding_params(n_layer=12, n_embd=768)
    if raw_model is None:
        raise ValueError(
            "automatic total_batch_size with a non-legacy target_param_count_basis requires "
            "the instantiated training model"
        )

    reference_config = replace(
        raw_model.config,
        n_layer=12,
        n_head=12,
        n_embd=768,
        mlp_hidden_dim=-1,
        attention_scale=-1.0,
    )
    token_class_ids = None
    if reference_config.output_factorization == "hierarchical_frequency_decile":
        output_class_count = int(raw_model.output_class_count)
        token_class_ids = torch.arange(reference_config.vocab_size, dtype=torch.long) % output_class_count
    with torch.device("meta"):
        reference_model = DiffusionTransformer(
            reference_config,
            rope_base=float(config.model.rope_base),
            compute_dtype=torch.float32,
            pad_vocab_to=int(config.model.pad_vocab_to),
            token_class_ids=token_class_ids,
        )
    return _selected_target_param_count(config, reference_model.parameter_counts())


def _selected_target_param_count(config: ExperimentConfig, counts: dict[str, int]) -> int:
    basis = config.train.target_param_count_basis
    if basis in counts:
        return int(counts[basis])
    compatibility_aliases = {
        "transformer_body_parameters": "non_embedding",
        "nanochat_scaling_parameters": "scaling",
        "total_parameters": "total",
    }
    alias = compatibility_aliases.get(basis)
    if alias is not None and alias in counts:
        return int(counts[alias])
    raise KeyError(f"parameter count view {basis!r} is missing from the instantiated-model inventory")


def _parameter_accounting_summary(
    config: ExperimentConfig,
    counts: dict[str, int],
    inventory: dict[str, object],
) -> dict[str, object]:
    return {
        "method": inventory["method"],
        "target_param_data_ratio": float(config.train.target_param_data_ratio),
        "target_param_count_basis": config.train.target_param_count_basis,
        "target_param_count": _selected_target_param_count(config, counts),
        "components": inventory["components"],
        "views": inventory["views"],
    }
