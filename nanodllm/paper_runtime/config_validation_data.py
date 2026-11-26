"""Data, packing and synthetic-input validation in historical error order."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config_validation_bos_only import validate_bos_only
from .data_bos_packing_config import validate_bos_packing

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_data_config(config: ExperimentConfig) -> None:
    sampler_eos_reachability_canary = bool(
        getattr(config.train, "strict_mdlm_sampler_eos_reachability_canary_only", False)
    )
    if config.data.tokenizer not in ("byte", "hf"):
        raise ValueError("data.tokenizer must be 'byte' or 'hf'")
    if config.data.tokenizer == "hf" and not config.data.tokenizer_path:
        raise ValueError("data.tokenizer_path is required when data.tokenizer is 'hf'")
    from .mod97_task import DATASETS as MOD97_DATASETS

    if config.data.synthetic_dataset not in {"", "eos_overfit_canary", *MOD97_DATASETS}:
        raise ValueError("data.synthetic_dataset is not a supported synthetic corpus")
    if config.data.synthetic_eos_overfit_loss_mask_policy not in ("eos_only", "all_lossable_tokens"):
        raise ValueError("data.synthetic_eos_overfit_loss_mask_policy must be 'eos_only' or 'all_lossable_tokens'")
    if not isinstance(config.data.synthetic_eos_overfit_unique_non_eos_tokens, bool):
        raise ValueError("data.synthetic_eos_overfit_unique_non_eos_tokens must be a boolean")
    if not isinstance(config.data.synthetic_eos_overfit_unique_non_eos_token_count, int):
        raise ValueError("data.synthetic_eos_overfit_unique_non_eos_token_count must be an integer")
    if config.data.synthetic_eos_overfit_unique_non_eos_token_count < 0:
        raise ValueError("data.synthetic_eos_overfit_unique_non_eos_token_count must be >= 0")
    if config.data.synthetic_eos_overfit_unique_non_eos_token_count > 62:
        raise ValueError("data.synthetic_eos_overfit_unique_non_eos_token_count must be <= 62")
    if (
        config.data.synthetic_eos_overfit_unique_non_eos_tokens
        and config.data.synthetic_dataset != "eos_overfit_canary"
    ):
        raise ValueError("data.synthetic_eos_overfit_unique_non_eos_tokens is only configurable for eos_overfit_canary")
    if (
        config.data.synthetic_eos_overfit_unique_non_eos_token_count
        and config.data.synthetic_dataset != "eos_overfit_canary"
    ):
        raise ValueError(
            "data.synthetic_eos_overfit_unique_non_eos_token_count is only configurable for eos_overfit_canary"
        )
    if config.data.synthetic_eos_overfit_fixed_row_text and config.data.synthetic_dataset != "eos_overfit_canary":
        raise ValueError("data.synthetic_eos_overfit_fixed_row_text is only configurable for eos_overfit_canary")
    if config.data.synthetic_eos_overfit_fixed_row_text and (
        config.data.synthetic_eos_overfit_unique_non_eos_tokens
        or config.data.synthetic_eos_overfit_unique_non_eos_token_count
    ):
        raise ValueError(
            "data.synthetic_eos_overfit_fixed_row_text is mutually exclusive with "
            "synthetic_eos_overfit_unique_non_eos_tokens and "
            "synthetic_eos_overfit_unique_non_eos_token_count"
        )
    document_lengths = getattr(
        config.data,
        "synthetic_eos_overfit_document_lengths",
        [1, 2, 3, 4, 1, 2, 3, 4],
    )
    if not isinstance(document_lengths, list):
        raise ValueError("data.synthetic_eos_overfit_document_lengths must be a list")
    if not document_lengths:
        raise ValueError("data.synthetic_eos_overfit_document_lengths must be non-empty")
    for length in document_lengths:
        if not isinstance(length, int):
            raise ValueError("data.synthetic_eos_overfit_document_lengths entries must be integers")
        if length <= 0:
            raise ValueError("data.synthetic_eos_overfit_document_lengths entries must be positive")
        if length > config.model.max_seq_len:
            raise ValueError("data.synthetic_eos_overfit_document_lengths entries must not exceed model.max_seq_len")
    if config.data.synthetic_dataset != "eos_overfit_canary" and document_lengths != [1, 2, 3, 4, 1, 2, 3, 4]:
        raise ValueError("data.synthetic_eos_overfit_document_lengths is only configurable for eos_overfit_canary")
    if (
        config.data.synthetic_eos_overfit_unique_non_eos_tokens
        and config.data.synthetic_eos_overfit_unique_non_eos_token_count
    ):
        raise ValueError(
            "set either synthetic_eos_overfit_unique_non_eos_tokens "
            "or synthetic_eos_overfit_unique_non_eos_token_count, not both"
        )
    if (
        config.data.synthetic_dataset != "eos_overfit_canary"
        and config.data.synthetic_eos_overfit_loss_mask_policy != "eos_only"
    ):
        raise ValueError("data.synthetic_eos_overfit_loss_mask_policy is only configurable for eos_overfit_canary")
    if config.data.train_exclude_data and not config.data.train_data:
        raise ValueError("data.train_exclude_data requires data.train_data")
    if config.data.train_exclude_data and config.data.synthetic_dataset:
        raise ValueError("data.train_exclude_data cannot be combined with data.synthetic_dataset")
    if not config.data.train_data and not config.data.synthetic_dataset and not sampler_eos_reachability_canary:
        raise ValueError(
            "data.train_data must contain at least one parquet file, directory, glob, or synthetic dataset"
        )
    if config.data.val_fraction < 0 or config.data.val_fraction >= 1:
        raise ValueError("data.val_fraction must be in [0, 1)")
    if config.data.tokenizer_batch_size <= 0:
        raise ValueError("data.tokenizer_batch_size must be positive")
    validate_bos_only(config)
    validate_bos_packing(config)
    if config.data.random_length_ratio < 0 or config.data.random_length_ratio > 1:
        raise ValueError("data.random_length_ratio must be in [0, 1]")
    if config.data.continuation_ratio < 0 or config.data.continuation_ratio > 1:
        raise ValueError("data.continuation_ratio must be in [0, 1]")
    if config.data.random_length_ratio + config.data.continuation_ratio > 1:
        raise ValueError("random_length_ratio + continuation_ratio must be <= 1")
    if config.data.random_length_min <= 0:
        raise ValueError("data.random_length_min must be positive")
    if config.data.continuation_min_prefix_frac < 0:
        raise ValueError("data.continuation_min_prefix_frac must be non-negative")
    if config.data.continuation_max_prefix_frac > 1:
        raise ValueError("data.continuation_max_prefix_frac must be <= 1")
    if config.data.continuation_min_prefix_frac >= config.data.continuation_max_prefix_frac:
        raise ValueError("data.continuation_min_prefix_frac must be < data.continuation_max_prefix_frac")
    if not isinstance(config.data.boundary_aware_packing, bool):
        raise ValueError("data.boundary_aware_packing must be a boolean")
    if not isinstance(config.data.content_budget_segmented_packing, bool):
        raise ValueError("data.content_budget_segmented_packing must be a boolean")
    if config.data.content_budget_segmented_packing:
        if config.data.boundary_aware_packing:
            raise ValueError("data.content_budget_segmented_packing cannot be combined with boundary_aware_packing")
        if config.data.short_document_curriculum:
            raise ValueError("data.content_budget_segmented_packing cannot be combined with short_document_curriculum")
        if config.model.causal:
            if not config.data.prepend_bos:
                raise ValueError("causal content-budget segmented packing requires data.prepend_bos=true")
            if config.data.loss_on_bos:
                raise ValueError("causal content-budget segmented packing requires data.loss_on_bos=false")
        if config.noise.type not in {"mdlm", "bd3lm"}:
            raise ValueError("data.content_budget_segmented_packing requires noise.type='mdlm' or 'bd3lm'")
        if config.noise.time_conditioning:
            raise ValueError("data.content_budget_segmented_packing requires time_conditioning=false")
        if config.noise.suffix_masking:
            raise ValueError("data.content_budget_segmented_packing requires suffix_masking=false")
    if not isinstance(config.data.short_document_curriculum, bool):
        raise ValueError("data.short_document_curriculum must be a boolean")
    if config.data.identity_mode not in ("off", "metadata", "full_sha256"):
        raise ValueError("data.identity_mode must be one of 'off', 'metadata', or 'full_sha256'")
    if config.data.identity_mode != "off" and config.data.synthetic_dataset:
        raise ValueError("data.identity_mode requires parquet data rather than data.synthetic_dataset")
    from .data_snapshot import validate_snapshot_config

    validate_snapshot_config(config.data)
    if config.data.synthetic_dataset in MOD97_DATASETS:
        if config.data.tokenizer != "hf":
            raise ValueError("mod97 task data requires the project HF tokenizer")
        if config.model.max_seq_len != 512:
            raise ValueError("mod97 task data requires a fixed 512-position row")
        if config.data.train_data or config.data.val_data or config.data.train_exclude_data:
            raise ValueError("mod97 task data is generated in memory and cannot be combined with parquet data")
        if not config.data.prepend_bos or not config.data.append_eos or not config.data.loss_on_eos:
            raise ValueError("mod97 task data requires BOS context and supervised EOS")
        if config.data.loss_on_bos:
            raise ValueError("mod97 task data does not supervise BOS")
        if config.data.random_length_ratio != 0 or config.data.continuation_ratio != 0:
            raise ValueError("mod97 task data forbids generic row-mask augmentation")
        if config.data.boundary_aware_packing or config.data.content_budget_segmented_packing:
            raise ValueError("mod97 task data uses one problem per fixed row")
        if config.data.short_document_curriculum:
            raise ValueError("mod97 task data cannot use the corpus short-document curriculum")
    if config.data.short_document_max_tokens <= 0:
        raise ValueError("data.short_document_max_tokens must be positive")
    if config.data.short_document_curriculum:
        if config.data.boundary_aware_packing:
            raise ValueError("data.short_document_curriculum cannot be combined with boundary_aware_packing")
        if config.data.short_document_max_tokens < 2:
            raise ValueError("data.short_document_max_tokens must be at least 2 when short curriculum is enabled")
        if config.data.short_document_max_tokens > config.model.max_seq_len:
            raise ValueError("data.short_document_max_tokens must not exceed model.max_seq_len")
