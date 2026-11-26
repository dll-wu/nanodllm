"""Fail-closed validation for the single-GPU one-mechanism block screen."""

import math
import re
from pathlib import Path

from .training_block_recipe import MECHANISMS


def project_disabled_block_recipe_defaults(value: dict) -> None:
    """Remove only proven additive defaults from frozen historical projections."""
    defaults = {
        ("train", "bd3lm_full_mask_batch_fraction"): 0.0,
        ("train", "bd3lm_partition_mode"): "fixed",
        ("train", "bd3lm_partition_lengths"): [],
        ("train", "bd3lm_partition_seed"): 0,
        ("model", "bd3lm_self_conditioning_dim"): 0,
        ("train", "model_init_allow_block_self_conditioning"): False,
        ("train", "bd3lm_recipe_mechanism"): "none",
        ("train", "bd3lm_recipe_start_step"): 0,
        ("train", "bd3lm_recipe_teacher_path"): "",
        ("train", "bd3lm_recipe_teacher_sha256"): "",
        ("train", "bd3lm_recipe_teacher_size_bytes"): 0,
        ("train", "bd3lm_recipe_teacher_weight"): 0.5,
    }
    for (section, field), expected in defaults.items():
        if value[section].pop(field) != expected:
            raise ValueError(f"the frozen historical projection requires {section}.{field}={expected!r}")


def validate_block_recipe(config) -> None:
    train, model = config.train, config.model
    name = train.bd3lm_recipe_mechanism
    if type(train.model_init_allow_block_self_conditioning) is not bool:
        raise ValueError("model_init_allow_block_self_conditioning must be boolean")
    if name not in MECHANISMS:
        raise ValueError(f"unknown bd3lm_recipe_mechanism: {name}")
    dim = model.bd3lm_self_conditioning_dim
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
        raise ValueError("bd3lm_self_conditioning_dim must be a nonnegative integer")
    start = train.bd3lm_recipe_start_step
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("bd3lm_recipe_start_step must be a nonnegative integer")
    teacher = (
        train.bd3lm_recipe_teacher_path,
        train.bd3lm_recipe_teacher_sha256,
        train.bd3lm_recipe_teacher_size_bytes,
    )
    weight = train.bd3lm_recipe_teacher_weight
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("bd3lm_recipe_teacher_weight must be finite and positive")
    if name == "none":
        if start or dim or any(teacher) or train.model_init_allow_block_self_conditioning:
            raise ValueError("disabled block recipe must not carry active extension settings")
        return
    if config.noise.type != "bd3lm" or model.bd3lm_clean_stream_mode != "causal":
        raise ValueError("block recipe requires causal-clean BD3 conversion")
    if config.parameterization.mode != "ordinary" or model.backbone_profile != "nanodllm":
        raise ValueError("block recipe screen supports the ordinary NanoDLLM backbone only")
    if config.runtime.compile:
        raise ValueError("block recipe mechanisms currently require eager training")
    if train.num_steps <= start:
        raise ValueError("block recipe start must precede the training horizon")
    if train.bd3lm_causal_ar_loss_weight != 0:
        raise ValueError("one-factor block recipe screen requires AR auxiliary weight zero")
    if train.bd3lm_uniform_t2t_batch_fraction != (0.25 if name.startswith("staged_") else 0.0):
        raise ValueError("staged block recipes require T2T fraction 0.25; other mechanisms require zero")
    if name in {"hold", "replay25", "replay50"} and model.max_seq_len % 2:
        raise ValueError("small-block replay requires a context divisible by two")
    if bool(dim) != (name == "self_condition"):
        raise ValueError("self_condition requires a positive adapter dimension, isolated from other mechanisms")
    if train.model_init_allow_block_self_conditioning != bool(dim and train.model_init_from):
        raise ValueError("self-conditioning base initialization requires its explicit extension flag")
    if name in {"anchor", "ar_teacher"}:
        if not isinstance(teacher[0], str) or not Path(teacher[0]).is_absolute():
            raise ValueError("block recipe requires an absolute frozen teacher checkpoint")
        if not isinstance(teacher[1], str) or re.fullmatch(r"[0-9a-f]{64}", teacher[1]) is None:
            raise ValueError("block recipe teacher requires an exact SHA-256")
        if isinstance(teacher[2], bool) or not isinstance(teacher[2], int) or teacher[2] <= 0:
            raise ValueError("block recipe teacher requires a positive byte size")
    elif any(teacher):
        raise ValueError("teacher settings require an anchor or AR-teacher mechanism")
