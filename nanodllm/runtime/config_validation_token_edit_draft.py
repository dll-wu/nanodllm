"""Token-edit corruption, draft checkpoint identity, and draft loss weighting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_draft_inputs(config: ExperimentConfig) -> None:
    for field_name in (
        "token_edit_corruption_batch_fraction",
        "token_edit_corruption_ratio",
        "token_edit_correct_draft_loss_weight",
        "token_edit_wrong_draft_loss_weight",
        "token_edit_mixed_canvas_draft_fraction",
        "token_edit_trajectory_completion_fraction",
        "token_edit_correction_aux_weight",
        "token_edit_expected_accuracy_power",
        "token_edit_expected_accuracy_corruption_threshold",
        "token_edit_cross_view_nonregression_weight",
        "token_edit_draft_m2t_loss_weight",
        "token_edit_draft_m2t_credit_strength",
        "token_edit_draft_m2t_credit_clip_min",
        "token_edit_draft_m2t_credit_clip_max",
        "token_edit_prefix_min_fraction",
        "token_edit_prefix_max_fraction",
    ):
        field_value = getattr(config.train, field_name)
        if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
            raise ValueError(f"train.{field_name} must be numeric")
        if not math.isfinite(float(field_value)):
            raise ValueError(f"train.{field_name} must be finite")
    if config.train.token_edit_corruption_batch_fraction < 0 or config.train.token_edit_corruption_batch_fraction > 1:
        raise ValueError("train.token_edit_corruption_batch_fraction must be in [0, 1]")
    if config.train.token_edit_batch_layout not in (
        "whole_microbatch_cadence",
        "fixed_12_m2t_4_t2t_rows",
    ):
        raise ValueError(
            "train.token_edit_batch_layout must be 'whole_microbatch_cadence' or 'fixed_12_m2t_4_t2t_rows'"
        )
    if config.train.token_edit_corruption_schedule not in ("legacy_fixed_ratio", "match_mdlm_per_row"):
        raise ValueError("train.token_edit_corruption_schedule must be 'legacy_fixed_ratio' or 'match_mdlm_per_row'")
    if config.train.token_edit_corruption_ratio < 0 or config.train.token_edit_corruption_ratio > 1:
        raise ValueError("train.token_edit_corruption_ratio must be in [0, 1]")
    if config.train.token_edit_input_source not in ("random_replacement", "self_generated_argmax"):
        raise ValueError("train.token_edit_input_source must be 'random_replacement' or 'self_generated_argmax'")
    if config.train.token_edit_draft_model_source not in ("live_student", "frozen_checkpoint"):
        raise ValueError("train.token_edit_draft_model_source must be 'live_student' or 'frozen_checkpoint'")
    draft_checkpoint = config.train.token_edit_draft_checkpoint
    draft_checkpoint_sha256 = config.train.token_edit_draft_checkpoint_expected_sha256
    draft_checkpoint_size = config.train.token_edit_draft_checkpoint_expected_size_bytes
    if not isinstance(draft_checkpoint, str) or not isinstance(draft_checkpoint_sha256, str):
        raise ValueError("train token-edit draft checkpoint path and digest must be strings")
    if isinstance(draft_checkpoint_size, bool) or not isinstance(draft_checkpoint_size, int):
        raise ValueError("train.token_edit_draft_checkpoint_expected_size_bytes must be an integer")
    if config.train.token_edit_draft_model_source == "live_student":
        if draft_checkpoint or draft_checkpoint_sha256 or draft_checkpoint_size != -1:
            raise ValueError("live-student token-edit drafts require empty checkpoint binding fields")
    else:
        if not Path(draft_checkpoint).expanduser().is_absolute():
            raise ValueError("frozen token-edit draft checkpoint must be an absolute path")
        if len(draft_checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in draft_checkpoint_sha256
        ):
            raise ValueError("frozen token-edit draft checkpoint SHA-256 must be lowercase 64-hex")
        if draft_checkpoint_size <= 0:
            raise ValueError("frozen token-edit draft checkpoint size must be positive")


def validate_draft_credit(config: ExperimentConfig) -> None:
    draft_m2t_weight = float(config.train.token_edit_draft_m2t_loss_weight)
    if not 0.0 <= draft_m2t_weight <= 1.0:
        raise ValueError("train.token_edit_draft_m2t_loss_weight must be in [0, 1]")
    credit_modes = {
        "uniform",
        "aligned_expected_error_reward",
        "shuffled_expected_error_reward",
    }
    credit_mode = config.train.token_edit_draft_m2t_credit_mode
    if credit_mode not in credit_modes:
        raise ValueError(
            "train.token_edit_draft_m2t_credit_mode must be 'uniform', "
            "'aligned_expected_error_reward', or 'shuffled_expected_error_reward'"
        )
    credit_strength = float(config.train.token_edit_draft_m2t_credit_strength)
    credit_clip_min = float(config.train.token_edit_draft_m2t_credit_clip_min)
    credit_clip_max = float(config.train.token_edit_draft_m2t_credit_clip_max)
    if credit_strength < 0.0:
        raise ValueError("train.token_edit_draft_m2t_credit_strength must be non-negative")
    if not 0.0 < credit_clip_min <= 1.0 <= credit_clip_max:
        raise ValueError("train draft-M2T credit clipping must satisfy 0 < clip_min <= 1 <= clip_max")
    if credit_mode == "uniform":
        if (credit_strength, credit_clip_min, credit_clip_max) != (1.0, 0.5, 1.5):
            raise ValueError("uniform draft-M2T credit controls must retain their defaults")
    else:
        if credit_strength <= 0.0:
            raise ValueError("non-uniform draft-M2T credit requires token_edit_draft_m2t_credit_strength>0")
        if draft_m2t_weight <= 0.0:
            raise ValueError("non-uniform draft-M2T credit requires token_edit_draft_m2t_loss_weight>0")
    if draft_m2t_weight > 0.0:
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("draft-M2T joint loss requires self-generated argmax drafts")
        if config.train.token_edit_batch_layout != "whole_microbatch_cadence":
            raise ValueError("draft-M2T joint loss requires whole_microbatch_cadence")
        if config.train.token_edit_canvas_mode != "full_draft":
            raise ValueError("draft-M2T joint loss requires token_edit_canvas_mode='full_draft'")
        if config.train.token_edit_prefix_enabled:
            raise ValueError("draft-M2T joint loss does not support token-edit prefix training")
        if config.train.token_edit_backbone_gradient_mode != "shared":
            raise ValueError("draft-M2T joint loss requires shared backbone gradients")
        if config.runtime.init_single_rank_process_group:
            raise ValueError("draft-M2T joint loss currently requires non-DDP single-process training")
        if config.train.token_edit_correction_objective != "expected_accuracy":
            raise ValueError("draft-M2T joint loss requires the expected_accuracy correction objective")
        if config.train.token_edit_correction_aux_weight != 0:
            raise ValueError("draft-M2T joint loss requires token_edit_correction_aux_weight=0")
