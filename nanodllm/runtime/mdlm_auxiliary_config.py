"""Explicit opt-in configuration for segment-local MDLM training views."""

import math
import re
from dataclasses import asdict, dataclass


@dataclass
class MDLMAuxiliaryConfig:
    mode: str = "none"
    start_step: int = 0
    fraction: float = 0.25
    seed: int = 42
    window_min_fraction: float = 0.5
    window_max_fraction: float = 1.0
    prefix_min_fraction: float = 0.125
    prefix_max_fraction: float = 0.75
    draft_fraction: float = 0.5
    proposal_teacher_path: str = ""
    proposal_teacher_sha256: str = ""
    proposal_tokenizer_sha256: str = ""
    proposal_first_error_weight: float = 1.0
    proposal_rounds: int = 2
    proposal_canvas_tokens: int = 128
    proposal_max_prefix: int = 128
    proposal_teacher_targets: str = "hard"


MODES = ("none", "boundary", "full_suffix", "uniform", "generated", "prefix_frontier", "proposal")


def strip_disabled_auxiliary(value: dict) -> None:
    """Preserve historical frozen projections only for the exact disabled default."""
    if value["train"].pop("mdlm_auxiliary") != asdict(MDLMAuxiliaryConfig()):
        raise ValueError("historical configuration forbids MDLM auxiliary views")


def validate_auxiliary(config) -> None:
    aux = config.train.mdlm_auxiliary
    if not isinstance(aux, MDLMAuxiliaryConfig) or aux.mode not in MODES:
        raise ValueError("unsupported MDLM auxiliary configuration")
    if type(aux.start_step) is not int or aux.start_step < 0 or type(aux.seed) is not int or aux.seed < 0:
        raise ValueError("MDLM auxiliary start step and seed must be nonnegative integers")
    values = (
        aux.fraction,
        aux.window_min_fraction,
        aux.window_max_fraction,
        aux.prefix_min_fraction,
        aux.prefix_max_fraction,
        aux.draft_fraction,
    )
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values):
        raise ValueError("MDLM auxiliary fractions must be finite numbers")
    if not (
        0 < aux.fraction <= 1
        and 0 < aux.window_min_fraction <= aux.window_max_fraction <= 1
        and (
            0 <= aux.prefix_min_fraction <= aux.prefix_max_fraction <= 1
            if aux.mode == "prefix_frontier"
            else 0 < aux.prefix_min_fraction <= aux.prefix_max_fraction < 1
        )
        and 0 < aux.draft_fraction <= 1
    ):
        raise ValueError("MDLM auxiliary fractions are outside their valid ranges")
    if aux.mode == "none":
        return
    if aux.mode == "proposal":
        if not (
            isinstance(aux.proposal_teacher_path, str)
            and aux.proposal_teacher_path.startswith("/")
            and all(
                isinstance(s, str) and re.fullmatch(r"[0-9a-f]{64}", s)
                for s in (aux.proposal_teacher_sha256, aux.proposal_tokenizer_sha256)
            )
            and type(aux.proposal_rounds) is int
            and 1 <= aux.proposal_rounds <= 8
            and type(aux.proposal_canvas_tokens) is int
            and aux.proposal_canvas_tokens >= 8 * aux.proposal_rounds
            and type(aux.proposal_max_prefix) is int
            and aux.proposal_max_prefix >= 1
            and aux.proposal_max_prefix + aux.proposal_canvas_tokens <= config.model.max_seq_len
            and type(aux.proposal_first_error_weight) in (int, float)
            and math.isfinite(aux.proposal_first_error_weight)
            and aux.proposal_first_error_weight >= 1
            and aux.proposal_teacher_targets in ("hard", "soft")
            and (aux.proposal_teacher_targets == "hard" or aux.proposal_first_error_weight == 1)
        ):
            raise ValueError("invalid proposal teacher, window or first-error weight")
        if config.runtime.compile or config.data.eos_mask_tail_mode != "none" or config.model.bd3lm_block_size:
            raise ValueError("proposal screen requires eager plain MDLM without mask-tail augmentation")
    if aux.mode == "prefix_frontier" and (
        aux.window_min_fraction != 1
        or aux.window_max_fraction != 1
        or config.noise.suffix_masking
        or config.train.position_decile_balanced_loss
        or config.model.length_head
        or config.model.target_length_conditioning
        or config.data.eos_mask_tail_mode != "none"
    ):
        raise ValueError("prefix frontier requires full segments and no competing loss/window/tail mechanism")
    if (
        config.noise.type != "mdlm"
        or config.model.causal
        or config.noise.time_conditioning
        or config.model.output_factorization != "flat"
        or config.model.block_causal_pilot_size
        or config.model.auxiliary_head_layers
        or config.train.token_edit_corruption_batch_fraction
        or config.noise.complement_masking
        or config.noise.per_token_noise_levels
        or config.noise.soft_input_curriculum
        or not config.noise.schedule_weight
        or not config.data.content_budget_segmented_packing
        or config.data.boundary_aware_packing
        or config.data.short_document_curriculum
        or config.data.loss_on_bos
        or not config.data.loss_on_eos
        or not config.data.prepend_bos
        or not config.data.append_eos
        or any(
            (
                config.train.auxiliary_loss_weight,
                config.train.entropy_bonus_beta,
                config.train.label_smoothing_epsilon,
                config.train.z_loss_weight,
                config.train.confidence_penalty_weight,
                config.train.eos_hazard_loss_weight,
                config.train.length_head_loss_weight,
            )
        )
    ):
        raise ValueError("MDLM auxiliary views require plain weighted MDLM and BOS/EOS segmented packing")


def selected(aux: MDLMAuxiliaryConfig, step: int, micro: int, accumulation: int) -> bool:
    if aux.mode == "none" or step < aux.start_step:
        return False
    index = (step - aux.start_step) * accumulation + micro
    return math.floor((index + 1) * aux.fraction) > math.floor(index * aux.fraction)
