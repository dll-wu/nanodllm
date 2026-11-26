"""Stateless, exact-resume phase policy for isolated block-recipe screens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class RecipeTrainSettings(Protocol):
    bd3lm_recipe_mechanism: str
    bd3lm_recipe_start_step: int


class RecipeOptimizerSettings(Protocol):
    grad_clip: float


class RecipeConfig(Protocol):
    train: RecipeTrainSettings
    optim: RecipeOptimizerSettings


MECHANISMS = (
    "none",
    "hold",
    "replay25",
    "replay50",
    "lr_half",
    "clip4",
    "anchor",
    "decode_mask",
    "staged_uniform",
    "staged_generated",
    "ar_teacher",
    "self_condition",
)


def active_mechanism(config: RecipeConfig, step: int) -> str:
    return config.train.bd3lm_recipe_mechanism if step >= config.train.bd3lm_recipe_start_step else "none"


def cadence(index: int, numerator: int, denominator: int) -> bool:
    """Whole-update cadence without RNG, including at a resumed boundary."""
    return (index + 1) * numerator // denominator > index * numerator // denominator


@dataclass(frozen=True)
class RecipePhase:
    mechanism: str
    block_size: int
    replay: bool
    lr_multiplier: float
    grad_clip: float


def recipe_phase(config: RecipeConfig, step: int, scheduled_block_size: int) -> RecipePhase:
    mechanism = active_mechanism(config, step)
    index = step - config.train.bd3lm_recipe_start_step
    fraction = {"hold": 4, "replay25": 1, "replay50": 2}.get(mechanism, 0)
    replay = fraction > 0 and cadence(index, fraction, 4)
    # The first TPP2 contains 1280 BL1 + 768 BL2 updates: 5/8 versus 3/8.
    replay_index = (index + 1) * fraction // 4 - 1
    block_size = (1 if cadence(replay_index, 5, 8) else 2) if replay else scheduled_block_size
    return RecipePhase(
        mechanism,
        block_size,
        replay,
        0.5 if mechanism == "lr_half" else 1.0,
        4.0 if mechanism == "clip4" else float(config.optim.grad_clip),
    )


def prepare_recipe_step(session, step: int) -> None:
    """Resolve actual clipping/LR/BL together, before optimizer work."""
    phase = recipe_phase(session.config, step, session.block_conversion_training_block_size)
    session.block_recipe_phase = phase
    session.block_conversion_training_block_size = phase.block_size
    session.lrm *= phase.lr_multiplier
    session.grad_clip_threshold = phase.grad_clip
    session.grad_clip_enabled = phase.grad_clip > 0
    session.raw_model.block_self_conditioning_enabled = phase.mechanism == "self_condition"


def recipe_step_metrics(session) -> dict[str, object]:
    phase = getattr(session, "block_recipe_phase", None)
    if phase is None or session.config.train.bd3lm_recipe_mechanism == "none":
        return {}
    return {
        "block_recipe_mechanism": session.config.train.bd3lm_recipe_mechanism,
        "block_recipe_active_mechanism": phase.mechanism,
        "block_recipe_replay_update": phase.replay,
        "block_recipe_lr_multiplier": phase.lr_multiplier,
        "block_recipe_cost_notice": "base_flops_excludes_extra_draft_teacher_and_self_conditioning_passes; use wall time and microbatch ledger",
    }
