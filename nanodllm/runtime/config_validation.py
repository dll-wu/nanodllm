"""Pure validation rules for experiment configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config_validation_data import validate_data_config
from .config_validation_data_audits import (
    validate_collator_audit,
    validate_corruption_audit,
    validate_packing_audit,
    validate_right_context_audit,
    validate_target_ce_audit,
)
from .config_validation_diagnostics import _validate_diagnostics_config
from .config_validation_eos_canaries import (
    validate_eos_overfit,
    validate_eos_reachability,
    validate_learned_eos_sampler,
)
from .config_validation_evaluation import validate_runtime_evaluation_config
from .config_validation_mdlm_audits import validate_all_token_bayes_audit, validate_mdlm_audit_options
from .config_validation_model import validate_model_config
from .config_validation_noise import validate_noise_config
from .config_validation_optim import validate_model_optimizer_profile, validate_optimizer
from .config_validation_parameterization import validate_parameterization_config
from .config_validation_recipe import validate_recipe
from .config_validation_token_edit import validate_token_edit
from .config_validation_training import (
    validate_loss_estimators,
    validate_replay_ids,
    validate_training_budget_and_initialization,
    validate_training_losses,
)
from .data_mask_tail_config import validate_mask_tail
from .lr_schedule import validate_lr_schedule
from .mdlm_auxiliary_config import validate_auxiliary
from .training_mtp import validate_mtp_config
from .training_variable_depth import validate_config as validate_depth

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_config(config: ExperimentConfig) -> None:
    validate_lr_schedule(config)
    validate_auxiliary(config)
    validate_mask_tail(config)
    validate_mtp_config(config)
    _validate_diagnostics_config(config)
    validate_model_optimizer_profile(config)
    validate_data_config(config)
    validate_model_config(config)
    validate_depth(config)
    validate_noise_config(config)
    validate_optimizer(config)
    validate_parameterization_config(config)
    validate_recipe(config)
    validate_training_budget_and_initialization(config)
    validate_mdlm_audit_options(config)
    validate_learned_eos_sampler(config)
    validate_eos_reachability(config)
    validate_eos_overfit(config)
    validate_all_token_bayes_audit(config)
    validate_collator_audit(config)
    validate_packing_audit(config)
    validate_corruption_audit(config)
    validate_target_ce_audit(config)
    validate_right_context_audit(config)
    validate_replay_ids(config)
    validate_training_losses(config)
    validate_runtime_evaluation_config(config)
    validate_loss_estimators(config)
    validate_token_edit(config)
