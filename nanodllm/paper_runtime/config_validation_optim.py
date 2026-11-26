"""Optimizer option validation; specialized backends retain their own scope."""

from dataclasses import replace

from .model_profiles import validate_nanollada_experiment
from .nanochat_muon import validate_config as validate_nanochat_muon


def validate_model_optimizer_profile(config):
    # Reuse the frozen model profile without changing evaluator source bytes.
    # The new optimizer owns its narrower scope and exact grouping separately.
    if config.optim.type == "nanochat_muon":
        validate_nanochat_muon(config)
        config = replace(config, optim=replace(config.optim, type="adamw"))
    validate_nanollada_experiment(config)


def validate_optimizer(config):
    validate_nanochat_muon(config)
    if config.optim.type not in ("adamw", "muon", "nanochat_muon"):
        raise ValueError("optim.type must be one of 'adamw', 'muon' or 'nanochat_muon'")
    if config.optim.lr <= 0:
        raise ValueError("optim.lr must be positive")
    if config.optim.muon_lr != -1.0 and config.optim.muon_lr <= 0:
        raise ValueError("optim.muon_lr must be positive or -1 to inherit optim.lr")
    if config.optim.adamw_lr != -1.0 and config.optim.adamw_lr <= 0:
        raise ValueError("optim.adamw_lr must be positive or -1 to inherit optim.lr")
    if config.optim.adam_eps <= 0:
        raise ValueError("optim.adam_eps must be positive")


def strip_disabled_nanochat_muon(value: dict) -> None:
    """Preserve frozen historical projections only for exact inactive defaults."""
    from dataclasses import asdict

    from .config import OptimConfig

    for name, default in asdict(OptimConfig()).items():
        if name.startswith("nanochat_") and value["optim"].pop(name, default) != default:
            raise ValueError("historical configuration requires disabled nanochat_muon defaults")
