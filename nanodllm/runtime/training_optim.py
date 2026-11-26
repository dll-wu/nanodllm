import torch
import torch.nn as nn

from . import ar_width_mup
from .config import ExperimentConfig
from .lr_schedule import get_lr_multiplier
from .nanochat_muon_schedule import metrics as _nanochat_optimizer_metrics
from .parameterization import (
    bind_parameterization_contract,
    optimizer_group_parameters,
    resolve_parameterization_contract,
)
from .training_mtp import validate_mtp_config
from .training_plan import _resolved_optimizer_lr


def _optimizer_base_lrs(config: ExperimentConfig, batch_lr_scale: float) -> tuple[float, ...]:
    if config.optim.type == "muon":
        return (
            _resolved_optimizer_lr(config, "muon", batch_lr_scale),
            _resolved_optimizer_lr(config, "adamw", batch_lr_scale),
        )
    return (_resolved_optimizer_lr(config, "adamw", batch_lr_scale),)


def _set_optimizer_base_lrs(optimizers, config: ExperimentConfig, batch_lr_scale: float) -> None:
    base_lrs = _optimizer_base_lrs(config, batch_lr_scale)
    optimizers = tuple(optimizers)
    if len(base_lrs) != len(optimizers):
        raise ValueError(f"expected {len(base_lrs)} optimizers for {config.optim.type}, got {len(optimizers)}")
    for optimizer, base_lr in zip(optimizers, base_lrs, strict=True):
        for group in optimizer.param_groups:
            resolved_base_lr = base_lr * float(group.get("parameterization_lr_multiplier", 1.0))
            group["base_lr"] = resolved_base_lr
            group["lr"] = resolved_base_lr


def _set_optimizer_lrs(optimizers, lr_multiplier: float) -> None:
    for group in _iter_optimizer_param_groups(optimizers):
        group["lr"] = float(group["base_lr"]) * lr_multiplier


def _step_lr_multiplier(config: ExperimentConfig, step: int, total_steps: int, warmup_steps: int) -> float:
    optim = config.optim
    return get_lr_multiplier(
        step, total_steps, warmup_steps, optim.warmdown_ratio, optim.final_lr_frac, optim.lr_decay_style
    )


def make_optimizer(model: nn.Module, config: ExperimentConfig, batch_lr_scale: float):
    validate_mtp_config(config)
    if ar_width_mup.active(config):
        ar_width_mup.validate_model(model)
        if batch_lr_scale != 1:
            raise ValueError("AR width muP forbids implicit batch learning-rate scaling")
    optim = config.optim
    base_lrs = _optimizer_base_lrs(config, batch_lr_scale)
    parameterization_contract = resolve_parameterization_contract(
        model,
        config,
        batch_lr_scale=batch_lr_scale,
    )
    bind_parameterization_contract(config, parameterization_contract)
    parameter_groups = optimizer_group_parameters(model, parameterization_contract)
    if optim.type == "nanochat_muon":
        from .nanochat_muon import make_optimizer as make_nanochat_muon

        return (make_nanochat_muon(parameter_groups, parameterization_contract, config),)
    if optim.type == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("optim.type='muon' requires a PyTorch build with torch.optim.Muon")
        muon_lr, adamw_lr = base_lrs
        muon_params = parameter_groups["muon.block_matrices"]
        adamw_decay = parameter_groups["adamw.decay"]
        adamw_no_decay = parameter_groups["adamw.no_decay"]
        muon_kwargs = {
            "lr": muon_lr,
            "weight_decay": optim.weight_decay,
            "momentum": optim.muon_momentum,
            "ns_steps": optim.muon_ns_steps,
        }
        if optim.muon_adjust_lr_fn:
            muon_kwargs["adjust_lr_fn"] = optim.muon_adjust_lr_fn
        return (
            torch.optim.Muon(muon_params, **muon_kwargs),
            torch.optim.AdamW(
                [
                    {"params": adamw_decay, "weight_decay": optim.weight_decay},
                    {"params": adamw_no_decay, "weight_decay": 0.0},
                ],
                lr=adamw_lr,
                betas=(optim.adam_beta1, optim.adam_beta2),
                eps=optim.adam_eps,
                fused=config.runtime.device_type == "cuda",
            ),
        )

    if config.parameterization.mode != "ordinary":
        resolved_groups = []
        for group in parameterization_contract["optimizer_groups"]:
            group_id = str(group["group_id"])
            resolved_groups.append(
                {
                    "params": parameter_groups[group_id],
                    "lr": float(group["resolved_lr"]),
                    "base_lr": float(group["resolved_lr"]),
                    "weight_decay": float(group["resolved_weight_decay"]),
                    "eps": float(group["resolved_adam_eps"]),
                    "parameterization_group_id": group_id,
                    "parameterization_role": str(group["role"]),
                    "parameterization_lr_multiplier": float(group["lr_multiplier"]),
                    "parameterization_weight_decay_multiplier": float(group["weight_decay_multiplier"]),
                    "parameterization_adam_eps_multiplier": float(group["adam_eps_multiplier"]),
                }
            )
        return (
            torch.optim.AdamW(
                resolved_groups,
                lr=base_lrs[0],
                betas=(optim.adam_beta1, optim.adam_beta2),
                eps=optim.adam_eps,
                **(
                    {"fused": False, "foreach": False}
                    if ar_width_mup.active(config)
                    else {"fused": config.runtime.device_type == "cuda"}
                ),
            ),
        )

    decay = parameter_groups["adamw.decay"]
    no_decay = parameter_groups.get("adamw.no_decay", [])
    return (
        torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": optim.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=base_lrs[0],
            betas=(optim.adam_beta1, optim.adam_beta2),
            eps=optim.adam_eps,
            fused=config.runtime.device_type == "cuda",
        ),
    )


def _optimizer_state_dict(optimizers) -> dict:
    optimizers = tuple(optimizers)
    if len(optimizers) == 1:
        return optimizers[0].state_dict()
    return {
        "format": "optimizer_list",
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
    }


def _load_optimizer_state_dict(optimizers, state: dict) -> None:
    optimizers = tuple(optimizers)
    if "optimizers" in state:
        if len(state["optimizers"]) != len(optimizers):
            raise ValueError(
                f"checkpoint has {len(state['optimizers'])} optimizer states, "
                f"but this run built {len(optimizers)} optimizers"
            )
        for optimizer, optimizer_state in zip(optimizers, state["optimizers"], strict=True):
            optimizer.load_state_dict(optimizer_state)
        return
    if len(optimizers) != 1:
        raise ValueError("single-optimizer checkpoint cannot be loaded into a multi-optimizer run")
    optimizers[0].load_state_dict(state)


def _iter_optimizer_param_groups(optimizers):
    for optimizer in optimizers:
        yield from optimizer.param_groups


def _optimizer_lr_metrics(optimizers) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for optimizer_idx, optimizer in enumerate(optimizers):
        group_lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
        if not group_lrs:
            continue
        metrics[f"optimizer_{optimizer_idx}_lr"] = group_lrs[0]
        if len(set(group_lrs)) > 1:
            metrics[f"optimizer_{optimizer_idx}_lr_min"] = min(group_lrs)
            metrics[f"optimizer_{optimizer_idx}_lr_max"] = max(group_lrs)
    metrics.update(_nanochat_optimizer_metrics(optimizers))
    return metrics


def _primary_optimizer_lr(optimizers) -> float:
    for group in _iter_optimizer_param_groups(optimizers):
        return float(group.get("lr", 0.0))
    return 0.0


def _zero_grad(optimizers) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def _unscale_optimizers(scaler, optimizers) -> None:
    for optimizer in optimizers:
        scaler.unscale_(optimizer)


def _step_optimizers(scaler, optimizers) -> None:
    for optimizer in optimizers:
        scaler.step(optimizer)
