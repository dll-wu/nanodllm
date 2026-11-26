"""Opt-in native momentum/decay time profiles, independent of the LR schedule.

The pinned nanochat trainer evaluates these at the zero-based update index.
No scheduler counter is persisted: exact resume reconstructs them from the
trainer's restored step and the original total horizon. Auxiliary AdamW roles
never receive these schedules. Historical constant contracts remain unchanged.
"""

import math

MOMENTUM_FIELD = "nanochat_momentum_schedule"
DECAY_FIELD = "nanochat_weight_decay_schedule"
SCHEDULE_VARIANT = "nanochat-92d63d4-scheduled-momentum-decay-v1"


def active(config) -> bool:
    return any(getattr(config.optim, name) != "constant" for name in (MOMENTUM_FIELD, DECAY_FIELD))


def strip_constant_schedules(value: dict) -> None:
    for name in (MOMENTUM_FIELD, DECAY_FIELD):
        if value["optim"].get(name) == "constant":
            value["optim"].pop(name)


def validate_config(config) -> None:
    o = config.optim
    if o.nanochat_momentum_schedule not in {"constant", "native"}:
        raise ValueError("nanochat_momentum_schedule must be constant or native")
    if o.nanochat_weight_decay_schedule not in {"constant", "cosine"}:
        raise ValueError("nanochat_weight_decay_schedule must be constant or cosine")
    if not active(config):
        return
    if (
        o.type != "nanochat_muon"
        or o.nanochat_matrix_optimizer != "muon"
        or o.nanochat_execution_scope
        not in {"ddp_replicated_d4_d8", "ddp_replicated_plain_d12", "ddp_replicated_plain_d20"}
        or config.parameterization.execution_scope != "ordinary"
    ):
        raise ValueError("scheduled nanochat Muon requires ordinary replicated D4/D8 or explicit plain D12/D20 Muon")
    if o.nanochat_momentum_schedule == "native" and o.muon_momentum != 0.95:
        raise ValueError("native momentum schedule requires the explicit historical momentum root 0.95")
    if not math.isfinite(o.warmdown_ratio) or not 0 <= o.warmdown_ratio <= 1:
        raise ValueError("scheduled nanochat Muon requires warmdown_ratio in [0,1]")


def values(config, step: int, total_steps: int) -> dict[str, float]:
    """Return the values used for an update, including the terminal boundary."""
    if total_steps <= 0 or not 0 <= step <= total_steps:
        raise ValueError("optimizer schedule needs a positive horizon and an in-range update index")
    o = config.optim
    momentum = o.muon_momentum
    if o.nanochat_momentum_schedule == "native":
        cooldown_steps = round(o.warmdown_ratio * total_steps)
        cooldown_start = total_steps - cooldown_steps
        # Preserve native branch order, including on very short test horizons.
        if step < 400:
            fraction = step / 400
            momentum = (1 - fraction) * 0.85 + fraction * 0.97
        elif cooldown_steps and step >= cooldown_start:
            fraction = (step - cooldown_start) / cooldown_steps
            momentum = 0.97 * (1 - fraction) + 0.90 * fraction
        else:
            momentum = 0.97
    decay = o.weight_decay
    if o.nanochat_weight_decay_schedule == "cosine":
        decay *= 0.5 * (1 + math.cos(math.pi * step / total_steps))
    return {"momentum": momentum, "weight_decay": decay}


def contract_fields(config) -> dict:
    """Add identity only on opt-in; never rewrite a historical constant digest."""
    if not active(config):
        return {}
    return {
        "variant": SCHEDULE_VARIANT,
        "momentum_schedule": config.optim.nanochat_momentum_schedule,
        "weight_decay_schedule": config.optim.nanochat_weight_decay_schedule,
        "optimizer_schedule": {
            "source": "nanochat@92d63d4e8bb4df75c3b71618f31ddde2378b2bcd:scripts/base_train.py",
            "step_convention": "zero_based_optimizer_update; reconstructed_on_resume",
            "momentum_warmup_steps": 400,
            "momentum_start_peak_final": [0.85, 0.97, 0.90],
            "momentum_cooldown_fraction": config.optim.warmdown_ratio,
            "decay_initial": config.optim.weight_decay,
            "decay_cosine_horizon": "complete_training_horizon",
            "auxiliary_adamw_scheduled": False,
            "lr_schedule_changed": False,
            "decay_integral_matched": False,
        },
    }


def apply(optimizers, config, step: int, total_steps: int) -> None:
    if not active(config):
        return
    current = values(config, step, total_steps)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            if group.get("kind") == "muon":
                group.update(current)


def metrics(optimizers) -> dict[str, float]:
    """Expose actual role settings without additional tensor reductions."""
    result = {}
    for index, optimizer in enumerate(optimizers):
        for group in optimizer.param_groups:
            kind = group.get("kind")
            if kind == "muon":
                for name in ("momentum", "weight_decay"):
                    result[f"optimizer_{index}_muon_{name}"] = float(group[name])
            role = group.get("parameterization_group_id", "")
            if role in {"adamw.token_embedding", "adamw.output_head"}:
                result[f"optimizer_{index}_{role.removeprefix('adamw.')}_lr"] = float(group["lr"])
    return result
