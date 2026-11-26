"""Exact D8 scratch-training matrix; no implicit batch/LR or objective changes."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig

SCOPE = "d8_mdlm_batch_objective"
RECIPE = "d8-mdlm-batch-objective-v1"
BASE_CONFIG = "configs/d8_mdlm_batch_objective.json"
TOKEN_BUDGET = 1_006_632_960
TPP_PARAMETERS = 50_331_648
BATCHES = (32_768, 131_072, 524_288)
LEARNING_RATES = (2e-4, 4e-4, 8e-4)
OBJECTIVES = ("inverse_t", "constant")
PROFILES = tuple(f"eb{eb // 1024}k_{objective}" for eb in BATCHES for objective in OBJECTIVES)
PROBE_STEPS = 8


def parse_profile(profile: str) -> tuple[int, str]:
    if profile not in PROFILES:
        raise ValueError(f"unknown D8 batch/objective profile: {profile}")
    batch, objective = profile.split("_", 1)
    return int(batch[2:-1]) * 1024, objective


def config_payload(eb: int, objective: str, lr: float, *, probe: bool = False) -> dict:
    """One full-horizon config or its explicit eight-update runtime probe."""
    if eb not in BATCHES or objective not in OBJECTIVES or lr not in LEARNING_RATES:
        raise ValueError("outside the locked D8 batch/objective/LR matrix")
    from .config import ExperimentConfig

    value = asdict(ExperimentConfig())
    base = json.loads((Path(__file__).resolve().parents[1] / BASE_CONFIG).read_text())
    for section, fields in base.items():
        value[section].update(fields)
    steps = PROBE_STEPS if probe else TOKEN_BUDGET // eb
    value["optim"]["lr"] = lr
    value["noise"].update(
        schedule_weight=objective == "inverse_t",
        # E[1(masked) / t] = 1. Match expected scalar target-weight mass,
        # not gradient norm or optimization dynamics, for the constant surrogate.
        mdlm_unweighted_loss_scale=1.0 if objective == "inverse_t" else 2.0 / 1.05,
    )
    value["train"].update(
        total_batch_size=eb,
        num_steps=steps,
        target_tokens=steps * eb,
        target_param_data_ratio=-1.0 if probe else 20.0,
        save_every=4 if probe else steps // 8,
        log_interval=1 if probe else max(1, steps // 300),
    )
    if probe:
        value["recipe"]["name"] = RECIPE + "-probe"
        # A kernel/loss/reload probe needs real packed rows, not a full-corpus
        # footer scan. Material configurations retain the complete corpus.
        value["data"]["train_data"] = [str(Path(value["data"]["val_data"][0]).with_name("shard_00000.parquet"))]
        value["data"]["train_exclude_data"] = []
    return value


def make_config(eb: int, objective: str, lr: float, *, probe: bool = False) -> ExperimentConfig:
    from .config import config_from_dict

    return config_from_dict(config_payload(eb, objective, lr, probe=probe))


def validate_config(config: ExperimentConfig) -> None:
    """Compare every resolved field, admitting only attempt paths and derived digest."""
    objective = "inverse_t" if config.noise.schedule_weight else "constant"
    probe = config.recipe.name == RECIPE + "-probe"
    expected = config_payload(config.train.total_batch_size, objective, config.optim.lr, probe=probe)
    observed = asdict(config)
    for section, field in (("train", "run_dir"), ("train", "resume_from"), ("parameterization", "contract_sha256")):
        observed[section][field] = expected[section][field]
    if config.train.resume_from and not Path(config.train.resume_from).is_absolute():
        raise ValueError("D8 batch/objective resume checkpoint must be absolute")
    if observed != expected:
        differences = [
            f"{section}.{key}"
            for section, fields in expected.items()
            for key, value in fields.items()
            if observed[section][key] != value
        ]
        raise ValueError(f"D8 batch/objective config differs: {', '.join(differences)}")


def parameterization_contract(contract: dict, mechanics: dict, *, probe: bool) -> dict:
    """Reuse the existing CompleteP mathematics with a new bounded execution scope."""
    material = copy.deepcopy({k: v for k, v in contract.items() if k != "contract_sha256"})
    material.update(
        schema="nanodllm.parameterization_contract.v9",
        adaptation_status="implemented_mechanics_d8_batch_objective_screen",
        mechanics=copy.deepcopy(mechanics),
        scientific_training_authorized=not probe,
        scientific_execution_scope=SCOPE,
        campaign_plan_binding_required=True,
        compiled_bf16_qualification_status="targeted_new_objective_probe_required_before_material_release",
        authorization_boundary="exact_D8_DB4_WS1_EB_objective_LR_screen_only",
    )
    return {
        **material,
        "contract_sha256": hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        ).hexdigest(),
    }
