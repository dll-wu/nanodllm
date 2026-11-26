"""Admission and parameterization metadata for the paired D4 init confirmation."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

from .parameterization_calibration import (
    D4_ROOT_CALIBRATION_EVALUATION_SEED_BLOCK,
    D4_ROOT_CALIBRATION_EXECUTION_SCOPE,
    D4_ROOT_CALIBRATION_NELBO_SEED,
    D4_ROOT_CALIBRATION_RECIPE,
    D4_ROOT_CALIBRATION_SEED_BLOCKS,
    calibration_parameterization_contract,
    validate_d4_root_calibration_config,
)

D4_HIDDEN_INIT_CONFIRMATION_EXECUTION_SCOPE = "d4_mdlm_hidden_init_confirmation"
D4_HIDDEN_INIT_CONFIRMATION_RECIPE = "d4-mdlm-hidden-init-0p5-independent-confirmation-v2"
D4_HIDDEN_INIT_CONFIRMATION_SEED_NAMESPACE = "nanodllm-d4-mdlm-hidden-init-0p5-independent-confirmation-v2"
D4_HIDDEN_INIT_CONFIRMATION_POINTS = ((4e-4, 1.0), (4e-4, 0.5))


def derive_hidden_init_confirmation_seed(block: str, role: str) -> int:
    """Derive a frozen unsigned 31-bit role seed from the v2 namespace."""

    material = f"{D4_HIDDEN_INIT_CONFIRMATION_SEED_NAMESPACE}:{block}:{role}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


D4_HIDDEN_INIT_CONFIRMATION_SEED_BLOCK = tuple(
    derive_hidden_init_confirmation_seed("training", role) for role in ("runtime", "model", "training", "data")
)
D4_HIDDEN_INIT_CONFIRMATION_EVALUATION_SEED_BLOCK = tuple(
    derive_hidden_init_confirmation_seed("evaluation", role) for role in ("runtime", "model", "training", "data")
)
D4_HIDDEN_INIT_CONFIRMATION_NELBO_SEED = derive_hidden_init_confirmation_seed("evaluation", "nelbo-plan")


def validate_hidden_init_confirmation_seed_independence() -> None:
    """Reject any replay of the calibration training or evaluation randomness."""

    if D4_HIDDEN_INIT_CONFIRMATION_SEED_BLOCK in D4_ROOT_CALIBRATION_SEED_BLOCKS:
        raise ValueError("D4 hidden-init confirmation training seeds replay a root-calibration block")
    if D4_HIDDEN_INIT_CONFIRMATION_EVALUATION_SEED_BLOCK == D4_ROOT_CALIBRATION_EVALUATION_SEED_BLOCK:
        raise ValueError("D4 hidden-init confirmation evaluation seeds replay root calibration")
    if D4_HIDDEN_INIT_CONFIRMATION_NELBO_SEED == D4_ROOT_CALIBRATION_NELBO_SEED:
        raise ValueError("D4 hidden-init confirmation NELBO plan replays root calibration")
    confirmation_values = {
        *D4_HIDDEN_INIT_CONFIRMATION_SEED_BLOCK,
        *D4_HIDDEN_INIT_CONFIRMATION_EVALUATION_SEED_BLOCK,
        D4_HIDDEN_INIT_CONFIRMATION_NELBO_SEED,
    }
    calibration_values = {
        *(value for block in D4_ROOT_CALIBRATION_SEED_BLOCKS for value in block),
        *D4_ROOT_CALIBRATION_EVALUATION_SEED_BLOCK,
        D4_ROOT_CALIBRATION_NELBO_SEED,
    }
    if not confirmation_values.isdisjoint(calibration_values):
        raise ValueError("D4 hidden-init confirmation reuses a root-calibration seed value")


def is_d4_hidden_init_confirmation(config) -> bool:
    return config.parameterization.execution_scope == D4_HIDDEN_INIT_CONFIRMATION_EXECUTION_SCOPE


def validate_d4_hidden_init_confirmation_config(config) -> str:
    """Reuse exact root mechanics while admitting only the paired confirmation."""

    validate_hidden_init_confirmation_seed_independence()
    if config.parameterization.mode != "ordinary":
        raise ValueError("D4 hidden-init confirmation requires ordinary parameterization")
    if config.recipe.name != D4_HIDDEN_INIT_CONFIRMATION_RECIPE:
        raise ValueError("D4 hidden-init confirmation recipe differs")
    if (float(config.optim.lr), float(config.parameterization.hidden_init_std_multiplier)) not in (
        D4_HIDDEN_INIT_CONFIRMATION_POINTS
    ):
        raise ValueError("D4 hidden-init confirmation permits only init 1.0 and 0.5 at root LR 4e-4")
    profile = "evaluation" if config.train.eval_only else "training"
    observed_seed_block = (
        int(config.runtime.seed),
        int(config.runtime.model_seed),
        int(config.runtime.training_seed),
        int(config.data.shuffle_seed),
    )
    expected_seed_block = (
        D4_HIDDEN_INIT_CONFIRMATION_EVALUATION_SEED_BLOCK
        if config.train.eval_only
        else D4_HIDDEN_INIT_CONFIRMATION_SEED_BLOCK
    )
    if observed_seed_block != expected_seed_block:
        raise ValueError(f"D4 hidden-init confirmation {profile} seed block differs")
    if int(config.eval.nelbo_seed) != D4_HIDDEN_INIT_CONFIRMATION_NELBO_SEED:
        raise ValueError("D4 hidden-init confirmation NELBO plan seed differs")

    shadow = copy.deepcopy(config)
    shadow.parameterization.execution_scope = D4_ROOT_CALIBRATION_EXECUTION_SCOPE
    shadow.recipe.name = D4_ROOT_CALIBRATION_RECIPE
    mechanics_seed_block = (
        D4_ROOT_CALIBRATION_EVALUATION_SEED_BLOCK if config.train.eval_only else min(D4_ROOT_CALIBRATION_SEED_BLOCKS)
    )
    (
        shadow.runtime.seed,
        shadow.runtime.model_seed,
        shadow.runtime.training_seed,
        shadow.data.shuffle_seed,
    ) = mechanics_seed_block
    shadow.eval.nelbo_seed = D4_ROOT_CALIBRATION_NELBO_SEED
    observed_profile = validate_d4_root_calibration_config(shadow)
    if observed_profile != profile:
        raise ValueError("D4 hidden-init confirmation profile differs from root mechanics")
    return profile


def hidden_init_confirmation_parameterization_contract(
    contract: Mapping[str, object], mechanics: Mapping[str, object]
) -> dict[str, object]:
    """Bind root mechanics to the confirmation-only authority envelope."""

    calibrated = calibration_parameterization_contract(contract, mechanics)
    material = copy.deepcopy({key: value for key, value in calibrated.items() if key != "contract_sha256"})
    material.update(
        {
            "adaptation_status": "implemented_mechanics_d4_hidden_init_confirmation_authorized",
            "compiled_bf16_qualification_status": "reuses_completed_d4_root_mechanics_qualification",
            "scientific_training_authorization_condition": "exact_paired_confirmation_contract_only",
            "scientific_execution_scope": D4_HIDDEN_INIT_CONFIRMATION_EXECUTION_SCOPE,
            "authorization_boundary": "exact_locked_d4_hidden_init_0p5_vs_1_confirmation_only",
            "qualification_evidence_owner": "completed_d4_root_calibration_mechanics_and_confirmation_plan",
        }
    )
    payload = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {**material, "contract_sha256": hashlib.sha256(payload).hexdigest()}
