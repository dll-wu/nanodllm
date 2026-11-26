"""Identify the training stage that supplied weights to a named comparison."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .telemetry_config import TELEMETRY_OPERATIONAL_FIELDS


def settings_digest(settings, world_size, *, initialized):
    """Hash the resolved training settings, independently of where the run was stored."""
    sections = {
        k: dict(settings.get(k, {}))
        for k in ("model", "noise", "optim", "parameterization", "runtime", "data", "train")
    }
    # The trainer fills this derived digest after instantiating the model.
    sections["parameterization"].pop("contract_sha256", None)
    for key in ("train_data", "tokenizer_path", "dataset_snapshot"):
        sections["data"].pop(key, None)
    for key in ("val_data", "train_exclude_data"):
        sections["data"][key] = [Path(p).name for p in sections["data"].get(key, [])]
    for key in (
        "run_dir",
        "resume_from",
        "model_init_from",
        "model_init_expected_sha256",
        "model_init_expected_size_bytes",
        "comparison_run_ids",
        "log_interval",
        *TELEMETRY_OPERATIONAL_FIELDS,
    ):
        sections["train"].pop(key, None)
    sections["runtime"].pop("min_free_gpu_mib", None)
    material = json.dumps([world_size, initialized, sections], sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(material.encode()).hexdigest()


@dataclass(frozen=True)
class ParentRequirement:
    name: str
    settings_sha256: str
    archived_sha256: str | None
    causal: bool
    noise_type: str
    steps: int
    batch_size: int


def native_profile_receipt(checkpoint, settings):
    """Exports retain a portable digest of the actual training config and rank count."""
    name = settings.get("recipe", {}).get("name")
    contract = checkpoint.get("resume_contract")
    if not name or not contract:
        return None
    from .training_checkpoint import _validated_trajectory_record

    trajectory = _validated_trajectory_record(contract.get("trajectory"))
    budget = trajectory["derived_training_budget"]
    return {
        "name": name,
        "settings_sha256": settings_digest(
            settings, contract["world_size"], initialized=bool(settings["train"].get("model_init_from"))
        ),
        "planned_steps": budget["total_steps"],
        "batch_size": budget["total_batch_size"],
    }


def verify_recipe_parent(required, checkpoint, *, portable, file_sha256):
    if portable:
        model, noise = checkpoint["config"], checkpoint["noise"]
        archived = checkpoint.get("source_sha256")
        receipt = checkpoint.get("profile_receipt")
    else:
        model = checkpoint.get("model_config", checkpoint["config"])
        settings = checkpoint["config"] if "model_config" in checkpoint else checkpoint["train_config"]
        noise = settings["noise"]
        archived = file_sha256
        receipt = None
    if model.get("causal") is not required.causal or noise.get("type") != required.noise_type:
        raise ValueError(f"wrong training objective for parent profile {required.name}")

    pinned = required.archived_sha256 is not None and archived == required.archived_sha256
    if not pinned:
        if not portable:
            receipt = native_profile_receipt(checkpoint, settings)
        if not receipt:
            raise ValueError(
                f"missing training profile receipt for {required.name}; use its native checkpoint or re-export it"
            )
        if receipt.get("name") != required.name or receipt.get("settings_sha256") != required.settings_sha256:
            raise ValueError(f"parent training profile or settings differ from {required.name}")

    # A partial checkpoint has the same recipe as the completed model.
    receipt = receipt or {}
    steps = required.steps if required.steps > 0 else receipt.get("planned_steps")
    batch = required.batch_size if required.batch_size > 0 else receipt.get("batch_size")
    if (
        type(steps) is not int
        or steps <= 0
        or type(batch) is not int
        or batch <= 0
        or checkpoint.get("step") != steps
        or checkpoint.get("train_tokens") != steps * batch
    ):
        raise ValueError(f"parent has not completed the prescribed training budget for {required.name}")
    return "archived_sha256" if pinned else "training_profile_receipt"
