"""Exact, stopped-gradient checkpoint teachers for self-generated token-edit drafts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch

from .config import ExperimentConfig
from .model import DiffusionConfig, DiffusionTransformer
from .training_features import _uses_frozen_token_edit_draft_model
from .training_initialization import load_model_state_initialization

FROZEN_DRAFT_MODEL_SOURCE_SCHEMA = "nanodllm.frozen_token_edit_draft_model_source.v1"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_token_edit_draft_model(
    config: ExperimentConfig,
    *,
    model_config: DiffusionConfig,
    device: torch.device,
    compute_dtype: torch.dtype,
    token_class_ids: torch.Tensor | None,
    rank: int,
    world_size: int,
    student_architecture_fingerprint: Mapping[str, object],
) -> tuple[DiffusionTransformer | None, dict[str, object] | None]:
    """Construct and admit the exact frozen draft teacher, if configured.

    The caller owns RNG preservation around this function. The returned module
    is excluded from optimizer construction, has no trainable parameters, and
    remains in evaluation mode for the lifetime of the training session.
    """

    if not _uses_frozen_token_edit_draft_model(config):
        return None, None

    teacher = DiffusionTransformer(
        model_config,
        rope_base=config.model.rope_base,
        compute_dtype=compute_dtype,
        pad_vocab_to=config.model.pad_vocab_to,
        token_class_ids=token_class_ids,
    ).to(device)
    source = load_model_state_initialization(
        teacher,
        checkpoint_path=config.train.token_edit_draft_checkpoint,
        expected_sha256=config.train.token_edit_draft_checkpoint_expected_sha256,
        expected_size_bytes=int(config.train.token_edit_draft_checkpoint_expected_size_bytes),
        rank=rank,
        world_size=world_size,
        allow_token_edit_extensions=False,
    )
    if source is None:  # Configuration validation requires an exact binding.
        raise RuntimeError("frozen token-edit draft checkpoint was not loaded")

    teacher_fingerprint = teacher.architecture_fingerprint()
    if teacher_fingerprint != dict(student_architecture_fingerprint):
        raise RuntimeError("frozen token-edit draft teacher architecture differs from the student")

    teacher.requires_grad_(False)
    teacher.eval()
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("frozen token-edit draft teacher retains trainable parameters")

    normalized_source = {key: value for key, value in source.items() if key not in {"path", "identity_seconds"}}
    artifact: dict[str, object] = {
        "schema": FROZEN_DRAFT_MODEL_SOURCE_SCHEMA,
        "draft_model_source": "frozen_checkpoint",
        "student_architecture_fingerprint_sha256": student_architecture_fingerprint["fingerprint_sha256"],
        "teacher_architecture_fingerprint_sha256": teacher_fingerprint["fingerprint_sha256"],
        "checkpoint_source": normalized_source,
        "gradient_policy": "stopped_gradient",
        "optimizer_membership": "excluded",
        "requires_grad": False,
        "module_training_mode": False,
        "runtime_compile": bool(config.runtime.compile),
        "rng_policy": "caller_restores_all_training_rng_streams_after_teacher_construction_and_load",
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return teacher, artifact


def validate_frozen_token_edit_draft_model_source_artifact(value: object) -> dict[str, object]:
    """Validate the immutable source receipt emitted by a frozen teacher run."""

    if not isinstance(value, Mapping):
        raise ValueError("frozen draft source artifact must be a mapping")
    artifact = dict(value)
    if artifact.get("schema") != FROZEN_DRAFT_MODEL_SOURCE_SCHEMA:
        raise ValueError("frozen draft source artifact schema differs")
    if artifact.get("draft_model_source") != "frozen_checkpoint":
        raise ValueError("frozen draft source role differs")
    if artifact.get("gradient_policy") != "stopped_gradient" or artifact.get("optimizer_membership") != "excluded":
        raise ValueError("frozen draft gradient or optimizer policy differs")
    if artifact.get("requires_grad") is not False or artifact.get("module_training_mode") is not False:
        raise ValueError("frozen draft module state differs")
    for field_name in (
        "student_architecture_fingerprint_sha256",
        "teacher_architecture_fingerprint_sha256",
        "artifact_sha256",
    ):
        digest = artifact.get(field_name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"frozen draft source {field_name} must be lowercase 64-hex")
    if artifact["student_architecture_fingerprint_sha256"] != artifact["teacher_architecture_fingerprint_sha256"]:
        raise ValueError("frozen draft teacher and student architectures differ")
    source = artifact.get("checkpoint_source")
    if not isinstance(source, Mapping):
        raise ValueError("frozen draft checkpoint receipt is absent")
    if (
        source.get("sha256") is None
        or source.get("size_bytes") is None
        or source.get("restored_model_state") is not True
    ):
        raise ValueError("frozen draft checkpoint receipt is incomplete")
    material = {key: item for key, item in artifact.items() if key != "artifact_sha256"}
    if artifact["artifact_sha256"] != _canonical_sha256(material):
        raise ValueError("frozen draft source artifact digest differs")
    return artifact
