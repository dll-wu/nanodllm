"""Hash-bound model-state initialization for fresh training trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from .nanollada_completep import validate_model_initialization
from .parameterization import validate_checkpoint_parameterization_contract
from .recipe_lineage import ParentRequirement, verify_recipe_parent
from .training_rng import _restore_rng_state, _rng_state

MODEL_STATE_INITIALIZATION_SCHEMA = "nanodllm.model_state_initialization.v1"
MODEL_STATE_INITIALIZATION_LOAD_MODE = (
    "strict_model_state_only_verified_private_file_without_optimizer_scaler_rng_progress_or_loader_state"
)
TOKEN_EDIT_EXTENSION_MODEL_STATE_INITIALIZATION_LOAD_MODE = "strict_base_model_state_only_with_seeded_token_edit_extensions_without_optimizer_scaler_rng_progress_or_loader_state"
TOKEN_EDIT_EXTENSION_STATE_PREFIXES = ("token_edit_adapters.", "token_edit_provenance_embedding.")


def checkpoint_file_identity(path: Path) -> tuple[str, int]:
    """Hash the exact checkpoint bytes before trusted deserialization."""

    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_state_descriptors(raw_model: nn.Module) -> list[dict[str, object]]:
    """Return the compact ordered tensor contract used before DDP transfer."""

    descriptors: list[dict[str, object]] = []
    for kind, tensors in (
        ("parameter", raw_model.named_parameters()),
        ("buffer", raw_model.named_buffers()),
    ):
        for name, value in sorted(tensors, key=lambda item: item[0]):
            if (
                value.layout != torch.strided
                or not value.is_contiguous()
                or value.is_quantized
                or value.device.type not in {"cpu", "cuda"}
            ):
                raise TypeError(
                    "distributed model-init supports only contiguous dense strided CPU/CUDA tensors: "
                    f"{kind} {name} has layout={value.layout}, device={value.device.type}, "
                    f"contiguous={value.is_contiguous()}, quantized={value.is_quantized}"
                )
            descriptors.append(
                {
                    "kind": kind,
                    "name": name,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "numel": int(value.numel()),
                    "layout": str(value.layout),
                    "device_type": value.device.type,
                    "requires_grad": bool(value.requires_grad),
                }
            )
    return descriptors


def _model_state_sha256(raw_model: nn.Module) -> str:
    """Hash exact parameter and buffer metadata/bytes after distributed import."""

    digest = hashlib.sha256()
    digest.update(b"nanodllm.imported_model_state.v1\0")
    for kind, tensors in (
        ("parameter", raw_model.named_parameters()),
        ("buffer", raw_model.named_buffers()),
    ):
        for name, tensor in sorted(tensors, key=lambda item: item[0]):
            value = tensor.detach().contiguous()
            metadata = {
                "kind": kind,
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "numel": int(value.numel()),
                "nbytes": int(value.numel() * value.element_size()),
            }
            digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\0")
            byte_view = value.reshape(-1).view(torch.uint8).cpu().numpy()
            digest.update(memoryview(byte_view))
            digest.update(b"\0")
    return digest.hexdigest()


def _named_tensor_subset_sha256(raw_model: nn.Module, names: list[str]) -> str:
    named_state = dict(raw_model.named_parameters()) | dict(raw_model.named_buffers())
    digest = hashlib.sha256()
    digest.update(b"nanodllm.named_tensor_subset.v1\0")
    for name in sorted(names):
        if name not in named_state:
            raise ValueError(f"declared extension tensor is absent from the child model: {name}")
        tensor = named_state[name].detach().contiguous()
        metadata = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).cpu().numpy()))
        digest.update(b"\0")
    return digest.hexdigest()


def _all_gather_local_result(*, operation: str, value_factory: Callable[[], object]) -> list[dict[str, object]]:
    """Run local work and exchange terminal status before any later collective."""

    try:
        local: dict[str, object] = {"status": "ok", "value": value_factory()}
    except Exception as exc:  # noqa: BLE001 - peers must learn that this rank cannot continue.
        local = {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}
    gathered: list[object | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    failures = []
    for rank, value in enumerate(gathered):
        if isinstance(value, Mapping) and value.get("status") == "ok":
            continue
        details = dict(value) if isinstance(value, Mapping) else {"receipt_type": type(value).__name__}
        failures.append({"rank": rank, **details})
    if failures:
        raise RuntimeError(f"distributed model-init {operation} failed coherently: {failures}")
    return [dict(value) for value in gathered if isinstance(value, Mapping)]


@contextmanager
def _verified_private_checkpoint_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
):
    """Copy and verify once, then deserialize through the same private descriptor."""

    with tempfile.TemporaryFile(mode="w+b", prefix="nanodllm-model-init-", suffix=".pt") as temporary:
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                size_bytes += len(chunk)
                temporary.write(chunk)
        temporary.flush()
        os.fsync(temporary.fileno())
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"model-init checkpoint SHA-256 mismatch: observed {observed_sha256}, expected {expected_sha256}"
            )
        if size_bytes != expected_size_bytes:
            raise ValueError(
                f"model-init checkpoint size mismatch: observed {size_bytes}, expected {expected_size_bytes}"
            )
        temporary.seek(0)
        yield temporary, observed_sha256, size_bytes


def initialize_from_parent(session):
    """Load the parent without advancing the child's seeded training RNGs."""
    config = session.config
    session.saved_state = _rng_state(config)
    try:
        session.model_initialization_source = load_model_state_initialization(
            session.raw_model,
            checkpoint_path=config.train.model_init_from,
            parent_requirement=config.recipe.parent_requirement,
            tokenizer=session.tokenizer,
            expected_sha256=config.train.model_init_expected_sha256,
            expected_size_bytes=int(config.train.model_init_expected_size_bytes),
            rank=session.rank,
            world_size=session.world_size,
            allow_token_edit_extensions=bool(config.train.model_init_allow_token_edit_extensions),
            allow_block_self_conditioning=config.train.model_init_allow_block_self_conditioning,
        )
    finally:
        _restore_rng_state(session.saved_state, config)


def load_model_state_initialization(
    raw_model: nn.Module,
    *,
    checkpoint_path: str,
    expected_sha256: str,
    expected_size_bytes: int,
    rank: int,
    world_size: int,
    allow_token_edit_extensions: bool = False,
    allow_block_self_conditioning: bool = False,
    tokenizer=None,
    parent_requirement: ParentRequirement | None = None,
) -> dict[str, object] | None:
    """Load only model parameters from an exact checkpoint into a fresh run.

    The caller constructs the model and all role-separated RNG streams normally.
    This function replaces the model state only. Optimizer, scaler, progress,
    schedule, RNG, and loader state from the source checkpoint are deliberately
    ignored and therefore start from the child run's own contract.
    """

    if not checkpoint_path:
        return None
    if allow_block_self_conditioning and (world_size != 1 or allow_token_edit_extensions):
        raise ValueError("self-conditioning initialization is isolated and single-GPU only")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid model-init rank topology: rank={rank}, world_size={world_size}")
    if world_size == 1:
        if rank != 0:
            raise ValueError(f"invalid single-rank model-init rank: {rank}")
        return _load_model_state_initialization_rank_zero(
            raw_model,
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            parent_requirement=parent_requirement,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            allow_token_edit_extensions=allow_token_edit_extensions,
            allow_block_self_conditioning=allow_block_self_conditioning,
        )

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed model initialization requires an initialized process group")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError(
            "distributed model-init topology differs from the active process group: "
            f"arguments=({rank},{world_size}), active=({dist.get_rank()},{dist.get_world_size()})"
        )

    descriptor_receipts = _all_gather_local_result(
        operation="tensor-descriptor preflight",
        value_factory=lambda: _model_state_descriptors(raw_model),
    )
    reference_descriptors = descriptor_receipts[0]["value"]
    if any(receipt["value"] != reference_descriptors for receipt in descriptor_receipts[1:]):
        descriptor_roots = [_canonical_sha256(receipt["value"]) for receipt in descriptor_receipts]
        raise RuntimeError(
            f"distributed model-init parameter/buffer descriptors differ across ranks: {descriptor_roots}"
        )

    admission: list[object | None] = [None]
    if rank == 0:
        try:
            source = _load_model_state_initialization_rank_zero(
                raw_model,
                checkpoint_path=checkpoint_path,
                tokenizer=tokenizer,
                parent_requirement=parent_requirement,
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
                allow_token_edit_extensions=allow_token_edit_extensions,
            )
            admission[0] = {"status": "ok", "source": source}
        except Exception as exc:  # noqa: BLE001 - every rank must receive a terminal collective result.
            admission[0] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
    dist.broadcast_object_list(admission, src=0)
    receipt = admission[0]
    if not isinstance(receipt, Mapping):
        raise RuntimeError("rank zero did not broadcast a model-init admission receipt")
    if receipt.get("status") != "ok":
        raise RuntimeError(
            "rank-zero model-init admission failed: "
            f"{receipt.get('error_type', 'Error')}: {receipt.get('message', 'unknown failure')}"
        )
    source = receipt.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError("rank-zero model-init admission receipt has no source metadata")

    # Only rank zero opens and deserializes the monolithic parent checkpoint. The
    # exact admitted model state is then transferred over the active process
    # group before any child optimizer is constructed.
    with torch.no_grad():
        for _name, parameter in sorted(raw_model.named_parameters(), key=lambda item: item[0]):
            dist.broadcast(parameter, src=0)
        for _name, buffer in sorted(raw_model.named_buffers(), key=lambda item: item[0]):
            dist.broadcast(buffer, src=0)

    digest_receipts = _all_gather_local_result(
        operation="imported parameter/buffer-state digest",
        value_factory=lambda: _model_state_sha256(raw_model),
    )
    model_state_digests = [receipt["value"] for receipt in digest_receipts]
    if (
        not isinstance(model_state_digests[0], str)
        or len(model_state_digests[0]) != 64
        or any(value != model_state_digests[0] for value in model_state_digests[1:])
    ):
        raise RuntimeError(
            f"distributed model-init parameter/buffer-state digests differ across ranks: {model_state_digests}"
        )

    result = dict(source)
    result.update(
        {
            "distributed_world_size": world_size,
            "distributed_transfer": "rank_zero_verified_load_then_parameter_and_buffer_broadcast",
            "source_checkpoint_deserialized_ranks": [0],
            "cross_rank_model_state_hash_method": (
                "sha256_domain_separated_sorted_named_parameter_and_buffer_metadata_and_raw_bytes_v1"
            ),
            "cross_rank_model_state_sha256": model_state_digests[0],
            "cross_rank_model_state_digest_verified": True,
            "cross_rank_model_state_digest_rank_count": world_size,
            "distributed_tensor_descriptor_sha256": _canonical_sha256(reference_descriptors),
            "distributed_tensor_descriptor_rank_count": world_size,
        }
    )
    return result


def _load_model_state_initialization_rank_zero(
    raw_model: nn.Module,
    *,
    checkpoint_path: str,
    expected_sha256: str,
    expected_size_bytes: int,
    allow_token_edit_extensions: bool = False,
    allow_block_self_conditioning: bool = False,
    tokenizer=None,
    parent_requirement: ParentRequirement | None = None,
) -> dict[str, object]:
    """Admit and load one parent checkpoint on the designated source rank."""

    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        raise ValueError("train.model_init_from must be an absolute checkpoint path")
    path = path.resolve(strict=True)

    identity_started = time.monotonic()
    with _verified_private_checkpoint_file(
        path,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    ) as (verified_file, observed_sha256, observed_size_bytes):
        identity_seconds = time.monotonic() - identity_started
        portable = path.name.endswith(".weights.pt")
        checkpoint = torch.load(verified_file, map_location="cpu", weights_only=portable)
        if parent_requirement is not None:
            parent_binding = verify_recipe_parent(
                parent_requirement, checkpoint, portable=portable, file_sha256=observed_sha256
            )
            if not portable and parent_binding == "training_profile_receipt":
                from ..evaluation.checkpoint import tokenizer_identity, training_tokenizer_sha256

                settings = checkpoint["config"] if "model_config" in checkpoint else checkpoint["train_config"]
                if settings["data"]["tokenizer"] == "hf" and (
                    tokenizer is None
                    or training_tokenizer_sha256(path, checkpoint=checkpoint) != tokenizer_identity(tokenizer)
                ):
                    raise ValueError("parent training tokenizer differs from the child tokenizer")
        if portable:
            from ..evaluation.checkpoint import tokenizer_identity
            from ..weights import portable_parent

            if tokenizer is None:
                raise ValueError("portable parent requires the child tokenizer")
            checkpoint = portable_parent(
                checkpoint, target_model=raw_model, expected_tokenizer_sha256=tokenizer_identity(tokenizer)
            )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("model-init checkpoint root must be a mapping")
    if hasattr(raw_model, "config"):
        validate_model_initialization(checkpoint, raw_model)
    source_parameterization = validate_checkpoint_parameterization_contract(
        checkpoint,
        raw_model,
        allow_token_edit_extensions=allow_token_edit_extensions,
        allow_block_self_conditioning=allow_block_self_conditioning,
    )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("model-init checkpoint must contain model_state_dict")
    extension_allowed = allow_token_edit_extensions or allow_block_self_conditioning
    incompatible = raw_model.load_state_dict(state_dict, strict=not extension_allowed)
    missing_keys = sorted(incompatible.missing_keys)
    unexpected_keys = sorted(incompatible.unexpected_keys)
    if unexpected_keys:
        raise ValueError(f"model-init state has unexpected tensors: {unexpected_keys}")
    if extension_allowed:
        if not missing_keys:
            raise ValueError("extended token-edit model initialization requires missing child extension tensors")
        invalid_missing = [
            key
            for key in missing_keys
            if not any(
                key.startswith(prefix)
                for prefix in (
                    ("block_self_conditioning.",)
                    if allow_block_self_conditioning
                    else TOKEN_EDIT_EXTENSION_STATE_PREFIXES
                )
            )
        ]
        if invalid_missing:
            raise ValueError(f"model-init state is missing non-extension tensors: {invalid_missing}")
        extension_state_sha256 = _named_tensor_subset_sha256(raw_model, missing_keys)
        load_mode = TOKEN_EDIT_EXTENSION_MODEL_STATE_INITIALIZATION_LOAD_MODE
        if allow_block_self_conditioning:
            load_mode = "strict_base_model_state_with_seeded_zero_output_block_self_conditioning"
    else:
        if missing_keys:
            raise ValueError(f"model-init state differs from the child architecture: {incompatible}")
        extension_state_sha256 = ""
        load_mode = MODEL_STATE_INITIALIZATION_LOAD_MODE

    source_step = checkpoint.get("step")
    source_train_tokens = checkpoint.get("train_tokens")
    source_train_valid_tokens = checkpoint.get("train_valid_tokens")
    for name, value in (
        ("step", source_step),
        ("train_tokens", source_train_tokens),
        ("train_valid_tokens", source_train_valid_tokens),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"model-init checkpoint {name} must be a nonnegative integer")

    embedded_model_config = checkpoint.get("config")
    embedded_train_config = checkpoint.get("train_config")
    if not isinstance(embedded_model_config, Mapping) or not embedded_model_config:
        raise ValueError("model-init checkpoint must contain a resolved model config")
    if not isinstance(embedded_train_config, Mapping) or not embedded_train_config:
        raise ValueError("model-init checkpoint must contain a complete train config")

    source = {
        "schema": MODEL_STATE_INITIALIZATION_SCHEMA,
        "mode": "model_state_only_checkpoint",
        "load_mode": load_mode,
        "verified_materialization": "private_temporary_file_same_descriptor_hash_and_deserialize",
        "path": str(path),
        "sha256": observed_sha256,
        "size_bytes": observed_size_bytes,
        "source_step": source_step,
        "source_train_tokens": source_train_tokens,
        "source_train_valid_tokens": source_train_valid_tokens,
        "embedded_model_config_sha256": _canonical_sha256(dict(embedded_model_config)),
        "embedded_train_config_sha256": _canonical_sha256(dict(embedded_train_config)),
        "identity_seconds": identity_seconds,
        "restored_model_state": True,
        "restored_optimizer_state": False,
        "restored_scaler_state": False,
        "restored_progress_counters": False,
        "restored_rng_state": False,
        "restored_loader_state": False,
        "retained_seeded_token_edit_extension_state": bool(missing_keys),
        "token_edit_extension_tensor_names": missing_keys,
        "token_edit_extension_state_sha256": extension_state_sha256,
    }
    if parent_requirement is not None:
        source["parent_profile"] = parent_requirement.name
        source["parent_profile_binding"] = parent_binding
    if source_parameterization is not None:
        source["source_parameterization_contract_sha256"] = source_parameterization["contract_sha256"]
    if allow_block_self_conditioning:
        source["retained_seeded_token_edit_extension_state"] = False
        source["token_edit_extension_tensor_names"] = []
        source["token_edit_extension_state_sha256"] = ""
        source["block_self_conditioning_tensor_names"] = missing_keys
        source["block_self_conditioning_state_sha256"] = extension_state_sha256
    del state_dict
    del checkpoint
    return source
