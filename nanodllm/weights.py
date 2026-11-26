"""Export a portable tensor checkpoint from an explicitly trusted trainer file."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from .evaluation.checkpoint import training_tokenizer_sha256
from .runtime.checkpoint_load_binding import load_checkpoint_with_optional_binding
from .runtime.config import DataConfig, ExperimentConfig, ModelConfig, NoiseConfig, TrainConfig
from .runtime.core_diffusion_adapters import DiffusionContinuationAdapter
from .runtime.core_eval_runtime import LoadedCoreModel
from .runtime.model import DiffusionConfig, DiffusionTransformer
from .runtime.nanollada_completep import MDLM_RULE_SET, validate_model_initialization
from .runtime.recipe_lineage import native_profile_receipt

SCHEMA = "nanodllm.weights.v1"
BLOCK_FIELDS = (
    "bd3lm_block_size_schedule",
    "bd3lm_block_size_schedule_start_step_indices",
    "bd3lm_block_size_schedule_values",
    "bd3lm_partition_mode",
    "bd3lm_partition_lengths",
    "bd3lm_partition_seed",
    "bd3lm_recipe_mechanism",
    "bd3lm_recipe_start_step",
)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_file(path, expected_sha256, *, trusted=False):
    path = Path(path).resolve(strict=True)
    identity = expected_sha256 if expected_sha256 is not None else file_sha256(path)
    payload = load_checkpoint_with_optional_binding(
        path,
        map_location="cpu",
        weights_only=not trusted,
        expected_sha256=identity,
        expected_size_bytes=path.stat().st_size,
        environ={},
    )
    return payload, identity


def _tokenizer(kind, tokenizer_path):
    from .evaluation.checkpoint import load_tokenizer, tokenizer_identity

    if kind not in {"byte", "hf"} or (kind == "hf") != (tokenizer_path is not None):
        raise ValueError("HF weights require a local tokenizer directory; byte weights use the built-in tokenizer")
    tokenizer = load_tokenizer(tokenizer_path)
    return tokenizer, tokenizer_identity(tokenizer)


def _check_architecture(architecture, model_settings, tokenizer):
    if (architecture.vocab_size, architecture.mask_token_id) != (
        tokenizer.get_vocab_size(),
        tokenizer.get_mask_token_id(),
    ):
        raise ValueError("tokenizer vocabulary or mask ID differs from the checkpoint")
    for field, value in (
        ("n_layer", model_settings.depth),
        ("n_embd", model_settings.n_embd),
        ("n_head", model_settings.n_head),
        ("sequence_len", model_settings.max_seq_len),
        ("bd3lm_block_size", model_settings.bd3lm_block_size),
        ("causal", model_settings.causal),
        ("backbone_profile", model_settings.backbone_profile),
    ):
        if getattr(architecture, field) != value:
            raise ValueError(f"checkpoint model configuration disagrees on {field}")


def _counters(source):
    out = {}
    for key in ("step", "train_tokens", "train_valid_tokens"):
        value = source[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid checkpoint {key}")
        out[key] = value
    return out


def export_weights(trusted_checkpoint, output, *, tokenizer_path=None, expected_source_sha256, data_identity_path=None):
    """Read a trusted local training pickle, strip training state and publish atomically."""
    if not expected_source_sha256:
        raise ValueError("expected_source_sha256 must bind the trusted training file")
    target = Path(output)
    if not target.name.endswith(".weights.pt"):
        raise ValueError("use a .weights.pt filename for portable weights")
    if target.exists():
        raise FileExistsError(target)
    source, source_digest = _read_file(trusted_checkpoint, expected_source_sha256, trusted=True)
    if source.get("parameterization_status"):
        raise ValueError("cannot export an unbound diagnostic checkpoint")
    # Historical block checkpoints use a different name for the resolved model.
    if "model_config" in source:
        architecture, settings = DiffusionConfig(**source["model_config"]), source["config"]
    else:
        architecture, settings = DiffusionConfig(**source["config"]), source["train_config"]
    model_config = ModelConfig(**settings["model"])
    noise = NoiseConfig(**settings["noise"])
    kind = settings["data"]["tokenizer"]
    tokenizer, token_identity = _tokenizer(kind, tokenizer_path)
    _check_architecture(architecture, model_config, tokenizer)
    if (
        kind == "hf"
        and training_tokenizer_sha256(trusted_checkpoint, data_identity_path, checkpoint=source) != token_identity
    ):
        raise ValueError("local tokenizer contents differ from the training data identity")
    if not isinstance(source["model_state_dict"], dict) or not all(
        type(name) is str and type(value) is torch.Tensor for name, value in source["model_state_dict"].items()
    ):
        raise ValueError("model state must be a dictionary of named tensors")
    payload = {
        "schema": SCHEMA,
        "source_sha256": source_digest,
        "tokenizer_kind": kind,
        "tokenizer_sha256": token_identity,
        "config": asdict(architecture),
        "model": asdict(model_config),
        "noise": asdict(noise),
        "block_training": {key: settings["train"][key] for key in BLOCK_FIELDS if key in settings["train"]},
        "model_state_dict": source["model_state_dict"],
        **_counters(source),
    }
    receipt = native_profile_receipt(source, settings)
    if receipt is not None:
        payload["profile_receipt"] = receipt
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".weights-", suffix=".tmp", delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        torch.save(payload, staging)
        # Hard-link creation is atomic and refuses to overwrite an existing file.
        os.link(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return file_sha256(target)


def portable_parent(payload, *, target_model, expected_tokenizer_sha256):
    """Map an exported parent to the training initializer after verifying its forward conventions."""
    if payload.get("schema") != SCHEMA or payload.get("tokenizer_sha256") != expected_tokenizer_sha256:
        raise ValueError("portable parent schema or tokenizer identity differs from the child")
    source = DiffusionConfig(**payload["config"])
    # AR to block changes the causal mask and block topology, without changing
    # the interpretation of the embedding, residual, MLP or readout weights.
    allowed = {"causal", "bd3lm_block_size"}
    target = asdict(target_model.config)
    comparable = asdict(source)
    if target_model.config.parameterization_rule_set == MDLM_RULE_SET:
        # The two objectives carry distinct rule names and hashes. The
        # native validator checks their actual depth/width forward factors.
        validate_model_initialization(
            {"config": payload["config"], "train_config": {"model": payload["model"]}}, target_model
        )
    elif any(target[key] != value for key, value in comparable.items() if key not in allowed):
        raise ValueError("portable parent differs from the child forward or parameterization conventions")
    if payload["model"]["rope_base"] != target_model.rope_base:
        raise ValueError("portable parent and child RoPE bases differ")
    return {
        "config": payload["config"],
        "train_config": {"model": payload["model"]},
        "model_state_dict": payload["model_state_dict"],
        **_counters(payload),
    }


def load_weights(path, *, tokenizer_path=None, expected_checkpoint_sha256=None, device="cpu", dtype="float32"):
    """Load a portable export with restricted tensor deserialization."""
    if torch.device(device).type not in {"cuda", "cpu"} or dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("unsupported device or precision")
    if torch.device(device).type == "cpu" and dtype != "float32":
        raise ValueError("CPU evaluations use float32")
    payload, identity = _read_file(path, expected_checkpoint_sha256)
    if payload.get("schema") != SCHEMA:
        raise ValueError("expected a nanodllm.weights.v1 export")
    tokenizer, token_identity = _tokenizer(payload["tokenizer_kind"], tokenizer_path)
    if payload["tokenizer_sha256"] != token_identity:
        raise ValueError("tokenizer contents differ from the exported weights")
    architecture = DiffusionConfig(**payload["config"])
    settings = ModelConfig(**payload["model"])
    _check_architecture(architecture, settings, tokenizer)
    if set(payload["block_training"]) - set(BLOCK_FIELDS):
        raise ValueError("unexpected block training field")
    config = ExperimentConfig(
        model=settings,
        noise=NoiseConfig(**payload["noise"]),
        data=DataConfig(tokenizer=payload["tokenizer_kind"], tokenizer_path=str(tokenizer_path or "")),
        train=TrainConfig(**payload["block_training"]),
    )
    model = DiffusionTransformer(
        architecture,
        rope_base=settings.rope_base,
        compute_dtype=getattr(torch, dtype),
        pad_vocab_to=settings.pad_vocab_to,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    adapter = (
        None
        if architecture.causal
        else DiffusionContinuationAdapter(
            model, noise_type=config.noise.type, clean_stream_mode=settings.bd3lm_clean_stream_mode, device=device
        )
    )
    counts = _counters(payload)
    inventory = {"kind": config.data.tokenizer, "content_sha256": token_identity, "inventory_sha256": token_identity}
    receipt = {
        "checkpoint_sha256": identity,
        "checkpoint_size_bytes": Path(path).stat().st_size,
        "source_sha256": payload["source_sha256"],
        "checkpoint_step": counts["step"],
        "checkpoint_train_tokens": counts["train_tokens"],
        "tokenizer_sha256": token_identity,
        "architecture_fingerprint": model.architecture_fingerprint(),
    }
    return LoadedCoreModel(model, config, tokenizer, inventory, receipt, adapter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-checkpoint", required=True, help="local training pickle you trust")
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", required=True, help="new .weights.pt file")
    parser.add_argument("--tokenizer-dir", help="local HF tokenizer directory")
    parser.add_argument("--data-identity", help="training data_identity.v1.json if stored separately")
    args = parser.parse_args()
    print(
        export_weights(
            args.trusted_checkpoint,
            args.output,
            tokenizer_path=args.tokenizer_dir,
            expected_source_sha256=args.source_sha256,
            data_identity_path=args.data_identity,
        )
    )


if __name__ == "__main__":
    main()
