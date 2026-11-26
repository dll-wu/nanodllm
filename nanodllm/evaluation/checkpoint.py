"""Load a local native transformer checkpoint and its matching tokenizer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ..runtime.data import ByteTokenizer, PretrainedHFTokenizer
from ..runtime.model import DiffusionConfig, DiffusionTransformer
from ..runtime.replication_eval_support import digest

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer_ordinary.json",
    "token_bytes.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
)


def stream_sha256(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def tokenizer_identity(tokenizer):
    if isinstance(tokenizer, ByteTokenizer):
        return digest("nanodllm.byte-tokenizer.v1")
    root = Path(tokenizer._tokenizer_path)
    files = {}
    # Identity covers model inputs and the denominator. Exclude corpus inventories,
    # machine-specific paths, timestamps and training receipts.
    for name in TOKENIZER_FILES:
        if (root / name).is_file():
            with (root / name).open("rb") as stream:
                files[name] = stream_sha256(stream)
    return digest(files)


def training_tokenizer_sha256(checkpoint_path, data_identity_path=None):
    """Recover a path-independent tokenizer binding from a trainer v1 or v2 data identity."""
    if data_identity_path is None:
        parent = Path(checkpoint_path).parent
        data_identity_path = next(
            (p for p in (parent / "data_identity.v2.json", parent / "data_identity.v1.json") if p.is_file()), None
        )
        if data_identity_path is None:
            raise ValueError("HF training checkpoint needs its data identity sidecar; pass --data-identity")
    data = json.loads(Path(data_identity_path).read_text())
    inventory = data.get("identity", data)["tokenizer"]
    if inventory["kind"] != "hf":
        raise ValueError("expected HF tokenizer in training data identity")
    return digest(
        {f["relative_path"]: f["sha256"] for f in inventory["files"] if f["relative_path"] in TOKENIZER_FILES}
    )


def load_tokenizer(tokenizer_path=None):
    if tokenizer_path is None:
        return ByteTokenizer()
    root = Path(tokenizer_path).resolve(strict=True)
    if not (root / "tokenizer.json").is_file() or not (root / "tokenizer_config.json").is_file():
        raise ValueError("expected a local tokenizer directory with tokenizer.json and tokenizer_config.json")
    if not (root / "token_bytes.json").is_file():
        raise ValueError("a local token_bytes.json is required for reproducible byte denominators")
    return PretrainedHFTokenizer.from_pretrained(root)


@dataclass(frozen=True)
class Checkpoint:
    model: DiffusionTransformer
    tokenizer: ByteTokenizer | PretrainedHFTokenizer
    checkpoint_sha256: str
    tokenizer_sha256: str
    clean_stream_mode: str
    noise_type: str


def load_checkpoint(
    path,
    *,
    tokenizer_path=None,
    device="cpu",
    dtype="float32",
    expected_sha256=None,
    trusted_training_checkpoint=False,
    data_identity_path=None,
):
    """Load a portable weights file, or explicitly opt into a trusted local training pickle."""
    if not trusted_training_checkpoint:
        from ..weights import load_weights

        if data_identity_path is not None:
            raise ValueError("data identity sidecars apply only to trusted training checkpoints")

        loaded = load_weights(
            path, tokenizer_path=tokenizer_path, expected_checkpoint_sha256=expected_sha256, device=device, dtype=dtype
        )
        return Checkpoint(
            loaded.model,
            loaded.tokenizer,
            loaded.checkpoint_receipt["checkpoint_sha256"],
            tokenizer_identity(loaded.tokenizer),
            loaded.config.model.bd3lm_clean_stream_mode,
            loaded.config.noise.type,
        )
    if dtype not in {"float32", "bfloat16"} or torch.device(device).type not in {"cpu", "cuda"}:
        raise ValueError("expected cpu/cuda and float32/bfloat16")
    if dtype == "bfloat16" and torch.device(device).type != "cuda":
        raise ValueError("CPU reference scoring uses float32")
    with Path(path).open("rb") as stream:
        identity = stream_sha256(stream)
        if expected_sha256 is not None and identity != expected_sha256:
            raise ValueError("checkpoint SHA-256 mismatch")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("expected a native checkpoint with model_state_dict")
    # Some block study checkpoints use the explicit model_config/config split.
    if "model_config" in payload:
        architecture, experiment = payload["model_config"], payload["config"]
    else:
        architecture, experiment = payload["config"], payload["train_config"]
    config = DiffusionConfig(**architecture)
    kind = experiment["data"]["tokenizer"]
    if kind not in {"byte", "hf"} or (kind == "hf") != (tokenizer_path is not None):
        raise ValueError("HF checkpoints require an explicit local tokenizer directory; byte checkpoints take none")
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_digest = tokenizer_identity(tokenizer)
    if kind == "hf" and training_tokenizer_sha256(path, data_identity_path) != tokenizer_digest:
        raise ValueError("tokenizer contents differ from training data identity")
    if (config.vocab_size, config.mask_token_id) != (tokenizer.get_vocab_size(), tokenizer.get_mask_token_id()):
        raise ValueError("checkpoint and tokenizer vocabulary or MASK ID differ")
    model_settings = experiment["model"]
    model = DiffusionTransformer(
        config,
        rope_base=model_settings["rope_base"],
        compute_dtype=getattr(torch, dtype),
        pad_vocab_to=model_settings["pad_vocab_to"],
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return Checkpoint(
        model.to(device).eval(),
        tokenizer,
        identity,
        tokenizer_digest,
        model_settings["bd3lm_clean_stream_mode"] if config.bd3lm_block_size else "causal",
        experiment["noise"]["type"],
    )
