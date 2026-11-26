"""Stable, per-example random streams used by the paper decoders."""

import hashlib
import json

import torch


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def keyed_seed(example_id: str, seed: int, stream: str) -> int:
    """Stable across worker, arm, shard ordering and Python hash randomization."""
    if not isinstance(example_id, str) or not example_id or type(seed) is not int:
        raise ValueError("RNG keys need an example identity and integer seed")
    return int(digest(["replication-evaluation-v1", example_id, seed, stream])[:15], 16)


def generator_for(example_id, seed, stream, *, device="cpu"):
    return torch.Generator(device=device).manual_seed(keyed_seed(example_id, seed, stream))


def device_of(model):
    return next(model.parameters()).device
