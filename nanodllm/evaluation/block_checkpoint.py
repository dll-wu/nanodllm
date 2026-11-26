"""Shared checkpoint admission and numerical settings for the two block readouts."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ..runtime.core_diffusion_adapters import DiffusionContinuationAdapter
from .checkpoint import Checkpoint, load_checkpoint


@dataclass(frozen=True)
class BlockCheckpoint:
    checkpoint: Checkpoint
    adapter: DiffusionContinuationAdapter

    @property
    def model(self):
        return self.checkpoint.model

    @property
    def tokenizer(self):
        return self.checkpoint.tokenizer


def load_block(
    path,
    tokenizer_dir,
    device,
    *,
    generation=False,
    sha256=None,
    tokenizer_sha256=None,
    training_checkpoint=False,
    data_identity_path=None,
):
    dtype = "bfloat16" if device == "cuda" and not generation else "float32"
    ckpt = load_checkpoint(
        path,
        tokenizer_path=tokenizer_dir,
        device=device,
        dtype=dtype,
        expected_sha256=sha256,
        trusted_training_checkpoint=training_checkpoint,
        data_identity_path=data_identity_path,
    )
    if tokenizer_sha256 is not None and tokenizer_sha256 != ckpt.tokenizer_sha256:
        raise ValueError("tokenizer content SHA-256 mismatch")
    if ckpt.noise_type != "bd3lm" or ckpt.clean_stream_mode != "causal":
        raise ValueError("these block readouts require a native causal-clean block checkpoint")
    adapter = DiffusionContinuationAdapter(
        ckpt.model, noise_type=ckpt.noise_type, clean_stream_mode=ckpt.clean_stream_mode, device=device
    )
    return BlockCheckpoint(ckpt, adapter)


def identity(loaded):
    return {
        "checkpoint_sha256": loaded.checkpoint.checkpoint_sha256,
        "tokenizer_content_sha256": loaded.checkpoint.tokenizer_sha256,
        "depth": loaded.model.config.n_layer,
        "native_block_size": loaded.adapter.block_size,
    }


def numeric_runtime(device):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
