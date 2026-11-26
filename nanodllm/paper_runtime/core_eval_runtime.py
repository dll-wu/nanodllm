"""Local checkpoint/tokenizer loading and family-checked AR inference for CORE diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint_load_binding import load_checkpoint_with_optional_binding
from .config import ExperimentConfig, config_from_dict
from .core_ar_scoring import ARExampleScore, padded_inputs, score_ar_target_logits
from .core_diffusion_adapters import DiffusionContinuationAdapter
from .core_prompting import PreparedExample
from .core_task_data import json_digest
from .data import ByteTokenizer, PretrainedHFTokenizer, make_tokenizer
from .data_identity import build_tokenizer_inventory
from .model import DiffusionTransformer
from .training_block_conversion import block_conversion_training_block_size
from .training_block_recipe import recipe_phase
from .training_plan import _resolve_diffusion_config
from .training_runtime import dtype_from_name


def load_local_tokenizer(
    kind: str, path: str | None, expected_inventory_sha256: str
) -> tuple[ByteTokenizer | PretrainedHFTokenizer, dict]:
    before = build_tokenizer_inventory(kind, path)
    if before["inventory_sha256"] != expected_inventory_sha256:
        raise ValueError("tokenizer inventory differs from the explicit task manifest")
    tokenizer = make_tokenizer(kind, path)
    after = build_tokenizer_inventory(kind, path)
    if after != before:
        raise ValueError("tokenizer changed during loading")
    return tokenizer, before


@dataclass(frozen=True)
class LoadedCoreModel:
    model: DiffusionTransformer
    config: ExperimentConfig
    tokenizer: ByteTokenizer | PretrainedHFTokenizer
    tokenizer_inventory: dict
    checkpoint_receipt: dict
    adapter: DiffusionContinuationAdapter | None


def load_core_model(
    checkpoint_path: str | Path,
    *,
    expected_tokenizer_inventory_sha256: str,
    scorer: str,
    device: str = "cpu",
    dtype: str = "float32",
    tokenizer_path: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> LoadedCoreModel:
    """Load a user-selected, trusted local NanoDLLM checkpoint, never a hub ID.

    Like the existing checkpoint tools this loads the full training pickle. A file
    hash binds bytes, not trust; do not pass downloaded/untrusted checkpoints here.
    """
    if scorer not in {"ar", "conditional_nelbo"}:
        raise ValueError("unsupported scorer")
    if device not in {"cpu", "cuda"} or dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("unsupported numeric runtime")
    if device == "cpu" and dtype != "float32":
        raise ValueError("the CPU reference runner uses float32")
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    with checkpoint_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected_checkpoint_sha256 is not None and digest != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 differs from the requested binding")
    size = checkpoint_path.stat().st_size
    receipt: dict = {}
    checkpoint = load_checkpoint_with_optional_binding(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        expected_sha256=digest,
        expected_size_bytes=size,
        receipt=receipt,
    )
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("train_config"), dict):
        raise ValueError("checkpoint must contain train_config")
    raw_config = checkpoint["train_config"]
    if not all(isinstance(raw_config.get(key), dict) for key in ("model", "data", "noise")):
        raise ValueError("checkpoint must declare its model, tokenizer and noise configuration")
    config = config_from_dict(raw_config)
    model_config = config.model
    if model_config.block_causal_pilot_size:
        raise ValueError("the historical single-stream block-causal pilot needs its own attention adapter")
    if (
        model_config.output_factorization != "flat"
        or model_config.target_length_conditioning
        or config.noise.time_conditioning
    ):
        raise ValueError("this runner requires time-independent flat logits without target-length conditioning")
    if scorer == "ar" and (not model_config.causal or model_config.bd3lm_block_size != 0):
        raise ValueError("AR scoring requires a causal, non-block checkpoint")
    if scorer == "conditional_nelbo" and (model_config.causal or config.noise.type not in {"mdlm", "bd3lm"}):
        raise ValueError("conditional NELBO requires an absorbing, non-AR checkpoint")
    resolved_path = tokenizer_path if tokenizer_path is not None else config.data.tokenizer_path
    tokenizer, inventory = load_local_tokenizer(
        config.data.tokenizer, resolved_path, expected_tokenizer_inventory_sha256
    )
    model = DiffusionTransformer(
        _resolve_diffusion_config(config, tokenizer),
        rope_base=config.model.rope_base,
        compute_dtype=dtype_from_name(dtype),
        pad_vocab_to=config.model.pad_vocab_to,
    ).to(device)
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("checkpoint must contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    adapter = (
        None
        if scorer == "ar"
        else DiffusionContinuationAdapter(
            model,
            noise_type=config.noise.type,
            clean_stream_mode=config.model.bd3lm_clean_stream_mode,
            device=device,
        )
    )
    receipt.update(
        {
            "checkpoint_sha256": digest,
            "checkpoint_size_bytes": size,
            "train_config_sha256": json_digest(raw_config),
            "checkpoint_step": checkpoint.get("step"),
            "checkpoint_train_tokens": checkpoint.get("train_tokens"),
            "declared_tokenizer_path": config.data.tokenizer_path,
            "architecture_fingerprint": model.architecture_fingerprint(),
        }
    )
    if config.noise.type == "bd3lm":
        step = checkpoint.get("step")
        if config.train.bd3lm_block_size_schedule == "stepwise" and (type(step) is not int or step < 0):
            raise ValueError("a curriculum checkpoint must record its completed-step count")
        receipt["training_block_size_schedule"] = config.train.bd3lm_block_size_schedule
        receipt["last_training_block_size"] = (
            block_conversion_training_block_size(config, step - 1) if type(step) is int and step > 0 else None
        )
        receipt["evaluation_block_size"] = config.model.bd3lm_block_size
        if config.train.bd3lm_partition_mode != "fixed":
            receipt["last_training_block_size"] = None
            receipt["training_partition"] = {
                "mode": config.train.bd3lm_partition_mode,
                "lengths": config.train.bd3lm_partition_lengths,
                "seed": config.train.bd3lm_partition_seed,
                "scalar_block_size_is_reference_only": True,
            }
        if config.train.bd3lm_recipe_mechanism != "none" and type(step) is int and step > 0:
            receipt["last_training_block_size"] = recipe_phase(
                config, step - 1, receipt["last_training_block_size"]
            ).block_size
            receipt["block_recipe_mechanism"] = config.train.bd3lm_recipe_mechanism
    return LoadedCoreModel(model, config, tokenizer, inventory, receipt, adapter)


@torch.inference_mode()
def score_ar_model(model: DiffusionTransformer, example: PreparedExample, *, device: str = "cpu") -> ARExampleScore:
    config = model.config
    if model.training or not config.causal or config.bd3lm_block_size:
        raise ValueError("AR inference requires an eval-mode causal non-block model")
    if config.time_conditioning or config.target_length_conditioning or config.output_factorization != "flat":
        raise ValueError("unsupported AR conditioning/output contract")
    inputs = padded_inputs(example, device=device)
    if inputs.shape[1] > config.sequence_len or int(inputs.max()) >= config.vocab_size:
        raise ValueError("AR input exceeds the model's sequence/vocabulary capacity")
    if (inputs == config.mask_token_id).any():
        raise ValueError("AR clean input contains the reserved MASK token")
    valid = torch.zeros_like(inputs, dtype=torch.bool)
    indices = []
    lengths = []
    for row, sequence in enumerate(example.sequences):
        valid[row, : len(sequence.tokens)] = True
        indices.extend(row * inputs.shape[1] + j for j in range(sequence.start - 1, sequence.end - 1))
        lengths.append(sequence.end - sequence.start)
    logits = model(
        inputs,
        logits_indices=torch.tensor(indices, device=device),
        attention_mask=valid,
        position_ids=torch.arange(inputs.shape[1], device=device),
    )
    return score_ar_target_logits(example, tuple(logits.split(lengths)))
