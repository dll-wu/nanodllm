#!/usr/bin/env python3
"""Emit the exact parameter/FLOP fingerprint for a launchable config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanodllm.config import load_config  # noqa: E402
from nanodllm.model import DiffusionConfig, DiffusionTransformer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment JSON config")
    parser.add_argument("--vocab-size", required=True, type=int, help="Unpadded tokenizer vocabulary size")
    parser.add_argument("--mask-token-id", type=int, default=0, help="Only recorded in model config; default: 0")
    parser.add_argument("--output", help="Optional JSON output path; stdout is always written")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = DiffusionConfig(
        sequence_len=int(config.model.max_seq_len),
        vocab_size=int(args.vocab_size),
        n_layer=int(config.model.depth),
        n_head=int(config.model.n_head),
        n_embd=int(config.model.n_embd),
        mask_token_id=int(args.mask_token_id),
        mlp_hidden_dim=int(config.model.mlp_hidden_dim),
        attention_scale=float(config.model.attention_scale),
        local_attention_lower_layers=int(config.model.local_attention_lower_layers),
        local_attention_window=int(config.model.local_attention_window),
        auxiliary_head_layers=tuple(int(i) for i in config.model.auxiliary_head_layers),
        time_conditioning=bool(config.noise.time_conditioning),
        bd3lm_block_size=int(config.model.bd3lm_block_size),
        output_factorization=config.model.output_factorization,
        causal=bool(config.model.causal),
        residual_output_projection_init=config.model.residual_output_projection_init,
        embedding_head_init_std_multiplier=float(config.model.embedding_head_init_std_multiplier),
        length_head=bool(config.model.length_head),
        target_length_conditioning=bool(config.model.target_length_conditioning),
        mask_position_embedding=bool(config.model.mask_position_embedding),
    )
    if model_config.output_factorization != "flat":
        raise ValueError("offline fingerprint utility currently requires model.output_factorization='flat'")
    with torch.device("meta"):
        model = DiffusionTransformer(
            model_config,
            rope_base=float(config.model.rope_base),
            compute_dtype=torch.float32,
            pad_vocab_to=int(config.model.pad_vocab_to),
        )
    record = model.architecture_fingerprint()
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
