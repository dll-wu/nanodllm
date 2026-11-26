"""Generate continuations with the native causal-prefix KV cache."""

import argparse
import json
from pathlib import Path

from ..evaluation.block_checkpoint import load_block
from .panel import study_prompts
from .reference import score_outputs
from .sampling import POLICIES, Decoder, generate_panel, read_prompts

SEEDS = (2026110201, 2026110202)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--training-checkpoint",
        action="store_true",
        help="Read the native training format for a trusted local checkpoint",
    )
    p.add_argument(
        "--data-identity", type=Path, help="Tokenizer identity sidecar for an explicitly trusted training checkpoint"
    )
    p.add_argument("--checkpoint-sha256")
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--tokenizer-sha256")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--study-parquet",
        type=Path,
        help="Reconstruct the 32 pinned generation prefixes from the local validation shard",
    )
    group.add_argument("--prompts", type=Path, help="Local JSONL rows with id and text or prefix_ids")
    group.add_argument("--prompt", action="append", help="One literal prompt; repeatable")
    p.add_argument("--seed", type=int, action="append", help="Proposal seed; repeatable")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--policy", choices=POLICIES, default="sample-redraw")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--capacity", type=int, help="Maximum generated tokens (default up to 256 for custom prompts)")
    p.add_argument("--block-length", type=int, help="Default: checkpoint-native block length")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--reference-model", type=Path, help="Local causal HF model and tokenizer directory")
    p.add_argument("--output", type=Path, help="Write the JSON report here (default: print the report to stdout)")
    args = p.parse_args(argv)
    model = load_block(
        args.checkpoint,
        args.tokenizer,
        args.device,
        generation=True,
        sha256=args.checkpoint_sha256,
        tokenizer_sha256=args.tokenizer_sha256,
        training_checkpoint=args.training_checkpoint,
        data_identity_path=args.data_identity,
    )
    if args.study_parquet:
        prompts = study_prompts(args.study_parquet, model.tokenizer)
    elif args.prompts:
        with args.prompts.open(encoding="utf-8") as f:
            prompts = read_prompts((json.loads(line) for line in f if line.strip()), model.tokenizer)
    else:
        prompts = read_prompts(({"id": f"prompt-{i}", "text": v} for i, v in enumerate(args.prompt)), model.tokenizer)
    capacity = args.capacity
    if capacity is None:
        capacity = 256
        if args.study_parquet is None:
            available = model.model.config.sequence_len - max(len(row["prefix_ids"]) for row in prompts)
            if available < 1:
                raise ValueError("prompt leaves no room for generation in the trained context")
            capacity = min(capacity, available)
    decoder = Decoder(
        args.k,
        args.policy,
        args.temperature,
        args.top_p,
        capacity,
        model.adapter.block_size if args.block_length is None else args.block_length,
    )
    result = generate_panel(model, prompts, decoder, args.seed or SEEDS, args.device)
    if args.reference_model:
        prefixes = {p["id"]: model.tokenizer.decode(p["prefix_ids"]) for p in prompts}
        del model
        result["reference"] = score_outputs(result["samples"], prefixes, args.reference_model, args.device)
    if args.output is None:
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("summary", "reference") if k in result}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
