"""Generate continuations with the native causal-prefix KV cache."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .paper_generation_panel import study_prompts
from .paper_generation_reference import score_outputs
from .paper_runtime.block_recipe_sampling import generate
from .paper_runtime.core_task_data import json_digest
from .paper_tasks_model import identity, load_block, numeric_runtime

SEEDS = (2026110201, 2026110202)
POLICIES = {"left-to-right": "left_to_right", "sample-keep": "confidence", "sample-redraw": "confidence_resample"}


@dataclass(frozen=True)
class Decoder:
    k: int = 4
    policy: str = "sample-redraw"
    temperature: float = 1.0
    top_p: float = 0.9
    capacity: int = 256
    block_length: int = 32

    def point(self):
        if min(self.k, self.capacity, self.block_length) < 1 or not 0 < self.top_p <= 1:
            raise ValueError("expected positive K, capacity and block length and top-p in (0,1]")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        return {"k": self.k, "policy": POLICIES[self.policy], "temperature": self.temperature, "top_p": self.top_p}


def read_prompts(rows, tokenizer):
    prompts = []
    seen = set()
    for row in rows:
        key = row["id"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("each prompt must have a unique nonempty string id")
        seen.add(key)
        if ("text" in row) == ("prefix_ids" in row):
            raise ValueError("supply either text or the complete prefix_ids for each prompt")
        if "text" in row:
            if not isinstance(row["text"], str):
                raise ValueError("prompt text must be a string")
            ids = [tokenizer.get_bos_token_id(), *tokenizer.encode(row["text"])]
        else:
            ids = row["prefix_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or not 0 <= i < tokenizer.get_vocab_size() for i in ids)
        ):
            raise ValueError("invalid prompt IDs")
        if ids[0] != tokenizer.get_bos_token_id() or set(ids[1:]) & tokenizer.get_special_token_ids():
            raise ValueError("prefix requires one document-start BOS followed by ordinary content")
        prompts.append({"id": key, "prefix_ids": ids})
    if not prompts:
        raise ValueError("empty prompt panel")
    return prompts


def generate_panel(loaded, prompts, decoder, seeds, device):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("supply distinct proposal seeds")
    point = decoder.point()
    if any(len(row["prefix_ids"]) + decoder.capacity > loaded.model.config.sequence_len for row in prompts):
        raise ValueError("prompt plus capacity exceeds the trained context")
    rows = []
    with numeric_runtime(device):
        for prompt in prompts:
            for seed in seeds:
                result = generate(
                    loaded.model,
                    loaded.tokenizer,
                    prompt,
                    point,
                    seed,
                    capacity=decoder.capacity,
                    block_length=decoder.block_length,
                )
                work = result["work"]
                rows.append(
                    {
                        "prompt_id": prompt["id"],
                        "seed": seed,
                        "text": result["text"],
                        "content_tokens": result["content_tokens"],
                        "eos_present": result["eos_present"],
                        "capacity_hit": result["capacity_hit"],
                        "repeated_4_fraction": result["repeated_4_fraction"],
                        "work": work,
                        "tokens_per_forward": result["content_tokens"] / work["model_forwards"],
                        "generation_seconds": result["elapsed_seconds"],
                        "tokens_per_second": result["content_tokens_per_second"],
                        "allocated_kv_bytes": result["allocated_kv_bytes"],
                    }
                )
    repeats = [r["repeated_4_fraction"] for r in rows if r["repeated_4_fraction"] is not None]
    return {
        "schema": "nanodllm.paper_generation.v1",
        "model": identity(loaded),
        "device": device,
        "model_compute_dtype": "float32",
        "forward_autocast": "bfloat16" if device == "cuda" else None,
        "decoder": {**point, "block_length": decoder.block_length, "capacity": decoder.capacity},
        "seeds": list(seeds),
        "prompts_sha256": json_digest(prompts),
        "summary": {
            "prompts": len(prompts),
            "samples": len(rows),
            "eos_count": sum(r["eos_present"] for r in rows),
            "capacity_hits": sum(r["capacity_hit"] for r in rows),
            "repeat4_defined": len(repeats),
            "mean_repeat4": mean(repeats) if repeats else None,
            "mean_content_tokens": mean(r["content_tokens"] for r in rows),
            "mean_tokens_per_forward": mean(r["tokens_per_forward"] for r in rows),
            "mean_tokens_per_second": mean(r["tokens_per_second"] for r in rows),
            "mean_seconds": mean(r["generation_seconds"] for r in rows),
        },
        "samples": rows,
    }


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
    p.add_argument("--capacity", type=int, default=256)
    p.add_argument("--block-length", type=int, help="Default: checkpoint-native block length")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--reference-model", type=Path, help="Local causal HF model and tokenizer directory")
    p.add_argument("--output", type=Path, required=True)
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
    decoder = Decoder(
        args.k,
        args.policy,
        args.temperature,
        args.top_p,
        args.capacity,
        model.adapter.block_size if args.block_length is None else args.block_length,
    )
    result = generate_panel(model, prompts, decoder, args.seed or SEEDS, args.device)
    if args.reference_model:
        prefixes = {p["id"]: model.tokenizer.decode(p["prefix_ids"]) for p in prompts}
        del model
        result["reference"] = score_outputs(result["samples"], prefixes, args.reference_model, args.device)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("summary", "reference") if k in result}, allow_nan=False))
    return 0
