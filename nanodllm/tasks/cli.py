"""Run corrected block CORE tasks on local data or compare two scored panels."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..evaluation.block_checkpoint import identity, load_block, numeric_runtime
from ..evaluation.checkpoint import load_tokenizer, tokenizer_identity
from .aggregate import aggregate, paired
from .data import check_study_tokenizer, is_study_roster, prepare_panel, read_roster, study_roster
from .scoring import OPTIONS, score_panel


def score(args):
    specs = read_roster(args.roster)
    labels = {s["label"] for s in specs}
    if args.task and not set(args.task) <= labels:
        raise ValueError("unrecognized task selection")
    loaded = load_block(
        args.checkpoint,
        args.tokenizer,
        args.device,
        sha256=args.checkpoint_sha256,
        tokenizer_sha256=args.tokenizer_sha256,
        training_checkpoint=args.training_checkpoint,
        data_identity_path=args.data_identity,
    )
    model = identity(loaded)
    paper = is_study_roster(args.roster)
    check_study_tokenizer(args.roster, model["tokenizer_content_sha256"])
    block = loaded.adapter.block_size
    context = min(1024, loaded.model.config.sequence_len)
    if paper and (context != 1024 or block != 32):
        raise ValueError("paper task roster requires BL32 and context 1024")
    tasks = {}
    with numeric_runtime(args.device):
        for spec in specs:
            if args.task and spec["label"] not in args.task:
                continue
            panel = prepare_panel(
                spec, args.bundle, loaded.tokenizer, block_size=block, context=context, limit=args.limit
            )
            tasks[spec["label"]] = score_panel(loaded.adapter, panel)
    return {
        "schema": "nanodllm.paper_tasks.v1",
        "model": model,
        "full_paper_panel": paper and len(tasks) == 21 and sum(len(t["rows"]) for t in tasks.values()) == 8845,
        "settings": {
            "task_order": list(tasks),
            "context": context,
            "block_size": block,
            "device": args.device,
            "dtype": "bf16" if args.device == "cuda" else "fp32",
            "device_batch": 8,
            "max_per_task": 500,
            "estimator": asdict(OPTIONS),
        },
        "primary": "mean_nelbo__block_full_mask",
        "aggregates": aggregate(tasks),
        "tasks": tasks,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Verify every selected example and demonstration without loading weights")
    audit.add_argument("--bundle", type=Path, required=True)
    audit.add_argument("--roster", type=Path, help="Local roster JSON (default: frozen CORE21 roster)")
    audit.add_argument("--tokenizer", type=Path)
    s = sub.add_parser("score")
    s.add_argument("--bundle", type=Path, required=True)
    s.add_argument("--roster", type=Path, help="Local roster JSON (default: frozen CORE21 roster)")
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument(
        "--training-checkpoint",
        action="store_true",
        help="Read the native training format for a trusted local checkpoint",
    )
    s.add_argument(
        "--data-identity", type=Path, help="Tokenizer identity sidecar for an explicitly trusted training checkpoint"
    )
    s.add_argument("--checkpoint-sha256")
    s.add_argument("--tokenizer-sha256")
    s.add_argument("--tokenizer", type=Path, help="Local HF tokenizer directory (omit for byte checkpoints)")
    s.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    s.add_argument("--task", action="append", help="Repeat to select tasks")
    s.add_argument("--limit", type=int, help="Limit scored rows per task after selecting the full demonstration pool")
    c = sub.add_parser("compare", help="Compute right minus left, in percentage points")
    c.add_argument("left", type=Path)
    c.add_argument("right", type=Path)
    c.add_argument("--draws", type=int, default=10000)
    c.add_argument("--seed", type=int)
    for cmd in (s, c):
        cmd.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command in ("audit", "score") and args.roster is None:
        args.roster = study_roster()
    if args.command == "audit":
        tokenizer = load_tokenizer(args.tokenizer)
        check_study_tokenizer(args.roster, tokenizer_identity(tokenizer))
        counts = {}
        for spec in read_roster(args.roster):
            panel = prepare_panel(spec, args.bundle, tokenizer)
            counts[spec["label"]] = len(panel.examples)
        print(json.dumps({"tasks": len(counts), "items": sum(counts.values()), "per_task": counts}))
        return 0
    result = (
        score(args)
        if args.command == "score"
        else paired(
            json.loads(args.left.read_text()), json.loads(args.right.read_text()), draws=args.draws, seed=args.seed
        )
    )
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result.get("aggregates", result.get("results")), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
