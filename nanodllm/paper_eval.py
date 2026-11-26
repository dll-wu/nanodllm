"""Command line for local paper corpus panel preparation, checkpoint scoring, and paired comparison."""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .paper_eval_checkpoint import load_paper_checkpoint, load_tokenizer
from .paper_eval_compare import paired_comparison
from .paper_eval_panels import KINDS, load_panel, prepare
from .paper_eval_scores import score_panel


def save_result(path, value, *, binary=False):
    """Expose either a complete output file or no new result if serialization fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, mode="wb" if binary else "w", delete=False) as f:
        try:
            if binary:
                torch.save(value, f)
            else:
                json.dump(value, f, indent=2, allow_nan=False)
                f.write("\n")
            f.close()
            os.replace(f.name, path)
        finally:
            Path(f.name).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("prepare", help="reconstruct a reference panel from local Parquet, or select a custom panel")
    build.add_argument("kind", choices=KINDS)
    build.add_argument("--parquet", required=True, type=Path)
    build.add_argument("--custom", action="store_true", help="select from user-supplied data")
    build.add_argument("--count", type=int, help="number of complete packed source rows (custom only)")
    build.add_argument("--width", type=int, help="packed source row width (custom only)")
    build.add_argument("--per-stratum", type=int, help="selected documents per length stratum (custom only)")
    score = sub.add_parser("score", help="evaluate one local checkpoint on an exact panel replay")
    score.add_argument("--checkpoint", required=True, type=Path)
    score.add_argument("--checkpoint-sha256")
    score.add_argument(
        "--data-identity", type=Path, help="tokenizer identity sidecar for a trusted training checkpoint"
    )
    score.add_argument(
        "--trusted-training-checkpoint",
        action="store_true",
        help="allow full pickle deserialization of a trusted native trainer checkpoint",
    )
    score.add_argument("--panel", required=True, type=Path)
    score.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    score.add_argument(
        "--dtype", choices=("float32", "bfloat16"), help="default uses the panel's paper precision on CUDA"
    )
    score.add_argument("--batch-size", type=int)
    pair = sub.add_parser("compare", help="compute candidate minus reference BPB and a paired percentile interval")
    pair.add_argument("--candidate", required=True, type=Path)
    pair.add_argument("--reference", required=True, type=Path)
    for command in (build, score):
        command.add_argument("--tokenizer", type=Path, help="local tokenizer directory (omit for native byte models)")
        command.add_argument(
            "--manifest", type=Path, help="location of the distributed corpus-v1.json reference manifest"
        )
    for command in (build, score, pair):
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    # Avoid overwriting a source checkpoint, tokenizer asset, panel, or report if a path is mistyped.
    inputs = [
        getattr(args, k, None)
        for k in ("checkpoint", "panel", "parquet", "manifest", "data_identity", "candidate", "reference")
    ]
    if any(isinstance(f, Path) and f.resolve() == args.output.resolve() for f in inputs):
        parser.error("--output must differ from input files")
    if getattr(args, "tokenizer", None) and args.output.resolve().is_relative_to(args.tokenizer.resolve()):
        parser.error("--output must be outside the tokenizer directory")
    torch.set_num_threads(4)
    if args.action == "prepare":
        result = prepare(
            args.kind,
            args.parquet,
            load_tokenizer(args.tokenizer),
            reference=not args.custom,
            manifest_path=args.manifest,
            count=args.count,
            width=args.width,
            per_stratum=args.per_stratum,
        )
        save_result(args.output, result, binary=True)
        print(json.dumps({k: v for k, v in result.items() if k != "data"}))
    elif args.action == "score":
        panel = load_panel(args.panel, load_tokenizer(args.tokenizer), manifest_path=args.manifest)
        dtype = args.dtype or ("bfloat16" if args.device == "cuda" and panel["kind"] == "packed" else "float32")
        loaded = load_paper_checkpoint(
            args.checkpoint,
            tokenizer_path=args.tokenizer,
            device=args.device,
            dtype=dtype,
            expected_sha256=args.checkpoint_sha256,
            trusted_training_checkpoint=args.trusted_training_checkpoint,
            data_identity_path=args.data_identity,
        )
        result = score_panel(
            loaded, panel, device=args.device, dtype=dtype, db=args.batch_size, manifest_path=args.manifest
        )
        save_result(args.output, result)
        print(json.dumps(result["summary"], allow_nan=False))
    else:
        result = paired_comparison(json.loads(args.candidate.read_text()), json.loads(args.reference.read_text()))
        save_result(args.output, result)
        print(json.dumps(result["metrics"], allow_nan=False))


if __name__ == "__main__":
    main()
