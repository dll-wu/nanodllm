"""Prepare a local corpus panel, score a checkpoint, or compare paired scores."""

import argparse
import json
from pathlib import Path

import torch

from ..release_assets import release_asset
from ..report_io import check_output, save_result
from .checkpoint import load_checkpoint, load_tokenizer
from .compare import paired_comparison
from .panels import KINDS, load_panel, prepare
from .scores import score_panel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("prepare", help="reconstruct a reference panel from local Parquet, or select a custom panel")
    build.add_argument("kind", choices=KINDS)
    build.add_argument("--parquet", required=True, type=Path)
    build.add_argument("--custom", action="store_true", help="select from user-supplied data")
    build.add_argument("--count", type=int, help="number of packed source rows or block documents (custom only)")
    build.add_argument("--width", type=int, help="packed source row width (custom only)")
    build.add_argument("--block-size", type=int, help="checkpoint block length for a custom block panel")
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
    torch.set_num_threads(4)
    if args.action == "prepare":
        reference = not args.custom and args.kind != "block"
        manifest = args.manifest
        if reference and manifest is None:
            manifest = release_asset("paper/panels/corpus-v1.json")
        check_output(parser, args.output, files=(args.parquet, manifest), tokenizer=args.tokenizer)
        result = prepare(
            args.kind,
            args.parquet,
            load_tokenizer(args.tokenizer),
            reference=reference,
            manifest_path=manifest,
            count=args.count,
            width=args.width,
            per_stratum=args.per_stratum,
            block_size=args.block_size,
        )
        save_result(args.output, result, binary=True)
        print(json.dumps({k: v for k, v in result.items() if k != "data"}))
    elif args.action == "score":
        check_output(
            parser,
            args.output,
            files=(args.panel, args.manifest),
            checkpoint=args.checkpoint,
            training=args.trusted_training_checkpoint,
            data_identity=args.data_identity,
            tokenizer=args.tokenizer,
        )
        panel = load_panel(args.panel, load_tokenizer(args.tokenizer), manifest_path=args.manifest)
        if panel["reference"] and args.manifest is None:
            check_output(parser, args.output, files=(release_asset("paper/panels/corpus-v1.json"),))
        dtype = args.dtype or ("bfloat16" if args.device == "cuda" and panel["kind"] == "packed" else "float32")
        loaded = load_checkpoint(
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
        check_output(parser, args.output, files=(args.candidate, args.reference))
        result = paired_comparison(json.loads(args.candidate.read_text()), json.loads(args.reference.read_text()))
        save_result(args.output, result)
        print(json.dumps(result["metrics"], allow_nan=False))


if __name__ == "__main__":
    main()
