"""Command-line entrypoints for nano_dllm training."""

import argparse
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from .config import apply_overrides, load_config
from .training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain a masked diffusion language model")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a plain JSON training config. Example configs are in the source repository.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional shortcut for --set train.run_dir=...",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value with a dotted key, e.g. --set train.num_steps=10.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    overrides = list(args.set)
    if args.run_dir is not None:
        overrides.append(f"train.run_dir={args.run_dir}")
    if overrides:
        config = apply_overrides(config, overrides)
    train(config)
