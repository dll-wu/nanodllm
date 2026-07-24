#!/usr/bin/env python3
"""Write a deterministic dataset/tokenizer/packing identity artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanodllm.config import apply_overrides, load_config  # noqa: E402
from nanodllm.data_identity import build_data_identity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record exact Parquet split membership, tokenizer hashes, and packing settings."
    )
    parser.add_argument("--config", required=True, help="Resolved experiment JSON config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply the same dotted config override syntax as the trainer; repeat as needed",
    )
    parser.add_argument("--output", help="Optional output JSON path; without it, the record is printed to stdout")
    parser.add_argument(
        "--full-sha256",
        action="store_true",
        help="Read and hash every dataset byte; expensive and disabled by default",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", "1")),
        help="DDP world size used for rank-membership summaries (default: WORLD_SIZE or 1)",
    )
    parser.add_argument(
        "--assignment-epoch",
        type=int,
        default=1,
        help="Epoch for the recorded shuffled rank-membership summary (default: 1)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.set:
        config = apply_overrides(config, list(args.set))
    record = build_data_identity(
        config,
        full_sha256=bool(args.full_sha256),
        world_size=int(args.world_size),
        assignment_epoch=int(args.assignment_epoch),
    )
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(
            f"wrote {output} (identity_sha256={record['hashes']['identity_sha256']})",
            file=sys.stderr,
        )
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
