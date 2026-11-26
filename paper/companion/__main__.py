"""Verify the numerical companion or materialize its TeX and derived statistics."""

import argparse
import json
from pathlib import Path

from .reproduce import ROOT, reproduce, sha256


def verify():
    manifest = json.loads((ROOT / "manifest.json").read_bytes())
    inputs = {p.name: sha256(p.read_bytes()) for p in (ROOT / "data").glob("*.json")}
    if inputs != manifest["inputs"]:
        raise ValueError("Numerical inputs differ from the pinned public snapshots")
    outputs, stats = reproduce()
    expected = manifest["outputs"]
    if outputs.keys() != expected.keys():
        raise ValueError("Exhibit inventory differs from the pinned edition")
    for name, value in outputs.items():
        if sha256(value.encode("utf-8")) != expected[name]["sha256"]:
            raise ValueError(f"Display differs from the pinned edition in {name}")
    return outputs, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="recompute and compare in memory (default)")
    mode.add_argument("--output", type=Path, metavar="DIR", help="write verified TeX and statistics.json")
    args = parser.parse_args()
    outputs, stats = verify()
    if args.output:
        for name, content in {
            **outputs,
            "statistics.json": json.dumps(stats, indent=2, allow_nan=False) + "\n",
        }.items():
            path = args.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
    print(f"Verified {len(outputs)} TeX files for 7 paper figures, 13 paper tables and supplementary exhibits.")


if __name__ == "__main__":
    main()
