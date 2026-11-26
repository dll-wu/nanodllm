"""List, materialize and launch the native training recipes from the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .paper_recipes_data import VALIDATION_SHARD, verify_corpus, verify_tokenizer
from .paper_runtime.config import ExperimentConfig, config_from_dict
from .release_assets import release_asset

GROUPS = ("full-adamw", "block-muon", "development", "development-adamw-grid", "full-bracket", "block-bracket", "tiny")


@dataclass(frozen=True)
class Profile:
    name: str
    group: str
    settings: dict
    world_size: int
    parent: str | None
    checkpoint_sha256: str | None
    checkpoint_bytes: int | None


def read_profiles() -> dict[str, Profile]:
    root = release_asset("configs/paper")
    profiles = {}
    for group in GROUPS:
        catalog = json.loads((root / f"{group}.json").read_text(encoding="utf-8"))
        for name, row in catalog["profiles"].items():
            settings = {section: dict(fields) for section, fields in catalog["defaults"].items()}
            for section, fields in row["overrides"].items():
                settings.setdefault(section, {}).update(fields)
            if name in profiles:
                raise ValueError(f"duplicate paper profile {name!r}")
            profiles[name] = Profile(
                name,
                group,
                settings,
                row["world_size"],
                row.get("parent"),
                row.get("checkpoint_sha256"),
                row.get("checkpoint_bytes"),
            )
    return profiles


def resolve(
    profiles: dict[str, Profile],
    name: str,
    *,
    corpus: Path,
    run_dir: Path,
    tokenizer: Path | None = None,
    parent: Path | None = None,
    parent_sha256: str | None = None,
    parent_bytes: int | None = None,
) -> ExperimentConfig:
    """Bind local paths. A new parent requires an explicit digest from this reproduction."""
    profile = profiles[name]
    settings = {section: dict(fields) for section, fields in profile.settings.items()}
    train, data = settings["train"], settings["data"]
    corpus = corpus.expanduser().resolve()
    train["run_dir"] = str(run_dir.expanduser().resolve())
    settings["recipe"]["name"] = name
    data["train_data"] = [str(corpus)]
    if profile.group == "tiny":
        if tokenizer is not None:
            raise ValueError("tiny profiles use the bundled byte tokenizer")
    else:
        if tokenizer is None:
            raise ValueError("pass --tokenizer with the local paper tokenizer directory")
        validation = str(corpus / VALIDATION_SHARD)
        data.update(
            tokenizer_path=str(tokenizer.expanduser().resolve()), val_data=[validation], train_exclude_data=[validation]
        )
    if profile.parent:
        if parent is None:
            raise ValueError(f"pass --parent with the checkpoint for {profile.parent}")
        source = profiles[profile.parent]
        parent = parent.expanduser().resolve()
        digest = parent_sha256 if parent_sha256 is not None else source.checkpoint_sha256
        if parent.name.endswith(".weights.pt") and not parent_sha256:
            raise ValueError("portable weights require --parent-sha256 for the exported file")
        if not digest or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("pass the trusted parent's lowercase --parent-sha256")
        if digest == source.checkpoint_sha256:
            size = source.checkpoint_bytes
            if parent_bytes is not None and parent_bytes != size:
                raise ValueError("--parent-bytes differs from the recorded checkpoint size")
        else:
            size = parent_bytes if parent_bytes is not None else parent.stat().st_size
        train.update(
            model_init_from=str(parent), model_init_expected_sha256=digest, model_init_expected_size_bytes=size
        )
    elif parent is not None or parent_sha256 is not None or parent_bytes is not None:
        raise ValueError(f"{name} trains from scratch and takes no parent options")
    return config_from_dict(settings)


def write_resolved(config: ExperimentConfig, output_dir: Path) -> tuple[Path, str]:
    # Include every dataclass field, including the explicit empty optimizer checkout.
    payload = (json.dumps(asdict(config), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    sha = hashlib.sha256(payload).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{config.recipe.name}-{sha}.json"
    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as stream:
        staged = Path(stream.name)
    try:
        staged.write_bytes(payload)
        try:
            os.link(staged, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"existing resolved config differs at {path}") from None
    finally:
        staged.unlink(missing_ok=True)
    return path, sha


def verify_inputs(profile: Profile, config: ExperimentConfig) -> None:
    if profile.group != "tiny":
        verify_tokenizer(Path(config.data.tokenizer_path))
        verify_corpus(Path(config.data.train_data[0]))
    elif not Path(config.data.train_data[0]).is_file():
        raise ValueError("tiny profiles require a local Parquet file")
    if profile.parent:
        path = Path(config.train.model_init_from)
        if not path.is_file() or path.stat().st_size != config.train.model_init_expected_size_bytes:
            raise ValueError(f"missing parent or incorrect byte size for {profile.parent}")
        # Native model-state initialization checks SHA-256 on the same descriptor it deserializes.


def launch_command(profile: Profile, config_file: Path) -> list[str]:
    cmd = [sys.executable, "-m"]
    if profile.world_size > 1:
        cmd += ["torch.distributed.run", "--standalone", f"--nproc-per-node={profile.world_size}", "--module"]
    return [*cmd, "nanodllm.paper_runtime.cli", "--config", str(config_file)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    listing = commands.add_parser("list", help="list registered training profiles and their parents")
    listing.add_argument("--group", choices=GROUPS)
    listing.add_argument("--checkpoints", action="store_true", help="include archived checkpoint SHA-256 and byte size")
    for action in ("materialize", "launch"):
        cmd = commands.add_parser(action)
        cmd.add_argument("profile")
        cmd.add_argument("--corpus", required=True, type=Path, help="ClimbMix directory, or tiny example Parquet file")
        cmd.add_argument("--tokenizer", type=Path)
        cmd.add_argument("--run-dir", required=True, type=Path)
        cmd.add_argument("--output-dir", type=Path, default=Path("resolved-paper-configs"))
        cmd.add_argument(
            "--parent", type=Path, help="verified parent .weights.pt or trusted native training checkpoint"
        )
        if action == "launch":
            cmd.add_argument(
                "--trusted-training-parent", action="store_true", help="allow loading a local full training pickle"
            )
        cmd.add_argument(
            "--parent-sha256", help="explicitly bind a locally reproduced parent instead of the archived digest"
        )
        cmd.add_argument(
            "--parent-bytes", type=int, help="size of a reproduced parent when materializing ahead of time"
        )
    verify = commands.add_parser("verify-data", help="verify local tokenizer bytes and corpus shard inventory")
    verify.add_argument("--tokenizer", type=Path, required=True)
    verify.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "verify-data":
            verify_tokenizer(args.tokenizer)
            verify_corpus(args.corpus)
            print("Verified four tokenizer files and the 6,543-shard name/size inventory.")
            return
        profiles = read_profiles()
        if args.action == "list":
            columns = ["profile", "group", "ranks", "parent"]
            if args.checkpoints:
                columns += ["checkpoint_sha256", "checkpoint_bytes"]
            print("\t".join(columns))
            for p in profiles.values():
                if not args.group or args.group == p.group:
                    row = [p.name, p.group, str(p.world_size), p.parent or "-"]
                    if args.checkpoints:
                        row += [p.checkpoint_sha256 or "-", str(p.checkpoint_bytes or "-")]
                    print("\t".join(row))
            return
        if args.profile not in profiles:
            parser.error(f"unknown profile {args.profile!r}; see list")
        profile = profiles[args.profile]
        config = resolve(
            profiles,
            args.profile,
            corpus=args.corpus,
            run_dir=args.run_dir,
            tokenizer=args.tokenizer,
            parent=args.parent,
            parent_sha256=args.parent_sha256,
            parent_bytes=args.parent_bytes,
        )
        if args.action == "launch":
            if (
                config.train.model_init_from
                and not config.train.model_init_from.endswith(".weights.pt")
                and not args.trusted_training_parent
            ):
                parser.error(
                    "full training pickles require --trusted-training-parent; portable exports end in .weights.pt"
                )
            if Path(config.train.run_dir).is_dir() and any(Path(config.train.run_dir).iterdir()):
                parser.error("choose an empty --run-dir for a new training stage")
            verify_inputs(profile, config)
        path, digest = write_resolved(config, args.output_dir)
        print(
            json.dumps(
                {
                    "profile": profile.name,
                    "config": str(path),
                    "sha256": digest,
                    "ranks": profile.world_size,
                    "parent": profile.parent,
                }
            ),
            flush=True,
        )
        if args.action == "launch":
            subprocess.run(launch_command(profile, path), check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
