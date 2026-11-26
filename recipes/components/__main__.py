"""Resolve, train and score the paper's paired 20M component recipes."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from nanodllm.runtime.config import ExperimentConfig, config_from_dict
from nanodllm.runtime.telemetry import write_json_atomic
from nanodllm.weights import file_sha256, load_weights

from .eval import compare, score_panel
from .train import read_ar_parent, train_stage


@dataclass
class Stage:
    name: str
    objective: str
    policy: dict
    config: ExperimentConfig


def resolve(preset, arm, seed, output):
    if seed not in (48, 49):
        raise ValueError("the paired training seeds are 48 and 49")
    preset = Path(preset)
    if not preset.exists() and preset.parent == Path("recipes/components"):
        preset = Path(__file__).with_name(preset.name)
    spec = json.loads(preset.read_text())
    if arm not in spec["arms"]:
        raise ValueError(f"choose an arm from {sorted(spec['arms'])}")
    stages = {"main": spec["arms"][arm]}
    if "tail" in spec:
        stages["tail"] = spec["tail"]
    base = json.loads(preset.with_name("base.json").read_text())
    result = []
    for name, stage in stages.items():
        merged = {section: {**values, **stage["overrides"].get(section, {})} for section, values in base.items()}
        if stage["overrides"].keys() - base.keys():
            raise ValueError("unknown preset section")
        merged["runtime"].update(seed=seed, model_seed=seed, training_seed=seed)
        merged["train"]["run_dir"] = str(Path(output) / name)
        merged["recipe"] = {"name": f"component/{preset.stem}/{arm}/{name}", "status": "historical"}
        cfg = config_from_dict(merged)
        result.append(Stage(name, stage["objective"], stage["policy"], cfg))
    return result


def train(args):
    from nanodllm.runtime.training import train as native_train

    stages = resolve(args.preset, args.arm, args.seed, args.output)
    if int(os.environ.get("WORLD_SIZE", 1)) != 1 or int(os.environ.get("RANK", 0)) != 0:
        raise ValueError("20M components require a single rank")
    needs_parent = stages[0].objective in {"B1", "B2", "B3", "native-bl32"}
    if (args.trusted_ar_parent is not None) != needs_parent or bool(args.parent_sha256) != needs_parent:
        raise ValueError("conversion requires --trusted-ar-parent and --parent-sha256; scratch and AR take neither")
    initial = stages[0].config
    if initial.data.tokenizer == "hf" and not Path(initial.data.tokenizer_path).is_dir():
        raise ValueError("set NANODLLM_TOKENIZER_DIR to the local paper tokenizer")
    if not all(Path(p).exists() for p in [*initial.data.train_data, *initial.data.val_data]):
        raise ValueError("set NANODLLM_DATA_DIR to the local corpus including the validation shard")
    parent = None
    if initial.data.tokenizer == "hf":
        from recipes.data import verify_corpus, verify_tokenizer

        verify_corpus(Path(initial.data.train_data[0]))
        verify_tokenizer(Path(initial.data.tokenizer_path))
    if needs_parent:
        ar = resolve(args.preset.with_name("ar10.json"), "parent", args.seed, args.output)[0].config
        parent = read_ar_parent(
            args.trusted_ar_parent,
            args.parent_sha256,
            seed=args.seed,
            expected_tokens=ar.train.target_tokens,
            expected_steps=ar.train.num_steps,
        )
        if parent["train_config"]["data"]["tokenizer"] != initial.data.tokenizer:
            raise ValueError("AR parent tokenizer kind differs")
        if initial.data.tokenizer == "hf":
            from nanodllm.evaluation.checkpoint import load_tokenizer, tokenizer_identity, training_tokenizer_sha256

            if training_tokenizer_sha256(args.trusted_ar_parent) != tokenizer_identity(
                load_tokenizer(initial.data.tokenizer_path)
            ):
                raise ValueError("AR parent tokenizer contents differ")
    args.output.mkdir(parents=True, exist_ok=False)
    parent_sha = args.parent_sha256
    for stage in stages:
        cfg = stage.config
        output = Path(cfg.train.run_dir)
        if stage.objective.startswith("native-"):
            if needs_parent:
                cfg.train.model_init_from = str(args.trusted_ar_parent.resolve())
                cfg.train.model_init_expected_sha256 = parent_sha
                cfg.train.model_init_expected_size_bytes = args.trusted_ar_parent.stat().st_size
            native_train(cfg)
            path = output / "model.pt"
        else:
            path = train_stage(cfg, stage.objective, stage.policy, output, parent=parent, parent_sha256=parent_sha)
        source_sha = file_sha256(path)
        receipt = {
            "schema": "nanodllm.component-endpoint.v1",
            "arm": f"{args.preset.stem}/{args.arm}",
            "phase": stage.name,
            "objective": stage.objective,
            "seed": args.seed,
            "step": cfg.train.num_steps,
            "source_positions": cfg.train.target_tokens,
            "ar_parent_sha256": args.parent_sha256,
            "phase_parent_sha256": parent_sha,
            "source_sha256": source_sha,
        }
        write_json_atomic(str(output / "component.json"), receipt)
        if stage is not stages[-1]:
            # Only a locally produced tensor-and-primitive endpoint enters the shared tail.
            parent = torch.load(path, map_location="cpu", weights_only=True)
            parent_sha = source_sha
    return receipt


def evaluate(args):
    from nanodllm.evaluation.panels import load_panel, prepare

    endpoint = json.loads(args.endpoint.read_text())
    if endpoint.get("schema") != "nanodllm.component-endpoint.v1" or endpoint["objective"] == "native-ar":
        raise ValueError("expected a diffusion component endpoint receipt")
    model = load_weights(
        args.weights,
        expected_checkpoint_sha256=args.weights_sha256,
        tokenizer_path=args.tokenizer_dir,
        device=args.device,
        dtype="float32",
    )
    receipt = model.checkpoint_receipt
    if (
        receipt["source_sha256"] != endpoint["source_sha256"]
        or receipt["checkpoint_step"] != endpoint["step"]
        or receipt["checkpoint_train_tokens"] != endpoint["source_positions"]
    ):
        raise ValueError("weights and component endpoint differ")
    if args.panel:
        panel = load_panel(args.panel, model.tokenizer, manifest_path=args.manifest)
    else:
        panel = prepare("dev32", args.parquet, model.tokenizer, manifest_path=args.manifest)
    if panel["kind"] != "dev32":
        raise ValueError("component comparisons use the shared dev32 panel")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    scores = score_panel(model.model, panel["data"], objective=endpoint["objective"])
    return {
        **endpoint,
        "schema": "nanodllm.component-scores.v1",
        "panel_sha256": panel["panel_sha256"],
        "reference_panel": panel["reference"],
        "weights_sha256": receipt["checkpoint_sha256"],
        **scores,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("show", "train"):
        command = commands.add_parser(name)
        command.add_argument("preset", type=Path, help="study JSON in recipes/components")
        command.add_argument("arm")
        command.add_argument("--seed", type=int, choices=(48, 49), required=True)
        command.add_argument("--output", type=Path, required=name == "train", default=Path("runs/components"))
        if name == "train":
            command.add_argument(
                "--trusted-ar-parent", type=Path, help="explicitly trusted, locally trained full AR10 checkpoint"
            )
            command.add_argument("--parent-sha256")
    ev = commands.add_parser("eval")
    ev.add_argument("--weights", required=True, type=Path, help="portable .weights.pt export")
    ev.add_argument("--weights-sha256")
    ev.add_argument("--endpoint", required=True, type=Path, help="matching component.json written by train")
    ev.add_argument("--tokenizer-dir", type=Path)
    ev.add_argument("--manifest", type=Path, help="public paper/panels/corpus-v1.json")
    ev.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    panel = ev.add_mutually_exclusive_group(required=True)
    panel.add_argument("--parquet", type=Path, help="local reference validation shard")
    panel.add_argument("--panel", type=Path, help="prepared dev32 tensor panel")
    cmp = commands.add_parser("compare")
    cmp.add_argument("--candidate", required=True, nargs=2, type=Path, help="two eval JSON files (seeds 48 and 49)")
    cmp.add_argument("--control", required=True, nargs=2, type=Path)
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train(args)
    elif args.command == "show":
        result = [asdict(s) for s in resolve(args.preset, args.arm, args.seed, args.output)]
    elif args.command == "eval":
        result = evaluate(args)
    else:
        result = compare(
            [json.loads(p.read_text()) for p in args.candidate], [json.loads(p.read_text()) for p in args.control]
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
