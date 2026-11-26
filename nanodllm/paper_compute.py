"""Summarize the compute of a scratch trajectory or a verified parent and child."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stage(run):
    metrics = _json(run / "metrics.json")
    summary = _json(run / "run_summary.json")
    if metrics["status"] != "ok":
        raise ValueError("a completed training run is required")
    tokens = metrics["total_tokens"]
    flops = metrics["flops_used"]
    if type(tokens) is not int or tokens <= 0 or not math.isfinite(flops) or flops <= 0:
        raise ValueError("expected positive training tokens and estimated FLOPs")
    resumed = bool(metrics.get("resume_consumed_microbatches", 0))
    seconds = metrics.get("training_seconds") if not resumed else None
    if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
        raise ValueError("invalid training time")
    gpu_seconds = (
        seconds * metrics["world_size"] if seconds is not None and metrics["runtime_device_type"] == "cuda" else None
    )
    return {
        "processed_tokens": tokens,
        "target_parameter_basis": summary["parameter_accounting"]["target_param_count_basis"],
        "target_parameters": summary["parameter_accounting"]["target_param_count"],
        "trainable_parameters": metrics["num_params"],
        "estimated_training_flops": flops,
        "flops_method": metrics["flops_accounting_method"],
        "training_seconds": seconds,
        "gpu_seconds": gpu_seconds,
        "resumed": resumed,
    }, summary["initial_state"].get("model_initialization_source")


def compute_ledger(run, *, parent_run=None, parent_checkpoint=None):
    child, source = _stage(Path(run))
    stages = []
    if bool(source) != (parent_run is not None):
        raise ValueError("supply the parent's run for a converted model and omit it for a scratch run")
    if parent_run is None and parent_checkpoint is not None:
        raise ValueError("a parent checkpoint also requires its run")
    if source:
        parent_root = Path(parent_run)
        parent, prior = _stage(parent_root)
        if prior:
            raise ValueError("this two-stage ledger expects a scratch parent")
        native = parent_root / "model.pt"
        bound = Path(parent_checkpoint) if parent_checkpoint else native
        with bound.open("rb") as stream:
            verified = hashlib.file_digest(stream, "sha256").hexdigest()
        if verified != source["sha256"] or parent["processed_tokens"] != source["source_train_tokens"]:
            raise ValueError("supplied parent differs from child initialization record")
        if bound.resolve() != native.resolve():
            from .paper_weights import SCHEMA, _read_file, file_sha256

            payload, _ = _read_file(bound, verified)
            if payload.get("schema") != SCHEMA or payload["source_sha256"] != file_sha256(native):
                raise ValueError("export does not originate from the supplied parent run")
        stages.append({"stage": "parent", **parent})
    stages.append({"stage": "continuation" if source else "scratch", **child})
    totals = {
        key: sum(s[key] for s in stages) if all(s[key] is not None for s in stages) else None
        for key in ("processed_tokens", "estimated_training_flops", "training_seconds", "gpu_seconds")
    }
    return {"schema": "nanodllm.training_compute.v1", "stages": stages, "totals": totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path, help="exported .weights.pt if the child used one")
    args = parser.parse_args()
    print(
        json.dumps(
            compute_ledger(args.run, parent_run=args.parent_run, parent_checkpoint=args.parent_checkpoint),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
