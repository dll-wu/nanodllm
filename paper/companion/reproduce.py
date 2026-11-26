"""Build the manuscript inputs from the public numerical snapshots."""

import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from . import block, components, development, historical, main_exhibits, scaling
from .claims import diagnostic_values, transfer
from .intervals import block_intervals, component_intervals
from .tasks import task_scores

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / "data" / f"{name}.json").read_bytes())


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def _block_data():
    data = load("block")
    development_panels, data["holdout"] = block_intervals(load("block-losses"))
    data["d4_paired_intervals"] = development_panels["d4"]
    data["d8"]["paired"] = []
    data["large"]["paired"] = []
    for profile, panel in development_panels.items():
        if profile == "d4":
            continue
        parts = profile.split("-")
        for metric, pair in panel.items():
            result = {"metric": metric, "delta": pair["delta_bpb"], "ci95": pair["confidence_interval_bpb"]}
            if parts[0] == "d8":
                data["d8"]["paired"].append({**result, "seed": int(parts[1][1:]), "conversion_lr": float(parts[2])})
            else:
                data["large"]["paired"].append({**result, "depth": int(parts[0][1:]), "panel": parts[1]})
    with np.load(ROOT / "data" / "task-draws.npz", allow_pickle=False) as draws:
        data["tasks"] = task_scores(load("task-counts"), data["task_panels"], draws=draws)
    for row in data["task_rows"]:
        pair = data["tasks"]["depths"][row["depth"]]["primary_accuracy"][row["task"]]
        for arm, accuracy in zip(("scratch_accuracy", "conversion_accuracy"), pair, strict=True):
            if float(row[arm]) != accuracy:
                raise ValueError("Task counts disagree with the per-task accuracy")
            row[arm] = accuracy
    for row in data["d8"]["generation"]:
        row["ppl"] = math.exp(row["nll_sum"] / row["scored_tokens"])
        row["reference_bpb"] = row["nll_sum"] / (math.log(2) * row["scored_bytes"])
    for row in data["large"]["generation"]:
        row["gemma_ppl"] = math.exp(row["nll_sum"] / row["scored_tokens"])
        row["gemma_bpb"] = row["nll_sum"] / (math.log(2) * row["scored_bytes"])
    return data, development_panels


def reproduce():
    """Return UTF-8 TeX sources and full-precision derived statistics."""
    history = load("historical")
    d4 = load("development")
    ladder = load("scaling")
    for point in ladder["points"]:
        plans = point["evaluator_values"]
        if plans:
            value = mean(plans)
            if not math.isclose(value, point["bpb"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Full-sequence endpoint disagrees with the evaluation plans")
            point["bpb"] = value
    component = load("components")
    contrasts = component_intervals(load("component-losses"))
    component["example_ppl"] = {
        name: math.exp(score["nll_sum"] / score["scored_tokens"])
        for name, score in component["decoder_example_scores"].items()
    }
    component["contrasts"] = contrasts
    component["compute"] = {
        "training": component["compute_training"],
        "predictive": component["compute_predictive"],
        "generation": component["compute_generation"],
        "prediction_contrasts": [{"panel": "fresh", "a": "fastv2", "b": "fixed", **contrasts["matched_compute"]}],
    }
    blocks, development_panels = _block_data()
    outputs = historical.render(history)
    outputs.update(development.render(d4))
    outputs.update(scaling.render(ladder))
    outputs.update(components.render(component))
    outputs.update(block.render(blocks))
    outputs.update(main_exhibits.render(blocks, component, ladder, d4, outputs))
    for path in sorted((ROOT / "templates").rglob("*.tex")):
        outputs[path.relative_to(ROOT / "templates").as_posix()] = path.read_text(encoding="utf-8")
    for name, rows in load("scope").items():
        body = [" & ".join(row) + r" \\" for row in rows]
        if name == "replication-coverage":
            body.insert(5, r"\midrule")
        outputs[f"tables/{name}-rows.tex"] = "\\def\\exhibitrows{%\n" + "\n".join(body) + "%\n}\n"
    diagnostics = load("diagnostics")
    statistics = {
        "diagnostics": diagnostic_values(diagnostics),
        "transfer": transfer(ladder, blocks, development_panels),
        "development_pairs": [
            {**row, "delta_bpb": row["candidate_bpb"] - row["control_bpb"]} for row in d4["paired_development"]
        ],
        "decoder_example_ppl": component["example_ppl"],
        "component_contrasts": contrasts,
        "block_development": development_panels,
        "block_holdout": blocks["holdout"],
        "block_tasks": blocks["tasks"],
        "large_model_generation": blocks["large"]["generation"],
    }
    return outputs, statistics
