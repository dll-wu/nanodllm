"""Equal-task means and paired item uncertainty for fixed checkpoints and masks."""

from collections import defaultdict
from itertools import product
from statistics import mean

import numpy as np


def combinations():
    return product(("mean_nelbo", "sum_nelbo"), ("block_full_mask", "sequential_masked"))


def aggregate(tasks):
    out = {}
    for ranking, completion in combinations():
        values = []
        for task in tasks.values():
            key = completion if task["task_type"] == "language_modeling" else ranking
            acc = task["accuracy"][key]
            values.append([acc, *((acc - task[k]) / (1 - task[k]) for k in ("baseline_v1", "baseline_v2"))])
        out[f"{ranking}__{completion}"] = dict(
            zip(("raw", "centered_v1", "centered_v2"), map(mean, zip(*values, strict=True)), strict=True)
        )
    return out


def paired(left, right, *, draws=10000, seed=None):
    if left["schema"] != right["schema"] or left["schema"] != "nanodllm.paper_tasks.v1":
        raise ValueError("unsupported task report")
    if left["settings"] != right["settings"] or left["model"]["depth"] != right["model"]["depth"]:
        raise ValueError("paired task settings or depths differ")
    if left["model"]["tokenizer_content_sha256"] != right["model"]["tokenizer_content_sha256"]:
        raise ValueError("paired tokenizers differ")
    a, b = left["tasks"], right["tasks"]
    if not a or a.keys() != b.keys() or draws < 1:
        raise ValueError("paired reports need matching tasks and positive draws")
    seed = 20261118 + left["model"]["depth"] if seed is None else seed
    per_task = {}
    groups = defaultdict(list)
    variants = list(combinations())
    order = left["settings"]["task_order"]
    if len(order) != len(a) or set(order) != a.keys():
        raise ValueError("invalid task order")
    for label in left["settings"]["task_order"]:
        x, y = a[label], b[label]
        for key in (
            "task_type",
            "num_fewshot",
            "data_sha256",
            "selection_sha256",
            "prepared_inputs_sha256",
            "baseline_v1",
            "baseline_v2",
        ):
            if x[key] != y[key]:
                raise ValueError(f"paired {key} differs for {label}")
        row_indices, differences = [], []
        for lrow, rrow in zip(x["rows"], y["rows"], strict=True):
            for k in ("row", "demonstration_rows", "mask_plan_sha256"):
                if lrow.get(k) != rrow.get(k):
                    raise ValueError(f"unaligned {k} for {label}")
            row_indices.append(lrow["row"])
            values = []
            for ranking, completion in variants:
                key = completion if x["task_type"] == "language_modeling" else ranking
                left_correct, right_correct = lrow["correct"][key], rrow["correct"][key]
                if type(left_correct) is not bool or type(right_correct) is not bool:
                    raise ValueError("expected Boolean item correctness")
                values.append(int(right_correct) - int(left_correct))
            differences.append(values)
        if not row_indices or len(row_indices) != len(set(row_indices)):
            raise ValueError("empty or duplicate item IDs")
        delta = np.array(differences, dtype=float)
        primary = delta[:, 0]
        per_task[label] = {
            "items": len(row_indices),
            "gained": int((primary > 0).sum()),
            "lost": int((primary < 0).sum()),
            "raw_delta_pp": float(100 * primary.mean()),
            "contribution_to_raw_mean_pp": float(100 * primary.mean() / len(a)),
        }
        groups[x["data_sha256"]].append((label, row_indices, delta))

    rng = np.random.default_rng(seed)
    samples = np.zeros((draws, len(variants), 2))
    effects = np.zeros((len(variants), 2))
    for tasks in groups.values():
        indices = tasks[0][1]
        if any(rows != indices for _, rows, _ in tasks):
            raise ValueError("shared dataset variants must have identical original row order")
        # The paper used multinomial counts over distinct item contrast vectors.
        # Keep all four scoring controls and both summaries in each joint draw.
        contribution = np.zeros((len(indices), len(variants), 2))
        for label, _, delta in tasks:
            contribution[:, :, 0] += delta / len(a)
            contribution[:, :, 1] += delta / (len(a) * (1 - a[label]["baseline_v2"]))
        n = len(indices)
        unique, counts = np.unique(contribution.reshape(n, -1), axis=0, return_counts=True)
        samples += (rng.multinomial(n, counts / n, size=draws) @ unique / n).reshape(samples.shape)
        effects += contribution.mean(axis=0)
    scale = 100
    intervals = scale * np.quantile(samples, [0.025, 0.975], axis=0)
    output = {}
    for i, (ranking, completion) in enumerate(variants):
        output[ranking + "__" + completion] = {
            key: {"delta_pp": float(effects[i, j] * scale), "ci95_pp": intervals[:, i, j].tolist()}
            for j, key in enumerate(("raw", "centered_v2"))
        }
    return {
        "direction": "right_minus_left",
        "seed": seed,
        "draws": draws,
        "tasks": len(a),
        "items": sum(len(t["rows"]) for t in a.values()),
        "left_checkpoint_sha256": left["model"]["checkpoint_sha256"],
        "right_checkpoint_sha256": right["model"]["checkpoint_sha256"],
        "results": output,
        "primary_by_task": per_task,
    }
