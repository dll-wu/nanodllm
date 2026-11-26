"""Paired, length-stratified corpus bootstrap with the original paper draw orders."""

import math
import random

import numpy as np

from .paper_eval_scores import REPORT_SCHEMA, summaries
from .paper_runtime.core_evaluation_cells import require


def bootstrap_indices(rows, kind, depth):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["stratum"], []).append(i)
    if kind == "dev32" and depth in (8, 12, 20):
        seed = 20261116 if depth == 8 else 20261118
        order = ("short", "long") if depth == 8 else sorted(groups)
        require(set(order) == set(groups), "development strata differ")
        rng = np.random.default_rng(seed)
        indices = np.concatenate([rng.choice(groups[k], (5000, len(groups[k])), replace=True) for k in order], axis=1)
        return seed, indices
    seed = 2026111201
    rng = random.Random(seed)
    # The D4 development and all holdout comparisons visited strata in order of
    # first appearance, and advanced Python's RNG separately for every document.
    return seed, np.asarray([[rng.choice(g) for g in groups.values() for _ in g] for _ in range(2000)])


def paired_comparison(candidate, reference):
    require(candidate["schema"] == reference["schema"] == REPORT_SCHEMA, "unknown score schema")
    for key in ("kind", "reference", "panel_sha256", "tokenizer_sha256", "dtype", "device", "device_batch"):
        require(candidate[key] == reference[key], f"paired {key} differs")
    if candidate["seeds"] and reference["seeds"]:
        require(
            candidate["seeds"] == reference["seeds"] and candidate["plans"] == reference["plans"],
            "paired noise differs",
        )
    left, right = candidate["rows"], reference["rows"]
    for rows, report in ((left, candidate), (right, reference)):
        require(rows and len({r["id"] for r in rows}) == len(rows), "empty or duplicate row identity")
        require(all(set(r["metrics"]) == set(rows[0]["metrics"]) for r in rows), "incomplete row metrics")
        require(summaries(rows) == report["summary"], "row scores disagree with their summary")
    require(
        [(r["id"], r["stratum"]) for r in left] == [(r["id"], r["stratum"]) for r in right],
        "paired rows or strata differ",
    )
    keys = sorted(left[0]["metrics"].keys() & right[0]["metrics"].keys())
    require(bool(keys), "no common estimand")
    if candidate["kind"] == "dev32":
        require(candidate["depth"] == reference["depth"], "development bootstrap requires a matched-depth pair")
    seed, indices = bootstrap_indices(left, candidate["kind"], candidate["depth"])
    metrics = {}
    for key in keys:
        candidate_parts = np.array([r["metrics"][key] for r in left], dtype=np.float64)
        reference_parts = np.array([r["metrics"][key] for r in right], dtype=np.float64)
        require(np.array_equal(candidate_parts[:, 1], reference_parts[:, 1]), "paired byte denominators differ")
        require(
            np.all(np.isfinite(candidate_parts))
            and np.all(np.isfinite(reference_parts))
            and np.all(candidate_parts >= 0)
            and np.all(reference_parts >= 0),
            "invalid paired score components",
        )
        byte_totals = candidate_parts[:, 1][indices].sum(axis=1)
        require(bool((byte_totals > 0).all()), "bootstrap replicate has no scoreable bytes")
        deltas = (candidate_parts[:, 0] - reference_parts[:, 0])[indices].sum(axis=1) / (math.log(2) * byte_totals)
        a, b = candidate["summary"][key]["bpb"], reference["summary"][key]["bpb"]
        metrics[key] = dict(
            candidate_bpb=a, reference_bpb=b, delta_bpb=a - b, ci95_bpb=np.quantile(deltas, [0.025, 0.975]).tolist()
        )
    return dict(
        schema="nanodllm.paper.corpus-paired.v1",
        kind=candidate["kind"],
        panel_sha256=candidate["panel_sha256"],
        candidate_sha256=candidate["checkpoint_sha256"],
        reference_sha256=reference["checkpoint_sha256"],
        units=len(left),
        bootstrap_seed=seed,
        bootstrap_draws=len(indices),
        metrics=metrics,
    )
