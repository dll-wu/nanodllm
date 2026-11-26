"""Paired document reductions. No text, token IDs or model runtime is required."""

import math
import random

import numpy as np

from .components import PAIRS

LOG2 = math.log(2)


def component_intervals(records):
    """Resample the same document indices across both arms and both fitted seeds."""
    sizes = np.array(records["content_bytes"])
    n = len(sizes)
    losses = {k: np.array(v) for k, v in records["content_nll_nats"].items()}
    if records["seeds"] != [48, 49] or any(a.shape != (2, n) for a in losses.values()):
        raise ValueError("Incomplete paired component losses")
    indices = np.random.default_rng(20261106).integers(0, n, size=(10000, n))
    # The later revised-control analysis used Python's choices with replacement.
    rng = random.Random(20261106)
    followup = np.array([rng.choices(range(n), k=n) for _ in range(10000)])

    def compare(candidate, control, draws):
        delta = losses[candidate] - losses[control]
        paired = delta.sum(axis=1) / (LOG2 * sizes.sum())
        boot = delta.mean(axis=0)[draws].sum(axis=1) / (LOG2 * sizes[draws].sum(axis=1))
        return {
            "delta_a_minus_b_by_training_seed": paired.tolist(),
            "mean_delta": float(paired.mean()),
            "paired_document_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
        }

    results = {candidate: compare(candidate, control, indices) for _, control, candidate in PAIRS}
    results["matched_compute"] = compare("short_fastv2", "revised_fixed", indices)
    results["full_budget"] = compare("fastv2", "revised_fixed", followup)
    results["control_change"] = compare("block_fixed", "revised_fixed", followup)
    results["shorter_conversion"] = compare("short_fastv2", "fastv2", followup)
    return results


def _stratified_indices(strata, seed, draws, protocol):
    groups = {}
    for i, label in enumerate(strata):
        groups.setdefault(label, []).append(i)
    if set(groups) not in ({"short", "long"}, {"opened_long"}):
        raise ValueError("Unknown source length strata")
    if protocol == "document_order":
        rng = random.Random(seed)
        return np.array([[rng.choice(g) for g in groups.values() for _ in g] for _ in range(draws)])
    order = ("short", "long") if protocol == "d8" else sorted(groups)
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(groups[k], (draws, len(groups[k]))) for k in order], axis=1)


def block_intervals(records):
    """Keep paired nats and byte denominators together in every stratified draw."""
    indices = {}

    def reduce_pair(pair, protocol):
        strata = tuple(pair["strata"])
        key = (strata, pair["seed"], pair["draws"], protocol)
        if key not in indices:
            indices[key] = _stratified_indices(*key)
        draws = indices[key]
        result = {}
        for name, data in pair["metrics"].items():
            sizes, ref, cand = [np.array(data[k]) for k in ("bytes", "reference_nats", "candidate_nats")]
            if any(a.shape != (len(strata),) for a in (sizes, ref, cand)) or np.any(sizes <= 0):
                raise ValueError("Invalid document counts")
            delta = cand - ref
            denominator = sizes.sum() * LOG2
            boot = delta[draws].sum(axis=1) / (LOG2 * sizes[draws].sum(axis=1))
            result[name] = {
                "reference_bpb": float(ref.sum() / denominator),
                "candidate_bpb": float(cand.sum() / denominator),
                "delta_bpb": float(delta.sum() / denominator),
                "confidence_interval_bpb": np.quantile(boot, [0.025, 0.975]).tolist(),
                "delta_by_length": {
                    label: float(delta[mask].sum() / (LOG2 * sizes[mask].sum()))
                    for label in sorted(set(strata))
                    for mask in [np.array(strata) == label]
                },
            }
        return result

    development = {}
    for key, pair in records["development"].items():
        protocol = "document_order" if key == "d4" else "d8" if key.startswith("d8-") else "large"
        development[key] = reduce_pair(pair, protocol)
    held = {
        profile: {"comparisons": {k: reduce_pair(pair, "document_order") for k, pair in arms.items()}}
        for profile, arms in records["holdout"].items()
    }
    return development, {"primary_metrics": ["common/block_end", "native/mean_eval_seeds"], "runs": held}
