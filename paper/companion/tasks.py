"""Equal-task score and paired item intervals from anonymous joint outcome counts."""

import numpy as np

VARIANTS = tuple(
    (rank, exact) for rank in ("mean_nelbo", "sum_nelbo") for exact in ("block_full_mask", "sequential_masked")
)


def task_scores(groups_by_depth, panels, *, draws=None):
    """Recompute task scores using the supplied bootstrap draws or a fresh seeded resample."""
    result = {}
    for depth, groups in groups_by_depth.items():
        rng = np.random.default_rng(20261118 + int(depth[1:])) if draws is None else None
        boots = np.zeros((10000, 2 * len(VARIANTS)))
        scores = {}
        for group in groups:
            tasks = group["tasks"]
            hist = group["outcomes"]
            counts = np.array([r["items"] for r in hist])
            correct = np.array([r["correct"] for r in hist])
            n = int(counts.sum())
            contribution = np.zeros((len(hist), boots.shape[1]))
            for i, task in enumerate(tasks):
                p = panels[task]
                if n != p["items"] or task in scores or correct.shape[1] != 4 * len(tasks):
                    raise ValueError(f"Invalid joint task counts for {task}")
                scores[task] = []
                for j, (rank, exact) in enumerate(VARIANTS):
                    alternative = (
                        exact == "sequential_masked" if p["task_type"] == "language_modeling" else rank == "sum_nelbo"
                    )
                    col = 4 * i + int(alternative)
                    scratch, conversion = correct[:, col], correct[:, col + 2]
                    scores[task].append([float(np.dot(counts, arm) / n) for arm in (scratch, conversion)])
                    delta = conversion - scratch
                    contribution[:, 2 * j] += delta / len(panels)
                    contribution[:, 2 * j + 1] += delta / (len(panels) * (1 - p["baseline_v2"]))
            # Few-shot and zero-shot HellaSwag share items. Their contributions
            # are already added to the same row when we resample each dataset.
            values, inverse = np.unique(contribution, axis=0, return_inverse=True)
            frequencies = np.bincount(inverse.ravel(), weights=counts)
            if draws is None:
                sampled = rng.multinomial(n, frequencies / n, size=len(boots))
            else:
                key = "__".join([depth, *tasks])
                sampled = draws[key]
                if (
                    sampled.shape != (len(boots), len(values))
                    or sampled.dtype.kind != "u"
                    or not np.all(sampled.sum(axis=1) == n)
                ):
                    raise ValueError(f"Invalid bootstrap counts for {key}")
            boots += sampled @ values / n
        if scores.keys() != panels.keys():
            raise ValueError(f"Missing task outcomes for {depth}")
        aggregates = {}
        for j, pair in enumerate(VARIANTS):
            entry = {}
            for i, arm in enumerate(("scratch", "conversion")):
                accuracy = [scores[t][j][i] for t in panels]
                entry[arm] = {
                    "mean_raw_accuracy": sum(accuracy) / len(panels),
                    "mean_centered_v2": sum(
                        (a - p["baseline_v2"]) / (1 - p["baseline_v2"])
                        for a, p in zip(accuracy, panels.values(), strict=True)
                    )
                    / len(panels),
                }
            entry["delta_raw_pp"] = 100 * (
                entry["conversion"]["mean_raw_accuracy"] - entry["scratch"]["mean_raw_accuracy"]
            )
            entry["conditional_raw_95ci_pp"] = (100 * np.quantile(boots[:, 2 * j], [0.025, 0.975])).tolist()
            entry["conditional_centered_v2_95ci_pp"] = (100 * np.quantile(boots[:, 2 * j + 1], [0.025, 0.975])).tolist()
            aggregates["__".join(pair)] = entry
        result[depth] = {"aggregates": aggregates, "primary_accuracy": {t: v[0] for t, v in scores.items()}}
    return {"depths": result}
