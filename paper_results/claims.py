"""Unrounded values behind the development, transfer and decoder comparisons."""

import math
from statistics import mean

from .formatting import one


def transfer(ladder, block, development_panels):
    full = []
    for child in (p for p in ladder["points"] if p["family"] == "conversion"):
        scratch = one(ladder["points"], depth=child["depth"], family="scratch")
        full.append(
            {
                "depth": child["depth"],
                "scratch_bpb": scratch["bpb"],
                "conversion_bpb": child["bpb"],
                "delta_bpb": child["bpb"] - scratch["bpb"],
            }
        )

    result = []
    for depth, seeds in [(4, [48]), (8, [50, 51, 52]), (12, [50]), (20, [50])]:
        for panel in ("development", "holdout"):
            for kind, metric in (("common", "common/block_end"), ("native", "native/mean_eval_seeds")):
                values = []
                for seed in seeds:
                    if panel == "development":
                        profile = {4: "d4", 8: f"d8-s{seed}-0.00175", 12: "d12-fresh", 20: "d20-fresh"}[depth]
                        pair = development_panels[profile][metric if depth == 4 else kind]
                    else:
                        profile = f"d8-s{seed}" if depth == 8 else f"d{depth}"
                        arm = "lr00175" if depth == 8 else "conversion"
                        pair = block["holdout"]["runs"][profile]["comparisons"][arm][metric]
                    values.append(pair)
                result.append(
                    {
                        "depth": depth,
                        "seeds": seeds,
                        "panel": panel,
                        "metric": metric,
                        **{
                            field: mean(v[field] for v in values)
                            for field in ("reference_bpb", "candidate_bpb", "delta_bpb")
                        },
                    }
                )
    return {"full_sequence": full, "block": result}


def diagnostic_values(diagnostics):
    cache = diagnostics["cache_checks"]
    bf16 = [r["by_dtype"]["bfloat16"]["max_probability_tv"] for r in cache]
    controls = diagnostics["scorer_and_commitment_controls"]
    rows = diagnostics["canvas_groups"]
    bins = []
    for seed in (48, 49):
        for canvas in ("legacy", "block_end"):
            for slots in ([1, 8], [9, 16], [17, 24], [25, 32]):
                values = [
                    one(rows, seed=seed, canvas=canvas, model=name, remaining_slots=slots)
                    for name in ("bd3_fixed", "bd3_optimized")
                ]
                fixed, calibrated = values
                difference = (calibrated["nll_nats"] - fixed["nll_nats"]) / (fixed["bytes"] * math.log(2))
                bins.append(
                    {"seed": seed, "canvas": canvas, "remaining_slots": slots, "calibrated_minus_fixed_bpb": difference}
                )
    return {
        "fp32_cache_comparisons": sum(r["by_dtype"]["float32"]["comparisons"] for r in cache),
        "fp32_max_logit_error": max(r["by_dtype"]["float32"]["max_absolute_error"] for r in cache),
        "bf16_range_of_worst_total_variation": [min(bf16), max(bf16)],
        "trainer_replay_mean_bpb": {k: mean(v) for k, v in diagnostics["trainer_replay"].items()},
        "canvas_bins": bins,
        "fresh_draw_k4_minus_k1_reference_nll": {
            str(s): one(controls, seed=s, group="mixed_bl:independent-k4-p0.9")["nll_micro"]
            - one(controls, seed=s, group="mixed_bl:independent-k1-p0.9")["nll_micro"]
            for s in (48, 49)
        },
    }
