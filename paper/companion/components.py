"""Corrected component comparisons and the matched-conversion-compute follow-up."""

import math
from functools import partial
from statistics import mean

from .formatting import bpb, bpb_delta, bpb_interval
from .tables import add_table

PAIRS = (
    ("BD3 optimized noise", "bd3_fixed", "bd3_optimized"),
    ("Fast-v2 construction", "block_fixed", "fastv2"),
    ("LLaDA 1 masking", "mdlm", "llada1"),
    ("Corrected curriculum", "new_fixed", "new_curriculum"),
)
LABELS = {
    "bd3_fixed": "BD3 fixed noise",
    "bd3_optimized": "BD3 optimized noise",
    "block_fixed": "Native fixed block",
    "fastv2": "Fast-v2 bundle",
    "mdlm": "Full MDLM",
    "llada1": "LLaDA 1 masking",
    "old_fixed": "Historical fixed + tail",
    "new_fixed": "Corrected fixed + tail",
    "old_curriculum": "Historical curriculum + tail",
    "new_curriculum": "Corrected curriculum + tail",
}


def render(data):
    outputs = {}
    table = partial(add_table, outputs, "components")
    pred = {}
    for row in data["predictive"]:
        if (
            row["panel"] == "fresh"
            and row["stratum"] == "all"
            and row["canvas"] == "block_end"
            and row["primary"]
            and row["phase"] != "main"
        ):
            key = row["model"], row["seed"]
            assert key not in pred
            pred[key] = row
    gen = {}
    for row in data["generation"]:
        if (
            row["panel"] == "fresh"
            and row["stratum"] == "all"
            and row["primary"]
            and row["phase"] != "main"
            and row["decoder"] in ("independent_k1_bl32", "independent_k4_bl32")
        ):
            key = row["model"], row["seed"], row["decoder"]
            assert key not in gen
            gen[key] = row
    assert len(pred) == 20 and len(gen) == 40

    overview, seeds = [], []
    for label, control, candidate in PAIRS:
        a, b = [pred[(control, s)] for s in (48, 49)], [pred[(candidate, s)] for s in (48, 49)]
        delta = [y["content_bpb"] - x["content_bpb"] for x, y in zip(a, b, strict=True)]
        contrast = data["contrasts"][candidate]
        assert all(abs(x - y) < 1e-12 for x, y in zip(delta, contrast["delta_a_minus_b_by_training_seed"], strict=True))
        lo, hi = contrast["paired_document_bootstrap_95"]
        overview.append(
            [
                label,
                bpb(mean(r["content_bpb"] for r in a)),
                bpb(mean(r["content_bpb"] for r in b)),
                bpb_delta(delta[0]),
                bpb_delta(delta[1]),
                bpb_interval(lo, hi),
            ]
        )
        for seed, x, y in zip((48, 49), a, b, strict=True):
            seeds.append(
                [
                    label,
                    str(seed),
                    bpb(x["content_bpb"]),
                    bpb(y["content_bpb"]),
                    bpb(x["native_bpb"]),
                    bpb(y["native_bpb"]),
                ]
            )
    table(
        "replication-refresh",
        "Corrected best-fit 20M component comparisons.",
        [
            "Candidate versus its control",
            "Control",
            "Candidate",
            r"$\Delta$, s48",
            r"$\Delta$, s49",
            "Document 95\\% interval",
        ],
        overview,
        "Common left-context BPB with a matched BL32 canvas; lower is better. Control/candidate columns average training seeds 48/49; differences are candidate minus control. The interval resamples documents jointly across these fitted seeds, not a training-population interval. Each row has its own matched control and budget; rows are not a common-cost leaderboard. Appendix~\\ref{app:replication-refresh} gives native scores, generation and protocol limits.",
        "lrrrrr",
    )
    table(
        "replication-refresh-seeds",
        "Per-seed matched prediction and native bounds.",
        ["Comparison", "Seed", "Common ctrl.", "Common cand.", "Native ctrl.", "Native cand."],
        seeds,
        "Fresh-panel primary inference policy, BL32 end-padded common prediction. Native columns are conditional NELBO BPB, a separate sampled quantity. Fast-v2 has no comparable native-bound entry. Corrected fixed/curriculum endpoints include the identical TPP2 tail; native fixed-block/Fast-v2 endpoints do not. No cross-row ranking or equivalence of these readouts is implied.",
        "lrrrrr",
    )
    generation_rows = []
    for model in LABELS:
        row = [LABELS[model]]
        for k in (1, 4):
            records = [gen[(model, seed, f"independent_k{k}_bl32")] for seed in (48, 49)]
            ppl = math.exp(mean(r["reference_nll"] for r in records))
            rep = mean(r["repeated_4_fraction"] for r in records)
            length = mean(r["content_tokens"] for r in records)
            row += [f"{ppl:.1f}", f"{100 * rep:.1f}", f"{length:.1f}"]
        generation_rows.append(row)
    table(
        "replication-refresh-generation",
        "Generation under independent commitment.",
        ["Model", "K1 PPL", "Rep.\\%", "Length", "K4 PPL", "Rep.\\%", "Length"],
        generation_rows,
        "BL32, temperature 1, top-p 0.9 and cap 256; fresh prompts, two proposals and training seeds 48/49. K is maximum committed tokens per forward. PPL is the geometric mean of each seed's pooled perplexity under a frozen 20M AR20 reference, not a semantic-quality score. Repeated four-gram fraction and length are arithmetic means; repetition excludes outputs shorter than four tokens. Lower PPL with different length or repetition is not an unconditional quality win.",
        "lrrrrrr",
    )
    compute = data["compute"]
    compute_pred, compute_gen = {}, {}
    for row in compute["predictive"]:
        if row["panel"] == "fresh" and row["stratum"] == "all" and row["canvas"] == "block_end" and row["primary"]:
            key = row["method"], row["seed"]
            assert key not in compute_pred
            compute_pred[key] = row
    for row in compute["generation"]:
        if (
            row["panel"] == "fresh"
            and row["stratum"] == "all"
            and row["primary"]
            and row["decoder"] in ("independent_k1_bl32", "independent_k4_bl32")
        ):
            key = row["method"], row["seed"], row["decoder"]
            assert key not in compute_gen
            compute_gen[key] = row
    training = {(r["method"], r["seed"]): r for r in compute["training"]}
    assert len(training) == len(compute["training"]) == len(compute_pred) == 4
    assert len(compute_gen) == 8
    cost_rows, sample_rows = [], []
    for method, label in (("fixed", "Fixed BL32"), ("fastv2", "Fast-v2")):
        for seed in (48, 49):
            train, prediction = training[method, seed], compute_pred[method, seed]
            cost_rows.append(
                [
                    label,
                    str(seed),
                    f"{train['child_tpp']:.3f}",
                    str(train["steps"]),
                    f"{train['estimated_flops'] / 1e16:.4f}",
                    f"{train['training_minutes']:.2f}",
                    bpb(prediction["content_bpb"]),
                ]
            )
        for k in (1, 4):
            records = [compute_gen[method, seed, f"independent_k{k}_bl32"] for seed in (48, 49)]
            record = {
                "method": method,
                "k": k,
                "geometric_mean_ppl": math.exp(mean(r["reference_nll"] for r in records)),
                "mean_repeat4": mean(r["repeated_4_fraction"] for r in records),
                "mean_content_tokens": mean(r["content_tokens"] for r in records),
                "mean_eos_rate": mean(r["eos_present"] for r in records),
            }
            sample_rows.append(
                [
                    label,
                    str(k),
                    f"{record['geometric_mean_ppl']:.1f}",
                    f"{100 * record['mean_repeat4']:.2f}",
                    f"{record['mean_content_tokens']:.1f}",
                    f"{100 * record['mean_eos_rate']:.1f}",
                ]
            )
    contrast = [r for r in compute["prediction_contrasts"] if r["panel"] == "fresh"]
    assert len(contrast) == 1 and contrast[0]["a"] == "fastv2" and contrast[0]["b"] == "fixed"
    deltas = [compute_pred["fastv2", s]["content_bpb"] - compute_pred["fixed", s]["content_bpb"] for s in (48, 49)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(deltas, contrast[0]["delta_a_minus_b_by_training_seed"], strict=True))
    table(
        "fastv2-compute",
        "Fast-v2 at approximately matched conversion compute.",
        ["Arm", "Seed", "Child TPP", "Updates", r"FLOPs/$10^{16}$", "Train min.", "Common BPB"],
        cost_rows,
        "Each pair reuses the same independently trained AR10 parent; parent cost is shared and excluded from the displayed conversion FLOPs/time. Fixed BL32 uses block-causal clean attention, and neither arm has a separate tail. Dense forward/backward matrix-work estimates include both views and alignment but exclude optimizer, pointwise and evaluation work. Fast-v2 uses 0.83--0.84\\% more estimated work. Each arm has one A100 and one A800 40GB run, crossed over seeds; timing is observational. The common prediction panel is now opened development data.",
        "lrrrrrr",
    )
    table(
        "fastv2-compute-generation",
        "Generation after the compute-budget comparison.",
        ["Arm", "K", "Reference PPL", "Repeat4 \\%", "Length", "EOS \\%"],
        sample_rows,
        "Independent commitment at BL32, temperature 1, top-p 0.9 and cap 256. Each endpoint uses 32 prompts and two proposal seeds on the now-opened main panel. PPL is the geometric mean of seed-level pooled PPLs under the frozen internal 20M AR20 scorer; other columns are arithmetic seed means. EOS is observed occurrence, not accuracy against a gold stopping point. K1's lower PPL accompanies longer output and fewer EOS emissions; K4 shows no consistent likelihood gain. This is not a general generation-quality ranking.",
        "lrrrrr",
    )
    return outputs
