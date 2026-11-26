"""Historical discovery, grid, follow-up and decoder tables."""

import math
import statistics
from functools import partial

from .formatting import bpb, bpb_delta, num, one
from .tables import add_table


def _task_scores(rows, sample_count):
    for row in rows:
        row["accuracy"] = {arm: count / row[sample_count] for arm, count in row["correct"].items()}
    result = {}
    for arm in rows[0]["accuracy"]:
        result[arm] = {
            "mean_raw_accuracy": statistics.mean(row["accuracy"][arm] for row in rows),
            "mean_centered_v2": statistics.mean(
                (row["accuracy"][arm] - row["baseline_v2"]) / (1 - row["baseline_v2"]) for row in rows
            ),
        }
    return result


def render(data):
    outputs = {}
    table = partial(add_table, outputs, "historical")
    b1 = data["component-discovery-1"]
    b2 = data["component-discovery-2"]
    labels = [
        ("F0", "AR reference", 20),
        ("F1", "Scratch MDLM", 20),
        ("F2", "Scratch LLaDA-1-style recipe", 20),
        ("F3", "BD3-LM BL32, fixed noise", 20),
        ("F4", "BD3-LM BL32, optimized noise", 20),
        ("B1", "AR to fixed BL32 masked diffusion", 40),
        ("B2", "AR to LLaDA-2.0-style block curriculum", 40),
        ("B3", "AR to Fast-dLLM-v2-style construction", 40),
        ("E0", "B2 plus masked-diffusion continuation", 42),
        ("E1", "B2 plus LLaDA-2.1-style positional repair", 42),
        ("E2", "B2 plus action-vocabulary positional control", 42),
        ("E3", "B2 plus LLaDA-2.2-style structural repair", 42),
        ("W0", "AR to bidirectional-current-block control", 40),
        ("W1", "AR to WeDLM-style reordered attention", 40),
        ("M0", "AR to uniform fixed-count masking", 40),
        ("M1", "AR to position-dependent masking", 40),
        ("A0", "MDLM to fixed BL32 control", 40),
        ("A1", "MDLM to adaptive irregular partitions", 40),
    ]
    rows = []
    for arm, label, tpp in labels:
        r = (
            one(b1["predictive_test"], arm=arm)
            if arm[0] in "FBE"
            else one(b2["predictive"], task=f"{arm}-predictive-test")
        )
        assert r["rows"] == 256
        rows.append([arm, label, str(tpp), bpb(r["content_bpb"]), bpb(r.get("conditional_native_bpb"))])
    table(
        "components",
        "20M component discovery screens.",
        ["ID", "Tested reconstruction", "Total TPP", "Common BPB", "Native BPB"],
        rows,
        "Seed42; 128 documents, two evaluator repeats. Four layers, width 256, context1024, DB16, EB64K. "
        "IDs are local to this replication series. Common BPB and native conditional NELBO are distinct; lower is better within a column, but native topologies differ. "
        "F/B stages use TPP20; E adds TPP2 to B2. W/M share an AR20 parent; A shares an MDLM20 parent. Budgets include parents, not equal FLOPs. "
        "Only the contrasts in Table~\\ref{tab:component-seeds} received the listed fresh-seed confirmation.",
        "llrrr",
    )

    rows = []
    for budget in (8, 32):
        estimates = [
            one(b1["estimator"], budget=budget, method=method) for method in ("nested", "all_times", "coupled")
        ]
        assert all(row["pairs"] == 32 for row in estimates)
        rows.append([str(budget), *[num(row["mean_loss_variance"], 5) for row in estimates]])
    table(
        "estimator-screen",
        "LLaDA-1.5-inspired estimator diagnostic.",
        ["Estimator budget", "Nested", "All-times", "Coupled"],
        rows,
        "Mean scalar preference-loss estimator variance across 32 fixed pairs, with the same nominated budget per method. "
        "Coupled denotes policy/reference noise coupling. These are estimator measurements, not preference-model training, "
        "full-gradient variance, downstream quality or a measured training-compute advantage.",
        "rrrr",
    )

    rows = []
    for arm in ("E0", "E1", "E2", "E3"):
        strata = [
            one(b1["repair"], arm=arm, stratum=kind) for kind in ("clean", "substitution", "insertion", "deletion")
        ]
        assert all(row["samples"] == 16 for row in strata)
        clean = strata[0]
        rows.append([arm, f"{sum(row['exact'] for row in strata)}/64", f"{clean['clean_unchanged']}/16"])
    table(
        "repair-screen",
        "Initial 20M local repair screen.",
        ["Arm", "Exact outputs", "Clean drafts unchanged"],
        rows,
        "32 cases with two evaluation seeds: clean, substitution, insertion and deletion strata. "
        "The fixed 32-slot panel uses a synthetic local EOS anchor. E0 is an all-MASK redrafting control, "
        "not an information-matched draft-conditioned editor. E1/E2/E3 are the reconstruction arms in Table~\\ref{tab:components}. "
        "Exact restoration and clean preservation differ from partial edit F1. This initial screen is not a summary of every later repair diagnostic "
        "or a falsification of published large-model editing.",
        "lrr",
    )

    pos = data["positive-seeds"]
    ada = data["adaptive-seeds"]
    rows = []
    for label, metric, stat in [
        ("F4 minus F3", "Common BPB", pos["fresh_pairs"]["F4_minus_F3"]["content_bpb"]),
        ("F4 minus F3", "Native BPB", pos["fresh_pairs"]["F4_minus_F3"]["conditional_native_bpb"]),
        ("B3 minus B1", "Common BPB", pos["fresh_pairs"]["B3_minus_B1"]["content_bpb"]),
        ("A1 minus A0", "Common BPB", ada["fresh_pairs"]["predictive/content_bpb"]),
        ("A1 minus A0", "Native BPB", ada["fresh_pairs"]["predictive/conditional_native_bpb"]),
    ]:
        vals = [stat["values"][str(seed)] for seed in (43, 44)]
        assert stat["n_pairs"] == 2
        assert math.isclose(stat["mean"], statistics.mean(vals), abs_tol=1e-12)
        assert math.isclose(stat["sample_sd"], statistics.stdev(vals), abs_tol=1e-12)
        rows.append([label, metric, *[bpb_delta(v) for v in vals], bpb_delta(stat["mean"]), bpb(stat["sample_sd"])])
    table(
        "component-seeds",
        "Fresh paired-seed component effects.",
        ["Contrast", "Readout", "Seed43", "Seed44", "Mean delta", "Paired SD"],
        rows,
        "Negative differences favor the first arm. The two fresh pairs exclude discovery seed42. F3/F4 change scratch initialization and training randomness; "
        "B1/B3 share one AR parent and A0/A1 share one MDLM parent. Data order is fixed. SD is the sample SD of two paired differences, not a confidence interval or an error bar for other methods.",
        "llrrrr",
    )

    gen = data["generation-replay"]
    rows = []
    for arm in ("F0", "F3", "F4", "B1", "B3"):
        decoders = (
            [("ar_t1.0_p0.9", "AR top-p0.9"), ("ar_t1.0_p1.0", "AR top-p1")]
            if arm == "F0"
            else [
                ("native_reference", "Native"),
                ("common_k1_t1.0_p1.0", "Confidence K1"),
                ("common_k4_t1.0_p1.0", "Confidence K4"),
            ]
        )
        for decoder, label in decoders:
            r = one(gen["generation"], arm=arm, decoder_id=decoder, kind="generation")
            t = one(gen["generation"], arm=arm, decoder_id=decoder, kind="timing")
            assert r["samples"] == 256 and t["samples"] == 18 and t["timing_context"] == "isolated"
            tps = t["mean_content_tokens"] * t["samples"] / t["work"]["decode_seconds"]
            rows.append(
                [
                    arm,
                    label,
                    num(r["mean_repeated_4_fraction"] * 100, 2),
                    num(r["secondary_AR_proxy"]["per_token_nll"], 3),
                    num(r["mean_content_tokens"], 1),
                    num(r["work"]["content_tokens_per_forward"], 3),
                    num(tps, 1),
                ]
            )
    table(
        "generation-d4",
        "20M held-out generation operating points.",
        ["Model", "Decoder", r"Repeat4 (\%)", "AR NLL", "Length", "TPF", "Tokens/s"],
        rows,
        "128 fresh documents, two sampling seeds, 256-token cap, temperature1. K1/K4 use top-p1; native settings are method-specific. "
        "All frozen points are shown, not a test-selected winner. NLL is from the same frozen 20M AR scorer; it is not semantic quality. "
        "Timing is isolated, uncached, batch1 decode-only on six prompts with three repeats, excluding loading/scoring. Timing prompts differ from the quality panel. "
        "B1 native duplicates K1 scientifically, with separate timing. No matched-quality speedup or AR parity is established.",
        "llrrrrr",
    )

    grids = data["batch-context-grid"]
    for family in ("ar", "mdlm"):
        rows = []
        for tpp in (20, 40, 80, 160):
            lrs = sorted({r["lr"] for r in grids["d4"] if r["family"] == family and r["tpp"] == tpp})
            assert len(lrs) == 3
            for lr in lrs:
                vals = [one(grids["d4"], family=family, tpp=tpp, lr=lr, eb=eb)["bpb"] for eb in (65536, 131072, 262144)]
                rows.append([str(tpp), f"{lr:.6g}", *[bpb(v) for v in vals]])
        table(
            f"{family}-grid",
            f"20M {family.upper()} AdamW batch/LR/horizon grid.",
            ["TPP", "Peak LR", "EB64K BPB", "EB128K BPB", "EB256K BPB"],
            rows,
            "All 36 cells, including reused historical endpoints; seed42, width256, context1024, DB16. "
            + (
                "AR uses exact causal content BPB and 30\\% cooldown. "
                if family == "ar"
                else "MDLM uses corpus NELBO and 10\\% cooldown under the historical objective, not the newer inverse-time Muon recipe. "
            )
            + "Each horizon has its own full schedule; 5\\% warmup. TPP uses 11,534,336 budget parameters (19,922,944 total). "
            "Training tokens are TPP times that count. Larger EB reduces updates at a fixed token budget. Upper-grid LR/batch winners are not established optima. "
            "Do not compare AR BPB with MDLM NELBO as identical likelihoods.",
            "rrrrr",
        )

    rows = []
    for r in grids["context"]:
        old = one(grids["d4"], cell=r["reference_cell"])
        rows.append(
            [
                r["family"].upper(),
                str(r["eb"] // 1024) + "K",
                f"{r['lr']:.6g}",
                bpb(old["bpb"]),
                bpb(r["bpb"]),
                bpb_delta(r["bpb"] - old["bpb"]),
            ]
        )
    table(
        "context",
        "20M training-context comparison at fixed source budget and EB.",
        ["Family", "EB", "Peak LR", "Train1024", "Train2048", "Delta BPB"],
        rows,
        "TPP20, seed42, AdamW, DB16; the same 1,024-token validation panel scores both training contexts. "
        "Doubling context halves gradient accumulation, not updates. Positive delta means training at 2048 is worse on this panel. "
        "AR and MDLM use different likelihood estimators. This does not measure the benefit of longer evaluation context or establish a context preference at 477M.",
        "llrrrr",
    )

    d12 = data["d12-muon"]
    rows = []
    for family, variants in [("ar", ("control", "low", "center")), ("mdlm", ("control", "lower", "low", "center"))]:
        for variant in variants:
            a, b = [one(d12["rows"], family=family, variant=variant, seed=s) for s in (43, 44)]
            rows.append(
                [
                    family.upper(),
                    str(a["eb"] // 1024) + "K",
                    f"{a['matrix_lr']:.5g}",
                    a["schedule"],
                    bpb(a["bpb"]),
                    bpb(b["bpb"]),
                ]
            )
    table(
        "d12-muon",
        "135M Muon batch/LR follow-up across two training seeds.",
        ["Family", "EB", "Matrix LR", "Momentum/decay", "Seed43 BPB", "Seed44 BPB"],
        rows,
        "Plain width768, context1024, TPP20: 2,202,009,600 training tokens. EB64K uses 33,600 updates and EB256K uses 8,400. "
        "MDLM at LR0.0035 holds optimizer settings fixed while changing batch/update count; AR candidates additionally change LR and momentum/decay schedules. "
        "All rows use Muon, so this is not Muon-versus-AdamW attribution. Seed43 informed selection; seed44 is fresh confirmation. "
        "The larger-batch MDLM advantage repeats, but no batch reversal at 477M has been measured.",
        "llrlrr",
    )

    new = data["d20-new"]
    core = data["d20-core"]
    corpus = data["d20-corpus"]
    oldgen = data["d20-generation"]
    settings = [data["d20-recipe-settings"][arm] for arm in ("native", "reconstruction", "muon_center")]
    budget = [f"{cfg['total_batch_size'] // 1024}K; {cfg['num_steps']:,}" for cfg in settings]
    schedule = [rf"{cfg['warmup_ratio']:.0%}; {cfg['warmdown_ratio']:.0%}".replace("%", r"\%") for cfg in settings]
    table(
        "d20-recipes",
        "Matched-size, matched-source-budget 477M recipe differences.",
        ["Factor", "Native NanoLLaDA replication", "NanoDLLM reconstruction", "Latest NanoDLLM Muon"],
        [
            ["Optimizer", "AdamW", "AdamW", "Muon + auxiliary AdamW"],
            ["Peak matrix LR", "0.0003", "0.0003", "0.00175 / 0.0035 / 0.007"],
            ["EB; updates", *budget],
            ["Warmup; cooldown", *schedule],
            [
                "Boundary targets",
                "BOS only; protect column0",
                "EOS trained; BOS protected",
                "EOS trained; BOS protected",
            ],
            ["Packing", "Best-fit; crop long documents", "Preserve document tails", "Preserve document tails"],
            ["Attention / positions", "Cross-document / continuous", "Same-document / reset", "Same-document / reset"],
            ["Clean-prefix rows", "None", r"10\%", r"10\%"],
            ["Loss denominator", "Actual batch times length", "Eligible target positions", "Eligible target positions"],
            ["Training seed", *[str(cfg["training_seed"]) for cfg in settings]],
        ],
        "All use the plain NanoLLaDA-style backbone, width1280, context1024, 477,102,080 total parameters and 435,159,040 TPP parameters; "
        "scratch TPP20 is 8,703,180,800 training tokens. ClimbMix/tokenizer identities match, but sample order, truncation and supervised positions differ. "
        "The historical Muon control uses 10\\% cooldown. Native means our upstream-code training replication, not the authors' checkpoint. "
        "This is not an optimizer-only comparison; Muon and AdamW nominal LRs are not equivalent update magnitudes.",
        "llll",
    )

    earlier_scores = _task_scores(core["task_rows"], "items")
    later_scores = _task_scores(new["task_rows"], "n")
    models = [
        ("Native", "native_nanollada_tpp20"),
        ("Reconstruction", "nanollada_tpp20"),
        ("Muon control (10\\%)", "control_cd10"),
        ("Muon low (30\\%)", "low"),
        ("Muon center (30\\%)", "center"),
        ("Muon high (30\\%)", "high"),
    ]
    rows, grows = [], []
    for label, key in models:
        old = key.endswith("tpp20") or key == "control_cd10"
        ck = "muon_tpp20" if key == "control_cd10" else key
        if old:
            corpus_bpb = statistics.mean(corpus["analysis"]["by_model"][ck]["values"])
            ag = earlier_scores["mdlm20" if key == "control_cd10" else key]
            g = oldgen["analysis"]["by_model"][ck]["all"]
        else:
            corpus_bpb = statistics.mean(new["training_nelbo"][key]["values"])
            ag = later_scores[key]
            g = new["generation_summary"][key]
        rows.append(
            [label, bpb(corpus_bpb), num(ag["mean_raw_accuracy"] * 100, 2), num(ag["mean_centered_v2"] * 100, 2)]
        )
        grows.append(
            [
                label,
                num(math.exp(g["micro"]["nll_sum"] / g["micro"]["scored_tokens"]), 2),
                num(g["repeated_4gram_fraction"]["mean"] * 100, 3),
                num(g["mean_content_tokens"], 2),
                str(g["eos_terminated"]) + "/256",
            ]
        )
    table(
        "d20-prediction",
        "477M common-corpus and task results.",
        ["Recipe", "Corpus NELBO BPB", r"Task raw (\%)", "Centered points"],
        rows,
        "The six recipes have the size/budget in Table~\\ref{tab:d20-recipes}. Corpus scores average four paired time/corruption plans on 1,024 frozen 1,024-token rows; "
        "EOS is excluded. Task scores use the corrected limited 21-task panel, not official full CORE. One training seed per recipe; historical controls are retained, not same-release repeats. "
        "LR labels are low0.00175/center0.0035/high0.007. Better corpus fit does not establish a task-aggregate win or optimizer attribution.",
        "lrrr",
    )
    scorer_params = data["scorer_parameters"]
    table(
        "d20-generation",
        "477M fixed-decoder generation diagnostics.",
        ["Recipe", "Reference PPL", r"Repeat4 (\%)", "Mean length", "EOS present"],
        grows,
        "128 paired validation prefixes, two sampling seeds, 256-token canvas, 64 forwards, temperature1 multinomial margin-jitter sampling (jitter0.02). "
        f"All outputs are scored by the same frozen {scorer_params / 1e6:.0f}M nanochat-port AR model. The full canvas is filled before truncating at the first EOS; EOS does not save compute. "
        "EOS generally occurs near the canvas end. Native has no explicit EOS supervision. These are predictability/repetition diagnostics, not semantic quality or useful stopping accuracy.",
        "lrrrr",
    )

    names = {
        "hellaswag_zeroshot": "HellaSwag zero-shot",
        "hellaswag": "HellaSwag few-shot",
        "bigbench_qa_wikidata": "Wikidata QA",
        "arc_easy": "ARC Easy",
        "arc_challenge": "ARC Challenge",
        "copa": "COPA",
        "commonsense_qa": "CommonsenseQA",
        "piqa": "PIQA",
        "openbook_qa": "OpenBookQA",
        "lambada_openai": "LAMBADA",
        "winograd": "Winograd",
        "winogrande": "WinoGrande",
        "bigbench_dyck_languages": "Dyck languages",
        "agi_eval_lsat_ar": "LSAT analytical reasoning",
        "bigbench_cs_algorithms": "CS Algorithms",
        "bigbench_operators": "Operators",
        "bigbench_repeat_copy_logic": "Repeat-copy logic",
        "squad": "SQuAD",
        "coqa": "CoQA",
        "boolq": "BoolQ",
        "bigbench_language_identification": "Language identification",
    }
    rows = []
    for r in new["task_rows"]:
        old = one(core["task_rows"], task=r["task"])
        assert r["n"] == old["items"]
        assert r["correct"]["control_cd10"] == old["correct"]["mdlm20"]
        values = [old["accuracy"][key] for key in ("native_nanollada_tpp20", "nanollada_tpp20")]
        values += [r["accuracy"][key] for key in ("control_cd10", "low", "center", "high")]
        label = names[r["task"]]
        rows.append([label, str(r["n"]), *[num(v * 100, 2) for v in values]])
    table(
        "d20-tasks",
        "477M per-task primary accuracies (percent).",
        ["Task", "Items", "Native", "Recon.", "Control", "Low", "Center", "High"],
        rows,
        "Same primary protocol as Table~\\ref{tab:d20-prediction}. Native and Recon. denote the two AdamW recipes; Control/Low/Center/High are Muon. "
        "The low-LR BoolQ gain is concentrated rather than broad and remains below the panel's 62.8\\% majority baseline. "
        "CommonsenseQA choice collapse affects both native and low-LR models; aggregate accuracy alone is not evidence that answer calibration is solved. "
        "Historical task counts and control accuracies were checked against the new replay. No tasks were removed after inspecting the results.",
        "lrrrrrrr",
    )
    return outputs
