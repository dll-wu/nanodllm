"""Matched block prediction, decoder, task and runtime tables."""

from statistics import mean

from .formatting import bpb, bpb_delta, bpb_interval
from .tables import add_table


def render(block):
    outputs = {}
    d8, large, core = block["d8"], block["large"], block["tasks"]
    task_rows, panels, grid = block["task_rows"], block["task_panels"], block["d8_discovery_grid"]
    comparisons = {"metrics": block["d4_paired_intervals"]}

    def table(name, cols, rows, caption, align, *, groups="", padding=4):
        add_table(outputs, "block", name, "", cols, rows, caption, align, groups=groups, padding=padding)

    selected = [
        r
        for r in d8["metrics"]
        if (r["recipe"] == "scratch40" and r["lr"] == 0.007) or (r["recipe"] == "AR10+block30" and r["lr"] == 0.00175)
    ]
    assert len(selected) == 6
    depths = (4, 8, 12, 20)
    params = {4: "19.92", 8: "58.72", 12: "135.27", 20: "477.10"}
    sizes = {depth: f"{float(count):.0f}M" for depth, count in params.items()}
    ar_common = {int(k): v for k, v in block["ar10_parent_common_bpb"].items()}
    assert set(ar_common) == set(depths)
    prediction = []
    for depth in depths:
        for recipe in ("scratch40", "AR10+MDLM30"):
            if depth == 4:
                which = "reference_bpb" if recipe == "scratch40" else "candidate_bpb"
                common = comparisons["metrics"]["common/block_end"][which]
                native = comparisons["metrics"]["native/mean_eval_seeds"][which]
                seeds = 1
            elif depth == 8:
                rows = [r for r in selected if r["recipe"] == ("AR10+block30" if recipe != "scratch40" else recipe)]
                common, native = mean(r["fresh_common"] for r in rows), mean(r["fresh_native"] for r in rows)
                seeds = len(rows)
            else:
                rows = [
                    r
                    for r in large["prediction"]
                    if r["depth"] == depth and r["recipe"] == recipe and r["panel"] == "fresh"
                ]
                assert len(rows) == 1
                common, native = rows[0]["common"], rows[0]["native"]
                seeds = 1
            prediction.append({"depth": depth, "recipe": recipe, "seeds": seeds, "common": common, "native": native})
    prediction_rows = []
    for depth in depths:
        prediction_rows.append([f"D{depth}", params[depth], "AR10 parent", "1", bpb(ar_common[depth]), "---"])
        for row in (r for r in prediction if r["depth"] == depth):
            label = "Scratch40" if row["recipe"] == "scratch40" else "AR10+block30"
            prediction_rows.append(
                [f"D{depth}", params[depth], label, str(row["seeds"]), bpb(row["common"]), bpb(row["native"])]
            )
    table(
        "block-prediction",
        ["Depth", "Params (M)", "Recipe", "Seeds", "Common BPB", "Native BPB"],
        prediction_rows,
        r"\textbf{Predictive scores on the balanced 32-document panel.} The two block diffusion recipes each use 40 tokens per counted parameter, a plain backbone, best-fit packing, mixed training block lengths 8/16/32/64 and Muon. The AR parent uses 10 tokens per counted parameter and has only a common next-token score. The 20M runs use seed 48. The 59M result averages child seeds 50--52 and shows its seed 50 AR parent. The 135M and 477M runs use seed 50. All diffusion models are evaluated at block length 32.",
        "lrllrr",
    )
    table(
        "block-seeds",
        ["Seed", "Recipe", "Matrix LR", "Common BPB", "Native BPB"],
        [
            [
                str(r["seed"]),
                "Scratch40" if r["recipe"] == "scratch40" else "AR10+block30",
                f"{r['lr']:g}",
                bpb(r["fresh_common"]),
                bpb(r["fresh_native"]),
            ]
            for r in sorted(d8["metrics"], key=lambda r: (r["recipe"], r["lr"], r["seed"]))
        ],
        "59M LR extension and confirmations, on the balanced development panel. Seed50 is discovery; seeds51/52 confirm the selected rates without per-seed retuning. Conversion0.0035 is retained as a different readout tradeoff, not omitted because it loses on common prediction. This table is the final bracket/confirmation subset, not the entire earlier EB/LR search.",
        "llrrr",
    )
    table(
        "block-d8-grid",
        ["Arm", "EB", "Matrix LR", "Common BPB", "Native BPB", "Child train h"],
        [
            [
                "AR10 parent" if r["profile"] == "parent" else "AR10+block30",
                f"{int(r['eb']):,}",
                r["lr"],
                bpb(float(r["fresh_bpb"])),
                bpb(float(r["fresh_native_nelbo"])) if r["fresh_native_nelbo"] else "---",
                f"{float(r['training_hours']):.2f}",
            ]
            for r in sorted(grid, key=lambda r: (r["profile"] != "parent", int(r["eb"]), float(r["lr"])))
        ],
        "59M conversion discovery and boundary extension, all seed50. The original transferred center was EB65,536/LR0.007. The selected common-score point is EB262,144/LR0.00175, at the smaller-LR/larger-batch edge; it is best tested, not bracketed as a joint optimum. Native NELBO instead prefers0.0035 on this panel. Children share the AR10 parent (shown for capability context, not equal total compute) and reset optimizer/stream state. Training hours exclude parent and evaluation; hardware is mixed A100/A800, so these are observational timings. Subsequent scratch and fresh-seed results appear in Table~\\ref{tab:block-seeds}.",
        "lrrrrr",
    )
    cirows = [
        [
            sizes[4],
            "48",
            label,
            bpb_delta(comparisons["metrics"][key]["delta_bpb"]),
            bpb_interval(*comparisons["metrics"][key]["confidence_interval_bpb"]),
        ]
        for label, key in (("common", "common/block_end"), ("native", "native/mean_eval_seeds"))
    ]
    cirows += [
        [sizes[8], str(r["seed"]), r["metric"], bpb_delta(r["delta"]), bpb_interval(*r["ci95"])]
        for r in d8["paired"]
        if r["conversion_lr"] == 0.00175
    ]
    cirows += [
        [sizes[r["depth"]], "50", r["metric"], bpb_delta(r["delta"]), bpb_interval(*r["ci95"])]
        for r in large["paired"]
        if r["panel"] == "fresh"
    ]
    table(
        "block-intervals",
        ["Size", "Seed", "Readout", r"$\Delta$ BPB", "Document 95\\% interval"],
        cirows,
        "Conversion minus scratch on the balanced panel. Intervals use paired length-stratified document bootstrap replicates at fixed checkpoints (2,000 at 20M and 5,000 at larger sizes). They do not estimate training-seed population uncertainty or correct for repeated selection/multiple comparisons. Both panels are already opened development data.",
        "llrrr",
    )

    task_summary, controls = [], []
    for depth, info in core["depths"].items():
        p = info["aggregates"]["mean_nelbo__block_full_mask"]
        task_summary.append(
            [
                sizes[int(depth[1:])],
                f"{100 * p['scratch']['mean_raw_accuracy']:.2f}",
                f"{100 * p['conversion']['mean_raw_accuracy']:.2f}",
                f"{p['delta_raw_pp']:+.2f}",
                f"[{p['conditional_raw_95ci_pp'][0]:+.2f}, {p['conditional_raw_95ci_pp'][1]:+.2f}]",
                f"{100 * p['scratch']['mean_centered_v2']:.2f}",
                f"{100 * p['conversion']['mean_centered_v2']:.2f}",
            ]
        )
        for name, p in info["aggregates"].items():
            label = (
                name.replace("mean_nelbo", "Mean")
                .replace("sum_nelbo", "Sum")
                .replace("block_full_mask", "full mask")
                .replace("sequential_masked", "sequential")
                .replace("__", " + ")
            )
            controls.append(
                [
                    sizes[int(depth[1:])],
                    label,
                    f"{100 * p['scratch']['mean_raw_accuracy']:.3f}",
                    f"{100 * p['conversion']['mean_raw_accuracy']:.3f}",
                    f"{100 * p['scratch']['mean_centered_v2']:.3f}",
                    f"{100 * p['conversion']['mean_centered_v2']:.3f}",
                    f"[{p['conditional_centered_v2_95ci_pp'][0]:+.2f}, {p['conditional_centered_v2_95ci_pp'][1]:+.2f}]",
                ]
            )
    table(
        "block-tasks-main",
        ["Size", "Scratch \\%", "Convert \\%", r"$\Delta$ pp", "Raw 95\\% interval", "Scratch cent.", "Convert cent."],
        task_summary,
        r"\textbf{Task results at 135M and 477M under the native block protocol.} Equal-task averages over21 tasks, capped at500 items/task (8,845 items/model). Primary scorer: mean-token conditional NELBO for ranking and block-full-mask exact completion. Centered scores use fixed per-task baselines and are multiplied by100. Paired item intervals condition on these checkpoints and corruption draws; tasks are fixed and overlapping HellaSwag variants resampled jointly. Both centered-score difference intervals include zero. This is not official full CORE, free generation, or the older full-sequence task protocol.",
        "lrrrrrr",
    )
    table(
        "block-task-controls",
        ["Size", "Scorers", "Scratch \\%", "Convert \\%", "S cent.", "C cent.", r"Cent. $\Delta$ 95\%"],
        controls,
        "Prespecified ranking/completion scoring controls. All four point differences favor conversion, but every raw/centered interval at 477M and every centered interval at 135M spans zero. The controls reuse predictions and are not independent replications. The complete aggregate records also retain raw-score intervals.",
        "llrrrrr",
    )
    names = {
        "bigbench_qa_wikidata": "Wikidata QA",
        "arc_easy": "ARC Easy",
        "arc_challenge": "ARC Challenge",
        "commonsense_qa": "CommonsenseQA",
        "openbookqa": "OpenBookQA",
        "hellaswag_zeroshot": "HellaSwag (zero)",
        "hellaswag": "HellaSwag",
        "lambada_openai": "LAMBADA",
        "winograd": "Winograd",
        "bigbench_cs_algorithms": "CS Algorithms",
        "bigbench_dyck_languages": "Dyck languages",
        "bigbench_repeat_copy_logic": "Repeat-copy logic",
        "bigbench_language_identification": "Language ID",
        "bigbench_logical_deduction": "Logical deduction",
        "bigbench_operators": "Operators",
        "bigbench_conceptual_combinations": "Conceptual combinations",
    }
    names.update(
        {
            "squad": "SQuAD",
            "coqa": "CoQA",
            "copa": "COPA",
            "piqa": "PIQA",
            "openbook_qa": "OpenBookQA",
            "winogrande": "WinoGrande",
            "agi_eval_lsat_ar": "LSAT analytical",
            "boolq": "BoolQ",
            "hellaswag": "HellaSwag (few)",
        }
    )
    tasks = []
    for a in [r for r in task_rows if r["depth"] == "d12"]:
        b = next(r for r in task_rows if r["depth"] == "d20" and r["task"] == a["task"])
        name = names.get(a["task"], a["task"].replace("_", " "))
        tasks.append(
            [
                name,
                "Exact" if a["type"] == "language_modeling" else "Rank",
                a["items"],
                f"{100 * panels[a['task']]['baseline_v2']:.2f}",
                *[f"{100 * float(r[k]):.2f}" for r in (a, b) for k in ("scratch_accuracy", "conversion_accuracy")],
            ]
        )
    table(
        "block-tasks",
        ["Task", "Type", "Items", r"$b_j$ \%", "135M S \\%", "135M C \\%", "477M S \\%", "477M C \\%"],
        tasks,
        "Complete primary21-task panel for current block models. S: scratch40; C: AR10+block30. The fixed baseline b used for centering is listed explicitly. Exact is gold-length token-exact completion with earlier blocks gold; Rank is candidate ranking. Neither measures free-running answer termination. Values across depths share examples and templates. Jeopardy is excluded. HellaSwag few/zero-shot are two scores on overlapping examples.",
        "llrrrrrr",
    )

    def grow(r):
        return [
            sizes[r["depth"]],
            "S" if r["recipe"] == "scratch40" else "C",
            r["decoder"].replace("_", "/"),
            f"{r['gemma_ppl']:.2f}",
            bpb(r["gemma_bpb"]),
            f"{100 * r['repeat4']:.2f}",
            str(r["eos_count"]),
            f"{r['content_tokens']:.1f}",
            f"{r['tpf']:.2f}",
        ]

    g = [r for r in large["generation"] if r["panel"] == "fresh"]
    caption = r"Gemma4 31B base scores generated text causally; no chat template or terminal EOS is scored. Each row pools64 generations (32 prompts, two sampling seeds), cap256, temperature1. K1 uses top-p1; K4 uses top-p0.9. PPL is token-weighted under the reference tokenizer, BPB byte-weighted. Repeat4 is mean repeated-four-gram fraction, EOS is count/64, length excludes EOS. Tokens/forward (TPF) includes block overhead. Reference likelihood is not semantic quality; K1/K4 changes both sampling and parallelism. S: scratch40; C: AR10+block30."
    table(
        "block-generation",
        ["Size", "Arm", "Decoder", "PPL", "BPB", "Rep. \\%", "EOS", "Length", "TPF"],
        [grow(r) for r in g if r["decoder"] in ("k1_bl32", "k4_bl32")],
        r"\textbf{Generation of the final block models.} " + caption,
        "llrrrrrrr",
    )
    table(
        "block-generation-lengths",
        ["Size", "Arm", "Decoder", "PPL", "BPB", "Rep. \\%", "EOS", "Length", "TPF"],
        [grow(r) for r in g if r["decoder"] not in ("k1_bl32", "k4_bl32")],
        "Runtime block-length controls. " + caption,
        "llrrrrrrr",
    )
    table(
        "block-d8-generation",
        ["Seed", "Arm", "Decoder", "Gemma PPL", "Rep. \\%", "EOS\\%", "Length", "TPF"],
        [
            [
                str(r["seed"]),
                "S" if r["recipe"] == "scratch40" else "C",
                r["decoder"].replace("_", "/"),
                f"{r['ppl']:.2f}",
                f"{100 * r['repeat4']:.2f}",
                f"{100 * r['eos']:.2f}",
                f"{r['tokens']:.1f}",
                f"{r['tpf']:.2f}",
            ]
            for r in d8["generation"]
            if r["decoder"] in ("k1_bl32", "k4_bl32")
            and (
                (r["recipe"] == "scratch40" and r["lr"] == 0.007) or (r["recipe"] != "scratch40" and r["lr"] == 0.00175)
            )
        ],
        "59M selected-recipe generation by training seed. Same reference and64-output protocol as Table~\\ref{tab:block-generation}. Conversion does not consistently improve either K1 or K4 PPL; its lowest K1 PPL coincides with increased repetition. Additional block lengths and alternate-LR outputs are retained in the portable data.",
        "llrrrrrr",
    )
    table(
        "block-runtime",
        ["Depth", "Arm", "Params M", "Budget M", "Child updates", "Child tokens B", "Attempts", "Job hours"],
        [
            [
                f"D{r['depth']}",
                "S" if r["recipe"] == "scratch40" else "C",
                f"{r['physical_params'] / 1e6:.3f}",
                f"{r['tpp_params'] / 1e6:.3f}",
                f"{r['child_updates']:,}",
                f"{r['child_tokens'] / 1e9:.4f}",
                str(r["attempts"]),
                f"{r['allocation_elapsed_hours']:.2f}",
            ]
            for r in large["runtime"]
        ],
        "Actual child-stage ledger on eight GPUs. Job hours sum attempts and include startup, evaluation, checkpointing and replay after time limits; they exclude queueing and AR-parent training. Multiplying by eight gives occupied GPU-hours for these child jobs, not total recipe training cost. S receives40TPP; C adds an AR10 parent to the displayed30TPP child. Parent-inclusive token budgets match to full-update rounding. DB16, EB262144, accumulation2 at both depths. Peak allocated memory is about19GiB at 135M and 45.2GiB at 477M, not total device reservation.",
        "llrrrrrr",
    )

    # A small dependency-free scientific figure: two separately labelled estimands.
    fig = [r"\begin{tikzpicture}[x=1cm,y=1cm,font=\small]"]
    for j, (metric, title) in enumerate((("common", "Common next-token BPB"), ("native", "Native block-bound BPB"))):
        x0 = j * 7.9
        fig += [f"\\begin{{scope}}[shift={{({x0},0)}}]", f"\\node[font=\\bfseries] at (3,3.75) {{{title}}};"]

        def ordinate(value):
            return 0.35 + (value - 0.85) * 6.2

        for val in (0.9, 1.0, 1.1, 1.2, 1.3):
            y = ordinate(val)
            fig += [
                f"\\draw[gray!20] (0.6,{y:.3f}) -- (6.6,{y:.3f});",
                f"\\node[anchor=east] at (0.48,{y:.3f}) {{{val:.1f}}};",
            ]
        fig += [r"\draw[gray] (0.6,0.32)--(0.6,3.37);", r"\draw[gray] (0.6,0.32)--(6.6,0.32);"]
        positions = {4: 1, 8: 2.7, 12: 4.4, 20: 6.1}
        for dep, x in positions.items():
            fig.append(f"\\node[align=center,anchor=north] at ({x},0.22) {{{sizes[dep]}}};")
        series = [
            ("scratch40", "blue!70!black", "circle", "solid"),
            ("AR10+MDLM30", "teal!80!black", "rectangle", "solid"),
        ]
        if metric == "common":
            series.append(("AR10", "black!65", "diamond", "dashed"))
        for recipe, color, mark, line in series:
            values = (
                ar_common if recipe == "AR10" else {r["depth"]: r[metric] for r in prediction if r["recipe"] == recipe}
            )
            points = [(x, ordinate(values[dep])) for dep, x in positions.items()]
            fig.append(f"\\draw[thick,{color},{line}] " + " -- ".join(f"({x},{y:.6f})" for x, y in points) + ";")
            for x, y in points:
                fill = "white" if recipe == "scratch40" else color
                size = "2.4pt" if recipe == "scratch40" else "1.6pt"
                fig.append(f"\\node[draw={color},fill={fill},{mark},inner sep={size}] at ({x},{y:.6f}) {{}};")
        fig.append(r"\end{scope}")
    for x, label, color, mark, line, fill in (
        (1.7, "Scratch40", "blue!70!black", "circle", "solid", "white"),
        (5.6, "AR10+block30", "teal!80!black", "rectangle", "solid", "teal!80!black"),
        (10.4, "AR10 parent", "black!65", "diamond", "dashed", "black!65"),
    ):
        fig += [
            f"\\draw[thick,{color},{line}] ({x},-.78)--({x + 0.6},-.78);",
            f"\\node[draw={color},fill={fill},{mark},inner sep=1.8pt] at ({x + 0.3},-.78) {{}};",
            f"\\node[anchor=west] at ({x + 0.65},-.78) {{{label}}};",
        ]
    fig.append(r"\end{tikzpicture}")
    outputs["figures/block-scaling.tex"] = "\n".join(fig) + "\n"
    holdout = block["holdout"]
    holdout_rows = []
    profiles = (
        ("d4", "20M", "48"),
        ("d8-s50", "59M", "50"),
        ("d8-s51", "59M", "51"),
        ("d8-s52", "59M", "52"),
        ("d12", "135M", "50"),
        ("d20", "477M", "50"),
    )
    for profile, size, seed in profiles:
        run = holdout["runs"][profile]
        for arm, metrics in run["comparisons"].items():
            rate = (
                "0.007" if profile == "d4" else {"lr0035": "0.0035", "lr00175": "0.00175", "conversion": "0.00175"}[arm]
            )
            row = [size, seed, rate]
            for metric in holdout["primary_metrics"]:
                m = metrics[metric]
                row.extend(
                    [
                        bpb(m["reference_bpb"]),
                        bpb(m["candidate_bpb"]),
                        "$" + bpb_delta(m["delta_bpb"]) + "$",
                        "$" + bpb_interval(*m["confidence_interval_bpb"]) + "$",
                    ]
                )
            holdout_rows.append(row)
    table(
        "block-holdout",
        [
            "Size",
            "Seed",
            "Conv. LR",
            "Scratch",
            "Conv.",
            r"$\Delta$",
            r"95\% CI",
            "Scratch",
            "Conv.",
            r"$\Delta$",
            r"95\% CI",
        ],
        holdout_rows,
        r"\textbf{Frozen-checkpoint replay on the same 256 held-out documents.} Scores are bits per byte; differences are conversion minus scratch. Intervals use 2,000 paired document resamples within source-length strata, conditional on the weights and fixed evaluation noise. The scratch learning rate is 0.007 throughout.",
        "llrrrrrrrrr",
        padding=3,
        groups=(
            r" & & & \multicolumn{4}{c}{Block-aligned next token} & \multicolumn{4}{c}{Native reconstruction} \\"
            "\n" + r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}" + "\n"
        ),
    )
    return outputs
