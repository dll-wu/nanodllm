"""Fixed block CORE readouts. All alternatives use the same candidate-independent mask seeds."""

from __future__ import annotations

import math
import statistics

from ..runtime.core_mask_plans import NelboOptions, make_mask_plan, reduce_mask_plan
from ..runtime.core_prompting import TargetSequence
from ..runtime.core_task_data import json_digest

OPTIONS = NelboOptions(exact_max_tokens=7, mc_draws=32, mc_replicates=4, seed=1337)


def rank_candidates(adapter, example, sample_id):
    candidates, plans = [], []
    for sequence in example.sequences:
        group_totals, group_plans = [], []
        for i, group in enumerate(adapter.target_groups(sequence)):
            plan = make_mask_plan(group.end - group.start, OPTIONS, sample_id=sample_id, group=i)
            losses = []
            for start in range(0, len(plan.draws), 8):
                losses.extend(adapter.masked_means(sequence, group, plan.draws[start : start + 8]))
            group_totals.append(reduce_mask_plan(plan, tuple(losses)))
            group_plans.append(plan.sha256)
        repeats = max(map(len, group_totals))
        total = statistics.mean(math.fsum(g[j] if len(g) > 1 else g[0] for g in group_totals) for j in range(repeats))
        candidates.append({"total_nelbo": total, "mean_nelbo": total / (sequence.end - sequence.start)})
        plans.append(group_plans)
    total_choice = min(range(len(candidates)), key=lambda i: candidates[i]["total_nelbo"])
    mean_choice = min(range(len(candidates)), key=lambda i: candidates[i]["mean_nelbo"])
    return {
        "candidates": candidates,
        "mask_plan_sha256": json_digest(plans),
        "mean_choice": mean_choice,
        "sum_choice": total_choice,
        "correct": {"mean_nelbo": mean_choice == example.gold, "sum_nelbo": total_choice == example.gold},
    }


def exact_completion(adapter, sequence):
    gold = sequence.tokens[sequence.start : sequence.end]
    full = tuple(t for g in adapter.target_groups(sequence) for t in adapter.masked_predictions(sequence, g))
    sequential = []
    for pos in range(sequence.start, sequence.end):
        step = TargetSequence(sequence.tokens[: pos + 1], pos, pos + 1, sequence.cropped_tokens)
        sequential.extend(adapter.masked_predictions(step, adapter.target_groups(step)[0]))
    return {
        "correct": {"block_full_mask": full == gold, "sequential_masked": tuple(sequential) == gold},
        "target_tokens": len(gold),
    }


def score_panel(adapter, panel):
    task = panel.task
    rows = []
    for item in panel.examples:
        if task.task_type == "language_modeling":
            result = exact_completion(adapter, item.example.sequences[0])
        else:
            result = rank_candidates(adapter, item.example, item.sample_id)
        rows.append({"row": item.original_index, "demonstration_rows": item.demonstration_indices, **result})
    return {
        "task_type": task.task_type,
        "data_sha256": task.data_sha256,
        "num_fewshot": task.num_fewshot,
        "baseline_v1": panel.baseline_v1,
        "baseline_v2": panel.baseline_v2,
        "selection_sha256": panel.selection_sha256,
        "prepared_inputs_sha256": panel.prepared_inputs_sha256,
        "rows": rows,
        "accuracy": {k: statistics.mean(r["correct"][k] for r in rows) for k in rows[0]["correct"]},
    }
