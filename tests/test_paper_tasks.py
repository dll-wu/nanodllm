import copy
import json
from dataclasses import replace

import pytest
from nanodllm.paper_runtime.core_prompting import PreparedExample, TargetSequence, fewshot_indices, task_indices
from nanodllm.paper_runtime.data import ByteTokenizer
from nanodllm.paper_tasks import main
from nanodllm.paper_tasks_aggregate import paired
from nanodllm.paper_tasks_data import PAPER_ROSTER, prepare_panel, read_roster, retain_bos
from nanodllm.paper_tasks_scoring import exact_completion, rank_candidates
from scripts.paper_tasks_demo import create


def test_roster_counts_and_corrected_boundaries(tmp_path):
    rows = read_roster(PAPER_ROSTER)
    assert len(rows) == 21 and sum(s["selected_count"] for s in rows) == 8845
    assert len({s["data_sha256"] for s in rows}) == 20
    assert sum(s["task_type"] == "language_modeling" for s in rows) == 8
    assert "jeopardy" not in {s["label"] for s in rows}
    assert next(s["delimiter"] for s in rows if s["label"] == "arc_easy") == "\nAnswer: "
    create(tmp_path)
    spec = next(s for s in read_roster(tmp_path / "roster.json") if s["label"] == "toy_choices")
    panel = prepare_panel(spec, tmp_path, ByteTokenizer(), context=23, block_size=8, limit=1)
    (row,) = panel.examples
    pool = task_indices(3, 500)
    assert row.original_index == pool[0]
    assert row.demonstration_indices == (pool[fewshot_indices(3, 0, 1)[0]],)
    for s in row.example.sequences:
        assert s.cropped_tokens > 0 and s.cropped_tokens % 8 == 0
        assert s.tokens[0] == 256 and s.end - s.start == 1
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_panel({**spec, "data_sha256": "a" * 64}, tmp_path, ByteTokenizer())


def test_crop_cannot_replace_answer():
    seq = TargetSequence((1, 2, 3), 1, 3, 2)
    ex = PreparedExample("language_modeling", (seq,), 256, None)
    with pytest.raises(ValueError, match="unaligned"):
        retain_bos(ex, 8)
    with pytest.raises(ValueError, match="restore BOS"):
        retain_bos(replace(ex, sequences=(replace(seq, cropped_tokens=0),)), 8)


class Oracle:
    def target_groups(self, seq):
        from nanodllm.paper_runtime.core_diffusion_adapters import TargetGroup

        groups = []
        start = seq.start
        while start < seq.end:
            end = min((start // 8 + 1) * 8, seq.end)
            groups.append(TargetGroup(start, end))
            start = end
        return groups

    def masked_means(self, seq, group, draws):
        return (float(seq.tokens[1]),) * len(draws)

    def masked_predictions(self, seq, group):
        return seq.tokens[group.start : group.end]


def test_mean_and_sum_rankings_and_gold_length():
    # Constant masked CE per token makes the expectation analytic, including MC groups.
    long = TargetSequence((256, 1, *range(3, 16)), 1, 15)
    short = TargetSequence((256, 2), 1, 2)
    example = PreparedExample("multiple_choice", (long, short), 256, 0)
    result = rank_candidates(Oracle(), example, "example:0")
    assert result["candidates"] == [{"mean_nelbo": 1, "total_nelbo": 14}, {"mean_nelbo": 2, "total_nelbo": 2}]
    assert result["correct"] == {"mean_nelbo": True, "sum_nelbo": False}
    assert exact_completion(Oracle(), long) == {
        "correct": {"block_full_mask": True, "sequential_masked": True},
        "target_tokens": 14,
    }


def test_score_compare_portable_weights_and_pairing(tmp_path):
    create(tmp_path)
    for seed in (11, 12):
        path = tmp_path / f"{seed}.json"
        assert (
            main(
                [
                    "score",
                    "--bundle",
                    str(tmp_path),
                    "--roster",
                    str(tmp_path / "roster.json"),
                    "--checkpoint",
                    str(tmp_path / f"demo-{seed}.weights.pt"),
                    "--output",
                    str(path),
                ]
            )
            == 0
        )
    left, right = [json.loads((tmp_path / f"{s}.json").read_text()) for s in (11, 12)]
    assert not left["full_paper_panel"]
    assert left["settings"]["estimator"] == dict(seed=1337, mc_draws=32, mc_replicates=4, exact_max_tokens=7)
    assert set(left["tasks"]) == {"toy_choices", "toy_schema", "toy_completion"}
    assert "Locker" not in json.dumps(left)
    assert (
        main(
            [
                "compare",
                str(tmp_path / "11.json"),
                str(tmp_path / "12.json"),
                "--draws",
                "25",
                "--output",
                str(tmp_path / "pair.json"),
            ]
        )
        == 0
    )
    comparison = json.loads((tmp_path / "pair.json").read_text())
    assert comparison["items"] == 9 and comparison["seed"] == 20261119
    broken = copy.deepcopy(right)
    broken["tasks"]["toy_choices"]["rows"][0]["mask_plan_sha256"] = "changed"
    with pytest.raises(ValueError, match="unaligned"):
        paired(left, broken)


def test_overlap_preserving_bootstrap_and_archive_input(tmp_path):
    import zipfile

    create(tmp_path)
    spec = read_roster(tmp_path / "roster.json")[0]
    path = tmp_path / spec["data_path"]
    spec = {**spec, "data_bytes": path.stat().st_size}
    archive = tmp_path / "tasks.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(path, "package/" + spec["data_path"])
        z.writestr("unused.jsonl", "unused")
    local = prepare_panel(spec, tmp_path, ByteTokenizer())
    assert prepare_panel(spec, archive, ByteTokenizer()) == local
    with pytest.raises(ValueError, match="wrong-sized"):
        prepare_panel({**spec, "data_bytes": 0}, archive, ByteTokenizer())

    def task(correct):
        return dict(
            rows=[
                {"row": i, "demonstration_rows": [], "correct": {"mean_nelbo": v, "sum_nelbo": v}}
                for i, v in enumerate(correct)
            ],
            task_type="multiple_choice",
            num_fewshot=0,
            data_sha256="shared",
            selection_sha256="same",
            prepared_inputs_sha256="same",
            baseline_v1=0.5,
            baseline_v2=0.5,
        )

    report = dict(
        settings={"task_order": ["a", "b"]},
        schema="nanodllm.paper_tasks.v1",
        model=dict(depth=12, tokenizer_content_sha256="same", checkpoint_sha256="demo"),
    )
    left = {**report, "tasks": {"a": task([True, False]), "b": task([False, True])}}
    right = {**report, "tasks": {"a": task([False, True]), "b": task([True, False])}}
    result = paired(left, right, draws=127)
    assert result["results"]["mean_nelbo__block_full_mask"]["raw"]["ci95_pp"] == [0, 0]
    assert result["primary_by_task"]["a"]["gained"] == result["primary_by_task"]["a"]["lost"] == 1
