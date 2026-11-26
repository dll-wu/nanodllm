import json
from pathlib import Path

import numpy as np
from paper.companion.reproduce import ROOT, load


def test_public_snapshot_boundary():
    # Science keys such as "common/block_end" are metric names. File locations,
    # free-text examples and model input IDs have no role in these reductions.
    excluded = {
        "prefix_text",
        "text",
        "prompt_tail",
        "opening",
        "input_ids",
        "token_ids",
        "document_id",
        "prompt_id",
        "gold",
        "demonstration_rows",
        "private_item_scores",
        "checkpoint",
        "source_commit",
        "git_head",
        "host",
        "hostname",
        "run_id",
        "nanochat_checkout",
        "path",
        "workdir",
        "log",
        "logs",
    }

    def inspect(value):
        if isinstance(value, dict):
            assert value.keys().isdisjoint(excluded)
            for content in value.values():
                inspect(content)
        elif isinstance(value, list):
            for content in value:
                inspect(content)
        elif isinstance(value, str):
            assert not Path(value).is_absolute() and "://" not in value

    for path in sorted((ROOT / "data").glob("*.json")):
        inspect(json.loads(path.read_bytes()))


def test_document_snapshots_are_paired_and_anonymous():
    data = load("block-losses")
    for profiles in (data["development"].values(), [p for arms in data["holdout"].values() for p in arms.values()]):
        for pair in profiles:
            n = len(pair["strata"])
            assert n in (32, 256)
            for metric in pair["metrics"].values():
                assert metric.keys() == {"bytes", "reference_nats", "candidate_nats"}
                assert all(len(values) == n for values in metric.values())
                assert all(type(v) is int and v > 0 for v in metric["bytes"])


def test_development_and_alternate_rate_coverage():
    dev = load("development")
    assert len(dev["development_path"]) == 6
    assert [r["ar_tpp"] for r in dev["allocation_grid"]] == [0, 5, 10, 20, 30, 35]
    assert {r["physical_parameters"] for r in dev["development_path"]} == {19922944, 36700242}
    data = load("block")
    assert len(data["d8"]["generation"]) == 48
    assert len(data["d8_discovery_grid"]) == 11
    assert {r["panel"] for r in data["large"]["generation"]} == {"opened", "fresh"}
    assert len(data["settings"]) == 4


def test_archived_task_draws_only_contain_resampling_counts():
    groups = load("task-counts")
    expected = {
        "__".join([depth, *group["tasks"]]): sum(outcome["items"] for outcome in group["outcomes"])
        for depth, datasets in groups.items()
        for group in datasets
    }
    with np.load(ROOT / "data" / "task-draws.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(expected)
        for key, items in expected.items():
            counts = archive[key]
            assert counts.dtype.kind == "u" and counts.ndim == 2 and len(counts) == 10000
            assert np.all(counts.sum(axis=1) == items)
