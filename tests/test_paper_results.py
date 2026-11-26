import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

import pytest
from paper_results.__main__ import verify
from paper_results.formatting import bpb, bpb_delta, bpb_interval, one
from paper_results.reproduce import ROOT, load


@pytest.fixture(scope="module")
def rebuilt():
    return verify()


def test_exact_exhibit_inventory(rebuilt):
    outputs, _ = rebuilt
    manifest = json.loads((ROOT / "manifest.json").read_bytes())
    paper = manifest["paper_exhibits"]
    assert sorted(v["number"] for k, v in paper.items() if k.startswith("tables/")) == list(range(1, 14))
    assert sorted(v["number"] for k, v in paper.items() if k.startswith("figures/")) == list(range(1, 8))
    assert len(outputs) == 67
    for name, record in manifest["outputs"].items():
        assert hashlib.sha256(outputs[name].encode()).hexdigest() == record["sha256"]
        assert all(f"{k}.json" in manifest["inputs"] for k in record["inputs"])


def test_display_policy_and_decoder_example(rebuilt):
    outputs, stats = rebuilt
    assert bpb(None) == "---"
    assert bpb_delta(-0.00078) == "-0.0008"
    assert bpb_delta(0.001) == "+0.001"
    assert bpb_interval(-0.0236, 0.0005) == "[-0.0236, +0.0005]"
    assert r"$[-0.0236, +0.0005]$" in outputs["tables/component-budget-rows.tex"]
    marks = outputs["figures/evaluation-marks.tex"]
    assert r"\reuserepeatrange{22.4--26.0\%}" in marks
    assert r"\freshrepeatrange{0.02--0.04\%}" in marks
    assert r"\freshsampleppl{214.1}" in marks
    assert all(word not in marks for word in ("opening", "promptending"))
    assert stats["decoder_example_ppl"]["reuse"] == pytest.approx(8.4, abs=0.05)


def test_component_document_resampling(rebuilt):
    _, stats = rebuilt
    contrasts = stats["component_contrasts"]
    assert contrasts["bd3_optimized"]["mean_delta"] == pytest.approx(-0.052152949866247855, abs=1e-12)
    assert contrasts["bd3_optimized"]["paired_document_bootstrap_95"] == pytest.approx(
        [-0.06747791433115304, -0.037846529522050004], abs=1e-12
    )
    assert contrasts["full_budget"]["paired_document_bootstrap_95"][1] < 0
    assert contrasts["matched_compute"]["paired_document_bootstrap_95"][1] > 0
    assert contrasts["shorter_conversion"]["paired_document_bootstrap_95"][0] < 0
    assert contrasts["control_change"]["mean_delta"] + contrasts["shorter_conversion"]["mean_delta"] == pytest.approx(
        contrasts["matched_compute"]["mean_delta"] - contrasts["fastv2"]["mean_delta"], abs=1e-12
    )


def test_transfer_scopes_and_weights(rebuilt):
    outputs, stats = rebuilt
    assert len(stats["transfer"]["full_sequence"]) == 6
    assert all(row["delta_bpb"] < 0 for row in stats["transfer"]["full_sequence"])
    d8 = one(stats["transfer"]["block"], panel="holdout", depth=8, metric="common/block_end")
    assert d8["seeds"] == [50, 51, 52]
    assert d8["delta_bpb"] == pytest.approx(-0.057, abs=0.0005)
    assert "confidence_interval_bpb" not in d8
    assert r"\nexttokenmean" in outputs["figures/transfer-marks.tex"]
    d12 = stats["block_holdout"]["runs"]["d12"]["comparisons"]["conversion"]
    assert d12["common/block_end"]["delta_bpb"] < 0 < d12["common/legacy"]["delta_bpb"]
    sig = stats["diagnostics"]
    assert sig["fp32_cache_comparisons"] == 324
    assert sig["fp32_max_logit_error"] == 5.7220458984375e-5
    assert sig["fp32_max_logit_error"] <= 5.73e-5
    assert sig["trainer_replay_mean_bpb"]["matched_nanodllm"] == pytest.approx(1.899184, abs=5e-7)


def test_primary_tasks_and_shared_item_intervals(rebuilt):
    _, stats = rebuilt
    panels = load("block")["task_panels"]
    assert len(panels) == 21 and sum(p["items"] for p in panels.values()) == 8845
    counts = load("task-counts")
    for depth, expected in [
        ("d12", [0.1531568569456766, 1.8389972622270756]),
        ("d20", [-0.5617059889730694, 1.0776791876170757]),
    ]:
        groups = counts[depth]
        assert len(groups) == 20
        (joint,) = [g for g in groups if len(g["tasks"]) > 1]
        assert set(joint["tasks"]) == {"hellaswag", "hellaswag_zeroshot"}
        a = stats["block_tasks"]["depths"][depth]["aggregates"]["mean_nelbo__block_full_mask"]
        assert a["conditional_raw_95ci_pp"] == pytest.approx(expected, abs=1e-12)
        assert a["conditional_centered_v2_95ci_pp"][0] < 0 < a["conditional_centered_v2_95ci_pp"][1]


def test_scoring_denominators(rebuilt):
    outputs, stats = rebuilt
    row = one(stats["large_model_generation"], depth=20, recipe="scratch40", panel="fresh", decoder="k4_bl32")
    assert row["gemma_ppl"] == math.exp(row["nll_sum"] / row["scored_tokens"])
    assert row["gemma_bpb"] == row["nll_sum"] / (row["scored_bytes"] * math.log(2))
    assert "180.3 & 95.3 & 1.502 & 0.83 & 3/64" in outputs["tables/final-model-rows.tex"]


def test_standalone_cli_and_input_drift(tmp_path):
    # Exercise the distribution boundary with only the companion on sys.path.
    shutil.copytree(ROOT, tmp_path / "paper_results", ignore=shutil.ignore_patterns("__pycache__"))
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, "-B", "-m", "paper_results"]
    output_dir = tmp_path / "out"
    subprocess.run([*cmd, "--output", str(output_dir)], cwd=tmp_path, env=env, check=True)
    assert len(list(output_dir.rglob("*.tex"))) == 67
    files = {p: p.read_bytes() for p in output_dir.rglob("*") if p.is_file()}
    subprocess.run([*cmd, "--output", str(output_dir)], cwd=tmp_path, env=env, check=True)
    assert all(p.read_bytes() == expected for p, expected in files.items())
    (tmp_path / "paper_results/data/component-losses.json").write_text("{}")
    failed = subprocess.run([*cmd, "--check"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert failed.returncode != 0 and "pinned public snapshots" in failed.stderr
