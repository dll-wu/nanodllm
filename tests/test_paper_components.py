"""Portable component presets, endpoint readout and paired evidence reductions."""

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from nanodllm.paper_components import main, resolve
from nanodllm.paper_components_eval import compare
from nanodllm.paper_components_train import dense_child_flops
from nanodllm.paper_eval_panels import prepare
from nanodllm.paper_runtime.data import ByteTokenizer
from nanodllm.paper_weights import export_weights, file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/components"


@pytest.mark.parametrize("seed", [48, 49])
def test_reference_budgets_and_parent_resets(seed):
    arms = {
        "bd3": {"fixed": 3520, "calibrated": 3520},
        "llada1": {"mdlm": 3520, "masking": 3520},
        "llada2": {"fixed": 4928, "curriculum": 4928},
        "ar10": {"parent": 1760},
        "fastv2": {"full": 5280, "token-causal": 5280, "block-causal": 5280, "matched": 1817},
    }
    for study, pairs in arms.items():
        for arm, steps in pairs.items():
            main_stage, *rest = resolve(CONFIG / (study + ".json"), arm, seed, "run")
            cfg = main_stage.config
            assert cfg.runtime.seed == cfg.runtime.model_seed == cfg.runtime.training_seed == seed
            assert cfg.train.num_steps == steps and cfg.train.target_tokens == steps * 65536
            assert cfg.model.depth == cfg.model.n_head == 4 and cfg.model.n_embd == 256
            if rest:
                (tail,) = rest
                assert study == "llada2" and tail.objective == "B1"
                assert tail.policy["clean_stream_mode"] == main_stage.policy["clean_stream_mode"] == "block_causal"
                assert tail.config.train.num_steps == 352 and tail.config.data.shuffle_seed == 319180587
                for key in ("lr", "nanochat_head_lr", "nanochat_embedding_lr"):
                    assert getattr(tail.config.optim, key) == getattr(cfg.optim, key) / 4


def test_fast_compute_horizon_from_physical_work():
    fixed = [dense_child_flops(692060160, n) for n in (173336969, 173214952)]
    fast = [dense_child_flops(x, y) for x, y in ((2210693120, 328706220), (2210029568, 328443530))]
    assert math.floor(5280 * sum(fixed) / sum(fast)) == 1817


def test_compact_evidence_reproduces_all_six_paired_comparisons():
    from paper_results.intervals import component_intervals

    compact = json.loads((ROOT / "paper_results/data/component-losses.json").read_text())
    published = component_intervals(compact)
    names = {
        "bd3_fixed": "bd3/fixed",
        "bd3_optimized": "bd3/calibrated",
        "llada1": "llada1/masking",
        "mdlm": "llada1/mdlm",
        "new_curriculum": "llada2/curriculum",
        "new_fixed": "llada2/fixed",
        "fastv2": "fastv2/full",
        "short_fastv2": "fastv2/matched",
        "revised_fixed": "fastv2/block-causal",
        "block_fixed": "fastv2/token-causal",
    }

    def reports(key):
        return [
            {
                "arm": names[key],
                "seed": s,
                "panel_sha256": "paired",
                "ar_parent_sha256": None if names[key].split("/")[0] in {"bd3", "llada1"} else str(s),
                "phase": "tail" if names[key].startswith("llada2") else "main",
                "documents": [
                    {"id": f"document-{i:02d}", "content_bytes": n, "common_nats": {"block_end": loss}}
                    for i, (n, loss) in enumerate(zip(compact["content_bytes"], values, strict=True))
                ],
            }
            for s, values in zip(compact["seeds"], compact["content_nll_nats"][key], strict=True)
        ]

    for left, right, contrast in [
        ("bd3_optimized", "bd3_fixed", "bd3_optimized"),
        ("llada1", "mdlm", "llada1"),
        ("new_curriculum", "new_fixed", "new_curriculum"),
        ("fastv2", "block_fixed", "fastv2"),
        ("fastv2", "revised_fixed", "full_budget"),
        ("short_fastv2", "revised_fixed", "matched_compute"),
    ]:
        result = compare(reports(left), reports(right))
        assert result["paired_document_95"] == pytest.approx(published[contrast]["paired_document_bootstrap_95"])
        assert result["mean_delta_bpb"] == pytest.approx(published[contrast]["mean_delta"])
    wrong = reports("fastv2")
    wrong[1]["ar_parent_sha256"] = "different"
    with pytest.raises(ValueError, match="parent"):
        compare(wrong, reports("revised_fixed"))


@pytest.fixture
def tiny(tmp_path):
    """Use a miniature native Muon model, local byte tokenizer and original fixture text."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    texts = ["warm silver paper " * 14, "the quiet snowy meadow " * 40]
    pq.write_table(pa.table({"text": texts * 3}), train)
    pq.write_table(pa.table({"text": texts}), val)
    base = json.loads((CONFIG / "base.json").read_text())
    base["data"].update(
        tokenizer="byte",
        tokenizer_path="",
        train_data=[str(train)],
        val_data=[str(val)],
        train_exclude_data=[],
        bos_packing_buffer_size=4,
        tokenizer_batch_size=2,
        continuation_ratio=0,
    )
    base["model"].update(depth=1, n_embd=16, n_head=2, mlp_hidden_dim=64, max_seq_len=320)
    base["train"].update(
        device_batch_size=1, total_batch_size=320, target_tokens=320, num_steps=1, telemetry_interval_seconds=0
    )
    base["runtime"].update(device_type="cpu", dtype="float32", min_free_gpu_mib=0, compile=False)
    (tmp_path / "base.json").write_text(json.dumps(base))
    for path in CONFIG.glob("*.json"):
        if path.name == "base.json":
            continue
        spec = json.loads(path.read_text())
        for stage in [*spec["arms"].values(), *([spec["tail"]] if "tail" in spec else [])]:
            stage["overrides"].update(
                train={"num_steps": 1, "target_tokens": 320, "eval_every": -1, "save_every": -1},
                runtime={"compile": False},
                eval={"enabled": False},
            )
            if stage["objective"] == "B2":
                stage["policy"].update(
                    warmup_fraction=0.4, decay_fraction=0.4, warmup_blocks=[1, 320], decay_blocks=[320, 32]
                )
                stage["overrides"]["train"].update(num_steps=5, target_tokens=1600)
            if stage["objective"] == "F4":
                stage["policy"].update(calibration_batch_count=2, noise_update_interval=1)
                stage["overrides"]["train"].update(num_steps=2, target_tokens=640)
        (tmp_path / path.name).write_text(json.dumps(spec))
    try:
        yield tmp_path
    finally:
        torch.set_num_threads(previous)


def launch(folder, study, arm, parent=None):
    output = folder / (study + "-" + arm)
    args = ["train", str(folder / (study + ".json")), arm, "--seed", "48", "--output", str(output)]
    if parent:
        args += ["--trusted-ar-parent", str(parent), "--parent-sha256", file_sha256(parent)]
    main(args)
    return output


def test_cpu_training_controls_tail_and_portable_scores(tiny, capsys):
    parent = launch(tiny, "ar10", "parent") / "main/model.pt"
    panel = prepare("dev32", tiny / "val.parquet", ByteTokenizer(), reference=False, per_stratum=1)
    path = tiny / "panel.pt"
    torch.save(panel, path)
    for study, arm in [
        ("bd3", "fixed"),
        ("bd3", "calibrated"),
        ("llada1", "mdlm"),
        ("llada1", "masking"),
        ("llada2", "fixed"),
        ("llada2", "curriculum"),
        ("fastv2", "full"),
        ("fastv2", "matched"),
        ("fastv2", "token-causal"),
        ("fastv2", "block-causal"),
    ]:
        output = launch(tiny, study, arm, parent if study in {"llada2", "fastv2"} else None)
        endpoint = output / ("tail" if study == "llada2" else "main")
        checkpoint = endpoint / "model.pt"
        identity = json.loads((endpoint / "component.json").read_text())
        if study == "llada2":
            assert identity["phase_parent_sha256"] == file_sha256(output / "main/model.pt")
            assert identity["ar_parent_sha256"] == file_sha256(parent)
        if arm == "calibrated":
            trained = json.loads((endpoint / "training_report.json").read_text())
            assert trained["method_state"]["calibration_forward_count"] == 8
        weights = endpoint / "test.weights.pt"
        export_weights(checkpoint, weights, expected_source_sha256=identity["source_sha256"])
        capsys.readouterr()
        main(["eval", "--weights", str(weights), "--endpoint", str(endpoint / "component.json"), "--panel", str(path)])
        scores = json.loads(capsys.readouterr().out)
        assert scores["targets"] == scores["content_bytes"] == 64
        assert math.isfinite(scores["common_bpb"]["block_end"])
        assert scores["reference_panel"] is False
        if arm in {"full", "matched"}:
            assert scores["native_bpb_by_seed"] is None
        else:
            assert set(scores["native_bpb_by_seed"]) == {"2026110201", "2026110202"}
    with pytest.raises(ValueError, match="weights and component endpoint differ"):
        main(
            [
                "eval",
                "--weights",
                str(weights),
                "--endpoint",
                str(tiny / "llada1-mdlm/main/component.json"),
                "--panel",
                str(path),
            ]
        )
