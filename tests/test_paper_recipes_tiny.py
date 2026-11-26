import json
import subprocess
import sys
from pathlib import Path

import pytest
from nanodllm.weights import export_weights
from recipes import launch
from recipes.data import file_sha256


def test_cpu_ar_scratch_and_portable_conversion(tmp_path, monkeypatch):
    corpus = tmp_path / "tiny.parquet"
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_tiny_corpus.py"
    subprocess.run([sys.executable, str(script), "--output", str(corpus)], check=True)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    common = ["--corpus", str(corpus), "--output-dir", str(tmp_path / "configs")]
    for profile in ("tiny-ar", "tiny-scratch"):
        launch.main(["launch", profile, *common, "--run-dir", str(tmp_path / profile)])
    original = tmp_path / "tiny-ar" / "model_latest.pt"
    parent = tmp_path / "parent.weights.pt"
    parent_hash = export_weights(original, parent, expected_source_sha256=file_sha256(original))
    run = tmp_path / "conversion"
    launch.main(
        [
            "launch",
            "tiny-convert",
            *common,
            "--run-dir",
            str(run),
            "--parent",
            str(parent),
            "--parent-sha256",
            parent_hash,
        ]
    )
    receipt = json.loads((run / "initial_state.v1.json").read_text())["model_initialization_source"]
    assert receipt["sha256"] == parent_hash
    assert receipt["source_step"] == 2 and receipt["source_train_tokens"] == 256
    assert receipt["restored_optimizer_state"] is False
    assert (run / "model_latest.pt").is_file()


def test_launch_requires_opt_in_for_resolved_native_parent(tmp_path, capsys):
    original = tmp_path / "native.pt"
    original.write_bytes(b"local training fixture")
    alias = tmp_path / "alias.weights.pt"
    alias.symlink_to(original)
    with pytest.raises(SystemExit) as exc:
        launch.main(
            [
                "launch",
                "tiny-convert",
                "--corpus",
                str(tmp_path / "data.parquet"),
                "--run-dir",
                str(tmp_path / "out"),
                "--parent",
                str(alias),
                "--parent-sha256",
                file_sha256(original),
            ]
        )
    assert exc.value.code == 2
    assert "--trusted-training-parent" in capsys.readouterr().err


def test_tiny_block_public_trainer(tmp_path, monkeypatch):
    from nanodllm.cli import main as pretrain

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    subprocess.run(
        [sys.executable, str(root / "scripts/prepare_tiny_corpus.py"), "--output", "data/tiny_corpus.parquet"],
        check=True,
    )
    pretrain(["--config", str(root / "recipes/tiny-block.json")])
    run = tmp_path / "runs/tiny-block"
    cfg = json.loads((run / "resolved_config.json").read_text())
    assert cfg["model"]["max_seq_len"] == 128
    assert cfg["model"]["bd3lm_block_size"] == 8 and cfg["noise"]["type"] == "bd3lm"
    assert cfg["runtime"]["device_type"] == "cpu" and cfg["runtime"]["dtype"] == "float32"
    steps = [json.loads(line)["step"] for line in (run / "train_log.jsonl").read_text().splitlines()]
    assert steps == [1, 2] and (run / "model.pt").is_file()
