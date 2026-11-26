import json
import subprocess
import sys
from pathlib import Path

import pytest
from nanodllm import paper_recipes
from nanodllm.paper_recipes_data import file_sha256
from nanodllm.paper_weights import export_weights


def test_cpu_ar_scratch_and_portable_conversion(tmp_path, monkeypatch):
    corpus = tmp_path / "tiny.parquet"
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_tiny_corpus.py"
    subprocess.run([sys.executable, str(script), "--output", str(corpus)], check=True)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    common = ["--corpus", str(corpus), "--output-dir", str(tmp_path / "configs")]
    for profile in ("tiny-ar", "tiny-scratch"):
        paper_recipes.main(["launch", profile, *common, "--run-dir", str(tmp_path / profile)])
    original = tmp_path / "tiny-ar" / "model_latest.pt"
    parent = tmp_path / "parent.weights.pt"
    parent_hash = export_weights(original, parent, expected_source_sha256=file_sha256(original))
    run = tmp_path / "conversion"
    paper_recipes.main(
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
        paper_recipes.main(
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
