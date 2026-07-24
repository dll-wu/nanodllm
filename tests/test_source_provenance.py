import json
import subprocess
from pathlib import Path

from nanodllm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, validate_config
from nanodllm.data import ByteTokenizer
from nanodllm.training import (
    _artifact_root_candidates,
    _label_smoothing_ar_metrics,
    _live_source_provenance,
    source_provenance_metrics,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def test_non_git_install_reports_unavailable_provenance_without_blocking(monkeypatch):
    def fake_run(args, **_kwargs):
        return _completed(args, returncode=128, stderr="not a git repository")

    monkeypatch.setattr(subprocess, "run", fake_run)

    live = _live_source_provenance()
    metrics = source_provenance_metrics()

    assert live["source_repository_detected"] is False
    assert live["source_git_status_available"] is False
    assert live["source_tracked_worktree_clean"] is None
    assert metrics["status"] == "ok"
    assert metrics["source_worktree_dirty"] is None
    assert metrics["source_provenance_is_informational"] is True
    assert metrics["source_provenance_warnings"]


def test_dirty_checkout_reports_counts_without_blocking_or_leaking_paths(monkeypatch):
    private_untracked_name = "private-local-notes.txt"

    def fake_run(args, **_kwargs):
        if args[-2:] == ["rev-parse", "--show-toplevel"]:
            return _completed(args, stdout=f"{REPOSITORY_ROOT}\n")
        if args[-2:] == ["rev-parse", "HEAD"]:
            return _completed(args, stdout=f"{'a' * 40}\n")
        if args[-2:] == ["status", "--porcelain"]:
            return _completed(
                args,
                stdout=f" M nanodllm/training.py\n?? {private_untracked_name}\n?? scratch-output/\n",
            )
        raise AssertionError(f"unexpected Git command: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metrics = source_provenance_metrics()

    assert metrics["status"] == "ok"
    assert metrics["source_repository_detected"] is True
    assert metrics["source_git_status_available"] is True
    assert metrics["source_git_commit"] == "a" * 40
    assert metrics["source_tracked_worktree_clean"] is False
    assert metrics["source_untracked_path_count"] == 2
    assert metrics["source_worktree_dirty"] is True
    assert private_untracked_name not in json.dumps(metrics)
    assert len(metrics["source_provenance_warnings"]) == 2


def test_causal_label_smoothing_accepts_the_configured_tokenizer_without_environment_contracts():
    config = ExperimentConfig(
        model=ModelConfig(causal=True),
        data=DataConfig(tokenizer="byte", tokenizer_path="", train_data=["unused.parquet"]),
        train=TrainConfig(label_smoothing_epsilon=0.1),
    )
    validate_config(config)

    metrics = _label_smoothing_ar_metrics(config, ByteTokenizer())

    assert metrics == {
        "ar_label_smoothing_enabled": True,
        "ar_label_smoothing_epsilon": 0.1,
        "ar_label_smoothing_vocab_size": 260,
    }


def test_comparison_artifact_roots_do_not_scan_prior_iteration_siblings(tmp_path, monkeypatch):
    current_root = tmp_path / "experiment_iter3"
    run_dir = current_root / "runs" / "current"
    working_directory = tmp_path / "working"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)

    candidates = _artifact_root_candidates(str(run_dir))

    assert candidates == [current_root.resolve(), working_directory.resolve()]
    assert tmp_path / "experiment_iter2" not in candidates
    assert tmp_path / "experiment_iter1" not in candidates
    assert tmp_path / "experiment" not in candidates
