import os
from pathlib import Path

import pytest
from nanodllm.runtime.training_checkpoint_paths import _resolve_resume_path


def _write_checkpoint(path: Path, *, mtime_ns: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.name.encode("utf-8"))
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_latest_discovers_root_level_step_checkpoint(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "model_step12.pt", mtime_ns=12)

    assert _resolve_resume_path(str(tmp_path), "latest") == str(checkpoint)


def test_latest_prefers_newer_root_step_over_stale_aliases(tmp_path):
    _write_checkpoint(tmp_path / "model_latest.pt", mtime_ns=10)
    _write_checkpoint(tmp_path / "model.pt", mtime_ns=10)
    checkpoint = _write_checkpoint(tmp_path / "model_step20.pt", mtime_ns=20)

    assert _resolve_resume_path(str(tmp_path), "latest") == str(checkpoint)


def test_latest_uses_numeric_step_when_stale_alias_is_an_older_hardlink(tmp_path):
    older = _write_checkpoint(tmp_path / "model_step10.pt", mtime_ns=100)
    (tmp_path / "model_latest.pt").hardlink_to(older)
    checkpoint = _write_checkpoint(tmp_path / "model_step20.pt", mtime_ns=20)

    assert _resolve_resume_path(str(tmp_path), "latest") == str(checkpoint)


def test_latest_orders_step_checkpoints_by_numeric_step_not_mtime(tmp_path):
    _write_checkpoint(tmp_path / "model_step9.pt", mtime_ns=90)
    checkpoint = _write_checkpoint(tmp_path / "model_step10.pt", mtime_ns=10)

    assert _resolve_resume_path(str(tmp_path), "latest") == str(checkpoint)


def test_latest_retains_canonical_alias_and_legacy_directory_behavior(tmp_path):
    canonical = _write_checkpoint(tmp_path / "model_latest.pt", mtime_ns=30)
    _write_checkpoint(tmp_path / "model.pt", mtime_ns=40)
    _write_checkpoint(tmp_path / "checkpoints" / "model_step20.pt", mtime_ns=20)

    assert _resolve_resume_path(str(tmp_path), "latest") == str(canonical)

    canonical.unlink()
    compatibility_alias = tmp_path / "model.pt"
    assert _resolve_resume_path(str(tmp_path), "latest") == str(compatibility_alias)

    compatibility_alias.unlink()
    legacy_step = tmp_path / "checkpoints" / "model_step20.pt"
    assert _resolve_resume_path(str(tmp_path), "latest") == str(legacy_step)


def test_latest_rejects_missing_or_malformed_step_checkpoints(tmp_path):
    _write_checkpoint(tmp_path / "model_stepnot-a-number.pt", mtime_ns=10)
    (tmp_path / "model_latest.pt").mkdir()

    with pytest.raises(FileNotFoundError, match="train.resume_from='latest'"):
        _resolve_resume_path(str(tmp_path), "latest")
