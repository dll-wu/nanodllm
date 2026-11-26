import json
from types import SimpleNamespace

import pytest
from nanodllm.config import ExperimentConfig, TrainConfig, to_dict
from nanodllm.telemetry import (
    append_jsonl,
    build_tensorboard_writer,
    close_tensorboard_writer,
    flush_tensorboard_writer,
    resolve_tensorboard_dir,
    write_json_atomic,
    write_tensorboard_scalars,
)


class _FakeSummaryWriter:
    instances = []

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.text = []
        self.scalars = []
        self.flush_count = 0
        self.close_count = 0
        self.__class__.instances.append(self)

    def add_text(self, tag: str, text: str, step: int) -> None:
        self.text.append((tag, text, step))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.close_count += 1


def test_write_json_atomic_preserves_format_and_replaces_target(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("stale", encoding="utf-8")

    write_json_atomic(str(path), {"b": 1, "a": [2]})

    assert path.read_text(encoding="utf-8") == '{\n  "b": 1,\n  "a": [\n    2\n  ]\n}'
    assert not (tmp_path / "metrics.json.tmp").exists()


def test_append_jsonl_sorts_keys_and_appends_records(tmp_path):
    path = tmp_path / "train_log.jsonl"

    append_jsonl(str(path), {"b": 2, "a": 1})
    append_jsonl(str(path), {"d": 4, "c": 3})

    assert path.read_text(encoding="utf-8") == '{"a": 1, "b": 2}\n{"c": 3, "d": 4}\n'


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("", "run/tensorboard"),
        ("events", "run/events"),
    ],
)
def test_resolve_tensorboard_dir_handles_default_and_relative_paths(configured, expected):
    config = SimpleNamespace(train=SimpleNamespace(run_dir="run", tensorboard_dir=configured))

    assert resolve_tensorboard_dir(config) == expected


def test_resolve_tensorboard_dir_preserves_absolute_path(tmp_path):
    config = SimpleNamespace(train=SimpleNamespace(run_dir="ignored", tensorboard_dir=str(tmp_path)))

    assert resolve_tensorboard_dir(config) == str(tmp_path)


def test_build_tensorboard_writer_records_resolved_config(tmp_path, monkeypatch):
    from nanodllm.runtime import telemetry

    _FakeSummaryWriter.instances.clear()
    monkeypatch.setattr(telemetry, "_load_summary_writer", lambda: _FakeSummaryWriter)
    config = ExperimentConfig(train=TrainConfig(run_dir=str(tmp_path), tensorboard=True))

    writer = build_tensorboard_writer(config, rank=0)

    assert writer is _FakeSummaryWriter.instances[0]
    assert writer.log_dir == str(tmp_path / "tensorboard")
    assert writer.text == [
        (
            "config/resolved",
            f"```json\n{json.dumps(to_dict(config), indent=2)}\n```",
            0,
        )
    ]


def test_build_tensorboard_writer_skips_disabled_and_nonzero_rank(monkeypatch):
    from nanodllm.runtime import telemetry

    load_calls = []
    monkeypatch.setattr(
        telemetry,
        "_load_summary_writer",
        lambda: load_calls.append("loaded") or _FakeSummaryWriter,
    )
    disabled = ExperimentConfig(train=TrainConfig(tensorboard=False))
    enabled = ExperimentConfig(train=TrainConfig(tensorboard=True))

    assert build_tensorboard_writer(disabled, rank=0) is None
    assert load_calls == []
    assert build_tensorboard_writer(enabled, rank=1) is None
    assert load_calls == ["loaded"]


def test_build_tensorboard_writer_closes_if_config_record_fails(tmp_path, monkeypatch):
    from nanodllm.runtime import telemetry

    class _FailingSummaryWriter(_FakeSummaryWriter):
        def add_text(self, tag: str, text: str, step: int) -> None:
            raise RuntimeError("config record failed")

    _FailingSummaryWriter.instances.clear()
    monkeypatch.setattr(telemetry, "_load_summary_writer", lambda: _FailingSummaryWriter)
    config = ExperimentConfig(train=TrainConfig(run_dir=str(tmp_path), tensorboard=True))

    with pytest.raises(RuntimeError, match="config record failed"):
        build_tensorboard_writer(config, rank=0)

    assert _FailingSummaryWriter.instances[0].close_count == 1


def test_tensorboard_scalar_filter_flush_and_idempotent_close():
    writer = _FakeSummaryWriter("events")
    record = {
        "step": 3,
        "total_steps": 10,
        "integer": 2,
        "floating": 1.5,
        "boolean": True,
        "text": "skip",
        "infinite": float("inf"),
        "not_a_number": float("nan"),
    }

    write_tensorboard_scalars(writer, "train", record, step=3)
    flush_tensorboard_writer(writer)
    close_tensorboard_writer(writer)
    close_tensorboard_writer(writer)

    assert writer.scalars == [
        ("train/integer", 2.0, 3),
        ("train/floating", 1.5, 3),
    ]
    assert writer.flush_count == 1
    assert writer.close_count == 1


def test_tensorboard_none_helpers_are_noops():
    write_tensorboard_scalars(None, "train", {"loss": 1.0}, step=1)
    flush_tensorboard_writer(None)
    close_tensorboard_writer(None)
