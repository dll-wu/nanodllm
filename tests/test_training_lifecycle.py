from types import SimpleNamespace

import nanodllm.training as training
import pytest


def test_restore_gc_state_unfreezes_and_restores_enabled_state(monkeypatch):
    events = []
    monkeypatch.setattr(
        training.gc,
        "get_freeze_count",
        lambda: events.append("get_freeze_count") or 1,
    )
    monkeypatch.setattr(training.gc, "unfreeze", lambda: events.append("unfreeze"))
    monkeypatch.setattr(training.gc, "enable", lambda: events.append("enable"))
    monkeypatch.setattr(training.gc, "disable", lambda: events.append("disable"))

    training._restore_gc_state(was_enabled=True, unfreeze=True)

    assert events == ["get_freeze_count", "unfreeze", "enable"]


def test_restore_gc_state_preserves_disabled_and_previously_frozen_state(monkeypatch):
    events = []
    monkeypatch.setattr(
        training.gc,
        "get_freeze_count",
        lambda: events.append("unexpected freeze-count query"),
    )
    monkeypatch.setattr(training.gc, "unfreeze", lambda: events.append("unexpected unfreeze"))
    monkeypatch.setattr(training.gc, "enable", lambda: events.append("unexpected enable"))
    monkeypatch.setattr(training.gc, "disable", lambda: events.append("disable"))

    training._restore_gc_state(was_enabled=False, unfreeze=False)

    assert events == ["disable"]


def test_train_unwinds_impl_resources_before_distributed_cleanup(monkeypatch):
    events = []
    config = object()

    def fake_train_impl(actual_config, resources):
        assert actual_config is config
        events.append("train_impl")
        resources.callback(events.append, "impl_resource")
        raise RuntimeError("training failed")

    monkeypatch.setattr(training, "_train_impl", fake_train_impl)
    monkeypatch.setattr(
        training,
        "_destroy_process_group_if_initialized",
        lambda: events.append("process_group"),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        training.train(config)

    assert events == ["train_impl", "impl_resource", "process_group"]


def test_train_preserves_process_group_owned_by_caller(monkeypatch):
    events = []

    monkeypatch.setattr(training.dist, "is_available", lambda: True)
    monkeypatch.setattr(training.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        training,
        "_train_impl",
        lambda _config, _resources: events.append("train_impl"),
    )
    monkeypatch.setattr(
        training,
        "_destroy_process_group_if_initialized",
        lambda: events.append("process_group"),
    )

    training.train(object())

    assert events == ["train_impl"]


def _publish_final_outputs(tmp_path, monkeypatch, *, save_checkpoint):
    events = []
    config = SimpleNamespace(train=SimpleNamespace(run_dir=str(tmp_path)))
    metrics = {"status": "ok", "loss": 1.25}
    writer = object()
    raw_model = object()
    optimizers = [object()]
    scaler = object()
    model_config = object()
    resume_contract = {"version": 1}

    monkeypatch.setattr(
        training,
        "_write_tensorboard_scalars",
        lambda *args: events.append(("tensorboard", args)),
    )
    monkeypatch.setattr(
        training,
        "flush_tensorboard_writer",
        lambda *args: events.append(("flush", args)),
    )

    def record_save(*args, **kwargs):
        events.append(("save", args, kwargs))
        return save_checkpoint(*args, **kwargs)

    monkeypatch.setattr(training, "save_checkpoint", record_save)
    monkeypatch.setattr(
        training,
        "alias_checkpoint",
        lambda *args: events.append(("alias", args)),
    )
    monkeypatch.setattr(
        training,
        "write_json_atomic",
        lambda *args: events.append(("metrics", args)),
    )
    monkeypatch.setattr(training, "print0", lambda *_args: None)

    def publish():
        return training._publish_final_outputs(
            config,
            rank=0,
            metrics=metrics,
            tensorboard_writer=writer,
            total_steps=7,
            raw_model=raw_model,
            optimizers=optimizers,
            scaler=scaler,
            model_config=model_config,
            train_tokens=101,
            train_valid_tokens=89,
            resume_contract=resume_contract,
        )

    expected = {
        "config": config,
        "metrics": metrics,
        "writer": writer,
        "raw_model": raw_model,
        "optimizers": optimizers,
        "scaler": scaler,
        "model_config": model_config,
        "resume_contract": resume_contract,
    }
    return publish, events, expected


def test_publish_final_outputs_checkpoints_before_success_metrics(tmp_path, monkeypatch):
    publish, events, expected = _publish_final_outputs(
        tmp_path,
        monkeypatch,
        save_checkpoint=lambda *_args, **_kwargs: None,
    )

    publish()

    assert [event[0] for event in events] == [
        "tensorboard",
        "flush",
        "save",
        "alias",
        "metrics",
    ]
    assert events[0][1] == (
        expected["writer"],
        "final",
        expected["metrics"],
        7,
    )
    assert events[1][1] == (expected["writer"],)

    save_args, save_kwargs = events[2][1:]
    assert save_args == (
        str(tmp_path / "model_latest.pt"),
        expected["raw_model"],
        expected["optimizers"],
        expected["scaler"],
        expected["config"],
        expected["model_config"],
        7,
        101,
        89,
        expected["metrics"],
    )
    assert save_kwargs == {"resume_contract": expected["resume_contract"]}
    assert events[3][1] == (
        str(tmp_path / "model_latest.pt"),
        str(tmp_path / "model.pt"),
    )
    assert events[4][1] == (
        str(tmp_path / "metrics.json"),
        expected["metrics"],
    )


def test_publish_final_outputs_does_not_write_success_metrics_when_checkpoint_fails(
    tmp_path,
    monkeypatch,
):
    def fail_checkpoint(*_args, **_kwargs):
        raise RuntimeError("checkpoint write failed")

    publish, events, _expected = _publish_final_outputs(
        tmp_path,
        monkeypatch,
        save_checkpoint=fail_checkpoint,
    )

    with pytest.raises(RuntimeError, match="checkpoint write failed"):
        publish()

    assert [event[0] for event in events] == ["tensorboard", "flush", "save"]
