"""Public telemetry API, implemented in :mod:`nanodllm.runtime.telemetry`."""

from .runtime.telemetry import (
    append_jsonl,
    build_tensorboard_writer,
    close_tensorboard_writer,
    flush_tensorboard_writer,
    resolve_tensorboard_dir,
    write_json_atomic,
    write_tensorboard_scalars,
)

__all__ = [
    "append_jsonl",
    "build_tensorboard_writer",
    "close_tensorboard_writer",
    "flush_tensorboard_writer",
    "resolve_tensorboard_dir",
    "write_json_atomic",
    "write_tensorboard_scalars",
]
