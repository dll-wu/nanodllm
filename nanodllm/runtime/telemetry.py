import json
import math
import os
from contextlib import suppress
from pathlib import Path

from .config import ExperimentConfig, to_dict

_TENSORBOARD_CLOSED_ATTR = "_nanodllm_tensorboard_writer_closed"


def write_json_atomic(path: str, obj: dict) -> None:
    """Write a JSON object through a sibling temporary file."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def append_jsonl(path: str, obj: dict) -> None:
    """Append one deterministically keyed JSON record."""
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def resolve_tensorboard_dir(config: ExperimentConfig) -> str:
    if not config.train.tensorboard_dir:
        return os.path.join(config.train.run_dir, "tensorboard")
    path = Path(config.train.tensorboard_dir)
    if path.is_absolute():
        return str(path)
    return os.path.join(config.train.run_dir, config.train.tensorboard_dir)


def _load_summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "train.tensorboard=true requires the tensorboard package; reinstall from pyproject.toml dependencies"
        ) from exc
    return SummaryWriter


def build_tensorboard_writer(config: ExperimentConfig, rank: int):
    if not config.train.tensorboard:
        return None

    summary_writer = _load_summary_writer()
    if rank != 0:
        return None

    writer = summary_writer(log_dir=resolve_tensorboard_dir(config))
    try:
        writer.add_text("config/resolved", f"```json\n{json.dumps(to_dict(config), indent=2)}\n```", 0)
    except Exception:
        with suppress(Exception):
            close_tensorboard_writer(writer)
        raise
    return writer


def write_tensorboard_scalars(writer, prefix: str, record: dict, step: int) -> None:
    if writer is None:
        return
    for key, value in record.items():
        if key in ("step", "total_steps") or isinstance(value, bool) or not isinstance(value, int | float):
            continue
        scalar = float(value)
        if math.isfinite(scalar):
            writer.add_scalar(f"{prefix}/{key}", scalar, step)


def flush_tensorboard_writer(writer) -> None:
    if writer is not None:
        writer.flush()


def close_tensorboard_writer(writer) -> None:
    """Close a writer once, including when cleanup paths converge."""
    if writer is None or bool(getattr(writer, _TENSORBOARD_CLOSED_ATTR, False)):
        return
    writer.close()
    setattr(writer, _TENSORBOARD_CLOSED_ATTR, True)
