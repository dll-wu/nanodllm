"""Descriptor-bound loading for checkpoints with an expected byte identity."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, BinaryIO

import torch

EXPECTED_CHECKPOINT_PATH_ENV = "NANODLLM_EXPECTED_CHECKPOINT_PATH"
EXPECTED_CHECKPOINT_SHA256_ENV = "NANODLLM_EXPECTED_CHECKPOINT_SHA256"
EXPECTED_CHECKPOINT_SIZE_BYTES_ENV = "NANODLLM_EXPECTED_CHECKPOINT_SIZE_BYTES"
_BINDING_ENV_NAMES = (
    EXPECTED_CHECKPOINT_PATH_ENV,
    EXPECTED_CHECKPOINT_SHA256_ENV,
    EXPECTED_CHECKPOINT_SIZE_BYTES_ENV,
)
_SHA256_HEX = frozenset("0123456789abcdef")


class CheckpointLoadBindingError(RuntimeError):
    """The checkpoint path or bytes differed from an explicit load binding."""


@dataclass(frozen=True, slots=True)
class ExpectedCheckpointBinding:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CheckpointLoadReceipt:
    path: str
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    def metrics(self) -> dict[str, object]:
        return {
            "source_checkpoint_load_binding_status": "exact_descriptor",
            "source_checkpoint_path": self.path,
            "source_checkpoint_sha256": self.sha256,
            "source_checkpoint_size_bytes": self.size_bytes,
            "source_checkpoint_device": self.device,
            "source_checkpoint_inode": self.inode,
            "source_checkpoint_mtime_ns": self.mtime_ns,
            "source_checkpoint_ctime_ns": self.ctime_ns,
        }


def _absolute_lexical_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def _validate_sha256(value: str, *, field: str) -> str:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise CheckpointLoadBindingError(f"{field} must be a lowercase 64-hex digest")
    return value


def _validate_size(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise CheckpointLoadBindingError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointLoadBindingError(f"{field} must be a positive integer") from exc
    if parsed <= 0 or str(parsed) != str(value):
        raise CheckpointLoadBindingError(f"{field} must be a positive canonical integer")
    return parsed


def expected_checkpoint_binding_from_env(
    environ: Mapping[str, str] | None = None,
) -> ExpectedCheckpointBinding | None:
    """Read the all-or-none expected checkpoint binding from the child environment."""

    source = os.environ if environ is None else environ
    values = {name: source.get(name, "") for name in _BINDING_ENV_NAMES}
    present = {name for name, value in values.items() if value}
    if not present:
        return None
    if present != set(_BINDING_ENV_NAMES):
        missing = sorted(set(_BINDING_ENV_NAMES) - present)
        raise CheckpointLoadBindingError(f"checkpoint load binding environment is incomplete: missing {missing}")
    path = _absolute_lexical_path(values[EXPECTED_CHECKPOINT_PATH_ENV])
    if path != values[EXPECTED_CHECKPOINT_PATH_ENV]:
        raise CheckpointLoadBindingError("expected checkpoint path must be absolute and lexically canonical")
    return ExpectedCheckpointBinding(
        path=path,
        sha256=_validate_sha256(values[EXPECTED_CHECKPOINT_SHA256_ENV], field=EXPECTED_CHECKPOINT_SHA256_ENV),
        size_bytes=_validate_size(values[EXPECTED_CHECKPOINT_SIZE_BYTES_ENV], field=EXPECTED_CHECKPOINT_SIZE_BYTES_ENV),
    )


def _merge_expected_binding(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None,
    expected_size_bytes: int | None,
    environ: Mapping[str, str] | None,
) -> ExpectedCheckpointBinding | None:
    environment_binding = expected_checkpoint_binding_from_env(environ)
    explicit_present = expected_sha256 not in (None, "") or expected_size_bytes not in (None, -1)
    if explicit_present:
        if expected_sha256 in (None, "") or expected_size_bytes in (None, -1):
            raise CheckpointLoadBindingError("explicit checkpoint SHA-256 and size must be provided together")
        explicit = ExpectedCheckpointBinding(
            path=_absolute_lexical_path(path),
            sha256=_validate_sha256(str(expected_sha256), field="expected checkpoint SHA-256"),
            size_bytes=_validate_size(expected_size_bytes, field="expected checkpoint size"),
        )
        if environment_binding is not None and environment_binding != explicit:
            raise CheckpointLoadBindingError("environment and explicit checkpoint load bindings differ")
        return explicit
    return environment_binding


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _require_stable_source(
    path: str,
    descriptor: int,
    admitted: os.stat_result,
    *,
    phase: str,
) -> None:
    observed = os.fstat(descriptor)
    try:
        lexical = os.lstat(path)
    except OSError as exc:
        raise CheckpointLoadBindingError(f"checkpoint path is unavailable {phase}: {exc}") from exc
    if _stat_identity(observed) != _stat_identity(admitted):
        raise CheckpointLoadBindingError(f"checkpoint descriptor changed {phase}")
    if (lexical.st_dev, lexical.st_ino) != (admitted.st_dev, admitted.st_ino):
        raise CheckpointLoadBindingError(f"checkpoint path changed {phase}")


def _copy_and_hash(source: BinaryIO, destination: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = source.read(8 * 1024 * 1024)
        if not chunk:
            break
        destination.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    destination.flush()
    destination.seek(0)
    return digest.hexdigest(), size


def _load_exact_descriptor(
    path: str,
    binding: ExpectedCheckpointBinding,
    *,
    map_location: Any,
    weights_only: bool,
) -> tuple[Any, CheckpointLoadReceipt]:
    if path != binding.path:
        raise CheckpointLoadBindingError(
            f"checkpoint path differs from binding: observed {path}, expected {binding.path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointLoadBindingError(f"cannot open bound checkpoint: {exc}") from exc
    try:
        admitted = os.fstat(descriptor)
        if not stat.S_ISREG(admitted.st_mode):
            raise CheckpointLoadBindingError("bound checkpoint is not a regular file")
        if admitted.st_size != binding.size_bytes:
            raise CheckpointLoadBindingError(
                f"checkpoint size mismatch: observed {admitted.st_size}, expected {binding.size_bytes}"
            )
        _require_stable_source(path, descriptor, admitted, phase="during admission")
        with os.fdopen(descriptor, "rb", closefd=False) as source, tempfile.TemporaryFile(mode="w+b") as private:
            digest, copied_size = _copy_and_hash(source, private)
            _require_stable_source(path, descriptor, admitted, phase="while copying")
            if copied_size != binding.size_bytes:
                raise CheckpointLoadBindingError(
                    f"checkpoint copied size mismatch: observed {copied_size}, expected {binding.size_bytes}"
                )
            if digest != binding.sha256:
                raise CheckpointLoadBindingError(
                    f"checkpoint SHA-256 mismatch: observed {digest}, expected {binding.sha256}"
                )
            private_before = os.fstat(private.fileno())
            value = torch.load(private, map_location=map_location, weights_only=weights_only)
            private_after = os.fstat(private.fileno())
            if _stat_identity(private_after) != _stat_identity(private_before):
                raise CheckpointLoadBindingError("private checkpoint descriptor changed while loading")
            _require_stable_source(path, descriptor, admitted, phase="while loading")
    except CheckpointLoadBindingError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointLoadBindingError(f"cannot safely load bound checkpoint: {exc}") from exc
    finally:
        os.close(descriptor)
    return value, CheckpointLoadReceipt(
        path=path,
        sha256=binding.sha256,
        size_bytes=binding.size_bytes,
        device=int(admitted.st_dev),
        inode=int(admitted.st_ino),
        mtime_ns=int(admitted.st_mtime_ns),
        ctime_ns=int(admitted.st_ctime_ns),
    )


def load_checkpoint_with_optional_binding(
    path: str | os.PathLike[str],
    *,
    map_location: Any,
    weights_only: bool,
    mmap_unbound: bool = False,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    receipt: MutableMapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Load a checkpoint, using a private same-descriptor copy when bound.

    Calls without an explicit or environment binding retain the historical
    ``torch.load(path, ...)`` behavior, including optional mmap loading.
    """

    lexical_path = _absolute_lexical_path(path)
    binding = _merge_expected_binding(
        lexical_path,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        environ=environ,
    )
    if binding is None:
        return torch.load(path, map_location=map_location, weights_only=weights_only, mmap=mmap_unbound)
    value, load_receipt = _load_exact_descriptor(
        lexical_path,
        binding,
        map_location=map_location,
        weights_only=weights_only,
    )
    if receipt is not None:
        receipt.clear()
        receipt.update(load_receipt.metrics())
    return value
