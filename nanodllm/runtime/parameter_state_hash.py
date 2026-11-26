"""Exact named-parameter state digests and restart-safe step receipts."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path

import torch
import torch.nn as nn

LEGACY_PRE_UPDATE_PARAMETER_STATE_HASH_METHOD = (
    "sha256_domain_separated_sorted_named_parameter_metadata_and_raw_bytes_v1"
)
PARAMETER_STATE_HASH_METHOD = "sha256_domain_separated_sorted_named_parameter_metadata_and_raw_bytes_v2"
PARAMETER_STATE_HASH_SCHEMA = "nanodllm.parameter_state_hash.v2"
_PARAMETER_STATE_HASH_RECORD_FIELDS = frozenset(
    {
        "schema",
        "step",
        "train_tokens",
        "hash_method",
        "parameter_state_sha256",
        "architecture_fingerprint_sha256",
        "parameterization_contract_sha256",
    }
)
_WRITER_TEMP_NONCE_HEX_LENGTH = 32


def resolve_parameter_state_hash_steps(steps: list[int], *, total_steps: int) -> frozenset[int]:
    """Bind configured receipt steps to the resolved optimizer-step horizon."""

    if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps <= 0:
        raise ValueError("resolved total optimizer steps must be a positive integer")
    if not isinstance(steps, list) or any(
        not isinstance(step, int) or isinstance(step, bool) or step <= 0 for step in steps
    ):
        raise ValueError("parameter state hash steps must be a list of positive integers")
    if steps != sorted(set(steps)):
        raise ValueError("parameter state hash steps must be unique and ascending")
    resolved = frozenset(steps)
    if resolved and max(resolved) > total_steps:
        raise ValueError(
            "diagnostics.parameter_state_hash_steps entries must not exceed the resolved total optimizer steps"
        )
    return resolved


def parameter_state_sha256(raw_model: nn.Module) -> tuple[str, int, int, int]:
    """Hash exact named-parameter state using the neutral reusable v2 domain."""

    return _named_parameter_state_sha256(raw_model, domain=b"nanodllm.named_parameter_state.v2\0")


def _legacy_pre_update_parameter_state_sha256(raw_model: nn.Module) -> tuple[str, int, int, int]:
    """Preserve the initial-state v1 digest bytes used by existing releases."""

    return _named_parameter_state_sha256(raw_model, domain=b"nanodllm.pre_update_parameter_state.v1\0")


def _named_parameter_state_sha256(
    raw_model: nn.Module,
    *,
    domain: bytes,
) -> tuple[str, int, int, int]:
    """Hash exact named-parameter metadata and bytes without a full CPU state copy."""

    digest = hashlib.sha256()
    digest.update(domain)
    parameter_count = 0
    parameter_numel = 0
    parameter_bytes = 0
    for name, parameter in sorted(raw_model.named_parameters(), key=lambda item: item[0]):
        value = parameter.detach().contiguous()
        metadata = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "nbytes": int(value.numel() * value.element_size()),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        byte_view = value.reshape(-1).view(torch.uint8).cpu().numpy()
        digest.update(memoryview(byte_view))
        digest.update(b"\0")
        parameter_count += 1
        parameter_numel += int(value.numel())
        parameter_bytes += int(value.numel() * value.element_size())
    return digest.hexdigest(), parameter_count, parameter_numel, parameter_bytes


def _validate_optional_sha256(value: str, *, field: str) -> None:
    if value and (len(value) != 64 or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{field} must be empty or a lowercase 64-hex digest")


def _validated_parameter_state_hash_record(
    value: object,
    *,
    expected_architecture_fingerprint_sha256: str | None = None,
    expected_parameterization_contract_sha256: str | None = None,
) -> dict[str, object]:
    """Validate one exact receipt without accepting schema extensions by accident."""

    if not isinstance(value, dict) or set(value) != _PARAMETER_STATE_HASH_RECORD_FIELDS:
        raise ValueError("parameter state hash record has an invalid exact field inventory")
    if value.get("schema") != PARAMETER_STATE_HASH_SCHEMA:
        raise ValueError("parameter state hash record schema differs")
    if value.get("hash_method") != PARAMETER_STATE_HASH_METHOD:
        raise ValueError("parameter state hash record method differs")
    step = value.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError("parameter state hash record step must be a positive integer")
    train_tokens = value.get("train_tokens")
    if not isinstance(train_tokens, int) or isinstance(train_tokens, bool) or train_tokens < 0:
        raise ValueError("parameter state hash record train_tokens must be a nonnegative integer")
    state_sha256 = value.get("parameter_state_sha256")
    architecture_sha256 = value.get("architecture_fingerprint_sha256")
    parameterization_sha256 = value.get("parameterization_contract_sha256")
    if not isinstance(state_sha256, str):
        raise ValueError("parameter_state_sha256 must be a lowercase 64-hex digest")
    _validate_optional_sha256(state_sha256, field="parameter_state_sha256")
    if not state_sha256:
        raise ValueError("parameter_state_sha256 must not be empty")
    if not isinstance(architecture_sha256, str):
        raise ValueError("architecture_fingerprint_sha256 must be a lowercase 64-hex digest")
    _validate_optional_sha256(architecture_sha256, field="architecture_fingerprint_sha256")
    if not architecture_sha256:
        raise ValueError("architecture_fingerprint_sha256 must not be empty")
    if not isinstance(parameterization_sha256, str):
        raise ValueError("parameterization_contract_sha256 must be empty or a lowercase 64-hex digest")
    _validate_optional_sha256(parameterization_sha256, field="parameterization_contract_sha256")
    if (
        expected_architecture_fingerprint_sha256 is not None
        and architecture_sha256 != expected_architecture_fingerprint_sha256
    ):
        raise ValueError("parameter state hash record architecture fingerprint differs")
    if (
        expected_parameterization_contract_sha256 is not None
        and parameterization_sha256 != expected_parameterization_contract_sha256
    ):
        raise ValueError("parameter state hash record parameterization contract differs")
    return dict(value)


def parameter_state_hash_record(
    raw_model: nn.Module,
    *,
    step: int,
    train_tokens: int,
    architecture_fingerprint_sha256: str,
    parameterization_contract_sha256: str,
) -> dict[str, object]:
    """Build one post-optimizer-step state receipt."""

    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError("parameter state hash step must be a positive integer")
    if not isinstance(train_tokens, int) or isinstance(train_tokens, bool) or train_tokens < 0:
        raise ValueError("parameter state hash train_tokens must be a nonnegative integer")
    _validate_optional_sha256(architecture_fingerprint_sha256, field="architecture_fingerprint_sha256")
    if not architecture_fingerprint_sha256:
        raise ValueError("architecture_fingerprint_sha256 must not be empty")
    _validate_optional_sha256(parameterization_contract_sha256, field="parameterization_contract_sha256")
    state_sha256, _parameter_count, _parameter_numel, _parameter_bytes = parameter_state_sha256(raw_model)
    return {
        "schema": PARAMETER_STATE_HASH_SCHEMA,
        "step": step,
        "train_tokens": train_tokens,
        "hash_method": PARAMETER_STATE_HASH_METHOD,
        "parameter_state_sha256": state_sha256,
        "architecture_fingerprint_sha256": architecture_fingerprint_sha256,
        "parameterization_contract_sha256": parameterization_contract_sha256,
    }


def _file_identity(observed: os.stat_result) -> tuple[int, ...]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_uid,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _directory_identity(observed: os.stat_result) -> tuple[int, ...]:
    """Bind a directory without treating child creation as path replacement."""

    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_uid,
    )


def _fsync_directory(descriptor: int) -> None:
    """Durably flush directory entries when the backing filesystem supports it."""

    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise


def _validate_parameter_state_hash_target(
    observed: os.stat_result,
    *,
    expected_nlink: int = 1,
) -> tuple[int, ...]:
    if not (
        stat.S_ISREG(observed.st_mode)
        and observed.st_uid == os.geteuid()
        and observed.st_nlink == expected_nlink
        and stat.S_IMODE(observed.st_mode) == 0o600
    ):
        link_description = "single-link" if expected_nlink == 1 else f"exactly {expected_nlink}-link"
        raise ValueError(f"parameter state hash artifact must be an owned, {link_description} regular file mode 0600")
    return _file_identity(observed)


def _open_parameter_state_hash_parent(artifact_path: Path, *, create: bool = True) -> int:
    if not artifact_path.is_absolute() or Path(os.path.abspath(artifact_path)) != artifact_path:
        raise ValueError("parameter state hash artifact path must be absolute and normalized")
    if artifact_path.name != "parameter_state_hashes.jsonl":
        raise ValueError("parameter state hash artifact must use the canonical filename")
    parent = artifact_path.parent
    container = parent.parent
    if not parent.name or parent.name in {".", ".."}:
        raise ValueError("parameter state hash artifact parent name is invalid")
    try:
        container_stat = os.lstat(container)
    except FileNotFoundError as exc:
        raise ValueError("parameter state hash artifact container does not exist") from exc
    if container.resolve(strict=True) != container:
        raise ValueError("parameter state hash artifact container path must not traverse symlinks")
    if not (
        stat.S_ISDIR(container_stat.st_mode)
        and container_stat.st_uid == os.geteuid()
        and stat.S_IMODE(container_stat.st_mode) & 0o022 == 0
    ):
        raise ValueError("parameter state hash artifact container must be an owned private non-symlink directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    container_descriptor = os.open(container, flags)
    try:
        if _directory_identity(os.fstat(container_descriptor)) != _directory_identity(container_stat):
            raise ValueError("parameter state hash artifact container changed during admission")
        if create:
            with suppress(FileExistsError):
                os.mkdir(parent.name, mode=0o700, dir_fd=container_descriptor)
        parent_stat = os.stat(parent.name, dir_fd=container_descriptor, follow_symlinks=False)
        if not (
            stat.S_ISDIR(parent_stat.st_mode)
            and parent_stat.st_uid == os.geteuid()
            and stat.S_IMODE(parent_stat.st_mode) & 0o022 == 0
        ):
            raise ValueError("parameter state hash artifact parent must be an owned private non-symlink directory")
        descriptor = os.open(parent.name, flags, dir_fd=container_descriptor)
        admitted = os.fstat(descriptor)
        if _directory_identity(admitted) != _directory_identity(parent_stat):
            os.close(descriptor)
            raise ValueError("parameter state hash artifact parent changed during admission")
        return descriptor
    finally:
        os.close(container_descriptor)


def _target_stat(parent_descriptor: int, artifact_name: str) -> os.stat_result | None:
    try:
        return os.stat(artifact_name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _writer_temp_pattern(artifact_name: str) -> re.Pattern[str]:
    return re.compile(rf"^\.{re.escape(artifact_name)}\.[0-9a-f]{{{_WRITER_TEMP_NONCE_HEX_LENGTH}}}\.tmp$")


def _snapshot_pattern(artifact_name: str) -> re.Pattern[str]:
    return re.compile(rf"^\.{re.escape(artifact_name)}\.step_([0-9]{{8,}})\.snapshot$")


def _snapshot_name(artifact_name: str, step: int) -> str:
    return f".{artifact_name}.step_{step:08d}.snapshot"


def _write_parameter_state_hash_records_temp(
    parent_descriptor: int,
    artifact_name: str,
    records: list[dict[str, object]],
) -> tuple[str, tuple[int, int]]:
    """Durably write one private temporary without publishing a target name."""

    temporary_name = f".{artifact_name}.{secrets.token_hex(_WRITER_TEMP_NONCE_HEX_LENGTH // 2)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_descriptor)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_stat = os.fstat(handle.fileno())
            _validate_parameter_state_hash_target(temporary_stat)
            temporary_inode = (temporary_stat.st_dev, temporary_stat.st_ino)
        return temporary_name, temporary_inode
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        raise


def _publish_no_replace(
    parent_descriptor: int,
    source_name: str,
    target_name: str,
    *,
    source_inode: tuple[int, int],
) -> None:
    """Publish one name without ever replacing an existing directory entry."""

    try:
        os.link(
            source_name,
            target_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise ValueError(f"parameter state hash publication target already exists: {target_name}") from exc
    try:
        source = _target_stat(parent_descriptor, source_name)
        promoted = _target_stat(parent_descriptor, target_name)
        if source is None or promoted is None:
            raise RuntimeError("parameter state hash no-replace publication lost a linked name")
        _validate_parameter_state_hash_target(source, expected_nlink=2)
        _validate_parameter_state_hash_target(promoted, expected_nlink=2)
        if (source.st_dev, source.st_ino) != source_inode or (promoted.st_dev, promoted.st_ino) != source_inode:
            raise ValueError("parameter state hash publication inode differs")
        _fsync_directory(parent_descriptor)
        os.unlink(source_name, dir_fd=parent_descriptor)
        _fsync_directory(parent_descriptor)
        promoted = _target_stat(parent_descriptor, target_name)
        if promoted is None:
            raise RuntimeError("parameter state hash artifact promotion lost its target")
        _validate_parameter_state_hash_target(promoted)
        if (promoted.st_dev, promoted.st_ino) != source_inode:
            raise ValueError("parameter state hash artifact changed during atomic promotion")
    except BaseException:
        # Never remove target_name here: after publication, a non-cooperating
        # writer may have replaced it. Leaving evidence is safer than deleting
        # a directory entry whose identity no longer belongs to this writer.
        raise


def _read_parameter_state_hash_records(
    parent_descriptor: int,
    artifact_name: str,
    *,
    expected_architecture_fingerprint_sha256: str | None = None,
    expected_parameterization_contract_sha256: str | None = None,
) -> tuple[list[dict[str, object]], tuple[int, ...] | None]:
    observed = _target_stat(parent_descriptor, artifact_name)
    if observed is None:
        return [], None
    target_identity = _validate_parameter_state_hash_target(observed)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(artifact_name, flags, dir_fd=parent_descriptor)
    admitted = os.fstat(descriptor)
    if _file_identity(admitted) != target_identity:
        os.close(descriptor)
        raise ValueError("parameter state hash artifact changed during admission")
    records: list[dict[str, object]] = []
    observed_steps: set[int] = set()
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                decoded = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"parameter state hash artifact has invalid JSON on line {line_number}") from exc
            try:
                record = _validated_parameter_state_hash_record(
                    decoded,
                    expected_architecture_fingerprint_sha256=expected_architecture_fingerprint_sha256,
                    expected_parameterization_contract_sha256=expected_parameterization_contract_sha256,
                )
            except ValueError as exc:
                raise ValueError(
                    f"parameter state hash artifact has malformed record on line {line_number}: {exc}"
                ) from exc
            step = int(record["step"])
            if step in observed_steps:
                raise ValueError(f"parameter state hash artifact contains duplicate step {step}")
            if records and step <= int(records[-1]["step"]):
                raise ValueError("parameter state hash artifact records are not in ascending optimizer-step order")
            observed_steps.add(step)
            records.append(record)
        if _file_identity(os.fstat(handle.fileno())) != target_identity:
            raise ValueError("parameter state hash artifact changed while it was read")
    if not records:
        raise ValueError("parameter state hash artifact must not be empty")
    return records, target_identity


def _safe_unlink_writer_name(
    parent_descriptor: int,
    name: str,
    *,
    expected_identity: tuple[int, ...],
) -> None:
    current = _target_stat(parent_descriptor, name)
    if current is None or _file_identity(current) != expected_identity:
        raise ValueError(f"parameter state hash recovery target changed before cleanup: {name}")
    os.unlink(name, dir_fd=parent_descriptor)
    _fsync_directory(parent_descriptor)


def _recover_writer_temporaries(parent_descriptor: int, artifact_name: str) -> None:
    """Remove only exact private writer temps, including one torn publish link."""

    temp_pattern = _writer_temp_pattern(artifact_name)
    snapshot_pattern = _snapshot_pattern(artifact_name)
    names = sorted(os.listdir(parent_descriptor))
    allowed_publish_names = {
        name for name in names if name == artifact_name or snapshot_pattern.fullmatch(name) is not None
    }
    for name in names:
        if temp_pattern.fullmatch(name) is None:
            if name.startswith(f".{artifact_name}.") and name.endswith(".tmp"):
                raise ValueError(f"parameter state hash artifact has an unsafe torn temporary: {name}")
            continue
        observed = _target_stat(parent_descriptor, name)
        if observed is None:
            continue
        if observed.st_nlink == 1:
            identity = _validate_parameter_state_hash_target(observed)
        elif observed.st_nlink == 2:
            identity = _validate_parameter_state_hash_target(observed, expected_nlink=2)
            peers = []
            for candidate in allowed_publish_names:
                peer = _target_stat(parent_descriptor, candidate)
                if peer is not None and (peer.st_dev, peer.st_ino) == (observed.st_dev, observed.st_ino):
                    peers.append(candidate)
            if len(peers) != 1:
                raise ValueError("parameter state hash torn temporary has an unsafe hard-link inventory")
        else:
            raise ValueError("parameter state hash torn temporary has an unsafe hard-link inventory")
        _safe_unlink_writer_name(parent_descriptor, name, expected_identity=identity)


def _read_available_parameter_state_hash_records(
    parent_descriptor: int,
    artifact_name: str,
    *,
    expected_architecture_fingerprint_sha256: str,
    expected_parameterization_contract_sha256: str,
    clean_completed_snapshots: bool,
) -> tuple[list[dict[str, object]], bool]:
    """Read canonical terminal state or the latest immutable prefix snapshot."""

    snapshot_pattern = _snapshot_pattern(artifact_name)
    snapshot_names: list[tuple[int, str]] = []
    for name in sorted(os.listdir(parent_descriptor)):
        match = snapshot_pattern.fullmatch(name)
        if match is not None:
            filename_step = int(match.group(1))
            if name != _snapshot_name(artifact_name, filename_step):
                raise ValueError(f"parameter state hash artifact has a noncanonical snapshot name: {name}")
            snapshot_names.append((filename_step, name))
        elif name.startswith(f".{artifact_name}.step_") and name.endswith(".snapshot"):
            raise ValueError(f"parameter state hash artifact has an unsafe snapshot name: {name}")
    snapshot_names.sort()

    canonical = _target_stat(parent_descriptor, artifact_name)
    if canonical is not None and canonical.st_nlink == 2:
        peers = []
        for _step, name in snapshot_names:
            snapshot = _target_stat(parent_descriptor, name)
            if snapshot is not None and (snapshot.st_dev, snapshot.st_ino) == (
                canonical.st_dev,
                canonical.st_ino,
            ):
                peers.append(name)
        if len(peers) != 1:
            raise ValueError("parameter state hash canonical artifact has an unsafe hard-link inventory")
        _validate_parameter_state_hash_target(canonical, expected_nlink=2)
        peer = _target_stat(parent_descriptor, peers[0])
        assert peer is not None
        peer_identity = _validate_parameter_state_hash_target(peer, expected_nlink=2)
        _safe_unlink_writer_name(parent_descriptor, peers[0], expected_identity=peer_identity)
        snapshot_names = [(step, name) for step, name in snapshot_names if name != peers[0]]
        canonical = _target_stat(parent_descriptor, artifact_name)

    canonical_records: list[dict[str, object]] = []
    if canonical is not None:
        canonical_records, _identity = _read_parameter_state_hash_records(
            parent_descriptor,
            artifact_name,
            expected_architecture_fingerprint_sha256=expected_architecture_fingerprint_sha256,
            expected_parameterization_contract_sha256=expected_parameterization_contract_sha256,
        )

    latest_records: list[dict[str, object]] = []
    for filename_step, name in snapshot_names:
        records, identity = _read_parameter_state_hash_records(
            parent_descriptor,
            name,
            expected_architecture_fingerprint_sha256=expected_architecture_fingerprint_sha256,
            expected_parameterization_contract_sha256=expected_parameterization_contract_sha256,
        )
        if int(records[-1]["step"]) != filename_step:
            raise ValueError("parameter state hash snapshot filename step differs from its ledger")
        if latest_records and records[: len(latest_records)] != latest_records:
            raise ValueError("parameter state hash immutable snapshots do not form one exact prefix chain")
        latest_records = records
        if canonical_records:
            if canonical_records[: len(records)] != records:
                raise ValueError("parameter state hash snapshot differs from the canonical ledger prefix")
            if clean_completed_snapshots:
                assert identity is not None
                _safe_unlink_writer_name(parent_descriptor, name, expected_identity=identity)

    if canonical_records:
        return canonical_records, True
    return latest_records, False


def _finalize_parameter_state_hash_ledger(
    parent_descriptor: int,
    artifact_name: str,
    *,
    terminal_step: int,
    architecture_fingerprint_sha256: str,
    parameterization_contract_sha256: str,
) -> None:
    """Publish the terminal immutable snapshot under the canonical name."""

    snapshot_name = _snapshot_name(artifact_name, terminal_step)
    snapshot = _target_stat(parent_descriptor, snapshot_name)
    if snapshot is None:
        raise RuntimeError("parameter state hash terminal snapshot disappeared")
    _validate_parameter_state_hash_target(snapshot)
    snapshot_inode = (snapshot.st_dev, snapshot.st_ino)
    _publish_no_replace(
        parent_descriptor,
        snapshot_name,
        artifact_name,
        source_inode=snapshot_inode,
    )
    _read_available_parameter_state_hash_records(
        parent_descriptor,
        artifact_name,
        expected_architecture_fingerprint_sha256=architecture_fingerprint_sha256,
        expected_parameterization_contract_sha256=parameterization_contract_sha256,
        clean_completed_snapshots=True,
    )
    canonical = _target_stat(parent_descriptor, artifact_name)
    if canonical is None or (canonical.st_dev, canonical.st_ino) != snapshot_inode:
        raise RuntimeError("parameter state hash terminal publication was not durably finalized")


def validate_resume_parameter_state_hashes(
    path: str | Path,
    raw_model: nn.Module,
    *,
    configured_steps: Sequence[int],
    checkpoint_step: int,
    checkpoint_train_tokens: int,
    architecture_fingerprint_sha256: str,
    parameterization_contract_sha256: str,
) -> list[dict[str, object]]:
    """Validate the complete ledger and bind a scheduled resume state exactly."""

    steps = resolve_parameter_state_hash_steps(
        list(configured_steps),
        total_steps=max([1, checkpoint_step, *configured_steps]),
    )
    artifact_path = Path(path)
    required_steps = {step for step in steps if step <= checkpoint_step}
    try:
        parent_descriptor = _open_parameter_state_hash_parent(artifact_path, create=False)
    except FileNotFoundError:
        if required_steps:
            raise ValueError("parameter state hash artifact is missing required resume receipts") from None
        return []
    try:
        fcntl.flock(parent_descriptor, fcntl.LOCK_EX)
        _recover_writer_temporaries(parent_descriptor, artifact_path.name)
        records, _canonical = _read_available_parameter_state_hash_records(
            parent_descriptor,
            artifact_path.name,
            expected_architecture_fingerprint_sha256=architecture_fingerprint_sha256,
            expected_parameterization_contract_sha256=parameterization_contract_sha256,
            clean_completed_snapshots=True,
        )
    finally:
        os.close(parent_descriptor)
    observed_by_step = {int(record["step"]): record for record in records}
    unexpected = sorted(set(observed_by_step) - set(steps))
    future = sorted(step for step in observed_by_step if step > checkpoint_step)
    missing = sorted(required_steps - set(observed_by_step))
    if unexpected or future or missing:
        raise ValueError(
            "parameter state hash resume ledger differs from the configured checkpoint prefix: "
            f"unexpected={unexpected}, future={future}, missing={missing}"
        )
    if checkpoint_step in steps:
        expected = parameter_state_hash_record(
            raw_model,
            step=checkpoint_step,
            train_tokens=checkpoint_train_tokens,
            architecture_fingerprint_sha256=architecture_fingerprint_sha256,
            parameterization_contract_sha256=parameterization_contract_sha256,
        )
        if observed_by_step.get(checkpoint_step) != expected:
            raise ValueError("resume checkpoint parameter state differs from its scheduled receipt")
    return records


def validate_checkpoint_parameter_state_hash(
    checkpoint: Mapping[str, object],
    raw_model: nn.Module,
    *,
    architecture_fingerprint_sha256: str,
    parameterization_contract_sha256: str,
) -> dict[str, object]:
    """Bind an opt-in checkpoint's exact loaded tensors to its embedded receipt."""

    observed = _validated_parameter_state_hash_record(
        checkpoint.get("parameter_state_hash"),
        expected_architecture_fingerprint_sha256=architecture_fingerprint_sha256,
        expected_parameterization_contract_sha256=parameterization_contract_sha256,
    )
    expected = parameter_state_hash_record(
        raw_model,
        step=int(checkpoint.get("step", 0)),
        train_tokens=int(checkpoint.get("train_tokens", 0)),
        architecture_fingerprint_sha256=architecture_fingerprint_sha256,
        parameterization_contract_sha256=parameterization_contract_sha256,
    )
    if observed != expected:
        raise ValueError("resume checkpoint parameter state differs from its embedded receipt")
    return observed


def append_parameter_state_hash_record_once(
    path: str | Path,
    record: dict[str, object],
    *,
    configured_steps: Sequence[int],
) -> None:
    """Publish one immutable prefix snapshot and a terminal no-replace ledger."""

    artifact_path = Path(path)
    if not artifact_path.name or artifact_path.name in {".", ".."}:
        raise ValueError("parameter state hash artifact name is invalid")
    record = _validated_parameter_state_hash_record(record)
    steps = resolve_parameter_state_hash_steps(
        list(configured_steps),
        total_steps=max([int(record["step"]), *configured_steps]),
    )
    if not steps or int(record["step"]) not in steps:
        raise ValueError("parameter state hash record step is not in the configured receipt schedule")
    ordered_steps = sorted(steps)
    parent_descriptor = _open_parameter_state_hash_parent(artifact_path)
    try:
        # The private directory is the stable serialization object even before
        # the ledger exists. Every cooperating writer therefore admits, reads,
        # and promotes under one process-safe lock.
        fcntl.flock(parent_descriptor, fcntl.LOCK_EX)
        _recover_writer_temporaries(parent_descriptor, artifact_path.name)
        existing_records, canonical_exists = _read_available_parameter_state_hash_records(
            parent_descriptor,
            artifact_path.name,
            expected_architecture_fingerprint_sha256=str(record["architecture_fingerprint_sha256"]),
            expected_parameterization_contract_sha256=str(record["parameterization_contract_sha256"]),
            clean_completed_snapshots=True,
        )
        existing_by_step = {int(existing["step"]): existing for existing in existing_records}
        unexpected = sorted(set(existing_by_step) - steps)
        if unexpected:
            raise ValueError(f"parameter state hash ledger has unconfigured steps: {unexpected}")
        if [int(existing["step"]) for existing in existing_records] != ordered_steps[: len(existing_records)]:
            raise ValueError("parameter state hash ledger is not the exact configured receipt prefix")
        if canonical_exists and int(existing_records[-1]["step"]) != ordered_steps[-1]:
            raise ValueError("parameter state hash canonical ledger was published before the terminal receipt")
        step = int(record["step"])
        existing = existing_by_step.get(step)
        if existing is not None:
            if existing != record:
                raise ValueError(f"parameter state hash record for step {step} differs from the existing receipt")
            if step == ordered_steps[-1] and not canonical_exists:
                _finalize_parameter_state_hash_ledger(
                    parent_descriptor,
                    artifact_path.name,
                    terminal_step=step,
                    architecture_fingerprint_sha256=str(record["architecture_fingerprint_sha256"]),
                    parameterization_contract_sha256=str(record["parameterization_contract_sha256"]),
                )
            return
        if existing_by_step and step <= max(existing_by_step):
            raise ValueError("parameter state hash records must be appended in ascending optimizer-step order")
        if canonical_exists:
            raise ValueError("parameter state hash canonical ledger is immutable")
        if step != ordered_steps[len(existing_records)]:
            raise ValueError("parameter state hash record is not the next configured receipt step")

        records = [*existing_records, record]
        temporary_name, temporary_inode = _write_parameter_state_hash_records_temp(
            parent_descriptor,
            artifact_path.name,
            records,
        )
        snapshot_name = _snapshot_name(artifact_path.name, step)
        try:
            _publish_no_replace(
                parent_descriptor,
                temporary_name,
                snapshot_name,
                source_inode=temporary_inode,
            )
            temporary_name = ""
        finally:
            if temporary_name:
                temporary = _target_stat(parent_descriptor, temporary_name)
                if temporary is not None and temporary.st_nlink == 1:
                    identity = _validate_parameter_state_hash_target(temporary)
                    _safe_unlink_writer_name(parent_descriptor, temporary_name, expected_identity=identity)

        if step == ordered_steps[-1]:
            _finalize_parameter_state_hash_ledger(
                parent_descriptor,
                artifact_path.name,
                terminal_step=step,
                architecture_fingerprint_sha256=str(record["architecture_fingerprint_sha256"]),
                parameterization_contract_sha256=str(record["parameterization_contract_sha256"]),
            )
    finally:
        os.close(parent_descriptor)
