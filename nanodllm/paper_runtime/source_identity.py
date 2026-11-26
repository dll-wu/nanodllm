"""Bounded Git source checks and the historical non-Git identity fallback."""

from __future__ import annotations

import errno
import hashlib
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

_LOGGER = logging.getLogger(__name__)
_GIT_TIMEOUT_SECONDS = 5
_GIT_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 0.25
_TRANSIENT_ERRNOS = {errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT}
_RUNTIME_PATHS = ("nanodllm", "pretrain.py", "pyproject.toml", "uv.lock")


@dataclass(frozen=True)
class _GitSourceState:
    head: str
    clean: bool


def _error_detail(error: OSError | subprocess.SubprocessError) -> str:
    # Keep scheduler diagnostics bounded and single-line; never dump the environment.
    if isinstance(error, subprocess.TimeoutExpired):
        detail = f"timeout after {error.timeout}s"
    elif isinstance(error, subprocess.CalledProcessError):
        detail = f"exit code {error.returncode}"
    else:
        detail = f"{type(error).__name__}: {str(error)[:500]!r}"
    stderr = getattr(error, "stderr", None)
    if stderr:
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        detail += f"; stderr={stderr[:2000]!r}"
    return detail


def _git_source_state(repo_root: Path) -> _GitSourceState | None:
    """Retry only transient probe failures, never dirty state or a Git error exit."""
    for attempt in range(1, _GIT_ATTEMPTS + 1):
        command = ["rev-parse", "HEAD"]
        try:
            head = subprocess.run(
                ["git", "-C", str(repo_root), *command],
                check=True,
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SECONDS,
            ).stdout.strip()
            if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
                _LOGGER.warning("Git source check rejected invalid HEAD; no retry; using source-tree fallback")
                return None
            command = ["status", "--porcelain", "--untracked-files=all", "--", *_RUNTIME_PATHS]
            status = subprocess.run(
                ["git", "-C", str(repo_root), *command],
                check=True,
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SECONDS,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            transient = isinstance(error, subprocess.TimeoutExpired) or (
                isinstance(error, OSError) and error.errno in _TRANSIENT_ERRNOS
            )
            retry = transient and attempt < _GIT_ATTEMPTS
            _LOGGER.warning(
                "Git source check failed (command=%s, attempt=%s/%s): %s; %s",
                " ".join(command),
                attempt,
                _GIT_ATTEMPTS,
                _error_detail(error),
                "retrying complete probe once" if retry else "no further retry; using source-tree fallback",
            )
            if retry:
                time.sleep(_RETRY_DELAY_SECONDS)
                continue
            return None
        if status:
            _LOGGER.warning(
                "Git source check found dirty runtime paths (%s entries); no retry; using source-tree fallback",
                len(status.splitlines()),
            )
        # Preserve the fallback's known HEAD for a successfully observed dirty tree.
        return _GitSourceState(head=head, clean=not status)
    raise AssertionError("unreachable Git source-check state")


def source_implementation_identity() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    state = _git_source_state(repo_root) if (repo_root / ".git").exists() else None
    if state is not None and state.clean:
        return {"mode": "clean_git_head", "git_head": state.head}
    head = state.head if state is not None else ""

    files = [repo_root / name for name in ("pretrain.py", "pyproject.toml", "uv.lock")]
    files.extend(sorted((repo_root / "nanodllm").rglob("*.py")))
    digest = hashlib.sha256(b"nanodllm-source-tree-v1\0")
    included = 0
    for path in sorted({path.resolve() for path in files if path.is_file()}):
        relative = path.relative_to(repo_root.resolve()).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
        included += 1
    if not included:
        raise RuntimeError(f"could not identify any runtime source files below {repo_root}")
    return {
        "mode": "source_tree_sha256_fallback",
        "git_head": head,
        "source_tree_sha256": digest.hexdigest(),
        "source_file_count": included,
    }
